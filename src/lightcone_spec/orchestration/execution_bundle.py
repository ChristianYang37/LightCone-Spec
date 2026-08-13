"""Path-bound reconstruction for one industrial dispatch assignment.

An :class:`IndustrialExecutionPlan` summary is intentionally not a loadable
execution instruction.  This module reopens the raw registry, scheduler,
topology, load, model, compile, and runtime artifacts and re-runs the existing
first-party reducers.  Read-only audit proves exact component and scheduler
replay without emitting an execution-plan identity.  A physical plan can be
built only after the global launch boundary revalidates every raw authority.

The boundary is read-only.  In particular, loading a bundle never creates an
evidence root, starts a process, imports CUDA, or contacts a model provider.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import stat
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.adaptation.parameters import TrainablePlan
from lightcone_spec.adaptation.plan_authority import (
    TrainablePlanAuthorityBinding,
    TrainablePlanRawJsonBinding,
    audit_trainable_plan_authority_for_method,
    require_trainable_plan_authority_for_method,
    trainable_plan_authority_binding_from_dict,
    trainable_plan_authority_binding_to_dict,
)
from lightcone_spec.config import load_run_config, run_config_sha256
from lightcone_spec.execution import ControlledExecutionPolicy
from lightcone_spec.experiments.budget_authority import (
    BudgetMaterializationBlockedError,
    bind_budget_materialization_authority,
    load_declared_budget_plan,
    replay_budget_activation_authority,
    require_ready_budget_materialization_authority_binding,
)
from lightcone_spec.experiments.capacity_authority import (
    CapacityAuthorityUnavailableError,
    bind_capacity_authority,
)
from lightcone_spec.experiments.completion_authority import (
    AssignmentTerminalAuthority,
    AssignmentTerminalBinding,
)
from lightcone_spec.experiments.failure_authority import (
    FailureExecutionAuthorityToken,
    FailureInjectionAuthorityBinding,
    FailureInjectionAuthorityBlocked,
    require_failure_injection_authority,
    revalidate_failure_injection_authority,
)
from lightcone_spec.experiments.gpu_pool import (
    AssignmentExecutionReceipt,
    AssignmentExecutionStatus,
    DispatchAttemptJournalReplay,
    DispatchAttemptJournalToken,
    DispatchExecutionPhase,
    DispatchScheduleReceipt,
    GpuAssignment,
    GpuDispatchExecutionContext,
    GpuDispatchPlan,
    GpuDispatchPlanningContext,
    GpuDispatchWave,
    GpuInventory,
    InterferenceEnvelope,
    _make_schedule_receipt,
    _make_wave_execution_receipt,
    execute_dispatch_plan,
    validate_dispatch_resume,
)
from lightcone_spec.experiments.interference_authority import (
    InterferenceCalibrationAuthority,
    InterferenceCalibrationBlockedError,
    InterferenceCalibrationSourceAuthority,
    materialize_interference_calibration_bootstrap_authority,
    require_calibrated_interference_execution_authority,
)
from lightcone_spec.experiments.inventory import build_serial_interference_envelope
from lightcone_spec.experiments.planning import (
    BudgetMaterializationAuthorityBinding,
    BudgetPlan,
    BudgetRawJsonBinding,
    ExperimentBudget,
    _budget_activation_raw_sources,
    budget_inventory_identity_from_gpu_inventory,
)
from lightcone_spec.experiments.planning_artifacts import (
    budget_load_binding_from_dict,
    production_load_plan_from_dict,
)
from lightcone_spec.experiments.registry import (
    ExperimentCell,
    ExperimentReceipt,
    ExperimentRegistry,
    LockedOutput,
    WorkloadClass,
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.serving import PinnedBenchServingTransport
from lightcone_spec.experiments.stage_activation import (
    RegistryStageActivationArtifact,
)
from lightcone_spec.locking.models import ModelLock
from lightcone_spec.locking.prepared_models import (
    PreparedModelContentAuthorityBlocked,
    has_prepared_model_content_release_manifest_sha256,
)
from lightcone_spec.orchestration.executor import (
    PREPARED_MODEL_CONTENT_RELEASE_MANIFEST_PIN_UNAVAILABLE_REASON,
    TRAINABLE_PLAN_RAW_AUTHORITY_UNAVAILABLE_REASON,
    ArtifactBinding,
    IndustrialExecutionPlan,
    TrainablePlanExecutionBlockedError,
    build_industrial_execution_plan,
    execute_industrial_plan,
    industrial_execution_split_contract,
    industrial_run_id,
    launch_server_subprocess,
)
from lightcone_spec.orchestration.industrial import (
    _require_registered_e1_execution_recipe,
    render_assigned_industrial_cell_runtime_plan,
)
from lightcone_spec.orchestration.native_terminal import NativeTerminalProvider
from lightcone_spec.orchestration.runtime import (
    ServerLaunch,
    _execution_argv,
    _execution_role,
    _render_server,
)
from lightcone_spec.runtime.attestation import RELEASE_TRUSTED_ATTESTER_POLICY
from lightcone_spec.runtime.compile_cache import (
    CompileCacheLaunchPlan,
    preflight_compile_cache_launch,
)
from lightcone_spec.runtime.distributed import (
    RankTopologyReceipt,
    TopologyIdentity,
    TopologyReceiptSet,
)
from lightcone_spec.sglang_bridge.checkout import verify_patched_checkout
from lightcone_spec.sglang_bridge.config import sglang_adaptation_payload
from lightcone_spec.telemetry.writer import EvidenceWriterPolicy

_REGISTRY_GENERATOR = "lightcone_spec.experiments.registry.build_industrial_registry:v2"
_SHA256_LENGTH = 64
_BUNDLE_KIND = "industrial_assignment_execution_bundle"
_CONTEXT_KIND = "gpu_dispatch_execution_context"
_TOPOLOGY_KIND = "industrial_topology_receipt_set"
_LAUNCH_KIND = "industrial_server_launch"
_EXECUTION_POLICY_KIND = "industrial_assignment_execution_policy"
_PREPARED_MODELS_KIND = "industrial_prepared_models"
_DISPATCH_RECEIPT_ENVELOPE_KIND = "industrial_dispatch_schedule_receipt_envelope"
_DISPATCH_ATTEMPT_JOURNAL_KIND = "industrial_dispatch_attempt_journal"
_DISPATCH_ATTEMPT_EVENT_KIND = "industrial_dispatch_attempt_event"
TRUSTED_DISPATCH_ATTESTER_UNAVAILABLE_REASON = "trusted_hardware_attester_unavailable"
_ACTIVATION_MANIFEST_ROLES = frozenset(
    {
        "registry_stage_activation_manifest",
        "e1_activation_authority_manifest",
        "e2_activation_authority_manifest",
        "confirmation_auxiliary_activation_authority_manifest",
        "confirmation_pilot_activation_authority_manifest",
        "confirmation_final_activation_authority_manifest",
        "confirmation_stage_aggregate_authority_manifest",
    }
)


class ExecutionBundleBlockedError(RuntimeError):
    """The raw inputs are honest but this release cannot execute them."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = _strict_text("bundle BLOCKED reason", reason_code)
        super().__init__(f"industrial execution bundle is BLOCKED: {reason_code}")


@dataclass(frozen=True)
class AssignmentLaunchMaterializationPolicy:
    """Non-result inputs for the first-party server-launch renderer."""

    schema_version: int
    kind: Literal["industrial_server_launch_materialization_policy"]
    patched_sglang_checkout: str
    adaptation_reserve_mb: int
    mem_fraction_static: float
    host: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.kind != ("industrial_server_launch_materialization_policy")
        ):
            raise ValueError("server-launch materialization policy is unsupported")
        checkout = Path(self.patched_sglang_checkout)
        if (
            not checkout.is_absolute()
            or checkout.resolve() != checkout
            or not checkout.is_dir()
        ):
            raise ValueError(
                "patched SGLang checkout must be an existing resolved directory"
            )
        if (
            isinstance(self.adaptation_reserve_mb, bool)
            or not isinstance(self.adaptation_reserve_mb, int)
            or self.adaptation_reserve_mb < 0
        ):
            raise ValueError("adaptation reserve must be a non-negative integer")
        if (
            isinstance(self.mem_fraction_static, bool)
            or not isinstance(self.mem_fraction_static, (int, float))
            or not math.isfinite(float(self.mem_fraction_static))
            or not 0.0 < float(self.mem_fraction_static) < 1.0
        ):
            raise ValueError("static memory fraction must lie in (0, 1)")
        _strict_text("server launch host", self.host)
        if self.host not in {"127.0.0.1", "localhost"}:
            raise ValueError("server launch materialization requires a loopback host")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "patched_sglang_checkout": self.patched_sglang_checkout,
            "adaptation_reserve_mb": self.adaptation_reserve_mb,
            "mem_fraction_static": float(self.mem_fraction_static),
            "host": self.host,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "server-launch materialization policy",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "patched_sglang_checkout",
                    "adaptation_reserve_mb",
                    "mem_fraction_static",
                    "host",
                }
            ),
        )
        return cls(
            schema_version=_strict_int(
                "server-launch policy schema", row["schema_version"]
            ),
            kind=_strict_text("server-launch policy kind", row["kind"]),
            patched_sglang_checkout=_strict_text(
                "patched SGLang checkout", row["patched_sglang_checkout"]
            ),
            adaptation_reserve_mb=_strict_int(
                "adaptation reserve", row["adaptation_reserve_mb"], minimum=0
            ),
            mem_fraction_static=_strict_float(
                "static memory fraction", row["mem_fraction_static"]
            ),
            host=_strict_text("server launch host", row["host"]),
        )


def _absolute_lexical_path(path: str | Path) -> Path:
    """Make a path absolute without following its leaf symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def require_release_dispatch_execution_authority() -> None:
    """Fail before execution mutation when no release signer exists.

    Dispatch-group entrypoints invoke this before bundle reads; the
    plan-materialization seam invokes it after read-only replay and immediately
    before its first renderer write.
    """

    RELEASE_TRUSTED_ATTESTER_POLICY.validate()
    if not RELEASE_TRUSTED_ATTESTER_POLICY.release_ready:
        raise ExecutionBundleBlockedError(TRUSTED_DISPATCH_ATTESTER_UNAVAILABLE_REASON)


def preflight_fresh_assignment_trace(
    plan: IndustrialExecutionPlan,
    *,
    output_root: str | Path,
    run_nonce_sha256: str,
    prior_failed_attempt_authorized: bool = False,
) -> str:
    """Reject implicit per-plan resume before the executor creates its root."""

    if type(plan) is not IndustrialExecutionPlan:
        raise TypeError("fresh-trace preflight requires an exact execution plan")
    plan.validate()
    nonce = _require_sha256("fresh-trace run nonce", run_nonce_sha256)
    root = Path(output_root)
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("fresh-trace output root must be absolute and resolved")
    registered = Path(plan.runtime_plan.cell.resources.evidence_root).resolve()
    if root != registered:
        raise ValueError("fresh-trace root differs from the registry reservation")
    run_id = industrial_run_id(plan, nonce)
    if not root.exists():
        return run_id
    if root.is_symlink() or not root.is_dir():
        raise ExecutionBundleBlockedError("fresh_trace_output_root_invalid")
    try:
        prior = tuple(
            sorted(
                entry.name
                for entry in root.iterdir()
                if entry.name == run_id
                or entry.name.startswith(f"{run_id}.")
                or entry.name.startswith(f".{run_id}.")
            )
        )
    except OSError as error:
        raise ExecutionBundleBlockedError("fresh_trace_preflight_unreadable") from error
    if prior:
        if prior_failed_attempt_authorized:
            return run_id
        raise ExecutionBundleBlockedError(
            "dispatch_wave_terminal_recovery_authority_unavailable"
        )
    return run_id


def dispatch_receipt_sidecar_path(path: str | Path) -> Path:
    """Return the structured scheduler-sidecar path used by this CLI slice."""

    output = _absolute_lexical_path(path)
    return output.with_name(f"{output.name}.sidecar.json")


def preflight_dispatch_receipt_output(path: str | Path) -> tuple[Path, Path]:
    """Require an empty immutable receipt target before any assignment launch."""

    output = _absolute_lexical_path(path)
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve() != parent:
        raise ExecutionBundleBlockedError("dispatch_receipt_parent_unavailable")
    sidecar = dispatch_receipt_sidecar_path(output)
    if (
        output.exists()
        or output.is_symlink()
        or sidecar.exists()
        or sidecar.is_symlink()
    ):
        raise ExecutionBundleBlockedError("dispatch_receipt_output_already_exists")
    return output, sidecar


def _write_fsynced_temporary(parent: Path, *, name: str, body: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(raw_path)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_immutable_file(path: Path, body: bytes) -> None:
    if path.exists() or path.is_symlink():
        if _read_regular_file(path, label="existing dispatch receipt artifact") != body:
            raise FileExistsError(
                f"immutable dispatch receipt artifact differs: {path}"
            )
        return
    temporary = _write_fsynced_temporary(path.parent, name=path.name, body=body)
    try:
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if (
                _read_regular_file(path, label="raced dispatch receipt artifact")
                != body
            ):
                raise FileExistsError(
                    f"immutable dispatch receipt artifact differs: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class DispatchAttemptJournalBinding:
    """Path and immutable hash-chain prefix bound by one schedule envelope."""

    journal_path: str
    manifest_sha256: str
    head_event_sha256: str
    event_count: int

    def __post_init__(self) -> None:
        path = Path(self.journal_path)
        if not path.is_absolute() or path.resolve() != path:
            raise ValueError(
                "attempt journal binding path must be absolute and resolved"
            )
        _require_sha256("attempt journal manifest", self.manifest_sha256)
        _require_sha256("attempt journal head event", self.head_event_sha256)
        _strict_int("attempt journal event count", self.event_count, minimum=1)

    def to_dict(self) -> dict[str, object]:
        return {
            "journal_path": self.journal_path,
            "manifest_sha256": self.manifest_sha256,
            "head_event_sha256": self.head_event_sha256,
            "event_count": self.event_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> DispatchAttemptJournalBinding:
        row = _strict_object(
            "dispatch attempt journal binding",
            value,
            frozenset(
                {
                    "journal_path",
                    "manifest_sha256",
                    "head_event_sha256",
                    "event_count",
                }
            ),
        )
        return cls(
            journal_path=_strict_text("journal_path", row["journal_path"]),
            manifest_sha256=_require_sha256("manifest_sha256", row["manifest_sha256"]),
            head_event_sha256=_require_sha256(
                "head_event_sha256", row["head_event_sha256"]
            ),
            event_count=_strict_int("event_count", row["event_count"], minimum=1),
        )


def publish_dispatch_schedule_receipt(
    path: str | Path,
    receipt: DispatchScheduleReceipt,
    *,
    attempt_journal: DispatchAttemptJournalBinding | None = None,
) -> tuple[Path, Path]:
    """Atomically publish one authoritative receipt envelope.

    The single immutable envelope contains both the receipt and its structured
    sidecar.  The adjacent sidecar is a derived convenience artifact only, so
    a crash after the envelope link cannot brick or erase resume authority.
    Retrying publication merely fills that identical optional derivative.
    """

    if type(receipt) is not DispatchScheduleReceipt:
        raise TypeError("dispatch receipt publisher requires an exact receipt")
    output = _absolute_lexical_path(path)
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve() != parent:
        raise ValueError(
            "dispatch receipt parent must be an existing resolved directory"
        )
    sidecar = dispatch_receipt_sidecar_path(output)
    envelope: dict[str, object] = {
        "schema_version": 1 if attempt_journal is None else 2,
        "kind": _DISPATCH_RECEIPT_ENVELOPE_KIND,
        "receipt": receipt.to_dict(),
        "sidecar": receipt.sidecar().to_dict(),
    }
    if attempt_journal is not None:
        if type(attempt_journal) is not DispatchAttemptJournalBinding:
            raise TypeError("dispatch receipt requires an exact journal binding")
        envelope["attempt_journal"] = attempt_journal.to_dict()
    envelope_body = (
        json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    sidecar_body = (
        json.dumps(
            receipt.sidecar().to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _publish_immutable_file(output, envelope_body)
    _fsync_directory(parent)
    _publish_immutable_file(sidecar, sidecar_body)
    _fsync_directory(parent)
    return output, sidecar


def _load_dispatch_schedule_envelope(
    path: str | Path,
    *,
    plan: GpuDispatchPlan,
    execution_context: GpuDispatchExecutionContext,
) -> tuple[DispatchScheduleReceipt, DispatchAttemptJournalBinding | None]:
    """Reopen one envelope; its receipt is evidence, never resume authority."""

    output = _absolute_lexical_path(path)
    decoded = _decode_json(
        _read_regular_file(output, label="dispatch schedule receipt envelope"),
        label="dispatch schedule receipt envelope",
    )
    if type(decoded) is not dict:
        raise TypeError("dispatch schedule receipt envelope must be a JSON object")
    schema_version = decoded.get("schema_version")
    if schema_version == 1:
        fields = frozenset({"schema_version", "kind", "receipt", "sidecar"})
    elif schema_version == 2:
        fields = frozenset(
            {"schema_version", "kind", "receipt", "sidecar", "attempt_journal"}
        )
    else:
        raise ValueError("dispatch schedule receipt envelope schema is unsupported")
    envelope = _strict_object("dispatch schedule receipt envelope", decoded, fields)
    if envelope["kind"] != _DISPATCH_RECEIPT_ENVELOPE_KIND:
        raise ValueError("dispatch schedule receipt envelope kind is unsupported")
    receipt = DispatchScheduleReceipt.from_dict(
        envelope["receipt"],
        plan=plan,
        execution_context=execution_context,
        sidecar=envelope["sidecar"],
    )
    journal = (
        None
        if schema_version == 1
        else DispatchAttemptJournalBinding.from_dict(envelope["attempt_journal"])
    )
    return receipt, journal


def load_dispatch_schedule_receipt(
    path: str | Path,
    *,
    plan: GpuDispatchPlan,
    execution_context: GpuDispatchExecutionContext,
) -> DispatchScheduleReceipt:
    """Reopen the single authoritative envelope against the exact scheduler."""

    receipt, _ = _load_dispatch_schedule_envelope(
        path,
        plan=plan,
        execution_context=execution_context,
    )
    return receipt


def dispatch_attempt_journal_path(receipt_output: str | Path) -> Path:
    """Return the deterministic journal root for a first-wave receipt target."""

    output = _absolute_lexical_path(receipt_output)
    return output.with_name(f"{output.name}.attempt-journal")


def _require_private_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ExecutionBundleBlockedError(f"{label}_unavailable") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or path.resolve() != path
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise ExecutionBundleBlockedError(f"{label}_invalid")


def _write_exclusive_fsynced(path: Path, body: bytes) -> None:
    """Publish one append-only file without a rename or overwrite window."""

    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory_descriptor = os.open(path.parent, directory_flags)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        directory_opened = os.fstat(directory_descriptor)
        directory_current = path.parent.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(directory_opened.st_mode)
            or directory_opened.st_dev != directory_current.st_dev
            or directory_opened.st_ino != directory_current.st_ino
        ):
            raise RuntimeError("append-only journal parent changed before write")
        descriptor = os.open(
            path.name,
            flags,
            0o600,
            dir_fd=directory_descriptor,
        )
        try:
            view = memoryview(body)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:  # pragma: no cover - defensive OS boundary
                    raise OSError("append-only journal write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


@dataclass(frozen=True)
class DispatchAttemptJournalSnapshot:
    """Fresh replay of every raw journal event against one frozen plan."""

    binding: DispatchAttemptJournalBinding | None
    receipt: DispatchScheduleReceipt | None
    replay_authority: DispatchAttemptJournalReplay | None
    terminal_bindings: tuple[AssignmentTerminalBinding, ...]
    latest_assignment_receipts: tuple[AssignmentExecutionReceipt, ...]
    incomplete_intent_sha256s: tuple[str, ...]
    event_sha256s: tuple[str, ...]

    def require_complete_cost_authority(self) -> None:
        if self.incomplete_intent_sha256s:
            raise ExecutionBundleBlockedError(
                "dispatch_attempt_intent_without_finish_cost_unresolved"
            )


class DispatchAttemptJournal:
    """Release-owned path-bound WAVE/INTENT/FINISH append-only hash chain.

    The journal is an exact recovery and accounting mechanism, not an external
    signature.  Formal success still requires reopening each trusted native
    terminal authority.  A durable INTENT without FINISH is deliberately not
    guessed from wall time and blocks all further dispatch from this journal.
    """

    _MANIFEST_FIELDS_V1 = frozenset(
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "journal_path",
            "events_path",
            "plan_sha256",
            "execution_context_sha256",
            "inventory_sha256",
            "fixed_instance_gpu_count",
        }
    )
    _MANIFEST_FIELDS_V2 = _MANIFEST_FIELDS_V1 | frozenset(
        {"execution_bundle_manifest_sha256"}
    )
    # Retain the v1 name for diagnostic callers/tests that deliberately build
    # an unbound journal. Formal dispatch always uses the v2 authority below.
    _MANIFEST_FIELDS = _MANIFEST_FIELDS_V1
    _EVENT_FIELDS = frozenset(
        {
            "schema_version",
            "kind",
            "manifest_sha256",
            "sequence",
            "previous_event_sha256",
            "event_type",
            "plan_sha256",
            "execution_context_sha256",
            "inventory_sha256",
            "wave_index",
            "wave_sha256",
            "payload",
        }
    )
    _ATTEMPT_FIELDS = frozenset(
        {
            "assignment_sha256",
            "budget_sha256",
            "retry_allowance",
            "gpu_uuids",
            "fixed_instance_gpu_count",
            "attempt",
            "prior_attempt_receipt_sha256",
        }
    )
    _PROTOCOL_SHA256 = content_sha256(
        "industrial_dispatch_attempt_journal.wave_intent_finish_hash_chain.v1"
    )
    _PUBLICATION_PROTOCOL_SHA256 = content_sha256(
        "industrial_dispatch_attempt_journal.wave_intent_finish_hash_chain."
        "execution_bundle_manifest_bound.v2"
    )

    def __init__(
        self,
        *,
        root: Path,
        plan: GpuDispatchPlan,
        execution_context: GpuDispatchExecutionContext,
        manifest_sha256: str,
        event_names: tuple[str, ...],
        event_sha256s: tuple[str, ...],
    ) -> None:
        self.root = root
        self.events_path = root / "events"
        self.plan = plan
        self.execution_context = execution_context
        self.manifest_sha256 = manifest_sha256
        self._event_names = event_names
        self._event_sha256s = event_sha256s
        self._append_lock = asyncio.Lock()
        self._active_wave_event_sha256: str | None = None
        self._active_wave_payload: dict[str, object] | None = None
        self._open_tokens: dict[str, dict[str, object]] = {}
        self._preissued_assignment_tokens: dict[str, DispatchAttemptJournalToken] = {}

    @classmethod
    def open_or_create(
        cls,
        root: str | Path,
        *,
        plan: GpuDispatchPlan,
        execution_context: GpuDispatchExecutionContext,
        expected_prefix: DispatchAttemptJournalBinding | None = None,
        execution_bundle_manifest_sha256: str | None = None,
    ) -> DispatchAttemptJournal:
        candidate = Path(os.path.abspath(os.fspath(root)))
        parent = candidate.parent
        if parent.is_symlink() or not parent.is_dir() or parent.resolve() != parent:
            raise ExecutionBundleBlockedError("dispatch_attempt_journal_parent_invalid")
        if execution_bundle_manifest_sha256 is not None:
            execution_bundle_manifest_sha256 = _require_sha256(
                "execution-bundle publication manifest",
                execution_bundle_manifest_sha256,
            )
        expected_manifest: dict[str, object] = {
            "schema_version": (1 if execution_bundle_manifest_sha256 is None else 2),
            "kind": _DISPATCH_ATTEMPT_JOURNAL_KIND,
            "protocol_sha256": (
                cls._PROTOCOL_SHA256
                if execution_bundle_manifest_sha256 is None
                else cls._PUBLICATION_PROTOCOL_SHA256
            ),
            "journal_path": str(candidate),
            "events_path": str(candidate / "events"),
            "plan_sha256": plan.sha256,
            "execution_context_sha256": execution_context.sha256,
            "inventory_sha256": execution_context.inventory.sha256,
            "fixed_instance_gpu_count": len(execution_context.inventory.devices),
        }
        if execution_bundle_manifest_sha256 is not None:
            expected_manifest["execution_bundle_manifest_sha256"] = (
                execution_bundle_manifest_sha256
            )
        expected_manifest_body = _canonical_bytes(expected_manifest) + b"\n"
        if not candidate.exists() and not candidate.is_symlink():
            try:
                os.mkdir(candidate, 0o700)
            except FileExistsError:
                pass
            _fsync_directory(parent)
        _require_private_directory(candidate, label="dispatch_attempt_journal")
        # A previous process may have died after mkdir but before syncing the
        # parent entry.  Repeating the directory sync is harmless and closes
        # that permitted initialization prefix without deleting it.
        _fsync_directory(parent)
        root_entries = cls._stable_directory_names(candidate)
        unknown_entries = set(root_entries) - {"events", "manifest.json"}
        if unknown_entries:
            raise ExecutionBundleBlockedError(
                "dispatch_attempt_journal_contains_unknown_entry"
            )
        if "events" not in root_entries:
            if root_entries:
                raise ExecutionBundleBlockedError(
                    "dispatch_attempt_journal_initialization_prefix_invalid"
                )
            try:
                os.mkdir(candidate / "events", 0o700)
            except FileExistsError:
                pass
            _fsync_directory(candidate)
            root_entries = cls._stable_directory_names(candidate)
            if root_entries != ("events",):
                raise ExecutionBundleBlockedError(
                    "dispatch_attempt_journal_initialization_prefix_invalid"
                )
        _require_private_directory(
            candidate / "events", label="dispatch_attempt_journal_events"
        )
        # Likewise make an already-present empty events directory durable on
        # retry before publishing or completing the manifest.
        _fsync_directory(candidate)
        names = cls._event_file_names(candidate / "events")
        if "manifest.json" not in root_entries:
            if names:
                raise ExecutionBundleBlockedError(
                    "dispatch_attempt_journal_events_precede_manifest"
                )
            try:
                _write_exclusive_fsynced(
                    candidate / "manifest.json", expected_manifest_body
                )
            except FileExistsError:
                pass
            root_entries = cls._stable_directory_names(candidate)
            if root_entries != ("events", "manifest.json"):
                raise ExecutionBundleBlockedError(
                    "dispatch_attempt_journal_initialization_prefix_invalid"
                )
        manifest_body = cls._complete_manifest_prefix(
            candidate / "manifest.json",
            expected=expected_manifest_body,
            events_are_empty=not names,
        )
        manifest_sha256 = hashlib.sha256(manifest_body).hexdigest()
        manifest = _strict_object(
            "dispatch attempt journal manifest",
            _decode_json(manifest_body, label="dispatch attempt journal manifest"),
            (
                cls._MANIFEST_FIELDS_V1
                if execution_bundle_manifest_sha256 is None
                else cls._MANIFEST_FIELDS_V2
            ),
        )
        if manifest != expected_manifest:
            raise ExecutionBundleBlockedError(
                "dispatch_attempt_journal_manifest_identity_mismatch"
            )
        journal = cls(
            root=candidate,
            plan=plan,
            execution_context=execution_context,
            manifest_sha256=manifest_sha256,
            event_names=names,
            event_sha256s=(),
        )
        snapshot = journal.replay()
        journal._event_sha256s = snapshot.event_sha256s
        if expected_prefix is not None:
            journal._validate_expected_prefix(expected_prefix, snapshot=snapshot)
        return journal

    @staticmethod
    def _stable_directory_names(path: Path) -> tuple[str, ...]:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise ExecutionBundleBlockedError(
                "dispatch_attempt_journal_directory_invalid"
            ) from error
        try:
            opened = os.fstat(descriptor)
            current = path.stat(follow_symlinks=False)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or opened.st_dev != current.st_dev
                or opened.st_ino != current.st_ino
            ):
                raise ExecutionBundleBlockedError(
                    "dispatch_attempt_journal_directory_changed"
                )
            before = tuple(sorted(os.listdir(descriptor)))
            after = tuple(sorted(os.listdir(descriptor)))
            if before != after:
                raise ExecutionBundleBlockedError(
                    "dispatch_attempt_journal_changed_during_enumeration"
                )
            return before
        finally:
            os.close(descriptor)

    @staticmethod
    def _complete_manifest_prefix(
        path: Path,
        *,
        expected: bytes,
        events_are_empty: bool,
    ) -> bytes:
        actual = _read_regular_file(path, label="dispatch attempt journal manifest")
        metadata = path.stat(follow_symlinks=False)
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise ExecutionBundleBlockedError(
                "dispatch_attempt_journal_manifest_identity_mismatch"
            )
        if actual == expected:
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                if (
                    opened.st_dev != metadata.st_dev
                    or opened.st_ino != metadata.st_ino
                    or opened.st_size != len(actual)
                ):
                    raise ExecutionBundleBlockedError(
                        "dispatch_attempt_journal_manifest_changed"
                    )
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _fsync_directory(path.parent)
            return actual
        if not events_are_empty or not expected.startswith(actual):
            raise ExecutionBundleBlockedError(
                "dispatch_attempt_journal_manifest_identity_mismatch"
            )
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        directory_descriptor = os.open(path.parent, directory_flags)
        flags = os.O_WRONLY | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            read_descriptor = os.open(
                path.name,
                os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0),
                dir_fd=directory_descriptor,
            )
            try:
                opened = os.fstat(read_descriptor)
                if (
                    opened.st_dev != metadata.st_dev
                    or opened.st_ino != metadata.st_ino
                    or opened.st_size != len(actual)
                ):
                    raise ExecutionBundleBlockedError(
                        "dispatch_attempt_journal_manifest_changed"
                    )
                os.fchmod(read_descriptor, 0o600)
            finally:
                os.close(read_descriptor)
            descriptor = os.open(path.name, flags, dir_fd=directory_descriptor)
            try:
                opened = os.fstat(descriptor)
                if (
                    opened.st_dev != metadata.st_dev
                    or opened.st_ino != metadata.st_ino
                    or opened.st_size != len(actual)
                ):
                    raise ExecutionBundleBlockedError(
                        "dispatch_attempt_journal_manifest_changed"
                    )
                remainder = memoryview(expected[len(actual) :])
                while remainder:
                    written = os.write(descriptor, remainder)
                    if written <= 0:  # pragma: no cover - defensive OS boundary
                        raise OSError("manifest prefix completion made no progress")
                    remainder = remainder[written:]
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        completed = _read_regular_file(
            path, label="completed dispatch attempt journal manifest"
        )
        if completed != expected:
            raise ExecutionBundleBlockedError(
                "dispatch_attempt_journal_manifest_identity_mismatch"
            )
        return completed

    @staticmethod
    def _event_file_names(events_path: Path) -> tuple[str, ...]:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(events_path, flags)
        except OSError as error:
            raise ExecutionBundleBlockedError(
                "dispatch_attempt_journal_events_invalid"
            ) from error
        try:
            opened = os.fstat(descriptor)
            current = events_path.stat(follow_symlinks=False)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or opened.st_dev != current.st_dev
                or opened.st_ino != current.st_ino
            ):
                raise ExecutionBundleBlockedError(
                    "dispatch_attempt_journal_events_changed"
                )
            before = tuple(sorted(os.listdir(descriptor)))
            for name in before:
                DispatchAttemptJournal._parse_event_file_name(name)
            after = tuple(sorted(os.listdir(descriptor)))
            if before != after:
                raise ExecutionBundleBlockedError(
                    "dispatch_attempt_journal_changed_during_enumeration"
                )
            return before
        finally:
            os.close(descriptor)

    @staticmethod
    def _parse_event_file_name(name: str) -> tuple[int, str]:
        parts = name.split(".")
        if (
            len(parts) != 3
            or len(parts[0]) != 12
            or not parts[0].isdigit()
            or not _is_sha256(parts[1])
            or parts[2] != "json"
        ):
            raise ExecutionBundleBlockedError(
                "dispatch_attempt_journal_contains_unknown_entry"
            )
        return int(parts[0]), parts[1]

    def _validate_expected_prefix(
        self,
        expected: DispatchAttemptJournalBinding,
        *,
        snapshot: DispatchAttemptJournalSnapshot,
    ) -> None:
        if (
            expected.journal_path != str(self.root)
            or expected.manifest_sha256 != self.manifest_sha256
            or expected.event_count > len(snapshot.event_sha256s)
            or snapshot.event_sha256s[expected.event_count - 1]
            != expected.head_event_sha256
        ):
            raise ExecutionBundleBlockedError(
                "dispatch_attempt_journal_receipt_prefix_mismatch"
            )

    def _attempt_identity(
        self,
        *,
        assignment: GpuAssignment,
        budget: ExperimentBudget,
        attempt: int,
        prior_attempt_receipt: AssignmentExecutionReceipt | None,
    ) -> dict[str, object]:
        return {
            "assignment_sha256": assignment.assignment_id,
            "budget_sha256": budget.sha256,
            "retry_allowance": budget.retry_allowance,
            "gpu_uuids": list(assignment.gpu_uuids),
            "fixed_instance_gpu_count": len(self.execution_context.inventory.devices),
            "attempt": attempt,
            "prior_attempt_receipt_sha256": (
                None if prior_attempt_receipt is None else prior_attempt_receipt.sha256
            ),
        }

    def _event_common(
        self,
        *,
        wave: GpuDispatchWave,
        event_type: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": _DISPATCH_ATTEMPT_EVENT_KIND,
            "manifest_sha256": self.manifest_sha256,
            "sequence": len(self._event_names),
            "previous_event_sha256": (
                None if not self._event_sha256s else self._event_sha256s[-1]
            ),
            "event_type": event_type,
            "plan_sha256": self.plan.sha256,
            "execution_context_sha256": self.execution_context.sha256,
            "inventory_sha256": self.execution_context.inventory.sha256,
            "wave_index": wave.wave_index,
            "wave_sha256": wave.sha256,
            "payload": payload,
        }

    def _verify_cached_head(self) -> None:
        names = self._event_file_names(self.events_path)
        if names != self._event_names:
            raise ExecutionBundleBlockedError(
                "dispatch_attempt_journal_changed_outside_writer"
            )
        if names:
            body = _read_regular_file(
                self.events_path / names[-1], label="dispatch attempt journal head"
            )
            if hashlib.sha256(body).hexdigest() != self._event_sha256s[-1]:
                raise ExecutionBundleBlockedError(
                    "dispatch_attempt_journal_head_changed"
                )

    def _append_event(self, event: dict[str, object]) -> str:
        self._verify_cached_head()
        body = _canonical_bytes(event) + b"\n"
        digest = hashlib.sha256(body).hexdigest()
        name = f"{len(self._event_names):012d}.{digest}.json"
        _write_exclusive_fsynced(self.events_path / name, body)
        self._event_names += (name,)
        self._event_sha256s += (digest,)
        return digest

    def _assignment_by_id(self, wave: GpuDispatchWave) -> dict[str, GpuAssignment]:
        return {assignment.assignment_id: assignment for assignment in wave.assignments}

    def _schedule_from_latest(
        self,
        latest: dict[str, AssignmentExecutionReceipt],
        *,
        prior_schedule_receipt_sha256: str | None,
    ) -> DispatchScheduleReceipt | None:
        wave_receipts = []
        for wave in self.plan.waves:
            if any(
                assignment.assignment_id not in latest
                for assignment in wave.assignments
            ):
                break
            wave_receipt = _make_wave_execution_receipt(
                plan=self.plan,
                wave=wave,
                assignment_receipts=tuple(
                    latest[assignment.assignment_id] for assignment in wave.assignments
                ),
                execution_context=self.execution_context,
            )
            wave_receipts.append(wave_receipt)
            if not wave_receipt.succeeded:
                break
        if not wave_receipts:
            return None
        if not wave_receipts[-1].succeeded:
            phase = DispatchExecutionPhase.FAILED
        elif len(wave_receipts) == len(self.plan.waves):
            phase = DispatchExecutionPhase.COMPLETE
        else:
            phase = DispatchExecutionPhase.RUNNING
        return _make_schedule_receipt(
            plan=self.plan,
            phase=phase,
            wave_receipts=tuple(wave_receipts),
            execution_context=self.execution_context,
            prior_schedule_receipt_sha256=prior_schedule_receipt_sha256,
        )

    def _expected_pending(
        self,
        latest: dict[str, AssignmentExecutionReceipt],
    ) -> tuple[GpuDispatchWave, tuple[dict[str, object], ...]]:
        target: GpuDispatchWave | None = None
        for wave in self.plan.waves:
            rows = tuple(latest.get(row.assignment_id) for row in wave.assignments)
            if all(
                row is not None and row.status is AssignmentExecutionStatus.SUCCEEDED
                for row in rows
            ):
                continue
            target = wave
            break
        if target is None:
            raise ValueError("complete dispatch journal cannot start another wave")
        identities = []
        for assignment in target.assignments:
            prior = latest.get(assignment.assignment_id)
            if (
                prior is not None
                and prior.status is AssignmentExecutionStatus.SUCCEEDED
            ):
                continue
            budget = self.execution_context.budgets_by_cell_id[
                assignment.work_item.item_id
            ]
            attempt = 1 if prior is None else prior.attempt + 1
            if attempt > budget.retry_allowance + 1:
                raise ValueError(
                    "dispatch retry would exceed the ExperimentBudget allowance"
                )
            identities.append(
                self._attempt_identity(
                    assignment=assignment,
                    budget=budget,
                    attempt=attempt,
                    prior_attempt_receipt=prior,
                )
            )
        return target, tuple(identities)

    def replay(
        self, *, event_count: int | None = None
    ) -> DispatchAttemptJournalSnapshot:
        all_names = self._event_file_names(self.events_path)
        if event_count is None:
            names = all_names
        else:
            _strict_int("attempt journal replay event count", event_count, minimum=1)
            if event_count > len(all_names):
                raise ValueError("attempt journal prefix exceeds available events")
            names = all_names[:event_count]
        latest: dict[str, AssignmentExecutionReceipt] = {}
        incomplete: dict[str, dict[str, object]] = {}
        wave_events: dict[str, dict[str, object]] = {}
        event_sha256s: list[str] = []
        previous: str | None = None
        last_finish_prior_schedule_sha256: str | None = None
        for expected_sequence, name in enumerate(names):
            sequence, declared_digest = self._parse_event_file_name(name)
            if sequence != expected_sequence:
                raise ExecutionBundleBlockedError(
                    "dispatch_attempt_journal_sequence_gap"
                )
            body = _read_regular_file(
                self.events_path / name,
                label="dispatch attempt journal event",
            )
            actual_digest = hashlib.sha256(body).hexdigest()
            if actual_digest != declared_digest:
                raise ExecutionBundleBlockedError(
                    "dispatch_attempt_journal_event_digest_mismatch"
                )
            event = _strict_object(
                "dispatch attempt journal event",
                _decode_json(body, label="dispatch attempt journal event"),
                self._EVENT_FIELDS,
            )
            if (
                event["schema_version"] != 1
                or event["kind"] != _DISPATCH_ATTEMPT_EVENT_KIND
                or event["manifest_sha256"] != self.manifest_sha256
                or event["sequence"] != sequence
                or event["previous_event_sha256"] != previous
                or event["plan_sha256"] != self.plan.sha256
                or event["execution_context_sha256"] != self.execution_context.sha256
                or event["inventory_sha256"] != self.execution_context.inventory.sha256
            ):
                raise ExecutionBundleBlockedError(
                    "dispatch_attempt_journal_event_identity_mismatch"
                )
            wave_index = _strict_int(
                "attempt event wave index", event["wave_index"], minimum=0
            )
            if wave_index >= len(self.plan.waves):
                raise ValueError("attempt journal wave index exceeds the plan")
            wave = self.plan.waves[wave_index]
            if event["wave_sha256"] != wave.sha256:
                raise ValueError("attempt journal event belongs to another wave")
            event_type = _strict_text("attempt event type", event["event_type"])
            payload = event["payload"]
            if event_type == "WAVE":
                if incomplete:
                    raise ExecutionBundleBlockedError(
                        "dispatch_attempt_intent_without_finish_cost_unresolved"
                    )
                wave_payload = _strict_object(
                    "dispatch wave-attempt event payload",
                    payload,
                    frozenset(
                        {
                            "prior_schedule_receipt_sha256",
                            "pending_attempts",
                        }
                    ),
                )
                expected_receipt = self._schedule_from_latest(
                    latest,
                    prior_schedule_receipt_sha256=(last_finish_prior_schedule_sha256),
                )
                expected_prior = (
                    None if expected_receipt is None else expected_receipt.sha256
                )
                if wave_payload["prior_schedule_receipt_sha256"] != expected_prior:
                    raise ValueError(
                        "dispatch journal wave prior schedule identity mismatch"
                    )
                expected_wave, expected_pending = self._expected_pending(latest)
                declared_pending = tuple(
                    _strict_object(
                        "dispatch pending attempt", row, self._ATTEMPT_FIELDS
                    )
                    for row in _strict_list(
                        "dispatch pending attempts", wave_payload["pending_attempts"]
                    )
                )
                if wave != expected_wave or declared_pending != expected_pending:
                    raise ValueError(
                        "dispatch journal wave pending attempts differ from replay"
                    )
                wave_events[actual_digest] = {
                    "wave_index": wave.wave_index,
                    "prior_schedule_receipt_sha256": expected_prior,
                    "pending_attempts": declared_pending,
                }
            elif event_type in {"INTENT", "FINISH"}:
                fields = set(self._ATTEMPT_FIELDS)
                fields.update(
                    {
                        "wave_event_sha256",
                        "started_monotonic_ns",
                        "finished_monotonic_ns",
                        "intent_event_sha256",
                        "assignment_receipt",
                    }
                )
                attempt_payload = _strict_object(
                    f"dispatch attempt {event_type.lower()} payload",
                    payload,
                    frozenset(fields),
                )
                identity = {key: attempt_payload[key] for key in self._ATTEMPT_FIELDS}
                wave_event_sha256 = _require_sha256(
                    "attempt wave event", attempt_payload["wave_event_sha256"]
                )
                wave_event = wave_events.get(wave_event_sha256)
                if wave_event is None or wave_event["wave_index"] != wave.wave_index:
                    raise ValueError("attempt event lacks its exact wave intent")
                if identity not in wave_event["pending_attempts"]:
                    raise ValueError("attempt event was not pending in its wave intent")
                started_ns = _strict_int(
                    "attempt started monotonic time",
                    attempt_payload["started_monotonic_ns"],
                    minimum=0,
                )
                if event_type == "INTENT":
                    if (
                        attempt_payload["finished_monotonic_ns"] is not None
                        or attempt_payload["intent_event_sha256"] is not None
                        or attempt_payload["assignment_receipt"] is not None
                        or any(
                            row["assignment_sha256"] == identity["assignment_sha256"]
                            and row["wave_event_sha256"] == wave_event_sha256
                            for row in incomplete.values()
                        )
                    ):
                        raise ValueError("dispatch attempt intent is not canonical")
                    incomplete[actual_digest] = {
                        **identity,
                        "wave_event_sha256": wave_event_sha256,
                        "started_monotonic_ns": started_ns,
                    }
                else:
                    intent_sha256 = _require_sha256(
                        "attempt intent event",
                        attempt_payload["intent_event_sha256"],
                    )
                    intent = incomplete.pop(intent_sha256, None)
                    finished_ns = _strict_int(
                        "attempt finished monotonic time",
                        attempt_payload["finished_monotonic_ns"],
                        minimum=started_ns,
                    )
                    if intent is None or intent != {
                        **identity,
                        "wave_event_sha256": wave_event_sha256,
                        "started_monotonic_ns": started_ns,
                    }:
                        raise ValueError(
                            "dispatch attempt finish lacks its exact intent"
                        )
                    receipt = AssignmentExecutionReceipt.from_dict(
                        attempt_payload["assignment_receipt"]
                    )
                    assignment_id = _strict_text(
                        "attempt assignment", identity["assignment_sha256"]
                    )
                    assignment = self._assignment_by_id(wave).get(assignment_id)
                    if assignment is None:
                        raise ValueError("attempt assignment is absent from its wave")
                    budget = self.execution_context.budgets_by_cell_id[
                        assignment.work_item.item_id
                    ]
                    prior = latest.get(assignment_id)
                    expected_identity = self._attempt_identity(
                        assignment=assignment,
                        budget=budget,
                        attempt=receipt.attempt,
                        prior_attempt_receipt=prior,
                    )
                    if identity != expected_identity:
                        raise ValueError("attempt finish identity differs from replay")
                    if (
                        receipt.plan_sha256 != self.plan.sha256
                        or receipt.wave_sha256 != wave.sha256
                        or receipt.assignment_sha256 != assignment_id
                        or receipt.budget_sha256 != budget.sha256
                        or receipt.attempt_intervals_monotonic_ns[-1]
                        != (started_ns, finished_ns)
                    ):
                        raise ValueError(
                            "attempt finish receipt differs from its raw interval"
                        )
                    if receipt.status is AssignmentExecutionStatus.SUCCEEDED:
                        binding = receipt.terminal_binding
                        if (
                            binding is None
                            or binding.cell_id != assignment.work_item.item_id
                            or binding.inventory_sha256
                            != self.execution_context.inventory.sha256
                            or binding.physical_gpu_uuids != assignment.gpu_uuids
                            or binding.dispatch_plan_sha256 != self.plan.sha256
                        ):
                            raise ValueError(
                                "attempt success terminal binding differs from "
                                "the frozen assignment"
                            )
                    latest[assignment_id] = receipt
                    last_finish_prior_schedule_sha256 = wave_event[
                        "prior_schedule_receipt_sha256"
                    ]
            else:
                raise ValueError("dispatch attempt journal event type is unsupported")
            event_sha256s.append(actual_digest)
            previous = actual_digest
        final_receipt = self._schedule_from_latest(
            latest,
            prior_schedule_receipt_sha256=last_finish_prior_schedule_sha256,
        )
        binding = (
            None
            if not event_sha256s
            else DispatchAttemptJournalBinding(
                journal_path=str(self.root),
                manifest_sha256=self.manifest_sha256,
                head_event_sha256=event_sha256s[-1],
                event_count=len(event_sha256s),
            )
        )
        replay_authority = (
            None
            if binding is None or final_receipt is None
            else DispatchAttemptJournalReplay(
                journal_path=binding.journal_path,
                manifest_sha256=binding.manifest_sha256,
                head_event_sha256=binding.head_event_sha256,
                event_count=binding.event_count,
                schedule_receipt_sha256=final_receipt.sha256,
            )
        )
        terminal_bindings = tuple(
            receipt.terminal_binding
            for receipt in latest.values()
            if receipt.status is AssignmentExecutionStatus.SUCCEEDED
            and receipt.terminal_binding is not None
        )
        return DispatchAttemptJournalSnapshot(
            binding=binding,
            receipt=final_receipt,
            replay_authority=replay_authority,
            terminal_bindings=terminal_bindings,
            latest_assignment_receipts=tuple(latest[key] for key in sorted(latest)),
            incomplete_intent_sha256s=tuple(sorted(incomplete)),
            event_sha256s=tuple(event_sha256s),
        )

    async def begin_wave_attempts(
        self,
        *,
        plan: GpuDispatchPlan,
        wave: GpuDispatchWave,
        attempts: tuple[
            tuple[
                GpuAssignment,
                int,
                ExperimentBudget,
                AssignmentExecutionReceipt | None,
            ],
            ...,
        ],
        prior_schedule_receipt_sha256: str | None,
    ) -> None:
        async with self._append_lock:
            if plan != self.plan:
                raise ValueError("attempt journal received another dispatch plan")
            snapshot = self.replay()
            snapshot.require_complete_cost_authority()
            latest = {
                receipt.assignment_sha256: receipt
                for receipt in snapshot.latest_assignment_receipts
            }
            expected_wave, expected_pending = self._expected_pending(latest)
            actual_pending = tuple(
                self._attempt_identity(
                    assignment=assignment,
                    budget=budget,
                    attempt=attempt,
                    prior_attempt_receipt=prior,
                )
                for assignment, attempt, budget, prior in attempts
            )
            expected_prior = (
                None if snapshot.receipt is None else snapshot.receipt.sha256
            )
            if (
                wave != expected_wave
                or actual_pending != expected_pending
                or prior_schedule_receipt_sha256 != expected_prior
            ):
                raise ValueError(
                    "attempt journal wave launch differs from raw replay state"
                )
            payload: dict[str, object] = {
                "prior_schedule_receipt_sha256": expected_prior,
                "pending_attempts": list(actual_pending),
            }
            event = self._event_common(
                wave=wave,
                event_type="WAVE",
                payload=payload,
            )
            self._active_wave_event_sha256 = self._append_event(event)
            self._active_wave_payload = payload
            for identity in actual_pending:
                started_ns = time.monotonic_ns()
                intent_payload = {
                    **identity,
                    "wave_event_sha256": self._active_wave_event_sha256,
                    "started_monotonic_ns": started_ns,
                    "finished_monotonic_ns": None,
                    "intent_event_sha256": None,
                    "assignment_receipt": None,
                }
                intent_event = self._event_common(
                    wave=wave,
                    event_type="INTENT",
                    payload=intent_payload,
                )
                digest = self._append_event(intent_event)
                token = DispatchAttemptJournalToken(started_ns, digest)
                assignment_sha256 = _strict_text(
                    "pending assignment", identity["assignment_sha256"]
                )
                self._open_tokens[digest] = intent_payload
                self._preissued_assignment_tokens[assignment_sha256] = token

    async def begin_attempt(
        self,
        *,
        plan: GpuDispatchPlan,
        wave: GpuDispatchWave,
        assignment: GpuAssignment,
        attempt: int,
        budget: ExperimentBudget,
        fixed_instance_gpu_count: int,
        prior_attempt_receipt: AssignmentExecutionReceipt | None,
        prior_schedule_receipt_sha256: str | None,
    ) -> DispatchAttemptJournalToken:
        async with self._append_lock:
            if (
                plan != self.plan
                or self._active_wave_event_sha256 is None
                or self._active_wave_payload is None
                or fixed_instance_gpu_count
                != len(self.execution_context.inventory.devices)
                or prior_schedule_receipt_sha256
                != self._active_wave_payload["prior_schedule_receipt_sha256"]
            ):
                raise ValueError("attempt intent differs from its durable wave intent")
            identity = self._attempt_identity(
                assignment=assignment,
                budget=budget,
                attempt=attempt,
                prior_attempt_receipt=prior_attempt_receipt,
            )
            if identity not in self._active_wave_payload["pending_attempts"]:
                raise ValueError("attempt was not declared by the durable wave intent")
            token = self._preissued_assignment_tokens.pop(
                assignment.assignment_id, None
            )
            if token is None:
                raise ValueError("attempt intent was not durably preissued")
            intent = self._open_tokens.get(token.intent_sha256)
            if intent is None or any(
                intent[key] != value for key, value in identity.items()
            ):
                raise ValueError("preissued attempt intent identity changed")
            return token

    async def finish_attempt(
        self,
        *,
        token: DispatchAttemptJournalToken,
        receipt: AssignmentExecutionReceipt,
    ) -> None:
        async with self._append_lock:
            intent = self._open_tokens.get(token.intent_sha256)
            if intent is None:
                raise ValueError(
                    "attempt finish lacks its process-local durable intent"
                )
            if (
                receipt.assignment_sha256 != intent["assignment_sha256"]
                or receipt.budget_sha256 != intent["budget_sha256"]
                or receipt.attempt != intent["attempt"]
                or receipt.prior_attempt_receipt_sha256
                != intent["prior_attempt_receipt_sha256"]
                or receipt.attempt_intervals_monotonic_ns[-1][0]
                != token.started_monotonic_ns
            ):
                raise ValueError("attempt finish receipt differs from its intent")
            wave_index = next(
                index
                for index, wave in enumerate(self.plan.waves)
                if wave.sha256 == receipt.wave_sha256
            )
            wave = self.plan.waves[wave_index]
            finished_ns = receipt.attempt_intervals_monotonic_ns[-1][1]
            payload = {key: intent[key] for key in self._ATTEMPT_FIELDS}
            payload.update(
                {
                    "wave_event_sha256": intent["wave_event_sha256"],
                    "started_monotonic_ns": token.started_monotonic_ns,
                    "finished_monotonic_ns": finished_ns,
                    "intent_event_sha256": token.intent_sha256,
                    "assignment_receipt": receipt.to_dict(),
                }
            )
            event = self._event_common(
                wave=wave,
                event_type="FINISH",
                payload=payload,
            )
            self._append_event(event)
            del self._open_tokens[token.intent_sha256]


_SHARED_BUNDLE_AUTHORITY_FIELDS = (
    "registry",
    "inventory",
    "interference_envelope",
    "interference_source_receipt",
    "budget_plan",
    "budget_policy",
    "budget_load_bindings",
    "capacity_envelope",
    "capacity_source_manifest",
    "capacity_verification_receipt",
    "dependency_receipts",
    "activation",
    "activation_runtime",
    "activation_split",
    "dispatch_context",
    "dispatch_plan",
)


def _require_shared_bundle_authority(
    bundles: tuple[IndustrialAssignmentExecutionBundle, ...],
) -> None:
    """Require one exact scheduler/raw-budget authority across all bundles."""

    if not bundles:
        raise ValueError("shared bundle authority requires at least one bundle")
    first = bundles[0]
    for bundle in bundles[1:]:
        for name in _SHARED_BUNDLE_AUTHORITY_FIELDS:
            if getattr(bundle, name) != getattr(first, name):
                raise ValueError(
                    f"execution bundles differ in shared authority field {name}"
                )


def _preflight_bundle_assignment_sources(
    bundle: IndustrialAssignmentExecutionBundle,
) -> None:
    """Hash every assignment-local raw source without constructing a plan."""

    for source in (
        bundle.topology_receipts,
        bundle.production_load,
        bundle.run_config,
        bundle.server_launch,
        bundle.execution_plan_summary,
        bundle.prepared_models,
        bundle.compile_cache_plan,
        bundle.execution_policy,
    ):
        source.load()
    for artifact in (
        *bundle.dependency_artifacts,
        bundle.split_artifact,
        bundle.sampling_artifact,
        bundle.model_lock_artifact,
        bundle.inventory_source_artifact,
        bundle.runtime_envelope_artifact,
    ):
        artifact.source.load()
    compile_plan = CompileCacheLaunchPlan.load(bundle.compile_cache_plan.path)
    if compile_plan.sha256 != bundle.compile_cache_plan.semantic_sha256:
        raise ValueError("bundle compile-cache plan semantic identity mismatch")
    preflight_compile_cache_launch(compile_plan)


def _preflight_bundle_trainable_plan_release_trust(
    bundle: IndustrialAssignmentExecutionBundle,
) -> None:
    """Reject absent adapted release trust before global wave mutation."""

    method = _strict_object(
        "bundle preflight RunConfig",
        bundle.run_config.load(),
        frozenset(
            {
                "schema_version",
                "method",
                "model",
                "runtime",
                "adaptation",
                "online_spec",
                "tenant_id",
            }
        ),
    )["method"]
    if method in {"target_only", "static"}:
        if (
            bundle.trainable_plan_authority is not None
            or bundle.prepared_model_content_release_manifest_sha256 is not None
        ):
            raise ValueError(
                "Target-only/Static bundle must not carry trainable-plan authority"
            )
        return
    if method not in {"tts", "l0"}:
        raise ExecutionBundleBlockedError(
            "current_release_core_trainable_plan_method_required"
        )
    authority = bundle.trainable_plan_authority
    if type(authority) is not TrainablePlanAuthorityBinding:
        raise ExecutionBundleBlockedError(
            TRAINABLE_PLAN_RAW_AUTHORITY_UNAVAILABLE_REASON
        )
    claimed = bundle.prepared_model_content_release_manifest_sha256
    if claimed is None:
        raise ExecutionBundleBlockedError(
            PREPARED_MODEL_CONTENT_RELEASE_MANIFEST_PIN_UNAVAILABLE_REASON
        )
    if not has_prepared_model_content_release_manifest_sha256(
        model_lock_sha256=authority.model_lock_sha256,
        prepared=authority.prepared_model_content_authority.prepared_model_set,
        claimed_manifest_sha256=claimed,
    ):
        raise ExecutionBundleBlockedError(
            PREPARED_MODEL_CONTENT_RELEASE_MANIFEST_PIN_UNAVAILABLE_REASON
        )


def _declared_wave_assignment_ids(
    bundle: IndustrialAssignmentExecutionBundle,
    *,
    wave_index: int,
) -> tuple[str, ...]:
    """Select a representative only; formal replay below grants authority."""

    value = bundle.dispatch_plan.load()
    if type(value) is not dict:
        raise TypeError("declared dispatch plan must be a JSON object")
    waves = _strict_list("declared dispatch waves", value.get("waves"))
    if wave_index >= len(waves):
        raise ValueError("dispatch wave index is outside the frozen plan")
    wave = GpuDispatchWave.from_dict(waves[wave_index])
    if wave.wave_index != wave_index:
        raise ValueError("declared dispatch wave index is not canonical")
    return tuple(assignment.assignment_id for assignment in wave.assignments)


async def execute_dispatch_wave_bundles(
    materialization_manifest_path: str | Path,
    *,
    wave_index: int,
    receipt_output: str | Path,
    resume_receipt_path: str | Path | None = None,
) -> DispatchScheduleReceipt:
    """Execute exactly one frozen wave through the first-party TaskGroup path.

    All gates that can be evaluated without allocation run before importing the
    pinned serving client, creating an evidence root, or launching a process.
    The current release exits at the source-owned trust/raw-authority gates, so
    this function is CPU-testable without ever reaching a GPU.
    """

    if not isinstance(materialization_manifest_path, (str, Path)):
        raise TypeError("formal dispatch execution requires one manifest path")
    unresolved_manifest = _absolute_lexical_path(materialization_manifest_path)
    if (
        not unresolved_manifest.is_absolute()
        or unresolved_manifest.is_symlink()
        or unresolved_manifest.resolve() != unresolved_manifest
    ):
        raise ExecutionBundleBlockedError(
            "industrial_dispatch_bundle_materialization_manifest_invalid"
        )
    require_release_dispatch_execution_authority()
    if (
        isinstance(wave_index, bool)
        or not isinstance(wave_index, int)
        or wave_index < 0
    ):
        raise ValueError("dispatch wave index must be a non-negative integer")
    receipt_target, _ = preflight_dispatch_receipt_output(receipt_output)
    if (
        resume_receipt_path is not None
        and Path(resume_receipt_path).resolve() == receipt_target
    ):
        raise ValueError("resume receipt and next-wave receipt output must differ")
    # Local import avoids making the source-owned materializer depend on an
    # executable bundle import cycle.  The manifest loader reopens every member
    # and its complete path-bound construction graph before returning here.
    from lightcone_spec.orchestration.execution_bundle_materializer import (
        load_materialized_dispatch_execution_bundle_publication,
    )

    publication = load_materialized_dispatch_execution_bundle_publication(
        unresolved_manifest
    )
    if not publication.bundles:
        raise ExecutionBundleBlockedError("industrial_execution_bundle_missing")

    bundles = publication.bundles
    _require_shared_bundle_authority(bundles)
    for bundle in bundles:
        _preflight_bundle_assignment_sources(bundle)
        _preflight_bundle_trainable_plan_release_trust(bundle)

    declared_wave_ids = _declared_wave_assignment_ids(
        bundles[0],
        wave_index=wave_index,
    )
    representative = next(
        (bundle for bundle in bundles if bundle.assignment_sha256 in declared_wave_ids),
        None,
    )
    if representative is None:
        raise ExecutionBundleBlockedError(
            "industrial_execution_bundle_coverage_incomplete"
        )
    representative_plan = representative.reconstruct_execution_plan()
    dispatch_plan = representative_plan.dispatch_plan
    base_context = representative_plan.dispatch_context
    if wave_index >= len(dispatch_plan.waves):
        raise ValueError("dispatch wave index is outside the frozen plan")
    wave = dispatch_plan.waves[wave_index]
    actual_wave_ids = tuple(assignment.assignment_id for assignment in wave.assignments)
    if actual_wave_ids != declared_wave_ids:
        raise ValueError("declared wave selection differs from formal dispatch replay")

    assignment_by_id = {
        assignment.assignment_id: assignment
        for dispatch_wave in dispatch_plan.waves
        for assignment in dispatch_wave.assignments
    }
    bundle_by_assignment: dict[str, IndustrialAssignmentExecutionBundle] = {}
    for bundle in bundles:
        if bundle.assignment_sha256 in bundle_by_assignment:
            raise ValueError("execution bundles duplicate a dispatch assignment")
        assignment = assignment_by_id.get(bundle.assignment_sha256)
        if assignment is None or assignment.work_item.item_id != bundle.cell_id:
            raise ValueError("execution bundle names another dispatch assignment")
        cell = assignment.work_item.cell
        if cell.resources.workload_class in {
            WorkloadClass.COMPILE,
            WorkloadClass.DOWNLOAD,
        }:
            raise ExecutionBundleBlockedError(
                "current_release_serving_execution_bundle_required"
            )
        bundle_by_assignment[bundle.assignment_sha256] = bundle
    if set(bundle_by_assignment) != set(assignment_by_id):
        raise ExecutionBundleBlockedError(
            "industrial_execution_bundle_coverage_incomplete"
        )

    plan_by_assignment = {
        representative.assignment_sha256: representative_plan,
    }

    def assignment_plan(assignment_sha256: str) -> IndustrialExecutionPlan:
        existing = plan_by_assignment.get(assignment_sha256)
        plan = (
            existing
            if existing is not None
            else bundle_by_assignment[assignment_sha256].reconstruct_execution_plan()
        )
        if plan.dispatch_plan != dispatch_plan or plan.dispatch_context != base_context:
            raise ValueError("execution bundle differs from shared dispatch authority")
        physical = plan.runtime_plan.physical_assignment
        if physical is None or physical.assignment_sha256 != assignment_sha256:
            raise ValueError("execution bundle plan lost its physical assignment")
        plan_by_assignment[assignment_sha256] = plan
        return plan

    for assignment in wave.assignments:
        assignment_plan(assignment.assignment_id)

    final_ready_budgets = base_context.require_ready_budget_authority()
    if final_ready_budgets != base_context.budgets:
        raise RuntimeError("final group budget replay changed the frozen scheduler")

    supplied_receipt: DispatchScheduleReceipt | None = None
    supplied_journal_binding: DispatchAttemptJournalBinding | None = None
    if resume_receipt_path is not None:
        supplied_receipt, supplied_journal_binding = _load_dispatch_schedule_envelope(
            resume_receipt_path,
            plan=dispatch_plan,
            execution_context=base_context,
        )
        if supplied_journal_binding is None:
            raise ExecutionBundleBlockedError(
                "dispatch_resume_receipt_lacks_raw_attempt_journal"
            )
        journal_root = Path(supplied_journal_binding.journal_path)
    else:
        journal_root = dispatch_attempt_journal_path(receipt_target)

    journal = DispatchAttemptJournal.open_or_create(
        journal_root,
        plan=dispatch_plan,
        execution_context=base_context,
        expected_prefix=supplied_journal_binding,
        execution_bundle_manifest_sha256=publication.manifest.sha256,
    )
    snapshot = journal.replay()
    snapshot.require_complete_cost_authority()
    if supplied_journal_binding is not None:
        prefix_snapshot = journal.replay(
            event_count=supplied_journal_binding.event_count
        )
        prefix_snapshot.require_complete_cost_authority()
        if (
            prefix_snapshot.binding != supplied_journal_binding
            or prefix_snapshot.receipt != supplied_receipt
        ):
            raise ExecutionBundleBlockedError(
                "dispatch_resume_receipt_differs_from_raw_attempt_journal"
            )

    def context_from_snapshot(
        value: DispatchAttemptJournalSnapshot,
    ) -> GpuDispatchExecutionContext:
        authorities: list[AssignmentTerminalAuthority] = []
        for binding in value.terminal_bindings:
            try:
                resumed_plan = assignment_plan(binding.assignment_sha256)
            except KeyError as error:
                raise ValueError(
                    "resume terminal binding lacks its raw execution bundle"
                ) from error
            authorities.append(
                AssignmentTerminalAuthority.from_binding(binding, plan=resumed_plan)
            )
        return replace(
            base_context,
            resume_terminal_authorities=tuple(authorities),
        )

    execution_context = context_from_snapshot(snapshot)
    resume_receipt = snapshot.receipt
    attempt_journal_replay = snapshot.replay_authority
    recovery_only = False
    if resume_receipt is None:
        if wave_index != 0:
            raise ExecutionBundleBlockedError(
                "dispatch_schedule_receipt_required_for_noninitial_wave"
            )
    elif supplied_receipt is None:
        last_wave_index = resume_receipt.wave_receipts[-1].wave_index
        if wave_index != last_wave_index:
            raise ValueError(
                "journal-only recovery must publish its last finished wave"
            )
        recovery_only = True
    elif resume_receipt != supplied_receipt:
        if wave_index != resume_receipt.wave_receipts[-1].wave_index:
            raise ValueError(
                "journal advanced beyond the supplied receipt on another wave"
            )
        recovery_only = True
    elif resume_receipt.phase is DispatchExecutionPhase.COMPLETE:
        raise ValueError("complete schedule receipt has no next wave")
    else:
        expected_wave_index = (
            resume_receipt.wave_receipts[-1].wave_index
            if resume_receipt.phase is DispatchExecutionPhase.FAILED
            else len(resume_receipt.wave_receipts)
        )
        if wave_index != expected_wave_index:
            raise ValueError("requested wave is not the journal-authorized wave")

    if recovery_only:
        if attempt_journal_replay is None or snapshot.binding is None:
            raise RuntimeError("finished journal recovery lacks its raw authority")
        validate_dispatch_resume(
            dispatch_plan,
            resume_receipt,
            execution_context=execution_context,
            attempt_journal_replay=attempt_journal_replay,
        )
        publish_dispatch_schedule_receipt(
            receipt_target,
            resume_receipt,
            attempt_journal=snapshot.binding,
        )
        return resume_receipt

    transports: dict[str, PinnedBenchServingTransport] = {}
    successful_assignment_ids = {
        receipt.assignment_sha256
        for receipt in snapshot.latest_assignment_receipts
        if receipt.status is AssignmentExecutionStatus.SUCCEEDED
    }
    latest_receipt_by_assignment = {
        receipt.assignment_sha256: receipt
        for receipt in snapshot.latest_assignment_receipts
    }
    for assignment in wave.assignments:
        if assignment.assignment_id in successful_assignment_ids:
            continue
        bundle = bundle_by_assignment[assignment.assignment_id]
        plan = assignment_plan(assignment.assignment_id)
        preflight_fresh_assignment_trace(
            plan,
            output_root=bundle.output_root,
            run_nonce_sha256=bundle.run_nonce_sha256,
            prior_failed_attempt_authorized=(
                latest_receipt_by_assignment.get(assignment.assignment_id) is not None
                and latest_receipt_by_assignment[assignment.assignment_id].status
                is AssignmentExecutionStatus.FAILED
            ),
        )
        checkout = Path(plan.server_launch.argv[4])
        transports[assignment.assignment_id] = await asyncio.to_thread(
            PinnedBenchServingTransport.from_checkout,
            checkout,
        )

    async def runner(assignment: GpuAssignment) -> AssignmentTerminalAuthority:
        bundle = bundle_by_assignment[assignment.assignment_id]
        plan = assignment_plan(assignment.assignment_id)
        transport = transports[assignment.assignment_id]
        result = await execute_industrial_plan(
            plan,
            output_root=bundle.output_root,
            run_nonce_sha256=bundle.run_nonce_sha256,
            launch_server=launch_server_subprocess,
            transport=transport,
            native_evidence=NativeTerminalProvider(
                transport,
                trusted_attester_policy=plan.trusted_attester_policy,
            ),
        )
        if result.resumed:
            prior = latest_receipt_by_assignment.get(assignment.assignment_id)
            if prior is None or prior.status is not AssignmentExecutionStatus.FAILED:
                raise RuntimeError(
                    "fresh dispatch wave cannot consume implicit per-plan resume"
                )
        return AssignmentTerminalAuthority(
            plan=plan,
            result=result,
            run_nonce_sha256=bundle.run_nonce_sha256,
        )

    receipt = await execute_dispatch_plan(
        dispatch_plan,
        execution_context=execution_context,
        runner=runner,
        resume_receipt=resume_receipt,
        attempt_journal=journal,
        attempt_journal_replay=attempt_journal_replay,
        stop_after_wave_index=wave_index,
    )
    final_snapshot = journal.replay()
    final_snapshot.require_complete_cost_authority()
    if final_snapshot.receipt != receipt or final_snapshot.binding is None:
        raise RuntimeError("raw attempt journal differs from the execution receipt")
    final_execution_context = context_from_snapshot(final_snapshot)
    if final_snapshot.replay_authority is None:
        raise RuntimeError("finished attempt journal lacks replay authority")
    validate_dispatch_resume(
        dispatch_plan,
        receipt,
        execution_context=final_execution_context,
        attempt_journal_replay=final_snapshot.replay_authority,
    )
    publish_dispatch_schedule_receipt(
        receipt_target,
        receipt,
        attempt_journal=final_snapshot.binding,
    )
    return receipt


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(name: str, value: object) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lower-case SHA-256")
    return value


def _strict_text(name: str, value: object) -> str:
    if type(value) is not str or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be non-empty single-line text")
    return value


def _strict_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a JSON integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _strict_float(name: str, value: object) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be a JSON number")
    converted = float(value)
    if not converted > 0 or not converted < float("inf"):
        raise ValueError(f"{name} must be finite and positive")
    return converted


def _strict_object(
    name: str,
    value: object,
    fields: frozenset[str],
) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing or unknown:
        raise ValueError(
            f"{name} fields differ: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return value


def _strict_list(name: str, value: object) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _read_regular_file(path: Path, *, label: str) -> bytes:
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError(f"{label} path must be absolute and resolved")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    try:
        directory_descriptor = os.open(path.parent, directory_flags)
    except OSError as error:
        raise RuntimeError(f"{label} parent is not a stable directory") from error
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        directory_opened = os.fstat(directory_descriptor)
        directory_current = path.parent.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(directory_opened.st_mode)
            or directory_opened.st_dev != directory_current.st_dev
            or directory_opened.st_ino != directory_current.st_ino
        ):
            raise RuntimeError(f"{label} parent changed before read")
        descriptor = os.open(path.name, flags, dir_fd=directory_descriptor)
    except OSError as error:
        os.close(directory_descriptor)
        raise RuntimeError(f"{label} is not a readable regular file") from error
    except BaseException:
        os.close(directory_descriptor)
        raise
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise RuntimeError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            body = handle.read()
        closed = os.fstat(descriptor)
        current = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or current.st_dev != opened.st_dev
            or current.st_ino != opened.st_ino
            or current.st_size != len(body)
            or closed.st_size != opened.st_size
            or closed.st_mtime_ns != opened.st_mtime_ns
            or closed.st_ctime_ns != opened.st_ctime_ns
            or current.st_mtime_ns != opened.st_mtime_ns
            or current.st_ctime_ns != opened.st_ctime_ns
        ):
            raise RuntimeError(f"{label} changed while it was read")
        return body
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)


def _decode_json(body: bytes, *, label: str) -> object:
    def finite_float(raw: str) -> float:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number {raw!r} is forbidden")
        return value

    try:
        return json.loads(
            body.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=finite_float,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, OverflowError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error


@dataclass(frozen=True)
class BoundJsonSource:
    """One immutable JSON path with raw, canonical, semantic, and sidecar IDs."""

    path: str
    canonical_sha256: str
    semantic_sha256: str
    file_sha256: str
    sidecar_file_sha256: str
    size: int

    @classmethod
    def bind(
        cls,
        path: str | Path,
        *,
        semantic_sha256: str | None = None,
    ) -> Self:
        source = Path(path).resolve()
        body = _read_regular_file(source, label="bundle source")
        value = _decode_json(body, label="bundle source")
        canonical_sha256 = hashlib.sha256(_canonical_bytes(value)).hexdigest()
        sidecar_body = _read_regular_file(
            Path(f"{source}.sha256"),
            label="bundle source sidecar",
        )
        if sidecar_body != f"{canonical_sha256}\n".encode("ascii"):
            raise ValueError("bundle source sidecar differs from canonical JSON")
        result = cls(
            path=str(source),
            canonical_sha256=canonical_sha256,
            semantic_sha256=semantic_sha256 or canonical_sha256,
            file_sha256=hashlib.sha256(body).hexdigest(),
            sidecar_file_sha256=hashlib.sha256(sidecar_body).hexdigest(),
            size=len(body),
        )
        result.load()
        return result

    def __post_init__(self) -> None:
        source = Path(self.path)
        if not source.is_absolute() or source.resolve() != source:
            raise ValueError("bundle source path must be absolute and resolved")
        for name in (
            "canonical_sha256",
            "semantic_sha256",
            "file_sha256",
            "sidecar_file_sha256",
        ):
            _require_sha256(f"bundle source {name}", getattr(self, name))
        _strict_int("bundle source size", self.size, minimum=1)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "bound JSON source",
            value,
            frozenset(
                {
                    "path",
                    "canonical_sha256",
                    "semantic_sha256",
                    "file_sha256",
                    "sidecar_file_sha256",
                    "size",
                }
            ),
        )
        return cls(
            path=_strict_text("bound JSON path", row["path"]),
            canonical_sha256=_require_sha256(
                "bound JSON canonical SHA-256", row["canonical_sha256"]
            ),
            semantic_sha256=_require_sha256(
                "bound JSON semantic SHA-256", row["semantic_sha256"]
            ),
            file_sha256=_require_sha256("bound JSON file SHA-256", row["file_sha256"]),
            sidecar_file_sha256=_require_sha256(
                "bound JSON sidecar SHA-256", row["sidecar_file_sha256"]
            ),
            size=_strict_int("bound JSON size", row["size"], minimum=1),
        )

    def load(self) -> object:
        source = Path(self.path)
        body = _read_regular_file(source, label="bound bundle source")
        sidecar_body = _read_regular_file(
            Path(f"{source}.sha256"),
            label="bound bundle source sidecar",
        )
        value = _decode_json(body, label="bound bundle source")
        canonical_sha256 = hashlib.sha256(_canonical_bytes(value)).hexdigest()
        if (
            len(body) != self.size
            or hashlib.sha256(body).hexdigest() != self.file_sha256
            or hashlib.sha256(sidecar_body).hexdigest() != self.sidecar_file_sha256
            or canonical_sha256 != self.canonical_sha256
            or sidecar_body != f"{canonical_sha256}\n".encode("ascii")
        ):
            raise RuntimeError("bound bundle source or sidecar changed")
        return value


@dataclass(frozen=True)
class BoundExecutionArtifact:
    """Path-bound artifact that must cover one dependency or execution role."""

    name: str
    experiment: str | None
    source: BoundJsonSource

    def __post_init__(self) -> None:
        _strict_text("execution artifact name", self.name)
        if self.experiment is not None:
            _strict_text("execution artifact experiment", self.experiment)
        if type(self.source) is not BoundJsonSource:
            raise TypeError("execution artifact source must be exact")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "experiment": self.experiment,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "execution artifact",
            value,
            frozenset({"name", "experiment", "source"}),
        )
        experiment = row["experiment"]
        if experiment is not None and type(experiment) is not str:
            raise TypeError("execution artifact experiment must be text or null")
        return cls(
            name=_strict_text("execution artifact name", row["name"]),
            experiment=experiment,
            source=BoundJsonSource.from_dict(row["source"]),
        )

    def artifact_binding(self) -> ArtifactBinding:
        self.source.load()
        return ArtifactBinding.from_path(
            name=self.name,
            path=self.source.path,
            expected_sha256=self.source.semantic_sha256,
            semantic_sha256=self.source.semantic_sha256,
            experiment=self.experiment,
        )


@dataclass(frozen=True)
class InterferenceCalibrationTerminalBundle:
    """One calibration run reconstructed from its raw plan and terminal files."""

    execution_bundle: BoundJsonSource
    terminal_binding: AssignmentTerminalBinding

    def __post_init__(self) -> None:
        if type(self.execution_bundle) is not BoundJsonSource:
            raise TypeError("calibration terminal requires an exact execution bundle")
        if type(self.terminal_binding) is not AssignmentTerminalBinding:
            raise TypeError("calibration terminal requires an exact terminal binding")

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_bundle": self.execution_bundle.to_dict(),
            "terminal_binding": self.terminal_binding.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "interference calibration terminal bundle",
            value,
            frozenset({"execution_bundle", "terminal_binding"}),
        )
        return cls(
            execution_bundle=BoundJsonSource.from_dict(row["execution_bundle"]),
            terminal_binding=AssignmentTerminalBinding.from_dict(
                row["terminal_binding"]
            ),
        )

    def reconstruct(self) -> AssignmentTerminalAuthority:
        raw_bundle = IndustrialAssignmentExecutionBundle.load(
            self.execution_bundle.path
        )
        if (
            raw_bundle.sha256 != self.execution_bundle.semantic_sha256
            or raw_bundle.sha256 != self.execution_bundle.canonical_sha256
        ):
            raise ValueError("calibration execution-bundle identity mismatch")
        if raw_bundle.interference_calibration_authority is not None:
            raise ValueError(
                "calibration terminal cannot recursively consume calibrated evidence"
            )
        plan = raw_bundle.reconstruct_execution_plan()
        return AssignmentTerminalAuthority.from_binding(
            self.terminal_binding,
            plan=plan,
        )


@dataclass(frozen=True)
class InterferenceCalibrationExecutionAuthority:
    """Path-bound raw calibration authority carried by a formal bundle."""

    schema_version: int
    kind: Literal["industrial_interference_calibration_execution_authority"]
    source: InterferenceCalibrationSourceAuthority
    terminals: tuple[InterferenceCalibrationTerminalBundle, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "industrial_interference_calibration_execution_authority"
        ):
            raise ValueError("calibration execution authority schema is unsupported")
        if type(self.source) is not InterferenceCalibrationSourceAuthority:
            raise TypeError("calibration execution authority requires raw sources")
        if any(
            type(terminal) is not InterferenceCalibrationTerminalBundle
            for terminal in self.terminals
        ):
            raise TypeError("calibration execution authority has a wrong terminal")
        identities = tuple(
            terminal.terminal_binding.authority_sha256 for terminal in self.terminals
        )
        if not identities or identities != tuple(sorted(set(identities))):
            raise ValueError(
                "calibration execution terminals must be authority-sorted and unique"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "source": self.source.to_dict(),
            "terminals": [terminal.to_dict() for terminal in self.terminals],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "interference calibration execution authority",
            value,
            frozenset({"schema_version", "kind", "source", "terminals"}),
        )
        return cls(
            schema_version=_strict_int(
                "calibration execution authority schema", row["schema_version"]
            ),
            kind=_strict_text("calibration execution authority kind", row["kind"]),
            source=InterferenceCalibrationSourceAuthority.from_dict(row["source"]),
            terminals=tuple(
                InterferenceCalibrationTerminalBundle.from_dict(item)
                for item in _strict_list(
                    "calibration execution terminals", row["terminals"]
                )
            ),
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def reconstruct(self) -> InterferenceCalibrationAuthority:
        authorities = tuple(terminal.reconstruct() for terminal in self.terminals)
        result = InterferenceCalibrationAuthority(
            schema_version=1,
            source=self.source,
            terminal_authorities=authorities,
        )
        if tuple(authority.sha256 for authority in authorities) != tuple(
            terminal.terminal_binding.authority_sha256 for terminal in self.terminals
        ):
            raise ValueError("calibration terminal binding identities changed")
        return result


@dataclass(frozen=True)
class IndustrialExecutionPlanAudit:
    """Non-executable proof of raw component and scheduler replay only."""

    schema_version: int
    kind: Literal["industrial_execution_plan_audit"]
    bundle_sha256: str
    assignment_sha256: str
    cell_id: str
    budget_plan_sha256: str
    budget_plan_status: Literal["READY", "UNRESOLVED"]
    budget_materialization_authority_sha256: str
    component_replay_sha256: str
    dispatch_plan_sha256: str
    execution_semantics_sha256: str | None
    execution_semantics_authority: Literal["diagnostic_non_authority"]
    exact_dispatch_replay: Literal[True]
    execution_plan_sha256: None
    execution_plan_status: Literal["NOT_VALIDATED"]
    execution_plan_reason_code: str

    def __post_init__(self) -> None:
        if self.schema_version != 2 or self.kind != "industrial_execution_plan_audit":
            raise ValueError("industrial execution-plan audit schema is unsupported")
        for name in (
            "bundle_sha256",
            "assignment_sha256",
            "cell_id",
            "budget_plan_sha256",
            "budget_materialization_authority_sha256",
            "component_replay_sha256",
            "dispatch_plan_sha256",
        ):
            _require_sha256(f"execution-plan audit {name}", getattr(self, name))
        if self.budget_plan_status not in {"READY", "UNRESOLVED"}:
            raise ValueError("execution-plan audit budget status is unsupported")
        if self.execution_semantics_sha256 is not None:
            _require_sha256(
                "execution-plan audit execution semantics",
                self.execution_semantics_sha256,
            )
        if self.execution_semantics_authority != "diagnostic_non_authority":
            raise ValueError("execution semantics audit cannot claim launch authority")
        if self.exact_dispatch_replay is not True:
            raise ValueError("execution-plan audit must record exact dispatch replay")
        if self.execution_plan_sha256 is not None:
            raise ValueError("audit-only output cannot claim an execution-plan SHA")
        if self.execution_plan_status != "NOT_VALIDATED":
            raise ValueError("audit-only output cannot validate an execution plan")
        _strict_text("execution-plan audit reason", self.execution_plan_reason_code)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class IndustrialAssignmentExecutionBundle:
    """All raw paths required to reconstruct one formal serving assignment."""

    schema_version: int
    kind: Literal["industrial_assignment_execution_bundle"]
    assignment_sha256: str
    cell_id: str
    execution_plan_sha256: str
    run_nonce_sha256: str
    output_root: str
    registry: BoundJsonSource
    inventory: BoundJsonSource
    interference_envelope: BoundJsonSource
    interference_source_receipt: BoundJsonSource
    interference_calibration_authority: InterferenceCalibrationExecutionAuthority | None
    budget_plan: BoundJsonSource
    budget_policy: BoundJsonSource
    budget_load_bindings: tuple[BoundJsonSource, ...]
    capacity_envelope: BoundJsonSource
    capacity_source_manifest: BoundJsonSource
    capacity_verification_receipt: BoundJsonSource
    dependency_receipts: tuple[BoundJsonSource, ...]
    activation: BoundJsonSource
    activation_runtime: BoundJsonSource
    activation_split: BoundJsonSource
    dispatch_context: BoundJsonSource
    dispatch_plan: BoundJsonSource
    topology_receipts: BoundJsonSource
    production_load: BoundJsonSource
    run_config: BoundJsonSource
    server_launch: BoundJsonSource
    execution_plan_summary: BoundJsonSource
    dependency_artifacts: tuple[BoundExecutionArtifact, ...]
    split_artifact: BoundExecutionArtifact
    sampling_artifact: BoundExecutionArtifact
    model_lock_artifact: BoundExecutionArtifact
    prepared_models: BoundJsonSource
    trainable_plan_authority: TrainablePlanAuthorityBinding | None
    failure_injection_authority: FailureInjectionAuthorityBinding | None
    prepared_model_content_release_manifest_sha256: str | None
    compile_cache_plan: BoundJsonSource
    inventory_source_artifact: BoundExecutionArtifact
    runtime_envelope_artifact: BoundExecutionArtifact
    execution_policy: BoundJsonSource

    def __post_init__(self) -> None:
        if self.schema_version != 4 or self.kind != _BUNDLE_KIND:
            raise ValueError("industrial execution-bundle schema is unsupported")
        for name in (
            "assignment_sha256",
            "cell_id",
            "execution_plan_sha256",
            "run_nonce_sha256",
        ):
            _require_sha256(f"execution bundle {name}", getattr(self, name))
        root = Path(self.output_root)
        if not root.is_absolute() or root.resolve() != root:
            raise ValueError(
                "execution bundle output root must be absolute and resolved"
            )
        sources = (
            self.registry,
            self.inventory,
            self.interference_envelope,
            self.interference_source_receipt,
            self.budget_plan,
            self.budget_policy,
            *self.budget_load_bindings,
            self.capacity_envelope,
            self.capacity_source_manifest,
            self.capacity_verification_receipt,
            *self.dependency_receipts,
            self.activation,
            self.activation_runtime,
            self.activation_split,
            self.dispatch_context,
            self.dispatch_plan,
            self.topology_receipts,
            self.production_load,
            self.run_config,
            self.server_launch,
            self.execution_plan_summary,
            self.prepared_models,
            self.compile_cache_plan,
            self.execution_policy,
        )
        if any(type(source) is not BoundJsonSource for source in sources):
            raise TypeError("execution bundle contains a non-exact JSON source")
        if (
            self.interference_calibration_authority is not None
            and type(self.interference_calibration_authority)
            is not InterferenceCalibrationExecutionAuthority
        ):
            raise TypeError("execution bundle has a wrong calibration authority")
        if (
            self.trainable_plan_authority is not None
            and type(self.trainable_plan_authority) is not TrainablePlanAuthorityBinding
        ):
            raise TypeError("execution bundle has a wrong trainable-plan authority")
        if (
            self.failure_injection_authority is not None
            and type(self.failure_injection_authority)
            is not FailureInjectionAuthorityBinding
        ):
            raise TypeError("execution bundle has a wrong failure authority")
        if self.prepared_model_content_release_manifest_sha256 is not None:
            _require_sha256(
                "bundle prepared model content release manifest",
                self.prepared_model_content_release_manifest_sha256,
            )
        artifacts = (
            *self.dependency_artifacts,
            self.split_artifact,
            self.sampling_artifact,
            self.model_lock_artifact,
            self.inventory_source_artifact,
            self.runtime_envelope_artifact,
        )
        if any(type(artifact) is not BoundExecutionArtifact for artifact in artifacts):
            raise TypeError("execution bundle contains a non-exact artifact binding")
        dependency_keys = tuple(
            (artifact.experiment, artifact.name)
            for artifact in self.dependency_artifacts
        )
        if dependency_keys != tuple(sorted(set(dependency_keys))):
            raise ValueError("dependency artifacts must be role-sorted and unique")
        receipt_paths = tuple(source.path for source in self.dependency_receipts)
        if len(receipt_paths) != len(set(receipt_paths)):
            raise ValueError("dependency receipt paths are duplicated")
        bindings_by_path: dict[str, BoundJsonSource] = {}
        for source in (*sources, *(artifact.source for artifact in artifacts)):
            previous = bindings_by_path.setdefault(source.path, source)
            if previous != source:
                raise ValueError(
                    "execution bundle gives one raw path conflicting identities"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "assignment_sha256": self.assignment_sha256,
            "cell_id": self.cell_id,
            "execution_plan_sha256": self.execution_plan_sha256,
            "run_nonce_sha256": self.run_nonce_sha256,
            "output_root": self.output_root,
            "registry": self.registry.to_dict(),
            "inventory": self.inventory.to_dict(),
            "interference_envelope": self.interference_envelope.to_dict(),
            "interference_source_receipt": (self.interference_source_receipt.to_dict()),
            "interference_calibration_authority": (
                None
                if self.interference_calibration_authority is None
                else self.interference_calibration_authority.to_dict()
            ),
            "budget_plan": self.budget_plan.to_dict(),
            "budget_policy": self.budget_policy.to_dict(),
            "budget_load_bindings": [
                source.to_dict() for source in self.budget_load_bindings
            ],
            "capacity_envelope": self.capacity_envelope.to_dict(),
            "capacity_source_manifest": self.capacity_source_manifest.to_dict(),
            "capacity_verification_receipt": (
                self.capacity_verification_receipt.to_dict()
            ),
            "dependency_receipts": [
                source.to_dict() for source in self.dependency_receipts
            ],
            "activation": self.activation.to_dict(),
            "activation_runtime": self.activation_runtime.to_dict(),
            "activation_split": self.activation_split.to_dict(),
            "dispatch_context": self.dispatch_context.to_dict(),
            "dispatch_plan": self.dispatch_plan.to_dict(),
            "topology_receipts": self.topology_receipts.to_dict(),
            "production_load": self.production_load.to_dict(),
            "run_config": self.run_config.to_dict(),
            "server_launch": self.server_launch.to_dict(),
            "execution_plan_summary": self.execution_plan_summary.to_dict(),
            "dependency_artifacts": [
                artifact.to_dict() for artifact in self.dependency_artifacts
            ],
            "split_artifact": self.split_artifact.to_dict(),
            "sampling_artifact": self.sampling_artifact.to_dict(),
            "model_lock_artifact": self.model_lock_artifact.to_dict(),
            "prepared_models": self.prepared_models.to_dict(),
            "trainable_plan_authority": (
                None
                if self.trainable_plan_authority is None
                else trainable_plan_authority_binding_to_dict(
                    self.trainable_plan_authority
                )
            ),
            "failure_injection_authority": (
                None
                if self.failure_injection_authority is None
                else self.failure_injection_authority.to_dict()
            ),
            "prepared_model_content_release_manifest_sha256": (
                self.prepared_model_content_release_manifest_sha256
            ),
            "compile_cache_plan": self.compile_cache_plan.to_dict(),
            "inventory_source_artifact": (self.inventory_source_artifact.to_dict()),
            "runtime_envelope_artifact": (self.runtime_envelope_artifact.to_dict()),
            "execution_policy": self.execution_policy.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = frozenset(
            {
                "schema_version",
                "kind",
                "assignment_sha256",
                "cell_id",
                "execution_plan_sha256",
                "run_nonce_sha256",
                "output_root",
                "registry",
                "inventory",
                "interference_envelope",
                "interference_source_receipt",
                "interference_calibration_authority",
                "budget_plan",
                "budget_policy",
                "budget_load_bindings",
                "capacity_envelope",
                "capacity_source_manifest",
                "capacity_verification_receipt",
                "dependency_receipts",
                "activation",
                "activation_runtime",
                "activation_split",
                "dispatch_context",
                "dispatch_plan",
                "topology_receipts",
                "production_load",
                "run_config",
                "server_launch",
                "execution_plan_summary",
                "dependency_artifacts",
                "split_artifact",
                "sampling_artifact",
                "model_lock_artifact",
                "prepared_models",
                "trainable_plan_authority",
                "failure_injection_authority",
                "prepared_model_content_release_manifest_sha256",
                "compile_cache_plan",
                "inventory_source_artifact",
                "runtime_envelope_artifact",
                "execution_policy",
            }
        )
        row = _strict_object("industrial execution bundle", value, fields)
        return cls(
            schema_version=_strict_int("bundle schema_version", row["schema_version"]),
            kind=_strict_text("bundle kind", row["kind"]),
            assignment_sha256=_require_sha256(
                "bundle assignment", row["assignment_sha256"]
            ),
            cell_id=_require_sha256("bundle cell", row["cell_id"]),
            execution_plan_sha256=_require_sha256(
                "bundle execution plan", row["execution_plan_sha256"]
            ),
            run_nonce_sha256=_require_sha256(
                "bundle run nonce", row["run_nonce_sha256"]
            ),
            output_root=_strict_text("bundle output root", row["output_root"]),
            registry=BoundJsonSource.from_dict(row["registry"]),
            inventory=BoundJsonSource.from_dict(row["inventory"]),
            interference_envelope=BoundJsonSource.from_dict(
                row["interference_envelope"]
            ),
            interference_source_receipt=BoundJsonSource.from_dict(
                row["interference_source_receipt"]
            ),
            interference_calibration_authority=(
                None
                if row["interference_calibration_authority"] is None
                else InterferenceCalibrationExecutionAuthority.from_dict(
                    row["interference_calibration_authority"]
                )
            ),
            budget_plan=BoundJsonSource.from_dict(row["budget_plan"]),
            budget_policy=BoundJsonSource.from_dict(row["budget_policy"]),
            budget_load_bindings=tuple(
                BoundJsonSource.from_dict(item)
                for item in _strict_list(
                    "bundle budget load bindings", row["budget_load_bindings"]
                )
            ),
            capacity_envelope=BoundJsonSource.from_dict(row["capacity_envelope"]),
            capacity_source_manifest=BoundJsonSource.from_dict(
                row["capacity_source_manifest"]
            ),
            capacity_verification_receipt=BoundJsonSource.from_dict(
                row["capacity_verification_receipt"]
            ),
            dependency_receipts=tuple(
                BoundJsonSource.from_dict(item)
                for item in _strict_list(
                    "bundle dependency receipts", row["dependency_receipts"]
                )
            ),
            activation=BoundJsonSource.from_dict(row["activation"]),
            activation_runtime=BoundJsonSource.from_dict(row["activation_runtime"]),
            activation_split=BoundJsonSource.from_dict(row["activation_split"]),
            dispatch_context=BoundJsonSource.from_dict(row["dispatch_context"]),
            dispatch_plan=BoundJsonSource.from_dict(row["dispatch_plan"]),
            topology_receipts=BoundJsonSource.from_dict(row["topology_receipts"]),
            production_load=BoundJsonSource.from_dict(row["production_load"]),
            run_config=BoundJsonSource.from_dict(row["run_config"]),
            server_launch=BoundJsonSource.from_dict(row["server_launch"]),
            execution_plan_summary=BoundJsonSource.from_dict(
                row["execution_plan_summary"]
            ),
            dependency_artifacts=tuple(
                BoundExecutionArtifact.from_dict(item)
                for item in _strict_list(
                    "bundle dependency artifacts", row["dependency_artifacts"]
                )
            ),
            split_artifact=BoundExecutionArtifact.from_dict(row["split_artifact"]),
            sampling_artifact=BoundExecutionArtifact.from_dict(
                row["sampling_artifact"]
            ),
            model_lock_artifact=BoundExecutionArtifact.from_dict(
                row["model_lock_artifact"]
            ),
            prepared_models=BoundJsonSource.from_dict(row["prepared_models"]),
            trainable_plan_authority=(
                None
                if row["trainable_plan_authority"] is None
                else trainable_plan_authority_binding_from_dict(
                    row["trainable_plan_authority"]
                )
            ),
            failure_injection_authority=(
                None
                if row["failure_injection_authority"] is None
                else FailureInjectionAuthorityBinding.from_dict(
                    row["failure_injection_authority"]
                )
            ),
            prepared_model_content_release_manifest_sha256=(
                None
                if row["prepared_model_content_release_manifest_sha256"] is None
                else _require_sha256(
                    "bundle prepared model content release manifest",
                    row["prepared_model_content_release_manifest_sha256"],
                )
            ),
            compile_cache_plan=BoundJsonSource.from_dict(row["compile_cache_plan"]),
            inventory_source_artifact=BoundExecutionArtifact.from_dict(
                row["inventory_source_artifact"]
            ),
            runtime_envelope_artifact=BoundExecutionArtifact.from_dict(
                row["runtime_envelope_artifact"]
            ),
            execution_policy=BoundJsonSource.from_dict(row["execution_policy"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = Path(path).resolve()
        body = _read_regular_file(source, label="industrial execution bundle")
        value = _decode_json(body, label="industrial execution bundle")
        digest = hashlib.sha256(_canonical_bytes(value)).hexdigest()
        sidecar = _read_regular_file(
            Path(f"{source}.sha256"),
            label="industrial execution bundle sidecar",
        )
        if sidecar != f"{digest}\n".encode("ascii"):
            raise ValueError("industrial execution bundle sidecar differs")
        bundle = cls.from_dict(value)
        if bundle.sha256 != digest:
            raise ValueError("industrial execution bundle is not canonical")
        return bundle

    def audit_execution_plan(self) -> IndustrialExecutionPlanAudit:
        """Replay raw components and dispatch without minting physical authority."""

        result = self._replay_execution_plan(diagnostic=True)
        if type(result) is not IndustrialExecutionPlanAudit:  # pragma: no cover
            raise RuntimeError("diagnostic replay returned an executable plan")
        return result

    def reconstruct_execution_plan(self) -> IndustrialExecutionPlan:
        """Rebuild a physical plan only from live path-bound raw authorities."""

        result = self._replay_execution_plan(diagnostic=False)
        if type(result) is not IndustrialExecutionPlan:  # pragma: no cover
            raise RuntimeError("formal replay returned an audit-only result")
        return result

    def reconstruct_execution_plan_for_materialization(
        self,
        launch_policy: AssignmentLaunchMaterializationPolicy,
        *,
        render_root: str | Path,
    ) -> IndustrialExecutionPlan:
        """Build the plan before its reducer-owned summary exists.

        This is the sole plan-to-bundle construction seam.  Every raw source,
        scheduler replay, release gate, and execution-plan validator remains
        active; only comparison with ``execution_plan_summary`` is deferred
        until the caller serializes the returned plan and constructs the final
        bundle.  Loaded bundles must continue to use
        :meth:`reconstruct_execution_plan`, which always checks that summary.
        """

        if type(launch_policy) is not AssignmentLaunchMaterializationPolicy:
            raise TypeError("materialization requires an exact launch policy")
        result = self._replay_execution_plan(
            diagnostic=False,
            verify_declared_summary=False,
            launch_materialization_policy=launch_policy,
            launch_materialization_root=render_root,
        )
        if type(result) is not IndustrialExecutionPlan:  # pragma: no cover
            raise RuntimeError("materialization replay returned an audit-only result")
        return result

    def preflight_execution_plan_materialization(
        self,
        launch_policy: AssignmentLaunchMaterializationPolicy,
        *,
        render_root: str | Path,
    ) -> None:
        """Replay every raw/source gate without writing renderer artifacts."""

        if type(launch_policy) is not AssignmentLaunchMaterializationPolicy:
            raise TypeError("materialization preflight requires an exact launch policy")
        result = self._replay_execution_plan(
            diagnostic=False,
            verify_declared_summary=False,
            launch_materialization_policy=launch_policy,
            launch_materialization_root=render_root,
            materialization_preflight_only=True,
        )
        if result is not None:  # pragma: no cover - private replay postcondition
            raise RuntimeError("materialization preflight returned an execution plan")

    def _replay_execution_plan(
        self,
        *,
        diagnostic: bool,
        verify_declared_summary: bool = True,
        launch_materialization_policy: (
            AssignmentLaunchMaterializationPolicy | None
        ) = None,
        launch_materialization_root: str | Path | None = None,
        materialization_preflight_only: bool = False,
    ) -> IndustrialExecutionPlan | IndustrialExecutionPlanAudit | None:
        """Replay planning inputs; create physical authority only when READY."""

        if diagnostic and not verify_declared_summary:
            raise ValueError("diagnostic replay cannot omit the declared summary")
        if diagnostic and launch_materialization_policy is not None:
            raise ValueError("diagnostic replay cannot materialize a server launch")
        if verify_declared_summary and launch_materialization_policy is not None:
            raise ValueError("loaded-bundle replay cannot replace its server launch")
        if (launch_materialization_policy is None) != (
            launch_materialization_root is None
        ):
            raise ValueError("materialization policy and render root must be paired")
        if materialization_preflight_only and launch_materialization_policy is None:
            raise ValueError("materialization preflight requires a launch policy")

        registry = _load_registry(self.registry.load())
        if registry.sha256 != self.registry.semantic_sha256:
            raise ValueError("bundle registry semantic identity mismatch")
        inventory = GpuInventory.from_dict(self.inventory.load())
        if inventory.sha256 != self.inventory.semantic_sha256:
            raise ValueError("bundle inventory semantic identity mismatch")
        envelope = InterferenceEnvelope.from_dict(self.interference_envelope.load())
        interference_calibration_authority = _replay_interference_authority(
            inventory=inventory,
            envelope=envelope,
            envelope_source=self.interference_envelope,
            receipt_source=self.interference_source_receipt,
            calibration_source=self.interference_calibration_authority,
        )
        receipts = tuple(
            _receipt_from_dict(source.load()) for source in self.dependency_receipts
        )
        if tuple(receipt.sha256 for receipt in receipts) != tuple(
            source.semantic_sha256 for source in self.dependency_receipts
        ):
            raise ValueError("bundle dependency-receipt semantic identity mismatch")
        (
            budget_plan,
            budget_load_bindings,
            budget_materialization_authority,
            activation_replay,
        ) = _rematerialize_budget_authority(
            registry=registry,
            inventory=inventory,
            activation_manifest_source=self.activation,
            registry_source=self.registry,
            activation_runtime_source=self.activation_runtime,
            activation_split_source=self.activation_split,
            dependency_receipt_sources=self.dependency_receipts,
            declared_source=self.budget_plan,
            policy_source=self.budget_policy,
            load_sources=self.budget_load_bindings,
            capacity_source=self.capacity_envelope,
            capacity_manifest_source=self.capacity_source_manifest,
            capacity_verification_source=self.capacity_verification_receipt,
        )
        activation = activation_replay.selected_activation
        if activation_replay.activation_sha256 != self.activation.semantic_sha256:
            raise ValueError("bundle activation semantic identity mismatch")
        if activation_replay.dependency_receipts != receipts:
            raise ValueError("bundle activation swapped dependency receipts")
        # These are two independent domains.  ``activation_runtime`` is the
        # shared reducer-lineage input and is replayed by the budget authority;
        # ``runtime_envelope_artifact`` is the per-execution doctor authority
        # and is reopened here, then structurally validated by plan.validate().
        self.activation_runtime.load()
        self.runtime_envelope_artifact.source.load()
        interference_bootstrap_authority = None
        if envelope.rules and interference_calibration_authority is None:
            interference_bootstrap_authority = _replay_interference_bootstrap_authority(
                registry=registry,
                inventory=inventory,
                activation=activation_replay.activation_artifact,
                envelope=envelope,
                receipt_source=self.interference_source_receipt,
            )
        budgets = budget_plan.diagnostic_budgets
        completion_authorities = (
            *activation_replay.prior_e2_stage_authorities,
            *activation_replay.prior_family_authorities,
        )
        completed_cell_ids = tuple(
            sorted(
                cell_id
                for authority in completion_authorities
                for cell_id in authority.derive_completed_cell_ids()
            )
        )
        if len(completed_cell_ids) != len(set(completed_cell_ids)):
            raise ValueError("bundle completion authorities overlap cells")
        context_arguments = {
            "registry": registry,
            "inventory": inventory,
            "interference_envelope": envelope,
            "budgets": budgets,
            "receipts": receipts,
            "completed_cell_ids": completed_cell_ids,
            "activation_artifact": activation_replay.activation_artifact,
            "family_activations": activation_replay.family_activations,
            "family_power_reductions": activation_replay.family_power_reductions,
            "port_start": _context_integer(self.dispatch_context, "port_start"),
            "port_end": _context_integer(self.dispatch_context, "port_end"),
            "seed": _context_integer(self.dispatch_context, "seed"),
        }
        if diagnostic:
            context: GpuDispatchPlanningContext = GpuDispatchPlanningContext(
                **context_arguments
            )
            capacity_authority = budget_plan.capacity_authority
            if capacity_authority is None:  # pragma: no cover - raw bundle invariant
                raise RuntimeError("raw capacity authority disappeared during audit")
            expected_context_value = context.authority_dict()
            expected_context_value.update(
                {
                    "schema_version": 4,
                    "kind": "gpu_dispatch_execution_context",
                    "interference_calibration_authority_sha256": (
                        None
                        if interference_calibration_authority is None
                        else interference_calibration_authority.sha256
                    ),
                    "interference_calibration_bootstrap_authority_sha256": (
                        None
                        if interference_bootstrap_authority is None
                        else interference_bootstrap_authority.sha256
                    ),
                    "budget_plan_sha256": budget_plan.sha256,
                    "capacity_authority_sha256": capacity_authority.sha256,
                    "budget_materialization_authority_sha256": (
                        budget_materialization_authority.sha256
                    ),
                    "completion_authority_sha256s": [
                        authority.sha256 for authority in completion_authorities
                    ],
                }
            )
        else:
            _require_ready_budget_authority(
                budget_plan,
                authority=budget_materialization_authority,
                registry=registry,
                inventory=inventory,
                activation=activation,
            )
            context = GpuDispatchExecutionContext(
                **{
                    name: value
                    for name, value in context_arguments.items()
                    if name != "completed_cell_ids"
                },
                budget_plan=budget_plan,
                budget_materialization_authority=(budget_materialization_authority),
                interference_calibration_authority=(interference_calibration_authority),
                interference_calibration_bootstrap_authority=(
                    interference_bootstrap_authority
                ),
                completion_authorities=completion_authorities,
            )
            expected_context_value = context.authority_dict()
        context_value = self.dispatch_context.load()
        if context_value != expected_context_value:
            raise ValueError(
                "dispatch-context summary differs from raw scheduler replay"
            )
        expected_context_sha256 = content_sha256(expected_context_value)
        if expected_context_sha256 != self.dispatch_context.semantic_sha256:
            raise ValueError("dispatch-context semantic identity mismatch")
        dispatch = GpuDispatchPlan.from_dict(
            self.dispatch_plan.load(), planning_context=context
        )
        expected_dispatch = (
            GpuDispatchPlanningContext.issue_plan(context)
            if diagnostic
            else context.issue_plan()
        )
        if dispatch != expected_dispatch:
            raise ValueError("dispatch artifact differs from exact scheduler replay")
        if dispatch.sha256 != self.dispatch_plan.semantic_sha256:
            raise ValueError("dispatch artifact semantic identity mismatch")
        assignment = _one_assignment(dispatch, self.assignment_sha256)
        if assignment.work_item.item_id != self.cell_id:
            raise ValueError("execution bundle assignment names another cell")
        cell = assignment.work_item.cell
        if cell.resources.workload_class in {
            WorkloadClass.COMPILE,
            WorkloadClass.DOWNLOAD,
        }:
            raise ExecutionBundleBlockedError(
                "current_release_serving_execution_bundle_required"
            )
        failure_authority_sha256, failure_execution_authority = (
            _require_bundle_failure_injection_authority(
                registry=registry,
                cell=cell,
                binding=self.failure_injection_authority,
                diagnostic=diagnostic,
            )
        )
        budget = _one_budget(budgets, self.cell_id)
        run_config = load_run_config(self.run_config.path)
        self.run_config.load()
        if (
            run_config_sha256(run_config) != self.run_config.semantic_sha256
            or run_config.model_dump(mode="json") != self.run_config.load()
        ):
            raise ValueError("run-config artifact changed or is noncanonical")
        topology = _topology_from_dict(self.topology_receipts.load())
        if topology.receipt_sha256 != self.topology_receipts.semantic_sha256:
            raise ValueError("topology receipt semantic identity mismatch")
        topology_devices = tuple(
            receipt.topology.device_id for receipt in topology.receipts
        )
        if (
            topology_devices != assignment.gpu_uuids
            or tuple(receipt.observed_world_size for receipt in topology.receipts)
            != (len(assignment.gpu_uuids),) * len(assignment.gpu_uuids)
            or run_config.runtime.device_identity != assignment.gpu_uuids[0]
        ):
            raise ValueError(
                "topology/run config differs from the planned physical assignment"
            )
        runtime = None
        load = production_load_plan_from_dict(self.production_load.load())
        if load.paired_replay_sha256 != self.production_load.semantic_sha256:
            raise ValueError("production-load semantic identity mismatch")
        selected_load_bindings = tuple(
            binding
            for binding in budget_load_bindings
            if binding.cell_id == self.cell_id
        )
        if (
            len(selected_load_bindings) != 1
            or load != selected_load_bindings[0].registered_load
            or load.paired_replay_sha256
            != selected_load_bindings[0].registered_load.paired_replay_sha256
        ):
            raise ValueError(
                "execution production load differs from the registered budget load"
            )
        execution_semantics = _resolve_bundle_execution_semantics(
            registry=registry,
            activation_replay=activation_replay,
            load_binding=selected_load_bindings[0],
            cell=cell,
            run_config=run_config,
            diagnostic=diagnostic,
        )
        compile_plan = CompileCacheLaunchPlan.load(self.compile_cache_plan.path)
        self.compile_cache_plan.load()
        if compile_plan.sha256 != self.compile_cache_plan.semantic_sha256:
            raise ValueError("compile-cache plan semantic identity mismatch")
        preflight_compile_cache_launch(compile_plan)
        sampling = SamplingProfile.load(self.sampling_artifact.source.path)
        self.sampling_artifact.source.load()
        if sampling.sha256 != self.sampling_artifact.source.semantic_sha256:
            raise ValueError("sampling artifact semantic identity mismatch")
        model_lock = ModelLock.load(self.model_lock_artifact.source.path)
        self.model_lock_artifact.source.load()
        if model_lock.sha256 != self.model_lock_artifact.source.semantic_sha256:
            raise ValueError("model-lock semantic identity mismatch")
        prepared_models = self.prepared_models.load()
        parameter_plan = _require_bundle_trainable_plan_authority(
            bundle=self,
            cell=cell,
            run_config=run_config,
            model_lock=model_lock,
            prepared_models=prepared_models,
            execution_semantics=execution_semantics,
            formal=not diagnostic,
        )
        if not diagnostic:
            if type(context) is not GpuDispatchExecutionContext:  # pragma: no cover
                raise RuntimeError("formal replay lost its execution context")
            runtime = render_assigned_industrial_cell_runtime_plan(
                registry=registry,
                cell_id=self.cell_id,
                assignment=assignment,
                dispatch_plan=dispatch,
                dispatch_context=context,
                budget=budget,
                inventory=inventory,
                dispatch_inventory_sha256=inventory.sha256,
                rank_configs=(run_config,),
                topology_receipts=topology,
                dependency_receipts=receipts,
                parameter_plan=parameter_plan,
                execution_semantics=execution_semantics,
            )
        writer_policy, startup, shutdown, abort, controlled_policy = (
            _execution_policy_from_dict(self.execution_policy.load())
        )
        if controlled_policy.sha256 != run_config.runtime.execution_policy_sha256:
            raise ValueError(
                "assignment execution policy differs from the bound RunConfig"
            )
        dependency_artifacts = tuple(
            artifact.artifact_binding() for artifact in self.dependency_artifacts
        )
        expected_dependency_outputs = tuple(
            sorted(
                (receipt.experiment, output.name, output.content_sha256)
                for receipt in receipts
                for output in receipt.outputs
            )
        )
        actual_dependency_outputs = tuple(
            sorted(
                (
                    str(artifact.experiment),
                    artifact.name,
                    artifact.content_sha256,
                )
                for artifact in dependency_artifacts
            )
        )
        if actual_dependency_outputs != expected_dependency_outputs:
            raise ValueError(
                "dependency artifacts do not cover the locked outputs exactly"
            )
        split = self.split_artifact.artifact_binding()
        expected_execution_split = industrial_execution_split_contract(
            registry_sha256=registry.sha256,
            cell=cell,
            load_plan=load,
            sampling_profile_sha256=sampling.sha256,
            model_lock_sha256=model_lock.sha256,
        )
        if (
            self.activation_split.semantic_sha256 != activation.split_sha256
            or self.split_artifact.source.load() != expected_execution_split
            or split.content_sha256 != content_sha256(expected_execution_split)
        ):
            raise ValueError("execution or activation split identity changed")
        sampling_binding = self.sampling_artifact.artifact_binding()
        model_lock_binding = self.model_lock_artifact.artifact_binding()
        inventory_source = self.inventory_source_artifact.artifact_binding()
        runtime_envelope = self.runtime_envelope_artifact.artifact_binding()
        if inventory_source.content_sha256 != inventory.source_receipt_sha256:
            raise ValueError(
                "inventory source artifact differs from the physical inventory"
            )
        summary = None
        if verify_declared_summary:
            summary = self.execution_plan_summary.load()
            summary_sha256 = hashlib.sha256(_canonical_bytes(summary)).hexdigest()
            if (
                summary_sha256 != self.execution_plan_sha256
                or self.execution_plan_summary.semantic_sha256
                != self.execution_plan_sha256
            ):
                raise ValueError(
                    "declared execution-plan summary identity is inconsistent"
                )
        expected_root = Path(cell.resources.evidence_root).resolve()
        if Path(self.output_root) != expected_root:
            raise ValueError("bundle output root differs from registry reservation")
        if launch_materialization_policy is None:
            launch = _server_launch_from_dict(self.server_launch.load())
        else:
            if launch_materialization_root is None:  # pragma: no cover - paired above
                raise RuntimeError("materialization render root disappeared")
            render_root = Path(launch_materialization_root)
            render_root_parent = render_root.parent
            if (
                not render_root.is_absolute()
                or render_root.resolve() != render_root
                or render_root.is_symlink()
                or render_root == expected_root
                or render_root.is_relative_to(expected_root)
                or expected_root.is_relative_to(render_root)
            ):
                raise ValueError(
                    "materialization render root must be a separate resolved directory"
                )
            if materialization_preflight_only:
                # The publication and assignment roots are intentionally absent
                # during the all-assignment preflight.  Validate their nearest
                # existing owner-private ancestor and require both future path
                # components to be fresh, including broken symlinks.
                if os.path.lexists(render_root) or os.path.lexists(render_root_parent):
                    raise ValueError(
                        "materialization render root must belong to a fresh publication"
                    )
                publication_parent = render_root_parent.parent
                if (
                    publication_parent.is_symlink()
                    or not publication_parent.is_dir()
                    or publication_parent.resolve() != publication_parent
                ):
                    raise ValueError(
                        "materialization publication parent must be resolved"
                    )
                publication_parent_metadata = publication_parent.stat(
                    follow_symlinks=False
                )
                if (
                    publication_parent_metadata.st_uid != os.geteuid()
                    or publication_parent_metadata.st_mode & 0o077
                ):
                    raise ValueError(
                        "materialization publication parent must be release-private"
                    )
            else:
                if (
                    not render_root.is_dir()
                    or render_root_parent.is_symlink()
                    or render_root_parent.resolve() != render_root_parent
                    or not render_root_parent.is_dir()
                ):
                    raise ValueError(
                        "materialization render root must be a separate resolved directory"
                    )
                render_metadata = render_root.stat(follow_symlinks=False)
                parent_metadata = render_root_parent.stat(follow_symlinks=False)
                if (
                    render_metadata.st_uid != os.geteuid()
                    or render_metadata.st_mode & 0o077
                    or parent_metadata.st_uid != os.geteuid()
                    or parent_metadata.st_mode & 0o077
                ):
                    raise ValueError(
                        "materialization render root must be release-private"
                    )
            roots = _validate_prepared_model_sources(
                model_lock=model_lock,
                prepared=prepared_models,
                run_config=run_config,
            )
            # This is the first filesystem-mutating step: the renderer writes
            # RunConfig/adaptation files.  All raw replay above and the
            # source-owned release trust gate must therefore pass first.
            adapted = run_config.method not in {"target_only", "static"}
            if adapted != (launch_materialization_policy.adaptation_reserve_mb > 0):
                raise ValueError(
                    "adaptation reserve must be positive exactly for adapted methods"
                )
            adaptation_payload = sglang_adaptation_payload(run_config)
            if adapted != (adaptation_payload is not None):
                raise ValueError(
                    "RunConfig method and rendered adaptation payload differ"
                )
            # These pure reducers cover every method/controlled-policy branch
            # that the renderer will use after its first immutable write.
            execution_role = _execution_role(run_config.method)
            _execution_argv(run_config.runtime, role=execution_role)
            verified_checkout = verify_patched_checkout(
                launch_materialization_policy.patched_sglang_checkout
            )
            require_release_dispatch_execution_authority()
            if materialization_preflight_only:
                # Renderer I/O and validations of those freshly rendered files
                # are the only intentionally deferred operations.
                return None
            launch = _render_server(
                output=render_root,
                method=run_config.method,
                config=run_config,
                verified_checkout=verified_checkout,
                roots=roots,
                target_id=run_config.model.target,
                drafter_id=run_config.model.drafter,
                adaptation_reserve_mb=(
                    launch_materialization_policy.adaptation_reserve_mb
                ),
                mem_fraction_static=(launch_materialization_policy.mem_fraction_static),
                host=launch_materialization_policy.host,
                port=assignment.ports[0],
                compile_cache_plan_path=self.compile_cache_plan.path,
            )
            launch = replace(launch, run_config=self.run_config.path)
        if (
            launch.run_config != self.run_config.path
            or launch.compile_cache_plan != self.compile_cache_plan.path
        ):
            raise ValueError("server launch swapped a bound execution source")
        _validate_model_inputs(
            model_lock=model_lock,
            prepared=prepared_models,
            run_config=run_config,
            launch=launch,
        )
        if diagnostic:
            component_replay_sha256 = content_sha256(
                {
                    "schema_version": 1,
                    "kind": "industrial_execution_bundle_component_replay",
                    "registry_sha256": registry.sha256,
                    "inventory_sha256": inventory.sha256,
                    "interference_envelope_sha256": envelope.sha256,
                    "budget_plan_sha256": budget_plan.sha256,
                    "budget_materialization_authority_sha256": (
                        budget_materialization_authority.sha256
                    ),
                    "activation_sha256": activation.sha256,
                    "dispatch_context_sha256": expected_context_sha256,
                    "dispatch_plan_sha256": dispatch.sha256,
                    "assignment_sha256": assignment.assignment_id,
                    "topology_receipt_sha256": topology.receipt_sha256,
                    "production_load_sha256": load.paired_replay_sha256,
                    "execution_semantics_authority": "diagnostic_non_authority",
                    "execution_semantics_sha256": (
                        None
                        if execution_semantics is None
                        else execution_semantics.sha256
                    ),
                    "run_config_sha256": run_config_sha256(run_config),
                    "server_launch_sha256": content_sha256(
                        server_launch_to_dict(launch)
                    ),
                    "dependency_artifact_sha256s": [
                        artifact.content_sha256 for artifact in dependency_artifacts
                    ],
                    "split_sha256": split.content_sha256,
                    "sampling_sha256": sampling_binding.content_sha256,
                    "model_lock_sha256": model_lock_binding.content_sha256,
                    "prepared_models_sha256": content_sha256(prepared_models),
                    "trainable_plan_authority_sha256": (
                        None
                        if self.trainable_plan_authority is None
                        else self.trainable_plan_authority.sha256
                    ),
                    "failure_injection_authority_sha256": (failure_authority_sha256),
                    "prepared_model_content_release_manifest_sha256": (
                        self.prepared_model_content_release_manifest_sha256
                    ),
                    "parameter_plan_sha256": (
                        None if parameter_plan is None else parameter_plan.sha256
                    ),
                    "compile_plan_sha256": compile_plan.sha256,
                    "inventory_source_sha256": inventory_source.content_sha256,
                    "runtime_envelope_sha256": runtime_envelope.content_sha256,
                    "execution_policy_sha256": content_sha256(
                        self.execution_policy.load()
                    ),
                }
            )
            return IndustrialExecutionPlanAudit(
                schema_version=2,
                kind="industrial_execution_plan_audit",
                bundle_sha256=self.sha256,
                assignment_sha256=self.assignment_sha256,
                cell_id=self.cell_id,
                budget_plan_sha256=budget_plan.sha256,
                budget_plan_status=budget_plan.status,
                budget_materialization_authority_sha256=(
                    budget_materialization_authority.sha256
                ),
                component_replay_sha256=component_replay_sha256,
                dispatch_plan_sha256=dispatch.sha256,
                execution_semantics_sha256=(
                    None if execution_semantics is None else execution_semantics.sha256
                ),
                execution_semantics_authority="diagnostic_non_authority",
                exact_dispatch_replay=True,
                execution_plan_sha256=None,
                execution_plan_status="NOT_VALIDATED",
                execution_plan_reason_code=(
                    "audit_does_not_construct_physical_execution_plan"
                ),
            )
        if type(context) is not GpuDispatchExecutionContext:  # pragma: no cover
            raise RuntimeError("formal replay lost its execution context")
        if runtime is None:  # pragma: no cover
            raise RuntimeError("formal replay lost its physical runtime plan")
        try:
            plan = build_industrial_execution_plan(
                runtime_plan=runtime,
                dispatch_plan=dispatch,
                dispatch_context=context,
                budget_plan=budget_plan,
                budget=budget,
                load_plan=load,
                server_launch=launch,
                dependency_receipts=receipts,
                dependency_artifacts=dependency_artifacts,
                split_artifact=split,
                sampling_artifact=sampling_binding,
                model_lock_artifact=model_lock_binding,
                compile_cache_plan=compile_plan,
                inventory_source_artifact=inventory_source,
                runtime_envelope_artifact=runtime_envelope,
                trainable_plan_authority=self.trainable_plan_authority,
                failure_execution_authority=failure_execution_authority,
                prepared_model_content_release_manifest_sha256=(
                    self.prepared_model_content_release_manifest_sha256
                ),
                evidence_writer_policy=writer_policy,
                trusted_attester_policy=RELEASE_TRUSTED_ATTESTER_POLICY,
                startup_timeout_s=startup,
                shutdown_timeout_s=shutdown,
                abort_grace_s=abort,
            )
        except TrainablePlanExecutionBlockedError as error:
            raise ExecutionBundleBlockedError(error.reason_code) from error
        if verify_declared_summary:
            if summary is None:  # pragma: no cover - guarded above
                raise RuntimeError("declared execution summary disappeared")
            if (
                _canonical_bytes(summary) != _canonical_bytes(plan.to_dict())
                or self.execution_plan_summary.semantic_sha256 != plan.sha256
                or plan.sha256 != self.execution_plan_sha256
            ):
                raise ValueError(
                    "declared execution-plan summary differs from raw artifact replay"
                )
        return plan


def finalize_materialized_execution_bundle(
    provisional: IndustrialAssignmentExecutionBundle,
    *,
    launch_policy: AssignmentLaunchMaterializationPolicy,
    render_root: str | Path,
    server_launch: BoundJsonSource,
    execution_plan_summary: BoundJsonSource,
) -> IndustrialAssignmentExecutionBundle:
    """Replace the construction placeholder with the reducer-owned plan.

    The plan is reconstructed from the provisional bundle's raw sources; the
    supplied summary path is accepted only when its complete JSON body equals
    that newly constructed plan.  A second ordinary bundle replay then proves
    the final schema-v4 object is load-equivalent to its source construction.
    """

    if type(provisional) is not IndustrialAssignmentExecutionBundle:
        raise TypeError("bundle finalization requires an exact provisional bundle")
    if type(launch_policy) is not AssignmentLaunchMaterializationPolicy:
        raise TypeError("bundle finalization requires an exact launch policy")
    if type(server_launch) is not BoundJsonSource:
        raise TypeError("bundle finalization requires an exact server-launch source")
    if type(execution_plan_summary) is not BoundJsonSource:
        raise TypeError("bundle finalization requires an exact summary source")
    plan = provisional.reconstruct_execution_plan_for_materialization(
        launch_policy,
        render_root=render_root,
    )
    launch_value = server_launch.load()
    if launch_value != server_launch_to_dict(
        plan.server_launch
    ) or server_launch.canonical_sha256 != content_sha256(launch_value):
        raise ValueError("materialized server launch differs from raw replay")
    summary = execution_plan_summary.load()
    if (
        summary != plan.to_dict()
        or execution_plan_summary.canonical_sha256 != plan.sha256
        or execution_plan_summary.semantic_sha256 != plan.sha256
    ):
        raise ValueError("materialized execution-plan summary differs from raw replay")
    final = replace(
        provisional,
        execution_plan_sha256=plan.sha256,
        server_launch=server_launch,
        execution_plan_summary=execution_plan_summary,
    )
    if final.reconstruct_execution_plan() != plan:
        raise RuntimeError("finalized execution bundle changed its reconstructed plan")
    return final


def _replay_interference_authority(
    *,
    inventory: GpuInventory,
    envelope: InterferenceEnvelope,
    envelope_source: BoundJsonSource,
    receipt_source: BoundJsonSource,
    calibration_source: InterferenceCalibrationExecutionAuthority | None,
) -> InterferenceCalibrationAuthority | None:
    """Replay serial or calibrated interference authority from raw files."""

    if envelope.rules:
        if calibration_source is None:
            return None
        raw_manifest = calibration_source.source.manifest
        if (
            receipt_source.path != raw_manifest.path
            or receipt_source.canonical_sha256 != raw_manifest.semantic_sha256
            or receipt_source.semantic_sha256 != raw_manifest.semantic_sha256
            or receipt_source.file_sha256 != raw_manifest.file_sha256
            or receipt_source.sidecar_file_sha256 != raw_manifest.sidecar_file_sha256
            or receipt_source.size != raw_manifest.size
        ):
            raise ValueError(
                "calibrated interference source receipt must bind its raw manifest"
            )
        authority = calibration_source.reconstruct()
        try:
            require_calibrated_interference_execution_authority(
                envelope,
                authority=authority,
            )
        except InterferenceCalibrationBlockedError as exc:
            raise ExecutionBundleBlockedError(exc.reason_code) from exc
        if envelope.sha256 != envelope_source.semantic_sha256:
            raise ValueError("calibrated interference envelope identity mismatch")
        return authority

    if calibration_source is not None:
        raise ValueError("serial interference envelope cannot carry calibration")

    receipt = receipt_source.load()
    expected_envelope, expected_receipt = build_serial_interference_envelope(inventory)
    if receipt != expected_receipt:
        raise ExecutionBundleBlockedError(
            "calibrated_interference_raw_authority_required"
        )
    receipt_sha256 = expected_receipt["receipt_sha256"]
    if (
        receipt_source.semantic_sha256 != receipt_sha256
        or envelope.source_receipt_sha256 != receipt_sha256
        or envelope != expected_envelope
        or envelope.sha256 != envelope_source.semantic_sha256
    ):
        raise ValueError("interference envelope differs from its raw serial receipt")
    return None


def _replay_interference_bootstrap_authority(
    *,
    registry: ExperimentRegistry,
    inventory: GpuInventory,
    activation: object,
    envelope: InterferenceEnvelope,
    receipt_source: BoundJsonSource,
):
    """Re-derive the calibration-only two-way envelope from raw stage inputs."""

    if type(activation) is not RegistryStageActivationArtifact:
        raise ExecutionBundleBlockedError(
            "calibrated_interference_raw_authority_required"
        )
    authority = materialize_interference_calibration_bootstrap_authority(
        registry,
        inventory,
        activation,
    )
    if envelope != authority.bootstrap_envelope:
        raise ExecutionBundleBlockedError(
            "calibrated_interference_raw_authority_required"
        )
    receipt = authority.source_receipt
    if (
        receipt_source.load() != receipt
        or receipt_source.semantic_sha256 != receipt["receipt_sha256"]
    ):
        raise ValueError("calibration bootstrap envelope differs from its raw receipt")
    return authority


def _rematerialize_budget_authority(
    *,
    registry: ExperimentRegistry,
    inventory: GpuInventory,
    activation_manifest_source: BoundJsonSource,
    registry_source: BoundJsonSource,
    activation_runtime_source: BoundJsonSource,
    activation_split_source: BoundJsonSource,
    dependency_receipt_sources: tuple[BoundJsonSource, ...],
    declared_source: BoundJsonSource,
    policy_source: BoundJsonSource,
    load_sources: tuple[BoundJsonSource, ...],
    capacity_source: BoundJsonSource,
    capacity_manifest_source: BoundJsonSource,
    capacity_verification_source: BoundJsonSource,
):
    capacity_authority = bind_capacity_authority(
        capacity_manifest_source.path,
        capacity_verification_source.path,
    )
    authority = bind_budget_materialization_authority(
        activation_manifest_path=activation_manifest_source.path,
        policy_path=policy_source.path,
        load_binding_paths=tuple(source.path for source in load_sources),
        capacity_envelope_path=capacity_source.path,
        capacity_authority=capacity_authority,
        declared_plan_path=declared_source.path,
    )
    activation_replay = replay_budget_activation_authority(authority.activation)
    replayed_registry = activation_replay.registry
    # The binder itself reopens every source and exactly reruns the first-party
    # reducer.  Reopen the declared/result rows through their strict codecs;
    # the formal READY boundary below independently revalidates the authority
    # again immediately before constructing an execution context.
    budget_plan = load_declared_budget_plan(authority)
    budget_load_bindings = tuple(
        budget_load_binding_from_dict(source.load()) for source in load_sources
    )
    expected_inventory = budget_inventory_identity_from_gpu_inventory(inventory)
    if (
        replayed_registry != registry
        or authority.registry_sha256 != registry.sha256
        or authority.activation_sha256 != activation_replay.activation_sha256
        or authority.budget_inventory_sha256 != expected_inventory.sha256
        or budget_plan.registry_sha256 != registry.sha256
        or budget_plan.inventory != expected_inventory
        or budget_plan.reducer_activation_sha256s
        != tuple(
            sorted(
                ()
                if activation_replay.activation_artifact is None
                else (activation_replay.activation_artifact.sha256,)
            )
        )
        or budget_plan.family_activation_sha256s
        != tuple(sorted(value.sha256 for value in activation_replay.family_activations))
        or budget_plan.family_power_reduction_sha256s
        != tuple(
            sorted(value.sha256 for value in activation_replay.family_power_reductions)
        )
    ):
        raise ValueError(
            "budget materialization authority differs from bundle execution inputs"
        )
    raw_pairs = (
        (
            "activation manifest",
            activation_manifest_source,
            authority.activation.manifest,
        ),
        (
            "generated registry",
            registry_source,
            authority.activation.generated_registry,
        ),
        ("declared BudgetPlan", declared_source, authority.declared_plan),
        ("budget policy", policy_source, authority.policy),
        ("capacity envelope", capacity_source, authority.capacity_envelope),
    )
    for label, source, binding in raw_pairs:
        _compare_budget_raw_binding(label, source, binding)
    runtime_binding = _one_activation_raw_binding(
        authority,
        role="activation_runtime",
        semantic_sha256=activation_replay.runtime_sha256,
    )
    split_binding = _one_activation_raw_binding(
        authority,
        role="activation_split",
        semantic_sha256=activation_replay.split_sha256,
    )
    _compare_budget_raw_binding(
        "activation runtime", activation_runtime_source, runtime_binding
    )
    _compare_budget_raw_binding(
        "activation split", activation_split_source, split_binding
    )
    if len(dependency_receipt_sources) != len(activation_replay.dependency_receipts):
        raise ValueError("budget activation dependency raw coverage differs")
    for source, receipt in zip(
        dependency_receipt_sources,
        activation_replay.dependency_receipts,
        strict=True,
    ):
        receipt_binding = _one_activation_raw_binding(
            authority,
            role="activation_dependency_receipt",
            semantic_sha256=receipt.sha256,
        )
        _compare_budget_raw_binding(
            "activation dependency receipt", source, receipt_binding
        )
    if len(load_sources) != len(authority.load_bindings):
        raise ValueError("budget load-binding raw authority coverage differs")
    for source, binding in zip(
        load_sources,
        authority.load_bindings,
        strict=True,
    ):
        _compare_budget_raw_binding("budget load binding", source, binding.source)
    for label, source, binding in (
        (
            "capacity source manifest",
            capacity_manifest_source,
            capacity_authority.source_manifest,
        ),
        (
            "capacity verification receipt",
            capacity_verification_source,
            capacity_authority.verification_receipt,
        ),
    ):
        if (
            source.path != binding.path
            or source.canonical_sha256 != binding.semantic_sha256
            or source.semantic_sha256 != binding.semantic_sha256
            or source.file_sha256 != binding.file_sha256
            or source.sidecar_file_sha256 != binding.sidecar_file_sha256
            or source.size != binding.size
        ):
            raise ValueError(f"{label} differs from its raw capacity binding")
    if tuple(value.cell_id for value in budget_load_bindings) != tuple(
        value.cell_id for value in authority.load_bindings
    ):
        raise ValueError("budget load cells differ from raw materialization authority")
    return budget_plan, budget_load_bindings, authority, activation_replay


def _resolve_bundle_execution_semantics(
    *,
    registry: ExperimentRegistry,
    activation_replay: object,
    load_binding: object,
    cell: ExperimentCell,
    run_config: object,
    diagnostic: bool,
):
    """Derive the E1 overlay only after caller-owned raw replay.

    The returned value is scientific identity.  It is never a release token;
    diagnostic callers additionally label it as non-authority in their audit.
    Stages whose source-owned execution semantics are not implemented retain a
    stable formal BLOCK instead of falling back to registry placeholders.
    """

    if cell.identity.experiment == "E1":
        from lightcone_spec.experiments.execution_semantics import (
            CellExecutionSemanticsBlockedError,
            resolve_cell_execution_semantics,
        )

        try:
            semantics = resolve_cell_execution_semantics(
                activation=activation_replay,
                load_binding=load_binding,
                cell=cell,
            )
            semantics = _require_registered_e1_execution_recipe(
                registry=registry,
                cell=cell,
                execution_semantics=semantics,
            )
            semantics.validate_run_config(run_config)
        except CellExecutionSemanticsBlockedError as error:
            raise ExecutionBundleBlockedError(error.reason_code) from error
        return semantics
    if cell.identity.experiment in {"E2", "E3b", "E5"} and not diagnostic:
        raise ExecutionBundleBlockedError(
            "cell_execution_semantics_experiment_unsupported"
        )
    return None


def _one_activation_raw_binding(
    authority: BudgetMaterializationAuthorityBinding,
    *,
    role: str,
    semantic_sha256: str,
) -> BudgetRawJsonBinding:
    matches = tuple(
        source
        for source in _budget_activation_raw_sources(authority.activation)
        if source.role == role and source.semantic_sha256 == semantic_sha256
    )
    if len(matches) != 1:
        raise ValueError(f"activation raw {role} binding is absent or path-ambiguous")
    return matches[0]


def _compare_budget_raw_binding(
    label: str,
    source: BoundJsonSource,
    binding: BudgetRawJsonBinding,
) -> None:
    """Require both authority layers to name the same immutable raw bytes."""

    if type(source) is not BoundJsonSource:
        raise TypeError(f"{label} requires one exact bundle source")
    if type(binding) is not BudgetRawJsonBinding:
        raise TypeError(f"{label} requires one exact budget raw binding")
    if (
        source.path != binding.path
        or binding.sidecar_path != f"{source.path}.sha256"
        or source.canonical_sha256 != binding.canonical_sha256
        or source.file_sha256 != binding.file_sha256
        or source.sidecar_file_sha256 != binding.sidecar_file_sha256
        or source.size != binding.size
        or binding.sidecar_size != 65
    ):
        raise ValueError(f"{label} differs from its raw budget binding")
    # An activation manifest is canonically bound by the budget reducer while
    # the bundle records the reducer output produced from that manifest.  The
    # two values intentionally inhabit different semantic domains.  Every
    # non-manifest role must retain the same semantic identity in both layers.
    if (
        binding.role not in _ACTIVATION_MANIFEST_ROLES
        and source.semantic_sha256 != binding.semantic_sha256
    ):
        raise ValueError(f"{label} semantic identity differs")


def _require_ready_budget_authority(
    plan: BudgetPlan,
    *,
    authority: BudgetMaterializationAuthorityBinding,
    registry: ExperimentRegistry,
    inventory: GpuInventory,
    activation,
) -> None:
    """Replay every raw reducer input and translate a named prelaunch block."""

    if type(plan) is not BudgetPlan:
        raise TypeError("formal execution requires one exact BudgetPlan")
    try:
        result = require_ready_budget_materialization_authority_binding(
            authority,
            expected_registry=registry,
            expected_inventory=budget_inventory_identity_from_gpu_inventory(inventory),
            expected_activation=activation,
            expected_plan=plan,
            expected_gpu_inventory=inventory,
        )
    except (
        BudgetMaterializationBlockedError,
        CapacityAuthorityUnavailableError,
    ) as error:
        raise ExecutionBundleBlockedError(error.reason_code) from error
    if result.budget_plan != plan:  # pragma: no cover - reducer postcondition
        raise RuntimeError("ready budget authority returned another BudgetPlan")


def _context_integer(source: BoundJsonSource, name: str) -> int:
    value = _strict_object(
        "dispatch execution context",
        source.load(),
        frozenset(
            {
                "schema_version",
                "kind",
                "registry_sha256",
                "inventory_sha256",
                "interference_envelope_sha256",
                "interference_calibration_authority_sha256",
                "interference_calibration_bootstrap_authority_sha256",
                "budget_sha256s",
                "receipt_sha256s",
                "completed_cell_ids",
                "activation_artifact_sha256",
                "family_activation_sha256s",
                "family_power_reduction_sha256s",
                "budget_plan_sha256",
                "capacity_authority_sha256",
                "budget_materialization_authority_sha256",
                "port_start",
                "port_end",
                "seed",
                "completion_authority_sha256s",
            }
        ),
    )
    if value["schema_version"] != 4 or value["kind"] != _CONTEXT_KIND:
        raise ValueError("bundle requires an execution dispatch context")
    calibration_sha256 = value["interference_calibration_authority_sha256"]
    if calibration_sha256 is not None:
        _require_sha256("dispatch context interference calibration", calibration_sha256)
    bootstrap_sha256 = value["interference_calibration_bootstrap_authority_sha256"]
    if bootstrap_sha256 is not None:
        _require_sha256("dispatch context interference bootstrap", bootstrap_sha256)
    _require_sha256("dispatch context budget plan", value["budget_plan_sha256"])
    _require_sha256(
        "dispatch context capacity authority", value["capacity_authority_sha256"]
    )
    _require_sha256(
        "dispatch context budget materialization authority",
        value["budget_materialization_authority_sha256"],
    )
    return _strict_int(f"dispatch context {name}", value[name], minimum=0)


def _load_registry(value: object) -> ExperimentRegistry:
    row = _strict_object(
        "industrial registry artifact",
        value,
        frozenset(
            {
                "schema_version",
                "generator",
                "parameters",
                "registry_sha256",
                "registry",
            }
        ),
    )
    if row["schema_version"] != 2 or row["generator"] != _REGISTRY_GENERATOR:
        raise ValueError("industrial registry generator identity mismatch")
    parameters = _strict_object(
        "industrial registry parameters",
        row["parameters"],
        frozenset(
            {
                "logical_gpu_slots",
                "base_port",
                "cache_root",
                "evidence_root",
                "seed",
            }
        ),
    )
    slots = tuple(
        _strict_text("logical GPU slot", item)
        for item in _strict_list(
            "industrial registry logical slots", parameters["logical_gpu_slots"]
        )
    )
    if not slots or len(slots) != len(set(slots)):
        raise ValueError("industrial registry logical slots are invalid")
    registry = build_industrial_registry(
        gpu_uuids=slots,
        base_port=_strict_int("registry base_port", parameters["base_port"]),
        cache_root=_strict_text("registry cache_root", parameters["cache_root"]),
        evidence_root=_strict_text(
            "registry evidence_root", parameters["evidence_root"]
        ),
        seed=_strict_int("registry seed", parameters["seed"], minimum=0),
    )
    if (
        row["registry_sha256"] != registry.sha256
        or row["registry"] != registry.to_dict()
    ):
        raise ValueError("industrial registry differs from generator replay")
    return registry


def _locked_output_from_dict(value: object) -> LockedOutput:
    row = _strict_object("locked output", value, frozenset({"name", "content_sha256"}))
    return LockedOutput(
        name=_strict_text("locked output name", row["name"]),
        content_sha256=_require_sha256("locked output content", row["content_sha256"]),
    )


def _receipt_from_dict(value: object) -> ExperimentReceipt:
    row = _strict_object(
        "experiment receipt",
        value,
        frozenset(
            {
                "experiment",
                "registry_sha256",
                "runtime_sha256",
                "split_sha256",
                "completed_cells_sha256",
                "dependency_receipts",
                "outputs",
                "selection_state",
            }
        ),
    )
    receipt = ExperimentReceipt(
        experiment=_strict_text("receipt experiment", row["experiment"]),
        registry_sha256=_require_sha256("receipt registry", row["registry_sha256"]),
        runtime_sha256=_require_sha256("receipt runtime", row["runtime_sha256"]),
        split_sha256=_require_sha256("receipt split", row["split_sha256"]),
        completed_cells_sha256=_require_sha256(
            "receipt completed cells", row["completed_cells_sha256"]
        ),
        dependency_receipts=tuple(
            _locked_output_from_dict(item)
            for item in _strict_list("receipt dependencies", row["dependency_receipts"])
        ),
        outputs=tuple(
            _locked_output_from_dict(item)
            for item in _strict_list("receipt outputs", row["outputs"])
        ),
        selection_state=_strict_text("receipt selection state", row["selection_state"]),
    )
    if receipt.to_dict() != value:
        raise ValueError("experiment receipt is not canonical")
    return receipt


def topology_receipt_set_to_dict(value: TopologyReceiptSet) -> dict[str, object]:
    """Return a complete topology receipt rather than a digest summary."""

    if type(value) is not TopologyReceiptSet:
        raise TypeError("topology receipt set must be exact")
    receipts = tuple(sorted(value.receipts, key=lambda row: row.topology.global_rank))
    return {
        "schema_version": 1,
        "kind": _TOPOLOGY_KIND,
        "receipts": [
            {
                "topology": asdict(receipt.topology),
                "process_id": receipt.process_id,
                "observed_world_size": receipt.observed_world_size,
                "receipt_sha256": receipt.sha256,
            }
            for receipt in receipts
        ],
        "topology_sha256": value.topology_sha256,
        "receipt_set_sha256": value.receipt_sha256,
    }


def _topology_from_dict(value: object) -> TopologyReceiptSet:
    row = _strict_object(
        "topology receipt set",
        value,
        frozenset(
            {
                "schema_version",
                "kind",
                "receipts",
                "topology_sha256",
                "receipt_set_sha256",
            }
        ),
    )
    if row["schema_version"] != 1 or row["kind"] != _TOPOLOGY_KIND:
        raise ValueError("topology receipt-set schema is unsupported")
    topology_fields = frozenset(TopologyIdentity.__dataclass_fields__)
    receipts: list[RankTopologyReceipt] = []
    for item in _strict_list("topology receipts", row["receipts"]):
        receipt_row = _strict_object(
            "rank topology receipt",
            item,
            frozenset(
                {
                    "topology",
                    "process_id",
                    "observed_world_size",
                    "receipt_sha256",
                }
            ),
        )
        topology_row = _strict_object(
            "rank topology identity", receipt_row["topology"], topology_fields
        )
        topology = TopologyIdentity(**topology_row)
        receipt = RankTopologyReceipt(
            topology=topology,
            process_id=_strict_text("topology process ID", receipt_row["process_id"]),
            observed_world_size=_strict_int(
                "topology observed world size",
                receipt_row["observed_world_size"],
                minimum=1,
            ),
        )
        if receipt.sha256 != receipt_row["receipt_sha256"]:
            raise ValueError("rank topology receipt SHA-256 mismatch")
        receipts.append(receipt)
    result = TopologyReceiptSet(tuple(receipts))
    if (
        row["topology_sha256"] != result.topology_sha256
        or row["receipt_set_sha256"] != result.receipt_sha256
        or topology_receipt_set_to_dict(result) != value
    ):
        raise ValueError("topology receipt-set identity mismatch")
    return result


def server_launch_to_dict(value: ServerLaunch) -> dict[str, object]:
    if type(value) is not ServerLaunch:
        raise TypeError("server launch must be exact")
    payload = asdict(value)
    payload["argv"] = list(value.argv)
    return {"schema_version": 1, "kind": _LAUNCH_KIND, **payload}


def _server_launch_from_dict(value: object) -> ServerLaunch:
    fields = frozenset(ServerLaunch.__dataclass_fields__) | frozenset(
        {"schema_version", "kind"}
    )
    row = _strict_object("server launch", value, fields)
    if row["schema_version"] != 1 or row["kind"] != _LAUNCH_KIND:
        raise ValueError("server-launch schema is unsupported")
    argv = tuple(
        _strict_text("server argv item", item)
        for item in _strict_list("server argv", row["argv"])
    )
    nullable = ("adaptation_config", "telemetry_path", "compile_cache_plan")
    nullable_sha = ("compile_cache_plan_sha256", "compile_cache_key_sha256")
    for name in nullable:
        if row[name] is not None:
            _strict_text(f"server launch {name}", row[name])
    for name in nullable_sha:
        if row[name] is not None:
            _require_sha256(f"server launch {name}", row[name])
    if type(row["exclusive_device"]) is not bool:
        raise TypeError("server launch exclusive_device must be boolean")
    launch = ServerLaunch(
        method=_strict_text("server launch method", row["method"]),
        base_url=_strict_text("server launch base URL", row["base_url"]),
        exclusive_device=row["exclusive_device"],
        run_config=_strict_text("server launch run config", row["run_config"]),
        adaptation_config=row["adaptation_config"],
        telemetry_path=row["telemetry_path"],
        argv=argv,
        compile_cache_plan=row["compile_cache_plan"],
        compile_cache_plan_sha256=row["compile_cache_plan_sha256"],
        compile_cache_key_sha256=row["compile_cache_key_sha256"],
    )
    if server_launch_to_dict(launch) != value:
        raise ValueError("server launch is not canonical")
    return launch


def execution_policy_to_dict(plan: IndustrialExecutionPlan) -> dict[str, object]:
    plan.validate()
    controlled = ControlledExecutionPolicy()
    if (
        plan.runtime_plan.rank_configs[0].runtime.execution_policy_sha256
        != controlled.sha256
    ):
        raise ValueError("execution plan uses an unregistered controlled policy")
    return {
        "schema_version": 1,
        "kind": _EXECUTION_POLICY_KIND,
        "patched_sglang_tree": plan.patched_sglang_tree,
        "evidence_writer_policy": plan.evidence_writer_policy.to_dict(),
        "evidence_writer_policy_sha256": plan.evidence_writer_policy.sha256,
        "controlled_execution_policy": controlled.to_dict(),
        "controlled_execution_policy_sha256": controlled.sha256,
        "startup_timeout_s": plan.startup_timeout_s,
        "shutdown_timeout_s": plan.shutdown_timeout_s,
        "abort_grace_s": plan.abort_grace_s,
    }


def _execution_policy_from_dict(
    value: object,
) -> tuple[
    EvidenceWriterPolicy,
    float,
    float,
    float,
    ControlledExecutionPolicy,
]:
    row = _strict_object(
        "assignment execution policy",
        value,
        frozenset(
            {
                "schema_version",
                "kind",
                "patched_sglang_tree",
                "evidence_writer_policy",
                "evidence_writer_policy_sha256",
                "controlled_execution_policy",
                "controlled_execution_policy_sha256",
                "startup_timeout_s",
                "shutdown_timeout_s",
                "abort_grace_s",
            }
        ),
    )
    if (
        row["schema_version"] != 1
        or row["kind"] != _EXECUTION_POLICY_KIND
        or row["patched_sglang_tree"] != PINNED_SGLANG_TREE
    ):
        raise ValueError("assignment execution policy identity mismatch")
    writer = EvidenceWriterPolicy.from_dict(row["evidence_writer_policy"])
    if writer.sha256 != row["evidence_writer_policy_sha256"]:
        raise ValueError("assignment writer-policy SHA-256 mismatch")
    controlled = ControlledExecutionPolicy.from_dict(row["controlled_execution_policy"])
    if controlled.sha256 != row["controlled_execution_policy_sha256"]:
        raise ValueError("assignment controlled-policy SHA-256 mismatch")
    return (
        writer,
        _strict_float("startup timeout", row["startup_timeout_s"]),
        _strict_float("shutdown timeout", row["shutdown_timeout_s"]),
        _strict_float("abort grace", row["abort_grace_s"]),
        controlled,
    )


def prepared_models_to_dict(
    model_lock: ModelLock,
    roots: dict[str, str],
) -> dict[str, object]:
    model_lock.validate()
    expected = tuple(sorted(model.model_id for model in model_lock.models))
    if tuple(sorted(roots)) != expected:
        raise ValueError("prepared model roots do not cover the model lock exactly")
    canonical_roots: dict[str, str] = {}
    for model_id in expected:
        root = Path(roots[model_id])
        if not root.is_absolute() or root.resolve() != root or not root.is_dir():
            raise ValueError(
                "prepared model root must be an existing resolved directory"
            )
        canonical_roots[model_id] = str(root)
    return {
        "schema_version": 1,
        "kind": _PREPARED_MODELS_KIND,
        "model_lock_sha256": model_lock.sha256,
        "roots": canonical_roots,
    }


def _require_trainable_raw_source_match(
    binding: TrainablePlanRawJsonBinding,
    source: BoundJsonSource,
    *,
    label: str,
) -> None:
    if (
        binding.path != source.path
        or binding.semantic_sha256 != source.canonical_sha256
        or binding.semantic_sha256 != source.semantic_sha256
        or binding.file_sha256 != source.file_sha256
        or binding.sidecar_file_sha256 != source.sidecar_file_sha256
        or binding.size != source.size
    ):
        raise ValueError(f"trainable-plan {label} differs from bundle raw source")


def _require_bundle_trainable_plan_authority(
    *,
    bundle: IndustrialAssignmentExecutionBundle,
    cell: ExperimentCell,
    run_config,
    model_lock: ModelLock,
    prepared_models: object,
    execution_semantics: object,
    formal: bool,
) -> TrainablePlan | None:
    method = cell.identity.method
    authority = bundle.trainable_plan_authority
    release_pin = bundle.prepared_model_content_release_manifest_sha256
    if method in {"target_only", "static"}:
        if authority is not None or release_pin is not None:
            raise ValueError(
                "Target-only/Static bundle must not carry trainable-plan authority"
            )
        return None
    if method not in {"tts", "l0"}:
        raise ExecutionBundleBlockedError(
            "current_release_core_trainable_plan_method_required"
        )
    from lightcone_spec.experiments.execution_semantics import (
        EXECUTION_SEMANTICS_RAW_ACTIVATION_UNAVAILABLE_REASON,
        EXECUTION_SEMANTICS_UNSUPPORTED_EXPERIMENT_REASON,
        CellExecutionSemantics,
    )

    if type(execution_semantics) is not CellExecutionSemantics:
        raise ExecutionBundleBlockedError(
            EXECUTION_SEMANTICS_RAW_ACTIVATION_UNAVAILABLE_REASON
            if cell.identity.experiment == "E1"
            else EXECUTION_SEMANTICS_UNSUPPORTED_EXPERIMENT_REASON
        )
    if release_pin is None:
        raise ExecutionBundleBlockedError(
            PREPARED_MODEL_CONTENT_RELEASE_MANIFEST_PIN_UNAVAILABLE_REASON
        )
    if type(authority) is not TrainablePlanAuthorityBinding:
        raise ExecutionBundleBlockedError(
            TRAINABLE_PLAN_RAW_AUTHORITY_UNAVAILABLE_REASON
        )
    if formal and not has_prepared_model_content_release_manifest_sha256(
        model_lock_sha256=authority.model_lock_sha256,
        prepared=authority.prepared_model_content_authority.prepared_model_set,
        claimed_manifest_sha256=release_pin,
    ):
        raise ExecutionBundleBlockedError(
            PREPARED_MODEL_CONTENT_RELEASE_MANIFEST_PIN_UNAVAILABLE_REASON
        )
    _require_trainable_raw_source_match(
        authority.model_lock,
        bundle.model_lock_artifact.source,
        label="model lock",
    )
    _require_trainable_raw_source_match(
        authority.run_config,
        bundle.run_config,
        label="RunConfig",
    )
    _require_trainable_raw_source_match(
        authority.split,
        bundle.split_artifact.source,
        label="execution split",
    )
    prepared = _strict_object(
        "prepared models",
        prepared_models,
        frozenset({"schema_version", "kind", "model_lock_sha256", "roots"}),
    )
    roots = prepared["roots"]
    if type(roots) is not dict or any(
        type(key) is not str or type(value) is not str for key, value in roots.items()
    ):
        raise TypeError("prepared-model roots must be a string mapping")
    content_binding = authority.prepared_model_content_authority
    content_roots = {
        snapshot.model_id: snapshot.root
        for snapshot in content_binding.prepared_model_set.snapshots
    }
    if (
        prepared["model_lock_sha256"] != model_lock.sha256
        or content_binding.model_lock_sha256 != model_lock.sha256
        or content_roots != roots
    ):
        raise ValueError(
            "trainable-plan prepared content differs from bundle model authority"
        )
    adaptation = run_config.adaptation
    if adaptation is None:
        raise ValueError("adapted bundle lacks an adaptation configuration")
    gate = (
        require_trainable_plan_authority_for_method
        if formal
        else audit_trainable_plan_authority_for_method
    )
    try:
        return gate(
            method,
            authority,
            expected_model_lock_sha256=model_lock.sha256,
            expected_prepared_model_content_manifest_sha256=release_pin,
            expected_run_config_sha256=run_config_sha256(run_config),
            expected_split_sha256=bundle.split_artifact.source.semantic_sha256,
            expected_cell_id=cell.cell_id,
            expected_cell_declaration_sha256=cell.sha256,
            expected_execution_semantics_sha256=execution_semantics.sha256,
            expected_target_model_id=run_config.model.target,
            expected_target_revision=run_config.model.target_revision,
            expected_drafter_model_id=run_config.model.drafter,
            expected_prepared_drafter_revision=run_config.model.drafter_revision,
            expected_backend=run_config.model.algorithm,
            expected_mode=adaptation.weight_update_mode,
            expected_scope=adaptation.parameter_scope,
            expected_optimizer=adaptation.optimizer.name,
            expected_rank=adaptation.rank,
            expected_lora_alpha=adaptation.lora_alpha,
        )
    except PreparedModelContentAuthorityBlocked as error:
        raise ExecutionBundleBlockedError(error.code) from error


def _validate_model_inputs(
    *,
    model_lock: ModelLock,
    prepared: object,
    run_config,
    launch: ServerLaunch,
) -> None:
    roots = _validate_prepared_model_sources(
        model_lock=model_lock,
        prepared=prepared,
        run_config=run_config,
    )
    target = run_config.model.target
    try:
        model_path_index = launch.argv.index("--model-path") + 1
        launched_root = launch.argv[model_path_index]
    except (ValueError, IndexError) as error:
        raise ValueError("server launch lacks one target model path") from error
    if launched_root != roots[target]:
        raise ValueError("server launch target root differs from prepared models")


def _validate_prepared_model_sources(
    *,
    model_lock: ModelLock,
    prepared: object,
    run_config,
) -> dict[str, str]:
    """Validate every model input before a renderer can create runtime files."""

    row = _strict_object(
        "prepared models",
        prepared,
        frozenset({"schema_version", "kind", "model_lock_sha256", "roots"}),
    )
    if (
        row["schema_version"] != 1
        or row["kind"] != _PREPARED_MODELS_KIND
        or row["model_lock_sha256"] != model_lock.sha256
    ):
        raise ValueError("prepared-model identity mismatch")
    roots = row["roots"]
    if type(roots) is not dict or any(
        type(key) is not str or type(value) is not str for key, value in roots.items()
    ):
        raise TypeError("prepared-model roots must be a JSON object")
    expected = {model.model_id: model.revision for model in model_lock.models}
    if set(roots) != set(expected):
        raise ValueError("prepared-model roots do not cover the lock exactly")
    validated_roots: dict[str, str] = {}
    for model_id, root_value in roots.items():
        root = Path(_strict_text("prepared model root", root_value))
        if not root.is_absolute() or root.resolve() != root or not root.is_dir():
            raise ValueError(
                "prepared model root is not an existing resolved directory"
            )
        if root.is_symlink():
            raise ValueError("prepared model root cannot be a symlink")
        validated_roots[model_id] = str(root)
    target = run_config.model.target
    if expected.get(target) != run_config.model.target_revision:
        raise ValueError("run config target revision differs from model lock")
    drafter = run_config.model.drafter
    if expected.get(drafter) != run_config.model.drafter_revision:
        raise ValueError("run config drafter revision differs from model lock")
    return validated_roots


def _one_assignment(plan: GpuDispatchPlan, assignment_sha256: str) -> GpuAssignment:
    matches = tuple(
        assignment
        for wave in plan.waves
        for assignment in wave.assignments
        if assignment.assignment_id == assignment_sha256
    )
    if len(matches) != 1:
        raise ValueError("bundle assignment is absent or duplicated in dispatch plan")
    return matches[0]


def _require_bundle_failure_injection_authority(
    *,
    registry: ExperimentRegistry,
    cell: ExperimentCell,
    binding: FailureInjectionAuthorityBinding | None,
    diagnostic: bool,
) -> tuple[str | None, FailureExecutionAuthorityToken | None]:
    """Revalidate the E5 plan and mint a token only at the formal boundary."""

    failure_cell = cell.identity.task == "failure_injection"
    if failure_cell != (binding is not None):
        if failure_cell:
            raise ExecutionBundleBlockedError(
                "failure_injection_raw_plan_authority_required"
            )
        raise ValueError("non-failure bundle cannot carry failure authority")
    if binding is None:
        return None, None
    replayed = revalidate_failure_injection_authority(binding, registry=registry)
    if (
        replayed.plan.cell_id != cell.cell_id
        or replayed.binding.registry_sha256 != registry.sha256
    ):
        raise ValueError("failure authority names another assignment cell")
    if diagnostic:
        return replayed.binding.sha256, None
    try:
        token = require_failure_injection_authority(
            replayed.binding,
            registry=registry,
        )
    except FailureInjectionAuthorityBlocked as error:
        raise ExecutionBundleBlockedError(error.reason) from error
    return replayed.binding.sha256, token


def _one_budget(
    budgets: tuple[ExperimentBudget, ...], cell_id: str
) -> ExperimentBudget:
    matches = tuple(budget for budget in budgets if budget.cell_id == cell_id)
    if len(matches) != 1:
        raise ValueError("bundle cell lacks one exact ExperimentBudget")
    return matches[0]


__all__ = [
    "TRUSTED_DISPATCH_ATTESTER_UNAVAILABLE_REASON",
    "BoundExecutionArtifact",
    "BoundJsonSource",
    "ExecutionBundleBlockedError",
    "IndustrialAssignmentExecutionBundle",
    "IndustrialExecutionPlanAudit",
    "InterferenceCalibrationExecutionAuthority",
    "dispatch_receipt_sidecar_path",
    "execute_dispatch_wave_bundles",
    "execution_policy_to_dict",
    "finalize_materialized_execution_bundle",
    "load_dispatch_schedule_receipt",
    "preflight_dispatch_receipt_output",
    "preflight_fresh_assignment_trace",
    "prepared_models_to_dict",
    "publish_dispatch_schedule_receipt",
    "require_release_dispatch_execution_authority",
    "server_launch_to_dict",
    "topology_receipt_set_to_dict",
]
