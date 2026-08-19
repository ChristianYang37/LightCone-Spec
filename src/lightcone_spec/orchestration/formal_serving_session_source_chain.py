"""Durable source-owned lifecycle evidence for one resident TP1 process.

This module is intentionally an empirical evidence recorder, not a reuse or
release authority.  It publishes every source response before the next
resident-session action and publishes a manifest only after the exact ordered
source close receipt has been validated.  The public revalidator starts from
that manifest path alone: it reopens every bound file, reparses the complete
``SourceOwned*`` chain, and deep-validates each unsigned native terminal
artifact under ``NO_TRUSTED_ATTESTERS``.

No in-memory live result, root attestation, or verifier-only token is accepted.
The resulting object therefore remains trusted-single-operator empirical
evidence with ``formal_measured=False`` and ``reuse_authorized=False``.  A
separate, fresh reset qualification remains responsible for authorizing reuse.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lightcone_spec.orchestration.native_terminal import (
    ValidatedNativeTerminalEvidence,
    canonical_sha256,
    validate_native_terminal_artifact,
)
from lightcone_spec.orchestration.session_reuse_authority import (
    ConnectionAccounting,
    SourceOwnedCloseReceipt,
    SourceOwnedInitialStateReceipt,
    SourceOwnedResetReceipt,
    SourceOwnedScoredClockReceipt,
    SourceOwnedSessionCapability,
    SourceOwnedTraceReceipt,
    SourceOwnedWarmupReceipt,
)
from lightcone_spec.runtime.attestation import NO_TRUSTED_ATTESTERS
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_MANIFEST = "resident-source-chain-manifest.json"
FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_EVIDENCE_LEVEL = (
    "trusted_single_operator_empirical_no_signature"
)
FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_PROTOCOL_SHA256 = canonical_sha256(
    {
        "schema_version": 1,
        "kind": "formal_serving_resident_source_chain",
        "scope": "one_ordered_resident_tp1_process",
        "source_chain": (
            "capability_initial_then_per_epoch_reset_warmup_scored_clock_"
            "trace_terminal_then_source_close"
        ),
        "terminal": "path_reopened_unsigned_native_terminal_artifact",
        "publication": "exclusive_step_files_then_close_manifest_last",
        "revalidation": "manifest_path_only_no_expected_live_result",
        "evidence_level": FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_EVIDENCE_LEVEL,
        "reuse_authorized": False,
        "formal_measured": False,
    }
)

_MANIFEST_KIND = "formal_serving_resident_source_chain_manifest"
_COMMIT_MARKER = "SOURCE_CHAIN_CLOSED"
_SOURCE_PROCESS_IDENTITY = re.compile(r"scheduler:([1-9][0-9]*)\Z")


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_object(label: str, value: object, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _resolved_evidence_directory(value: str | Path) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path != path.resolve(strict=False)
        or not path.is_dir()
        or path.is_symlink()
    ):
        raise ValueError(
            "resident source evidence directory must be existing, absolute, "
            "resolved, and symlink-free"
        )
    status = path.stat(follow_symlinks=False)
    if status.st_uid != os.geteuid() or status.st_mode & 0o022:
        raise ValueError(
            "resident source evidence directory must be current-user-owned and "
            "not group/world writable"
        )
    return path


def _source_filename(step: str, *, epoch_index: int | None = None) -> str:
    if epoch_index is None:
        return f"resident-source-{step}.json"
    return f"resident-source-epoch-{epoch_index:04d}-{step}.json"


def _publish_source_response(
    directory: Path,
    *,
    filename: str,
    value: object,
) -> CanonicalJsonProofBinding:
    if type(value) is not dict:
        raise TypeError("resident source response must be one exact JSON object")
    path = directory / filename
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _reopen_binding(
    value: object,
    *,
    label: str,
    expected_path: Path | None = None,
) -> CanonicalJsonProofBinding:
    binding = CanonicalJsonProofBinding.from_dict(value)
    if expected_path is not None and binding.absolute_path != str(expected_path):
        raise ValueError(f"{label} path differs from the closed source chain")
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError(f"{label} changed after publication")
    return binding


def _coerce_manifest_binding(
    value: str | Path | CanonicalJsonProofBinding,
) -> CanonicalJsonProofBinding:
    if type(value) is CanonicalJsonProofBinding:
        binding = value
        if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
            raise ValueError("resident source-chain manifest changed")
        return binding
    if type(value) is not str and not isinstance(value, Path):
        raise TypeError("resident source-chain revalidation requires one path")
    return CanonicalJsonProofBinding.bind(value)


@dataclass(frozen=True)
class FormalServingResidentSourceEpochEvidence:
    """Deep-reopened source and terminal evidence for one logical epoch."""

    epoch_index: int
    execution_plan_sha256: str
    reset_receipt_binding: CanonicalJsonProofBinding
    warmup_receipt_binding: CanonicalJsonProofBinding
    scored_clock_receipt_binding: CanonicalJsonProofBinding
    trace_receipt_binding: CanonicalJsonProofBinding
    terminal_artifact: CanonicalJsonProofBinding
    reset: SourceOwnedResetReceipt
    warmup: SourceOwnedWarmupReceipt
    scored_clock: SourceOwnedScoredClockReceipt
    trace: SourceOwnedTraceReceipt
    terminal: ValidatedNativeTerminalEvidence


@dataclass(frozen=True)
class RevalidatedFormalServingResidentSourceChain:
    """Path-reopened empirical evidence; never a formal or reuse authority."""

    protocol_sha256: str
    manifest: CanonicalJsonProofBinding
    session_plan_sha256: str
    execution_plan_sha256s: tuple[str, ...]
    capability_binding: CanonicalJsonProofBinding
    initial_state_binding: CanonicalJsonProofBinding
    close_receipt_binding: CanonicalJsonProofBinding
    capability: SourceOwnedSessionCapability
    initial_state: SourceOwnedInitialStateReceipt
    epochs: tuple[FormalServingResidentSourceEpochEvidence, ...]
    close: SourceOwnedCloseReceipt
    evidence_level: Literal["trusted_single_operator_empirical_no_signature"]
    reuse_authorized: Literal[False]
    formal_measured: Literal[False]

    @property
    def sha256(self) -> str:
        """The immutable semantic identity of the path-reopened manifest."""

        return self.manifest.semantic_sha256


def _validate_terminal_link(
    *,
    capability: SourceOwnedSessionCapability,
    execution_plan_sha256: str,
    epoch_index: int,
    previous_terminal: ValidatedNativeTerminalEvidence | None,
    scored_clock: SourceOwnedScoredClockReceipt,
    trace: SourceOwnedTraceReceipt,
    terminal_artifact: CanonicalJsonProofBinding,
) -> ValidatedNativeTerminalEvidence:
    terminal = validate_native_terminal_artifact(
        terminal_artifact.reopen(),
        trusted_attester_policy=NO_TRUSTED_ATTESTERS,
    )
    terminal.binding.validate()
    process_match = _SOURCE_PROCESS_IDENTITY.fullmatch(capability.process_identity)
    if process_match is None:
        raise ValueError(
            "source capability process identity is not the pinned scheduler identity"
        )
    prior_run_id = (
        None if previous_terminal is None else previous_terminal.binding.run_id
    )
    if (
        terminal.authority_kind != "untrusted_raw_terminal"
        or terminal.binding.execution_plan_sha256 != execution_plan_sha256
        or terminal.binding.session_id != capability.session_plan_sha256
        or terminal.binding.session_epoch != epoch_index
        or terminal.binding.previous_run_id != prior_run_id
        or terminal.begin_receipt.server_process_id != int(process_match.group(1))
        or terminal.reset_receipt.server_process_id != int(process_match.group(1))
        or terminal.begin_receipt.server_process_started_ns
        != capability.process_started_ns
        or terminal.reset_receipt.server_process_started_ns
        != capability.process_started_ns
        or scored_clock.native_reset_sha256 != terminal.reset_receipt.reset_sha256
        or trace.terminal_receipt_sha256 != terminal.terminal_sha256
    ):
        raise ValueError(
            "resident finalized terminal breaks the source-owned epoch chain"
        )
    return terminal


def _manifest_without_digest(value: dict[str, object]) -> dict[str, object]:
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    return unsigned


def revalidate_formal_serving_resident_source_chain(
    manifest_path: str | Path | CanonicalJsonProofBinding,
) -> RevalidatedFormalServingResidentSourceChain:
    """Deep-reopen a complete source chain from its manifest path alone."""

    manifest_binding = _coerce_manifest_binding(manifest_path)
    raw = _strict_object(
        "resident source-chain manifest",
        manifest_binding.reopen(),
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "commit_marker",
            "session_plan_sha256",
            "execution_plan_sha256s",
            "capability",
            "initial_state",
            "epochs",
            "close_receipt",
            "evidence_level",
            "reuse_authorized",
            "formal_measured",
            "manifest_sha256",
        },
    )
    manifest_sha256 = _sha256("resident source-chain manifest", raw["manifest_sha256"])
    if (
        raw["schema_version"] != 1
        or raw["kind"] != _MANIFEST_KIND
        or raw["protocol_sha256"]
        != FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_PROTOCOL_SHA256
        or raw["commit_marker"] != _COMMIT_MARKER
        or raw["evidence_level"] != FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_EVIDENCE_LEVEL
        or raw["reuse_authorized"] is not False
        or raw["formal_measured"] is not False
        or canonical_sha256(_manifest_without_digest(raw)) != manifest_sha256
    ):
        raise ValueError("resident source-chain manifest identity differs")

    session_plan_sha256 = _sha256(
        "resident source-chain session plan", raw["session_plan_sha256"]
    )
    raw_execution_plans = raw["execution_plan_sha256s"]
    if type(raw_execution_plans) is not list:
        raise TypeError("resident source-chain plans must be one ordered JSON array")
    execution_plan_sha256s = tuple(
        _sha256("resident source-chain execution plan", value)
        for value in raw_execution_plans
    )
    if not execution_plan_sha256s or len(execution_plan_sha256s) != len(
        set(execution_plan_sha256s)
    ):
        raise ValueError("resident source-chain plans must be nonempty and unique")

    manifest_identity = Path(manifest_binding.absolute_path)
    if manifest_identity.name != FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_MANIFEST:
        raise ValueError("resident source-chain manifest filename differs")
    source_directory = manifest_identity.parent
    capability_binding = _reopen_binding(
        raw["capability"],
        label="resident source capability",
        expected_path=source_directory / _source_filename("capability"),
    )
    capability = SourceOwnedSessionCapability.parse(
        capability_binding.reopen(),
        session_plan_sha256=session_plan_sha256,
        execution_plan_sha256s=execution_plan_sha256s,
    )
    if not capability.continuous_connection_accounting:
        raise ValueError("resident source capability lacks connection accounting")
    initial_binding = _reopen_binding(
        raw["initial_state"],
        label="resident source initial state",
        expected_path=source_directory / _source_filename("initial-state"),
    )
    initial = SourceOwnedInitialStateReceipt.parse(
        initial_binding.reopen(), capability=capability
    )

    raw_epochs = raw["epochs"]
    if type(raw_epochs) is not list or len(raw_epochs) != len(execution_plan_sha256s):
        raise ValueError("resident source-chain epoch coverage is incomplete")
    epochs: list[FormalServingResidentSourceEpochEvidence] = []
    accounting: ConnectionAccounting = initial.state.connection_accounting
    reset_generation = initial.state.reset_generation
    clock_generation = 0
    previous_plan: str | None = None
    previous_terminal: ValidatedNativeTerminalEvidence | None = None
    for epoch_index, (execution_plan_sha256, epoch_value) in enumerate(
        zip(execution_plan_sha256s, raw_epochs, strict=True), start=1
    ):
        epoch = _strict_object(
            "resident source-chain epoch",
            epoch_value,
            {
                "epoch_index",
                "execution_plan_sha256",
                "reset_receipt",
                "warmup_receipt",
                "scored_clock_receipt",
                "trace_receipt",
                "terminal_artifact",
            },
        )
        if (
            epoch["epoch_index"] != epoch_index
            or epoch["execution_plan_sha256"] != execution_plan_sha256
        ):
            raise ValueError("resident source-chain epoch order differs")
        reset_binding = _reopen_binding(
            epoch["reset_receipt"],
            label="resident source reset receipt",
            expected_path=source_directory
            / _source_filename("reset", epoch_index=epoch_index),
        )
        reset = SourceOwnedResetReceipt.parse(
            reset_binding.reopen(),
            capability=capability,
            prior_execution_plan_sha256=previous_plan,
            next_execution_plan_sha256=execution_plan_sha256,
            initial_state_receipt_sha256=initial.initial_state_receipt_sha256,
            clean_state_sha256=initial.state.clean_state_sha256,
            expected_reset_generation=reset_generation,
            prior_accounting=accounting,
        )
        reset_generation = reset.after.reset_generation
        accounting = reset.after.connection_accounting
        warmup_binding = _reopen_binding(
            epoch["warmup_receipt"],
            label="resident source warmup receipt",
            expected_path=source_directory
            / _source_filename("warmup", epoch_index=epoch_index),
        )
        warmup = SourceOwnedWarmupReceipt.parse(
            warmup_binding.reopen(),
            execution_plan_sha256=execution_plan_sha256,
            prior_accounting=accounting,
        )
        accounting = warmup.connection_accounting
        clock_binding = _reopen_binding(
            epoch["scored_clock_receipt"],
            label="resident source scored-clock receipt",
            expected_path=source_directory
            / _source_filename("scored-clock", epoch_index=epoch_index),
        )
        scored_clock = SourceOwnedScoredClockReceipt.parse(
            clock_binding.reopen(),
            execution_plan_sha256=execution_plan_sha256,
            warmup=warmup,
            prior_clock_generation=clock_generation,
        )
        clock_generation = scored_clock.clock_generation
        trace_binding = _reopen_binding(
            epoch["trace_receipt"],
            label="resident source trace receipt",
            expected_path=source_directory
            / _source_filename("trace", epoch_index=epoch_index),
        )
        trace = SourceOwnedTraceReceipt.parse(
            trace_binding.reopen(),
            execution_plan_sha256=execution_plan_sha256,
            clock=scored_clock,
            prior_accounting=accounting,
        )
        if trace.aborted:
            raise ValueError("resident source chain contains an aborted trace")
        terminal_artifact = _reopen_binding(
            epoch["terminal_artifact"], label="resident native terminal artifact"
        )
        terminal = _validate_terminal_link(
            capability=capability,
            execution_plan_sha256=execution_plan_sha256,
            epoch_index=epoch_index,
            previous_terminal=previous_terminal,
            scored_clock=scored_clock,
            trace=trace,
            terminal_artifact=terminal_artifact,
        )
        epochs.append(
            FormalServingResidentSourceEpochEvidence(
                epoch_index=epoch_index,
                execution_plan_sha256=execution_plan_sha256,
                reset_receipt_binding=reset_binding,
                warmup_receipt_binding=warmup_binding,
                scored_clock_receipt_binding=clock_binding,
                trace_receipt_binding=trace_binding,
                terminal_artifact=terminal_artifact,
                reset=reset,
                warmup=warmup,
                scored_clock=scored_clock,
                trace=trace,
                terminal=terminal,
            )
        )
        accounting = trace.connection_accounting
        previous_plan = execution_plan_sha256
        previous_terminal = terminal

    close_binding = _reopen_binding(
        raw["close_receipt"],
        label="resident source close receipt",
        expected_path=source_directory / _source_filename("close"),
    )
    close = SourceOwnedCloseReceipt.parse(
        close_binding.reopen(),
        capability=capability,
        prior_accounting=accounting,
        initial_state_receipt_sha256=initial.initial_state_receipt_sha256,
        execution_plan_sha256s=execution_plan_sha256s,
        reset_receipt_sha256s=tuple(
            epoch.reset.reset_receipt_sha256 for epoch in epochs
        ),
        warmup_receipt_sha256s=tuple(
            epoch.warmup.warmup_receipt_sha256 for epoch in epochs
        ),
        clock_receipt_sha256s=tuple(
            epoch.scored_clock.clock_receipt_sha256 for epoch in epochs
        ),
        trace_receipt_sha256s=tuple(
            epoch.trace.trace_receipt_sha256 for epoch in epochs
        ),
        terminal_receipt_sha256s=tuple(
            epoch.terminal.terminal_sha256 for epoch in epochs
        ),
    )
    return RevalidatedFormalServingResidentSourceChain(
        protocol_sha256=FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_PROTOCOL_SHA256,
        manifest=manifest_binding,
        session_plan_sha256=session_plan_sha256,
        execution_plan_sha256s=execution_plan_sha256s,
        capability_binding=capability_binding,
        initial_state_binding=initial_binding,
        close_receipt_binding=close_binding,
        capability=capability,
        initial_state=initial,
        epochs=tuple(epochs),
        close=close,
        evidence_level=FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_EVIDENCE_LEVEL,
        reuse_authorized=False,
        formal_measured=False,
    )


@dataclass
class _PendingEpoch:
    epoch_index: int
    execution_plan_sha256: str
    reset_receipt: CanonicalJsonProofBinding | None = None
    warmup_receipt: CanonicalJsonProofBinding | None = None
    scored_clock_receipt: CanonicalJsonProofBinding | None = None
    trace_receipt: CanonicalJsonProofBinding | None = None
    terminal_artifact: CanonicalJsonProofBinding | None = None
    reset: SourceOwnedResetReceipt | None = None
    warmup: SourceOwnedWarmupReceipt | None = None
    scored_clock: SourceOwnedScoredClockReceipt | None = None
    trace: SourceOwnedTraceReceipt | None = None
    terminal: ValidatedNativeTerminalEvidence | None = None


class FormalServingResidentSourceChainPublisher:
    """Exclusive, prefix-durable publisher for a live resident source chain."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        session_plan_sha256: str,
        execution_plan_sha256s: tuple[str, ...],
    ) -> None:
        self._directory = _resolved_evidence_directory(output_dir)
        self._session_plan_sha256 = _sha256(
            "resident source session plan", session_plan_sha256
        )
        if (
            type(execution_plan_sha256s) is not tuple
            or not execution_plan_sha256s
            or len(execution_plan_sha256s) != len(set(execution_plan_sha256s))
        ):
            raise ValueError(
                "resident source publisher requires unique ordered execution plans"
            )
        self._execution_plan_sha256s = tuple(
            _sha256("resident source execution plan", value)
            for value in execution_plan_sha256s
        )
        self._capability_binding: CanonicalJsonProofBinding | None = None
        self._capability: SourceOwnedSessionCapability | None = None
        self._initial_binding: CanonicalJsonProofBinding | None = None
        self._initial: SourceOwnedInitialStateReceipt | None = None
        self._accounting: ConnectionAccounting | None = None
        self._reset_generation = 0
        self._clock_generation = 0
        self._previous_plan: str | None = None
        self._previous_terminal: ValidatedNativeTerminalEvidence | None = None
        self._epochs: list[_PendingEpoch] = []
        self._closed = False

    @property
    def manifest_path(self) -> Path:
        return self._directory / FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_MANIFEST

    def record_capability(self, value: object) -> CanonicalJsonProofBinding:
        if self._capability_binding is not None or self._closed:
            raise RuntimeError("resident source capability was already recorded")
        binding = _publish_source_response(
            self._directory,
            filename=_source_filename("capability"),
            value=value,
        )
        capability = SourceOwnedSessionCapability.parse(
            binding.reopen(),
            session_plan_sha256=self._session_plan_sha256,
            execution_plan_sha256s=self._execution_plan_sha256s,
        )
        if not capability.continuous_connection_accounting:
            raise ValueError("resident source capability lacks connection accounting")
        self._capability_binding = binding
        self._capability = capability
        return binding

    def record_initial_state(self, value: object) -> CanonicalJsonProofBinding:
        if (
            self._capability is None
            or self._initial_binding is not None
            or self._closed
        ):
            raise RuntimeError("resident source initial state is out of order")
        binding = _publish_source_response(
            self._directory,
            filename=_source_filename("initial-state"),
            value=value,
        )
        initial = SourceOwnedInitialStateReceipt.parse(
            binding.reopen(), capability=self._capability
        )
        self._initial_binding = binding
        self._initial = initial
        self._accounting = initial.state.connection_accounting
        self._reset_generation = initial.state.reset_generation
        return binding

    def record_reset(self, value: object) -> CanonicalJsonProofBinding:
        if self._initial is None or self._accounting is None or self._closed:
            raise RuntimeError("resident source reset is out of order")
        if self._epochs and self._epochs[-1].trace_receipt is None:
            raise RuntimeError("prior resident source epoch is incomplete")
        epoch_index = len(self._epochs) + 1
        if epoch_index > len(self._execution_plan_sha256s):
            raise RuntimeError("resident source reset exceeds registered epochs")
        execution_plan_sha256 = self._execution_plan_sha256s[epoch_index - 1]
        binding = _publish_source_response(
            self._directory,
            filename=_source_filename("reset", epoch_index=epoch_index),
            value=value,
        )
        assert self._capability is not None
        reset = SourceOwnedResetReceipt.parse(
            binding.reopen(),
            capability=self._capability,
            prior_execution_plan_sha256=self._previous_plan,
            next_execution_plan_sha256=execution_plan_sha256,
            initial_state_receipt_sha256=self._initial.initial_state_receipt_sha256,
            clean_state_sha256=self._initial.state.clean_state_sha256,
            expected_reset_generation=self._reset_generation,
            prior_accounting=self._accounting,
        )
        self._reset_generation = reset.after.reset_generation
        self._accounting = reset.after.connection_accounting
        self._epochs.append(
            _PendingEpoch(
                epoch_index=epoch_index,
                execution_plan_sha256=execution_plan_sha256,
                reset_receipt=binding,
                reset=reset,
            )
        )
        return binding

    def record_warmup(self, value: object) -> CanonicalJsonProofBinding:
        epoch = self._current_epoch(required="reset")
        if epoch.warmup_receipt is not None:
            raise RuntimeError("resident source warmup was already recorded")
        assert self._accounting is not None
        binding = _publish_source_response(
            self._directory,
            filename=_source_filename("warmup", epoch_index=epoch.epoch_index),
            value=value,
        )
        warmup = SourceOwnedWarmupReceipt.parse(
            binding.reopen(),
            execution_plan_sha256=epoch.execution_plan_sha256,
            prior_accounting=self._accounting,
        )
        self._accounting = warmup.connection_accounting
        epoch.warmup_receipt = binding
        epoch.warmup = warmup
        return binding

    def record_scored_clock(self, value: object) -> CanonicalJsonProofBinding:
        epoch = self._current_epoch(required="warmup")
        if epoch.scored_clock_receipt is not None:
            raise RuntimeError("resident source scored clock was already recorded")
        assert epoch.warmup is not None
        binding = _publish_source_response(
            self._directory,
            filename=_source_filename("scored-clock", epoch_index=epoch.epoch_index),
            value=value,
        )
        clock = SourceOwnedScoredClockReceipt.parse(
            binding.reopen(),
            execution_plan_sha256=epoch.execution_plan_sha256,
            warmup=epoch.warmup,
            prior_clock_generation=self._clock_generation,
        )
        self._clock_generation = clock.clock_generation
        epoch.scored_clock_receipt = binding
        epoch.scored_clock = clock
        return binding

    def record_trace(
        self,
        value: object,
        *,
        terminal_artifact: CanonicalJsonProofBinding,
    ) -> CanonicalJsonProofBinding:
        epoch = self._current_epoch(required="scored-clock")
        if epoch.trace_receipt is not None:
            raise RuntimeError("resident source trace was already recorded")
        if type(terminal_artifact) is not CanonicalJsonProofBinding:
            raise TypeError("resident source trace requires one terminal binding")
        binding = _publish_source_response(
            self._directory,
            filename=_source_filename("trace", epoch_index=epoch.epoch_index),
            value=value,
        )
        assert epoch.scored_clock is not None
        assert self._accounting is not None
        trace = SourceOwnedTraceReceipt.parse(
            binding.reopen(),
            execution_plan_sha256=epoch.execution_plan_sha256,
            clock=epoch.scored_clock,
            prior_accounting=self._accounting,
        )
        if trace.aborted:
            raise ValueError("resident source trace was aborted")
        assert self._capability is not None
        stable_terminal_binding = CanonicalJsonProofBinding.bind(
            terminal_artifact.absolute_path
        )
        if stable_terminal_binding != terminal_artifact:
            raise ValueError("resident terminal artifact changed before binding")
        terminal = _validate_terminal_link(
            capability=self._capability,
            execution_plan_sha256=epoch.execution_plan_sha256,
            epoch_index=epoch.epoch_index,
            previous_terminal=self._previous_terminal,
            scored_clock=epoch.scored_clock,
            trace=trace,
            terminal_artifact=stable_terminal_binding,
        )
        epoch.trace_receipt = binding
        epoch.terminal_artifact = stable_terminal_binding
        epoch.trace = trace
        epoch.terminal = terminal
        self._accounting = trace.connection_accounting
        self._previous_plan = epoch.execution_plan_sha256
        self._previous_terminal = terminal
        return binding

    def record_close(
        self, value: object
    ) -> RevalidatedFormalServingResidentSourceChain:
        if self._closed:
            raise RuntimeError("resident source chain was already closed")
        if (
            self._capability is None
            or self._capability_binding is None
            or self._initial is None
            or self._initial_binding is None
            or self._accounting is None
            or len(self._epochs) != len(self._execution_plan_sha256s)
            or any(
                epoch.trace_receipt is None
                or epoch.terminal_artifact is None
                or epoch.trace is None
                or epoch.terminal is None
                for epoch in self._epochs
            )
        ):
            raise RuntimeError("resident source close precedes complete epoch coverage")
        close_binding = _publish_source_response(
            self._directory,
            filename=_source_filename("close"),
            value=value,
        )
        parsed_epochs = [self._evidence_from_pending(epoch) for epoch in self._epochs]
        SourceOwnedCloseReceipt.parse(
            close_binding.reopen(),
            capability=self._capability,
            prior_accounting=self._accounting,
            initial_state_receipt_sha256=self._initial.initial_state_receipt_sha256,
            execution_plan_sha256s=self._execution_plan_sha256s,
            reset_receipt_sha256s=tuple(
                epoch.reset.reset_receipt_sha256 for epoch in parsed_epochs
            ),
            warmup_receipt_sha256s=tuple(
                epoch.warmup.warmup_receipt_sha256 for epoch in parsed_epochs
            ),
            clock_receipt_sha256s=tuple(
                epoch.scored_clock.clock_receipt_sha256 for epoch in parsed_epochs
            ),
            trace_receipt_sha256s=tuple(
                epoch.trace.trace_receipt_sha256 for epoch in parsed_epochs
            ),
            terminal_receipt_sha256s=tuple(
                epoch.terminal.terminal_sha256 for epoch in parsed_epochs
            ),
        )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "kind": _MANIFEST_KIND,
            "protocol_sha256": (FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_PROTOCOL_SHA256),
            "commit_marker": _COMMIT_MARKER,
            "session_plan_sha256": self._session_plan_sha256,
            "execution_plan_sha256s": list(self._execution_plan_sha256s),
            "capability": self._capability_binding.to_dict(),
            "initial_state": self._initial_binding.to_dict(),
            "epochs": [self._epoch_manifest(epoch) for epoch in self._epochs],
            "close_receipt": close_binding.to_dict(),
            "evidence_level": (FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_EVIDENCE_LEVEL),
            "reuse_authorized": False,
            "formal_measured": False,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        publish_canonical_json_no_replace(self.manifest_path, manifest)
        self._closed = True
        return revalidate_formal_serving_resident_source_chain(self.manifest_path)

    def _current_epoch(self, *, required: str) -> _PendingEpoch:
        if not self._epochs or self._closed:
            raise RuntimeError(f"resident source {required} is out of order")
        epoch = self._epochs[-1]
        requirements = {
            "reset": epoch.reset_receipt,
            "warmup": epoch.warmup_receipt,
            "scored-clock": epoch.scored_clock_receipt,
        }
        if requirements[required] is None:
            raise RuntimeError(f"resident source {required} is missing")
        return epoch

    @staticmethod
    def _evidence_from_pending(
        epoch: _PendingEpoch,
    ) -> FormalServingResidentSourceEpochEvidence:
        assert epoch.reset_receipt is not None
        assert epoch.warmup_receipt is not None
        assert epoch.scored_clock_receipt is not None
        assert epoch.trace_receipt is not None
        assert epoch.terminal_artifact is not None
        assert epoch.reset is not None
        assert epoch.warmup is not None
        assert epoch.scored_clock is not None
        assert epoch.trace is not None
        assert epoch.terminal is not None
        return FormalServingResidentSourceEpochEvidence(
            epoch_index=epoch.epoch_index,
            execution_plan_sha256=epoch.execution_plan_sha256,
            reset_receipt_binding=epoch.reset_receipt,
            warmup_receipt_binding=epoch.warmup_receipt,
            scored_clock_receipt_binding=epoch.scored_clock_receipt,
            trace_receipt_binding=epoch.trace_receipt,
            terminal_artifact=epoch.terminal_artifact,
            reset=epoch.reset,
            warmup=epoch.warmup,
            scored_clock=epoch.scored_clock,
            trace=epoch.trace,
            terminal=epoch.terminal,
        )

    @staticmethod
    def _epoch_manifest(epoch: _PendingEpoch) -> dict[str, object]:
        assert epoch.reset_receipt is not None
        assert epoch.warmup_receipt is not None
        assert epoch.scored_clock_receipt is not None
        assert epoch.trace_receipt is not None
        assert epoch.terminal_artifact is not None
        return {
            "epoch_index": epoch.epoch_index,
            "execution_plan_sha256": epoch.execution_plan_sha256,
            "reset_receipt": epoch.reset_receipt.to_dict(),
            "warmup_receipt": epoch.warmup_receipt.to_dict(),
            "scored_clock_receipt": epoch.scored_clock_receipt.to_dict(),
            "trace_receipt": epoch.trace_receipt.to_dict(),
            "terminal_artifact": epoch.terminal_artifact.to_dict(),
        }


__all__ = (
    "FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_EVIDENCE_LEVEL",
    "FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_MANIFEST",
    "FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_PROTOCOL_SHA256",
    "FormalServingResidentSourceChainPublisher",
    "FormalServingResidentSourceEpochEvidence",
    "RevalidatedFormalServingResidentSourceChain",
    "revalidate_formal_serving_resident_source_chain",
)
