"""First-party reduction authority for the disjoint TTS numeric calibration.

The reducer accepts only the exact 288 staged cells and path-bearing serving
evidence.  SLO-goodput and safety are recomputed from raw request/performance
tables; callers cannot submit scores or terminal digests as measurements.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

from lightcone_spec.experiments.formal_protocol import (
    SignedTtsCalibrationSeal,
    TtsCalibrationAuthority,
    TtsCalibrationSeal,
    reject_banned_model_identity,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.industrial_analysis import (
    RawTtsCalibrationEvidenceManifest,
)
from lightcone_spec.experiments.registry import (
    PILOT_BLOCKS,
    ExperimentRegistry,
    content_sha256,
)
from lightcone_spec.experiments.stage_materialization import (
    StageCoverageReceipt,
    StageMaterializationReceipt,
)
from lightcone_spec.experiments.statistics import (
    TTFT_LIMIT_MS,
    HardwareEnvelope,
    SloRequest,
    account_slo,
)
from lightcone_spec.runtime.attestation import TrustedAttesterPolicy
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_SHA256_LENGTH = 64
_FIRST_PARTY_REDUCTION_SEAL = object()


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _strict_text(label: str, value: object) -> str:
    if type(value) is not str or not value or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be exact single-line text")
    return value


@dataclass(frozen=True)
class _TtsCalibrationRunIdentity:
    experiment: str
    runtime_sha256: str
    split_sha256: str


@dataclass(frozen=True)
class _ControlledTtsCalibrationTerminal:
    registry_cell_id: str
    canonical_raw_sha256: str
    terminal_sha256: str
    run_id: str
    run_nonce_sha256: str
    execution_plan_sha256: str
    rank_config_sha256: str
    method: str
    control_binding_sha256: str
    control_envelope_sha256: str
    control_reservation_sha256: str
    control_policy_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("registry cell", self.registry_cell_id),
            ("canonical raw", self.canonical_raw_sha256),
            ("terminal", self.terminal_sha256),
            ("run nonce", self.run_nonce_sha256),
            ("execution plan", self.execution_plan_sha256),
            ("rank config", self.rank_config_sha256),
            ("control binding", self.control_binding_sha256),
            ("control envelope", self.control_envelope_sha256),
            ("control reservation", self.control_reservation_sha256),
            ("control policy", self.control_policy_sha256),
        ):
            _require_sha256(f"controlled TTS terminal {label}", value)
        _strict_text("controlled TTS terminal run ID", self.run_id)
        if self.method != "tts":
            raise ValueError("TTS calibration terminal must use fixed-barrier TTS")


@dataclass(frozen=True)
class TtsCalibrationCellObservation:
    """Reducer output for one path-reopened candidate/pilot execution."""

    materialized_cell_id: str
    registry_cell_id: str
    candidate_id: str
    learning_rate: float
    stride: int
    block: int
    disposition: str
    reason_codes: tuple[str, ...]
    slo_goodput_tps: float | None
    slo_accounting_sha256: str
    raw_run_binding_sha256: str
    run_id: str
    run_nonce_sha256: str
    terminal_receipt_sha256: str
    qualification_lock_sha256: str
    terminal_control_binding_sha256: str
    terminal_control_envelope_sha256: str
    terminal_control_reservation_sha256: str
    terminal_control_policy_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("materialized cell", self.materialized_cell_id),
            ("registry cell", self.registry_cell_id),
            ("candidate", self.candidate_id),
            ("SLO accounting", self.slo_accounting_sha256),
            ("raw run binding", self.raw_run_binding_sha256),
            ("run nonce", self.run_nonce_sha256),
            ("terminal receipt", self.terminal_receipt_sha256),
            ("qualification lock", self.qualification_lock_sha256),
            ("terminal control binding", self.terminal_control_binding_sha256),
            ("terminal control envelope", self.terminal_control_envelope_sha256),
            (
                "terminal control reservation",
                self.terminal_control_reservation_sha256,
            ),
            ("terminal control policy", self.terminal_control_policy_sha256),
        ):
            _require_sha256(f"TTS calibration {label}", value)
        _strict_text("TTS calibration run ID", self.run_id)
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or type(self.stride) is not int
            or self.stride < 1
            or type(self.block) is not int
            or self.block not in PILOT_BLOCKS
        ):
            raise ValueError("TTS calibration observation has invalid numeric values")
        if self.disposition not in {"ELIGIBLE", "ELIMINATED"}:
            raise ValueError("TTS calibration disposition is invalid")
        if (
            type(self.reason_codes) is not tuple
            or self.reason_codes != tuple(sorted(set(self.reason_codes)))
            or any(
                type(reason) is not str or not reason for reason in self.reason_codes
            )
        ):
            raise ValueError("TTS calibration elimination reasons are not canonical")
        if self.disposition == "ELIGIBLE":
            if self.reason_codes:
                raise ValueError("eligible TTS calibration row cannot have a reason")
            if (
                type(self.slo_goodput_tps) is not float
                or not math.isfinite(self.slo_goodput_tps)
                or self.slo_goodput_tps <= 0
            ):
                raise ValueError("eligible TTS calibration row requires SLO-goodput")
        elif not self.reason_codes or self.slo_goodput_tps is not None:
            raise ValueError(
                "eliminated TTS calibration row requires reasons and no score"
            )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, init=False)
class TtsCalibrationReductionReceipt:
    """Unforgeable-in-process result of reopening all 288 raw executions."""

    schema_version: int
    protocol_lock_sha256: str
    authority_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    raw_manifest_sha256: str
    tuning_window_sha256: str
    registry_sha256: str
    runtime_sha256: str
    split_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    terminal_control_reservation_sha256: str
    terminal_control_policy_sha256: str
    observations: tuple[TtsCalibrationCellObservation, ...]
    selected_candidate_id: str
    selected_learning_rate: float
    selected_stride: int
    selected_mean_slo_goodput_tps: float
    selection_rule: str

    def __init__(
        self,
        *,
        schema_version: int,
        protocol_lock_sha256: str,
        authority_sha256: str,
        materialization_receipt_sha256: str,
        coverage_receipt_sha256: str,
        raw_manifest_sha256: str,
        tuning_window_sha256: str,
        registry_sha256: str,
        runtime_sha256: str,
        split_sha256: str,
        inventory_sha256: str,
        hardware_envelope_sha256: str,
        terminal_control_reservation_sha256: str,
        terminal_control_policy_sha256: str,
        observations: tuple[TtsCalibrationCellObservation, ...],
        selected_candidate_id: str,
        selected_learning_rate: float,
        selected_stride: int,
        selected_mean_slo_goodput_tps: float,
        selection_rule: str,
        _construction_seal: object,
    ) -> None:
        if _construction_seal is not _FIRST_PARTY_REDUCTION_SEAL:
            raise TypeError(
                "TTS calibration reduction must come from raw first-party evidence"
            )
        for name, value in locals().copy().items():
            if name not in {"self", "_construction_seal"}:
                object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("only TTS calibration reduction schema 2 is supported")
        for label, value in (
            ("protocol lock", self.protocol_lock_sha256),
            ("authority", self.authority_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
            ("raw manifest", self.raw_manifest_sha256),
            ("tuning window", self.tuning_window_sha256),
            ("registry", self.registry_sha256),
            ("runtime", self.runtime_sha256),
            ("split", self.split_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
            (
                "terminal control reservation",
                self.terminal_control_reservation_sha256,
            ),
            ("terminal control policy", self.terminal_control_policy_sha256),
            ("selected candidate", self.selected_candidate_id),
        ):
            _require_sha256(f"TTS reduction {label}", value)
        if (
            type(self.observations) is not tuple
            or len(self.observations) != 288
            or any(
                type(row) is not TtsCalibrationCellObservation
                for row in self.observations
            )
            or tuple(row.materialized_cell_id for row in self.observations)
            != tuple(sorted({row.materialized_cell_id for row in self.observations}))
        ):
            raise ValueError("TTS reduction must canonically cover exactly 288 cells")
        candidate_blocks: dict[str, set[int]] = {}
        for row in self.observations:
            candidate_blocks.setdefault(row.candidate_id, set()).add(row.block)
        if len(candidate_blocks) != 72 or any(
            blocks != set(PILOT_BLOCKS) for blocks in candidate_blocks.values()
        ):
            raise ValueError("TTS reduction lacks 72 candidates x four pilots")
        if (
            len({row.run_id for row in self.observations}) != 288
            or len({row.run_nonce_sha256 for row in self.observations}) != 288
        ):
            raise ValueError("TTS reduction reuses a run identity or nonce")
        if len({row.terminal_receipt_sha256 for row in self.observations}) != 288:
            raise ValueError("TTS reduction reuses a terminal receipt")
        if (
            len({row.terminal_control_binding_sha256 for row in self.observations})
            != 288
            or len({row.terminal_control_envelope_sha256 for row in self.observations})
            != 288
            or {row.terminal_control_reservation_sha256 for row in self.observations}
            != {self.terminal_control_reservation_sha256}
            or {row.terminal_control_policy_sha256 for row in self.observations}
            != {self.terminal_control_policy_sha256}
        ):
            raise ValueError(
                "TTS reduction lacks one atomic 288-terminal external-control batch"
            )
        feasible_candidates = {
            candidate_id
            for candidate_id in candidate_blocks
            if all(
                row.disposition == "ELIGIBLE"
                for row in self.observations
                if row.candidate_id == candidate_id
            )
        }
        if not feasible_candidates:
            raise ValueError("TTS reduction has no four-pilot feasible candidate")
        selected = tuple(
            row
            for row in self.observations
            if row.candidate_id == self.selected_candidate_id
        )
        if (
            len(selected) != 4
            or self.selected_candidate_id not in feasible_candidates
            or {row.block for row in selected} != set(PILOT_BLOCKS)
            or {row.learning_rate for row in selected} != {self.selected_learning_rate}
            or {row.stride for row in selected} != {self.selected_stride}
        ):
            raise ValueError(
                "TTS reduction selected candidate identity is inconsistent"
            )
        selected_mean = statistics.fmean(
            row.slo_goodput_tps for row in selected if row.slo_goodput_tps is not None
        )
        if not math.isclose(
            selected_mean,
            self.selected_mean_slo_goodput_tps,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("TTS reduction selected mean differs from raw pilots")
        ranked = sorted(
            (
                -statistics.fmean(
                    row.slo_goodput_tps
                    for row in self.observations
                    if row.candidate_id == candidate_id
                ),
                candidate_id,
            )
            for candidate_id in feasible_candidates
        )
        if not ranked or ranked[0][1] != self.selected_candidate_id:
            raise ValueError("TTS reduction does not select maximum SLO-goodput")
        if self.selection_rule != "safety_first_then_maximize_slo_goodput":
            raise ValueError("TTS reduction selection rule differs from protocol")
        reject_banned_model_identity(self)

    def validate_against(
        self,
        *,
        authority: TtsCalibrationAuthority,
        materialization: StageMaterializationReceipt,
        coverage: StageCoverageReceipt,
    ) -> None:
        if type(authority) is not TtsCalibrationAuthority:
            raise TypeError("TTS reduction requires an exact authority")
        if type(materialization) is not StageMaterializationReceipt:
            raise TypeError("TTS reduction requires an exact materialization")
        if type(coverage) is not StageCoverageReceipt:
            raise TypeError("TTS reduction requires an exact coverage receipt")
        coverage.validate_against(materialization)
        if (
            materialization.stage != "TTS-Cal"
            or self.protocol_lock_sha256 != materialization.protocol_lock_sha256
            or self.authority_sha256 != authority.sha256
            or self.materialization_receipt_sha256 != materialization.sha256
            or self.coverage_receipt_sha256 != coverage.sha256
            or self.tuning_window_sha256 != authority.tuning_window_sha256
            or self.selection_rule != authority.selection_rule
        ):
            raise ValueError("TTS reduction lineage differs from sealed inputs")
        expected = authority.candidate_id(
            learning_rate=self.selected_learning_rate,
            stride=self.selected_stride,
        )
        if expected != self.selected_candidate_id:
            raise ValueError("TTS reduction selected recipe differs from authority")

    @property
    def selected_pilot_run_binding_sha256s(self) -> tuple[str, ...]:
        return tuple(
            row.raw_run_binding_sha256
            for row in sorted(
                (
                    row
                    for row in self.observations
                    if row.candidate_id == self.selected_candidate_id
                ),
                key=lambda row: row.block,
            )
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def _qualification_rows(
    *,
    analysis: Any,
    block: int,
    reference: object,
    registry: ExperimentRegistry,
    protocol_lock_sha256: str,
    materialization: StageMaterializationReceipt,
    authority: TtsCalibrationAuthority,
    manifest: RawTtsCalibrationEvidenceManifest,
    runtime_sha256: str,
    split_sha256: str,
    loaded: dict[str, object],
) -> tuple[tuple[str, str, bool], ...]:
    value = analysis._bound_json(
        reference.path, reference.sha256, label="TTS calibration qualification lock"
    )
    required = {
        "schema_version",
        "kind",
        "registry_sha256",
        "protocol_lock_sha256",
        "materialization_receipt_sha256",
        "authority_sha256",
        "tuning_window_sha256",
        "runtime_sha256",
        "split_sha256",
        "block",
        "corpus_sha256",
        "arrival_trace_sha256",
        "request_ids_sha256",
        "sampling_profile_sha256",
        "model_lock_sha256",
        "rows",
    }
    first = loaded[min(loaded)].run_rows[0]
    if set(value) != required or any(
        value.get(name) != expected
        for name, expected in (
            ("schema_version", 1),
            ("kind", "tts_calibration_request_qualification_lock"),
            ("registry_sha256", registry.sha256),
            ("protocol_lock_sha256", protocol_lock_sha256),
            ("materialization_receipt_sha256", materialization.sha256),
            ("authority_sha256", authority.sha256),
            ("tuning_window_sha256", manifest.tuning_window.sha256),
            ("runtime_sha256", runtime_sha256),
            ("split_sha256", split_sha256),
            ("block", block),
            ("corpus_sha256", first["corpus_sha256"]),
            ("arrival_trace_sha256", first["arrival_trace_sha256"]),
            ("request_ids_sha256", first["request_ids_sha256"]),
            ("sampling_profile_sha256", first["sampling_profile_sha256"]),
            ("model_lock_sha256", first["model_lock_sha256"]),
        )
    ):
        raise ValueError("TTS qualification lock differs from run identity")
    common = (
        "corpus_sha256",
        "arrival_trace_sha256",
        "request_ids_sha256",
        "sampling_profile_sha256",
        "model_lock_sha256",
    )
    if any(
        any(cell.run_rows[0][name] != first[name] for name in common)
        for cell in loaded.values()
    ):
        raise ValueError("TTS pilot candidates do not share one tuning request set")
    rows = value.get("rows")
    if type(rows) is not list or not rows:
        raise ValueError("TTS qualification lock requires request rows")
    parsed: list[tuple[str, str, bool]] = []
    for row in rows:
        if type(row) is not dict or set(row) != {
            "request_id",
            "prompt_bucket",
            "eligible",
        }:
            raise ValueError("TTS qualification row has an ambiguous schema")
        request_id = _strict_text("TTS qualification request ID", row["request_id"])
        bucket = row["prompt_bucket"]
        eligible = row["eligible"]
        if bucket not in TTFT_LIMIT_MS or type(eligible) is not bool:
            raise ValueError("TTS qualification row is incomplete")
        parsed.append((request_id, str(bucket), eligible))
    request_ids = tuple(row[0] for row in parsed)
    if (
        len(request_ids) != len(set(request_ids))
        or content_sha256(list(request_ids)) != first["request_ids_sha256"]
        or any(
            tuple(row.get("request_id") for row in cell.request_rows) != request_ids
            for cell in loaded.values()
        )
    ):
        raise ValueError("TTS qualification request coverage differs from raw evidence")
    return tuple(parsed)


def _validate_tts_calibration_native_terminal_batch(
    *,
    analysis: Any,
    registry: ExperimentRegistry,
    manifest: RawTtsCalibrationEvidenceManifest,
    loaded: dict[str, object],
    inventory: GpuInventory,
    replay_store: ChallengeReplayStore | None,
    replay_reservation: ChallengeReplayReservationBinding | None = None,
    now_ns: int,
) -> tuple[_ControlledTtsCalibrationTerminal, ...]:
    """Deep-open and atomically authorize all 288 pulled native terminals."""

    from lightcone_spec.orchestration import (
        PreparedNativeTerminalExternalControl,
        finalize_prepared_native_terminal_external_controls,
        prepare_native_terminal_external_control,
        validate_native_terminal_artifacts_with_external_controls,
    )

    if (type(replay_store) is ChallengeReplayStore) == (
        type(replay_reservation) is ChallengeReplayReservationBinding
    ):
        raise TypeError(
            "TTS terminal controls require exactly one replay store or reservation"
        )
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("TTS terminal control verification time is invalid")
    prepared: list[PreparedNativeTerminalExternalControl] = []
    contexts: list[tuple[str, object, object]] = []
    for pilot in manifest.pilots:
        for cell, control_reference in zip(
            pilot.cells,
            pilot.terminal_control_attestations,
            strict=True,
        ):
            if len(cell.terminal_receipts) != 1:
                raise ValueError(
                    "TTS calibration requires exactly one TP1 native terminal"
                )
            terminal_reference = cell.terminal_receipts[0]
            terminal_value = analysis._bound_json(
                terminal_reference.path,
                terminal_reference.sha256,
                label="TTS calibration native terminal",
            )
            control_value = analysis._bound_json(
                control_reference.path,
                control_reference.sha256,
                label="TTS calibration native terminal external control",
            )
            control = ControlArtifactAttestation.from_dict(control_value)
            row = prepare_native_terminal_external_control(
                terminal_value,
                control_attestation=control,
                expected_inventory_sha256=inventory.sha256,
                expected_registry_sha256=registry.sha256,
            )
            if row.binding.canonical_raw_sha256 != terminal_reference.sha256:
                raise ValueError(
                    "TTS native terminal file is not the controlled canonical bytes"
                )
            loaded_cell = loaded[cell.cell_id]
            run_rows = loaded_cell.run_rows
            if len(run_rows) != 1:
                raise ValueError("TTS native terminal lacks one exact run identity")
            run = run_rows[0]
            binding = row.evidence.binding
            request_ids = tuple(
                str(request["request_id"]) for request in loaded_cell.request_rows
            )
            if (
                binding.run_id != run["run_id"]
                or binding.run_nonce_sha256 != run["run_nonce_sha256"]
                or binding.execution_plan_sha256 != run["runtime_sha256"]
                or binding.rank_config_sha256 != run["rank_config_sha256"]
                or binding.method != "tts"
                or binding.scored_request_ids != request_ids
                or binding.warmup_request_ids
            ):
                raise ValueError(
                    "TTS native terminal differs from the path-reopened run"
                )
            prepared.append(row)
            contexts.append((cell.cell_id, terminal_reference, run))
    if len(prepared) != 288:
        raise ValueError("TTS calibration external controls do not cover 288 cells")
    if replay_store is not None:
        validated = validate_native_terminal_artifacts_with_external_controls(
            tuple(prepared),
            replay_store=replay_store,
            now_ns=now_ns,
        )
    else:
        assert replay_reservation is not None
        if replay_reservation.reserved_ns > now_ns:
            raise ValueError("TTS terminal reservation is from the future")
        verified_controls = tuple(
            verify_release_control_artifact_attestation(
                row.control_attestation,
                expected_inventory_sha256=row.expected_inventory_sha256,
                now_ns=replay_reservation.reserved_ns,
                consumed_challenge_sha256s=(),
            )
            for row in prepared
        )
        validated = finalize_prepared_native_terminal_external_controls(
            tuple(prepared),
            verified_controls=verified_controls,
            replay_reservation=replay_reservation,
        )
    controlled: list[_ControlledTtsCalibrationTerminal] = []
    for row, evidence, context in zip(prepared, validated, contexts, strict=True):
        cell_id, terminal_reference, run = context
        if (
            evidence.external_control_binding_sha256 is None
            or evidence.external_control_envelope_sha256 is None
            or evidence.external_control_reservation_sha256 is None
            or evidence.external_control_trusted_policy_sha256 is None
        ):
            raise RuntimeError("TTS native terminal batch lost its external control")
        controlled.append(
            _ControlledTtsCalibrationTerminal(
                registry_cell_id=cell_id,
                canonical_raw_sha256=terminal_reference.sha256,
                terminal_sha256=evidence.terminal_sha256,
                run_id=str(run["run_id"]),
                run_nonce_sha256=str(run["run_nonce_sha256"]),
                execution_plan_sha256=str(run["runtime_sha256"]),
                rank_config_sha256=str(run["rank_config_sha256"]),
                method=evidence.binding.method,
                control_binding_sha256=(evidence.external_control_binding_sha256),
                control_envelope_sha256=(evidence.external_control_envelope_sha256),
                control_reservation_sha256=(
                    evidence.external_control_reservation_sha256
                ),
                control_policy_sha256=(evidence.external_control_trusted_policy_sha256),
            )
        )
    return tuple(sorted(controlled, key=lambda row: row.registry_cell_id))


def reduce_tts_calibration_from_raw(
    *,
    registry: ExperimentRegistry,
    materialization: StageMaterializationReceipt,
    coverage: StageCoverageReceipt,
    authority: TtsCalibrationAuthority,
    manifest: RawTtsCalibrationEvidenceManifest,
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
    runtime_sha256: str,
    split_sha256: str,
    replay_store: ChallengeReplayStore | None = None,
    replay_reservation: ChallengeReplayReservationBinding | None = None,
    now_ns: int,
) -> TtsCalibrationReductionReceipt:
    """Deep-reopen the exact 288 cells and deterministically select one recipe."""

    from lightcone_spec.experiments import industrial_analysis as analysis
    from lightcone_spec.experiments import selection_authority as selection

    if type(registry) is not ExperimentRegistry:
        raise TypeError("TTS reduction requires an exact ExperimentRegistry")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("TTS reduction requires an exact materialization")
    if type(coverage) is not StageCoverageReceipt:
        raise TypeError("TTS reduction requires an exact coverage receipt")
    if type(authority) is not TtsCalibrationAuthority:
        raise TypeError("TTS reduction requires an exact calibration authority")
    if type(manifest) is not RawTtsCalibrationEvidenceManifest:
        raise TypeError("TTS reduction requires its exact raw evidence manifest")
    if (
        type(hardware_envelope) is not HardwareEnvelope
        or type(inventory) is not GpuInventory
    ):
        raise TypeError("TTS reduction requires exact hardware and inventory authority")
    for label, value in (("runtime", runtime_sha256), ("split", split_sha256)):
        _require_sha256(f"TTS reduction {label}", value)
    coverage.validate_against(materialization)
    if (
        materialization.stage != "TTS-Cal"
        or materialization.expected_cell_count != 288
        or materialization.source_decision_sha256 != authority.sha256
        or authority.tuning_window_sha256 != manifest.tuning_window.sha256
        or any(row.status != "COMPLETE" for row in coverage.dispositions)
    ):
        raise ValueError(
            "TTS reduction inputs do not authorize exact complete calibration"
        )
    analysis.validate_raw_evidence_manifest_sidecars(manifest)
    registry_cells = {cell.cell_id: cell for cell in registry.cells_for("TTS-Cal")}
    if len(registry_cells) != 288:
        raise ValueError("TTS reduction registry lacks exact 288-cell calibration")
    materialized_by_registry: dict[str, object] = {}
    for cell in materialization.cells:
        dimensions = dict(cell.dimensions)
        registry_cell_id = dimensions.get("registry_cell_id")
        _require_sha256("TTS materialized registry cell", registry_cell_id)
        if registry_cell_id in materialized_by_registry:
            raise ValueError("TTS materialization repeats a registry cell")
        materialized_by_registry[str(registry_cell_id)] = cell
    if set(materialized_by_registry) != set(registry_cells):
        raise ValueError("TTS materialization differs from exact staged registry")
    manifest_ids = {cell.cell_id for cell in manifest.cells}
    if manifest_ids != set(registry_cells) or len(manifest.cells) != 288:
        raise ValueError("TTS raw evidence differs from exact staged registry")
    family = _TtsCalibrationRunIdentity("TTS-Cal", runtime_sha256, split_sha256)
    references = {cell.cell_id: cell for cell in manifest.cells}
    loaded = {
        cell_id: analysis._load_cell(
            references[cell_id],
            registry=registry,
            family=family,
            cells_by_id=registry_cells,
            envelope=hardware_envelope,
            inventory=inventory,
        )
        for cell_id in sorted(registry_cells)
    }
    disposition_by_id = {row.cell_id: row for row in coverage.dispositions}
    pending_observations: list[dict[str, object]] = []
    block_refs = {pilot.block: pilot for pilot in manifest.pilots}
    for block in PILOT_BLOCKS:
        registry_ids = tuple(
            sorted(
                cell_id
                for cell_id, cell in registry_cells.items()
                if cell.identity.block == block
            )
        )
        if len(registry_ids) != 72 or {
            cell.cell_id for cell in block_refs[block].cells
        } != set(registry_ids):
            raise ValueError("TTS raw pilot lacks its exact 72 candidate cells")
        block_loaded = {cell_id: loaded[cell_id] for cell_id in registry_ids}
        qualification = _qualification_rows(
            analysis=analysis,
            block=block,
            reference=block_refs[block].qualification_lock,
            registry=registry,
            protocol_lock_sha256=materialization.protocol_lock_sha256,
            materialization=materialization,
            authority=authority,
            manifest=manifest,
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
            loaded=block_loaded,
        )
        qualification_by_id = {
            request_id: (bucket, eligible)
            for request_id, bucket, eligible in qualification
        }
        for registry_cell_id in registry_ids:
            loaded_cell = block_loaded[registry_cell_id]
            metrics = selection._metrics(loaded_cell)
            slo_rows = tuple(
                SloRequest(
                    request_id=metric.request_id,
                    prompt_bucket=qualification_by_id[metric.request_id][0],
                    eligible=qualification_by_id[metric.request_id][1],
                    completed=metric.completed,
                    error=metric.error,
                    ttft_ms=metric.ttft_ms,
                    within_request_p99_itl_ms=metric.within_request_p99_itl_ms,
                )
                for metric in metrics
            )
            slo = account_slo(slo_rows)
            reasons = set(selection._safety_reasons(loaded_cell))
            if not slo.passed:
                reasons.add("slo:request_qualification_failed")
            qualified_ids = {
                row.request_id
                for row in slo_rows
                if row.eligible
                and row.completed
                and not row.error
                and row.ttft_ms is not None
                and row.within_request_p99_itl_ms is not None
                and row.ttft_ms <= TTFT_LIMIT_MS[row.prompt_bucket]
                and row.within_request_p99_itl_ms <= slo.within_request_p99_itl_limit_ms
            }
            request_rows = loaded_cell.request_rows
            elapsed_ns = max(int(row["completed_ns"]) for row in request_rows) - min(
                int(row["arrival_ns"]) for row in request_rows
            )
            goodput = sum(
                metric.output_tokens
                for metric in metrics
                if metric.request_id in qualified_ids
            ) / (elapsed_ns / 1_000_000_000.0)
            if not math.isfinite(goodput) or goodput <= 0:
                reasons.add("slo:no_positive_qualified_goodput")
            cell = registry_cells[registry_cell_id]
            materialized = materialized_by_registry[registry_cell_id]
            dimensions = dict(materialized.dimensions)
            stride = int(cell.identity.variant.removeprefix("tts_calibration:stride="))
            learning_rate = cell.identity.learning_rate
            if type(learning_rate) is not float:
                raise ValueError("TTS registry learning rate is not exact")
            candidate_id = authority.candidate_id(
                learning_rate=learning_rate, stride=stride
            )
            if (
                materialized.recipe_sha256 != candidate_id
                or dimensions.get("block") != block
                or dimensions.get("learning_rate") != learning_rate
                or dimensions.get("stride") != stride
            ):
                raise ValueError("TTS materialized candidate differs from registry")
            reference = references[registry_cell_id]
            if len(reference.terminal_receipts) != 1:
                raise ValueError("TTS calibration requires one TP1 terminal per cell")
            disposition = disposition_by_id[materialized.cell_id]
            terminal_sha256 = reference.terminal_receipts[0].sha256
            if disposition.terminal_receipt_sha256 != terminal_sha256:
                raise ValueError("TTS coverage terminal differs from raw evidence")
            run = loaded_cell.run_rows[0]
            raw_binding = {
                "schema_version": 1,
                "kind": "tts_calibration_raw_run_binding",
                "materialized_cell_id": materialized.cell_id,
                "registry_cell_id": registry_cell_id,
                "candidate_id": candidate_id,
                "block": block,
                "run_id": run["run_id"],
                "run_nonce_sha256": run["run_nonce_sha256"],
                "config_sha256": run["config_sha256"],
                "rank_config_sha256": run["rank_config_sha256"],
                "runtime_sha256": runtime_sha256,
                "split_sha256": split_sha256,
                "execution_plan_sha256": run["runtime_sha256"],
                "execution_split_sha256": run["split_sha256"],
                "terminal_receipt_sha256": terminal_sha256,
                "hardware_receipt_sha256": loaded_cell.hardware_receipt_sha256,
                "budget_observation_sha256": loaded_cell.budget_observation_sha256,
                "qualification_lock_sha256": block_refs[
                    block
                ].qualification_lock.sha256,
            }
            pending_observations.append(
                {
                    "materialized_cell_id": materialized.cell_id,
                    "registry_cell_id": registry_cell_id,
                    "candidate_id": candidate_id,
                    "learning_rate": learning_rate,
                    "stride": stride,
                    "block": block,
                    "disposition": "ELIMINATED" if reasons else "ELIGIBLE",
                    "reason_codes": tuple(sorted(reasons)),
                    "slo_goodput_tps": None if reasons else float(goodput),
                    "slo_accounting_sha256": content_sha256(slo),
                    "raw_run_binding_sha256": content_sha256(raw_binding),
                    "run_id": str(run["run_id"]),
                    "run_nonce_sha256": str(run["run_nonce_sha256"]),
                    "terminal_receipt_sha256": terminal_sha256,
                    "qualification_lock_sha256": block_refs[
                        block
                    ].qualification_lock.sha256,
                }
            )
    feasible_candidates = {
        candidate_id
        for candidate_id in authority.candidate_ids
        if all(
            pending["disposition"] == "ELIGIBLE"
            for pending in pending_observations
            if pending["candidate_id"] == candidate_id
        )
    }
    if not feasible_candidates:
        raise ValueError("TTS calibration has no safe SLO-feasible candidate")
    candidate_means = {
        candidate_id: statistics.fmean(
            float(pending["slo_goodput_tps"])
            for pending in pending_observations
            if pending["candidate_id"] == candidate_id
            and pending["slo_goodput_tps"] is not None
        )
        for candidate_id in feasible_candidates
    }
    selected_candidate_id = min(
        candidate_means,
        key=lambda candidate_id: (-candidate_means[candidate_id], candidate_id),
    )
    controlled = _validate_tts_calibration_native_terminal_batch(
        analysis=analysis,
        registry=registry,
        manifest=manifest,
        loaded=loaded,
        inventory=inventory,
        replay_store=replay_store,
        replay_reservation=replay_reservation,
        now_ns=now_ns,
    )
    controlled_by_registry = {row.registry_cell_id: row for row in controlled}
    if len(controlled_by_registry) != 288:
        raise RuntimeError("TTS terminal control batch changed cell coverage")
    observations = tuple(
        TtsCalibrationCellObservation(
            **pending,
            terminal_control_binding_sha256=(
                controlled_by_registry[
                    str(pending["registry_cell_id"])
                ].control_binding_sha256
            ),
            terminal_control_envelope_sha256=(
                controlled_by_registry[
                    str(pending["registry_cell_id"])
                ].control_envelope_sha256
            ),
            terminal_control_reservation_sha256=(
                controlled_by_registry[
                    str(pending["registry_cell_id"])
                ].control_reservation_sha256
            ),
            terminal_control_policy_sha256=(
                controlled_by_registry[
                    str(pending["registry_cell_id"])
                ].control_policy_sha256
            ),
        )
        for pending in pending_observations
    )
    ordered = tuple(sorted(observations, key=lambda row: row.materialized_cell_id))
    selected = next(row for row in ordered if row.candidate_id == selected_candidate_id)
    terminal_control_reservations = {
        row.terminal_control_reservation_sha256 for row in ordered
    }
    terminal_control_policies = {row.terminal_control_policy_sha256 for row in ordered}
    if len(terminal_control_reservations) != 1 or len(terminal_control_policies) != 1:
        raise RuntimeError("TTS terminal controls did not share one atomic authority")
    return TtsCalibrationReductionReceipt(
        schema_version=2,
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        authority_sha256=authority.sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        raw_manifest_sha256=manifest.sha256,
        tuning_window_sha256=manifest.tuning_window.sha256,
        registry_sha256=registry.sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=content_sha256(hardware_envelope),
        terminal_control_reservation_sha256=next(iter(terminal_control_reservations)),
        terminal_control_policy_sha256=next(iter(terminal_control_policies)),
        observations=ordered,
        selected_candidate_id=selected_candidate_id,
        selected_learning_rate=selected.learning_rate,
        selected_stride=selected.stride,
        selected_mean_slo_goodput_tps=float(candidate_means[selected_candidate_id]),
        selection_rule=authority.selection_rule,
        _construction_seal=_FIRST_PARTY_REDUCTION_SEAL,
    )


def seal_tts_calibration_reduction(
    reduction: TtsCalibrationReductionReceipt,
    *,
    authority: TtsCalibrationAuthority,
    materialization: StageMaterializationReceipt,
    coverage: StageCoverageReceipt,
) -> TtsCalibrationSeal:
    """Construct the only signable TTS recipe seal from a first-party reduction."""

    if type(reduction) is not TtsCalibrationReductionReceipt:
        raise TypeError("TTS calibration seal requires an exact raw reduction")
    reduction.validate_against(
        authority=authority, materialization=materialization, coverage=coverage
    )
    return TtsCalibrationSeal._from_reduction(
        reduction=reduction,
        authority=authority,
        materialization=materialization,
        coverage=coverage,
    )


FORMAL_TTS_CALIBRATION_REDUCTION_PROOF_KIND = (
    "formal_tts_calibration_reduction_proof_artifact"
)
FORMAL_TTS_CALIBRATION_REDUCTION_PROOF_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_formal_tts_calibration_reduction_proof_protocol",
        "coverage_authority": (
            "registry_rooted_portable_tts_calibration_coverage_replay"
        ),
        "reduction": "exact_288_raw_cells_safe_then_maximum_slo_goodput",
        "seal": "deterministic_first_party_payload_exact_compare",
        "caller_inputs": "no_authority_materialization_coverage_or_manifest",
    }
)


def _reduction_to_dict(
    reduction: TtsCalibrationReductionReceipt,
) -> dict[str, object]:
    if type(reduction) is not TtsCalibrationReductionReceipt:
        raise TypeError("TTS reduction proof requires an exact reduction")
    value = asdict(reduction)
    observations: list[dict[str, object]] = []
    for row in reduction.observations:
        encoded = asdict(row)
        encoded["reason_codes"] = list(row.reason_codes)
        observations.append(encoded)
    value["observations"] = observations
    return value


def _untrusted_reduction_from_dict(value: object) -> TtsCalibrationReductionReceipt:
    """Decode only inside the proof revalidator; never export this constructor."""

    expected = set(TtsCalibrationReductionReceipt.__dataclass_fields__)
    if type(value) is not dict or set(value) != expected:
        raise ValueError("TTS reduction proof payload fields differ")
    row = dict(value)
    raw_observations = row.pop("observations")
    if type(raw_observations) is not list:
        raise TypeError("TTS reduction proof observations must be an array")
    observations: list[TtsCalibrationCellObservation] = []
    observation_fields = set(TtsCalibrationCellObservation.__dataclass_fields__)
    for raw in raw_observations:
        if type(raw) is not dict or set(raw) != observation_fields:
            raise ValueError("TTS reduction proof observation fields differ")
        item = dict(raw)
        reasons = item["reason_codes"]
        if type(reasons) not in {list, tuple}:
            raise TypeError("TTS reduction proof reason codes must be an array")
        item["reason_codes"] = tuple(reasons)
        observations.append(TtsCalibrationCellObservation(**item))
    return TtsCalibrationReductionReceipt(
        **row,
        observations=tuple(observations),
        _construction_seal=_FIRST_PARTY_REDUCTION_SEAL,
    )


def _open_proof_binding(
    binding: CanonicalJsonProofBinding,
    *,
    label: str,
) -> dict[str, object]:
    observed = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if observed != binding:
        raise ValueError(f"{label} path identity changed")
    value = observed.reopen()
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != observed:
        raise RuntimeError(f"{label} changed while reopened")
    return value


@dataclass(frozen=True)
class FormalTtsCalibrationReductionProofArtifact:
    """Portable raw-evidence proof for the unique 288-cell TTS winner."""

    schema_version: Literal[2]
    kind: Literal["formal_tts_calibration_reduction_proof_artifact"]
    protocol_sha256: str
    portable_coverage_proof_source: CanonicalJsonProofBinding
    hardware_envelope_source: CanonicalJsonProofBinding
    replay_reservation: ChallengeReplayReservationBinding
    runtime_sha256: str
    split_sha256: str
    reduction_payload: dict[str, object]
    expected_reduction_sha256: str
    expected_seal_payload_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 2
            or self.kind != FORMAL_TTS_CALIBRATION_REDUCTION_PROOF_KIND
            or self.protocol_sha256
            != FORMAL_TTS_CALIBRATION_REDUCTION_PROOF_PROTOCOL_SHA256
        ):
            raise ValueError("TTS reduction proof schema is unsupported")
        bindings = (
            self.portable_coverage_proof_source,
            self.hardware_envelope_source,
        )
        if any(type(row) is not CanonicalJsonProofBinding for row in bindings):
            raise TypeError("TTS reduction proof sources must be path-bound")
        paths = tuple(row.absolute_path for row in bindings)
        if len(paths) != len(set(paths)):
            raise ValueError("TTS reduction proof reuses a source path")
        if type(self.replay_reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("TTS reduction proof replay reservation is not exact")
        for label, digest in (
            ("runtime", self.runtime_sha256),
            ("split", self.split_sha256),
            ("reduction", self.expected_reduction_sha256),
            ("seal payload", self.expected_seal_payload_sha256),
        ):
            _require_sha256(f"TTS reduction proof {label}", digest)
        reduction = _untrusted_reduction_from_dict(self.reduction_payload)
        if reduction.sha256 != self.expected_reduction_sha256:
            raise ValueError("TTS reduction proof payload digest differs")
        if (
            reduction.runtime_sha256 != self.runtime_sha256
            or reduction.split_sha256 != self.split_sha256
            or reduction.terminal_control_reservation_sha256
            != self.replay_reservation.reservation_sha256
        ):
            raise ValueError("TTS reduction proof payload lineage differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "portable_coverage_proof_source": (
                self.portable_coverage_proof_source.to_dict()
            ),
            "hardware_envelope_source": self.hardware_envelope_source.to_dict(),
            "replay_reservation": self.replay_reservation.to_dict(),
            "runtime_sha256": self.runtime_sha256,
            "split_sha256": self.split_sha256,
            "reduction_payload": self.reduction_payload,
            "expected_reduction_sha256": self.expected_reduction_sha256,
            "expected_seal_payload_sha256": self.expected_seal_payload_sha256,
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "artifact_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("TTS reduction proof artifact fields differ")
        row = dict(value)
        declared = _require_sha256(
            "TTS reduction proof artifact", row.pop("artifact_sha256")
        )
        for name in (
            "portable_coverage_proof_source",
            "hardware_envelope_source",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        row["replay_reservation"] = ChallengeReplayReservationBinding.from_dict(
            row["replay_reservation"]
        )
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("TTS reduction proof artifact digest differs")
        return artifact


def build_formal_tts_calibration_reduction_proof_artifact(
    *,
    portable_coverage_proof_path: str | Path,
    hardware_envelope_source_path: str | Path,
    replay_reservation: ChallengeReplayReservationBinding,
    runtime_sha256: str,
    split_sha256: str,
    now_ns: int,
) -> FormalTtsCalibrationReductionProofArtifact:
    """Derive the unique reduction proof without a caller-authored winner."""

    from lightcone_spec.experiments.formal_stage_coverage_portable import (
        revalidate_portable_formal_stage_coverage_proof_artifact,
    )
    from lightcone_spec.experiments.registry import build_industrial_registry

    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("TTS reduction proof binding time is invalid")
    _require_sha256("TTS reduction proof runtime", runtime_sha256)
    _require_sha256("TTS reduction proof split", split_sha256)
    portable_binding = CanonicalJsonProofBinding.bind(portable_coverage_proof_path)
    context = revalidate_portable_formal_stage_coverage_proof_artifact(
        portable_binding.absolute_path,
        now_ns=now_ns,
    )
    authority = context.tts_calibration_authority
    manifest = context.raw_tts_evidence_manifest
    if (
        context.materialization.stage != "TTS-Cal"
        or authority is None
        or manifest is None
    ):
        raise ValueError("TTS reduction proof requires portable TTS-Cal coverage")
    hardware_binding = CanonicalJsonProofBinding.bind(hardware_envelope_source_path)
    hardware = _hardware_envelope_from_proof(hardware_binding.reopen())
    rebuilt = reduce_tts_calibration_from_raw(
        registry=build_industrial_registry(),
        materialization=context.materialization,
        coverage=context.coverage,
        authority=authority,
        manifest=manifest,
        hardware_envelope=hardware,
        inventory=context.inventory,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        replay_reservation=replay_reservation,
        now_ns=now_ns,
    )
    derived_seal = seal_tts_calibration_reduction(
        rebuilt,
        authority=authority,
        materialization=context.materialization,
        coverage=context.coverage,
    )
    artifact = FormalTtsCalibrationReductionProofArtifact(
        schema_version=2,
        kind=FORMAL_TTS_CALIBRATION_REDUCTION_PROOF_KIND,
        protocol_sha256=FORMAL_TTS_CALIBRATION_REDUCTION_PROOF_PROTOCOL_SHA256,
        portable_coverage_proof_source=portable_binding,
        hardware_envelope_source=hardware_binding,
        replay_reservation=replay_reservation,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        reduction_payload=_reduction_to_dict(rebuilt),
        expected_reduction_sha256=rebuilt.sha256,
        expected_seal_payload_sha256=derived_seal.sha256,
    )
    artifact.__post_init__()
    return artifact


def bind_formal_tts_calibration_reduction_proof_artifact(
    reduction: TtsCalibrationReductionReceipt,
    *,
    portable_coverage_proof_path: str | Path,
    hardware_envelope_source_path: str | Path,
    replay_reservation: ChallengeReplayReservationBinding,
    now_ns: int,
) -> FormalTtsCalibrationReductionProofArtifact:
    """Compatibility binder that exact-compares a first-party reduction."""

    if type(reduction) is not TtsCalibrationReductionReceipt:
        raise TypeError("TTS reduction proof binder requires an exact reduction")
    artifact = build_formal_tts_calibration_reduction_proof_artifact(
        portable_coverage_proof_path=portable_coverage_proof_path,
        hardware_envelope_source_path=hardware_envelope_source_path,
        replay_reservation=replay_reservation,
        runtime_sha256=reduction.runtime_sha256,
        split_sha256=reduction.split_sha256,
        now_ns=now_ns,
    )
    rebuilt = _untrusted_reduction_from_dict(artifact.reduction_payload)
    if rebuilt != reduction or rebuilt.sha256 != reduction.sha256:
        raise ValueError("TTS reduction proof binder replay selected another result")
    return artifact


def publish_formal_tts_calibration_reduction_proof_artifact(
    artifact: FormalTtsCalibrationReductionProofArtifact,
    output_path: str | Path,
    *,
    now_ns: int,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not FormalTtsCalibrationReductionProofArtifact:
        raise TypeError("TTS reduction proof publisher requires exact input")
    _rebuild_formal_tts_calibration_reduction_proof_artifact(
        artifact,
        now_ns=now_ns,
    )
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(output_path)


def _hardware_envelope_from_proof(value: object) -> HardwareEnvelope:
    fields = set(HardwareEnvelope.__dataclass_fields__)
    if type(value) is not dict or set(value) != fields:
        raise ValueError("TTS reduction hardware envelope fields differ")
    row = dict(value)
    for name in ("allowed_throttling_reasons", "allowed_background_processes"):
        raw = row[name]
        if type(raw) is not list:
            raise TypeError("TTS reduction hardware tuple must be an array")
        row[name] = tuple(raw)
    return HardwareEnvelope(**row)  # type: ignore[arg-type]


def _rebuild_formal_tts_calibration_reduction_proof_artifact(
    artifact: FormalTtsCalibrationReductionProofArtifact,
    *,
    now_ns: int,
) -> tuple[
    TtsCalibrationReductionReceipt,
    TtsCalibrationSeal,
    TtsCalibrationAuthority,
]:
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("TTS reduction proof verification time is invalid")
    from lightcone_spec.experiments.formal_stage_coverage_portable import (
        revalidate_portable_formal_stage_coverage_proof_artifact,
    )
    from lightcone_spec.experiments.registry import build_industrial_registry

    portable_binding = CanonicalJsonProofBinding.bind(
        artifact.portable_coverage_proof_source.absolute_path
    )
    if portable_binding != artifact.portable_coverage_proof_source:
        raise ValueError("TTS reduction portable coverage identity changed")
    context = revalidate_portable_formal_stage_coverage_proof_artifact(
        artifact.portable_coverage_proof_source.absolute_path,
        now_ns=now_ns,
    )
    if (
        CanonicalJsonProofBinding.bind(
            artifact.portable_coverage_proof_source.absolute_path
        )
        != portable_binding
    ):
        raise RuntimeError("TTS reduction portable coverage changed while reopened")
    authority = context.tts_calibration_authority
    manifest = context.raw_tts_evidence_manifest
    if (
        context.materialization.stage != "TTS-Cal"
        or authority is None
        or manifest is None
    ):
        raise ValueError("TTS reduction proof portable coverage is not TTS-Cal")
    hardware = _hardware_envelope_from_proof(
        _open_proof_binding(
            artifact.hardware_envelope_source,
            label="TTS hardware envelope",
        )
    )
    expected = _untrusted_reduction_from_dict(artifact.reduction_payload)
    rebuilt = reduce_tts_calibration_from_raw(
        registry=build_industrial_registry(),
        materialization=context.materialization,
        coverage=context.coverage,
        authority=authority,
        manifest=manifest,
        hardware_envelope=hardware,
        inventory=context.inventory,
        runtime_sha256=artifact.runtime_sha256,
        split_sha256=artifact.split_sha256,
        replay_reservation=artifact.replay_reservation,
        now_ns=now_ns,
    )
    if rebuilt != expected or rebuilt.sha256 != artifact.expected_reduction_sha256:
        raise ValueError("TTS reduction proof replay selected another result")
    seal = seal_tts_calibration_reduction(
        rebuilt,
        authority=authority,
        materialization=context.materialization,
        coverage=context.coverage,
    )
    if seal.sha256 != artifact.expected_seal_payload_sha256:
        raise ValueError("TTS reduction proof replay selected another seal")
    return rebuilt, seal, authority


def revalidate_formal_tts_calibration_reduction_proof_artifact(
    artifact_path: str | Path,
    *,
    now_ns: int,
) -> tuple[TtsCalibrationReductionReceipt, TtsCalibrationSeal]:
    """Deep-reduce all 288 raw rows under the durable replay reservation."""

    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = FormalTtsCalibrationReductionProofArtifact.from_dict(binding.reopen())
    rebuilt, seal, _authority = (
        _rebuild_formal_tts_calibration_reduction_proof_artifact(
            artifact,
            now_ns=now_ns,
        )
    )
    if CanonicalJsonProofBinding.bind(artifact_path) != binding:
        raise RuntimeError("TTS reduction proof changed while reopened")
    return rebuilt, seal


def revalidate_signed_tts_calibration_seal_from_reduction_proof(
    artifact_path: str | Path,
    *,
    signed_seal: SignedTtsCalibrationSeal,
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    now_ns: int,
) -> TtsCalibrationSeal:
    """Deep-reduce before trusting the signed recipe payload."""

    if type(signed_seal) is not SignedTtsCalibrationSeal:
        raise TypeError("TTS reduction proof requires an exact signed seal")
    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = FormalTtsCalibrationReductionProofArtifact.from_dict(binding.reopen())
    _reduction, expected, authority = (
        _rebuild_formal_tts_calibration_reduction_proof_artifact(
            artifact,
            now_ns=now_ns,
        )
    )
    verified = signed_seal.verify(
        authority=authority,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    if verified != expected or verified.sha256 != expected.sha256:
        raise ValueError("signed TTS seal differs from raw reduction proof")
    if CanonicalJsonProofBinding.bind(artifact_path) != binding:
        raise RuntimeError("TTS reduction proof changed while verifying its seal")
    return verified


__all__ = (
    "FORMAL_TTS_CALIBRATION_REDUCTION_PROOF_KIND",
    "FORMAL_TTS_CALIBRATION_REDUCTION_PROOF_PROTOCOL_SHA256",
    "FormalTtsCalibrationReductionProofArtifact",
    "TtsCalibrationCellObservation",
    "TtsCalibrationReductionReceipt",
    "bind_formal_tts_calibration_reduction_proof_artifact",
    "build_formal_tts_calibration_reduction_proof_artifact",
    "publish_formal_tts_calibration_reduction_proof_artifact",
    "reduce_tts_calibration_from_raw",
    "revalidate_formal_tts_calibration_reduction_proof_artifact",
    "revalidate_signed_tts_calibration_seal_from_reduction_proof",
    "seal_tts_calibration_reduction",
)
