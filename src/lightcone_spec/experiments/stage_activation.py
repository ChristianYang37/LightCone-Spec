"""Reducer-owned activation for registry stages without a bespoke reducer.

The generic reducer never consumes a cell list.  It replays the immutable
registry, the exact dependency-receipt chain, runtime/split identities, and the
release dispatchability predicate to disposition every stage cell.  E1, E2,
and confirmation-family stages retain their dedicated reducers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Literal

from lightcone_spec.experiments.registry import (
    FROZEN_TTS_RECIPE_SENTINEL,
    INDUSTRIAL_EXPERIMENT_ORDER,
    SEALED_E2_RECIPE_SENTINEL,
    CellStatus,
    ExperimentCell,
    ExperimentReceipt,
    ExperimentRegistry,
    LockedOutput,
    WorkloadClass,
    content_sha256,
    serving_cell_rejection_reason,
)
from lightcone_spec.runtime.compile_cache import (
    RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE,
)

_SHA256_LENGTH = 64
_SPECIAL_REDUCER_STAGES = frozenset({"E1", "E2", "E3b", "E5"})
_GENERIC_REGISTRY_STAGES = tuple(
    stage
    for stage in INDUSTRIAL_EXPERIMENT_ORDER
    if stage not in _SPECIAL_REDUCER_STAGES
)

RELEASE_DOWNLOAD_ASSIGNMENT_CONTRACT_UNAVAILABLE = (
    "release_download_assignment_contract_unavailable"
)

REGISTRY_STAGE_RELEASE_CAPABILITY_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "industrial_registry_stage_release_capability",
        "generic_stages": _GENERIC_REGISTRY_STAGES,
        "preflight": (
            "only_registered_static_interference_calibration_cells_are_serving"
        ),
        "compile": "blocked_without_first_party_prewarm_and_result_pointer",
        "download": "blocked_without_first_party_download_terminal_contract",
        "serving": (
            "target_only_static_and_single_rank_dflash_core_when_exact_semantics_"
            "are_bound_by_a_first_party_reducer"
        ),
        "scientific_identity": (
            "frozen_tts_and_l0_naive_blocked_l0_candidates_e1_e2_only_"
            "lightcone_requires_sealed_e2_receipt"
        ),
        "unsupported_methods": (
            "onlinespec_ogd",
            "onlinespec_opt",
            "onlinespec_ens",
        ),
        "missing_semantics": "blocked",
    }
)
REGISTRY_STAGE_ACTIVATION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "industrial_registry_stage_activation_reducer",
        "inputs": (
            "exact_registry",
            "complete_validated_dependency_receipt_prefix",
            "content_bound_runtime_identity",
            "content_bound_split_identity",
            "release_dispatchability_predicate",
        ),
        "root_authority": "canonical_registry_genesis_v1",
        "cell_selection_input_forbidden": True,
        "all_stage_cells_dispositioned": True,
        "release_capability_sha256": REGISTRY_STAGE_RELEASE_CAPABILITY_SHA256,
    }
)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(name: str, value: object) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be lower-case SHA-256")
    return value


def _require_text(name: str, value: object) -> str:
    if type(value) is not str or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be non-empty single-line text")
    return value


def _strict_object(name: str, value: object, fields: frozenset[str]) -> dict[str, Any]:
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


class RegistryStageDispositionStatus(str, Enum):
    ACTIVATED = "ACTIVATED"
    BLOCKED = "BLOCKED"
    NOT_APPLICABLE = "N/A"


@dataclass(frozen=True)
class RegistryStageCellDisposition:
    cell_id: str
    status: RegistryStageDispositionStatus
    reason_code: str

    def __post_init__(self) -> None:
        _require_sha256("registry-stage disposition cell", self.cell_id)
        if not isinstance(self.status, RegistryStageDispositionStatus):
            raise TypeError("registry-stage disposition status is invalid")
        reason = _require_text("registry-stage disposition reason", self.reason_code)
        if any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
            for character in reason
        ):
            raise ValueError("registry-stage disposition reason code is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class RegistryGenesisAuthority:
    schema_version: int
    registry_sha256: str
    root_experiment: str
    reducer_protocol_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only registry genesis authority schema 1 is supported")
        _require_sha256("genesis registry", self.registry_sha256)
        if self.root_experiment != INDUSTRIAL_EXPERIMENT_ORDER[0]:
            raise ValueError("registry genesis authority names a non-root experiment")
        if self.reducer_protocol_sha256 != (REGISTRY_STAGE_ACTIVATION_PROTOCOL_SHA256):
            raise ValueError("registry genesis authority uses another reducer")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "industrial_registry_genesis_authority",
            "registry_sha256": self.registry_sha256,
            "root_experiment": self.root_experiment,
            "dependency_experiments": [],
            "reducer_protocol_sha256": self.reducer_protocol_sha256,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def _source_authority_sha256(
    *,
    registry_sha256: str,
    experiment: str,
    runtime_sha256: str,
    split_sha256: str,
    dependency_receipt_sha256s: tuple[str, ...],
    genesis_authority_sha256: str | None,
) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "industrial_registry_stage_activation_source",
            "registry_sha256": registry_sha256,
            "experiment": experiment,
            "runtime_sha256": runtime_sha256,
            "split_sha256": split_sha256,
            "dependency_receipt_sha256s": dependency_receipt_sha256s,
            "genesis_authority_sha256": genesis_authority_sha256,
            "release_capability_sha256": (REGISTRY_STAGE_RELEASE_CAPABILITY_SHA256),
            "reducer_protocol_sha256": (REGISTRY_STAGE_ACTIVATION_PROTOCOL_SHA256),
        }
    )


@dataclass(frozen=True)
class RegistryStageActivationArtifact:
    schema_version: int
    registry_sha256: str
    experiment: str
    runtime_sha256: str
    split_sha256: str
    dependency_receipts: tuple[ExperimentReceipt, ...]
    genesis_authority: RegistryGenesisAuthority | None
    release_capability_sha256: str
    reducer_protocol_sha256: str
    source_authority_sha256: str
    activation_round: Literal["registry_release_dispatchability_v1"]
    status: Literal["AVAILABLE", "BLOCKED"]
    dispositions: tuple[RegistryStageCellDisposition, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "only registry-stage activation artifact schema 1 is supported"
            )
        for name in ("registry_sha256", "runtime_sha256", "split_sha256"):
            _require_sha256(f"registry-stage {name}", getattr(self, name))
        if self.experiment not in _GENERIC_REGISTRY_STAGES:
            raise ValueError("stage requires a bespoke activation reducer")
        if self.release_capability_sha256 != (REGISTRY_STAGE_RELEASE_CAPABILITY_SHA256):
            raise ValueError("registry-stage activation uses another release policy")
        if self.reducer_protocol_sha256 != (REGISTRY_STAGE_ACTIVATION_PROTOCOL_SHA256):
            raise ValueError("registry-stage activation uses another reducer")
        if self.activation_round != "registry_release_dispatchability_v1":
            raise ValueError("registry-stage activation round is invalid")
        if any(
            type(receipt) is not ExperimentReceipt
            for receipt in self.dependency_receipts
        ):
            raise TypeError("registry-stage dependencies must be exact receipts")
        stage_index = INDUSTRIAL_EXPERIMENT_ORDER.index(self.experiment)
        expected_dependencies = INDUSTRIAL_EXPERIMENT_ORDER[:stage_index]
        if (
            tuple(receipt.experiment for receipt in self.dependency_receipts)
            != expected_dependencies
        ):
            raise ValueError(
                "registry-stage activation lacks the exact dependency receipt prefix"
            )
        if any(
            receipt.registry_sha256 != self.registry_sha256
            for receipt in self.dependency_receipts
        ):
            raise ValueError("registry-stage dependency belongs to another registry")
        if stage_index == 0:
            if (
                self.genesis_authority is None
                or self.genesis_authority.registry_sha256 != self.registry_sha256
                or self.genesis_authority.root_experiment != self.experiment
            ):
                raise ValueError("root activation lacks canonical genesis authority")
        elif self.genesis_authority is not None:
            raise ValueError("non-root activation cannot claim genesis authority")
        expected_source = _source_authority_sha256(
            registry_sha256=self.registry_sha256,
            experiment=self.experiment,
            runtime_sha256=self.runtime_sha256,
            split_sha256=self.split_sha256,
            dependency_receipt_sha256s=tuple(
                receipt.sha256 for receipt in self.dependency_receipts
            ),
            genesis_authority_sha256=(
                None
                if self.genesis_authority is None
                else self.genesis_authority.sha256
            ),
        )
        if self.source_authority_sha256 != expected_source:
            raise ValueError("registry-stage source authority identity mismatch")
        if not self.dispositions:
            raise ValueError("registry-stage activation requires cell dispositions")
        if self.dispositions != tuple(
            sorted(self.dispositions, key=lambda row: row.cell_id)
        ) or len({row.cell_id for row in self.dispositions}) != len(self.dispositions):
            raise ValueError(
                "registry-stage dispositions must be cell-sorted and unique"
            )
        active = self.activated_cell_ids
        expected_status = "AVAILABLE" if active else "BLOCKED"
        if self.status != expected_status:
            raise ValueError("registry-stage status differs from its dispositions")

    @property
    def activated_cell_ids(self) -> tuple[str, ...]:
        return tuple(
            row.cell_id
            for row in self.dispositions
            if row.status is RegistryStageDispositionStatus.ACTIVATED
        )

    @property
    def direct_dependency_receipt_sha256(self) -> str | None:
        if not self.dependency_receipts:
            return None
        return self.dependency_receipts[-1].sha256

    def _payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "registry_sha256": self.registry_sha256,
            "experiment": self.experiment,
            "runtime_sha256": self.runtime_sha256,
            "split_sha256": self.split_sha256,
            "dependency_receipts": [
                receipt.to_dict() for receipt in self.dependency_receipts
            ],
            "genesis_authority": (
                None
                if self.genesis_authority is None
                else self.genesis_authority.to_dict()
            ),
            "release_capability_sha256": self.release_capability_sha256,
            "reducer_protocol_sha256": self.reducer_protocol_sha256,
            "source_authority_sha256": self.source_authority_sha256,
            "activation_round": self.activation_round,
            "status": self.status,
            "dispositions": [row.to_dict() for row in self.dispositions],
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self._payload_dict())


def release_execution_capability_rejection_reason(
    cell: ExperimentCell,
) -> str | None:
    """Check release method/topology capability without resolving template axes.

    This narrower predicate is only for cells selected by a verified bespoke
    reducer.  It cannot make a generic registry template executable: callers
    without reducer authority must use :func:`release_dispatch_rejection_reason`.
    """

    if type(cell) is not ExperimentCell:
        raise TypeError("release capability requires an exact registry cell")
    if cell.status is not CellStatus.UNMEASURED:
        return cell.reason_code
    # Resource isolation is only a placement property.  It is not terminal
    # authority.  This release has neither an exact compile-workload manifest
    # and atomic cache result pointer nor a first-party download receipt
    # contract, so neither non-serving class may enter the scheduler.
    if cell.resources.workload_class is WorkloadClass.COMPILE:
        return RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE
    if cell.resources.workload_class is WorkloadClass.DOWNLOAD:
        return RELEASE_DOWNLOAD_ASSIGNMENT_CONTRACT_UNAVAILABLE
    if cell.identity.experiment == "preflight":
        if is_serving_interference_calibration_cell(cell):
            return None
        return "release_preflight_method_unsupported"
    if cell.identity.topology != "tp1_dp1":
        return "release_topology_executor_unsupported"
    identity = cell.identity
    method = identity.method
    if method in {"target_only", "static"}:
        return None
    recipe_markers = {identity.optimizer, identity.schedule}
    if FROZEN_TTS_RECIPE_SENTINEL in recipe_markers:
        return "tts_official_recipe_unavailable"
    if SEALED_E2_RECIPE_SENTINEL in recipe_markers:
        return "sealed_e2_recipe_receipt_required"
    if method == "tts":
        return "tts_frozen_recipe_authority_required"
    if method == "l0" and identity.experiment not in {"E1", "E2"}:
        return "sealed_e2_recipe_receipt_required"
    if method == "l0" and identity.backend == "DFLASH":
        return None
    if method == "l0":
        return "release_adaptive_backend_unsupported"
    if method.startswith("onlinespec_"):
        return "release_onlinespec_execution_contract_unavailable"
    return "release_method_capability_unsupported"


def release_dispatch_rejection_reason(cell: ExperimentCell) -> str | None:
    """Return why a cell lacks both release capability and exact semantics."""

    capability = release_execution_capability_rejection_reason(cell)
    if capability is not None:
        return capability
    if is_serving_interference_calibration_cell(cell):
        return None
    if serving_cell_rejection_reason(cell) is not None:
        return "release_serving_contract_unresolved"
    if cell.identity.method not in {"target_only", "static", "tts", "l0"}:
        return "release_method_capability_unsupported"
    return None


def is_serving_interference_calibration_cell(cell: ExperimentCell) -> bool:
    """Identify the only preflight cells executed through serving evidence."""

    if type(cell) is not ExperimentCell:
        raise TypeError("interference calibration predicate requires an exact cell")
    return (
        cell.identity.experiment == "preflight"
        and cell.identity.task == "simultaneous_single_gpu_interference"
        and cell.identity.method == "static"
        and cell.resources.workload_class is WorkloadClass.CORRECTNESS
    )


def _genesis_authority(registry: ExperimentRegistry) -> RegistryGenesisAuthority:
    return RegistryGenesisAuthority(
        schema_version=1,
        registry_sha256=registry.sha256,
        root_experiment=INDUSTRIAL_EXPERIMENT_ORDER[0],
        reducer_protocol_sha256=REGISTRY_STAGE_ACTIVATION_PROTOCOL_SHA256,
    )


def materialize_registry_stage_activation(
    registry: ExperimentRegistry,
    *,
    experiment: str,
    dependency_receipts: Sequence[ExperimentReceipt] = (),
    runtime_sha256: str,
    split_sha256: str,
) -> RegistryStageActivationArtifact:
    """Disposition a generic stage solely from validated release authority."""

    if type(registry) is not ExperimentRegistry:
        raise TypeError("registry-stage activation requires an exact registry")
    if experiment not in _GENERIC_REGISTRY_STAGES:
        raise ValueError("stage requires its registered bespoke activation reducer")
    _require_sha256("registry-stage runtime", runtime_sha256)
    _require_sha256("registry-stage split", split_sha256)
    stage_index = INDUSTRIAL_EXPERIMENT_ORDER.index(experiment)
    expected_dependencies = INDUSTRIAL_EXPERIMENT_ORDER[:stage_index]
    receipts = tuple(dependency_receipts)
    if any(type(receipt) is not ExperimentReceipt for receipt in receipts):
        raise TypeError("registry-stage dependencies must be exact receipts")
    validated = registry.validate_receipts(receipts)
    if set(validated) != set(expected_dependencies):
        raise ValueError(
            "registry-stage activation requires the complete dependency receipt prefix"
        )
    ordered_receipts = tuple(validated[name] for name in expected_dependencies)
    genesis = _genesis_authority(registry) if stage_index == 0 else None
    rows: list[RegistryStageCellDisposition] = []
    for cell in registry.cells_for(experiment):
        if cell.status is CellStatus.NOT_APPLICABLE:
            status = RegistryStageDispositionStatus.NOT_APPLICABLE
            reason = cell.reason_code
        elif cell.status is not CellStatus.UNMEASURED:
            status = RegistryStageDispositionStatus.BLOCKED
            reason = cell.reason_code
        else:
            rejection = release_dispatch_rejection_reason(cell)
            status = (
                RegistryStageDispositionStatus.ACTIVATED
                if rejection is None
                else RegistryStageDispositionStatus.BLOCKED
            )
            reason = rejection or "release_dispatchability_verified"
        rows.append(
            RegistryStageCellDisposition(
                cell_id=cell.cell_id,
                status=status,
                reason_code=reason,
            )
        )
    return RegistryStageActivationArtifact(
        schema_version=1,
        registry_sha256=registry.sha256,
        experiment=experiment,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        dependency_receipts=ordered_receipts,
        genesis_authority=genesis,
        release_capability_sha256=REGISTRY_STAGE_RELEASE_CAPABILITY_SHA256,
        reducer_protocol_sha256=REGISTRY_STAGE_ACTIVATION_PROTOCOL_SHA256,
        source_authority_sha256=_source_authority_sha256(
            registry_sha256=registry.sha256,
            experiment=experiment,
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
            dependency_receipt_sha256s=tuple(
                receipt.sha256 for receipt in ordered_receipts
            ),
            genesis_authority_sha256=None if genesis is None else genesis.sha256,
        ),
        activation_round="registry_release_dispatchability_v1",
        status=(
            "AVAILABLE"
            if any(
                row.status is RegistryStageDispositionStatus.ACTIVATED for row in rows
            )
            else "BLOCKED"
        ),
        dispositions=tuple(sorted(rows, key=lambda row: row.cell_id)),
    )


def verify_registry_stage_activation(
    registry: ExperimentRegistry,
    artifact: RegistryStageActivationArtifact,
) -> None:
    """Replay the reducer and reject any edited or self-authorizing artifact."""

    if type(artifact) is not RegistryStageActivationArtifact:
        raise TypeError("registry-stage authority must be the exact artifact type")
    expected = materialize_registry_stage_activation(
        registry,
        experiment=artifact.experiment,
        dependency_receipts=artifact.dependency_receipts,
        runtime_sha256=artifact.runtime_sha256,
        split_sha256=artifact.split_sha256,
    )
    if artifact != expected:
        raise ValueError(
            "registry-stage activation is not the exact reducer-generated artifact"
        )


def _locked_output_from_dict(value: object) -> LockedOutput:
    row = _strict_object("locked output", value, frozenset({"name", "content_sha256"}))
    return LockedOutput(
        name=_require_text("locked output name", row["name"]),
        content_sha256=_require_sha256("locked output content", row["content_sha256"]),
    )


def _receipt_from_dict(value: object) -> ExperimentReceipt:
    row = _strict_object(
        "registry-stage dependency receipt",
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
    return ExperimentReceipt(
        experiment=_require_text("receipt experiment", row["experiment"]),
        registry_sha256=_require_sha256("receipt registry", row["registry_sha256"]),
        runtime_sha256=_require_sha256("receipt runtime", row["runtime_sha256"]),
        split_sha256=_require_sha256("receipt split", row["split_sha256"]),
        completed_cells_sha256=_require_sha256(
            "receipt completed cells", row["completed_cells_sha256"]
        ),
        dependency_receipts=tuple(
            _locked_output_from_dict(item)
            for item in _strict_list(
                "receipt dependency receipts", row["dependency_receipts"]
            )
        ),
        outputs=tuple(
            _locked_output_from_dict(item)
            for item in _strict_list("receipt outputs", row["outputs"])
        ),
        selection_state=_require_text(
            "receipt selection state", row["selection_state"]
        ),
    )


def _genesis_from_dict(value: object) -> RegistryGenesisAuthority:
    row = _strict_object(
        "registry genesis authority",
        value,
        frozenset(
            {
                "schema_version",
                "kind",
                "registry_sha256",
                "root_experiment",
                "dependency_experiments",
                "reducer_protocol_sha256",
            }
        ),
    )
    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != 1
        or row["kind"] != ("industrial_registry_genesis_authority")
    ):
        raise ValueError("registry genesis authority identity mismatch")
    if _strict_list("genesis dependency experiments", row["dependency_experiments"]):
        raise ValueError("registry genesis authority cannot have dependencies")
    return RegistryGenesisAuthority(
        schema_version=1,
        registry_sha256=_require_sha256("genesis registry", row["registry_sha256"]),
        root_experiment=_require_text(
            "genesis root experiment", row["root_experiment"]
        ),
        reducer_protocol_sha256=_require_sha256(
            "genesis reducer", row["reducer_protocol_sha256"]
        ),
    )


def registry_stage_activation_to_dict(
    artifact: RegistryStageActivationArtifact,
) -> dict[str, object]:
    if type(artifact) is not RegistryStageActivationArtifact:
        raise TypeError("registry-stage serializer requires the exact artifact")
    return {
        "artifact_kind": "registry_stage_activation",
        "artifact_sha256": artifact.sha256,
        **artifact._payload_dict(),
    }


def registry_stage_activation_from_dict(
    value: object,
) -> RegistryStageActivationArtifact:
    row = _strict_object(
        "registry-stage activation artifact",
        value,
        frozenset(
            {
                "artifact_kind",
                "artifact_sha256",
                "schema_version",
                "registry_sha256",
                "experiment",
                "runtime_sha256",
                "split_sha256",
                "dependency_receipts",
                "genesis_authority",
                "release_capability_sha256",
                "reducer_protocol_sha256",
                "source_authority_sha256",
                "activation_round",
                "status",
                "dispositions",
            }
        ),
    )
    if row["artifact_kind"] != "registry_stage_activation":
        raise ValueError("registry-stage artifact kind mismatch")
    dispositions = tuple(
        RegistryStageCellDisposition(
            cell_id=_require_sha256("disposition cell", disposition["cell_id"]),
            status=RegistryStageDispositionStatus(
                _require_text("disposition status", disposition["status"])
            ),
            reason_code=_require_text("disposition reason", disposition["reason_code"]),
        )
        for disposition in (
            _strict_object(
                "registry-stage disposition",
                item,
                frozenset({"cell_id", "status", "reason_code"}),
            )
            for item in _strict_list("registry-stage dispositions", row["dispositions"])
        )
    )
    genesis_value = row["genesis_authority"]
    artifact = RegistryStageActivationArtifact(
        schema_version=row["schema_version"],
        registry_sha256=_require_sha256(
            "registry-stage registry", row["registry_sha256"]
        ),
        experiment=_require_text("registry-stage experiment", row["experiment"]),
        runtime_sha256=_require_sha256("registry-stage runtime", row["runtime_sha256"]),
        split_sha256=_require_sha256("registry-stage split", row["split_sha256"]),
        dependency_receipts=tuple(
            _receipt_from_dict(item)
            for item in _strict_list(
                "registry-stage dependency receipts", row["dependency_receipts"]
            )
        ),
        genesis_authority=(
            None if genesis_value is None else _genesis_from_dict(genesis_value)
        ),
        release_capability_sha256=_require_sha256(
            "registry-stage release capability", row["release_capability_sha256"]
        ),
        reducer_protocol_sha256=_require_sha256(
            "registry-stage reducer", row["reducer_protocol_sha256"]
        ),
        source_authority_sha256=_require_sha256(
            "registry-stage source authority", row["source_authority_sha256"]
        ),
        activation_round=_require_text(
            "registry-stage activation round", row["activation_round"]
        ),
        status=_require_text("registry-stage status", row["status"]),
        dispositions=dispositions,
    )
    if row["artifact_sha256"] != artifact.sha256:
        raise ValueError("registry-stage artifact SHA-256 mismatch")
    return artifact


__all__ = [
    "REGISTRY_STAGE_ACTIVATION_PROTOCOL_SHA256",
    "REGISTRY_STAGE_RELEASE_CAPABILITY_SHA256",
    "RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE",
    "RELEASE_DOWNLOAD_ASSIGNMENT_CONTRACT_UNAVAILABLE",
    "RegistryGenesisAuthority",
    "RegistryStageActivationArtifact",
    "RegistryStageCellDisposition",
    "RegistryStageDispositionStatus",
    "is_serving_interference_calibration_cell",
    "materialize_registry_stage_activation",
    "registry_stage_activation_from_dict",
    "registry_stage_activation_to_dict",
    "release_dispatch_rejection_reason",
    "release_execution_capability_rejection_reason",
    "verify_registry_stage_activation",
]
