"""Path-bound replay authority for industrial budget materialization.

Budget arithmetic is first-party only when the reducer reopens the generated
registry, one strict tagged raw activation authority and every nested source,
policy, cell-sorted load bindings, capacity envelope and capacity authority,
then exactly reproduces the declared :class:`BudgetPlan`.  A serialized plan,
activation, selection, power summary, or cell-ID list is never execution
authority by itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightcone_spec.experiments.gpu_pool import GpuInventory

from lightcone_spec.experiments.capacity_authority import bind_capacity_authority
from lightcone_spec.experiments.completion_authority import (
    CompletedCellAuthority,
    CompletionAuthorityUnavailableError,
    DurableJsonArtifactBinding,
)
from lightcone_spec.experiments.planning import (
    BUDGET_MATERIALIZATION_AUTHORITY_PROTOCOL_SHA256,
    BudgetActivationAuthorityBinding,
    BudgetInventoryIdentity,
    BudgetLoadBinding,
    BudgetLoadRawBinding,
    BudgetMaterializationAuthorityBinding,
    BudgetPlan,
    BudgetPolicy,
    BudgetRawJsonBinding,
    CapacityAuthorityBinding,
    CapacityEnvelope,
    ConfirmationAuxiliaryActivationAuthorityBinding,
    ConfirmationAuxiliaryCompletionAuthorityBinding,
    ConfirmationFamilyCompletionAuthorityBinding,
    ConfirmationFamilyPowerReductionArtifact,
    ConfirmationFinalActivationAuthorityBinding,
    ConfirmationPilotActivationAuthorityBinding,
    ConfirmationStageAggregateAuthorityBinding,
    ConfirmationStageFamilyAuthorityBinding,
    DependencyGpuInventoryAuthorityBinding,
    DependencyLockedOutputAuthorityBinding,
    E1ActivationAuthorityBinding,
    E1ParetoArtifact,
    E2ActivationAuthorityBinding,
    E2StageCompletionAuthorityBinding,
    E2StageReductionArtifact,
    FamilyActivationArtifact,
    FamilyPilotCompletionAuthorityBinding,
    ReducerActivationArtifact,
    RegistryStageActivationAuthorityBinding,
    RegistryStageDependencyCompletionAuthorityBinding,
    SealedE3aSelection,
    _budget_activation_raw_sources,
    derive_confirmation_family,
    derive_confirmation_stage_partition,
    materialize_confirmation_auxiliary_activation,
    materialize_confirmation_pilots,
    materialize_confirmation_prefix,
    materialize_industrial_budgets,
    reduce_e1_activation,
    reduce_e2_activation,
)
from lightcone_spec.experiments.planning_artifacts import (
    budget_load_binding_from_dict,
    budget_plan_from_dict,
    budget_policy_from_dict,
    capacity_envelope_from_dict,
)
from lightcone_spec.experiments.registry import (
    E2_HALVING_STAGES,
    INDUSTRIAL_EXPERIMENT_ORDER,
    CellStatus,
    ExperimentReceipt,
    ExperimentRegistry,
    LockedOutput,
    WorkloadClass,
    build_legacy_industrial_registry,
    content_sha256,
)

# Historical budget/completion fixtures are defined against the eager
# diagnostic registry.  Keep the module-local legacy name for those explicit
# compatibility callers; the public registry default remains signed-staged.
build_industrial_registry = build_legacy_industrial_registry
from lightcone_spec.experiments.stage_activation import (
    RegistryStageActivationArtifact,
    is_serving_interference_calibration_cell,
    materialize_registry_stage_activation,
    verify_registry_stage_activation,
)

BUDGET_MATERIALIZATION_UNRESOLVED_REASON = "industrial_budget_plan_unresolved"
DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON = (
    "dependency_completion_manifest_authority_missing"
)
DEPENDENCY_COMPLETION_LOCKED_OUTPUT_UNSUPPORTED_REASON = (
    "dependency_completion_non_json_locked_output_authority_unsupported"
)
DEPENDENCY_COMPLETION_SPECIALIZED_ACTIVATION_REASON = (
    "dependency_completion_specialized_activation_authority_unsupported"
)
DEPENDENCY_COMPLETION_GPU_INVENTORY_MISSING_REASON = (
    "dependency_completion_full_gpu_inventory_missing"
)
DEPENDENCY_COMPLETION_FAMILY_STAGE_AGGREGATION_MISSING_REASON = (
    "dependency_completion_family_stage_aggregation_missing"
)
E2_STAGE_COMPLETION_AUTHORITY_MISSING_REASON = (
    "e2_prior_stage_schema_v4_completion_authority_missing"
)

_REGISTRY_GENERATOR = (
    "lightcone_spec.experiments.registry.build_legacy_industrial_registry:v3"
)
_MANIFEST_V1_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "registry_artifact",
        "experiment",
        "runtime_artifact",
        "split_artifact",
        "dependency_receipts",
    }
)
_MANIFEST_V2_FIELDS = _MANIFEST_V1_FIELDS | frozenset(
    {"dependency_completion_authorities"}
)
_E1_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "registry_artifact",
        "runtime_artifact",
        "split_artifact",
        "dependency_receipt",
        "dependency_completion_authority",
        "selection_manifest",
        "gpu_inventory",
        "inventory_source_receipt",
        "hardware_envelope",
    }
)
_E2_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "registry_artifact",
        "runtime_artifact",
        "split_artifact",
        "dependency_receipt",
        "dependency_completion_authority",
        "pareto_manifest",
        "prior_stage_manifests",
        "prior_stage_completion_authorities",
        "stage_index",
        "gpu_inventory",
        "inventory_source_receipt",
        "hardware_envelope",
    }
)
_E2_STAGE_COMPLETION_FIELDS = frozenset(
    {"completed_cells_artifact", "activation_manifest"}
)
_CONFIRMATION_PILOT_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "registry_artifact",
        "runtime_artifact",
        "split_artifact",
        "trace_artifact",
        "sampling_artifact",
        "dependency_receipts",
        "dependency_completion_authorities",
        "family_seed_cell_id",
        "gpu_inventory",
        "inventory_source_receipt",
        "hardware_envelope",
    }
)
_CONFIRMATION_AUXILIARY_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "registry_artifact",
        "experiment",
        "runtime_artifact",
        "split_artifact",
        "trace_artifact",
        "sampling_artifact",
        "dependency_receipts",
        "dependency_completion_authorities",
        "gpu_inventory",
        "inventory_source_receipt",
        "hardware_envelope",
    }
)
_CONFIRMATION_FINAL_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "pilot_activation_manifest",
        "pilot_completed_cells",
        "power_manifest",
    }
)
_CONFIRMATION_STAGE_AGGREGATE_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "registry_artifact",
        "experiment",
        "stage_receipt",
        "stage_completed_cells",
        "runtime_artifact",
        "split_artifact",
        "gpu_inventory",
        "inventory_source_receipt",
        "families",
        "auxiliary",
    }
)
_CONFIRMATION_STAGE_FAMILY_FIELDS = frozenset(
    {"family_sha256", "final_activation_manifest", "completed_cells_artifact"}
)
_CONFIRMATION_STAGE_AUXILIARY_FIELDS = frozenset(
    {"activation_manifest", "completed_cells_artifact"}
)
_DEPENDENCY_COMPLETION_FIELDS = frozenset(
    {
        "receipt_artifact",
        "completed_cells_artifact",
        "activation_manifest",
        "inventory_artifact",
        "inventory_source_receipt",
        "locked_outputs",
    }
)
_DEPENDENCY_LOCKED_OUTPUT_FIELDS = frozenset({"name", "artifact"})
_INVENTORY_SOURCE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "challenge_nonce_sha256",
        "host_id",
        "hostname",
        "machine_id_sha256",
        "commands",
        "parsed_topology",
        "pci_locality",
        "receipt_sha256",
    }
)
_INVENTORY_PCI_FIELDS = frozenset(
    {"index", "uuid", "pci_bus_id", "pci_root", "numa_node"}
)
_REGISTRY_FIELDS = frozenset(
    {
        "schema_version",
        "generator",
        "parameters",
        "registry_sha256",
        "registry",
    }
)
_REGISTRY_PARAMETER_FIELDS = frozenset(
    {
        "logical_gpu_slots",
        "base_port",
        "cache_root",
        "evidence_root",
        "seed",
    }
)
_RECEIPT_FIELDS = frozenset(
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
)
_LOCKED_OUTPUT_FIELDS = frozenset({"name", "content_sha256"})
_WRAPPER_KIND_BY_ROLE = {
    "budget_policy": "budget_policy",
    "budget_load_binding": "budget_load_binding",
    "capacity_envelope": "capacity_envelope",
    "declared_budget_plan": "budget_plan",
}


class BudgetMaterializationBlockedError(RuntimeError):
    """Named fail-closed outcome from an exact but non-READY raw replay."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(
            f"formal budget materialization is BLOCKED: {self.reason_code}"
        )


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


def _strict_text(name: str, value: object) -> str:
    if type(value) is not str or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be non-empty single-line text")
    return value


def _strict_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a JSON integer")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _parse_json(body: bytes, *, label: str) -> object:
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    _validate_finite_json(value, label=label)
    return value


def _validate_finite_json(value: object, *, label: str) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if type(value) is list:
        for item in value:
            _validate_finite_json(item, label=label)
    elif type(value) is dict:
        for item in value.values():
            _validate_finite_json(item, label=label)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _exact_existing_path(path: str | Path, *, label: str) -> Path:
    source = Path(path)
    if not source.is_absolute():
        raise ValueError(f"{label} path must be absolute and resolved")
    try:
        resolved = source.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} source is missing") from error
    if resolved != source:
        raise ValueError(f"{label} path must be absolute, resolved, and non-symlink")
    return source


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    source = _exact_existing_path(path, label=label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise RuntimeError(f"{label} is not a readable regular file") from error
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            body = handle.read()
        opened_after = os.fstat(descriptor)
        current = source.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or (opened_before.st_dev, opened_before.st_ino)
            != (opened_after.st_dev, opened_after.st_ino)
            or opened_before.st_size != opened_after.st_size
            or opened_before.st_mtime_ns != opened_after.st_mtime_ns
            or current.st_dev != opened_after.st_dev
            or current.st_ino != opened_after.st_ino
            or current.st_size != opened_after.st_size
            or current.st_mtime_ns != opened_after.st_mtime_ns
            or opened_after.st_size != len(body)
        ):
            raise RuntimeError(f"{label} changed while it was read")
        return body
    finally:
        os.close(descriptor)


def _locked_output_from_value(value: object, *, label: str) -> LockedOutput:
    row = _strict_object(label, value, _LOCKED_OUTPUT_FIELDS)
    return LockedOutput(
        name=_strict_text(f"{label} name", row["name"]),
        content_sha256=_strict_text(f"{label} content SHA-256", row["content_sha256"]),
    )


def _receipt_from_value(value: object) -> ExperimentReceipt:
    row = _strict_object("activation dependency receipt", value, _RECEIPT_FIELDS)
    receipt = ExperimentReceipt(
        experiment=_strict_text("receipt experiment", row["experiment"]),
        registry_sha256=_strict_text("receipt registry", row["registry_sha256"]),
        runtime_sha256=_strict_text("receipt runtime", row["runtime_sha256"]),
        split_sha256=_strict_text("receipt split", row["split_sha256"]),
        completed_cells_sha256=_strict_text(
            "receipt completed cells", row["completed_cells_sha256"]
        ),
        dependency_receipts=tuple(
            _locked_output_from_value(item, label="receipt dependency")
            for item in _strict_list("receipt dependencies", row["dependency_receipts"])
        ),
        outputs=tuple(
            _locked_output_from_value(item, label="receipt output")
            for item in _strict_list("receipt outputs", row["outputs"])
        ),
        selection_state=_strict_text("receipt selection state", row["selection_state"]),
    )
    if row != receipt.to_dict():
        raise ValueError("activation dependency receipt is not canonical")
    return receipt


@cache
def _regenerate_registry(
    *,
    slots: tuple[str, ...],
    base_port: int,
    cache_root: str,
    evidence_root: str,
    seed: int,
) -> ExperimentRegistry:
    """Memoize only the deterministic reducer, never a raw artifact read."""

    return build_legacy_industrial_registry(
        gpu_uuids=slots,
        base_port=base_port,
        cache_root=cache_root,
        evidence_root=evidence_root,
        seed=seed,
    )


def _generated_registry_from_value(value: object) -> ExperimentRegistry:
    row = _strict_object("generated registry artifact", value, _REGISTRY_FIELDS)
    if (
        _strict_int("generated registry schema", row["schema_version"]) != 3
        or row["generator"] != _REGISTRY_GENERATOR
    ):
        raise ValueError("generated registry identity mismatch")
    parameters = _strict_object(
        "generated registry parameters",
        row["parameters"],
        _REGISTRY_PARAMETER_FIELDS,
    )
    slots = tuple(
        _strict_text("generated registry logical slot", item)
        for item in _strict_list(
            "generated registry logical slots", parameters["logical_gpu_slots"]
        )
    )
    if not slots or len(slots) != len(set(slots)):
        raise ValueError("generated registry requires unique logical slots")
    registry = _regenerate_registry(
        slots=slots,
        base_port=_strict_int("generated registry base port", parameters["base_port"]),
        cache_root=_strict_text(
            "generated registry cache root", parameters["cache_root"]
        ),
        evidence_root=_strict_text(
            "generated registry evidence root", parameters["evidence_root"]
        ),
        seed=_strict_int("generated registry seed", parameters["seed"]),
    )
    if (
        row["registry_sha256"] != registry.sha256
        or row["registry"] != registry.to_dict()
    ):
        raise ValueError("generated registry differs from first-party generation")
    return registry


def _gpu_inventory_from_value(value: object):
    from lightcone_spec.experiments.gpu_pool import GpuInventory

    try:
        inventory = GpuInventory.from_dict(value)
    except (TypeError, ValueError) as error:
        raise ValueError("dependency GPU inventory is not canonical") from error
    if inventory.to_dict() != value:
        raise ValueError("dependency GPU inventory is not canonical")
    return inventory


def _inventory_source_receipt_sha256(value: object) -> str:
    row = _strict_object(
        "dependency GPU inventory source receipt",
        value,
        _INVENTORY_SOURCE_RECEIPT_FIELDS,
    )
    if (
        _strict_int("inventory source receipt schema", row["schema_version"]) != 1
        or row["kind"] != "gpu_inventory_probe_receipt"
    ):
        raise ValueError("dependency GPU inventory source receipt schema is invalid")
    for name in (
        "challenge_nonce_sha256",
        "host_id",
        "hostname",
        "machine_id_sha256",
    ):
        _strict_text(f"inventory source receipt {name}", row[name])
    _strict_object(
        "inventory source receipt commands",
        row["commands"],
        frozenset({"gpu", "processes", "topology"}),
    )
    if type(row["parsed_topology"]) is not dict:
        raise TypeError("inventory source parsed topology must be a JSON object")
    pci_rows = tuple(
        _strict_object("inventory source PCI row", item, _INVENTORY_PCI_FIELDS)
        for item in _strict_list("inventory source PCI locality", row["pci_locality"])
    )
    pci_uuids = tuple(
        _strict_text("inventory source PCI UUID", item["uuid"]) for item in pci_rows
    )
    if len(pci_uuids) != len(set(pci_uuids)):
        raise ValueError("inventory source PCI identities are duplicated")
    declared = _strict_text("inventory source receipt SHA-256", row["receipt_sha256"])
    content = {key: item for key, item in row.items() if key != "receipt_sha256"}
    if declared != hashlib.sha256(_canonical_bytes(content)).hexdigest():
        raise ValueError("inventory source receipt digest is forged")
    return declared


def _validate_gpu_inventory_source(inventory, source_value: object) -> None:
    source_sha256 = _inventory_source_receipt_sha256(source_value)
    row = source_value
    if type(row) is not dict:  # pragma: no cover - strict decoder above
        raise TypeError("inventory source receipt must be an object")
    pci_rows = tuple(
        _strict_object("inventory source PCI row", item, _INVENTORY_PCI_FIELDS)
        for item in _strict_list("inventory source PCI locality", row["pci_locality"])
    )
    expected = tuple(
        sorted(
            (
                device.uuid,
                device.pci_bus_id,
                device.pci_root,
                device.numa_node,
            )
            for device in inventory.devices
        )
    )
    actual = tuple(
        sorted(
            (
                _strict_text("inventory source PCI UUID", item["uuid"]),
                _strict_text("inventory source PCI bus", item["pci_bus_id"]),
                _strict_text("inventory source PCI root", item["pci_root"]),
                _strict_int("inventory source NUMA node", item["numa_node"]),
            )
            for item in pci_rows
        )
    )
    if (
        source_sha256 != inventory.source_receipt_sha256
        or row["host_id"] not in inventory.host_ids
        or len(inventory.host_ids) != 1
        or actual != expected
    ):
        raise ValueError("dependency GPU inventory differs from its raw source")


def _hardware_envelope_from_value(value: object):
    from lightcone_spec.experiments.statistics import HardwareEnvelope

    row = _strict_object(
        "activation hardware envelope",
        value,
        frozenset(
            {
                "gpu_clock_mhz_min",
                "gpu_clock_mhz_max",
                "memory_clock_mhz_min",
                "memory_clock_mhz_max",
                "temperature_c_max",
                "power_watts_min",
                "power_watts_max",
                "power_state",
            }
        ),
    )
    numeric: dict[str, float] = {}
    for name in (
        "gpu_clock_mhz_min",
        "gpu_clock_mhz_max",
        "memory_clock_mhz_min",
        "memory_clock_mhz_max",
        "temperature_c_max",
        "power_watts_min",
        "power_watts_max",
    ):
        value = row[name]
        if (
            type(value) not in {int, float}
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"activation hardware envelope {name} is invalid")
        numeric[name] = float(value)
    envelope = HardwareEnvelope(
        **numeric,
        power_state=_strict_text(
            "activation hardware envelope power state", row["power_state"]
        ),
    )
    if {
        **{name: getattr(envelope, name) for name in numeric},
        "power_state": envelope.power_state,
    } != {**numeric, "power_state": row["power_state"]}:
        raise ValueError("activation hardware envelope is not canonical")
    return envelope


def _load_hardware_envelope(binding: BudgetRawJsonBinding):
    if binding.role != "activation_hardware_envelope":
        raise TypeError("activation requires its raw hardware envelope")
    return _hardware_envelope_from_value(load_budget_raw_json(binding))


def _strict_path_field(row: dict[str, Any], name: str, *, label: str) -> str:
    return _strict_text(f"{label} {name}", row[name])


def _strict_optional_path(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _strict_text(label, value)


def _strict_path_list(value: object, *, label: str) -> tuple[str, ...]:
    rows = tuple(
        _strict_text(f"{label} path", item) for item in _strict_list(label, value)
    )
    if len(rows) != len(set(rows)):
        raise ValueError(f"{label} paths must be unique")
    return rows


def _artifact_wrapper_sha256(
    value: object,
    *,
    role: str,
) -> str:
    row = value if type(value) is dict else None
    expected_kind = _WRAPPER_KIND_BY_ROLE[role]
    if row is None or row.get("artifact_kind") != expected_kind:
        raise ValueError(f"{role} requires the exact {expected_kind} wrapper")
    semantic = row.get("artifact_sha256")
    if type(semantic) is not str:
        raise TypeError(f"{role} artifact SHA-256 must be text")
    return semantic


def _semantic_sha256(role: str, value: object) -> str:
    if role == "generated_registry":
        return _generated_registry_from_value(value).sha256
    if role == "dependency_gpu_inventory":
        return _gpu_inventory_from_value(value).sha256
    if role == "dependency_gpu_inventory_source_receipt":
        return _inventory_source_receipt_sha256(value)
    if role == "activation_dependency_receipt":
        return _receipt_from_value(value).sha256
    if role in _WRAPPER_KIND_BY_ROLE:
        semantic = _artifact_wrapper_sha256(value, role=role)
        if role == "budget_policy":
            decoded = budget_policy_from_dict(value)
        elif role == "budget_load_binding":
            decoded = budget_load_binding_from_dict(value)
        elif role == "capacity_envelope":
            decoded = capacity_envelope_from_dict(value)
        else:
            decoded = budget_plan_from_dict(value)
        if decoded.sha256 != semantic:
            raise ValueError(f"{role} semantic identity mismatch")
        return semantic
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _bound_semantic_sha256(role: str, value: object) -> str:
    """Read redundant semantics cheaply after exact bytes are already bound.

    Formal callers immediately feed the returned JSON through the role's
    first-party decoder.  Repeating that potentially large decode here would
    hash every request corpus twice at every consumer boundary.
    """

    if role == "generated_registry":
        row = _strict_object("generated registry artifact", value, _REGISTRY_FIELDS)
        semantic = row["registry_sha256"]
        if type(semantic) is not str:
            raise TypeError("generated registry SHA-256 must be text")
        return semantic
    if role == "dependency_gpu_inventory":
        if type(value) is not dict:
            raise TypeError("dependency GPU inventory must be an object")
        semantic = value.get("inventory_sha256")
        if semantic is None:
            return _gpu_inventory_from_value(value).sha256
        if type(semantic) is not str:
            raise TypeError("dependency GPU inventory SHA-256 must be text")
        return semantic
    if role == "dependency_gpu_inventory_source_receipt":
        if type(value) is not dict:
            raise TypeError("dependency GPU inventory source must be an object")
        semantic = value.get("receipt_sha256")
        if type(semantic) is not str:
            raise TypeError("dependency GPU inventory source SHA-256 must be text")
        return semantic
    if role == "activation_dependency_receipt":
        _strict_object("activation dependency receipt", value, _RECEIPT_FIELDS)
        return hashlib.sha256(_canonical_bytes(value)).hexdigest()
    if role in _WRAPPER_KIND_BY_ROLE:
        return _artifact_wrapper_sha256(value, role=role)
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def bind_budget_raw_json(
    path: str | Path,
    *,
    role: str,
) -> BudgetRawJsonBinding:
    """Bind one current CLI JSON artifact and its canonical ``.sha256``."""

    source = _exact_existing_path(path, label=f"{role} raw JSON")
    sidecar = _exact_existing_path(
        Path(f"{source}.sha256"), label=f"{role} raw JSON sidecar"
    )
    body = _regular_file_bytes(source, label=f"{role} raw JSON")
    sidecar_body = _regular_file_bytes(sidecar, label=f"{role} raw JSON sidecar")
    value = _parse_json(body, label=f"{role} raw JSON")
    canonical_sha256 = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    if sidecar_body != f"{canonical_sha256}\n".encode("ascii"):
        raise ValueError(f"{role} raw JSON sidecar is missing or invalid")
    binding = BudgetRawJsonBinding(
        schema_version=1,
        role=role,
        path=str(source),
        sidecar_path=str(sidecar),
        canonical_sha256=canonical_sha256,
        semantic_sha256=_semantic_sha256(role, value),
        file_sha256=hashlib.sha256(body).hexdigest(),
        sidecar_file_sha256=hashlib.sha256(sidecar_body).hexdigest(),
        size=len(body),
        sidecar_size=len(sidecar_body),
    )
    load_budget_raw_json(binding)
    return binding


def load_budget_raw_json(binding: BudgetRawJsonBinding) -> object:
    """Reopen one raw binding and reject path, bytes, sidecar, or semantic drift."""

    if type(binding) is not BudgetRawJsonBinding:
        raise TypeError("budget source must be an exact raw JSON binding")
    source = Path(binding.path)
    sidecar = Path(binding.sidecar_path)
    body = _regular_file_bytes(source, label=f"bound {binding.role} JSON")
    sidecar_body = _regular_file_bytes(
        sidecar, label=f"bound {binding.role} JSON sidecar"
    )
    value = _parse_json(body, label=f"bound {binding.role} JSON")
    canonical_sha256 = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    if (
        len(body) != binding.size
        or len(sidecar_body) != binding.sidecar_size
        or hashlib.sha256(body).hexdigest() != binding.file_sha256
        or hashlib.sha256(sidecar_body).hexdigest() != binding.sidecar_file_sha256
        or canonical_sha256 != binding.canonical_sha256
        or sidecar_body != f"{canonical_sha256}\n".encode("ascii")
    ):
        raise RuntimeError(f"bound {binding.role} JSON source or sidecar changed")
    if _bound_semantic_sha256(binding.role, value) != binding.semantic_sha256:
        raise RuntimeError(f"bound {binding.role} semantic identity changed")
    return value


def _activation_manifest_from_value(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("registry-stage activation manifest must be a JSON object")
    schema_version = _strict_int(
        "activation manifest schema", value.get("schema_version")
    )
    if schema_version not in {1, 2}:
        raise ValueError("registry-stage activation manifest schema is unsupported")
    row = _strict_object(
        "registry-stage activation manifest",
        value,
        _MANIFEST_V1_FIELDS if schema_version == 1 else _MANIFEST_V2_FIELDS,
    )
    if row["kind"] != "industrial_registry_stage_activation_manifest":
        raise ValueError(
            "budget authority supports only a raw registry-stage activation manifest"
        )
    _strict_text("activation manifest experiment", row["experiment"])
    for name in ("registry_artifact", "runtime_artifact", "split_artifact"):
        _strict_text(f"activation manifest {name}", row[name])
    dependencies = tuple(
        _strict_text("activation dependency path", item)
        for item in _strict_list(
            "activation dependency paths", row["dependency_receipts"]
        )
    )
    if len(dependencies) != len(set(dependencies)):
        raise ValueError("activation dependency paths must be unique")
    if schema_version == 2:
        completion_specs = tuple(
            _dependency_completion_spec_from_value(item)
            for item in _strict_list(
                "activation dependency completion authorities",
                row["dependency_completion_authorities"],
            )
        )
        if (
            len(completion_specs) != len(dependencies)
            or tuple(item["receipt_artifact"] for item in completion_specs)
            != dependencies
        ):
            raise ValueError(
                "activation completion authorities must exactly cover receipt paths"
            )
    return row


def _dependency_completion_spec_from_value(value: object) -> dict[str, Any]:
    row = _strict_object(
        "registry-stage dependency completion authority",
        value,
        _DEPENDENCY_COMPLETION_FIELDS,
    )
    for name in (
        "receipt_artifact",
        "completed_cells_artifact",
        "activation_manifest",
        "inventory_artifact",
        "inventory_source_receipt",
    ):
        _strict_text(f"dependency completion {name}", row[name])
    outputs = tuple(
        _strict_object(
            "dependency completion locked output",
            item,
            _DEPENDENCY_LOCKED_OUTPUT_FIELDS,
        )
        for item in _strict_list(
            "dependency completion locked outputs", row["locked_outputs"]
        )
    )
    output_names = tuple(
        _strict_text("dependency completion locked-output name", item["name"])
        for item in outputs
    )
    if output_names != tuple(sorted(set(output_names))):
        raise ValueError(
            "dependency completion locked outputs must be name-sorted and unique"
        )
    for item in outputs:
        _strict_text("dependency completion locked-output path", item["artifact"])
    return row


@dataclass(frozen=True)
class _DependencyCompletionRecord:
    binding: RegistryStageDependencyCompletionAuthorityBinding
    receipt: ExperimentReceipt
    authority: CompletedCellAuthority
    lineage_records: tuple[_DependencyCompletionRecord, ...] = ()


@dataclass(frozen=True)
class BudgetActivationAuthorityResult:
    """Exact reducer outputs reconstructed from one tagged raw authority."""

    binding: BudgetActivationAuthorityBinding
    registry: ExperimentRegistry
    activation_artifact: (
        RegistryStageActivationArtifact | ReducerActivationArtifact | None
    )
    family_activations: tuple[FamilyActivationArtifact, ...]
    family_power_reductions: tuple[ConfirmationFamilyPowerReductionArtifact, ...]
    dependency_records: tuple[_DependencyCompletionRecord, ...]
    prior_family_authorities: tuple[CompletedCellAuthority, ...]
    stage_family_authorities: tuple[CompletedCellAuthority, ...] = ()
    auxiliary_authority: CompletedCellAuthority | None = None
    prior_e2_stage_authorities: tuple[CompletedCellAuthority, ...] = ()
    e3a_selection: SealedE3aSelection | None = None
    e1_pareto: E1ParetoArtifact | None = None
    prior_e2_reductions: tuple[E2StageReductionArtifact, ...] = ()

    def __post_init__(self) -> None:
        if type(self.binding) not in {
            RegistryStageActivationAuthorityBinding,
            E1ActivationAuthorityBinding,
            E2ActivationAuthorityBinding,
            ConfirmationAuxiliaryActivationAuthorityBinding,
            ConfirmationPilotActivationAuthorityBinding,
            ConfirmationFinalActivationAuthorityBinding,
            ConfirmationStageAggregateAuthorityBinding,
        }:
            raise TypeError("activation authority result requires a tagged binding")
        if type(self.registry) is not ExperimentRegistry:
            raise TypeError("activation authority result requires an exact registry")
        if self.activation_artifact is not None and type(
            self.activation_artifact
        ) not in {RegistryStageActivationArtifact, ReducerActivationArtifact}:
            raise TypeError("activation authority result has an invalid stage artifact")
        if any(
            type(value) is not FamilyActivationArtifact
            for value in self.family_activations
        ):
            raise TypeError("activation authority result has invalid family activation")
        if any(
            type(value) is not ConfirmationFamilyPowerReductionArtifact
            for value in self.family_power_reductions
        ):
            raise TypeError("activation authority result has invalid family power")
        if (
            self.e3a_selection is not None
            and type(self.e3a_selection) is not SealedE3aSelection
        ):
            raise TypeError("activation authority result has invalid E3a selection")
        if self.e1_pareto is not None and type(self.e1_pareto) is not E1ParetoArtifact:
            raise TypeError("activation authority result has invalid E1 Pareto")
        if any(
            type(value) is not E2StageReductionArtifact
            for value in self.prior_e2_reductions
        ):
            raise TypeError("activation authority result has invalid E2 reduction")
        if any(
            type(value) is not CompletedCellAuthority
            for value in self.prior_e2_stage_authorities
        ):
            raise TypeError("activation authority result has invalid E2 completion")
        if any(
            type(value) is not CompletedCellAuthority
            for value in self.stage_family_authorities
        ):
            raise TypeError("activation authority result has invalid family completion")
        if (
            self.auxiliary_authority is not None
            and type(self.auxiliary_authority) is not CompletedCellAuthority
        ):
            raise TypeError(
                "activation authority result has invalid auxiliary completion"
            )
        if (
            self.activation_artifact is not None
            and type(self.binding) is not ConfirmationStageAggregateAuthorityBinding
            and (
                self.family_activations
                or self.family_power_reductions
                or self.prior_family_authorities
            )
        ):
            raise ValueError("stage and family activation outputs cannot be mixed")
        if self.activation_artifact is None and not self.family_activations:
            raise ValueError("activation authority result produced no activation")
        if (
            type(self.binding)
            in {
                RegistryStageActivationAuthorityBinding,
                E1ActivationAuthorityBinding,
                E2ActivationAuthorityBinding,
                ConfirmationAuxiliaryActivationAuthorityBinding,
            }
            and self.activation_artifact is None
        ):
            raise ValueError("stage activation binding produced family output")
        if (
            type(self.binding)
            in {
                ConfirmationPilotActivationAuthorityBinding,
                ConfirmationFinalActivationAuthorityBinding,
            }
            and self.activation_artifact is not None
        ):
            raise ValueError("family activation binding produced a stage artifact")
        if type(self.binding) is E2ActivationAuthorityBinding:
            if len(self.prior_e2_stage_authorities) != self.binding.stage_index:
                raise ValueError("E2 activation result lacks prior round completions")
        elif self.prior_e2_stage_authorities:
            raise ValueError("non-E2 activation carries E2 round completions")
        if type(self.binding) is ConfirmationStageAggregateAuthorityBinding:
            if len(self.stage_family_authorities) != len(self.binding.families):
                raise ValueError("confirmation aggregate lacks family completions")
            if (self.binding.auxiliary_completion_authority is None) != (
                self.auxiliary_authority is None
            ):
                raise ValueError("confirmation aggregate auxiliary completion differs")
        elif self.stage_family_authorities:
            raise ValueError("non-aggregate activation carries family completions")
        elif self.auxiliary_authority is not None:
            raise ValueError("non-aggregate activation carries auxiliary completion")

    @property
    def activation_sha256(self) -> str:
        if type(self.binding) is ConfirmationStageAggregateAuthorityBinding:
            return self.binding.activation_sha256
        if self.activation_artifact is not None:
            return self.activation_artifact.sha256
        if not self.family_activations:
            raise ValueError("activation authority produced no activation")
        return self.family_activations[-1].sha256

    @property
    def selected_activation(
        self,
    ) -> (
        RegistryStageActivationArtifact
        | ReducerActivationArtifact
        | FamilyActivationArtifact
    ):
        if type(self.binding) is ConfirmationStageAggregateAuthorityBinding:
            return self.family_activations[-1]
        if self.activation_artifact is not None:
            return self.activation_artifact
        if not self.family_activations:
            raise ValueError("activation authority produced no activation")
        return self.family_activations[-1]

    @property
    def runtime_sha256(self) -> str:
        if self.activation_artifact is not None:
            return (
                self.activation_artifact.plan.runtime_sha256
                if type(self.activation_artifact) is ReducerActivationArtifact
                else self.activation_artifact.runtime_sha256
            )
        return self.family_activations[0].family.runtime_sha256

    @property
    def split_sha256(self) -> str:
        if self.activation_artifact is not None:
            return (
                self.activation_artifact.plan.split_sha256
                if type(self.activation_artifact) is ReducerActivationArtifact
                else self.activation_artifact.split_sha256
            )
        return self.family_activations[0].family.split_sha256

    @property
    def experiment(self) -> str:
        if self.activation_artifact is not None:
            return (
                self.activation_artifact.plan.experiment
                if type(self.activation_artifact) is ReducerActivationArtifact
                else self.activation_artifact.experiment
            )
        return self.family_activations[0].family.experiment

    @property
    def dependency_receipts(self) -> tuple[ExperimentReceipt, ...]:
        if type(self.activation_artifact) is RegistryStageActivationArtifact:
            return self.activation_artifact.dependency_receipts
        return tuple(record.receipt for record in self.dependency_records)


def _durable_completed_cells(
    source: BudgetRawJsonBinding,
) -> DurableJsonArtifactBinding:
    durable = DurableJsonArtifactBinding.from_path(source.path)
    if (
        durable.path != source.path
        or durable.sidecar_path != source.sidecar_path
        or durable.semantic_sha256 != source.semantic_sha256
        or durable.file_sha256 != source.file_sha256
        or durable.sidecar_file_sha256 != source.sidecar_file_sha256
        or durable.size != source.size
        or durable.sidecar_size != source.sidecar_size
    ):
        raise ValueError("dependency completed cells differ from their raw binding")
    return durable


def _dependency_inventory_authority(
    *, inventory_path: str, source_receipt_path: str
) -> tuple[DependencyGpuInventoryAuthorityBinding, object]:
    inventory_source = bind_budget_raw_json(
        inventory_path, role="dependency_gpu_inventory"
    )
    receipt_source = bind_budget_raw_json(
        source_receipt_path,
        role="dependency_gpu_inventory_source_receipt",
    )
    inventory = _gpu_inventory_from_value(load_budget_raw_json(inventory_source))
    _validate_gpu_inventory_source(
        inventory,
        load_budget_raw_json(receipt_source),
    )
    binding = DependencyGpuInventoryAuthorityBinding(
        schema_version=1,
        inventory=inventory_source,
        source_receipt=receipt_source,
        inventory_sha256=inventory.sha256,
        source_receipt_sha256=inventory.source_receipt_sha256,
    )
    return binding, inventory


def _bind_dependency_completion(
    spec: dict[str, Any],
    *,
    expected_receipt_source: BudgetRawJsonBinding,
    expected_registry: ExperimentRegistry,
    earlier_records: tuple[_DependencyCompletionRecord, ...] | None,
    manifest_stack: tuple[Path, ...],
) -> _DependencyCompletionRecord:
    receipt_source = bind_budget_raw_json(
        _strict_text("dependency completion receipt path", spec["receipt_artifact"]),
        role="activation_dependency_receipt",
    )
    if receipt_source != expected_receipt_source:
        raise ValueError("dependency completion swapped its raw receipt")
    receipt = _receipt_from_value(load_budget_raw_json(receipt_source))
    replay = _bind_stage_activation_authority(
        _strict_text(
            "dependency completion activation manifest",
            spec["activation_manifest"],
        ),
        manifest_stack=manifest_stack,
    )
    activation_binding = replay.binding
    registry = replay.registry
    nested_records = replay.dependency_records
    if replay.dependency_receipts and not nested_records:
        raise BudgetMaterializationBlockedError(
            DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON
        )
    if (
        registry != expected_registry
        or replay.experiment != receipt.experiment
        or replay.runtime_sha256 != receipt.runtime_sha256
        or replay.split_sha256 != receipt.split_sha256
        or (
            earlier_records is not None
            and tuple(record.binding for record in nested_records)
            != tuple(record.binding for record in earlier_records)
        )
        or replay.dependency_receipts
        != tuple(record.receipt for record in nested_records)
    ):
        raise ValueError(
            "dependency completion activation or recursive lineage differs"
        )
    if receipt.experiment in {"E3b", "E5"} and (
        type(activation_binding) is not ConfirmationStageAggregateAuthorityBinding
        or activation_binding.stage_receipt != receipt_source
    ):
        # A per-family activation remains incremental authority even when one
        # registry happens to contain a single family.  Only the independent,
        # exact-registry aggregate can mint a stage receipt.
        raise BudgetMaterializationBlockedError(
            DEPENDENCY_COMPLETION_FAMILY_STAGE_AGGREGATION_MISSING_REASON
        )
    if (
        replay.family_activations
        and type(activation_binding) is not ConfirmationStageAggregateAuthorityBinding
    ):
        raise BudgetMaterializationBlockedError(
            DEPENDENCY_COMPLETION_FAMILY_STAGE_AGGREGATION_MISSING_REASON
        )
    definition = registry.definition(receipt.experiment)
    if receipt.experiment not in INDUSTRIAL_EXPERIMENT_ORDER:
        raise ValueError("dependency completion stage is outside the registry order")
    stage_index = INDUSTRIAL_EXPERIMENT_ORDER.index(receipt.experiment)
    if stage_index != len(nested_records):
        raise BudgetMaterializationBlockedError(
            DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON
        )
    if definition.dependencies and not nested_records:
        raise BudgetMaterializationBlockedError(
            DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON
        )

    completed_source = bind_budget_raw_json(
        _strict_text(
            "dependency completion completed-cells path",
            spec["completed_cells_artifact"],
        ),
        role="dependency_completed_cells",
    )
    if (
        type(activation_binding) is ConfirmationStageAggregateAuthorityBinding
        and activation_binding.stage_completed_cells != completed_source
    ):
        raise ValueError("confirmation aggregate stage completion artifact was swapped")
    inventory_binding, inventory = _dependency_inventory_authority(
        inventory_path=_strict_text(
            "dependency completion inventory path", spec["inventory_artifact"]
        ),
        source_receipt_path=_strict_text(
            "dependency completion inventory source path",
            spec["inventory_source_receipt"],
        ),
    )
    if nested_records and any(
        record.authority.inventory != inventory for record in nested_records
    ):
        raise ValueError("dependency completion prefix swaps the full GPU inventory")
    locked_output_specs = tuple(
        _strict_object(
            "dependency completion locked output",
            item,
            _DEPENDENCY_LOCKED_OUTPUT_FIELDS,
        )
        for item in _strict_list(
            "dependency completion locked outputs", spec["locked_outputs"]
        )
    )
    locked_output_names = tuple(
        _strict_text("dependency completion locked-output name", row["name"])
        for row in locked_output_specs
    )
    if len(locked_output_names) != len(definition.locked_outputs) or set(
        locked_output_names
    ) != set(definition.locked_outputs):
        raise ValueError(
            "dependency completion locked outputs are incomplete, duplicated, or extra"
        )
    if receipt.experiment == "E3a":
        from lightcone_spec.experiments.selection_authority import (
            SelectionReductionAuthorityUnavailableError,
            require_e3a_locked_output_reduction_authority,
        )

        try:
            require_e3a_locked_output_reduction_authority()
        except SelectionReductionAuthorityUnavailableError as error:
            raise BudgetMaterializationBlockedError(error.reason_code) from error
    elif receipt.experiment == "E1":
        from lightcone_spec.experiments.selection_authority import (
            SelectionReductionAuthorityUnavailableError,
            require_e1_common_load_reduction_authority,
        )

        try:
            require_e1_common_load_reduction_authority()
        except SelectionReductionAuthorityUnavailableError as error:
            raise BudgetMaterializationBlockedError(error.reason_code) from error
    output_bindings: list[DependencyLockedOutputAuthorityBinding] = []
    output_sha256s: dict[str, str] = {}
    for output in locked_output_specs:
        name = _strict_text("dependency completion locked-output name", output["name"])
        path = Path(
            _strict_text("dependency completion locked-output path", output["artifact"])
        )
        if path.suffix.lower() != ".json":
            raise BudgetMaterializationBlockedError(
                DEPENDENCY_COMPLETION_LOCKED_OUTPUT_UNSUPPORTED_REASON
            )
        source = bind_budget_raw_json(path, role="dependency_locked_output")
        output_bindings.append(
            DependencyLockedOutputAuthorityBinding(name=name, artifact=source)
        )
        output_sha256s[name] = source.semantic_sha256

    expected_receipt = registry.make_receipt(
        receipt.experiment,
        output_sha256s,
        runtime_sha256=replay.runtime_sha256,
        split_sha256=replay.split_sha256,
        completed_cells_sha256=completed_source.semantic_sha256,
        dependencies=tuple(record.receipt for record in nested_records),
    )
    if receipt != expected_receipt:
        raise ValueError(
            "dependency receipt differs from completed cells, lineage, or outputs"
        )
    direct_dependency = None
    dependency_authority = None
    if definition.dependencies:
        direct_name = definition.dependencies[-1]
        direct = next(
            (
                record
                for record in nested_records
                if record.receipt.experiment == direct_name
            ),
            None,
        )
        if direct is None:
            raise BudgetMaterializationBlockedError(
                DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON
            )
        direct_dependency = direct.receipt
        dependency_authority = direct.authority
    authority = CompletedCellAuthority(
        completed_cells=_durable_completed_cells(completed_source),
        registry=registry,
        inventory=inventory,
        direct_dependency_receipt=direct_dependency,
        dependency_authority=dependency_authority,
        activation_artifact=replay.activation_artifact,
        family_activations=replay.family_activations,
        family_power_reductions=replay.family_power_reductions,
        prior_family_authorities=replay.prior_family_authorities,
        raw_activation_authority=activation_binding,
    )
    binding = RegistryStageDependencyCompletionAuthorityBinding(
        schema_version=1,
        receipt=receipt_source,
        completed_cells=completed_source,
        activation=activation_binding,
        inventory_authority=inventory_binding,
        locked_outputs=tuple(output_bindings),
        receipt_sha256=receipt.sha256,
        completed_authority_sha256=authority.sha256,
    )
    return _DependencyCompletionRecord(
        binding=binding,
        receipt=receipt,
        authority=authority,
        lineage_records=nested_records,
    )


def _bind_registry_stage_activation(
    manifest_path: str | Path,
    *,
    manifest_stack: tuple[Path, ...] = (),
) -> tuple[
    RegistryStageActivationAuthorityBinding,
    ExperimentRegistry,
    RegistryStageActivationArtifact,
    tuple[_DependencyCompletionRecord, ...],
]:
    resolved_manifest = _exact_existing_path(
        manifest_path, label="registry-stage activation manifest"
    )
    if resolved_manifest in manifest_stack:
        raise ValueError("registry-stage activation manifests contain a cycle")
    next_stack = (*manifest_stack, resolved_manifest)
    manifest_binding = bind_budget_raw_json(
        resolved_manifest,
        role="registry_stage_activation_manifest",
    )
    manifest = _activation_manifest_from_value(load_budget_raw_json(manifest_binding))
    registry_binding = bind_budget_raw_json(
        _strict_text("activation registry path", manifest["registry_artifact"]),
        role="generated_registry",
    )
    runtime_binding = bind_budget_raw_json(
        _strict_text("activation runtime path", manifest["runtime_artifact"]),
        role="activation_runtime",
    )
    split_binding = bind_budget_raw_json(
        _strict_text("activation split path", manifest["split_artifact"]),
        role="activation_split",
    )
    receipt_bindings = tuple(
        bind_budget_raw_json(path, role="activation_dependency_receipt")
        for path in _strict_list(
            "activation dependency paths", manifest["dependency_receipts"]
        )
    )
    registry = _generated_registry_from_value(load_budget_raw_json(registry_binding))
    receipts = tuple(
        _receipt_from_value(load_budget_raw_json(binding))
        for binding in receipt_bindings
    )
    completion_records: list[_DependencyCompletionRecord] = []
    if manifest["schema_version"] == 2:
        for source, spec in zip(
            receipt_bindings,
            _strict_list(
                "activation dependency completion authorities",
                manifest["dependency_completion_authorities"],
            ),
            strict=True,
        ):
            completion_records.append(
                _bind_dependency_completion(
                    _dependency_completion_spec_from_value(spec),
                    expected_receipt_source=source,
                    expected_registry=registry,
                    earlier_records=tuple(completion_records),
                    manifest_stack=next_stack,
                )
            )
    activation = materialize_registry_stage_activation(
        registry,
        experiment=_strict_text("activation experiment", manifest["experiment"]),
        dependency_receipts=receipts,
        runtime_sha256=runtime_binding.canonical_sha256,
        split_sha256=split_binding.canonical_sha256,
    )
    verify_registry_stage_activation(registry, activation)
    binding = RegistryStageActivationAuthorityBinding(
        schema_version=1,
        kind="registry_stage_activation_manifest",
        manifest=manifest_binding,
        generated_registry=registry_binding,
        runtime=runtime_binding,
        split=split_binding,
        dependency_receipts=receipt_bindings,
        dependency_completion_authorities=tuple(
            record.binding for record in completion_records
        ),
        activation_sha256=activation.sha256,
    )
    return binding, registry, activation, tuple(completion_records)


def _specialized_manifest(
    manifest_path: str | Path,
    *,
    role: str,
    expected_kind: str,
    expected_fields: frozenset[str],
    manifest_stack: tuple[Path, ...],
) -> tuple[Path, BudgetRawJsonBinding, dict[str, Any], tuple[Path, ...]]:
    resolved = _exact_existing_path(manifest_path, label=f"{expected_kind} manifest")
    if resolved in manifest_stack:
        raise ValueError("raw activation authority manifests contain a cycle")
    source = bind_budget_raw_json(resolved, role=role)
    value = load_budget_raw_json(source)
    if (
        expected_kind == "industrial_e2_activation_authority_manifest"
        and type(value) is dict
        and type(value.get("stage_index")) is int
        and value["stage_index"] > 0
        and "prior_stage_completion_authorities" not in value
    ):
        raise BudgetMaterializationBlockedError(
            E2_STAGE_COMPLETION_AUTHORITY_MISSING_REASON
        )
    row = _strict_object(
        f"{expected_kind} manifest",
        value,
        expected_fields,
    )
    if row["schema_version"] != 1 or row["kind"] != expected_kind:
        raise ValueError(f"{expected_kind} manifest identity is invalid")
    return resolved, source, row, (*manifest_stack, resolved)


def _bind_generated_registry(path: str, *, label: str):
    source = bind_budget_raw_json(path, role="generated_registry")
    return source, _generated_registry_from_value(load_budget_raw_json(source))


def _bind_runtime_split(
    row: dict[str, Any], *, label: str
) -> tuple[BudgetRawJsonBinding, BudgetRawJsonBinding]:
    runtime = bind_budget_raw_json(
        _strict_path_field(row, "runtime_artifact", label=label),
        role="activation_runtime",
    )
    split = bind_budget_raw_json(
        _strict_path_field(row, "split_artifact", label=label),
        role="activation_split",
    )
    return runtime, split


def _bind_activation_inventory(
    row: dict[str, Any], *, label: str
) -> tuple[DependencyGpuInventoryAuthorityBinding, object]:
    return _dependency_inventory_authority(
        inventory_path=_strict_path_field(row, "gpu_inventory", label=label),
        source_receipt_path=_strict_path_field(
            row, "inventory_source_receipt", label=label
        ),
    )


def _bind_activation_hardware(
    row: dict[str, Any], *, label: str
) -> tuple[BudgetRawJsonBinding, object]:
    source = bind_budget_raw_json(
        _strict_path_field(row, "hardware_envelope", label=label),
        role="activation_hardware_envelope",
    )
    return source, _load_hardware_envelope(source)


def _bind_e1_activation_authority(
    manifest_path: str | Path,
    *,
    manifest_stack: tuple[Path, ...] = (),
) -> BudgetActivationAuthorityResult:
    _, manifest_source, row, next_stack = _specialized_manifest(
        manifest_path,
        role="e1_activation_authority_manifest",
        expected_kind="industrial_e1_activation_authority_manifest",
        expected_fields=_E1_AUTHORITY_FIELDS,
        manifest_stack=manifest_stack,
    )
    registry_source, registry = _bind_generated_registry(
        _strict_path_field(row, "registry_artifact", label="E1 activation"),
        label="E1 activation",
    )
    runtime, split = _bind_runtime_split(row, label="E1 activation")
    receipt_source = bind_budget_raw_json(
        _strict_path_field(row, "dependency_receipt", label="E1 activation"),
        role="activation_dependency_receipt",
    )
    receipt = _receipt_from_value(load_budget_raw_json(receipt_source))
    if receipt.experiment != "E3a":
        raise ValueError("E1 activation requires the exact E3a dependency receipt")
    completion = _bind_dependency_completion(
        _dependency_completion_spec_from_value(row["dependency_completion_authority"]),
        expected_receipt_source=receipt_source,
        expected_registry=registry,
        earlier_records=None,
        manifest_stack=next_stack,
    )
    inventory_binding, inventory = _bind_activation_inventory(
        row, label="E1 activation"
    )
    hardware_source, hardware = _bind_activation_hardware(row, label="E1 activation")
    if completion.authority.inventory != inventory:
        raise ValueError("E1 activation swaps dependency and selection inventories")
    selection_source = bind_budget_raw_json(
        _strict_path_field(row, "selection_manifest", label="E1 activation"),
        role="e3a_selection_raw_manifest",
    )
    from lightcone_spec.experiments.industrial_analysis import (
        raw_e3a_selection_manifest_from_dict,
        validate_raw_evidence_manifest_sidecars,
    )
    from lightcone_spec.experiments.selection_authority import (
        SelectionReductionAuthorityUnavailableError,
        reduce_e3a_selection_from_raw,
    )

    raw_selection = raw_e3a_selection_manifest_from_dict(
        load_budget_raw_json(selection_source)
    )
    validate_raw_evidence_manifest_sidecars(raw_selection)
    try:
        selection = reduce_e3a_selection_from_raw(
            registry=registry,
            manifest=raw_selection,
            hardware_envelope=hardware,
            inventory=inventory,
            runtime_sha256=runtime.canonical_sha256,
            split_sha256=split.canonical_sha256,
            confirmation_data_visible=False,
        )
    except SelectionReductionAuthorityUnavailableError as error:
        raise BudgetMaterializationBlockedError(
            "trusted_hardware_attester_unavailable"
        ) from error
    activation = reduce_e1_activation(
        registry,
        e3a_receipt=receipt,
        selection=selection,
    )
    binding = E1ActivationAuthorityBinding(
        schema_version=1,
        kind="e1_activation_manifest",
        manifest=manifest_source,
        generated_registry=registry_source,
        runtime=runtime,
        split=split,
        dependency_receipt=receipt_source,
        dependency_completion_authority=completion.binding,
        selection_manifest=selection_source,
        inventory_authority=inventory_binding,
        hardware_envelope=hardware_source,
        activation_sha256=activation.sha256,
        selection_sha256=selection.sha256,
    )
    return BudgetActivationAuthorityResult(
        binding=binding,
        registry=registry,
        activation_artifact=activation,
        family_activations=(),
        family_power_reductions=(),
        dependency_records=(*completion.lineage_records, completion),
        prior_family_authorities=(),
        e3a_selection=selection,
    )


def _bind_e2_activation_authority(
    manifest_path: str | Path,
    *,
    manifest_stack: tuple[Path, ...] = (),
) -> BudgetActivationAuthorityResult:
    _, manifest_source, row, next_stack = _specialized_manifest(
        manifest_path,
        role="e2_activation_authority_manifest",
        expected_kind="industrial_e2_activation_authority_manifest",
        expected_fields=_E2_AUTHORITY_FIELDS,
        manifest_stack=manifest_stack,
    )
    stage_index = _strict_int("E2 activation stage index", row["stage_index"])
    if stage_index not in range(len(E2_HALVING_STAGES)):
        raise ValueError("E2 activation stage index is invalid")
    raw_completion_specs = _strict_list(
        "E2 prior stage completion authorities",
        row["prior_stage_completion_authorities"],
    )
    if len(raw_completion_specs) != stage_index:
        raise BudgetMaterializationBlockedError(
            E2_STAGE_COMPLETION_AUTHORITY_MISSING_REASON
        )
    registry_source, registry = _bind_generated_registry(
        _strict_path_field(row, "registry_artifact", label="E2 activation"),
        label="E2 activation",
    )
    runtime, split = _bind_runtime_split(row, label="E2 activation")
    receipt_source = bind_budget_raw_json(
        _strict_path_field(row, "dependency_receipt", label="E2 activation"),
        role="activation_dependency_receipt",
    )
    receipt = _receipt_from_value(load_budget_raw_json(receipt_source))
    if receipt.experiment != "E1":
        raise ValueError("E2 activation requires the exact E1 dependency receipt")
    completion = _bind_dependency_completion(
        _dependency_completion_spec_from_value(row["dependency_completion_authority"]),
        expected_receipt_source=receipt_source,
        expected_registry=registry,
        earlier_records=None,
        manifest_stack=next_stack,
    )
    e1_replay = _bind_stage_activation_authority(
        completion.binding.activation.manifest.path,
        manifest_stack=next_stack,
    )
    if (
        type(e1_replay.binding) is not E1ActivationAuthorityBinding
        or type(e1_replay.activation_artifact) is not ReducerActivationArtifact
        or e1_replay.e3a_selection is None
        or e1_replay.registry != registry
    ):
        raise ValueError("E2 activation lacks raw E1 selection lineage")
    inventory_binding, inventory = _bind_activation_inventory(
        row, label="E2 activation"
    )
    hardware_source, hardware = _bind_activation_hardware(row, label="E2 activation")
    if completion.authority.inventory != inventory:
        raise ValueError("E2 activation swaps dependency and tuning inventories")
    pareto_source = bind_budget_raw_json(
        _strict_path_field(row, "pareto_manifest", label="E2 activation"),
        role="e1_pareto_raw_manifest",
    )
    from lightcone_spec.experiments.industrial_analysis import (
        raw_e1_pareto_manifest_from_dict,
        raw_e2_stage_manifest_from_dict,
        reduce_e2_stage_from_raw,
        validate_raw_evidence_manifest_sidecars,
    )
    from lightcone_spec.experiments.selection_authority import (
        SelectionReductionAuthorityUnavailableError,
        reduce_e1_pareto_from_raw,
        require_e1_common_load_reduction_authority,
    )

    raw_pareto = raw_e1_pareto_manifest_from_dict(load_budget_raw_json(pareto_source))
    validate_raw_evidence_manifest_sidecars(raw_pareto)
    e3a_record = e1_replay.dependency_records[-1]
    try:
        pareto = reduce_e1_pareto_from_raw(
            registry=registry,
            activation=e1_replay.activation_artifact,
            manifest=raw_pareto,
            hardware_envelope=hardware,
            inventory=inventory,
            e3a_receipt=e3a_record.receipt,
            e3a_selection=e1_replay.e3a_selection,
            source_activation_authority_sha256=e1_replay.binding.sha256,
            confirmation_data_visible=False,
        )
    except SelectionReductionAuthorityUnavailableError as error:
        raise BudgetMaterializationBlockedError(
            "trusted_hardware_attester_unavailable"
        ) from error
    if (
        runtime.canonical_sha256 != pareto.runtime_sha256
        or split.canonical_sha256 != pareto.split_sha256
    ):
        raise ValueError("E2 runtime/split differs from the raw E1 Pareto lineage")
    try:
        require_e1_common_load_reduction_authority()
    except SelectionReductionAuthorityUnavailableError as error:
        raise BudgetMaterializationBlockedError(error.reason_code) from error
    e1_outputs = {output.name: output.content_sha256 for output in receipt.outputs}
    if e1_outputs.get("dflash_pareto_set") != pareto.sha256:
        raise ValueError("E1 receipt does not bind the raw Pareto replay")
    prior_paths = _strict_path_list(
        row["prior_stage_manifests"], label="E2 prior stage manifests"
    )
    if len(prior_paths) != stage_index:
        raise ValueError("E2 activation lacks an exact prior-stage manifest prefix")
    prior_sources: list[BudgetRawJsonBinding] = []
    prior_reductions: list[E2StageReductionArtifact] = []
    prior: E2StageReductionArtifact | None = None
    for expected_stage, path in enumerate(prior_paths):
        source = bind_budget_raw_json(path, role="e2_stage_raw_manifest")
        raw_stage = raw_e2_stage_manifest_from_dict(load_budget_raw_json(source))
        validate_raw_evidence_manifest_sidecars(raw_stage)
        if raw_stage.stage_index != expected_stage:
            raise ValueError("E2 prior raw stages are not the exact ordered prefix")
        prior = reduce_e2_stage_from_raw(
            registry=registry,
            e1_receipt=receipt,
            pareto=pareto,
            stage_index=expected_stage,
            cells=raw_stage.cells,
            hardware_envelope=hardware,
            inventory=inventory,
            prior_stage_reduction=prior,
            confirmation_data_visible=False,
        )
        prior_sources.append(source)
        prior_reductions.append(prior)
    completion_specs = tuple(
        _strict_object(
            "E2 prior stage completion authority",
            value,
            _E2_STAGE_COMPLETION_FIELDS,
        )
        for value in raw_completion_specs
    )
    prior_completion_bindings: list[E2StageCompletionAuthorityBinding] = []
    prior_completion_authorities: list[CompletedCellAuthority] = []
    for expected_stage, spec in enumerate(completion_specs):
        completed_path = _strict_text(
            "E2 prior stage completed cells", spec["completed_cells_artifact"]
        )
        activation_path = _strict_text(
            "E2 prior stage activation manifest", spec["activation_manifest"]
        )
        stage_replay = _bind_e2_activation_authority(
            activation_path,
            manifest_stack=next_stack,
        )
        if (
            type(stage_replay.binding) is not E2ActivationAuthorityBinding
            or stage_replay.binding.stage_index != expected_stage
            or stage_replay.registry != registry
            or stage_replay.e1_pareto != pareto
            or stage_replay.prior_e2_reductions
            != tuple(prior_reductions[:expected_stage])
        ):
            raise ValueError(
                "E2 prior completion activation differs from the raw round prefix"
            )
        completed = bind_budget_raw_json(
            completed_path,
            role="e2_stage_completed_cells",
        )
        direct = stage_replay.dependency_records[-1]
        authority = CompletedCellAuthority(
            completed_cells=_durable_completed_cells(completed),
            registry=registry,
            inventory=inventory,
            direct_dependency_receipt=direct.receipt,
            dependency_authority=direct.authority,
            activation_artifact=stage_replay.activation_artifact,
            family_activations=(),
            family_power_reductions=(),
            prior_family_authorities=(),
            raw_activation_authority=stage_replay.binding,
        )
        completion_binding = E2StageCompletionAuthorityBinding(
            schema_version=1,
            completed_cells=completed,
            stage_activation=stage_replay.binding,
            inventory_authority=inventory_binding,
            completed_authority_sha256=authority.sha256,
        )
        prior_completion_bindings.append(completion_binding)
        prior_completion_authorities.append(authority)
    activation = reduce_e2_activation(
        registry,
        e1_receipt=receipt,
        pareto=pareto,
        stage_index=stage_index,
        prior_reduction=prior,
    )
    binding = E2ActivationAuthorityBinding(
        schema_version=1,
        kind="e2_activation_manifest",
        manifest=manifest_source,
        generated_registry=registry_source,
        runtime=runtime,
        split=split,
        dependency_receipt=receipt_source,
        dependency_completion_authority=completion.binding,
        pareto_manifest=pareto_source,
        prior_stage_manifests=tuple(prior_sources),
        prior_stage_completion_authorities=tuple(prior_completion_bindings),
        inventory_authority=inventory_binding,
        hardware_envelope=hardware_source,
        stage_index=stage_index,
        activation_sha256=activation.sha256,
        pareto_sha256=pareto.sha256,
        prior_stage_reduction_sha256=None if prior is None else prior.sha256,
    )
    return BudgetActivationAuthorityResult(
        binding=binding,
        registry=registry,
        activation_artifact=activation,
        family_activations=(),
        family_power_reductions=(),
        dependency_records=(*completion.lineage_records, completion),
        prior_family_authorities=(),
        prior_e2_stage_authorities=tuple(prior_completion_authorities),
        e1_pareto=pareto,
        prior_e2_reductions=tuple(prior_reductions),
    )


def _bind_confirmation_pilot_activation_authority(
    manifest_path: str | Path,
    *,
    manifest_stack: tuple[Path, ...] = (),
) -> BudgetActivationAuthorityResult:
    _, manifest_source, row, next_stack = _specialized_manifest(
        manifest_path,
        role="confirmation_pilot_activation_authority_manifest",
        expected_kind="industrial_confirmation_pilot_activation_authority_manifest",
        expected_fields=_CONFIRMATION_PILOT_AUTHORITY_FIELDS,
        manifest_stack=manifest_stack,
    )
    registry_source, registry = _bind_generated_registry(
        _strict_path_field(row, "registry_artifact", label="confirmation pilot"),
        label="confirmation pilot",
    )
    runtime, split = _bind_runtime_split(row, label="confirmation pilot")
    trace = bind_budget_raw_json(
        _strict_path_field(row, "trace_artifact", label="confirmation pilot"),
        role="activation_trace",
    )
    sampling = bind_budget_raw_json(
        _strict_path_field(row, "sampling_artifact", label="confirmation pilot"),
        role="activation_sampling",
    )
    receipt_paths = _strict_path_list(
        row["dependency_receipts"], label="confirmation pilot dependencies"
    )
    receipt_sources = tuple(
        bind_budget_raw_json(path, role="activation_dependency_receipt")
        for path in receipt_paths
    )
    receipts = tuple(
        _receipt_from_value(load_budget_raw_json(source)) for source in receipt_sources
    )
    specs = tuple(
        _dependency_completion_spec_from_value(value)
        for value in _strict_list(
            "confirmation pilot dependency completion authorities",
            row["dependency_completion_authorities"],
        )
    )
    if len(specs) != len(receipt_sources):
        raise ValueError("confirmation pilot completion coverage is incomplete")
    records: list[_DependencyCompletionRecord] = []
    for source, spec in zip(receipt_sources, specs, strict=True):
        records.append(
            _bind_dependency_completion(
                spec,
                expected_receipt_source=source,
                expected_registry=registry,
                earlier_records=tuple(records),
                manifest_stack=next_stack,
            )
        )
    if tuple(record.receipt for record in records) != receipts:
        raise ValueError("confirmation pilot dependency receipts were swapped")
    inventory_binding, inventory = _bind_activation_inventory(
        row, label="confirmation pilot"
    )
    hardware_source, hardware = _bind_activation_hardware(
        row, label="confirmation pilot"
    )
    if any(record.authority.inventory != inventory for record in records):
        raise ValueError("confirmation pilot swaps dependency GPU inventory")
    family_seed_cell_id = _strict_text(
        "confirmation pilot family seed", row["family_seed_cell_id"]
    )
    family = derive_confirmation_family(
        registry,
        cell_id=family_seed_cell_id,
        runtime_sha256=runtime.canonical_sha256,
        split_sha256=split.canonical_sha256,
        trace_sha256=trace.canonical_sha256,
        sampling_sha256=sampling.canonical_sha256,
        hardware_envelope_sha256=content_sha256(hardware),
    )
    stage_index = INDUSTRIAL_EXPERIMENT_ORDER.index(family.experiment)
    expected_dependencies = INDUSTRIAL_EXPERIMENT_ORDER[:stage_index]
    if tuple(receipt.experiment for receipt in receipts) != expected_dependencies:
        raise ValueError("confirmation pilot lacks the exact dependency receipt prefix")
    registry.validate_receipts(receipts)
    pilot = materialize_confirmation_pilots(registry, family)
    binding = ConfirmationPilotActivationAuthorityBinding(
        schema_version=1,
        kind="confirmation_pilot_activation_manifest",
        manifest=manifest_source,
        generated_registry=registry_source,
        runtime=runtime,
        split=split,
        trace=trace,
        sampling=sampling,
        dependency_receipts=receipt_sources,
        dependency_completion_authorities=tuple(record.binding for record in records),
        inventory_authority=inventory_binding,
        hardware_envelope=hardware_source,
        family_sha256=family.sha256,
        activation_sha256=pilot.sha256,
    )
    return BudgetActivationAuthorityResult(
        binding=binding,
        registry=registry,
        activation_artifact=None,
        family_activations=(pilot,),
        family_power_reductions=(),
        dependency_records=tuple(records),
        prior_family_authorities=(),
    )


def _bind_confirmation_auxiliary_activation_authority(
    manifest_path: str | Path,
    *,
    manifest_stack: tuple[Path, ...] = (),
) -> BudgetActivationAuthorityResult:
    _, manifest_source, row, next_stack = _specialized_manifest(
        manifest_path,
        role="confirmation_auxiliary_activation_authority_manifest",
        expected_kind="industrial_confirmation_auxiliary_activation_authority_manifest",
        expected_fields=_CONFIRMATION_AUXILIARY_AUTHORITY_FIELDS,
        manifest_stack=manifest_stack,
    )
    experiment = _strict_text("confirmation auxiliary experiment", row["experiment"])
    if experiment not in {"E3b", "E5"}:
        raise ValueError("confirmation auxiliary supports only E3b/E5")
    registry_source, registry = _bind_generated_registry(
        _strict_path_field(row, "registry_artifact", label="confirmation auxiliary"),
        label="confirmation auxiliary",
    )
    runtime, split = _bind_runtime_split(row, label="confirmation auxiliary")
    trace = bind_budget_raw_json(
        _strict_path_field(row, "trace_artifact", label="confirmation auxiliary"),
        role="activation_trace",
    )
    sampling = bind_budget_raw_json(
        _strict_path_field(row, "sampling_artifact", label="confirmation auxiliary"),
        role="activation_sampling",
    )
    receipt_sources = tuple(
        bind_budget_raw_json(path, role="activation_dependency_receipt")
        for path in _strict_path_list(
            row["dependency_receipts"],
            label="confirmation auxiliary dependencies",
        )
    )
    receipts = tuple(
        _receipt_from_value(load_budget_raw_json(source)) for source in receipt_sources
    )
    specs = tuple(
        _dependency_completion_spec_from_value(value)
        for value in _strict_list(
            "confirmation auxiliary dependency completion authorities",
            row["dependency_completion_authorities"],
        )
    )
    if len(specs) != len(receipt_sources):
        raise ValueError("confirmation auxiliary completion coverage is incomplete")
    records: list[_DependencyCompletionRecord] = []
    for source, spec in zip(receipt_sources, specs, strict=True):
        records.append(
            _bind_dependency_completion(
                spec,
                expected_receipt_source=source,
                expected_registry=registry,
                earlier_records=tuple(records),
                manifest_stack=next_stack,
            )
        )
    if tuple(record.receipt for record in records) != receipts:
        raise ValueError("confirmation auxiliary dependency receipts were swapped")
    inventory_binding, inventory = _bind_activation_inventory(
        row, label="confirmation auxiliary"
    )
    hardware_source, hardware = _bind_activation_hardware(
        row, label="confirmation auxiliary"
    )
    if any(record.authority.inventory != inventory for record in records):
        raise ValueError("confirmation auxiliary swaps dependency GPU inventory")
    activation = materialize_confirmation_auxiliary_activation(
        registry,
        experiment=experiment,
        dependency_receipts=receipts,
        runtime_sha256=runtime.canonical_sha256,
        split_sha256=split.canonical_sha256,
        trace_sha256=trace.canonical_sha256,
        sampling_sha256=sampling.canonical_sha256,
        hardware_envelope_sha256=content_sha256(hardware),
    )
    binding = ConfirmationAuxiliaryActivationAuthorityBinding(
        schema_version=1,
        kind="confirmation_auxiliary_activation_manifest",
        manifest=manifest_source,
        generated_registry=registry_source,
        runtime=runtime,
        split=split,
        trace=trace,
        sampling=sampling,
        dependency_receipts=receipt_sources,
        dependency_completion_authorities=tuple(record.binding for record in records),
        inventory_authority=inventory_binding,
        hardware_envelope=hardware_source,
        experiment=experiment,
        activation_sha256=activation.sha256,
    )
    return BudgetActivationAuthorityResult(
        binding=binding,
        registry=registry,
        activation_artifact=activation,
        family_activations=(),
        family_power_reductions=(),
        dependency_records=tuple(records),
        prior_family_authorities=(),
    )


def _bind_family_pilot_completion(
    *,
    completed_cells_path: str,
    pilot_replay: BudgetActivationAuthorityResult,
    inventory_binding: DependencyGpuInventoryAuthorityBinding,
    inventory: object,
) -> tuple[FamilyPilotCompletionAuthorityBinding, CompletedCellAuthority]:
    if (
        type(pilot_replay.binding) is not ConfirmationPilotActivationAuthorityBinding
        or len(pilot_replay.family_activations) != 1
    ):
        raise TypeError("family pilot completion requires raw pilot activation")
    completed = bind_budget_raw_json(
        completed_cells_path,
        role="family_pilot_completed_cells",
    )
    direct = (
        pilot_replay.dependency_records[-1] if pilot_replay.dependency_records else None
    )
    authority = CompletedCellAuthority(
        completed_cells=_durable_completed_cells(completed),
        registry=pilot_replay.registry,
        inventory=inventory,
        direct_dependency_receipt=None if direct is None else direct.receipt,
        dependency_authority=None if direct is None else direct.authority,
        activation_artifact=None,
        family_activations=pilot_replay.family_activations,
        family_power_reductions=(),
        prior_family_authorities=(),
        raw_activation_authority=pilot_replay.binding,
    )
    binding = FamilyPilotCompletionAuthorityBinding(
        schema_version=1,
        completed_cells=completed,
        pilot_activation=pilot_replay.binding,
        inventory_authority=inventory_binding,
        completed_authority_sha256=authority.sha256,
    )
    return binding, authority


def _bind_confirmation_final_activation_authority(
    manifest_path: str | Path,
    *,
    manifest_stack: tuple[Path, ...] = (),
) -> BudgetActivationAuthorityResult:
    _, manifest_source, row, next_stack = _specialized_manifest(
        manifest_path,
        role="confirmation_final_activation_authority_manifest",
        expected_kind="industrial_confirmation_final_activation_authority_manifest",
        expected_fields=_CONFIRMATION_FINAL_AUTHORITY_FIELDS,
        manifest_stack=manifest_stack,
    )
    pilot_replay = _bind_confirmation_pilot_activation_authority(
        _strict_path_field(
            row, "pilot_activation_manifest", label="confirmation final"
        ),
        manifest_stack=next_stack,
    )
    pilot_binding = pilot_replay.binding
    if type(pilot_binding) is not ConfirmationPilotActivationAuthorityBinding:
        raise TypeError("confirmation final lost its pilot activation authority")
    inventory = _gpu_inventory_from_value(
        load_budget_raw_json(pilot_binding.inventory_authority.inventory)
    )
    pilot_completion_binding, pilot_completion = _bind_family_pilot_completion(
        completed_cells_path=_strict_path_field(
            row, "pilot_completed_cells", label="confirmation final"
        ),
        pilot_replay=pilot_replay,
        inventory_binding=pilot_binding.inventory_authority,
        inventory=inventory,
    )
    power_source = bind_budget_raw_json(
        _strict_path_field(row, "power_manifest", label="confirmation final"),
        role="confirmation_family_power_raw_manifest",
    )
    from lightcone_spec.experiments.industrial_analysis import (
        raw_confirmation_family_power_manifest_from_dict,
        reduce_confirmation_family_power,
        validate_raw_evidence_manifest_sidecars,
    )

    raw_power = raw_confirmation_family_power_manifest_from_dict(
        load_budget_raw_json(power_source)
    )
    validate_raw_evidence_manifest_sidecars(raw_power)
    pilot = pilot_replay.family_activations[0]
    reduction = reduce_confirmation_family_power(
        registry=pilot_replay.registry,
        pilot_activation=pilot,
        blocks=raw_power.blocks,
        hardware_envelope=_load_hardware_envelope(pilot_binding.hardware_envelope),
        inventory=inventory,
        confirmation_data_visible=False,
    )
    final = materialize_confirmation_prefix(
        pilot_replay.registry,
        family=pilot.family,
        reduction=reduction,
        pilot_activation=pilot,
    )
    binding = ConfirmationFinalActivationAuthorityBinding(
        schema_version=1,
        kind="confirmation_final_activation_manifest",
        manifest=manifest_source,
        generated_registry=pilot_binding.generated_registry,
        pilot_activation_authority=pilot_binding,
        pilot_completion_authority=pilot_completion_binding,
        power_manifest=power_source,
        family_sha256=pilot.family.sha256,
        power_reduction_sha256=reduction.sha256,
        activation_sha256=final.sha256,
    )
    return BudgetActivationAuthorityResult(
        binding=binding,
        registry=pilot_replay.registry,
        activation_artifact=None,
        family_activations=(pilot, final),
        family_power_reductions=(reduction,),
        dependency_records=pilot_replay.dependency_records,
        prior_family_authorities=(pilot_completion,),
    )


def _bind_confirmation_stage_aggregate_authority(
    manifest_path: str | Path,
    *,
    manifest_stack: tuple[Path, ...] = (),
) -> BudgetActivationAuthorityResult:
    """Rebuild one exact E3b/E5 stage from all per-family raw authorities."""

    _, manifest_source, row, next_stack = _specialized_manifest(
        manifest_path,
        role="confirmation_stage_aggregate_authority_manifest",
        expected_kind="industrial_confirmation_stage_aggregate_authority_manifest",
        expected_fields=_CONFIRMATION_STAGE_AGGREGATE_FIELDS,
        manifest_stack=manifest_stack,
    )
    experiment = _strict_text(
        "confirmation stage aggregate experiment", row["experiment"]
    )
    if experiment not in {"E3b", "E5"}:
        raise ValueError("confirmation stage aggregate supports only E3b/E5")
    registry_source, registry = _bind_generated_registry(
        _strict_path_field(row, "registry_artifact", label="confirmation aggregate"),
        label="confirmation aggregate",
    )
    runtime, split = _bind_runtime_split(row, label="confirmation aggregate")
    inventory_binding, inventory = _bind_activation_inventory(
        row, label="confirmation aggregate"
    )
    stage_receipt_source = bind_budget_raw_json(
        _strict_path_field(row, "stage_receipt", label="confirmation aggregate"),
        role="activation_dependency_receipt",
    )
    stage_receipt = _receipt_from_value(load_budget_raw_json(stage_receipt_source))
    stage_completed_source = bind_budget_raw_json(
        _strict_path_field(
            row, "stage_completed_cells", label="confirmation aggregate"
        ),
        role="dependency_completed_cells",
    )
    if (
        stage_receipt.registry_sha256 != registry.sha256
        or stage_receipt.experiment != experiment
        or stage_receipt.runtime_sha256 != runtime.canonical_sha256
        or stage_receipt.split_sha256 != split.canonical_sha256
        or stage_receipt.completed_cells_sha256
        != stage_completed_source.semantic_sha256
    ):
        raise ValueError(
            "confirmation aggregate receipt differs from registry/runtime/split"
        )

    family_rows = tuple(
        _strict_object(
            "confirmation stage aggregate family",
            value,
            _CONFIRMATION_STAGE_FAMILY_FIELDS,
        )
        for value in _strict_list(
            "confirmation stage aggregate families", row["families"]
        )
    )
    declared_family_sha256s = tuple(
        _strict_text("confirmation aggregate family SHA-256", value["family_sha256"])
        for value in family_rows
    )
    if not declared_family_sha256s or declared_family_sha256s != tuple(
        sorted(set(declared_family_sha256s))
    ):
        raise ValueError(
            "confirmation aggregate families must be SHA-sorted and unique"
        )
    final_paths = tuple(
        _strict_path_field(
            value,
            "final_activation_manifest",
            label="confirmation aggregate family",
        )
        for value in family_rows
    )
    completed_paths = tuple(
        _strict_path_field(
            value,
            "completed_cells_artifact",
            label="confirmation aggregate family",
        )
        for value in family_rows
    )
    if (
        len(set(final_paths)) != len(final_paths)
        or len(set(completed_paths)) != len(completed_paths)
        or set(final_paths) & set(completed_paths)
    ):
        raise ValueError("confirmation aggregate family raw paths must be unique")

    family_bindings: list[ConfirmationStageFamilyAuthorityBinding] = []
    family_completions: list[CompletedCellAuthority] = []
    pilot_completions: list[CompletedCellAuthority] = []
    family_activations: list[FamilyActivationArtifact] = []
    family_power: list[ConfirmationFamilyPowerReductionArtifact] = []
    family_scopes: list[set[str]] = []
    common_dependency_bindings = None
    common_external_identity: tuple[str, str, str] | None = None
    common_records: tuple[_DependencyCompletionRecord, ...] = ()
    for declared_family_sha256, final_path, completed_path in zip(
        declared_family_sha256s, final_paths, completed_paths, strict=True
    ):
        final_replay = _bind_confirmation_final_activation_authority(
            final_path,
            manifest_stack=next_stack,
        )
        if (
            type(final_replay.binding)
            is not ConfirmationFinalActivationAuthorityBinding
            or len(final_replay.family_activations) != 2
            or len(final_replay.family_power_reductions) != 1
            or len(final_replay.prior_family_authorities) != 1
            or final_replay.registry != registry
        ):
            raise ValueError(
                "confirmation aggregate family lacks exact raw final lineage"
            )
        final_binding = final_replay.binding
        pilot_binding = final_binding.pilot_activation_authority
        pilot, final = final_replay.family_activations
        family = final.family
        if (
            family.sha256 != declared_family_sha256
            or family.experiment != experiment
            or pilot.family != family
            or pilot_binding.runtime != runtime
            or pilot_binding.split != split
            or pilot_binding.generated_registry != registry_source
            or pilot_binding.inventory_authority != inventory_binding
        ):
            raise ValueError(
                "confirmation aggregate family differs from stage identity"
            )
        external_identity = (
            family.trace_sha256,
            family.sampling_sha256,
            family.hardware_envelope_sha256,
        )
        if common_external_identity is None:
            common_external_identity = external_identity
        elif external_identity != common_external_identity:
            raise ValueError(
                "confirmation aggregate families use different stage authorities"
            )
        dependency_bindings = tuple(
            record.binding for record in final_replay.dependency_records
        )
        if common_dependency_bindings is None:
            common_dependency_bindings = dependency_bindings
            common_records = final_replay.dependency_records
        elif dependency_bindings != common_dependency_bindings:
            raise ValueError(
                "confirmation aggregate families use different dependency lineage"
            )
        completed_source = bind_budget_raw_json(
            completed_path,
            role="confirmation_family_completed_cells",
        )
        direct = (
            final_replay.dependency_records[-1]
            if final_replay.dependency_records
            else None
        )
        completion = CompletedCellAuthority(
            completed_cells=_durable_completed_cells(completed_source),
            registry=registry,
            inventory=inventory,
            direct_dependency_receipt=None if direct is None else direct.receipt,
            dependency_authority=None if direct is None else direct.authority,
            activation_artifact=None,
            family_activations=final_replay.family_activations,
            family_power_reductions=final_replay.family_power_reductions,
            prior_family_authorities=final_replay.prior_family_authorities,
            raw_activation_authority=final_binding,
        )
        completion_binding = ConfirmationFamilyCompletionAuthorityBinding(
            schema_version=1,
            completed_cells=completed_source,
            final_activation=final_binding,
            inventory_authority=inventory_binding,
            completed_authority_sha256=completion.sha256,
        )
        family_bindings.append(
            ConfirmationStageFamilyAuthorityBinding(
                schema_version=1,
                family_sha256=family.sha256,
                final_activation_authority=final_binding,
                completion_authority=completion_binding,
            )
        )
        family_completions.append(completion)
        pilot_completions.extend(final_replay.prior_family_authorities)
        family_activations.extend((pilot, final))
        family_power.extend(final_replay.family_power_reductions)
        family_scopes.append({value.cell_id for value in final.dispositions})

    if common_external_identity is None:  # guarded by non-empty family rows
        raise RuntimeError("confirmation aggregate lost its family identity")
    trace_sha256, sampling_sha256, hardware_sha256 = common_external_identity
    expected_families, expected_auxiliary_cells = derive_confirmation_stage_partition(
        registry,
        experiment=experiment,
        runtime_sha256=runtime.canonical_sha256,
        split_sha256=split.canonical_sha256,
        trace_sha256=trace_sha256,
        sampling_sha256=sampling_sha256,
        hardware_envelope_sha256=hardware_sha256,
    )
    expected_family_sha256s = tuple(family.sha256 for family in expected_families)
    if declared_family_sha256s != expected_family_sha256s:
        raise ValueError(
            "confirmation aggregate family set differs from the exact registry stage"
        )
    auxiliary_binding: ConfirmationAuxiliaryCompletionAuthorityBinding | None = None
    auxiliary_authority: CompletedCellAuthority | None = None
    auxiliary_activation: ReducerActivationArtifact | None = None
    raw_auxiliary = row["auxiliary"]
    if expected_auxiliary_cells:
        auxiliary_row = _strict_object(
            "confirmation stage aggregate auxiliary",
            raw_auxiliary,
            _CONFIRMATION_STAGE_AUXILIARY_FIELDS,
        )
        auxiliary_replay = _bind_confirmation_auxiliary_activation_authority(
            _strict_path_field(
                auxiliary_row,
                "activation_manifest",
                label="confirmation aggregate auxiliary",
            ),
            manifest_stack=next_stack,
        )
        if (
            type(auxiliary_replay.binding)
            is not ConfirmationAuxiliaryActivationAuthorityBinding
            or type(auxiliary_replay.activation_artifact)
            is not ReducerActivationArtifact
            or auxiliary_replay.registry != registry
            or auxiliary_replay.binding.generated_registry != registry_source
            or auxiliary_replay.binding.runtime != runtime
            or auxiliary_replay.binding.split != split
            or auxiliary_replay.binding.inventory_authority != inventory_binding
            or auxiliary_replay.binding.experiment != experiment
            or (
                auxiliary_replay.binding.trace.canonical_sha256,
                auxiliary_replay.binding.sampling.canonical_sha256,
                content_sha256(
                    _load_hardware_envelope(auxiliary_replay.binding.hardware_envelope)
                ),
            )
            != common_external_identity
            or tuple(record.binding for record in auxiliary_replay.dependency_records)
            != common_dependency_bindings
        ):
            raise ValueError(
                "confirmation aggregate auxiliary differs from stage lineage"
            )
        expected_auxiliary_ids = tuple(
            cell.cell_id for cell in expected_auxiliary_cells
        )
        if (
            tuple(
                disposition.cell_id
                for disposition in auxiliary_replay.activation_artifact.dispositions
            )
            != expected_auxiliary_ids
        ):
            raise ValueError(
                "confirmation aggregate auxiliary does not cover exact registry remainder"
            )
        auxiliary_completed_source = bind_budget_raw_json(
            _strict_path_field(
                auxiliary_row,
                "completed_cells_artifact",
                label="confirmation aggregate auxiliary",
            ),
            role="confirmation_auxiliary_completed_cells",
        )
        direct = (
            auxiliary_replay.dependency_records[-1]
            if auxiliary_replay.dependency_records
            else None
        )
        auxiliary_authority = CompletedCellAuthority(
            completed_cells=_durable_completed_cells(auxiliary_completed_source),
            registry=registry,
            inventory=inventory,
            direct_dependency_receipt=None if direct is None else direct.receipt,
            dependency_authority=None if direct is None else direct.authority,
            activation_artifact=auxiliary_replay.activation_artifact,
            family_activations=(),
            family_power_reductions=(),
            prior_family_authorities=(),
            raw_activation_authority=auxiliary_replay.binding,
        )
        auxiliary_binding = ConfirmationAuxiliaryCompletionAuthorityBinding(
            schema_version=1,
            completed_cells=auxiliary_completed_source,
            activation=auxiliary_replay.binding,
            inventory_authority=inventory_binding,
            completed_authority_sha256=auxiliary_authority.sha256,
        )
        auxiliary_activation = auxiliary_replay.activation_artifact
    elif raw_auxiliary is not None:
        raise ValueError(
            "confirmation aggregate cannot add an empty-stage auxiliary authority"
        )
    stage_cell_ids = {cell.cell_id for cell in registry.cells_for(experiment)}
    auxiliary_scope = (
        set()
        if auxiliary_activation is None
        else {row.cell_id for row in auxiliary_activation.dispositions}
    )
    disposition_cell_ids = set().union(*family_scopes, auxiliary_scope)
    if disposition_cell_ids != stage_cell_ids or sum(
        len(scope) for scope in family_scopes
    ) + len(auxiliary_scope) != len(stage_cell_ids):
        raise ValueError(
            "confirmation aggregate dispositions do not exactly partition the stage"
        )
    final_activations = tuple(family_activations[1::2])
    disposition_rows = tuple(
        sorted(
            tuple(
                {
                    "cell_id": disposition.cell_id,
                    "status": disposition.status.value,
                    "reason_code": disposition.reason_code,
                }
                for activation in final_activations
                for disposition in activation.dispositions
            )
            + tuple(
                {
                    "cell_id": disposition.cell_id,
                    "status": disposition.status.value,
                    "reason_code": disposition.reason_code,
                }
                for disposition in (
                    ()
                    if auxiliary_activation is None
                    else auxiliary_activation.dispositions
                )
            ),
            key=lambda value: value["cell_id"],
        )
    )
    activated_cell_ids = tuple(
        sorted(
            tuple(
                cell_id
                for activation in final_activations
                for cell_id in activation.activated_cell_ids
            )
            + (
                ()
                if auxiliary_activation is None
                else auxiliary_activation.plan.activated_cell_ids
            )
        )
    )
    dispositions_sha256 = content_sha256(disposition_rows)
    activation_sha256 = content_sha256(
        {
            "reducer_activation_sha256s": (
                () if auxiliary_activation is None else (auxiliary_activation.sha256,)
            ),
            "family_activation_sha256s": tuple(
                sorted(value.sha256 for value in family_activations)
            ),
            "family_power_reduction_sha256s": tuple(
                sorted(value.sha256 for value in family_power)
            ),
        }
    )
    binding = ConfirmationStageAggregateAuthorityBinding(
        schema_version=1,
        kind="confirmation_stage_aggregate_manifest",
        manifest=manifest_source,
        generated_registry=registry_source,
        stage_receipt=stage_receipt_source,
        stage_completed_cells=stage_completed_source,
        runtime=runtime,
        split=split,
        inventory_authority=inventory_binding,
        experiment=experiment,
        families=tuple(family_bindings),
        auxiliary_completion_authority=auxiliary_binding,
        stage_receipt_sha256=stage_receipt.sha256,
        family_sha256s=declared_family_sha256s,
        activated_cell_ids=activated_cell_ids,
        dispositions_sha256=dispositions_sha256,
        activation_sha256=activation_sha256,
    )
    _budget_activation_raw_sources(binding)
    return BudgetActivationAuthorityResult(
        binding=binding,
        registry=registry,
        activation_artifact=auxiliary_activation,
        family_activations=tuple(family_activations),
        family_power_reductions=tuple(family_power),
        dependency_records=common_records,
        prior_family_authorities=tuple(pilot_completions),
        stage_family_authorities=tuple(family_completions),
        auxiliary_authority=auxiliary_authority,
    )


def _bind_stage_activation_authority(
    manifest_path: str | Path,
    *,
    manifest_stack: tuple[Path, ...] = (),
) -> BudgetActivationAuthorityResult:
    source = _exact_existing_path(manifest_path, label="activation authority manifest")
    value = _parse_json(
        _regular_file_bytes(source, label="activation authority manifest"),
        label="activation authority manifest",
    )
    if type(value) is not dict:
        raise TypeError("activation authority manifest must be a JSON object")
    kind = value.get("kind")
    if kind == "industrial_registry_stage_activation_manifest":
        binding, registry, activation, records = _bind_registry_stage_activation(
            source,
            manifest_stack=manifest_stack,
        )
        return BudgetActivationAuthorityResult(
            binding=binding,
            registry=registry,
            activation_artifact=activation,
            family_activations=(),
            family_power_reductions=(),
            dependency_records=records,
            prior_family_authorities=(),
        )
    if kind == "industrial_e1_activation_authority_manifest":
        return _bind_e1_activation_authority(source, manifest_stack=manifest_stack)
    if kind == "industrial_e2_activation_authority_manifest":
        return _bind_e2_activation_authority(source, manifest_stack=manifest_stack)
    if kind == "industrial_confirmation_auxiliary_activation_authority_manifest":
        return _bind_confirmation_auxiliary_activation_authority(
            source, manifest_stack=manifest_stack
        )
    if kind == "industrial_confirmation_pilot_activation_authority_manifest":
        return _bind_confirmation_pilot_activation_authority(
            source, manifest_stack=manifest_stack
        )
    if kind == "industrial_confirmation_final_activation_authority_manifest":
        return _bind_confirmation_final_activation_authority(
            source, manifest_stack=manifest_stack
        )
    if kind == "industrial_confirmation_stage_aggregate_authority_manifest":
        return _bind_confirmation_stage_aggregate_authority(
            source, manifest_stack=manifest_stack
        )
    raise ValueError("activation authority manifest has an unsupported tagged kind")


def bind_budget_activation_authority(
    manifest_path: str | Path,
) -> BudgetActivationAuthorityBinding:
    """Bind one strict tagged raw activation authority and rerun its reducer."""

    return _bind_stage_activation_authority(manifest_path).binding


def replay_budget_activation_authority(
    binding: BudgetActivationAuthorityBinding,
) -> BudgetActivationAuthorityResult:
    """Reopen every path in a tagged authority and exact-compare reducer output."""

    if type(binding) not in {
        RegistryStageActivationAuthorityBinding,
        E1ActivationAuthorityBinding,
        E2ActivationAuthorityBinding,
        ConfirmationAuxiliaryActivationAuthorityBinding,
        ConfirmationPilotActivationAuthorityBinding,
        ConfirmationFinalActivationAuthorityBinding,
        ConfirmationStageAggregateAuthorityBinding,
    }:
        raise TypeError("activation replay requires an exact tagged authority")
    rebound = _bind_stage_activation_authority(binding.manifest.path)
    if rebound.binding != binding:
        raise ValueError("activation reducer output differs from its raw binding")
    activated = {cell.cell_id: cell for cell in rebound.registry.cells}
    if type(binding) is ConfirmationStageAggregateAuthorityBinding:
        activated_ids = tuple(
            sorted(
                (
                    ()
                    if rebound.activation_artifact is None
                    else rebound.activation_artifact.plan.activated_cell_ids
                )
                + tuple(
                    cell_id
                    for artifact in rebound.family_activations
                    for cell_id in artifact.activated_cell_ids
                )
            )
        )
    elif rebound.activation_artifact is not None:
        activated_ids = (
            rebound.activation_artifact.activated_cell_ids
            if type(rebound.activation_artifact) is RegistryStageActivationArtifact
            else (rebound.activation_artifact.plan.activated_cell_ids)
        )
    else:
        activated_ids = tuple(
            cell_id
            for artifact in rebound.family_activations
            for cell_id in artifact.activated_cell_ids
        )
    for cell_id in set(activated_ids):
        cell = activated[cell_id]
        if (
            cell.status is not CellStatus.UNMEASURED
            or cell.resources.workload_class
            in {WorkloadClass.COMPILE, WorkloadClass.DOWNLOAD}
        ):
            raise ValueError(
                "budget authority supports only runnable serving activation"
            )
    if type(binding) is RegistryStageActivationAuthorityBinding:
        artifact = rebound.activation_artifact
        if type(artifact) is not RegistryStageActivationArtifact:  # pragma: no cover
            raise TypeError("generic authority replay lost its stage activation")
        if any(
            activated[cell_id].identity.method != "target_only"
            and not (
                artifact.experiment == "E3a"
                and activated[cell_id].identity.method == "static"
            )
            and not is_serving_interference_calibration_cell(activated[cell_id])
            for cell_id in set(activated_ids)
        ):
            raise ValueError(
                "generic budget authority supports only Target-only, E3a Static, "
                "or the registered Static interference calibration activation"
            )
    return rebound


def bind_registry_stage_activation_authority(
    manifest_path: str | Path,
) -> RegistryStageActivationAuthorityBinding:
    """Bind a generic-stage manifest and its recursive completion path graph."""

    binding, _, _, _ = _bind_registry_stage_activation(manifest_path)
    return binding


def replay_registry_stage_activation_authority(
    binding: RegistryStageActivationAuthorityBinding,
) -> tuple[ExperimentRegistry, RegistryStageActivationArtifact]:
    """Reopen a generic-stage activation manifest and rerun its reducer."""

    if type(binding) is not RegistryStageActivationAuthorityBinding:
        raise TypeError("activation replay requires an exact authority binding")
    replay = replay_budget_activation_authority(binding)
    if type(replay.activation_artifact) is not RegistryStageActivationArtifact:
        raise TypeError("generic activation replay produced another artifact kind")
    return replay.registry, replay.activation_artifact


def _completion_block_reason(error: CompletionAuthorityUnavailableError) -> str:
    message = str(error)
    if E2_STAGE_COMPLETION_AUTHORITY_MISSING_REASON in message:
        return E2_STAGE_COMPLETION_AUTHORITY_MISSING_REASON
    if "trusted_hardware_attester_unavailable" in message:
        return "trusted_hardware_attester_unavailable"
    if "non-serving execution has no release terminal contract" in message:
        return "dependency_completion_non_serving_terminal_authority_unavailable"
    if "E1/E2 raw reducer source bundle is unavailable" in message:
        return DEPENDENCY_COMPLETION_SPECIALIZED_ACTIVATION_REASON
    if "dependency" in message and "authority missing" in message:
        return DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON
    return "dependency_completion_terminal_authority_unavailable"


def require_ready_registry_stage_dependency_completions(
    binding: RegistryStageActivationAuthorityBinding,
    *,
    expected_registry: ExperimentRegistry,
    expected_gpu_inventory: object,
) -> tuple[CompletedCellAuthority, ...]:
    """Reopen and replay every prior schema-v4 completion in prefix order."""

    if type(binding) is not RegistryStageActivationAuthorityBinding:
        raise TypeError("dependency completion replay requires exact activation")
    return require_ready_budget_activation_dependency_completions(
        binding,
        expected_registry=expected_registry,
        expected_gpu_inventory=expected_gpu_inventory,
    )


def require_ready_budget_activation_dependency_completions(
    binding: BudgetActivationAuthorityBinding,
    *,
    expected_registry: ExperimentRegistry,
    expected_gpu_inventory: object,
) -> tuple[CompletedCellAuthority, ...]:
    """Require every raw dependency and prior family completion to validate."""

    from lightcone_spec.experiments.gpu_pool import GpuInventory

    if type(expected_registry) is not ExperimentRegistry:
        raise TypeError("dependency completion replay requires an exact registry")
    if type(expected_gpu_inventory) is not GpuInventory:
        raise TypeError("dependency completion replay requires the full GPU inventory")
    replay = replay_budget_activation_authority(binding)
    if replay.registry != expected_registry:
        raise ValueError("dependency completion authority differs from execution")
    if replay.dependency_receipts and not replay.dependency_records:
        raise BudgetMaterializationBlockedError(
            DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON
        )
    if len(replay.dependency_records) != len(replay.dependency_receipts):
        raise ValueError("dependency completion authority has incomplete coverage")
    completion_results: dict[str, object] = {}
    for record, receipt in zip(
        replay.dependency_records, replay.dependency_receipts, strict=True
    ):
        authority = record.authority
        if (
            record.receipt != receipt
            or authority.registry != expected_registry
            or authority.inventory != expected_gpu_inventory
            or authority.sha256 != record.binding.completed_authority_sha256
        ):
            raise ValueError(
                "dependency completion receipt, registry, or inventory was swapped"
            )
        try:
            result = authority.revalidate()
        except CompletionAuthorityUnavailableError as error:
            raise BudgetMaterializationBlockedError(
                _completion_block_reason(error)
            ) from error
        if (
            result.experiment != receipt.experiment
            or result.completed_cells_sha256 != receipt.completed_cells_sha256
        ):
            raise ValueError(
                "dependency receipt differs from validated completed-cell authority"
            )
        completion_results[result.experiment] = result

    raw_dependency_manifest = None
    raw_dependency_experiment = None
    if type(binding) is E1ActivationAuthorityBinding:
        from lightcone_spec.experiments.industrial_analysis import (
            raw_e3a_selection_manifest_from_dict,
        )

        raw_dependency_manifest = raw_e3a_selection_manifest_from_dict(
            load_budget_raw_json(binding.selection_manifest)
        )
        raw_dependency_experiment = "E3a"
    elif type(binding) is E2ActivationAuthorityBinding:
        from lightcone_spec.experiments.industrial_analysis import (
            raw_e1_pareto_manifest_from_dict,
        )

        raw_dependency_manifest = raw_e1_pareto_manifest_from_dict(
            load_budget_raw_json(binding.pareto_manifest)
        )
        raw_dependency_experiment = "E1"
    if raw_dependency_manifest is not None:
        completed = completion_results.get(str(raw_dependency_experiment))
        if completed is None:
            raise ValueError("raw selection dependency lacks completed authority")
        manifest_cell_ids = tuple(
            sorted(cell.cell_id for cell in raw_dependency_manifest.cells)
        )
        manifest_terminal_sha256s = tuple(
            sorted(
                receipt.sha256
                for cell in raw_dependency_manifest.cells
                for receipt in cell.terminal_receipts
            )
        )
        completed_cell_ids = tuple(sorted(completed.completed_cell_ids))
        completed_terminal_sha256s = tuple(
            sorted(
                terminal.terminal_receipt_sha256
                for terminal in completed.terminal_bindings
            )
        )
        if (
            manifest_cell_ids != completed_cell_ids
            or manifest_terminal_sha256s != completed_terminal_sha256s
        ):
            raise ValueError(
                "raw selection evidence differs from dependency completion"
            )

    if type(binding) is E2ActivationAuthorityBinding:
        from lightcone_spec.experiments.industrial_analysis import (
            raw_e2_stage_manifest_from_dict,
        )

        if (
            len(replay.prior_e2_stage_authorities) != binding.stage_index
            or len(binding.prior_stage_completion_authorities) != binding.stage_index
        ):
            raise BudgetMaterializationBlockedError(
                E2_STAGE_COMPLETION_AUTHORITY_MISSING_REASON
            )
        for expected_stage, (
            authority,
            completion_binding,
            manifest_source,
        ) in enumerate(
            zip(
                replay.prior_e2_stage_authorities,
                binding.prior_stage_completion_authorities,
                binding.prior_stage_manifests,
                strict=True,
            )
        ):
            if (
                authority.registry != expected_registry
                or authority.inventory != expected_gpu_inventory
                or authority.sha256 != completion_binding.completed_authority_sha256
                or completion_binding.stage_activation.stage_index != expected_stage
            ):
                raise ValueError("E2 prior completion authority was swapped")
            try:
                completed = authority.revalidate()
            except CompletionAuthorityUnavailableError as error:
                raise BudgetMaterializationBlockedError(
                    _completion_block_reason(error)
                ) from error
            raw_stage = raw_e2_stage_manifest_from_dict(
                load_budget_raw_json(manifest_source)
            )
            raw_cell_ids = tuple(sorted(cell.cell_id for cell in raw_stage.cells))
            raw_terminal_sha256s = tuple(
                sorted(
                    receipt.sha256
                    for cell in raw_stage.cells
                    for receipt in cell.terminal_receipts
                )
            )
            completed_terminal_sha256s = tuple(
                sorted(
                    terminal.terminal_receipt_sha256
                    for terminal in completed.terminal_bindings
                )
            )
            if (
                raw_stage.stage_index != expected_stage
                or completed.experiment != "E2"
                or tuple(sorted(completed.completed_cell_ids)) != raw_cell_ids
                or completed_terminal_sha256s != raw_terminal_sha256s
            ):
                raise ValueError(
                    "E2 raw round differs from schema-v4 completion authority"
                )
    if type(binding) is ConfirmationStageAggregateAuthorityBinding:
        if (
            len(replay.stage_family_authorities) != len(binding.families)
            or binding.stage_receipt_sha256 != binding.stage_receipt.semantic_sha256
            or (binding.auxiliary_completion_authority is None)
            != (replay.auxiliary_authority is None)
        ):
            raise BudgetMaterializationBlockedError(
                DEPENDENCY_COMPLETION_FAMILY_STAGE_AGGREGATION_MISSING_REASON
            )
        final_by_family = {
            activation.family.sha256: activation
            for activation in replay.family_activations
            if activation.activation_round == "final_prefix"
        }
        if tuple(sorted(final_by_family)) != binding.family_sha256s:
            raise ValueError("confirmation aggregate final family set changed")
        for family_binding, authority in zip(
            binding.families, replay.stage_family_authorities, strict=True
        ):
            final = family_binding.final_activation_authority
            if (
                authority.registry != expected_registry
                or authority.inventory != expected_gpu_inventory
                or authority.sha256
                != family_binding.completion_authority.completed_authority_sha256
                or family_binding.completion_authority.final_activation != final
            ):
                raise ValueError("confirmation aggregate family completion was swapped")
            try:
                completed = authority.revalidate()
            except CompletionAuthorityUnavailableError as error:
                raise BudgetMaterializationBlockedError(
                    _completion_block_reason(error)
                ) from error
            expected_cell_ids = tuple(
                sorted(final_by_family[family_binding.family_sha256].activated_cell_ids)
            )
            if (
                completed.experiment != binding.experiment
                or completed.completed_cell_ids != expected_cell_ids
            ):
                raise ValueError(
                    "confirmation aggregate final completion differs from activation"
                )
        if binding.auxiliary_completion_authority is not None:
            auxiliary_authority = replay.auxiliary_authority
            auxiliary_binding = binding.auxiliary_completion_authority
            if (
                auxiliary_authority is None
                or auxiliary_authority.registry != expected_registry
                or auxiliary_authority.inventory != expected_gpu_inventory
                or auxiliary_authority.sha256
                != auxiliary_binding.completed_authority_sha256
                or replay.activation_artifact is None
                or auxiliary_binding.activation.activation_sha256
                != replay.activation_artifact.sha256
            ):
                raise ValueError(
                    "confirmation aggregate auxiliary completion was swapped"
                )
            try:
                auxiliary_completed = auxiliary_authority.revalidate()
            except CompletionAuthorityUnavailableError as error:
                raise BudgetMaterializationBlockedError(
                    _completion_block_reason(error)
                ) from error
            if (
                auxiliary_completed.experiment != binding.experiment
                or auxiliary_completed.completed_cell_ids
                != tuple(sorted(replay.activation_artifact.plan.activated_cell_ids))
            ):
                raise ValueError(
                    "confirmation aggregate auxiliary completion differs from activation"
                )
    pilot_scope_by_family = {
        activation.family.sha256: frozenset(activation.activated_cell_ids)
        for activation in replay.family_activations
        if activation.activation_round == "excluded_pilots"
    }
    final_families = {
        activation.family.sha256
        for activation in replay.family_activations
        if activation.activation_round == "final_prefix"
    }
    if final_families - pilot_scope_by_family.keys():
        raise ValueError("final family activation lacks its pilot activation")
    expected_prior_scopes = {
        pilot_scope_by_family[family_sha256] for family_sha256 in final_families
    }
    actual_prior_scopes: set[frozenset[str]] = set()
    for authority in replay.prior_family_authorities:
        if (
            authority.registry != expected_registry
            or authority.inventory != expected_gpu_inventory
        ):
            raise ValueError("prior family completion registry/inventory was swapped")
        try:
            prior = authority.revalidate()
        except CompletionAuthorityUnavailableError as error:
            raise BudgetMaterializationBlockedError(
                _completion_block_reason(error)
            ) from error
        scope = frozenset(prior.completed_cell_ids)
        if scope in actual_prior_scopes:
            raise ValueError("prior family completion scopes are duplicated")
        actual_prior_scopes.add(scope)
        if scope not in expected_prior_scopes:
            raise ValueError(
                "prior family completion differs from raw pilot activation"
            )
    if actual_prior_scopes != expected_prior_scopes:
        raise ValueError("prior family completion coverage is incomplete")
    return tuple(
        [record.authority for record in replay.dependency_records]
        + list(replay.prior_e2_stage_authorities)
        + list(replay.stage_family_authorities)
        + ([] if replay.auxiliary_authority is None else [replay.auxiliary_authority])
        + list(replay.prior_family_authorities)
    )


def load_declared_budget_plan(
    binding: BudgetMaterializationAuthorityBinding,
) -> BudgetPlan:
    """Strictly reopen the declared plan without claiming rematerialization."""

    if type(binding) is not BudgetMaterializationAuthorityBinding:
        raise TypeError("declared plan loader requires an exact authority binding")
    plan = budget_plan_from_dict(load_budget_raw_json(binding.declared_plan))
    if plan.sha256 != binding.declared_plan_sha256:
        raise ValueError("declared BudgetPlan differs from its authority binding")
    return plan


@dataclass(frozen=True)
class BudgetMaterializationAuthorityResult:
    registry: ExperimentRegistry
    activation: (
        RegistryStageActivationArtifact
        | ReducerActivationArtifact
        | FamilyActivationArtifact
    )
    family_activations: tuple[FamilyActivationArtifact, ...]
    family_power_reductions: tuple[ConfirmationFamilyPowerReductionArtifact, ...]
    policy: BudgetPolicy
    load_bindings: tuple[BudgetLoadBinding, ...]
    capacity_envelope: CapacityEnvelope
    capacity_authority: CapacityAuthorityBinding
    budget_plan: BudgetPlan

    def __post_init__(self) -> None:
        if type(self.registry) is not ExperimentRegistry:
            raise TypeError("budget authority result requires an exact registry")
        if type(self.activation) not in {
            RegistryStageActivationArtifact,
            ReducerActivationArtifact,
            FamilyActivationArtifact,
        }:
            raise TypeError("budget authority result requires an exact activation")
        if any(
            type(value) is not FamilyActivationArtifact
            for value in self.family_activations
        ):
            raise TypeError("budget authority result has invalid family activations")
        if any(
            type(value) is not ConfirmationFamilyPowerReductionArtifact
            for value in self.family_power_reductions
        ):
            raise TypeError("budget authority result has invalid family power")
        if type(self.policy) is not BudgetPolicy:
            raise TypeError("budget authority result requires an exact policy")
        if any(type(value) is not BudgetLoadBinding for value in self.load_bindings):
            raise TypeError("budget authority result requires exact load bindings")
        if type(self.capacity_envelope) is not CapacityEnvelope:
            raise TypeError("budget authority result requires exact capacity")
        if type(self.capacity_authority) is not CapacityAuthorityBinding:
            raise TypeError("budget authority result requires exact capacity authority")
        if type(self.budget_plan) is not BudgetPlan:
            raise TypeError("budget authority result requires an exact BudgetPlan")


def _rematerialize_budget_plan(
    replay: BudgetActivationAuthorityResult,
    *,
    load_bindings: tuple[BudgetLoadBinding, ...],
    policy: BudgetPolicy,
    inventory: BudgetInventoryIdentity,
    capacity_envelope: CapacityEnvelope,
    capacity_authority: CapacityAuthorityBinding,
) -> BudgetPlan:
    activations = (
        () if replay.activation_artifact is None else (replay.activation_artifact,)
    )
    return materialize_industrial_budgets(
        replay.registry,
        activations=activations,
        family_activations=replay.family_activations,
        family_power_reductions=replay.family_power_reductions,
        load_bindings=load_bindings,
        policy=policy,
        inventory=inventory,
        capacity_envelope=capacity_envelope,
        capacity_authority=capacity_authority,
        require_complete=False,
    )


def bind_budget_materialization_authority(
    *,
    activation_manifest_path: str | Path,
    policy_path: str | Path,
    load_binding_paths: tuple[str | Path, ...],
    capacity_envelope_path: str | Path,
    capacity_authority: CapacityAuthorityBinding,
    declared_plan_path: str | Path,
) -> BudgetMaterializationAuthorityBinding:
    """Bind and fully replay all raw budget sources without requiring READY."""

    if type(capacity_authority) is not CapacityAuthorityBinding:
        raise TypeError("budget authority requires an exact capacity authority")
    rebound_capacity = bind_capacity_authority(
        capacity_authority.source_manifest.path,
        capacity_authority.verification_receipt.path,
    )
    if rebound_capacity != capacity_authority:
        raise ValueError("capacity authority changed before budget binding")
    activation_replay = _bind_stage_activation_authority(activation_manifest_path)
    activation_binding = activation_replay.binding
    registry = activation_replay.registry
    policy_binding = bind_budget_raw_json(policy_path, role="budget_policy")
    policy = budget_policy_from_dict(load_budget_raw_json(policy_binding))
    raw_loads: list[BudgetLoadRawBinding] = []
    decoded_loads: list[BudgetLoadBinding] = []
    for path in load_binding_paths:
        source = bind_budget_raw_json(path, role="budget_load_binding")
        load = budget_load_binding_from_dict(load_budget_raw_json(source))
        raw_loads.append(BudgetLoadRawBinding(cell_id=load.cell_id, source=source))
        decoded_loads.append(load)
    load_bindings = tuple(raw_loads)
    cell_ids = tuple(value.cell_id for value in load_bindings)
    if cell_ids != tuple(sorted(set(cell_ids))):
        raise ValueError("budget load binding paths must be cell-sorted and unique")
    capacity_binding = bind_budget_raw_json(
        capacity_envelope_path,
        role="capacity_envelope",
    )
    capacity_envelope = capacity_envelope_from_dict(
        load_budget_raw_json(capacity_binding)
    )
    declared_binding = bind_budget_raw_json(
        declared_plan_path,
        role="declared_budget_plan",
    )
    declared = budget_plan_from_dict(load_budget_raw_json(declared_binding))
    binding = BudgetMaterializationAuthorityBinding(
        schema_version=1,
        activation=activation_binding,
        policy=policy_binding,
        load_bindings=load_bindings,
        capacity_envelope=capacity_binding,
        capacity_authority=capacity_authority,
        declared_plan=declared_binding,
        registry_sha256=registry.sha256,
        budget_inventory_sha256=declared.inventory.sha256,
        activation_sha256=activation_replay.activation_sha256,
        budget_policy_sha256=policy_binding.semantic_sha256,
        budget_load_binding_sha256s=tuple(
            value.source.semantic_sha256 for value in load_bindings
        ),
        capacity_envelope_sha256=capacity_binding.semantic_sha256,
        capacity_authority_sha256=capacity_authority.sha256,
        declared_plan_sha256=declared.sha256,
        authority_protocol_sha256=(BUDGET_MATERIALIZATION_AUTHORITY_PROTOCOL_SHA256),
    )
    rematerialized = _rematerialize_budget_plan(
        activation_replay,
        load_bindings=tuple(decoded_loads),
        policy=policy,
        inventory=declared.inventory,
        capacity_envelope=capacity_envelope,
        capacity_authority=rebound_capacity,
    )
    if declared != rematerialized or declared.sha256 != rematerialized.sha256:
        raise ValueError(
            "declared BudgetPlan differs from first-party raw rematerialization"
        )
    return binding


def revalidate_budget_materialization_authority_binding(
    binding: BudgetMaterializationAuthorityBinding,
    *,
    expected_registry: ExperimentRegistry,
    expected_inventory: BudgetInventoryIdentity,
    expected_activation: (
        RegistryStageActivationArtifact
        | ReducerActivationArtifact
        | FamilyActivationArtifact
        | None
    ) = None,
    expected_plan: BudgetPlan | None = None,
) -> BudgetMaterializationAuthorityResult:
    """Reopen raw inputs, rerun the reducer, and exact-compare the plan."""

    if type(binding) is not BudgetMaterializationAuthorityBinding:
        raise TypeError("formal budget authority requires an exact binding")
    if type(expected_registry) is not ExperimentRegistry:
        raise TypeError("formal budget authority requires an exact registry")
    if type(expected_inventory) is not BudgetInventoryIdentity:
        raise TypeError("formal budget authority requires an exact inventory")
    if expected_activation is not None and type(expected_activation) not in {
        RegistryStageActivationArtifact,
        ReducerActivationArtifact,
        FamilyActivationArtifact,
    }:
        raise TypeError("expected activation must be an exact activation artifact")
    if expected_plan is not None and type(expected_plan) is not BudgetPlan:
        raise TypeError("expected plan must be an exact BudgetPlan")
    activation_replay = replay_budget_activation_authority(binding.activation)
    registry = activation_replay.registry
    activation = activation_replay.selected_activation
    if (
        registry != expected_registry
        or registry.sha256 != binding.registry_sha256
        or expected_inventory.sha256 != binding.budget_inventory_sha256
        or activation_replay.activation_sha256 != binding.activation_sha256
        or (expected_activation is not None and activation != expected_activation)
    ):
        raise ValueError("budget materialization authority differs from execution")
    policy = budget_policy_from_dict(load_budget_raw_json(binding.policy))
    load_bindings = tuple(
        budget_load_binding_from_dict(load_budget_raw_json(value.source))
        for value in binding.load_bindings
    )
    if tuple(value.cell_id for value in load_bindings) != tuple(
        value.cell_id for value in binding.load_bindings
    ):
        raise ValueError("budget load binding cells differ from raw authority")
    capacity_envelope = capacity_envelope_from_dict(
        load_budget_raw_json(binding.capacity_envelope)
    )
    rebound_capacity = bind_capacity_authority(
        binding.capacity_authority.source_manifest.path,
        binding.capacity_authority.verification_receipt.path,
    )
    if rebound_capacity != binding.capacity_authority:
        raise ValueError("raw capacity authority differs from budget authority")
    declared = load_declared_budget_plan(binding)
    rematerialized = _rematerialize_budget_plan(
        activation_replay,
        load_bindings=load_bindings,
        policy=policy,
        inventory=expected_inventory,
        capacity_envelope=capacity_envelope,
        capacity_authority=rebound_capacity,
    )
    if (
        policy.sha256 != binding.budget_policy_sha256
        or tuple(value.sha256 for value in load_bindings)
        != binding.budget_load_binding_sha256s
        or capacity_envelope.sha256 != binding.capacity_envelope_sha256
        or rebound_capacity.sha256 != binding.capacity_authority_sha256
        or declared != rematerialized
        or declared.sha256 != rematerialized.sha256
        or (expected_plan is not None and declared != expected_plan)
    ):
        raise ValueError(
            "declared BudgetPlan differs from first-party raw rematerialization"
        )
    return BudgetMaterializationAuthorityResult(
        registry=registry,
        activation=activation,
        family_activations=activation_replay.family_activations,
        family_power_reductions=activation_replay.family_power_reductions,
        policy=policy,
        load_bindings=load_bindings,
        capacity_envelope=capacity_envelope,
        capacity_authority=rebound_capacity,
        budget_plan=rematerialized,
    )


def require_ready_budget_materialization_authority_binding(
    binding: BudgetMaterializationAuthorityBinding,
    *,
    expected_registry: ExperimentRegistry,
    expected_inventory: BudgetInventoryIdentity,
    expected_activation: (
        RegistryStageActivationArtifact
        | ReducerActivationArtifact
        | FamilyActivationArtifact
        | None
    ) = None,
    expected_plan: BudgetPlan | None = None,
    expected_gpu_inventory: GpuInventory,
) -> BudgetMaterializationAuthorityResult:
    """Replay all raw inputs and require a complete source-authorized plan."""

    result = revalidate_budget_materialization_authority_binding(
        binding,
        expected_registry=expected_registry,
        expected_inventory=expected_inventory,
        expected_activation=expected_activation,
        expected_plan=expected_plan,
    )
    if expected_gpu_inventory is None:
        raise BudgetMaterializationBlockedError(
            DEPENDENCY_COMPLETION_GPU_INVENTORY_MISSING_REASON
        )
    require_ready_budget_activation_dependency_completions(
        binding.activation,
        expected_registry=expected_registry,
        expected_gpu_inventory=expected_gpu_inventory,
    )
    if result.budget_plan.status != "READY":
        reasons = tuple(
            sorted({value.reason_code for value in result.budget_plan.dispositions})
        )
        reason = (
            reasons[0]
            if len(reasons) == 1
            else BUDGET_MATERIALIZATION_UNRESOLVED_REASON
        )
        raise BudgetMaterializationBlockedError(reason)
    result.budget_plan.require_ready()
    return result


__all__ = [
    "BUDGET_MATERIALIZATION_UNRESOLVED_REASON",
    "DEPENDENCY_COMPLETION_FAMILY_STAGE_AGGREGATION_MISSING_REASON",
    "DEPENDENCY_COMPLETION_GPU_INVENTORY_MISSING_REASON",
    "DEPENDENCY_COMPLETION_LOCKED_OUTPUT_UNSUPPORTED_REASON",
    "DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON",
    "DEPENDENCY_COMPLETION_SPECIALIZED_ACTIVATION_REASON",
    "E2_STAGE_COMPLETION_AUTHORITY_MISSING_REASON",
    "BudgetActivationAuthorityResult",
    "BudgetMaterializationAuthorityResult",
    "BudgetMaterializationBlockedError",
    "bind_budget_activation_authority",
    "bind_budget_materialization_authority",
    "bind_budget_raw_json",
    "bind_registry_stage_activation_authority",
    "load_budget_raw_json",
    "load_declared_budget_plan",
    "replay_budget_activation_authority",
    "replay_registry_stage_activation_authority",
    "require_ready_budget_activation_dependency_completions",
    "require_ready_budget_materialization_authority_binding",
    "require_ready_registry_stage_dependency_completions",
    "revalidate_budget_materialization_authority_binding",
]
