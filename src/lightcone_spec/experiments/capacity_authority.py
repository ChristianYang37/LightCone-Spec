"""Path-bound raw capacity authority for formal industrial budgets.

``CapacityEnvelope`` is useful arithmetic, but its numbers and receipt digest
are caller-authored declarations.  This module reopens the exact envelope,
physical inventory and inventory probe, provider quota, host filesystem
capacity, and one sizing receipt (including three raw provenance sources) per
cell.  A release verifier then signs the complete source manifest.  Only the
source-owned release policy can turn that replay into execution authority.

The current release policy is intentionally empty.  Consequently these APIs
can validate and preserve a diagnostic source bundle, but formal revalidation
returns a named BLOCKED condition rather than treating a test signature as
authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.planning import (
    CAPACITY_AUTHORITY_PROTOCOL_SHA256,
    CAPACITY_MAXIMUM_SOURCE_AGE_NS,
    CELL_CAPACITY_SIZING_PROTOCOL_SHA256,
    BudgetInventoryIdentity,
    CapacityAuthorityBinding,
    CapacityEnvelope,
    CapacityRawJsonBinding,
    CellCapacityRequirement,
    budget_inventory_identity_from_gpu_inventory,
)
from lightcone_spec.experiments.planning_artifacts import (
    capacity_envelope_from_dict,
)
from lightcone_spec.experiments.registry import content_sha256
from lightcone_spec.runtime.attestation import (
    RELEASE_TRUSTED_ATTESTER_POLICY,
    AttestationChallenge,
    SignedAttestation,
    require_release_trusted_attester_policy,
)

TRUSTED_CAPACITY_VERIFIER_UNAVAILABLE_REASON = "trusted_capacity_verifier_unavailable"

_SHA256_LENGTH = 64
_RAW_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "path",
        "sidecar_path",
        "semantic_sha256",
        "file_sha256",
        "sidecar_file_sha256",
        "size",
        "sidecar_size",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "authority_protocol_sha256",
        "registry_sha256",
        "budget_inventory_sha256",
        "collection_nonce_sha256",
        "maximum_source_age_ns",
        "sources",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "capacity_envelope",
        "gpu_inventory",
        "gpu_inventory_source_receipt",
        "provider_quota_receipt",
        "host_capacity_receipt",
        "cell_sizing_receipts",
    }
)
_INVENTORY_RECEIPT_FIELDS = frozenset(
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
_PROVIDER_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "budget_inventory_sha256",
        "gpu_inventory_sha256",
        "inventory_source_receipt_sha256",
        "provider_scope_sha256",
        "collection_nonce_sha256",
        "captured_at_ns",
        "total_quota_gpu_ms",
        "consumed_gpu_ms",
        "available_gpu_ms",
    }
)
_HOST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "budget_inventory_sha256",
        "gpu_inventory_sha256",
        "inventory_source_receipt_sha256",
        "host_sha256",
        "filesystem_sha256",
        "collection_nonce_sha256",
        "captured_at_ns",
        "host_free_bytes",
        "host_quota_bytes",
    }
)
_SIZING_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "registry_sha256",
        "budget_inventory_sha256",
        "cell_id",
        "maximum_evidence_bytes",
        "model_staging_bytes",
        "compile_overlay_bytes",
        "evidence_contract_source",
        "model_staging_source",
        "compile_overlay_source",
        "sizing_protocol_sha256",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "cell_id",
        "maximum_bytes",
        "derivation_sha256",
    }
)
_VERIFICATION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "authority_protocol_sha256",
        "source_manifest_sha256",
        "verification_payload",
        "challenge",
        "attestation",
    }
)
_VERIFICATION_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "authority_protocol_sha256",
        "source_manifest_sha256",
        "registry_sha256",
        "budget_inventory_sha256",
        "capacity_envelope_sha256",
        "capacity_source_receipt_sha256",
        "gpu_inventory_sha256",
        "inventory_source_receipt_sha256",
        "provider_quota_receipt_sha256",
        "host_capacity_receipt_sha256",
        "cell_sizing_receipt_sha256s",
    }
)
_CHALLENGE_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "challenge_id",
        "nonce_base64",
        "subject_sha256",
        "issued_ns",
        "expires_ns",
    }
)
_ATTESTATION_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "algorithm",
        "attester_id",
        "key_id",
        "environment",
        "public_key_base64",
        "challenge_sha256",
        "payload_sha256",
        "signature_base64",
    }
)


class CapacityAuthorityUnavailableError(RuntimeError):
    """The raw bundle is inspectable but cannot authorize this release."""

    def __init__(self, reason_code: str = TRUSTED_CAPACITY_VERIFIER_UNAVAILABLE_REASON):
        self.reason_code = reason_code
        super().__init__(f"capacity authority is BLOCKED: {reason_code}")


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


def _strict_text(name: str, value: object) -> str:
    if type(value) is not str or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be non-empty single-line text")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _strict_object(
    name: str, value: object, expected_fields: frozenset[str]
) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys")
    if set(value) != expected_fields:
        missing = sorted(expected_fields - set(value))
        unknown = sorted(set(value) - expected_fields)
        raise ValueError(f"{name} fields differ: missing={missing}, unknown={unknown}")
    return value


def _strict_list(name: str, value: object) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        value[key] = item
    return value


def _parse_json(body: bytes, *, label: str) -> object:
    try:
        return json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError(f"{label} path must be absolute and resolved")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{label} is not a readable regular file") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            body = handle.read()
        current = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != opened.st_dev
            or current.st_ino != opened.st_ino
            or current.st_size != len(body)
        ):
            raise RuntimeError(f"{label} changed while it was read")
        return body
    finally:
        os.close(descriptor)


def bind_capacity_raw_json(path: str | Path) -> CapacityRawJsonBinding:
    """Bind a raw JSON file and its canonical-hash sidecar without trusting it."""

    source = Path(path).resolve()
    sidecar = Path(f"{source}.sha256").resolve()
    body = _regular_file_bytes(source, label="capacity raw JSON")
    sidecar_body = _regular_file_bytes(sidecar, label="capacity raw JSON sidecar")
    value = _parse_json(body, label="capacity raw JSON")
    semantic_sha256 = content_sha256(value)
    if sidecar_body != f"{semantic_sha256}\n".encode("ascii"):
        raise ValueError("capacity raw JSON sidecar is missing or invalid")
    return CapacityRawJsonBinding(
        schema_version=1,
        path=str(source),
        sidecar_path=str(sidecar),
        semantic_sha256=semantic_sha256,
        file_sha256=hashlib.sha256(body).hexdigest(),
        sidecar_file_sha256=hashlib.sha256(sidecar_body).hexdigest(),
        size=len(body),
        sidecar_size=len(sidecar_body),
    )


def _binding_from_dict(value: object, *, label: str) -> CapacityRawJsonBinding:
    row = _strict_object(label, value, _RAW_BINDING_FIELDS)
    return CapacityRawJsonBinding(
        schema_version=row["schema_version"],
        path=row["path"],
        sidecar_path=row["sidecar_path"],
        semantic_sha256=row["semantic_sha256"],
        file_sha256=row["file_sha256"],
        sidecar_file_sha256=row["sidecar_file_sha256"],
        size=row["size"],
        sidecar_size=row["sidecar_size"],
    )


def load_capacity_raw_json(binding: CapacityRawJsonBinding) -> object:
    """Reopen a raw binding and reject byte, sidecar, path, or semantic drift."""

    if type(binding) is not CapacityRawJsonBinding:
        raise TypeError("capacity source must be an exact raw JSON binding")
    source = Path(binding.path)
    sidecar = Path(binding.sidecar_path)
    body = _regular_file_bytes(source, label="bound capacity JSON")
    sidecar_body = _regular_file_bytes(sidecar, label="bound capacity JSON sidecar")
    if (
        len(body) != binding.size
        or len(sidecar_body) != binding.sidecar_size
        or hashlib.sha256(body).hexdigest() != binding.file_sha256
        or hashlib.sha256(sidecar_body).hexdigest() != binding.sidecar_file_sha256
        or sidecar_body != f"{binding.semantic_sha256}\n".encode("ascii")
    ):
        raise RuntimeError("bound capacity JSON source or sidecar changed")
    value = _parse_json(body, label="bound capacity JSON")
    if content_sha256(value) != binding.semantic_sha256:
        raise RuntimeError("bound capacity JSON semantic identity changed")
    return value


def capacity_source_receipt_sha256_from_paths(
    *,
    inventory_source_receipt_path: str | Path,
    provider_quota_receipt_path: str | Path,
    host_capacity_receipt_path: str | Path,
    cell_sizing_receipt_paths: tuple[str | Path, ...],
) -> str:
    """Derive the non-circular source-set identity used by an envelope."""

    inventory_receipt = bind_capacity_raw_json(inventory_source_receipt_path)
    inventory_value = _strict_object(
        "GPU inventory source receipt",
        load_capacity_raw_json(inventory_receipt),
        _INVENTORY_RECEIPT_FIELDS,
    )
    inventory_receipt_sha256 = _require_sha256(
        "GPU inventory source receipt",
        inventory_value["receipt_sha256"],
    )
    provider = bind_capacity_raw_json(provider_quota_receipt_path)
    host = bind_capacity_raw_json(host_capacity_receipt_path)
    sizing = tuple(bind_capacity_raw_json(path) for path in cell_sizing_receipt_paths)
    return _capacity_source_receipt_sha256(
        inventory_source_receipt_sha256=inventory_receipt_sha256,
        provider_quota_receipt_sha256=provider.semantic_sha256,
        host_capacity_receipt_sha256=host.semantic_sha256,
        cell_sizing_receipt_sha256s=tuple(row.semantic_sha256 for row in sizing),
    )


def _capacity_source_receipt_sha256(
    *,
    inventory_source_receipt_sha256: str,
    provider_quota_receipt_sha256: str,
    host_capacity_receipt_sha256: str,
    cell_sizing_receipt_sha256s: tuple[str, ...],
) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "industrial_capacity_source_receipt_set",
            "inventory_source_receipt_sha256": inventory_source_receipt_sha256,
            "provider_quota_receipt_sha256": provider_quota_receipt_sha256,
            "host_capacity_receipt_sha256": host_capacity_receipt_sha256,
            "cell_sizing_receipt_sha256s": cell_sizing_receipt_sha256s,
            "authority_protocol_sha256": CAPACITY_AUTHORITY_PROTOCOL_SHA256,
        }
    )


def build_capacity_source_manifest(
    *,
    registry_sha256: str,
    budget_inventory_sha256: str,
    collection_nonce_sha256: str,
    capacity_envelope_path: str | Path,
    gpu_inventory_path: str | Path,
    inventory_source_receipt_path: str | Path,
    provider_quota_receipt_path: str | Path,
    host_capacity_receipt_path: str | Path,
    cell_sizing_receipt_paths: tuple[str | Path, ...],
) -> dict[str, object]:
    """Build a strict path manifest; writing it does not confer authority."""

    _require_sha256("capacity manifest registry", registry_sha256)
    _require_sha256("capacity manifest budget inventory", budget_inventory_sha256)
    _require_sha256("capacity collection nonce", collection_nonce_sha256)
    sources = {
        "capacity_envelope": bind_capacity_raw_json(capacity_envelope_path).to_dict(),
        "gpu_inventory": bind_capacity_raw_json(gpu_inventory_path).to_dict(),
        "gpu_inventory_source_receipt": bind_capacity_raw_json(
            inventory_source_receipt_path
        ).to_dict(),
        "provider_quota_receipt": bind_capacity_raw_json(
            provider_quota_receipt_path
        ).to_dict(),
        "host_capacity_receipt": bind_capacity_raw_json(
            host_capacity_receipt_path
        ).to_dict(),
        "cell_sizing_receipts": [
            bind_capacity_raw_json(path).to_dict() for path in cell_sizing_receipt_paths
        ],
    }
    value: dict[str, object] = {
        "schema_version": 1,
        "kind": "industrial_capacity_source_manifest",
        "authority_protocol_sha256": CAPACITY_AUTHORITY_PROTOCOL_SHA256,
        "registry_sha256": registry_sha256,
        "budget_inventory_sha256": budget_inventory_sha256,
        "collection_nonce_sha256": collection_nonce_sha256,
        "maximum_source_age_ns": CAPACITY_MAXIMUM_SOURCE_AGE_NS,
        "sources": sources,
    }
    _validate_capacity_sources(value)
    return value


@dataclass(frozen=True)
class CapacityAuthorityResult:
    """Freshly replayed and release-verified capacity inputs."""

    capacity_envelope: CapacityEnvelope
    budget_inventory: BudgetInventoryIdentity
    gpu_inventory: GpuInventory
    registry_sha256: str
    source_manifest_sha256: str
    verification_receipt_sha256: str
    provider_quota_receipt_sha256: str
    host_capacity_receipt_sha256: str
    cell_sizing_receipt_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.capacity_envelope) is not CapacityEnvelope:
            raise TypeError("capacity result envelope is invalid")
        if type(self.budget_inventory) is not BudgetInventoryIdentity:
            raise TypeError("capacity result budget inventory is invalid")
        if type(self.gpu_inventory) is not GpuInventory:
            raise TypeError("capacity result GPU inventory is invalid")
        for name in (
            "registry_sha256",
            "source_manifest_sha256",
            "verification_receipt_sha256",
            "provider_quota_receipt_sha256",
            "host_capacity_receipt_sha256",
        ):
            _require_sha256(f"capacity result {name}", getattr(self, name))
        if any(not _is_sha256(row) for row in self.cell_sizing_receipt_sha256s):
            raise ValueError("capacity result sizing receipt SHA-256 is invalid")


@dataclass(frozen=True)
class _ValidatedCapacitySources:
    envelope: CapacityEnvelope
    budget_inventory: BudgetInventoryIdentity
    gpu_inventory: GpuInventory
    registry_sha256: str
    manifest_sha256: str
    inventory_source_receipt_sha256: str
    provider_receipt_sha256: str
    host_receipt_sha256: str
    sizing_receipt_sha256s: tuple[str, ...]
    captured_at_ns: int
    raw_paths: tuple[str, ...]

    def verification_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "industrial_capacity_verification_payload",
            "authority_protocol_sha256": CAPACITY_AUTHORITY_PROTOCOL_SHA256,
            "source_manifest_sha256": self.manifest_sha256,
            "registry_sha256": self.registry_sha256,
            "budget_inventory_sha256": self.budget_inventory.sha256,
            "capacity_envelope_sha256": self.envelope.sha256,
            "capacity_source_receipt_sha256": (self.envelope.source_receipt_sha256),
            "gpu_inventory_sha256": self.gpu_inventory.sha256,
            "inventory_source_receipt_sha256": (self.inventory_source_receipt_sha256),
            "provider_quota_receipt_sha256": self.provider_receipt_sha256,
            "host_capacity_receipt_sha256": self.host_receipt_sha256,
            "cell_sizing_receipt_sha256s": list(self.sizing_receipt_sha256s),
        }


def build_capacity_verification_payload(
    source_manifest: object,
) -> dict[str, object]:
    """Replay a manifest into the exact payload a release verifier must sign."""

    return _validate_capacity_sources(source_manifest).verification_payload()


def _validate_capacity_sources(value: object) -> _ValidatedCapacitySources:
    manifest = _strict_object("capacity source manifest", value, _MANIFEST_FIELDS)
    if (
        manifest["schema_version"] != 1
        or manifest["kind"] != "industrial_capacity_source_manifest"
        or manifest["authority_protocol_sha256"] != CAPACITY_AUTHORITY_PROTOCOL_SHA256
        or manifest["maximum_source_age_ns"] != CAPACITY_MAXIMUM_SOURCE_AGE_NS
    ):
        raise ValueError("capacity source manifest protocol is unsupported")
    registry_sha256 = _require_sha256(
        "capacity source registry", manifest["registry_sha256"]
    )
    budget_inventory_sha256 = _require_sha256(
        "capacity source budget inventory", manifest["budget_inventory_sha256"]
    )
    collection_nonce_sha256 = _require_sha256(
        "capacity source collection nonce", manifest["collection_nonce_sha256"]
    )
    source_rows = _strict_object(
        "capacity source manifest sources", manifest["sources"], _SOURCE_FIELDS
    )
    envelope_binding = _binding_from_dict(
        source_rows["capacity_envelope"], label="capacity envelope binding"
    )
    inventory_binding = _binding_from_dict(
        source_rows["gpu_inventory"], label="GPU inventory binding"
    )
    inventory_receipt_binding = _binding_from_dict(
        source_rows["gpu_inventory_source_receipt"],
        label="GPU inventory source receipt binding",
    )
    provider_binding = _binding_from_dict(
        source_rows["provider_quota_receipt"],
        label="provider quota receipt binding",
    )
    host_binding = _binding_from_dict(
        source_rows["host_capacity_receipt"],
        label="host capacity receipt binding",
    )
    sizing_bindings = tuple(
        _binding_from_dict(row, label="cell sizing receipt binding")
        for row in _strict_list(
            "cell sizing receipt bindings", source_rows["cell_sizing_receipts"]
        )
    )
    direct_bindings = (
        envelope_binding,
        inventory_binding,
        inventory_receipt_binding,
        provider_binding,
        host_binding,
        *sizing_bindings,
    )
    if not sizing_bindings:
        raise ValueError("capacity source manifest has no cell sizing receipts")
    if len({row.path for row in direct_bindings}) != len(direct_bindings):
        raise ValueError("capacity source manifest aliases distinct raw sources")

    envelope = capacity_envelope_from_dict(load_capacity_raw_json(envelope_binding))
    inventory = GpuInventory.from_dict(load_capacity_raw_json(inventory_binding))
    budget_inventory = budget_inventory_identity_from_gpu_inventory(inventory)
    if (
        budget_inventory.sha256 != budget_inventory_sha256
        or envelope.budget_inventory_sha256 != budget_inventory_sha256
    ):
        raise ValueError("capacity envelope and raw inventory identity differ")
    inventory_receipt = _validate_inventory_receipt(
        load_capacity_raw_json(inventory_receipt_binding), inventory=inventory
    )
    provider = _validate_provider_receipt(
        load_capacity_raw_json(provider_binding),
        inventory=inventory,
        budget_inventory=budget_inventory,
        collection_nonce_sha256=collection_nonce_sha256,
    )
    host = _validate_host_receipt(
        load_capacity_raw_json(host_binding),
        inventory=inventory,
        budget_inventory=budget_inventory,
        collection_nonce_sha256=collection_nonce_sha256,
    )
    if provider["captured_at_ns"] != host["captured_at_ns"]:
        raise ValueError("provider and host capacity were not captured atomically")
    if envelope.provider_quota_gpu_ms != provider["available_gpu_ms"]:
        raise ValueError("capacity envelope differs from raw provider availability")
    if (
        envelope.host_free_bytes != host["host_free_bytes"]
        or envelope.host_quota_bytes != host["host_quota_bytes"]
    ):
        raise ValueError("capacity envelope differs from raw host capacity")

    requirements: list[CellCapacityRequirement] = []
    nested_paths: list[str] = []
    for binding in sizing_bindings:
        requirement, provenance_paths = _validate_sizing_receipt(
            load_capacity_raw_json(binding),
            registry_sha256=registry_sha256,
            budget_inventory_sha256=budget_inventory_sha256,
        )
        requirements.append(requirement)
        nested_paths.extend(provenance_paths)
    ordered = tuple(sorted(requirements, key=lambda row: row.cell_id))
    if tuple(requirements) != ordered or len(
        {row.cell_id for row in requirements}
    ) != len(requirements):
        raise ValueError("cell sizing receipts must be cell-sorted and unique")
    if envelope.cell_requirements != ordered:
        raise ValueError("capacity envelope differs from per-cell sizing provenance")
    all_paths = tuple(row.path for row in direct_bindings) + tuple(nested_paths)
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("capacity authority aliases raw source paths")

    expected_source_receipt = _capacity_source_receipt_sha256(
        inventory_source_receipt_sha256=inventory_receipt["receipt_sha256"],
        provider_quota_receipt_sha256=provider_binding.semantic_sha256,
        host_capacity_receipt_sha256=host_binding.semantic_sha256,
        cell_sizing_receipt_sha256s=tuple(
            binding.semantic_sha256 for binding in sizing_bindings
        ),
    )
    if envelope.source_receipt_sha256 != expected_source_receipt:
        raise ValueError("capacity envelope source receipt set is forged")
    return _ValidatedCapacitySources(
        envelope=envelope,
        budget_inventory=budget_inventory,
        gpu_inventory=inventory,
        registry_sha256=registry_sha256,
        manifest_sha256=content_sha256(manifest),
        inventory_source_receipt_sha256=inventory_receipt["receipt_sha256"],
        provider_receipt_sha256=provider_binding.semantic_sha256,
        host_receipt_sha256=host_binding.semantic_sha256,
        sizing_receipt_sha256s=tuple(
            binding.semantic_sha256 for binding in sizing_bindings
        ),
        captured_at_ns=provider["captured_at_ns"],
        raw_paths=all_paths,
    )


def _validate_inventory_receipt(
    value: object, *, inventory: GpuInventory
) -> dict[str, Any]:
    receipt = _strict_object(
        "GPU inventory source receipt", value, _INVENTORY_RECEIPT_FIELDS
    )
    if receipt["schema_version"] != 1 or receipt["kind"] != (
        "gpu_inventory_probe_receipt"
    ):
        raise ValueError("GPU inventory source receipt schema is unsupported")
    declared = _require_sha256(
        "GPU inventory source receipt SHA-256", receipt["receipt_sha256"]
    )
    content = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if (
        declared != content_sha256(content)
        or declared != inventory.source_receipt_sha256
    ):
        raise ValueError("GPU inventory source receipt identity mismatch")
    if len(inventory.host_ids) != 1 or receipt["host_id"] != inventory.host_ids[0]:
        raise ValueError("GPU inventory source receipt belongs to another host")
    _strict_text("GPU inventory source hostname", receipt["hostname"])
    _require_sha256("GPU inventory machine identity", receipt["machine_id_sha256"])
    _require_sha256("GPU inventory challenge nonce", receipt["challenge_nonce_sha256"])
    commands = _strict_object(
        "GPU inventory commands",
        receipt["commands"],
        frozenset({"gpu", "processes", "topology"}),
    )
    for name, command in commands.items():
        row = _strict_object(
            f"GPU inventory {name} command",
            command,
            frozenset({"argv", "stdout"}),
        )
        argv = _strict_list(f"GPU inventory {name} argv", row["argv"])
        if not argv or any(type(item) is not str or not item for item in argv):
            raise ValueError("GPU inventory command argv is invalid")
        if type(row["stdout"]) is not str:
            raise TypeError("GPU inventory command stdout must be text")
    if (
        type(receipt["parsed_topology"]) is not dict
        or type(receipt["pci_locality"]) is not list
    ):
        raise TypeError("GPU inventory parsed topology/locality is invalid")
    return receipt


def _validate_provider_receipt(
    value: object,
    *,
    inventory: GpuInventory,
    budget_inventory: BudgetInventoryIdentity,
    collection_nonce_sha256: str,
) -> dict[str, Any]:
    receipt = _strict_object("provider quota receipt", value, _PROVIDER_FIELDS)
    if receipt["schema_version"] != 1 or receipt["kind"] != (
        "industrial_provider_quota_receipt"
    ):
        raise ValueError("provider quota receipt schema is unsupported")
    if (
        receipt["budget_inventory_sha256"] != budget_inventory.sha256
        or receipt["gpu_inventory_sha256"] != inventory.sha256
        or receipt["inventory_source_receipt_sha256"] != inventory.source_receipt_sha256
        or receipt["collection_nonce_sha256"] != collection_nonce_sha256
    ):
        raise ValueError("provider quota receipt identity mismatch")
    _require_sha256("provider scope", receipt["provider_scope_sha256"])
    _nonnegative_int("provider capture time", receipt["captured_at_ns"])
    total = _nonnegative_int("provider total quota", receipt["total_quota_gpu_ms"])
    consumed = _nonnegative_int("provider consumed quota", receipt["consumed_gpu_ms"])
    available = _nonnegative_int(
        "provider available quota", receipt["available_gpu_ms"]
    )
    if consumed > total or available != total - consumed:
        raise ValueError("provider quota arithmetic is inconsistent")
    return receipt


def _validate_host_receipt(
    value: object,
    *,
    inventory: GpuInventory,
    budget_inventory: BudgetInventoryIdentity,
    collection_nonce_sha256: str,
) -> dict[str, Any]:
    receipt = _strict_object("host capacity receipt", value, _HOST_FIELDS)
    if receipt["schema_version"] != 1 or receipt["kind"] != (
        "industrial_host_capacity_receipt"
    ):
        raise ValueError("host capacity receipt schema is unsupported")
    if (
        receipt["budget_inventory_sha256"] != budget_inventory.sha256
        or receipt["gpu_inventory_sha256"] != inventory.sha256
        or receipt["inventory_source_receipt_sha256"] != inventory.source_receipt_sha256
        or receipt["host_sha256"] != budget_inventory.host_sha256
        or receipt["collection_nonce_sha256"] != collection_nonce_sha256
    ):
        raise ValueError("host capacity receipt identity mismatch")
    _require_sha256("host filesystem identity", receipt["filesystem_sha256"])
    _nonnegative_int("host capture time", receipt["captured_at_ns"])
    _nonnegative_int("host free bytes", receipt["host_free_bytes"])
    _nonnegative_int("host quota bytes", receipt["host_quota_bytes"])
    return receipt


def _validate_sizing_receipt(
    value: object,
    *,
    registry_sha256: str,
    budget_inventory_sha256: str,
) -> tuple[CellCapacityRequirement, tuple[str, ...]]:
    receipt = _strict_object("cell capacity sizing receipt", value, _SIZING_FIELDS)
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "industrial_cell_capacity_sizing_receipt"
        or receipt["registry_sha256"] != registry_sha256
        or receipt["budget_inventory_sha256"] != budget_inventory_sha256
        or receipt["sizing_protocol_sha256"] != CELL_CAPACITY_SIZING_PROTOCOL_SHA256
    ):
        raise ValueError("cell capacity sizing receipt identity mismatch")
    cell_id = _require_sha256("capacity sizing cell", receipt["cell_id"])
    requirements = (
        (
            "maximum_evidence_bytes",
            "industrial_evidence_capacity_provenance",
            "evidence_contract_source",
        ),
        (
            "model_staging_bytes",
            "industrial_model_staging_capacity_provenance",
            "model_staging_source",
        ),
        (
            "compile_overlay_bytes",
            "industrial_compile_overlay_capacity_provenance",
            "compile_overlay_source",
        ),
    )
    values: dict[str, int] = {}
    paths: list[str] = []
    for field, kind, source_name in requirements:
        amount = _nonnegative_int(f"cell sizing {field}", receipt[field])
        source = _binding_from_dict(
            receipt[source_name], label=f"cell sizing {source_name} binding"
        )
        provenance = _strict_object(
            f"cell sizing {source_name}",
            load_capacity_raw_json(source),
            _PROVENANCE_FIELDS,
        )
        if (
            provenance["schema_version"] != 1
            or provenance["kind"] != kind
            or provenance["cell_id"] != cell_id
            or provenance["maximum_bytes"] != amount
        ):
            raise ValueError("cell sizing receipt differs from raw provenance")
        _require_sha256("cell sizing derivation", provenance["derivation_sha256"])
        values[field] = amount
        paths.append(source.path)
    return (
        CellCapacityRequirement(
            cell_id=cell_id,
            maximum_evidence_bytes=values["maximum_evidence_bytes"],
            model_staging_bytes=values["model_staging_bytes"],
            compile_overlay_bytes=values["compile_overlay_bytes"],
        ),
        tuple(paths),
    )


def _challenge_from_dict(value: object) -> AttestationChallenge:
    row = _strict_object("capacity verifier challenge", value, _CHALLENGE_FIELDS)
    challenge = AttestationChallenge(**row)
    challenge.validate()
    return challenge


def _attestation_from_dict(value: object) -> SignedAttestation:
    row = _strict_object("capacity verifier attestation", value, _ATTESTATION_FIELDS)
    attestation = SignedAttestation(**row)
    attestation.validate()
    return attestation


def _validate_verification_receipt(
    value: object,
    *,
    sources: _ValidatedCapacitySources,
) -> tuple[AttestationChallenge, SignedAttestation]:
    receipt = _strict_object(
        "capacity verification receipt", value, _VERIFICATION_RECEIPT_FIELDS
    )
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "industrial_capacity_verification_receipt"
        or receipt["authority_protocol_sha256"] != CAPACITY_AUTHORITY_PROTOCOL_SHA256
        or receipt["source_manifest_sha256"] != sources.manifest_sha256
    ):
        raise ValueError("capacity verification receipt identity mismatch")
    payload = _strict_object(
        "capacity verification payload",
        receipt["verification_payload"],
        _VERIFICATION_PAYLOAD_FIELDS,
    )
    if payload != sources.verification_payload():
        raise ValueError("capacity verification payload differs from raw replay")
    challenge = _challenge_from_dict(receipt["challenge"])
    attestation = _attestation_from_dict(receipt["attestation"])
    payload_sha256 = content_sha256(payload)
    if (
        challenge.subject_sha256 != sources.manifest_sha256
        or attestation.challenge_sha256 != challenge.sha256
        or attestation.payload_sha256 != payload_sha256
    ):
        raise ValueError("capacity verifier signature binding is inconsistent")
    if (
        challenge.issued_ns < sources.captured_at_ns
        or challenge.issued_ns - sources.captured_at_ns > CAPACITY_MAXIMUM_SOURCE_AGE_NS
        or challenge.expires_ns - challenge.issued_ns > CAPACITY_MAXIMUM_SOURCE_AGE_NS
    ):
        raise ValueError("capacity sources or verifier challenge are stale")
    return challenge, attestation


def bind_capacity_authority(
    source_manifest_path: str | Path,
    verification_receipt_path: str | Path,
) -> CapacityAuthorityBinding:
    """Bind and structurally replay raw capacity files without promoting them."""

    manifest_binding = bind_capacity_raw_json(source_manifest_path)
    receipt_binding = bind_capacity_raw_json(verification_receipt_path)
    sources = _validate_capacity_sources(load_capacity_raw_json(manifest_binding))
    _validate_verification_receipt(
        load_capacity_raw_json(receipt_binding), sources=sources
    )
    if manifest_binding.path in sources.raw_paths or receipt_binding.path in (
        *sources.raw_paths,
        manifest_binding.path,
    ):
        raise ValueError("capacity manifest/receipt aliases a raw source")
    return CapacityAuthorityBinding(
        schema_version=1,
        source_manifest=manifest_binding,
        verification_receipt=receipt_binding,
        registry_sha256=sources.registry_sha256,
        budget_inventory_sha256=sources.budget_inventory.sha256,
        capacity_envelope_sha256=sources.envelope.sha256,
        gpu_inventory_sha256=sources.gpu_inventory.sha256,
        inventory_source_receipt_sha256=(sources.inventory_source_receipt_sha256),
        trusted_verifier_policy_sha256=RELEASE_TRUSTED_ATTESTER_POLICY.sha256,
        authority_protocol_sha256=CAPACITY_AUTHORITY_PROTOCOL_SHA256,
    )


def revalidate_capacity_authority_binding(
    binding: CapacityAuthorityBinding,
    *,
    expected_registry_sha256: str,
    expected_inventory: BudgetInventoryIdentity,
    expected_envelope: CapacityEnvelope,
) -> CapacityAuthorityResult:
    """Reopen every raw source and require the source-owned release verifier."""

    if type(binding) is not CapacityAuthorityBinding:
        raise TypeError("formal capacity authority requires an exact binding")
    if type(expected_inventory) is not BudgetInventoryIdentity:
        raise TypeError("formal capacity authority requires an exact inventory")
    if type(expected_envelope) is not CapacityEnvelope:
        raise TypeError("formal capacity authority requires an exact envelope")
    _require_sha256("expected capacity registry", expected_registry_sha256)
    policy = require_release_trusted_attester_policy(RELEASE_TRUSTED_ATTESTER_POLICY)
    if binding.trusted_verifier_policy_sha256 != policy.sha256:
        raise ValueError("capacity binding names another verifier policy")
    sources = _validate_capacity_sources(
        load_capacity_raw_json(binding.source_manifest)
    )
    receipt_value = load_capacity_raw_json(binding.verification_receipt)
    challenge, attestation = _validate_verification_receipt(
        receipt_value, sources=sources
    )
    if (
        binding.registry_sha256 != sources.registry_sha256
        or binding.budget_inventory_sha256 != sources.budget_inventory.sha256
        or binding.capacity_envelope_sha256 != sources.envelope.sha256
        or binding.gpu_inventory_sha256 != sources.gpu_inventory.sha256
        or binding.inventory_source_receipt_sha256
        != sources.inventory_source_receipt_sha256
        or expected_registry_sha256 != sources.registry_sha256
        or expected_inventory != sources.budget_inventory
        or expected_envelope != sources.envelope
    ):
        raise ValueError("capacity authority differs from the exact budget plan")
    now_ns = time.time_ns()
    challenge.validate(now_ns=now_ns)
    if (
        now_ns < sources.captured_at_ns
        or now_ns - sources.captured_at_ns > CAPACITY_MAXIMUM_SOURCE_AGE_NS
    ):
        raise ValueError("capacity provider/host observations are stale")
    if not policy.release_ready:
        raise CapacityAuthorityUnavailableError()
    policy.verify_release(
        challenge,
        attestation,
        payload_sha256=content_sha256(sources.verification_payload()),
        now_ns=now_ns,
    )
    return CapacityAuthorityResult(
        capacity_envelope=sources.envelope,
        budget_inventory=sources.budget_inventory,
        gpu_inventory=sources.gpu_inventory,
        registry_sha256=sources.registry_sha256,
        source_manifest_sha256=sources.manifest_sha256,
        verification_receipt_sha256=binding.verification_receipt.semantic_sha256,
        provider_quota_receipt_sha256=sources.provider_receipt_sha256,
        host_capacity_receipt_sha256=sources.host_receipt_sha256,
        cell_sizing_receipt_sha256s=sources.sizing_receipt_sha256s,
    )


@dataclass(frozen=True)
class CapacityAuthority:
    """Convenience owner for a durable raw capacity binding."""

    binding: CapacityAuthorityBinding

    def __post_init__(self) -> None:
        if type(self.binding) is not CapacityAuthorityBinding:
            raise TypeError("capacity authority requires an exact binding")

    @classmethod
    def from_paths(
        cls,
        source_manifest_path: str | Path,
        verification_receipt_path: str | Path,
    ) -> CapacityAuthority:
        return cls(
            bind_capacity_authority(
                source_manifest_path,
                verification_receipt_path,
            )
        )

    @property
    def sha256(self) -> str:
        return self.binding.sha256

    def revalidate(
        self,
        *,
        registry_sha256: str,
        inventory: BudgetInventoryIdentity,
        envelope: CapacityEnvelope,
    ) -> CapacityAuthorityResult:
        return revalidate_capacity_authority_binding(
            self.binding,
            expected_registry_sha256=registry_sha256,
            expected_inventory=inventory,
            expected_envelope=envelope,
        )


def capacity_verification_receipt_template(
    *,
    source_manifest: object,
    challenge: AttestationChallenge,
    attestation: SignedAttestation,
) -> dict[str, object]:
    """Assemble a receipt from an out-of-band signature and replay it strictly."""

    sources = _validate_capacity_sources(source_manifest)
    value: dict[str, object] = {
        "schema_version": 1,
        "kind": "industrial_capacity_verification_receipt",
        "authority_protocol_sha256": CAPACITY_AUTHORITY_PROTOCOL_SHA256,
        "source_manifest_sha256": sources.manifest_sha256,
        "verification_payload": sources.verification_payload(),
        "challenge": asdict(challenge),
        "attestation": asdict(attestation),
    }
    _validate_verification_receipt(value, sources=sources)
    return value


__all__ = [
    "TRUSTED_CAPACITY_VERIFIER_UNAVAILABLE_REASON",
    "CapacityAuthority",
    "CapacityAuthorityResult",
    "CapacityAuthorityUnavailableError",
    "bind_capacity_authority",
    "bind_capacity_raw_json",
    "build_capacity_source_manifest",
    "build_capacity_verification_payload",
    "capacity_source_receipt_sha256_from_paths",
    "capacity_verification_receipt_template",
    "load_capacity_raw_json",
    "revalidate_capacity_authority_binding",
]
