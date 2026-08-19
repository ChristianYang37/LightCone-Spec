"""Source-owned proof replay for signable formal GPU-hour envelopes.

An envelope is a compact numeric projection and is not scientific authority on
its own.  This module binds the exact source manifest, protocol/runtime roots,
inventory, compact registry layer, materialization transport, and (for
preflight) proof-derived coverage graph needed to reproduce the envelope.
Only the reproduced envelope may enter the offline signer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry import (
    formal_runtime_authority_manifest_from_dict,
    protocol_lock_from_dict,
    stage_gpu_hour_envelope_from_dict,
)
from lightcone_spec.experiments.formal_registry_layers import (
    FORMAL_REGISTRY_LAYER_ARTIFACT_KIND,
    load_formal_registry_verification_receipt_path,
)
from lightcone_spec.experiments.formal_stage_coverage import (
    rebuild_formal_stage_bound_materialization,
)
from lightcone_spec.experiments.gpu_hour_authority import (
    ProspectiveGpuHourSourceManifest,
    revalidate_persisted_prospective_gpu_hour_source_manifest,
    revalidate_persisted_stage_gpu_hour_source_manifest,
    revalidate_persisted_staged_prospective_gpu_hour_source_manifest,
    verify_registered_prospective_gpu_hour_authority,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.stage_materialization import (
    StageGpuHourEnvelope,
    StageMaterializationReceipt,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_STAGE_GPU_HOUR_ENVELOPE_PROOF_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_formal_stage_gpu_hour_envelope_proof_protocol",
        "source_union": (
            "lifecycle_actual",
            "preflight_fail_closed_until_admission_bound_qualification_schedule",
            "staged_prospective",
            "downstream_prospective",
        ),
        "registry": "proof_carrying_schema5_layer_only",
        "materialization": "direct_or_bounded_shard_index_deep_rebuild",
        "envelope": "exact_reducer_output_byte_compare",
        "caller_numeric_authority": "forbidden",
    }
)

FormalGpuHourEnvelopeProofSourceKind = Literal[
    "lifecycle_actual",
    "preflight",
    "staged_prospective",
    "downstream_prospective",
]


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _strict(value: object, fields: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _stable_open(binding: CanonicalJsonProofBinding) -> object:
    if type(binding) is not CanonicalJsonProofBinding:
        raise TypeError("formal GPU-hour proof source is not path-bound")
    before = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if before != binding:
        raise ValueError("formal GPU-hour proof source binding changed")
    value = binding.reopen()
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != before:
        raise RuntimeError("formal GPU-hour proof source changed while reopened")
    return value


def _optional_binding(value: object) -> CanonicalJsonProofBinding | None:
    return None if value is None else CanonicalJsonProofBinding.from_dict(value)


@dataclass(frozen=True)
class FormalStageGpuHourEnvelopeProofArtifact:
    """Compact, portable graph that deterministically reproduces an envelope."""

    schema_version: Literal[1]
    kind: Literal["formal_stage_gpu_hour_envelope_proof_artifact"]
    protocol_sha256: str
    source_kind: FormalGpuHourEnvelopeProofSourceKind
    verified_ns: int
    protocol_lock_sha256: str
    runtime_authority_sha256: str
    registry_receipt_sha256: str
    inventory_sha256: str
    final_materialization_receipt_sha256: str
    pilot_materialization_receipt_sha256: str | None
    envelope_sha256: str
    protocol_lock_source: CanonicalJsonProofBinding
    runtime_authority_source: CanonicalJsonProofBinding
    registry_layer_source: CanonicalJsonProofBinding
    inventory_source: CanonicalJsonProofBinding
    final_materialization_source: CanonicalJsonProofBinding
    pilot_materialization_source: CanonicalJsonProofBinding | None
    gpu_hour_source_manifest: CanonicalJsonProofBinding
    envelope_source: CanonicalJsonProofBinding
    preflight_coverage_proof_source: CanonicalJsonProofBinding | None

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_stage_gpu_hour_envelope_proof_artifact"
            or self.protocol_sha256
            != FORMAL_STAGE_GPU_HOUR_ENVELOPE_PROOF_PROTOCOL_SHA256
            or self.source_kind
            not in {
                "lifecycle_actual",
                "preflight",
                "staged_prospective",
                "downstream_prospective",
            }
        ):
            raise ValueError("formal GPU-hour envelope proof schema differs")
        if type(self.verified_ns) is not int or self.verified_ns < 1:
            raise ValueError("formal GPU-hour proof time must be positive")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime authority", self.runtime_authority_sha256),
            ("registry receipt", self.registry_receipt_sha256),
            ("inventory", self.inventory_sha256),
            ("final materialization", self.final_materialization_receipt_sha256),
            ("envelope", self.envelope_sha256),
        ):
            _sha256(f"formal GPU-hour proof {label}", digest)
        if self.pilot_materialization_receipt_sha256 is not None:
            _sha256(
                "formal GPU-hour proof pilot materialization",
                self.pilot_materialization_receipt_sha256,
            )
        required = (
            self.protocol_lock_source,
            self.runtime_authority_source,
            self.registry_layer_source,
            self.inventory_source,
            self.final_materialization_source,
            self.gpu_hour_source_manifest,
            self.envelope_source,
        )
        for binding in required:
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("formal GPU-hour proof source is not path-bound")
            binding.__post_init__()
        for binding in (
            self.pilot_materialization_source,
            self.preflight_coverage_proof_source,
        ):
            if binding is not None and type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("formal GPU-hour optional source is not path-bound")
        if (
            (self.source_kind == "downstream_prospective")
            != (self.pilot_materialization_source is not None)
            or (self.source_kind == "downstream_prospective")
            != (self.pilot_materialization_receipt_sha256 is not None)
            or (self.source_kind == "preflight")
            != (self.preflight_coverage_proof_source is not None)
        ):
            raise ValueError("formal GPU-hour proof source union differs")
        bindings = (
            *required,
            *(
                ()
                if self.pilot_materialization_source is None
                else (self.pilot_materialization_source,)
            ),
            *(
                ()
                if self.preflight_coverage_proof_source is None
                else (self.preflight_coverage_proof_source,)
            ),
        )
        paths = tuple(row.absolute_path for row in bindings)
        if len(paths) != len(set(paths)):
            raise ValueError("formal GPU-hour proof aliases direct sources")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        def optional(
            value: CanonicalJsonProofBinding | None,
        ) -> dict[str, object] | None:
            return None if value is None else value.to_dict()

        value: dict[str, object] = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field
            not in {
                "protocol_lock_source",
                "runtime_authority_source",
                "registry_layer_source",
                "inventory_source",
                "final_materialization_source",
                "pilot_materialization_source",
                "gpu_hour_source_manifest",
                "envelope_source",
                "preflight_coverage_proof_source",
            }
        }
        for field in (
            "protocol_lock_source",
            "runtime_authority_source",
            "registry_layer_source",
            "inventory_source",
            "final_materialization_source",
            "gpu_hour_source_manifest",
            "envelope_source",
        ):
            value[field] = getattr(self, field).to_dict()
        value["pilot_materialization_source"] = optional(
            self.pilot_materialization_source
        )
        value["preflight_coverage_proof_source"] = optional(
            self.preflight_coverage_proof_source
        )
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            value,
            {*cls.__dataclass_fields__, "artifact_sha256"},
            label="formal GPU-hour envelope proof",
        )
        declared = _sha256("formal GPU-hour envelope proof", row.pop("artifact_sha256"))
        for field in (
            "protocol_lock_source",
            "runtime_authority_source",
            "registry_layer_source",
            "inventory_source",
            "final_materialization_source",
            "gpu_hour_source_manifest",
            "envelope_source",
        ):
            row[field] = CanonicalJsonProofBinding.from_dict(row[field])
        row["pilot_materialization_source"] = _optional_binding(
            row["pilot_materialization_source"]
        )
        row["preflight_coverage_proof_source"] = _optional_binding(
            row["preflight_coverage_proof_source"]
        )
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("formal GPU-hour envelope proof digest differs")
        return artifact


def _source_kind(value: object) -> FormalGpuHourEnvelopeProofSourceKind:
    if type(value) is not dict:
        raise TypeError("formal GPU-hour source manifest must be an object")
    kind = value.get("kind")
    try:
        return {
            "lifecycle_gpu_hour_source_manifest": "lifecycle_actual",
            "preflight_gpu_hour_source_manifest": "preflight",
            "staged_prospective_gpu_hour_source_manifest": "staged_prospective",
            "prospective_gpu_hour_source_manifest": "downstream_prospective",
        }[kind]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "formal GPU-hour source manifest kind is unsupported"
        ) from error


def _require_registered_final_materialization(
    registry_receipt: object,
    materialization: StageMaterializationReceipt,
) -> None:
    rows = tuple(
        row.payload
        for row in registry_receipt.cumulative_signed_materializations
        if row.payload.sha256 == materialization.sha256
    )
    if rows != (materialization,):
        raise ValueError(
            "formal GPU-hour final materialization lacks exact registry lineage"
        )


def _require_source_kind_for_stage(
    source_kind: FormalGpuHourEnvelopeProofSourceKind,
    stage: str,
) -> None:
    allowed = {
        "preflight": frozenset({"preflight"}),
        "E3a": frozenset({"lifecycle_actual", "staged_prospective"}),
        "TTS-Cal": frozenset({"lifecycle_actual", "staged_prospective"}),
        "E1": frozenset({"lifecycle_actual", "staged_prospective"}),
        "E2": frozenset({"lifecycle_actual", "staged_prospective"}),
        "E4": frozenset({"lifecycle_actual", "staged_prospective"}),
        "E1a": frozenset({"lifecycle_actual", "staged_prospective"}),
        "E3b": frozenset({"downstream_prospective"}),
        "E5": frozenset({"downstream_prospective"}),
        "E6": frozenset({"downstream_prospective"}),
        "E0": frozenset({"downstream_prospective"}),
    }.get(stage)
    if allowed is None or source_kind not in allowed:
        raise ValueError("formal GPU-hour proof source kind differs from stage")


def _rebuild(
    artifact: FormalStageGpuHourEnvelopeProofArtifact,
    *,
    now_ns: int,
) -> StageGpuHourEnvelope:
    if type(now_ns) is not int or now_ns < artifact.verified_ns:
        raise ValueError("formal GPU-hour proof replay predates publication")
    for binding in (
        artifact.protocol_lock_source,
        artifact.runtime_authority_source,
        artifact.registry_layer_source,
        artifact.inventory_source,
        artifact.final_materialization_source,
        artifact.gpu_hour_source_manifest,
        artifact.envelope_source,
        *(
            ()
            if artifact.pilot_materialization_source is None
            else (artifact.pilot_materialization_source,)
        ),
        *(
            ()
            if artifact.preflight_coverage_proof_source is None
            else (artifact.preflight_coverage_proof_source,)
        ),
    ):
        _stable_open(binding)
    registry_layer = _stable_open(artifact.registry_layer_source)
    if (
        type(registry_layer) is not dict
        or registry_layer.get("kind") != FORMAL_REGISTRY_LAYER_ARTIFACT_KIND
    ):
        raise ValueError("formal GPU-hour proof requires a schema5 registry layer")
    registry_receipt = load_formal_registry_verification_receipt_path(
        artifact.registry_layer_source.absolute_path,
        now_ns=now_ns,
    )
    protocol_lock = protocol_lock_from_dict(_stable_open(artifact.protocol_lock_source))
    runtime = formal_runtime_authority_manifest_from_dict(
        _stable_open(artifact.runtime_authority_source)
    )
    inventory = GpuInventory.from_dict(_stable_open(artifact.inventory_source))
    envelope = stage_gpu_hour_envelope_from_dict(_stable_open(artifact.envelope_source))
    final_materialization = rebuild_formal_stage_bound_materialization(
        artifact.final_materialization_source,
        expected_receipt_sha256=artifact.final_materialization_receipt_sha256,
    )
    if (
        registry_receipt.sha256 != artifact.registry_receipt_sha256
        or registry_receipt.signed_protocol_lock.payload != protocol_lock
        or runtime.sha256 != artifact.runtime_authority_sha256
        or runtime.sha256 != protocol_lock.formal_runtime_authority_manifest_sha256
        or inventory.sha256 != artifact.inventory_sha256
        or inventory.sha256 != registry_receipt.inventory_sha256
        or final_materialization.protocol_lock_sha256 != protocol_lock.sha256
        or envelope.sha256 != artifact.envelope_sha256
        or envelope.materialization_receipt_sha256 != final_materialization.sha256
    ):
        raise ValueError("formal GPU-hour proof immutable lineage differs")
    _require_registered_final_materialization(registry_receipt, final_materialization)
    _require_source_kind_for_stage(artifact.source_kind, final_materialization.stage)
    if artifact.source_kind == "lifecycle_actual":
        revalidate_persisted_stage_gpu_hour_source_manifest(
            artifact.gpu_hour_source_manifest.absolute_path,
            envelope=envelope,
            protocol_lock=protocol_lock,
            formal_runtime_authority_manifest=runtime,
            materialization=final_materialization,
            inventory=inventory,
            now_ns=now_ns,
        )
    elif artifact.source_kind == "staged_prospective":
        revalidate_persisted_staged_prospective_gpu_hour_source_manifest(
            artifact.gpu_hour_source_manifest.absolute_path,
            envelope=envelope,
            protocol_lock=protocol_lock,
            formal_runtime_authority_manifest=runtime,
            materialization=final_materialization,
            inventory=inventory,
            now_ns=now_ns,
        )
    elif artifact.source_kind == "preflight":
        # Schema-1 preflight GPU-hour sources account only the scientific
        # 1+1+8 rows.  They omit the mandatory native/distributed qualification
        # processes and do not bind their caller timeout to launch admission.
        # Never promote that undercount into an offline-signable envelope.
        raise ValueError(
            "preflight GPU-hour proof lacks admission-bound qualification costs"
        )
    elif artifact.source_kind == "downstream_prospective":
        if (
            artifact.pilot_materialization_source is None
            or artifact.pilot_materialization_receipt_sha256 is None
        ):  # pragma: no cover - constructor closes
            raise AssertionError("prospective GPU-hour proof lost pilots")
        pilot_materialization = rebuild_formal_stage_bound_materialization(
            artifact.pilot_materialization_source,
            expected_receipt_sha256=(artifact.pilot_materialization_receipt_sha256),
        )
        authority = verify_registered_prospective_gpu_hour_authority(
            registry_receipt=registry_receipt,
            pilot_materialization=pilot_materialization,
            final_materialization=final_materialization,
            current_ns=now_ns,
        )
        revalidate_persisted_prospective_gpu_hour_source_manifest(
            artifact.gpu_hour_source_manifest.absolute_path,
            envelope=envelope,
            authority=authority,
            protocol_lock=protocol_lock,
            formal_runtime_authority_manifest=runtime,
            pilot_materialization=pilot_materialization,
            final_materialization=final_materialization,
            inventory=inventory,
            now_ns=now_ns,
        )
    else:  # pragma: no cover - source union is closed
        raise AssertionError("unsupported formal GPU-hour proof source")
    return envelope


def bind_formal_stage_gpu_hour_envelope_proof_artifact(
    *,
    protocol_lock_path: str | Path,
    runtime_authority_path: str | Path,
    registry_layer_path: str | Path,
    inventory_path: str | Path,
    final_materialization_path: str | Path,
    gpu_hour_source_manifest_path: str | Path,
    envelope_path: str | Path,
    now_ns: int,
    pilot_materialization_path: str | Path | None = None,
    preflight_coverage_proof_path: str | Path | None = None,
) -> FormalStageGpuHourEnvelopeProofArtifact:
    """Bind and deep-replay existing reducer outputs before publication."""

    protocol_source = CanonicalJsonProofBinding.bind(protocol_lock_path)
    runtime_source = CanonicalJsonProofBinding.bind(runtime_authority_path)
    registry_source = CanonicalJsonProofBinding.bind(registry_layer_path)
    inventory_source = CanonicalJsonProofBinding.bind(inventory_path)
    final_source = CanonicalJsonProofBinding.bind(final_materialization_path)
    source = CanonicalJsonProofBinding.bind(gpu_hour_source_manifest_path)
    envelope_source = CanonicalJsonProofBinding.bind(envelope_path)
    source_kind = _source_kind(_stable_open(source))
    envelope = stage_gpu_hour_envelope_from_dict(_stable_open(envelope_source))
    final_materialization = rebuild_formal_stage_bound_materialization(
        final_source,
        expected_receipt_sha256=envelope.materialization_receipt_sha256,
    )
    pilot_source = (
        None
        if pilot_materialization_path is None
        else CanonicalJsonProofBinding.bind(pilot_materialization_path)
    )
    pilot_sha256: str | None = None
    if source_kind == "downstream_prospective":
        prospective = ProspectiveGpuHourSourceManifest.from_dict(_stable_open(source))
        if pilot_source is None:
            raise ValueError(
                "prospective GPU-hour proof requires pilot materialization"
            )
        pilot = rebuild_formal_stage_bound_materialization(
            pilot_source,
            expected_receipt_sha256=prospective.pilot_materialization_receipt_sha256,
        )
        pilot_sha256 = pilot.sha256
    elif pilot_source is not None:
        raise ValueError("non-prospective GPU-hour proof cannot bind pilots")
    coverage_source = (
        None
        if preflight_coverage_proof_path is None
        else CanonicalJsonProofBinding.bind(preflight_coverage_proof_path)
    )
    protocol_lock = protocol_lock_from_dict(_stable_open(protocol_source))
    runtime = formal_runtime_authority_manifest_from_dict(_stable_open(runtime_source))
    inventory = GpuInventory.from_dict(_stable_open(inventory_source))
    registry_receipt = load_formal_registry_verification_receipt_path(
        registry_source.absolute_path,
        now_ns=now_ns,
    )
    artifact = FormalStageGpuHourEnvelopeProofArtifact(
        schema_version=1,
        kind="formal_stage_gpu_hour_envelope_proof_artifact",
        protocol_sha256=FORMAL_STAGE_GPU_HOUR_ENVELOPE_PROOF_PROTOCOL_SHA256,
        source_kind=source_kind,
        verified_ns=now_ns,
        protocol_lock_sha256=protocol_lock.sha256,
        runtime_authority_sha256=runtime.sha256,
        registry_receipt_sha256=registry_receipt.sha256,
        inventory_sha256=inventory.sha256,
        final_materialization_receipt_sha256=final_materialization.sha256,
        pilot_materialization_receipt_sha256=pilot_sha256,
        envelope_sha256=envelope.sha256,
        protocol_lock_source=protocol_source,
        runtime_authority_source=runtime_source,
        registry_layer_source=registry_source,
        inventory_source=inventory_source,
        final_materialization_source=final_source,
        pilot_materialization_source=pilot_source,
        gpu_hour_source_manifest=source,
        envelope_source=envelope_source,
        preflight_coverage_proof_source=coverage_source,
    )
    if _rebuild(artifact, now_ns=now_ns) != envelope:
        raise RuntimeError("formal GPU-hour proof binding changed reducer output")
    return artifact


def publish_formal_stage_gpu_hour_envelope_proof_artifact(
    artifact: FormalStageGpuHourEnvelopeProofArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not FormalStageGpuHourEnvelopeProofArtifact:
        raise TypeError("formal GPU-hour proof publisher requires exact artifact")
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    rebuilt = revalidate_formal_stage_gpu_hour_envelope_proof_artifact(
        binding.absolute_path,
        now_ns=artifact.verified_ns,
    )
    if rebuilt.sha256 != artifact.envelope_sha256:
        raise RuntimeError("published formal GPU-hour proof changed")
    return binding


def revalidate_formal_stage_gpu_hour_envelope_proof_artifact(
    artifact_path: str | Path,
    *,
    now_ns: int,
    relocatable_bundle_manifest_path: str | Path | None = None,
) -> StageGpuHourEnvelope:
    """Deep-replay one local or stable-pulled GPU-hour proof graph."""

    if relocatable_bundle_manifest_path is not None:
        from lightcone_spec.runtime.relocatable_evidence import (
            activate_relocatable_evidence_bundle,
        )

        with activate_relocatable_evidence_bundle(
            relocatable_bundle_manifest_path
        ) as bundle:
            entry = str(artifact_path)
            if entry not in bundle.artifact.entry_remote_absolute_paths:
                raise ValueError("formal GPU-hour proof is not a pulled-evidence entry")
            return revalidate_formal_stage_gpu_hour_envelope_proof_artifact(
                entry,
                now_ns=now_ns,
            )
    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = FormalStageGpuHourEnvelopeProofArtifact.from_dict(binding.reopen())
    envelope = _rebuild(artifact, now_ns=now_ns)
    if CanonicalJsonProofBinding.bind(artifact_path) != binding:
        raise RuntimeError("formal GPU-hour proof changed while replayed")
    return envelope


__all__ = (
    "FORMAL_STAGE_GPU_HOUR_ENVELOPE_PROOF_PROTOCOL_SHA256",
    "FormalGpuHourEnvelopeProofSourceKind",
    "FormalStageGpuHourEnvelopeProofArtifact",
    "bind_formal_stage_gpu_hour_envelope_proof_artifact",
    "publish_formal_stage_gpu_hour_envelope_proof_artifact",
    "revalidate_formal_stage_gpu_hour_envelope_proof_artifact",
)
