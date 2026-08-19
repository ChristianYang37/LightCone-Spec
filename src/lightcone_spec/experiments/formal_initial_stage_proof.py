"""Source-owned replay proofs for initial formal-stage materializations.

The first dynamic cells do not have a prior :mod:`formal_stage_prefix` node.
They are nevertheless deterministic outputs of the public materializers.  A
small append-only proof therefore binds the exact schema-3 registry layer that
precedes the stage and, only for TTS-Cal, the source-owned numerical authority.
The signer reopens this graph and reruns the materializer; caller-authored cell
rows are never accepted as authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.formal_method_authority import (
    load_tts_calibration_authority_artifact,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry import (
    FormalRegistryVerificationReceipt,
)
from lightcone_spec.experiments.formal_registry_layers import (
    FORMAL_REGISTRY_LAYER_ARTIFACT_KIND,
    load_formal_registry_verification_receipt_path,
)
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    StageCoverageReceipt,
    StageMaterializationReceipt,
    materialize_e1_first_slice,
    materialize_e3a,
    materialize_preflight,
    materialize_tts_calibration,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_INITIAL_STAGE_MATERIALIZATION_PROOF_KIND = (
    "formal_initial_stage_materialization_proof_artifact"
)

FormalInitialStageMaterializationPhase = Literal[
    "preflight",
    "e3a",
    "tts_calibration",
    "e1",
]

_PHASE_STAGE = {
    "preflight": "preflight",
    "e3a": "E3a",
    "tts_calibration": "TTS-Cal",
    "e1": "E1",
}


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


def _stable_open(binding: CanonicalJsonProofBinding, *, label: str) -> object:
    if type(binding) is not CanonicalJsonProofBinding:
        raise TypeError(f"{label} is not path-bound")
    rebound = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if rebound != binding:
        raise ValueError(f"{label} path identity changed")
    value = binding.reopen()
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != rebound:
        raise RuntimeError(f"{label} changed while reopened")
    return value


def _one_stage_payload(
    rows: tuple[object, ...],
    *,
    stage: str,
    label: str,
) -> object:
    matches = tuple(row.payload for row in rows if row.payload.stage == stage)
    if len(matches) != 1:
        raise ValueError(f"initial materialization requires one exact {label}")
    return matches[0]


def _derive_initial_materialization(
    phase: FormalInitialStageMaterializationPhase,
    *,
    registry_receipt: FormalRegistryVerificationReceipt,
    tts_authority_source: CanonicalJsonProofBinding | None,
    now_ns: int,
) -> StageMaterializationReceipt:
    if type(registry_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("initial materialization proof requires durable registry")
    manifest = registry_receipt.revalidate(current_ns=now_ns)
    lock = registry_receipt.signed_protocol_lock.payload
    unmeasured = GpuHourEstimate.unmeasured()
    if phase == "preflight":
        if (
            tts_authority_source is not None
            or registry_receipt.cumulative_signed_materializations
            or registry_receipt.cumulative_signed_coverage
            or manifest.materializations
            or manifest.coverage
            or manifest.source_authorities
        ):
            raise ValueError(
                "preflight materialization proof requires the exact bootstrap root"
            )
        result = materialize_preflight(
            protocol_lock_sha256=lock.sha256,
            gpu_hours=unmeasured,
        )
        if result.expected_cell_count != 10:
            raise AssertionError(
                "preflight public materializer no longer emits 10 cells"
            )
        return result

    if phase == "e3a":
        if tts_authority_source is not None:
            raise ValueError("E3a materialization cannot bind a TTS authority")
        preflight = _one_stage_payload(
            registry_receipt.cumulative_signed_materializations,
            stage="preflight",
            label="signed preflight materialization",
        )
        coverage = _one_stage_payload(
            registry_receipt.cumulative_signed_coverage,
            stage="preflight",
            label="signed preflight coverage",
        )
        assert type(preflight) is StageMaterializationReceipt
        assert type(coverage) is StageCoverageReceipt
        return materialize_e3a(
            registry_verification_receipt=registry_receipt,
            protocol_lock=lock,
            preflight_materialization=preflight,
            preflight_coverage=coverage,
            now_ns=now_ns,
            gpu_hours=unmeasured,
        )

    if phase == "tts_calibration":
        if type(tts_authority_source) is not CanonicalJsonProofBinding:
            raise ValueError(
                "TTS-Cal materialization requires its source-owned authority"
            )
        authority_artifact = load_tts_calibration_authority_artifact(
            tts_authority_source.absolute_path
        )
        e3a = _one_stage_payload(
            registry_receipt.cumulative_signed_materializations,
            stage="E3a",
            label="signed E3a materialization",
        )
        coverage = _one_stage_payload(
            registry_receipt.cumulative_signed_coverage,
            stage="E3a",
            label="signed E3a coverage",
        )
        assert type(e3a) is StageMaterializationReceipt
        assert type(coverage) is StageCoverageReceipt
        return materialize_tts_calibration(
            registry_verification_receipt=registry_receipt,
            protocol_lock=lock,
            tts_calibration_authority=authority_artifact.authority,
            e3a_materialization=e3a,
            e3a_coverage=coverage,
            now_ns=now_ns,
            gpu_hours=unmeasured,
        )

    if phase == "e1":
        if tts_authority_source is not None:
            raise ValueError("E1 materialization cannot bind a second TTS authority")
        e3a = _one_stage_payload(
            registry_receipt.cumulative_signed_materializations,
            stage="E3a",
            label="signed E3a materialization",
        )
        e3a_coverage = _one_stage_payload(
            registry_receipt.cumulative_signed_coverage,
            stage="E3a",
            label="signed E3a coverage",
        )
        tts = _one_stage_payload(
            registry_receipt.cumulative_signed_materializations,
            stage="TTS-Cal",
            label="signed TTS-Cal materialization",
        )
        tts_coverage = _one_stage_payload(
            registry_receipt.cumulative_signed_coverage,
            stage="TTS-Cal",
            label="signed TTS-Cal coverage",
        )
        assert type(e3a) is StageMaterializationReceipt
        assert type(e3a_coverage) is StageCoverageReceipt
        assert type(tts) is StageMaterializationReceipt
        assert type(tts_coverage) is StageCoverageReceipt
        return materialize_e1_first_slice(
            registry_verification_receipt=registry_receipt,
            protocol_lock=lock,
            tts_calibration_materialization=tts,
            tts_calibration_coverage=tts_coverage,
            e3a_materialization=e3a,
            e3a_coverage=e3a_coverage,
            now_ns=now_ns,
            gpu_hours=unmeasured,
        )
    raise ValueError("initial materialization proof phase is unsupported")


@dataclass(frozen=True)
class FormalInitialStageMaterializationProofArtifact:
    """Path-bound proof that replays one initial public materializer."""

    schema_version: Literal[1]
    kind: Literal["formal_initial_stage_materialization_proof_artifact"]
    phase: FormalInitialStageMaterializationPhase
    verified_ns: int
    protocol_lock_sha256: str
    registry_receipt_sha256: str
    expected_materialization_sha256: str
    registry_layer_source: CanonicalJsonProofBinding
    tts_calibration_authority_source: CanonicalJsonProofBinding | None

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != FORMAL_INITIAL_STAGE_MATERIALIZATION_PROOF_KIND
            or self.phase not in _PHASE_STAGE
        ):
            raise ValueError("initial materialization proof schema differs")
        if type(self.verified_ns) is not int or self.verified_ns < 1:
            raise ValueError("initial materialization proof time must be positive")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry receipt", self.registry_receipt_sha256),
            ("materialization", self.expected_materialization_sha256),
        ):
            _sha256(f"initial materialization proof {label}", digest)
        if type(self.registry_layer_source) is not CanonicalJsonProofBinding:
            raise TypeError("initial materialization registry is not path-bound")
        if (self.phase == "tts_calibration") != (
            type(self.tts_calibration_authority_source) is CanonicalJsonProofBinding
        ):
            raise ValueError("initial materialization TTS authority coverage differs")
        if (
            self.tts_calibration_authority_source is not None
            and self.tts_calibration_authority_source.absolute_path
            == self.registry_layer_source.absolute_path
        ):
            raise ValueError("initial materialization proof aliases its sources")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "phase": self.phase,
            "verified_ns": self.verified_ns,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "registry_receipt_sha256": self.registry_receipt_sha256,
            "expected_materialization_sha256": (self.expected_materialization_sha256),
            "registry_layer_source": self.registry_layer_source.to_dict(),
            "tts_calibration_authority_source": (
                None
                if self.tts_calibration_authority_source is None
                else self.tts_calibration_authority_source.to_dict()
            ),
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            value,
            {*cls.__dataclass_fields__, "artifact_sha256"},
            label="initial materialization proof",
        )
        declared = _sha256("initial materialization proof", row.pop("artifact_sha256"))
        row["registry_layer_source"] = CanonicalJsonProofBinding.from_dict(
            row["registry_layer_source"]
        )
        row["tts_calibration_authority_source"] = (
            None
            if row["tts_calibration_authority_source"] is None
            else CanonicalJsonProofBinding.from_dict(
                row["tts_calibration_authority_source"]
            )
        )
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("initial materialization proof digest differs")
        return artifact


def _rebuild(
    artifact: FormalInitialStageMaterializationProofArtifact,
    *,
    now_ns: int,
) -> StageMaterializationReceipt:
    if type(now_ns) is not int or now_ns < artifact.verified_ns:
        raise ValueError("initial materialization replay predates publication")
    raw_registry = _stable_open(
        artifact.registry_layer_source,
        label="initial materialization registry layer",
    )
    if (
        type(raw_registry) is not dict
        or raw_registry.get("kind") != FORMAL_REGISTRY_LAYER_ARTIFACT_KIND
    ):
        raise ValueError(
            "initial materialization proof requires a schema3 registry layer"
        )
    if artifact.tts_calibration_authority_source is not None:
        _stable_open(
            artifact.tts_calibration_authority_source,
            label="initial materialization TTS authority",
        )
    receipt = load_formal_registry_verification_receipt_path(
        artifact.registry_layer_source.absolute_path,
        now_ns=now_ns,
    )
    materialization = _derive_initial_materialization(
        artifact.phase,
        registry_receipt=receipt,
        tts_authority_source=artifact.tts_calibration_authority_source,
        now_ns=now_ns,
    )
    if (
        receipt.sha256 != artifact.registry_receipt_sha256
        or receipt.signed_protocol_lock.payload.sha256 != artifact.protocol_lock_sha256
        or materialization.stage != _PHASE_STAGE[artifact.phase]
        or materialization.expected_cell_count
        != {"preflight": 10, "e3a": 360, "tts_calibration": 288, "e1": 68}[
            artifact.phase
        ]
        or materialization.sha256 != artifact.expected_materialization_sha256
    ):
        raise ValueError("initial materialization proof replay differs")
    return materialization


def bind_formal_initial_stage_materialization_proof_artifact(
    *,
    phase: FormalInitialStageMaterializationPhase,
    registry_layer_path: str | Path,
    now_ns: int,
    tts_calibration_authority_path: str | Path | None = None,
) -> FormalInitialStageMaterializationProofArtifact:
    """Bind the immediate durable prefix and replay its public materializer."""

    registry_source = CanonicalJsonProofBinding.bind(registry_layer_path)
    authority_source = (
        None
        if tts_calibration_authority_path is None
        else CanonicalJsonProofBinding.bind(tts_calibration_authority_path)
    )
    raw_registry = _stable_open(
        registry_source,
        label="initial materialization registry layer",
    )
    if (
        type(raw_registry) is not dict
        or raw_registry.get("kind") != FORMAL_REGISTRY_LAYER_ARTIFACT_KIND
    ):
        raise ValueError(
            "initial materialization proof requires a schema3 registry layer"
        )
    receipt = load_formal_registry_verification_receipt_path(
        registry_source.absolute_path,
        now_ns=now_ns,
    )
    materialization = _derive_initial_materialization(
        phase,
        registry_receipt=receipt,
        tts_authority_source=authority_source,
        now_ns=now_ns,
    )
    artifact = FormalInitialStageMaterializationProofArtifact(
        schema_version=1,
        kind=FORMAL_INITIAL_STAGE_MATERIALIZATION_PROOF_KIND,
        phase=phase,
        verified_ns=now_ns,
        protocol_lock_sha256=receipt.signed_protocol_lock.payload.sha256,
        registry_receipt_sha256=receipt.sha256,
        expected_materialization_sha256=materialization.sha256,
        registry_layer_source=registry_source,
        tts_calibration_authority_source=authority_source,
    )
    if _rebuild(artifact, now_ns=now_ns) != materialization:
        raise RuntimeError("initial materialization proof changed during binding")
    return artifact


def publish_formal_initial_stage_materialization_proof_artifact(
    artifact: FormalInitialStageMaterializationProofArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not FormalInitialStageMaterializationProofArtifact:
        raise TypeError("initial materialization proof publisher requires exact input")
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    rebuilt = revalidate_formal_initial_stage_materialization_proof_artifact(
        binding.absolute_path,
        now_ns=artifact.verified_ns,
    )
    if rebuilt.sha256 != artifact.expected_materialization_sha256:
        raise RuntimeError("published initial materialization proof changed")
    return binding


def revalidate_formal_initial_stage_materialization_proof_artifact(
    artifact_path: str | Path,
    *,
    now_ns: int,
) -> StageMaterializationReceipt:
    """Deep-reopen and replay one initial materialization proof."""

    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = FormalInitialStageMaterializationProofArtifact.from_dict(
        binding.reopen()
    )
    result = _rebuild(artifact, now_ns=now_ns)
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise RuntimeError("initial materialization proof changed while replayed")
    return result


__all__ = (
    "FORMAL_INITIAL_STAGE_MATERIALIZATION_PROOF_KIND",
    "FormalInitialStageMaterializationPhase",
    "FormalInitialStageMaterializationProofArtifact",
    "bind_formal_initial_stage_materialization_proof_artifact",
    "publish_formal_initial_stage_materialization_proof_artifact",
    "revalidate_formal_initial_stage_materialization_proof_artifact",
)
