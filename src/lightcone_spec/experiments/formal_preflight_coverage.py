"""Durable, zero-caller preflight stage-coverage reconstruction.

The preflight finalizer produces several useful typed values, but those values
are in-memory projections.  This module persists only the path-bound inputs and
one reducer-owned output.  A clean verifier can therefore reopen the sealed
dispatch and every raw/control proof, rerun the 1+1+8 reducer, and byte-compare
the exact :class:`StageCoverageReceipt` without trusting caller status fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.formal_dispatch import (
    FormalPreflightDispatchReceipt,
    load_formal_preflight_dispatch_receipt,
)
from lightcone_spec.experiments.formal_preflight_execution import (
    FormalPreflightFinalEvidence,
    finalize_formal_preflight_evidence,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry import (
    stage_coverage_receipt_from_dict,
    stage_coverage_receipt_to_dict,
    tts_l0_candidate_state_coverage_from_dict,
)
from lightcone_spec.experiments.preflight_authority import (
    PREFLIGHT_REQUIRED_QUALIFICATION_SUITES,
    PreflightQualificationProofSource,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_PREFLIGHT_STAGE_COVERAGE_PROOF_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_preflight_stage_coverage_proof_protocol",
        "authority": "sealed_dispatch_plus_deep_reopened_remote_and_local_proofs",
        "reducer": "finalize_formal_preflight_evidence",
        "coverage": "exact_1_compile_1_exactness_8_interference",
        "candidate_state": "exact_path_bound_tts_l0_replay_pair",
        "caller_status_or_digest": "forbidden",
        "portability": "relocatable_evidence_closed_binding_union",
    }
)


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
        raise TypeError("preflight coverage source is not path-bound")
    before = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if before != binding:
        raise ValueError("preflight coverage source binding changed")
    value = binding.reopen()
    after = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if after != before:
        raise RuntimeError("preflight coverage source changed while reopened")
    return value


@dataclass(frozen=True)
class FormalPreflightStageCoverageProofArtifact:
    """Compact path graph for one exact preflight coverage reduction."""

    schema_version: Literal[1]
    kind: Literal["formal_preflight_stage_coverage_proof_artifact"]
    protocol_sha256: str
    verified_ns: int
    protocol_lock_sha256: str
    registry_sha256: str
    runtime_sha256: str
    inventory_sha256: str
    materialization_receipt_sha256: str
    source_authority_sha256: str
    activation_sha256: str
    pointer_coverage_sha256: str
    coverage_receipt_sha256: str
    dispatch_receipt_source: CanonicalJsonProofBinding
    remote_raw_receipt_source: CanonicalJsonProofBinding
    exactness_result_source: CanonicalJsonProofBinding
    interference_proof_source: CanonicalJsonProofBinding
    qualification_proof_sources: tuple[PreflightQualificationProofSource, ...]
    candidate_state_coverage_source: CanonicalJsonProofBinding
    candidate_replay_proof_sources: tuple[
        CanonicalJsonProofBinding, CanonicalJsonProofBinding
    ]
    derived_stage_coverage_source: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_preflight_stage_coverage_proof_artifact"
            or self.protocol_sha256
            != FORMAL_PREFLIGHT_STAGE_COVERAGE_PROOF_PROTOCOL_SHA256
        ):
            raise ValueError("preflight stage coverage proof schema differs")
        if type(self.verified_ns) is not int or self.verified_ns < 0:
            raise ValueError("preflight stage coverage verification time is invalid")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("runtime", self.runtime_sha256),
            ("inventory", self.inventory_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("source authority", self.source_authority_sha256),
            ("activation", self.activation_sha256),
            ("pointer coverage", self.pointer_coverage_sha256),
            ("coverage", self.coverage_receipt_sha256),
        ):
            _sha256(f"preflight coverage proof {label}", digest)
        for binding in (
            self.dispatch_receipt_source,
            self.remote_raw_receipt_source,
            self.exactness_result_source,
            self.interference_proof_source,
            self.candidate_state_coverage_source,
            self.derived_stage_coverage_source,
            *self.candidate_replay_proof_sources,
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("preflight coverage proof source is not path-bound")
            binding.__post_init__()
        if (
            type(self.qualification_proof_sources) is not tuple
            or tuple(row.suite_id for row in self.qualification_proof_sources)
            != PREFLIGHT_REQUIRED_QUALIFICATION_SUITES
            or any(
                type(row) is not PreflightQualificationProofSource
                for row in self.qualification_proof_sources
            )
        ):
            raise ValueError("preflight coverage proof qualification universe differs")
        if (
            type(self.candidate_replay_proof_sources) is not tuple
            or len(self.candidate_replay_proof_sources) != 2
            or len({row.absolute_path for row in self.candidate_replay_proof_sources})
            != 2
        ):
            raise ValueError("preflight coverage proof replay pair differs")
        direct_paths = (
            self.dispatch_receipt_source.absolute_path,
            self.remote_raw_receipt_source.absolute_path,
            self.exactness_result_source.absolute_path,
            self.interference_proof_source.absolute_path,
            self.candidate_state_coverage_source.absolute_path,
            self.derived_stage_coverage_source.absolute_path,
            *(row.absolute_path for row in self.candidate_replay_proof_sources),
            *(
                path
                for row in self.qualification_proof_sources
                for path in (
                    row.result_pointer.absolute_path,
                    row.proof_artifact.absolute_path,
                )
            ),
        )
        if len(direct_paths) != len(set(direct_paths)):
            raise ValueError("preflight coverage proof aliases direct sources")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "verified_ns": self.verified_ns,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "registry_sha256": self.registry_sha256,
            "runtime_sha256": self.runtime_sha256,
            "inventory_sha256": self.inventory_sha256,
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "source_authority_sha256": self.source_authority_sha256,
            "activation_sha256": self.activation_sha256,
            "pointer_coverage_sha256": self.pointer_coverage_sha256,
            "coverage_receipt_sha256": self.coverage_receipt_sha256,
            "dispatch_receipt_source": self.dispatch_receipt_source.to_dict(),
            "remote_raw_receipt_source": self.remote_raw_receipt_source.to_dict(),
            "exactness_result_source": self.exactness_result_source.to_dict(),
            "interference_proof_source": self.interference_proof_source.to_dict(),
            "qualification_proof_sources": [
                row.to_dict() for row in self.qualification_proof_sources
            ],
            "candidate_state_coverage_source": (
                self.candidate_state_coverage_source.to_dict()
            ),
            "candidate_replay_proof_sources": [
                row.to_dict() for row in self.candidate_replay_proof_sources
            ],
            "derived_stage_coverage_source": (
                self.derived_stage_coverage_source.to_dict()
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
            label="preflight stage coverage proof",
        )
        declared = _sha256("preflight stage coverage proof", row.pop("artifact_sha256"))
        for name in (
            "dispatch_receipt_source",
            "remote_raw_receipt_source",
            "exactness_result_source",
            "interference_proof_source",
            "candidate_state_coverage_source",
            "derived_stage_coverage_source",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        raw_qualification = row["qualification_proof_sources"]
        raw_replay = row["candidate_replay_proof_sources"]
        if type(raw_qualification) is not list or type(raw_replay) is not list:
            raise TypeError("preflight coverage proof source arrays differ")
        row["qualification_proof_sources"] = tuple(
            PreflightQualificationProofSource.from_dict(item)
            for item in raw_qualification
        )
        row["candidate_replay_proof_sources"] = tuple(
            CanonicalJsonProofBinding.from_dict(item) for item in raw_replay
        )
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("preflight stage coverage proof digest differs")
        return artifact


def _qualification_paths(
    rows: tuple[PreflightQualificationProofSource, ...],
) -> dict[str, tuple[str, str]]:
    return {
        row.suite_id: (
            row.result_pointer.absolute_path,
            row.proof_artifact.absolute_path,
        )
        for row in rows
    }


def _rebuild(
    artifact: FormalPreflightStageCoverageProofArtifact,
    *,
    now_ns: int,
) -> FormalPreflightFinalEvidence:
    if type(now_ns) is not int or now_ns < artifact.verified_ns:
        raise ValueError("preflight coverage replay predates publication")
    for binding in (
        artifact.dispatch_receipt_source,
        artifact.remote_raw_receipt_source,
        artifact.exactness_result_source,
        artifact.interference_proof_source,
        artifact.candidate_state_coverage_source,
        artifact.derived_stage_coverage_source,
        *artifact.candidate_replay_proof_sources,
    ):
        _stable_open(binding)
    for row in artifact.qualification_proof_sources:
        _stable_open(row.result_pointer)
        _stable_open(row.proof_artifact)

    dispatch = FormalPreflightDispatchReceipt.from_dict(
        _stable_open(artifact.dispatch_receipt_source)
    )
    token = load_formal_preflight_dispatch_receipt(
        artifact.dispatch_receipt_source.absolute_path,
        current_ns=now_ns,
    )
    if dispatch.revalidate(current_ns=now_ns).sha256 != token.sha256:
        raise ValueError("preflight coverage dispatch changed during replay")
    candidate_state_coverage = tts_l0_candidate_state_coverage_from_dict(
        _stable_open(artifact.candidate_state_coverage_source)
    )
    evidence = finalize_formal_preflight_evidence(
        token,
        remote_raw_receipt_path=artifact.remote_raw_receipt_source.absolute_path,
        exactness_result_path=artifact.exactness_result_source.absolute_path,
        interference_proof_artifact_path=(
            artifact.interference_proof_source.absolute_path
        ),
        qualification_proof_paths=_qualification_paths(
            artifact.qualification_proof_sources
        ),
        materialization=dispatch.signed_materialization.payload,
        candidate_state_coverage=candidate_state_coverage,
        candidate_replay_proof_paths=tuple(
            row.absolute_path for row in artifact.candidate_replay_proof_sources
        ),
        now_ns=now_ns,
    )
    derived = stage_coverage_receipt_from_dict(
        _stable_open(artifact.derived_stage_coverage_source)
    )
    if (
        evidence.stage_coverage != derived
        or evidence.stage_coverage.sha256 != artifact.coverage_receipt_sha256
        or evidence.materialization.sha256 != artifact.materialization_receipt_sha256
        or evidence.source_authority.sha256 != artifact.source_authority_sha256
        or evidence.activation.sha256 != artifact.activation_sha256
        or evidence.coverage.sha256 != artifact.pointer_coverage_sha256
        or token.protocol_lock.sha256 != artifact.protocol_lock_sha256
        or token.dispatch_context.registry.sha256 != artifact.registry_sha256
        or evidence.source_authority.runtime_sha256 != artifact.runtime_sha256
        or evidence.source_authority.inventory_sha256 != artifact.inventory_sha256
    ):
        raise ValueError("preflight stage coverage proof replay differs")
    return evidence


def publish_formal_preflight_stage_coverage_proof_artifact(
    *,
    dispatch_receipt_path: str | Path,
    remote_raw_receipt_path: str | Path,
    exactness_result_path: str | Path,
    interference_proof_artifact_path: str | Path,
    qualification_proof_paths: dict[str, tuple[str | Path, str | Path]],
    candidate_state_coverage_path: str | Path,
    candidate_replay_proof_paths: tuple[str | Path, str | Path],
    derived_stage_coverage_output_path: str | Path,
    output_path: str | Path,
    now_ns: int,
) -> CanonicalJsonProofBinding:
    """Deep-reduce raw sources and atomically publish the durable proof graph."""

    if tuple(sorted(qualification_proof_paths)) != (
        PREFLIGHT_REQUIRED_QUALIFICATION_SUITES
    ):
        raise ValueError("preflight coverage publisher requires exact suites")
    qualification = tuple(
        PreflightQualificationProofSource.bind(
            suite_id=suite_id,
            result_pointer_path=qualification_proof_paths[suite_id][0],
            proof_artifact_path=qualification_proof_paths[suite_id][1],
        )
        for suite_id in PREFLIGHT_REQUIRED_QUALIFICATION_SUITES
    )
    replay = tuple(
        CanonicalJsonProofBinding.bind(path) for path in candidate_replay_proof_paths
    )
    if len(replay) != 2:  # pragma: no cover - tuple annotation is not runtime
        raise ValueError("preflight coverage publisher requires exact replay pair")
    dispatch_binding = CanonicalJsonProofBinding.bind(dispatch_receipt_path)
    dispatch = FormalPreflightDispatchReceipt.from_dict(_stable_open(dispatch_binding))
    token = load_formal_preflight_dispatch_receipt(
        dispatch_binding.absolute_path,
        current_ns=now_ns,
    )
    candidate_binding = CanonicalJsonProofBinding.bind(candidate_state_coverage_path)
    evidence = finalize_formal_preflight_evidence(
        token,
        remote_raw_receipt_path=remote_raw_receipt_path,
        exactness_result_path=exactness_result_path,
        interference_proof_artifact_path=interference_proof_artifact_path,
        qualification_proof_paths={
            suite_id: (
                str(qualification_proof_paths[suite_id][0]),
                str(qualification_proof_paths[suite_id][1]),
            )
            for suite_id in PREFLIGHT_REQUIRED_QUALIFICATION_SUITES
        },
        materialization=dispatch.signed_materialization.payload,
        candidate_state_coverage=tts_l0_candidate_state_coverage_from_dict(
            _stable_open(candidate_binding)
        ),
        candidate_replay_proof_paths=tuple(row.absolute_path for row in replay),
        now_ns=now_ns,
    )
    publish_canonical_json_no_replace(
        derived_stage_coverage_output_path,
        stage_coverage_receipt_to_dict(evidence.stage_coverage),
    )
    artifact = FormalPreflightStageCoverageProofArtifact(
        schema_version=1,
        kind="formal_preflight_stage_coverage_proof_artifact",
        protocol_sha256=FORMAL_PREFLIGHT_STAGE_COVERAGE_PROOF_PROTOCOL_SHA256,
        verified_ns=now_ns,
        protocol_lock_sha256=token.protocol_lock.sha256,
        registry_sha256=token.dispatch_context.registry.sha256,
        runtime_sha256=evidence.source_authority.runtime_sha256,
        inventory_sha256=evidence.source_authority.inventory_sha256,
        materialization_receipt_sha256=evidence.materialization.sha256,
        source_authority_sha256=evidence.source_authority.sha256,
        activation_sha256=evidence.activation.sha256,
        pointer_coverage_sha256=evidence.coverage.sha256,
        coverage_receipt_sha256=evidence.stage_coverage.sha256,
        dispatch_receipt_source=dispatch_binding,
        remote_raw_receipt_source=CanonicalJsonProofBinding.bind(
            remote_raw_receipt_path
        ),
        exactness_result_source=CanonicalJsonProofBinding.bind(exactness_result_path),
        interference_proof_source=CanonicalJsonProofBinding.bind(
            interference_proof_artifact_path
        ),
        qualification_proof_sources=qualification,
        candidate_state_coverage_source=candidate_binding,
        candidate_replay_proof_sources=replay,  # type: ignore[arg-type]
        derived_stage_coverage_source=CanonicalJsonProofBinding.bind(
            derived_stage_coverage_output_path
        ),
    )
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    proof = CanonicalJsonProofBinding.bind(output_path)
    rebuilt = revalidate_formal_preflight_stage_coverage_proof_artifact(
        proof.absolute_path,
        now_ns=now_ns,
    )
    if rebuilt.stage_coverage != evidence.stage_coverage:
        raise RuntimeError("published preflight coverage proof changed")
    return proof


def revalidate_formal_preflight_stage_coverage_proof_artifact(
    artifact_path: str | Path,
    *,
    now_ns: int,
    relocatable_bundle_manifest_path: str | Path | None = None,
) -> FormalPreflightFinalEvidence:
    """Deep-replay one local or stable-pulled preflight proof graph."""

    if relocatable_bundle_manifest_path is not None:
        from lightcone_spec.runtime.relocatable_evidence import (
            activate_relocatable_evidence_bundle,
        )

        with activate_relocatable_evidence_bundle(
            relocatable_bundle_manifest_path
        ) as bundle:
            artifact_text = str(artifact_path)
            if artifact_text not in bundle.artifact.entry_remote_absolute_paths:
                raise ValueError(
                    "preflight coverage proof is not a pulled-evidence entry"
                )
            return revalidate_formal_preflight_stage_coverage_proof_artifact(
                artifact_text,
                now_ns=now_ns,
            )
    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = FormalPreflightStageCoverageProofArtifact.from_dict(binding.reopen())
    evidence = _rebuild(artifact, now_ns=now_ns)
    if CanonicalJsonProofBinding.bind(artifact_path) != binding:
        raise RuntimeError("preflight coverage proof changed while replayed")
    return evidence


__all__ = (
    "FORMAL_PREFLIGHT_STAGE_COVERAGE_PROOF_PROTOCOL_SHA256",
    "FormalPreflightStageCoverageProofArtifact",
    "publish_formal_preflight_stage_coverage_proof_artifact",
    "revalidate_formal_preflight_stage_coverage_proof_artifact",
)
