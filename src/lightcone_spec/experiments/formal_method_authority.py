"""Source-owned method authorities consumed by :class:`ProtocolLock`.

The scientific lock must not accept caller supplied digests for method
semantics.  These artifacts retain path-bound source inputs, replay them on
every load, and deterministically rebuild the public TTS and ChronoBelief
authorities.  Publication is canonical JSON and no-replace; a serialized
authority without its reopenable sources is deliberately unusable here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.adaptation.plan_authority import (
    TrainablePlanAuthorityBinding,
    trainable_plan_authority_binding_from_dict,
)
from lightcone_spec.experiments.formal_protocol import (
    TTS_PRIMARY_SOURCE_ID,
    TTS_PRIMARY_SOURCE_VERSION,
    ChronoBeliefAuthority,
    TtsCalibrationAuthority,
    content_sha256,
)
from lightcone_spec.runtime.content_authorization import (
    TtsCalibrationTuningWindow,
)
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

TTS_CALIBRATION_SOURCE_AUTHORITY_ID = "tts-primary-source-reconstruction-v2"
TTS_CALIBRATION_SOURCE_ARTIFACT_KIND = (
    "lightcone_tts_calibration_source_authority_artifact"
)
CHRONOBELIEF_SOURCE_AUTHORITY_ID = "lightcone-chronobelief-equations-5.5-5.8-v1"
CHRONOBELIEF_SOURCE_ARTIFACT_KIND = "lightcone_chronobelief_source_authority_artifact"
TTS_TUNING_WINDOW_SOURCE_KIND = "lightcone_tts_disjoint_tuning_window_source"
TTS_DRAFTER_NATIVE_LOSS_SOURCE_KIND = "lightcone_tts_drafter_native_loss_source"

TTS_DRAFTER_NATIVE_LOSS_SOURCE = {
    "schema_version": 1,
    "kind": TTS_DRAFTER_NATIVE_LOSS_SOURCE_KIND,
    "teacher_rows": "latest_round_only",
    "position_weights": "drafter_native",
    "proximal_anchor": "source_point",
    "request_reset": True,
    "execution_stream": "side_stream",
    "optimization_steps_per_update": 1,
}


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _strict_object(
    label: str,
    value: object,
    fields: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


def _raw_binding_from_dict(value: object, *, label: str) -> EvidenceFileBinding:
    return EvidenceFileBinding.from_dict(value, label=label)


def _reopen_raw(value: EvidenceFileBinding, *, label: str) -> None:
    if type(value) is not EvidenceFileBinding:
        raise TypeError(f"{label} requires an exact raw source binding")
    value.reopen(label=label)


def _reopen_json(
    value: CanonicalJsonProofBinding,
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not CanonicalJsonProofBinding:
        raise TypeError(f"{label} requires an exact canonical JSON binding")
    if CanonicalJsonProofBinding.bind(value.absolute_path) != value:
        raise ValueError(f"{label} changed")
    reopened = value.reopen()
    if type(reopened) is not dict:
        raise TypeError(f"{label} must be one canonical JSON object")
    return reopened


def _reopen_trainable_plan(
    value: CanonicalJsonProofBinding,
    *,
    require_full_drafter: bool,
) -> TrainablePlanAuthorityBinding:
    raw = _reopen_json(value, label="trainable-plan authority")
    binding = trainable_plan_authority_binding_from_dict(raw)
    if (
        type(binding) is not TrainablePlanAuthorityBinding
        or binding.sha256 != value.semantic_sha256
    ):
        raise ValueError("trainable-plan source identity differs")
    result = binding.revalidate()
    if result.binding != binding or result.plan.sha256 != binding.trainable_plan_sha256:
        raise ValueError("trainable-plan source did not replay exactly")
    if require_full_drafter and (
        result.plan.backend != "DFLASH"
        or result.plan.mode != "full"
        or result.plan.scope != "all"
        or binding.method not in {"tts", "l0"}
    ):
        raise ValueError("TTS source requires the complete DFlash drafter plan")
    return binding


def _reopen_tuning_window(value: CanonicalJsonProofBinding) -> None:
    window = TtsCalibrationTuningWindow.from_dict(
        _reopen_json(value, label="TTS tuning-window source")
    )
    if window.kind != TTS_TUNING_WINDOW_SOURCE_KIND:
        raise ValueError("TTS tuning-window source kind differs")


def _reopen_loss_source(value: CanonicalJsonProofBinding) -> None:
    if _reopen_json(value, label="TTS drafter-native loss source") != (
        TTS_DRAFTER_NATIVE_LOSS_SOURCE
    ):
        raise ValueError("TTS drafter-native loss source differs from protocol")


def tts_calibration_authority_to_dict(
    value: TtsCalibrationAuthority,
) -> dict[str, object]:
    if type(value) is not TtsCalibrationAuthority:
        raise TypeError("TTS authority codec requires an exact value")
    value.__post_init__()
    row = asdict(value)
    row["learning_rates"] = list(value.learning_rates)
    row["strides"] = list(value.strides)
    row["excluded_pilot_blocks"] = list(value.excluded_pilot_blocks)
    row["authority_sha256"] = value.sha256
    return row


def tts_calibration_authority_from_dict(value: object) -> TtsCalibrationAuthority:
    row = _strict_object(
        "TTS calibration authority",
        value,
        frozenset(TtsCalibrationAuthority.__dataclass_fields__) | {"authority_sha256"},
    )
    declared = _require_sha256("TTS declared authority", row.pop("authority_sha256"))
    for field in ("learning_rates", "strides", "excluded_pilot_blocks"):
        raw = row[field]
        if type(raw) is not list:
            raise TypeError(f"TTS authority {field} must be an array")
        row[field] = tuple(raw)
    authority = TtsCalibrationAuthority(**row)  # type: ignore[arg-type]
    if authority.sha256 != declared:
        raise ValueError("TTS authority digest differs from content")
    return authority


def chronobelief_authority_to_dict(value: ChronoBeliefAuthority) -> dict[str, object]:
    if type(value) is not ChronoBeliefAuthority:
        raise TypeError("ChronoBelief authority codec requires an exact value")
    value.__post_init__()
    row = asdict(value)
    row["equations"] = list(value.equations)
    row["authority_sha256"] = value.sha256
    return row


def chronobelief_authority_from_dict(value: object) -> ChronoBeliefAuthority:
    row = _strict_object(
        "ChronoBelief authority",
        value,
        frozenset(ChronoBeliefAuthority.__dataclass_fields__) | {"authority_sha256"},
    )
    declared = _require_sha256(
        "ChronoBelief declared authority", row.pop("authority_sha256")
    )
    equations = row["equations"]
    if type(equations) is not list:
        raise TypeError("ChronoBelief equations must be an array")
    row["equations"] = tuple(equations)
    authority = ChronoBeliefAuthority(**row)  # type: ignore[arg-type]
    if authority.sha256 != declared:
        raise ValueError("ChronoBelief authority digest differs from content")
    return authority


@dataclass(frozen=True)
class TtsCalibrationSourceAuthorityArtifact:
    schema_version: Literal[1]
    kind: Literal["lightcone_tts_calibration_source_authority_artifact"]
    paper_pdf_source: EvidenceFileBinding
    paper_source: EvidenceFileBinding
    tuning_window_source: CanonicalJsonProofBinding
    trainable_plan_authority_source: CanonicalJsonProofBinding
    drafter_native_loss_source: CanonicalJsonProofBinding
    authority: TtsCalibrationAuthority

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != TTS_CALIBRATION_SOURCE_ARTIFACT_KIND
        ):
            raise ValueError("TTS source authority artifact schema is unsupported")
        _reopen_raw(self.paper_pdf_source, label="TTS paper PDF")
        _reopen_raw(self.paper_source, label="TTS paper source")
        _reopen_tuning_window(self.tuning_window_source)
        plan = _reopen_trainable_plan(
            self.trainable_plan_authority_source,
            require_full_drafter=True,
        )
        _reopen_loss_source(self.drafter_native_loss_source)
        expected = TtsCalibrationAuthority(
            schema_version=1,
            authority_id=TTS_CALIBRATION_SOURCE_AUTHORITY_ID,
            primary_source_id=TTS_PRIMARY_SOURCE_ID,
            primary_source_version=TTS_PRIMARY_SOURCE_VERSION,
            paper_pdf_sha256=self.paper_pdf_source.raw_sha256,
            paper_source_sha256=self.paper_source.raw_sha256,
            tuning_window_sha256=self.tuning_window_source.semantic_sha256,
            trainable_plan_sha256=plan.trainable_plan_sha256,
            drafter_native_loss_recipe_sha256=(
                self.drafter_native_loss_source.semantic_sha256
            ),
        )
        if (
            type(self.authority) is not TtsCalibrationAuthority
            or self.authority != expected
        ):
            raise ValueError("TTS authority differs from source-owned replay")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        row: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "paper_pdf_source": self.paper_pdf_source.to_dict(),
            "paper_source": self.paper_source.to_dict(),
            "tuning_window_source": self.tuning_window_source.to_dict(),
            "trainable_plan_authority_source": (
                self.trainable_plan_authority_source.to_dict()
            ),
            "drafter_native_loss_source": self.drafter_native_loss_source.to_dict(),
            "authority": tts_calibration_authority_to_dict(self.authority),
        }
        if include_sha256:
            row["artifact_sha256"] = self.sha256
        return row

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "TTS source authority artifact",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "paper_pdf_source",
                    "paper_source",
                    "tuning_window_source",
                    "trainable_plan_authority_source",
                    "drafter_native_loss_source",
                    "authority",
                    "artifact_sha256",
                }
            ),
        )
        declared = _require_sha256("TTS source artifact", row.pop("artifact_sha256"))
        artifact = cls(
            schema_version=row["schema_version"],  # type: ignore[arg-type]
            kind=row["kind"],  # type: ignore[arg-type]
            paper_pdf_source=_raw_binding_from_dict(
                row["paper_pdf_source"], label="TTS paper PDF"
            ),
            paper_source=_raw_binding_from_dict(
                row["paper_source"], label="TTS paper source"
            ),
            tuning_window_source=CanonicalJsonProofBinding.from_dict(
                row["tuning_window_source"]
            ),
            trainable_plan_authority_source=CanonicalJsonProofBinding.from_dict(
                row["trainable_plan_authority_source"]
            ),
            drafter_native_loss_source=CanonicalJsonProofBinding.from_dict(
                row["drafter_native_loss_source"]
            ),
            authority=tts_calibration_authority_from_dict(row["authority"]),
        )
        if artifact.sha256 != declared:
            raise ValueError("TTS source artifact digest differs from content")
        return artifact


@dataclass(frozen=True)
class ChronoBeliefSourceAuthorityArtifact:
    schema_version: Literal[1]
    kind: Literal["lightcone_chronobelief_source_authority_artifact"]
    paper_pdf_source: EvidenceFileBinding
    tex_source: EvidenceFileBinding
    authority: ChronoBeliefAuthority

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != CHRONOBELIEF_SOURCE_ARTIFACT_KIND:
            raise ValueError(
                "ChronoBelief source authority artifact schema is unsupported"
            )
        _reopen_raw(self.paper_pdf_source, label="ChronoBelief paper PDF")
        _reopen_raw(self.tex_source, label="ChronoBelief TeX source")
        expected = ChronoBeliefAuthority(
            schema_version=1,
            authority_id=CHRONOBELIEF_SOURCE_AUTHORITY_ID,
            paper_pdf_sha256=self.paper_pdf_source.raw_sha256,
            tex_source_sha256=self.tex_source.raw_sha256,
        )
        if (
            type(self.authority) is not ChronoBeliefAuthority
            or self.authority != expected
        ):
            raise ValueError("ChronoBelief authority differs from source-owned replay")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        row: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "paper_pdf_source": self.paper_pdf_source.to_dict(),
            "tex_source": self.tex_source.to_dict(),
            "authority": chronobelief_authority_to_dict(self.authority),
        }
        if include_sha256:
            row["artifact_sha256"] = self.sha256
        return row

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "ChronoBelief source authority artifact",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "paper_pdf_source",
                    "tex_source",
                    "authority",
                    "artifact_sha256",
                }
            ),
        )
        declared = _require_sha256(
            "ChronoBelief source artifact", row.pop("artifact_sha256")
        )
        artifact = cls(
            schema_version=row["schema_version"],  # type: ignore[arg-type]
            kind=row["kind"],  # type: ignore[arg-type]
            paper_pdf_source=_raw_binding_from_dict(
                row["paper_pdf_source"], label="ChronoBelief paper PDF"
            ),
            tex_source=_raw_binding_from_dict(
                row["tex_source"], label="ChronoBelief TeX source"
            ),
            authority=chronobelief_authority_from_dict(row["authority"]),
        )
        if artifact.sha256 != declared:
            raise ValueError("ChronoBelief source artifact digest differs from content")
        return artifact


def build_source_tts_calibration_authority_artifact(
    *,
    paper_pdf_path: str | Path,
    paper_source_path: str | Path,
    tuning_window_path: str | Path,
    trainable_plan_authority_path: str | Path,
    drafter_native_loss_path: str | Path,
) -> TtsCalibrationSourceAuthorityArtifact:
    pdf = EvidenceFileBinding.bind(Path(paper_pdf_path), label="TTS paper PDF")
    paper = EvidenceFileBinding.bind(Path(paper_source_path), label="TTS paper source")
    tuning = CanonicalJsonProofBinding.bind(tuning_window_path)
    plan_source = CanonicalJsonProofBinding.bind(trainable_plan_authority_path)
    plan = _reopen_trainable_plan(plan_source, require_full_drafter=True)
    loss = CanonicalJsonProofBinding.bind(drafter_native_loss_path)
    _reopen_tuning_window(tuning)
    _reopen_loss_source(loss)
    authority = TtsCalibrationAuthority(
        schema_version=1,
        authority_id=TTS_CALIBRATION_SOURCE_AUTHORITY_ID,
        primary_source_id=TTS_PRIMARY_SOURCE_ID,
        primary_source_version=TTS_PRIMARY_SOURCE_VERSION,
        paper_pdf_sha256=pdf.raw_sha256,
        paper_source_sha256=paper.raw_sha256,
        tuning_window_sha256=tuning.semantic_sha256,
        trainable_plan_sha256=plan.trainable_plan_sha256,
        drafter_native_loss_recipe_sha256=loss.semantic_sha256,
    )
    return TtsCalibrationSourceAuthorityArtifact(
        schema_version=1,
        kind=TTS_CALIBRATION_SOURCE_ARTIFACT_KIND,
        paper_pdf_source=pdf,
        paper_source=paper,
        tuning_window_source=tuning,
        trainable_plan_authority_source=plan_source,
        drafter_native_loss_source=loss,
        authority=authority,
    )


def build_source_chronobelief_authority_artifact(
    *,
    paper_pdf_path: str | Path,
    tex_source_path: str | Path,
) -> ChronoBeliefSourceAuthorityArtifact:
    pdf = EvidenceFileBinding.bind(Path(paper_pdf_path), label="ChronoBelief paper PDF")
    tex = EvidenceFileBinding.bind(
        Path(tex_source_path), label="ChronoBelief TeX source"
    )
    authority = ChronoBeliefAuthority(
        schema_version=1,
        authority_id=CHRONOBELIEF_SOURCE_AUTHORITY_ID,
        paper_pdf_sha256=pdf.raw_sha256,
        tex_source_sha256=tex.raw_sha256,
    )
    return ChronoBeliefSourceAuthorityArtifact(
        schema_version=1,
        kind=CHRONOBELIEF_SOURCE_ARTIFACT_KIND,
        paper_pdf_source=pdf,
        tex_source=tex,
        authority=authority,
    )


def _publish_and_reopen(
    artifact: TtsCalibrationSourceAuthorityArtifact
    | ChronoBeliefSourceAuthorityArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    if isinstance(artifact, TtsCalibrationSourceAuthorityArtifact):
        reopened = TtsCalibrationSourceAuthorityArtifact.from_dict(binding.reopen())
    else:
        reopened = ChronoBeliefSourceAuthorityArtifact.from_dict(binding.reopen())
    if reopened != artifact:
        raise RuntimeError("method authority changed during publication")
    return binding


def publish_tts_calibration_authority_artifact(
    artifact: TtsCalibrationSourceAuthorityArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not TtsCalibrationSourceAuthorityArtifact:
        raise TypeError("TTS publisher requires an exact source artifact")
    artifact.__post_init__()
    return _publish_and_reopen(artifact, output_path)


def publish_chronobelief_authority_artifact(
    artifact: ChronoBeliefSourceAuthorityArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not ChronoBeliefSourceAuthorityArtifact:
        raise TypeError("ChronoBelief publisher requires an exact source artifact")
    artifact.__post_init__()
    return _publish_and_reopen(artifact, output_path)


def load_tts_calibration_authority_artifact(
    path: str | Path,
) -> TtsCalibrationSourceAuthorityArtifact:
    before = CanonicalJsonProofBinding.bind(path)
    artifact = TtsCalibrationSourceAuthorityArtifact.from_dict(before.reopen())
    if CanonicalJsonProofBinding.bind(before.absolute_path) != before:
        raise RuntimeError("TTS source authority changed while loaded")
    return artifact


def load_chronobelief_authority_artifact(
    path: str | Path,
) -> ChronoBeliefSourceAuthorityArtifact:
    before = CanonicalJsonProofBinding.bind(path)
    artifact = ChronoBeliefSourceAuthorityArtifact.from_dict(before.reopen())
    if CanonicalJsonProofBinding.bind(before.absolute_path) != before:
        raise RuntimeError("ChronoBelief source authority changed while loaded")
    return artifact


__all__ = (
    "CHRONOBELIEF_SOURCE_ARTIFACT_KIND",
    "CHRONOBELIEF_SOURCE_AUTHORITY_ID",
    "TTS_CALIBRATION_SOURCE_ARTIFACT_KIND",
    "TTS_CALIBRATION_SOURCE_AUTHORITY_ID",
    "TTS_DRAFTER_NATIVE_LOSS_SOURCE",
    "TTS_DRAFTER_NATIVE_LOSS_SOURCE_KIND",
    "TTS_TUNING_WINDOW_SOURCE_KIND",
    "ChronoBeliefSourceAuthorityArtifact",
    "TtsCalibrationSourceAuthorityArtifact",
    "build_source_chronobelief_authority_artifact",
    "build_source_tts_calibration_authority_artifact",
    "chronobelief_authority_from_dict",
    "chronobelief_authority_to_dict",
    "load_chronobelief_authority_artifact",
    "load_tts_calibration_authority_artifact",
    "publish_chronobelief_authority_artifact",
    "publish_tts_calibration_authority_artifact",
    "tts_calibration_authority_from_dict",
    "tts_calibration_authority_to_dict",
)
