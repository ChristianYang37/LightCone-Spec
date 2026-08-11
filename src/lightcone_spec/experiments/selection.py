"""Leakage-safe maximin selection shared by TTS and L0."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments.protocol import (
    FORMAL_CONCURRENCY_GRID,
    TUNING_STAGES,
    TuningCandidate,
    successive_halving,
    tuning_stage,
)


@dataclass(frozen=True)
class LossPoint:
    prefix_len_min: int
    prefix_len_max: int
    prefix_len_mean: float
    loss: float


@dataclass(frozen=True)
class SliceMeasurement:
    """Immutable endpoint measurement used before held-out confirmation."""

    schema_version: int
    phase: str
    stage: int
    method: str
    candidate_id: str | None
    manifest_sha256: str
    config_sha256: str
    model_lock_sha256: str
    sampling_profile_sha256: str
    window_sha256: str
    output_set_sha256: str
    prompt_count: int
    context_limit: int
    concurrency: int
    decode_goodput_tps: float
    itl_p99_ms: float
    peak_hbm_bytes: int
    kv_bytes: int
    kv_token_capacity: int
    optimizer_bytes: int
    trainable_parameters: int
    exposed_update_ms: float
    updates_launched: int
    updates_published: int
    exactness_violations: int
    version_mismatches: int
    fallbacks: int
    nonfinite_updates: int
    oom_events: int
    retractions: int
    loss_points: tuple[LossPoint, ...] = ()

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError("slice measurement must use schema version 2")
        if self.phase not in {
            "static_load_screen",
            "shared_config_tuning",
            "onlinespec_tuning",
        }:
            raise ValueError("slice measurement phase is not selectable")
        allowed_methods = {
            "static_load_screen": {"static"},
            "shared_config_tuning": {"static", "tts", "l0"},
            "onlinespec_tuning": {
                "static",
                "onlinespec_ogd",
                "onlinespec_opt",
                "onlinespec_ens",
            },
        }[self.phase]
        if self.method not in allowed_methods:
            raise ValueError("slice measurement has an unknown method")
        if self.prompt_count < 1 or self.context_limit < 1 or self.concurrency < 1:
            raise ValueError("slice dimensions must be positive")
        for name in (
            "manifest_sha256",
            "config_sha256",
            "model_lock_sha256",
            "sampling_profile_sha256",
            "window_sha256",
            "output_set_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        numeric = (
            self.decode_goodput_tps,
            self.itl_p99_ms,
            self.exposed_update_ms,
        )
        if not all(math.isfinite(value) and value >= 0 for value in numeric):
            raise ValueError("slice performance values must be finite and non-negative")
        if self.decode_goodput_tps <= 0:
            raise ValueError("slice decode goodput must be positive")
        counters = (
            self.peak_hbm_bytes,
            self.kv_bytes,
            self.kv_token_capacity,
            self.optimizer_bytes,
            self.trainable_parameters,
            self.updates_launched,
            self.updates_published,
            self.exactness_violations,
            self.version_mismatches,
            self.fallbacks,
            self.nonfinite_updates,
            self.oom_events,
            self.retractions,
        )
        if any(value < 0 for value in counters):
            raise ValueError("slice counters cannot be negative")
        if self.method == "static":
            if self.candidate_id is not None or any(
                (
                    self.optimizer_bytes,
                    self.trainable_parameters,
                    self.updates_launched,
                    self.updates_published,
                )
            ):
                raise ValueError("Static slice must have no adaptation state")
            if self.loss_points:
                raise ValueError("Static slice cannot contain adaptation loss")
        else:
            candidate = self.candidate_id or ""
            if len(candidate) != 64 or any(
                character not in "0123456789abcdef" for character in candidate
            ):
                raise ValueError("adapted slice requires a candidate SHA-256")
            if self.trainable_parameters < 1 or self.updates_launched < 1:
                raise ValueError("adapted slice did not execute an update")
            if not self.loss_points:
                raise ValueError("adapted slice lacks loss-by-prefix evidence")
        for point in self.loss_points:
            if (
                point.prefix_len_min < 1
                or point.prefix_len_max < point.prefix_len_min
                or not point.prefix_len_min
                <= point.prefix_len_mean
                <= point.prefix_len_max
                or not math.isfinite(point.loss)
                or point.loss < -1e-6
            ):
                raise ValueError("loss-by-prefix evidence is invalid")

    @property
    def sha256(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(body).hexdigest()

    def write(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        if output.exists() and output.read_text(encoding="utf-8") != body:
            raise ValueError("slice measurement is immutable")
        output.write_text(body, encoding="utf-8")
        Path(f"{output}.sha256").write_text(self.sha256 + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SliceMeasurement:
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        points = tuple(LossPoint(**row) for row in value.pop("loss_points", ()))
        artifact = cls(loss_points=points, **value)
        artifact.validate()
        sidecar = Path(f"{source}.sha256")
        if not sidecar.is_file() or sidecar.read_text().strip() != artifact.sha256:
            raise ValueError("slice measurement sidecar is missing or invalid")
        return artifact


@dataclass(frozen=True)
class CandidateMeasurement:
    candidate_id: str
    method: str
    phase: str
    goodput_ratio_to_static: float
    peak_hbm_bytes: int
    itl_p99_ms: float
    exposed_update_ms: float
    exactness_violations: int = 0
    version_mismatches: int = 0
    fallbacks: int = 0
    nonfinite_updates: int = 0
    oom_events: int = 0
    retractions: int = 0
    updates_launched: int = 0
    updates_published: int = 0

    def validate(self) -> None:
        if self.phase != "tune":
            raise ValueError("confirmation evidence must never enter selection")
        if self.method not in {"tts", "l0"}:
            raise ValueError("selection accepts only TTS and L0 measurements")
        for name in (
            "goodput_ratio_to_static",
            "itl_p99_ms",
            "exposed_update_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.goodput_ratio_to_static <= 0:
            raise ValueError("goodput ratio must be positive")
        if self.peak_hbm_bytes < 0:
            raise ValueError("peak HBM cannot be negative")
        for name in (
            "exactness_violations",
            "version_mismatches",
            "fallbacks",
            "nonfinite_updates",
            "oom_events",
            "retractions",
            "updates_launched",
            "updates_published",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")

    @property
    def safe(self) -> bool:
        return (
            self.updates_launched > 0
            and self.updates_published > 0
            and all(
                int(getattr(self, name)) == 0
                for name in (
                    "exactness_violations",
                    "version_mismatches",
                    "fallbacks",
                    "nonfinite_updates",
                    "oom_events",
                    "retractions",
                )
            )
        )


def reduce_tuning_stage(
    measurements: list[SliceMeasurement],
    *,
    candidates: dict[str, TuningCandidate],
    active_candidate_ids: tuple[str, ...],
    stage: int,
) -> tuple[tuple[str, ...], tuple[CandidateMeasurement, ...]]:
    """Validate one tuning stage and return survivors plus selection rows."""
    prompt_count, context_limit = tuning_stage(stage)
    if not active_candidate_ids or len(active_candidate_ids) != len(
        set(active_candidate_ids)
    ):
        raise ValueError("active tuning candidate identities must be unique")
    if set(active_candidate_ids) - candidates.keys():
        raise ValueError("active tuning set references an unknown candidate")
    for measurement in measurements:
        measurement.validate()
        if (
            measurement.phase != "shared_config_tuning"
            or measurement.stage != stage
            or measurement.prompt_count != prompt_count
            or measurement.context_limit != context_limit
        ):
            raise ValueError("measurement belongs to another tuning stage")
    static_rows = [row for row in measurements if row.method == "static"]
    if len(static_rows) != 1:
        raise ValueError("each tuning stage requires exactly one Static baseline")
    static = static_rows[0]
    grouped: dict[str, dict[str, SliceMeasurement]] = {}
    for row in measurements:
        if row.method == "static":
            continue
        if row.candidate_id not in active_candidate_ids:
            raise ValueError("measurement is outside the active tuning set")
        methods = grouped.setdefault(str(row.candidate_id), {})
        if row.method in methods:
            raise ValueError("duplicate candidate/method tuning slice")
        methods[row.method] = row
        if (
            row.manifest_sha256 != static.manifest_sha256
            or row.model_lock_sha256 != static.model_lock_sha256
            or row.sampling_profile_sha256 != static.sampling_profile_sha256
            or row.window_sha256 != static.window_sha256
            or row.output_set_sha256 != static.output_set_sha256
            or row.concurrency != static.concurrency
        ):
            raise ValueError("tuning slices are not paired to the Static baseline")
    if set(grouped) != set(active_candidate_ids) or any(
        set(methods) != {"tts", "l0"} for methods in grouped.values()
    ):
        raise ValueError("tuning stage coverage is incomplete")
    rows: list[CandidateMeasurement] = []
    scores: dict[str, float] = {}
    safe_candidates: list[str] = []
    for candidate_id in active_candidate_ids:
        methods = grouped[candidate_id]
        ratios = []
        candidate_rows: list[CandidateMeasurement] = []
        for method in ("tts", "l0"):
            source = methods[method]
            ratio = source.decode_goodput_tps / static.decode_goodput_tps
            ratios.append(ratio)
            candidate_rows.append(
                CandidateMeasurement(
                    candidate_id=candidate_id,
                    method=method,
                    phase="tune",
                    goodput_ratio_to_static=ratio,
                    peak_hbm_bytes=source.peak_hbm_bytes,
                    itl_p99_ms=source.itl_p99_ms,
                    exposed_update_ms=source.exposed_update_ms,
                    exactness_violations=source.exactness_violations,
                    version_mismatches=source.version_mismatches,
                    fallbacks=source.fallbacks,
                    nonfinite_updates=source.nonfinite_updates,
                    oom_events=source.oom_events,
                    retractions=source.retractions,
                    updates_launched=source.updates_launched,
                    updates_published=source.updates_published,
                )
            )
        rows.extend(candidate_rows)
        scores[candidate_id] = min(ratios)
        if all(row.safe for row in candidate_rows):
            safe_candidates.append(candidate_id)
    if not safe_candidates:
        raise ValueError("no tuning candidate passes the stage safety gate")
    safe_ids = tuple(safe_candidates)
    survivors = (
        safe_ids
        if stage == len(TUNING_STAGES) - 1
        else successive_halving(safe_ids, scores)
    )
    return survivors, tuple(rows)


@dataclass(frozen=True)
class SelectionArtifact:
    schema_version: int
    candidate: TuningCandidate
    selected_concurrency: int
    minimum_goodput_ratio: float
    peak_hbm_bytes: int
    itl_p99_ms: float
    exposed_update_ms: float
    manifest_sha256: str
    sampling_profile_sha256: str
    tuning_grid_sha256: str
    load_screen_sha256: str
    tuning_window_sha256: str
    model_lock_sha256: str
    patched_sglang_tree: str
    tuning_evidence_sha256: str
    selection_protocol: str = "successive_halving"

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def sha256(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(body).hexdigest()

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError("selection artifact must use schema version 2")
        if self.selected_concurrency not in FORMAL_CONCURRENCY_GRID:
            raise ValueError("selection artifact has an invalid concurrency")
        if self.patched_sglang_tree != PINNED_SGLANG_TREE:
            raise ValueError("selection artifact uses the wrong runtime tree")
        if self.selection_protocol not in {
            "successive_halving",
            "heldout_anchor",
        }:
            raise ValueError("selection artifact uses an unknown protocol")
        for name in (
            "minimum_goodput_ratio",
            "itl_p99_ms",
            "exposed_update_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.minimum_goodput_ratio <= 0 or self.peak_hbm_bytes < 0:
            raise ValueError("selection performance and memory must be valid")
        for name in (
            "manifest_sha256",
            "sampling_profile_sha256",
            "tuning_grid_sha256",
            "load_screen_sha256",
            "tuning_window_sha256",
            "model_lock_sha256",
            "tuning_evidence_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")

    def write(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        if output.exists() and output.read_text(encoding="utf-8") != body:
            raise ValueError("selection artifact is immutable")
        output.write_text(body, encoding="utf-8")
        Path(f"{output}.sha256").write_text(self.sha256 + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SelectionArtifact:
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        value.setdefault("selection_protocol", "successive_halving")
        candidate = TuningCandidate(**value.pop("candidate"))
        artifact = cls(candidate=candidate, **value)
        artifact.validate()
        sidecar = Path(f"{source}.sha256")
        if not sidecar.is_file() or sidecar.read_text().strip() != artifact.sha256:
            raise ValueError("selection artifact sidecar is missing or invalid")
        return artifact


def select_shared_config(
    measurements: list[CandidateMeasurement],
    *,
    candidates: dict[str, TuningCandidate],
    selected_concurrency: int,
    manifest_sha256: str,
    sampling_profile_sha256: str,
    tuning_grid_sha256: str,
    load_screen_sha256: str,
    tuning_window_sha256: str,
    model_lock_sha256: str,
    tuning_evidence_sha256: str | None = None,
    selection_protocol: str = "successive_halving",
) -> SelectionArtifact:
    if not measurements:
        raise ValueError("selection requires tuning measurements")
    for row in measurements:
        row.validate()
    grouped: dict[str, dict[str, CandidateMeasurement]] = {}
    for row in measurements:
        if row.candidate_id not in candidates:
            raise ValueError("measurement references an unknown candidate")
        if candidates[row.candidate_id].candidate_id != row.candidate_id:
            raise ValueError("candidate identity does not match its contents")
        methods = grouped.setdefault(row.candidate_id, {})
        if row.method in methods:
            raise ValueError("duplicate candidate/method measurement")
        methods[row.method] = row
    eligible: list[tuple[tuple[float, int, float, float, str], str]] = []
    for candidate_id, methods in grouped.items():
        if set(methods) != {"tts", "l0"}:
            continue
        rows = tuple(methods.values())
        if not all(row.safe for row in rows):
            continue
        minimum_ratio = min(row.goodput_ratio_to_static for row in rows)
        peak_hbm = max(row.peak_hbm_bytes for row in rows)
        p99 = max(row.itl_p99_ms for row in rows)
        exposed = max(row.exposed_update_ms for row in rows)
        eligible.append(
            ((-minimum_ratio, peak_hbm, p99, exposed, candidate_id), candidate_id)
        )
    if not eligible:
        raise ValueError("no shared configuration passes safety requirements")
    eligible.sort()
    winner_id = eligible[0][1]
    winner = grouped[winner_id]
    canonical_rows = [
        asdict(row)
        for row in sorted(measurements, key=lambda row: (row.candidate_id, row.method))
    ]
    rows_hash = hashlib.sha256(
        json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    evidence_hash = tuning_evidence_sha256 or rows_hash
    artifact = SelectionArtifact(
        schema_version=2,
        candidate=candidates[winner_id],
        selected_concurrency=selected_concurrency,
        minimum_goodput_ratio=min(
            row.goodput_ratio_to_static for row in winner.values()
        ),
        peak_hbm_bytes=max(row.peak_hbm_bytes for row in winner.values()),
        itl_p99_ms=max(row.itl_p99_ms for row in winner.values()),
        exposed_update_ms=max(row.exposed_update_ms for row in winner.values()),
        manifest_sha256=manifest_sha256,
        sampling_profile_sha256=sampling_profile_sha256,
        tuning_grid_sha256=tuning_grid_sha256,
        load_screen_sha256=load_screen_sha256,
        tuning_window_sha256=tuning_window_sha256,
        model_lock_sha256=model_lock_sha256,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        tuning_evidence_sha256=evidence_hash,
        selection_protocol=selection_protocol,
    )
    artifact.validate()
    return artifact


def select_heldout_anchor(
    measurements: list[SliceMeasurement],
    *,
    candidate: TuningCandidate,
    selected_concurrency: int,
    manifest_sha256: str,
    sampling_profile_sha256: str,
    tuning_grid_sha256: str,
    load_screen_sha256: str,
    tuning_window_sha256: str,
    model_lock_sha256: str,
    tuning_evidence_sha256: str,
) -> SelectionArtifact:
    """Lock one tuning-window anchor without claiming grid optimality.

    This path exists for a faithful held-out reproduction: the anchor may be
    chosen from diagnostic tuning evidence, but it must be re-measured on the
    complete terminal tuning slice before any confirmation prompt is touched.
    It deliberately shares the same safety, pairing, and confirmation runtime
    as the exhaustive successive-halving path.
    """
    survivors, rows = reduce_tuning_stage(
        measurements,
        candidates={candidate.candidate_id: candidate},
        active_candidate_ids=(candidate.candidate_id,),
        stage=len(TUNING_STAGES) - 1,
    )
    if survivors != (candidate.candidate_id,):
        raise ValueError("held-out anchor did not survive its tuning safety gate")
    if any(
        measurement.concurrency != selected_concurrency for measurement in measurements
    ):
        raise ValueError("held-out anchor load differs from the Static load screen")
    return select_shared_config(
        list(rows),
        candidates={candidate.candidate_id: candidate},
        selected_concurrency=selected_concurrency,
        manifest_sha256=manifest_sha256,
        sampling_profile_sha256=sampling_profile_sha256,
        tuning_grid_sha256=tuning_grid_sha256,
        load_screen_sha256=load_screen_sha256,
        tuning_window_sha256=tuning_window_sha256,
        model_lock_sha256=model_lock_sha256,
        tuning_evidence_sha256=tuning_evidence_sha256,
        selection_protocol="heldout_anchor",
    )
