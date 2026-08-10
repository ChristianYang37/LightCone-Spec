"""Registered clean-room OnlineSPEC baseline protocol and analysis."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments.data import (
    DFLASH_SAFE_CONTEXT_LIMIT,
    LongContinuationAdapter,
    sample_set_sha256,
)
from lightcone_spec.experiments.protocol import (
    FORMAL_CONCURRENCY_GRID,
    TUNING_STAGES,
    successive_halving,
    tuning_stage,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.selection import SliceMeasurement
from lightcone_spec.experiments.statistics import bca_mean_interval

ONLINE_SPEC_REPOSITORY = "https://github.com/ZinYY/OnlineSPEC"
ONLINE_SPEC_COMMIT = "e58f82eb3f3adca3a686211236bf4f6e9e7e3a2b"
ONLINE_SPEC_TREE = "e037a463f16bcbb19c909d4a626c4c25a983c289"
ONLINE_SPEC_PAPER = "https://arxiv.org/abs/2603.12617v2"
ONLINE_SPEC_SOURCE_AUDIT_SHA256 = (
    "20d0843e7eff72331656cdebcd443edc94077337bac8baa6d7bb4c3d7f73db87"
)
ONLINE_SPEC_CLAIM_SCOPE = (
    "clean-room-online-learner-equation-complete-not-official-system-reproduction"
)
ONLINE_SPEC_METHODS = (
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
)
ONLINE_SPEC_STUDY_METHODS = ("static", *ONLINE_SPEC_METHODS)


def _git_output(checkout: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(checkout), *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"cannot inspect OnlineSPEC source checkout with git: {checkout}"
        ) from exc
    return completed.stdout.strip()


def _sha256_value(value: object) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _write_bound(path: str | Path, value: object) -> None:
    output = Path(path)
    body = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if output.exists() and output.read_text(encoding="utf-8") != body:
        raise ValueError(f"refusing to overwrite immutable artifact {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")
    Path(f"{output}.sha256").write_text(_sha256_value(value) + "\n", encoding="utf-8")


def _load_bound(path: str | Path) -> dict:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    sidecar = Path(f"{source}.sha256")
    if not sidecar.is_file() or sidecar.read_text().strip() != _sha256_value(value):
        raise ValueError(f"artifact sidecar is missing or invalid: {source}")
    if not isinstance(value, dict):
        raise TypeError("bound artifact must be an object")
    return value


def verify_onlinespec_source_checkout(
    checkout: str | Path,
    audit_path: str | Path,
    *,
    expected_audit_sha256: str = ONLINE_SPEC_SOURCE_AUDIT_SHA256,
) -> dict:
    """Verify a clean source checkout against the content-bound source audit.

    The upstream repository has no license file at the audited commit, so its
    files are never vendored. This verifier proves which external tree was
    inspected without importing or redistributing that source.
    """

    source = Path(checkout).resolve()
    if not source.is_dir() or not (source / ".git").exists():
        raise ValueError("OnlineSPEC source checkout is not a Git worktree")
    audit = _load_bound(audit_path)
    audit_sha256 = _sha256_value(audit)
    if audit_sha256 != expected_audit_sha256:
        raise ValueError("OnlineSPEC source audit identity is not registered")
    required = {
        "schema_version",
        "repository",
        "commit",
        "tree",
        "key_files",
        "license_files",
        "license_status",
    }
    if not required <= set(audit) or audit["schema_version"] != 2:
        raise ValueError("OnlineSPEC source audit schema is incomplete")
    key_files = audit["key_files"]
    license_files = audit["license_files"]
    if (
        not isinstance(key_files, dict)
        or not key_files
        or not isinstance(license_files, list)
        or not all(isinstance(value, str) for value in license_files)
    ):
        raise ValueError("OnlineSPEC source audit file inventory is malformed")

    if _git_output(source, "rev-parse", "--is-inside-work-tree") != "true":
        raise ValueError("OnlineSPEC source path is not inside a Git worktree")
    head = _git_output(source, "rev-parse", "HEAD")
    tree = _git_output(source, "rev-parse", "HEAD^{tree}")
    if head != audit["commit"] or tree != audit["tree"]:
        raise ValueError("OnlineSPEC checkout commit/tree does not match the audit")
    if _git_output(source, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("OnlineSPEC source checkout must be clean")

    tracked = set(
        _git_output(source, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    )
    verified: dict[str, str] = {}
    for raw_path, expected in key_files.items():
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("OnlineSPEC key-file audit entry is malformed")
        relative = Path(raw_path)
        candidate = (source / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not candidate.is_relative_to(source)
            or raw_path not in tracked
            or not candidate.is_file()
        ):
            raise ValueError(f"OnlineSPEC audited file is unavailable: {raw_path}")
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"OnlineSPEC audited file hash mismatch: {raw_path}")
        verified[raw_path] = actual

    license_names = {
        "license",
        "license.md",
        "license.txt",
        "copying",
        "copying.md",
        "copying.txt",
    }
    discovered_licenses = sorted(
        path for path in tracked if Path(path).name.casefold() in license_names
    )
    if discovered_licenses != sorted(license_files):
        raise ValueError(
            "OnlineSPEC license-file inventory no longer matches the audit"
        )

    return {
        "schema_version": 2,
        "scope": "onlinespec-source-checkout-verification",
        "repository": audit["repository"],
        "commit": head,
        "tree": tree,
        "source_audit_sha256": audit_sha256,
        "clean": True,
        "verified_key_files": len(verified),
        "key_files_sha256": _sha256_value(verified),
        "license_status": audit["license_status"],
        "license_files": discovered_licenses,
    }


@dataclass(frozen=True)
class OnlineSpecCandidate:
    method: str
    weight_update_mode: str
    parameter_scope: str
    learning_rate: float
    rank: int | None
    stride: int
    projection_radius: float | None = None
    additional_learning_rates: tuple[float, ...] = ()
    hedge_learning_rate: float | None = None
    grad_clip: float = 1.0

    def validate(self) -> None:
        if self.method not in ONLINE_SPEC_METHODS:
            raise ValueError("unknown OnlineSPEC method")
        if self.weight_update_mode not in {"residual", "lora", "full"}:
            raise ValueError("unknown OnlineSPEC update mode")
        if self.parameter_scope not in {"tail", "drafter"}:
            raise ValueError("unknown OnlineSPEC parameter scope")
        if self.weight_update_mode == "residual" and self.parameter_scope != "tail":
            raise ValueError("residual OnlineSPEC is tail-only")
        if self.weight_update_mode == "full" and self.rank is not None:
            raise ValueError("full OnlineSPEC requires rank=null")
        if self.weight_update_mode != "full" and (self.rank is None or self.rank < 1):
            raise ValueError("factorized OnlineSPEC requires a positive rank")
        numeric = (self.learning_rate, self.grad_clip)
        if any(not math.isfinite(value) or value <= 0 for value in numeric):
            raise ValueError("OnlineSPEC optimizer values must be positive and finite")
        if self.stride < 1:
            raise ValueError("OnlineSPEC stride must be positive")
        if self.projection_radius is not None and (
            not math.isfinite(self.projection_radius) or self.projection_radius <= 0
        ):
            raise ValueError("projection radius must be positive and finite")
        rates = self.additional_learning_rates
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("expert learning rates must be positive and finite")
        if tuple(sorted(rates)) != rates or len(set(rates)) != len(rates):
            raise ValueError("expert learning rates must be unique and increasing")
        if self.method == "onlinespec_ens":
            if self.weight_update_mode != "full":
                raise ValueError("Hedge combines full parameter decisions")
            if (
                not rates
                or rates[0] <= self.learning_rate
                or self.hedge_learning_rate is None
                or not math.isfinite(self.hedge_learning_rate)
                or self.hedge_learning_rate <= 0
            ):
                raise ValueError("Hedge requires an ordered expert grid and meta rate")
        elif rates or self.hedge_learning_rate is not None:
            raise ValueError("expert fields are only valid for OnlineSPEC Hedge")

    @property
    def candidate_id(self) -> str:
        self.validate()
        return _sha256_value(asdict(self))


def onlinespec_candidates() -> tuple[OnlineSpecCandidate, ...]:
    """DFlash drafter candidates; confirmation never participates in tuning."""
    rows: list[OnlineSpecCandidate] = []
    for method in ("onlinespec_ogd", "onlinespec_opt"):
        # OnlineSPEC uses globally clipped projected SGD, not Adam's
        # coordinate-normalized step. Its rate is therefore the parameter-space
        # displacement scale. The logarithmic grid spans the scales used by the
        # source instantiations without selecting from confirmation evidence.
        for stride in (20, 40, 80, 160):
            for learning_rate in (1e-4, 1e-3, 1e-2, 1e-1):
                rows.append(
                    OnlineSpecCandidate(
                        method,
                        "full",
                        "drafter",
                        learning_rate,
                        None,
                        stride,
                    )
                )
            for rank in (8, 16, 32):
                for learning_rate in (1e-4, 1e-3, 1e-2, 1e-1):
                    rows.append(
                        OnlineSpecCandidate(
                            method,
                            "lora",
                            "drafter",
                            learning_rate,
                            rank,
                            stride,
                        )
                    )
    for stride in (40, 80, 160):
        # The pinned EAGLE3 ensemble recipe includes a 1e-4 base rate.  The
        # normalized single-step loss used here is not numerically identical
        # to that multi-epoch recipe, so retain a logarithmic tuning grid
        # instead of copying a claimed winner, but do not omit its scale.
        for learning_rate in (1e-4, 1e-3, 1e-2):
            for hedge_learning_rate in (0.1, 0.5, 1.0):
                rows.append(
                    OnlineSpecCandidate(
                        "onlinespec_ens",
                        "full",
                        "drafter",
                        learning_rate,
                        None,
                        stride,
                        additional_learning_rates=(
                            learning_rate * 3,
                            learning_rate * 10,
                        ),
                        hedge_learning_rate=hedge_learning_rate,
                    )
                )
    identities = [row.candidate_id for row in rows]
    if len(identities) != len(set(identities)):
        raise AssertionError("OnlineSPEC tuning grid contains duplicate candidates")
    return tuple(rows)


@dataclass(frozen=True)
class OnlineSpecManifest:
    schema_version: int
    name: str
    methods: tuple[str, ...]
    phases: tuple[str, ...]
    official_repository: str
    official_commit: str
    official_tree: str
    official_paper: str
    implementation: str
    claim_scope: str
    source_audit_sha256: str
    tuning_grid_sha256: str
    sampling_profile_sha256: str
    tuning_window_sha256: str
    confirmation_window_sha256: str
    confirmation_repetitions: int
    confirmation_schedule_seed: int
    request_scheduling: str
    headline_timing_unit: str
    inference_cluster_unit: str
    formal_context_start: int
    safe_context_limit: int
    gpu_evidence: str

    @classmethod
    def default(cls) -> OnlineSpecManifest:
        data = LongContinuationAdapter()
        return cls(
            schema_version=2,
            name="onlinespec-clean-room-baseline",
            methods=ONLINE_SPEC_STUDY_METHODS,
            phases=(
                "tuning_only_selection",
                "paired_controlled_confirmation",
                "independent_profiler",
            ),
            official_repository=ONLINE_SPEC_REPOSITORY,
            official_commit=ONLINE_SPEC_COMMIT,
            official_tree=ONLINE_SPEC_TREE,
            official_paper=ONLINE_SPEC_PAPER,
            implementation="clean-room-paper-equations",
            claim_scope=ONLINE_SPEC_CLAIM_SCOPE,
            source_audit_sha256=ONLINE_SPEC_SOURCE_AUDIT_SHA256,
            tuning_grid_sha256=_sha256_value(
                [asdict(candidate) for candidate in onlinespec_candidates()]
            ),
            sampling_profile_sha256=SamplingProfile().sha256,
            tuning_window_sha256=sample_set_sha256(data.window("tune")),
            confirmation_window_sha256=sample_set_sha256(data.window("confirm")),
            confirmation_repetitions=8,
            confirmation_schedule_seed=20260810,
            request_scheduling="ordered_native_batch_cohort_queue",
            headline_timing_unit="method_repetition_batch",
            inference_cluster_unit="repetition_block",
            formal_context_start=16384,
            safe_context_limit=DFLASH_SAFE_CONTEXT_LIMIT,
            gpu_evidence="UNMEASURED",
        )

    def validate(self) -> None:
        if self != type(self).default():
            raise ValueError(
                "OnlineSPEC source manifest differs from the registered protocol"
            )

    @property
    def sha256(self) -> str:
        self.validate()
        return _sha256_value(asdict(self))

    def write(self, path: str | Path) -> None:
        self.validate()
        _write_bound(path, asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> OnlineSpecManifest:
        value = _load_bound(path)
        artifact = cls(
            **{
                **value,
                "methods": tuple(value["methods"]),
                "phases": tuple(value["phases"]),
            }
        )
        artifact.validate()
        return artifact


@dataclass(frozen=True)
class OnlineSpecTuningMeasurement:
    method: str
    candidate_id: str
    goodput_ratio_to_static: float
    peak_hbm_bytes: int
    itl_p99_ms: float
    exposed_update_ms: float
    updates_launched: int
    updates_published: int
    safety_violations: int

    def validate(self) -> None:
        if self.method not in ONLINE_SPEC_METHODS:
            raise ValueError("unknown OnlineSPEC tuning method")
        if len(self.candidate_id) != 64:
            raise ValueError("OnlineSPEC candidate identity must be a SHA-256")
        if (
            any(
                not math.isfinite(value) or value < 0
                for value in (
                    self.goodput_ratio_to_static,
                    self.itl_p99_ms,
                    self.exposed_update_ms,
                )
            )
            or self.goodput_ratio_to_static <= 0
        ):
            raise ValueError("OnlineSPEC tuning metrics are invalid")
        if (
            min(
                self.peak_hbm_bytes,
                self.updates_launched,
                self.updates_published,
                self.safety_violations,
            )
            < 0
        ):
            raise ValueError("OnlineSPEC tuning counters cannot be negative")

    @property
    def safe(self) -> bool:
        return (
            self.safety_violations == 0
            and self.updates_launched > 0
            and self.updates_published > 0
        )


def reduce_onlinespec_tuning_stage(
    measurements: Iterable[SliceMeasurement],
    *,
    candidates: dict[str, OnlineSpecCandidate],
    active_candidate_ids: tuple[str, ...],
    stage: int,
) -> tuple[tuple[str, ...], tuple[OnlineSpecTuningMeasurement, ...]]:
    """Validate one paired stage and halve candidates within each learner."""
    prompt_count, context_limit = tuning_stage(stage)
    active = tuple(active_candidate_ids)
    if not active or len(active) != len(set(active)):
        raise ValueError("active OnlineSPEC candidates must be unique")
    if set(active) - candidates.keys():
        raise ValueError("active OnlineSPEC set contains an unknown candidate")
    active_methods = {candidates[candidate_id].method for candidate_id in active}
    if active_methods != set(ONLINE_SPEC_METHODS):
        raise ValueError("each tuning stage must retain every OnlineSPEC learner")
    rows = tuple(measurements)
    for row in rows:
        row.validate()
        if (
            row.phase != "onlinespec_tuning"
            or row.stage != stage
            or row.prompt_count != prompt_count
            or row.context_limit != context_limit
        ):
            raise ValueError("measurement belongs to another OnlineSPEC stage")
    static_rows = [row for row in rows if row.method == "static"]
    if len(static_rows) != 1:
        raise ValueError("each OnlineSPEC stage requires one Static reference")
    static = static_rows[0]
    if any(
        (
            static.exactness_violations,
            static.version_mismatches,
            static.fallbacks,
            static.nonfinite_updates,
            static.oom_events,
            static.retractions,
        )
    ):
        raise ValueError("OnlineSPEC Static reference failed its safety contract")
    adapted: dict[str, SliceMeasurement] = {}
    for row in rows:
        if row.method == "static":
            continue
        candidate_id = row.candidate_id or ""
        candidate = candidates.get(candidate_id)
        if candidate_id not in active or candidate is None:
            raise ValueError("OnlineSPEC slice is outside the active set")
        if candidate.method != row.method or candidate_id in adapted:
            raise ValueError("duplicate or mismatched OnlineSPEC candidate slice")
        if any(
            (
                row.manifest_sha256 != static.manifest_sha256,
                row.model_lock_sha256 != static.model_lock_sha256,
                row.sampling_profile_sha256 != static.sampling_profile_sha256,
                row.window_sha256 != static.window_sha256,
                row.output_set_sha256 != static.output_set_sha256,
                row.concurrency != static.concurrency,
            )
        ):
            raise ValueError("OnlineSPEC slice is not paired to Static")
        adapted[candidate_id] = row
    if set(adapted) != set(active):
        raise ValueError("OnlineSPEC tuning-stage coverage is incomplete")
    reduced: list[OnlineSpecTuningMeasurement] = []
    safe_by_method: dict[str, list[str]] = {
        method: [] for method in ONLINE_SPEC_METHODS
    }
    scores: dict[str, float] = {}
    for candidate_id in active:
        row = adapted[candidate_id]
        safety = sum(
            (
                row.exactness_violations,
                row.version_mismatches,
                row.fallbacks,
                row.nonfinite_updates,
                row.oom_events,
                row.retractions,
            )
        )
        measurement = OnlineSpecTuningMeasurement(
            method=row.method,
            candidate_id=candidate_id,
            goodput_ratio_to_static=(
                row.decode_goodput_tps / static.decode_goodput_tps
            ),
            peak_hbm_bytes=row.peak_hbm_bytes,
            itl_p99_ms=row.itl_p99_ms,
            exposed_update_ms=row.exposed_update_ms,
            updates_launched=row.updates_launched,
            updates_published=row.updates_published,
            safety_violations=safety,
        )
        measurement.validate()
        reduced.append(measurement)
        scores[candidate_id] = measurement.goodput_ratio_to_static
        if measurement.safe:
            safe_by_method[row.method].append(candidate_id)
    if any(not values for values in safe_by_method.values()):
        raise ValueError("a learner has no safe candidate in this tuning stage")
    survivors: list[str] = []
    for method in ONLINE_SPEC_METHODS:
        safe_ids = tuple(safe_by_method[method])
        survivors.extend(
            safe_ids
            if stage == len(TUNING_STAGES) - 1
            else successive_halving(safe_ids, scores)
        )
    return tuple(sorted(survivors)), tuple(
        sorted(reduced, key=lambda row: (row.method, row.candidate_id))
    )


@dataclass(frozen=True)
class OnlineSpecSelection:
    schema_version: int
    selected: tuple[OnlineSpecCandidate, ...]
    selected_concurrency: int
    manifest_sha256: str
    model_lock_sha256: str
    sampling_profile_sha256: str
    tuning_evidence_sha256: str
    reference_core_selection_sha256: str
    patched_sglang_tree: str

    def validate(self) -> None:
        if self.schema_version != 2 or self.patched_sglang_tree != PINNED_SGLANG_TREE:
            raise ValueError("OnlineSPEC selection runtime identity is invalid")
        if (
            tuple(candidate.method for candidate in self.selected)
            != ONLINE_SPEC_METHODS
        ):
            raise ValueError("selection requires one ordered candidate per method")
        if self.selected_concurrency not in FORMAL_CONCURRENCY_GRID:
            raise ValueError("OnlineSPEC selection load is invalid")
        for candidate in self.selected:
            candidate.validate()
        for value in (
            self.manifest_sha256,
            self.model_lock_sha256,
            self.sampling_profile_sha256,
            self.tuning_evidence_sha256,
            self.reference_core_selection_sha256,
        ):
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError("selection identity must be a SHA-256")

    @property
    def sha256(self) -> str:
        self.validate()
        return _sha256_value(asdict(self))

    def write(self, path: str | Path) -> None:
        self.validate()
        _write_bound(path, asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> OnlineSpecSelection:
        value = _load_bound(path)
        selected = tuple(
            OnlineSpecCandidate(
                **{
                    **row,
                    "additional_learning_rates": tuple(
                        row.get("additional_learning_rates", ())
                    ),
                }
            )
            for row in value.pop("selected")
        )
        artifact = cls(selected=selected, **value)
        artifact.validate()
        return artifact


def select_onlinespec(
    measurements: Iterable[OnlineSpecTuningMeasurement],
    *,
    candidates: dict[str, OnlineSpecCandidate],
    selected_concurrency: int,
    manifest_sha256: str,
    model_lock_sha256: str,
    sampling_profile_sha256: str,
    reference_core_selection_sha256: str,
    tuning_evidence_sha256: str | None = None,
) -> OnlineSpecSelection:
    rows = tuple(measurements)
    if not rows:
        raise ValueError("OnlineSPEC selection requires tuning evidence")
    identities = [(row.method, row.candidate_id) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("OnlineSPEC tuning evidence contains duplicate candidates")
    for row in rows:
        row.validate()
        candidate = candidates.get(row.candidate_id)
        if candidate is None or candidate.method != row.method:
            raise ValueError("OnlineSPEC tuning row references the wrong candidate")
    selected = []
    for method in ONLINE_SPEC_METHODS:
        eligible = [row for row in rows if row.method == method and row.safe]
        if not eligible:
            raise ValueError(f"no safe tuning candidate for {method}")
        winner = min(
            eligible,
            key=lambda row: (
                -row.goodput_ratio_to_static,
                row.peak_hbm_bytes,
                row.itl_p99_ms,
                row.exposed_update_ms,
                row.candidate_id,
            ),
        )
        selected.append(candidates[winner.candidate_id])
    evidence = tuning_evidence_sha256 or _sha256_value(
        [asdict(row) for row in sorted(rows, key=lambda row: row.candidate_id)]
    )
    if len(evidence) != 64 or any(char not in "0123456789abcdef" for char in evidence):
        raise ValueError("OnlineSPEC tuning evidence identity must be a SHA-256")
    artifact = OnlineSpecSelection(
        schema_version=2,
        selected=tuple(selected),
        selected_concurrency=selected_concurrency,
        manifest_sha256=manifest_sha256,
        model_lock_sha256=model_lock_sha256,
        sampling_profile_sha256=sampling_profile_sha256,
        tuning_evidence_sha256=evidence,
        reference_core_selection_sha256=reference_core_selection_sha256,
        patched_sglang_tree=PINNED_SGLANG_TREE,
    )
    artifact.validate()
    return artifact


@dataclass(frozen=True)
class OnlineSpecComparison:
    method: str
    mean_speedup: float
    ci_lower: float
    ci_upper: float
    safety_pass: bool


@dataclass(frozen=True)
class OnlineSpecGpuAttestation:
    schema_version: int
    status: str
    manifest_sha256: str
    selection_sha256: str
    model_lock_sha256: str
    performance_sha256: str
    patched_sglang_tree: str
    hardware_sha256: str
    methods: tuple[str, ...]
    repetitions: int

    def validate(self) -> None:
        if self.schema_version != 2 or self.status != "MEASURED":
            raise ValueError("OnlineSPEC GPU attestation must be schema-v2 MEASURED")
        if self.methods != ONLINE_SPEC_STUDY_METHODS or self.repetitions != 8:
            raise ValueError("OnlineSPEC attestation coverage is invalid")
        if self.patched_sglang_tree != PINNED_SGLANG_TREE:
            raise ValueError("OnlineSPEC attestation uses the wrong runtime tree")
        for value in (
            self.manifest_sha256,
            self.selection_sha256,
            self.model_lock_sha256,
            self.performance_sha256,
            self.hardware_sha256,
        ):
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError("OnlineSPEC attestation identity must be a SHA-256")

    @property
    def sha256(self) -> str:
        self.validate()
        return _sha256_value(asdict(self))

    def write(self, path: str | Path) -> None:
        self.validate()
        _write_bound(path, asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> OnlineSpecGpuAttestation:
        value = _load_bound(path)
        artifact = cls(**{**value, "methods": tuple(value["methods"])})
        artifact.validate()
        return artifact


def compare_onlinespec(
    rows: list[dict], *, seed: int = 0
) -> tuple[OnlineSpecComparison, ...]:
    """Paired diagnostics only; these rows never drive the core speed gate."""
    if {str(row["method"]) for row in rows} != set(ONLINE_SPEC_STUDY_METHODS):
        raise ValueError("OnlineSPEC comparison coverage has the wrong methods")
    filtered = [
        row
        for row in rows
        if row.get("region") == "long_region"
    ]
    grouped: dict[int, dict[str, dict]] = {}
    for row in filtered:
        key = int(row["repetition_block"])
        if str(row["method"]) in grouped.setdefault(key, {}):
            raise ValueError("duplicate OnlineSPEC paired cell")
        grouped[key][str(row["method"])] = row
    if not grouped or any(
        set(group) != set(ONLINE_SPEC_STUDY_METHODS) for group in grouped.values()
    ):
        raise ValueError("OnlineSPEC paired coverage is incomplete")
    prompt_batches = {
        str(row["prompt_id"]) for group in grouped.values() for row in group.values()
    }
    blocks = set(grouped)
    if (
        len(prompt_batches) != 1
        or not next(iter(prompt_batches)).startswith("batch-")
        or blocks != set(range(8))
    ):
        raise ValueError("OnlineSPEC comparison does not cover the registered matrix")
    concurrencies = {
        int(row["concurrency"]) for group in grouped.values() for row in group.values()
    }
    if len(concurrencies) != 1 or next(iter(concurrencies)) < 1:
        raise ValueError("OnlineSPEC methods must share one positive load")
    for group in grouped.values():
        at_risk = {int(row["at_risk_requests"]) for row in group.values()}
        output = {int(row["output_tokens"]) for row in group.values()}
        starts = {int(row["generated_bucket_start"]) for row in group.values()}
        ends = {int(row["generated_bucket_end"]) for row in group.values()}
        if (
            len(at_risk) != 1
            or len(output) != 1
            or min(at_risk | output) < 1
            or starts != {16384}
            or ends != {DFLASH_SAFE_CONTEXT_LIMIT}
        ):
            raise ValueError("OnlineSPEC methods do not share the paired work")
    full_rows = [
        row for row in rows if str(row.get("region")) == "full_trajectory"
    ]
    full_by_block: dict[int, dict[str, dict]] = {}
    for row in full_rows:
        block = int(row["repetition_block"])
        method = str(row["method"])
        methods = full_by_block.setdefault(block, {})
        if method in methods:
            raise ValueError("duplicate OnlineSPEC run-scope row")
        methods[method] = row
    if set(full_by_block) != set(range(8)) or any(
        set(methods) != set(ONLINE_SPEC_STUDY_METHODS)
        for methods in full_by_block.values()
    ):
        raise ValueError("OnlineSPEC run-scope safety coverage is incomplete")
    if any(
        str(row.get("prompt_id")) not in prompt_batches
        or int(row.get("concurrency", -1)) not in concurrencies
        for methods in full_by_block.values()
        for row in methods.values()
    ):
        raise ValueError("OnlineSPEC run-scope evidence uses another batch")
    results = []
    safety_fields = (
        "exactness_violations",
        "version_mismatches",
        "fallbacks",
        "nonfinite_updates",
        "oom_events",
        "retractions",
    )
    for index, method in enumerate(ONLINE_SPEC_METHODS):
        clusters: dict[str, list[float]] = {}
        safe = True
        for block, group in grouped.items():
            baseline = float(group["static"]["decode_goodput_tps"])
            measured = float(group[method]["decode_goodput_tps"])
            if (
                min(baseline, measured) <= 0
                or not np.isfinite([baseline, measured]).all()
            ):
                raise ValueError("OnlineSPEC goodput must be finite and positive")
            clusters[str(block)] = [measured / baseline - 1.0]
        for group in full_by_block.values():
            safe &= all(int(group[method][field]) == 0 for field in safety_fields)
            safe &= all(int(group["static"][field]) == 0 for field in safety_fields)
            safe &= int(group[method]["updates_launched"]) > 0
            safe &= int(group[method]["updates_published"]) > 0
        estimate, lower, upper = bca_mean_interval(
            {key: np.asarray(value) for key, value in clusters.items()},
            seed=seed + index,
        )
        results.append(OnlineSpecComparison(method, estimate, lower, upper, safe))
    return tuple(results)
