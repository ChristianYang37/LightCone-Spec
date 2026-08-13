"""Registered clean-room OnlineSPEC baseline protocol and analysis."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NoReturn

import numpy as np

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.adaptation.parameters import LAYER_SCOPES, LORA_RANKS
from lightcone_spec.execution import ControlledExecutionPolicy
from lightcone_spec.experiments.data import (
    DFLASH_SAFE_CONTEXT_LIMIT,
    LongContinuationAdapter,
    sample_set_sha256,
)
from lightcone_spec.experiments.protocol import (
    FORMAL_CONCURRENCY_GRID,
    successive_halving,
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
ONLINE_SPEC_MANIFEST_KIND = "preliminary_diagnostic_onlinespec_manifest"
ONLINE_SPEC_EVIDENCE_SCOPE = "PRELIMINARY_DIAGNOSTIC_ONLY"
ONLINE_SPEC_METHODS = (
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
)
ONLINE_SPEC_STUDY_METHODS = ("static", *ONLINE_SPEC_METHODS)
# OnlineSPEC candidates can be intentionally weak near the origin and become
# useful only after enough online feedback has accumulated.  Its resource axis
# therefore starts at the headline region instead of borrowing the core
# Static/TTS/L0 4K/8K stages.  Prompt count still grows by successive halving,
# while every survivor is measured on an increasingly long tuning trajectory
# that remains disjoint from confirmation.  The complete schedule is also
# embedded in the manifest below.
ONLINE_SPEC_TUNING_STAGES = (
    (2, 16384),
    (4, 24576),
    (8, 32768),
    (16, DFLASH_SAFE_CONTEXT_LIMIT),
)


def onlinespec_tuning_stage(stage: int) -> tuple[int, int]:
    if stage not in range(len(ONLINE_SPEC_TUNING_STAGES)):
        raise ValueError("OnlineSPEC tuning stage must be in [0, 4)")
    return ONLINE_SPEC_TUNING_STAGES[stage]


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
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"bound artifact must be a regular file: {source}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"bound artifact contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"bound artifact contains non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            source.read_bytes().decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"bound artifact is not strict UTF-8 JSON: {source}"
        ) from error

    def reject_nonfinite(item: object) -> None:
        if type(item) is float and not math.isfinite(item):
            raise ValueError("bound artifact contains a non-finite JSON number")
        if type(item) is dict:
            for nested in item.values():
                reject_nonfinite(nested)
        elif type(item) is list:
            for nested in item:
                reject_nonfinite(nested)

    reject_nonfinite(value)
    sidecar = Path(f"{source}.sha256")
    expected_sidecar = f"{_sha256_value(value)}\n".encode("ascii")
    if (
        sidecar.is_symlink()
        or not sidecar.is_file()
        or sidecar.read_bytes() != expected_sidecar
    ):
        raise ValueError(f"artifact sidecar is missing or invalid: {source}")
    if type(value) is not dict:
        raise TypeError("bound artifact must be an object")
    return value


def _historical_onlinespec_payload(current: dict[str, object]) -> dict[str, object]:
    value = current.copy()
    for field_name in (
        "kind",
        "evidence_scope",
        "formal_execution_authorized",
        "industrial_authority_consumption",
    ):
        value.pop(field_name)
    value["name"] = "onlinespec-clean-room-baseline"
    value["formal_context_start"] = value.pop("diagnostic_context_start")
    value["gpu_evidence"] = "UNMEASURED"
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
        if self.weight_update_mode not in {"lora", "full"}:
            raise ValueError("unknown OnlineSPEC update mode")
        if self.parameter_scope not in LAYER_SCOPES:
            raise ValueError("unknown OnlineSPEC parameter scope")
        if self.weight_update_mode == "full" and self.rank is not None:
            raise ValueError("full OnlineSPEC requires rank=null")
        if self.weight_update_mode == "lora" and self.rank not in LORA_RANKS:
            raise ValueError("factorized OnlineSPEC requires a registered LoRA rank")
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
                        "all",
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
                            "all",
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
        for weight_update_mode, rank in (
            ("full", None),
            ("lora", 8),
            ("lora", 16),
            ("lora", 32),
        ):
            for learning_rate in (1e-4, 1e-3, 1e-2):
                for hedge_learning_rate in (0.1, 0.5, 1.0):
                    rows.append(
                        OnlineSpecCandidate(
                            "onlinespec_ens",
                            weight_update_mode,
                            "all",
                            learning_rate,
                            rank,
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
    kind: str
    name: str
    evidence_scope: str
    formal_execution_authorized: bool
    industrial_authority_consumption: str
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
    tuning_stages: tuple[tuple[int, int], ...]
    sampling_profile_sha256: str
    execution_policy_sha256: str
    tuning_window_sha256: str
    confirmation_window_sha256: str
    confirmation_repetitions: int
    confirmation_schedule_seed: int
    request_scheduling: str
    headline_timing_unit: str
    inference_cluster_unit: str
    diagnostic_context_start: int
    safe_context_limit: int
    gpu_evidence: str
    _historical_source_sha256: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def default(cls) -> OnlineSpecManifest:
        data = LongContinuationAdapter()
        return cls(
            schema_version=2,
            kind=ONLINE_SPEC_MANIFEST_KIND,
            name="preliminary-diagnostic-onlinespec-clean-room-baseline",
            evidence_scope=ONLINE_SPEC_EVIDENCE_SCOPE,
            formal_execution_authorized=False,
            industrial_authority_consumption="FORBIDDEN",
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
            tuning_stages=ONLINE_SPEC_TUNING_STAGES,
            sampling_profile_sha256=SamplingProfile().sha256,
            execution_policy_sha256=ControlledExecutionPolicy().sha256,
            tuning_window_sha256=sample_set_sha256(data.window("tune")),
            confirmation_window_sha256=sample_set_sha256(data.window("confirm")),
            confirmation_repetitions=8,
            confirmation_schedule_seed=20260810,
            request_scheduling="ordered_native_batch_cohort_queue",
            headline_timing_unit="method_repetition_batch",
            inference_cluster_unit="repetition_block",
            diagnostic_context_start=16384,
            safe_context_limit=DFLASH_SAFE_CONTEXT_LIMIT,
            gpu_evidence=ONLINE_SPEC_EVIDENCE_SCOPE,
        )

    def validate(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError("OnlineSPEC schema version must be an exact integer")
        if self.formal_execution_authorized is not False:
            raise ValueError("OnlineSPEC diagnostics cannot authorize formal execution")
        exact_integer_fields = (
            self.confirmation_repetitions,
            self.confirmation_schedule_seed,
            self.diagnostic_context_start,
            self.safe_context_limit,
        )
        if any(type(value) is not int for value in exact_integer_fields):
            raise TypeError("OnlineSPEC scalar identities must be exact integers")
        if type(self.tuning_stages) is not tuple or any(
            type(stage) is not tuple
            or len(stage) != 2
            or any(type(value) is not int for value in stage)
            for stage in self.tuning_stages
        ):
            raise TypeError("OnlineSPEC tuning stages must contain exact integer pairs")
        if self._payload() != type(self).default()._payload():
            raise ValueError(
                "OnlineSPEC source manifest differs from the registered protocol"
            )
        if self._historical_source_sha256 is not None:
            expected_historical_sha256 = _sha256_value(
                _historical_onlinespec_payload(self._payload())
            )
            if self._historical_source_sha256 != expected_historical_sha256:
                raise ValueError("historical OnlineSPEC source identity mismatch")

    def _payload(self) -> dict:
        value = asdict(self)
        value.pop("_historical_source_sha256")
        return value

    @property
    def sha256(self) -> str:
        self.validate()
        return self._historical_source_sha256 or _sha256_value(self._payload())

    def write(self, path: str | Path) -> None:
        self.validate()
        if self._historical_source_sha256 is not None:
            raise ValueError(
                "historical OnlineSPEC manifests are read-only preliminary evidence"
            )
        _write_bound(path, self._payload())

    @classmethod
    def load(cls, path: str | Path) -> OnlineSpecManifest:
        value = _load_bound(path)
        source_sha256 = _sha256_value(value)
        current_fields = set(cls.default()._payload())
        historical_fields = set(
            _historical_onlinespec_payload(cls.default()._payload())
        )
        if set(value) == current_fields:
            historical = False
        elif set(value) == historical_fields:
            historical = True
        else:
            raise ValueError("OnlineSPEC manifest fields do not match schema")
        if historical and not (
            type(value.get("schema_version")) is int
            and value.get("schema_version") == 2
            and value.get("name") == "onlinespec-clean-room-baseline"
            and value.get("gpu_evidence") == "UNMEASURED"
            and "kind" not in value
            and "evidence_scope" not in value
            and "formal_execution_authorized" not in value
            and "industrial_authority_consumption" not in value
            and "formal_context_start" in value
            and "diagnostic_context_start" not in value
        ):
            raise ValueError("historical OnlineSPEC manifest identity mismatch")
        for field_name in ("methods", "phases", "tuning_stages"):
            if type(value[field_name]) is not list:
                raise TypeError(
                    "OnlineSPEC manifest sequence fields must be JSON arrays"
                )
        if any(
            type(stage) is not list or len(stage) != 2
            for stage in value["tuning_stages"]
        ):
            raise TypeError("OnlineSPEC tuning stages must be two-element JSON arrays")
        if historical:
            value = {
                **value,
                "kind": ONLINE_SPEC_MANIFEST_KIND,
                "name": "preliminary-diagnostic-onlinespec-clean-room-baseline",
                "evidence_scope": ONLINE_SPEC_EVIDENCE_SCOPE,
                "formal_execution_authorized": False,
                "industrial_authority_consumption": "FORBIDDEN",
                "diagnostic_context_start": value["formal_context_start"],
                "gpu_evidence": ONLINE_SPEC_EVIDENCE_SCOPE,
            }
            value.pop("formal_context_start")
        artifact = cls(
            **{
                **value,
                "methods": tuple(value["methods"]),
                "phases": tuple(value["phases"]),
                "tuning_stages": tuple(
                    tuple(stage) for stage in value["tuning_stages"]
                ),
                "_historical_source_sha256": (source_sha256 if historical else None),
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
    prompt_count, context_limit = onlinespec_tuning_stage(stage)
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
            if stage == len(ONLINE_SPEC_TUNING_STAGES) - 1
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
    selection_protocol: str = "successive_halving"

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
        if self.selection_protocol not in {
            "successive_halving",
            "heldout_anchor",
        }:
            raise ValueError("OnlineSPEC selection uses an unknown protocol")
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
        value.setdefault("selection_protocol", "successive_halving")
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
    selection_protocol: str = "successive_halving",
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
        selection_protocol=selection_protocol,
    )
    artifact.validate()
    return artifact


def select_onlinespec_heldout_anchor(
    measurements: Iterable[SliceMeasurement],
    *,
    candidates: dict[str, OnlineSpecCandidate],
    selected_concurrency: int,
    manifest_sha256: str,
    model_lock_sha256: str,
    sampling_profile_sha256: str,
    reference_core_selection_sha256: str,
    tuning_evidence_sha256: str,
) -> OnlineSpecSelection:
    """Lock one safe terminal candidate per learner without a grid-optimum claim."""
    if (
        len(candidates) != len(ONLINE_SPEC_METHODS)
        or {candidate.method for candidate in candidates.values()}
        != set(ONLINE_SPEC_METHODS)
        or any(
            candidate_id != candidate.candidate_id
            for candidate_id, candidate in candidates.items()
        )
    ):
        raise ValueError(
            "OnlineSPEC anchor requires one registered candidate per learner"
        )
    rows = tuple(measurements)
    if any(row.concurrency != selected_concurrency for row in rows):
        raise ValueError("OnlineSPEC anchor load differs from the core Static load")
    survivors, reduced = reduce_onlinespec_tuning_stage(
        rows,
        candidates=candidates,
        active_candidate_ids=tuple(candidates),
        stage=len(ONLINE_SPEC_TUNING_STAGES) - 1,
    )
    if survivors != tuple(sorted(candidates)):
        raise ValueError("an OnlineSPEC anchor failed its terminal safety gate")
    return select_onlinespec(
        reduced,
        candidates=candidates,
        selected_concurrency=selected_concurrency,
        manifest_sha256=manifest_sha256,
        model_lock_sha256=model_lock_sha256,
        sampling_profile_sha256=sampling_profile_sha256,
        reference_core_selection_sha256=reference_core_selection_sha256,
        tuning_evidence_sha256=tuning_evidence_sha256,
        selection_protocol="heldout_anchor",
    )


@dataclass(frozen=True)
class OnlineSpecComparison:
    method: str
    mean_speedup: float
    ci_lower: float
    ci_upper: float
    safety_pass: bool
    acceleration_pass: bool

    @property
    def passed(self) -> bool:
        return self.safety_pass and self.acceleration_pass


@dataclass(frozen=True)
class OnlineSpecGpuAttestation:
    """Disabled legacy attestation schema retained only for clear rejection."""

    schema_version: int
    status: str
    manifest_sha256: str
    selection_sha256: str
    model_lock_sha256: str
    performance_sha256: str
    target_reference_sha256: str
    patched_sglang_tree: str
    hardware_sha256: str
    methods: tuple[str, ...]
    repetitions: int

    def validate(self) -> None:
        raise RuntimeError(
            "onlinespec_gpu_attestation_api_disabled: comparison evidence cannot "
            "enter the core industrial gate"
        )

    @property
    def sha256(self) -> str:
        self.validate()
        return _sha256_value(asdict(self))

    def write(self, path: str | Path) -> None:
        self.validate()
        _write_bound(path, asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> OnlineSpecGpuAttestation:
        del path
        raise RuntimeError(
            "onlinespec_gpu_attestation_api_disabled: comparison evidence cannot "
            "enter the core industrial gate"
        )


def compare_onlinespec(
    rows: list[dict], *, minimum_speedup: float = 0.03, seed: int = 0
) -> tuple[OnlineSpecComparison, ...]:
    """Paired diagnostics only; these rows never drive the core speed gate."""
    if minimum_speedup < 0:
        raise ValueError("minimum speedup cannot be negative")
    if {str(row["method"]) for row in rows} != set(ONLINE_SPEC_STUDY_METHODS):
        raise ValueError("OnlineSPEC comparison coverage has the wrong methods")
    filtered = [row for row in rows if row.get("region") == "long_region"]
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
    long_ends = {
        int(row["generated_bucket_end"])
        for group in grouped.values()
        for row in group.values()
    }
    if (
        len(long_ends) != 1
        or not 16384 < next(iter(long_ends)) < DFLASH_SAFE_CONTEXT_LIMIT
    ):
        raise ValueError(
            "OnlineSPEC long bounds must be shared generated-token positions"
        )
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
            or ends != long_ends
        ):
            raise ValueError("OnlineSPEC methods do not share the paired work")
    full_rows = [row for row in rows if str(row.get("region")) == "full_trajectory"]
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
        results.append(
            OnlineSpecComparison(
                method=method,
                mean_speedup=estimate,
                ci_lower=lower,
                ci_upper=upper,
                safety_pass=safe,
                acceleration_pass=estimate >= minimum_speedup and lower > 0.0,
            )
        )
    return tuple(results)
