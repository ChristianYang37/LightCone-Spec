"""Derive controller manifests from an attested P5 stride winner.

The helper in this module is deliberately free of filesystem I/O.  Receipt
and evidence verification belongs to the thin handoff script; identity and
cell matching live here so they can be unit tested without manufacturing an
artifact tree.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from lightcone_spec.exit_codes import ConfigError
from lightcone_spec.locking.hashing import canonical_json, sha256_json
from lightcone_spec.orchestration.catalog import (
    p5_priority_dflash_l3_evaluation_manifest,
    p5_priority_dflash_paired_trace_manifest,
)
from lightcone_spec.orchestration.manifest import ExperimentManifest
from lightcone_spec.orchestration.units import RunUnit


SELECTION_RULE = "p5_stride_screen_selection_v2"
SOURCE_MANIFEST_NAME = "p5_priority_dflash_stride_screen_v1"
SCREEN_STRIDES = (1, 4, 8, 16)
MATCHED_LR = 1e-4
MATCHED_WEIGHT_DECAY = 1e-2
MATCHED_PYTORCH_CUDA_ALLOC_CONF = "backend:native,expandable_segments:True"

# Runtime drift is never authorized by a pathname (for example, merely because
# only ``runtime_config.py`` changed).  Every exception must be code-reviewed
# as one exact pair of already validated aggregate fingerprints, together with
# the exact structural delta it is expected to contain.  The v7 -> current-v8
# pair (screen SHA ``b65e57c...``) is intentionally absent: its DFlash client
# and tail-adaptation sources changed as well as measurement/configuration
# code, so it requires a fresh v8 screen rather than retrospective equivalence.
EXACT_RUNTIME_TRANSITION_AUTHORIZATIONS: dict[tuple[str, str], dict[str, Any]] = {}

_SHA_BINDINGS = (
    "terminal_receipt_sha256",
    "execution_receipt_sha256",
    "selection_receipt_sha256",
    "source_manifest_file_sha256",
    "source_manifest_sha256",
    "lockfile_sha256",
    "model_roots_sha256",
    "screen_runtime_implementation_sha256",
    "consumer_runtime_implementation_sha256",
    "queue_source_sha256",
    "selector_source_sha256",
    "builder_source_sha256",
    "helper_source_sha256",
)


@dataclass(frozen=True)
class MatchedControllerManifests:
    """The two manifests and the identity that binds their controller data."""

    update_stride: int
    source_unit_ids: dict[str, str]
    identity: dict[str, Any]
    trace: ExperimentManifest
    l3_phase2: ExperimentManifest


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"stride selection {field} must be an object")
    return value


def exact_runtime_transition_authorization(
    *,
    screen_sha256: str,
    consumer_sha256: str,
    changed_files: list[str],
    added_files: list[str],
    removed_files: list[str],
    locked_reference_changed: bool,
) -> str | None:
    """Return a reviewed transition id only for one exact fingerprint pair.

    Aggregate SHA-256 equality alone would bind the pair, but checking the
    declared structural delta as well prevents an accidentally misdocumented
    allow-list entry from presenting semantic drift as measurement-only.
    """

    record = EXACT_RUNTIME_TRANSITION_AUTHORIZATIONS.get(
        (screen_sha256, consumer_sha256)
    )
    if record is None:
        return None
    expected = {
        "changed_files": changed_files,
        "added_files": added_files,
        "removed_files": removed_files,
        "locked_reference_changed": locked_reference_changed,
    }
    if set(record) != ({"id"} | set(expected)) or any(
        record.get(field) != value for field, value in expected.items()
    ):
        return None
    transition_id = record.get("id")
    if not isinstance(transition_id, str) or not transition_id:
        return None
    return transition_id


def _winner(
    selection: Mapping[str, Any],
    key: str,
    *,
    family: str,
    method: str,
) -> Mapping[str, Any]:
    winners = _mapping(selection.get("winners"), field="winners")
    row = _mapping(winners.get(key), field=f"winners.{key}")
    _require(row.get("family") == family, f"{key} has the wrong family")
    _require(row.get("method") == method, f"{key} has the wrong method")
    _require(row.get("eligible") is True, f"{key} is not eligible")
    stride = row.get("stride")
    _require(
        isinstance(stride, int)
        and not isinstance(stride, bool)
        and stride in SCREEN_STRIDES,
        f"{key} has an unsupported stride",
    )
    unit_id = row.get("unit_id")
    _require(
        isinstance(unit_id, str) and len(unit_id) == 64,
        f"{key} lacks a 64-character unit_id",
    )
    return row


def _canonical_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    canonical = json.loads(canonical_json(dict(selection)))
    version = canonical.get("schema_version")
    winners = _mapping(canonical.get("winners"), field="winners")
    if version == 2:
        _require("tts_best" not in winners, "schema-v2 contains legacy tts_best")
        return canonical
    _require(version == 1, "unsupported selection schema")
    legacy = _mapping(winners.get("tts_best"), field="winners.tts_best")
    # Schema v1 had one TTS role.  Preserve its attested meaning for both v2
    # roles rather than retrospectively selecting a different unit.
    engineering = legacy
    canonical["schema_version"] = 2
    canonical["selection_rule"]["id"] = SELECTION_RULE
    canonical["winners"] = {
        "tts_acceptance_best": legacy,
        "tts_engineering_best": engineering,
        "l0_best": winners.get("l0_best"),
        "same_stride_tts_for_l0": winners.get("same_stride_tts_for_l0"),
    }
    confirmation = canonical.get("confirmation_unit_ids")
    if isinstance(confirmation, list) and confirmation:
        ordered = [confirmation[0]]
        for row in canonical["winners"].values():
            if isinstance(row, Mapping) and row.get("unit_id") not in ordered:
                ordered.append(row["unit_id"])
        canonical["confirmation_unit_ids"] = ordered
    return canonical


def _validate_selection(
    selection: Mapping[str, Any],
    *,
    allow_l0_not_superior: bool = False,
) -> dict[str, Mapping[str, Any]]:
    """Validate a screen selection before it can seed controller manifests.

    ``allow_l0_not_superior`` admits a screen that resolved every candidate role
    but did not establish L0 over the acceptance-best TTS.  That ordering is the
    hypothesis the controller phase exists to test, so requiring it as an entry
    condition would make the L1/L2 oracle ceiling permanently unmeasurable.
    Every candidate role, safety column, cardinality and rule check below still
    applies, and no downstream acceptance or utility gate is affected.
    """

    selection = _canonical_selection(selection)
    _require(selection.get("schema_version") == 2, "unsupported selection schema")
    if allow_l0_not_superior:
        expected_status = "scientifically_blocked"
        expected_objective = False
    else:
        expected_status = "winner_selected"
        expected_objective = True
    _require(
        selection.get("status") == expected_status,
        "selection has no winner",
    )
    _require(
        selection.get("scope") == "candidate_screen_only_no_claim",
        "selection scope mismatch",
    )
    _require(
        selection.get("objective_screen_pass") is expected_objective,
        "selection objective did not pass",
    )
    rule = _mapping(selection.get("selection_rule"), field="selection_rule")
    _require(rule.get("id") == SELECTION_RULE, "selection rule mismatch")
    cardinality = _mapping(selection.get("cardinality"), field="cardinality")
    _require(
        cardinality.get("contexts") == [4096, 16384]
        and cardinality.get("strides") == list(SCREEN_STRIDES)
        and cardinality.get("no_stride_pooling") is True,
        "selection cardinality mismatch",
    )

    tts_acceptance_best = _winner(
        selection, "tts_acceptance_best", family="tts", method="tts"
    )
    tts_engineering_best = _winner(
        selection, "tts_engineering_best", family="tts", method="tts"
    )
    l0_best = _winner(
        selection, "l0_best", family="l0", method="naive_async"
    )
    same_tts = _winner(
        selection,
        "same_stride_tts_for_l0",
        family="tts",
        method="tts",
    )
    candidates = _mapping(selection.get("candidates"), field="candidates")
    for role, family, method, selected_row in (
        ("tts_acceptance_best", "tts", "tts", tts_acceptance_best),
        ("tts_engineering_best", "tts", "tts", tts_engineering_best),
        ("l0_best", "l0", "naive_async", l0_best),
    ):
        rows = candidates.get(family)
        _require(
            isinstance(rows, list) and len(rows) == len(SCREEN_STRIDES),
            f"selection candidates.{family} must contain four rows",
        )
        by_stride: dict[int, Mapping[str, Any]] = {}
        for value in rows:
            row = _mapping(value, field=f"candidates.{family}[]")
            stride = row.get("stride")
            _require(
                row.get("family") == family
                and row.get("method") == method
                and isinstance(stride, int)
                and not isinstance(stride, bool)
                and stride in SCREEN_STRIDES,
                f"selection candidates.{family} identity mismatch",
            )
            _require(
                stride not in by_stride,
                f"selection candidates.{family} repeats stride {stride}",
            )
            by_stride[stride] = row
        _require(
            set(by_stride) == set(SCREEN_STRIDES),
            f"selection candidates.{family} stride coverage mismatch",
        )
        candidate = by_stride[int(selected_row["stride"])]
        if role == "tts_engineering_best" and selected_row.get(
            "engineering_eligible"
        ) is False:
            fallback = dict(selected_row)
            reason = fallback.pop("engineering_fallback_reason", None)
            _require(
                fallback == candidate
                and reason
                == "no_eligible_engineering_candidate_used_tts_acceptance_best"
                and selected_row["unit_id"] == tts_acceptance_best["unit_id"],
                "TTS engineering fallback does not match acceptance-best",
            )
        else:
            _require(
                candidate == selected_row,
                f"{family} winner does not match its candidate row",
            )
    _require(
        l0_best["stride"] == same_tts["stride"],
        "L0 winner and matched TTS use different strides",
    )
    _require(
        l0_best["unit_id"] != same_tts["unit_id"],
        "L0 winner and matched TTS cannot share a unit_id",
    )
    confirmation = selection.get("confirmation_unit_ids")
    _require(
        isinstance(confirmation, list)
        and all(isinstance(value, str) for value in confirmation),
        "confirmation_unit_ids must be a string list",
    )
    _require(
        {
            str(tts_acceptance_best["unit_id"]),
            str(tts_engineering_best["unit_id"]),
            str(l0_best["unit_id"]),
            str(same_tts["unit_id"]),
        }.issubset(set(confirmation)),
        "selection confirmation units omit a winner",
    )
    return {
        "tts_acceptance_best": tts_acceptance_best,
        "tts_engineering_best": tts_engineering_best,
        "l0_best": l0_best,
        "same_tts": same_tts,
    }


def _validate_screen_unit(unit: RunUnit) -> None:
    _require(unit.phase == SOURCE_MANIFEST_NAME, "source unit phase mismatch")
    _require(unit.model_pair == "qwen3_4b_dflash16", "source model pair mismatch")
    _require(unit.dataset == "livecodebench", "source dataset mismatch")
    _require(
        unit.prompt_subset == "p5_ctx_4096-16384",
        "source prompt subset mismatch",
    )
    _require(unit.seed == 0, "source seed mismatch")
    _require(unit.lifecycle == "stream", "source lifecycle mismatch")
    _require(unit.sampling_profile == "greedy_t0", "source sampling mismatch")
    _require(unit.trainable_scope == "tail_lora", "source tail layout mismatch")
    _require(unit.weight_update_mode == "lora", "source update mode mismatch")
    _require(unit.parameter_scope == "tail", "source parameter scope mismatch")
    _require(unit.adapter_rank == 16, "source LoRA rank mismatch")
    _require(unit.logical_delay == 0, "source logical delay mismatch")
    _require(unit.concurrency == 20, "source concurrency mismatch")
    _require(unit.transport_variant is None, "source transport variant mismatch")
    _require(unit.required and not unit.allow_resource_skip, "source unit is optional")


def _validate_source_manifest(source: ExperimentManifest) -> dict[tuple[str, int], RunUnit]:
    _require(source.name == SOURCE_MANIFEST_NAME, "source manifest name mismatch")
    _require(source.phase == SOURCE_MANIFEST_NAME, "source manifest phase mismatch")
    _require(source.profile == "local_1x96gb", "source profile mismatch")
    _require(
        source.engine_params.get("prompt_limit") == 40
        and source.engine_params.get("prompt_offset") == 0,
        "source prompt slice mismatch",
    )
    try:
        lr = float(source.engine_params["lr"])
        weight_decay = float(source.engine_params["weight_decay"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError("source manifest lacks the matched optimizer tier") from exc
    _require(
        math.isclose(lr, MATCHED_LR, rel_tol=0.0, abs_tol=0.0),
        "source learning rate mismatch",
    )
    _require(
        math.isclose(
            weight_decay, MATCHED_WEIGHT_DECAY, rel_tol=0.0, abs_tol=0.0
        ),
        "source weight decay mismatch",
    )
    allocator = source.engine_params.get("pytorch_cuda_alloc_conf")
    _require(
        isinstance(allocator, str)
        and allocator.strip()
        and allocator == MATCHED_PYTORCH_CUDA_ALLOC_CONF,
        "source CUDA allocator contract mismatch",
    )

    index: dict[tuple[str, int], RunUnit] = {}
    for unit in source.units:
        _validate_screen_unit(unit)
        key = (unit.method, unit.stride)
        _require(key not in index, f"duplicate source method/stride cell: {key}")
        index[key] = unit
    expected = {
        ("static", 1),
        *((method, stride) for method in ("tts", "naive_async") for stride in SCREEN_STRIDES),
    }
    _require(set(index) == expected, "source manifest is not the nine-unit stride screen")
    for (method, _stride), unit in index.items():
        expected_contention = "realistic_async" if method == "naive_async" else "none"
        _require(
            unit.contention_condition == expected_contention,
            f"source contention mismatch for {method}/stride={unit.stride}",
        )
    return index


def _validated_bindings(
    bindings: Mapping[str, Any],
    *,
    source_manifest: ExperimentManifest,
    static_unit_id: str,
) -> dict[str, Any]:
    _require(bindings.get("schema_version") == 1, "binding schema mismatch")
    for field in _SHA_BINDINGS:
        value = bindings.get(field)
        _require(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value),
            f"binding {field} is not a lowercase SHA-256",
        )
    _require(
        bindings["source_manifest_sha256"] == source_manifest.content_sha256(),
        "binding source manifest content hash mismatch",
    )
    _require(
        bindings.get("source_static_unit_id") == static_unit_id,
        "binding source Static unit mismatch",
    )
    _require(
        bindings.get("pytorch_cuda_alloc_conf")
        == MATCHED_PYTORCH_CUDA_ALLOC_CONF,
        "binding CUDA allocator mismatch",
    )
    revisions = _mapping(bindings.get("model_revisions"), field="model_revisions")
    _require(
        set(revisions) == {"target", "drafter", "tokenizer"}
        and all(
            isinstance(value, str)
            and len(value) == 40
            and all(char in "0123456789abcdef" for char in value)
            for value in revisions.values()
        ),
        "binding model revisions are incomplete",
    )
    runtimes = {}
    for name in (
        "screen_runtime_implementation_fingerprint",
        "runtime_implementation_fingerprint",
    ):
        runtime = _mapping(bindings.get(name), field=name)
        _require(
            runtime.get("schema_version") == 1
            and isinstance(runtime.get("sha256"), str)
            and len(runtime["sha256"]) == 64,
            f"binding {name} is invalid",
        )
        runtime_body = dict(runtime)
        runtime_sha256 = runtime_body.pop("sha256", None)
        _require(
            sha256_json(runtime_body) == runtime_sha256,
            f"binding {name} digest mismatch",
        )
        runtimes[name] = runtime
    _require(
        bindings["screen_runtime_implementation_sha256"]
        == runtimes["screen_runtime_implementation_fingerprint"]["sha256"]
        and bindings["consumer_runtime_implementation_sha256"]
        == runtimes["runtime_implementation_fingerprint"]["sha256"],
        "binding runtime summary hashes differ from their fingerprints",
    )
    transition = _mapping(
        bindings.get("runtime_transition"), field="runtime_transition"
    )
    screen_files = runtimes[
        "screen_runtime_implementation_fingerprint"
    ].get("files")
    consumer_files = runtimes["runtime_implementation_fingerprint"].get(
        "files"
    )
    _require(
        isinstance(screen_files, Mapping) and isinstance(consumer_files, Mapping),
        "binding runtime transition file maps are invalid",
    )
    screen_runtime = runtimes["screen_runtime_implementation_fingerprint"]
    consumer_runtime = runtimes["runtime_implementation_fingerprint"]
    common_files = set(screen_files) & set(consumer_files)
    changed_files = sorted(
        path
        for path in common_files
        if screen_files[path] != consumer_files[path]
    )
    added_files = sorted(set(consumer_files) - set(screen_files))
    removed_files = sorted(set(screen_files) - set(consumer_files))
    locked_reference_changed = screen_runtime.get(
        "locked_reference"
    ) != consumer_runtime.get("locked_reference")
    equal_runtime = screen_runtime == consumer_runtime
    if equal_runtime:
        authorization_id = "identical_runtime"
    else:
        authorization_id = exact_runtime_transition_authorization(
            screen_sha256=screen_runtime["sha256"],
            consumer_sha256=consumer_runtime["sha256"],
            changed_files=changed_files,
            added_files=added_files,
            removed_files=removed_files,
            locked_reference_changed=locked_reference_changed,
        )
    _require(
        authorization_id is not None
        and transition.get("schema_version") == 2
        and transition.get("screen_sha256")
        == bindings["screen_runtime_implementation_sha256"]
        and transition.get("consumer_sha256")
        == bindings["consumer_runtime_implementation_sha256"]
        and transition.get("changed_files") == changed_files
        and transition.get("added_files") == added_files
        and transition.get("removed_files") == removed_files
        and transition.get("locked_reference_changed")
        is locked_reference_changed
        and transition.get("authorization_id") == authorization_id
        and transition.get("authorization_basis")
        == "exact_runtime_fingerprint_pair"
        and transition.get("equal") is equal_runtime
        and transition.get("screen_measurements_reusable") is equal_runtime
        and transition.get("selection_reuse_only") is (not equal_runtime)
        and transition.get("scientific_equivalence_claim") is False
        and transition.get("requires_matched_confirmation")
        is (not equal_runtime),
        "binding runtime transition mismatch",
    )
    contention = _mapping(
        bindings.get("contention_mapping"), field="contention_mapping"
    )
    _require(
        contention
        == {
            "phase1_naive_async": "realistic_async",
            "phase1_tts": "none",
            "phase2_lc_transport": "realistic_async",
        },
        "binding contention mapping mismatch",
    )
    # Freeze nested caller-owned mappings before placing them in identities or
    # engine parameters.
    return json.loads(canonical_json(dict(bindings)))


def _matched_unit(unit: RunUnit, *, phase: str, stride: int) -> RunUnit:
    return replace(unit, phase=phase, stride=stride)


def _cell(unit: RunUnit) -> tuple[Any, ...]:
    """Cell identity shared by phase-1 TTS and evaluation-only phase-2 L3."""

    return (
        unit.model_pair,
        unit.dataset,
        unit.prompt_subset,
        unit.seed,
        unit.lifecycle,
        unit.sampling_profile,
        unit.trainable_scope,
        unit.stride,
        unit.logical_delay,
        unit.concurrency,
        unit.adapter_rank,
        unit.transport_variant,
        unit.parameter_scope,
        unit.parameter_allowlist,
        unit.required,
        unit.allow_resource_skip,
    )


def build_matched_controller_manifests(
    selection: Mapping[str, Any],
    source_manifest: ExperimentManifest,
    *,
    bindings: Mapping[str, Any],
    allow_l0_not_superior: bool = False,
) -> MatchedControllerManifests:
    """Build same-stride trace and L3 manifests from a validated screen winner."""

    selection = _canonical_selection(selection)
    selected = _validate_selection(
        selection, allow_l0_not_superior=allow_l0_not_superior
    )
    source_index = _validate_source_manifest(source_manifest)
    frozen_bindings = _validated_bindings(
        bindings,
        source_manifest=source_manifest,
        static_unit_id=source_index[("static", 1)].unit_id,
    )
    for label, row in selected.items():
        method = "naive_async" if label == "l0_best" else "tts"
        source = source_index[(method, int(row["stride"]))]
        _require(
            source.unit_id == row["unit_id"],
            f"{label} unit_id does not bind to the source manifest",
        )
    expected_confirmation = []
    for unit_id in (
        source_index[("static", 1)].unit_id,
        str(selected["tts_acceptance_best"]["unit_id"]),
        str(selected["tts_engineering_best"]["unit_id"]),
        str(selected["l0_best"]["unit_id"]),
        str(selected["same_tts"]["unit_id"]),
    ):
        if unit_id not in expected_confirmation:
            expected_confirmation.append(unit_id)
    _require(
        selection.get("confirmation_unit_ids") == expected_confirmation,
        "selection confirmation unit ordering or coverage mismatch",
    )

    stride = int(selected["l0_best"]["stride"])
    source_unit_ids = {
        "static": source_index[("static", 1)].unit_id,
        "tts_acceptance_best": str(
            selected["tts_acceptance_best"]["unit_id"]
        ),
        "tts_engineering_best": str(
            selected["tts_engineering_best"]["unit_id"]
        ),
        "l0_best": str(selected["l0_best"]["unit_id"]),
        "same_stride_tts_for_l0": str(selected["same_tts"]["unit_id"]),
    }
    identity_body = {
        "schema_version": 2,
        "model_pair": "qwen3_4b_dflash16",
        "weight_update_mode": "lora",
        "tail_layout_mode": "tail_lora",
        "parameter_scope": "tail",
        "adapter_rank": 16,
        "optimizer": "adamw",
        "lr": MATCHED_LR,
        "weight_decay": MATCHED_WEIGHT_DECAY,
        "pytorch_cuda_alloc_conf": MATCHED_PYTORCH_CUDA_ALLOC_CONF,
        "sampling_profile": "greedy_t0",
        "lifecycle": "stream",
        "update_stride": stride,
        "source_unit_ids": source_unit_ids,
        "bindings": frozen_bindings,
    }
    identity = {**identity_body, "sha256": sha256_json(identity_body)}
    slug = (
        f"s{stride}_lora_r16_lr1e4_wd1e2_"
        f"b{identity['sha256'][:12]}_v1"
    )
    trace_name = f"p5_priority_dflash_paired_trace_{slug}"
    l3_name = f"p5_priority_dflash_l3_evaluation_{slug}"

    trace_template = p5_priority_dflash_paired_trace_manifest()
    trace_engine = {
        **trace_template.engine_params,
        "lr": MATCHED_LR,
        "weight_decay": MATCHED_WEIGHT_DECAY,
        "pytorch_cuda_alloc_conf": MATCHED_PYTORCH_CUDA_ALLOC_CONF,
        "model_roots_sha256": frozen_bindings["model_roots_sha256"],
        "runtime_implementation_fingerprint": frozen_bindings[
            "runtime_implementation_fingerprint"
        ],
        "matched_controller_identity_sha256": identity["sha256"],
    }
    trace = replace(
        trace_template,
        name=trace_name,
        phase=trace_name,
        description=(
            "Winner-matched DFlash4B L0/TTS controller traces using the "
            f"attested same-stride candidate at stride {stride}, tail-LoRA "
            "rank 16, AdamW lr=1e-4 and weight_decay=1e-2."
        ),
        lockfile_sha256=frozen_bindings["lockfile_sha256"],
        engine_params=trace_engine,
        units=[
            _matched_unit(unit, phase=trace_name, stride=stride)
            for unit in trace_template.units
        ],
    )

    l3_template = p5_priority_dflash_l3_evaluation_manifest()
    l3_engine = {
        **l3_template.engine_params,
        "lr": MATCHED_LR,
        "weight_decay": MATCHED_WEIGHT_DECAY,
        "pytorch_cuda_alloc_conf": MATCHED_PYTORCH_CUDA_ALLOC_CONF,
        "model_roots_sha256": frozen_bindings["model_roots_sha256"],
        "runtime_implementation_fingerprint": frozen_bindings[
            "runtime_implementation_fingerprint"
        ],
        "matched_controller_identity_sha256": identity["sha256"],
    }
    l3 = replace(
        l3_template,
        name=l3_name,
        phase=l3_name,
        description=(
            "Evaluation-only phase-2 DFlash4B L3 cells exactly mirroring the "
            f"winner-matched phase-1 TTS cells at stride {stride}, tail-LoRA "
            "rank 16, AdamW lr=1e-4 and weight_decay=1e-2."
        ),
        lockfile_sha256=frozen_bindings["lockfile_sha256"],
        engine_params=l3_engine,
        units=[
            _matched_unit(unit, phase=l3_name, stride=stride)
            for unit in l3_template.units
        ],
    )

    trace_tts_cells = sorted(_cell(unit) for unit in trace.units if unit.method == "tts")
    l3_cells = sorted(_cell(unit) for unit in l3.units)
    _require(trace_tts_cells == l3_cells, "phase-2 L3 cells do not mirror phase-1 TTS")
    _require(
        {unit.method for unit in trace.units} == {"naive_async", "tts"},
        "trace manifest method set drifted",
    )
    _require(
        {unit.method for unit in l3.units} == {"lc_transport"}
        and l3.engine_params.get("l3_evaluation_only") is True,
        "phase-2 manifest is not evaluation-only lc_transport",
    )
    trace_normalized_engine = {
        key: value
        for key, value in trace.engine_params.items()
        if key not in {"trace_producer_methods", "l3_evaluation_only"}
    }
    l3_normalized_engine = {
        key: value
        for key, value in l3.engine_params.items()
        if key not in {"trace_producer_methods", "l3_evaluation_only"}
    }
    _require(
        trace_normalized_engine == l3_normalized_engine,
        "phase-1/phase-2 engine parameters differ outside allowed roles",
    )
    _require(
        frozen_bindings["contention_mapping"]
        == {
            "phase1_naive_async": next(
                unit.contention_condition
                for unit in trace.units
                if unit.method == "naive_async"
            ),
            "phase1_tts": next(
                unit.contention_condition
                for unit in trace.units
                if unit.method == "tts"
            ),
            "phase2_lc_transport": next(
                unit.contention_condition for unit in l3.units
            ),
        },
        "published contention mapping differs from the bound exception",
    )
    return MatchedControllerManifests(
        update_stride=stride,
        source_unit_ids=source_unit_ids,
        identity=identity,
        trace=trace,
        l3_phase2=l3,
    )
