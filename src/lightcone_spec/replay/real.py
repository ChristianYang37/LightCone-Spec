"""Load bounded real-model replay labels and fit/gate controller artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from lightcone_spec.exit_codes import ArtifactValidationFailure
from lightcone_spec.locking.hashing import sha256_file, sha256_json
from lightcone_spec.replay.counterfactual import ReplayUpdateRecord
from lightcone_spec.replay.pipeline import fit_replay_pipeline
from lightcone_spec.replay.splits import split_of_group
from lightcone_spec.statistics.bootstrap import B_DEFAULT
from lightcone_spec.trajectory.features import UpdateFeatureRow, design_matrix
from lightcone_spec.trajectory.zvector import default_zvectorizer


def _owning_runtime_config_path(path: Path, root: Path) -> Path | None:
    """Find the nearest run config without escaping the requested root."""

    root = root.resolve()
    for parent in path.resolve().parents:
        candidate = parent / "adaptation.runtime.yaml"
        if candidate.is_file():
            return candidate
        if parent == root:
            break
    return None


def _trace_owner(path: Path, root: Path) -> dict[str, object] | None:
    """Bind a trace shard to its immutable run phase and prompt window."""

    root = root.resolve()
    for parent in path.resolve().parents:
        candidate = parent / "manifest.json"
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ArtifactValidationFailure(
                    f"invalid owning run manifest for replay trace: {candidate}"
                ) from exc
            engine = payload.get("engine_params")
            if not isinstance(engine, dict):
                raise ArtifactValidationFailure(
                    f"owning replay run lacks engine_params: {candidate}"
                )
            phase = payload.get("phase")
            offset = engine.get("prompt_offset")
            limit = engine.get("prompt_limit")
            explicit_phase2 = bool(
                engine.get("phase2_tts_reference_only") is True
                or engine.get("l3_evaluation_only") is True
            )
            if (
                not isinstance(phase, str)
                or not phase
                or isinstance(offset, bool)
                or not isinstance(offset, int)
                or isinstance(limit, bool)
                or not isinstance(limit, int)
                or offset < 0
                or limit <= 0
            ):
                if not explicit_phase2:
                    return None
                raise ArtifactValidationFailure(
                    f"owning replay phase/window identity is incomplete: {candidate}"
                )
            if engine.get("phase2_tts_reference_only") is True:
                role = "phase2_tts_reference"
                evaluation_only = True
            elif engine.get("l3_evaluation_only") is True:
                role = "phase2_l3"
                evaluation_only = True
            else:
                role = "phase1_producer"
                evaluation_only = False
            return {
                "phase": phase,
                "prompt_offset": offset,
                "prompt_limit": limit,
                "role": role,
                "evaluation_only": evaluation_only,
                "manifest_path": str(candidate.resolve()),
                "manifest_sha256": sha256_file(candidate),
            }
        if parent == root:
            break
    return None


def _select_pair_owned_paths(
    root: Path,
    paths: list[Path],
    model_pair_id: str | None,
    *,
    require_owner: bool,
    allow_empty: bool = False,
) -> list[Path]:
    if model_pair_id is None and not require_owner:
        return paths
    from lightcone_spec.config.loader import load_adaptation_config

    selected: list[Path] = []
    for path in paths:
        config_path = _owning_runtime_config_path(path, root)
        if config_path is None:
            if require_owner or model_pair_id is not None:
                raise ArtifactValidationFailure(
                    f"real replay evidence has no owning adaptation.runtime.yaml: {path}"
                )
            selected.append(path)
            continue
        config = load_adaptation_config(config_path)
        if model_pair_id is None or config.model.pair_id == model_pair_id:
            selected.append(path)
    if model_pair_id is not None and not selected and not allow_empty:
        raise ArtifactValidationFailure(
            f"no real replay evidence for model pair {model_pair_id!r} under {root}"
        )
    return selected


def _controller_runtime_identity(
    root: Path, model_pair_id: str | None = None
) -> tuple[dict, str]:
    from lightcone_spec.config.loader import load_adaptation_config
    from lightcone_spec.methods.registry import controller_runtime_identity

    identities: dict[str, dict] = {}
    parameter_layouts: set[str | None] = set()
    indexes = _select_pair_owned_paths(
        root,
        sorted(root.rglob("index*.jsonl")),
        model_pair_id,
        require_owner=True,
    )
    for index in indexes:
        config_path = _owning_runtime_config_path(index, root)
        assert config_path is not None
        config = load_adaptation_config(config_path)
        identity = controller_runtime_identity(config)
        identities[sha256_json(identity)] = identity
        for line in index.read_text().splitlines():
            if line.strip():
                parameter_layouts.add(
                    json.loads(line).get("parameter_layout_sha256")
                )
    if len(identities) != 1:
        raise ArtifactValidationFailure(
            "real replay mixes incompatible candidate/runtime identities: "
            f"{sorted(identities)}"
        )
    if None in parameter_layouts or len(parameter_layouts) != 1:
        raise ArtifactValidationFailure(
            "real replay lacks one unambiguous parameter-layout hash; "
            "recapture traces with the active tail manager"
        )
    return next(iter(identities.values())), next(iter(parameter_layouts))


def load_real_replay_records(
    root: str | Path, *, model_pair_id: str | None = None
) -> list[ReplayUpdateRecord]:
    root = Path(root)
    indexes = _select_pair_owned_paths(
        root,
        sorted(root.rglob("index*.jsonl")),
        model_pair_id,
        require_owner=model_pair_id is not None,
    )
    if not indexes:
        raise ArtifactValidationFailure(f"no real replay index found under {root}")
    records: list[ReplayUpdateRecord] = []
    for index in indexes:
        owner_config = None
        owner_config_path = _owning_runtime_config_path(index, root)
        if owner_config_path is not None:
            from lightcone_spec.config.loader import load_adaptation_config

            owner_config = load_adaptation_config(owner_config_path)
        trace_owner = _trace_owner(index, root)
        for line in index.read_text().splitlines():
            item = json.loads(line)
            path = index.parent / item["path"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise ArtifactValidationFailure(
                    f"real replay shard missing or hash-drifted: {path}"
                )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            schema_version = payload.get("schema_version")
            provenance_method = payload.get("provenance_method")
            evaluation_pair_id = None
            trace_stage_index = None
            trace_stage_count = None
            trace_capture_sampling = None
            evaluation_concurrency = None
            fresh_gradient_scope = None
            transported_candidate_utility = None
            paired_l2_utility = None
            transport_evaluation_contract = None
            transport_variant = None
            transport_map_sha256 = None
            if schema_version in (2, 3):
                missing = [
                    name
                    for name in (
                        "provenance_method",
                        "controller_label_source",
                        "actual_published_utility",
                        "full_candidate_utility",
                        "candidate_arrival_round",
                        "actual_arrival_round",
                        "paired_tts_barrier",
                        "prefix_feature_exact",
                    )
                    if name not in payload
                ]
                if missing:
                    raise ArtifactValidationFailure(
                        f"replay schema v{schema_version} lacks {missing} in {path}"
                    )
                if payload["controller_label_source"] != "full_candidate_utility":
                    raise ArtifactValidationFailure(
                        "replay schema v2 controller label is not the raw full "
                        f"candidate utility in {path}"
                    )
                actual_utility = float(payload["actual_published_utility"])
                controller_utility = float(payload["full_candidate_utility"])
                candidate_arrival_round = int(payload["candidate_arrival_round"])
                actual_arrival_round = int(payload["actual_arrival_round"])
                paired_tts_barrier = payload["paired_tts_barrier"]
                if not isinstance(paired_tts_barrier, bool):
                    raise ArtifactValidationFailure(
                        f"paired_tts_barrier must be boolean in {path}"
                    )
                if actual_arrival_round < candidate_arrival_round:
                    raise ArtifactValidationFailure(
                        "actual TTS arrival precedes candidate-ready arrival in "
                        f"{path}"
                    )
                source_round = int(payload["source_round"])
                minimum_candidate_round = source_round + 1
                if owner_config is not None:
                    minimum_candidate_round += (
                        owner_config.async_.logical_delay_rounds
                    )
                if candidate_arrival_round < minimum_candidate_round:
                    raise ArtifactValidationFailure(
                        "candidate arrival violates the pipeline/logical-delay "
                        f"lower bound in {path}"
                    )
                if not np.isclose(
                    float(payload["round_delay"]),
                    candidate_arrival_round - source_round,
                ):
                    raise ArtifactValidationFailure(
                        "round_delay disagrees with candidate arrival in "
                        f"{path}"
                    )
                if int(payload["arrival_round"]) != candidate_arrival_round:
                    raise ArtifactValidationFailure(
                        f"schema-v{schema_version} arrival_round must denote "
                        "candidate-ready "
                        f"arrival in {path}"
                    )
                if paired_tts_barrier and provenance_method != "tts":
                    raise ArtifactValidationFailure(
                        "paired_tts_barrier evidence must come from the TTS "
                        f"publisher in {path}"
                    )
                if (
                    paired_tts_barrier
                    and owner_config is not None
                    and actual_arrival_round % owner_config.update_stride != 0
                ):
                    raise ArtifactValidationFailure(
                        "paired TTS actual arrival is not a fixed-stride "
                        f"barrier in {path}"
                    )
                if (
                    not paired_tts_barrier
                    and actual_arrival_round != candidate_arrival_round
                ):
                    raise ArtifactValidationFailure(
                        "non-paired replay cannot claim distinct candidate and "
                        f"actual arrivals in {path}"
                    )
                if schema_version == 3:
                    missing_publish_features = [
                        name
                        for name in (
                            "source_prefix_len",
                            "source_acceptance",
                            "source_training_loss",
                            "source_grad_norm",
                            "source_z_raw",
                            "arrival_z_raw",
                            "fresh_gradient_scope",
                        )
                        if name not in payload
                    ]
                    if missing_publish_features:
                        raise ArtifactValidationFailure(
                            "replay schema v3 lacks exact publish-time controller "
                            f"features {missing_publish_features} in {path}"
                        )
                    evaluation_pair_id = payload.get("evaluation_pair_id")
                    if (
                        not isinstance(evaluation_pair_id, str)
                        or not evaluation_pair_id
                    ):
                        raise ArtifactValidationFailure(
                            "replay schema v3 lacks a non-empty "
                            f"evaluation_pair_id in {path}"
                        )
                    try:
                        trace_stage_index = int(payload["trace_stage_index"])
                        trace_stage_count = int(payload["trace_stage_count"])
                        trace_capture_sampling = str(
                            payload["trace_capture_sampling"]
                        )
                        evaluation_concurrency = int(
                            payload["evaluation_concurrency"]
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ArtifactValidationFailure(
                            f"replay schema v3 lacks a valid trace stage in {path}"
                        ) from exc
                    if (
                        trace_stage_count <= 0
                        or not 0 <= trace_stage_index < trace_stage_count
                        or trace_capture_sampling not in ("first", "staged")
                        or evaluation_concurrency <= 0
                    ):
                        raise ArtifactValidationFailure(
                            f"replay schema v3 has an invalid trace stage in {path}"
                        )
                    fresh_gradient_scope = str(payload["fresh_gradient_scope"])
                    if fresh_gradient_scope not in (
                        "writer_rank_local_v1",
                        "tp_consensus_v1",
                    ):
                        raise ArtifactValidationFailure(
                            f"replay schema v3 has an unknown fresh-gradient scope in {path}"
                        )
                    if (
                        owner_config is not None
                        and owner_config.runtime.tensor_parallel_size > 1
                        and fresh_gradient_scope != "tp_consensus_v1"
                    ):
                        raise ArtifactValidationFailure(
                            "TP real replay mixes a consensus stale gradient with "
                            f"a rank-local fresh gradient in {path}; capture the "
                            "fresh gradient through the same TP consensus hook"
                        )
                    if provenance_method == "lc_transport":
                        required_l3 = (
                            "transported_candidate_utility",
                            "paired_l2_utility",
                            "transport_evaluation_contract",
                            "transport_variant",
                            "transport_map_sha256",
                        )
                        missing_l3 = [
                            name for name in required_l3 if name not in payload
                        ]
                        if missing_l3:
                            raise ArtifactValidationFailure(
                                f"L3 replay schema v3 lacks {missing_l3} in {path}"
                            )
                        transported_candidate_utility = float(
                            payload["transported_candidate_utility"]
                        )
                        paired_l2_utility = float(payload["paired_l2_utility"])
                        transport_evaluation_contract = str(
                            payload["transport_evaluation_contract"]
                        )
                        transport_variant = str(payload["transport_variant"])
                        transport_map_sha256 = str(
                            payload["transport_map_sha256"]
                        )
                        if transport_evaluation_contract != (
                            "joint_fisher_transport_adamw_damping_v1"
                        ):
                            raise ArtifactValidationFailure(
                                "L3 replay did not execute the complete joint "
                                f"Fisher/transport/AdamW/damping path in {path}"
                            )
                        if transport_variant != "joint":
                            raise ArtifactValidationFailure(
                                "L3 replay is not the joint transport variant "
                                f"in {path}"
                            )
                        if len(transport_map_sha256) != 64 or any(
                            char not in "0123456789abcdef"
                            for char in transport_map_sha256
                        ):
                            raise ArtifactValidationFailure(
                                "L3 replay has an invalid transport-map hash "
                                f"in {path}"
                            )
                        if not np.isclose(
                            transported_candidate_utility,
                            actual_utility,
                            rtol=1e-6,
                            atol=1e-8,
                        ):
                            raise ArtifactValidationFailure(
                                "L3 transported utility is not the actual "
                                f"published utility in {path}"
                            )
            elif schema_version == 1:
                # Schema v1 stored the utility of whichever policy happened to
                # publish.  It is a valid raw-candidate label only for L0, whose
                # publication is exactly the full candidate by definition.
                if provenance_method != "naive_async":
                    raise ArtifactValidationFailure(
                        "legacy replay schema v1 is policy-label ambiguous; only "
                        "provenance_method='naive_async' can be migrated safely: "
                        f"{path}"
                    )
                actual_utility = float(payload["utility"])
                controller_utility = float(
                    payload.get("full_candidate_utility", actual_utility)
                )
                candidate_arrival_round = int(payload["arrival_round"])
                actual_arrival_round = int(payload["arrival_round"])
                paired_tts_barrier = False
                if not np.isclose(
                    actual_utility,
                    controller_utility,
                    rtol=1e-5,
                    atol=1e-7,
                ):
                    raise ArtifactValidationFailure(
                        "legacy naive_async replay has inconsistent actual/full "
                        f"candidate utility in {path}"
                    )
            else:
                raise ArtifactValidationFailure(
                    f"unsupported real replay schema {schema_version!r} in {path}"
                )
            if payload.get("prefix_feature_exact") is not True:
                raise ArtifactValidationFailure(
                    "real controller replay requires exact prefix/token-delay "
                    f"features; missing or approximate evidence in {path}"
                )
            if not np.isfinite(actual_utility) or not np.isfinite(controller_utility):
                raise ArtifactValidationFailure(
                    f"real replay contains non-finite utility in {path}"
                )
            for name in (
                "oracle_l1_utility",
                "oracle_l2_utility",
                "oracle_l2_kappa",
            ):
                if name in payload and not np.isfinite(float(payload[name])):
                    raise ArtifactValidationFailure(
                        f"real replay contains non-finite {name} in {path}"
                    )
            for name, value in (
                ("transported_candidate_utility", transported_candidate_utility),
                ("paired_l2_utility", paired_l2_utility),
            ):
                if value is not None and not np.isfinite(value):
                    raise ArtifactValidationFailure(
                        f"real replay contains non-finite {name} in {path}"
                    )
            utility_by_kappa = None
            if schema_version in (2, 3) and "utility_by_kappa" in payload:
                raw_grid = payload["utility_by_kappa"]
                if not isinstance(raw_grid, dict) or not raw_grid:
                    raise ArtifactValidationFailure(
                        f"real replay has an invalid kappa utility grid in {path}"
                    )
                try:
                    utility_by_kappa = {
                        float(kappa): float(utility)
                        for kappa, utility in raw_grid.items()
                    }
                except (TypeError, ValueError) as exc:
                    raise ArtifactValidationFailure(
                        f"real replay has a non-numeric kappa grid in {path}"
                    ) from exc
                if (
                    any(
                        not np.isfinite(kappa)
                        or not np.isfinite(utility)
                        or not 0.0 <= kappa <= 1.0
                        for kappa, utility in utility_by_kappa.items()
                    )
                    or 0.0 not in utility_by_kappa
                    or 1.0 not in utility_by_kappa
                ):
                    raise ArtifactValidationFailure(
                        "real replay kappa grid must contain finite utilities "
                        f"at 0 and 1 over [0, 1] in {path}"
                    )
                if not np.isclose(
                    utility_by_kappa[1.0], controller_utility,
                    rtol=1e-5, atol=1e-7,
                ):
                    raise ArtifactValidationFailure(
                        "real replay kappa=1 utility disagrees with the full "
                        f"candidate utility in {path}"
                    )
            oracle_names = (
                "oracle_l1_utility",
                "oracle_l2_utility",
                "oracle_l2_kappa",
            )
            oracle_present = [name in payload for name in oracle_names]
            if any(oracle_present):
                if not all(oracle_present) or utility_by_kappa is None:
                    raise ArtifactValidationFailure(
                        "real replay oracle evidence must include one complete "
                        f"kappa-grid contract in {path}"
                    )
                if not np.isclose(
                    utility_by_kappa[0.0], 0.0, rtol=0.0, atol=1e-7
                ):
                    raise ArtifactValidationFailure(
                        f"real replay kappa=0 utility is not zero in {path}"
                    )
                expected_l1 = max(
                    utility_by_kappa[0.0], utility_by_kappa[1.0]
                )
                expected_l2 = max(utility_by_kappa.values())
                oracle_kappa = float(payload["oracle_l2_kappa"])
                matched_kappas = [
                    kappa
                    for kappa in utility_by_kappa
                    if np.isclose(kappa, oracle_kappa, rtol=0.0, atol=1e-8)
                ]
                if (
                    not np.isclose(
                        float(payload["oracle_l1_utility"]),
                        expected_l1,
                        rtol=1e-5,
                        atol=1e-7,
                    )
                    or not np.isclose(
                        float(payload["oracle_l2_utility"]),
                        expected_l2,
                        rtol=1e-5,
                        atol=1e-7,
                    )
                    or len(matched_kappas) != 1
                    or not np.isclose(
                        utility_by_kappa[matched_kappas[0]],
                        expected_l2,
                        rtol=1e-5,
                        atol=1e-7,
                    )
                ):
                    raise ArtifactValidationFailure(
                        "real replay oracle labels disagree with their captured "
                        f"kappa utility grid in {path}"
                    )
            row = UpdateFeatureRow(
                sequence_id=payload["sequence_id"],
                update_id=payload["update_id"],
                round_delay=payload["round_delay"],
                token_delay=payload["token_delay"],
                wall_us=payload["wall_us"],
                endpoint_distance=payload["endpoint_distance"],
                rho_path=payload["rho_path"],
                parameter_displacement=payload["parameter_displacement"],
                # Controller labels always describe the same-arrival, undamped
                # raw candidate.  The policy's actual published effect is kept
                # separately for systems diagnostics only.
                utility=controller_utility,
                relative_gradient_mismatch=payload[
                    "relative_gradient_mismatch"
                ],
                harmful=int(controller_utility < 0.0),
                source_prefix_len=float(payload.get("source_prefix_len", 0.0)),
                source_acceptance=float(payload.get("source_acceptance", 0.0)),
                source_training_loss=float(
                    payload.get("source_training_loss", 0.0)
                ),
                source_grad_norm=float(payload.get("source_grad_norm", 0.0)),
            )
            feature_values = row.features()
            invalid_features = sorted(
                name
                for name, value in feature_values.items()
                if not np.isfinite(value)
            )
            if invalid_features:
                raise ArtifactValidationFailure(
                    "real replay contains non-finite publish-time controller "
                    f"features {invalid_features} in {path}"
                )
            for name in ("relative_gradient_mismatch", "cosine"):
                if not np.isfinite(float(payload[name])):
                    raise ArtifactValidationFailure(
                        f"real replay contains non-finite {name} in {path}"
                    )
            for name in ("delta_g", "delta_z"):
                if not np.isfinite(
                    np.asarray(payload[name], dtype=np.float64)
                ).all():
                    raise ArtifactValidationFailure(
                        f"real replay contains non-finite {name} in {path}"
                    )
            record = ReplayUpdateRecord(
                    row=row,
                    utilities={8: controller_utility},
                    cosine=float(payload["cosine"]),
                    delta_g=np.asarray(payload["delta_g"], dtype=np.float64),
                    delta_z=np.asarray(payload["delta_z"], dtype=np.float64),
                    source_round=int(payload["source_round"]),
                    arrival_round=int(payload["arrival_round"]),
                    source_z_raw=(
                        np.asarray(payload["source_z_raw"], dtype=np.float64)
                        if "source_z_raw" in payload
                        else None
                    ),
                    arrival_z_raw=(
                        np.asarray(payload["arrival_z_raw"], dtype=np.float64)
                        if "arrival_z_raw" in payload
                        else None
                    ),
                    utility_metric=str(
                        payload.get("utility_metric", "training_loss_gain_v1")
                    ),
                    training_loss_gain=(
                        float(payload["training_loss_gain"])
                        if "training_loss_gain" in payload
                        else None
                    ),
                    full_candidate_utility=controller_utility,
                    actual_published_utility=actual_utility,
                    provenance_method=str(provenance_method),
                    candidate_arrival_round=candidate_arrival_round,
                    actual_arrival_round=actual_arrival_round,
                    paired_tts_barrier=paired_tts_barrier,
                    prefix_feature_exact=True,
                    oracle_l1_utility=(
                        float(payload["oracle_l1_utility"])
                        if "oracle_l1_utility" in payload
                        else None
                    ),
                    oracle_l2_utility=(
                        float(payload["oracle_l2_utility"])
                        if "oracle_l2_utility" in payload
                        else None
                    ),
                    oracle_l2_kappa=(
                        float(payload["oracle_l2_kappa"])
                        if "oracle_l2_kappa" in payload
                        else None
                    ),
                    utility_by_kappa=utility_by_kappa,
                    evaluation_pair_id=evaluation_pair_id,
                    trace_stage_index=trace_stage_index,
                    trace_stage_count=trace_stage_count,
                    trace_capture_sampling=trace_capture_sampling,
                    evaluation_concurrency=evaluation_concurrency,
                    fresh_gradient_scope=fresh_gradient_scope,
                    transported_candidate_utility=(
                        transported_candidate_utility
                    ),
                    paired_l2_utility=paired_l2_utility,
                    transport_evaluation_contract=(
                        transport_evaluation_contract
                    ),
                    transport_variant=transport_variant,
                    transport_map_sha256=transport_map_sha256,
                )
            if trace_owner is not None:
                record.trace_owner_phase = trace_owner["phase"]
                record.trace_prompt_offset = trace_owner["prompt_offset"]
                record.trace_prompt_limit = trace_owner["prompt_limit"]
                record.trace_owner_role = trace_owner["role"]
                record.trace_evaluation_only = trace_owner["evaluation_only"]
                record.trace_owner_manifest_sha256 = trace_owner["manifest_sha256"]
            records.append(record)
    utility_metrics = {record.utility_metric for record in records}
    if len(utility_metrics) != 1:
        raise ArtifactValidationFailure(
            "real replay mixes incompatible utility metrics: "
            f"{sorted(utility_metrics)}"
        )
    return records


def _scan_exactness_evidence(
    root: str | Path,
    *,
    model_pair_id: str | None = None,
    owner_method: str | None = None,
    owner_methods: tuple[str, ...] | None = None,
    require_l3_evaluation_only: bool | None = None,
) -> dict:
    """Require positive canary evidence, not merely absence of an exception."""
    if owner_method is not None and owner_methods is not None:
        raise ValueError("owner_method and owner_methods are mutually exclusive")
    allowed_owner_methods = (
        frozenset(owner_methods)
        if owner_methods is not None
        else frozenset((owner_method,)) if owner_method is not None else None
    )
    root = Path(root)
    paths = _select_pair_owned_paths(
        root,
        sorted(root.rglob("adaptation-telemetry-*.jsonl")),
        model_pair_id,
        require_owner=model_pair_id is not None,
        allow_empty=True,
    )
    if allowed_owner_methods is not None or require_l3_evaluation_only is not None:
        from lightcone_spec.config.loader import load_adaptation_config

        filtered: list[Path] = []
        for path in paths:
            config_path = _owning_runtime_config_path(path, root)
            if config_path is None:
                continue
            config = load_adaptation_config(config_path)
            if (
                allowed_owner_methods is not None
                and config.method not in allowed_owner_methods
            ):
                continue
            if (
                require_l3_evaluation_only is not None
                and config.trace.l3_evaluation_only
                is not require_l3_evaluation_only
            ):
                continue
            filtered.append(path)
        paths = filtered
    rounds = 0
    violations: list[dict] = []
    for path in paths:
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactValidationFailure(
                    f"malformed telemetry {path}:{line_no}: {exc}"
                ) from exc
            if item.get("kind") == "round":
                rounds += 1
                if item.get("version_canary_ok") is not True:
                    violations.append(
                        {
                            "path": str(path),
                            "line": line_no,
                            "reason": "version_canary_failed",
                        }
                    )
                if item.get("prefix_feature_exact") is not True:
                    violations.append(
                        {
                            "path": str(path),
                            "line": line_no,
                            "reason": "prefix_feature_inexact",
                        }
                    )
            reason = item.get("failure_reason")
            if isinstance(reason, str) and reason.startswith("exactness:"):
                violations.append(
                    {"path": str(path), "line": line_no, "reason": reason}
                )
    return {
        "verified": bool(paths and rounds > 0 and not violations),
        "telemetry_files": len(paths),
        "rounds_checked": rounds,
        "violation_count": len(violations),
        "violations": violations[:32],
        "owner_method": owner_method,
        "owner_methods": (
            sorted(allowed_owner_methods)
            if allowed_owner_methods is not None
            else None
        ),
        "l3_evaluation_only": require_l3_evaluation_only,
    }


def _l3_transport_gate(
    records,
    transport_map,
    *,
    expected_transport_map_sha256: str | None = None,
    bootstrap_b: int = B_DEFAULT,
    min_test_groups: int = 8,
    seed: int = 0,
) -> dict:
    """Gate L3 on real acceptance utility, never gradient reconstruction.

    A phase-1 fit freezes the transport map.  A bounded phase-2 evaluation
    then records the complete joint Fisher/transport/AdamW/damping utility and
    a same-state L2-no-transport counterfactual.  The exact prompt/checkpoint/
    seed and source-update unit is paired with a TTS trace.  Both held-out
    prompt-cluster BCa
    lower bounds must be positive; the caller applies the independent
    exactness gate afterwards.
    """

    test = [
        r
        for r in records
        if split_of_group(r.row.sequence_id, seed) == "test"
    ]

    # Keep map reconstruction quality as a diagnostic only.  In particular,
    # neither its point estimate nor CI participates in `enabled`.
    baseline_errors: list[float] = []
    transport_errors: list[float] = []
    diagnostic_groups: list[str] = []
    for record in test:
        predicted = transport_map.state_correction(record.delta_z)
        baseline_error = float(np.dot(record.delta_g, record.delta_g))
        residual = record.delta_g - predicted
        baseline_errors.append(baseline_error)
        transport_errors.append(float(np.dot(residual, residual)))
        diagnostic_groups.append(record.row.sequence_id)
    baseline = np.asarray(baseline_errors, dtype=np.float64)
    transport = np.asarray(transport_errors, dtype=np.float64)
    diagnostic_groups_a = np.asarray(diagnostic_groups)
    diagnostic_unique = np.unique(diagnostic_groups_a)

    def relative_reduction(indices: np.ndarray) -> float:
        baseline_total = float(baseline[indices].sum())
        transport_total = float(transport[indices].sum())
        if baseline_total <= np.finfo(np.float64).eps:
            return 0.0 if transport_total <= baseline_total else -1.0
        return 1.0 - transport_total / baseline_total

    diagnostic_boot: list[float] = []
    if len(diagnostic_unique):
        rng = np.random.Generator(np.random.PCG64(0))
        for _ in range(bootstrap_b):
            chosen = rng.choice(
                diagnostic_unique, size=len(diagnostic_unique), replace=True
            )
            idx = np.concatenate(
                [np.where(diagnostic_groups_a == group)[0] for group in chosen]
            )
            diagnostic_boot.append(relative_reduction(idx))
    diagnostic_ci = (
        [
            float(np.percentile(diagnostic_boot, 2.5)),
            float(np.percentile(diagnostic_boot, 97.5)),
        ]
        if diagnostic_boot
        else [float("nan"), float("nan")]
    )
    relative = (
        relative_reduction(np.arange(len(test))) if test else float("nan")
    )
    baseline_mse = float(baseline.mean()) if len(baseline) else float("nan")
    transport_mse = float(transport.mean()) if len(transport) else float("nan")
    diagnostic = {
        "metric": "fresh_minus_stale_gradient_reconstruction_mse",
        "used_for_enable": False,
        "relative_error_reduction_vs_drop": relative,
        "mean_squared_error_reduction_vs_drop": baseline_mse - transport_mse,
        "baseline_mse": baseline_mse,
        "transport_mse": transport_mse,
        "ci95": diagnostic_ci,
        "n_test": len(test),
        "n_test_groups": len(diagnostic_unique),
    }

    all_l3_records = [
        r for r in test if getattr(r, "provenance_method", None) == "lc_transport"
    ]
    l3_records = [
        r for r in all_l3_records
        if getattr(r, "trace_owner_role", None) == "phase2_l3"
        and getattr(r, "trace_evaluation_only", None) is True
    ]
    tts_records = [
        r for r in test
        if getattr(r, "provenance_method", None) == "tts"
        and getattr(r, "trace_owner_role", None) == "phase2_tts_reference"
        and getattr(r, "trace_evaluation_only", None) is True
    ]

    def incomplete(reason: str, **extra) -> dict:
        utility_gate = {
            "complete": False,
            "eligible": False,
            "disabled_reason": reason,
            "n_l3_test": len(l3_records),
            "n_tts_test": len(tts_records),
            **extra,
        }
        return {
            "enabled": False,
            "evidence_insufficient": True,
            "disabled_reason": reason,
            "heldout_transported_utility_gate": utility_gate,
            "transport_fit_diagnostic": diagnostic,
            # Compatibility aliases for existing reports.  They remain
            # explicitly diagnostic and cannot open the gate.
            **{
                key: diagnostic[key]
                for key in (
                    "relative_error_reduction_vs_drop",
                    "mean_squared_error_reduction_vs_drop",
                    "baseline_mse",
                    "transport_mse",
                    "ci95",
                    "n_test",
                    "n_test_groups",
                )
            },
            "exactness_required": True,
        }

    if all_l3_records and len(l3_records) != len(all_l3_records):
        return incomplete(
            "L3 held-out labels lack explicit phase/window ownership"
        )
    if not l3_records:
        return incomplete(
            "missing schema-v3 held-out joint transported acceptance labels"
        )
    l3_windows = {
        (
            getattr(record, "trace_prompt_offset", None),
            getattr(record, "trace_prompt_limit", None),
        )
        for record in l3_records
    }
    tts_windows = {
        (
            getattr(record, "trace_prompt_offset", None),
            getattr(record, "trace_prompt_limit", None),
        )
        for record in tts_records
    }
    if len(l3_windows) != 1 or tts_windows != l3_windows:
        return incomplete(
            "L3/TTS evaluation does not bind one identical held-out window"
        )
    required_contract = "joint_fisher_transport_adamw_damping_v1"
    if any(
        record.evaluation_pair_id is None
        or record.trace_stage_index is None
        or record.trace_stage_count is None
        or record.trace_capture_sampling not in ("first", "staged")
        or record.evaluation_concurrency is None
        or record.transported_candidate_utility is None
        or record.paired_l2_utility is None
        or record.transport_evaluation_contract != required_contract
        or record.transport_variant != "joint"
        for record in l3_records
    ):
        return incomplete(
            "L3 held-out labels are incomplete or did not execute the full "
            "joint Fisher/transport/AdamW/damping path"
        )
    if expected_transport_map_sha256 is None or any(
        record.transport_map_sha256 != expected_transport_map_sha256
        for record in l3_records
    ):
        return incomplete(
            "L3 held-out labels were not produced by the final frozen "
            "transport map"
        )

    def pair_key(record) -> tuple[str, int, int]:
        return (
            str(record.evaluation_pair_id),
            int(record.evaluation_concurrency),
            int(record.trace_stage_index),
        )

    tts_by_key: dict[tuple[str, int, int], object] = {}
    duplicate_tts = 0
    for record in tts_records:
        if (
            record.evaluation_pair_id is None
            or record.evaluation_concurrency is None
            or record.trace_stage_index is None
        ):
            continue
        key = pair_key(record)
        if key in tts_by_key:
            duplicate_tts += 1
        else:
            tts_by_key[key] = record
    l3_by_key: dict[tuple[str, int, int], object] = {}
    duplicate_l3 = 0
    for record in l3_records:
        key = pair_key(record)
        if key in l3_by_key:
            duplicate_l3 += 1
        else:
            l3_by_key[key] = record
    if duplicate_l3 or duplicate_tts:
        return incomplete(
            "L3/TTS evaluation contains duplicate pairing keys",
            duplicate_l3=duplicate_l3,
            duplicate_tts=duplicate_tts,
        )
    missing_tts = sorted(set(l3_by_key).difference(tts_by_key))
    extra_tts = sorted(set(tts_by_key).difference(l3_by_key))
    if missing_tts or extra_tts:
        return incomplete(
            "L3 and TTS held-out prompt/checkpoint/seed trace-stage sets differ",
            missing_tts_pairs=len(missing_tts),
            extra_tts_pairs=len(extra_tts),
        )

    paired = [(l3_by_key[key], tts_by_key[key]) for key in sorted(l3_by_key)]
    if any(
        l3.trace_stage_count != tts.trace_stage_count
        or l3.trace_capture_sampling != tts.trace_capture_sampling
        or l3.evaluation_concurrency != tts.evaluation_concurrency
        for l3, tts in paired
    ):
        return incomplete(
            "paired L3/TTS traces used different concurrency, stage counts, "
            "or sampling policies"
        )
    source_round_differences = np.asarray(
        [abs(int(l3.source_round) - int(tts.source_round)) for l3, tts in paired],
        dtype=np.float64,
    )
    candidate_arrival_differences = np.asarray(
        [
            abs(
                int(l3.candidate_arrival_round)
                - int(tts.candidate_arrival_round)
            )
            for l3, tts in paired
        ],
        dtype=np.float64,
    )
    prefix_differences = np.asarray(
        [
            abs(float(l3.row.source_prefix_len) - float(tts.row.source_prefix_len))
            for l3, tts in paired
        ],
        dtype=np.float64,
    )
    prefix_relative_differences = np.asarray(
        [
            difference
            / max(
                float(l3.row.source_prefix_len),
                float(tts.row.source_prefix_len),
                1.0,
            )
            for difference, (l3, tts) in zip(prefix_differences, paired)
        ],
        dtype=np.float64,
    )
    groups = np.asarray([l3.row.sequence_id for l3, _ in paired])
    unique_groups = np.unique(groups)
    required_groups = max(int(min_test_groups), 2)
    if len(unique_groups) < required_groups:
        return incomplete(
            "too few independent held-out prompt groups for L3",
            n_pairs=len(paired),
            n_test_groups=len(unique_groups),
            minimum_test_groups=required_groups,
        )

    l3_values = np.asarray(
        [l3.transported_candidate_utility for l3, _ in paired],
        dtype=np.float64,
    )
    l2_values = np.asarray(
        [l3.paired_l2_utility for l3, _ in paired], dtype=np.float64
    )
    tts_values = np.asarray(
        [tts.actual_published_utility for _, tts in paired], dtype=np.float64
    )
    if not np.isfinite(
        np.concatenate((l3_values, l2_values, tts_values))
    ).all():
        return incomplete("non-finite held-out L3/L2/TTS acceptance utility")

    from lightcone_spec.statistics.bootstrap import cluster_bca

    def interval(delta: np.ndarray):
        result = cluster_bca(delta, groups, np.mean, b=bootstrap_b, seed=0)
        return result.estimate, [result.ci_low, result.ci_high]

    gain_vs_tts, ci_vs_tts = interval(l3_values - tts_values)
    gain_vs_l2, ci_vs_l2 = interval(l3_values - l2_values)
    eligible = bool(ci_vs_tts[0] > 0.0 and ci_vs_l2[0] > 0.0)
    utility_gate = {
        "complete": True,
        "eligible": eligible,
        "utility_metric": "survival_weighted_accepted_prefix_v1",
        "l3_contract": required_contract,
        "pairing_contract": (
            "exact_request_seed_concurrency_trace_stage_v1"
        ),
        "transport_map_sha256": expected_transport_map_sha256,
        "tts_reference": "paired_actual_tts_barrier",
        "l2_reference": "same_state_same_arrival_l2_no_transport",
        "horizon_alignment": "independent_H_from_each_actual_arrival",
        "n_pairs": len(paired),
        "n_test_groups": len(unique_groups),
        "trace_capture_sampling": paired[0][0].trace_capture_sampling,
        "trace_stage_count": paired[0][0].trace_stage_count,
        "evaluation_concurrency": paired[0][0].evaluation_concurrency,
        "mean_abs_source_round_difference": float(
            source_round_differences.mean()
        ),
        "max_abs_source_round_difference": float(
            source_round_differences.max()
        ),
        "mean_abs_candidate_arrival_round_difference": float(
            candidate_arrival_differences.mean()
        ),
        "max_abs_candidate_arrival_round_difference": float(
            candidate_arrival_differences.max()
        ),
        "mean_abs_source_prefix_difference": float(prefix_differences.mean()),
        "max_abs_source_prefix_difference": float(prefix_differences.max()),
        "mean_relative_source_prefix_difference": float(
            prefix_relative_differences.mean()
        ),
        "max_relative_source_prefix_difference": float(
            prefix_relative_differences.max()
        ),
        "gain_vs_tts": gain_vs_tts,
        "ci95_vs_tts": ci_vs_tts,
        "gain_vs_l2": gain_vs_l2,
        "ci95_vs_l2": ci_vs_l2,
    }
    return {
        "enabled": eligible,
        "evidence_insufficient": False,
        "disabled_reason": (
            None
            if eligible
            else "transported acceptance utility did not beat both paired "
            "TTS and L2 with positive held-out 95% CI lower bounds"
        ),
        "heldout_transported_utility_gate": utility_gate,
        "transport_fit_diagnostic": diagnostic,
        **{
            key: diagnostic[key]
            for key in (
                "relative_error_reduction_vs_drop",
                "mean_squared_error_reduction_vs_drop",
                "baseline_mse",
                "transport_mse",
                "ci95",
                "n_test",
                "n_test_groups",
            )
        },
        "exactness_required": True,
    }


def _oracle_replay_gate(
    records,
    bootstrap_b: int = B_DEFAULT,
    min_test_groups: int = 8,
    seed: int = 0,
) -> dict:
    """Held-out upper bounds before spending complexity on L1/L2.

    These deltas compare oracle gate/damping with the same-arrival full raw
    candidate (L0).  A real TTS barrier comparison remains a separate paired
    systems experiment and must not be inferred from this replay ceiling.
    """
    test = [
        r
        for r in records
        if split_of_group(r.row.sequence_id, seed) == "test"
    ]
    complete = bool(test) and all(
        r.full_candidate_utility is not None
        and r.oracle_l1_utility is not None
        and r.oracle_l2_utility is not None
        for r in test
    )
    if not complete:
        return {
            "complete": False,
            "l1_eligible": False,
            "l2_eligible": False,
            "disabled_reason": "trace lacks oracle kappa replay labels",
        }
    groups = np.asarray([r.row.sequence_id for r in test])
    unique = np.unique(groups)
    required_groups = max(int(min_test_groups), 2)
    if len(unique) < required_groups:
        return {
            "complete": False,
            "l1_eligible": False,
            "l2_eligible": False,
            "n_test": len(test),
            "n_test_groups": len(unique),
            "minimum_test_groups": required_groups,
            "disabled_reason": "too few independent held-out prompt groups",
        }
    full = np.asarray([r.full_candidate_utility for r in test], dtype=np.float64)
    l1 = np.asarray([r.oracle_l1_utility for r in test], dtype=np.float64)
    l2 = np.asarray([r.oracle_l2_utility for r in test], dtype=np.float64)

    def clustered_ci(delta: np.ndarray) -> tuple[float, list[float]]:
        from lightcone_spec.statistics.bootstrap import cluster_bca

        interval = cluster_bca(delta, groups, np.mean, b=bootstrap_b, seed=0)
        return interval.estimate, [interval.ci_low, interval.ci_high]

    l1_gain, l1_ci = clustered_ci(l1 - full)
    l2_gain, l2_ci = clustered_ci(l2 - full)
    return {
        "complete": True,
        "reference": "same_arrival_full_candidate_l0",
        "tts_barrier_comparison_required": True,
        "n_test": len(test),
        "n_test_groups": len(unique),
        "l1_oracle_gain": l1_gain,
        "l1_ci95": l1_ci,
        "l1_eligible": bool(l1_ci[0] > 0.0),
        "l2_oracle_gain": l2_gain,
        "l2_ci95": l2_ci,
        "l2_eligible": bool(l2_ci[0] > 0.0),
    }


def _tts_paired_gate(
    records,
    *,
    bootstrap_b: int = B_DEFAULT,
    min_test_groups: int = 8,
    incomplete_pairs: int = 0,
    seed: int = 0,
) -> dict:
    """Paired early-arrival oracle gain over the same candidate's TTS barrier.

    All utilities come from one TTS request trace.  The undamped candidate at
    its first ready boundary is exactly L0's same-source action; no
    independently evolved L0/L1/L2 trajectory is admitted as evidence for
    this systems gate.
    """

    tts_records = [r for r in records if r.provenance_method == "tts"]
    test = [
        r
        for r in tts_records
        if split_of_group(r.row.sequence_id, seed) == "test"
    ]
    all_pairs_complete = bool(tts_records) and all(
        r.paired_tts_barrier is True
        and r.prefix_feature_exact is True
        and r.candidate_arrival_round is not None
        and r.actual_arrival_round is not None
        and r.actual_arrival_round >= r.candidate_arrival_round
        and r.full_candidate_utility is not None
        and r.actual_published_utility is not None
        and r.oracle_l1_utility is not None
        and r.oracle_l2_utility is not None
        and np.isfinite(float(r.full_candidate_utility))
        and np.isfinite(float(r.actual_published_utility))
        and np.isfinite(float(r.oracle_l1_utility))
        and np.isfinite(float(r.oracle_l2_utility))
        for r in tts_records
    )
    complete = bool(test) and incomplete_pairs == 0 and all_pairs_complete
    if not complete:
        return {
            "complete": False,
            "l0_eligible": False,
            "l1_eligible": False,
            "l2_eligible": False,
            "n_test": len(test),
            "incomplete_pairs": int(incomplete_pairs),
            "disabled_reason": (
                "trace contains unfinished paired TTS labels"
                if incomplete_pairs
                else "held-out trace lacks a complete same-candidate TTS pair"
            ),
        }
    groups = np.asarray([r.row.sequence_id for r in test])
    unique = np.unique(groups)
    # A bootstrap distribution made from one prompt cluster is a repeated
    # copy of one observation, not uncertainty evidence.  Keep this invariant
    # even in focused tests that lower the normal eight-cluster threshold.
    required_groups = max(int(min_test_groups), 2)
    if len(unique) < required_groups:
        return {
            "complete": False,
            "l0_eligible": False,
            "l1_eligible": False,
            "l2_eligible": False,
            "n_test": len(test),
            "n_test_groups": len(unique),
            "minimum_test_groups": required_groups,
            "incomplete_pairs": 0,
            "disabled_reason": "too few independent held-out prompt groups",
        }

    from lightcone_spec.statistics.bootstrap import cluster_bca

    actual_tts = np.asarray(
        [r.actual_published_utility for r in test], dtype=np.float64
    )
    l0 = np.asarray(
        [r.full_candidate_utility for r in test], dtype=np.float64
    )
    l1 = np.asarray([r.oracle_l1_utility for r in test], dtype=np.float64)
    l2 = np.asarray([r.oracle_l2_utility for r in test], dtype=np.float64)

    def interval(delta: np.ndarray):
        result = cluster_bca(delta, groups, np.mean, b=bootstrap_b, seed=0)
        return result.estimate, [result.ci_low, result.ci_high]

    l0_gain, l0_ci = interval(l0 - actual_tts)
    l1_gain, l1_ci = interval(l1 - actual_tts)
    l2_gain, l2_ci = interval(l2 - actual_tts)
    return {
        "complete": True,
        "reference": "same_candidate_actual_tts_barrier",
        "horizon_alignment": "independent_H_from_each_arrival",
        "n_test": len(test),
        "n_test_groups": len(unique),
        "incomplete_pairs": 0,
        "l0_gain_vs_tts": l0_gain,
        "l0_ci95": l0_ci,
        "l0_eligible": bool(l0_ci[0] > 0.0),
        "l1_gain_vs_tts": l1_gain,
        "l1_ci95": l1_ci,
        "l1_eligible": bool(l1_ci[0] > 0.0),
        "l2_gain_vs_tts": l2_gain,
        "l2_ci95": l2_ci,
        "l2_eligible": bool(l2_ci[0] > 0.0),
    }


def _learned_policy_gate(
    records,
    artifact,
    *,
    bootstrap_b: int = B_DEFAULT,
    min_test_groups: int = 8,
    incomplete_pairs: int = 0,
    seed: int = 0,
) -> dict:
    """Held-out utility of the fitted L1/L2 policy against real TTS.

    The controller is fitted on grouped train/calibration records before this
    function is called.  Policy decisions are evaluated only on held-out TTS
    records and reuse the same candidate's captured kappa grid, so this is not
    an oracle ceiling and does not mix independently evolved trajectories.
    """

    tts_records = [r for r in records if r.provenance_method == "tts"]
    test = [
        r
        for r in tts_records
        if split_of_group(r.row.sequence_id, seed) == "test"
    ]
    complete = (
        bool(test)
        and incomplete_pairs == 0
        and all(
            r.paired_tts_barrier is True
            and r.prefix_feature_exact is True
            and r.actual_published_utility is not None
            and getattr(r, "utility_by_kappa", None) is not None
            and 0.0 in r.utility_by_kappa
            and 1.0 in r.utility_by_kappa
            and np.isfinite(r.row.round_delay)
            and float(r.row.round_delay).is_integer()
            for r in tts_records
        )
    )
    if not complete:
        return {
            "complete": False,
            "l1_eligible": False,
            "l2_eligible": False,
            "n_test": len(test),
            "incomplete_pairs": int(incomplete_pairs),
            "disabled_reason": (
                "trace contains unfinished paired TTS labels"
                if incomplete_pairs
                else "held-out TTS trace lacks complete fitted-policy kappa evidence"
            ),
        }
    groups = np.asarray([r.row.sequence_id for r in test])
    unique = np.unique(groups)
    required_groups = max(int(min_test_groups), 2)
    if len(unique) < required_groups:
        return {
            "complete": False,
            "l1_eligible": False,
            "l2_eligible": False,
            "n_test": len(test),
            "n_test_groups": len(unique),
            "minimum_test_groups": required_groups,
            "incomplete_pairs": 0,
            "disabled_reason": "too few independent held-out prompt groups",
        }

    from lightcone_spec.controller.damping import damping_factor
    from lightcone_spec.statistics.bootstrap import cluster_bca

    x, _ = design_matrix([record.row for record in test], artifact.feature_set)
    harm = np.nan_to_num(
        artifact.harmful_classifier.probability(x), nan=1.0
    )
    mismatch = np.nan_to_num(
        artifact.mismatch_predictor.predict(x), nan=np.inf, posinf=np.inf
    )
    fixed_discards = {
        int(delay)
        for delay in artifact.extra.get("gate_constant_discard_delays", ())
    }
    fixed_applies = {
        int(delay)
        for delay in artifact.extra.get("gate_constant_apply_delays", ())
    }
    profiles = artifact.extra.get("constant_controller_profiles", {})
    l1_values = []
    l2_values = []
    l1_applied = []
    l2_kappas = []
    l1_zero_delay_fast = []
    l1_constant_apply_fast = []
    l1_constant_discard_fast = []
    l1_predictor_path = []
    l2_constant_profile_fast = []
    l2_zero_delay_fast = []
    l2_unit_kappa_fast = []
    l2_predictor_path = []
    grids = set()
    for index, record in enumerate(test):
        delay = int(record.row.round_delay)
        grid = record.utility_by_kappa
        assert grid is not None
        grids.update(grid)

        apply_l1 = delay == 0 or (
            not artifact.gate_discard_all
            and (
                delay in fixed_applies
                or not (
                    delay in fixed_discards
                    or harm[index] > artifact.gate_threshold
                )
            )
        )
        l1_zero_delay_fast.append(delay == 0)
        l1_constant_apply_fast.append(delay > 0 and delay in fixed_applies)
        l1_constant_discard_fast.append(
            delay > 0
            and (artifact.gate_discard_all or delay in fixed_discards)
        )
        l1_predictor_path.append(
            delay > 0
            and not artifact.gate_discard_all
            and delay not in fixed_applies
            and delay not in fixed_discards
        )
        l1_applied.append(apply_l1)
        l1_values.append(grid[1.0] if apply_l1 else grid[0.0])

        profile = profiles.get(str(delay))
        if delay == 0:
            kappa = 1.0
        elif profile is not None:
            kappa = float(profile["damping_factor"])
        else:
            kappa = float(
                damping_factor(
                    mismatch[index],
                    artifact.damping_radius,
                    artifact.damping_kernel,
                )
            )
        l2_constant_profile_fast.append(delay > 0 and profile is not None)
        l2_zero_delay_fast.append(delay == 0)
        l2_predictor_path.append(delay > 0 and profile is None)
        l2_unit_kappa_fast.append(float(kappa) == 1.0)
        if not np.isfinite(kappa) or not 0.0 <= kappa <= 1.0:
            return {
                "complete": False,
                "l1_eligible": False,
                "l2_eligible": False,
                "n_test": len(test),
                "n_test_groups": len(unique),
                "disabled_reason": "fitted L2 policy produced an invalid damping factor",
            }
        ordered = sorted(grid.items())
        l2_kappas.append(kappa)
        l2_values.append(
            float(
                np.interp(
                    kappa,
                    [item[0] for item in ordered],
                    [item[1] for item in ordered],
                )
            )
        )

    actual_tts = np.asarray(
        [r.actual_published_utility for r in test], dtype=np.float64
    )
    l1 = np.asarray(l1_values, dtype=np.float64)
    l2 = np.asarray(l2_values, dtype=np.float64)

    def interval(delta: np.ndarray):
        result = cluster_bca(delta, groups, np.mean, b=bootstrap_b, seed=0)
        return result.estimate, [result.ci_low, result.ci_high]

    l1_gain, l1_ci = interval(l1 - actual_tts)
    l2_gain, l2_ci = interval(l2 - actual_tts)
    return {
        "complete": True,
        "reference": "learned_policy_same_candidate_actual_tts_barrier",
        "evaluation_split": "heldout_prompt_groups",
        "kappa_utility_estimator": "piecewise_linear_captured_kappa_grid_v1",
        "captured_kappa_grid": sorted(float(value) for value in grids),
        "n_test": len(test),
        "n_test_groups": len(unique),
        "incomplete_pairs": 0,
        "l1_gain_vs_tts": l1_gain,
        "l1_ci95": l1_ci,
        "l1_apply_fraction": float(np.mean(l1_applied)),
        "l1_zero_delay_fastpath_fraction": float(
            np.mean(l1_zero_delay_fast)
        ),
        "l1_constant_apply_fastpath_fraction": float(
            np.mean(l1_constant_apply_fast)
        ),
        "l1_constant_discard_fastpath_fraction": float(
            np.mean(l1_constant_discard_fast)
        ),
        "l1_predictor_path_fraction": float(np.mean(l1_predictor_path)),
        "l1_eligible": bool(l1_ci[0] > 0.0),
        "l2_gain_vs_tts": l2_gain,
        "l2_ci95": l2_ci,
        "l2_mean_kappa": float(np.mean(l2_kappas)),
        "l2_constant_profile_fastpath_fraction": float(
            np.mean(l2_constant_profile_fast)
        ),
        "l2_zero_delay_fastpath_fraction": float(
            np.mean(l2_zero_delay_fast)
        ),
        "l2_unit_kappa_fastpath_fraction": float(
            np.mean(l2_unit_kappa_fast)
        ),
        "l2_predictor_path_fraction": float(np.mean(l2_predictor_path)),
        "l2_eligible": bool(l2_ci[0] > 0.0),
    }


def _count_incomplete_tts_pairs(
    root: str | Path, *, model_pair_id: str | None = None
) -> int:
    root = Path(root)
    count = 0
    paths = _select_pair_owned_paths(
        root,
        sorted(root.rglob("incomplete-paired-tts-*.jsonl")),
        model_pair_id,
        require_owner=model_pair_id is not None,
        allow_empty=True,
    )
    for path in paths:
        count += sum(bool(line.strip()) for line in path.read_text().splitlines())
    return count


def fit_real_replay(
    root: str | Path,
    *,
    model_pair_id: str,
    transport_rank: int = 16,
    seed: int = 0,
):
    root_path = Path(root)
    runtime_identity, parameter_layout_sha256 = _controller_runtime_identity(
        root_path, model_pair_id
    )
    records = load_real_replay_records(
        root_path, model_pair_id=model_pair_id
    )
    utility_metrics = {record.utility_metric for record in records}
    if utility_metrics != {"survival_weighted_accepted_prefix_v1"}:
        raise ArtifactValidationFailure(
            "real controller fitting requires survival-weighted accepted-prefix "
            f"utility; observed {sorted(utility_metrics)}"
        )
    provenance_methods = sorted({
        str(record.provenance_method) for record in records
    })
    required_provenance = {"naive_async", "tts"}
    missing_provenance = sorted(
        required_provenance.difference(provenance_methods)
    )
    if missing_provenance:
        raise ArtifactValidationFailure(
            "real replay producer contract requires same-pair naive_async and "
            f"paired TTS traces; missing {missing_provenance} for {model_pair_id}"
        )
    # Freeze the controller and transport map exclusively from the phase-1
    # candidate family.  Phase-2 lc_transport evaluation follows a different
    # parameter trajectory; feeding it back into the fit would change the map
    # after its utility was measured and invalidate the evidence circularly.
    has_l3_evaluation = any(
        record.provenance_method == "lc_transport" for record in records
    )
    fit_records = [
        record
        for record in records
        if record.provenance_method in required_provenance
        and (
            not has_l3_evaluation
            or (
                getattr(record, "trace_owner_role", None) == "phase1_producer"
                and getattr(record, "trace_evaluation_only", None) is False
            )
        )
    ]
    phase2_producers = [
        record
        for record in records
        if record.provenance_method in required_provenance
        and getattr(record, "trace_evaluation_only", None) is True
    ]
    unowned_producers = [
        record
        for record in records
        if record.provenance_method in required_provenance
        and getattr(record, "trace_owner_role", None) is None
    ]
    if has_l3_evaluation and unowned_producers:
        raise ArtifactValidationFailure(
            "L3 evaluation mixes explicitly owned phase-2 references with "
            "legacy unowned controller producers"
        )
    if not fit_records:
        raise ArtifactValidationFailure(
            "controller fit has no explicitly owned phase-1 producer records"
        )
    missing_fit_provenance = required_provenance.difference(
        {record.provenance_method for record in fit_records}
    )
    if missing_fit_provenance:
        raise ArtifactValidationFailure(
            "controller fit phase-1 ownership omits producer methods: "
            f"{sorted(missing_fit_provenance)}"
        )
    split_counts = {
        name: sum(
            split_of_group(record.row.sequence_id, seed) == name
            for record in fit_records
        )
        for name in ("train", "calibration", "test")
    }
    if min(split_counts.values(), default=0) == 0:
        raise ArtifactValidationFailure(
            "real replay grouped split is incomplete for the requested split "
            f"seed {seed}; collect more independent prompts (counts={split_counts})"
        )
    # Compute the held-out algorithmic ceiling before fitting a controller.
    # A learned controller cannot recover utility absent from its oracle.
    oracle_gate = _oracle_replay_gate(fit_records, seed=seed)
    incomplete_tts_pairs = _count_incomplete_tts_pairs(
        root_path, model_pair_id=model_pair_id
    )
    tts_paired_gate = _tts_paired_gate(
        fit_records,
        incomplete_pairs=incomplete_tts_pairs,
        seed=seed,
    )
    zvec = default_zvectorizer()
    train = [
        r
        for r in fit_records
        if split_of_group(r.row.sequence_id, seed) == "train"
    ]
    if any(r.source_z_raw is None or r.arrival_z_raw is None for r in records):
        raise ArtifactValidationFailure(
            "real replay shards lack raw trajectory vectors required for the "
            "train-only controller normalizer"
        )
    train_states = np.stack(
        [
            z
            for record in train
            for z in (record.source_z_raw, record.arrival_z_raw)
        ]
    )
    zvec.mean = train_states.mean(axis=0)
    zvec.std = np.maximum(train_states.std(axis=0), 1e-8)
    for record in records:
        record.delta_z = (
            record.arrival_z_raw - record.source_z_raw
        ) / zvec.std
    result = fit_replay_pipeline(
        fit_records,
        model_pair_id=model_pair_id,
        zvec=zvec,
        transport_rank=transport_rank,
        seed=seed,
    )
    learned_policy_gate = _learned_policy_gate(
        fit_records,
        result.artifact,
        incomplete_pairs=incomplete_tts_pairs,
        seed=seed,
    )
    transport_map_sha256 = sha256_json(result.artifact.transport_map.to_dict())
    l3_gate = _l3_transport_gate(
        records,
        result.artifact.transport_map,
        expected_transport_map_sha256=transport_map_sha256,
        seed=seed,
    )
    exactness = _scan_exactness_evidence(
        root,
        model_pair_id=model_pair_id,
        owner_methods=tuple(sorted(required_provenance)),
        require_l3_evaluation_only=False,
    )
    l3_evaluation_exactness = _scan_exactness_evidence(
        root,
        model_pair_id=model_pair_id,
        owner_method="lc_transport",
        require_l3_evaluation_only=True,
    )
    for gate in (oracle_gate, tts_paired_gate, learned_policy_gate):
        gate["trace_exactness_verified"] = bool(exactness["verified"])
        if not exactness["verified"]:
            if "l0_eligible" in gate:
                gate["l0_eligible"] = False
            gate["l1_eligible"] = False
            gate["l2_eligible"] = False
            gate["exactness_disabled_reason"] = (
                "trace exactness evidence missing"
                if exactness["rounds_checked"] == 0
                else "trace exactness violation observed"
            )
    l3_gate["exactness"] = l3_evaluation_exactness
    l3_gate["enabled"] = bool(
        l3_gate["enabled"] and l3_evaluation_exactness["verified"]
    )
    # Evaluation readiness is not a performance claim.  It only breaks the
    # two-pass circularity for an explicit bounded benchmark run; production
    # remains closed until the paired utility gate above passes.
    l3_gate["evaluation_ready"] = bool(
        exactness["verified"] and result.artifact.transport_map is not None
    )
    if not l3_evaluation_exactness["verified"]:
        l3_gate["exactness_disabled_reason"] = (
            "L3 evaluation exactness evidence missing"
            if l3_evaluation_exactness["rounds_checked"] == 0
            else "L3 evaluation exactness violation observed"
        )
        if l3_gate.get("heldout_transported_utility_gate", {}).get(
            "eligible", False
        ):
            l3_gate["disabled_reason"] = l3_gate[
                "exactness_disabled_reason"
            ]
    result.artifact.extra["l3_gate"] = l3_gate
    result.artifact.extra["transport_map_sha256"] = transport_map_sha256
    result.artifact.extra["trace_exactness"] = exactness
    result.artifact.extra["oracle_replay_gate"] = oracle_gate
    result.artifact.extra["tts_paired_gate"] = tts_paired_gate
    result.artifact.extra["learned_policy_gate"] = learned_policy_gate
    result.artifact.extra["trace_producer_contract"] = {
        "schema_version": 3,
        "required_provenance_methods": sorted(required_provenance),
        "observed_provenance_methods": provenance_methods,
        "pair_filtered": True,
        "tts_pairing": "same_candidate_early_vs_fixed_barrier_v1",
        "l3_evaluation": "two_pass_joint_fisher_transport_adamw_damping_v1",
    }
    result.artifact.extra["real_replay_records"] = len(records)
    result.artifact.extra["controller_fit_records"] = len(fit_records)
    result.artifact.extra["fresh_gradient_scopes"] = sorted(
        {
            str(record.fresh_gradient_scope)
            for record in fit_records
            if record.fresh_gradient_scope is not None
        }
    )
    result.artifact.extra["controller_utility_metric"] = next(
        iter({record.utility_metric for record in records})
    )
    loss_diagnostics = [
        record.training_loss_gain
        for record in fit_records
        if record.training_loss_gain is not None
    ]
    utility_values = np.asarray([record.row.utility for record in fit_records])
    utility_diagnostics = {
        "mean_utility": float(utility_values.mean()),
        "harmful_rate": float((utility_values < 0.0).mean()),
        "mean_training_loss_gain": None,
        "utility_loss_pearson": None,
    }
    if len(loss_diagnostics) == len(fit_records):
        loss_values = np.asarray(loss_diagnostics, dtype=np.float64)
        utility_diagnostics["mean_training_loss_gain"] = float(
            loss_values.mean()
        )
        if utility_values.std() > 0 and loss_values.std() > 0:
            utility_diagnostics["utility_loss_pearson"] = float(
                np.corrcoef(utility_values, loss_values)[0, 1]
            )
    result.artifact.extra["utility_diagnostics"] = utility_diagnostics
    result.artifact.extra["controller_runtime_identity"] = runtime_identity
    result.artifact.extra["controller_runtime_identity_sha256"] = sha256_json(
        runtime_identity
    )
    result.artifact.extra["parameter_layout_sha256"] = (
        parameter_layout_sha256
    )
    result.artifact.extra["split_seed"] = int(seed)
    selected_evidence_paths = _select_pair_owned_paths(
        root_path,
        sorted(root_path.rglob("index*.jsonl")),
        model_pair_id,
        require_owner=True,
    ) + _select_pair_owned_paths(
        root_path,
        sorted(root_path.rglob("incomplete-paired-tts-*.jsonl")),
        model_pair_id,
        require_owner=True,
        allow_empty=True,
    )
    result.artifact.extra["real_replay_data_sha256"] = sha256_json(
        [
            (str(path.relative_to(root_path)), sha256_file(path))
            for path in selected_evidence_paths
        ]
    )
    telemetry_paths = _select_pair_owned_paths(
        root_path,
        sorted(root_path.rglob("adaptation-telemetry-*.jsonl")),
        model_pair_id,
        require_owner=True,
        allow_empty=True,
    )
    result.artifact.extra["exactness_evidence_sha256"] = sha256_json(
        [
            (str(path.relative_to(root_path)), sha256_file(path))
            for path in telemetry_paths
        ]
    )
    owner_configs = sorted(
        {
            config
            for path in (*selected_evidence_paths, *telemetry_paths)
            if (config := _owning_runtime_config_path(path, root_path)) is not None
        }
    )
    result.artifact.extra["runtime_config_evidence_sha256"] = sha256_json(
        [
            (str(path.relative_to(root_path)), sha256_file(path))
            for path in owner_configs
        ]
    )
    result.artifact.extra["test_group_hash"] = sha256_json(
        sorted(
            {
                record.row.sequence_id
                for record in fit_records
                if split_of_group(record.row.sequence_id, seed) == "test"
            }
        )
    )
    return result
