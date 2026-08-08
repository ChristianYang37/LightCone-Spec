"""Method construction from an AdaptationConfig (spec 6.4, 10.2)."""

from __future__ import annotations

import torch

from lightcone_spec.adapters.adapter_params import AdapterShapes
from lightcone_spec.config.schema import (
    MODEL_PAIRS,
    AdaptationConfig,
    effective_proposal_depth,
)
from lightcone_spec.controller.artifact import ControllerArtifact
from lightcone_spec.exit_codes import ConfigError
from lightcone_spec.locking.hashing import sha256_json
from lightcone_spec.methods.base import CandidateGeneratorConfig, MethodRuntime
from lightcone_spec.methods.lightcone import (
    EndpointGateMethod,
    LCDampMethod,
    LCGateMethod,
    LCTransportMethod,
    ParameterOnlyMethod,
    RandomTransportMethod,
    RoundDiscardMethod,
    WallDampMethod,
)
from lightcone_spec.methods.onlinespec import (
    OnlineSpecEnsemble,
    OnlineSpecOGD,
    OnlineSpecOptimistic,
)
from lightcone_spec.methods.simple import (
    NaiveAsyncMethod,
    OracleCurrentMethod,
    StaticMethod,
    SyncFreshMethod,
    TTSDSparkMethod,
)


def controller_runtime_identity(
    config: AdaptationConfig,
) -> dict:
    """Method-agnostic identity of the candidate and feature distribution.

    Logical delay and dataset are intentionally absent: a pooled controller is
    trained across both, with their effects represented in its input features.
    """
    pair = MODEL_PAIRS.get(config.model.pair_id)
    proposal_depth = (
        effective_proposal_depth(
            pair, config.runtime.speculative_num_draft_tokens
        )
        if pair is not None
        else int(config.runtime.speculative_num_draft_tokens)
    )
    return {
        # v4 binds the effective proposal depth.  Utility labels such as the
        # survival-weighted accepted prefix change scale with this depth, so a
        # controller fitted at one gamma must never be reused at another.
        # For catalogued backends this is the backend-visible proposal count,
        # not the raw engine window (DSpark/DFlash reserve one anchor row).
        "schema_version": 4,
        "model": {
            "pair_id": config.model.pair_id,
            "target_revision": config.model.target_revision,
            "drafter_revision": config.model.drafter_revision,
            "tokenizer_revision": config.model.tokenizer_revision,
        },
        "candidate": {
            # Public configs expose residual/lora/full; this field names the
            # concrete tail-bank layout used by the artifact.
            "weight_update_mode": config.tail_layout_mode,
            "adapter_rank": config.effective_adapter_rank,
            "optimizer": config.optimizer,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
            "trust_region_radius": config.trust_region_radius,
            "confidence_loss_weight": config.confidence_loss_weight,
            # For the shared single-step candidate the source-bound proximal KL
            # has exactly zero first-order gradient.  Binding an otherwise
            # irrelevant lambda value would prevent a TTS trace from training
            # the same L1/L2 candidate family.
            "proximal_contract": "source_bound_zero_gradient_single_step_v1",
            "update_stride": config.update_stride,
            "effective_proposal_depth": proposal_depth,
            "max_in_flight": config.async_.max_in_flight,
            "lifecycle": config.lifecycle,
        },
        "sampling": {
            "temperature": config.sampling.temperature,
            "top_p": config.sampling.top_p,
        },
        "trajectory": config.trajectory.model_dump(mode="json"),
    }


def validate_controller_artifact(
    config: AdaptationConfig,
    artifact: ControllerArtifact,
    parameter_layout_sha256: str | None = None,
) -> None:
    """Validate every frozen controller dependency before model loading."""
    key = config.method
    if artifact.model_pair_id != config.model.pair_id:
        raise ConfigError(
            "controller artifact model pair mismatch: "
            f"{artifact.model_pair_id!r} != {config.model.pair_id!r}"
        )
    if artifact.clock_variant != config.trajectory.clock_variant:
        raise ConfigError(
            "controller artifact trajectory clock mismatch: "
            f"{artifact.clock_variant!r} != {config.trajectory.clock_variant!r}"
        )
    from lightcone_spec.trajectory.features import FEATURE_SETS

    if artifact.feature_set not in FEATURE_SETS:
        raise ConfigError(
            f"controller artifact feature_set is unknown: {artifact.feature_set!r}"
        )
    n_features = len(FEATURE_SETS[artifact.feature_set])
    for name, predictor in (
        ("utility", artifact.utility_predictor),
        ("mismatch", artifact.mismatch_predictor),
        ("harmful", artifact.harmful_classifier),
    ):
        if not all(
            len(value) == n_features
            for value in (predictor.mean, predictor.std, predictor.coef)
        ):
            raise ConfigError(
                f"controller {name} predictor dimension does not match "
                f"feature_set {artifact.feature_set!r} ({n_features})"
            )
    if len(artifact.harmful_classifier.iso_x) != len(
        artifact.harmful_classifier.iso_y
    ):
        raise ConfigError("controller isotonic calibration arrays differ in length")
    if not artifact.distance_weights.frozen:
        raise ConfigError("controller artifact distance weights are not frozen")
    if not config.model.pair_id.startswith("toy_") and (
        not artifact.train_group_hash or not artifact.calibration_group_hash
    ):
        raise ConfigError(
            "real-model controller artifact must contain non-empty grouped "
            "train and calibration hashes"
        )
    if not config.model.pair_id.startswith("toy_") and (
        artifact.zvectorizer is None
        or artifact.zvectorizer.mean is None
        or artifact.zvectorizer.std is None
    ):
        raise ConfigError(
            "real-model controller artifact must contain its train-only "
            "trajectory normalizer"
        )
    if (
        not config.model.pair_id.startswith("toy_")
        and not artifact.extra.get("real_replay_data_sha256")
    ):
        raise ConfigError(
            "real-model controller artifact must contain its replay-data hash"
        )
    if (
        not config.model.pair_id.startswith("toy_")
        and artifact.extra.get("controller_utility_metric")
        != "survival_weighted_accepted_prefix_v1"
    ):
        raise ConfigError(
            "real-model controller must be fitted on survival-weighted "
            "accepted-prefix utility; KL/training loss is diagnostic only"
        )
    if not config.model.pair_id.startswith("toy_"):
        artifact_layout = artifact.extra.get("parameter_layout_sha256")
        if not isinstance(artifact_layout, str) or len(artifact_layout) != 64:
            raise ConfigError(
                "real-model controller artifact lacks its 64-character "
                "parameter-layout hash; recapture and refit before enabling "
                "adaptation"
            )
        if (
            parameter_layout_sha256 is not None
            and artifact_layout != parameter_layout_sha256
        ):
            raise ConfigError(
                "controller parameter layout does not match the active "
                "algorithm, update mode, rank, TP shard or model head"
            )
    safe_static_gate = key == "lc_gate" and artifact.gate_discard_all
    if (
        not config.model.pair_id.startswith("toy_")
        and key in ("lc_gate", "lc_damp")
        and not safe_static_gate
        and not artifact.extra.get("trace_exactness", {}).get("verified", False)
    ):
        raise ConfigError(
            f"{key} is disabled: controller trace exactness evidence is missing "
            "or contains a violation"
        )
    if (
        not config.model.pair_id.startswith("toy_")
        and key in ("lc_gate", "lc_damp")
        and not safe_static_gate
    ):
        oracle_gate = artifact.extra.get("oracle_replay_gate", {})
        oracle_key = "l1_eligible" if key == "lc_gate" else "l2_eligible"
        if not oracle_gate.get("complete") or not oracle_gate.get(oracle_key):
            raise ConfigError(
                f"{key} is disabled: held-out oracle replay did not establish "
                "a positive same-arrival utility ceiling"
            )
        tts_gate = artifact.extra.get("tts_paired_gate", {})
        if not tts_gate.get("complete") or not tts_gate.get(oracle_key):
            raise ConfigError(
                f"{key} is disabled: paired real-TTS barrier evidence is "
                "missing or its 95% CI does not establish positive utility"
            )
        learned_gate = artifact.extra.get("learned_policy_gate", {})
        if not learned_gate.get("complete") or not learned_gate.get(oracle_key):
            raise ConfigError(
                f"{key} is disabled: the fitted policy did not establish "
                "positive held-out utility over the same candidate's real "
                "TTS barrier"
            )
    if not config.model.pair_id.startswith("toy_") and not safe_static_gate:
        expected_identity = controller_runtime_identity(config)
        actual_identity = artifact.extra.get("controller_runtime_identity")
        actual_sha = artifact.extra.get("controller_runtime_identity_sha256")
        if actual_identity is None or actual_sha is None:
            raise ConfigError(
                "real-model controller artifact lacks its candidate/runtime "
                "identity; recapture and refit before enabling adaptation"
            )
        if sha256_json(actual_identity) != actual_sha:
            raise ConfigError("controller runtime identity hash is inconsistent")
        if actual_identity != expected_identity:
            raise ConfigError(
                "controller runtime identity does not match the active candidate "
                "generator, lifecycle, sampling, model revisions or trajectory"
            )
    if key == "lc_transport":
        l3_gate = artifact.extra.get("l3_gate", {})
        if artifact.transport_map is None:
            raise ConfigError("L3 transport artifact has no transport map")
        if not config.model.pair_id.startswith("toy_"):
            expected_map_sha = artifact.extra.get("transport_map_sha256")
            actual_map_sha = sha256_json(artifact.transport_map.to_dict())
            if expected_map_sha != actual_map_sha:
                raise ConfigError(
                    "L3 transport map hash is missing or inconsistent with "
                    "the fitted controller artifact"
                )
            l3_exactness_ok = bool(
                l3_gate.get("exactness", {}).get("verified", False)
            )
            if config.trace.l3_evaluation_only:
                # Break the otherwise circular evidence dependency without
                # opening the production gate: a bounded, benchmark-only pass
                # may execute the frozen map solely to collect the real
                # transported/L2/TTS utility labels required by the next fit.
                phase1_exactness_ok = bool(
                    artifact.extra.get("trace_exactness", {}).get(
                        "verified", False
                    )
                )
                if (
                    not l3_gate.get("evaluation_ready", False)
                    or not phase1_exactness_ok
                ):
                    raise ConfigError(
                        "L3 evaluation is disabled: the phase-1 artifact lacks "
                        "a frozen map with verified trace exactness"
                    )
            else:
                utility_gate = l3_gate.get(
                    "heldout_transported_utility_gate", {}
                )
                ci_tts = utility_gate.get("ci95_vs_tts")
                ci_l2 = utility_gate.get("ci95_vs_l2")
                evidence_ok = bool(
                    utility_gate.get("complete")
                    and utility_gate.get("eligible")
                    and utility_gate.get("utility_metric")
                    == "survival_weighted_accepted_prefix_v1"
                    and utility_gate.get("l3_contract")
                    == "joint_fisher_transport_adamw_damping_v1"
                    and utility_gate.get("pairing_contract")
                    == "exact_request_seed_concurrency_trace_stage_v1"
                    and utility_gate.get("transport_map_sha256")
                    == expected_map_sha
                    and isinstance(ci_tts, (list, tuple))
                    and len(ci_tts) == 2
                    and ci_tts[0] > 0.0
                    and isinstance(ci_l2, (list, tuple))
                    and len(ci_l2) == 2
                    and ci_l2[0] > 0.0
                )
                if not (
                    l3_gate.get("enabled", False)
                    and evidence_ok
                    and l3_exactness_ok
                ):
                    raise ConfigError(
                        "L3 transport is disabled: held-out replay did not pass "
                        "both paired TTS/L2 positive-95%-CI gates and the "
                        "L3-evaluation-specific zero-violation exactness gate"
                    )


def build_method(
    config: AdaptationConfig,
    shapes: AdapterShapes,
    basis: torch.Tensor,
    transport_variant: str = "joint",
    controller_artifact: ControllerArtifact | None = None,
    parameter_layout_sha256: str | None = None,
) -> MethodRuntime:
    gen_cfg = CandidateGeneratorConfig(
        lr=config.lr,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        trust_region_radius=config.trust_region_radius,
        confidence_loss_weight=config.confidence_loss_weight,
        lambda_prox=config.lambda_prox if config.method == "tts" else 0.0,
    )
    key = config.method
    if key == "static":
        return StaticMethod()
    if key == "sync_fresh":
        return SyncFreshMethod(shapes, basis, gen_cfg)
    if key == "tts":
        return TTSDSparkMethod(shapes, basis, gen_cfg)
    if key == "naive_async":
        return NaiveAsyncMethod(shapes, basis, gen_cfg)
    if key == "oracle_current":
        return OracleCurrentMethod(shapes, basis, gen_cfg)
    if key in ("onlinespec_ogd", "onlinespec_opt", "onlinespec_ens"):
        cls = {
            "onlinespec_ogd": OnlineSpecOGD,
            "onlinespec_opt": OnlineSpecOptimistic,
            "onlinespec_ens": OnlineSpecEnsemble,
        }[key]
        return cls(
            shapes,
            basis,
            lr=config.lr,
            grad_clip=config.grad_clip,
            trust_region_radius=config.trust_region_radius,
            confidence_loss_weight=config.confidence_loss_weight,
            seed=config.runtime.seed,
        )
    # Controller-driven methods need a frozen artifact.
    if config.controller.artifact_path is None:
        raise ConfigError(f"method {key} requires controller.artifact_path")
    artifact = controller_artifact or ControllerArtifact.load(
        config.controller.artifact_path
    )
    validate_controller_artifact(
        config,
        artifact,
        parameter_layout_sha256=parameter_layout_sha256,
    )
    if key == "lc_gate":
        return LCGateMethod(shapes, basis, gen_cfg, artifact)
    if key == "lc_damp":
        return LCDampMethod(shapes, basis, gen_cfg, artifact)
    if key == "lc_transport":
        return LCTransportMethod(
            shapes, basis, gen_cfg, artifact, variant=transport_variant
        )
    if key == "round_discard":
        return RoundDiscardMethod(shapes, basis, gen_cfg, artifact)
    if key == "wall_damp":
        return WallDampMethod(shapes, basis, gen_cfg, artifact)
    if key == "endpoint_gate":
        return EndpointGateMethod(shapes, basis, gen_cfg, artifact)
    if key == "parameter_only":
        return ParameterOnlyMethod(shapes, basis, gen_cfg, artifact)
    if key == "random_transport":
        return RandomTransportMethod(shapes, basis, gen_cfg, artifact)
    raise ConfigError(f"unknown method {key!r}")
