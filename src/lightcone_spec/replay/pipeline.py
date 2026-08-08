"""Replay fitting pipeline (spec 13.2): distance-weight selection,
predictor fitting/calibration/evaluation, gate threshold, damping
radius, transport PCA/ridge, artifact freezing and the H1 verdict.

All fitting happens on sequence-grouped train/calibration splits; test
groups are only ever evaluated, never used for selection.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from lightcone_spec.controller.artifact import ControllerArtifact
from lightcone_spec.controller.damping import (
    calibration_radius,
    damping_factor,
    select_utility_calibrated_radius,
)
from lightcone_spec.controller.gate import select_gate_threshold
from lightcone_spec.locking.hashing import sha256_json
from lightcone_spec.replay.counterfactual import ReplayUpdateRecord
from lightcone_spec.replay.splits import split_of_group
from lightcone_spec.trajectory.features import FEATURE_SETS, design_matrix
from lightcone_spec.trajectory.predictors import HarmfulClassifier, RidgePredictor
from lightcone_spec.trajectory.zvector import ZVectorizer
from lightcone_spec.transport.fit import TransportMap, fit_transport_map
from lightcone_spec.transport.fit import hash_group

# Coarse simplex grid for (a_p, a_h, a_e) selection (spec 7.2): weights
# are fitted on train/calibration only and frozen afterwards.
SIMPLEX_STEP = 0.25


def simplex_grid(step: float = SIMPLEX_STEP) -> list[tuple[float, float, float]]:
    n = int(round(1.0 / step))
    out = []
    for i, j in itertools.product(range(n + 1), repeat=2):
        if i + j <= n:
            out.append((i * step, j * step, 1.0 - (i + j) * step))
    return [w for w in out if max(w) < 1.0 + 1e-9 and sum(w) > 0]


@dataclass
class PredictorEvaluation:
    feature_set: str
    mae: float
    spearman: float
    auroc: float
    auprc: float
    calibration_error: float
    n_test: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ReplayFitResult:
    artifact: ControllerArtifact
    evaluations: dict[str, PredictorEvaluation]
    h1: dict
    harmful_rate: float
    mean_cosine: float
    split_sizes: dict[str, int] = field(default_factory=dict)


def _by_split(records: list[ReplayUpdateRecord], seed: int = 0):
    out = {"train": [], "calibration": [], "test": []}
    for rec in records:
        out[split_of_group(rec.row.sequence_id, seed)].append(rec)
    return out


def _groups(records: list[ReplayUpdateRecord]) -> np.ndarray:
    return np.asarray([hash_group(r.row.sequence_id) for r in records])


def _calibration_constant_gate_delays(
    round_delays: np.ndarray,
    features: np.ndarray,
    utilities: np.ndarray,
    harm_probabilities: np.ndarray,
    *,
    threshold: float,
    discard_all: bool,
) -> tuple[dict[int, np.ndarray], list[int], list[int]]:
    """Select no-predictor L1 paths from calibration rows only.

    The returned hints are not sufficient to enable L1.  Real replay later
    evaluates their exact runtime semantics on disjoint held-out groups and
    requires a positive paired-TTS CI.
    """

    if not (
        len(round_delays)
        == len(features)
        == len(utilities)
        == len(harm_probabilities)
    ):
        raise ValueError("constant gate calibration arrays differ in length")
    constant_features: dict[int, np.ndarray] = {}
    constant_discards: list[int] = []
    constant_applies: list[int] = []
    for delay in sorted(set(round_delays.tolist())):
        mask = round_delays == delay
        bucket = features[mask]
        bucket_harm = harm_probabilities[mask]
        # A delay-only fast path is semantically identical to the fitted
        # predictor only when *all* predictor inputs in that delay bucket are
        # the same.  Otherwise freezing apply/discard by delay silently drops
        # source quality and trajectory state at runtime.
        feature_is_constant = bool(
            len(bucket)
            and np.allclose(bucket, bucket[0], rtol=0.0, atol=1e-12)
        )
        if (
            delay > 0
            and feature_is_constant
            and np.all(bucket_harm > threshold)
        ):
            constant_discards.append(int(delay))
        elif (
            delay > 0
            and feature_is_constant
            and not discard_all
            and np.all(bucket_harm <= threshold)
            and np.all(utilities[mask] >= 0.0)
        ):
            constant_applies.append(int(delay))
        if feature_is_constant:
            constant_features[int(delay)] = bucket[0]
    return constant_features, constant_discards, constant_applies


def _fit_eval_predictor(
    feature_set: str,
    splits: dict[str, list[ReplayUpdateRecord]],
) -> tuple[RidgePredictor, RidgePredictor, HarmfulClassifier, PredictorEvaluation]:
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows_tr = [r.row for r in splits["train"]]
    rows_cal = [r.row for r in splits["calibration"]]
    rows_te = [r.row for r in splits["test"]]
    x_tr, _ = design_matrix(rows_tr, feature_set)
    x_cal, _ = design_matrix(rows_cal, feature_set)
    x_te, _ = design_matrix(rows_te, feature_set)
    y_tr = np.asarray([r.utility for r in rows_tr])
    y_te = np.asarray([r.utility for r in rows_te])
    g_tr = _groups(splits["train"])

    util = RidgePredictor().fit(x_tr, y_tr, g_tr)
    mism = RidgePredictor(log1p_target=True).fit(
        x_tr,
        np.asarray([r.relative_gradient_mismatch for r in rows_tr]),
        g_tr,
    )
    harm = HarmfulClassifier().fit(
        x_tr, np.asarray([r.harmful for r in rows_tr], dtype=np.float64), g_tr
    )
    harm.calibrate(x_cal, np.asarray([r.harmful for r in rows_cal], dtype=np.float64))

    pred_u = util.predict(x_te)
    mae = float(np.abs(pred_u - y_te).mean()) if len(y_te) else float("nan")
    if len(y_te) >= 3 and np.std(pred_u) > 0 and np.std(y_te) > 0:
        rho = float(spearmanr(pred_u, y_te).statistic)
    else:
        rho = float("nan")
    y_h = np.asarray([r.harmful for r in rows_te], dtype=np.float64)
    p_h = np.nan_to_num(harm.probability(x_te), nan=0.5)
    if len(np.unique(y_h)) == 2:
        auroc = float(roc_auc_score(y_h, p_h))
        auprc = float(average_precision_score(y_h, p_h))
    else:
        auroc, auprc = float("nan"), float("nan")
    cal_err = float(np.abs(p_h.mean() - y_h.mean())) if len(y_h) else float("nan")
    return (
        util,
        mism,
        harm,
        PredictorEvaluation(
            feature_set=feature_set,
            mae=mae,
            spearman=rho,
            auroc=auroc,
            auprc=auprc,
            calibration_error=cal_err,
            n_test=len(rows_te),
        ),
    )


def fit_replay_pipeline(
    records: list[ReplayUpdateRecord],
    model_pair_id: str,
    zvec: ZVectorizer,
    transport_rank: int = 16,
    damping_kernel: str = "exponential",
    unsafe_apply_limit: float = 0.10,
    seed: int = 0,
    bootstrap_b: int = 5000,
) -> ReplayFitResult:
    splits = _by_split(records, seed)
    if any(not splits[name] for name in ("train", "calibration", "test")):
        raise ValueError(
            "replay needs non-empty grouped train, calibration and test splits "
            f"for split seed {seed}"
        )

    evaluations: dict[str, PredictorEvaluation] = {}
    fitted: dict[str, tuple] = {}
    for fs in FEATURE_SETS:
        fitted[fs] = _fit_eval_predictor(fs, splits)
        evaluations[fs] = fitted[fs][3]

    # ---- H1 verdict (spec 14.7) ---------------------------------------
    delay_mae = evaluations["delay_only"].mae
    # `delay` is the best held-out MAE among round/token/wall delay
    # combinations; wall_only is the other delay-only member.
    delay_best = min(delay_mae, evaluations["wall_only"].mae)
    path_mae = evaluations["path_length"].mae
    improvement = (delay_best - path_mae) / delay_best if delay_best > 0 else 0.0

    # Cluster bootstrap CI on the improvement (clusters = sequence groups).
    rows_te = [r.row for r in splits["test"]]
    y_te = np.asarray([r.utility for r in rows_te])
    groups_te = np.asarray([r.sequence_id for r in rows_te])
    uniq = np.unique(groups_te)
    rng = np.random.Generator(np.random.PCG64(0))
    boot: list[float] = []
    util_delay = fitted["delay_only"][0]
    util_wall = fitted["wall_only"][0]
    util_path = fitted["path_length"][0]
    x_delay, _ = design_matrix(rows_te, "delay_only")
    x_wall, _ = design_matrix(rows_te, "wall_only")
    x_path, _ = design_matrix(rows_te, "path_length")
    err_delay = np.abs(util_delay.predict(x_delay) - y_te)
    err_wall = np.abs(util_wall.predict(x_wall) - y_te)
    err_path = np.abs(util_path.predict(x_path) - y_te)
    for _ in range(min(bootstrap_b, 5000)):
        take = rng.choice(len(uniq), size=len(uniq), replace=True)
        mask_idx = np.concatenate(
            [np.where(groups_te == uniq[i])[0] for i in take]
        )
        d = min(err_delay[mask_idx].mean(), err_wall[mask_idx].mean())
        p = err_path[mask_idx].mean()
        boot.append((d - p) / d if d > 0 else 0.0)
    ci = (
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    ) if boot else (float("nan"), float("nan"))
    h1 = {
        "delay_mae": delay_best,
        "path_mae": path_mae,
        "relative_improvement": improvement,
        "ci95": ci,
        "threshold": 0.15,
        "pass": bool(improvement >= 0.15 and ci[0] > 0),
    }

    # ---- controller pieces from the confirmatory path-length clock -----
    util_p, mism_p, harm_p, _ = fitted["path_length"]
    rows_cal = [r.row for r in splits["calibration"]]
    x_cal, _ = design_matrix(rows_cal, "path_length")
    harm_probs_cal = harm_p.probability(x_cal)
    utils_cal = np.asarray([r.utility for r in rows_cal])
    gate = select_gate_threshold(harm_probs_cal, utils_cal, unsafe_apply_limit)
    # Some causal arrival buckets collapse the selected controller feature to
    # one deterministic value (notably path length at the one-round pipeline
    # minimum). Freeze those all-discard buckets so runtime can skip predictor
    # kernels and, crucially, avoid publishing a zero adapter/version.
    round_delays = np.asarray([r.round_delay for r in rows_cal], dtype=np.int64)
    (
        constant_features_by_delay,
        constant_discard_delays,
        constant_apply_delays,
    ) = _calibration_constant_gate_delays(
        round_delays,
        x_cal,
        utils_cal,
        harm_p.probability(x_cal),
        threshold=gate.threshold,
        discard_all=gate.discard_all,
    )
    mism_pred_cal = mism_p.predict(x_cal)
    calibration_grids = [record.utility_by_kappa for record in splits["calibration"]]
    if calibration_grids and all(grid is not None for grid in calibration_grids):
        radius, damping_calibration = select_utility_calibrated_radius(
            mism_pred_cal,
            [grid for grid in calibration_grids if grid is not None],
            kernel=damping_kernel,
        )
    else:
        radius = calibration_radius(mism_pred_cal, utils_cal)
        damping_calibration = {
            "contract": "positive_utility_mismatch_q90_v1",
            "candidate_count": None,
            "mean_calibration_utility": None,
            "mean_kappa": None,
        }
    constant_controller_profiles = {}
    for delay, features in constant_features_by_delay.items():
        one = np.asarray(features, dtype=np.float64).reshape(1, -1)
        predicted_mismatch = float(mism_p.predict(one)[0])
        constant_controller_profiles[str(delay)] = {
            "features": one[0].tolist(),
            "predicted_utility": float(util_p.predict(one)[0]),
            "predicted_mismatch": predicted_mismatch,
            "predicted_harm_probability": float(harm_p.probability(one)[0]),
            "damping_factor": float(
                damping_factor(predicted_mismatch, radius, damping_kernel)
            ),
        }
    calibration_count = max(len(round_delays), 1)
    constant_fast_path_calibration_coverage = {
        "records": int(len(round_delays)),
        "l1_constant_apply_fraction": float(
            np.isin(round_delays, constant_apply_delays).sum()
            / calibration_count
        ),
        "l1_constant_discard_fraction": float(
            np.isin(round_delays, constant_discard_delays).sum()
            / calibration_count
        ),
        "l2_constant_profile_fraction": float(
            np.isin(
                round_delays,
                [int(delay) for delay in constant_controller_profiles],
            ).sum()
            / calibration_count
        ),
    }

    # ---- transport (train split only) ----------------------------------
    train_dg = np.stack([r.delta_g for r in splits["train"]])
    train_dz = np.stack([r.delta_z for r in splits["train"]])
    tmap = fit_transport_map(
        train_dg,
        train_dz,
        [r.row.sequence_id for r in splits["train"]],
        rank=transport_rank,
    )

    artifact = ControllerArtifact(
        model_pair_id=model_pair_id,
        clock_variant="target_only",
        feature_set="path_length",
        distance_weights=_frozen_weights(),
        utility_predictor=util_p,
        mismatch_predictor=mism_p,
        harmful_classifier=harm_p,
        gate_threshold=gate.threshold,
        gate_discard_all=gate.discard_all,
        damping_radius=radius,
        damping_kernel=damping_kernel,
        zvectorizer=zvec,
        transport_map=tmap,
        train_group_hash=sha256_json(
            sorted({r.row.sequence_id for r in splits["train"]})
        ),
        calibration_group_hash=sha256_json(
            sorted({r.row.sequence_id for r in splits["calibration"]})
        ),
        extra={
            "gate_unsafe_apply_rate": gate.unsafe_apply_rate,
            "gate_apply_fraction": gate.apply_fraction,
            "gate_constant_discard_delays": constant_discard_delays,
            "gate_constant_apply_delays": constant_apply_delays,
            "constant_controller_profiles": constant_controller_profiles,
            "constant_fast_path_source": "calibration_only_v1",
            "constant_fast_path_calibration_coverage": (
                constant_fast_path_calibration_coverage
            ),
            "damping_calibration": damping_calibration,
            "round_discard_D": _round_discard_threshold(splits["calibration"]),
            "wall_damp_radius_us": _wall_damp_radius(splits["calibration"]),
            "fisher_decay": 0.99,
        },
    )
    all_rows = [r.row for r in records]
    return ReplayFitResult(
        artifact=artifact,
        evaluations=evaluations,
        h1=h1,
        harmful_rate=float(np.mean([r.harmful for r in all_rows])),
        mean_cosine=float(np.mean([r.cosine for r in records])),
        split_sizes={k: len(v) for k, v in splits.items()},
    )


def _frozen_weights():
    from lightcone_spec.trajectory.distance import DistanceWeights

    w = DistanceWeights(a_p=1 / 3, a_h=1 / 3, a_e=1 / 3)
    w.frozen = True
    return w


def select_distance_weights(
    records_fn,
    seed: int = 0,
):
    """Grid-select (a_p, a_h, a_e) on train/calibration only: for each
    simplex point recompute rho/endpoint features via `records_fn(w)` and
    keep the weights whose path-length ridge has the lowest calibration
    MAE (ties -> more uniform). Returns (weights, best_mae)."""
    from lightcone_spec.trajectory.distance import DistanceWeights

    best_w, best_mae, best_spread = None, np.inf, np.inf
    for a_p, a_h, a_e in simplex_grid():
        if a_p + a_h + a_e <= 0:
            continue
        w = DistanceWeights(a_p=a_p, a_h=a_h, a_e=a_e)
        records = records_fn(w)
        splits = _by_split(records, seed)
        if not splits["train"] or not splits["calibration"]:
            continue
        rows_tr = [r.row for r in splits["train"]]
        rows_cal = [r.row for r in splits["calibration"]]
        x_tr, _ = design_matrix(rows_tr, "path_length")
        x_cal, _ = design_matrix(rows_cal, "path_length")
        y_tr = np.asarray([r.utility for r in rows_tr])
        y_cal = np.asarray([r.utility for r in rows_cal])
        pred = RidgePredictor().fit(x_tr, y_tr, _groups(splits["train"]))
        mae = float(np.abs(pred.predict(x_cal) - y_cal).mean())
        spread = float(np.var([a_p, a_h, a_e]))
        if mae < best_mae - 1e-12 or (
            abs(mae - best_mae) <= 1e-12 and spread < best_spread
        ):
            best_w, best_mae, best_spread = w, mae, spread
    if best_w is None:
        raise ValueError("no feasible distance weights")
    best_w.frozen = True
    return best_w, best_mae


def _round_discard_threshold(cal: list[ReplayUpdateRecord]) -> float:
    """Calibration-chosen D for the Round-Discard control: the delay whose
    cutoff maximizes mean utility of kept updates."""
    if not cal:
        return 5.0
    delays = sorted({r.row.round_delay for r in cal})
    best_d, best_u = delays[-1], -np.inf
    for d in delays:
        kept = [r.row.utility for r in cal if r.row.round_delay <= d]
        u = float(np.mean([*(kept or [0.0])]))
        if u > best_u:
            best_d, best_u = d, u
    return float(best_d)


def _wall_damp_radius(cal: list[ReplayUpdateRecord]) -> float:
    if not cal:
        return 1e6
    walls = np.asarray([r.row.wall_us for r in cal if r.row.utility > 0])
    if walls.size == 0:
        walls = np.asarray([r.row.wall_us for r in cal])
    return float(max(np.quantile(walls, 0.9), 1.0))
