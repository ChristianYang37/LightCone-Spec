"""Teacher-forced counterfactual replay (spec 13.2, 7.6).

For each source round r on the update cadence, with injected delay d:

  - stale gradient g_r at the parameters phi_r current at r;
  - candidate delta u_r via the shared single-step AdamW generator;
  - arrival a = r + d; fresh gradient g_a at phi_a on the arrival prefix;
  - U_r(H) = sum_{j=a}^{a+H-1} [ l_j(phi_before) - l_j(phi_after) ] on
    the frozen trace windows;
  - features: delays, wall, endpoint, rho, parameter displacement,
    delta_z; labels: utility per horizon, harmful, relative mismatch,
    stale/fresh cosine; fresh-minus-stale gradient for transport fitting.

The replay applies each update at its arrival (in arrival order) so the
parameter trajectory and displacement are realistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from lightcone_spec.adapters.adapter_params import clip_gradient_global_norm
from lightcone_spec.adapters.losses import confidence_soft_targets
from lightcone_spec.methods.base import (
    TeacherSignal,
    apply_delta_with_trust_region,
    evaluate_loss_and_grad,
)
from lightcone_spec.methods.optim import AdamWDeltaState, adamw_delta
from lightcone_spec.replay.trace import SequenceTrace
from lightcone_spec.runtime.toy_model import ToyModelPair
from lightcone_spec.trajectory.distance import DistanceWeights, d_z
from lightcone_spec.trajectory.features import UpdateFeatureRow
from lightcone_spec.trajectory.labels import (
    MAIN_HORIZON,
    gradient_cosine,
    relative_gradient_mismatch,
)
from lightcone_spec.trajectory.zvector import ZVectorizer


@dataclass
class ReplayUpdateRecord:
    row: UpdateFeatureRow
    utilities: dict[int, float]
    cosine: float
    delta_g: np.ndarray  # fresh - stale (for transport fitting)
    delta_z: np.ndarray
    source_round: int
    arrival_round: int
    source_z_raw: np.ndarray | None = None
    arrival_z_raw: np.ndarray | None = None
    utility_metric: str = "training_loss_gain_v1"
    training_loss_gain: float | None = None
    full_candidate_utility: float | None = None
    actual_published_utility: float | None = None
    provenance_method: str | None = None
    candidate_arrival_round: int | None = None
    actual_arrival_round: int | None = None
    paired_tts_barrier: bool = False
    prefix_feature_exact: bool = True
    oracle_l1_utility: float | None = None
    oracle_l2_utility: float | None = None
    oracle_l2_kappa: float | None = None
    utility_by_kappa: dict[float, float] | None = None
    # Schema-v3 L3 evaluation evidence.  These values are populated only by
    # an explicit evaluation replay of the complete
    # Fisher -> transport -> AdamW -> damping path.  Gradient reconstruction
    # error is deliberately kept separate and can never stand in for these
    # survival-weighted acceptance utilities.
    evaluation_pair_id: str | None = None
    trace_stage_index: int | None = None
    trace_stage_count: int | None = None
    trace_capture_sampling: str | None = None
    evaluation_concurrency: int | None = None
    fresh_gradient_scope: str | None = None
    transported_candidate_utility: float | None = None
    paired_l2_utility: float | None = None
    transport_evaluation_contract: str | None = None
    transport_variant: str | None = None
    transport_map_sha256: str | None = None


@dataclass
class ReplayConfig:
    cadence: int = 12
    delay: int = 5
    horizons: tuple[int, ...] = (1, 4, MAIN_HORIZON, 16)
    lr: float = 1e-3
    grad_clip: float = 1.0
    trust_region_radius: float = 1.0
    confidence_loss_weight: float = 1.0
    window: int = 3  # draft positions per supervision window


def _window_signal(
    pair: ToyModelPair,
    trace: SequenceTrace,
    round_id: int,
    phi_ref: torch.Tensor,
    window: int,
) -> TeacherSignal:
    prefix = trace.prefix_at_round(round_id)
    prefixes = [prefix]
    cur = prefix
    # Teacher-forced window: the actual future tokens of the trace.
    start = len(prefix)
    for k in range(window - 1):
        if start + k < len(trace.tokens):
            cur = cur + (trace.tokens[start + k],)
        prefixes.append(cur)
    k = len(prefixes)
    u = np.stack([pair.projected_hidden(p) for p in prefixes])
    m = np.stack([pair.markov_embedding(p) for p in prefixes])
    base = np.stack([pair.base_drafter_logits(p, round_id) for p in prefixes])
    tgt = np.stack([pair.target_logits(p, round_id) for p in prefixes])
    conf = np.asarray([pair.base_confidence_logit(i) for i in range(k)])
    src = np.stack(
        [pair.proposal_logits(prefixes[i], round_id, phi_ref, i) for i in range(k)]
    )
    t_tgt = torch.from_numpy(tgt.astype(np.float32))
    t_src = torch.from_numpy(src.astype(np.float32))
    return TeacherSignal(
        source_round=round_id,
        source_version=0,
        u=torch.from_numpy(u.astype(np.float32)),
        m_prev=torch.from_numpy(m.astype(np.float32)),
        base_proposal_logits=torch.from_numpy(base.astype(np.float32)),
        base_confidence_logits=torch.from_numpy(conf.astype(np.float32)),
        target_logits=t_tgt,
        valid_mask=torch.ones(k, dtype=torch.bool),
        source_proposal_logits=t_src,
        confidence_targets=confidence_soft_targets(t_tgt, t_src),
    )


def _loss_at(
    pair: ToyModelPair,
    trace: SequenceTrace,
    round_id: int,
    phi: torch.Tensor,
    cfg: ReplayConfig,
) -> float:
    signal = _window_signal(pair, trace, round_id, phi, cfg.window)
    breakdown, _ = evaluate_loss_and_grad(
        phi,
        signal,
        pair.shapes,
        pair.basis,
        confidence_loss_weight=cfg.confidence_loss_weight,
        need_grad=False,
    )
    return float(breakdown.total.detach())


def replay_trace(
    pair: ToyModelPair,
    trace: SequenceTrace,
    cfg: ReplayConfig,
    weights: DistanceWeights,
    zvec: ZVectorizer,
) -> list[ReplayUpdateRecord]:
    records: list[ReplayUpdateRecord] = []
    phi = torch.zeros(pair.shapes.num_params(), dtype=torch.float32)
    phi0 = phi.clone()
    adam = AdamWDeltaState(num_params=pair.shapes.num_params())
    max_round = trace.max_round()
    by_round = {s.round_id: s for s in trace.states}
    wall_by_round = {r["round_id"]: r["wall_us"] for r in trace.rounds}
    prefix_len = {r["round_id"]: r["prefix_len"] for r in trace.rounds}

    for r in range(0, max_round + 1, cfg.cadence):
        a = r + cfg.delay
        if a > max_round or a + max(cfg.horizons) > max_round + max(cfg.horizons):
            if a > max_round:
                continue
        phi_source = phi.clone()
        sig_r = _window_signal(pair, trace, r, phi_source, cfg.window)
        _, g_stale = evaluate_loss_and_grad(
            phi_source, sig_r, pair.shapes, pair.basis,
            confidence_loss_weight=cfg.confidence_loss_weight,
        )
        assert g_stale is not None
        g_stale, _ = clip_gradient_global_norm(g_stale, cfg.grad_clip)

        # Parameter trajectory: previous updates already applied; phi at
        # arrival equals current phi (updates apply in arrival order).
        phi_arrival = phi.clone()
        sig_a = _window_signal(pair, trace, a, phi_arrival, cfg.window)
        _, g_fresh = evaluate_loss_and_grad(
            phi_arrival, sig_a, pair.shapes, pair.basis,
            confidence_loss_weight=cfg.confidence_loss_weight,
        )
        assert g_fresh is not None
        g_fresh, _ = clip_gradient_global_norm(g_fresh, cfg.grad_clip)

        u_r = adamw_delta(g_stale, adam, cfg.lr)
        phi_after = apply_delta_with_trust_region(
            phi_arrival, u_r, phi0, cfg.trust_region_radius
        )

        utilities: dict[int, float] = {}
        for h in cfg.horizons:
            lb, la = [], []
            for j in range(a, min(a + h, max_round + 1)):
                lb.append(_loss_at(pair, trace, j, phi_arrival, cfg))
                la.append(_loss_at(pair, trace, j, phi_after, cfg))
            utilities[h] = float(np.sum(lb) - np.sum(la)) if lb else 0.0

        rho = 0.0
        for j in range(r + 1, a + 1):
            rho += d_z(by_round[j - 1], by_round[j], weights)
        endp = d_z(by_round[r], by_round[a], weights)
        delta_z = zvec.delta_z(by_round[r], by_round[a])
        u_main = utilities.get(MAIN_HORIZON, 0.0)
        row = UpdateFeatureRow(
            sequence_id=trace.group_id,
            update_id=f"{trace.trajectory_id}-r{r}-d{cfg.delay}",
            round_delay=cfg.delay,
            token_delay=prefix_len[a] - prefix_len[r],
            wall_us=wall_by_round[a] - wall_by_round[r],
            endpoint_distance=endp,
            rho_path=rho,
            parameter_displacement=float(
                torch.linalg.vector_norm(phi_arrival - phi_source)
            ),
            utility=u_main,
            relative_gradient_mismatch=relative_gradient_mismatch(g_stale, g_fresh),
            harmful=int(u_main < 0.0),
        )
        records.append(
            ReplayUpdateRecord(
                row=row,
                utilities=utilities,
                cosine=gradient_cosine(g_stale, g_fresh),
                delta_g=(g_fresh - g_stale).numpy().astype(np.float64),
                delta_z=delta_z,
                source_round=r,
                arrival_round=a,
            )
        )
        # Apply the update (arrival order) so later sources see movement.
        phi = phi_after
    return records
