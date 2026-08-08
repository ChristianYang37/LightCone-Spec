"""Exactness harness (spec 3.4, 13.1 P0-B, 15.4).

Three layers:
  1. exact enumeration on the 4-symbol horizon-6 toy: TV <= 1e-10 for the
     correct protocol, including version changes across rounds;
  2. a deliberately injected wrong-version race whose TV must be >= 0.05
     -- if the harness fails to flag it, the harness itself fails;
  3. Monte Carlo conformance of the actual Philox sampler against the
     enumerated law (chi-square on preregistered bins, Holm-corrected).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from lightcone_spec.benchmarks.synthetic import make_two_regime_world
from lightcone_spec.runtime.exact_sampler import (
    enumerate_speculative_law,
    enumerate_target_law,
    run_speculative_round,
    total_variation_between_laws,
)
from lightcone_spec.runtime.toy_model import make_toy_pair

TV_EXACT_LIMIT = 1e-10
TV_CANARY_MIN = 0.05
MC_ALPHA = 0.01


@dataclass
class ExactnessReport:
    tv_correct_static: float = float("nan")
    tv_correct_version_change: float = float("nan")
    tv_race_canary: float = float("nan")
    race_flagged: bool = False
    mc_pvalues: list[float] = field(default_factory=list)
    mc_holm_pass: bool = False
    engine_canary_caught: bool = False
    passed: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def run_exactness_suite(
    horizon: int = 6,
    depth: int = 3,
    mc_samples: int = 20000,
    seed: int = 0,
) -> ExactnessReport:
    report = ExactnessReport()
    world = make_two_regime_world(seed)
    pair = make_toy_pair(world, seed=seed)
    phi_a = torch.zeros(pair.shapes.num_params())
    torch.manual_seed(seed)
    phi_b = torch.randn(pair.shapes.num_params()) * 0.3

    def target(prefix):
        return pair.target_dist(prefix, 0)

    def proposal(phi):
        return lambda prefix: pair.proposal_dist(prefix, 0, phi)

    tgt_law = enumerate_target_law(target, horizon, 4)

    # 1a. static proposal.
    law_static = enumerate_speculative_law(
        target, lambda r: proposal(phi_a), horizon, depth
    )
    report.tv_correct_static = total_variation_between_laws(law_static, tgt_law)

    # 1b. version changes across rounds (correct protocol).
    law_vc = enumerate_speculative_law(
        target,
        lambda r: proposal(phi_a if r % 2 == 0 else phi_b),
        horizon,
        depth,
    )
    report.tv_correct_version_change = total_variation_between_laws(law_vc, tgt_law)

    # 2. deliberate race canary.
    law_race = enumerate_speculative_law(
        target,
        lambda r: proposal(phi_a),
        horizon,
        depth,
        denominator_for_round=lambda r: proposal(phi_b),
    )
    report.tv_race_canary = total_variation_between_laws(law_race, tgt_law)
    report.race_flagged = report.tv_race_canary >= TV_CANARY_MIN

    # 3. Monte Carlo conformance of the Philox sampler (first committed
    # token = preregistered bins over the alphabet, per round version).
    from scipy.stats import chisquare

    pvals = []
    for version_phi, tag in ((phi_a, "v0"), (phi_b, "v1")):
        counts = np.zeros(4)
        expected_first = _first_token_marginal(target, proposal(version_phi), depth)
        for i in range(mc_samples):
            outcome = run_speculative_round(
                (), target, proposal(version_phi), depth,
                seed, f"mc-{tag}-{i}", 0,
            )
            counts[outcome.committed_tokens[0]] += 1
        expected = expected_first * mc_samples
        stat = chisquare(counts, expected)
        pvals.append(float(stat.pvalue))
    report.mc_pvalues = pvals
    report.mc_holm_pass = _holm_all_pass(pvals, MC_ALPHA)

    # 4. engine-level injected canary must be caught as failed_exactness.
    report.engine_canary_caught = _engine_canary_caught(pair, seed)

    report.passed = (
        report.tv_correct_static <= TV_EXACT_LIMIT
        and report.tv_correct_version_change <= TV_EXACT_LIMIT
        and report.race_flagged
        and report.mc_holm_pass
        and report.engine_canary_caught
    )
    return report


def _first_token_marginal(target, proposal, depth) -> np.ndarray:
    """Exact marginal of the first committed token (equals the target
    first-token law under the correct protocol)."""
    law = enumerate_speculative_law(target, lambda r: proposal, 1, depth)
    out = np.zeros(4)
    for seq, p in law.items():
        out[seq[0]] += p
    return out


def _holm_all_pass(pvals: list[float], alpha: float) -> bool:
    """Holm step-down: conformance requires that *no* hypothesis is
    rejected."""
    m = len(pvals)
    for rank, p in enumerate(sorted(pvals)):
        if p < alpha / (m - rank):
            return False
    return True


def _engine_canary_caught(pair, seed: int) -> bool:
    from lightcone_spec.methods.base import CandidateGeneratorConfig
    from lightcone_spec.methods.simple import NaiveAsyncMethod
    from lightcone_spec.runtime.engine import EngineConfig, ReferenceEngine

    method = NaiveAsyncMethod(
        pair.shapes,
        pair.basis,
        CandidateGeneratorConfig(
            lr=1e-3, grad_clip=1.0, trust_region_radius=1.0,
            confidence_loss_weight=1.0, lambda_prox=0.0,
        ),
    )
    engine = ReferenceEngine(
        pair,
        method,
        EngineConfig(
            method_key="naive_async",
            seed=seed,
            update_stride=4,
            max_rounds=8,
            max_new_tokens=16,
            inject_version_race=True,
        ),
    )
    res = engine.run_request("canary-req")
    return res.status == "failed_exactness"
