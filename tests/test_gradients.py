"""Finite-difference gradient check for the common loss and the TTS
single-step lambda-invariance conformance test (spec 15.2)."""

from __future__ import annotations

from dataclasses import replace

import torch

from lightcone_spec.adapters.adapter_params import (
    AdapterParams,
    canonicalize_master_vector,
    parameter_views,
    rmsnorm,
)
from lightcone_spec.adapters.losses import common_loss
from lightcone_spec.methods.base import (
    CandidateGeneratorConfig,
    CommonCandidateGenerator,
    evaluate_loss_and_grad,
    survival_weighted_acceptance,
)
from lightcone_spec.methods.simple import NaiveAsyncMethod, TTSDSparkMethod
from lightcone_spec.methods.optim import AdamWDeltaState, adamw_delta

from conftest import make_signal


def test_finite_difference_gradient(shapes, basis):
    torch.manual_seed(1)
    signal = make_signal(seed=1)
    phi = torch.randn(shapes.num_params()) * 0.05

    loss, grad = evaluate_loss_and_grad(phi, signal, shapes, basis)
    assert grad is not None
    assert all(
        value.grad_fn is None
        for value in (
            loss.total,
            loss.distillation,
            loss.confidence,
            loss.proximal,
        )
    )

    eps = 1e-3
    rng = torch.Generator().manual_seed(2)
    idx = torch.randperm(shapes.num_params(), generator=rng)[:12]
    for i in idx.tolist():
        e = torch.zeros_like(phi)
        e[i] = eps
        lp, _ = evaluate_loss_and_grad(
            phi + e, signal, shapes, basis, need_grad=False
        )
        lm, _ = evaluate_loss_and_grad(
            phi - e, signal, shapes, basis, need_grad=False
        )
        fd = (float(lp.total.detach()) - float(lm.total.detach())) / (2 * eps)
        assert abs(fd - float(grad[i])) < 5e-3, (
            f"param {i}: finite diff {fd} vs autograd {float(grad[i])}"
        )


def test_closed_form_gradient_matches_autograd_oracle(shapes, basis):
    torch.manual_seed(17)
    original = make_signal(seed=17)
    phi = torch.randn(shapes.num_params()) * 0.03
    mask = original.valid_mask.clone()
    mask[-1] = False

    for confidence_targets in (original.confidence_targets, None):
        for lambda_prox in (0.0, 0.4):
            scale = torch.linspace(0.5, 1.5, original.u.shape[0])
            signal = replace(
                original,
                valid_mask=mask,
                confidence_targets=confidence_targets,
                proposal_logit_scale=scale,
            )
            loss, actual = evaluate_loss_and_grad(
                phi,
                signal,
                shapes,
                basis,
                confidence_loss_weight=0.7,
                lambda_prox=lambda_prox,
            )

            adapter = AdapterParams(shapes, basis)
            adapter.load_flat(phi)
            q_logits = (
                signal.base_proposal_logits
                + adapter.draft_logit_residual(signal.u)
                + adapter.markov_logit_residual(signal.m_prev)
            ) * scale[:, None]
            conf_logits = (
                signal.base_confidence_logits
                + adapter.confidence_residual(signal.u, signal.m_prev)
            )
            expected_loss = common_loss(
                signal.target_logits * scale[:, None],
                q_logits,
                conf_logits,
                confidence_targets,
                mask,
                confidence_loss_weight=0.7,
                source_proposal_logits=signal.source_proposal_logits,
                lambda_prox=lambda_prox,
            )
            expected_loss.total.backward()
            expected = adapter.grad_flat()

            assert torch.allclose(loss.total, expected_loss.total.detach(), atol=1e-7)
            assert actual is not None
            assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_tts_single_step_lambda_invariance(shapes, basis):
    """At phi = phi_source the proximal-KL gradient is exactly zero, so
    the single-step TTS candidate must not depend on lambda_prox."""
    signal = make_signal(seed=3)
    phi_source = torch.zeros(shapes.num_params())
    deltas = []
    for lam in (0.0, 0.1, 10.0):
        gen = CommonCandidateGenerator(
            shapes,
            basis,
            CandidateGeneratorConfig(
                lr=1e-3, grad_clip=1.0, trust_region_radius=1.0,
                confidence_loss_weight=1.0, lambda_prox=lam,
            ),
        )
        cand = gen.candidate(phi_source, signal)
        deltas.append(cand.candidate_delta)
    assert torch.allclose(deltas[0], deltas[1], atol=1e-6)
    assert torch.allclose(deltas[0], deltas[2], atol=1e-6)


def test_tts_and_l0_share_the_same_source_bound_candidate(shapes, basis):
    """Scheduling policy is the only intended TTS/L0 difference here."""
    signal = make_signal(seed=31)
    phi_source = torch.zeros(shapes.num_params())
    common = dict(
        lr=1e-3,
        grad_clip=1.0,
        trust_region_radius=1.0,
        confidence_loss_weight=1.0,
    )
    tts = TTSDSparkMethod(
        shapes,
        basis,
        CandidateGeneratorConfig(**common, lambda_prox=4.0),
    )
    l0 = NaiveAsyncMethod(
        shapes,
        basis,
        CandidateGeneratorConfig(**common, lambda_prox=0.0),
    )

    tts_candidate = tts.make_candidate(phi_source, signal)
    l0_candidate = l0.make_candidate(phi_source, signal)

    assert tts_candidate is not None and l0_candidate is not None
    assert torch.equal(tts_candidate.raw_gradient, l0_candidate.raw_gradient)
    assert torch.equal(
        tts_candidate.candidate_delta, l0_candidate.candidate_delta
    )


def test_common_candidate_applies_gradient_consensus_before_clip_and_adam(
    shapes, basis
):
    cfg = CandidateGeneratorConfig(
        lr=1e-3,
        grad_clip=1e6,
        trust_region_radius=1.0,
        confidence_loss_weight=1.0,
        lambda_prox=0.0,
    )
    phi = torch.zeros(shapes.num_params())
    signal = make_signal(seed=33)
    baseline = CommonCandidateGenerator(shapes, basis, cfg).candidate(phi, signal)
    seen = []

    def half_consensus(grad, finite_t):
        seen.append((grad.clone(), finite_t))
        return grad * 0.5, finite_t

    generator = CommonCandidateGenerator(shapes, basis, cfg)
    generator.bind_gradient_consensus(half_consensus)
    candidate = generator.candidate(phi, signal)

    assert len(seen) == 1
    assert torch.allclose(candidate.raw_gradient, baseline.raw_gradient * 0.5)
    # Adam's first normalized step is scale invariant, but it must consume the
    # consensus gradient and advance exactly once.
    assert generator.state.step == 1


def test_invalid_adam_candidate_does_not_mutate_moments_or_step():
    state = AdamWDeltaState(num_params=4)
    state.exp_avg.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    state.exp_avg_sq.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    before = state.state_dict()

    delta = adamw_delta(torch.zeros(4), state, 1e-3, valid=False)

    assert torch.count_nonzero(delta) == 0
    assert state.step == before["step"]
    assert torch.equal(state.exp_avg, before["exp_avg"])
    assert torch.equal(state.exp_avg_sq, before["exp_avg_sq"])


def test_adamw_delta_matches_torch_decoupled_weight_decay():
    parameter = torch.tensor([1.5, -0.25, 2.0], dtype=torch.float32)
    gradient = torch.tensor([0.2, -0.4, 0.1], dtype=torch.float32)
    lr = 3e-4
    weight_decay = 1e-2

    reference = torch.nn.Parameter(parameter.clone())
    optimizer = torch.optim.AdamW(
        [reference],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    reference.grad = gradient.clone()
    optimizer.step()

    state = AdamWDeltaState(num_params=parameter.numel())
    delta = adamw_delta(
        gradient,
        state,
        lr,
        parameter=parameter,
        weight_decay=weight_decay,
    )

    torch.testing.assert_close(parameter + delta, reference.detach())


def test_invalid_adamw_candidate_does_not_apply_weight_decay():
    parameter = torch.tensor([1.0, -2.0])
    state = AdamWDeltaState(num_params=2)

    delta = adamw_delta(
        torch.zeros(2),
        state,
        1e-3,
        valid=False,
        parameter=parameter,
        weight_decay=0.1,
    )

    assert torch.equal(delta, torch.zeros_like(delta))
    assert state.step == 0


def test_bfloat16_online_add_preserves_tts_l0_source_parity(shapes, basis):
    """The source q must include the same BF16 casts/add order as serving."""
    signal = make_signal(seed=37)
    phi_source = torch.randn(shapes.num_params()) * 0.02
    adapter = AdapterParams(shapes, basis)
    adapter.load_flat(phi_source)
    base = signal.base_proposal_logits.to(torch.bfloat16)
    online_q = base + adapter.draft_logit_residual(signal.u).to(torch.bfloat16)
    online_q = online_q + adapter.markov_logit_residual(signal.m_prev).to(
        torch.bfloat16
    )
    signal = replace(
        signal,
        base_proposal_logits=base,
        source_proposal_logits=online_q.float(),
    )
    common = dict(
        lr=1e-3,
        grad_clip=1.0,
        trust_region_radius=1.0,
        confidence_loss_weight=1.0,
    )
    tts = TTSDSparkMethod(
        shapes,
        basis,
        CandidateGeneratorConfig(**common, lambda_prox=8.0),
    )
    l0 = NaiveAsyncMethod(
        shapes,
        basis,
        CandidateGeneratorConfig(**common, lambda_prox=0.0),
    )

    tts_candidate = tts.make_candidate(phi_source, signal)
    l0_candidate = l0.make_candidate(phi_source, signal)

    assert tts_candidate is not None and l0_candidate is not None
    assert torch.equal(tts_candidate.raw_gradient, l0_candidate.raw_gradient)
    assert torch.equal(
        tts_candidate.candidate_delta, l0_candidate.candidate_delta
    )


def test_bfloat16_closed_form_gradient_matches_online_autograd(shapes, basis):
    signal = make_signal(seed=39)
    basis_native = basis.to(torch.bfloat16)
    signal = replace(
        signal,
        u=signal.u.to(torch.bfloat16),
        m_prev=signal.m_prev.to(torch.bfloat16),
        base_proposal_logits=signal.base_proposal_logits.to(torch.bfloat16),
    )
    # The serving bank stores a Q-DQ canonical FP32 master and exposes the
    # corresponding model-dtype row.  The oracle must differentiate through
    # that exact native graph; a legacy FP32 AdapterParams graph can happen to
    # produce the same rounded logits while yielding a different gradient.
    phi = canonicalize_master_vector(
        torch.randn(shapes.num_params()) * 0.02,
        torch.bfloat16,
    )
    forward_phi = phi.to(torch.bfloat16)
    actual_loss, actual_grad = evaluate_loss_and_grad(
        phi,
        signal,
        shapes,
        basis_native,
        confidence_loss_weight=0.7,
        forward_phi=forward_phi,
    )

    oracle_master = phi.detach().clone().requires_grad_(True)
    views = parameter_views(oracle_master.to(torch.bfloat16), shapes)
    hidden_coordinates = torch.bmm(
        signal.u.unsqueeze(0), views["a_h"].T.unsqueeze(0)
    ).squeeze(0)
    markov_coordinates = torch.bmm(
        signal.m_prev.unsqueeze(0), views["a_m"].T.unsqueeze(0)
    ).squeeze(0)
    q_native = (
        signal.base_proposal_logits + hidden_coordinates @ basis_native.T
    )
    q_native = q_native + markov_coordinates @ basis_native.T
    confidence_features = torch.cat(
        [
            signal.u,
            rmsnorm(signal.m_prev),
            torch.ones(
                signal.u.shape[0], 1, dtype=torch.bfloat16
            ),
        ],
        dim=-1,
    )
    expected_loss = common_loss(
        signal.target_logits,
        q_native.float(),
        signal.base_confidence_logits
        + (
            views["a_c"][: signal.u.shape[0]] * confidence_features
        ).sum(-1).float(),
        signal.confidence_targets,
        signal.valid_mask,
        confidence_loss_weight=0.7,
        source_proposal_logits=signal.source_proposal_logits,
    )
    expected_loss.total.backward()

    assert actual_grad is not None
    assert torch.equal(actual_loss.total, expected_loss.total.detach())
    assert oracle_master.grad is not None
    assert torch.equal(actual_grad, oracle_master.grad)


def test_deterministic_proposal_survival_uses_one_hot_q(shapes, basis):
    signal = make_signal(seed=41, k=2)
    score = torch.zeros_like(signal.base_proposal_logits)
    score[0, 3] = 2.0
    score[1, 7] = 2.0
    probabilities = torch.full_like(score, 1e-4)
    probabilities[0, 3] = 0.7
    probabilities[1, 7] = 0.4
    probabilities /= probabilities.sum(dim=-1, keepdim=True)
    signal = replace(
        signal,
        base_proposal_logits=score,
        source_proposal_logits=score,
        target_logits=probabilities.log(),
        confidence_targets=None,
        proposal_distribution_kind="deterministic_argmax",
    )

    actual = survival_weighted_acceptance(
        torch.zeros(shapes.num_params()), signal, shapes, basis
    )
    c0 = probabilities[0, 3]
    c1 = probabilities[1, 7]
    expected = c0 + c0 * c1

    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)


def test_lazy_confidence_target_matches_precomputed_signal(shapes, basis):
    signal = make_signal(seed=4)
    phi = torch.randn(shapes.num_params()) * 0.01
    eager_loss, eager_grad = evaluate_loss_and_grad(phi, signal, shapes, basis)
    lazy_loss, lazy_grad = evaluate_loss_and_grad(
        phi,
        replace(signal, confidence_targets=None),
        shapes,
        basis,
    )
    assert torch.allclose(lazy_loss.total, eager_loss.total, atol=1e-7, rtol=0)
    assert torch.allclose(lazy_grad, eager_grad, atol=1e-7, rtol=0)


def test_grad_clip_scale_applied(shapes, basis):
    from lightcone_spec.adapters.adapter_params import clip_gradient_global_norm

    g = torch.ones(10) * 10.0
    clipped, scale = clip_gradient_global_norm(g, 1.0)
    assert abs(float(torch.linalg.vector_norm(clipped)) - 1.0) < 1e-6
    assert scale < 1.0


def test_temperature_scaled_proposal_reconstruction_matches_online(shapes, basis):
    """A_d/A_m are applied before temperature online; training must rebuild
    the same effective q logits that sampling and rejection use."""
    signal = make_signal(seed=11)
    k = signal.u.shape[0]
    signal.proposal_logit_scale = torch.linspace(0.5, 2.0, k)
    phi = torch.randn(shapes.num_params()) * 0.01
    adapter = AdapterParams(shapes, basis)
    adapter.load_flat(phi)
    online_q = (
        signal.base_proposal_logits
        + adapter.draft_logit_residual(signal.u)
        + adapter.markov_logit_residual(signal.m_prev)
    ) * signal.proposal_logit_scale[:, None]
    online_conf = signal.base_confidence_logits + adapter.confidence_residual(
        signal.u, signal.m_prev
    )
    expected = common_loss(
        signal.target_logits * signal.proposal_logit_scale[:, None],
        online_q,
        online_conf,
        signal.confidence_targets,
        signal.valid_mask,
    )
    rebuilt, _ = evaluate_loss_and_grad(
        phi, signal, shapes, basis, need_grad=False
    )
    assert torch.allclose(rebuilt.total, expected.total, atol=1e-7, rtol=0)


def test_survival_weighted_acceptance_uses_prefix_mask(shapes, basis):
    signal = make_signal(seed=12)
    mask = torch.zeros_like(signal.valid_mask)
    mask[:2] = True
    perfect = replace(
        signal,
        base_proposal_logits=signal.target_logits.clone(),
        source_proposal_logits=signal.target_logits.clone(),
        valid_mask=mask,
        proposal_logit_scale=None,
    )
    phi = torch.zeros(shapes.num_params())
    stochastic = survival_weighted_acceptance(phi, perfect, shapes, basis)
    greedy = survival_weighted_acceptance(
        phi, perfect, shapes, basis, greedy=True
    )
    assert stochastic == 2.0
    assert greedy == 2.0


def test_l3_preview_uses_fixed_moments_and_advances_shared_state_once(shapes, basis):
    gen = CommonCandidateGenerator(
        shapes,
        basis,
        CandidateGeneratorConfig(
            lr=1e-3,
            grad_clip=1.0,
            trust_region_radius=1.0,
            confidence_loss_weight=1.0,
            lambda_prox=0.0,
        ),
    )
    preview_avg = torch.zeros(shapes.num_params())
    preview_sq = torch.zeros(shapes.num_params())
    gen.bind_preview_state(preview_avg, preview_sq)
    ptrs = (preview_avg.data_ptr(), preview_sq.data_ptr())
    gen.prepare_preview_state()
    cand = gen.candidate(
        torch.zeros(shapes.num_params()), make_signal(seed=21), defer_state_advance=True
    )
    assert gen.state.step == 0
    assert gen.preview_state.step == 1
    assert (gen.preview_state.exp_avg.data_ptr(), gen.preview_state.exp_avg_sq.data_ptr()) == ptrs
    gen.delta_from_transported_gradient(cand.raw_gradient)
    assert gen.state.step == 1
