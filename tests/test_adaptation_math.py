from __future__ import annotations

import pytest
import torch

from lightcone_spec.adaptation.memory import AdaptationMemoryLedger, tensor_bytes
from lightcone_spec.adaptation.optimizer import FixedAddressBank, GPUOptimizer
from lightcone_spec.adaptation.parameters import (
    TRAINABLE_PLAN_OPTIMIZERS,
    DFlashParameterPlan,
    DSparkParameterPlan,
    LoRAFactors,
    NativeLayerParameterPlan,
)
from lightcone_spec.config.schema import OptimizerConfig
from lightcone_spec.experiments.formal_protocol import (
    ChronoBeliefState,
    chronobelief_reference_transition,
)
from lightcone_spec.runtime.dflash_canvas import (
    CanvasReconstruction,
    DifferentiableCanvasContract,
    position_weighted_kl,
    rms_norm,
    scaled_dot_product_canvas,
)
from lightcone_spec.runtime.exactness import greedy_exact, rejection_sample


def optimizer_config(name: str, *, weight_decay: float = 0.0) -> OptimizerConfig:
    values = {
        "name": name,
        "learning_rate": 0.05,
        "weight_decay": weight_decay,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "grad_clip": 1.0,
    }
    if name in {"sgdm", "nag", "muon"}:
        values["momentum"] = 0.9
    if name == "muon":
        values.update(
            muon_ns_steps=5,
            muon_auxiliary_learning_rate=0.005,
            muon_auxiliary_weight_decay=0.01,
        )
    return OptimizerConfig(**values)


@pytest.mark.parametrize(
    ("name", "weight_decay", "reference"),
    [
        ("adam", 0.0, torch.optim.Adam),
        ("adamw", 0.01, torch.optim.AdamW),
    ],
)
def test_optimizer_matches_torch_one_step(name, weight_decay, reference) -> None:
    initial = torch.tensor([1.0, -2.0])
    gradient = torch.tensor([0.25, -0.5])
    ours = GPUOptimizer((initial,), optimizer_config(name, weight_decay=weight_decay))
    proposal = ours.propose((gradient,))

    parameter = torch.nn.Parameter(initial.clone())
    baseline = reference(
        (parameter,),
        lr=0.05,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    parameter.grad = gradient.clone()
    baseline.step()
    torch.testing.assert_close(proposal.parameters[0], parameter.detach())
    torch.testing.assert_close(ours.master[0], initial)
    ours.commit(proposal)
    torch.testing.assert_close(ours.master[0], parameter.detach())


@pytest.mark.parametrize(("name", "nesterov"), [("sgdm", False), ("nag", True)])
def test_momentum_optimizer_matches_torch(name: str, nesterov: bool) -> None:
    initial = torch.tensor([1.0, -2.0])
    gradients = (
        torch.tensor([0.25, -0.5]),
        torch.tensor([-0.3, 0.2]),
    )
    ours = GPUOptimizer((initial,), optimizer_config(name, weight_decay=0.01))
    parameter = torch.nn.Parameter(initial.clone())
    baseline = torch.optim.SGD(
        (parameter,),
        lr=0.05,
        momentum=0.9,
        nesterov=nesterov,
        weight_decay=0.01,
    )
    for gradient in gradients:
        proposal = ours.propose((gradient,))
        ours.commit(proposal)
        parameter.grad = gradient.clone()
        baseline.step()
        torch.testing.assert_close(ours.master[0], parameter.detach())


def test_lion_matches_reference_and_uses_one_state_tensor() -> None:
    initial = torch.tensor([1.0, -2.0])
    gradient = torch.tensor([0.25, -0.5])
    config = optimizer_config("lion", weight_decay=0.01)
    ours = GPUOptimizer((initial,), config)
    proposal = ours.propose((gradient,))
    direction = ((1.0 - config.beta1) * gradient).sign()
    expected = initial * (1.0 - 0.05 * 0.01) - 0.05 * direction
    torch.testing.assert_close(proposal.parameters[0], expected)
    torch.testing.assert_close(
        proposal.first_moments[0], (1.0 - config.beta2) * gradient
    )
    assert proposal.second_moments[0].numel() == 0


def test_muon_matches_reference_and_uses_adamw_for_non_matrices() -> None:
    matrix = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    vector = torch.tensor([1.0, -1.0])
    matrix_gradient = torch.tensor([[0.25, -0.5], [0.1, 0.2]])
    vector_gradient = torch.tensor([0.3, -0.4])
    config = optimizer_config("muon", weight_decay=0.01)
    ours = GPUOptimizer((matrix, vector), config)
    proposal = ours.propose((matrix_gradient, vector_gradient))

    momentum_buffer = (1.0 - 0.9) * matrix_gradient
    nesterov = (1.0 - 0.9) * matrix_gradient + 0.9 * momentum_buffer
    from lightcone_spec.adaptation.optimizer import zeroth_power_newton_schulz

    direction = zeroth_power_newton_schulz(nesterov, steps=5, epsilon=1e-7)
    expected_matrix = matrix * (1.0 - 0.05 * 0.01) - 0.05 * direction
    torch.testing.assert_close(proposal.parameters[0], expected_matrix)

    auxiliary = torch.nn.Parameter(vector.clone())
    baseline = torch.optim.AdamW(
        (auxiliary,),
        lr=0.005,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    auxiliary.grad = vector_gradient.clone()
    baseline.step()
    torch.testing.assert_close(proposal.parameters[1], auxiliary.detach())
    assert proposal.second_moments[0].numel() == 0
    assert proposal.second_moments[1].numel() == vector.numel()


def test_seven_registered_optimizer_reference_paths_are_implemented() -> None:
    assert TRAINABLE_PLAN_OPTIMIZERS == (
        "adam",
        "adamw",
        "chronobelief",
        "lion",
        "muon",
        "nag",
        "sgdm",
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_chronobelief_matches_cpu_reference_and_mixed_precision(dtype) -> None:
    initial = torch.tensor([0.75, -1.25], dtype=dtype)
    gradient = torch.tensor([0.2, -0.4], dtype=dtype)
    config = optimizer_config("chronobelief", weight_decay=0.01)
    optimizer = GPUOptimizer((initial,), config, initial_safe_boundary_version=3)
    proposal = optimizer.propose((gradient,), source_version=0, safe_boundary_version=3)
    reference = chronobelief_reference_transition(
        ChronoBeliefState(
            tuple(float(value) for value in initial.float()),
            (0.0, 0.0),
            (0.0, 0.0),
            0,
        ),
        tuple(float(value) for value in gradient.float()),
        safe_boundary_age=3,
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        epsilon=config.epsilon,
        weight_decay=config.weight_decay,
    )
    torch.testing.assert_close(
        proposal.parameters[0],
        torch.tensor(reference.parameters),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        proposal.first_moments[0],
        torch.tensor(reference.first_moments),
    )
    torch.testing.assert_close(
        proposal.second_moments[0],
        torch.tensor(reference.second_moments),
        rtol=2e-5,
        atol=2e-5,
    )
    assert proposal.safe_boundary_age == 3


def test_chronobelief_age_is_exact_and_abort_does_not_advance_state() -> None:
    optimizer = GPUOptimizer(
        (torch.tensor([1.0, -2.0]),),
        optimizer_config("chronobelief", weight_decay=0.01),
        initial_safe_boundary_version=5,
    )
    with pytest.raises(ValueError, match="source versions"):
        optimizer.propose((torch.tensor([0.25, -0.5]),))
    with pytest.raises(ValueError, match="derived from source versions"):
        optimizer.propose(
            (torch.tensor([0.25, -0.5]),),
            safe_boundary_age=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="current safe boundary"):
        optimizer.propose(
            (torch.tensor([0.25, -0.5]),),
            source_version=0,
            safe_boundary_version=4,
        )

    first = optimizer.propose(
        (torch.tensor([0.25, -0.5]),),
        source_version=5,
        safe_boundary_version=5,
    )
    stale = optimizer.propose(
        (torch.tensor([0.25, -0.5]),),
        source_version=0,
        safe_boundary_version=5,
    )
    assert first.step == stale.step == 1
    assert optimizer.step_number == 0
    assert all(bool((moment == 0).all()) for moment in optimizer.first_moments)
    assert all(bool((moment == 0).all()) for moment in optimizer.second_moments)
    assert not torch.equal(first.parameters[0], stale.parameters[0])

    optimizer.commit(stale)
    assert optimizer.step_number == 1
    assert optimizer.safe_boundary_version == 6
    torch.testing.assert_close(optimizer.master[0], stale.parameters[0])
    torch.testing.assert_close(optimizer.first_moments[0], stale.first_moments[0])
    torch.testing.assert_close(optimizer.second_moments[0], stale.second_moments[0])
    with pytest.raises(ValueError, match="step conflict"):
        optimizer.commit(first)

    other = GPUOptimizer(
        (torch.tensor([1.0, -2.0]),),
        optimizer_config("chronobelief", weight_decay=0.01),
        initial_safe_boundary_version=5,
    )
    foreign = other.propose(
        (torch.tensor([0.25, -0.5]),),
        source_version=5,
        safe_boundary_version=5,
    )
    with pytest.raises(ValueError, match="another state owner"):
        optimizer.commit(foreign)


def test_chronobelief_large_age_is_safe_and_nonfinite_candidates_are_rejected() -> None:
    config = OptimizerConfig(
        name="chronobelief",
        learning_rate=0.01,
        weight_decay=0.0,
        beta1=0.99,
        beta2=0.01,
        epsilon=1e-8,
        grad_clip=1.0,
    )
    optimizer = GPUOptimizer(
        (torch.tensor([1.0]),),
        config,
        initial_safe_boundary_version=10**9,
    )
    proposal = optimizer.propose(
        (torch.tensor([0.25]),),
        source_version=0,
        safe_boundary_version=10**9,
    )
    assert bool(torch.isfinite(proposal.parameters[0]).all())
    assert bool(torch.isfinite(proposal.first_moments[0]).all())
    assert bool(torch.isfinite(proposal.second_moments[0]).all())

    overflow = GPUOptimizer(
        (torch.tensor([3e38]),),
        OptimizerConfig(
            name="chronobelief",
            learning_rate=1e38,
            weight_decay=1e38,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            grad_clip=1.0,
        ),
    )
    with pytest.raises(ValueError, match="proposal is non-finite"):
        overflow.propose(
            (torch.tensor([1.0]),),
            source_version=0,
            safe_boundary_version=0,
        )


def test_tts_none_gradient_clipping_is_exact_and_l0_candidate_bytes_match() -> None:
    config = OptimizerConfig(
        name="adam",
        learning_rate=0.1,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        grad_clip=None,
    )
    assert OptimizerConfig.model_validate(config.model_dump()).grad_clip is None
    initial = torch.tensor([1.0, -2.0])
    gradient = torch.tensor([100.0, -200.0])
    tts = GPUOptimizer((initial,), config).propose((gradient,))
    l0_naive = GPUOptimizer((initial,), config).propose((gradient,))
    expected_first = 0.1 * gradient
    expected_second = 0.001 * gradient.square()
    expected_direction = (expected_first / 0.1) / (
        (expected_second / 0.001).sqrt() + 1e-8
    )
    torch.testing.assert_close(tts.parameters[0], initial - 0.1 * expected_direction)
    assert torch.equal(tts.parameters[0], l0_naive.parameters[0])
    assert torch.equal(tts.first_moments[0], l0_naive.first_moments[0])
    assert torch.equal(tts.second_moments[0], l0_naive.second_moments[0])

    clipped = GPUOptimizer(
        (initial,),
        OptimizerConfig(**{**config.model_dump(), "grad_clip": 1.0}),
    ).propose((gradient,))
    assert not torch.equal(tts.first_moments[0], clipped.first_moments[0])


def test_optimizer_rejects_layout_and_step_conflicts() -> None:
    optimizer = GPUOptimizer((torch.ones(2),), optimizer_config("adam"))
    with pytest.raises(ValueError, match="one gradient"):
        optimizer.propose(())
    with pytest.raises(ValueError, match="layout"):
        optimizer.propose((torch.ones(3),))
    proposal = optimizer.propose((torch.ones(2),))
    optimizer.commit(proposal)
    with pytest.raises(ValueError, match="step conflict"):
        optimizer.commit(proposal)
    with pytest.raises(ValueError, match="finite"):
        GPUOptimizer((torch.tensor([float("nan")]),), optimizer_config("adam"))
    finite = GPUOptimizer((torch.ones(2),), optimizer_config("adam"))
    with pytest.raises(ValueError, match="finite"):
        finite.propose((torch.tensor([1.0, float("inf")]),))


def test_schedules_advance_only_on_published_updates() -> None:
    inverse = GPUOptimizer(
        (torch.ones(1),),
        OptimizerConfig(
            name="sgd",
            learning_rate=0.1,
            schedule="inverse_sqrt_published_update",
        ),
    )
    first = inverse.propose((torch.ones(1),))
    repeated = inverse.propose((torch.ones(1),))
    assert first.step == repeated.step == 1
    torch.testing.assert_close(first.parameters[0], repeated.parameters[0])
    inverse.commit(first)
    second = inverse.propose((torch.ones(1),))
    torch.testing.assert_close(
        second.parameters[0],
        torch.tensor([0.9 - 0.1 / (2**0.5)]),
    )

    cosine = GPUOptimizer(
        (torch.ones(1),),
        OptimizerConfig(
            name="sgd",
            learning_rate=0.1,
            schedule="cosine_to_zero",
            schedule_total_published_updates=2,
        ),
    )
    cosine.commit(cosine.propose((torch.ones(1),)))
    final = cosine.propose((torch.ones(1),))
    torch.testing.assert_close(final.parameters[0], torch.tensor([0.9]))


def test_cosine_schedule_requires_exact_horizon() -> None:
    with pytest.raises(ValueError, match="published-update horizon"):
        OptimizerConfig(
            name="sgd",
            learning_rate=0.1,
            schedule="cosine_to_zero",
        )
    with pytest.raises(ValueError, match="published-update horizon"):
        OptimizerConfig(
            name="sgd",
            learning_rate=0.1,
            schedule_total_published_updates=2,
        )


def test_fixed_bank_preserves_storage_and_validates_layout() -> None:
    active = torch.zeros(4, dtype=torch.float16)
    bank = FixedAddressBank((active,))
    address = active.data_ptr()
    bank.stage((torch.ones(4, dtype=torch.float32),))
    bank.publish()
    assert active.data_ptr() == address
    torch.testing.assert_close(active, torch.ones_like(active))
    with pytest.raises(ValueError, match="shape or device"):
        bank.stage((torch.ones(5),))


def named_parameters() -> dict[str, torch.Tensor]:
    return {
        "fc.weight": torch.zeros(8, 4),
        "layers.0.self_attn.qkv_proj.weight": torch.zeros(12, 4),
        "layers.0.self_attn.o_proj.weight": torch.zeros(4, 4),
        "layers.0.mlp.gate_up_proj.weight": torch.zeros(16, 4),
        "layers.0.mlp.down_proj.weight": torch.zeros(4, 8),
        "layers.0.input_layernorm.weight": torch.ones(4),
        "norm.weight": torch.ones(4),
        "lm_head.weight": torch.zeros(32, 4),
        "target_model.layers.0.weight": torch.zeros(4, 4),
    }


def test_full_selects_all_and_only_drafter_owned_float_parameters() -> None:
    plan = DFlashParameterPlan.build(named_parameters(), mode="full", scope="all")
    names = {entry.name for entry in plan.entries}
    assert "layers.0.input_layernorm.weight" in names
    assert "fc.weight" not in names
    assert "norm.weight" not in names
    assert "lm_head.weight" not in names
    assert not any(name.startswith("target_model") for name in names)


def test_parameter_ownership_uses_exact_dotted_components() -> None:
    parameters = {
        "layers.0.draft_lm_head_adapter.weight": torch.zeros(2, 2),
        "layers.0.target_model_adapter.weight": torch.zeros(2, 2),
        "layers.0.target_model.weight": torch.zeros(2, 2),
    }
    plan = DFlashParameterPlan.build(parameters, mode="full", scope="all")
    assert {entry.name for entry in plan.entries} == {
        "layers.0.draft_lm_head_adapter.weight",
        "layers.0.target_model_adapter.weight",
    }


def test_lora_selects_actual_dflash_linear_matrices() -> None:
    plan = DFlashParameterPlan.build(
        named_parameters(), mode="lora", scope="all", rank=2
    )
    names = {entry.name for entry in plan.entries}
    assert names == {
        "layers.0.self_attn.qkv_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate_up_proj.weight",
        "layers.0.mlp.down_proj.weight",
    }
    assert plan.trainable_parameter_count == sum(
        2 * (entry.shape[0] + entry.shape[1]) for entry in plan.entries
    )


def test_dspark_native_hybrid_uses_lora_layers_and_full_replicated_heads() -> None:
    parameters: dict[str, torch.Tensor] = {}
    for layer in range(5):
        parameters[f"layers.{layer}.self_attn.q_proj.weight"] = torch.zeros(4, 4)
        parameters[f"layers.{layer}.norm.weight"] = torch.ones(4)
    parameters.update(
        {
            "markov.w1.weight": torch.zeros(4, 4),
            "markov.w2.weight": torch.zeros(4, 4),
            "acceptance.projection": torch.zeros(1),
        }
    )
    plan = DSparkParameterPlan.build(
        parameters,
        mode="lora",
        scope="last3_native_heads",
        rank=4,
        w1_name="markov.w1.weight",
        w2_name="markov.w2.weight",
        acceptance_name="acceptance.projection",
    )
    entries = {entry.name: entry for entry in plan.entries}
    assert entries["layers.2.self_attn.q_proj.weight"].parameterization == "lora"
    assert "layers.1.self_attn.q_proj.weight" not in entries
    for name in ("markov.w1.weight", "markov.w2.weight", "acceptance.projection"):
        assert entries[name].parameterization == "full"
        assert entries[name].ownership == "replicated"
    assert len(plan.sha256) == 64
    assert plan.predict_memory("adamw").peak_bytes > 0


@pytest.mark.parametrize("backend", ["EAGLE", "EAGLE3", "NEXTN"])
def test_other_native_backends_have_digest_bound_layer_plans(backend: str) -> None:
    plan = NativeLayerParameterPlan.build(
        named_parameters(),
        backend=backend,
        mode="lora",
        scope="last1",
        rank=2,
    )
    assert plan.backend == backend
    assert plan.rank == plan.lora_alpha == 2
    assert {entry.parameterization for entry in plan.entries} == {"lora"}
    assert len(plan.sha256) == 64


def test_lora_zero_initialization_preserves_base_and_is_seeded() -> None:
    base = torch.randn(8, 4)
    first = LoRAFactors.initialize(base, 2, seed=7)
    second = LoRAFactors.initialize(base, 2, seed=7)
    torch.testing.assert_close(first.a, second.a)
    torch.testing.assert_close(first.b, torch.zeros_like(first.b))
    torch.testing.assert_close(first.merged(base), base.float())
    first.b[0, 0] = 1.0
    assert not torch.equal(first.merged(base), base.float())


def test_lora_memory_charges_full_fixed_banks_and_merge_temporaries() -> None:
    plan = DFlashParameterPlan.build(
        {"layers.0.self_attn.q_proj.weight": torch.zeros(4, 4)},
        mode="lora",
        scope="last1",
        rank=2,
    )
    prediction = plan.predict_memory("adamw")
    assert prediction.active_merged == 4 * 4 * 4
    assert prediction.staging == prediction.active_merged
    assert prediction.merge_scratch == 3 * 4 * 4 * 4
    assert prediction.peak_bytes > prediction.resident_bytes


def test_chronobelief_memory_reserves_both_fp32_moment_vectors() -> None:
    plan = DFlashParameterPlan.build(
        {"layers.0.self_attn.q_proj.weight": torch.zeros(4, 4)},
        mode="full",
        scope="last1",
    )
    prediction = plan.predict_memory("chronobelief")
    assert prediction.optimizer_first == 4 * plan.trainable_parameter_count
    assert prediction.optimizer_second == 4 * plan.trainable_parameter_count
    assert prediction == plan.predict_memory("adamw")


def test_memory_ledger_reserves_adaptation_before_kv() -> None:
    ledger = AdaptationMemoryLedger(
        active_base=10,
        master_fp32=20,
        gradients=20,
        first_moments=20,
        second_moments=20,
        staging=10,
        training_activations=30,
        kv_gather_scratch=40,
        candidate_scratch=15,
        graph_buffers=5,
        telemetry=5,
    )
    assert ledger.resident_bytes == 90
    assert ledger.peak_bytes == 195
    assert ledger.kv_budget(300, reserve_bytes=25) == 80
    with pytest.raises(MemoryError, match="no silent offload"):
        ledger.kv_budget(100)
    assert tensor_bytes((torch.zeros(4), torch.zeros(2, dtype=torch.float16))) == 20


def test_memory_ledger_rejects_negative_categories() -> None:
    with pytest.raises(ValueError):
        AdaptationMemoryLedger(staging=-1)


def test_rms_norm_matches_reference() -> None:
    hidden = torch.tensor([[1.0, 2.0, 3.0]])
    weight = torch.tensor([1.0, 0.5, 2.0])
    expected = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)
    expected = expected * weight
    torch.testing.assert_close(rms_norm(hidden, weight, 1e-6), expected)


def test_canvas_attention_detaches_history_but_not_current() -> None:
    query = torch.randn(1, 1, 1, 4, requires_grad=True)
    historical_key = torch.randn(1, 1, 2, 4, requires_grad=True)
    historical_value = torch.randn(1, 1, 2, 4, requires_grad=True)
    current_key = torch.randn(1, 1, 1, 4, requires_grad=True)
    current_value = torch.randn(1, 1, 1, 4, requires_grad=True)
    result = scaled_dot_product_canvas(
        query,
        historical_key,
        historical_value,
        current_key,
        current_value,
    )
    result.sum().backward()
    assert historical_key.grad is None
    assert historical_value.grad is None
    assert current_key.grad is not None
    assert current_value.grad is not None


def test_position_weighted_kl_zero_and_gradient() -> None:
    target = torch.randn(2, 3, 7)
    identical = target.clone().requires_grad_(True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    zero = position_weighted_kl(target, identical, mask, decay=0.9)
    assert abs(float(zero.detach())) < 1e-6
    draft = torch.randn(2, 3, 7, requires_grad=True)
    loss = position_weighted_kl(target, draft, mask, decay=0.9)
    loss.backward()
    assert float(loss.detach()) >= 0
    assert draft.grad is not None
    assert torch.count_nonzero(draft.grad[0, 2]) == 0


def test_differentiable_canvas_contract_accepts_and_rejects_reconstruction() -> None:
    raw = torch.randn(1, 2, 3)

    def differentiable(offset: float = 0.0):
        history = torch.randn(1, 1).detach()
        current_key = torch.randn(1, 1, requires_grad=True)
        current_value = torch.randn(1, 1, requires_grad=True)
        return CanvasReconstruction(
            raw_logits=raw,
            differentiable_logits=raw.clone().requires_grad_(True) + offset,
            historical_key=history,
            historical_value=history.clone(),
            current_key=current_key,
            current_value=current_value,
        )

    contract = DifferentiableCanvasContract(lambda: raw, lambda: differentiable())
    contract.reconstruct()
    failing = DifferentiableCanvasContract(
        lambda: raw, lambda: differentiable(1.0), atol=0.0, rtol=0.0
    )
    with pytest.raises(RuntimeError, match="does not reconstruct"):
        failing.reconstruct()


def test_exact_rejection_accept_and_residual_paths() -> None:
    target = torch.tensor([[0.7, 0.3], [0.1, 0.9]])
    proposal = torch.tensor([[0.5, 0.5], [0.8, 0.2]])
    tokens = torch.tensor([0, 0])
    uniforms = torch.tensor([0.1, 0.9])
    generator = torch.Generator().manual_seed(0)
    sampled, accepted = rejection_sample(
        target, proposal, tokens, uniforms, generator=generator
    )
    assert accepted.tolist() == [True, False]
    assert sampled[0].item() == 0
    assert sampled[1].item() == 1
    assert greedy_exact(torch.tensor([[1.0, 2.0]])).item() == 1


@pytest.mark.parametrize(
    ("target", "proposal", "tokens", "uniforms", "message"),
    [
        ([0.8, 0.3], [0.5, 0.5], [0], [0.1], "sum to one"),
        ([0.7, 0.3], [1.1, -0.1], [0], [0.1], "negative"),
        ([0.7, 0.3], [0.0, 1.0], [0], [0.1], "positive mass"),
        ([0.7, 0.3], [0.5, 0.5], [2], [0.1], "vocabulary"),
        ([0.7, 0.3], [0.5, 0.5], [0], [1.0], r"\[0, 1\)"),
    ],
)
def test_exact_rejection_fails_closed_on_invalid_inputs(
    target, proposal, tokens, uniforms, message
) -> None:
    with pytest.raises(ValueError, match=message):
        rejection_sample(
            torch.tensor([target]),
            torch.tensor([proposal]),
            torch.tensor(tokens),
            torch.tensor(uniforms),
        )
