from __future__ import annotations

import pytest
import torch

from lightcone_spec.adaptation.memory import AdaptationMemoryLedger, tensor_bytes
from lightcone_spec.adaptation.optimizer import FixedAddressBank, GPUOptimizer
from lightcone_spec.adaptation.parameters import DFlashParameterPlan, LoRAFactors
from lightcone_spec.config.schema import OptimizerConfig
from lightcone_spec.runtime.dflash_canvas import (
    CanvasReconstruction,
    DifferentiableCanvasContract,
    position_weighted_kl,
    rms_norm,
    scaled_dot_product_canvas,
)
from lightcone_spec.runtime.exactness import greedy_exact, rejection_sample


def optimizer_config(name: str, *, weight_decay: float = 0.0) -> OptimizerConfig:
    return OptimizerConfig(
        name=name,
        learning_rate=0.05,
        weight_decay=weight_decay,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        grad_clip=1.0,
    )


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
    plan = DFlashParameterPlan.build(
        named_parameters(), mode="full", scope="drafter"
    )
    names = {entry.name for entry in plan.entries}
    assert "fc.weight" in names
    assert "layers.0.input_layernorm.weight" in names
    assert "norm.weight" in names
    assert "lm_head.weight" not in names
    assert not any(name.startswith("target_model") for name in names)


def test_parameter_ownership_uses_exact_dotted_components() -> None:
    parameters = {
        "draft_lm_head_adapter.weight": torch.zeros(2, 2),
        "layers.target_model_adapter.weight": torch.zeros(2, 2),
        "target_model.weight": torch.zeros(2, 2),
    }
    plan = DFlashParameterPlan.build(
        parameters, mode="full", scope="drafter"
    )
    assert {entry.name for entry in plan.entries} == {
        "draft_lm_head_adapter.weight",
        "layers.target_model_adapter.weight",
    }


def test_lora_selects_actual_dflash_linear_matrices() -> None:
    plan = DFlashParameterPlan.build(
        named_parameters(), mode="lora", scope="drafter", rank=2
    )
    names = {entry.name for entry in plan.entries}
    assert names == {
        "fc.weight",
        "layers.0.self_attn.qkv_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate_up_proj.weight",
        "layers.0.mlp.down_proj.weight",
    }
    assert plan.trainable_parameter_count == sum(
        2 * (entry.shape[0] + entry.shape[1]) for entry in plan.entries
    )


def test_tail_selection_requires_explicit_allowlist() -> None:
    with pytest.raises(ValueError, match="allowlist"):
        DFlashParameterPlan.build(named_parameters(), mode="full", scope="tail")
    plan = DFlashParameterPlan.build(
        named_parameters(),
        mode="full",
        scope="tail",
        tail_names=("norm.weight",),
    )
    assert [entry.name for entry in plan.entries] == ["norm.weight"]


def test_lora_zero_initialization_preserves_base_and_is_seeded() -> None:
    base = torch.randn(8, 4)
    first = LoRAFactors.initialize(base, 2, seed=7)
    second = LoRAFactors.initialize(base, 2, seed=7)
    torch.testing.assert_close(first.a, second.a)
    torch.testing.assert_close(first.b, torch.zeros_like(first.b))
    torch.testing.assert_close(first.merged(base), base.float())
    first.b[0, 0] = 1.0
    assert not torch.equal(first.merged(base), base.float())


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
