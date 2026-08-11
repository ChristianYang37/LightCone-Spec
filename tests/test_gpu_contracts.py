from __future__ import annotations

import pytest
import torch

from lightcone_spec.adaptation.optimizer import FixedAddressBank
from lightcone_spec.runtime.dflash_canvas import position_weighted_kl
from lightcone_spec.runtime.exactness import rejection_sample
from lightcone_spec.runtime.publication import CudaPublicationCoordinator

pytestmark = pytest.mark.gpu


def require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")


def test_cuda_graph_replay_observes_fixed_address_publication_without_recapture() -> (
    None
):
    require_cuda()
    active = torch.zeros(32, device="cuda", dtype=torch.float16)
    static_input = torch.arange(32, device="cuda", dtype=torch.float16)
    candidate = torch.ones_like(active)
    captured_output = torch.empty_like(active)
    bank = FixedAddressBank((active,))
    coordinator = CudaPublicationCoordinator("cuda")
    assert not coordinator.ready()
    with pytest.raises(RuntimeError, match="no completed"):
        coordinator.publish_boundary(publish=bank.publish, tensors=(active,))
    parameter_address = active.data_ptr()
    output_address = captured_output.data_ptr()

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        captured_output.copy_(static_input * active + 1)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output.copy_(static_input * active + 1)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured_output, torch.ones_like(captured_output))

    with coordinator.update_window((active,)):
        with (
            pytest.raises(RuntimeError, match="max_in_flight"),
            coordinator.update_window((active,)),
        ):
            pass
        bank.stage((candidate,))
    coordinator.publish_boundary(publish=bank.publish, tensors=(active,))
    graph.replay()
    torch.cuda.synchronize()
    assert active.data_ptr() == parameter_address
    assert captured_output.data_ptr() == output_address
    torch.testing.assert_close(active, candidate)
    torch.testing.assert_close(captured_output, static_input + 1)


def test_differentiable_logits_have_finite_gpu_gradient() -> None:
    require_cuda()
    target = torch.randn(4, 15, 257, device="cuda")
    draft = torch.randn(4, 15, 257, device="cuda", requires_grad=True)
    mask = torch.ones(4, 15, device="cuda", dtype=torch.bool)
    loss = position_weighted_kl(target, draft, mask, decay=0.95)
    loss.backward()
    assert draft.grad is not None
    assert bool(torch.isfinite(draft.grad).all())


def test_stochastic_rejection_distribution_matches_target() -> None:
    require_cuda()
    count = 100_000
    target = torch.tensor([0.2, 0.3, 0.5], device="cuda").expand(count, -1)
    proposal = torch.tensor([0.6, 0.2, 0.2], device="cuda").expand(count, -1)
    generator = torch.Generator(device="cuda").manual_seed(7)
    proposal_tokens = torch.multinomial(proposal, 1, generator=generator).squeeze(-1)
    uniforms = torch.rand(count, device="cuda", generator=generator)
    sampled, _ = rejection_sample(
        target,
        proposal,
        proposal_tokens,
        uniforms,
        generator=generator,
    )
    observed = torch.bincount(sampled, minlength=3).float() / count
    torch.testing.assert_close(observed, target[0], atol=0.01, rtol=0.0)


def test_high_batch_fixed_bank_allocator_stability() -> None:
    require_cuda()
    active = torch.zeros((48, 16, 256), device="cuda", dtype=torch.float16)
    candidate = torch.empty_like(active)
    bank = FixedAddressBank((active,))
    address = active.data_ptr()
    bank.stage((candidate,))
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    for value in range(20):
        candidate.fill_(float(value))
        bank.stage((candidate,))
        bank.publish()
    torch.cuda.synchronize()
    assert active.data_ptr() == address
    assert torch.cuda.memory_allocated() == allocated
    assert torch.cuda.memory_reserved() == reserved
    torch.testing.assert_close(active, candidate)
