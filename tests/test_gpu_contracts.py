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


def test_cuda_publication_preserves_addresses_without_recapture() -> None:
    require_cuda()
    active = torch.zeros(32, device="cuda", dtype=torch.float16)
    bank = FixedAddressBank((active,))
    coordinator = CudaPublicationCoordinator("cuda")
    assert not coordinator.ready()
    with pytest.raises(RuntimeError, match="no completed"):
        coordinator.publish_boundary(publish=bank.publish, tensors=(active,))
    address = active.data_ptr()
    with coordinator.update_window((active,)):
        with (
            pytest.raises(RuntimeError, match="max_in_flight"),
            coordinator.update_window((active,)),
        ):
            pass
        bank.stage((torch.ones_like(active),))
    coordinator.publish_boundary(publish=bank.publish, tensors=(active,))
    torch.cuda.synchronize()
    assert active.data_ptr() == address
    torch.testing.assert_close(active, torch.ones_like(active))


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
    torch.testing.assert_close(
        observed, target[0], atol=0.01, rtol=0.0
    )


def test_high_batch_fixed_bank_allocator_stability() -> None:
    require_cuda()
    active = torch.zeros((48, 16, 256), device="cuda", dtype=torch.float16)
    bank = FixedAddressBank((active,))
    address = active.data_ptr()
    for value in range(20):
        bank.stage((torch.full_like(active, float(value)),))
        bank.publish()
    assert active.data_ptr() == address
