"""OnlineSpec DSpark ports (spec 6.9-6.11): projected OGD, optimistic
two-step, and the three-learner Hedge ensemble.

Clean-room note (spec 18.1): implemented from the paper formulas only;
no code copied from the OnlineSPEC repository snapshot, which is used
solely for behavioral cross-checking. These are unified DSpark ports,
not official Opt-Hydra / Ens-EAGLE reproductions.
"""

from __future__ import annotations

from typing import Optional

import torch

from lightcone_spec.adapters.adapter_params import (
    AdapterShapes,
    canonicalize_master_vector,
    clip_gradient_global_norm,
    initial_parameter_vector,
    trust_region_project,
)
from lightcone_spec.methods.base import (
    ArrivalContext,
    CandidateUpdate,
    Decision,
    DecisionKind,
    MethodRuntime,
    PublishPolicy,
    TeacherSignal,
    consensus_gradient,
    evaluate_loss_and_grad,
)

HEDGE_EPSILON = 10.0
ENSEMBLE_LR_MULTIPLIERS = (1.0, 2.0, 4.0)


class _OnlineSpecBase(MethodRuntime):
    publish_policy = PublishPolicy.ASYNC_BOUNDARY

    def __init__(
        self,
        shapes: AdapterShapes,
        basis: torch.Tensor,
        lr: float,
        grad_clip: float,
        trust_region_radius: float,
        confidence_loss_weight: float,
        seed: int = 0,
    ):
        self.shapes = shapes
        self.basis = basis
        self.lr = lr
        self.grad_clip = grad_clip
        self.radius = trust_region_radius
        self.confidence_loss_weight = confidence_loss_weight
        forward_dtype = getattr(basis, "dtype", None)
        if forward_dtype is None:
            forward_dtype = getattr(getattr(basis, "weight", None), "dtype", None)
        forward_dtype = forward_dtype or torch.float32
        self.phi0 = canonicalize_master_vector(
            initial_parameter_vector(shapes), forward_dtype
        )
        self._counter = 0
        self.arrival_index = 0  # update event index t, ordered by arrival
        self.gradient_consensus_fn = None

    def bind_gradient_consensus(self, callback) -> None:
        self.gradient_consensus_fn = callback

    def _grad_at(
        self,
        phi: torch.Tensor,
        signal: TeacherSignal,
        forward_phi: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        float | torch.Tensor,
        float | torch.Tensor,
        object,
        bool | torch.Tensor,
    ]:
        if self.phi0.device != phi.device:
            self.phi0 = self.phi0.to(phi.device)
            if hasattr(self, "hat_phi"):
                self.hat_phi = self.hat_phi.to(phi.device)
                self.hint = self.hint.to(phi.device)
            if hasattr(self, "learners"):
                self.learners = [x.to(phi.device) for x in self.learners]
                self.log_weights = self.log_weights.to(phi.device)
        breakdown, grad = evaluate_loss_and_grad(
            phi,
            signal,
            self.shapes,
            self.basis,
            confidence_loss_weight=self.confidence_loss_weight,
            lambda_prox=0.0,
            forward_phi=forward_phi,
        )
        assert grad is not None
        finite_t = torch.isfinite(grad).all() & torch.isfinite(breakdown.total)
        grad, numerical_ok = consensus_gradient(
            grad, finite_t, self.gradient_consensus_fn
        )
        norm_t = torch.linalg.vector_norm(grad)
        norm = norm_t if grad.is_cuda else float(norm_t)
        clipped, scale = clip_gradient_global_norm(grad, self.grad_clip)
        return clipped, norm, scale, breakdown, numerical_ok

    def _project(self, phi: torch.Tensor) -> torch.Tensor:
        return trust_region_project(phi, self.phi0, self.radius)

    def make_candidate(
        self,
        phi_source: torch.Tensor,
        signal: TeacherSignal,
        forward_phi_source: Optional[torch.Tensor] = None,
        cuda_timing_ref: Optional[dict[str, object]] = None,
    ) -> Optional[CandidateUpdate]:
        """Gradient evaluated at the immutable source snapshot on the side
        stream. Skipped teacher signals are never accumulated."""
        self._counter += 1
        if phi_source.is_cuda:
            if cuda_timing_ref is None:
                raise ValueError(
                    "CUDA candidates require an externally leased timing bundle"
                )
            required = {
                "backward_start",
                "backward_end",
                "optimizer_end",
                "optimizer_step_out",
            }
            missing = required.difference(cuda_timing_ref)
            if missing:
                raise ValueError(
                    "CUDA candidate timing bundle is missing "
                    + ", ".join(sorted(missing))
                )
            stream = torch.cuda.current_stream(phi_source.device)
            cuda_timing_ref["backward_start"].record(stream)
        clipped, norm, scale, breakdown, numerical_ok = self._grad_at(
            phi_source,
            signal,
            forward_phi=forward_phi_source,
        )
        optimizer_step: int | torch.Tensor = self._counter
        if cuda_timing_ref is not None:
            stream = torch.cuda.current_stream(phi_source.device)
            cuda_timing_ref["backward_end"].record(stream)
            optimizer_step = cuda_timing_ref["optimizer_step_out"]
            if (
                not isinstance(optimizer_step, torch.Tensor)
                or not optimizer_step.is_cuda
                or optimizer_step.numel() != 1
                or optimizer_step.dtype != torch.int64
            ):
                raise TypeError("optimizer_step_out must be one CUDA int64 scalar")
            cuda_timing_ref["optimizer_end"].record(stream)
            optimizer_step.fill_(self._counter)
        return CandidateUpdate(
            update_id=f"os{signal.source_round}-{self._counter}",
            source_round=signal.source_round,
            source_version=signal.source_version,
            raw_gradient=clipped,
            candidate_delta=torch.zeros_like(clipped),  # SGD applies at arrival
            grad_norm=norm,
            grad_clip_scale=scale,
            loss=breakdown,
            optimizer_step=optimizer_step,
            signal=signal,
            phi_source=phi_source,
            numerical_ok=numerical_ok,
            cuda_timing_ref=cuda_timing_ref,
        )


class OnlineSpecOGD(_OnlineSpecBase):
    """phi_{t+1} = Pi_Phi[phi_t - eta g_t]; Euclidean projected SGD, no
    proximal KL / momentum / Adam state / gradient averaging. Event index
    t follows feedback arrival order; the source round is logged
    separately (spec 6.9)."""

    key = "onlinespec_ogd"

    def decide(self, candidate: CandidateUpdate, ctx: ArrivalContext) -> Decision:
        self.arrival_index += 1
        new_phi = self._project(ctx.phi_active - self.lr * candidate.raw_gradient)
        return Decision(
            kind=DecisionKind.APPLY, published_delta=new_phi - ctx.phi_active
        )


class OnlineSpecOptimistic(_OnlineSpecBase):
    """Two-step optimistic update (spec 6.10):

        phi_t      = Pi[hat_phi_t - eta h_t]
        hat_phi_{t+1} = Pi[hat_phi_t - eta g_t(phi_t)]
        h_{t+1}    = g_t   (the last *arrived and accepted* gradient)
        phi_{t+1}  = Pi[hat_phi_{t+1} - eta h_{t+1}]

    hat_phi_1 = phi_0, h_1 = 0. Forbidden: momentum/EMA/Adam moments as
    hints, unarrived gradients, source-round reordering, or silently
    absorbing version-conflicted gradients into the hint.
    """

    key = "onlinespec_opt"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hat_phi = self.phi0.clone()
        self.hint = torch.zeros_like(self.phi0)

    def decide(self, candidate: CandidateUpdate, ctx: ArrivalContext) -> Decision:
        self.arrival_index += 1
        g_t = candidate.raw_gradient
        self.hat_phi = self._project(self.hat_phi - self.lr * g_t)
        self.hint = g_t.clone()
        new_phi = self._project(self.hat_phi - self.lr * self.hint)
        return Decision(
            kind=DecisionKind.APPLY, published_delta=new_phi - ctx.phi_active
        )


class OnlineSpecEnsemble(_OnlineSpecBase):
    """Three projected-SGD learners (eta, 2eta, 4eta) mixed by Hedge with
    epsilon = 10 (spec 6.11). Log weights are log-sum-exp stabilized; the
    Hedge loss of learner i is the common loss evaluated on learner i's
    own (virtual) proposal for the arrived signal; mixing happens in
    adapter-coefficient space because adapters are linear in the logits.
    The active proposal always uses an immutable copied mixture snapshot.
    """

    key = "onlinespec_ens"
    retain_source_signal = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learners = [self.phi0.clone() for _ in ENSEMBLE_LR_MULTIPLIERS]
        self.log_weights = torch.log(
            torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
        )
        self.history: list[dict] = []

    def mixture(self) -> torch.Tensor:
        pi = torch.softmax(self.log_weights, dim=0)
        mix = torch.zeros_like(self.phi0)
        for w, phi in zip(pi, self.learners):
            mix.add_(phi * w)
        return mix

    def decide(self, candidate: CandidateUpdate, ctx: ArrivalContext) -> Decision:
        self.arrival_index += 1
        signal = candidate.signal
        assert signal is not None
        losses = []
        for i, phi_i in enumerate(self.learners):
            breakdown, grad = evaluate_loss_and_grad(
                phi_i,
                signal,
                self.shapes,
                self.basis,
                confidence_loss_weight=self.confidence_loss_weight,
            )
            assert grad is not None
            finite_t = torch.isfinite(grad).all() & torch.isfinite(
                breakdown.total
            )
            grad, global_ok = consensus_gradient(
                grad, finite_t, self.gradient_consensus_fn
            )
            candidate.numerical_ok = candidate.numerical_ok & global_ok
            clipped, _ = clip_gradient_global_norm(grad, self.grad_clip)
            eta_i = self.lr * ENSEMBLE_LR_MULTIPLIERS[i]
            self.learners[i] = self._project(phi_i - eta_i * clipped)
            ok_t = torch.as_tensor(
                global_ok,
                device=breakdown.total.device,
                dtype=torch.bool,
            )
            losses.append(
                torch.where(
                    ok_t,
                    breakdown.total.detach(),
                    torch.zeros_like(breakdown.total),
                )
            )
        # Hedge on the pre-update learner losses.
        losses_t = torch.stack(losses).to(torch.float64)
        self.log_weights = self.log_weights - HEDGE_EPSILON * losses_t
        # log-sum-exp stabilization.
        self.log_weights = self.log_weights - torch.logsumexp(
            self.log_weights, dim=0
        )
        pi = torch.softmax(self.log_weights, dim=0)
        multipliers = torch.as_tensor(
            ENSEMBLE_LR_MULTIPLIERS,
            device=pi.device,
            dtype=pi.dtype,
        )
        eff_lr = self.lr * torch.dot(pi, multipliers)
        self.history.append(
            {
                "losses": losses_t,
                "weights": pi.detach(),
                "effective_lr": eff_lr.detach(),
            }
        )
        new_mix = self.mixture()
        return Decision(
            kind=DecisionKind.APPLY, published_delta=new_mix - ctx.phi_active
        )

    def telemetry(self) -> dict:
        return {
            "hedge_history": [
                {
                    "losses": item["losses"].cpu().tolist(),
                    "weights": item["weights"].cpu().tolist(),
                    "effective_lr": float(item["effective_lr"].cpu()),
                }
                for item in self.history
            ]
        }
