"""Optimizer primitives (spec 6.3, 6.8).

AdamWDelta: exactly one AdamW step per trigger, decoupled weight decay,
betas (0.9, 0.999), eps 1e-8. The function returns the candidate delta
u_r that L0 would publish; controller scaling (L2/L3) is applied to this
optimizer-transformed delta, never to the gradient fed into Adam.

SGD: plain projected SGD without momentum or weight decay (OnlineSpec
family).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from lightcone_spec.exit_codes import NumericalFailure

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


@dataclass
class AdamWDeltaState:
    """Request-local AdamW state; lifecycle-bound (destroyed with the
    request in request mode)."""

    num_params: int
    step: int | torch.Tensor = 0
    exp_avg: torch.Tensor = field(default=None)  # type: ignore[assignment]
    exp_avg_sq: torch.Tensor = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.exp_avg is None:
            self.exp_avg = torch.zeros(self.num_params, dtype=torch.float32)
        if self.exp_avg_sq is None:
            self.exp_avg_sq = torch.zeros(self.num_params, dtype=torch.float32)

    def clone(self) -> "AdamWDeltaState":
        return AdamWDeltaState(
            num_params=self.num_params,
            step=(self.step.clone() if isinstance(self.step, torch.Tensor) else self.step),
            exp_avg=self.exp_avg.clone(),
            exp_avg_sq=self.exp_avg_sq.clone(),
        )

    def ensure_device(self, device: torch.device) -> None:
        """Move persistent optimizer state once, before the queued update.

        This is deliberately lazy so CPU reference tests retain their exact
        behaviour while GPU runtimes keep moments on the side-stream device.
        """
        if self.exp_avg.device != device:
            self.exp_avg = self.exp_avg.to(device=device)
            self.exp_avg_sq = self.exp_avg_sq.to(device=device)
        if device.type == "cuda" and not isinstance(self.step, torch.Tensor):
            self.step = torch.tensor(
                self.step, dtype=torch.int64, device=device
            )

    def bind(self, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor) -> None:
        """Bind request-local state to fixed-address slot storage."""
        expected = (self.num_params,)
        if tuple(exp_avg.shape) != expected or tuple(exp_avg_sq.shape) != expected:
            raise ValueError("optimizer slot state shape mismatch")
        self.exp_avg = exp_avg
        self.exp_avg_sq = exp_avg_sq
        self.step = (
            torch.zeros((), dtype=torch.int64, device=exp_avg.device)
            if exp_avg.is_cuda
            else 0
        )

    def state_dict(self) -> dict:
        return {
            "num_params": self.num_params,
            "step": (
                self.step.clone()
                if isinstance(self.step, torch.Tensor)
                else self.step
            ),
            "exp_avg": self.exp_avg.clone(),
            "exp_avg_sq": self.exp_avg_sq.clone(),
        }


def adamw_delta(
    grad: torch.Tensor,
    state: AdamWDeltaState,
    lr: float,
    advance_state: bool = True,
    valid: bool | torch.Tensor = True,
    *,
    parameter: torch.Tensor | None = None,
    weight_decay: float = 0.0,
) -> torch.Tensor:
    """Return one decoupled AdamW delta to add to ``parameter``.

    ``parameter`` is required only for non-zero decay, preserving every
    historical zero-decay caller.  Invalid GPU candidates advance neither
    moments nor parameters, including the decay term.
    """
    if not 0.0 <= float(weight_decay) < float("inf"):
        raise ValueError("weight_decay must be finite and non-negative")
    g = grad.to(torch.float32)
    parameter_fp32 = None
    if weight_decay != 0.0:
        if parameter is None:
            raise ValueError("non-zero AdamW weight_decay requires parameter")
        if tuple(parameter.shape) != tuple(g.shape):
            raise ValueError("AdamW parameter and gradient shapes differ")
        parameter_fp32 = parameter.to(device=g.device, dtype=torch.float32)
    if not g.is_cuda and not torch.isfinite(g).all():
        raise NumericalFailure("non-finite gradient passed to AdamWDelta")
    st = state if advance_state else state.clone()
    st.ensure_device(g.device)
    if isinstance(st.step, torch.Tensor):
        valid_t = torch.as_tensor(valid, device=g.device, dtype=torch.bool)
        if valid_t.numel() != 1:
            raise ValueError("AdamW validity flag must be scalar")
        next_step = st.step + 1
        next_exp_avg = st.exp_avg * ADAM_BETA1 + g * (1.0 - ADAM_BETA1)
        next_exp_avg_sq = (
            st.exp_avg_sq * ADAM_BETA2 + g.square() * (1.0 - ADAM_BETA2)
        )
        step_f = next_step.to(torch.float32)
        bias1 = 1.0 - torch.pow(
            torch.as_tensor(ADAM_BETA1, device=g.device), step_f
        )
        bias2 = 1.0 - torch.pow(
            torch.as_tensor(ADAM_BETA2, device=g.device), step_f
        )
        m_hat = next_exp_avg / bias1
        v_hat = next_exp_avg_sq / bias2
        update = m_hat / (v_hat.sqrt() + ADAM_EPS)
        if parameter_fp32 is not None:
            update = update + float(weight_decay) * parameter_fp32
        delta = -lr * update
        delta = torch.where(valid_t, delta, torch.zeros_like(delta))
        st.exp_avg.copy_(torch.where(valid_t, next_exp_avg, st.exp_avg))
        st.exp_avg_sq.copy_(
            torch.where(valid_t, next_exp_avg_sq, st.exp_avg_sq)
        )
        st.step.copy_(torch.where(valid_t, next_step, st.step))
    else:
        if not bool(valid):
            return torch.zeros_like(g)
        st.step += 1
        st.exp_avg.mul_(ADAM_BETA1).add_(g, alpha=1.0 - ADAM_BETA1)
        st.exp_avg_sq.mul_(ADAM_BETA2).addcmul_(g, g, value=1.0 - ADAM_BETA2)
        bias1 = 1.0 - ADAM_BETA1**st.step
        bias2 = 1.0 - ADAM_BETA2**st.step
        m_hat = st.exp_avg / bias1
        v_hat = st.exp_avg_sq / bias2
        update = m_hat / (v_hat.sqrt() + ADAM_EPS)
        if parameter_fp32 is not None:
            update = update + float(weight_decay) * parameter_fp32
        delta = -lr * update
    if not advance_state:
        pass  # state clone discarded; caller wanted a pure preview
    if not delta.is_cuda and not torch.isfinite(delta).all():
        raise NumericalFailure("non-finite AdamW delta")
    return delta


def adamw_delta_batched(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    steps: torch.Tensor,
    lr: float,
    valid: torch.Tensor,
    *,
    parameter: torch.Tensor | None = None,
    weight_decay: float = 0.0,
) -> torch.Tensor:
    """Vectorized request-independent AdamW deltas.

    Every row is one request-local optimizer.  ``exp_avg``, ``exp_avg_sq``
    and ``steps`` are mutated in place, but there is deliberately no state
    sharing across rows.  This primitive lets one verification batch use a
    handful of elementwise kernels instead of launching an AdamW expression
    once per request.

    The caller owns gathering/scattering non-contiguous slot rows.  Keeping
    that policy outside this numerical primitive makes the same code usable
    by the CPU reference tests and by the fixed-address GPU bank.
    """

    if grad.ndim != 2:
        raise ValueError("batched AdamW gradient must have shape (B, P)")
    if exp_avg.shape != grad.shape or exp_avg_sq.shape != grad.shape:
        raise ValueError("batched AdamW moment and gradient shapes differ")
    batch_size = int(grad.shape[0])
    if steps.shape != (batch_size,) or steps.dtype != torch.int64:
        raise ValueError("batched AdamW steps must be int64 with shape (B,)")
    if steps.device != grad.device:
        raise ValueError("batched AdamW steps and gradients must share a device")
    if valid.shape != (batch_size,) or valid.dtype != torch.bool:
        raise ValueError("batched AdamW validity must be bool with shape (B,)")
    if valid.device != grad.device:
        raise ValueError("batched AdamW validity and gradients must share a device")
    if not 0.0 <= float(weight_decay) < float("inf"):
        raise ValueError("weight_decay must be finite and non-negative")

    g = grad.to(torch.float32)
    parameter_fp32 = None
    if weight_decay != 0.0:
        if parameter is None:
            raise ValueError("non-zero AdamW weight_decay requires parameter")
        if parameter.shape != grad.shape:
            raise ValueError("batched AdamW parameter and gradient shapes differ")
        parameter_fp32 = parameter.to(device=g.device, dtype=torch.float32)
    if not g.is_cuda and not torch.isfinite(g).all():
        raise NumericalFailure("non-finite gradient passed to batched AdamWDelta")

    valid_col = valid[:, None]
    next_steps = steps + 1
    next_exp_avg = exp_avg * ADAM_BETA1 + g * (1.0 - ADAM_BETA1)
    next_exp_avg_sq = (
        exp_avg_sq * ADAM_BETA2 + g.square() * (1.0 - ADAM_BETA2)
    )
    step_f = next_steps.to(torch.float32)[:, None]
    beta1 = torch.as_tensor(ADAM_BETA1, device=g.device, dtype=torch.float32)
    beta2 = torch.as_tensor(ADAM_BETA2, device=g.device, dtype=torch.float32)
    bias1 = 1.0 - torch.pow(beta1, step_f)
    bias2 = 1.0 - torch.pow(beta2, step_f)
    update = (next_exp_avg / bias1) / (
        (next_exp_avg_sq / bias2).sqrt() + ADAM_EPS
    )
    if parameter_fp32 is not None:
        update = update + float(weight_decay) * parameter_fp32
    delta = torch.where(valid_col, -float(lr) * update, torch.zeros_like(update))

    exp_avg.copy_(torch.where(valid_col, next_exp_avg, exp_avg))
    exp_avg_sq.copy_(torch.where(valid_col, next_exp_avg_sq, exp_avg_sq))
    steps.copy_(torch.where(valid, next_steps, steps))
    if not delta.is_cuda and not torch.isfinite(delta).all():
        raise NumericalFailure("non-finite batched AdamW delta")
    return delta


def sgd_step(
    phi: torch.Tensor,
    grad: torch.Tensor,
    lr: float,
    phi0: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Projected OGD step: phi_{t+1} = Pi_Phi[phi_t - eta * g_t]."""
    from lightcone_spec.adapters.adapter_params import trust_region_project

    g = grad.to(torch.float32)
    if not g.is_cuda and not torch.isfinite(g).all():
        raise NumericalFailure("non-finite gradient passed to SGD")
    return trust_region_project(phi.to(torch.float32) - lr * g, phi0, radius)
