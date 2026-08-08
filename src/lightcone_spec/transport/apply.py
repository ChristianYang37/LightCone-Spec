"""Arrival-time transport (spec 7.9).

g_tilde_r = g_r + diag(F_r) s_r^phi + P_g A delta_z_r

The transported gradient goes through the same request-local AdamW state
as L0, then L2 damping scales the resulting delta. Variants:
parameter-only, state-only, joint, random-basis, discard, L2
no-transport -- all must be implemented and reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from lightcone_spec.transport.fisher import FisherEMA
from lightcone_spec.transport.fit import TransportMap

TRANSPORT_VARIANTS = (
    "joint",
    "parameter_only",
    "state_only",
    "random",
    "discard",
    "l2_no_transport",
)


@dataclass
class TransportResult:
    transported_grad: torch.Tensor
    parameter_comp_norm: float | torch.Tensor
    state_transport_norm: float | torch.Tensor
    random_transport: bool


def transport_gradient(
    raw_grad: torch.Tensor,
    fisher: Optional[FisherEMA | torch.Tensor],
    s_phi: Optional[torch.Tensor],
    transport_map: Optional[TransportMap],
    delta_z: Optional[np.ndarray | torch.Tensor],
    variant: str = "joint",
) -> TransportResult:
    if variant not in TRANSPORT_VARIANTS:
        raise ValueError(f"unknown transport variant {variant!r}")
    g = raw_grad.to(torch.float32).clone()
    param_norm = 0.0
    state_norm = 0.0
    random_used = False

    use_param = variant in ("joint", "parameter_only", "random")
    use_state = variant in ("joint", "state_only", "random")

    if use_param and fisher is not None and s_phi is not None:
        comp = (
            fisher.parameter_compensation(s_phi)
            if isinstance(fisher, FisherEMA)
            else fisher.to(s_phi.device) * s_phi.to(torch.float32)
        )
        norm = torch.linalg.vector_norm(comp)
        param_norm = norm if norm.is_cuda else float(norm)
        g = g + comp
    if use_state and transport_map is not None and delta_z is not None:
        if isinstance(delta_z, torch.Tensor):
            corr_t = transport_map.state_correction_tensor(delta_z.to(g.device))
        else:
            corr = transport_map.state_correction(delta_z)
            corr_t = torch.from_numpy(np.asarray(corr, dtype=np.float32)).to(g.device)
        norm = torch.linalg.vector_norm(corr_t)
        state_norm = norm if norm.is_cuda else float(norm)
        g = g + corr_t
        random_used = transport_map.random_basis
    return TransportResult(
        transported_grad=g,
        parameter_comp_norm=param_norm,
        state_transport_norm=state_norm,
        random_transport=random_used,
    )
