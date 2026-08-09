"""Registered clean-room comparisons; never selected into the core gate."""

from .onlinespec import (
    OnlineSpecHedge,
    OnlineSpecOGD,
    OnlineSpecOptimistic,
    OnlineSpecProposal,
    ogd_update,
    project_l2_ball,
)

__all__ = [
    "OnlineSpecHedge",
    "OnlineSpecOGD",
    "OnlineSpecOptimistic",
    "OnlineSpecProposal",
    "ogd_update",
    "project_l2_ball",
]
