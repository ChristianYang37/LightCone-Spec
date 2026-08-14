"""Canonical method semantics and registered comparison learners."""

from .core import (
    CandidateReplayBinding,
    CandidateTermination,
    CandidateUpdate,
    MethodPolicy,
    PublicationDelay,
    assert_candidate_equivalence,
    policy_for,
    publication_delay,
    publication_round,
)
from .onlinespec import (
    OnlineSpecHedge,
    OnlineSpecOGD,
    OnlineSpecOptimistic,
    OnlineSpecProposal,
    ogd_update,
    project_l2_ball,
)

__all__ = [
    "CandidateReplayBinding",
    "CandidateTermination",
    "CandidateUpdate",
    "MethodPolicy",
    "OnlineSpecHedge",
    "OnlineSpecOGD",
    "OnlineSpecOptimistic",
    "OnlineSpecProposal",
    "PublicationDelay",
    "assert_candidate_equivalence",
    "ogd_update",
    "policy_for",
    "project_l2_ball",
    "publication_delay",
    "publication_round",
]
