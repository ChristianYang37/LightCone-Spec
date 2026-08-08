"""Typed hook payloads for the seven DSparkWorkerV2 boundaries
(spec 8.2).

The fork's `DSparkAdaptationManager` builds these records and calls the
runtime's matching methods. Every payload carries the version fields the
exactness proof needs; hooks are only invoked when the adaptation config
is set (upstream parity otherwise).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DraftInputsReady:
    """Hook 1: draft inputs ready (before proposal computation)."""

    request_id: str
    round_id: int
    request_epoch: int
    slot_index: int
    active_version: int
    prefix_len: int
    hidden_ref: Any = None  # device tensor reference (never copied here)
    markov_ref: Any = None


@dataclass
class ProposalReady:
    """Hook 2: proposal distribution ready (canvas binding point)."""

    request_id: str
    round_id: int
    proposal_version: int
    draft_tokens: list[int] = field(default_factory=list)
    proposal_probs_ref: Any = None
    confidence_logits_ref: Any = None
    rng_substream_ids: list[str] = field(default_factory=list)


@dataclass
class VerifyLogitsReady:
    """Hook 3: target verify logits ready (teacher signal source)."""

    request_id: str
    round_id: int
    proposal_version: int
    target_logits_ref: Any = None
    target_hidden_ref: Any = None
    valid_positions: int = 0


@dataclass
class AcceptanceDone:
    """Hook 4: acceptance/rejection resolved for the round."""

    request_id: str
    round_id: int
    proposal_version: int
    denominator_version: int
    residual_version: int
    accepted_drafts: int = 0
    committed_tokens: list[int] = field(default_factory=list)
    used_bonus: bool = False


@dataclass
class RoundCommitted:
    """Hook 5: round outcome committed to the sequence state."""

    request_id: str
    round_id: int
    prefix_len_after: Optional[int]
    active_version: int
    proposal_version: int = 0
    draft_tokens: int = 0
    accepted_drafts: int = 0
    committed_per_verify: int = 0
    target_calls: int = 1
    rng_substream_id: str = ""
    round_wall_us: float = 0.0
    draft_cpu_us: float = 0.0
    verify_cpu_us: float = 0.0
    draft_cuda_us: float = 0.0
    verify_cuda_us: float = 0.0
    accept_cuda_us: float = 0.0
    verify_len: int = 0
    batch_size: int = 1
    offered_concurrency: int = 1
    algorithmic_censored: bool = False
    # Whether ``prefix_len_after`` was observed exactly at this commit
    # boundary. Device-only backends that can only reuse the previous host
    # length must say so explicitly; controller labels then fail closed.
    prefix_feature_exact: bool = True
    cuda_timing_ref: Any = None


def rng_substream_identity(
    *,
    request_id: str,
    sampling_seed: int | None,
    is_greedy: bool,
    round_id: int,
    prefix_len: int,
) -> str:
    """Describe the exact request-level sampling substream for one round."""

    round_id = int(round_id)
    prefix_len = int(prefix_len)
    if round_id < 0 or prefix_len < 0:
        raise ValueError("RNG substream round and prefix must be non-negative")
    if sampling_seed is not None:
        sampling_seed = int(sampling_seed)
        if sampling_seed < 0:
            raise ValueError("sampling_seed must be non-negative")
        request_key = f"seed-{sampling_seed}"
    elif is_greedy:
        if not request_id:
            raise ValueError("greedy RNG identity requires a request id or seed")
        request_key = "request-" + hashlib.sha256(
            str(request_id).encode("utf-8")
        ).hexdigest()[:16]
    else:
        raise ValueError(
            "stochastic speculative decoding requires a request-level sampling_seed"
        )
    mode = "deterministic-greedy" if is_greedy else "seeded-stochastic"
    return f"{mode}-v1:{request_key}:round-{round_id}:prefix-{prefix_len}"


@dataclass
class UpdatePollPoint:
    """Hook 6: legal graph boundary; controller may publish staged
    parameters here and only here."""

    request_id: str
    round_id: int
    request_epoch: int
    slot_index: int
    active_version: int
    in_replay: bool = False


@dataclass
class RequestLifecycle:
    """Hook 7: request/stream begin & end (slot allocate/reset/free)."""

    request_id: str
    event: str  # begin | end | stream_begin | stream_end
    request_epoch: int
    slot_index: int
    stream_id: Optional[str] = None
    tenant_id_hash: str = ""


class AdaptationHooks:
    """Interface the fork calls; implemented by
    `lightcone_spec.sglang_bridge.runtime.AdaptationRuntime`."""

    def on_draft_inputs_ready(self, ev: DraftInputsReady) -> None: ...

    def on_proposal_ready(self, ev: ProposalReady) -> None: ...

    def on_verify_logits_ready(self, ev: VerifyLogitsReady) -> None: ...

    def on_acceptance_done(self, ev: AcceptanceDone) -> None: ...

    def on_round_committed(self, ev: RoundCommitted) -> None: ...

    def on_update_poll(self, ev: UpdatePollPoint) -> Optional[int]:
        """May return a new active version if a publish happened."""
        ...

    def on_request_lifecycle(self, ev: RequestLifecycle) -> None: ...
