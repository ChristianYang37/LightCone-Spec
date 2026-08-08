"""DSpark-Static, Sync-Fresh, TTS-DSpark, L0-NaiveAsync and
Oracle-Current (spec 6.5-6.8, 6.12)."""

from __future__ import annotations

from typing import Optional

import torch

from lightcone_spec.adapters.adapter_params import (
    AdapterShapes,
    clip_gradient_global_norm,
)
from lightcone_spec.methods.base import (
    ArrivalContext,
    CandidateGeneratorConfig,
    CandidateUpdate,
    CommonCandidateGenerator,
    Decision,
    DecisionKind,
    MethodRuntime,
    PublishPolicy,
    TeacherSignal,
    consensus_gradient,
)
from lightcone_spec.methods.optim import adamw_delta


class StaticMethod(MethodRuntime):
    """DSpark-Static: adapter stays zero, no optimizer state is allocated,
    light telemetry only. The unique speedup denominator (spec 6.5)."""

    key = "static"
    publish_policy = PublishPolicy.NONE

    def make_candidate(
        self,
        phi_source: torch.Tensor,
        signal: TeacherSignal,
        forward_phi_source: Optional[torch.Tensor] = None,
        cuda_timing_ref: Optional[dict[str, object]] = None,
    ) -> Optional[CandidateUpdate]:
        del forward_phi_source, cuda_timing_ref
        return None

    def decide(self, candidate: CandidateUpdate, ctx: ArrivalContext) -> Decision:
        raise RuntimeError("static never produces candidates")


class _AdamWCandidateMethod(MethodRuntime):
    """Shared base for the TTS-candidate-generator family."""

    def __init__(
        self,
        shapes: AdapterShapes,
        basis: torch.Tensor,
        cfg: CandidateGeneratorConfig,
        defer_state_advance: bool = False,
    ):
        self.generator = CommonCandidateGenerator(shapes, basis, cfg)
        self.defer_state_advance = defer_state_advance

    def bind_slot_state(self, exp_avg, exp_avg_sq, fisher=None) -> None:
        del fisher
        if exp_avg is None or exp_avg_sq is None:
            raise ValueError("AdamW method requires fixed optimizer state")
        self.generator.state.bind(exp_avg, exp_avg_sq)

    def bind_candidate_preview(self, exp_avg, exp_avg_sq) -> None:
        if not self.defer_state_advance:
            return
        self.generator.bind_preview_state(exp_avg, exp_avg_sq)

    def bind_gradient_consensus(self, callback) -> None:
        self.generator.bind_gradient_consensus(callback)

    def prepare_candidate_preview(self) -> None:
        if self.defer_state_advance:
            self.generator.prepare_preview_state()

    def common_candidate_generator(self) -> CommonCandidateGenerator:
        return self.generator

    def make_candidate(
        self,
        phi_source: torch.Tensor,
        signal: TeacherSignal,
        forward_phi_source: Optional[torch.Tensor] = None,
        cuda_timing_ref: Optional[dict[str, object]] = None,
    ) -> Optional[CandidateUpdate]:
        return self.generator.candidate(
            phi_source,
            signal,
            defer_state_advance=self.defer_state_advance,
            forward_phi_source=forward_phi_source,
            cuda_timing_ref=cuda_timing_ref,
        )


class SyncFreshMethod(_AdamWCandidateMethod):
    """Sync-Fresh: same candidate generator as L0-L3, but the main stream
    blocks after the teacher signal is ready; the update completes and is
    published at a legal boundary before the next round. c(r) = r + 1.
    A low-staleness *utility* upper bound, not a throughput bound."""

    key = "sync_fresh"
    publish_policy = PublishPolicy.BLOCKING

    def decide(self, candidate: CandidateUpdate, ctx: ArrivalContext) -> Decision:
        return Decision(kind=DecisionKind.APPLY, published_delta=candidate.candidate_delta)


class TTSDSparkMethod(_AdamWCandidateMethod):
    """TTS-DSpark: TTS loss (common loss + lambda * proximal KL against
    the immutable source proposal), single-step AdamW,
    latest-signal-only, fixed stride pipeline (spec 6.7).

    Conformance note: at the source parameter point the first-order
    gradient of the proximal KL is exactly zero, so a single step should
    not vary with lambda beyond numerical tolerance; a visible lambda
    dependence indicates an implementation/source-binding bug and must
    not be hidden.
    """

    key = "tts"
    publish_policy = PublishPolicy.FIXED_STRIDE_BARRIER

    def decide(self, candidate: CandidateUpdate, ctx: ArrivalContext) -> Decision:
        return Decision(kind=DecisionKind.APPLY, published_delta=candidate.candidate_delta)


class NaiveAsyncMethod(_AdamWCandidateMethod):
    """L0-NaiveAsync: same candidate as TTS; applied unconditionally at
    the first legal graph boundary after the candidate is ready; never
    waits for the next stride barrier; never consults delay/rho/utility.
    A lagging source version with max_in_flight=1 is discarded and logged
    as `version_conflict` -- never silently rebased (spec 6.8)."""

    key = "naive_async"
    publish_policy = PublishPolicy.ASYNC_BOUNDARY

    def decide(self, candidate: CandidateUpdate, ctx: ArrivalContext) -> Decision:
        if candidate.source_version != ctx.active_version:
            return Decision(
                kind=DecisionKind.VERSION_CONFLICT,
                published_delta=torch.zeros_like(candidate.candidate_delta),
            )
        return Decision(kind=DecisionKind.APPLY, published_delta=candidate.candidate_delta)


class OracleCurrentMethod(_AdamWCandidateMethod):
    """Oracle-Current: on arrival, recomputes a fresh teacher signal and
    gradient g_a on the arrival prefix, then uses the same TTS candidate
    generator / AdamW state / lr / trust region. Source gradients are
    never used; the extra target/backward cost is fully recorded. Upper
    bound only (spec 6.12)."""

    key = "oracle_current"
    publish_policy = PublishPolicy.ASYNC_BOUNDARY
    needs_fresh_gradient_at_arrival = True

    def __init__(self, shapes, basis, cfg):
        super().__init__(shapes, basis, cfg, defer_state_advance=True)

    def decide(self, candidate: CandidateUpdate, ctx: ArrivalContext) -> Decision:
        if ctx.fresh_gradient is None:
            raise RuntimeError("oracle_current requires a fresh arrival gradient")
        finite_t = torch.isfinite(ctx.fresh_gradient).all()
        fresh_gradient, global_ok = consensus_gradient(
            ctx.fresh_gradient,
            finite_t,
            self.generator.gradient_consensus_fn,
        )
        candidate.numerical_ok = candidate.numerical_ok & global_ok
        clipped, _ = clip_gradient_global_norm(
            fresh_gradient, self.generator.cfg.grad_clip
        )
        delta = adamw_delta(
            clipped,
            self.generator.state,
            self.generator.cfg.lr,
            valid=candidate.numerical_ok,
            parameter=ctx.phi_active,
            weight_decay=self.generator.cfg.weight_decay,
        )
        return Decision(kind=DecisionKind.APPLY, published_delta=delta)
