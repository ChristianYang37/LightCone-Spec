"""LightCone controller methods L1-L3 (spec 6.8, 7.7-7.9) and the five
diagnostic negative controls (spec 6.14).

All of them consume the exact same TTS candidate generator output; the
controller only decides apply/discard/damp/transport on the
optimizer-produced candidate delta u_r.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from lightcone_spec.adapters.adapter_params import (
    AdapterShapes,
    clip_gradient_global_norm,
)
from lightcone_spec.controller.artifact import ControllerArtifact
from lightcone_spec.controller.damping import damping_factor
from lightcone_spec.controller.gate import gate_decision
from lightcone_spec.methods.base import (
    ArrivalContext,
    CandidateGeneratorConfig,
    CandidateUpdate,
    Decision,
    DecisionKind,
    PublishPolicy,
)
from lightcone_spec.methods.simple import _AdamWCandidateMethod
from lightcone_spec.transport.apply import transport_gradient
from lightcone_spec.transport.fisher import FisherEMA
from lightcone_spec.trajectory.features import UpdateFeatureRow


def _feature_vector(ctx: ArrivalContext, feature_set: str) -> np.ndarray | torch.Tensor:
    from lightcone_spec.trajectory.features import FEATURE_SETS

    tensor_values = (
        ctx.endpoint_distance,
        ctx.rho_path,
        ctx.parameter_displacement,
        ctx.source_acceptance,
        ctx.source_training_loss,
        ctx.source_grad_norm,
    )
    if any(isinstance(value, torch.Tensor) for value in tensor_values):
        device = ctx.phi_active.device

        def tensor(value):
            return torch.as_tensor(value, device=device, dtype=torch.float32)

        feats = {
            "log1p_source_prefix_len": torch.log1p(
                tensor(max(ctx.source_prefix_len, 0))
            ),
            "source_acceptance": tensor(ctx.source_acceptance),
            "log1p_source_training_loss": torch.log1p(
                tensor(ctx.source_training_loss).clamp_min(0.0)
            ),
            "log1p_source_grad_norm": torch.log1p(
                tensor(ctx.source_grad_norm).clamp_min(0.0)
            ),
            "log1p_round_delay": torch.log1p(tensor(max(ctx.delay_rounds, 0))),
            "log1p_token_delay": torch.log1p(tensor(max(ctx.delay_tokens, 0))),
            "log1p_wall_us": torch.log1p(tensor(max(ctx.delay_wall_us, 0.0))),
            "endpoint_distance": tensor(ctx.endpoint_distance),
            "rho_path": tensor(ctx.rho_path),
            "parameter_displacement": tensor(ctx.parameter_displacement),
        }
        return torch.stack([feats[name] for name in FEATURE_SETS[feature_set]])
    row = UpdateFeatureRow(
        sequence_id="",
        update_id="",
        round_delay=ctx.delay_rounds,
        token_delay=ctx.delay_tokens,
        wall_us=ctx.delay_wall_us,
        endpoint_distance=ctx.endpoint_distance,
        rho_path=ctx.rho_path,
        parameter_displacement=ctx.parameter_displacement,
        utility=0.0,
        relative_gradient_mismatch=0.0,
        harmful=0,
        source_prefix_len=ctx.source_prefix_len,
        source_acceptance=ctx.source_acceptance,
        source_training_loss=ctx.source_training_loss,
        source_grad_norm=ctx.source_grad_norm,
    )
    feats = row.features()
    return np.asarray([feats[n] for n in FEATURE_SETS[feature_set]], dtype=np.float64)


class _ControllerMethod(_AdamWCandidateMethod):
    """Base for controller-driven methods; holds the frozen artifact."""

    def __init__(
        self,
        shapes: AdapterShapes,
        basis: torch.Tensor,
        cfg: CandidateGeneratorConfig,
        artifact: ControllerArtifact,
        defer_state_advance: bool = False,
    ):
        super().__init__(shapes, basis, cfg, defer_state_advance=defer_state_advance)
        self.artifact = artifact

    def _predictions(self, ctx: ArrivalContext):
        x = _feature_vector(ctx, self.artifact.feature_set)
        if isinstance(x, torch.Tensor):
            return (
                self.artifact.utility_predictor.predict_tensor(x),
                self.artifact.mismatch_predictor.predict_tensor(x),
                self.artifact.harmful_classifier.probability_tensor(x),
            )
        util = float(self.artifact.utility_predictor.predict(x)[0])
        mism = float(self.artifact.mismatch_predictor.predict(x)[0])
        harm = float(self.artifact.harmful_classifier.probability(x)[0])
        return util, mism, harm


class LCGateMethod(_ControllerMethod):
    """L1-LC-Gate: trajectory-aware apply/discard on the full candidate
    delta u_r (never a recomputed alternative update)."""

    key = "lc_gate"
    publish_policy = PublishPolicy.ASYNC_BOUNDARY

    def decide(self, candidate: CandidateUpdate, ctx: ArrivalContext) -> Decision:
        if ctx.delay_rounds == 0:
            # Fresh parity is an unconditional identity action.  Avoid all
            # feature and predictor kernels on this common path.
            return Decision(
                kind=DecisionKind.APPLY,
                published_delta=candidate.candidate_delta,
                threshold=self.artifact.gate_threshold,
                gate_applied=True,
            )
        # A controller fitted with no safe threshold is an explicit static
        # fallback.  This is host-resident artifact state, so honoring it here
        # avoids predictor kernels, a zero-valued publish, and a spurious
        # adapter-version advance without introducing a device sync.
        fixed_discard = ctx.delay_rounds in set(
            getattr(self.artifact, "extra", {}).get(
                "gate_constant_discard_delays", ()
            )
        )
        if ctx.delay_rounds > 0 and (
            self.artifact.gate_discard_all or fixed_discard
        ):
            return Decision(
                kind=DecisionKind.DISCARD,
                published_delta=torch.zeros_like(candidate.candidate_delta),
                threshold=self.artifact.gate_threshold,
                gate_applied=False,
            )
        fixed_apply = ctx.delay_rounds in set(
            getattr(self.artifact, "extra", {}).get(
                "gate_constant_apply_delays", ()
            )
        )
        if fixed_apply:
            profile = getattr(self.artifact, "extra", {}).get(
                "constant_controller_profiles", {}
            ).get(str(ctx.delay_rounds), {})
            return Decision(
                kind=DecisionKind.APPLY,
                published_delta=candidate.candidate_delta,
                predicted_utility=profile.get("predicted_utility"),
                predicted_mismatch=profile.get("predicted_mismatch"),
                predicted_harm_probability=profile.get(
                    "predicted_harm_probability"
                ),
                threshold=self.artifact.gate_threshold,
                gate_applied=True,
            )
        util, mism, harm = self._predictions(ctx)
        apply = gate_decision(harm, self.artifact.gate_threshold)
        if isinstance(apply, torch.Tensor):
            published = torch.where(
                apply, candidate.candidate_delta, torch.zeros_like(candidate.candidate_delta)
            )
            kind = DecisionKind.APPLY
        else:
            published = (
                candidate.candidate_delta
                if apply
                else torch.zeros_like(candidate.candidate_delta)
            )
            kind = DecisionKind.APPLY if apply else DecisionKind.DISCARD
        return Decision(
            kind=kind,
            published_delta=published,
            predicted_utility=util,
            predicted_mismatch=mism,
            predicted_harm_probability=harm,
            threshold=self.artifact.gate_threshold,
            gate_applied=apply,
        )


class LCDampMethod(_ControllerMethod):
    """L2-LC-Damp: mismatch-aware continuous scaling kappa_r * u_r with
    the exponential kernel and calibrated radius (spec 7.8)."""

    key = "lc_damp"
    publish_policy = PublishPolicy.ASYNC_BOUNDARY

    def decide(self, candidate: CandidateUpdate, ctx: ArrivalContext) -> Decision:
        profile = getattr(self.artifact, "extra", {}).get(
            "constant_controller_profiles", {}
        ).get(str(ctx.delay_rounds))
        if profile is None:
            util, mism, harm = self._predictions(ctx)
        else:
            util = float(profile["predicted_utility"])
            mism = float(profile["predicted_mismatch"])
            harm = float(profile["predicted_harm_probability"])
        if ctx.delay_rounds == 0:
            kappa = 1.0
        elif profile is not None:
            kappa = float(profile["damping_factor"])
        else:
            kappa = damping_factor(
                mism, self.artifact.damping_radius, self.artifact.damping_kernel
            )
        published_delta = (
            candidate.candidate_delta
            if isinstance(kappa, (float, int)) and float(kappa) == 1.0
            else kappa * candidate.candidate_delta
        )
        return Decision(
            kind=DecisionKind.DAMP,
            published_delta=published_delta,
            damping_factor=kappa,
            predicted_utility=util,
            predicted_mismatch=mism,
            predicted_harm_probability=harm,
        )


class LCTransportMethod(_ControllerMethod):
    """L3-LC-Transport: parameter compensation + low-rank state transport
    + damping (spec 7.9). Maintains the Fisher EMA over raw gradients;
    feeds the transported gradient through the same shared AdamW state,
    then applies L2 damping to the resulting delta."""

    key = "lc_transport"
    publish_policy = PublishPolicy.ASYNC_BOUNDARY
    needs_delta_z = True

    def __init__(self, shapes, basis, cfg, artifact, variant: str = "joint"):
        super().__init__(shapes, basis, cfg, artifact, defer_state_advance=True)
        self.variant = variant
        self.fisher = FisherEMA(
            shapes.num_params(),
            decay=artifact.extra.get("fisher_decay", 0.99),
        )

    def bind_slot_state(self, exp_avg, exp_avg_sq, fisher=None) -> None:
        super().bind_slot_state(exp_avg, exp_avg_sq)
        if fisher is None:
            raise ValueError("LC transport requires fixed Fisher state")
        self.fisher.bind(fisher)

    def make_candidate(
        self,
        phi_source,
        signal,
        forward_phi_source=None,
        cuda_timing_ref=None,
    ):
        cand = super().make_candidate(
            phi_source,
            signal,
            forward_phi_source=forward_phi_source,
            cuda_timing_ref=cuda_timing_ref,
        )
        if cand is not None:
            self.after_batched_candidate(cand)
        return cand

    def after_batched_candidate(self, candidate: CandidateUpdate) -> None:
        """Preserve L3's request-local Fisher update after shared backward."""

        self.fisher.update(
            candidate.raw_gradient, valid=candidate.numerical_ok
        )
        candidate.fisher_snapshot = (
            self.fisher.value
            if self.fisher.value.is_cuda
            else self.fisher.value.detach().clone()
        )

    def decide(self, candidate: CandidateUpdate, ctx: ArrivalContext) -> Decision:
        util, mism, harm = self._predictions(ctx)
        phi_src = (
            candidate.phi_source
            if candidate.phi_source is not None
            else torch.zeros_like(ctx.phi_active)
        )
        s_phi = ctx.phi_active - phi_src
        if self.variant == "discard":
            return Decision(
                kind=DecisionKind.DISCARD,
                published_delta=torch.zeros_like(candidate.candidate_delta),
                predicted_utility=util,
                predicted_mismatch=mism,
                predicted_harm_probability=harm,
                transport_rank=self.artifact.transport_map.rank
                if self.artifact.transport_map
                else None,
            )
        if self.variant == "l2_no_transport":
            # This branch publishes a real optimizer step, so advance the
            # shared state exactly once just like the transported path.
            base_delta = self.generator.delta_from_transported_gradient(
                candidate.raw_gradient,
                parameter=ctx.phi_active,
                valid=candidate.numerical_ok,
            )
            kappa = (
                1.0
                if ctx.delay_rounds == 0
                else damping_factor(
                    mism, self.artifact.damping_radius, self.artifact.damping_kernel
                )
            )
            return Decision(
                kind=DecisionKind.DAMP,
                published_delta=kappa * base_delta,
                damping_factor=kappa,
                predicted_utility=util,
                predicted_mismatch=mism,
                predicted_harm_probability=harm,
            )
        result = transport_gradient(
            candidate.raw_gradient,
            (
                candidate.fisher_snapshot
                if candidate.fisher_snapshot is not None
                else self.fisher
            ),
            s_phi,
            self.artifact.transport_map,
            ctx.delta_z,
            variant=self.variant,
        )
        clipped, _ = clip_gradient_global_norm(
            result.transported_grad, self.generator.cfg.grad_clip
        )
        u_tilde = self.generator.delta_from_transported_gradient(
            clipped,
            parameter=ctx.phi_active,
            valid=candidate.numerical_ok,
        )
        kappa = (
            1.0
            if ctx.delay_rounds == 0
            else damping_factor(
                mism, self.artifact.damping_radius, self.artifact.damping_kernel
            )
        )
        return Decision(
            kind=DecisionKind.TRANSPORT,
            published_delta=kappa * u_tilde,
            damping_factor=kappa,
            predicted_utility=util,
            predicted_mismatch=mism,
            predicted_harm_probability=harm,
            transport_rank=(
                self.artifact.transport_map.rank if self.artifact.transport_map else None
            ),
            parameter_comp_norm=result.parameter_comp_norm,
            state_transport_norm=result.state_transport_norm,
            random_transport=result.random_transport,
        )


# ---------------------------------------------------------------------------
# Diagnostic negative controls (spec 6.14): P0/P1/P3 mechanism figures only.
# ---------------------------------------------------------------------------


class RoundDiscardMethod(_ControllerMethod):
    """Discard when round delay exceeds a calibration-chosen D."""

    key = "round_discard"

    def __init__(self, shapes, basis, cfg, artifact):
        super().__init__(shapes, basis, cfg, artifact)
        self.delay_threshold = float(artifact.extra.get("round_discard_D", 5))

    def decide(self, candidate: CandidateUpdate, ctx: ArrivalContext) -> Decision:
        apply = ctx.delay_rounds <= self.delay_threshold
        return Decision(
            kind=DecisionKind.APPLY if apply else DecisionKind.DISCARD,
            published_delta=(
                candidate.candidate_delta
                if apply
                else torch.zeros_like(candidate.candidate_delta)
            ),
            threshold=self.delay_threshold,
        )


class WallDampMethod(_ControllerMethod):
    """Exponential damping with the same kernel shape as L2, driven purely
    by wall-clock delay."""

    key = "wall_damp"

    def __init__(self, shapes, basis, cfg, artifact):
        super().__init__(shapes, basis, cfg, artifact)
        self.wall_radius = float(artifact.extra.get("wall_damp_radius_us", 1e6))

    def decide(self, candidate: CandidateUpdate, ctx: ArrivalContext) -> Decision:
        kappa = damping_factor(ctx.delay_wall_us, self.wall_radius, "exponential")
        if ctx.delay_rounds == 0:
            kappa = 1.0
        return Decision(
            kind=DecisionKind.DAMP,
            published_delta=kappa * candidate.candidate_delta,
            damping_factor=kappa,
        )


class EndpointGateMethod(LCGateMethod):
    """L1 with endpoint distance replacing rho (feature_set='endpoint');
    everything else identical to the gate."""

    key = "endpoint_gate"


class ParameterOnlyMethod(LCTransportMethod):
    """L3 keeping only diag(F) s_phi (DC-ASGD-style compensation)."""

    key = "parameter_only"

    def __init__(self, shapes, basis, cfg, artifact):
        super().__init__(shapes, basis, cfg, artifact, variant="parameter_only")


class RandomTransportMethod(LCTransportMethod):
    """Same rank / same kernels as L3 but a random orthonormal basis, to
    rule out capacity/compute artifacts."""

    key = "random_transport"

    def __init__(self, shapes, basis, cfg, artifact):
        super().__init__(shapes, basis, cfg, artifact, variant="random")
