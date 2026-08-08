"""CPU reference engine.

Drives the complete request lifecycle (spec 4) against the toy model
pair: version-locked canvases, target-exact acceptance with Philox
substreams, trigger-round teacher signals, side-stream candidates with
logical delay, publish policies per method, trajectory recording,
controller decisions, and artifact row emission matching the spec 11
schemas.

The GPU path replaces the toy model with the SGLang fork via
`sglang_bridge`; the lifecycle, version rules and row schemas are
identical, which is what makes the CPU engine a reference
implementation rather than a separate system.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from lightcone_spec.adapters.losses import confidence_soft_targets
from lightcone_spec.exit_codes import ExactnessViolation, NumericalFailure
from lightcone_spec.methods.base import (
    ArrivalContext,
    CandidateUpdate,
    Decision,
    DecisionKind,
    MethodRuntime,
    PublishPolicy,
    TeacherSignal,
    apply_delta_with_trust_region,
    evaluate_loss_and_grad,
)
from lightcone_spec.runtime.canary import CanaryCounter
from lightcone_spec.runtime.canvas import Canvas
from lightcone_spec.runtime.double_buffer import (
    DoubleBufferStore,
    PendingUpdate,
    ReadyEvent,
)
from lightcone_spec.runtime.events import UpdateEventChain, monotonic_us
from lightcone_spec.runtime.exact_sampler import (
    acceptance_probability,
    normalize_sampling_dist,
    residual_distribution,
)
from lightcone_spec.runtime.rng import (
    DrawKind,
    categorical_draw,
    request_id_hash,
    substream_id,
    uniform_draw,
)
from lightcone_spec.runtime.toy_model import ToyModelPair
from lightcone_spec.runtime.versions import RequestVersionState
from lightcone_spec.trajectory.distance import (
    DistanceWeights,
    d_z,
    endpoint_distance as traj_endpoint,
    rho_path as traj_rho,
)
from lightcone_spec.trajectory.state import TrajectoryState, make_state
from lightcone_spec.trajectory.zvector import ZVectorizer, default_zvectorizer


def _us() -> float:
    return time.perf_counter_ns() / 1000.0


@dataclass
class EngineConfig:
    method_key: str
    seed: int = 0
    update_stride: int = 10
    logical_delay_rounds: int = 0
    update_latency_rounds: int = 1
    max_rounds: int = 32
    max_new_tokens: int = 64
    draft_depth: int = 3
    temperature: float = 1.0
    top_p: float = 1.0
    trust_region_radius: float = 1.0
    lifecycle: str = "request"
    max_in_flight: int = 1
    # Deliberate canary injection (exactness harness only).
    inject_version_race: bool = False
    # Delay-twin profile hooks (spec 12.3).
    idle_dilation: int = 0
    extra_wall_us: float = 0.0


@dataclass
class ScheduledUpdate:
    candidate: CandidateUpdate
    pending: PendingUpdate
    ready_round: int
    source_prefix_len: int
    phi_source: torch.Tensor


@dataclass
class RequestResult:
    request_id: str
    status: str
    committed_tokens: list[int]
    rounds_rows: list[dict]
    updates_rows: list[dict]
    decisions_rows: list[dict]
    summary_row: dict
    canaries: CanaryCounter
    trajectory_states: list[TrajectoryState]
    method_telemetry: dict


class ReferenceEngine:
    def __init__(
        self,
        pair: ToyModelPair,
        method: MethodRuntime,
        cfg: EngineConfig,
        distance_weights: Optional[DistanceWeights] = None,
        zvectorizer: Optional[ZVectorizer] = None,
        tenant_id_hash: str = "tenant-0",
        stream_id: Optional[str] = None,
    ):
        self.pair = pair
        self.method = method
        self.cfg = cfg
        self.weights = distance_weights or DistanceWeights(
            a_p=1 / 3, a_h=1 / 3, a_e=1 / 3
        )
        self.zvec = zvectorizer or default_zvectorizer()
        self.tenant_id_hash = tenant_id_hash
        self.stream_id = stream_id
        num_params = pair.shapes.num_params()
        self.store = DoubleBufferStore(num_params, max_in_flight=cfg.max_in_flight)
        self.phi0 = torch.zeros(num_params, dtype=torch.float32)
        self._epoch_counter = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _proposal_dist(
        self, prefix: tuple[int, ...], round_id: int, phi: torch.Tensor, pos: int
    ) -> np.ndarray:
        logits = self.pair.proposal_logits(prefix, round_id, phi, pos)
        return normalize_sampling_dist(logits, self.cfg.temperature, self.cfg.top_p)

    def _target_dist(self, prefix: tuple[int, ...], round_id: int) -> np.ndarray:
        logits = self.pair.target_logits(prefix, round_id)
        return normalize_sampling_dist(logits, self.cfg.temperature, self.cfg.top_p)

    def _make_teacher_signal(
        self,
        canvas: Canvas,
        prefixes: list[tuple[int, ...]],
        round_id: int,
        phi_source: torch.Tensor,
        source_version: int,
    ) -> TeacherSignal:
        """Full-window verification signal (spec 5.2): every draft position
        of the trigger round supervises, never accepted-only."""
        k = len(canvas.draft_tokens)
        if k == 0:
            raise NumericalFailure("empty supervision window at trigger round")
        u = np.stack([self.pair.projected_hidden(p) for p in prefixes[:k]])
        m = np.stack([self.pair.markov_embedding(p) for p in prefixes[:k]])
        base_logits = np.stack(
            [
                self.pair.base_drafter_logits(prefixes[i], round_id)
                for i in range(k)
            ]
        )
        target_logits = np.stack(
            [self.pair.target_logits(prefixes[i], round_id) for i in range(k)]
        )
        base_conf = np.asarray(
            [self.pair.base_confidence_logit(i) for i in range(k)]
        )
        source_logits = np.stack(
            [
                self.pair.proposal_logits(prefixes[i], round_id, phi_source, i)
                for i in range(k)
            ]
        )
        t_target = torch.from_numpy(target_logits.astype(np.float32))
        t_source = torch.from_numpy(source_logits.astype(np.float32))
        conf_targets = confidence_soft_targets(t_target, t_source)
        return TeacherSignal(
            source_round=round_id,
            source_version=source_version,
            u=torch.from_numpy(u.astype(np.float32)),
            m_prev=torch.from_numpy(m.astype(np.float32)),
            base_proposal_logits=torch.from_numpy(base_logits.astype(np.float32)),
            base_confidence_logits=torch.from_numpy(base_conf.astype(np.float32)),
            target_logits=t_target,
            valid_mask=torch.ones(k, dtype=torch.bool),
            source_proposal_logits=t_source,
            confidence_targets=conf_targets,
        )

    def _fresh_gradient(
        self,
        prefix: tuple[int, ...],
        round_id: int,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """Oracle-Current support: fresh teacher signal + gradient on the
        arrival prefix (extra target/backward cost recorded by caller)."""
        depth = self.cfg.draft_depth
        prefixes = [prefix]
        cur = prefix
        # Teacher-forced expansion along the target's top tokens.
        for _ in range(depth - 1):
            p = self._target_dist(cur, round_id)
            cur = cur + (int(np.argmax(p)),)
            prefixes.append(cur)
        k = len(prefixes)
        u = np.stack([self.pair.projected_hidden(p) for p in prefixes])
        m = np.stack([self.pair.markov_embedding(p) for p in prefixes])
        base_logits = np.stack(
            [self.pair.base_drafter_logits(p, round_id) for p in prefixes]
        )
        target_logits = np.stack(
            [self.pair.target_logits(p, round_id) for p in prefixes]
        )
        base_conf = np.asarray([self.pair.base_confidence_logit(i) for i in range(k)])
        cur_logits = np.stack(
            [
                self.pair.proposal_logits(prefixes[i], round_id, phi, i)
                for i in range(k)
            ]
        )
        t_target = torch.from_numpy(target_logits.astype(np.float32))
        t_cur = torch.from_numpy(cur_logits.astype(np.float32))
        signal = TeacherSignal(
            source_round=round_id,
            source_version=self.store.active_version,
            u=torch.from_numpy(u.astype(np.float32)),
            m_prev=torch.from_numpy(m.astype(np.float32)),
            base_proposal_logits=torch.from_numpy(base_logits.astype(np.float32)),
            base_confidence_logits=torch.from_numpy(base_conf.astype(np.float32)),
            target_logits=t_target,
            valid_mask=torch.ones(k, dtype=torch.bool),
            source_proposal_logits=t_cur,
            confidence_targets=confidence_soft_targets(t_target, t_cur),
        )
        _, grad = evaluate_loss_and_grad(
            phi, signal, self.pair.shapes, self.pair.basis
        )
        assert grad is not None
        return grad

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run_request(self, request_id: str, run_meta: dict | None = None) -> RequestResult:
        cfg = self.cfg
        self._epoch_counter += 1
        rid_hash = request_id_hash(request_id)
        version_state = RequestVersionState(
            request_id=request_id,
            tenant_id_hash=self.tenant_id_hash,
            stream_id=self.stream_id,
            request_epoch=self._epoch_counter,
        )
        canaries = CanaryCounter()
        if cfg.lifecycle == "request":
            # Reset adapter, optimizer, controller history, trace.
            self.store = DoubleBufferStore(
                self.pair.shapes.num_params(), max_in_flight=cfg.max_in_flight
            )
        committed: list[int] = []
        rounds_rows: list[dict] = []
        updates_rows: list[dict] = []
        decisions_rows: list[dict] = []
        traj_states: list[TrajectoryState] = []
        scheduled: list[ScheduledUpdate] = []
        chains: dict[str, UpdateEventChain] = {}
        source_prefix_lens: dict[str, int] = {}
        phi_sources: dict[str, torch.Tensor] = {}
        status = "complete_valid"
        request_start_us = _us()
        first_decode_us: Optional[float] = None
        target_calls_total = 0
        version_mismatch_count = 0
        update_id_seq = 0

        meta = run_meta or {}

        try:
            for round_id in range(cfg.max_rounds):
                if len(committed) >= cfg.max_new_tokens:
                    break
                prefix = tuple(committed)
                round_cpu_start = _us()

                # ---- graph boundary: poll ready updates, publish -------
                controller_cpu_us = 0.0
                for sched in list(scheduled):
                    if sched.ready_round > round_id:
                        continue
                    cand = sched.candidate
                    chain = chains[cand.update_id]
                    ctrl_start = _us()
                    ctx = self._arrival_context(
                        cand, sched, round_id, traj_states, prefix
                    )
                    decision = self.method.decide(cand, ctx)
                    controller_cpu_us += _us() - ctrl_start
                    self._execute_decision(
                        cand, sched, decision, chain, round_id, version_state
                    )
                    decisions_rows.append(
                        self._decision_row(cand, ctx, decision, controller_cpu_us, meta)
                    )
                    updates_rows.append(
                        self._update_row(
                            cand,
                            chain,
                            decision,
                            len(committed),
                            sched.source_prefix_len,
                            meta,
                        )
                    )
                    if decision.kind == DecisionKind.VERSION_CONFLICT:
                        version_mismatch_count += 1
                    scheduled.remove(sched)

                # ---- draft: version-locked canvas ------------------------
                if first_decode_us is None:
                    first_decode_us = _us()
                draft_start = _us()
                self.store.begin_replay()
                phi_active = self.store.read_active()
                proposal_version = self.store.active_version
                version_state.proposal_version = proposal_version
                drafts: list[int] = []
                proposal_probs: list[np.ndarray] = []
                sub_ids: list[str] = []
                cur = prefix
                remaining = cfg.max_new_tokens - len(committed)
                gamma = min(cfg.draft_depth, max(remaining - 1, 0))
                for k in range(gamma):
                    q = self._proposal_dist(cur, round_id, phi_active, k)
                    tok = categorical_draw(
                        q, cfg.seed, rid_hash, round_id, k, DrawKind.PROPOSAL
                    )
                    drafts.append(tok)
                    proposal_probs.append(q)
                    sub_ids.append(
                        substream_id(cfg.seed, rid_hash, round_id, k, DrawKind.PROPOSAL)
                    )
                    cur = cur + (tok,)
                self.store.end_replay()
                denominator_version = proposal_version
                if cfg.inject_version_race and round_id % 2 == 1:
                    # Deliberate canary: denominator bound to a wrong version.
                    denominator_version = proposal_version + 1
                canvas = Canvas(
                    request_id=request_id,
                    round_id=round_id,
                    proposal_version=proposal_version,
                    denominator_version=denominator_version,
                    residual_version=denominator_version,
                    draft_tokens=drafts,
                    proposal_probs=proposal_probs,
                    confidence_logits=np.asarray(
                        [self.pair.base_confidence_logit(k) for k in range(gamma)]
                    ),
                    rng_substream_ids=sub_ids,
                )
                draft_cpu_us = _us() - draft_start

                # ---- verify + accept (target-exact) ----------------------
                verify_start = _us()
                version_state.check_canvas_consistency(
                    canvas.proposal_version,
                    canvas.denominator_version,
                    canvas.residual_version,
                )
                canvas.consume()
                prefixes = [prefix]
                cur = prefix
                for tok in drafts:
                    cur = cur + (tok,)
                    prefixes.append(cur)
                round_committed: list[int] = []
                accepted = 0
                rejected = False
                cur = prefix
                for k, tok in enumerate(drafts):
                    p = self._target_dist(cur, round_id)
                    target_calls_total += 1
                    q = canvas.proposal_probs[k]
                    a = acceptance_probability(float(p[tok]), float(q[tok]))
                    u_draw = uniform_draw(
                        cfg.seed, rid_hash, round_id, k, DrawKind.ACCEPTANCE
                    )
                    if u_draw <= a:
                        round_committed.append(tok)
                        accepted += 1
                        cur = cur + (tok,)
                    else:
                        r = residual_distribution(p, q)
                        rtok = categorical_draw(
                            r, cfg.seed, rid_hash, round_id, k, DrawKind.RESIDUAL
                        )
                        round_committed.append(rtok)
                        rejected = True
                        break
                if not rejected:
                    p = self._target_dist(cur, round_id)
                    target_calls_total += 1
                    btok = categorical_draw(
                        p, cfg.seed, rid_hash, round_id, gamma, DrawKind.BONUS
                    )
                    round_committed.append(btok)
                verify_cpu_us = _us() - verify_start

                # ---- trajectory state (target-side, reuses verify tensors)
                p_first = self._target_dist(prefix, round_id)
                z = make_state(
                    round_id=round_id,
                    target_probs=p_first,
                    hidden_projected=self.pair.projected_hidden(prefix),
                    topk=min(64, p_first.shape[0]),
                )
                traj_states.append(z)
                for _ in range(cfg.idle_dilation):
                    # Idle-insertion twin: identical repeated states.
                    traj_states.append(
                        make_state(
                            round_id=round_id,
                            target_probs=p_first,
                            hidden_projected=self.pair.projected_hidden(prefix),
                            topk=min(64, p_first.shape[0]),
                        )
                    )

                prev_len = len(committed)
                committed.extend(round_committed)

                rounds_rows.append(
                    {
                        **self._common_keys(request_id, meta),
                        "round_id": round_id,
                        "prefix_pos_before": prev_len,
                        "prefix_pos_after": len(committed),
                        "active_version": self.store.active_version,
                        "proposal_version": proposal_version,
                        "draft_tokens": gamma,
                        "accepted_drafts": accepted,
                        "committed_per_verify": len(round_committed),
                        "target_calls": len(drafts) + (0 if rejected else 1),
                        "draft_cpu_us": draft_cpu_us,
                        "draft_cuda_us": 0.0,
                        "verify_cpu_us": verify_cpu_us,
                        "verify_cuda_us": 0.0,
                        "accept_cuda_us": 0.0,
                        "target_topk_token_ids": z.topk_token_ids.tolist(),
                        "target_topk_probs": z.topk_probs.tolist(),
                        "target_other_mass": z.other_mass,
                        "proposal_topk_token_ids": (
                            np.argsort(-proposal_probs[0])[:64].astype(int).tolist()
                            if proposal_probs
                            else []
                        ),
                        "proposal_topk_probs": (
                            np.sort(proposal_probs[0])[::-1][:64].astype(float).tolist()
                            if proposal_probs
                            else []
                        ),
                        "proposal_other_mass": 0.0,
                        "hidden_proj": z.hidden_proj.tolist(),
                        "event_sketch": z.event_sketch.tolist(),
                        "endpoint_from_previous": (
                            d_z(traj_states[-2], z, self.weights)
                            if len(traj_states) >= 2
                            else 0.0
                        ),
                        "rng_substream_id": sub_ids[0] if sub_ids else "",
                        "version_canary_ok": canaries.total() == 0,
                    }
                )

                # ---- trigger update (spec 4.5: r mod S == 0) --------------
                if (
                    self.method.publish_policy != PublishPolicy.NONE
                    and round_id % cfg.update_stride == 0
                ):
                    if not self.store.can_launch():
                        # max_in_flight budget exhausted; skipped signals are
                        # never accumulated (spec 6.3).
                        pass
                    elif len(canvas.draft_tokens) == 0:
                        # No draft positions this round (e.g. token budget hit
                        # before drafting): no supervision window, skip the
                        # trigger rather than failing the request.
                        pass
                    else:
                        chain = UpdateEventChain(
                            update_id=f"{request_id}-u{update_id_seq}",
                            source_round=round_id,
                            source_version=proposal_version,
                        )
                        update_id_seq += 1
                        chain.mark("snapshot")
                        phi_source = phi_active.clone()
                        signal = self._make_teacher_signal(
                            canvas, prefixes, round_id, phi_source, proposal_version
                        )
                        chain.mark("teacher")
                        try:
                            cand = self.method.make_candidate(phi_source, signal)
                        except NumericalFailure:
                            cand = None
                        if cand is not None:
                            cand.update_id = chain.update_id
                            chains[chain.update_id] = chain
                            source_prefix_lens[chain.update_id] = len(committed)
                            phi_sources[chain.update_id] = phi_source
                            pending = PendingUpdate(
                                update_id=chain.update_id,
                                source_round=round_id,
                                source_version=proposal_version,
                                candidate_delta=cand.candidate_delta,
                                raw_gradient=cand.raw_gradient,
                                events=chain,
                                ready=ReadyEvent(event_id=chain.done_event_id),
                                source_snapshot=(
                                    phi_source.clone()
                                    if cfg.max_in_flight == 2
                                    else None
                                ),
                                source_optimizer_state=(
                                    {} if cfg.max_in_flight == 2 else None
                                ),
                                version_ancestry=(proposal_version,),
                            )
                            self.store.launch(pending)
                            self.store.write_staging(pending, phi_source + cand.candidate_delta)
                            ready_round = self._ready_round(round_id)
                            scheduled.append(
                                ScheduledUpdate(
                                    candidate=cand,
                                    pending=pending,
                                    ready_round=ready_round,
                                    source_prefix_len=len(committed),
                                    phi_source=phi_source,
                                )
                            )
                canvas.release()
        except ExactnessViolation as exc:
            status = "failed_exactness"
            canaries.record("version_mismatch", str(exc))
        except NumericalFailure:
            status = "failed_runtime"

        end_us = _us()
        decode_wall_s = max(end_us - (first_decode_us or end_us), 0.0) / 1e6
        e2e_wall_s = max(end_us - request_start_us, 0.0) / 1e6 + (
            self.cfg.extra_wall_us / 1e6
        )
        n_rounds = max(len(rounds_rows), 1)
        round_ms = [
            (r["draft_cpu_us"] + r["verify_cpu_us"]) / 1000.0 for r in rounds_rows
        ] or [0.0]
        itl_ms = [
            latency / max(int(row["committed_per_verify"]), 1)
            for latency, row in zip(round_ms, rounds_rows)
        ] or [0.0]
        summary = {
            **self._common_keys(request_id, meta),
            "prompt_id_hash": request_id_hash(request_id),
            "task_type": meta.get("task_type", "synthetic"),
            "output_tokens": len(committed),
            "quality_metric_name": meta.get("quality_metric_name", "none"),
            "quality_value": meta.get("quality_value"),
            "decode_wall_s": decode_wall_s if status == "complete_valid" else 0.0,
            "e2e_wall_s": e2e_wall_s if status == "complete_valid" else 0.0,
            "decode_tps": (
                len(committed) / decode_wall_s
                if status == "complete_valid" and decode_wall_s > 0
                else 0.0
            ),
            "e2e_tps": (
                len(committed) / e2e_wall_s
                if status == "complete_valid" and e2e_wall_s > 0
                else 0.0
            ),
            "goodput_tps": (
                len(committed) / decode_wall_s
                if status == "complete_valid" and decode_wall_s > 0
                else 0.0
            ),
            "offered_concurrency": int(meta.get("offered_concurrency", 1)),
            "ttft_ms": (
                max((first_decode_us or request_start_us) - request_start_us, 0.0)
                / 1000.0
            ),
            "queue_ms": 0.0,
            "estimated_perf_scope": "unavailable_toy_runtime",
            "estimated_tflops_per_gpu": None,
            "estimated_mfu": None,
            "estimated_read_gbps_per_gpu": None,
            "estimated_write_gbps_per_gpu": None,
            "peak_tflops_per_gpu": None,
            "adaptation_fallback_count": 0,
            "kv_retracted_requests": 0,
            "peak_running_requests": int(meta.get("offered_concurrency", 1)),
            "peak_queue_requests": 0,
            "model_weight_hbm_bytes": 0,
            "kv_cache_hbm_bytes": 0,
            "cuda_graph_hbm_bytes": 0,
            "kv_token_capacity": 0,
            "adaptation_fixed_bytes": 0,
            "adaptation_reserve_bytes": 0,
            "mean_accepted_drafts": float(
                np.mean([r["accepted_drafts"] for r in rounds_rows]) if rounds_rows else 0.0
            ),
            "mean_committed_per_verify": float(
                np.mean([r["committed_per_verify"] for r in rounds_rows])
                if rounds_rows
                else 0.0
            ),
            "target_calls_per_output_token": (
                target_calls_total / len(committed) if committed else 0.0
            ),
            "p50_round_ms": float(np.percentile(round_ms, 50)),
            "p95_round_ms": float(np.percentile(round_ms, 95)),
            "p99_round_ms": float(np.percentile(round_ms, 99)),
            "p50_itl_ms": float(np.percentile(itl_ms, 50)),
            "p95_itl_ms": float(np.percentile(itl_ms, 95)),
            "p99_itl_ms": float(np.percentile(itl_ms, 99)),
            "energy_per_token_j": None,
            "peak_hbm_bytes": 0,
            "version_mismatch_count": version_mismatch_count + canaries.total(),
            "status": status,
        }
        if status != "complete_valid":
            # No throughput summaries for failed-exactness units (spec 3.3).
            for k in ("decode_tps", "e2e_tps", "goodput_tps"):
                summary[k] = 0.0
        return RequestResult(
            request_id=request_id,
            status=status,
            committed_tokens=committed,
            rounds_rows=rounds_rows,
            updates_rows=updates_rows,
            decisions_rows=decisions_rows,
            summary_row=summary,
            canaries=canaries,
            trajectory_states=traj_states,
            method_telemetry=self.method.telemetry(),
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _ready_round(self, trigger_round: int) -> int:
        cfg = self.cfg
        if self.method.publish_policy == PublishPolicy.BLOCKING:
            return trigger_round + 1  # c(r) = r + 1
        if self.method.publish_policy == PublishPolicy.FIXED_STRIDE_BARRIER:
            return trigger_round + cfg.update_stride
        return trigger_round + max(cfg.update_latency_rounds, 1) + cfg.logical_delay_rounds

    def _arrival_context(
        self,
        cand: CandidateUpdate,
        sched: ScheduledUpdate,
        arrival_round: int,
        traj_states: list[TrajectoryState],
        prefix: tuple[int, ...],
    ) -> ArrivalContext:
        source_round = cand.source_round
        by_round = {s.round_id: s for s in traj_states}
        have_both = source_round in by_round and (arrival_round - 1) in by_round
        # Exposure at the canvas of `arrival_round`; states cover rounds up
        # to arrival_round - 1, so rho integrates through the last recorded
        # state (the arrival round's own state is recorded after drafting).
        end_round = min(arrival_round, max(by_round) if by_round else source_round)
        if have_both or (source_round in by_round and end_round in by_round):
            rho = traj_rho(traj_states, source_round, end_round, self.weights)
            endp = traj_endpoint(traj_states, source_round, end_round, self.weights)
            delta_z = torch.from_numpy(
                self.zvec.delta_z(by_round[source_round], by_round[end_round]).astype(
                    np.float32
                )
            )
        else:
            rho, endp, delta_z = 0.0, 0.0, None
        phi_active = self.store.read_active()
        disp = float(torch.linalg.vector_norm(phi_active - sched.phi_source))
        fresh_grad = None
        if self.method.needs_fresh_gradient_at_arrival:
            fresh_grad = self._fresh_gradient(prefix, arrival_round, phi_active)
        return ArrivalContext(
            arrival_round=arrival_round,
            active_version=self.store.active_version,
            phi_active=phi_active,
            delay_rounds=arrival_round - source_round,
            delay_tokens=len(prefix) - sched.source_prefix_len,
            delay_wall_us=float(
                (arrival_round - source_round) * 1000.0 + self.cfg.extra_wall_us
            ),
            delay_versions=self.store.active_version - cand.source_version,
            rho_path=rho,
            endpoint_distance=endp,
            parameter_displacement=disp,
            delta_z=delta_z,
            fresh_gradient=fresh_grad,
        )

    def _execute_decision(
        self,
        cand: CandidateUpdate,
        sched: ScheduledUpdate,
        decision: Decision,
        chain: UpdateEventChain,
        round_id: int,
        version_state: RequestVersionState,
    ) -> None:
        pending = sched.pending
        if decision.kind == DecisionKind.VERSION_CONFLICT or (
            torch.count_nonzero(decision.published_delta) == 0
            and decision.kind == DecisionKind.DISCARD
        ):
            self.store.discard(pending)
            return
        phi_active = self.store.read_active()
        new_params = apply_delta_with_trust_region(
            phi_active,
            decision.published_delta,
            self.phi0,
            self.cfg.trust_region_radius,
        )
        chain.apply_round = round_id
        new_version = self.store.publish(pending, new_params)
        chain.exposure_round = round_id
        chain.mark("exposure")
        version_state.source_version = cand.source_version
        version_state.active_version = new_version
        chain.validate(allow_incomplete=True)

    def _common_keys(self, request_id: str, meta: dict) -> dict:
        return {
            "schema_version": 1,
            "run_id": meta.get("run_id", "local"),
            "unit_id": meta.get("unit_id", "local"),
            "request_id": request_id,
            "stream_id": self.stream_id,
            "tenant_id_hash": self.tenant_id_hash,
            "model_pair_id": meta.get("model_pair_id", "toy_markov4"),
            "method": self.method.key,
            "dataset": meta.get("dataset", "markov4_world"),
            "seed": self.cfg.seed,
            "lifecycle": self.cfg.lifecycle,
        }

    def _update_row(
        self,
        cand: CandidateUpdate,
        chain: UpdateEventChain,
        decision: Decision,
        committed_len: int,
        source_prefix_len: int,
        meta: dict,
    ) -> dict:
        applied = decision.kind in (
            DecisionKind.APPLY,
            DecisionKind.DAMP,
            DecisionKind.TRANSPORT,
        ) and torch.count_nonzero(decision.published_delta) > 0
        return {
            **self._common_keys(chain.update_id.rsplit("-u", 1)[0], meta),
            "update_id": chain.update_id,
            "source_round": chain.source_round,
            "apply_round": chain.apply_round,
            "exposure_round": chain.exposure_round if applied else None,
            "source_version": cand.source_version,
            "source_training_loss": float(cand.loss.total),
            "source_expected_accepted_prefix": float(
                cand.loss.expected_accepted_prefix
            ),
            "source_prefix_len": int(source_prefix_len),
            "active_version_at_arrival": self.store.active_version
            - (1 if applied else 0),
            "staging_version": cand.source_version + 1,
            "published_version": self.store.active_version if applied else None,
            "delay_rounds": (chain.apply_round or chain.source_round)
            - chain.source_round,
            "delay_tokens": committed_len,
            "delay_wall_us": float(
                ((chain.commit_ts_us or chain.snapshot_ts_us or 0.0))
                - (chain.snapshot_ts_us or 0.0)
            ),
            "delay_versions": max(
                self.store.active_version - cand.source_version - (1 if applied else 0),
                0,
            ),
            "snapshot_ts_us": chain.snapshot_ts_us,
            "teacher_ts_us": chain.teacher_ts_us,
            "launch_ts_us": chain.launch_ts_us,
            "done_ts_us": chain.done_ts_us,
            "commit_ts_us": chain.commit_ts_us,
            "exposure_ts_us": chain.exposure_ts_us,
            "launch_event_id": chain.launch_event_id,
            "done_event_id": chain.done_event_id,
            "commit_event_id": chain.commit_event_id,
            "grad_norm": cand.grad_norm,
            "grad_clip_scale": cand.grad_clip_scale,
            "grad_sketch": cand.raw_gradient[:16].tolist(),
            "candidate_delta_norm": float(
                torch.linalg.vector_norm(cand.candidate_delta)
            ),
            "side_queue_cuda_us": 0.0,
            "candidate_cuda_us": 0.0,
            "barrier_wait_cpu_us": 0.0,
            "publish_cuda_us": 0.0,
            "optimizer_step": cand.optimizer_step,
            "numerical_ok": cand.numerical_ok,
            "failure_reason": cand.failure_reason,
        }

    def _decision_row(
        self,
        cand: CandidateUpdate,
        ctx: ArrivalContext,
        decision: Decision,
        controller_cpu_us: float,
        meta: dict,
    ) -> dict:
        return {
            **self._common_keys(cand.update_id.rsplit("-u", 1)[0], meta),
            "update_id": cand.update_id,
            "rho_path": ctx.rho_path,
            "endpoint_distance": ctx.endpoint_distance,
            "parameter_displacement": ctx.parameter_displacement,
            "predicted_utility": decision.predicted_utility,
            "predicted_mismatch": decision.predicted_mismatch,
            "predicted_harm_probability": decision.predicted_harm_probability,
            "threshold": decision.threshold,
            "decision": decision.kind.value,
            "damping_factor": decision.damping_factor,
            "transport_rank": decision.transport_rank,
            "parameter_comp_norm": decision.parameter_comp_norm,
            "state_transport_norm": decision.state_transport_norm,
            "random_transport": decision.random_transport,
            "controller_cpu_us": controller_cpu_us,
            "controller_cuda_us": 0.0,
        }
