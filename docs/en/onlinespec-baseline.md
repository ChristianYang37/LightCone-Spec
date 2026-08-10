# OnlineSPEC baseline

[中文](../zh-CN/onlinespec-baseline.md) · [Home](../../README.md)

## Status and boundary

OnlineSPEC is an important registered comparison baseline, not part of the
Static/TTS/L0 hypothesis or its formal speed gate. It has its own tuning data,
selection artifact, paired confirmation queue, performance table, GPU
attestation, and diagnostic analysis. Its result cannot select a core method or
change a core `PASS`, `BLOCKED`, or `UNMEASURED` decision.

The implementation is clean-room and targets the paper's test-time drafter
update abstraction. It is based on the [OnlineSPEC
paper](https://arxiv.org/abs/2603.12617v2) and records the [official repository
at commit
`e58f82e`](https://github.com/ZinYY/OnlineSPEC/tree/e58f82eb3f3adca3a686211236bf4f6e9e7e3a2b).
The audited repository had no project-level license file at that commit; some
individual files carry their own notices. No source file, checkpoint utility,
or training script was copied into LightCone-Spec.

The machine-readable audit is
[`manifests/provenance/onlinespec_source_audit_v2.json`](../../manifests/provenance/onlinespec_source_audit_v2.json).
It binds the upstream commit and Git tree, the SHA-256 of every implementation
file used in this comparison, the three architecture-specific pipelines, and
the adopted/rejected design decisions. The OnlineSPEC study manifest binds the
audit artifact by content hash, so changing the source interpretation creates
a different experiment identity.

Online-LR's reasoning-level DPO pipeline is not presented as a token-level
drafter baseline. LightCone-Spec implements the three online-learning rules
that can be compared under the same speculative-decoding feedback and
exactness contract: OGD, optimistic OGD, and Hedge over OGD experts.

GPU status is `UNMEASURED`. This page describes protocol and implementation,
not a performance result.

## Source audit

| Upstream idea or implementation detail | LightCone-Spec decision |
|---|---|
| Predict on a chunk, observe feedback, then update for the next chunk | Retained as a strict prequential lifecycle |
| OGD learner | Reimplemented from the published projected-gradient equation |
| Optimistic online learning | Implemented as the paper's two-state projected update, not momentum |
| Ensemble of learners with different learning rates | Retained as independent learners with cumulative-loss Hedge |
| Target-to-draft KL/CE supervision | Retained with the same semantic mask and frozen target used by the core runtime |
| Per-chunk subprocess training and disk checkpoints | Replaced by GPU-resident transactional candidates and fixed-address publication |
| CPU checkpoint averaging for ensemble inference | Replaced by a device-resident weighted full-parameter decision |
| Hard-coded devices and dataset paths | Replaced by immutable schema, model locks, and rendered launch plans |
| Skip a failed or OOM update and continue the experiment | Rejected; safety failures are recorded and invalidate the affected evidence |
| Unaccounted synchronization and training time | Replaced by asynchronous CUDA timing, exposed-update accounting, and an independent profiler |

The pinned repository contains more than one EAGLE ensemble pipeline. The
paper-aligned `pipeline_ens.py` and `pipeline_eagle3_ens.py` keep independent
base learners and form the next ensemble from cumulative losses. The older
`pipeline_hedge.py` variants instead weight only the preceding chunk loss and
reset every learner from the current meta checkpoint before training. That is
not cumulative-loss Hedge and is not reproduced here. At the audited commit,
the README examples and both `eagle-ens.sh` entrypoints still invoke these
older `*_hedge.py` variants rather than the cumulative `*_ens.py` files.
Likewise, the README calls momentum “optimism” and the Hydra source implements
momentum SGD, while the paper specifies the two-state historical-gradient-hint
update. LightCone-Spec implements the published state transition and does not
relabel momentum as optimistic online learning.

Several paper recipes and pinned-source defaults also disagree. The appendix
states a global Online-LR batch of 16 while `LR/pipeline.py` launches 12; it
describes EAGLE updates as Adam while `EAGLE/train.py` constructs AdamW; and it
sets the Hedge meta rate to 10 while the paper-aligned source pipeline defaults
to 0.2 on a differently scaled accumulated loss. These are provenance facts,
not hyperparameters that can be silently chosen after seeing confirmation
data. LightCone-Spec registers and tunes its normalized-loss rates only on the
tuning window.

### The three upstream instantiations are not interchangeable

The official project is a workspace containing three architecture-specific
systems, rather than one drop-in optimizer. The table below records the full
test-time-update boundary audited at the pinned commit.

| Instantiation | Prediction object | Feedback and update recipe in the pinned project | Online unit and publication | LightCone-Spec treatment |
|---|---|---|---|---|
| Online-LR | A standalone reasoning draft LM used by Lookahead Reasoning | Verifier-derived chosen/rejected responses; DPO with AdamW, β=(0.9, 0.95), learning rate 5e-7, DPO β=0.1, and three epochs. The launcher uses global batch 12 and micro batch 2. | 25-example chunks; a DeepSpeed subprocess rewrites a model directory. | The reasoning-level preference pipeline is documented but not relabelled as token-level OGD. It requires a separate judge, data, and memory protocol and is outside this registered baseline. |
| Opt-Hydra | A multi-head Hydra draft head | Feature reconstruction plus teacher/token losses. The released “optimistic” path uses SGD momentum 0.9, learning rate 0.1, and three epochs, rather than the paper's two-state transition. | 80-example chunks; trainer checkpoints and optimizer state are loaded from disk between chunks. | Implements the published two-state optimistic learner on the supported drafter parameters. Momentum is not treated as mathematical optimism. |
| Ens-EAGLE / EAGLE3 | Three independent EAGLE or EAGLE3 draft heads | Each learner is trained at a different rate. Registered scripts use EAGLE rates 3e-5/6e-5/1.2e-4 for five epochs and EAGLE3 rates 1e-4/2e-4/4e-4 for two epochs; source variants disagree about cumulative versus last-chunk weighting. | 40-example chunks; checkpoints are loaded and merged on CPU before evaluation. | Keeps independent projected-OGD experts and cumulative-loss Hedge, but performs the update and full-parameter merge on device. Expert backward passes are streamed sequentially so only one expert-gradient scratch is live. |

“Complete” therefore has a precise meaning here: all three published online
learner transitions used for a common speculative-decoding comparison—OGD,
two-state optimistic OGD, and cumulative-loss Hedge—have complete schema,
runtime, tuning, confirmation, telemetry, and safety implementations. It does
not mean that Online-LR, Hydra, and EAGLE have been made architecture-identical,
because doing so would change the model pair, supervision, and systems budget
rather than isolate the online learner.

### Paper, pinned source, and LightCone-Spec

This table is the implementation-level comparison used to decide what was
retained. It is intentionally about test-time drafter updates, not reported
benchmark values.

| Concern | Paper | Pinned official source | LightCone-Spec |
|---|---|---|---|
| Online unit | One predict-feedback-update round | Dataset chunks: EAGLE 40, Hydra 80, reasoning 25; maximum sequence length 2,048 | One versioned speculative update round; stride is explicit and tuned |
| OGD state | One projected gradient step | Online-LR uses AdamW; EAGLE training also uses AdamW | Exact projected SGD state, one transactional proposal per update |
| Optimism | Anchor plus previous-gradient hint | Hydra uses SGD with momentum 0.9 | Exact anchor/hint transition from the paper; hint error is measured |
| Ensemble | Independent OGD experts and cumulative-loss exponential weighting | The `*_ens.py` path follows this structure; older `*_hedge.py` files do not | Independent GPU-resident experts, per-expert loss and gradient, cumulative-loss Hedge |
| EAGLE objective | Cross entropy is the theoretical loss | Feature SmoothL1 plus token-distribution loss, with AdamW and multiple epochs | Target-to-draft KL/CE on the exact semantic mask so all backends share one feedback contract |
| Gradient clipping | Bounded-gradient assumption | EAGLE source uses value clipping; recipes vary by backend | Per-expert global-norm clipping, configured and evidence-bound |
| Parameter publication | Abstract next-round decision | Subprocess training and checkpoint replacement | Functional candidate followed by fixed-address atomic publication |
| Ensemble merge | Weighted parameter decision | CPU checkpoint load, average, and save | Device-resident weighted full-parameter decision; no disk or CPU merge |
| Historical KV | Not specified | No explicit version contract for pre-update KV | Frozen, detached, versioned history; only future KV uses a published version |
| Exact sampling | Target/draft likelihood-ratio verification | Most released evaluations use greedy decoding | The actual proposal distribution is retained for exact rejection sampling |
| Failure handling | Not specified | Some training paths skip invalid samples, NaNs, or OOM batches | Fail closed for the affected cohort and invalidate unsafe evidence |
| Timing | Acceleration includes online evolution | Per-question TPS; some scripts separately add train time | Paired clean-server goodput plus update/barrier/overlap/HBM evidence |

The numeric values of optimizer and Hedge meta learning rates are not copied
from the source recipe. Its accumulated epoch loss has a different scale from
the normalized per-round loss used here, so copying its epsilon would not
preserve the same exponential weights. All learner and meta rates are selected
only on the registered tuning window.

Because every OnlineSPEC gradient is globally clipped, its learning rate is a
parameter-space displacement scale rather than an Adam-style coordinate step.
The registered OGD/optimistic grid therefore uses rates
`1e-4, 1e-3, 1e-2, 1e-1` and strides `20, 40, 80, 160`. Hedge uses ordered
triples starting at `1e-4`, `1e-3`, or `1e-2`, with strides `40, 80, 160`.
These are
tuning-only protocol bounds, not reported best settings.

## Online learners

Let $K$ be the permitted parameter set, Π its Euclidean projection, $w_t$ the
decision used for proposal round $t$, and $g_t=\nabla\ell_t(w_t)$ the loss
gradient revealed by verification. In the executable contract, each learner
independently applies the configured global-norm clip to $g_t$ before the
transition below.

Projected OGD uses

\[
w_{t+1}=\Pi_K(w_t-\eta g_t).
\]

The optimistic learner keeps an anchor \(\hat w_t\) and a hint \(h_t\). Its
published decision and post-feedback transition are

\[
w_t=\Pi_K(\hat w_t-\eta h_t),\qquad
\hat w_{t+1}=\Pi_K(\hat w_t-\eta g_t),\qquad
h_{t+1}=g_t.
\]

The next decision is

\[
w_{t+1}=\Pi_K(\hat w_{t+1}-\eta h_{t+1}).
\]

For Hedge, expert $i$ owns its own parameters, gradient, learning rate
ηᵢ, and cumulative loss $L_{t,i}$. After every expert is evaluated and
updated at its own decision,

\[
p_{t+1,i}=\frac{\exp(-\gamma L_{t+1,i})}
{\sum_j\exp(-\gamma L_{t+1,j})},\qquad
w_{t+1}=\sum_i p_{t+1,i}w_{t+1,i}.
\]

Gradients are never reused across experts. Factor averaging is not equivalent
to parameter averaging, so Hedge is restricted to `weight_update_mode=full`.

Target-to-draft cross entropy and
\(D_{\mathrm{KL}}(p_{\mathrm{target}}\Vert q_{\mathrm{draft}})\) differ by the
target entropy, which is independent of the drafter parameters. They therefore
produce the same drafter gradient, and the common additive term does not change
Hedge probabilities.

## Runtime lifecycle

Every OnlineSPEC method follows one ordering:

1. Decode with the currently published decision and preserve the actual
   proposal distribution used to sample draft tokens.
2. Verification reveals target supervision and the semantic valid mask.
3. On an update round, compute the learner-specific candidate on the side CUDA
   stream. The candidate does not mutate active parameters or learner state.
4. At the next prediction boundary, validate cohort epoch, slot generation,
   source version, finite values, and readiness, then atomically commit or
   discard the complete candidate.
5. The next proposal uses the committed decision. Exact target rejection
   sampling continues to use the original recorded proposal distribution.

This boundary is deliberately different from TTS and L0 publication
scheduling. OnlineSPEC is neither silently mapped onto a TTS stride barrier nor
allowed to publish midway through the proposal it trained on.

Historical drafter KV is frozen and versioned exactly as in the core runtime.
The history is a detached state input; only the current canvas is
differentiable. Publishing new weights does not rewrite old KV. This is a
truncated-online baseline, and telemetry records the version used by every new
KV segment.

## Backend and parameter support

- DFlash supports drafter-scope `full` and `lora`, plus the shared tail
  ablations. Only update rounds run the differentiable current-canvas path;
  the whole homogeneous cohort is reconstructed as one batched SDPA/MLP graph,
  not as a Python loop over requests.
- DSpark and EAGLE/EAGLE3 use the cache-safe tail path and their existing
  backend restrictions. They do not pretend to provide drafter-wide gradients.
- OGD and optimistic OGD support Full and LoRA in the registered DFlash grid.
- Hedge supports Full only and uses at least two ordered expert learning rates.
- TP and DP must both equal one. Unsupported, quantized, or incompatible graph
  paths fail before OnlineSPEC state allocation.

OnlineSPEC owns plain projected SGD arithmetic. It does not inherit AdamW,
Muon, or another core optimizer through a configuration alias.

## Memory and telemetry

The HBM preflight includes active decisions, initial projection anchors,
candidate state, optimistic anchor and hint, all resident Hedge expert
parameters, cumulative losses, weighted-merge staging, one reusable
expert-gradient scratch, training activations, KV gather scratch, graph
buffers, and bounded telemetry. Hedge experts are differentiated one at a
time; each updated expert candidate is retained, while its differentiable
leaves and gradient are released before the next expert. This preserves the
same Hedge decision without reserving one full gradient copy per expert. These
tensors are non-evictable and are never silently offloaded or downgraded.
KV-pool sizing uses only the remaining memory.

Update telemetry records learner step, source and published versions, loss,
gradient norm, differentiable/inference-logit reconstruction diagnostics,
update and publication timing, and safety dispositions. Optimistic runs also
record hint error. Hedge records per-expert gradient norms, expert
probabilities, cumulative losses, ensemble entropy, and effective expert
count. Headline collection does not copy these values to the CPU per round;
diagnostics drain the bounded device buffer outside the measured hot path.

## Registered experiment

The tracked protocol is
`manifests/speed-study/onlinespec_baseline_v2.json`. It uses the same controlled
greedy sampling semantics and DFlash model pair as the core study. Greedy
confirmation keeps the target-token trajectory identical across learners;
paired output digests are verified before any timing comparison, while raw
generated text is not retained. Stochastic exactness is verified separately.
The protocol preserves
separate evidence identities:

1. Run successive-halving tuning on the tuning-only window. Halving occurs
   independently inside OGD, optimistic OGD, and Hedge so one learner cannot
   eliminate another.
2. Select one safe candidate per learner. The CLI requires the core
   Static/TTS/L0 selection, inherits its selected concurrency, and recursively
   binds its SHA-256. The OnlineSPEC selection also binds the complete terminal
   tuning artifact, model lock, sampling profile, manifest, and patched SGLang
   tree. A manually supplied or mismatched load is rejected.
3. In each method/block interval, submit all 32 disjoint prompts exactly once
   in one start-gated burst. The formal admission limit cannot exceed those 32
   unique prompts; SGLang's locked admission limit drains the queue without
   resetting the cohort. Run eight independently randomized blocks over the
   16K-to-40,960 long region. The batch owns the union of its active decode
   intervals; request-level rows are diagnostic only.
4. Collect one diagnostic comparison per learner against the paired Static
   rows and bind the evidence to a GPU attestation.
5. Profile in a separate run; synchronized traces cannot enter headline
   timing.

The command family is:

```text
build-onlinespec-study        list-onlinespec-candidates
verify-onlinespec-source
render-onlinespec-tuning-runtime
run-onlinespec-tuning-slice  advance-onlinespec-tuning-stage
select-onlinespec-config     render-onlinespec-runtime
build-onlinespec-queue       run-onlinespec-confirmation
collect-onlinespec-study     attest-onlinespec-study
analyze-onlinespec-study
```

Use `lightcone-spec COMMAND --help` for exact arguments. Generated selections,
performance data, and attestations remain under the ignored artifact root.
The comparison is important evidence, but `analyze-onlinespec-study` explicitly
reports `core_speed_gate_affected=false`.

## Reproduction claim

LightCone-Spec claims a clean-room, paper-equation implementation of the
OnlineSPEC online drafter learners under one industrial speculative-decoding
runtime. It does not claim byte-for-byte reproduction of the official scripts,
does not redistribute their code, and does not treat Online-LR's reasoning DPO
pipeline as if it were token-level draft-model adaptation. Any future extension
of that pipeline requires its own objective, data, memory contract, and
registered comparison.

For a repeatable source audit, clone the official repository outside this
project and detach at the recorded commit:

```bash
git clone https://github.com/ZinYY/OnlineSPEC.git OnlineSPEC-upstream
git -C OnlineSPEC-upstream checkout --detach e58f82eb3f3adca3a686211236bf4f6e9e7e3a2b
lightcone-spec verify-onlinespec-source \
  --checkout OnlineSPEC-upstream \
  --audit manifests/provenance/onlinespec_source_audit_v2.json \
  --output /path/to/ignored-artifacts/onlinespec-source-verification.json
```

The verifier fails unless the checkout is clean and its commit, tree, 18
audited file hashes, and license inventory all match. The checkout and receipt
are audit inputs only and must not be copied, vendored, or committed into
LightCone-Spec.
