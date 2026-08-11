# Mathematical method

[中文](../zh-CN/mathematical-method.md) · [README](../../README.md)

## Objective and semantic rows

The equations below specify the registered target method. They do not imply
that every backend or topology is executable in this release. The industrial
executor currently completes only TP1/DP1 Target-only. The pinned patch now
contains the native
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v1` lifecycle and
a lower-level adaptive implementation only for TP1/DP1 DFlash, but no trusted
hardware signer is configured. Static/TTS/L0 remain blocked before mutation,
and no new GPU result is available.

For target distribution (p), reconstructed proposal (q_\theta), semantic
mask (m), and optional position weight (w_k), the common proposal objective
is target-to-draft cross entropy (equivalently KL up to target entropy):

\[
\mathcal L_{\mathrm{proposal}}(\theta)=
\frac{\sum_{b,k}m_{b,k}w_k
[-\sum_v p_{b,k,v}\log q_{\theta,b,k,v}]}
{\sum_{b,k}m_{b,k}w_k}.
\]

The mask excludes rows beyond the request budget and after terminal state, but
retains verifier-produced teacher rows for the complete legal canvas, including
the sampled counterfactual suffix after an earlier rejection. Cohort
normalization prevents batch size from silently scaling the learning rate. An
empty or non-finite denominator produces an invalid device candidate and is
discarded; it is never converted to zero loss.

For the registered block-16 DFlash recipe,

\[
w_k=\exp(-(k-1)/7).
\]

Position weighting, teacher-row policy, canvas width, and source version are
part of configuration/evidence identity rather than result-derived choices.

## Backend evidence and reconstruction

Let (E_b) be a backend-native `ProposalEvidence` envelope. Its common fields
contain adapter-free logits (z^0), deployed proposal logits, the normalized
sampling distribution (q^{\mathrm{sample}}), mask and teacher rows, actual
sampled predecessor, cohort/source identity, and backend payload. A registered
backend reconstruction

\[
R_b(E_b,\Delta_\theta)
  \longrightarrow (z_\theta,q_\theta,c_\theta?)
\]

must preserve row/vocabulary shapes and confidence availability. At zero
functional delta it is checked against native inference. If inference already
applied the adapter, passing another nonzero delta is rejected so the update is
not counted twice.

Structural validation runs on the host before expensive work. Finiteness,
normalization, and reconstruction/supervision validity produce a three-byte
device receipt that is copied asynchronously to pinned host memory on the side
stream before its completion event. A legal publication boundary first makes a
nonblocking event query; only after completion does it read the receipt outside
the measured update interval. It uses neither `.item()` nor a blocking event
synchronization. An invalid receipt advances neither optimizer step nor active
version and publishes no tensors.

The current lower-level patch reconstructs DFlash's differentiable canvas for
TP1/DP1. The target contracts for DSpark require native Markov W1/W2 features
and the actual sampled predecessor; EAGLE/EAGLE3 would bind tree state and one
source version across the proposal chain; NEXTN would bind native MTP hidden
state and a content digest of the upstream interface. These payloads are not
interchangeable, and DSpark/EAGLE/EAGLE3/NEXTN adaptation remains `BLOCKED`.

## DSpark composite objective

The non-executable target DSpark contract may also train its native
confidence/acceptance state in a hybrid plan.
For a verified target row (p) and the sampling-bound proposal (q), define a
stop-gradient conditional-survival target

\[
s=1-\operatorname{TV}(p,q)
=1-\frac12\sum_v|p_v-q_v|,\qquad s\in[0,1].
\]

With confidence logit (a_\theta) and tuning-locked nonnegative coefficient
\(\gamma\), the masked hybrid objective is

\[
\mathcal L_{\mathrm{DSpark}}
=\mathcal L_{\mathrm{proposal}}
+\gamma\,\operatorname{BCEWithLogits}(a_\theta,s).
\]

The survival target is detached, so the confidence term cannot move the
proposal by differentiating through its own label. Layer-only DSpark freezes
W1, W2, and confidence and rejects \(\gamma\). Hybrid DSpark trains W1, W2, and
the scalar acceptance/confidence parameter as Full replicated state while the
selected backbone layer matrices use the chosen Full or LoRA plan.

Fixed-budget verification is a tuning control. Native-scheduler confirmation
is separate; missing supervision rows cannot be repaired by changing the mask
after observing results.

## Parameterization and memory

For a selected native matrix (W), LoRA uses

\[
W'=W+BA,\qquad
A\in\mathbb R^{r\times d_{in}},\quad
B\in\mathbb R^{d_{out}\times r}.
\]

Initialization is deterministic and has zero functional delta. Registered
ranks are (1,2,4,8,16,32,64), with \(\alpha/r=1\). Full uses FP32 masters for
every eligible floating parameter in the selected `last1`, `last3`, `last5`,
or `all` native layer scope. Borrowed target embeddings, target head, and
target model are frozen.

DSpark additionally permits last-1/3/5 hybrid scopes with native heads. The
complete E1a geometry is (4\times8+3\times8=56) configurations. Full/LoRA,
scope, rank, alpha, selected and frozen names, dtype, and sharded/replicated
ownership are all digest-bound.

Memory is derived from the actual trainable coordinates: active merged values,
FP32 master and gradient, allocated optimizer moments, functional candidate,
staging, activation/scratch, graph, KV, and telemetry. Admission uses the least
headroom across ranks; no method may be silently changed to fit.

## Functional optimizer and publication

Each optimizer implements a functional proposal

\[
(\theta_t,s_t,g_t)\mapsto(\widehat\theta_{t+1},\widehat s_{t+1}).
\]

Active parameters and optimizer state change only when the complete candidate
commits. Adam/AdamW use two FP32 moments; SGDm/NAG/Lion use one. Muon uses a
matrix momentum and registered Newton--Schulz transform, plus auxiliary AdamW
moments for non-matrices. Schedules advance on published updates, not attempted
or discarded candidates.

For source round (r), TTS and L0 compute the same candidate (u_r). With
stride (S), TTS waits until the next registered update boundary at or after
readiness, whereas L0 uses the first legal graph boundary after readiness.
Loss, trainable plan, optimizer arithmetic, source rows, and candidate bytes
must match; a numerical difference is an implementation failure.

In the target CPU coordinator contract, TP2 sharded coordinates stay on their
inference owner and replicated coordinates reduce inside the TP replica; DP2
cohorts are sticky and replica-local. All ranks would prepare one update
identity and derive one commit or abort decision. The current `RunConfig`
rejects every TP2/DP2 plan before model loading, so these equations are not a
multi-rank execution claim.

## OnlineSPEC comparison learners

The separately registered OnlineSPEC baseline uses projected SGD decisions.
Projected OGD is

\[
w_{t+1}=\Pi_K(w_t-\eta g_t).
\]

Optimistic OGD keeps an anchor and the last revealed gradient as a hint. Hedge
keeps independent experts, computes each expert's loss and gradient at its own
decision, and weights future decisions by exponentiated negative cumulative
loss. Full and LoRA factor-coordinate decisions are distinct classes; factor
averaging is not equivalent to averaging the dense products (BA).

OnlineSPEC is transactional and TP1/DP1 under its independent protocol. Its
equations and evidence cannot change the TTS/L0 candidate, selection, power
plan, or core gate.

## Exact speculative sampling

The recorded proposal probability (q) must be the exact distribution that
generated the draft token. Given target probability (p), token (x) is
accepted with

\[
\alpha(x)=\min(1,p(x)/q(x)).
\]

On rejection, replacement is sampled from normalized \((p-q)_+\). This is the
positive-part rejection residual required by exact speculative sampling. It is
not an adaptation parameterization, trainable module, schema value, or legacy
alias. The distinction is important: removing the old adaptation option does
not change the sampler's exactness rule.

Greedy controlled traces keep paired token trajectories equal. Stochastic
coupled-RNG and distributional tests remain separate GPU requirements.

## Speed and claim condition

Measured decode time is decomposed as

\[
T_m=T_{\mathrm{static}}-\Delta T_{\mathrm{target}}
+T_{\mathrm{update}}^{\mathrm{exposed}}
+T_{\mathrm{draft}}^{\mathrm{extra}}+T_{\mathrm{barrier}}.
\]

Acceptance alone is not a speed result. A claim requires registered paired
goodput inference, target-work consistency, production SLO and completion
accounting, exactness/version/publication safety, HBM and energy evidence,
hardware-envelope validity, durable receipts, and a content-bound GPU
attestation. Without that evidence every new outcome remains `UNMEASURED`.
