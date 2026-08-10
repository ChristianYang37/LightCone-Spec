# Mathematical method

[中文](../zh-CN/mathematical-method.md) · [README](../../README.md)

## Objective

Let \(p_t\) be the target distribution and \(q_\theta\) the DFlash proposal at
one valid draft position. The update minimizes a position-weighted target-to-
draft KL:

\[
\mathcal L(\theta)=
\frac{\sum_{b,k}m_{b,k}\lambda^k
D_{\mathrm{KL}}(p_{b,k}\Vert q_{\theta,b,k})}
{\sum_{b,k}m_{b,k}\lambda^k}.
\]

The semantic mask \(m\) removes positions beyond the request limit and after a
terminal token. It deliberately retains the sampled counterfactual suffix after
the first rejection: target verification produced a valid teacher distribution
for that complete canvas, exactly as in the TTS objective. The gradient is
normalized across the cohort, so batch size does not silently change the
learning rate. For the registered block-16 DFlash checkpoint, the weight is
bound to its training recipe,

\[
w_k=\exp\!\left(-\frac{k-1}{7}\right),
\qquad \lambda=\exp(-1/7).
\]

Uniform weighting is not used for DFlash because an error near the front of a
canvas prevents every later token from surviving prefix verification. The
decay is therefore part of the run identity, not a silent default or a
result-derived hyperparameter. See the
[DFlash paper](https://arxiv.org/abs/2602.06036).

## Identical candidates, different publication

For source round \(r\), both adapted methods compute the same functional
optimizer proposal \(u_r\). Active parameters and moments remain unchanged
until publication.

With update stride \(S\), TTS publishes no earlier than the next fixed update
boundary:

\[
a_{\mathrm{TTS}}=
\max\!\left(a_{\mathrm{ready}},
(\lfloor r/S\rfloor+1)S\right).
\]

L0 publishes at the first legal decode boundary after the side event is ready:

\[
a_{\mathrm{L0}}=a_{\mathrm{ready}}.
\]

Candidate tensors, loss, optimizer, rank, stride, supervision, and source
version are held equal. Any numerical candidate difference is an implementation
failure, not a method distinction.

## Truncated online gradient

At an update round, historical paged KV is gathered and detached. Current
canvas K/V is recomputed through differentiable DFlash linear layers, RMSNorm,
NeoX RoPE, and non-causal SDPA. A device-side reconstruction predicate compares
the recomputed hidden state with the actual inference hidden state. A mismatch
turns the proposal into a no-op and disables further cohort adaptation.

The target embedding and LM head are frozen. The final target head maps the
differentiable DFlash hidden state to draft logits. A source-point proximal KL
has zero first derivative, so the implementation does not materialize that
ineffective term or expose its coefficient as a tuning dimension.

## Parameterizations

For a selected base matrix \(W\), LoRA uses

\[
W'=W+BA,
\qquad A\in\mathbb R^{r\times d_{in}},\quad
B\in\mathbb R^{d_{out}\times r},
\]

with seeded \(A\), zero \(B\), and both factors trainable. Inference sees only
the merged fixed-address matrix. Full uses an FP32 master for every DFlash-owned
floating parameter. Tail LoRA/full instead computes \(h'=h+\Delta h\) and
projects \(h'\) once through the frozen target head. Residual tail mode applies
a low-rank correction directly to logits.

For DSpark, \(h\) is specifically the normalized LM-head input; the Markov
state remains a separate frozen feature. For EAGLE/EAGLE3, every hidden in one
proposal chain is evaluated against a tail bank pinned to one source version.

## Optimizer dynamics

Every optimizer is implemented as a functional proposal
\((\theta_t,s_t,g_t)\mapsto(\hat\theta_{t+1},\hat s_{t+1})\). The active
parameter and state pair changes only when publication commits the candidate.
Thus TTS and L0 receive identical optimizer arithmetic even if their candidates
become publishable at different decode boundaries.

Adam and AdamW use bias-corrected FP32 first and second moments; AdamW applies
decoupled decay. For coupled-decay momentum methods, let
\(\tilde g_t=g_t+\lambda\theta_t\). SGDm and NAG use

\[
v_t=\mu v_{t-1}+\tilde g_t,\qquad
d_t^{\mathrm{SGDm}}=v_t,\qquad
d_t^{\mathrm{NAG}}=\tilde g_t+\mu v_t.
\]

Lion keeps one moment and uses

\[
d_t=\operatorname{sign}(\beta_1m_{t-1}+(1-\beta_1)g_t),\qquad
m_t=\beta_2m_{t-1}+(1-\beta_2)g_t,
\]

with decoupled weight decay. Muon uses a Nesterov momentum proposal for every
two-dimensional parameter, applies the registered quintic Newton--Schulz
orthogonalization, and scales the matrix step by
\(\sqrt{\max(1,d_{out}/d_{in})}\). Biases, norms, and other non-matrix
parameters take an explicitly configured auxiliary AdamW step. No optimizer
silently creates unused moment tensors or falls back to another rule.

## OnlineSPEC comparison learners

The separate OnlineSPEC baseline uses projected OGD rather than a core
optimizer. Given revealed gradient $g_t$, OGD uses

\[
w_{t+1}=\Pi_K(w_t-\eta g_t).
\]

Optimistic OGD keeps an anchor \(\hat w_t\) and uses the last revealed gradient
as its next hint. Hedge keeps independent OGD experts, evaluates each loss and
gradient at that expert's own decision, and forms the next decision in one
registered coordinate class with probabilities proportional to exponentiated
negative cumulative loss. Full uses dense parameters; LoRA uses a fixed-rank
factor-coordinate decision and is reported separately. The implementation is
transactional and commits only after feedback, so
the decision that receives a gradient is exactly the decision that generated
the proposal.

These equations, projection semantics, and clean-room source boundary are
specified in [OnlineSPEC baseline](onlinespec-baseline.md). They are comparison
math and do not change the Static/TTS/L0 objective or speed gate.

## Exact speculative sampling

The proposal probability \(q\) recorded for verification is exactly the
distribution that generated the draft token. Given target probability \(p\), a
proposal token \(x\) is accepted with

\[
\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right).
\]

On rejection, the replacement is sampled from normalized
\((p-q)_+\). This preserves the target distribution even though historical KV
was produced by older drafter versions. The formal controlled speed profile is
greedy so all methods follow the same token trajectory; stochastic coupled-RNG
and distributional exactness are checked separately.

## Speed condition

The measured decode time is decomposed as

\[
T_m=T_{\mathrm{static}}-\Delta T_{\mathrm{target}}
+T_{\mathrm{update}}^{\mathrm{exposed}}
+T_{\mathrm{draft}}^{\mathrm{extra}}+T_{\mathrm{barrier}}.
\]

Acceptance alone is insufficient. A method clears the engineering gate only
when paired long-region decode goodput improves, target calls do not contradict
that improvement, and exactness, version, fallback, non-finite, OOM, and
retraction counters remain zero.
