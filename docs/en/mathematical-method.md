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

The semantic mask \(m\) removes invalid request positions, tokens after an
accepted terminal token, and draft suffixes conditioned on a rejected token.
The gradient is normalized across the cohort, so batch size does not silently
change the learning rate.

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

## Exact speculative sampling

The proposal probability \(q\) recorded for verification is exactly the
distribution that generated the draft token. Given target probability \(p\), a
proposal token \(x\) is accepted with

\[
\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right).
\]

On rejection, the replacement is sampled from normalized
\((p-q)_+\). This preserves the target distribution even though historical KV
was produced by older drafter versions. Greedy exactness is checked separately.

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
