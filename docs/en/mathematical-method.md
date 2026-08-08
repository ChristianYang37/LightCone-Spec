# Mathematical method

[中文](../zh-CN/mathematical-method.md) · [Home](../../README.md)

Let `h` be the final hidden state before the frozen proposal head `W`, and let
`m` be an optional DSpark Markov feature. LightCone applies one of three
cache-safe tail parameterizations:

\[
\Delta \ell=B(A_hR_h^\top h+A_m m),
\]

\[
\Delta h=(hA_h+mA_m)B_h,\qquad \Delta\ell=W\Delta h,
\]

\[
\Delta h=hD_h+mD_m,\qquad \Delta\ell=W\Delta h.
\]

They correspond to `residual`, `lora`, and `full`. Here `full` means a
full-rank tail matrix; the drafter backbone, target embeddings, and LM head stay
frozen, preserving historical KV validity.

L0 publishes at the first legal boundary. L1 uses a binary gate, L2 predicts a
damping factor in `[0,1]`, and L3 transports a candidate before evaluation.
Their purpose is to manage arrival-time utility under staleness, not to change
the speculative-decoding exactness rule. Sampling and rejection share the same
corrected proposal distribution.
