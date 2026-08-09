# Troubleshooting

[中文](../zh-CN/troubleshooting.md) · [Home](../../README.md)

## Patch or runtime identity failure

- Confirm the full upstream commit, detached clean status, patch digests, and
  expected tree in `patches/sglang/manifest.json`.
- Do not edit a runtime after load screening or tuning begins. Any source,
  package, model, sampling, or launch-argument drift needs a fresh selection
  and evidence root.
- Never point a test at an implicit workspace `sglang/`; pass the explicit clean
  upstream or disposable patched checkout.

## Model or configuration refusal

- Regenerate the model lock when an upstream revision changes. A model-roots
  file must bind that exact lock and every selected local directory must exist.
- DFlash adaptation rejects TP/DP greater than one, quantized draft paths,
  block/canvas mismatch, unsupported speculative options, and insufficient
  explicit HBM reserve.
- Formal Static/TTS/L0 endpoints require the explicit speed-study metric flag
  and exact rejection sampling. If the exact kernel is unavailable, fix that
  environment; do not replace it with a greedy DFlash fallback.
- Static must have no adaptation object. TTS and L0 must have identical
  adaptation fields and the exact selected concurrency.

## Memory pressure

Adaptation state is sized before the KV pool. It is resident, not evictable,
never automatically offloaded, and never silently changed from Full to LoRA or
tail. Reduce admission, context, or the explicitly selected parameter layout;
then repeat load screening because the runtime identity changed.

An OOM or retraction is a safety event in the formal study. Do not remove the
counter, increase a timeout, or drop the failed request from the denominator.

## Interrupted confirmation

Run the identical `run-confirmation` command against the same immutable root.
Hash-bound completed receipts are verified and skipped. Parquet shards without
a terminal receipt are interrupted attempts and are excluded. If a receipt is
corrupt, duplicated, or bound to another identity, preserve the directory for
audit and start a new evidence root; do not hand-edit it.

## Unexpected `UNMEASURED` or `BLOCKED`

`UNMEASURED` means no valid GPU attestation was supplied, even if diagnostic
arithmetic is positive. `BLOCKED` means attested GPU evidence exists but at
least one registered speed, interval, safety, coverage, or publication
criterion failed. Both are valid outcomes and must not be relabeled.

When reporting a bug, include sanitized commands, package and platform
versions, exact upstream and patched tree IDs, and the smallest reproducible
config. Never publish tokens, passwords, private prompts, instance addresses,
or machine-specific model paths.
