# Excluded hot-path audit (2026-09-10)

Scope: Qwen3-8B + DFlash, TP2/c1, ChronoBelief LoRA rank 8, last1,
LR 1e-3, constant, S10. No formal 480-request rerun, paper edit, push, PR,
release, model download, or automatic resumption of the full DAG.

Baseline: `5b2cc6b`, SGLang v69. Candidate commits and runtimes remain separate.
GPU evidence is stored under the server diagnostic directory
`diagnostics/hotpath-940f95a5`; local compact results accompany the final report.

## Screening ledger

Each pair uses the same frozen calibration inputs (36,864 prefill tokens),
independent 10-second warmup/reset and a 30-second streaming window.
Three paired repetitions per domain; ratios use geometric averaging.
Loading and input construction are recorded separately. Window throughput is
not a completed-request benchmark and includes the window's prefill cost.

| Candidate | Code gain | Math gain | Screening decision |
|---|---:|---:|---|
| Explicit attention VJP (`f9fbcda6`, v70) | -16.82% | +2.01% | Rejected; original attention retained |
| Analytic RMS VJP only (`a4096a15`, v72) | +1.04% | +1.26% | Borderline candidate; requires independent confirmation |
| Local scheduler rank collector vs global `sys.settrace` (same v69) | +16.07% | +12.62% | Candidate; requires independent confirmation |

Synthetic small-tensor microbenchmarks favored both VJPs, but did not predict
the attention end-to-end result. They are explicitly not acceptance evidence.
BF16 attention gradients differed by small rounding amounts; the candidate is
not promoted. RMS preserves the original FP32 accumulation order, including
the residual gradient addition; a simplification that failed exact regression
was discarded without relaxing tolerances.

The retained real snapshot at round 4091/version 4090 was replayed on both
ranks: production logits matched exactly, the strong mismatch control was
rejected, and the tested parameter gradient matched exactly. That particular
saved-state gradient uses the earlier audit's quadratic projection objective,
not the training KL; separate bounded live KL/proposal checks are prepared for
the final excluded requests.

## Other paths and evidence boundaries

- L0 returns without waiting when its candidate event is not ready. That does
  not eliminate CPU launch costs, stream/resource contention, or TP publication
  synchronization. Existing `barrier=0` is not proof of zero total blocking.
- RoPE already uses an analytic inverse-rotation VJP. Its native forward is
  in-place; the independent input and dummy-K storage cannot simply be aliased.
- TP validity reduction, host reads, and post-publication barrier are preserved.
  Reusing vocabulary-gather storage requires proof that no live backward graph
  still owns the output. No unproven synchronization/buffer change is retained.
- An asynchronously read context flag cannot be used a round late while claiming
  identical activation semantics. The exact threshold check is retained and
  measured separately around the 20K crossing.
- New local instrumentation covers attention/RMS/RoPE forward/backward,
  vocabulary gather, collectives, preparation, publication and round/gate setup.
  Full timeline data is excluded from candidate ranking. Inclusive timings and
  rank clocks are not added into wall time or interpreted as kernel overlap.
- Request-end update/loss traces are retained before reset. This does not imply
  all old evidence was missing: the pre-existing diagnostics method already
  writes a telemetry file when configured.

## Final excluded acceptance

Independent confirmation uses the pre-frozen confirmation prompts, not the
search prompts. Final full-output QA covers short input, a 20K crossing and
36K input, with complete reset and both-rank safety accounting. Bounded live
KL/gradient/optimizer-proposal replay deliberately duplicates work and
synchronizes only at the selected diagnostic updates; its request latency is
not a performance result.

All 12 independent confirmation windows passed: combined old-to-new window
goodput gains were Code +16.316% and Math +9.756%. Six 4096-output-token requests
(old/new at short, 20K crossing, and 36K inputs) passed. Within each input pair,
all output token IDs, publication counts and gate activation contexts matched.
Both ranks' real KL, gradients, ChronoBelief proposals and uncommitted optimizer
state matched exactly at updates 1 and 100 for all three inputs. No request
reached update 300; do not describe that event as tested.

Retain RMS-only v72 and the local collector; attention remains original.
This is excluded correctness/short-window acceptance, not a formal 480-request
performance rerun or proof of beating Static. The formal SQLite logical hashes
and row counts remained unchanged, integrity=ok.

Harness-only failures were preserved and repaired: request-reset eligibility,
tokenizer BatchEncoding normalization, and an invalid duplicate flush method
call after the first successful full request. That successful request was
reused; no completed performance window was rerun. Final CPU suite: 399 passed.
Power state and complete local evidence links belong in the delivered report.
