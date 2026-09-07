# GitHub preview v3: 96 registered cells

Status: implementation and excluded validation in progress; all v3 GPU results
are UNMEASURED. This changes GitHub preview only, not paper or formal S10 recipes.

| Panel | Methods | Four-block cells |
|---|---|---:|
| Qwen3-8B, LiveCodeBench / MATH-500 long generation | Target-only, Static EAGLE3, Static DFlash, OnlineSPEC-Ensemble DFlash transfer, LightCone DFlash | 40 |
| Qwen3-8B DSpark c1/c8/c32 | Static, LightCone | 24 |
| Qwen3-8B DSpark shared BurstGPT trace | Static, LightCone | 8 |
| Qwen3.8-27B | Target-only, native MTP, community DSpark, community DFlash2, OnlineSPEC-Ensemble DFlash2 transfer, LightCone DFlash2 | 24 |

The existing prompt records, input/output budgets, seeds 0/1/2/3, normal EOS,
trace arrivals and request counts remain frozen. All methods in a comparison
share the minimum completely validated TP1 or TP2 topology. No one-method TP
upgrade, quantization, offload, workload shortening, or default DSpark calibration.

## Frozen recipes and interpretation

Preview LightCone is ChronoBelief, LoRA rank8, last1, LR=1e-3, constant, S=1.
This is the user-specified deployment recipe, not a claim of optimality at S1.
DSpark retains its seven fitted confidence temperatures. Formal S10 is unchanged.

OnlineSPEC-Ensemble is an explicitly named **algorithm transfer**, not the
authors' engine or best-on-DFlash claim. Source: [OnlineSPEC v2, B.10](https://arxiv.org/html/2603.12617v2#A2.SS10).
Three independent Full/all Adam experts use LR={1e-4,2e-4,4e-4}, beta=(.9,.95),
epsilon=1e-8, clipping=.5, WD=0, cumulative-loss Hedge inverse-temperature=10.
The transfer performs one update on current valid supervision per S10 event.
The source's chunk=40 / epochs=2 are provenance, not executed chunk training.
The [public pipeline](https://github.com/ZinYY/OnlineSPEC/blob/main/EAGLE/pipeline_eagle3_hedge.py)
uses a previous-loss softmax and differs from the paper's cumulative-loss rule;
this transfer explicitly chooses the paper rule. Updates, merge and synchronization
are included in end-to-end timing. Rejected proposals commit no weights, Adam
moments, cumulative losses or steps. The legacy E0 OGD-based ensemble is unchanged.
S1 vs S10 is not an equal-update-budget mechanism ablation.

TTS is removed from new preview execution, tables and both video method lists.
All legacy raw TTS and other attempts remain auditable under their old protocol.
No old metric is rewritten. New identities and frozen pairing keys prevent mixing
legacy S10 timings with v3; supersession is protocol-driven, never gain-driven.

## Required correctness review

The old greedy MATH third request reaches 32768 tokens for Static and 771 for
LightCone. Current throughput formulas reproduce raw counters, but this does not
resolve trajectory correctness. Old raw requests lack output IDs, so they cannot
locate the first divergent token. `scripts/diagnose_preview_trajectory.py` captures
the first three original requests (same order/seeds) with target, Static, frozen
and active adapters with native output IDs in an excluded bounded run.
DFlash rejects `return_logprob`; only target-only captures top-2 probabilities.
The first common prefix is scored through target-only; verification-side logits
need separate instrumentation if that does not explain the difference.
Generated text stays private. No captured run is an automatic pass.
The optional `--verify-trace --reference <prior requests.json>` diagnostic
loads an excluded-only Python trace hook. It records verification top-2 logits,
argmax/committed/bonus IDs, positions, KV locations and adapter versions only
around request 2 output positions 620--660. It changes no SGLang patch or
sampler. Synchronization can perturb execution: a missing trace or any output-ID
change from the successful uninstrumented capture rejects that diagnostic.
Its timings must never be used for performance claims.
The additional active-only `--target-state-audit` checks exact target-parameter
bytes from first decode to a due update in that window, publication/optimizer
storage disjointness, and the request's committed target KV before/after that
update. It supports only unquantized token-major MHA, synchronizes explicitly,
hashes host chunks without retaining weights, and fails on unsupported layouts.
It does not prove all-request/all-time KV invariance. Missing evidence or output
changes from the uninstrumented reference reject the capture.
Review verification/bonus/stop indices, target state, KV/version/reset, and
numeric margins before declaring a root cause or accepting speedup claims.
The diagnostic output cap is not a change to any formal request budget.

`scripts/validate_preview_updates.py` separately validates the proposed update
paths without accepting any formal node. It uses one frozen 8B MATH prompt,
two 512-token greedy reset repetitions, then either a forced-32K S1 pressure
request or a 512-token sampled ensemble request. The forced EOS override is
excluded pressure testing, not the normal-EOS benchmark. A separate GPU process
checks actual Adam experts against native Adam, rejection transactionality,
cumulative Hedge and reset, then exits before server memory measurements.
Short passing requests cannot stand in for full-condition common-TP acceptance.
The separate `--reset-diagnostic <original QA directory>` mode runs only the two
512-token requests, fingerprints post-reset optimizer/master/active/staging and
target state, and traces verification offsets 150--190. It never writes a passing
QA result or formal acceptance. It records whether instrumentation preserves each
original trajectory; synchronized traces that change an async publication schedule
cannot establish the cause of an uninstrumented divergence. Epoch/generation IDs
are recorded separately from byte-equality checks because they intentionally advance.

## Metrics, memory and release

Four raw block points and block-level paired log-ratio t95% intervals (df=3) are
reported; requests are not independent runs. Greedy repeats characterize timing,
not response-population uncertainty. Report absolute throughput, native per-user
speed, AL excluding bonus tokens, completion and negative outcomes. Target draft
AL is N/A. All grouping includes backend, topology, policy and preview revision.

Report allocator allocated/reserved peaks, NVML peak, update resident/peak budget,
KV capacity and TP rank-local values. Sum of rank-local peaks is explicitly a sum
of individual peaks, not a simultaneous system peak. Allocator peak resets at
each successful cache flush, including flushes without emptying the CUDA cache;
KV sizing and allocation policy are unchanged. Upper bounds are not measured
update peaks. A larger KV pool is not automatically larger adaptation overhead.
No memory advantage is claimed until compatible measurements exist.

Both six-method videos replace the TTS slot with OnlineSPEC transfer. Real c8,
common TP, eight 16K inputs and 1024 output limit; server-local Chromium/loopback,
independent recordings with 1x aligned playback. Real committed token chunks are
displayed together; no artificial typing. Video data is excluded from 96 cells.

## Resume contract

Freeze `formal_preview_manifest_v3` with version=3 and original stimuli. Enable
`formal_preview_v3` only after CI and matched GPU QA. Acceptance stores the exact
manifest, `trajectory_diagnosis=reviewed` and per-node accepted status. Old v1/v2
acceptance cannot authorize v3. Unsupported 27B paths must wait for real adapter
support; downloads and CPU tests do not constitute GPU acceptance. The runner can
finish accepted 8B nodes first and stop at unaccepted 27B. After all v3 nodes finish,
resume the original full DAG/E5-last. Preserve SQLite/attempt/log backups and use
one atomic-claim runner. Models remain server-only. No paid GPU waits on development.
