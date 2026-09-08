# GitHub preview v3: 96 registered cells

Status: first-40 collection is running; remaining-group acceptance and video
validation are separate gates. Consult the live SQLite for counts, not this document.
This changes GitHub preview only, not paper or formal S10 recipes.

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
For excluded TP2 QA, `/get_server_info` reports a DP leader, not one entry per
TP rank. An opt-in local trace records each scheduler's existing metrics before
its IPC response is filtered; QA requires fresh rank0/rank1 records and checks
both ranks' update/reset counters. This does not change production telemetry or
claim unobserved rank-local numbers from the DP-leader API response.
The explicit `--pressure-only` mode is an independent single forced-32K S1
stress request (TP1 or TP2), not a rerun of the failed greedy reset-equivalence
test. It preserves safety/count/publication checks and labels reset coverage
false. The original cross-reset trajectory assertion is unchanged; passing this
stress request never grants reset, sampling, common-group or formal acceptance.

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

The server-local `scripts/continue_preview.py` controller waits for all first 40
cells and their runner/server to exit, then backs up SQLite and fast-forwards the
pre-staged CI-green commit. It never edits the active runner's checkout. Its order
is DSpark full-condition QA / 32 cells, 27B full-condition QA / 24 cells, both
six-method video sets, then the original full DAG with E5-last. The controller
and its children share an exclusive lease; another new runner cannot recover or
claim their active work. An interrupted QA/take without a passing result is not
automatically overwritten or silently rerun.

`scripts/validate_remaining_preview.py` uses block-0's complete frozen workloads,
normal execution/reset and rank-complete checks in a separate excluded database.
Common TP1 is attempted first, then TP2 only for explicit capacity evidence.
Runtime, numerical and reconstruction failures stop for diagnosis. If DSpark
needs TP2, supplemental matched Target/Static anchors are measured separately;
the frozen arrival sequence and sample budget remain unchanged. QA and these
anchors are not extra preview cells. Acceptance binds exact group rows, so an
unstarted 27B topology change cannot invalidate or remap the first forty rows.
Each node rereads its acceptance; completed raw evidence is never rewritten.

Video `--qa` validates the full c8/16K/1024 workload without a capture or acceptance
write. All six methods must pass at one TP before recording. The event sink is
checked against every final native token trajectory, timestamps, stop record and
throughput denominator. Post-warmup reset bounds allocator peaks; the separate
100-ms NVML file bounds generation-only sampling. TP2 captures both ranks.
Chromium and FFmpeg are installed by `scripts/prepare_preview_recording.py` in a
separate Linux-server tools directory, never in the experiment venv or local model
cache. Each set retains six originals and its 1x composite; technical failed
takes remain visible and require an explicit audited retry directory.

`scripts/validate_preview_group.py` prepares excluded full-condition checks for
the 8B long-generation comparison. Each invocation uses one block-0 method/task
with all eight frozen requests, the original normal-EOS 32K limit, seed, cache,
recipe and budget rules. Select the same TP for all five methods; both tasks
must be covered before calling the complete group validated. The formal SQLite
is read-only; attempts and copied selections live in a separate excluded SQLite.
The normal cell executor supplies request/reset/safety/metrics semantics. TP2
captures both actual rank metrics before DP-leader filtering; missing ranks or
any rank's safety error fails the check. Passing one case writes no formal
acceptance and does not replace the original trajectory diagnosis. No benchmark
gain may be inferred from these excluded runs.

If formal v3 freezing has not yet occurred, this QA explicitly constructs an
excluded v3 candidate from the frozen legacy inputs and records its provenance.
It does not write that candidate back to the formal database, authorize legacy
TTS jobs, or freeze the tested TP. An existing malformed v3 manifest still fails
closed instead of falling back. The 27B manifest remains a separate prerequisite.

For preview-v3 native EAGLE3, use Triton for draft attention only: the pinned
FlashInfer multi-step draft backend cannot initialize the registered sliding
window with its single-wrapper buffer. Target attention, draft window, TP,
sampling and request budgets stay unchanged; historical E0 commands are not
rewritten. Preserve the failed initialization take and require GPU revalidation.
The pinned Triton multi-step backend also omitted speculative sliding-window
metadata. The patched path gathers each step/branch's actual packed KV tail,
uses the registered window length, and refills stable window buffers for graph
capture/replay as well as eager decode. It does not substitute ordinary request
pool indices, disable windows, or disable CUDA graphs. CPU ragged-buffer tests
are not GPU acceptance; retain both initialization failures and require a new
excluded full-condition take on the incremented runtime marker.
Decode follows the existing prefill inequality `query_pos - key_pos <= W`:
there are at most W left positions plus the current token (W+1 KV slots).
Draft-extend instead retains at most W prefix slots because current/extended
K/V are separate kernel inputs. Both capture and replay must supply non-null
prefix window metadata, including when the prefix is empty. A passing decode
capture alone does not establish draft-extend or full-request correctness.

Freeze `formal_preview_manifest_v3` with version=3 and original stimuli. Enable
`formal_preview_v3` only after CI and matched GPU QA. Acceptance stores the exact
manifest, `trajectory_diagnosis=reviewed` and per-node accepted status. Old v1/v2
acceptance cannot authorize v3. Unsupported 27B paths must wait for real adapter
support; downloads and CPU tests do not constitute GPU acceptance. The runner can
finish accepted 8B nodes first and stop at unaccepted 27B. After all v3 nodes finish,
resume the original full DAG/E5-last. Preserve SQLite/attempt/log backups and use
one atomic-claim runner. Models remain server-only. No paid GPU waits on development.
