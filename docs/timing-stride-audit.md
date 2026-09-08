# Timing audit and global stride freeze

Status: native-RoPE retained-candidate TP2 forward/gradient QA passed on v66/marker-v48.
The bounded-window harness passed six excluded TP2 baseline windows, including
cancellation/reset acceptance. Optimization gains and global stride selection
remain **UNMEASURED**. No formal recipe is changed.
Preview performance numbers do not propagate into the manuscript.

## Fast feedback supersedes the long-baseline prerequisite

The first implementation candidate reuses only the parameter-independent frozen
layer prefix within one update, between active-weight reconstruction and gradient
replay. It never caches across requests or updates; selected/non-layer parameters
opt out when independence is not established. Reconstruction, gradient anchoring,
and publication gates remain unchanged. A separate candidate runtime is compared
against the baseline with the same S=1 configuration before any acceptance.

The second, separately measured candidate prepares the owned teacher probability,
log-probability and masked position weights once per update. The original KL
operations and gradient are preserved; teacher data are not cached across windows.
Its comparison baseline must be the last retained implementation, not an unrelated
historical speed. Neither candidate is accepted for formal results by CPU tests.

The approved quick-tuning protocol no longer waits for all 24 long-baseline cells.
Preserve completed/partial old evidence, stop its supervisor from claiming more
work, and record an intentional interruption separately from a safety failure.
Do not alter formal results or the first 40 completed preview cells.

`scripts/quick_tune_lightcone.py` has independent baseline/compare/stride phases:

- Same 8B DFlash, TP2, c1 and frozen non-display Code/Math calibration pool.
  Compatible jobs reuse a server; each warmup and measured window resets state.
  Incompatible resource/layout or runtime changes restart a separate server session.
- Ten-second warmup, reset, then thirty seconds from first token. EOS advances
  the fixed prompt sequence, cycling only within this excluded diagnostic window.
  All methods use the same prompt/seed sequence; request counts are not repetitions.
- At cutoff, abort only the known active diagnostic request. Preserve every event,
  including late arrivals; count only arrivals within the window. Abort, stream exit,
  pre-reset rank safety, flush and reset validation have a combined ten-second limit.
  Failure discards that server. A missing first token after sixty seconds fails;
  it does not silently become a zero-throughput capacity result.
- Report stream-observed committed tokens / (prefill + window) and / window
  separately. These include online learning but are not full-request benchmark
  measurements. Per-user prefix speed requires valid native timestamps; absent
  evidence stays UNMEASURED. Loading/configuration/warmup/cleanup are separate.
- First comparison: old/new in two domains, thirty seconds each. Both domains
  worse than -3% discard immediately; both better than +3% trigger confirmation.
  Otherwise repeat up to three pairs per domain (ninety seconds per side).
  A mixed direction or less than 1% aggregate gain retains the old implementation.
  Comparisons cannot change science parameters or runtime precision.
- Baseline is six short windows (Static, LC S1, LC S10, two domains). Stride
  screening is sixteen short windows (seven LC strides and Static, two domains).
  Short evidence only nominates candidates; it does not freeze a recipe.

Keep at least three documented single-hotspot optimization or exclusion rounds.
Deep profiling is a separate short diagnostic, never used to rank performance.
Only retained candidates incur long-request/multi-update/reset/TP regression and
the original independent confirmation/common-method acceptance. All raw evidence
is retained; the formal public CLI, YAML and SQLite schemas remain unchanged.

## Immediate failure repair

A completed capacity-negative TP2 Static c32 QA omitted already-collected rank-local
metrics. The reviewer stopped on missing evidence. Preserve before/after rank counters
and actual allocation even when registered requests time out; never call that row a
performance success. Native KV admission failures require the existing log evidence
of pool capacity and a valid model context; connection/numerical errors do not qualify.

The first excluded Code/S1/TP2 baseline (timing off) subsequently reproduced a
reconstruction rejection at round 4091 / version 4090: RMS 0.022035593 exceeded
0.01953125, while KL 0.001462524 remained below its unchanged limit. Bounded
two-rank capture and fixed-state replay found unchanged proposal/boundary KV and
bit-identical repeated replay logits. Native-module replay matched production
bit-for-bit. Single-operator interventions isolated RoPE: native RoPE alone made
hidden states and logits bit-identical (RMS/KL zero); attention, Q/K norm and MLP
substitutions alone did not change the rejection. This is one retained-state
diagnosis, not general runtime acceptance or a speedup measurement.

CUDA replay now uses the same native RoPE primitive, retaining the mathematical
rotary VJP and immutable captured inputs. CPU tests cover the VJP, long positions,
partial rotary suffix, repeatability and non-aliasing. The new runtime still needs
retained-candidate GPU/gradient checks and full-budget long-request regression;
no threshold, stride or formal result is changed by this repair. Rejected metrics
also preserve measured request/rank evidence rather than inventing zero completions.

## Measurement scope

`scripts/audit_lightcone_timing.py` creates its own excluded SQLite and uses the
formal continuation GPU lease. The formal SQLite is read-only; old raw attempts,
completed preview cells, and frozen selections are not rewritten. A fresh process
per cell keeps clean-server pairs. SIGINT stops new claims at the cell boundary.

The import-time hooks in `scripts/timing_hooks` avoid Python line tracing. CUDA
events are reclaimed by query, with event waits only at the excluded cell boundary.
Overflow is explicit and invalidates timing attribution. Request resets preserve
the cell recorder. Per-rank event clocks remain separate. CPU timings include
client waits and are not called GPU time. Deep CPU/CUDA traces use shapes and
allocator memory, without stack tracing; NVTX ranges allow Nsys attribution.

The HTML/JSON/CSV report distinguishes inclusive durations, interval unions, and
interval overlap. CUDA event intervals can include GPU waits or contention: they
are not kernel occupancy or direct causal decoding blockage. Hardware/kernel
overlap, main-line blockage, and unattributed wall time remain UNMEASURED until a
deep trace supports their attribution. Existing `main_side_overlap_ratio` and
`exposed_update_ms` are not substitutes for that evidence. Hardware counters
unavailable without permission remain unavailable.

## Registered work

- Baseline: Static / LightCone S1 / LightCone S10, two domains, two blocks,
  full timing on/off = **24 excluded cells**. Same 4 search prompts, TP2, c1,
  normal EOS and original 32K output limit. Observation effects above 1% prohibit
  using full instrumentation for formal speed measurements.
- Optimization: at least three documented rounds, one main hypothesis each;
  candidate vs prior implementation across two domains/two blocks = 8 cells per
  candidate. Keep failed/no-gain trials; stop blind changes after two consecutive
  rounds without reproducible >=1% improvement. Final independent confirmation
  is separate from selecting candidates.
- Stride coarse: S={1,4,10,32}, two domains/two blocks + four Static = **20**.
- Refine: up to two nearest unmeasured neighbours in {1,2,4,8,10,16,32} = **<=8**.
- Confirmation: frozen candidate vs Static, two domains/four independent blocks,
  eight held-out prompts/domain = **16**. No reselection using confirmation.
- Other-method feasibility and cross-model/backend/topology transfer QA are extra;
  do not count them as formal or hide them in the 44-cell search budget.

CalibrationMix APPS/Code and OpenR1-Math provide disjoint search/confirmation IDs
and text, excluding preview inputs. Freeze order and seeds before running. Keep
ChronoBelief/rank8/last1/LR1e-3/constant, model precision, safety and update rules.
Select LightCone/Static geometric mean throughput; within 1%, use measured update
cost, p99 ITL, then larger S. Missing cost is not zero. Confirm other adaptive methods
in the frozen ranking order; no common feasible S means no global freeze.

## Rollout gates still outstanding

Validate the new hooks on GPUs, measure observer overhead, retain only supported
optimizations, then perform the stride search and independent confirmation. Only
after shared-method and transfer QA should a separate versioned freeze/replacement
migration touch pending formal work. S10 winning does not invalidate unchanged S10
evidence, but hot-path performance versions must never be silently pooled.
All ETA strata without compatible measured runtime remain UNMEASURED.
