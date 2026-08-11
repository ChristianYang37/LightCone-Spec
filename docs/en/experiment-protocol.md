# Experiment protocol

[中文](../zh-CN/experiment-protocol.md) · [Home](../../README.md)

## Registered question

The study tests whether paper-faithful TTS and first-ready L0 each increase
decode goodput over unchanged Static. It does not assume that acceptance gains
pay for training or scheduling cost. GPU status remains `UNMEASURED` until the
complete protocol produces a content-bound attestation.

The primary model pair is Qwen3-8B + DFlash. The checkpoint/model context limit
is 40,960 prompt-plus-generated tokens. Formal measurement caps that context at
40,928, leaving two block-16 speculative KV reservations at the request
boundary. The formal long region begins at 16K generated tokens and ends at
that safe request limit.
TTS and L0 must share candidate computation, optimizer, update mode, parameter
scope, rank, learning rate, stride, supervision, sampling, and load. Their only
difference is publication policy.

## Data isolation

The controlled adapter deterministically generates copyright-independent
prompts from a finite vocabulary. Three content-disjoint, hash-bound windows
contain eight load prompts, sixteen tuning prompts, and thirty-two confirmation
prompts. Confirmation data cannot influence load or hyperparameter selection.

Context is recorded as the actual `prefix_len_before` each proposal. Generated
position buckets are 0, 2K, 4K, 8K, 16K, 24K, 32K, and the safe limit. Natural
EOS side tables use locked LiveCodeBench and Math500 revisions, thirty-two
prompts each, and report the number of requests still at risk in every bucket.
They replicate behavior but do not determine the formal gate.

## Load and tuning

Static independently scans concurrency 1, 2, 4, 8, 16, 32, and 48. A load is
eligible only with zero OOM and retractions, KV capacity for
`concurrency × 40,928`, and p99 ITL no greater than twice Static-c1. The
eligible load with highest decode goodput is fixed for all three methods.
Selection also requires no more active requests than the 32-prompt confirmation
window can supply; the c48 screen remains a capacity diagnostic rather than an
underfilled formal load.

Tuning searches drafter Full and LoRA across Adam, AdamW, SGDm, NAG, Muon, and
Lion with optimizer-specific learning-rate ranges, rank, stride, weight decay,
and global gradient clipping. Muon's matrix step and auxiliary AdamW fields are
one bound candidate identity; they cannot be selected independently after the
run. Successive halving reads only the tuning window. The final shared
configuration maximizes

\[
\min\left(V_{\mathrm{TTS}}/V_{\mathrm{Static}},
V_{\mathrm{L0}}/V_{\mathrm{Static}}\right),
\]

after exactness and stability checks. Ties prefer lower peak HBM, then lower
p99 ITL, then lower exposed update time. The immutable selection artifact binds
the grid, tuning window, model lock, patched tree, load, and tuning evidence.

For a narrowly scoped reproduction, a registered-grid anchor may instead be
locked from a complete terminal tuning triplet. That artifact is explicitly
labeled `heldout_anchor`: it uses the same independent confirmation and GPU
gate, but makes no claim that the anchor is the grid optimum.

## Independent confirmation

Confirmation has eight repetition blocks. Every block independently resets the
cohort before each method, randomly orders Static/TTS/L0, and submits all 32
distinct prompts once in one ordered native batch request. This removes
host-thread arrival races without throttling the GPU: SGLang's locked admission
limit owns continuous batching, and the engine and cohort are not reset while
the queue drains. A warmup occurs outside the interval. One method/block batch owns the
union of its active decode intervals, excluding request queue and prefill gaps;
repetitions are never merged and batch metrics are never copied onto prompt or
bucket rows.

Request-level ITL, TTFT, output identity, and per-request decode diagnostics are
recorded separately. They do not masquerade as aggregate system goodput. Load
and early tuning stages fill an undersized prompt window round-robin only until
the registered concurrency is occupied; confirmation and natural-task runs
never replicate prompts to manufacture load.

The registered controlled profile is greedy. This makes the target-token
trajectory identical across Static and every exact adapted method, so a paired
timing effect cannot be caused by method-dependent random-number consumption.
Tuning and confirmation store only a format-tagged SHA-256 of each complete
output-token-ID trajectory and fail closed unless every paired method has the
same digest. A decoded-text digest is explicitly insufficient. Formal collection
additionally requires one 32-prompt target-only greedy reference captured from
the locked target snapshot at the same load and safe context limit. Every
method/block digest must match that reference; cross-method agreement by itself
is insufficient. The reference, derived table, and attestation bind the same
reference SHA-256, and neither generated text nor raw token IDs are retained in
evidence artifacts.
Stochastic coupled-RNG and distribution checks remain mandatory GPU tests;
stochastic natural-task runs are robustness side tables, not the causal speed
headline.

Every complete run has a terminal receipt binding its normalized Parquet
shards. Resume reuses only identity-matched receipts. Interrupted shards are
not evidence, so a stopped multi-hour study can continue without duplicating
or silently dropping completed cells.

## Metrics and inference

Headline metrics are paired batch decode-goodput effects for one dedicated
`long_region` row spanning 16K-to-limit and for the full trajectory. Position
buckets remain explanatory and are never averaged into the headline. TTS and L0 are evaluated separately against
Static using repetition-block BCa 95% intervals. The repetition block is the
independent timing and randomization unit; treating 32 requests that share one
wall-clock interval as 32 independent goodput samples would be
pseudoreplication. Each method must reach the registered mean threshold and
have a confidence lower bound above zero. A third paired interval compares L0
directly with TTS. The fixed zero-margin contract requires L0's mean relative
goodput and its BCa lower bound to be non-negative; a run cannot pass merely
because both methods separately beat Static.

Interpretation retains survival-weighted accepted prefix, committed and
verified drafts per verification, verification waste, target calls per output
token, TTFT, ITL percentiles, CUDA lane times, exposed update time, overlap,
graph replay, batch fill, queue occupancy, peak HBM, KV and optimizer memory,
loss, and trainable parameter count. Target-only estimated MFU and profiler
device utilization use different names.

Optimizer memory is reported by category rather than inferred from parameter
count. This distinguishes two-moment Adam/AdamW, one-moment SGDm/NAG/Lion, and
Muon's matrix momentum plus non-matrix auxiliary AdamW state.

Adapted runs also retain one semantic record per verification round, including
the real `prefix_len_before`, valid verified/accepted/committed counts, proposal
source version, and frozen-KV version segments. Static enables only aggregate
study counters and never allocates adaptation trace buffers.

Detailed profiling is a separate run; headline decoding never synchronizes
once per round. Acceptance alone is not a speed claim.

## OnlineSPEC comparison protocol

OnlineSPEC is registered as an important comparison with a separate manifest
and evidence namespace. It uses the controlled tuning and confirmation windows
but cannot consume core TTS/L0 tuning rows or confirmation outcomes. Its three
learners are reduced independently during successive halving. OnlineSPEC owns
a manifest-bound long-trajectory schedule of 2/16K, 4/24K, 8/32K, and
16/40,928 prompt/context pairs; it does not reuse the core study's 4K/8K early
stages. One safe configuration per learner is then compared with a paired
Static reference.

Confirmation uses 32 held-out prompts submitted once per method to one SGLang
queue, eight randomized blocks, identical seeds, one locked concurrency, and
the same 16K-to-safe-limit region. Aggregate goodput is inferred over the eight
independent method/block active-interval unions; prompt-level rows remain
diagnostic.
The derived table reports OGD, optimistic OGD, and Hedge separately; it never
collapses them into a best-of-baselines result. Each comparison includes paired
BCa intervals, safety counters, update counts, HBM categories, and the
learner-specific diagnostics defined in [OnlineSPEC
baseline](onlinespec-baseline.md).

The comparison has its own content-bound GPU attestation. Without it the result
is `UNMEASURED`. With it, the result is still diagnostic and cannot alter the
core formal gate or its selected configuration.

## Formal gate

Both adapted methods must independently show at least three percent mean
goodput improvement over Static, with paired BCa 95% lower bounds above zero.
Exactness violations, version mismatches, fallbacks, non-finite updates, OOM,
and retractions must all be zero; adapted runs must launch and publish updates.

The gate returns `PASS` only with an attestation binding the manifest,
selection, exact evidence bytes, target-only reference, model revisions,
patched SGLang tree, and GPU hardware report. An unattested calculation is
always `UNMEASURED`; a measured
run that misses any criterion is `BLOCKED`. This repository contains protocol
code, not result artifacts or performance claims.
