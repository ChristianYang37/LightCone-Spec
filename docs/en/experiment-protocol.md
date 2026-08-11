# Experiment protocol

[中文](../zh-CN/experiment-protocol.md) · [Home](../../README.md)

## Question and current status

The industrial study asks where online drafter adaptation helps, why it helps,
and when its cost or operational risk outweighs saved target work. Target-only,
Static, TTS, and L0 are kept distinct. TTS and L0 use the same candidates and
differ only in publication time.

All new GPU outcomes are `UNMEASURED`. The code, CPU tests, and registry
establish target protocol and coordinator contracts, not a fully runnable
speculative surface or a benchmark result. The industrial executor currently
runs only TP1/DP1 Target-only. Static/TTS/L0 are `BLOCKED` before mutation
because the pinned native
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v1`
begin/reset/finalize hook has no configured trusted hardware signer. Stage B is
also blocked on immutable model/data/trace locks, provider credentials,
registered hardware, GPU smoke, and the exact interference envelope.
Historical v2 evidence is regression/debugging material only and is excluded
from schema-v3 selection, power sizing, confirmation, and claims.

## Immutable dependency DAG

The registry fixes this order:

```text
preflight -> E3a -> E1 -> E2 -> E4 -> E3b -> E1a -> E5 -> E6 -> E0
```

Each target definition names dependencies, locked outputs, and scientific
axes. Each cell then binds its complete identity: experiment/model/backend/
task/method, parameterization and native scope, optimizer and schedule,
context/regime/width/arrival/SLO, cohort/topology, seed/block, two logical rank
slots, logical port/cache/evidence claims, workload class, and truthful
status/reason. A separate frozen physical assignment binds inventory UUIDs,
rank layout, concrete ports, topology, exact per-cell budget, and whole-instance
billing; changing hosts does not rewrite the scientific registry.

A stage could dispatch only after exact dependency receipts validate and the
cell passes executable-release preflight. A receipt binds registry, runtime,
split, dependency-output, and locked-output SHA-256 values and asserts that
selection was sealed before downstream unblinding.
Editing a registry artifact or reserializing a dependency under another digest
fails closed.

## Stages and locked decisions

| Stage | Purpose | Output locked before downstream use |
|---|---|---|
| Preflight | source/runtime/model/data identity, exactness, HBM, telemetry, inventory/topology, audit-only session-reset schema, cache/HTTP/writer, and interference calibration | runtime envelope |
| E3a | Target-only/Static context, regime, concurrency, and draft-width capacity | reference load, matched width, crossover and drift witness |
| E1 | DFlash layer scope and Full/LoRA geometry at AdamW/SGDm anchors | safe Pareto set and common load |
| E2 | optimizer, log learning-rate grid, schedule, and successive halving | one DFlash recipe |
| E4 | cumulative systems mechanisms and isolated profiling | mechanism gate |
| E3b | paired long-context Target-only/Static/TTS/L0 confirmation | long-context confirmation |
| E1a | native DSpark transfer and retuning | one DSpark recipe |
| E5 | production arrivals, cohorts, topologies, SLOs, and failures | production and topology surfaces |
| E6 | native NEXTN interface and two-rank fit, then transfer | native MTP transfer surface |
| E0 | model/backend/task breadth including isolated OnlineSPEC | breadth surface |

E1 crosses four native layer scopes with Full plus seven LoRA ranks and two
optimizer anchors: exactly 64 geometry cells before downstream optimizer
search. E2 keeps optimizer-specific fields and the schedules `constant`,
inverse-square-root by published update, and cosine-to-zero as separate
identities. ChronoBelief declarations are explicitly `BLOCKED`: no authoritative
update equation or source identity is registered, and substituting another
optimizer is forbidden.

E1a has exactly 56 adaptive configurations. Its 32 layer-only cells cross
`last1/last3/last5/all` with Full plus seven LoRA ranks while freezing DSpark
native heads. Its 24 hybrid cells cross `last1/last3/last5` with the same eight
backbone parameterizations and additionally train native W1, W2, and scalar
acceptance/confidence state as Full. Fixed verification budget is a tuning
control; the transferred candidate must also survive the native scheduler.

These are registered scientific envelopes, not instructions to execute every
template and not current release support. E1 activation consumes the sealed
E3a selection and materializes exactly one 130-cell width/load slice; every
other E1 template receives an immutable disposition. E2 materializes only one
successive-halving round at a time, preserves matched TTS/L0 pairs and family
floors, and derives the next round only from the prior sealed survivor receipt.
Confirmation materialization is family-local: four excluded pilots are reduced
before confirmation is visible, then only a sealed 12--20-block final prefix is
activated. The pinned patch still rejects DSpark/EAGLE/EAGLE3/NEXTN adaptation,
nonconstant schedules, and all multi-rank execution. Its TP1/DP1 DFlash path
cannot produce a claim without the out-of-band trusted signer.

## Data, contexts, and traces

Controlled prompt windows are content-disjoint and digest-bound. Selection
data cannot enter confirmation. Model and tokenizer revisions, prompt compiler,
task split, maximum context, generation budget, and safe speculative headroom
are immutable inputs. Natural datasets require exact external revisions and
remain outside the repository.

Long-context axes are 1K, 2K, 4K, 8K, 16K, 24K, 32K, and 40,928 tokens across
long-input/short-output, short-input/long-generation, and multi-turn
shared-prefix regimes. DFlash draft widths are 4, 8, and 16. E3b reports a
matched-width panel and a deployment-optimal-width panel separately; changing
width after seeing confirmation is forbidden.

The controlled profile is greedy. Exactness diagnostics hash each complete,
ordered output-token-ID trajectory with an explicit format tag; decoded text
is not an exactness witness. Legacy formal collection also binds a locked
32-prompt Target-only greedy reference captured at the same load and safe
context limit, and every method/block must match it. Cross-method agreement by
itself is insufficient. This reference remains an `UNMEASURED` diagnostic in
the current release and cannot bypass speculative execution or attestation
blocks.

Production traces separate content identity from arrival identity. Open-loop
Poisson, immediate-burst, BurstGPT-shaped, and soak traces bind exact arrival
offsets. Closed-loop runs instead bind a maximum request pool, population, and
per-client order; every realized offer time must follow the prior terminal
event for that client. Each method may consume a different contiguous prefix,
and exhausting any client pool before the fixed arrival window is a
nonclaimable failure. A synthetic BurstGPT-shaped trace is never labelled as
the real dataset without an immutable external corpus digest.

Paired open-loop methods consume identical trace bytes; paired closed-loop
methods consume the same pool and client ordering. Every actual offer is
accounted exactly once as rejected, completed, timed out, cancelled, or
unfinished. Unfinished work remains in the denominator through its registered
timeout boundary. Missing rows are not filled with zero.

## GPU-pool staging

The registry uses two logical rank slots; physical devices come only from a
content-bound `GpuInventory` and frozen assignment. The sole deterministic
same-host scheduler supports arbitrary inventory size, with explicit tests for
1, 2, 4, 8, and 16 GPUs. It allocates a k-GPU TP/DP shape atomically on one
valid topology group, rejects partial gangs and cross-host placement, rotates
independent blocks over eligible UUIDs, and prevents overlap in GPU, port,
cache writer, evidence root, and exclusive resources.

Before concurrent work, preflight records clocks, temperature, power state,
background processes, driver/runtime identity, topology, per-rank HBM, and an
exact `InterferenceEnvelope`. The envelope is keyed by hardware, workload,
co-run signature and count, gang shape, thermal/power/load state, and host
contention. If only two-way concurrency was calibrated on an eight-GPU host,
the frozen headline waves remain two-way. Runtime completion never creates
result-dependent co-tenancy. Profiler, download, and compile work is
exclusive-host and cannot contend with headline timing.

Each dispatch plan binds the exact `ExperimentBudget` digest for every
assignment. Wall time, requested-gang compute GPU time, reserved GPU time, and
fixed-instance billed GPU time remain separate; a two-GPU gang consumes twice
its wall time, while whole-instance billing charges the entire frozen inventory
for the observed wall interval. A queue is data, not an instruction to launch
every assignment simultaneously.

TP2 and sticky-replica DP2 are target coordinator contracts. A future release
would require a verified patched-runtime capability receipt and all-rank
prepare/decide/apply/receipt evidence for one publication identity. The current
`RunConfig` rejects every TP2/DP2 cell before model loading; the CPU `gloo`
harness is only a state-machine test and cannot enable those cells.

No stage claims multi-node, an executable TP2/DP2 path, Kubernetes scheduling,
elastic membership, or automatic failover. Larger inventory support means more
independent TP1/DP1 work, not release support for larger rank groups.

## Production metrics and safety

Every system point reports offered/admitted/terminal request accounting,
throughput and decode goodput, TTFT by prompt bucket, within-request ITL,
completion and error rates, target calls/work, accepted/verified/committed
drafts, update/candidate/publication counts, exposed and overlapped update time,
queue/batch occupancy, HBM categories, energy per output token, and hardware
envelope validity.

The registered production SLO requires TTFT no greater than 2/5/10 seconds for
short/medium/long prompts, within-request p99 ITL no greater than 100 ms,
qualification at least 99%, error at most 0.1%, and completion at least 99.9%.
A p99 claim is `UNRESOLVED` below 10,000 completed requests; it is not estimated
from a smaller sample and presented as qualified.

Exactness violations, duplicate/mixed identities, non-finite candidates,
partial publication, fallback, OOM, retraction, evidence drops, incomplete
terminal accounting, or out-of-envelope hardware observations invalidate the
affected block. Profilers and synchronizing diagnostics run separately with
headline evidence forbidden.

## Power and statistical inference

For every exact `ConfirmationFamilyIdentity`, exactly four paired pilot blocks
estimate the log-effect variance for L0--Static and L0--TTS. The identity binds
experiment/model/backend/task, context/regime/load/arrival, width panel,
topology, cohort and method family, runtime/split/trace/sampling, and hardware
envelope. One family's pilots cannot power another. Pilot IDs are permanently
excluded from confirmation. With family alpha 0.05, the first Holm threshold,
a 3% minimum relative effect, and 80% target power fixed in advance, the reducer
selects the smallest common final-block prefix from 12 through 20 that powers
both contrasts. It seals that prefix before confirmation is visible; if no
count qualifies, status is `UNDERPOWERED` and confirmation cannot start.

Final goodput effects are paired log ratios with 95% BCa intervals over
independent repetition blocks. The primary family contains exactly
L0--Static and L0--TTS and uses Holm family-wise adjustment. Secondary breadth
hypotheses are grouped explicitly and use Benjamini--Hochberg FDR; they cannot
be promoted into the primary family after results are known.

Long-context request-level summaries use hierarchical bootstrap: resample
independent blocks, then requests inside the sampled blocks. Production
arrival experiments resample whole time blocks to preserve within-block tail
dependence. Both use fixed 95% intervals and registered seeds/repetition counts.
Requests sharing one service interval are not treated as independent system
replicates.

A legal evidence alias shares one byte-equivalent Target-only observation; it
does not copy rows. The alias validates model/runtime/tree, sampling/seed,
corpus/trace, limits, hardware/topology/ranks, method/server configuration,
schema, output-token trajectory, and timing contract. Static is not
automatically aliasable and TTS/L0 are never aliases. The dependence map makes
all aliased consumers one resampling/covariance unit. The current formal
reducer fails closed on a non-singleton unit unless execution plans and terminal
evidence recompute the equivalence; a structurally valid self-described alias
is not claim evidence.

Power, clocks, temperature, throttling reasons, and background processes are
locked as a per-block hardware envelope. Missing or out-of-range observations
invalidate the block rather than becoming covariates chosen after the fact.

## Evidence, resume, and claims

Evidence is written incrementally through a bounded single-writer queue to
batched Parquet WAL row groups. Registered row/time checkpoints and terminal
boundaries retain WAL fsync, directory sync, uniqueness, negative-row
durability, and zero-drop coverage; temporarily emptying the queue is not an
implicit fsync boundary. Interrupted and aborted attempts remain inspectable,
but their rows are excluded. Resume skips only a single receipt whose full
identity and file digests validate; competing completed attempts are an error.

The immutable session-key and reset/finalize receipt schemas describe the
evidence a future compatible method/block server reuse path would need. Live
shared-session execution is unavailable in this release: no release-owned
trusted boundary can prove drain/reset/finalize, the session receipts are not
durably bound into a terminal envelope, and whole-inventory accounting does not
yet cover the continuous launch-to-termination interval. All reuse mutation
entry points therefore fail before launch or evidence creation. Within the
supported single-trace path, submit and abort share one caller-owned official
HTTP pool, with timeouts bound to the request deadline and abort grace.
Immutable compile-cache bases remain content-addressed and verified, each
process gets a private writable overlay, and CUDA Graphs never cross a process
or GPU. Fault-injection cells always use a fresh process in this release.

The native hook now binds capability, begin, reset, finalize, exact terminal
request coverage and ordered token IDs, plus Static aggregate safety or TTS/L0
request/round/update/KV/performance evidence. Provider object attributes are not
trust evidence. Every successful serving run must additionally publish a
terminal-bound `BudgetObservationReceipt` covering all registered phases,
measured gang GPU time, whole-instance billed time, and exact deltas. Missing
components remain missing or explicit N/A, never zero.

An empirical gate additionally requires an attestation binding the registry or
manifest, selections and stage receipts, exact runtime/capability and patched
tree, model/tokenizer/data/trace identities, the Target-only reference,
hardware and power report, and every final Parquet digest. Without it, status
is `UNMEASURED` even when local arithmetic is positive. Valid attested evidence
that fails a registered criterion is `BLOCKED`. The repository contains
protocol code, not performance claims or result artifacts, and this release
contains no GPU results.
No trusted hardware-attester identity is configured in this release. Therefore
content-consistent caller-authored attestation files cannot promote either the
industrial or legacy analyzers to `MEASURED`, and Static/TTS/L0 stay blocked
before mutation despite the implemented hook.
