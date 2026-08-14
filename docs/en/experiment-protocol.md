# Experiment protocol

[中文](../zh-CN/experiment-protocol.md) · [Home](../../README.md)

## Question and current status

The industrial study asks where online drafter adaptation helps, why it helps,
and when its cost or operational risk outweighs saved target work. The formal
roles are Target-only, Static, TTS, L0-naive, and LightCone. Recipe authority
and publication policy are orthogonal: TTS uses a frozen primary-source recipe
with a fixed barrier; L0-naive uses that authority with first-ready publication;
LC-candidates and the E2-sealed LightCone winner use the L0 policy with search
and sealed recipes, respectively.

All formal industrial GPU outcomes are `UNMEASURED`. The code, CPU tests, and registry
establish target protocol and coordinator contracts, not a fully runnable
speculative surface or a benchmark result. The industrial executor currently
runs only TP1/DP1 Target-only. Static/TTS/L0-naive/LightCone are `BLOCKED` before mutation
because the pinned native
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v1`
begin/reset/finalize hook has no configured trusted hardware signer. Stage B is
also blocked on immutable model/data/trace locks, provider credentials,
registered hardware, GPU smoke, and the exact interference envelope.
Historical shared-tuned-AdamW rows are matched-recipe publication-policy
diagnostics, not TTS-paper reproduction. They are regression/debugging material
only and are excluded from schema-v3 selection, power sizing, confirmation,
and claims.

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

Same-host interference calibration uses paired isolated and simultaneous
Static blocks. Both committed-token goodput and raw within-request p99 ITL must
have at most a 1% paired mean relative difference, and each paired BCa 95%
interval must include zero. Missing per-token timestamps remain UNRESOLVED;
request latency and evenly distributed SSE gaps are not ITL substitutes. A
passing rule authorizes only the measured cardinality and still requires the
release-owned trusted-attester chain. In its absence, headline scheduling stays
serial and formal measurements remain blocked.

## Stages and locked decisions

| Stage | Purpose | Output locked before downstream use |
|---|---|---|
| Preflight | source/runtime/model/data identity, exactness, HBM, telemetry, inventory/topology, audit-only session-reset schema, cache/HTTP/writer, and interference calibration | runtime envelope |
| E3a | Target-only/Static context, regime, concurrency, and draft-width capacity | reference load, matched width, crossover and drift witness |
| E1 | 1,428 templates; LC-candidate geometry plus stage-level frozen TTS/L0-naive anchors | safe LC Pareto set and common load |
| E2 | 11,920 templates; LC-candidate optimizer/schedule successive halving only | one sealed LightCone recipe |
| E4 | cumulative systems mechanisms and isolated profiling | mechanism gate |
| E3b | 11,520 long-context cells over the five formal roles | long-context confirmation |
| E1a | native DSpark transfer and retuning | one DSpark recipe |
| E5 | 11,064 production/failure cells over the five formal roles | production and topology surfaces |
| E6 | 3,747 NEXTN transfer cells with frozen TTS and sealed LightCone recipes | native MTP transfer surface |
| E0 | 2,144 structural cells: 864 compatibility templates plus 1,280 selected-core anchors | blocked, non-formal breadth surface |

Each activated E1 width/load slice has 68 cells: two fixed references, two
L0-policy LC-candidate optimizer anchors under each of 32 Full/LoRA geometries,
and one stage-level frozen anchor each for TTS and L0-naive. The 21 possible
slices form the 1,428-template envelope. E2 tunes only LC-candidates. In each
of four rounds it carries 2 fixed references, 32 geometries times 93 search
recipes, and one stage-level frozen anchor for each baseline, for 2,980 cells
per round and 11,920 templates total. The E1 selector consumes a 67-cell
subset and the E2 selector a 2,979-cell subset: Target-only exactness, Static,
frozen TTS, and LC-candidates. The separately planned L0-naive anchor is reserved
for mechanism/decomposition evidence and is excluded from candidate safety,
Pareto, and ranking inputs. E2 keeps optimizer-specific fields and the schedules `constant`,
inverse-square-root by published update, and cosine-to-zero as separate
identities. ChronoBelief declarations are explicitly `BLOCKED`: no authoritative
update equation or source identity is registered, and substituting another
optimizer is forbidden.

The TTS anchor is `TTS-paper-reconstruction`. Its update-recipe authority fixes
Adam, one step, latest-round-only supervision, position-weighted distillation
plus a source-point proximal term, request reset, and strided side-stream
execution. Orthogonally, TTS publication uses the paper's fixed barrier. Its
undisclosed numeric optimizer/loss/trainable/stride fields
keep both frozen TTS and L0-naive anchors `BLOCKED`. They never inherit a search
candidate, schema default, historical AdamW recipe, or result-derived winner.

E1a has exactly 56 adaptive configurations. Its 32 layer-only cells cross
`last1/last3/last5/all` with Full plus seven LoRA ranks while freezing DSpark
native heads. Its 24 hybrid cells cross `last1/last3/last5` with the same eight
backbone parameterizations and additionally train native W1, W2, and scalar
acceptance/confidence state as Full. Fixed verification budget is a tuning
control; the transferred candidate must also survive the native scheduler.

These are registered scientific envelopes, not instructions to execute every
template and not current release support. E1 activation consumes the sealed
E3a selection and materializes exactly one 68-cell width/load slice; every
other E1 template receives an immutable disposition. E2 materializes only one
successive-halving round at a time, ranks each LC-candidate against fixed Static
and frozen-TTS references, preserves family floors, and derives the next round
only from the prior sealed survivor receipt. Only the final tuning-only E2
receipt can seal the recipe named LightCone.
Formal E2 reduction requires genuine per-token observation times. The official
SGLang SSE client may coalesce several token IDs in one chunk, so the adapter
records those token timestamps as unavailable rather than inventing evenly
spaced ITLs. E2 therefore remains `BLOCKED` until a native per-token timestamp
hook is bound or one-token-per-chunk delivery is proven by the pinned runtime.
Confirmation materialization is family-local: four excluded pilots are reduced
before confirmation is visible, then only a sealed 12--20-block final prefix is
activated. A family remains the incremental scheduling/power unit, but a stage
dependency receipt is issued only from a separate exact-coverage aggregate:
families are SHA-sorted and replayed from raw pilot/final completion authority.
E5's 264 non-family failure-injection cells use a deterministic auxiliary
activation/completion, and the family plus auxiliary dispositions must be an
exact disjoint cover of the stage. The pinned patch still rejects
DSpark/EAGLE/EAGLE3/NEXTN adaptation and all multi-rank execution. TP1/DP1
DFlash implements the registered E2 schedules and logical delays, while a real
quota-shadow acquisition need remains a named backend-capability block. This
path cannot produce a claim without the out-of-band trusted signer.

E3b contains 24 blocks by five roles, eight contexts, three regimes, two loads,
and two width panels (11,520 cells). E5 contains 24 blocks over its registered
production panels and 264 failure injections (11,064 cells). E6 contains three
preflights; four headline roles over three models, two tasks, three contexts,
two loads, and 24 blocks; plus L0-naive mechanism anchors for LiveCodeBench at
16K/32K and both loads (3,747 cells). E6 already has one TTS role: it always
uses the frozen external recipe, never an E2-derived recipe. LightCone
uses the sealed E1/E2 winner without E6 retuning. E0 has 2,144 structural cells:
864 compatibility templates (`4 models x 3 backends x 9 tasks x 8 roles`) plus
1,280 preregistered selected-core anchors (`4 models x 1 DFlash backend x 2
tasks x 2 loads x (4 excluded pilots + 12 final blocks) x 5 formal roles`).
Both surfaces remain blocked and non-reportable until their exact recipe, load,
repetition, and evidence authorities are sealed.
The 24-block axis explicitly separates four excluded pilots from 20 potential
final paired blocks at each registered load. A family reports only the sealed
12--20 final prefix, so every reported E6 contrast has a valid 95% interval.

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

`GpuFleetInventory` composes multiple independently verified hosts above that
same-host scheduler. It balances independent cells and whole confirmation
blocks across eligible hosts while preserving host-local inventory,
interference, port, cache, evidence, and materialization identities. Matched
baseline anchors and complete confirmation blocks stay on one host/GPU. A host's concurrent headline work is limited
only by that host's calibrated envelope; heterogeneous hardware envelopes do
not enter one family. Cross-host gangs are rejected with
`cross_host_collectives_unvalidated`.

Before concurrent work, preflight records clocks, temperature, power state,
background processes, driver/runtime identity, topology, per-rank HBM, and an
exact `InterferenceEnvelope`. The envelope is keyed by hardware, workload,
co-run signature and count, gang shape, thermal/power/load state, and host
contention. If only two-way concurrency was calibrated on an eight-GPU host,
the frozen headline waves remain two-way. Runtime completion never creates
result-dependent co-tenancy. Profiler, download, and compile work is
exclusive-host and cannot contend with headline timing.

Exclusive-host classification is not execution authority. This release does
not dispatch COMPILE or DOWNLOAD cells: compile lacks an exact release-owned
prewarm/finalization manifest and atomic cache-result pointer, and download
lacks a first-party terminal receipt contract.

A future compile contract must bind the assignment, budget, inventory and GPU
UUIDs; compile-plan/key and model revisions; TP, context, concurrency, and graph
buckets; deterministic prewarm payloads; and a graceful-shutdown acknowledgement.
Its atomically published result pointer must name and hash the manifest, attempt
receipt, final cache receipt, and immutable cache object. Resume must reopen all
of those files rather than accept the pointer's serialized summary.

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

Multi-host control distributes independent host-local work through
content-bound requests and receipts. No stage claims cross-host collectives,
an executable TP2/DP2 path, Kubernetes scheduling, elastic membership, or
automatic failover. Fleet SSH transport concurrency is explicitly bounded. A
failed host preserves completed peer receipts. Any post-dispatch loss of
authority is `REMOTE_OUTCOME_UNKNOWN` and requires an independent
content-addressed reconcile over the exact original destination, port, and
known-host-key authority; endpoint values, key bytes, and credentials are not
persisted. Only a terminal-negative outcome may create a new same-host
receipt-bound attempt. An in-flight attempt cannot migrate silently.

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
estimate the log-effect variance for LightCone--TTS and LightCone--Static. The identity binds
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
LightCone--TTS and LightCone--Static and uses Holm family-wise adjustment. The
preregistered secondary decomposition is L0-naive--TTS and
LightCone--L0-naive. Other secondary breadth hypotheses are grouped explicitly
and use Benjamini--Hochberg FDR; they cannot be promoted into the primary family
after results are known. The registered reporting target includes
method-by-model, method-by-context, and method-by-load interactions, without
assuming that any interaction or large-model effect is positive or choosing a
model from observed LightCone gain.
`CrossFamilyInteractionReducerArtifact` currently records only a content-bound,
structural, non-formal `UNRESOLVED` contract. It does not prove completed GPU
coverage, interval validity, or a formal interaction effect; those require the
registered statistical reducer to consume complete attested final evidence.

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
automatically aliasable and adaptive scientific roles are never aliases. The dependence map makes
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
Each execution plan also binds the producer queue and batch limits, writer
queue and Parquet row-group limits, checkpoint interval, overflow mode, SQLite
WAL/FULL settings, and file/checkpoint/directory fsync gates. The checkpoint,
prepared receipt, terminal receipt, and resume path must agree on that policy.

The pinned source now produces content-bound all-reset capability, initial
state, and boundary receipts. The receipt proves the registered drain,
KV/prefix, RNG/counter, scheduler/telemetry, adaptation-state, allocator/HBM,
and completion-event predicates at the CPU contract boundary. Its GPU reset
semantics remain `PENDING`. On the supported single-tokenizer HTTP/1.1 uvicorn
paths, continuous HTTP accounting comes from real protocol connection creation/
closure events and remains bound to one HTTP process generation with monotonic,
conserved totals; request counters and client fields are rejected as
substitutes. Granian HTTP/2 and multiple-tokenizer HTTP-process paths fail
closed before producing this capability. The receipt is not yet durably bound
into the terminal envelope or continuous launch-to-termination inventory accounting.
The source patch covers reset-state accounting only and does not yet produce
native warm-up, trace, or close receipts. All live GPU reuse entry points
therefore remain blocked and fall back to a
distinct clean process and HTTP pool per trace; no reset or startup-saving
claim is recorded. Within each supported single-trace path, submit and abort
share that official HTTP pool, with timeouts bound to the request deadline and
abort grace.
Immutable compile-cache bases remain content-addressed and verified, each
process gets a private writable overlay, and CUDA Graphs never cross a process
or GPU. Fault-injection cells always use a fresh process in this release.

The native hook now binds capability, begin, reset, finalize, exact terminal
request coverage and ordered token IDs, plus Static aggregate safety or adaptive-role
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
industrial or legacy analyzers to `MEASURED`, and Static/TTS/L0-naive/LightCone stay blocked
before mutation despite the implemented hook.

The external `TrustedAttesterPolicyBundle` format binds public verification
keys, nonce freshness/replay policy, hardware-envelope allowlists, and validity
to a separately provisioned anchor. Private signing material never belongs in
the repository or experiment artifacts. The source-release anchor is currently
unconfigured, so loading an operator bundle cannot by itself authorize the
formal DAG.

The protocol labels code-only closure `CPU_READY`, a prepared but unexecuted
device check `GPU_SMOKE_READY`, and only qualifying attested terminal evidence
`MEASURED`. These labels are not successive claims: a CPU or diagnostic smoke
success remains non-measured.
