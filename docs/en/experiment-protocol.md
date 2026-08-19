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

All formal industrial GPU outcomes are `UNMEASURED`. The staged registry, CPU
tests, and runtime contracts do not by themselves authorize a benchmark result.
The trusted single-operator path may collect complete empirical evidence from
one clean Git HEAD/tree, the complete semantic SGLang patch series, a schema-v5
trusted ProtocolLock, exact external-source locks, and fresh dynamic GPU
qualification. Without a current root-authorized attestation, that evidence is
`trusted_single_operator_empirical_no_signature`, has
`formal_measured=false`, and remains `UNMEASURED`. A release-level `MEASURED`
outcome additionally requires the repository-owned Ed25519 root chain. Until
the relevant run-specific receipts exist, each assignment remains fail-closed
before allocation. CPU state-machine tests, caller-authored digests, and an old
smoke result cannot promote TP2/DP2, native ITL, DSpark, NEXTN, or EAGLE3 to
formal support.
Historical shared-tuned-AdamW rows are matched-recipe publication-policy
diagnostics, not TTS-paper reproduction. They are regression/debugging material
only and are excluded from schema-v3 selection, power sizing, confirmation,
and claims.

## Immutable dependency DAG

The registry fixes this order:

```text
preflight -> E3a -> TTS-Cal -> E1 -> E2 -> E4 -> E3b -> E1a -> E5 -> E6 -> E0
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
| TTS-Cal | 288 disjoint tuning-only rows: 9 learning rates x 8 strides x 4 excluded pilots | one frozen TTS recipe |
| E1 | exactly 68 rows after E3a/TTS-Cal sealing | safe LC Pareto set and common load |
| E2 | four successively materialized rounds; `n0=105g`, `n(k+1)=max(ceil(nk/4),21)`, plus four anchors per round | one sealed LightCone recipe |
| E4 | 48 strength-2 screen rows, 96 local-factorial rows, and 3 isolated profiler rows | mechanism gate |
| E3b | `480B` long-context rows, where `B=4+N` and `N` is a sealed 12--20 final prefix | long-context confirmation |
| E1a | 58 configurations x 2 verification modes = 116 rows | one DSpark recipe |
| E5 | `450B+264` rows: powered headline families plus the locked 11 faults x 2 backends x 3 topologies x 4 cohort-count one-shot diagnostic matrix | production and topology surfaces |
| E6 | `2+60B` rows over exactly two NEXTN target models | native MTP transfer surface |
| E0 | 108 signed compatibility decisions, then `16VB` rows for the `V` valid model/backend/task combinations | OnlineSPEC breadth surface |

E1 has exactly 68 rows: Target-only, Static, frozen TTS, frozen L0-naive, and
two optimizer anchors under each of 32 Full/LoRA LightCone geometries. The
L0-naive row is a mechanism anchor and cannot enter candidate ranking. E2 tunes
only LightCone candidates. For every surviving E1 geometry it registers seven
optimizers (`Adam`, `AdamW`, `SGDM`, `NAG`, `Muon`, `Lion`, and project-owned
`ChronoBelief`), three schedules, and five learning rates, hence 105 recipes
per geometry in round zero. Each later round is an exact subset of its sealed
predecessor; no candidate may re-enter. Four fixed role anchors are added to
each round, so the complete E2 count is `16 + sum(n0..n3)` rather than an eager
sentinel matrix.

The TTS reconstruction authority fixes Adam `(beta1=.9, beta2=.999,
epsilon=1e-8, weight_decay=0)`, one update step, full-drafter training, no
clipping, latest-round-only supervision, drafter-native position weights, a
source-point proximal term, request reset, and a side stream. TTS-Cal crosses
learning rates `1e-7, 3e-7, ..., 1e-3` with strides
`1,5,10,15,20,30,40,50` over four excluded pilots. Selection is safety-first,
then maximum SLO-goodput, and is signed before E1. TTS publishes at a fixed
barrier; L0-naive consumes the identical frozen recipe/candidate bytes and
changes only publication to first-ready. Neither may inherit an E2 winner.

ChronoBelief is a project-owned preregistered optimizer. Its identity binds the
four update equations, PDF/TeX digests, standard update-count bias correction,
decoupled weight decay, and safe-boundary age `d_r`. Skip and abort leave
moments and the update counter byte-identical; only a committed proposal may
advance optimizer state.

E1a has exactly 58 configurations: 56 adaptive configurations plus two fixed
references. Its 32 layer-only adaptive configurations cross
`last1/last3/last5/all` with Full plus seven LoRA ranks while freezing DSpark
native heads. Its 24 hybrid cells cross `last1/last3/last5` with the same eight
backbone parameterizations and additionally train native W1, W2, and scalar
acceptance/confidence state as Full. Every configuration runs once with a fixed
verification budget and once with the native scheduler, yielding 116 rows.

These are signed staged materializations, not instructions to expand a future
matrix. E1 consumes sealed E3a and TTS-Cal receipts and materializes exactly
one 68-row width/load slice. E2 materializes only one
successive-halving round at a time, ranks each LC-candidate against fixed Static
and frozen-TTS references, preserves family floors, and derives the next round
only from the prior sealed survivor receipt. Only the final tuning-only E2
receipt can seal the recipe named LightCone.
Formal E2 reduction requires native per-token production/commit timestamps.
Coalesced SSE chunks are never divided into invented intervals; absent or
incomplete native timestamp coverage blocks the reducer.
Confirmation materialization is family-local: four excluded pilots are reduced
before confirmation is visible, then only a sealed 12--20-block final prefix is
activated. A family remains the incremental scheduling/power unit, but a stage
dependency receipt is issued only from a separate exact-coverage aggregate:
families are SHA-sorted and replayed from raw pilot/final completion authority.
E5's 264 deterministic failure injections are one-shot diagnostic passes;
they are not multiplied by confirmation blocks. Selected p99 anchors require
at least 10,000 completed requests. The pilot reducer seals each selected
backend/topology family before final unblinding; all five paired method cells
in that existing family then consume the same exact 11,000-offer extension
pool in every included block. The extension creates no extra materialized row.
Headline families and the 264 failure diagnostics form the exact signed
materialization.

E3b materializes 480 rows per block. E5 materializes 450 headline rows per
block plus 264 one-shot failure diagnostics; the sealed selected-p99 set only
adds execution requirements to matching headline cells.
E6 has one globally deduplicated immutable-interface preflight per target model
and 60 rows per block over exactly `Qwen/Qwen3.6-35B-A3B` and
`Qwen/Qwen3.5-122B-A10B-FP8`. Its source-owned producer derives exactly two TP2
launches from the built-in `mtp.*` component of those same frozen target
snapshots; a separate draft-model path is forbidden. TTS always uses its frozen
external recipe; LightCone uses the sealed E1/E2 winner without E6 retuning.

E0 first publishes exactly 12 source-owned pre-probe interfaces for the four
models by three backends. Those interfaces contain no pre-existing task proof.
The physical campaign then executes 12 launch groups, each covering the nine
task-native probes, to produce the exact 108 model/backend/task compatibility
terminals. A task-specific EAGLE3 proof row may be published only after its
successful one-request core evidence exists. Only the resulting `V` valid
decisions materialize serving rows: eight roles, task-native requests,
concurrency one and common SLO load, giving `16VB`. An all-N/A compatibility
receipt legitimately emits zero timing rows.

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
methods consume the same pool and client ordering. Pairing is bound to the
digest of that complete registered source pool, not to a method's realized
offered prefix or response tokens. Observed outputs remain in the evidence and
a separate exactness gate compares every request completed by two methods.
Every actual offer is accounted exactly once as rejected, completed, timed
out, cancelled, or unfinished. Unfinished work remains in the denominator
through its registered timeout boundary. Missing rows are not filled with zero.

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

Disk admission uses a signed stage-specific capacity envelope. It combines
static input sizing, compile/cache high-water, evidence high-water, retry
reserve, and a safety margin against the exact dispatch schedule. The legacy
100 GB free-space rule is used only when that authority is absent or
unavailable; a bad signature or tampered envelope never falls back. Additional
disk allocation is not an automatic pass, and admission never deletes user
data.

Exclusive-host classification is not execution authority. COMPILE and DOWNLOAD
use distinct first-party runners. Their typed assignments bind budget,
inventory/GPU UUIDs, source checkout, prepared model/tokenizer content,
RunConfig, launch argv/ports, deterministic prewarm or download plan, and
graceful shutdown. Each runner publishes an immutable terminal and atomic
no-replace result pointer; resume reopens every named raw file and sidecar.
Missing dynamic control, partial terminal coverage, or a cache-pointer mismatch
blocks activation.

Each dispatch plan binds the exact `ExperimentBudget` digest for every
assignment. Wall time, requested-gang compute GPU time, reserved GPU time, and
fixed-instance billed GPU time remain separate; a two-GPU gang consumes twice
its wall time, while whole-instance billing charges the entire frozen inventory
for the observed wall interval. A queue is data, not an instruction to launch
every assignment simultaneously.

`RunConfig` supports only `tp1_dp1`, `tp2_dp1`, and `tp1_dp2`, on one host and
at most two ranks. TP2 requires all-rank prepare/decide/apply/receipt evidence
for one publication identity and proves one-rank abort produces zero partial
publication. DP2 is two TP1 replicas behind a sticky cohort router and forbids
cross-replica gradient averaging. Neither topology becomes executable from CPU
state-machine coverage: a fresh root-authorized mode-specific GPU qualification
receipt must bind the exact patched tree and two-GPU inventory.

Multi-host control distributes independent host-local work through
content-bound requests and receipts. No stage claims cross-host collectives,
world size greater than two, Kubernetes scheduling, elastic membership, or
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

The pinned source produces content-bound all-reset capability, initial-state,
and boundary receipts. The receipt covers drain, KV/prefix, RNG/counter,
scheduler/telemetry, adaptation state, allocator/HBM, and completion events.
GPU reuse remains disabled until the exact identity passes reset/address/value/
HBM qualification; a verified proof enables only the measured identity and
topology. On the supported single-tokenizer HTTP/1.1 uvicorn
paths, continuous HTTP accounting comes from real protocol connection creation/
closure events and remains bound to one HTTP process generation with monotonic,
conserved totals; request counters and client fields are rejected as
substitutes. Granian HTTP/2 and multiple-tokenizer HTTP-process paths fail
closed before producing this capability. The terminal envelope binds warm-up,
trace, reset, close, and continuous launch-to-termination inventory accounting.
If any receipt is absent or the dynamic proof does not match, the executor
falls back to a distinct clean process and HTTP pool per trace and records no
startup-saving claim.
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

The trusted single-operator final audit may close complete, reproducible
empirical evidence without a signature, but it must label that closure
`trusted_single_operator_empirical_no_signature` and keep
`formal_measured=false`. Promotion to `MEASURED` additionally requires an
attestation binding the registry or manifest, selections and stage receipts,
exact runtime/capability and patched tree, model/tokenizer/data/trace
identities, the Target-only reference, hardware and power report, and every
final Parquet digest. Without it, status is `UNMEASURED` even when the empirical
campaign and local arithmetic are complete. Valid attested evidence that fails
a registered criterion is `BLOCKED`. The repository contains protocol code,
not performance claims or result artifacts, and this release contains no GPU
results.
The repository contains one offline Ed25519 public root and its fingerprint.
Private signing material never belongs in the repository, remote instance,
argv, environment, logs, or experiment artifacts. Root-authorized deployment
policies bind control type, nonce freshness/replay, exact inventory/hardware,
validity, and source lineage. Dispatch, compile, non-serving terminal,
capacity, interference, and rank aggregate controls share that policy and one
atomic replay reservation. A content-consistent caller-authored key or bundle
cannot authorize the formal DAG.

The protocol labels code-only closure `CPU_READY`, a prepared but unexecuted
device check `GPU_SMOKE_READY`, and only qualifying attested terminal evidence
`MEASURED`. These labels are not successive claims: a CPU or diagnostic smoke
success remains non-measured.
