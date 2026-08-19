# Architecture

[中文](../zh-CN/architecture.md) · [README](../../README.md)

## Boundaries and status

LightCone-Spec separates the Python protocol, a semantic mail-patch integration
for a disposable SGLang checkout, and external models/data/run artifacts. The
only runtime source identity is the pinned upstream commit plus the complete
patch series and expected final Git tree. A workspace checkout is never an
implicit dependency.

CPU tests validate contracts, not GPU speed. Every new GPU cell is
`UNMEASURED`. The current end-to-end industrial executor supports only
TP1/DP1 Target-only. The pinned patch now implements the exact
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v1`
begin/reset/finalize hook, and the host validates its content rather than
trusting provider attributes. The release contains no allowlisted, out-of-band
hardware signer, however, so Static/TTS/L0-naive/LightCone still fail closed before mutation.
Empirical Stage B is also `BLOCKED` until immutable model/data/trace inputs,
provider access, registered hardware and interference evidence, and the trusted
signer are available. Historical v2 evidence is usable for regression and
debugging only and cannot support a new claim.

## One decode and candidate lifecycle

Target-only disables speculation and is the only currently claimable path.
Static is an allocation-free native speculative target path, but release
preflight blocks it pending trusted terminal attestation.
Neither method's schema contract imports adaptation state or allocates
optimizer, gradient, candidate, or adaptation trace storage.

TTS, L0-naive, and LC-candidates reuse one bounded candidate-lifecycle
implementation. The pinned patch also contains the registered DSpark, NEXTN,
compatible EAGLE3, TP2, and sticky DP2 source paths. Recipe authority and
publication policy remain orthogonal, and shared machinery never merges live
candidates, optimizer state, configs, or evidence. Every exact path remains
non-claimable until its dynamic GPU qualification and trusted external-control
proof pass:

1. Verification emits one `ProposalEvidence` envelope for the exact proposal
   that was sampled.
2. The cohort keeps only each request's newest legal supervision row and binds
   the batch to cohort, epoch, slot/buffer generation, source round, and source
   adapter version.
3. A side device stream reconstructs the backend-native differentiable
   proposal, applies the selected trainable plan, computes the loss, and makes
   a functional optimizer candidate without mutating active parameters or
   optimizer state.
4. Device checks cover finiteness, reconstruction, and supervision validity.
   Their three-byte receipt is copied asynchronously to pinned host memory and
   read only after a nonblocking ready-event query, outside measured update
   timing. Ready-event identity and reserved scratch bytes are part of the
   candidate.
5. TTS waits for its fixed update barrier. L0-naive, LC-candidates, and
   LightCone publish at the first legal safe graph boundary after readiness.
6. Commit copies a complete candidate into fixed-address inference storage
   exactly once. Stale, duplicate, cancelled, generation-mismatched,
   non-finite, or conflicting candidates are discarded with a terminal reason.

Candidate creation, commit, discard, cancellation, reset, and disable each have
one terminal transition. Only one candidate may be in flight per cohort, so
side-stream work and scratch memory remain bounded.
Candidate equality is checked only by controlled replay with identical
source-state and proposal-evidence digests. Live TTS and L0-naive histories may
diverge after different publication decisions.

## Common backend evidence and reconstruction

`ProposalEvidence` is deliberately small and backend-neutral. It contains:

- adapter-free and deployed proposal logits;
- the normalized corrected distribution actually used to sample;
- semantic valid mask and target teacher rows;
- sampled predecessor token IDs and embeddings;
- optional confidence rows;
- unique request IDs, cohort digest, source version, and a typed backend
  payload.

The target contract requires construction to check shape, dtype, device,
uniqueness, and identity while keeping numerical checks as device predicates.
It requires a backend validator to reconstruct `proposal_logits`,
`corrected_distribution`, and optional confidence, and to reject an adapter
delta when native inference already applied it. The pinned patch implements
this reconstruction for DFlash, DSpark, NEXTN, and an
official-selector-gated EAGLE3 path. These are source capabilities, not
hardware claims: each remains fail-closed until its exact fresh GPU
qualification is verified.

The cross-backend semantics are:

- DFlash binds the deployed differentiable canvas and the sampling-time
  proposal correction; only this backend has a lower-level adaptive patch path.
- DSpark binds real inference-native Markov W1/W2 features, the actual sampled
  predecessor, scheduler mode, proposal distribution, and confidence state.
- EAGLE3 binds tree state, top-k-one execution, and one source version for the
  entire proposal/verification chain, but only for a prepared-model selector
  decision marked compatible. Generic EAGLE remains unsupported.
- NEXTN binds native MTP hidden state and an immutable upstream interface
  digest, plus interface and memory-fit preflight.

DSpark, NEXTN, and compatible EAGLE3 are
`implemented_pending_dynamic_gpu_proof`; absent or mismatched proof is rejected
before worker allocation. EAGLE is unsupported. Historical drafter KV is
detached, immutable, and versioned. An eligible update path may use it as state
but must never rebuild or differentiate it; future KV records the newly
published version.

## Trainable plans

The target adaptation contract has exactly two parameterizations: Full and
LoRA. Every plan lists selected and frozen names, shapes, dtypes, Full/LoRA
coordinate identity, and sharded/replicated ownership; its digest enters
configuration and evidence.
Full keeps FP32 masters for the selected native parameters. LoRA selects only
registered two-dimensional matrices, starts with zero functional delta, uses
ranks 1/2/4/8/16/32/64, and fixes `alpha/r=1`.

DFlash, EAGLE/EAGLE3, and NEXTN use `last1`, `last3`, `last5`, or `all` native
layer scopes. Borrowed target embeddings, LM heads, and target models remain
frozen.

DSpark has the same four layer-only scopes with all native W1/W2/acceptance
parameters frozen. It also has `last1_native_heads`, `last3_native_heads`, and
`last5_native_heads`: selected backbone matrices follow Full or LoRA, while W1,
W2, and the scalar acceptance/confidence parameter are always Full and
replicated. The registered E1a set is exactly

```text
4 layer-only scopes x (Full + 7 LoRA ranks) = 32
3 hybrid scopes     x (Full + 7 LoRA ranks) = 24
total                                             56
```

The plan computes memory from actual coordinates and optimizer state rather
than a fixed multiplier of model size.

## Topology and publication

The schema and coordinator describe one-host TP1/DP1, TP2/DP1, and TP1/DP2.
`RunConfig` accepts a two-rank configuration only with the exact source-owned
`patched_two_gpu_v1` capability identity; formal dispatch additionally requires
a fresh root-authorized dynamic GPU proof. A caller-authored receipt cannot
enable it. The multi-rank identity binds global/local/TP/DP rank, device, node,
process, rendezvous, router, clock, runtime, and model identity.

In this coordinator contract, TP trainable state follows
inference ownership. Sharded parameters remain local; replicated parameters
reduce only within the owning TP replica. DP uses sticky cohort routing: a
cohort stays replica-local and adaptation gradients are never averaged between
replicas.

Distributed publication is two-phase. Each rank prepares the same
retry-stable update/candidate identity and reports source version, epoch,
buffer/optimizer generation, readiness, finiteness, memory reservation, safe
boundary, and process-group health. One all-rank decision either commits
everywhere or aborts everywhere. Post-copy receipts reject partial application.
A collective or split-decision failure makes the service unready, blocks new
admission, and requires an explicit same-topology restart.

`GlooPublicationTransport` is a real two-process CPU harness for this state
machine. It intentionally rejects NCCL and cannot validate CUDA events, graph
capture, fixed-address device copies, GPU numerics, performance, or contention.

## Fleet control plane and release states

`GpuFleetScheduler` is only a host-affinity and serial-partitioning control
layer. Every child placement is delegated to the sole same-host
`GpuPoolScheduler`; fleet composition does not add a second placement
algorithm. A `HostInventoryBinding` joins exactly one host's content-bound
`GpuInventory`
and `InterferenceEnvelope`. `GpuFleetInventory` sorts and deduplicates those
bindings, while each `HostExecutionBinding` assigns port, cache, evidence, and
contention namespaces that are collision-free within that host; literal values
may repeat on different hosts. The remote execution binding then fixes the
host-local materialization manifest. Independent cells are balanced over
eligible hosts, matched-geometry baseline anchors and a complete confirmation block remain on
one host/GPU, and heterogeneous hardware envelopes are never pooled into one
statistical family.

Gang placement is still topology-aware and atomic inside one host. A requested
shape that can be satisfied only by combining hosts fails before dispatch with
`cross_host_collectives_unvalidated`. This is a hard capability boundary:
multi-host inventory and SSH dispatch are not evidence for cross-host NCCL,
TP/DP rendezvous, or model-parallel execution.

The coordinator transmits a canonical, content-bound host request to
`execute-dispatch-wave --host-request-stdin`. `SshHostRoute` remains local and
nonserializable; OpenSSH uses an agent, batch public-key authentication, strict
host-key checking against a fixed file, no forwarding, and bounded stdout,
stderr, and runtime. The worker reopens a host-local materialization manifest
and returns a content-bound response without routing data or raw logs. If a
request may have reached the worker but no authoritative response returns, its
outcome is `REMOTE_OUTCOME_UNKNOWN`, never a retryable failure. Reconciliation
uses the exact original destination, port, and known-host-key authority to
independently fetch the exact receipt envelope and evidence bytes, recomputes
every content identity in memory under strict size bounds, and publishes only a
path-free digest projection. Endpoint values, host-key bytes, and credentials
are never persisted in requests, receipts, evidence, or logs. Completed receipts
stay in the fleet receipt; an in-flight attempt cannot move to another host
under the same identity. A bounded semaphore limits fleet transport concurrency.

The architecture uses three noninterchangeable labels. `CPU_READY` covers
non-device contract validation. `GPU_SMOKE_READY` only marks a bounded device
check as prepared. `MEASURED` requires the registered workload, complete
terminal evidence, hardware envelope, and release-owned attestation. These
labels do not authorize a device or performance claim; each execution path must
close its own runtime, evidence, and attestation gates.

## HBM and cohort governance

The HBM ledger charges active/base state, FP32 masters, gradients, allocated
optimizer moments, metadata, candidate and staging buffers, training
activations, KV gather scratch, graph buffers, telemetry, and safety margins.
Admission is decided by the least-headroom rank; a sum that fits one rank but
not another is rejected before expensive allocation.

Pressure handling follows a fixed order: preserve immutable/runtime
correctness, preserve active KV and published state, evict only native inactive
prefixes, abort pending adaptation and release temporary state, then queue or
reject new work. Optional cold-cohort offload is last, explicit, timed, and
disabled by default. No path silently changes parameterization, precision, or
optimizer.

The cohort manager uses a bounded number of fixed-size slabs with tenant
quotas. Keys include tenant, cohort digest, and replica. Slot and optimizer
generations prevent stale reuse. Only inactive cohorts may be reclaimed or
cold-offloaded; transfer/reclamation receipts preserve slab and byte identity.

## Traces and durable telemetry

Load traces bind independent identities for request content, arrivals,
timeouts/cancellations, and warmup/scored windows. Synthetic Poisson and
immediate-burst generators are deterministic from their seeds and explicitly
labelled synthetic. A BurstGPT-shaped trace is not represented as the real
dataset without an immutable external corpus digest. Paired methods must bind
the same complete registered source-pool digest; a closed-loop method-specific
offered prefix and its observed output tokens do not redefine pairing.
Completed outputs are checked by a separate exactness gate. Every offered
request is accounted as rejected, completed, timed out, cancelled, or
unfinished.

The evidence schema can represent per-process/rank run, request, round, update,
and performance records through a bounded queue. Duplicate primary identities
are rejected by a durable index. Periodic or capacity-triggered flushes create
fsynced Parquet WAL segments and a checkpoint; backpressure and any explicit
drop are counted.
The native terminal lifecycle binds capability, begin, reset, and finalize
receipts to one process/session/run/nonce/plan/rank identity and exact ordered
token IDs. Static retains zero round/update detailed-trace allocation and emits
only the required aggregate speculative safety/accounting. Adaptive scientific roles additionally
bind request, round, update, KV-version, publication, performance, and safety
rows. The release verifier still blocks all three methods before mutation when
the signer is unavailable.

The repository defines immutable session-key and reset/finalize receipt data
contracts for compatible adjacent traces. The complete key includes
source/capability, run configuration, model/drafter revisions, method/backend,
topology and physical GPU UUIDs, memory/graph/telemetry configuration,
compile-cache identity, and ports. The pinned SGLang source now implements the
all-reset capability/state/receipt producer. On the supported single-tokenizer
HTTP/1.1 uvicorn paths, it now binds cumulative HTTP process/generation/
created/closed/current totals measured at the real transport lifecycle;
Granian HTTP/2 and multiple-tokenizer HTTP-process paths fail closed before
producing that capability. Its GPU semantics remain `PENDING`, and durable
terminal plus continuous whole-inventory accounting from launch through
termination are absent. This is a reset-state accounting slice only; the patch
does not produce native warm-up, trace, or close receipts. This release
therefore does **not** execute or claim a shared server session. Every shared-session
mutation entry point therefore fails before launch, network access, reset, or
evidence-root creation with
`shared_session_gpu_reset_and_durable_terminal_authority_unavailable`.
The high-level block executor handles this state by validating all members and
then using one clean process, HTTP pool, native lifecycle, terminal receipt, and
budget observation per trace. It does not claim reuse. Single-trace execution
remains the only claimable Target-only path.

On successful close, WAL segments are coverage-checked and assembled into
process-unique Parquet shards and a durable prepared receipt. The executor then
finishes the native terminal lifecycle and publishes the mandatory budget
observation bound to that prepared receipt's exact bytes. Only after both exist
does it publish a distinct exclusive terminal envelope. That final envelope
retains the prepared schema, row-group coverage, counters, file sizes, schema
digests, and SHA-256 values, and additionally binds the prepared receipt's raw
digest and size plus the budget observation's semantic digest, raw receipt and
sidecar digests, sizes, and safe paths. A crash between observation and terminal
publication can construct the final envelope from the validated prepared
receipt without re-execution; a prepared receipt without its observation is
nonclaimable. Interrupted or aborted WALs remain inspectable but are excluded
from analysis. A completed receipt is never overwritten or combined with a
competing attempt.

## Industrial registry boundary

The immutable registry declares the order
`preflight -> E3a -> E1 -> E2 -> E4 -> E3b -> E1a -> E5 -> E6 -> E0`.
Every cell binds axes, seed, status/reason, two logical rank slots, ports,
cache/evidence roots, and workload isolation. Physical UUIDs are never registry
identity: a content-bound inventory and deterministic frozen assignment bind
UUIDs, rank layout, topology, ports, the complete per-cell `ExperimentBudget`,
and whole-instance billing before launch.

One same-host `GpuPoolScheduler` is the sole scheduler. It accepts arbitrary
inventories and has explicit regression coverage for 1, 2, 4, 8, and 16 GPUs;
gang placement is atomic and topology-aware, and each headline wave is frozen
before execution. Concurrent work is bounded by an exact
`InterferenceEnvelope`, not by the number of idle devices. Resources, ports,
cache writers, and evidence roots cannot overlap, and receipt-only resume never
infers completion from a directory. Dispatch writes a path-bound WAVE/INTENT/
FINISH hash chain before and after each runner; partial sibling failures retain
successful raw terminal authorities, and an unfinished intent blocks rather
than inventing retry cost.

Reducer-owned, signed materialization consumes the sealed upstream receipts
and creates exactly one 68-cell E1 stage: Target-only, Static, frozen TTS,
frozen L0-naive, and 64 LightCone candidates over 32 geometries and two
optimizer anchors. L0-naive is a mechanism anchor and is not ranked as a
candidate. For each of the `g` surviving geometries, E2 materializes
`n0 = 105g` recipes (seven optimizers by three schedules by five learning
rates), then three successive-halving rounds with
`n(k+1) = max(ceil(nk/4), 21)`; every round also carries four fixed anchors.
There is no eager sentinel matrix. An exact sealed E2 final-recipe receipt is
the only path to the LightCone role. Confirmation planning is family-local:
exactly four
excluded pilots select `POWERED` with a 12--20-block final prefix or
`UNDERPOWERED` before confirmation is visible. Legal Target-only reuse requires
a byte-equivalent, content-bound evidence alias, and analysis carries its
dependence unit rather than counting the alias as an independent sample. A
self-described non-singleton alias remains blocked until execution evidence
recomputes the equivalence.

Every launched cell binds an exact budget for startup, compile, excluded
warm-up, scored arrivals, deadlines, drain, reset, evidence close, special-job
durations, retries, tokens, p99 status, and GPU accounting. A mandatory
observation receipt records every observed component plus measured and
whole-instance billed GPU time; missing components are not zero. Immutable
compile-cache bases use private per-process overlays, the official serving
client reuses one caller-owned HTTP pool, and evidence writes batch durable WAL
row groups without weakening terminal fsync and coverage checks.

Registry and pool planning make no device-state or empirical claim. Their
declarations do not override release preflight: every formal role still needs
its signed content, capacity, compile, execution, terminal, and qualification
chain. TP2/DP2 and backend-specific paths remain blocked until their exact
dynamic proofs exist.

This release claims no completed industrial measurement, cross-host collective,
Kubernetes scheduling, elastic membership, remote evidence storage, or
automatic failover. Its multi-host control plane distributes independent
host-local work only. Source implementation is not a substitute for a fresh
GPU result.
