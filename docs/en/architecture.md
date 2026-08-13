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
hardware signer, however, so Static/TTS/L0 still fail closed before mutation.
Empirical Stage B is also `BLOCKED` until immutable model/data/trace inputs,
provider access, registered hardware and interference evidence, and the trusted
signer are available. Historical v2 evidence is usable for regression and
debugging only and cannot support a new claim.

## One decode and candidate lifecycle

Target-only disables speculation and is the only path the industrial executor
can currently complete. Static is an allocation-free native speculative target
path, but release preflight blocks it pending trusted terminal attestation.
Neither method's schema contract imports adaptation state or allocates
optimizer, gradient, candidate, or adaptation trace storage.

The target TTS/L0 contract shares one candidate lifecycle; the lower-level
pinned patch and native terminal lifecycle implement this only for TP1/DP1
DFlash. That is still not release-executable without the trusted signer:

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
5. TTS waits for its next fixed update boundary. L0 publishes at the first legal
   graph boundary after the same candidate becomes ready.
6. Commit copies a complete candidate into fixed-address inference storage
   exactly once. Stale, duplicate, cancelled, generation-mismatched,
   non-finite, or conflicting candidates are discarded with a terminal reason.

Candidate creation, commit, discard, cancellation, reset, and disable each have
one terminal transition. Only one candidate may be in flight per cohort, so
side-stream work and scratch memory remain bounded.

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
delta when native inference already applied it. The current patch implements
that lower-level adaptive reconstruction only for TP1/DP1 DFlash.

The target envelope declares, but this release does not execute, the following
cross-backend semantics:

- DFlash binds the deployed differentiable canvas and the sampling-time
  proposal correction; only this backend has a lower-level adaptive patch path.
- DSpark binds real inference-native Markov W1/W2 features, the actual sampled
  predecessor, scheduler mode, proposal distribution, and confidence state.
- EAGLE/EAGLE3 bind tree state, top-k-one execution, and one source version for
  the entire proposal/verification chain.
- NEXTN binds native MTP hidden state and an immutable upstream interface
  digest. Its target contract would additionally require interface and
  memory-fit preflight; the current schema rejects it independently.

DSpark, EAGLE, EAGLE3, and NEXTN adaptation remain `BLOCKED` before model
loading. Historical drafter KV is detached, immutable, and versioned in the
target contract. An eligible update path may use it as state but must never
rebuild or differentiate it; future KV records the newly published version.

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

The target schema and CPU coordinator vocabulary describe one-host TP1/DP1,
TP2/DP1, and TP1/DP2 identities. The current `RunConfig` accepts only TP1/DP1;
it rejects every TP2/DP2 configuration before model loading because this
release cannot issue a content-bound `patched_two_gpu_v1` capability receipt.
The target multi-rank receipt would bind global/local/TP/DP rank, device, node,
process, rendezvous, router, clock, runtime, and model identity.

In that non-executable target coordinator contract, TP trainable state follows
inference ownership. Sharded parameters remain local; replicated parameters
reduce only within the owning TP replica. DP uses sticky cohort routing: a
cohort stays replica-local and adaptation gradients are never averaged between
replicas.

Target distributed publication is two-phase. Each rank prepares the same
retry-stable update/candidate identity and reports source version, epoch,
buffer/optimizer generation, readiness, finiteness, memory reservation, safe
boundary, and process-group health. One all-rank decision either commits
everywhere or aborts everywhere. Post-copy receipts reject partial application.
A collective or split-decision failure makes the service unready, blocks new
admission, and requires an explicit same-topology restart.

`GlooPublicationTransport` is a real two-process CPU harness for this state
machine. It intentionally rejects NCCL and cannot validate CUDA events, graph
capture, fixed-address device copies, GPU numerics, performance, or contention.

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
the same trace digest and account for every offered request as rejected,
completed, timed out, cancelled, or unfinished.

The evidence schema can represent per-process/rank run, request, round, update,
and performance records through a bounded queue. Duplicate primary identities
are rejected by a durable index. Periodic or capacity-triggered flushes create
fsynced Parquet WAL segments and a checkpoint; backpressure and any explicit
drop are counted.
The native terminal lifecycle binds capability, begin, reset, and finalize
receipts to one process/session/run/nonce/plan/rank identity and exact ordered
token IDs. Static retains zero round/update detailed-trace allocation and emits
only the required aggregate speculative safety/accounting. TTS/L0 additionally
bind request, round, update, KV-version, publication, performance, and safety
rows. The release verifier still blocks all three methods before mutation when
the signer is unavailable.

The repository defines immutable session-key and reset/finalize receipt data
contracts for compatible adjacent traces. The complete key includes
source/capability, run configuration, model/drafter revisions, method/backend,
topology and physical GPU UUIDs, memory/graph/telemetry configuration,
compile-cache identity, and ports. The pinned SGLang source now implements the
all-reset capability/state/receipt producer, but its GPU semantics remain
`PENDING`; it explicitly advertises continuous HTTP connection accounting as
unavailable and lacks durable terminal binding and continuous whole-inventory
accounting from launch through termination. This release therefore does
**not** execute or claim a shared server session. Every shared-session
mutation entry point therefore fails before launch, network access, reset, or
evidence-root creation with
`shared_session_trusted_durable_boundary_and_continuous_accounting_unavailable`.
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

Reducer-owned activation artifacts materialize the one 130-cell E1 slice and
each E2 successive-halving round, with an immutable disposition for every
nonactivated template. Confirmation planning is family-local: exactly four
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
target declarations do not override release preflight: Static/TTS/L0 remain
`BLOCKED` on the trusted signer; all TP2/DP2 and unsupported adaptive backends
remain `BLOCKED` on their separate implementation gates.

This release claims no speculative industrial execution, multi-rank execution,
multi-node execution, Kubernetes scheduling, elastic membership, remote
evidence storage, or automatic failover. It contains no new GPU result.
