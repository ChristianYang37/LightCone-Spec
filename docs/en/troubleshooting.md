# Troubleshooting

[中文](../zh-CN/troubleshooting.md) · [Home](../../README.md)

## Patch or runtime identity failure

- Confirm the exact detached upstream commit, clean status, series order,
  per-patch digests, modified-file inventory, and expected final tree in
  `patches/sglang/manifest.json`.
- Run the full disposable apply/compile/test/reverse verifier. A successful CPU
  package test does not verify a pending SGLang migration.
- Never point a run at an implicit workspace checkout. Pass the explicit
  verified disposable checkout and bind its tree receipt.
- Any source, package, model, sampling, trace, topology, or launch-argument
  drift requires a fresh runtime identity and downstream evidence root.

## Configuration or backend refusal

- Schema v3 uses canonical `target_only`, `static`, `tts`, and `l0`. Unknown
  method names and retired adaptation fields are errors.
- Target-only requires speculation off and no adaptation. Static requires
  speculation on but no adaptation. TTS requires the frozen TTS recipe plus
  fixed-barrier publication; L0-naive requires that same recipe authority plus
  first-ready publication. An L0-policy search recipe is an LC-candidate until
  the exact E2 final-recipe receipt seals it as LightCone.
- Do not interpret source support or a valid config as `READY`. Formal execution
  requires a fresh root-authorized deployment/hardware policy and the exact
  prepared-content, workload, compile, qualification, terminal, interference,
  and capacity authorities. A checkout deliberately contains none of that
  session evidence.
- TTS-Cal fixes the registered reconstruction grid and structural semantics: Adam, one step,
  `(beta1=.9, beta2=.999, epsilon=1e-8)`, zero decay, no clipping,
  full-drafter/latest-update-round-only update, request reset, side stream, the
  learning-rate grid, and eight strides. Its exact pinned DFlash loss is
  float32 target-to-draft forward KL at temperature one with valid-row masking,
  `exp(-(k-1)/7)` position weights, and masked weighted normalization. The
  source-point value correction is not an independent proximal penalty, so
  there is no `lambda` input or additional search axis. Reject inheritance from
  E1/E2, schema defaults, or a historical AdamW recipe.
  Candidate equality is valid only for controlled replay with identical
  source-state and proposal-evidence digests.
- Adaptation is Full or LoRA over `last1`, `last3`, `last5`, or `all`. LoRA
  requires a registered rank and `alpha/r=1`. Borrowed target parameters and
  quantized/unowned coordinates cannot become trainable.
- DSpark layer-only and `*_native_heads` hybrid paths are implemented source
  contracts but require a DSpark GPU proof over the real sampled predecessor,
  W1/W2/confidence state, 56-candidate selector, and fixed/native schedulers.
  Placeholder Markov features or an inferred predecessor are rejected.
- NEXTN likewise requires its native MTP/interface proof and, for TP2, exact
  two-model shard authority. Adaptive EAGLE remains unsupported. EAGLE3 is
  limited to combinations admitted by a signed official model/selector
  compatibility decision.
- Optimizer-specific fields are strict: SGDm/NAG/Muon require momentum; Muon
  also requires Newton--Schulz and auxiliary AdamW values. Unused fields are
  not ignored.

## Distributed-rank refusal or collective failure

- `RunConfig` recognizes only `tp1_dp1`, `tp2_dp1`, and `tp1_dp2`. A
  distributed row must carry the exact source-owned `patched_two_gpu_v1`
  identity and a receipt claim; formal execution additionally requires the
  matching durable GPU qualification artifact. A caller-authored digest cannot
  enable multi-rank work.
- One-node/two-rank two-phase TP publication and sticky DP replica isolation
  are implemented contracts pending dynamic GPU proof. Check the proof's exact
  rank/UUID/topology, all-rank terminal coverage, one-rank abort behavior, and
  zero cross-replica gradient evidence.
- A missing prepare vote, foreign topology receipt, generation mismatch,
  non-finite candidate, unsafe boundary, or incomplete post-copy receipt aborts
  the update everywhere. A transport/split-decision failure also disables
  admission until an explicit same-topology restart.
- The CPU `gloo` harness must use a live gloo process group. It intentionally
  rejects NCCL and cannot be used as GPU capability evidence.
- Multi-node, more than two ranks, Kubernetes, elastic membership, and
  automatic failover are unsupported rather than hidden flags.

## Memory pressure

Inspect the per-category, per-rank HBM ledger. Admission is limited by the
least-headroom rank after model/KV, FP32 master, gradient, actual moments,
candidate/staging, activation, graph, telemetry, and safety margin are charged.
An estimate based only on trainable parameter count is incomplete.

Pressure handling preserves active correctness state, may evict only native
inactive prefixes, then aborts pending adaptation and queues or rejects new
work. Optional inactive-cohort offload is explicit and last. Do not hide an OOM
by changing Full to LoRA, rank, scope, optimizer, precision, context, or
admission; each change needs a new config and load screen.

Disk admission is separate from HBM. A schema-3 stage capacity gate reopens
the real provider/host/sizing snapshot and exact execution-wave schedule. It
sums retained evidence and registered retries, adds the maximum concurrently
resident staging/compile bytes per wave, then the safety margin. If that
root-authorized chain is absent, the legacy 100 GB threshold remains the
fail-closed fallback. Adding 30 GB does not itself authorize a stage; rerun the
raw capacity reducer and obtain a fresh local `capacity` control attestation.
Never delete user data to force the gate.

The trusted single-operator v03 path uses a separate unsigned empirical
schema-3 authority. Its 31 GiB initial admission is exactly one 16 GiB physical
wave plus a 15 GiB safety margin. Fresh free space already includes bytes
written by an adopted RUNNING process, so restart binds that process in the
durable ledger without charging its full lifetime high-water twice. Capacity
loss stops new dispatch or durably blocks the current DAG node; reconciliation
of an existing RUNNING physical or E6/E0 auxiliary process remains available.
Automatic retries are disabled because this path has no reachable failed-attempt
archive producer. Retain the failed evidence and resolve the block explicitly.

For slab exhaustion, inspect tenant quota, active references, replica identity,
generation counters, and optional cold-offload timers. Only inactive cohorts
can be reclaimed. Reusing a slab without a matching reclamation/transfer
receipt is a stale-state bug.

## Telemetry backpressure or interruption

The evidence writer has a bounded queued-row count and batches queue drains into
Parquet WAL row groups. Registered row/time checkpoints and terminal boundaries
perform the durable fsync; a temporarily empty queue is not an fsync request.
Under `backpressure` the producer waits; under explicit `drop` the triggering
negative row remains durable, drop counters increment, and the attempt cannot
satisfy complete evidence. Increasing a bound changes runtime identity and must
be decided before measurement.

Duplicate request/round/update/performance keys are rejected by the durable
index. Checkpoint, WAL row-group coverage, final shard schema/digest, and
completion-receipt counters must agree. Do not rename, combine, truncate, or
hand-edit segments.

On interruption, keep the WAL/checkpoint/aborted marker for audit and resume
from the same immutable cell. Only a single valid exclusive terminal receipt is
skipped. A final Parquet file without a receipt, or multiple completed attempts
for one run/rank, is not valid evidence.

## Registry or dispatch refusal

- Load the generated registry through the CLI; a generator, parameter, embedded
  declaration, or SHA-256 mismatch means it was edited.
- The registry takes two logical rank slots, never physical device arguments.
  Supply a strict content-bound inventory, exact per-cell budget sequence, and
  interference envelope to the pool planner; physical UUIDs appear only in the
  frozen assignment.
- Supply exactly the receipts declared by the ready stage. Locked-output names
  must match the definition and use lowercase SHA-256 values.
- E1/E2 completion requires the exact reducer activation and a disposition for
  every template. Confirmation completion requires family-local pilot/final
  activation and its exact power plan. A forged or cross-round survivor is an
  error, not runnable work.
- A completed serving row needs a terminal receipt, physical assignment,
  `ExperimentBudget`, and terminal-bound `BudgetObservationReceipt`. All
  observed phase components and measured/billed GPU-time closure must agree;
  directory presence is not completion.
- Concurrency is capped by the exact matching `InterferenceEnvelope`. More idle
  GPUs do not authorize a co-run class that was not calibrated. Gang jobs are
  atomic; profiler/download/compile work is exclusive-host. Never launch every
  assignment merely because it appears in one plan.
- Resume replays the path-bound append-only attempt journal and reopens every
  successful terminal authority. A failed sibling does not erase durable
  successful receipts and only failed siblings consume a retry. An intent
  without a finish is `dispatch_attempt_intent_without_finish_cost_unresolved`;
  do not delete it or impute its cost.

## Unexpected `UNMEASURED`, `BLOCKED`, or `UNDERPOWERED`

`UNMEASURED` means no eligible release-attested, content-bound GPU evidence
exists. Positive diagnostics, CPU mocks, historical v2 results, or acceptance
changes do not alter it. In the release-attested lane, a fresh campaign is
`BLOCKED` until its provider state, root-authorized dynamic hardware policy,
immutable content locks, GPU smoke, compile/exactness/interference terminals,
and stage capacity control all verify. The trusted
`formal_single_operator_v1` lane does not need the external signer, but it still
requires the corresponding source/content/runtime, fresh GPU qualification,
capacity, terminal, and coverage gates and can conclude only
`trusted_single_operator_empirical_no_signature`, `formal_measured=false`, and
`UNMEASURED`. TP2/DP2, DSpark, NEXTN, native ITL, and session reuse specifically
remain `implemented_pending_dynamic_gpu_proof`; TTS/L0-naive require the exact
TTS-Cal seal, LightCone the E2 seal, and EAGLE3 an applicable compatibility
decision. None may be relabelled READY from source capability alone.

`BLOCKED` is also the correct outcome for a registered cell whose prerequisite
or attested criterion failed. `UNDERPOWERED` means four excluded pilot blocks
could not select 12--20 final blocks with registered power; confirmation must
not start. `UNRESOLVED` p99 means fewer than 10,000 completions. Do not relabel,
omit, impute, or optimize away any of these states.

## Security and bug reports

Report sanitized commands, package/platform versions, exact upstream and
patched tree IDs, registry/cell/trace digests, topology receipt, and the
smallest reproducible schema-v3 config. Never publish tokens, passwords,
provider keys, temporary URLs, private prompts, instance addresses, model
paths, or raw provider state.
