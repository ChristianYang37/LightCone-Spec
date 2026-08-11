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
  speculation on but no adaptation. TTS/L0 require byte-equivalent adaptation
  after removing the method field.
- For the industrial executor, only TP1/DP1 Target-only is currently READY.
  The pinned tree implements
  `sglang.schema_v3.content_bound_terminal_speculative_evidence.v1`, but
  Static/TTS/L0 fail before mutation because no release-trusted hardware signer
  is configured. This is the expected fail-closed boundary.
- Adaptation is Full or LoRA over `last1`, `last3`, `last5`, or `all`. LoRA
  requires a registered rank and `alpha/r=1`. Borrowed target parameters and
  quantized/unowned coordinates cannot become trainable.
- DSpark layer-only and `*_native_heads` hybrid fields are target contracts;
  the current adaptive schema rejects DSpark before model loading. A future
  implementation must resolve real W1/W2/confidence state and may not use
  placeholder Markov features or an inferred predecessor.
- EAGLE/EAGLE3 and NEXTN validators/hooks are target contracts, not current
  implementations. Their adaptive configs are rejected before model loading.
- Optimizer-specific fields are strict: SGDm/NAG/Muon require momentum; Muon
  also requires Newton--Schulz and auxiliary AdamW values. Unused fields are
  not ignored.

## Distributed-rank refusal or collective failure

- The current `RunConfig` rejects every TP2/DP2 value before model loading. A
  caller-authored `patched_two_gpu_v1` digest cannot enable multi-rank work.
- One-node/two-rank identity, sticky DP routing, and all-rank receipt fields are
  CPU coordinator target contracts only. They describe what a future pinned
  implementation must prove; they are not current SGLang support.
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
- Resume accepts only complete assignment/wave/schedule receipts. A failed
  sibling does not erase durable successful receipts, but it also cannot be
  relabelled as success.

## Unexpected `UNMEASURED`, `BLOCKED`, or `UNDERPOWERED`

`UNMEASURED` means no eligible content-bound GPU evidence/attestation exists.
Positive diagnostics, CPU mocks, historical v2 results, or acceptance changes
do not alter it. The current new GPU phase remains `UNMEASURED` and Stage B is
`BLOCKED` on the missing trusted hardware signer, provider credentials,
immutable model/data/trace locks, registered hardware, GPU smoke, and
interference evidence. Static/TTS/L0, all
DSpark/EAGLE/EAGLE3/NEXTN adaptive cells, and all TP2/DP2 cells are blocked;
only TP1/DP1 Target-only is end-to-end executable.

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
