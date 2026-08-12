# CLI

[中文](../zh-CN/cli.md) · [Home](../../README.md)

## Command surface

`lightcone-spec --help` and `lightcone-spec COMMAND --help` are the argument
authority. The schema-v3 and industrial commands are:

| Command | Purpose |
|---|---|
| `doctor` | Read-only host, Python, CUDA, and source identity report |
| `validate-config` | Validate one strict schema-v3 run config and sidecar |
| `build-industrial-registry` | Bind one or more stable logical rank slots and materialize the immutable experiment DAG |
| `collect-gpu-inventory` | Collect a nonce-bound physical GPU/topology inventory and raw probe receipt |
| `build-interference-envelope` | Derive the current serial interference envelope and its inventory-bound raw receipt |
| `reduce-e1-activation` | Derive the single 130-cell E1 slice from sealed E3a evidence |
| `reduce-e2-activation` | Materialize one E2 round from the E1 Pareto artifact or prior survivors |
| `reduce-e2-successive-halving` | Reduce exact E2 stage evidence into a sealed survivor receipt |
| `materialize-confirmation-pilots` | Activate exactly four excluded pilots for one confirmation family |
| `reduce-confirmation-family-power` | Reduce one family's exact four pilot blocks into a sealed power plan |
| `materialize-confirmation-prefix` | Activate only a powered family's sealed 12--20-block final prefix |
| `validate-evidence-alias` | Replay one raw alias manifest against registry, hardware, inventory, and terminal authority |
| `build-evidence-dependence-map` | Preserve shared-observation dependence from reducer-issued alias artifacts |
| `materialize-stage-activation` | Replay generic registry-stage dispatchability from raw registry, lineage, runtime, and split authority |
| `materialize-industrial-budgets` | Derive one fail-closed `BudgetPlan` from reducer, load, inventory, policy, and capacity authority |
| `estimate-industrial-budget` | Replay one ready `BudgetPlan` on an exact physical inventory and interference envelope |
| `plan-industrial-dispatch` | Freeze deterministic topology-aware GPU-pool waves and physical assignments |
| `execute-dispatch-wave` | Replay path-bound assignment bundles and execute one receipt-bounded frozen wave when all release authorities are available |
| `seal-industrial-stage` | Bind activated completion, dispositions, budgets, runtime, split, dependencies, and locked outputs |
| `analyze-industrial` | Validate schema-v3 terminal, budget, family-power, and hardware evidence |
| `build-speed-study` | Materialize the smaller core source protocol |
| `lock-models` | Resolve model IDs to immutable revisions |
| `prepare-models` | Download or offline-verify locked snapshots |
| `list-tuning-candidates` | Write the registered Full/LoRA tuning grid |
| `render-target-only-runtime` | Render a speculation-disabled Target-only endpoint |
| `render-static-load-runtime` | Render an allocation-free Static endpoint |
| `render-tuning-runtime` | Render matched TTS/L0 tuning endpoints |
| `run-controlled-slice` | Measure one registered controlled slice |
| `collect-static-load-screen` | Validate Static load coverage and select the reference load |
| `advance-tuning-stage` | Validate a halving stage and seal survivors |
| `select-speed-config` | Apply the tuning-only registered selection rule |
| `select-anchor-config` | Lock a terminal registered anchor without an optimum claim |
| `render-runtime` | Emit matched sequential core configs and launch argv |
| `build-confirmation-queue` | Materialize clean-server confirmation jobs |
| `run-confirmation` | Execute one method/block confirmation slice |
| `run-target-reference` | Capture a locked Target-only greedy token-ID trajectory diagnostic |
| `collect-speed-study` | Derive a table from completed receipt-bound evidence |
| `render-replication-runtime` | Render natural-task or profiler-only slices |
| `run-natural-slice` | Run one locked natural-task slice |
| `build-profiler-plan` | Build an isolated profile plan with headline evidence forbidden |
| `attest-speed-study` | Bind GPU, runtime, model, selection, trace, and evidence identities |
| `analyze-speed-study` | Evaluate the registered paired gate |

The separate OnlineSPEC family is unchanged in isolation:

| Command | Purpose |
|---|---|
| `build-onlinespec-study` | Build the provenance-bound comparison protocol |
| `verify-onlinespec-source` | Verify an external clean checkout against its source audit |
| `list-onlinespec-candidates` | Write OGD, optimistic, and ensemble candidates |
| `render-onlinespec-tuning-runtime` | Render paired Static/candidate tuning endpoints |
| `run-onlinespec-tuning-slice` | Measure one learner tuning slice |
| `advance-onlinespec-tuning-stage` | Halve candidates independently per learner |
| `select-onlinespec-config` | Select one safe terminal candidate per learner |
| `select-onlinespec-anchor-config` | Lock three terminal anchors without a grid-optimum claim |
| `render-onlinespec-runtime` | Render sequential Static and learner endpoints |
| `build-onlinespec-queue` | Materialize randomized clean-server comparison jobs |
| `run-onlinespec-confirmation` | Execute one learner/method block |
| `collect-onlinespec-study` | Derive the isolated comparison table |
| `attest-onlinespec-study` | Bind comparison evidence and source identities |
| `analyze-onlinespec-study` | Produce diagnostic learner-versus-Static intervals |

There is no generic method override or command that converts an unreceipted
directory into completed evidence.

## Industrial registry workflow

The registry contains scientific identities and one or more logical rank slots, not
host-specific GPU UUIDs:

```bash
lightcone-spec build-industrial-registry \
  --logical-gpu-slot logical-rank-slot-0 logical-rank-slot-1 \
  --base-port 24000 \
  --cache-root runtime-cache/industrial \
  --evidence-root artifacts/industrial \
  --seed 20260811 \
  --output artifacts/industrial/registry.json
```

The output embeds its generator identity, input parameters, complete
declarations, and registry SHA-256. Loading regenerates the registry and
compares exact content, so hand-edited cells are rejected. A separate
content-bound inventory supplies physical UUIDs, model/memory/capability,
PCI/NUMA/interconnect topology, availability, and topology groups.

Stages without the E1/E2 or confirmation-family reducers use a first-party
generic reducer. Its raw manifest has exactly the registry, stage name,
runtime, split, and complete ordered dependency-receipt paths; there is no
caller cell list or bare activation-plan input. Preflight has an empty receipt
list and derives canonical genesis authority from the exact registry:

```json
{
  "schema_version": 1,
  "kind": "industrial_registry_stage_activation_manifest",
  "registry_artifact": "artifacts/industrial/registry.json",
  "experiment": "preflight",
  "runtime_artifact": "artifacts/industrial/runtime.json",
  "split_artifact": "artifacts/industrial/preflight-split.json",
  "dependency_receipts": []
}
```

Bind that manifest with its adjacent `.sha256`, then materialize the canonical
artifact for inspection:

```bash
lightcone-spec materialize-stage-activation \
  --manifest artifacts/industrial/preflight-activation-manifest.json \
  --output artifacts/industrial/preflight-activation.json
```

Consumers take the raw manifest path through `--activation-plan` and replay
the reducer; they reject the serialized output as standalone authority. The
reducer dispositions every declared stage cell from registry status and the
release dispatch predicate. In this release every preflight cell is blocked:
the Target-only compile cell has no release-owned exact prewarm manifest,
graceful-finalization acknowledgement, or atomic attempt/final-cache result
pointer, while Static/TTS/L0 also lack a supported execution contract. The
command therefore writes the canonical `BLOCKED` artifact and returns 42. E6
download cells are blocked independently because no first-party download
terminal contract ships.

Budget materialization is fail-closed. It consumes reducer-generated stage or
family activation, a complete `BudgetPolicy`, one `BudgetLoadBinding` for every
activated serving cell, an independently sourced `CapacityEnvelope`, and the
complete physical inventory. Formal capacity authority additionally requires
the paired `--capacity-manifest` and `--capacity-verification-receipt` raw
artifacts. The verifier reopens every path-bound inventory/source, provider
quota, host-capacity, and per-cell sizing input and accepts only the
release-owned trust root. It writes `READY` only when load coverage, provider
GPU quota, and maximum-attempt disk capacity all close; otherwise it preserves
the derived budgets in an immutable diagnostic `UNRESOLVED` plan and exits 42.
A bare envelope or SHA-256 is never execution authority. Repeat
`--budget-load-binding PATH` once per activated serving cell:

```bash
lightcone-spec materialize-industrial-budgets \
  --registry artifacts/industrial/registry.json \
  --activation-plan artifacts/industrial/preflight-activation-manifest.json \
  --inventory artifacts/industrial/inventory.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --capacity-manifest artifacts/industrial/capacity-source-manifest.json \
  --capacity-verification-receipt artifacts/industrial/capacity-verification.json \
  --output artifacts/industrial/budget-plan.json
```

There is no caller-selected-cell or duration-sequence fallback. Preflight now
closes through the generic reducer; omitting its bound raw manifest is still an
explicit unresolved authority error, and a caller-authored activation cannot
bridge it. Serving activations additionally repeat `--budget-load-binding PATH`
once per activated serving cell.

Planning reloads that `BudgetPlan`, reruns materialization from the same raw
inputs, and requires exact equality before scheduling. It also requires the
paired interference authority:

```bash
lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
  --activation-plan artifacts/industrial/preflight-activation-manifest.json \
  --inventory artifacts/industrial/inventory.json \
  --interference-envelope artifacts/industrial/interference-envelope.json \
  --budget-plan artifacts/industrial/budget-plan.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --capacity-manifest artifacts/industrial/capacity-source-manifest.json \
  --capacity-verification-receipt artifacts/industrial/capacity-verification.json \
  --output artifacts/industrial/dispatch.json
```

The frozen plan binds the exact inventory and interference envelope, every
cell's budget digest, physical UUID/rank/port/topology assignment, wave
membership, and scheduler identity. The scheduler accepts arbitrary same-host
inventories; 1/2/4/8/16-GPU cases are regression-tested. A wave never exceeds
the calibrated co-run class, so eight idle GPUs do not imply eight-way headline
execution. Structural loading of an envelope does not prove its calibration;
formal timing still requires raw isolated/co-run evidence and trusted hardware
binding.

`execute-dispatch-wave` never treats the planner JSON or an
`IndustrialExecutionPlan.to_dict()` summary as launch authority. Repeat
`--bundle` for every assignment in the frozen dispatch plan. Each schema-v1
bundle binds absolute resolved paths, raw/canonical/semantic/file identities,
and adjacent sidecars for the registry and inventory, serial-interference raw
receipt, budget policy/load/capacity inputs, raw generic-stage activation
manifest and its exact runtime envelope/split/receipt chain, context and
dispatch, topology/load/config/launch, sampling/model lock/prepared roots,
compile plan, and inventory/runtime evidence. Activation is a tagged raw
authority: generic, E1, E2, confirmation-pilot, confirmation-final, and
stage-aggregate manifests are replayed from their complete path closure. A
final prefix derives prior pilot completion only from schema-v4/native
completion authority; bare completed-cell IDs are rejected. E3b/E5 stage
receipts use a separate family-SHA-sorted aggregate. The E5 aggregate also
requires deterministic auxiliary activation/completion for its 264 non-family
failure-injection cells, and the family plus auxiliary dispositions must form
an exact disjoint cover of the registry stage. Its read-only audit reruns the
raw reducers and exact planning scheduler, but deliberately reports the
execution-plan SHA as `null`/`NOT_VALIDATED`; a self-consistent serialized plan
summary cannot authorize itself. Formal reconstruction first requires a live
`READY` `BudgetPlan` plus path-bound budget materialization authority for its
policy, loads, activation, inventory, and capacity sources. Only then can it
build the physical plan and compare the redundant summary exactly. This
initial execution slice accepts only those Target-only serving assignments
that pass the release capability, raw activation/completion, capacity,
interference, and trusted-attestation boundaries. Compile/download and every
unsupported serving assignment remain explicitly blocked.
Before launch, the command linearly checks every assignment-local source and
requires complete bundle coverage plus one exact shared schema-v3 dispatch
context/raw budget authority. It replays that shared scheduler authority as a
group, then constructs only the requested wave's physical plans (and any prior
plans named by a resume receipt); each plan that can actually launch still
runs the full public validation boundary.

The command executes exactly one `--wave-index`. Wave zero has no resume input;
a successful prefix requires the previous `--resume-receipt`, while a failed
wave resumes at that same index. Before any sibling runner starts, the command
durably appends one wave intent and every per-assignment intent to a private
`RECEIPT.attempt-journal`; each terminal or failed return appends its finish,
exact retry identity, and cumulative monotonic cost. A partial wave therefore
keeps successful siblings and retries only failed siblings. An intent without
a finish has unknowable cost and blocks under
`dispatch_attempt_intent_without_finish_cost_unresolved`.

The immutable file at `--receipt-output` is one canonical schema-v2 envelope
containing the receipt, its structured sidecar, and the exact journal
manifest/head/event-count prefix. Resume treats the caller receipt only as an
anchor: it replays the raw journal and reopens every successful
`AssignmentTerminalAuthority`. A crash after all finish rows but before the
envelope is recoverable by rerunning the same wave/output command; no runner is
repeated. `RECEIPT.sidecar.json` is only a derived convenience copy, so a crash
before that copy cannot destroy resume authority. The parent directory must
already exist.

The hash chain detects deletion, substitution, symlinks, and coordinated event
rehashing relative to an already published envelope prefix. It is not an
external signature: an actor who can replace the entire unsigned journal and
every anchor is outside this software-only recovery threat model. Formal GPU
claims still require the release-owned trusted attester below.

```bash
lightcone-spec execute-dispatch-wave \
  --bundle artifacts/industrial/bundles/assignment-000.json \
  --bundle artifacts/industrial/bundles/assignment-001.json \
  --wave-index 0 \
  --receipt-output artifacts/industrial/dispatch-wave-0.json
```

The source release has no trusted hardware attester in its release-owned trust
policy. A caller or test signer cannot unlock formal execution, even with a
cryptographically valid signature, and a bare `CapacityEnvelope` cannot grant
execution authority. It therefore returns `BLOCKED`/42 before reading a bundle,
creating the receipt parent or evidence root, importing the serving client, or
launching a process. A missing bundle is likewise never converted into a
planner-summary fallback. Fresh execution rejects unjournaled pre-existing
per-plan trace files. Resume requires the raw append-only attempt chain plus
revalidated structured terminal bindings; neither a bare terminal digest nor a
caller-rehashed schedule JSON can skip work. If a coordinator dies after
durable intent but before durable finish, execution remains blocked rather than
inventing a monotonic cost.

Estimate the same activated set before dispatch. Every budget field is
explicit; the report separates wall, compute, reserved, and whole-instance
billed GPU time in optimistic, registered, and quota-envelope scenarios:

```bash
lightcone-spec estimate-industrial-budget \
  --registry artifacts/industrial/registry.json \
  --activation-plan artifacts/industrial/preflight-activation-manifest.json \
  --inventory artifacts/industrial/inventory.json \
  --interference-envelope artifacts/industrial/interference-envelope.json \
  --budget-plan artifacts/industrial/budget-plan.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --capacity-manifest artifacts/industrial/capacity-source-manifest.json \
  --capacity-verification-receipt artifacts/industrial/capacity-verification.json \
  --output artifacts/industrial/budget-report.json
```

The estimator performs the same raw-input rematerialization and live capacity
revalidation. Its report binds both the scheduler inventory SHA-256 and
interference-envelope SHA-256. If the exact scheduler replay is not executable,
it retains the real derived GPU-hour diagnostics, records a named unresolved
assumption, and exits 42; the report cannot be used as launch authority.

After a stage's activated cells and dispositions are durable, seal them.
`--inventory PATH` is the authority for the physical GPU identities and
topology used by the completed evidence. Every repeated `--locked-output` has
the single syntax `NAME=PATH`: names must be unique, and `PATH` must identify a
content-bound artifact. A SHA-256 literal is not a substitute for that artifact.
Runtime and split are also bound files, and dependencies are receipt files
rather than copied hash strings:

```bash
lightcone-spec seal-industrial-stage \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/inventory.json \
  --experiment preflight \
  --runtime-artifact artifacts/industrial/runtime.json \
  --split-artifact artifacts/industrial/preflight-split.json \
  --activation-plan artifacts/industrial/preflight-activation-manifest.json \
  --completed-cells artifacts/industrial/completed-cells.json \
  --locked-output runtime_envelope=artifacts/industrial/runtime-envelope.json \
  --output artifacts/industrial/receipts/preflight.json
```

Sealing E2 adds one mandatory authority, `--e2-final-stage-manifest PATH`. It
must be the raw `halving_3` manifest; the command reruns the first-party raw
reducer and requires a `FINAL_RECIPE` result. The `dflash_recipe=PATH` artifact
must then bind exactly that final candidate, registry, runtime, split, and final
stage reduction. A digest in place of `PATH`, a caller-authored summary, or a
recipe for a different candidate is rejected:

```bash
lightcone-spec seal-industrial-stage \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/inventory.json \
  --experiment E2 \
  --runtime-artifact artifacts/industrial/runtime.json \
  --split-artifact artifacts/industrial/e2-split.json \
  --completed-cells artifacts/industrial/e2-completed-cells.json \
  --e2-final-stage-manifest artifacts/industrial/e2-halving_3-manifest.json \
  --dependency-receipt artifacts/industrial/receipts/E1.json \
  --locked-output dflash_recipe=artifacts/industrial/e2-dflash-recipe.json \
  --output artifacts/industrial/receipts/E2.json
```

These checks define the claim-bearing path once a trusted hardware attester is
registered. In the current release, sealing still returns `BLOCKED` before it
can issue a receipt; this raw-authority contract does not make final execution
or a performance claim reachable.

For a downstream stage, repeat `--dependency-receipt` for its exact declared
dependencies. Then pass all completed receipts to the planner:

```bash
lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/inventory.json \
  --interference-envelope artifacts/industrial/interference-envelope.json \
  --budget-plan artifacts/industrial/budget-plan.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --budget-load-binding artifacts/industrial/load-cell-000.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --receipt artifacts/industrial/receipts/preflight.json \
  --completed-cells artifacts/industrial/completed-cells.json \
  --activation-plan artifacts/industrial/next-stage-activation-manifest.json \
  --output artifacts/industrial/next-dispatch.json
```

Formal completed-cell rows bind the frozen physical assignment, exact budget,
terminal receipt, mandatory `BudgetObservationReceipt`, and three-way budget/
terminal digest closure. Missing phase timing, compile/download N/A reasoning,
or GPU accounting is not replaced with zero. Resume accepts only a complete
receipt and never directory presence.

The dispatch plan is target-protocol data, not proof that a cell is executable.
The pinned tree implements the exact native terminal begin/reset/finalize hook,
but no trusted hardware signer is configured. Generic activation records a
canonical blocked preflight disposition; it does not create a compile runner,
execution authority, or performance claim. Static/TTS/L0 remain blocked without
their validated native capability and trusted signer. The CLI does not silently
provision hardware or start a GPU.

## Identity and topology flow

The minimum industrial identity chain is:

```text
source + patched tree + model/data/trace locks
                         |
                         v
          registry logical slots + activation reducers
                         |
                         v
 inventory + interference + budgets -> frozen physical waves
                         |
                         v
       terminal + reset + budget-observation receipts
                         |
                         v
 family power + dependence-aware statistics + trusted attestation
```

TP2 and sticky DP2 fields are target registry/CPU-coordinator vocabulary. The
current `RunConfig` rejects every TP2/DP2 value before model loading, and the
CPU `gloo` contract or a caller-authored receipt cannot enable them. A future
implementation would have to bind rank, UUID, rendezvous, router, clock,
process-group, ownership, and receipt identity exactly.

Generated JSON uses an adjacent `.sha256` sidecar where the command contract
requires one. The loaders validate canonical content, not only filenames.

## Core tuning and confirmation

Target-only and Static render without an adaptation object or reserve. TTS and
L0 target declarations share a Full/LoRA candidate from registered native
layer scopes. Rendering is pure planning: in the current release only
Target-only may proceed through industrial execution. Static/TTS/L0 fail the
trusted-terminal-attester preflight before an endpoint starts.

The industrial E1/E2 commands are reducer-owned. E1 consumes sealed E3a output
and activates exactly one width/load slice. E2 materializes stage zero and then
requires the prior sealed survivor receipt for each successive-halving round,
while preserving matched TTS/L0 pairs and family floors. Confirmation planning
activates four excluded pilots per exact family, seals `POWERED` with 12--20
final blocks or `UNDERPOWERED`, and materializes only the sealed prefix before
confirmation data is visible.

Family power is never accepted as a bare score or hand-authored count:

```bash
lightcone-spec reduce-confirmation-family-power \
  --manifest artifacts/industrial/family-power-manifest.json \
  --output artifacts/industrial/family-power-plan.json
```

The exact `industrial_family_power_manifest` binds the registry artifact,
pilot activation, hardware envelope, and four pilot blocks. Every pilot cell
supplies terminal receipts, a hardware receipt, and a budget observation. The
reducer keeps confirmation hidden and rejects missing, extra, cross-family, or
unsafe evidence.

`validate-evidence-alias` does not create scientific equivalence from labels or
round-trip a caller-authored receipt. It replays a bound raw manifest against
the current registry, hardware envelope, GPU inventory and its source receipt,
execution/load/config/split/sampling/model/budget artifacts, source terminal
evidence, and native terminal artifacts:

```bash
lightcone-spec validate-evidence-alias \
  --manifest artifacts/industrial/raw-alias-manifest.json \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/gpu-inventory.json \
  --hardware-envelope artifacts/industrial/hardware-envelope.json \
  --output artifacts/industrial/alias-reduction.json

lightcone-spec build-evidence-dependence-map \
  --direct-map artifacts/industrial/direct-dependence-map.json \
  --alias-reduction artifacts/industrial/alias-reduction.json \
  --output artifacts/industrial/evidence-dependence-map.json
```

The dependence map accepts only first-party reduction artifacts and makes
shared controls one observation for covariance/bootstrap purposes. Static is
backend-specific and TTS/L0 are never aliases. The schema-v3 industrial
analysis manifest lists the raw alias manifests; the formal reducer replays
them and rejects a serialized map that differs from that replay. Legacy alias
receipts, caller-authored reduction summaries, and the old `--alias` flags are
rejected.

Every completed slice ends with content-bound evidence. A resume skips it only
after manifest/config/method/block/data/trace and every shard digest validate.
Interrupted Parquet WAL segments have no terminal receipt and are excluded.
Profiler output has `headline_evidence_forbidden=true` and cannot be merged into
the measured timing table.

## OnlineSPEC workflow

OnlineSPEC reuses model locking, controlled data, and a paired Static reference,
but never core tuning or confirmation rows. `verify-onlinespec-source` checks
the external checkout's exact commit/tree, clean status, audited key-file
hashes, and license inventory. The checkout stays outside this repository.

Successive halving runs independently inside OGD, optimistic OGD, and the
ensemble. Selection also binds the core reference load, but it cannot change
that load or affect the core gate. An anchor selection labels
`optimized_grid_claim=false`. OnlineSPEC remains TP1/DP1 and uses its own GPU
attestation.

## Status, resume, and exit codes

`analyze-speed-study` may produce diagnostics without attestation, but status is
`UNMEASURED` and the process exits 42. With valid content-bound GPU evidence it
exits zero only for a complete registered pass. Valid evidence that misses a
criterion is `BLOCKED` and also exits 42. Identity, schema, I/O, receipt, or
runtime errors are ordinary nonzero failures, not scientific outcomes.
The current release has no trusted hardware-attester identity, so a supplied
legacy attestation is rejected and `analyze-industrial` cannot emit `MEASURED`;
it exits 42 even for otherwise self-consistent diagnostic JSON.

The industrial registry may declare target cells as `UNMEASURED`, but that
declarative status is not executable readiness. The native hook is present;
executor preflight resolves Static/TTS/L0 to `BLOCKED` because the trusted
signer is absent. All TP2/DP2 and DSpark/EAGLE/EAGLE3/NEXTN adaptive cells are
likewise blocked by separate current release gates. Historical v2 artifacts
are regression-only and cannot be supplied as schema-v3 stage receipts.

### Target-output reference diagnostic

Against a separate locked Target-only server, `run-target-reference` records
per-prompt token counts and format-tagged SHA-256 values of complete ordered
output-token-ID arrays. It never substitutes a decoded-text digest. Legacy
collectors require `--target-reference` and reject a method/block unless its
trajectory matches the reference; agreement among speculative methods alone is
not an exactness proof. In this release the result remains an `UNMEASURED`
diagnostic because Static/TTS/L0 execution and trusted hardware attestation are
both blocked.

## Credentials and output roots

Artifacts, model roots, caches, traces, provider state, profiles, selections,
attestations, and handoff files belong under ignored external roots. Pass model
access through a temporary `HF_TOKEN` environment variable or another secure
channel. Do not place tokens, passwords, provider API keys, private prompts,
instance addresses, or machine-specific paths in arguments, manifests, logs,
documentation, or Git.
