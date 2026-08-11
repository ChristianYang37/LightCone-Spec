# CLI

[中文](../zh-CN/cli.md) · [Home](../../README.md)

## Command surface

`lightcone-spec --help` and `lightcone-spec COMMAND --help` are the argument
authority. The schema-v3 and industrial commands are:

| Command | Purpose |
|---|---|
| `doctor` | Read-only host, Python, CUDA, and source identity report |
| `validate-config` | Validate one strict schema-v3 run config and sidecar |
| `build-industrial-registry` | Bind two stable logical rank slots and materialize the immutable experiment DAG |
| `reduce-e1-activation` | Derive the single 130-cell E1 slice from sealed E3a evidence |
| `reduce-e2-activation` | Materialize one E2 round from the E1 Pareto artifact or prior survivors |
| `reduce-e2-successive-halving` | Reduce exact E2 stage evidence into a sealed survivor receipt |
| `materialize-confirmation-pilots` | Activate exactly four excluded pilots for one confirmation family |
| `reduce-confirmation-family-power` | Reduce one family's exact four pilot blocks into a sealed power plan |
| `materialize-confirmation-prefix` | Activate only a powered family's sealed 12--20-block final prefix |
| `validate-evidence-alias` | Strictly load and rewrite one content-bound evidence alias receipt |
| `build-evidence-dependence-map` | Preserve shared-observation dependence from legal aliases |
| `estimate-industrial-budget` | Reduce exact per-cell budgets for one inventory and activation bundle |
| `plan-industrial-dispatch` | Freeze deterministic topology-aware GPU-pool waves and physical assignments |
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

The registry contains scientific identities and two logical rank slots, not
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

Planning is fail-closed: it requires an inventory, an exact per-cell budget
sequence, and an interference envelope. It also requires reducer-generated
stage or family activation for template stages; a hand-authored cell list is
not accepted.

```bash
lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/inventory.json \
  --interference-envelope artifacts/industrial/interference-envelope.json \
  --budget-plan artifacts/industrial/budgets.json \
  --output artifacts/industrial/preflight-dispatch.json
```

The frozen plan binds the exact inventory and interference envelope, every
cell's budget digest, physical UUID/rank/port/topology assignment, wave
membership, and scheduler identity. The scheduler accepts arbitrary same-host
inventories; 1/2/4/8/16-GPU cases are regression-tested. A wave never exceeds
the calibrated co-run class, so eight idle GPUs do not imply eight-way headline
execution. Structural loading of an envelope does not prove its calibration;
formal timing still requires raw isolated/co-run evidence and trusted hardware
binding. There is currently no `execute-dispatch-wave` CLI command; the
structured library executor requires a first-party, content-bound launch
adapter and does not infer launch configuration from planner JSON.

Estimate the same activated set before dispatch. Every budget field is
explicit; the report separates wall, compute, reserved, and whole-instance
billed GPU time in optimistic, registered, and quota-envelope scenarios:

```bash
lightcone-spec estimate-industrial-budget \
  --registry artifacts/industrial/registry.json \
  --activation-plan artifacts/industrial/activation.json \
  --inventory artifacts/industrial/inventory.json \
  --budgets artifacts/industrial/budgets.json \
  --output artifacts/industrial/budget-report.json
```

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
  --budget-plan artifacts/industrial/budgets.json \
  --receipt artifacts/industrial/receipts/preflight.json \
  --completed-cells artifacts/industrial/completed-cells.json \
  --activation-plan artifacts/industrial/activation.json \
  --output artifacts/industrial/next-dispatch.json
```

Formal completed-cell rows bind the frozen physical assignment, exact budget,
terminal receipt, mandatory `BudgetObservationReceipt`, and three-way budget/
terminal digest closure. Missing phase timing, compile/download N/A reasoning,
or GPU accounting is not replaced with zero. Resume accepts only a complete
receipt and never directory presence.

The dispatch plan is target-protocol data, not proof that a cell is executable.
The pinned tree implements the exact native terminal begin/reset/finalize hook,
but no trusted hardware signer is configured. Release preflight therefore runs
only TP1/DP1 Target-only and blocks Static/TTS/L0 before process, filesystem,
root, or network mutation. The CLI does not silently provision hardware.

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

`validate-evidence-alias` does not create scientific equivalence from labels;
it accepts only a fully content-bound byte-equivalent receipt. The dependence
map makes shared controls one observation for covariance/bootstrap purposes.
Static is backend-specific and TTS/L0 are never aliases. This release's formal
reducer rejects a non-singleton dependence unit unless the alias is recomputed
from execution plans and terminal evidence; a self-described alias can validate
structurally but cannot enter a claim.

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
