# CLI

[中文](../zh-CN/cli.md) · [Home](../../README.md)

## Command surface

`lightcone-spec --help` and `lightcone-spec COMMAND --help` are the argument
authority. The schema-v3 and industrial commands are:

| Command | Purpose |
|---|---|
| `doctor` | Read-only host, Python, CUDA, and source identity report |
| `validate-config` | Validate one strict schema-v3 run config and sidecar |
| `build-industrial-registry` | Bind two GPU UUIDs and materialize the immutable experiment DAG |
| `plan-industrial-dispatch` | Validate receipts and emit deterministic one/two-GPU waves |
| `seal-industrial-stage` | Bind a completed stage's runtime, split, dependencies, and locked outputs |
| `analyze-industrial` | Validate schema-v3 terminal evidence and write a bound E3b/E5 analysis manifest |
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

Use immutable device UUIDs, not CUDA ordinals:

```bash
lightcone-spec build-industrial-registry \
  --gpu-uuid GPU-UUID-0 GPU-UUID-1 \
  --base-port 24000 \
  --cache-root runtime-cache/industrial \
  --evidence-root artifacts/industrial \
  --seed 20260811 \
  --output artifacts/industrial/registry.json
```

The output embeds its generator identity, input parameters, complete
declarations, and registry SHA-256. Loading regenerates the registry and
compares the exact content, so hand-edited cells are rejected.

Before any receipt exists, the planner emits only `preflight` waves and keeps
single-GPU work serial because the interference gate has not passed:

```bash
lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
  --output artifacts/industrial/preflight-dispatch.json
```

After a stage's registered outputs are durable, seal them. Every
`--locked-output` is `NAME=LOWERCASE_SHA256`; dependencies are receipt files,
not copied hash strings:

```bash
lightcone-spec seal-industrial-stage \
  --registry artifacts/industrial/registry.json \
  --experiment preflight \
  --runtime-sha256 RUNTIME_SHA256 \
  --split-sha256 SPLIT_SHA256 \
  --locked-output runtime_envelope=OUTPUT_SHA256 \
  --output artifacts/industrial/receipts/preflight.json
```

For a downstream stage, repeat `--dependency-receipt` for its exact declared
dependencies. Then pass all completed receipts to the planner:

```bash
lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
  --receipt artifacts/industrial/receipts/preflight.json \
  --completed-cells artifacts/industrial/completed-cells.json \
  --interference-receipt artifacts/industrial/interference.json \
  --output artifacts/industrial/next-dispatch.json
```

The optional completed-cell artifact must bind each cell to measured evidence
and a terminal receipt. The optional interference artifact must be a matching
`PASS` for the same registry and GPU UUIDs. Without it, cells remain serial.
Exclusive work is never paired, even after it passes.

The dispatch plan is target-protocol data, not proof that a cell is executable.
The library industrial executor validates provider state before launch and
currently runs only TP1/DP1 Target-only end to end. It blocks every speculative
cell before process/network mutation unless an injected provider implements
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v1` for the exact
pinned tree. No such provider ships in this release. The CLI does not silently
launch servers or provision hardware.

## Identity and topology flow

The minimum industrial identity chain is:

```text
source + patch tree + model/data locks + two-GPU capability
                         |
                         v
              immutable registry + traces
                         |
                         v
           dependency receipts + dispatch waves
                         |
                         v
              terminal evidence receipts
                         |
                         v
        derived statistics + GPU attestation + gate
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
native terminal-evidence preflight before an endpoint starts.

Load screening and tuning use only their registered data windows. A later
halving stage must bind the prior survivor artifact. `select-speed-config`
requires complete safe coverage; `select-anchor-config` is a narrower
reproduction route and records that it did not optimize the grid. Confirmation
data cannot enter either selection.

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
legacy attestation is rejected and `analyze-industrial` remains
`UNRESOLVED`/`UNMEASURED` with exit 42 even for self-consistent JSON.

The industrial registry may declare target cells as `UNMEASURED`, but that
declarative status is not executable readiness. Executor preflight resolves
Static/TTS/L0 to `BLOCKED` when the exact native terminal hook is absent; all
TP2/DP2 and DSpark/EAGLE/EAGLE3/NEXTN adaptive cells are likewise blocked by
the current schema/patch. Historical v2 artifacts are regression-only and
cannot be supplied as schema-v3 stage receipts.

## Credentials and output roots

Artifacts, model roots, caches, traces, provider state, profiles, selections,
attestations, and handoff files belong under ignored external roots. Pass model
access through a temporary `HF_TOKEN` environment variable or another secure
channel. Do not place tokens, passwords, provider API keys, private prompts,
instance addresses, or machine-specific paths in arguments, manifests, logs,
documentation, or Git.
