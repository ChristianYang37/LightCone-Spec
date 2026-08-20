# CLI

[中文](../zh-CN/cli.md) · [Home](../../README.md)

## Command surface

`lightcone-spec --help` and `lightcone-spec COMMAND --help` are the argument
authority. The schema-v3 and industrial commands are:

| Command | Purpose |
|---|---|
| `doctor` | Read-only host, Python, CUDA, and source identity report |
| `validate-config` | Validate one strict schema-v3 run config and sidecar |
| `bind-formal-workload-authority` | Bind a release-allowlisted local LiveCodeBench v6 Hard or MATH-500 Level 5 raw source |
| `revalidate-formal-workload-authority` | Reopen a diagnostic workload binding and replay its path, revision, bytes, and complete selection |
| `build-industrial-registry` | Bind one or more stable logical rank slots and materialize the immutable experiment DAG |
| `collect-gpu-inventory` | Collect a nonce-bound physical GPU/topology inventory and raw probe receipt |
| `publish-tts-drafter-native-loss-source` | Publish the sole code-owned pinned DFlash loss descriptor |
| `assemble-gpu-fleet-inventory` | Join repeated single-host inventory/interference pairs into one content-bound fleet inventory |
| `build-interference-envelope` | Derive the current serial interference envelope and its inventory-bound raw receipt |
| `materialize-interference-calibration-bootstrap` | Derive the calibration-only two-way execution envelope from a raw preflight activation and exact inventory |
| `reduce-interference-calibration` | Reopen path-bound execution bundles and terminal authority, then reduce raw isolated/simultaneous evidence into exact-cardinality rules |
| `reduce-e1-activation` | Derive the single 68-cell E1 slice from sealed E3a evidence |
| `reduce-e2-activation` | Materialize one E2 round from the E1 Pareto artifact or prior survivors |
| `reduce-e2-successive-halving` | Reduce exact E2 stage evidence into a sealed survivor receipt |
| `materialize-confirmation-pilots` | Activate exactly four excluded pilots for one confirmation family |
| `reduce-confirmation-family-power` | Reduce one family's exact four pilot blocks into a sealed power plan |
| `materialize-confirmation-prefix` | Activate only a powered family's sealed 12--20-block final prefix |
| `validate-evidence-alias` | Replay one raw alias manifest against registry, hardware, inventory, and terminal authority |
| `build-evidence-dependence-map` | Preserve shared-observation dependence from reducer-issued alias artifacts |
| `materialize-stage-activation` | Replay generic registry-stage dispatchability from raw registry, lineage, runtime, and split authority |
| `materialize-preflight-pointer-coverage` | Deep-reopen one compile, one exactness, and eight interference result authorities into the exact ten-cell preflight activation/coverage receipt |
| `materialize-stage-capacity-gate` | Derive a stage-specific capacity gate from path-bound raw capacity sources and an exact execution-wave schedule |
| `materialize-industrial-budgets` | Derive one fail-closed `BudgetPlan` from reducer, load, inventory, policy, and capacity authority |
| `bind-industrial-budget-authority` | Bind a declared `BudgetPlan` to its complete tagged raw activation/load/capacity closure |
| `estimate-industrial-budget` | Replay one ready `BudgetPlan` on an exact physical inventory and interference envelope |
| `plan-industrial-dispatch` | Freeze deterministic topology-aware GPU-pool waves and physical assignments |
| `materialize-dispatch-execution-bundles` | Bind one path-only raw input graph and publish a complete schema-v5 assignment-bundle set after all-assignment preflight |
| `execute-dispatch-wave` | Reopen one committed host-local manifest for a receipt-bounded wave; `--host-request-stdin` is the noninteractive remote-worker protocol |
| `seal-industrial-stage` | Bind activated completion, dispositions, budgets, runtime, split, dependencies, and locked outputs |
| `analyze-industrial` | Validate schema-v3 terminal, budget, family-power, and hardware evidence |
| `analyze-e3b-long-context` | Validate and reduce the isolated E3b long-context evidence family |
| `build-preliminary-speed-study` | Materialize the historical `PRELIMINARY_DIAGNOSTIC_ONLY` protocol; never industrial authority |
| `lock-models` | Resolve model IDs to immutable revisions |
| `prepare-models` | Download or offline-verify locked snapshots |
| `list-preliminary-tuning-candidates` | Write the historical Full/LoRA diagnostic grid |
| `render-preliminary-target-only-runtime` | Render a diagnostic speculation-disabled Target-only endpoint |
| `render-preliminary-static-load-runtime` | Render a diagnostic allocation-free Static endpoint |
| `render-preliminary-tuning-runtime` | Render historical matched-recipe publication-policy diagnostic endpoints; not TTS reproduction |
| `run-preliminary-controlled-slice` | Measure one preliminary controlled slice |
| `collect-preliminary-static-load-screen` | Validate preliminary Static load coverage |
| `advance-preliminary-tuning-stage` | Validate a preliminary halving stage |
| `select-preliminary-speed-config` | Apply the historical tuning-only rule |
| `select-preliminary-anchor-config` | Lock a preliminary anchor without a claim |
| `render-preliminary-runtime` | Emit historical matched-recipe diagnostic configs and launch argv |
| `build-preliminary-confirmation-queue` | Materialize clean-server diagnostic jobs |
| `run-preliminary-confirmation` | Execute one preliminary method/block slice |
| `run-preliminary-target-reference` | Capture a preliminary Target-only token-ID trajectory |
| `collect-preliminary-speed-study` | Derive a preliminary diagnostic table |
| `render-preliminary-replication-runtime` | Render preliminary natural-task or profiler slices |
| `run-preliminary-natural-slice` | Run one preliminary natural-task slice |
| `build-preliminary-profiler-plan` | Build an isolated diagnostic profile plan |
| `attest-preliminary-speed-study` | Emit a categorical non-authority decision (exit 42) |
| `analyze-preliminary-speed-study` | Produce preliminary statistics only (exit 42) |

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
| `attest-onlinespec-study` | Emit a categorical non-authority decision (exit 42) |
| `analyze-onlinespec-study` | Produce diagnostic learner-versus-Static intervals only (exit 42) |

The historical speed-study manifest/API/CLI surface is strictly
`PRELIMINARY_DIAGNOSTIC_ONLY`. It cannot consume an industrial registry,
activation, budget, completion authority, or materialized bundle, and its
tables and receipts are never accepted by `analyze-industrial`. Historical
schema-v2 manifests remain readable under this forced scope; they cannot be
rewritten or relabelled as formal. The only formal execution command is
`execute-dispatch-wave`, which reopens a materialized bundle and enters the
industrial executor. There is no generic method override or command that
converts an unreceipted directory into completed evidence.

The bootstrap envelope is permission to generate only the eight registered
Static calibration observations. It cannot authorize headline co-tenancy or a
larger cardinality. `reduce-interference-calibration` checks the release trust
root before opening caller paths; the current no-signer release therefore
writes a `BLOCKED` decision and no calibrated envelope. A preflight stage can
seal `runtime_envelope=PATH` only by reopening that same raw execution
authority, never from a hand-authored rule list or digest.

## Compile-cache plan binding

Every preliminary and OnlineSPEC runtime-render command requires
`--compile-cache-plan /absolute/path/to/plan.json`. This is diagnostic launch
authority, not attestation or permission to enter the formal DAG. Derive its
`CompileCacheKey` programmatically from a passing `doctor` report, the exact
model lock, and the rendered RunConfig; do not type expected host values from
memory. The current diagnostic launcher accepts only `bfloat16`,
`cuda_malloc_async`, graph buckets `(1,)`, and no caller build flags. It
re-observes Python, Torch, Triton, the Torch CUDA build, `nvcc`, driver, selected
GPU model, and SM before creating a cache attempt, and applies the allocator
environment before importing Torch.

Derive the key and issue an immutable diagnostic build plan with the source API:

```python
import json
from pathlib import Path

from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.locking.models import ModelLock
from lightcone_spec.orchestration.runtime import (
    build_target_only_run_config,
    derive_diagnostic_compile_cache_key,
)
from lightcone_spec.runtime.compile_cache import CompileCacheLaunchPlan

doctor = json.loads(Path("/absolute/runtime/doctor.json").read_text())
model_lock = ModelLock.load("/absolute/runtime/model-lock.json")
sampling = SamplingProfile.load("/absolute/runtime/sampling-profile.json")
devices = doctor["gpu"]["parsed_inventory"]["devices"]
if len(devices) != 1:
    raise RuntimeError("Target-only diagnostic requires one visible doctor GPU")
gpu_uuid = devices[0]["uuid"]
config = build_target_only_run_config(
    concurrency=1,
    gpu_uuid=gpu_uuid,
    model_lock=model_lock,
    sampling_profile=sampling,
)
key = derive_diagnostic_compile_cache_key(
    doctor_report=doctor,
    model_lock=model_lock,
    config=config,
)
cache_root = Path("/absolute/runtime/compile-cache")
build = CompileCacheLaunchPlan.issue(
    key=key,
    cache_root=cache_root,
    cache_mode="build",
)
build_path = build.write(Path("/absolute/runtime/build-plan.json"))
```

Pass `build_path` and the same `gpu_uuid`, concurrency, model lock, and sampling
profile to the render command. The renderer reconstructs the exact source-owned
RunConfig and embeds the expected plan, key, and RunConfig SHA-256 values in the
child argv. For example:

```bash
lightcone-spec render-preliminary-target-only-runtime \
  ... \
  --gpu-uuid GPU-01234567-89ab-cdef-0123-456789abcdef \
  --compile-cache-plan /absolute/runtime/build-plan.json
```

`--gpu-uuid` is required and must be the single physical `GPU-*` UUID selected
from the passing doctor inventory. The renderer serializes it into the existing
`RuntimeConfig.device_identity`; the official launcher reopens that exact
RunConfig and binds the same UUID to `CUDA_VISIBLE_DEVICES`. Comma-separated,
whitespace-containing, and non-`GPU-*` selectors are rejected before runtime
files or a subprocess are created.

The child stably reopens both canonical artifacts and their sidecars, derives
the exact algorithm and draft width from the RunConfig, and rejects identity or
argv changes before cache, model, or GPU mutation. It also requires exactly one
usable Torch CUDA device and refuses to initialize the allocator after Torch
was imported. Do not edit, regenerate, move, or replace an artifact after
rendering.

Preliminary model roots bind locked revision paths, but do not carry the formal
`PreparedModelContentAuthority` replay. The launcher therefore accepts build
mode only and seals its cache receipt as unattributed: it cannot authorize
reuse, formal execution, attestation, or `MEASURED`. Although the source API
supports a reuse plan for a completed release-builder receipt with the same key
and cache root, no preliminary receipt qualifies. Derive a new build plan until
the formal content-authority execution path supplies that receipt.

## Formal external workload authority

Formal benchmark data is never downloaded by these commands. The bind command
accepts only `livecodebench_v6_hard` or `math500_level5`; source code owns the
exact repository revision, raw-file SHA-256, row counts, filtering protocol,
and complete selected-row SHA-256. It selects every exact protocol match in raw
order and never takes the first 32 rows.

```bash
lightcone-spec bind-formal-workload-authority \
  --workload math500_level5 \
  --source /absolute/path/to/math500-locked.json \
  --content-verification-receipt /absolute/path/to/content-E3a.json \
  --now-ns 1786900000000000000 \
  --output artifacts/industrial/math500-workload-authority.json

lightcone-spec revalidate-formal-workload-authority \
  --authority artifacts/industrial/math500-workload-authority.json \
  --content-verification-receipt /absolute/path/to/content-E3a.json \
  --now-ns 1786900000000000000
```

Both commands require a current, root-verified content-verification receipt and
an explicit verification time. Missing or invalid content authority fails
closed before the raw workload can be bound. A successful bind reports
`BOUND_AUTHORIZED_CONTENT`, but the emitted wrapper still permanently carries
`formal_execution_authorized=false`; it is workload input identity that must be
replayed with the same verified content authority, not GPU dispatch,
attestation, or a formal result. The legacy source-allowlist API remains a
non-authorizing diagnostic and is not the production CLI path.

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
generic command always writes the canonical diagnostic decision and returns
zero when that write/reload succeeds, including when the decision is
`BLOCKED`. Exit status is not execution permission. Formal preflight becomes
`AVAILABLE` only through the first-party pointer reducer, which deep-reopens
exactly one compile result, one exactness result, and eight interference
serving results and requires all ten mandatory cells terminal with zero skip:

```bash
lightcone-spec materialize-preflight-pointer-coverage \
  --registry artifacts/formal/registry.json \
  --runtime-artifact artifacts/formal/runtime.json \
  --split-artifact artifacts/formal/preflight-split.json \
  --compile-result artifacts/formal/compile-result.json \
  --exactness-result artifacts/formal/exactness-result.json \
  --interference-authority artifacts/formal/interference-execution.json \
  --source-output artifacts/formal/preflight-source.json \
  --activation-output artifacts/formal/preflight-activation.json \
  --coverage-output artifacts/formal/preflight-coverage.json
```

The reducer reopens the compile cache/result pointer, JUnit and rank/native
exactness artifacts, and every interference terminal authority. A caller-entered
terminal digest, partial coverage, failed/error/skipped case, missing rank, or
foreign runtime/split identity is rejected.

Budget materialization is fail-closed. The legacy diagnostic command consumes reducer-generated stage or
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

Formal staged execution uses the separate schema-3 capacity gate. It reopens
the raw provider/host/sizing sources and exact stage schedule, sums retained
evidence (including registered retries), adds only the maximum concurrently
resident staging/compile bytes for each execution wave, then adds the fixed
safety margin. The command accepts no caller-entered free-space or high-water
number:

```bash
lightcone-spec materialize-stage-capacity-gate \
  --registry artifacts/formal/registry.json \
  --capacity-source-manifest artifacts/formal/capacity-sources.json \
  --stage-schedule artifacts/formal/preflight-capacity-schedule.json \
  --now-ns 1786900000000000000 \
  --output artifacts/formal/preflight-capacity-gate.json
```

Formal dispatch then requires a local dynamic `capacity` control attestation
for that exact gate and atomically reserves its challenge with the other
controls. Missing signed authority retains the 100 GB fail-closed fallback;
source tamper, schedule drift, or an undersized observed envelope is an error,
not permission to fall back.

There is no caller-selected-cell or duration-sequence fallback. Formal
preflight closes through the typed pointer reducer; omitting its bound raw
sources is an explicit unresolved authority error, and a caller-authored
activation cannot bridge it. Serving activations additionally repeat
`--budget-load-binding PATH` once per activated serving cell.

Formal consumers do not accept the `BudgetPlan` alone. After materialization,
publish its path-bound raw closure with the tagged activation manifest (not a
serialized activation summary):

```bash
lightcone-spec bind-industrial-budget-authority \
  --activation-manifest artifacts/industrial/preflight-activation-manifest.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --capacity-manifest artifacts/industrial/capacity-source-manifest.json \
  --capacity-verification-receipt artifacts/industrial/capacity-verification.json \
  --budget-plan artifacts/industrial/budget-plan.json \
  --output artifacts/industrial/budget-authority.json
```

Repeat `--budget-load-binding PATH` for every activated cell. The command
reopens all sources, reruns the registered reducer, exact-compares the declared
plan, and writes only the resulting binding; it never turns an `UNRESOLVED`
plan into execution permission.

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

`materialize-dispatch-execution-bundles` closes the structural boundary between
the frozen planner output and formal execution. Its schema-v1 request contains
only absolute resolved raw-artifact paths, their source roles, and one
assignment-runtime path set for every dispatch cell. It deliberately contains
no caller-supplied assignment SHA, execution-plan SHA or summary, output root,
or semantic hash. The reducer reopens the request and every sidecar, derives
the assignments from the frozen dispatch plan, and requires an exact one-to-one
assignment cover before it can construct schema-v5 bundles.

Every assignment's raw activation/completion, budget, capacity, interference,
topology, runtime, sampling, model, compile, nonce, launch-policy, and optional
trainable/failure authority is preflighted before the fresh publication
directory or any renderer artifact is created. Tagged generic, E1, E2,
confirmation-pilot, confirmation-final, and stage-aggregate activation
authorities are replayed from their complete path closure. Final prefixes use
only schema-v4/native completion authority; bare completed-cell IDs are
rejected. E3b/E5 family aggregates remain separate, and E5 additionally
requires the deterministic auxiliary activation/completion cover for its 264
non-family failure-injection cells. Formal reconstruction also requires a live
`READY` `BudgetPlan` and its path-bound materialization authority; a
self-consistent planner or execution summary cannot authorize itself.

The output path must be absolute, resolved, absent, and below a stable
owner-private parent. After all-assignment preflight succeeds, the command
creates a private `0700` directory, writes each schema-v5 bundle and adjacent
sidecars exclusively, and writes
`dispatch-execution-bundle-manifest.json` plus its sidecar last as the commit
marker. An interrupted partial directory is retained but has no consumable
manifest; retry uses a different fresh directory rather than deleting evidence.

```bash
lightcone-spec materialize-dispatch-execution-bundles \
  --request /absolute/path/to/dispatch-bundle-request.json \
  --output-directory /absolute/private/path/dispatch-bundles-attempt-0001
```

`execute-dispatch-wave` never treats the planner JSON, an
`IndustrialExecutionPlan.to_dict()` summary, or raw bundle paths as launch
authority. It accepts exactly one required `--materialization-manifest`. After
the release dispatch trust gate, the loader reopens the manifest, its request
and dispatch plan, every schema-v5 member, and the complete path-bound
construction graph. It rejects changed input bytes, swapped membership,
incomplete assignment coverage, or bundles outside the manifest directory.
Only then does execution replay the shared scheduler authority and construct
the requested wave's physical plans (plus prior plans named by a resume
receipt). Compile/download and unsupported serving assignments remain
explicitly blocked, and every potentially launchable plan still crosses the
full public validation boundary.

The command executes exactly one `--wave-index`. Wave zero has no resume input;
a successful prefix requires the previous `--resume-receipt`, while a failed
wave resumes at that same index. Before any sibling runner starts, the command
durably appends one wave intent and every per-assignment intent to a private
`RECEIPT.attempt-journal`; each terminal or failed return appends its finish,
exact retry identity, and cumulative monotonic cost. A partial wave therefore
keeps successful siblings and retries only failed siblings. An intent without
a finish has unknowable cost and blocks under
`dispatch_attempt_intent_without_finish_cost_unresolved`.

The attempt-journal manifest is schema-v2 for formal manifest-based dispatch
and binds the exact materialization-manifest SHA-256. Reopening the same journal
through a different bundle publication is therefore an identity mismatch, even
if the dispatch plan is otherwise unchanged.

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
  --materialization-manifest \
    /absolute/private/path/dispatch-bundles-attempt-0001/dispatch-execution-bundle-manifest.json \
  --wave-index 0 \
  --receipt-output artifacts/industrial/dispatch-wave-0.json
```

The source release has no trusted hardware attester in its release-owned trust
policy. A caller or test signer cannot unlock formal execution, even with a
cryptographically valid signature, and a bare `CapacityEnvelope` cannot grant
execution authority. Given an otherwise complete request, materialization
therefore returns `BLOCKED`/42 at the all-assignment release preflight before
creating its publication directory. Formal execution independently returns
`BLOCKED`/42 at its entry trust gate before reading the materialization
manifest, creating the receipt parent or evidence root, importing the serving
client, or launching a process. A bundle publication is structural authority,
not GPU attestation or permission to execute. Fresh execution rejects
unjournaled pre-existing per-plan trace files. Resume requires the raw
append-only attempt chain plus revalidated structured terminal bindings;
neither a bare terminal digest nor a caller-rehashed schedule JSON can skip
work. If a coordinator dies after durable intent but before durable finish,
execution remains blocked rather than inventing a monotonic cost.

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

Actual preflight GPU hours are materialized only from the finalized, sealed
preflight chain. The command accepts no duration, run-count, or reserve scalar.
It reopens the sole compile and exactness lifecycle authorities transitively
from `--source-authority`, and it requires exactly eight distinct path-bound
interference lifecycle proofs whose cell IDs match the finalized 1+1+8 source:

```bash
lightcone-spec materialize-preflight-gpu-hour-envelope \
  --dispatch-receipt /absolute/evidence/preflight-dispatch.json \
  --remote-raw-receipt /absolute/evidence/preflight-remote-raw.json \
  --source-authority /absolute/evidence/preflight-source.json \
  --activation /absolute/evidence/preflight-activation.json \
  --coverage /absolute/evidence/preflight-pointer-coverage.json \
  --stage-coverage /absolute/evidence/preflight-stage-coverage.json \
  --interference-lifecycle-proof 9a4d4b4f84399bbd9e33e542d110e237d33ef0d1f738d60406399611cadaf6d6=/absolute/evidence/lifecycle-0.json \
  --interference-lifecycle-proof a88676a576111058c46b44dc2754a4d171e6c8f2226c3ebb65f2db4716c5c253=/absolute/evidence/lifecycle-1.json \
  --interference-lifecycle-proof 11de28835716cb8fa5af2dea84571e2ae930f6f3642a6ccecaaa46529ca76f10=/absolute/evidence/lifecycle-2.json \
  --interference-lifecycle-proof 2761af3f18adceab26d68bc7887b50f21575c8f54062738ff893d402ffa5c32e=/absolute/evidence/lifecycle-3.json \
  --interference-lifecycle-proof 1e7933ad9c1b95af086d1a8626d882beb4f376987786e5799b2e440f4ae536dd=/absolute/evidence/lifecycle-4.json \
  --interference-lifecycle-proof 8902eb962b1ab7703a29446c1f4bb56f183d620496043e02814bfd1d0d94a630=/absolute/evidence/lifecycle-5.json \
  --interference-lifecycle-proof 52f11f9c36ea5016a0cba90b4a1ae843b77a15be00936df846f8a9a6cfe620f8=/absolute/evidence/lifecycle-6.json \
  --interference-lifecycle-proof e93df280d22db443b27ee32e6dc95eed9bc7ecb6a1f4a439dddf57c93c5206e5=/absolute/evidence/lifecycle-7.json \
  --formal-runtime-authority-manifest /absolute/evidence/runtime-authority.json \
  --source-output /absolute/evidence/preflight-gpu-hour-source.json \
  --now-ns 2000000000 \
  --output /absolute/evidence/preflight-gpu-hour-envelope.json
```

Both output paths must be distinct, absolute, normalized, and absent. The
source manifest and schema-2 envelope are published atomically without
replacement. The envelope is accepted directly by the offline scientific
signer's `stage-gpu-hour-envelope` decoder; the finalized signed wrapper and
the source manifest then enter `reserve-formal-stage-gpu-hours`. An ordinary
formal-stage counterpart is intentionally unavailable until the public,
path-bound stage-source rebuild bundle exists; a digest, generic JSON, or a
serialized private execution seal is never a substitute.

### Source-owned method authorities

Before creating `ProtocolLock`, publish the three code-owned method inputs from
their exact bound sources:

```bash
lightcone-spec publish-tts-calibration-tuning-window \
  --tuning-workload-authority /absolute/source/lcb-hard-authority.json \
  --content-verification-receipt /absolute/evidence/content-master.json \
  --output /absolute/source/tts-tuning-window.json

lightcone-spec publish-tts-drafter-native-loss-source \
  --output /absolute/source/drafter-native-loss.json

lightcone-spec publish-tts-calibration-source-authority \
  --paper-pdf /absolute/source/tts-v2.pdf \
  --paper-source /absolute/source/tts-v2-source.tar.gz \
  --tuning-workload-authority /absolute/source/lcb-hard-authority.json \
  --content-verification-receipt /absolute/evidence/content-master.json \
  --tuning-window /absolute/source/tts-tuning-window.json \
  --trainable-plan-authority /absolute/source/trainable-plan.json \
  --drafter-native-loss /absolute/source/drafter-native-loss.json \
  --output /absolute/evidence/tts-calibration-authority.json

lightcone-spec publish-chronobelief-source-authority \
  --paper-pdf /absolute/source/chronobelief.pdf \
  --tex-source /absolute/source/chronobelief.tex \
  --output /absolute/evidence/chronobelief-authority.json

lightcone-spec publish-e1-recipe-anchor-authority \
  --trusted-content-bundle /absolute/source/trusted-content.json \
  --trainable-plan-authority /absolute/source/trainable-plan.json \
  --output /absolute/evidence/e1-recipe-anchor-authority.json
```

The schema-2 master content ceremony must run first and explicitly forbids the
TTS tuning window from its content set. These commands then deep-read the
durable master receipt, its exact replay-reservation record, the original
signed workload authorization at the receipt's recorded verification time,
and the bound LCB authority. A raw or expired workload authorization is not a
substitute for the receipt. The schema-4 window binds the receipt digest,
verification time, and reservation identity and is published atomically
without replacement. The commands accept no caller-supplied authority digest,
tuning partition, time, or recipe override. The code-owned tuning selector
hashes every exact LCB-hard problem ID under its versioned namespace, excludes
the four smallest hashes, and binds the complete complement. The TTS source
publisher reopens an exact descriptor of the pinned DFlash loss: float32
target-to-draft forward KL, valid-row masking, position weights
`exp(-(k-1)/7)`, temperature one, and masked weighted normalization. The
source-point value correction is not an independent proximal penalty, so there
is no proximal coefficient to supply or tune. With the exact source pins,
post-master window, canonical plan, and loss descriptor, it can publish the
`ProtocolLock` input. This is a project-calibrated runtime baseline rather than
paper reproduction. The ChronoBelief command remains a separate project-owned
preregistration.
The current E1 publisher additionally requires one runtime-`BOUND` trusted
content bundle. Its schema-3 artifact binds and reopens that exact bundle and
joins the plan's target/drafter revisions and prepared roots to it. Historical
schema-2 E1 artifacts remain load-only and cannot enter a trusted schema-5
ProtocolLock. The E1 anchor publisher accepts only its explicit structural selector:
the unique Qwen3-8B/DFlash LC-candidate `full`/`last1` AdamW anchor at the
registered width-8, concurrency-4 slot. A different but otherwise valid plan
is rejected; this selector does not encode an observed winner.

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

These checks define the claim-bearing path once the current session's dynamic
hardware and typed control chain is verified. Without that chain, sealing
returns `BLOCKED`; the raw-authority contract alone does not make execution or
a performance claim reachable. The sealed output can authorize only the
LightCone role. It can never rewrite the separately frozen TTS or L0-naive
recipe authority.

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
The pinned tree implements the compile, exact native terminal, and coverage
contracts, while the package pins only the offline public root. Generic
release-attested activation records a diagnostic disposition; it does not
create result pointers, a fresh dynamic hardware policy, execution authority,
or a performance claim. Every role remains `BLOCKED` in that lane until its
exact content, qualification, terminal, and control artifacts are present. The
trusted `formal_single_operator_v1` CLI below is a separate empirical lane: it
does not require an external signer, but it does require the exact
source/content/runtime, fresh GPU qualification, capacity, terminal, and
coverage artifacts. The CLI never silently provisions hardware or starts a
GPU.

## Fleet inventory and remote host waves

Fleet assembly consumes one `--inventory PATH` and one
`--interference-envelope PATH` per host. Both options are repeated, and their
positions form the pairs; mismatched counts are rejected. Every input inventory
must describe exactly one host, and every envelope remains bound to that host's
hardware identities.

```bash
lightcone-spec assemble-gpu-fleet-inventory \
  --inventory /external/host-a/inventory.json \
  --interference-envelope /external/host-a/interference.json \
  --inventory /external/host-b/inventory.json \
  --interference-envelope /external/host-b/interference.json \
  --output /external/fleet/inventory.json
```

`GpuFleetInventory`, `GpuFleetScheduler`, and `GpuFleetDispatchPlan` are the
Python control-plane API for distributing independent assignments.
`GpuFleetScheduler` only selects a host and serial partition; every child plan
is issued by the sole same-host `GpuPoolScheduler` and binds `host_id`, the host
inventory digest, physical GPU UUIDs, and port/cache/evidence/contention
resources that are collision-free within that host. Literal resource values may
repeat on different hosts. The separate remote execution binding fixes the
host-local execution manifest. A complete gang, matched-geometry baseline
anchor set, and confirmation block stay on one host. A gang that would
span hosts is rejected with `cross_host_collectives_unvalidated`; fleet
composition never creates a cross-host rendezvous.

Remote orchestration is also currently a Python-library coordinator API. It uses
coordinator-local `SshHostRoute` values and `execute_fleet_wave`; there is no
public fleet-coordinator CLI. Routes require an SSH agent and a fixed
`known_hosts` file. A path-free route-authority digest binds the destination,
port, and exact known-host bytes to the attempt. Address, user, agent socket,
known-hosts path/key bytes, and raw stdout/stderr remain outside serialized
requests and receipts. Passwords,
tokens, private keys, and provider credentials must not be placed in argv,
stdin, artifacts, or logs.

The remote node exposes exactly one worker entry point:

```bash
lightcone-spec execute-dispatch-wave --host-request-stdin
```

This mode is for the coordinator, not an interactive shell. It is mutually
exclusive with `--materialization-manifest`, `--wave-index`,
`--resume-receipt`, and `--receipt-output`; it reads one size-bounded canonical
request from standard input. The worker reopens the declared absolute
host-local manifest and writes one bounded canonical response to standard
output. A local pre-dispatch rejection is a failed transport outcome. Once
dispatch may have occurred, timeout, connection loss, truncated or invalid
output, and missing authority all produce `REMOTE_OUTCOME_UNKNOWN`; none can be
interpreted as completion or retried directly.

A failed host does not invalidate already completed receipts from other hosts.
An unknown outcome is reconciled only by an independent fetch under the exact
original destination, port, and known-host-key authority. The exact receipt and
evidence bytes are then content-validated locally; missing, forged, incomplete,
or oversized input remains unknown.
Only a terminal-negative result may create a new receipt-bound attempt for that
same host. Download, compile, profiler, and shared-I/O contention remain
host-exclusive, headline concurrency never exceeds that host's calibrated
envelope, and fleet SSH concurrency is bounded separately.

No command accepts a caller-selected trusted-attester bundle as release
authority. The installed package pins the offline Ed25519 public root and
fingerprint. A fresh root-signed deployment policy supplies the observed
hardware allowlist and typed control keys without changing source HEAD; the
private key remains offline. Missing, expired, replayed, wrong-key, or
hardware-mismatched policy keeps formal dispatch fail-closed.

The local-only signer is invoked as a Python module, not on the GPU host:

```bash
python -m lightcone_spec.runtime.offline_signer sign-deployment --help
python -m lightcone_spec.runtime.offline_signer sign-control --help
python -m lightcone_spec.runtime.offline_signer sign-scientific --help
python -m lightcone_spec.runtime.offline_signer finalize-scientific --help
```

No signing subcommand has a key-path or key-bytes option. Without `--key-fd`, it
requires an interactive unechoed TTY path prompt. With `--key-fd`, the argument
is only the numeric inherited descriptor. Outputs are canonical public JSON
and are created atomically without replacement; signing errors use a generic
message so a prompted private path cannot enter logs. `sign-scientific`
accepts only the closed set of typed scientific wrappers; its candidate must
then pass `finalize-scientific`, which revalidates policy and atomically records
the challenge in a private single-use ledger. There is no generic JSON signer.

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

TP2 and sticky DP2 are implemented source contracts, but the source capability
table is CPU-audited identity rather than GPU proof. A distributed `RunConfig`
must bind that exact capability plus a receipt claim, and formal dispatch must
verify the matching durable GPU qualification artifact. The CPU `gloo`
contract or a caller-authored digest cannot enable execution. The proof binds
rank, UUID, rendezvous, router, clock, process group, ownership, all-rank
publication, and terminal identity exactly.

Generated JSON uses an adjacent `.sha256` sidecar where the command contract
requires one. The loaders validate canonical content, not only filenames.

## Core tuning and confirmation

Target-only and Static render without adaptation state or reserve. TTS uses the
recipe frozen by the exact TTS-Cal seal with fixed-barrier publication;
L0-naive uses that same frozen recipe authority with first-ready publication.
The release-attested lane signs that seal; the trusted single-operator lane
uses the reducer-owned content identity without an external signer.
The historical `TTS-paper-reconstruction` authority is diagnostic only. E1/E2 `l0`
search recipes are LC-candidates, and only the exact E2-sealed winner is
LightCone. Shared runtime code does not merge live candidate, optimizer, or
evidence identity. Rendering is pure planning and grants no role execution
permission. TTS-Cal fixes the numeric reconstruction grid and semantics; TTS
and L0-naive remain blocked until the disjoint tuning evidence seals one exact
winner. All five roles still require their fresh content, runtime, terminal,
capacity, and dynamic GPU authorities before an endpoint starts. Only
release-level `MEASURED` promotion additionally requires the external
attestation chain.

The industrial E1/E2 commands are reducer-owned. E1 consumes sealed E3a and
TTS-Cal receipts and materializes exactly 68 cells: four fixed roles plus two
LightCone candidates for each of 32 geometries. L0-naive remains a mechanism
anchor and never enters candidate ranking. For `g` surviving geometries, E2
materializes `n0 = 105g` recipes (seven optimizers by three schedules by five
learning rates), then requires the prior sealed survivor receipt for each of
three later rounds with `n(k+1) = max(ceil(nk/4), 21)`. Four fixed anchors are
added per round. It ranks only LightCone candidates against fixed Static and
frozen-TTS references; no command expands an eager sentinel matrix.
Confirmation planning
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
backend-specific and adaptive scientific roles are never aliases. The schema-v3 industrial
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
`optimized_grid_claim=false`. OnlineSPEC remains TP1/DP1 and diagnostic-only;
its attestation command can only emit a non-authority decision.

## Status, resume, and exit codes

`analyze-preliminary-speed-study` always writes
`PRELIMINARY_DIAGNOSTIC_ONLY` and exits 42. It rejects every attestation, even
if a future trusted attester is configured. `attest-preliminary-speed-study`
and `attest-onlinespec-study` likewise emit categorical non-authority decisions
and exit 42. Identity, schema, I/O, receipt, or runtime errors are ordinary
nonzero failures, not scientific outcomes.
The source release pins a public trust root but no reusable hardware identity.
A supplied legacy attestation is rejected. `analyze-industrial` can emit
`MEASURED` only after the exact current-session root-authorized hardware and
typed control chain is verified; self-consistent diagnostic JSON still exits
42.

The industrial registry may declare target cells as `UNMEASURED`, but that
declarative status is not executable readiness. The source paths for TP2/DP2,
DSpark, NEXTN, native ITL, and session reuse are
`implemented_pending_dynamic_gpu_proof`; they remain `BLOCKED` without those
exact proofs. TTS/L0-naive require the sealed TTS-Cal winner, LightCone the
sealed E2 winner, and EAGLE3 an official compatible model/selector decision.
Historical v2 artifacts are regression-only and cannot be supplied as
schema-v3 stage receipts.

### Target-output reference diagnostic

Against a separate locked Target-only server,
`run-preliminary-target-reference` records
per-prompt token counts and format-tagged SHA-256 values of complete ordered
output-token-ID arrays. It never substitutes a decoded-text digest. Legacy
collectors require `--target-reference` and reject a method/block unless its
trajectory matches the reference; agreement among speculative methods alone is
not an exactness proof. The result remains
`PRELIMINARY_DIAGNOSTIC_ONLY` regardless of hardware availability.

## Credentials and output roots

Artifacts, model roots, caches, traces, provider state, profiles, selections,
attestations, and handoff files belong under ignored external roots. Pass model
access through a temporary `HF_TOKEN` environment variable or another secure
channel. Do not place tokens, passwords, provider API keys, private prompts,
instance addresses, or machine-specific paths in arguments, manifests, logs,
documentation, or Git.

## Trusted single-operator formal workflow

### Claim boundary

`formal_single_operator_v1` is the path for a trusted operator to collect and
reduce the complete v03 empirical campaign from one controlled clean checkout.
It records canonical SHA-256 provenance and publishes only to new,
non-overwriting directories. It is neither an adversarial attestation mode nor
permission to call an unsigned result formal `MEASURED` evidence. Without the
current release-root attestation, finalization must report
`trusted_single_operator_empirical_no_signature`,
`formal_measured=false`, and `UNMEASURED`.

The public supervisor commands are:

| Command | Purpose |
|---|---|
| `formal-single-operator status` | Report source-code capability for all 21 nodes without reading run evidence or allocating a GPU |
| `publish-v03-model-lock` | Publish the code-owned exact-19 model/revision lock without network resolution |
| `write-v03-e0-raw-source-path-inputs` | Canonically publish the registry-exact seven raw E0 paths |
| `publish-v03-e0-source-authorities` | Scan those seven sources and publish their typed authorities |
| `write-v03-content-path-inputs` | Canonically publish exact model/data/E0/inventory and future-doctor paths |
| `publish-v03-content-path-spec` | Derive the digest-free pre-doctor content path spec |
| `publish-content-replay-authority` | Deep-scan each unique model snapshot once and publish its path-bound replay receipt |
| `publish-stage-capacity` | Bind the pre-doctor spec, run-root filesystem, and required free-byte threshold |
| `publish-trusted-content` | Deep-bind one typed content path spec and its runtime observations into a BOUND bundle |
| `publish-preflight-workload` | Derive the preflight workload authority from that exact BOUND bundle |
| `publish-tts-cal-trainable-plan` | Derive the fixed trusted TTS-Cal plan from BOUND content |
| `publish-e1-anchor-trainable-plan` | Derive the fixed trusted E1 anchor plan from BOUND content |
| `publish-onlinespec-source-authority` | Bind the audited external OnlineSPEC checkout for E0 only |
| `build-trusted-protocol-lock` | Build the schema-v5 trusted ProtocolLock from path-bound sources |
| `write-dag-driver-config` | Publish one immutable path-only 21-node driver config |
| `write-bootstrap-config` | Publish the non-LLM supervisor config |
| `bootstrap-once` | Advance at most one durable controller/scheduler cycle |
| `bootstrap-run` | Block until the DAG completes or reaches a genuine unresolved block |

The lower-level `materialize-node`, `prepare-run`, `execute-run`,
`finalize-run`, and reducer commands remain useful for focused replay and
debugging. Do not interleave them with a running bootstrap supervisor or start
a second scheduler.

### Publish immutable inputs

Pre-create a private source directory, an empty E0-authority directory, and the
private run root outside Git. Then use this order. It deliberately publishes a
pre-doctor path spec first, derives capacity from that spec, publishes the fresh
doctor report, and only then seals the runtime-`BOUND` content bundle. No step
depends on an artifact produced later in the sequence.

If the exact 19 snapshots are not already present, publish the code-owned lock
and let the existing model preparer fetch only those immutable revisions. The
lock publisher takes no model, revision, digest, token, or network input. Use
`--offline` when auditing an already-populated cache:

```bash
lightcone-spec formal-single-operator publish-v03-model-lock \
  --output /absolute/sources/formal-v03-model-lock.json

lightcone-spec prepare-models \
  --lockfile /absolute/sources/formal-v03-model-lock.json \
  --model-cache /absolute/models \
  --output /absolute/sources/formal-v03-prepared-model-roots.json
```

Collect the nonce-bound physical inventory, write the exact-seven E0 raw-source
handoff, and derive the seven source authorities:

```bash
lightcone-spec collect-gpu-inventory \
  --challenge-nonce-sha256 FRESH_64_LOWERCASE_HEX_NONCE_SHA256 \
  --receipt-output /absolute/sources/gpu-inventory-receipt.json \
  --output /absolute/sources/gpu-inventory.json

lightcone-spec formal-single-operator write-v03-e0-raw-source-path-inputs \
  --source AIME-2025=/absolute/cache/e0/aime25-test.jsonl \
  --source Alpaca=/absolute/cache/e0/alpaca-eval.json \
  --source Arena-Hard=/absolute/cache/e0/arena-hard-v2.0-question.jsonl \
  --source GSM8K=/absolute/cache/e0/gsm8k-test.jsonl \
  --source HumanEval=/absolute/cache/e0/humaneval.jsonl.gz \
  --source MBPP=/absolute/cache/e0/mbpp-sanitized.json \
  --source MT-Bench=/absolute/cache/e0/mt-bench-question.jsonl \
  --output /absolute/sources/e0-raw-source-path-inputs.json

lightcone-spec formal-single-operator publish-v03-e0-source-authorities \
  --inputs /absolute/sources/e0-raw-source-path-inputs.json \
  --output-directory /absolute/sources/e0-authorities
```

Next write the registry-complete content handoff. The 19 model keys and paths
below are source-owned: the CLI accepts no revision or digest override. The six
BurstGPT and seven E0 names must also appear exactly once. The doctor output
must still be absent, although its resolved parent already exists.

```bash
lightcone-spec formal-single-operator write-v03-content-path-inputs \
  --repository-root /absolute/clean/lightcone-checkout \
  --model-snapshot gemma4_12b_dflash_e0=/absolute/models/models--deepseek-ai--dflash_gemma4_12b_block7/snapshots/7490ce60c7630107917fe558e2bbe3dcec6195cb \
  --model-snapshot gemma4_12b_dspark_e0=/absolute/models/models--deepseek-ai--dspark_gemma4_12b_block7/snapshots/2fa72e765eec2965fc4d86a8663ce6769eba6218 \
  --model-snapshot gemma4_12b_eagle3_e0=/absolute/models/models--deepseek-ai--eagle3_gemma4_12b_ttt7/snapshots/0bc24c312350910419cf371e54082f040d65cc82 \
  --model-snapshot gemma4_12b_target=/absolute/models/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --model-snapshot qwen35_122b_a10b_fp8_nextn=/absolute/models/models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9 \
  --model-snapshot qwen36_35b_a3b_nextn=/absolute/models/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --model-snapshot qwen3_14b_dflash_e0=/absolute/models/models--deepseek-ai--dflash_qwen3_14b_block7/snapshots/ab0a8b28236654620bb41d64b336d00a14cb467f \
  --model-snapshot qwen3_14b_dspark_e0=/absolute/models/models--deepseek-ai--dspark_qwen3_14b_block7/snapshots/83207b416acf99f41c2184648923632fccea6dd0 \
  --model-snapshot qwen3_14b_eagle3_e0=/absolute/models/models--deepseek-ai--eagle3_qwen3_14b_ttt7/snapshots/d7ea05d0b0009badfff0df2dcaedf82cce0f74f8 \
  --model-snapshot qwen3_14b_target=/absolute/models/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18 \
  --model-snapshot qwen3_4b_dflash_e0=/absolute/models/models--deepseek-ai--dflash_qwen3_4b_block7/snapshots/02d530b7962ea1412beaf41a05c0b8e36d5f9b1d \
  --model-snapshot qwen3_4b_dspark_e0=/absolute/models/models--deepseek-ai--dspark_qwen3_4b_block7/snapshots/3457dff1417cb84927f6098a5fcb7cee85c934b7 \
  --model-snapshot qwen3_4b_eagle3_e0=/absolute/models/models--deepseek-ai--eagle3_qwen3_4b_ttt7/snapshots/b0b90fd15d052217c226be5e46d468d8d129e0cd \
  --model-snapshot qwen3_4b_target=/absolute/models/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --model-snapshot qwen3_8b_dflash_core=/absolute/models/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4 \
  --model-snapshot qwen3_8b_dflash_e0=/absolute/models/models--deepseek-ai--dflash_qwen3_8b_block7/snapshots/9e44dbbb6cb68b0c943abf9c5fc3c17c00897cdf \
  --model-snapshot qwen3_8b_dspark_e0_core=/absolute/models/models--deepseek-ai--dspark_qwen3_8b_block7/snapshots/03326e5043815da1f81b109078b2889737c26017 \
  --model-snapshot qwen3_8b_eagle3_e0=/absolute/models/models--deepseek-ai--eagle3_qwen3_8b_ttt7/snapshots/f6485ba8d21e11942958617dbe7e71b467f38f38 \
  --model-snapshot qwen3_8b_target=/absolute/models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --livecodebench-raw /absolute/cache/livecodebench/test6.jsonl \
  --math500-raw /absolute/cache/math-500/test.jsonl \
  --burstgpt-asset BurstGPT_1.csv=/absolute/cache/burstgpt/BurstGPT_1.csv \
  --burstgpt-asset BurstGPT_2.csv=/absolute/cache/burstgpt/BurstGPT_2.csv \
  --burstgpt-asset BurstGPT_3.csv=/absolute/cache/burstgpt/BurstGPT_3.csv \
  --burstgpt-asset BurstGPT_without_fails_1.csv=/absolute/cache/burstgpt/BurstGPT_without_fails_1.csv \
  --burstgpt-asset BurstGPT_without_fails_2.csv=/absolute/cache/burstgpt/BurstGPT_without_fails_2.csv \
  --burstgpt-asset BurstGPT_without_fails_3.csv=/absolute/cache/burstgpt/BurstGPT_without_fails_3.csv \
  --e0-source-authority AIME-2025=/absolute/sources/e0-authorities/formal-v03-e0-aime_2025-source-authority.json \
  --e0-source-authority Alpaca=/absolute/sources/e0-authorities/formal-v03-e0-alpaca-source-authority.json \
  --e0-source-authority Arena-Hard=/absolute/sources/e0-authorities/formal-v03-e0-arena_hard-source-authority.json \
  --e0-source-authority GSM8K=/absolute/sources/e0-authorities/formal-v03-e0-gsm8k-source-authority.json \
  --e0-source-authority HumanEval=/absolute/sources/e0-authorities/formal-v03-e0-humaneval-source-authority.json \
  --e0-source-authority MBPP=/absolute/sources/e0-authorities/formal-v03-e0-mbpp-source-authority.json \
  --e0-source-authority MT-Bench=/absolute/sources/e0-authorities/formal-v03-e0-mt_bench-source-authority.json \
  --inventory /absolute/sources/gpu-inventory.json \
  --doctor-output /absolute/sources/doctor.json \
  --content-replay-authority-output /absolute/sources/v03-content-replay-authority.json \
  --output /absolute/sources/v03-content-path-inputs.json

lightcone-spec formal-single-operator publish-v03-content-path-spec \
  --inputs /absolute/sources/v03-content-path-inputs.json \
  --output /absolute/sources/v03-content-path-spec.json

lightcone-spec formal-single-operator publish-content-replay-authority \
  --spec /absolute/sources/v03-content-path-spec.json \
  --output /absolute/sources/v03-content-replay-authority.json
```

The replay publisher reads every byte of each unique physical snapshot once.
Derive capacity from that receipt-bound pre-doctor spec, then publish the fresh
canonical doctor report directly. A non-`PASS` report is still retained and
exits 42; it cannot be reused as `BOUND` content.

```bash
lightcone-spec formal-single-operator publish-stage-capacity \
  --content-path-spec /absolute/sources/v03-content-path-spec.json \
  --run-root /absolute/run/formal-v03-study \
  --output /absolute/sources/v03-stage-capacity.json

lightcone-spec doctor \
  --project-root /absolute/clean/lightcone-checkout \
  --sglang-root /absolute/runtime/patched-sglang \
  --trusted-single-operator-capacity /absolute/sources/v03-stage-capacity.json \
  --output /absolute/sources/doctor.json
```

Only after doctor `PASS`, bind content and derive every downstream workload,
trainable plan, and method source. Trusted TTS and E1 authority commands
consume the same BOUND bundle directly; release-signed workload/receipt
arguments do not belong in this lane.

```bash
lightcone-spec formal-single-operator publish-trusted-content \
  --spec /absolute/sources/v03-content-path-spec.json \
  --output /absolute/sources/trusted-content.json

lightcone-spec formal-single-operator publish-preflight-workload \
  --content-source /absolute/sources/trusted-content.json \
  --output /absolute/sources/preflight-workload.json

lightcone-spec formal-single-operator publish-tts-cal-trainable-plan \
  --trusted-content-bundle /absolute/sources/trusted-content.json \
  --output /absolute/sources/tts-trainable-plan.json

lightcone-spec formal-single-operator publish-e1-anchor-trainable-plan \
  --trusted-content-bundle /absolute/sources/trusted-content.json \
  --output /absolute/sources/e1-trainable-plan.json

lightcone-spec publish-tts-calibration-tuning-window \
  --trusted-content-bundle /absolute/sources/trusted-content.json \
  --output /absolute/sources/tts-tuning-window.json

lightcone-spec publish-tts-drafter-native-loss-source \
  --output /absolute/sources/tts-native-loss.json

lightcone-spec publish-tts-calibration-source-authority \
  --paper-pdf /absolute/sources/tts-paper.pdf \
  --paper-source /absolute/sources/tts-v2-source.tar.gz \
  --trusted-content-bundle /absolute/sources/trusted-content.json \
  --tuning-window /absolute/sources/tts-tuning-window.json \
  --trainable-plan-authority /absolute/sources/tts-trainable-plan.json \
  --drafter-native-loss /absolute/sources/tts-native-loss.json \
  --output /absolute/sources/tts-calibration-authority.json

lightcone-spec publish-chronobelief-source-authority \
  --paper-pdf /absolute/sources/chronobelief-paper.pdf \
  --tex-source /absolute/sources/chronobelief-paper.tex \
  --output /absolute/sources/chronobelief-authority.json

lightcone-spec publish-e1-recipe-anchor-authority \
  --trusted-content-bundle /absolute/sources/trusted-content.json \
  --trainable-plan-authority /absolute/sources/e1-trainable-plan.json \
  --output /absolute/sources/e1-recipe-anchor-authority.json

lightcone-spec formal-single-operator publish-onlinespec-source-authority \
  --checkout /absolute/sources/onlinespec-checkout \
  --audit /absolute/sources/onlinespec-source-audit.json \
  --output /absolute/sources/onlinespec-source-authority.json

lightcone-spec publish-formal-runtime-authority-manifest \
  --repository-root /absolute/clean/lightcone-checkout \
  --output /absolute/sources/runtime-authority.json

lightcone-spec formal-single-operator build-trusted-protocol-lock \
  --protocol-id lightcone-v03-study \
  --trusted-content-bundle /absolute/sources/trusted-content.json \
  --runtime-authority-manifest /absolute/sources/runtime-authority.json \
  --tts-calibration-authority /absolute/sources/tts-calibration-authority.json \
  --chronobelief-authority /absolute/sources/chronobelief-authority.json \
  --e1-recipe-anchor-authority /absolute/sources/e1-recipe-anchor-authority.json \
  --output /absolute/sources/protocol-lock.json
```

All source and output paths must be absolute and normalized. Canonical outputs
use atomic no-replace publication and are deep-reopened before the next step.
Model/data payloads, provider state, credentials, and all run artifacts stay
outside the checkout.

### Configure and cross the first GPU boundary

Create new private run and prerequisite-catalog directories outside the Git
checkout, then publish the two path-only configs:

```bash
lightcone-spec formal-single-operator write-dag-driver-config \
  --repository-root /absolute/clean/lightcone-checkout \
  --run-root /absolute/run/formal-v03-study \
  --protocol-lock /absolute/sources/protocol-lock.json \
  --content-source /absolute/sources/trusted-content.json \
  --runtime-authority-manifest /absolute/sources/runtime-authority.json \
  --inventory /absolute/sources/gpu-inventory.json \
  --doctor-report /absolute/sources/doctor.json \
  --preflight-workload-authority /absolute/sources/preflight-workload.json \
  --profiler-tool /absolute/tools/nsys \
  --profiler-tool /absolute/tools/ncu \
  --prerequisite-catalog /absolute/run/formal-v03-prerequisites \
  --output /absolute/run/formal-v03-study/driver-config.json

lightcone-spec formal-single-operator write-bootstrap-config \
  --driver-config /absolute/run/formal-v03-study/driver-config.json \
  --onlinespec-source-authority /absolute/sources/onlinespec-source-authority.json \
  --output /absolute/run/formal-v03-study/bootstrap-config.json
```

`status` has `readiness_scope=code_capability_only`. A 21/21 result confirms
that each node has callable materializer/producer/mapper/executor/finalizer
code; it does not establish that this run's files, models, tools, GPUs, or
upstream receipts are ready:

```bash
lightcone-spec formal-single-operator status
```

The capacity and doctor steps above freshly join live GPU
UUID/model/compute-capability rows to the exact path-spec inventory. This
empirical mode admits one 16 GiB wave plus a 15 GiB safety margin, disables
automatic physical and E6/E0 auxiliary retries, and never authorizes a formal
`MEASURED` claim.

A cold start has an intentionally inspectable boundary. The first cycle
materializes preflight. The second plans the exact one compile, one exactness,
and eight interference cells and commits all ten attempts plus their two-GPU
physical group as `PENDING`, with no PID/PGID. The third cycle is the first one
that may atomically mark the group `RUNNING` and spawn its `setsid` child:

```bash
lightcone-spec formal-single-operator bootstrap-once \
  --config /absolute/run/formal-v03-study/bootstrap-config.json
lightcone-spec formal-single-operator bootstrap-once \
  --config /absolute/run/formal-v03-study/bootstrap-config.json
```

Inspect the exported progress and host boundary before deliberately invoking a
third cycle or `bootstrap-run`.

### Automated 21-node progression

Once the GPU host, rolling archive, and finalizer inputs are ready, the
non-LLM supervisor resumes the same durable state:

```bash
lightcone-spec formal-single-operator bootstrap-run \
  --config /absolute/run/formal-v03-study/bootstrap-config.json
```

One `/absolute/run/formal-v03-study/operator.sqlite3` WAL is the lifecycle
authority. One `formal-dag-driver.lock` prevents a competing scheduler. The
watchdog reconciles PID/PGID, heartbeat, GPU state, log growth, timeout/OOM,
terminal publication, and capacity at its fixed 30-second cadence; progress
tables are atomically exported under the run root. Exit `42` means a retained
real block, `43` means the whole DAG reached controller completion, and `0`
means an intermediate cycle changed state.

The 21 nodes are the exact sequential expansion of
`preflight -> E3a -> TTS-Cal -> E1 -> E2(r0..r3) ->
E4(screen,local,profiler) -> E3b(pilot,final) -> E1a ->
E5(pilot,final) -> E6(pilot,final) -> E0(tuning,pilot,final)`. Only the current
node is materialized. Prerequisite and auxiliary catalogs are append-only and
source-owned; a future result never unlocks an earlier node.

Preflight retains exactly ten stage cells while the exactness execution also
runs the six source-defined native qualification suites. Its eight fresh
interference observations are reduced with paired BCa 95% intervals. Only a
passing <=1% goodput and native-p99-ITL gate may switch the scheduler to two
independent single-GPU headline workers; every other outcome remains isolated.

E6 publishes exactly two TP2 launch descriptors from the built-in `mtp.*`
component in the frozen Qwen target checkpoints. It neither downloads nor
accepts an external drafter. E0 first publishes 12 model/backend pre-probe
interfaces, then runs 12 launch groups over nine task-native probes to produce
the exact 108 compatibility terminals. EAGLE3 task authority is post-probe
only. The resulting VALID/N/A bundle, not a caller-entered `V`, controls E0
tuning and serving materialization.

For E5, the pilot reducer seals each selected backend/topology p99 family
before final unblinding. The existing cells for all five paired methods in that
family receive the same exact 11,000-offer extension pool and must reach at
least 10,000 completions; no diagnostic or headline row is added.

### Rolling archive and restoration

Run the local non-LLM companion alongside the DAG when the remote spool is
small. Its endpoint file contains routing and pinned SSH/rsync tool paths, not
credentials. It polls every 30 seconds, archives only sealed safe boundaries,
verifies every SHA-256, performs a full local rehydrate, and only then asks the
remote operator to evict exact inode-bound v03 files. It never recursively
deletes directories, model caches, or old runs:

```bash
/absolute/runtime/python -m lightcone_spec.orchestration.formal_rolling_archive_companion \
  --endpoint /absolute/archive-host/control/archive-endpoint.json \
  --remote-run-root /absolute/gpu-host/formal-v03-study \
  --local-results-root /absolute/archive-host/formal-v03-study/results \
  --state-root /absolute/archive-host/formal-v03-study/archive-state \
  --lock /absolute/archive-host/formal-v03-study/archive.lock \
  run
```

If a later reducer needs an evicted member, the companion streams only that
member back to its original path with a remote free-space gate, no-replace
publication, and per-file restart records. `restore-all --order reverse`
restores every retained archive in reverse node order before a manual audit;
the remote scientific closure also consumes its exact rehydration catalog.

### Cross-host archive, shutdown, and billing

Finalization has three irreversible parts. Config files for this boundary must
be created with the typed `publish_remote_closure_config`,
`publish_cross_host_ssh_endpoint`, and `publish_cross_host_finalizer_config`
source APIs; do not hand-author their digests or place a provider token in
them.

Part A runs on the GPU host only after all 21 nodes are `REDUCED`. It stops
dispatch, restores any catalogued rolling members, proves zero running attempts
and writers, exports progress, and seals one digest-addressed whole-run payload.
After its receipt is published, the remote SQLite is permanently read-only:

```bash
/absolute/runtime/python \
  /absolute/checkout/scripts/formal_experiment_production_finalizer.py \
  remote-seal \
  --config /absolute/gpu-host/formal-v03-study/remote-closure-config.json
```

Parts B and C run once from the archive host. `run` fetches the closure and
payload, checks the local SHA manifest, fully rehydrates the archive, obtains a
fresh remote zero-writer probe, and publishes the pre-power composite. Only
then does it journal at most one AutoDL `power_off` mutation, require
`code=="Success"`, verify the same instance as `shutdown` through both status
and list responses, close the provider boot interval, and publish compute,
reserved, allocated-billed, whole-instance-billed, archive, idle, and wall-time
accounting:

```bash
/absolute/runtime/python \
  /absolute/checkout/scripts/formal_experiment_production_finalizer.py \
  run \
  --config /absolute/archive-host/formal-v03-study/local-finalizer-config.json
```

The provider token is read only from the local process environment. A crash
after the power intent is retained as indeterminate and cannot issue a second
mutation; restart reopens the journals and continues status/list confirmation.
Successful cross-host completion still has
`formal_measured=false` unless the separate release-root attestation exists.

Before pilots, `gpu-hours-pre` reports fixed cell counts,
`duration_unmeasured`, and the minimum pilot set. After pilots,
`gpu-hours-post` consumes actual single-operator run manifests and lifecycle
timings to separate actual pilot cost, same-stratum projection, and one-shot
diagnostics. It never promotes a registry prefix into whole-study completion.
