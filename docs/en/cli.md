# CLI

[中文](../zh-CN/cli.md) · [Home](../../README.md)

## Commands

`lightcone-spec --help` is the argument authority. Version 0.2.0 exposes:

| Command | Purpose |
|---|---|
| `doctor` | Read-only host, Python, CUDA, and source-tree report |
| `validate-config` | Parse one strict schema-v2 run config |
| `build-speed-study` | Materialize the immutable source protocol |
| `lock-models` | Resolve model IDs to immutable revisions |
| `prepare-models` | Download or offline-verify locked snapshots |
| `list-tuning-candidates` | Materialize the registered Full/LoRA search grid |
| `render-static-load-runtime` | Render one allocation-free Static load endpoint |
| `render-tuning-runtime` | Render exclusive TTS/L0 slices for one candidate |
| `run-controlled-slice` | Measure one load-screen or tuning slice |
| `collect-static-load-screen` | Validate the Static load grid and select the load |
| `advance-tuning-stage` | Validate a tuning stage and write its survivor set |
| `select-speed-config` | Apply the tuning-only maximin rule |
| `render-runtime` | Emit matched sequential confirmation configs and argv |
| `build-confirmation-queue` | Register the 24 clean-server confirmation jobs |
| `run-confirmation` | Execute one method/block confirmation slice |
| `collect-speed-study` | Derive the formal table from completed receipts |
| `render-replication-runtime` | Render natural-task or profiler-only slices |
| `run-natural-slice` | Run one locked natural-EOS side-table slice |
| `build-profiler-plan` | Build an isolated Nsight/device-monitor plan |
| `attest-speed-study` | Bind GPU, runtime, model, selection, and evidence IDs |
| `analyze-speed-study` | Evaluate the registered paired speed gate |

The important but gate-isolated OnlineSPEC comparison has a parallel command
family:

| Command | Purpose |
|---|---|
| `build-onlinespec-study` | Materialize the provenance-bound comparison protocol |
| `verify-onlinespec-source` | Verify an external clean checkout against the registered source audit |
| `list-onlinespec-candidates` | Write the OGD, optimistic, and Hedge tuning grid |
| `render-onlinespec-tuning-runtime` | Render paired Static/candidate tuning endpoints |
| `run-onlinespec-tuning-slice` | Measure one learner tuning slice |
| `advance-onlinespec-tuning-stage` | Halve candidates independently per learner |
| `select-onlinespec-config` | Select one safe terminal candidate per learner |
| `render-onlinespec-runtime` | Render Static plus three exclusive comparison endpoints |
| `build-onlinespec-queue` | Register randomized clean-server comparison jobs |
| `run-onlinespec-confirmation` | Execute one method/block comparison slice |
| `collect-onlinespec-study` | Derive the paired comparison table |
| `attest-onlinespec-study` | Bind comparison evidence to GPU and source identities |
| `analyze-onlinespec-study` | Produce diagnostic learner-versus-Static intervals |

There is no generic method override or replay command. The formal workflow is
intentionally narrow, and source manifests never contain a measured result.

## Identity flow

The minimum dependency chain is:

```text
manifest + model lock + sampling profile + registered grid
                            |
                            v
              load screen + staged tuning
                            |
                            v
                   selection artifact
                            |
                            v
       sequential launch plan + completed evidence
                            |
                            v
          derived table + attestation + speed gate
```

Each artifact has a sidecar or embedded digest. A mismatch is an actionable
error; the CLI never creates an empty successful run or substitutes a default
configuration. `select-speed-config` requires the terminal tuning artifact,
the complete Static load-screen artifact, source manifest, controlled sampling
profile, and model lock. The selection binds those identities, the full tuning
grid, and the patched SGLang tree.

## OnlineSPEC workflow

OnlineSPEC reuses model locking, host preflight, controlled sampling, and the
selected Static load, but it never reuses core tuning or confirmation rows.
Before running the comparison, `verify-onlinespec-source` checks the external
checkout's exact commit and tree, clean status, all registered key-file hashes,
and license-file inventory. Its receipt is content-bound and belongs in the
ignored artifact root; the upstream source remains outside this repository.
Start from the tracked
`manifests/speed-study/onlinespec_baseline_v2.json`, render one candidate and
its single paired Static reference per stage, and advance all registered stages
before selection. Successive halving runs independently inside each learner.

The terminal selection binds the SHA-256 of the complete terminal-stage
artifact, not merely a reserialized winner row. It also requires
`--core-selection`, inherits that artifact's selected Static concurrency, and
binds its SHA-256; there is no independent OnlineSPEC concurrency override.
Because confirmation contains 32 unique prompts, a formal selection cannot
claim concurrency 48. `render-onlinespec-runtime` then emits four descriptions
sharing one port and device; execute Static, OGD, optimistic, and Hedge
sequentially in the manifest's randomized order.

```bash
lightcone-spec select-onlinespec-config \
  --measurements artifacts/onlinespec/terminal-tuning.json \
  --manifest manifests/speed-study/onlinespec_baseline_v2.json \
  --model-lock artifacts/locks/models.json \
  --sampling-profile manifests/speed-study/sampling_profile_v2.json \
  --core-selection artifacts/selection.json \
  --output artifacts/onlinespec/selection.json
```

`analyze-onlinespec-study` requires the same content-bound attestation discipline
as the core analysis. Without it, the status is `UNMEASURED` and exit code is
42. Attested evidence with any safety failure is `BLOCKED` and also exits 42.
A safe measured output remains diagnostic and sets
`core_speed_gate_affected=false`. Full equations, source-audit boundaries, and
memory accounting are documented in [OnlineSPEC baseline](onlinespec-baseline.md).

## Load screen and tuning

Both phases use the controlled sampling profile. Static load screening must
produce one `run-controlled-slice --phase static-load` measurement for every
registered concurrency. `collect-static-load-screen` rejects incomplete,
duplicate, OOM, retracted, or identity-mismatched evidence.

For each load point, `render-static-load-runtime --concurrency C` emits exactly
one native Static endpoint. It accepts neither an adaptation group nor an HBM
adaptation reserve, and its launch vector contains no adaptation flag. Stop the
server before rendering and starting the next load point. Every renderer also
requires `--sglang-checkout /path/to/patched-sglang`; the checkout must be clean
and have the exact patchset tree recorded by this release.

Tuning starts from `list-tuning-candidates`. Render exactly one Static endpoint
at the selected concurrency and reuse its immutable launch description for the
single Static baseline in each stage. For each active candidate and stage,
`render-tuning-runtime` emits only the TTS and L0 slices on one port; it cannot
accidentally duplicate Static evidence. Launch only the slice being measured, run
`run-controlled-slice --phase tune`, terminate the server, and continue.
Candidate JSON includes the complete optimizer identity: Adam/AdamW betas and
decay, SGDm/NAG momentum, Lion betas, or Muon momentum, Newton--Schulz steps,
and auxiliary AdamW fields. Do not remove fields or hand-normalize two
optimizer candidates into one identity.
`advance-tuning-stage` enforces the registered prompt count and context limit,
paired Static/TTS/L0 coverage, safety counters, and survivor identity. A later
stage must name the prior survivor artifact; confirmation data is never an
input.

## Confirmation and resume

`render-runtime` emits three method descriptions that deliberately share one
port and require exclusive ownership of the device. `build-confirmation-queue`
turns the manifest's seeded method order into 24 jobs. For each job, launch its
`launch_argv`, wait for health, run its `run_argv`, then stop that server before
starting the next job. Running all three endpoints simultaneously is invalid
because it changes HBM, KV capacity, batching, and contention.

The launch wrapper imports SGLang only from the verified checkout. The server's
diagnostics must report the SHA-256 of the exact adaptation configuration; a
method, optimizer, learning-rate, rank, stride, or cohort mismatch invalidates
the slice before evidence is committed.

`run-confirmation` executes exactly one method/block slice. It resets the
engine/cohort once, performs an unmeasured warmup by default, then submits all
32 distinct prompts once in one ordered native batch request. SGLang's locked
`max_running_requests` performs admission while the cohort remains continuous.
It records the union of active decode intervals plus request-level absolute
streaming arrivals. The model's 40,960-token limit includes the tokenized
prompt; each prompt is capped at the registered 40,928 safe limit so two
block-16 KV reservations remain available.

Each completed slice ends with a SHA-256-bound receipt. Re-running the same job
skips it only after validating its manifest, config, method, block, prompt,
batch window, load, and every bound shard. Interrupted shards without a receipt never enter
`speed_study.parquet`; a changed or duplicate terminal fails closed. Use
`--no-warmup` only for diagnostics, not the formal protocol.

## Natural replication and profiling

Natural side tables use `render-replication-runtime --phase natural` with the
separate EOS-enabled sampling profile, followed by `run-natural-slice` for each
method and locked dataset revision. They report at-risk requests but cannot
affect selection or the formal gate.

Detailed profiling uses `--phase profile` and `build-profiler-plan`. Its plan
sets `headline_evidence_forbidden=true`; synchronized profiler output must never
be merged into headline timing evidence.

## Attestation and exit status

`analyze-speed-study` without an attestation can compute diagnostics but writes
status `UNMEASURED` and exits with code 42. With a valid content-bound GPU
attestation it exits zero only when both adapted methods pass; a valid measured
failure is `BLOCKED` and also exits 42. Input, identity, runtime, or evidence
errors are ordinary nonzero failures and must not be reported as scientific
results.

Artifacts, model roots, profiler traces, and generated selections belong under
an ignored output root. Never place secrets in CLI arguments; model access uses
the temporary `HF_TOKEN` environment variable.
