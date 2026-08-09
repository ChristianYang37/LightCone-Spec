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
engine/cohort, performs an unmeasured warmup by default, and records absolute
streaming arrival times. The model's 40,960-token limit includes the tokenized
prompt; generation is capped separately for every prompt.

Each completed slice ends with a SHA-256-bound receipt. Re-running the same job
skips it only after validating its manifest, config, method, block, prompt,
load, and every bound shard. Interrupted shards without a receipt never enter
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
