# LightCone-Spec

[简体中文](README_zh-CN.md) · [Documentation](docs/en/architecture.md) · [License](LICENSE)

LightCone-Spec is an evidence-first framework for testing online drafter
adaptation in speculative decoding. It separates runtime engineering from
empirical claims: CPU contracts can establish identity, durability, and
fail-closed behavior, but only registered, attested GPU evidence can establish
speed or capacity.

> Alpha software. Every formal industrial GPU outcome is `UNMEASURED` until a
> new clean source/patch/protocol/registry identity passes the registered GPU
> gates. A release-level `MEASURED` claim is authorized only by fresh
> root-signed controls that bind the exact external inputs, inventory,
> hardware, and native terminal evidence; the trusted single-operator path may
> collect unsigned empirical evidence but cannot cross that claim boundary.
> Historical v2 artifacts are regression/debugging evidence only; they are not
> evidence for the schema-v3 protocol or a new performance claim.

<!-- RESULT_TRUTH_GATE_BEGIN -->
## Result truth gate

These fields deliberately separate evidence state, executable-release state,
host operations, and source identity. Powering a GPU host off does not complete
an experiment, and a historical measurement does not become a formal result.

| Truth field | Value | Scope |
|---|---|---|
| `formal_industrial_gpu_evidence` | `UNMEASURED` | Current schema-v3 formal evidence; no industrial performance claim is released. |
| `formal_industrial_execution` | `BLOCKED_PENDING_QUALIFICATION` | The release public trust root is configured, but fresh root-authorized source, model, hardware, control, and mandatory-preflight receipts do not yet exist; execution fails closed before mutation. |
| `historical_snapshot_evidence` | `PRELIMINARY_NON_FORMAL` | The numerical snapshot below is historical engineering evidence only. |
| `historical_snapshot_host_at_archive` | `POWERED_OFF_NOT_RELEASED` | Operational state at archive time; the instance was shut down but not released or deleted. |
| `current_sglang_upstream_commit` | `3312645a307453893a00778592f105581e3d1c3d` | Full Git commit pinned by the current patch manifest. |
| `current_patched_sglang_tree` | `c6571336b70cd5f0e0f609d731a65fa98fd7e0b2` | Full Git tree expected after applying the current patch series. |
| `current_patch_payload_sha256` | `38b5ec81b9d75950558f8c72c1297bab47badf89d855b3e13dc1ad1c639f7d95` | SHA-256 of the latest semantic mail-patch bytes. |
| `current_patch_manifest_sha256` | `cc8355703fe83c8a73ecdbf9cd656140e257e69570fbfd8d8bc08f657e72fd71` | SHA-256 of the current canonical patch-manifest JSON. |
| `historical_main_code_prefix` | `0db2ff4` | Short code prefix bound only to the preliminary snapshot. |
| `historical_patched_tree_prefix` | `e795ecc` | Short tree prefix bound only to the preliminary snapshot. |

The current commit, tree, patch payload, and manifest digest are distinct
identity domains. The two seven-character historical prefixes are not current
release identities and cannot satisfy any formal identity gate.
<!-- RESULT_TRUTH_GATE_END -->

## Scope

The formal comparison has five scientific roles. Recipe authority and
publication policy are orthogonal identities; a shared runtime implementation
does not merge configurations, live candidates, optimizer state, or evidence.

| Scientific role | Runtime method | Recipe authority | Publication |
|---|---|---|---|
| Target-only | `target_only` | structural zero adaptation | native target decoding |
| Static | `static` | structural zero adaptation | native speculative decoding |
| TTS | `tts` | frozen, primary-source-bound TTS recipe | fixed TTS barrier |
| L0-naive | `l0` | the same frozen TTS recipe authority | first-ready safe boundary |
| LightCone | `l0` | one tuning-sealed E2 winner | first-ready safe boundary |

An `l0` run using a registered E1/E2 search recipe is an **LC-candidate**, not
LightCone. Only an exact sealed E2 final-recipe receipt may materialize or name
LightCone. Target-only and Static structurally allocate no adaptation,
optimizer, gradient, candidate, or adaptation-telemetry state.

The historical `TTS-paper-reconstruction` baseline remains diagnostic and
`BLOCKED`: the primary-source audit as of 2026-08-15 found no official author
code/config, and the paper does not fix all numerical fields. The formal path
therefore uses a separate preregistered TTS-Cal authority. It fixes Adam, one
optimization step, `(beta1=.9, beta2=.999, epsilon=1e-8)`, zero weight decay,
no clipping, full-drafter/latest-round-only updates, a digest-bound
drafter-native position/proximal loss recipe, per-request reset, side stream,
learning rates `1e-7, 3e-7, ..., 1e-3`, and strides
`{1,5,10,15,20,30,40,50}`. Four excluded pilots reduce that grid with the
registered safety-first rule; only the signed winner may freeze TTS and
L0-naive. Before that seal both remain `BLOCKED`. Neither may inherit an E1/E2
winner, schema default, or historical AdamW configuration. TTS publication is
independently fixed to the paper's synchronization barrier.
The [provenance authority](manifests/provenance/tts_recipe_authority_v1.json)
binds arXiv `2605.09329v2`, PDF SHA-256
`7688b05bab7696f4a47a5987f2fcad13d46f1d84cec9f90caf661fb397f3ee20`,
and source SHA-256
`22c549c0297fc0a2a71af002c3721f71ddfd06d86bc46b2f41592bd6748afe59`.

The target industrial registry covers DFlash and DSpark first, followed by
production load, multi-GPU topology, native NEXTN preflight, and breadth
templates. EAGLE and EAGLE3 remain target backend contracts with narrow
compatibility guards; registration is not executable release support.
OnlineSPEC remains an important comparison with separate tuning, evidence,
attestation, and analysis; it cannot select or alter the core gate.

The current pinned SGLang patch contains source implementations for the DFlash,
DSpark, NEXTN, and official-selector-compatible EAGLE3 paths, including TP1,
two-rank TP2 publication, and sticky two-replica DP2 isolation where the formal
matrix registers them. DFlash executes the registered constant,
inverse-square-root, and finite-horizon cosine schedules plus non-negative
logical publication delay; DSpark binds its actual predecessor, W1/W2 and
confidence head; NEXTN binds its MTP teacher/interface and TP2 shard authority.
The patch also exposes exact begin/reset/finalize terminal evidence and the
host validates its content. These are implementation claims, not GPU results:
the repository ships no trusted hardware signer and no fresh dynamic GPU
qualification receipt. Static/TTS/L0-naive/LightCone and every adaptive
backend/topology therefore remain non-claimable and fail release preflight
before mutation. Generic EAGLE is unsupported, and EAGLE3 combinations without
an official signed compatibility decision remain N/A or `BLOCKED`. A requested
quota-shadow row is identity- and quota-recorded; unsupported acquisition is
rejected instead of fabricating a teacher row.

## Runtime contract

- Schema v3 rejects unknown or retired fields before model allocation.
- A common backend envelope binds adapter-free logits, the exact proposal
  distribution used for sampling, semantic masks, target teacher rows, the
  sampled predecessor, cohort identity, source version, and a backend-native
  payload.
- The target backend contract requires a validator to reconstruct the
  differentiable proposal without applying an adapter twice. The pinned patch
  implements DFlash, DSpark, NEXTN, and compatibility-authorized EAGLE3
  surfaces, but each exact backend/topology pair stays blocked until its named
  dynamic GPU suite and durable external-control proof pass. Generic EAGLE is
  not implemented.
- Adaptation is only `full` or `lora`. Layer scopes are `last1`, `last3`,
  `last5`, and `all`; LoRA ranks are 1, 2, 4, 8, 16, 32, and 64 with
  `alpha/r=1`.
- DSpark additionally registers three hybrid scopes: the last 1, 3, or 5
  backbone layers plus native W1, W2, and acceptance/confidence parameters.
  Its E1a grid has exactly 56 adaptive configurations: 32 layer-only and 24
  hybrid configurations.
- Historical drafter KV is frozen and versioned. Publishing a candidate affects
  future KV only; rebuilding or differentiating old KV is a different method.
- Candidate equality is a controlled mechanism-replay assertion only. It
  requires identical source-state and proposal-evidence digests; live TTS and
  L0-naive runs may diverge after their publication histories diverge.

The rejection sampler still uses normalized $(p-q)_+$ after rejecting a
proposal token. That positive-part rejection distribution is sampling math; it
is not an adaptation mode or configuration alias.

## Industrial execution

The declarative dependency order is:

```text
preflight -> E3a -> TTS-Cal -> E1 -> E2 -> E4 -> E3b -> E1a -> E5 -> E6 -> E0
```

Every cell binds scientific axes, seed, logical rank slots, ports,
cache/evidence roots, resource isolation, and a truthful initial status. A
stage receipt seals the runtime and split digests, exact dependency receipts,
activated/disposition artifacts, and locked outputs before the next stage can
be dispatched. One deterministic same-host GPU-pool scheduler maps those
logical declarations onto content-bound physical UUIDs for 1, 2, 4, 8, 16, or
more GPUs. It serializes exclusive headline/profile/download/compile work and
permits independent work to overlap only under an exact registered
interference-envelope rule.

Isolation alone is not terminal authority. COMPILE and DOWNLOAD use typed,
first-party assignments, immutable terminals, and atomic no-replace result
pointers; missing dynamic control or incomplete raw coverage blocks activation.

TTS-Cal has 288 disjoint tuning-only rows and seals one numeric TTS recipe.
E1 then materializes exactly 68 rows: four fixed roles plus two optimizer
anchors over 32 LightCone geometries. E2 registers 105 recipes per surviving
geometry and materializes four successive-halving rounds with a floor of 21
families and four fixed anchors per round. E4 is 48 screen + 96 local-factorial
+ 3 profiler rows; E3b is `480B`; E1a is 116 rows; E5 is `450B+264`; E6 is
`2+60B` over exactly two target models. E0 first signs 108 compatibility
decisions and materializes `16VB` rows only for the `V` valid combinations.
No blocked sentinel matrix is expanded. Family-specific four-pilot evidence fixes either a
12--20 final prefix or `UNDERPOWERED` before confirmation. Per-cell budgets,
physical assignments, terminal evidence, and observed-versus-registered
GPU-time receipts are all content-bound. The pinned patch now ships a
first-party all-reset producer for source-owned capability, initial-state, and
reset receipts. It covers drain, KV/prefix, RNG/counters, scheduler/telemetry,
adaptation state, allocator/HBM, and a completion event. GPU reuse remains
disabled until the exact runtime identity passes the registered device reset
qualification. On the supported single-tokenizer HTTP/1.1 uvicorn
paths, the HTTP process counts real protocol `connection_made`/
`connection_lost` events and injects cumulative process, generation, created,
closed, and current totals into reset state; scheduler request counters and
client fields cannot substitute. Granian HTTP/2 and multiple-tokenizer HTTP-
process paths fail closed before producing this capability. The receipts are
not yet durably integrated with terminal and whole-inventory accounting, and
native warm-up/trace/close receipts are not part of this reset-state slice.
Live GPU reuse therefore remains blocked and falls back to one clean process
and HTTP pool per logical trace.

The target schema and CPU coordinator vocabulary define one-node TP2 and
sticky-replica DP2 identities, inference-sharded TP state, replica-local DP
cohorts, and all-rank prepare/decision/application/receipt transitions. They do
not make those shapes executable. The current `RunConfig` and pinned SGLang
patch reject every TP2/DP2 run before model loading and cannot issue a
`patched_two_gpu_v1` capability receipt.

The real CPU `gloo` harness tests collective state-machine behavior only. It
does not attest NCCL, CUDA streams, graph boundaries, device copies, throughput,
or two-GPU correctness.

## Deployment scale and readiness

The resource-pool control plane has two independent scale dimensions. A single
host may expose any number of GPUs. `GpuFleetScheduler` is only the host-affinity
and serial-partitioning layer; after it selects a host, the sole
`GpuPoolScheduler` performs every child placement against that host's topology
and interference envelope. A `GpuFleetInventory` may combine multiple such
hosts and distribute independent cells or complete confirmation blocks between
them. It does not turn the fleet into one model-parallel rank group. Every
TP/DP gang remains inside one host, and a placement that would require a
cross-host collective is rejected with `cross_host_collectives_unvalidated`.

Each host retains its own content-bound inventory digest, physical GPU UUIDs,
interference envelope, ports, cache namespace, evidence namespace, and
host-local materialization manifest. Remote waves use an SSH agent, a fixed
known-hosts file, and canonical JSON on standard input; routing data and
credentials are coordinator-local and are never copied into artifacts or logs.
A host failure preserves completed receipts and leaves in-flight work on that
host. Once a request may have reached the worker, loss of an authoritative
response is `REMOTE_OUTCOME_UNKNOWN`: it cannot be retried until an independent
content-addressed receipt/evidence fetch over the exact original destination,
port, and known-host-key authority reconciles the attempt. Endpoint values,
host-key bytes, and credentials are not persisted in requests, receipts,
evidence, or logs. Only a terminal-negative result may create a new
receipt-bound attempt, and fleet transport concurrency is explicitly bounded.

Readiness labels are deliberately not performance claims:

- `CPU_READY` means non-GPU schemas, scheduling, identity, failure, and receipt
  contracts passed CPU/mock gates.
- `GPU_SMOKE_READY` means the exact device path and external inputs are assembled
  for a bounded smoke; it does not mean that smoke passed or that a formal cell
  is authorized.
- `MEASURED` requires completed registered GPU evidence and the release-owned
  attestation chain. Neither of the first two labels may be reported as
  `MEASURED`.

## Memory, traces, and evidence

HBM admission is governed by the least-feasible rank. The ledger separately
charges model/KV state, FP32 masters, gradients, actual optimizer moments,
candidate/staging state, activations, graph buffers, and telemetry. Pressure
handling preserves immutable and active correctness state first, may evict only
native inactive prefixes, then aborts pending adaptation and queues or rejects
new work. It never silently changes Full to LoRA.

Cohort state uses fixed-size slabs, tenant quotas, replica identity, generation
counters, and deterministic reclamation. Optional cold offload is explicit,
timed, and limited to inactive cohorts; it is not an automatic OOM escape.

Open-loop production traces are immutable and method-independent. Closed-loop
runs bind one immutable maximum request pool and per-client order; realized
offer times and contiguous client prefixes are recorded because faster methods
may replenish clients sooner. Exhausting that pool before the fixed arrival
window makes the run nonclaimable. Synthetic Poisson and immediate-burst traces
are labelled synthetic, and BurstGPT naming requires a bound external identity.

Evidence writes to a bounded in-memory queue and durable, process-unique
Parquet WAL segments. Flush/checkpoint state, duplicate identities,
backpressure, and any explicit drops are counted. Final Parquet shards are
published only after coverage checks and filesystem durability, followed by an
exclusive content-bound completion receipt. Interrupted WALs remain
inspectable but cannot enter analysis without that receipt.
The execution-plan identity registers the producer queue/batch, writer queue,
Parquet row-group, checkpoint interval, overflow mode, SQLite durability, and
WAL/checkpoint/directory fsync policy; resume requires the same policy digest.

## Installation and quick start

Framework-only development:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Create a disposable SGLang checkout at the exact pin:

```bash
git clone https://github.com/sgl-project/sglang.git /path/to/sglang
git -C /path/to/sglang checkout --detach \
  3312645a307453893a00778592f105581e3d1c3d
patches/sglang/apply.sh /path/to/sglang
python scripts/verify_sglang_patchset.py \
  --upstream-checkout /path/to/clean-upstream --compile-only
```

The current schema-v3 SGLang series must pass clean-pin, patch-digest,
expected-tree, compile/test, and reverse-removal verification before any GPU
run is evidence-eligible. Documentation does not substitute for that gate.

Build the registry with logical rank slots; physical identities enter only
through a separately locked inventory and dispatch assignment:

```bash
lightcone-spec build-industrial-registry \
  --logical-gpu-slot logical-rank-slot-0 logical-rank-slot-1 \
  --cache-root runtime-cache/industrial \
  --evidence-root artifacts/industrial \
  --output artifacts/industrial/registry.json

lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/inventory.json \
  --interference-envelope artifacts/industrial/interference.json \
  --budget-plan artifacts/industrial/budget-plan.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --budget-load-binding artifacts/industrial/load-cell-000.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --capacity-manifest artifacts/industrial/capacity-source-manifest.json \
  --capacity-verification-receipt artifacts/industrial/capacity-verification.json \
  --activation-plan artifacts/industrial/stage-activation-manifest.json \
  --output artifacts/industrial/dispatch.json
```

For a fleet, collect and calibrate each host separately, then assemble the
content-bound host pairs in matching repeated-argument order:

```bash
lightcone-spec assemble-gpu-fleet-inventory \
  --inventory artifacts/host-a/inventory.json \
  --interference-envelope artifacts/host-a/interference.json \
  --inventory artifacts/host-b/inventory.json \
  --interference-envelope artifacts/host-b/interference.json \
  --output artifacts/fleet/inventory.json
```

The remote coordinator is currently a Python-library API. The only remote
worker CLI entry point is `lightcone-spec execute-dispatch-wave
--host-request-stdin`; it accepts the canonical coordination request on standard
input and is not an interactive operator command.

This command materializes target declarations only. It does not bypass the
executor's native-evidence preflight or make a speculative or multi-rank cell
runnable.

After completing a stage, seal its content-bound outputs and pass the resulting
receipt back to the planner. Use `lightcone-spec COMMAND --help` for exact
arguments; the CLI never executes the registry as an implicit shell script.
Model locks, evidence, traces, credentials, provider state, and selections
belong under ignored external roots and must not be committed.

## Trusted v03 reproduction

The implemented `formal-single-operator` path runs the immutable 21-node DAG
with one SQLite WAL, one `flock`-guarded scheduler, and source-owned physical
producers. It is a trusted-operator empirical path, not a substitute for the
release attestation chain. Without a current root-authorized attestation, its
complete outcome is
`trusted_single_operator_empirical_no_signature` and remains `UNMEASURED`; it
must never be reported as `MEASURED`.

After publishing the source authorities, BOUND trusted-content bundle,
preflight workload, and schema-v5 ProtocolLock, write the path-only DAG and
bootstrap configs. Inspect code capability without allocating a GPU, then run
the non-LLM supervisor:

```bash
lightcone-spec formal-single-operator status
lightcone-spec formal-single-operator bootstrap-run \
  --config /absolute/results/formal-v03-run/bootstrap-config.json
```

`status` reports code capability for all 21 nodes; it is not run-specific GPU
readiness. On a cold start, exactly two `bootstrap-once` cycles reach a queue
with all ten preflight attempts `PENDING`; the third cycle is the first
GPU-launch boundary.
Fresh paired interference evidence alone decides whether later headline work
uses two independent single-GPU workers or isolated scheduling.

E6 derives exactly two TP2 launches from the built-in `mtp.*` component in the
two frozen target checkpoints and forbids an external draft model. E0 first
publishes exactly 12 model/backend pre-probe interfaces, then runs the 108
model/backend/task compatibility probes; task-specific EAGLE3 authority is
published only after successful post-probe evidence. The rolling archive and
three-part cross-host closure keep the GPU host spool bounded while retaining
a fully rehydrated local archive. See the
[trusted workflow CLI](docs/en/cli.md#trusted-single-operator-formal-workflow)
for the complete absolute-path command sequence and shutdown boundary.

## Statistics and claims

Four excluded pilot blocks estimate variance only. The power plan fixes 12--20
final blocks before downstream unblinding and records `UNDERPOWERED` unless
both primary contrasts reach 80% power for a 3% minimum effect. Primary
LightCone--TTS and LightCone--Static hypotheses use Holm family-wise
correction. The preregistered secondary decomposition is L0-naive--TTS and
LightCone--L0-naive. The registered reporting target includes method-by-model,
method-by-context, and method-by-load interactions without assuming their sign.
The current `CrossFamilyInteractionReducerArtifact` is only a content-bound,
structural, non-formal `UNRESOLVED` contract. It does not prove completed GPU
coverage, interval validity, or any formal interaction; those require the later
registered statistical reduction over attested final evidence. Other secondary
breadth families use Benjamini--Hochberg FDR. Long-context/request data use hierarchical
block-then-request bootstrap; production traces use time-block bootstrap. P99
latency is eligible only with its registered minimum completion count.

Every reported system point includes throughput/goodput, TTFT/ITL, completion
and error accounting, target work, update/publication timing, HBM, energy per
output token, and the locked power/clock/thermal envelope. Missing values are
not replaced with zero.

No gate can return `PASS` without a content-bound GPU attestation covering the
registry/manifest, selection, model and data revisions, runtime capability and
patched tree, hardware/power report, trace identity, and exact Parquet inputs.
Local mocks, historical v2 evidence, or positive arithmetic remain
`UNMEASURED`. The release ships only its offline Ed25519 public trust root; it
does not ship a private key or a pre-authorized hardware identity. A fresh,
challenge-bound deployment policy and source-owned control attestations must
bind the observed inventory and new immutable source identity. Caller-authored
doctor/attestation JSON is rejected, and no analyzer can emit a new `MEASURED`
GPU outcome before that qualification chain and the mandatory preflight pass.

For preliminary exactness diagnostics, `run-preliminary-target-reference` can
capture a locked Target-only greedy reference whose per-prompt outputs are
format-tagged hashes of complete ordered token-ID trajectories. Legacy
collectors require every method/block to match that reference; decoded-text
agreement or agreement only among speculative methods is insufficient. The
reference remains `PRELIMINARY_DIAGNOSTIC_ONLY`: it does not make Static/TTS/L0-naive/LightCone
formally executable and cannot substitute for industrial authority.

## Historical preliminary mechanism snapshot

This separately scoped snapshot comes from the older main-line implementation
and is retained only as preliminary engineering evidence. It does not change
the current industrial protocol's formal truth above: schema-v3 Stage B remains
`UNMEASURED` and `BLOCKED`, and these numbers cannot activate a cell, select a
configuration, or pass a release gate.

One RTX PRO 6000 Blackwell (96 GB) ran Qwen3-8B + DFlash-b16 on 16 repetitive,
controlled prompts, at concurrency 8 and a 40,928-token safe context limit.
Each diagnostic path generated 654,042 tokens in one timing block:

| Method | Decode goodput | vs. Static | p99 ITL | Peak HBM |
|---|---:|---:|---:|---:|
| Static | 1,342.0 tok/s | 1.00x | 45.04 ms | 90.36 GiB |
| Matched-recipe fixed-barrier diagnostic | 2,497.9 tok/s | +86.13% | 45.94 ms | 90.52 GiB |
| Matched-recipe first-ready diagnostic | 2,519.5 tok/s | +87.74% | 46.21 ms | 90.53 GiB |

The first-ready diagnostic was 0.86% faster than the fixed-barrier diagnostic
in this snapshot. All three complete ordered
token-ID trajectories matched, while exactness-violation, version-mismatch,
fallback, non-finite-update, OOM, and retraction counters were zero. The bound
historical identity is main code `0db2ff4`, old patched SGLang tree `e795ecc`,
execution-policy SHA `231ca579`, and tuning-window SHA `132019ee`; CUDA Graph
and Radix Cache were disabled by that policy.

The archived selection and runtime configs bind both adaptive rows to one
shared tuned Adam recipe (learning rate 0.001, weight decay 0, LoRA rank 8,
stride 80). They are matched-recipe publication-policy diagnostics, **not** TTS
paper reproduction, and cannot tune the new LightCone recipe or satisfy a
formal gate. The mechanism diagnostics
suggest that online work was already mostly hidden: adaptive main-side overlap
was about 96.7%, and roughly 8.4--8.7 seconds of
aggregate training, optimizer, merge, and publication work exposed only about
0.27 seconds on the decode critical path. The first-ready diagnostic's advantage
instead tracked fewer target calls: 52,083 versus the fixed-barrier diagnostic's
52,879 (12.558 versus 12.369 committed tokens
per call). The adaptation-memory ledger was dominated by candidate scratch
(7.88 GiB) and resident buffers (5.91 GiB), not the approximately 31.3 MiB
optimizer state. The follow-up optimization order is therefore candidate
scratch, staging, training-activation lifetime, dynamic reserve calibration,
and only then optimizer moments. The fixed 16 GiB reserve reduced reported KV
capacity from 461,703 to 359,227 tokens, about 22.2%.

This is an `n=1` mechanism check with no formal BCa confidence interval. The
cyclic prompts are strongly online-learnable, so this is neither a natural-task
result nor a reproduction of the paper's LiveCodeBench, MATH-500, or OnlineSPEC
experiments. The next scientific priority is the locked LiveCodeBench v6 Hard
and MATH-500 Level 5 protocol, not the provisional first-32-item loader. An
earlier CUDA-Graph-on diagnostic matched only 14 of 16 token trajectories and
was about 2.85% slower, so graph replay remains off until exactness is repaired.

The operator was intentionally stopped and the host powered off. It correctly
sealed `failed_resumable` with exit code 143 rather than reporting completion;
the unfinished Target-only reference has no final JSON and must be rerun. The
60-file raw archive remains ignored and is not committed to the public tree.

## Documentation

- [Architecture](docs/en/architecture.md)
- [Mathematical method](docs/en/mathematical-method.md)
- [Installation](docs/en/installation.md)
- [Configuration](docs/en/configuration.md)
- [CLI](docs/en/cli.md)
- [SGLang patch workflow](docs/en/sglang-patches.md)
- [Experiment protocol](docs/en/experiment-protocol.md)
- [OnlineSPEC baseline](docs/en/onlinespec-baseline.md)
- [Troubleshooting](docs/en/troubleshooting.md)

## Limitations

- All formal industrial GPU outcomes are `UNMEASURED` until the new clean code,
  patch, protocol, registry, external-source, and hardware identities produce
  fresh signed GPU receipts.
- TP2/DP2, native ITL, DSpark, NEXTN, and compatible EAGLE3 paths require exact
  mode-specific device qualification. CPU contracts and caller-provided keys
  cannot authorize them. Multi-host execution remains independent host-local
  work only; cross-host collectives, world size greater than two, Kubernetes,
  elastic membership, and automatic failover remain unsupported.
- TTS and L0-naive require the signed TTS-Cal winner. LightCone additionally
  requires the sealed E2 final recipe. No method may inherit a result-derived
  or legacy diagnostic recipe.
- ChronoBelief is registered with exact equations and state semantics, but it
  still requires the same runtime/GPU qualification as every adaptive recipe.
- Historical KV remains frozen by design. Recomputing it requires a new
  algorithm, memory envelope, protocol, and claim.

## Contributing and license

Read [CONTRIBUTING.md](CONTRIBUTING.md),
[CONTRIBUTING_zh-CN.md](CONTRIBUTING_zh-CN.md), and
[SECURITY.md](SECURITY.md). LightCone-Spec is licensed under
[Apache-2.0](LICENSE); external models, datasets, and SGLang retain their own
licenses.
