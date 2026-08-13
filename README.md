# LightCone-Spec

[简体中文](README_zh-CN.md) · [Documentation](docs/en/architecture.md) · [License](LICENSE)

LightCone-Spec is an evidence-first framework for testing online drafter
adaptation in speculative decoding. It separates runtime engineering from
empirical claims: CPU contracts can establish identity, durability, and
fail-closed behavior, but only registered, attested GPU evidence can establish
speed or capacity.

> Alpha software. Every formal industrial GPU outcome is `UNMEASURED`. The industrial
> executor currently runs only Target-only end to end. Static, TTS, L0, and
> every other speculative method are `BLOCKED` before process or network
> mutation because no trusted hardware signer is provisioned for the pinned
> `sglang.schema_v3.content_bound_terminal_speculative_evidence.v1` hook.
> Empirical Stage B is also `BLOCKED` on provider credentials, resolved
> model/data/trace locks, and registered hardware.
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
| `formal_industrial_execution` | `BLOCKED` | Current release authorization; the missing trusted signer and unresolved Stage-B inputs fail closed before mutation. |
| `historical_snapshot_evidence` | `PRELIMINARY_NON_FORMAL` | The numerical snapshot below is historical engineering evidence only. |
| `historical_snapshot_host_at_archive` | `POWERED_OFF_NOT_RELEASED` | Operational state at archive time; the instance was shut down but not released or deleted. |
| `current_sglang_upstream_commit` | `3312645a307453893a00778592f105581e3d1c3d` | Full Git commit pinned by the current patch manifest. |
| `current_patched_sglang_tree` | `c6accb514b9d10ee95e704e69aa11e058adbe77a` | Full Git tree expected after applying the current patch series. |
| `current_patch_payload_sha256` | `d640d5fe0ac55cb542d0b885fac51d64dc6a83a142ac52480cbfc99d9b866b6e` | SHA-256 of the latest semantic mail-patch bytes. |
| `current_patch_manifest_sha256` | `dfa7256fed0335331643a6930e473130f85494ce2f89df8355202474d1e06d69` | SHA-256 of the current canonical patch-manifest JSON. |
| `historical_main_code_prefix` | `0db2ff4` | Short code prefix bound only to the preliminary snapshot. |
| `historical_patched_tree_prefix` | `e795ecc` | Short tree prefix bound only to the preliminary snapshot. |

The current commit, tree, patch payload, and manifest digest are distinct
identity domains. The two seven-character historical prefixes are not current
release identities and cannot satisfy any formal identity gate.
<!-- RESULT_TRUTH_GATE_END -->

## Scope

The core comparison has four methods:

| Method | Speculation | Online candidate | Publication |
|---|---:|---:|---|
| Target-only (`target_only`) | no | none | native target decoding |
| Static (`static`) | yes | none | native speculative decoding |
| TTS (`tts`) | yes | side-stream update | next fixed update boundary |
| L0 (`l0`) | yes | byte-equivalent to TTS | first legal boundary after readiness |

TTS and L0 share the same evidence, trainable plan, loss, optimizer candidate,
and reconstruction path. Their only experimental difference is publication
timing. Target-only and Static allocate no adaptation, optimizer, candidate, or
adaptation telemetry state.

The target industrial registry covers DFlash and DSpark first, followed by
production load, multi-GPU topology, native NEXTN preflight, and breadth
templates. EAGLE and EAGLE3 remain target backend contracts with narrow
compatibility guards; registration is not executable release support.
OnlineSPEC remains an important comparison with separate tuning, evidence,
attestation, and analysis; it cannot select or alter the core gate.

The current pinned SGLang patch contains a lower-level executable adaptive path only for
TP1/DP1 DFlash. It executes the registered constant, inverse-square-root, and
finite-horizon cosine schedules plus non-negative logical publication delay.
It exposes the exact begin/reset/finalize terminal-evidence hook and
the host adapter validates its content, but the repository ships no trusted
hardware signer. Static/TTS/L0 therefore remain non-claimable and fail release
preflight before mutation.
DSpark now has a CPU-only native contract for adapter-free reconstruction,
actual-predecessor W1 features, native-head selection, composite loss, and
verification-budget semantics. It does not connect a CUDA worker or authorize
runtime adaptation. DSpark/EAGLE/EAGLE3/NEXTN adaptation and all TP2/DP2 execution still fail
closed. A requested quota-shadow row is identity- and quota-recorded, but the
current DFlash backend capability blocks acquisition instead of fabricating a
teacher row.

## Runtime contract

- Schema v3 rejects unknown or retired fields before model allocation.
- A common backend envelope binds adapter-free logits, the exact proposal
  distribution used for sampling, semantic masks, target teacher rows, the
  sampled predecessor, cohort identity, source version, and a backend-native
  payload.
- The target backend contract requires a validator to reconstruct the
  differentiable proposal without applying an adapter twice. Only the pinned
  patch's lower-level TP1/DP1 DFlash path implements that adaptive surface;
  DSpark, EAGLE/EAGLE3, and NEXTN remain non-executable contracts.
- Adaptation is only `full` or `lora`. Layer scopes are `last1`, `last3`,
  `last5`, and `all`; LoRA ranks are 1, 2, 4, 8, 16, 32, and 64 with
  `alpha/r=1`.
- DSpark additionally registers three hybrid scopes: the last 1, 3, or 5
  backbone layers plus native W1, W2, and acceptance/confidence parameters.
  Its E1a grid has exactly 56 adaptive configurations: 32 layer-only and 24
  hybrid configurations.
- Historical drafter KV is frozen and versioned. Publishing a candidate affects
  future KV only; rebuilding or differentiating old KV is a different method.

The rejection sampler still uses normalized \((p-q)_+\) after rejecting a
proposal token. That positive-part rejection distribution is sampling math; it
is not an adaptation mode or configuration alias.

## Industrial execution

The declarative dependency order is:

```text
preflight -> E3a -> E1 -> E2 -> E4 -> E3b -> E1a -> E5 -> E6 -> E0
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

Isolation alone is not terminal authority. The current release blocks all
COMPILE and DOWNLOAD cells before budgeting or dispatch because it ships no
release-owned compile prewarm/finalization result-pointer contract and no
first-party download terminal receipt contract.

Reducer-owned E1 activation materializes one 130-cell width/load slice from
the 2,730-cell envelope. E2 materializes only the current quarter-retention
successive-halving round. Family-specific four-pilot evidence fixes either a
12--20 final prefix or `UNDERPOWERED` before confirmation. Per-cell budgets,
physical assignments, terminal evidence, and observed-versus-registered
GPU-time receipts are all content-bound. The pinned patch now ships a
first-party all-reset producer for source-owned capability, initial-state, and
reset receipts. It covers drain, KV/prefix, RNG/counters, scheduler/telemetry,
adaptation state, allocator/HBM, and a completion event. Its GPU reset semantics
remain explicitly `PENDING`. On the supported single-tokenizer HTTP/1.1 uvicorn
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

This command materializes target declarations only. It does not bypass the
executor's native-evidence preflight or make a speculative or multi-rank cell
runnable.

After completing a stage, seal its content-bound outputs and pass the resulting
receipt back to the planner. Use `lightcone-spec COMMAND --help` for exact
arguments; the CLI never executes the registry as an implicit shell script.
Model locks, evidence, traces, credentials, provider state, and selections
belong under ignored external roots and must not be committed.

## Statistics and claims

Four excluded pilot blocks estimate variance only. The power plan fixes 12--20
final blocks before downstream unblinding and records `UNDERPOWERED` if neither
contrast reaches 80% power for a 3% minimum effect. Primary L0--Static and
L0--TTS hypotheses use Holm family-wise correction. Secondary breadth families
use Benjamini--Hochberg FDR. Long-context/request data use hierarchical
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
`UNMEASURED`. This release intentionally has no configured trusted hardware
attester: caller-authored doctor/attestation JSON is rejected, and no analyzer
can emit a new `MEASURED` GPU outcome.

For preliminary exactness diagnostics, `run-preliminary-target-reference` can
capture a locked Target-only greedy reference whose per-prompt outputs are
format-tagged hashes of complete ordered token-ID trajectories. Legacy
collectors require every method/block to match that reference; decoded-text
agreement or agreement only among speculative methods is insufficient. The
reference remains `PRELIMINARY_DIAGNOSTIC_ONLY`: it does not make Static/TTS/L0
formally executable and cannot substitute for industrial authority.

## Historical preliminary mechanism snapshot

This separately scoped snapshot comes from the older main-line implementation
and is retained only as preliminary engineering evidence. It does not change
the current industrial protocol's formal truth above: schema-v3 Stage B remains
`UNMEASURED` and `BLOCKED`, and these numbers cannot activate a cell, select a
configuration, or pass a release gate.

One RTX PRO 6000 Blackwell (96 GB) ran Qwen3-8B + DFlash-b16 on 16 repetitive,
controlled prompts, at concurrency 8 and a 40,928-token safe context limit.
Each method generated 654,042 tokens in one timing block:

| Method | Decode goodput | vs. Static | p99 ITL | Peak HBM |
|---|---:|---:|---:|---:|
| Static | 1,342.0 tok/s | 1.00x | 45.04 ms | 90.36 GiB |
| TTS | 2,497.9 tok/s | +86.13% | 45.94 ms | 90.52 GiB |
| L0 | 2,519.5 tok/s | +87.74% | 46.21 ms | 90.53 GiB |

L0 was 0.86% faster than TTS in this snapshot. All three complete ordered
token-ID trajectories matched, while exactness-violation, version-mismatch,
fallback, non-finite-update, OOM, and retraction counters were zero. The bound
historical identity is main code `0db2ff4`, old patched SGLang tree `e795ecc`,
execution-policy SHA `231ca579`, and tuning-window SHA `132019ee`; CUDA Graph
and Radix Cache were disabled by that policy.

The mechanism diagnostics suggest that online work was already mostly hidden:
TTS/L0 main-side overlap was about 96.7%, and roughly 8.4--8.7 seconds of
aggregate training, optimizer, merge, and publication work exposed only about
0.27 seconds on the decode critical path. L0's advantage instead tracked fewer
target calls: 52,083 versus TTS's 52,879 (12.558 versus 12.369 committed tokens
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

- All formal industrial GPU outcomes are `UNMEASURED`. The native terminal hook is present,
  but Stage B is `BLOCKED` on the absent trusted hardware signer, provider
  credentials, resolved model/data/trace locks, and registered hardware.
- The current end-to-end industrial execution surface is Target-only at
  TP1/DP1. TP2/DP2 appear only in target registry/coordinator contracts and are
  rejected by this release. There is no multi-node, Kubernetes,
  elastic-cluster, remote-object-store, or automatic failover claim.
- The CPU `gloo` contract is not GPU/NCCL evidence. A topology configuration is
  only target vocabulary; this release rejects TP2/DP2 regardless of any
  caller-supplied capability receipt.
- Static/TTS/L0 remain `BLOCKED` until an out-of-band trusted signer is bound to
  the exact native terminal hook and pinned tree. DSpark/EAGLE/EAGLE3/NEXTN
  adaptive cells and all TP2/DP2 cells are also `BLOCKED`; they are never
  implicit successes.
- ChronoBelief tuning cells stay `BLOCKED` until an authoritative update
  equation and source identity are registered; no substitute optimizer is used.
- Historical KV remains frozen by design. Recomputing it requires a new
  algorithm, memory envelope, protocol, and claim.

## Contributing and license

Read [CONTRIBUTING.md](CONTRIBUTING.md),
[CONTRIBUTING_zh-CN.md](CONTRIBUTING_zh-CN.md), and
[SECURITY.md](SECURITY.md). LightCone-Spec is licensed under
[Apache-2.0](LICENSE); external models, datasets, and SGLang retain their own
licenses.
