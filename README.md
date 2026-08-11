# LightCone-Spec

[简体中文](README_zh-CN.md) · [Documentation](docs/en/architecture.md) · [License](LICENSE)

LightCone-Spec is an evidence-first framework for testing online drafter
adaptation in speculative decoding. It separates runtime engineering from
empirical claims: CPU contracts can establish identity, durability, and
fail-closed behavior, but only registered, attested GPU evidence can establish
speed or capacity.

> Alpha software. Every new GPU outcome is `UNMEASURED`. The industrial
> executor currently runs only Target-only end to end. Static, TTS, L0, and
> every other speculative method are `BLOCKED` before process or network
> mutation because the pinned SGLang integration does not provide
> `sglang.schema_v3.content_bound_terminal_speculative_evidence.v1`. Empirical
> Stage B is also `BLOCKED` on provider credentials and registered hardware.
> Historical v2 artifacts are regression/debugging evidence only; they are not
> evidence for the schema-v3 protocol or a new performance claim.

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
production load, two-GPU topology, native NEXTN preflight, and breadth
templates. EAGLE and EAGLE3 remain target backend contracts with narrow
compatibility guards; registration is not executable release support.
OnlineSPEC remains an important comparison with separate tuning, evidence,
attestation, and analysis; it cannot select or alter the core gate.

The current pinned SGLang patch contains a lower-level adaptive path only for
TP1/DP1 DFlash with a constant optimizer schedule and zero extra logical
delay. It does not expose the content-bound terminal evidence provider required
by the industrial executor, so that path is not end-to-end runnable or
claimable there. Static/TTS/L0 fail executor preflight before mutation.
DSpark/EAGLE/EAGLE3/NEXTN adaptation, all TP2/DP2 execution, nonconstant
schedules, and positive extra delay also fail closed; their registry cells are
`BLOCKED` until a new patch and provider identity implement them.

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

Every cell binds scientific axes, seed, GPU UUIDs, ports, cache/evidence roots,
resource isolation, and a truthful initial status. A stage receipt seals the
runtime and split digests, exact dependency receipts, and locked outputs before
the next stage can be dispatched. The two-GPU scheduler serializes exclusive
headline/profile/download/compile work and permits independent single-GPU work
to overlap only after an interference gate passes.

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

Build the two-GPU industrial registry with real immutable device identities:

```bash
lightcone-spec build-industrial-registry \
  --gpu-uuid GPU-UUID-0 GPU-UUID-1 \
  --cache-root runtime-cache/industrial \
  --evidence-root artifacts/industrial \
  --output artifacts/industrial/registry.json

lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
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

For exactness diagnostics, `run-target-reference` can capture a locked
Target-only greedy reference whose per-prompt outputs are format-tagged hashes
of complete ordered token-ID trajectories. Legacy collectors require every
method/block to match that reference; decoded-text agreement or agreement only
among speculative methods is insufficient. The reference strengthens an
`UNMEASURED` diagnostic only: it does not make Static/TTS/L0 executable and
cannot substitute for the absent trusted attester.

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

- All new GPU outcomes are `UNMEASURED`. Stage B is `BLOCKED` on the missing
  native terminal speculative-evidence provider, provider credentials, and
  registered hardware.
- The current end-to-end industrial execution surface is Target-only at
  TP1/DP1. TP2/DP2 appear only in target registry/coordinator contracts and are
  rejected by this release. There is no multi-node, Kubernetes,
  elastic-cluster, remote-object-store, or automatic failover claim.
- The CPU `gloo` contract is not GPU/NCCL evidence. A topology configuration is
  only target vocabulary; this release rejects TP2/DP2 regardless of any
  caller-supplied capability receipt.
- Static/TTS/L0 remain `BLOCKED` until the exact native terminal evidence hook
  is implemented and bound to the pinned tree. DSpark/EAGLE/EAGLE3/NEXTN
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
