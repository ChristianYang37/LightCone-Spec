# LightCone-Spec

[简体中文](README_zh-CN.md) · [Documentation](docs/en/architecture.md) · [License](LICENSE)

LightCone-Spec is an evidence-first research framework for testing whether
test-time drafter updates can make speculative decoding faster. Version 0.2.0
has one deliberately narrow question: can paper-faithful TTS and a first-ready
publication policy both improve decode goodput over an unchanged Static
baseline?

> Alpha software. GPU performance is `UNMEASURED` until an immutable formal
> run passes the registered statistical and safety gate. This repository does
> not publish benchmark results or performance claims.

## Scope

The formal study contains exactly three methods:

| Method | Candidate computation | Publication policy |
|---|---|---|
| Static | none | native SGLang speculative decoding |
| TTS | side-CUDA-stream update every stride | synchronize and publish at the next fixed update boundary |
| L0 (`naive_async`) | byte-for-byte equivalent update to TTS | publish at the first legal graph boundary after the ready event |

TTS follows the scheduling definition in the [Test-Time Speculation
paper](https://arxiv.org/abs/2605.09329). L0 changes only the publication time;
it is not a different loss or optimizer.

The formal speed study remains Qwen3-8B + DFlash with TP=DP=1. The patchset
also implements the same Static/TTS/L0 publication contract for DSpark and
single-layer, top-k-one EAGLE/EAGLE3 through cache-safe tail updates. Those
compatibility paths are GPU-`UNMEASURED` and are not silently substituted into
the formal DFlash gate. Three clean-room OnlineSpec equations are kept under an
isolated `baselines` package and never enter the default study.

## Performance model

The study does not assume an update is beneficial:

\[
T_m=T_{\mathrm{static}}-\Delta T_{\mathrm{target}}
+T_{\mathrm{update}}^{\mathrm{exposed}}
+T_{\mathrm{draft}}^{\mathrm{extra}}+T_{\mathrm{barrier}}.
\]

An adapted method is faster only if fewer target calls save more time than
training, contention, publication, and barrier overhead consume. Formal claims
therefore require paired decode goodput, exactness counters, target-call
counts, CUDA timing, HBM accounting, and confidence intervals together.

## Architecture

- `lightcone_spec` owns strict schema-v2 configuration, deterministic data
  windows, selection, evidence records, and statistical gates.
- `patches/sglang` is a reproducible six-patch mail series against one exact
  upstream commit. The repository never vendors or edits SGLang in place.
- A cohort runtime keeps optimizer state on GPU, publishes into fixed-address
  inference tensors, and binds every candidate to epoch, slot generation, and
  source version. Adam, AdamW, SGDm, NAG, Muon, and Lion share this functional
  propose-then-commit path.
- Headline telemetry uses asynchronous CUDA events. Synchronizing diagnostics
  and profilers run after the measured interval or in a separate run.

See [Architecture](docs/en/architecture.md) and [Mathematical
method](docs/en/mathematical-method.md).

## Installation

Framework-only development:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Create a disposable patched SGLang checkout from the exact upstream pin:

```bash
git clone https://github.com/sgl-project/sglang.git /path/to/sglang
git -C /path/to/sglang checkout 3312645a307453893a00778592f105581e3d1c3d
patches/sglang/apply.sh /path/to/sglang
python scripts/verify_sglang_patchset.py \
  --upstream-checkout /path/to/clean-upstream --compile-only
```

See [Installation](docs/en/installation.md) for the GPU environment contract.

## Quick start

Build the immutable source protocol and sampling profile:

```bash
lightcone-spec build-speed-study \
  --output artifacts/protocol/static_tts_l0.json
```

Lock model revisions before downloading:

```bash
lightcone-spec lock-models --output artifacts/locks/models.json \
  Qwen/Qwen3-8B z-lab/Qwen3-8B-DFlash-b16
lightcone-spec prepare-models --lockfile artifacts/locks/models.json \
  --model-cache /path/to/model-cache \
  --output artifacts/locks/model-roots.json
```

Render one allocation-free Static endpoint per registered concurrency before
tuning (shown for `C=48`):

```bash
lightcone-spec render-static-load-runtime --concurrency 48 \
  --sglang-checkout /path/to/patched-sglang \
  --model-lock artifacts/locks/models.json \
  --model-roots artifacts/locks/model-roots.json \
  --sampling-profile manifests/speed-study/sampling_profile_v2.json \
  --mem-fraction-static MEMORY_FRACTION \
  --output-root artifacts/load/c48
```

After the independent Static load screen and tuning phase have produced a
selection artifact, render one exclusive-device launch plan. All methods reuse
one port and one GPU **sequentially**; never start the three server argv vectors
at the same time:

```bash
lightcone-spec render-runtime \
  --sglang-checkout /path/to/patched-sglang \
  --selection artifacts/selection.json \
  --model-lock artifacts/locks/models.json \
  --model-roots artifacts/locks/model-roots.json \
  --sampling-profile manifests/speed-study/sampling_profile_v2.json \
  --adaptation-group-id formal-a --adaptation-reserve-mb RESERVE_MB \
  --mem-fraction-static MEMORY_FRACTION \
  --output-root artifacts/runtime

lightcone-spec build-confirmation-queue \
  --manifest manifests/speed-study/static_tts_l0_v2.json \
  --selection artifacts/selection.json \
  --model-lock artifacts/locks/models.json \
  --sampling-profile manifests/speed-study/sampling_profile_v2.json \
  --launch-plan artifacts/runtime/launch-plan.json \
  --evidence-root artifacts/confirmation \
  --output artifacts/confirmation/queue.json
```

For each queue job, start its `launch_argv`, wait for health, run its
`run_argv`, then terminate that server before the next job. Finally use
`collect-speed-study` to derive the formal table. The queue is data, not a
shell script: orchestration must preserve the registered order and clean-server
boundaries.

`RESERVE_MB` and `MEMORY_FRACTION` are intentionally not source defaults. They
must come from hardware preflight and the selected parameter layout.

## Update modes and cache contract

The public schema accepts:

- `residual`: tail-only low-rank logit correction;
- `lora`: low-rank factors for the drafter or a tail ablation, merged into
  fixed-address inference weights at publication;
- `full`: all DFlash-owned floating parameters for drafter scope, or a
  full-rank tail ablation. Target embeddings, target LM head, and target model
  remain frozen.

DFlash supports drafter Full/LoRA and all three tail modes. DSpark and
EAGLE/EAGLE3 deliberately support tail scope only: residual, tail LoRA, and
full-rank tail. DSpark applies the tail to the exact post-normalization LM-head
input before its Markov correction. EAGLE pins the proposal version from
draft-extend through verification and cannot publish while that proposal is
outstanding.

The selectable online optimizers are `adam`, `adamw`, `sgdm`, `nag`, `muon`,
and `lion`. Muon uses its matrix orthogonalization path for two-dimensional
parameters and an explicitly configured auxiliary AdamW path for non-matrix
parameters. Optimizer identity and all optimizer-specific fields are part of
the cohort, selection, layout, and evidence hashes.

Historical drafter KV is immutable. KV created before publication is neither
rebuilt nor differentiated; new KV records the newly published source version.
The actual proposal distribution remains the distribution used by exact
speculative rejection sampling.

## Evidence and safety

Formal confirmation uses 32 held-out controlled prompts, eight independent
method-order blocks, one selected load, and generated-token buckets through each
prompt's checkpoint-safe end. The 40,960-token model limit always includes the
tokenized prompt; the headline region begins at 16K generated tokens. Both TTS
and L0 must independently clear the registered speed threshold with a paired
prompt-cluster BCa interval and zero safety events.

The gate cannot report `PASS` without a content-bound GPU attestation covering
the manifest, tuning selection, patched SGLang tree, model revisions, hardware
report, and exact Parquet inputs. A local or synthetic table remains
`UNMEASURED`, even when its arithmetic effect is positive.

## Documentation

- [Architecture](docs/en/architecture.md)
- [Mathematical method](docs/en/mathematical-method.md)
- [Installation](docs/en/installation.md)
- [Configuration](docs/en/configuration.md)
- [CLI](docs/en/cli.md)
- [SGLang patch workflow](docs/en/sglang-patches.md)
- [Experiment protocol](docs/en/experiment-protocol.md)
- [Troubleshooting](docs/en/troubleshooting.md)

## Limitations and roadmap

- GPU status is currently `UNMEASURED`; no speedup is asserted.
- Adaptation requires TP=DP=1 and an unquantized draft/KV path. Drafter-scope
  Full/LoRA is DFlash-only; DSpark requires verify-all execution, and adapted
  EAGLE/EAGLE3 requires a single layer, fixed depth, top-k one, and exact
  full-vocabulary rejection sampling. Unsupported combinations fail before
  adaptation allocation.
- Historical KV is frozen by design. Recomputing old KV would define a
  different method and memory envelope.
- GPU certification outside the formal DFlash pair and multi-GPU adaptation
  are future work. Implemented compatibility does not imply measured speedup.

## Contributing and license

Read [CONTRIBUTING.md](CONTRIBUTING.md),
[CONTRIBUTING_zh-CN.md](CONTRIBUTING_zh-CN.md), and
[SECURITY.md](SECURITY.md). LightCone-Spec is licensed under
[Apache-2.0](LICENSE); external models, datasets, and SGLang retain their own
licenses.
