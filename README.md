# LightCone-Spec

[简体中文](README_zh-CN.md) · [Documentation](docs/en/architecture.md) · [License](LICENSE)

LightCone-Spec is an evidence-first research framework for testing whether
test-time drafter updates can make speculative decoding faster. Version 0.2.0
has one deliberately narrow question: can paper-faithful TTS and a first-ready
publication policy both improve decode goodput over an unchanged Static
baseline?

> Alpha software. The preliminary GPU snapshot below is a controlled
> mechanism check, not a formal paper-reproduction claim. Formal GPU status
> remains `UNMEASURED` until an immutable run passes the registered
> statistical, target-exactness, and safety gate.

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
the formal DFlash gate. Three clean-room OnlineSPEC learners are kept under an
isolated `baselines` package. OnlineSPEC is an important registered comparison
with its own tuning and paired evidence path; it never changes core selection
or the Static/TTS/L0 speed gate. Its pinned, machine-readable
[source audit](manifests/provenance/onlinespec_source_audit_v2.json) distinguishes
the paper equations, the released recipes, and this project's implementation.
The official paper's Qwen3-8B result uses a Qwen3-0.6B Lookahead Reasoning
drafter; applying the clean-room learners to DFlash is a separate
cross-architecture comparison, not a reproduction of that system result.

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

## Preliminary controlled GPU snapshot

The following snapshot tests the mechanism on one RTX PRO 6000 Blackwell
(96 GB) with Qwen3-8B + DFlash-b16, concurrency 8, greedy decoding, and a
40,928-token safe context limit. Each method processed the same 16 deterministic
long-continuation prompts (654,042 generated tokens per method). CUDA Graph and
Radix Cache were disabled by the registered exactness-safe policy. TTS and L0
used the same drafter LoRA rank 8, Adam learning rate `1e-3`, and update stride
80; only publication timing differed.

| Method | Decode goodput | vs. Static | p99 ITL | Peak HBM |
|---|---:|---:|---:|---:|
| Static DFlash | 1,342.0 tok/s | 1.00x | 45.04 ms | 90.36 GiB |
| TTS | 2,497.9 tok/s | 1.86x | 45.94 ms | 90.52 GiB |
| L0 | **2,519.5 tok/s** | **1.88x** | 46.21 ms | 90.53 GiB |

L0 was 0.86% faster than TTS in this snapshot. Complete token-ID trajectories
were identical across all three methods, and exactness-violation, version-
mismatch, fallback, non-finite-update, OOM, and retraction counters were zero.
The reproducibility identity is `LightCone-Spec@0db2ff4`, patched SGLang tree
`e795ecc`, execution-policy SHA `231ca579`, and tuning-window SHA `132019ee`.

These are single-block tuning-window measurements (`n=1` timing block), not
BCa confidence intervals or natural-task results. The repetitive controlled
prompts intentionally make within-request adaptation observable; they do not
establish the paper's LiveCodeBench, mathematics, or OnlineSPEC claims. Legacy
repeated timings that hashed decoded text instead of complete token IDs are
deliberately excluded from this table.

## Architecture

- `lightcone_spec` owns strict schema-v2 configuration, deterministic data
  windows, selection, evidence records, and statistical gates.
- `patches/sglang` is a reproducible ten-patch mail series against one exact
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

Build the immutable source protocol. Formal launches also bind the checked-in
sampling profile and role-bound execution policy:

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
render the target-only launch plan, run `run-target-reference` once against that
separate server at the same load, then pass the artifact to
`collect-speed-study`:

```bash
lightcone-spec render-target-runtime --concurrency SELECTED_CONCURRENCY \
  --sglang-checkout /path/to/patched-sglang \
  --model-lock artifacts/locks/models.json \
  --model-roots artifacts/locks/model-roots.json \
  --sampling-profile manifests/speed-study/sampling_profile_v2.json \
  --mem-fraction-static MEMORY_FRACTION \
  --output-root artifacts/target-runtime
```

The registered policy fixes the model context and server seed and disables
radix caching and CUDA graphs on both endpoint roles. The target-only reference
also disables overlap scheduling, while measured DFlash endpoints retain the
native overlap path; both assignments are part of one content identity.
Changing any of those controls creates a different, non-formal runtime
identity. Agreement among speculative methods is insufficient: every
method/block must match the target-only greedy token-ID hashes. Decoded text
alone is not an exactness witness because distinct token sequences can decode
to the same text.
The queue is data, not a
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

Formal confirmation submits 32 distinct held-out controlled prompts to SGLang
in one ordered native batch request per method across eight independent
method-order blocks. The locked server admission limit controls the active batch and the
cohort is not reset while the queue drains. Each method/block owns the union of
its active decode intervals; request diagnostics remain separate, so shared
batch timing is not pseudoreplicated as 32 goodput samples. The 40,960-token
model limit includes the tokenized prompt. Formal prompt-plus-generated context
stops at 40,928 so DFlash retains two block-16 speculative KV reservations; the
headline region
begins at 16K generated tokens. TTS and L0 must each clear the registered speed
threshold, repetition-block BCa interval, and zero safety events.

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
- [OnlineSPEC baseline](docs/en/onlinespec-baseline.md)
- [Troubleshooting](docs/en/troubleshooting.md)

## Limitations and roadmap

- Formal GPU status remains `UNMEASURED`; the published table is explicitly a
  single-block controlled snapshot rather than an attested paper-reproduction
  result.
- Adaptation requires TP=DP=1 and an unquantized draft/KV path. Drafter-scope
  Full/LoRA is DFlash-only; DSpark requires verify-all execution, and adapted
  EAGLE/EAGLE3 requires a single layer, fixed depth, top-k one, and exact
  full-vocabulary rejection sampling. Unsupported combinations fail before
  adaptation allocation.
- Historical KV is frozen by design. Recomputing old KV would define a
  different method and memory envelope.
- GPU certification outside the formal DFlash pair and multi-GPU adaptation
  are future work. Implemented compatibility does not imply measured speedup.
- The OnlineSPEC comparison is a clean-room implementation of the published
  online-learner equations. Its official repository had no project-level
  license file at the pinned audit commit, so no upstream source is
  redistributed or copied.

## Contributing and license

Read [CONTRIBUTING.md](CONTRIBUTING.md),
[CONTRIBUTING_zh-CN.md](CONTRIBUTING_zh-CN.md), and
[SECURITY.md](SECURITY.md). LightCone-Spec is licensed under
[Apache-2.0](LICENSE); external models, datasets, and SGLang retain their own
licenses.
