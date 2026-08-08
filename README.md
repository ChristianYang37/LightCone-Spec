# LightCone-Spec

[简体中文](README_zh-CN.md) · [Documentation](docs/en/architecture.md) · [License](LICENSE)

LightCone-Spec is a research framework for version-safe asynchronous test-time
adaptation in speculative decoding. It separates proposal correction from the
drafter backbone, publishes updates only at legal decode boundaries, and binds
every controller and run to reproducible model, runtime, and data identities.

> Alpha software. This repository publishes implementation and evaluation
> protocols, not performance claims or experiment results.

## Architecture

The system has three deliberately separate layers:

1. `lightcone_spec` builds candidates, controllers, immutable manifests, and
   evidence artifacts.
2. A pinned patch series adds backend-neutral proposal signals and a versioned
   tail bank to a disposable SGLang checkout.
3. Experiment runners enforce exactness, model locks, request-level splits,
   telemetry integrity, and fail-closed scientific gates.

Adaptation tensors remain GPU-resident. Active buffers keep stable addresses
for graph replay; staging, optimizer, and candidate work run on a side stream.
Request epoch, slot generation, and source version jointly prevent stale or ABA
publication.

## Supported backends

| Backend | Supported scope | Current boundary |
|---|---|---|
| DSpark | residual, LoRA, full-rank tail; Markov and confidence signals | checkpoint and proposal-depth constraints are validated before model load |
| DFlash | residual, LoRA, full-rank tail | deterministic proposals and certified rejection-sampling paths only |
| EAGLE / EAGLE3 | residual, LoRA, full-rank tail | single-layer, `topk=1`; unsupported tree/multi-layer combinations fail closed |

The unmodified SGLang path is preserved when adaptation is disabled. The
repository does not vendor SGLang or model weights.

## Installation

For framework-only development:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
lightcone-spec --help
```

For a GPU runtime, first run the read-only preflight, then use the native
installer. The installer creates a disposable checkout from the exact upstream
commit and applies the verified patch series; it never edits system Python or
CUDA and does nothing without `--execute`.

```bash
lightcone-spec doctor --output doctor.json
python scripts/install_native.py --runtime-root ~/lightcone-spec-runtime
python scripts/install_native.py --runtime-root ~/lightcone-spec-runtime --execute
```

See [installation](docs/en/installation.md) and the
[SGLang patch workflow](docs/en/sglang-patches.md).

## Quick start

Create immutable input locks before downloading or loading a model:

```bash
lightcone-spec lock --output lightcone.lock.json \
  --pairs qwen3_4b_dflash16 --datasets livecodebench
lightcone-spec prepare-models --lockfile lightcone.lock.json \
  --model-cache /path/to/cache --pairs qwen3_4b_dflash16 \
  --output model-roots.json
lightcone-spec run-manifest --manifest manifests/smoke/smoke_gpu_qwen3_4b.json \
  --lockfile lightcone.lock.json --model-roots model-roots.json \
  --runtime-root /path/to/runtime --model-cache /path/to/cache
```

Real runs require explicit selection and controller artifacts where the
manifest calls for them. Missing, mismatched, or result-derived implicit
defaults are rejected before model loading.

## Update modes and schedulers

`--weight-update-mode` accepts exactly:

- `residual`: a compressed output-logit residual;
- `lora`: a low-rank update at the proposal tail;
- `full`: a full-rank proposal-tail update, not full-drafter fine-tuning.

L0 publishes a completed candidate at the first legal boundary. L1 gates
candidates, L2 damps their magnitude, and L3 evaluates transported candidates.
Controller stages remain diagnostic unless their immutable evidence gates pass.

## Correctness, memory, and evidence

- Proposal sampling and rejection use the same corrected distribution `q`.
- Semantic masks exclude rejected suffixes, post-stop tokens, bonus tokens past
  the request boundary, and tokens beyond `max_new_tokens`.
- Adaptation memory is reserved before KV-pool sizing and is not silently
  offloaded or evicted. Admission control and KV retraction handle pressure.
- Lightweight CUDA-event telemetry is separate from synchronized profiling.
- Artifacts bind model revisions, parameter layout, runtime sources, data
  windows, seeds, and upstream/patch identities.

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

GPU and model compatibility is intentionally narrower than SGLang itself.
Unsupported speculative trees, parameter scopes, checkpoint context windows,
and sampling combinations fail closed. Planned work includes broader backend
coverage, more cache-safe trainable scopes, multi-GPU certification, and public
evidence only after the corresponding gates are complete.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md),
[CONTRIBUTING_zh-CN.md](CONTRIBUTING_zh-CN.md), and [SECURITY.md](SECURITY.md).
LightCone-Spec is licensed under [Apache-2.0](LICENSE).
