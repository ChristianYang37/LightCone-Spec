# Installation

[中文](../zh-CN/installation.md) · [Home](../../README.md)

## Framework environment

Use an isolated environment. The core package has no SGLang source dependency:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
lightcone-spec doctor
pytest -q
```

The optional `gpu` extra installs the dataset loader used by the natural-task
side tables. Controlled prompts and all CPU tests work without it.

## Patched SGLang checkout

SGLang must live outside this repository. Clone the exact pin and apply the
mail series to that clean checkout:

```bash
git clone https://github.com/sgl-project/sglang.git /path/to/sglang
git -C /path/to/sglang checkout --detach \
  3312645a307453893a00778592f105581e3d1c3d
patches/sglang/apply.sh /path/to/sglang
```

To audit the series without changing the source checkout:

```bash
python scripts/verify_sglang_patchset.py \
  --upstream-checkout /path/to/clean-upstream --compile-only
```

Omit `--compile-only` only in an environment that has the pinned SGLang test
dependencies. The verifier clones into a temporary directory, applies all
patches, checks the expected Git tree, compiles the changed Python surface,
runs the focused test when requested, reverses the patches, and confirms that
the supplied upstream checkout remained clean.

## GPU environment contract

Run `lightcone-spec doctor --path /path/to/patched-sglang` before model loading.
Record the NVIDIA driver, toolkit, PyTorch runtime, compiler, free storage, and
patched tree with the eventual evidence. Do not replace system Python or CUDA,
use `sudo`, or reuse an unidentified environment.

The repository deliberately has no one-command GPU installer: CUDA and wheel
compatibility depend on the measured host. Create a user-space environment,
pin its resolved packages, and keep the model cache and run artifacts outside
the source tree. A failed preflight is a stop condition, not permission to
silently change precision, update mode, or memory policy.

## Model preparation

Resolve immutable revisions before downloading:

```bash
lightcone-spec lock-models --output artifacts/locks/models.json \
  Qwen/Qwen3-8B z-lab/Qwen3-8B-DFlash-b16
lightcone-spec prepare-models \
  --lockfile artifacts/locks/models.json \
  --model-cache /path/to/model-cache \
  --output artifacts/locks/model-roots.json
```

Pass credentials only through a temporary `HF_TOKEN` environment variable or
another secure credential channel. Never put a token in a command argument,
manifest, log, or repository file.
