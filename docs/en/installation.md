# Installation

[中文](../zh-CN/installation.md) · [Home](../../README.md)

## Framework development

Use an isolated Python environment and install the project in editable mode:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

## GPU runtime

Run `lightcone-spec doctor` first. The native installer requires a supported
NVIDIA driver/toolkit combination and user-space build tools. Its default mode
is read-only. With `--execute`, it creates a fresh runtime root, clones the
pinned SGLang commit, applies `patches/sglang/series`, creates a virtual
environment, and records dependency and source provenance.

```bash
python scripts/install_native.py --runtime-root /absolute/runtime/path
python scripts/install_native.py --runtime-root /absolute/runtime/path --execute
```

The installer does not use `sudo`, replace system CUDA, accept a dirty SGLang
tree, or reuse an unknown virtual environment. Model preparation is a separate,
lock-controlled operation.
