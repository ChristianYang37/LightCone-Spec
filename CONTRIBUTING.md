# Contributing to LightCone-Spec

[简体中文](CONTRIBUTING_zh-CN.md)

Thank you for helping improve LightCone-Spec. By participating, you agree to
keep changes reproducible, reviewable, and safe for users running large models.

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Run `python -m compileall -q src scripts tests`, `git diff --check`, and the
relevant focused tests before opening a pull request. GPU behavior needs a
separate GPU-marked test and must not make CPU tests require CUDA.

## Design rules

- Prefer the smallest reusable abstraction; avoid backend copies of lifecycle,
  version-bank, or controller logic.
- Fail closed before model loading when identity, exactness, or compatibility
  cannot be established.
- Keep the disabled speculative path allocation-free and upstream-compatible.
- Never commit model weights, datasets, telemetry, profiles, experiment
  results, credentials, provider state, or machine-specific paths.
- Do not add a SGLang checkout or submodule. Author changes against the pinned
  upstream as a semantic mail patch and update the manifest, checksums, tests,
  expected tree, and documentation together.

## Pull requests

Use focused commits and explain the invariant being changed. Include CPU tests,
GPU-test requirements, compatibility impact, memory-accounting impact, and
evidence that exactness is preserved. Performance claims require an approved
experiment protocol and are not accepted from ad hoc measurements.

Contributions are licensed under Apache-2.0 unless explicitly stated otherwise.
