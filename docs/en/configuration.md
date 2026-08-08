# Configuration

[中文](../zh-CN/configuration.md) · [Home](../../README.md)

Configuration is resolved in layers: schema defaults, immutable manifest
values, and a small set of explicit CLI overrides. `--weight-update-mode` is
the single public override for `residual`, `lora`, or `full`; explicit overrides
recompute unit and artifact identities.

Real manifests must bind a lockfile, model-roots digest, runtime fingerprint,
dataset window, sampling profile, and seed. Controller methods additionally
require an artifact whose model pair, normalization, distance weights, data
hash, parameter layout, and lifecycle match the run.

Static units do not allocate adaptation state. Unsupported combinations and
ambiguous legacy/new configuration sources are rejected before model loading.
Never place access tokens, passwords, machine paths, or result-derived selected
hyperparameters in a manifest committed to the repository.
