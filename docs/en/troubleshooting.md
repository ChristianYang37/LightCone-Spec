# Troubleshooting

[中文](../zh-CN/troubleshooting.md) · [Home](../../README.md)

- **Patch application is rejected:** confirm the full upstream commit, detached
  clean status, patch checksums, and `patches/sglang/manifest.json`.
- **Runtime fingerprint mismatch:** do not edit a screened runtime. Build a new
  disposable checkout and start a fresh evidence root.
- **Model loading is refused:** regenerate the lock and model-roots file, then
  verify the requested pair is fully present. Never weaken hash checks.
- **Controller is rejected:** compare its model pair, parameter-layout hash,
  data window, sampling, lifecycle, and runtime identity with the run.
- **Out of device memory:** reduce admitted requests or context/load. Adaptation
  buffers are intentionally not offloaded or silently downgraded.
- **Queue stops with a scientific block:** inspect the attested terminal receipt.
  Resume only through the final gate's recursive `verify-resume` interface.

When reporting a bug, include sanitized commands, platform versions, the exact
upstream and patch tree IDs, and the smallest reproducible manifest. Never post
tokens, passwords, private model paths, or raw private prompts.
