# SGLang patch workflow

[中文](../zh-CN/sglang-patches.md) · [Home](../../README.md)

## Source boundary

SGLang is an external Apache-2.0 project. LightCone-Spec pins upstream commit
`3312645a307453893a00778592f105581e3d1c3d` and distributes only mail-formatted
patches. The repository contains no SGLang source, submodule, or modified
checkout. A local `sglang/` directory is ignored and is never an integration
identity.

## Series layers

The six-patch series has one-way semantic dependencies:

1. strict cross-backend schema, preflight, and disabled fast path;
2. cohort, resident optimizer, source-version, CUDA-event, and publication
   runtime;
3. differentiable DFlash drafter Full/LoRA update path;
4. cache-safe DFlash, DSpark, EAGLE, and EAGLE3 tail paths;
5. memory accounting, lifecycle, telemetry, and profiling integration;
6. cross-backend optimizer, proposal, exactness, and regression tests.

Only the complete series is supported. Intermediate patch states are review
boundaries, not runnable product variants.

## Application

`patches/sglang/apply.sh` accepts only a clean checkout at the exact upstream
HEAD. It verifies every patch digest, applies with `git am`, and checks the
final Git tree recorded in `manifest.json`:

```bash
patches/sglang/apply.sh /path/to/clean-sglang
```

Any dirty state, wrong commit, changed patch, failed mail application, or final
tree mismatch stops immediately. The script never rebases, stashes, resets, or
edits the supplied checkout to make a mismatch disappear.

## Verification and authoring

Run the disposable verifier before publishing:

```bash
python scripts/verify_sglang_patchset.py \
  --upstream-checkout /path/to/clean-upstream
```

New integration changes are patch-first: create a temporary branch from the
pin, make one semantic commit, add focused tests, export with `git format-patch`,
then update `series`, patch SHA-256 values, modified-file lists, expected final
tree, Python pin constants, NOTICE, and documentation together. Finally verify
application and reverse removal while confirming the original upstream source
remains clean.
