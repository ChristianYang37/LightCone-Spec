# Changelog

All notable changes to LightCone-Spec are documented here. The format follows
Keep a Changelog. The project remains pre-release; a stable public API has not
yet been declared.

## [Unreleased]

### Added

- Added cache-safe tail implementations of Static/TTS/L0 for DSpark,
  EAGLE, and EAGLE3, with backend-specific version and proposal contracts.
- Added selectable Adam, AdamW, SGDm, NAG, Muon, and Lion online optimizers
  with strict configuration identity and state-aware HBM accounting.
- Added clean-room projected OGD, two-state optimistic OGD, and cumulative-loss
  Hedge as a provenance-bound OnlineSPEC comparison, with separate tuning,
  paired confirmation, telemetry, memory accounting, and GPU attestation.
- Streamed OnlineSPEC expert backward passes through one reusable gradient
  scratch while preserving the same cumulative-loss Hedge decision.
- Added a separately identified LoRA-coordinate Hedge decision class for
  memory-bounded OnlineSPEC comparisons; dense Full Hedge remains available
  and is never silently downgraded.
- Covered the source-audited EAGLE3 learning-rate scale in the Hedge tuning
  grid without selecting hyperparameters from confirmation evidence.

### Changed

- Replaced per-prompt pseudo-load timing and host-thread admission races with
  one ordered native SGLang batch queue, active-decode interval unions, and
  repetition-block inference for both the core and OnlineSPEC protocols.
- Bound OnlineSPEC source-point gradients to the exact inference proposal
  value while retaining the differentiable surrogate Jacobian.
- Bound OnlineSPEC selection to the core Static load selection and capped
  formal admission at the 32 unique confirmation prompts.

### Planned

- GPU certification of the registered Qwen3-8B + DFlash speed study.
- Multi-GPU validation and GPU certification of the compatibility backends.

## [0.2.0] - 2026-08-09

### Changed

- Focused the formal runtime and experiment protocol on Static, paper-faithful
  TTS, and first-ready L0.
- Replaced the public configuration with strict schema-v2 identities.
- Added DFlash drafter Full/LoRA updates, cache-safe tail ablations, frozen
  historical KV versioning, and cohort-scoped publication.
- Replaced the SGLang integration with a reproducible eight-patch series against
  one exact upstream commit.
- Added deterministic disjoint data windows, tuning-only maximin selection,
  independent confirmation timing, resumable evidence receipts, and a
  content-bound GPU speed gate.
- Rewrote English and Chinese documentation without experimental results or
  performance claims.

GPU status for this source version is `UNMEASURED`. No release tag is created.
