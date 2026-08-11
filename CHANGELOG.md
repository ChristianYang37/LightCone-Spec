# Changelog

All notable changes to LightCone-Spec are documented here. The format follows
Keep a Changelog. The project remains pre-release; a stable public API has not
yet been declared.

## [Unreleased]

No unreleased entries.

## [0.3.0] - 2026-08-11

### Added

- Added canonical Target-only, Static, TTS, and `l0` method identities and a
  strict schema-v3 runtime configuration. Target-only and Static retain
  allocation-free adaptation paths.
- Added a backend-neutral proposal-evidence envelope with adapter-free/deployed
  logits, exact sampling distribution, semantic masks, teacher rows, sampled
  predecessors, cohort/source identity, device predicates, a lower-level
  TP1/DP1 DFlash reconstruction path, and non-executable target contracts for
  DSpark, EAGLE/EAGLE3, and NEXTN.
- Added Full/LoRA native layer plans for `last1`, `last3`, `last5`, and `all`,
  with LoRA ranks 1/2/4/8/16/32/64 at `alpha/r=1`, explicit selected/frozen
  names, inference-aligned ownership, and exact plan memory prediction.
- Added DSpark layer-only and native-head hybrid plans. The registered E1a grid
  has exactly 56 adaptive configurations and uses native W1/W2 features,
  sampled-predecessor evidence, and a proper conditional-survival confidence
  objective.
- Added target one-node TP2 and sticky-replica DP2 identities, capability
  receipt vocabulary, sharded/replicated ownership, a CPU all-rank two-phase
  publication contract, and a real CPU `gloo` collective harness. The release
  validator still rejects all TP2/DP2 execution.
- Added least-rank HBM admission, an ordered pressure policy, and tenant/
  replica-isolated fixed cohort slabs with generation-safe reclamation and
  optional timed cold offload.
- Added immutable method-independent production traces, deterministic labelled
  synthetic Poisson and burst generators, timeout/cancellation accounting, and
  strict paired-trace identity.
- Added bounded process-unique evidence writing with durable Parquet WAL
  segments, checkpoints, duplicate detection, backpressure/drop counters,
  crash-retained attempts, final coverage validation, and exclusive
  content-bound completion receipts.
- Added an immutable industrial experiment registry and deterministic two-GPU
  resource planner for
  `preflight -> E3a -> E1 -> E2 -> E4 -> E3b -> E1a -> E5 -> E6 -> E0`, plus
  CLI commands to build the registry, plan dispatch, and seal stage receipts.
- Added preregistered four-pilot power sizing for 12--20 final blocks, Holm
  adjustment for L0--Static/L0--TTS, Benjamini--Hochberg FDR for secondary
  families, hierarchical block/request and time-block bootstrap, p99 completion
  guards, production SLO accounting, and hardware/power-envelope validation.

### Changed

- Made `l0` the sole canonical first-ready method key and removed the legacy
  method alias from the public protocol.
- Restricted adaptation to Full and LoRA native parameter plans; retired
  correction-only adaptation fields and scope aliases are rejected rather than
  normalized.
- Kept TTS and L0 on one candidate, optimizer, reconstruction, and evidence
  path; their only experimental distinction is publication timing.
- Made historical drafter KV immutable and versioned across backends. New
  publication affects future KV only.
- Made the rejection sampler's normalized positive part `(p-q)_+` explicitly a
  sampling exactness rule, not an adaptation configuration.
- Made the pinned official SGLang serving adapter retain ordered server
  `output_ids` across cumulative/incremental streaming and non-streaming
  responses, without text retokenization.
- Expanded public EN/zh documentation for schema v3, Target-only, backend
  evidence, DSpark, topology receipts, HBM/cohort governance, durable telemetry,
  immutable traces, the industrial DAG, two-GPU staging, and statistics.
- Kept the clean-room OnlineSPEC comparison isolated under its own tuning,
  evidence, attestation, and analysis identities; it cannot alter the core
  selection or gate.
- Required a target-only greedy reference artifact in both formal
  diagnostic collectors and analyzers; it cannot mint `MEASURED` evidence in
  this release.
- Made cross-method output agreement insufficient for exactness: every formal
  method/block must also match the locked target-only trajectory.
- Replaced decoded-text output digests with format-tagged hashes of complete
  output-token-ID trajectories so formal exactness cannot alias two tokenizations.

### Evidence status

- All new GPU outcomes remain `UNMEASURED`; this release contains no new
  benchmark number or speed/capacity claim.
- No trusted hardware attester is configured. Self-authored doctor or
  attestation JSON cannot mint `MEASURED`; legacy attestation inputs are
  rejected and industrial analysis remains `UNRESOLVED`/`UNMEASURED`.
- The industrial executor runs only TP1/DP1 Target-only end to end.
  Static/TTS/L0 are `BLOCKED` before mutation because the pinned integration
  lacks `sglang.schema_v3.content_bound_terminal_speculative_evidence.v1`.
- Empirical Stage B is also `BLOCKED` until provider credentials and registered
  hardware are available. DSpark/EAGLE/EAGLE3/NEXTN adaptation and all TP2/DP2
  execution remain blocked independently of hardware access.
- Historical v2 artifacts are regression/debugging inputs only. They do not
  satisfy the schema-v3 registry, capability, trace, telemetry, statistics, or
  attestation contract.
- ChronoBelief registry cells are `BLOCKED` because no authoritative equation
  and source identity are registered; the runtime does not invent or substitute
  an optimizer implementation.
- The CPU `gloo` harness validates collective state-machine behavior only and
  does not attest NCCL, CUDA, two-GPU numerics, or performance.
- The schema-v3 SGLang mail series must pass the complete clean-pin,
  patch-digest, expected-tree, compile/test, and reverse-removal gate before any
  GPU evidence is eligible.

## [0.2.0] - 2026-08-09

### Changed

- Focused the formal runtime and experiment protocol on Static, paper-faithful
  TTS, and first-ready L0.
- Introduced strict schema-v2 identities, DFlash Full/LoRA adaptation, frozen
  historical KV versioning, and cohort-scoped publication.
- Replaced the SGLang integration with a reproducible semantic mail series
  against one exact upstream commit.
- Added deterministic disjoint data windows, tuning-only maximin selection,
  independent confirmation timing, resumable evidence receipts, and a
  content-bound GPU speed gate.
- Rewrote English and Chinese documentation without experimental results or
  performance claims.

GPU status for the 0.2.0 source version was `UNMEASURED`; no release tag was
created.
