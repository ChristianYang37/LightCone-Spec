# LightCone-Spec project standards

These rules apply to every source, patch, test, document, and experiment
change. `AGENTS.md` points here so there is one normative list.

## Design

1. Apply Occam's razor. Prefer the smallest abstraction that preserves the
   invariant; reuse one lifecycle, optimizer, publication, and evidence path.
2. Update recipe and publication policy are orthogonal scientific identities.
   Sharing runtime, reconstruction, evidence, or publication machinery never
   implies that methods share configs, evidence rows, candidates, optimizer
   states, or tuning authority. Registered OnlineSPEC learners are important
   comparisons, but use separate tuning, evidence, attestation, and analysis
   and can never influence the core selection or gate.
3. Fail closed before expensive allocation whenever identity, compatibility,
   exactness, memory capacity, or evidence completeness cannot be established.
4. Target-only, Static, and every disabled path must preserve upstream behavior
   and allocate no adaptation, optimizer, gradient, candidate, or adaptation-
   telemetry state.
5. Historical drafter KV is frozen and versioned. Rebuilding or differentiating
   old KV is a new algorithm and requires a separate proposal and protocol.
6. Never replace a missing measurement with zero, copy run-level metrics into
   buckets, or synchronize the headline CUDA hot path to simplify telemetry.

## Scientific method identity

1. TTS uses a frozen, primary-source-bound TTS update recipe and the TTS fixed-
   barrier publication policy. It must never inherit an E1/E2 winner, a schema
   default, or a historical AdamW recipe.
2. L0-naive uses that same frozen TTS recipe with first-ready publication at a
   safe boundary. Recipe identity is shared; live candidate, optimizer, state,
   and evidence identity is not.
3. LightCone candidates use the L0 first-ready publication policy with
   registered E1/E2 recipe candidates. E1/E2 tune only these candidates.
4. Only the tuning-sealed E2 winner may be materialized or reported as
   `LightCone`. An L0-policy run without the exact sealed final-recipe receipt
   is never LightCone.
5. Candidate equivalence is required only for a controlled mechanism replay
   with identical source-state and proposal-evidence digests. Live TTS and
   L0-naive histories may diverge after different publication decisions and
   must not be required to produce byte-identical future candidates.
6. Formal method rows are distinct identities: Target-only, Static, TTS,
   L0-naive, and LightCone. Historical matched-recipe publication-policy runs
   remain diagnostics and cannot establish TTS reproduction, tune LightCone,
   or satisfy formal gates.

## SGLang boundary

1. Do not vendor, submodule, or commit a modified SGLang checkout.
2. All integration changes are semantic mail patches against the exact pinned
   upstream commit and are tested only in disposable patched checkouts.
3. Update patch digests, file lists, expected tree identity, tests, NOTICE, and
   documentation atomically. The complete series is the supported state.
4. Never weaken clean-HEAD, patch-hash, final-tree, or reverse-removal checks to
   make a patch apply.

## Evidence and claims

1. Evidence precedes claims. Local arithmetic, CPU mocks, acceptance changes,
   or an unattested table cannot establish GPU speedup.
2. Formal results require immutable model/runtime/data/sampling identities,
   paired independent timing, registered statistics, complete safety counters,
   and a content-bound GPU attestation.
3. `UNMEASURED` and `BLOCKED` are valid outcomes. Never relabel, omit, or
   optimize away a negative result.
4. Do not commit raw experimental results, result-derived selections, model or
   dataset payloads, telemetry, traces, profiles, provider state, credentials,
   personal paths, or handoff material. A compact public result summary is
   allowed only when the user explicitly requests it and the text identifies
   the immutable code/runtime/data scope, sample size, evidence level, and
   limitations. A preliminary summary must never be relabelled as a formal
   attested result.

## Engineering and review

1. Validate external input and numerical finiteness at the boundary. Reject
   duplicate IDs, stale versions, incomplete coverage, and ambiguous resumes.
2. Destructive cleanup is never a prerequisite for recovery. Completed evidence
   is receipt-bound; interrupted evidence is retained but excluded.
3. Every behavior change needs proportionate CPU/mock coverage and a GPU-marked
   contract where device semantics matter.
4. Before commit: run CPU tests, compileall, Ruff, patch verification, package
   build/install smoke, public-tree checks, link/parity checks, secret/path/size
   scans, and `git diff --check`.
5. Use focused commits, preserve public history, and never force-push `main`.
