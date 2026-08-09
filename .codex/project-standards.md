# LightCone-Spec project standards

These rules apply to every source, patch, test, document, and experiment
change. `AGENTS.md` points here so there is one normative list.

## Design

1. Apply Occam's razor. Prefer the smallest abstraction that preserves the
   invariant; reuse one lifecycle, optimizer, publication, and evidence path.
2. The formal research surface is Static, TTS, and L0. TTS and L0 share one
   candidate implementation and differ only in publication timing.
3. Fail closed before expensive allocation whenever identity, compatibility,
   exactness, memory capacity, or evidence completeness cannot be established.
4. Static and every disabled path must preserve upstream behavior and allocate
   no adaptation, optimizer, gradient, candidate, or trace state.
5. Historical drafter KV is frozen and versioned. Rebuilding or differentiating
   old KV is a new algorithm and requires a separate proposal and protocol.
6. Never replace a missing measurement with zero, copy run-level metrics into
   buckets, or synchronize the headline CUDA hot path to simplify telemetry.

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
4. Do not commit experimental results, selections derived from results, model
   or dataset payloads, telemetry, traces, profiles, provider state, credentials,
   personal paths, or handoff material.

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
