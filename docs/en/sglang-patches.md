# SGLang patch workflow

[中文](../zh-CN/sglang-patches.md) · [Home](../../README.md)

## Source boundary

SGLang is an external Apache-2.0 project. LightCone-Spec pins upstream commit
`3312645a307453893a00778592f105581e3d1c3d` and distributes only semantic
mail patches. The repository contains no SGLang source, submodule, or modified
checkout. A local `sglang/` directory is ignored and is never an integration
identity.

Only the complete ordered series is supported. Patch count, filenames,
SHA-256 values, modified-file inventory, upstream identity, and expected final
tree are read from `patches/sglang/series` and
`patches/sglang/manifest.json`; documentation does not duplicate them as an
independent authority.

## Schema-v3 target and verified patch surface

The schema-v3 envelope defines one coherent **target** runtime surface. Items
in this list are contracts and registry vocabulary; they are not claims that
the current patch implements every item:

1. strict Target-only/Static/TTS/L0 and backend-native configuration with a
   disabled path that allocates no adaptation state;
2. common proposal evidence, backend payload validation, differentiable
   reconstruction, and exact sampling-distribution preservation;
3. Full/LoRA native layer plans, DSpark W1/W2/acceptance hybrids, functional
   optimizer candidates, source/buffer/optimizer generations, and fixed-address
   publication;
4. DFlash, DSpark, EAGLE/EAGLE3, and NEXTN native proposal-hook contracts without
   double-applying an adapter;
5. one-node TP2 and sticky DP2 identity/ownership, all-rank prepare/decision/
   receipt publication, and fail-closed process-group handling;
6. least-rank HBM admission, fixed cohort slabs, bounded device/event telemetry,
   durable Parquet WAL evidence, lifecycle cleanup, and focused regressions;
7. OnlineSPEC comparison hooks folded into the same version/exactness/evidence
   infrastructure while remaining gate-isolated.

Intermediate patch states are review boundaries, not product variants. An
addition that depends on an earlier patch must remain in the complete series;
the verifier never skips a patch to make a partial combination apply.

The current pinned patch implements and tests strict schema-v3 parsing,
allocation-free Target-only/Static, and TP1/DP1 DFlash native-layer Full/LoRA
adaptation with fixed-address device-predicated publication. That is a
lower-level patched-server surface, not end-to-end industrial executor support.
It also implements a CPU-only DSpark native contract: adapter-free backbone
reconstruction applies one candidate delta, actual sampled predecessors select
real Markov W1 embeddings, layer-only scopes freeze W1/W2/acceptance state,
hybrid scopes train those heads only in Full mode, and the composite objective
uses exact proposal/teacher distributions with a stop-gradient `1-TV`
confidence target. Fixed-budget decisions are exact total-token budgets while
native-scheduler decisions remain authoritative. This contract is not wired to
the DSpark worker/CUDA publication path and leaves its runtime capability gate
closed.
It also makes the official SGLang serving benchmark expose the server-provided,
ordered `output_ids` for both cumulative/incremental streaming and non-streaming
responses. Those IDs are never reconstructed by retokenizing generated text;
missing, discontinuous, or rewritten trajectories fail the claim-grade
exactness gate. The benchmark client now also accepts a caller-owned async HTTP
session used by submit and abort, so one server session can reuse a registered
connection pool without duplicating the official request parser.

The patch implements the content-bound
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v1` capability
and begin/reset/finalize endpoints. The exact latest-patch SHA-256 is
`05ab7ae2074f2e9ffa2387f1897e85ea1527a6daf44e1527dfd908adfb547f12`
and the final tree is `fae3c1538ed4934fb3b47c7ebc82393306c43f06`; the manifest remains the
authority. The lifecycle binds run/nonce/plan/rank, process/session/reset
lineage, expected request IDs, exact ordered token IDs, terminal coverage,
Static aggregate safety, and TTS/L0 request/round/update/KV/performance rows.
It exposes a signer plugin boundary but bundles no trusted hardware key or
release signer. The executor therefore still runs only Target-only end to end
and blocks Static/TTS/L0 before mutation.

The second patch exposes an opt-in native committed-token observation
contract. Each timestamp is sampled by the CPU while the streamer enumerates
an already-committed final token prefix after stop handling; it is not a decode
production time or CUDA event. The producer is explicitly
`CPU_CONTRACT_ONLY`, the formal release allowlist remains empty, and these
events cannot support E2 p99 ITL. Coalesced SSE chunks are no longer divided
into invented equal ITLs. The benchmark now reports expected, observed,
coalesced, and missing client intervals plus exact coverage. Any incomplete
coverage makes every aggregate ITL statistic, including p99, explicitly
`UNSUPPORTED`/`null`; a sparse subset is never promoted to a distribution.

The executor also rejects the following before model loading because their
execution contracts are not implemented:

- DSpark, EAGLE, EAGLE3, and NEXTN adaptation;
- every TP2 or DP2 run;
- quota-shadow teacher acquisition when an update round lacks native
  supervision (the fixed ledger records and capability-blocks that need);
- DSpark worker/CUDA composite-head training and NEXTN training interfaces.

TP1/DP1 DFlash TTS/L0 now execute constant, inverse-square-root-by-published-
update, and finite-horizon cosine schedules. They also record intrinsic
readiness and apply non-negative logical delay without changing the TTS
fixed-boundary versus L0 first-ready publication distinction. These CPU/native
contracts do not claim CUDA performance evidence.

The same patch now provides the first-party
`sglang.schema_v3.source_owned_all_reset_session.v1` producer and admin
endpoints for capability, initial-state, and reset receipts. It requires a
drained scheduler and cleared KV/prefix state, restores DFlash adaptation via
the native cache-reset path, and binds RNG/counters, scheduler/telemetry,
weights/master/moments/candidate/cohort/update state, allocator/HBM, and the
completion event. The third patch adds source-owned HTTP accounting at the
protocol boundary of the supported single-tokenizer HTTP/1.1 uvicorn paths:
actual `connection_made`/`connection_lost` events produce cumulative process/
generation/created/closed/current snapshots that the HTTP process injects and
the producer checks for conservation and monotonic continuity. Granian HTTP/2
and multiple-tokenizer HTTP-process paths fail closed before producing this
capability. The accumulator is initialized explicitly at the actual HTTP
serving-process startup boundary: duplicate initialization in one PID is
rejected, while a forked child replaces inherited ownership with its own
generation. Request counters, headers, and caller payloads cannot substitute.
The initial-state endpoint finalizes exactly once and content-binds its own
post-capability snapshot; it cannot return a stale provisional snapshot. The
generic terminal-evidence route rejects all reserved session actions before
manager dispatch, so only the dedicated HTTP wrappers can inject accounting.
GPU reset semantics remain `PENDING`; CPU-valid receipts do not authorize GPU
reuse, and every failed transition requires a fresh process. This patch closes
the reset-state accounting slice. The fourth patch registers the exact ordered
trace members, binds each reset to the existing native terminal begin/reset/
finalize lifecycle, derives excluded-warm-up and independent scored-clock
receipts from scheduler-owned state, and seals the complete native digest chain
with a lifecycle-terminal receipt. The close receipt explicitly reports
`transport_close_pending=true`: the HTTP response that carries it cannot
truthfully have observed its own `connection_lost` yet. The caller must still
close the pool and terminate the process. Durable live-process reuse, formal
GPU reset semantics, and GPU validation remain absent and `BLOCKED`.

The fifth patch adds the source-owned
`sglang.schema_v3.source_owned_compile_cache_lifecycle.v1` producer and its
dedicated begin/finalize endpoints. The scheduler binds the exact plan, cache
key, model lock, sampling profile, prewarm manifest, physical assignment,
budget, inventory, patched tree, process, drained boundaries, and ordered
request terminals. The host consumer reuses the same pinned official bench
pool; a submitter response is never accepted as evidence. This is strictly a
`CPU_CONTRACT_ONLY`. JIT time, cache hit/miss/write counts, and CUDA Graph
capture/replay counts remain `null` with
`gpu_compile_semantics_unavailable`; the formal plan/source allowlists and the
independent GPU-vetted source registry remain empty, so COMPILE stays
`BLOCKED` before mutation.

Those rows remain `BLOCKED`, not simulated through DFlash or reported as
`UNMEASURED` runnable work. The lower-level DFlash implementation and valid
terminal envelope cannot be promoted to claimable evidence without an
allowlisted out-of-band signer bound to the pinned tree and challenge. No CUDA
graph, multi-GPU, speed, capacity, or other GPU result is reported by this
release.

## Application

`patches/sglang/apply.sh` accepts only a clean checkout at the exact upstream
HEAD. It checks the registered patch digests, applies in order with `git am`,
and compares the resulting Git tree with the manifest:

```bash
patches/sglang/apply.sh /path/to/clean-sglang
```

Dirty state, a wrong commit, modified patch bytes, mail-application failure, or
final-tree mismatch stops immediately. The script does not stash, reset,
rebase, or edit the supplied checkout to hide a mismatch.

## Verification and authoring

Use a separate clean upstream checkout:

```bash
python scripts/verify_sglang_patchset.py \
  --upstream-checkout /path/to/clean-upstream
```

The verifier must apply the full series in a disposable clone, confirm the
expected tree and modified-file inventory, compile changed Python, run the
requested focused tests, reverse the series, and prove the caller's upstream
checkout stayed clean. The focused adaptation-protocol tests load only the
patched config, runtime, parameter-plan, and DSpark CPU-contract modules through a stub package;
patch-integrity CI therefore does not depend on SGLang's unrelated optional
top-level serving packages.

New integration work is patch-first: branch temporarily from the pin, make a
focused semantic commit with tests, export through `git format-patch`, and
update series order, patch digests, modified files, expected final tree, Python
pin constants, NOTICE, and EN/zh documentation atomically. Never edit or
commit a patched checkout.

## Current evidence gate

The final schema-v3 patch has passed the repository's complete
apply/compile/focused-test/reverse verifier. That is a patch-integrity result,
not GPU validation. A CPU package test or an older patched-tree receipt is
insufficient. TP1/DP1 DFlash and its strict native terminal lifecycle are
implemented, but the release trust policy has no configured signer;
Static/TTS/L0 industrial cells are therefore `BLOCKED` before mutation rather
than runnable `UNMEASURED` work. A test signer, provider object attribute, or
caller-supplied verifier cannot unlock that policy. The DSpark CPU contract
does not authorize execution. DSpark/EAGLE/EAGLE3/NEXTN adaptive cells and every TP2/DP2 cell remain
`BLOCKED`. Target-only is the only release-executable path.

Historical v2 evidence remains useful for regression comparison only. It does
not carry the new Target-only, backend-plan, topology, registry, trace,
statistics, or telemetry identities and cannot be upgraded by changing a
label.
