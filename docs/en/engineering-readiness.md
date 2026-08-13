# Engineering readiness and acceptance matrix

[中文](../zh-CN/engineering-readiness.md) · [Home](../../README.md)

This page maps the two industrial engineering specifications to the current
source boundary. It is an implementation/readiness ledger, not an experiment
result. GPU measurements, host addresses, credentials, model/data payloads,
provider state, and result-derived selections remain outside the repository.

## Status model

The three release labels are deliberately narrow:

| Label | Meaning | What it does not mean |
|---|---|---|
| `CPU_READY` | The source, schema, deterministic reducer/scheduler, failure semantics, and CPU/mock receipt contract exist and pass their applicable non-GPU gates. | No CUDA/NCCL behavior, speed, memory capacity, or formal execution claim. |
| `GPU_SMOKE_READY` | The source path is eligible for a bounded device check once the named external inputs are present. | The smoke has not necessarily run; it is never a benchmark or `MEASURED` evidence. |
| `MEASURED` | A registered run completed with content-bound terminal evidence, immutable inputs, hardware/interference coverage, and the release-owned trusted-attester chain. | A local mock, diagnostic signer, historical v2 artifact, or positive arithmetic cannot produce this label. |

`BLOCKED`, `UNRESOLVED`, `UNDERPOWERED`, `INVALIDATED`, and `N/A` remain valid
terminal dispositions. A component may be `CPU_READY` while its formal GPU path
is `BLOCKED`; those facts are recorded separately below.

## Current release boundary

| Surface | Source/control status | Next executable gate |
|---|---|---|
| Single-host arbitrary-N inventory and independent work | `CPU_READY`; the same-host scheduler has explicit 1/2/4/8/16-GPU contract coverage. | Collect a content-bound inventory and run a bounded host smoke. |
| Multi-host independent cells | `CPU_READY`; fleet composition, host-local namespaces, deterministic balancing, bounded transport concurrency, failure isolation, and unknown-outcome reconciliation are implemented. | Exercise the Python coordinator over pinned SSH routes; no public coordinator CLI exists. |
| Same-host gang placement | `CPU_READY` as a scheduler/state-machine contract. | A real TP2/DP2 launch remains blocked by the runtime/source capability below. |
| Cross-host TP/DP collective | `BLOCKED` with `cross_host_collectives_unvalidated`. | A separate source release plus real multi-host NCCL/rendezvous validation. |
| TP1/DP1 Target-only | `GPU_SMOKE_READY` after exact model/runtime/data locks and host preparation. | Formal evidence still needs the release attester and registered DAG closure. |
| TP1/DP1 DFlash Static/TTS/L0 | Lower-level source lifecycle is `CPU_READY`; release execution is `BLOCKED` on trust and device validation. | Pinned-tree GPU smoke, exact terminal evidence, and a reviewed source-release attester anchor. |
| DSpark adaptation | Proposal/reconstruction/selector/loss/scheduler-mode contract is `CPU_READY`; the worker/CUDA candidate path is `BLOCKED`. | Implement and patch the native worker update/publication path, then GPU-validate it. |
| EAGLE/EAGLE3/NEXTN adaptation | Target schemas and strict compatibility guards only; `BLOCKED`. | Exact pinned upstream training interfaces and a semantic SGLang patch. |
| TP2/DP2 adaptation/publication | CPU/gloo coordinator contract only; formal source path is `BLOCKED` before model load. | Vocab-parallel loss, real all-rank producer/publication, native routing, and same-host GPU/NCCL validation. |
| Server-session reuse | Source-owned lifecycle and incremental evidence are `CPU_READY`, but this release authorizes no live reuse. | Use the clean-process fallback; a later release would additionally require device reset, durable close, and whole-inventory continuity. |
| Trusted attestation | Anchored external public-bundle loader is `CPU_READY`; private material is forbidden. The source-release anchor is unset. | Provision and review a source-owned anchor, live signer, nonce replay store, and allowed hardware envelope. |
| Formal industrial results | `UNMEASURED`. | Every code, workload, hardware, evidence, power, and attestation gate must close first. |

## Native-system specification acceptance

This matrix follows the major numbered requirements in the first engineering
specification. “Accepted” means the smallest truthful source/CPU contract is
present; it never upgrades a missing device path.

| Spec item | Acceptance in this source | Evidence boundary / remaining work |
|---|---|---|
| §0 audit, preserve, reconcile | Accepted under the later user instruction to continue in the existing industrial worktree/branch; unrelated work is preserved and result artifacts remain external. | The local handoff records Git/test identities; public source contains no host/result material. |
| §0.1 isolated branch/worktree | The original instruction to create another isolated branch/worktree was superseded by the later user direction to continue on the already isolated `codex/lightcone-industrial-20260811` worktree. No additional branch was created. | G0 stops on this branch; `main` remains unchanged and no push or automatic merge is implied. |
| §0.2 scientific/service invariants | The source/CPU contracts preserve Target-only, zero-adaptation Static, one TTS/L0 candidate with publication-only differences, frozen historical KV, fail-closed publication, explicit memory accounting, and missing-as-missing evidence. | CUDA exactness, service behavior, and performance remain `UNMEASURED`; CPU acceptance cannot authorize a positive GPU claim. |
| §1 common backend contract | Accepted for common proposal evidence, trainable plan, candidate identity, reconstruction, and strict backend payload validation. | DFlash has the lower-level adaptive path; DSpark is CPU-only; other backends remain guarded target contracts. |
| §2 GPU-resident update/publication | Accepted at the pinned DFlash source-contract level, including one candidate lifecycle, fixed-address publication, events, and exact termination semantics. | CUDA graphs, stream priority, pointers, and timing require GPU smoke; no device result is claimed. |
| §3 Full/LoRA and native selectors | Accepted for the rank grid, zero functional initialization, native scopes, exact memory-plan accounting, and the 56-cell DSpark selector. | DSpark composite-head runtime execution remains blocked. |
| §4 TP and replica-local DP | Accepted as topology, sharded/replicated ownership, sticky cohort, atomic publication, failure, and CPU/gloo contracts. | TP2/DP2 runtime remains blocked; cross-host collective is explicitly unsupported. |
| §5 HBM/cohort governance | Accepted for one ledger, least-feasible-rank admission, eviction order, bounded slabs, quotas, generations, and explicit cold offload. | Verify real allocator/HBM receipts and pressure behavior on the target GPU. |
| §6 production request/failure path | Accepted for bounded admission, terminal states, timeouts/cancellation, clean failure evidence, and fail-closed release preflight. | Only Target-only is currently eligible for end-to-end GPU smoke; speculative and multi-rank gates remain above. |
| §7 telemetry/profiling/useful work | Accepted for bounded writer/WAL durability, exact coverage, budget observations, safety counters, and isolated profiler contracts. | Device timing, power, HBM, and collective telemetry remain unmeasured. |
| §8 CLI/identity/CI/release | Accepted for strict CLI inputs, content identities, semantic patch verification, packaging/public-tree gates, and categorical blocked outcomes. | The final local gate transcript belongs in the uncommitted handoff, not in public docs. |
| §9 declarative registry | Accepted with one registry/DAG, immutable cells, reducer-owned activation, exact dependency receipts, and one executor/evidence path. | Declarations do not authorize blocked methods. |
| §10 load/corpus | Accepted as immutable open/closed-loop trace contracts, request accounting, synthetic labels, and external data locks. | Exact formal dataset revisions and payloads are external gates. |
| §11 registered E3a–E0 cells | Accepted as truthful registry, activation, budget, power, and completion contracts. | No formal timing has started; DSpark, NEXTN, TP2/DP2, compile, and download cells retain their named blockers. |
| §12 analysis/power/evidence | Accepted for four excluded pilots, family-local 12–20-block power, Holm/BH, hierarchical/time-block bootstrap, evidence dependence, and no missing-as-zero. | Formal analysis requires attested GPU terminal inputs. |
| §13 CPU-ready stage | Source implementation is `CPU_READY`; GPU-marked tests are collected but not executed in this phase. | Use the final handoff for the exact gate transcript and package identities. |
| §14 no-card SSH preparation | Coordinator/worker protocol is `CPU_READY`. | Actual SSH routing, patched checkout, caches, data locks, and public trust bundle are external and uncommitted. |
| §15 provider and GPU smoke | `GPU_SMOKE_READY` only for the bounded paths named above; no smoke was run in this phase. | Requires the user-provided host and immutable external inputs. |
| §16 formal two-GPU schedule | `BLOCKED`; no formal experiment has started. | TP2/DP2 source capability, signer, smoke, interference, and all registered authorities must pass first. |
| D1 single-GPU baseline/tuning | Scheduling, activation, halving, Pareto, and budget contracts are `CPU_READY`; execution is `UNMEASURED`. | Requires Target-only/Static device evidence, per-host interference calibration, immutable inputs, and formal authority before any E3a/E1/E2 run. |
| D2 mechanism/confirmation | Profiler isolation and family-level four-pilot/12–20-block power contracts are `CPU_READY`; execution is `UNMEASURED`. | E4/E3b and DSpark confirmation remain blocked by selected-recipe device evidence and, for DSpark, the missing native worker path. |
| D3 production single-GPU load | Registered trace, request-accounting, lambda/p99, soak, and paired-block contracts are `CPU_READY`; no production trace was run. | Requires exact replay inputs, at least 10,000 completions at a preregistered p99 anchor, terminal evidence, and release attestation. |
| D4 two-GPU topology/scale | Placement/state-machine vocabulary is `CPU_READY`, while real TP2/DP2 and E6 two-rank execution are `BLOCKED`. | Requires a later patched-runtime capability plus same-host GPU/NCCL validation; cross-host collectives remain fail-closed. |
| D5 breadth last | Frozen-recipe ordering, sequential staging, alias, and OnlineSPEC-isolation contracts exist; execution is `UNMEASURED`/`N/A` by backend. | E0 remains last and blocked until recipes/interfaces are frozen; absent exact model/backend interfaces stay `N/A`. |
| §17 return package | Code/docs/contracts are deliverables; numerical results remain `UNMEASURED`. | Final SHAs and local gate output go to the local handoff; immutable evidence stays external. |

## Optimization specification acceptance

This matrix follows the second, optimization and pre-experiment specification.

| Spec item | Acceptance in this source | Evidence boundary / remaining work |
|---|---|---|
| §0 starting point and isolation | Reconciled with the later instruction to continue directly on the existing industrial branch and stop after G0 stabilization. | No new branch, destructive cleanup, result import, or automatic merge is implied by this document. |
| §0.1 current capability boundary | Re-audited against the current source rather than the prompt's historical counts and identities; the current release boundary is recorded above. Only TP1/DP1 Target-only is eligible for an end-to-end bounded GPU smoke. | Static/TTS/L0 release execution, DSpark, other adaptive backends, and TP2/DP2 remain explicitly `BLOCKED`; there is no trusted source-release anchor or `MEASURED` result. |
| §0.2 scientific/optimization invariants | Immutable cells, paired TTS/L0 semantics, clean independent starts, excluded pilots, fixed-duration traces, p99 minimums, explicit GPU-hours, and external-only artifacts are enforced as source/CPU contracts. | Session reuse is unauthorized; every empirical value remains `UNMEASURED` until exact device and attestation evidence exists. |
| §1 first-class budgets | Accepted: immutable per-cell budgets distinguish startup, compile, warm-up, scored time, deadlines, drain, reset, evidence, retries, special jobs, requests/tokens/p99, and compute/reserved/billed GPU time. | Estimates remain planning artifacts; actual deltas require terminal observations. |
| §2 E1 activation/E2 halving | Accepted: E1 materializes one 130-cell slice; E2 is sealed quarter-retention, round-by-round, pair-preserving, and adversarially validated. | Real survivor evidence is not in the repository. |
| §3 family power | Accepted: exactly four excluded pilots select a sealed 12–20 final prefix or `UNDERPOWERED`, independently per family. | Confirmation stays unavailable until pilot terminal evidence exists. |
| §4 evidence alias | Accepted: aliases are content-bound, equivalence-checked, and dependence-preserving; non-singleton formal use fails closed without recomputation. | An alias never creates a new observation or GPU saving claim by itself. |
| §5 clean server sessions | Accepted as a source lifecycle/audit contract plus deterministic fresh-process fallback; its receipts cannot authorize reuse in this release. | A later live-reuse release would require complete device reset/close/whole-inventory evidence. |
| §6 native terminal/session evidence | Accepted for source-owned begin/reset/finalize identity, incremental durable steps, terminal coverage, and incomplete-session retention. | Source-release signer is absent; device reset semantics still require smoke. |
| §7 HTTP/writer/compile cache | Accepted for the supported HTTP/1.1 lifecycle counters, one caller-owned pool per execution, bounded durable writer, immutable cache bases, private overlays, and source-owned compile lifecycle. | Unsupported HTTP paths fail closed; compile GPU prewarm/finalization must be validated on the target host. |
| §8 arbitrary GPU-pool scheduler | Accepted for arbitrary same-host inventories and explicit 1/2/4/8/16 tests; fleet composition adds independent multi-host scaling without changing placement semantics. | Cross-host collectives remain blocked. |
| §9 code/release gates | Applicable non-GPU checks are required for G0 and recorded in the local handoff before any later merge decision. | GPU/integration/system tests remain queued for SSH smoke. |
| §10 exact dry-run plan | Reducers, budgets, assignments, and bundle materialization are implemented. | Exact model/data/provider/inventory inputs must be frozen externally before a real dry run. |
| §11 no-card/provider preparation | Software path is prepared; credentials, routes, addresses, provider state, and model payloads are intentionally absent. | External operator action after SSH handoff. |
| §12 GPU smoke/session equivalence | Not run. The order and fail-closed expectations are specified below. | Must not be relabelled as formal evidence. |
| §13 optimized DAG execution | Not started; formal status is `UNMEASURED`. | Execute only after all source, smoke, budget, interference, and trust gates close. |
| §13.1 E3a baseline/capacity | Activation, balancing, width-selection, and capacity-envelope contracts are `CPU_READY`; no E3a cell ran. | Target-only/Static native evidence, exact external inputs, and per-host calibration must pass before materializing the 130-cell E1 slice. |
| §13.2 E1/E2 tuning | Reducer-owned E1 activation and pair-preserving quarter-retention E2 halving are `CPU_READY`; execution is `UNMEASURED`. | Device-capable DFlash, exact receipts, and formal authority are required; later-stage losers remain unmaterialized. |
| §13.3 E4 mechanism/profilers | Isolation, compile-cache identity, private-overlay, and profiler accounting contracts are `CPU_READY`; no profiler or headline timing ran. | Requires the frozen selected recipe and device validation; profiler timing can never become headline evidence. |
| §13.4 E3b pilot/power/confirmation | Four excluded pilots, family-local power, whole-block scheduling, legal aliases, and per-host interference limits are `CPU_READY`; confirmation is unavailable. | Pilot terminal evidence must seal 12–20 blocks before unblinding; no early favorable stopping is permitted. |
| §13.5 E1a/E5 production | Registry, trace, lambda/p99, paired-arrival, soak, and gang-scheduling contracts are `CPU_READY`; DSpark and TP2/DP2 execution are `BLOCKED`. | Requires the native DSpark path, frozen anchors, exact replay evidence, same-host capability, and release attestation. |
| §13.6 E6/E0 | Compatibility-first gating, frozen-recipe transfer/breadth order, exact aliases, and OnlineSPEC isolation are represented; execution is `UNMEASURED`/`N/A`. | Missing exact model/backend interfaces remain `N/A`; no download or transfer run begins before those gates pass. |
| §14 continuous budget/evidence control | Accepted as observation receipts, receipt-only resume, immutable attempts, cost accounting, and completion sealing. | Populated receipts arise only from real external runs. |
| §15 completion/return | Public source contains no results or secrets; local handoff carries operational status. | A future result package must remain content-bound and evidence-first. |

## SSH-to-experiment runbook

Run this sequence after the operator supplies SSH access. Stop at the first
`BLOCKED`, identity mismatch, missing external authority, or failed smoke.

1. Run `doctor` on every host and bind the repository commit, patched SGLang
   tree, driver/runtime/toolchain, storage, clocks/power, and background state.
2. On every host, create a separate nonce-bound `GpuInventory` and that host's
   serial/calibrated `InterferenceEnvelope`; do not copy an envelope between
   nominally identical machines.
3. Assemble the fleet with paired repeated `--inventory` and
   `--interference-envelope` options. Confirm unique host IDs/GPU UUIDs and
   collision-free port, cache, evidence, and manifest namespaces within each
   host; literal values may repeat on different hosts.
4. Re-run patch apply/tree/file-list/compile/focused-test/reverse verification
   and runtime-manifest checks in a disposable checkout.
5. Run bounded device, terminal, session, reset, compile-cache, HTTP, writer,
   and evidence smokes. This release must still select the fresh-process
   fallback; a complete CPU audit receipt must not be treated as reuse authority.
6. Run the single-GPU Target-only smoke first. Run Static/DFlash TTS/L0 only as
   diagnostics when their exact capability is present; without the reviewed
   source-release signer they remain non-formal and `BLOCKED` for the DAG.
7. Exercise same-host TP2/DP2 preflight and diagnostic contracts. The current
   source must reject real launch before model load; do not attempt a formal
   two-GPU run until a later source capability and same-host NCCL smoke exist.
8. Exercise the Python fleet coordinator over SSH. The remote worker is
   `execute-dispatch-wave --host-request-stdin`; preserve completed host
   receipts, reconcile every unknown outcome through the independent same-host
   receipt/evidence fetch, and retry only terminal-negative work on that host
   under a new attempt.
9. Calibrate interference separately per host. Only the exact passing
   cardinality may enable concurrent headline work; profiler, download,
   compile, and shared-I/O domains remain exclusive.
10. Load the externally anchored public attester bundle, nonce replay store,
    immutable model/data/trace locks, and registered hardware envelope. Private
    keys never enter the repository, argv, artifacts, or logs.
11. Materialize and execute the formal DAG only when every preceding gate is
    satisfied. Otherwise seal the exact `BLOCKED`/`UNMEASURED` disposition and
    retain interrupted evidence without admitting it to analysis.

## Completion rule

G0 is complete when the target branch is clean at the handed-off SHA, `main`
remains intentionally unchanged, all applicable non-GPU/package/public-tree
gates pass, and every unrun device/formal surface has an explicit contract and
blocker.
The empirical phase is complete only after immutable external evidence closes
the registered DAG. Until then, the truthful project-level result is
`UNMEASURED`, regardless of code coverage or smoke readiness.
