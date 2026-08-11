# Configuration

[中文](../zh-CN/configuration.md) · [Home](../../README.md)

## Schema identity

Run configurations use schema version 3, are immutable after validation, and
reject unknown fields. Core methods are `target_only`, `static`, `tts`, and
`l0`. The isolated OnlineSPEC comparison adds
`onlinespec_ogd`, `onlinespec_opt`, and `onlinespec_ens`; it has separate
selection and evidence identities and cannot replace a core method.

Every run binds exact target/drafter revisions, backend algorithm, context and
draft depth, pinned SGLang commit, sampling-profile SHA-256, tenant, and runtime
topology. Target-only requires `speculation_enabled=false`. Every other method
requires speculation, and its verification width must equal draft depth plus
one. Unknown schemas and retired adaptation fields fail before model loading.

These names include target protocol vocabulary, not a promise that every valid
scientific declaration is executable. The current end-to-end industrial
executor accepts only TP1/DP1 Target-only. Static/TTS/L0 are blocked before
mutation because the pinned native begin/reset/finalize hook has no configured
trusted hardware signer. The lower-level adaptive patch path and terminal
schema are limited to TP1/DP1 DFlash, constant schedule, zero extra logical
delay, and `update_round` teacher rows.

## Method and disabled-path contract

Target-only and Static schema configs require both `adaptation: null` and
`online_spec: null`. Target-only launches the target path without speculation;
Static describes native speculative decoding. Neither may allocate optimizer,
gradient, master, candidate, or cohort-adaptation state. Only Target-only is
currently end-to-end executable. Static requires content-bound request,
performance, and aggregate speculative safety evidence, while preserving zero
round/update detailed-trace allocation; it therefore fails the trusted-signer
release preflight.

TTS and L0 require byte-equivalent model, runtime, adaptation, sampling, and
load identities after removing the method field. They use one candidate
implementation and may differ only in publication timing. Exact
full-vocabulary rejection sampling is mandatory; absence of the registered
kernel is an error, never a silent greedy fallback.

## Adaptation object

| Field | Schema-v3 contract |
|---|---|
| `weight_update_mode` | `full` or `lora` |
| `parameter_scope` | `last1`, `last3`, `last5`, `all`; DSpark also permits the three `*_native_heads` scopes |
| `kv_history_policy` | exactly `frozen` |
| `adaptation_scope` | exactly `cohort` |
| `adaptation_group_id` | explicit non-empty cohort identity |
| `rank`, `lora_alpha` | both `null` for Full; same registered rank for LoRA so `alpha/r=1` |
| `lora_matrix_policy` | exactly `registered_matrices_v1` |
| `native_head_policy` | `frozen` for layer-only; `full` for a DSpark hybrid |
| `stride` | positive integer |
| `max_in_flight` | exactly one |
| `canvas_tokens` | equals speculative verification width |
| `teacher_row_policy` | `update_round` or registered `quota_shadow` |

LoRA ranks are exactly 1, 2, 4, 8, 16, 32, and 64. A LoRA plan selects only
registered two-dimensional native matrices and begins with zero functional
delta. Full selects all eligible floating parameters in the named native layer
scope. Borrowed target embeddings, target LM head, and target model remain
frozen. Quantized or unowned trainable coordinates fail preflight.

The trainable-plan digest covers selected and frozen parameters, shapes,
dtypes, parameterization, LoRA rank/alpha, and sharded versus replicated
ownership. Changing any of these values creates a new configuration, memory
plan, selection, and evidence identity.

## Backend-specific fields

This section specifies target contracts. In the current executable schema,
adaptive configurations must use DFlash; DSpark, EAGLE, EAGLE3, and NEXTN are
rejected before model loading. DFlash's target contract uses native
differentiable-canvas evidence. EAGLE/EAGLE3 would require
`speculative_eagle_topk=1` for adaptation and pin one proposal source version.
NEXTN would require a separately preflighted native interface digest; registry
E6 also targets a two-rank memory-fit receipt. Those prerequisites do not make
the cells executable in the current schema.

DSpark layer-only scopes freeze W1, W2, and acceptance/confidence state. Hybrid
scopes `last1_native_heads`, `last3_native_heads`, and `last5_native_heads`
select the named backbone scope while training W1, W2, and the scalar native
acceptance/confidence parameter as Full replicated state. A hybrid requires a
non-null `confidence_loss_weight`; a layer-only plan rejects it.

`verification_mode` is `fixed_budget` only when
`fixed_verification_budget` is present; otherwise it is `native_scheduler`.
The fixed-budget phase is a tuning control, while confirmation uses the native
scheduler. Proposal cross-entropy and the proper confidence loss use the real
native Markov features and actual sampled predecessor. A configuration cannot
substitute reconstructed placeholder features.

The E1a registry contains exactly 56 adaptive configurations: four layer-only
scopes and three hybrid scopes, each crossed with Full plus seven LoRA ranks.

## Optimizer contract

The target TTS/L0 optimizer registry contains `adam`, `adamw`, `sgdm`, `nag`,
`muon`, and `lion`. They make a functional parameter/state proposal and mutate
active state only on commit. Plain `sgd` is reserved for OnlineSPEC.

| Optimizer | Required identity | Resident moment rule |
|---|---|---|
| Adam | learning rate, betas, epsilon | FP32 first and second moments; no decay |
| AdamW | learning rate, betas, epsilon, decay | FP32 first and second moments; decoupled decay |
| SGDm | learning rate, momentum, optional decay | one FP32 momentum; coupled decay |
| NAG | learning rate, momentum, optional decay | one FP32 momentum; coupled decay |
| Lion | learning rate, betas, optional decay | one FP32 moment; decoupled decay |
| Muon | learning rate, momentum, Newton--Schulz steps, auxiliary AdamW fields | matrix momentum plus two auxiliary moments for non-matrices |

Global `grad_clip` is positive and evidence-bound. Unused optimizer fields are
rejected. The target schedule vocabulary includes `constant`,
`inverse_sqrt_published_update`, and `cosine_to_zero`, advancing from
published-update count rather than attempted work. The current adaptive
`RunConfig` accepts only `constant`; the other schedules remain registered but
non-executable. Optimizer and schedule identities enter cohort, plan,
selection, and evidence digests.

## Runtime topology

The target registry/coordinator vocabulary describes one node and at most two
ranks:

| Shape | Required fields |
|---|---|
| TP1/DP1 | `distributed_runtime_capability=single_rank`, no capability receipt |
| TP2/DP1 | `patched_two_gpu_v1`, capability receipt, distinct rank/device identities |
| TP1/DP2 | same capability receipt plus an explicit sticky `router_identity` |

Rank fields must lie inside their TP/DP dimensions. Device, rendezvous, router,
clock, process-group backend, and capability receipt are part of target runtime
identity. The current release accepts only the TP1/DP1 row and rejects all
TP2/DP2 `RunConfig` values before model loading; a caller-authored receipt
cannot enable them.

`process_group_backend=gloo` is valid for the CPU collective contract. It does
not certify NCCL/CUDA behavior and must not be used as a GPU capability receipt.
The CPU contract preserves future receipt vocabulary only. Production TP2/DP2
work remains `BLOCKED` until a new pinned runtime implements and emits the GPU
receipt plus all-rank publication evidence.

Prefill/decode disaggregation and two-batch overlap remain disabled. Multi-node
and more-than-two-rank configurations fail validation; there is no Kubernetes,
elastic, or automatic-failover setting.

Runtime topology and pool capacity are separate identities. The scientific
registry uses two stable logical rank slots. A strict `GpuInventory` may contain
any number of devices; the sole scheduler has explicit 1/2/4/8/16-GPU
regression coverage and freezes physical UUIDs, rank layout, ports, topology,
and whole-instance size in an `IndustrialPhysicalAssignment`. This does not
enable TP2/DP2: release `RunConfig` validation remains TP1/DP1-only.

## HBM and cohort policy

Runtime rendering takes explicit adaptation reserve and model/KV memory
fractions from preflight; there is no universal source default. Admission uses
the least-headroom rank after charging all model, KV, optimizer, candidate,
activation, graph, telemetry, and safety-margin categories.

Every materialized cell also requires an immutable `ExperimentBudget`; short,
p99-anchor, soak, failure, profiler, compile, and download jobs do not share a
hidden duration. Startup/load, compile/prewarm, excluded warm-up and request
pool, scored arrivals, deadline, drain, reset/finalization, evidence close,
retry, tokens, minimum completions, topology, reserved GPU time, and
whole-instance billed time are explicit. The execution copy keeps
`measured_gpu_ms=null`; a separate terminal-bound observation records every
phase and exact deltas. Missing and N/A are distinct from zero.

Fixed cohort slabs are quota-bound per tenant and replica. Optional cold
offload must be configured explicitly and applies only to inactive cohorts.
Memory pressure never silently changes Full to LoRA, precision, optimizer, or
scope. Any such change requires a new config, load screen, selection, and
evidence root.

## OnlineSPEC comparison

The isolated OnlineSPEC protocol adds a required `online_spec` object. Its
optimizer must be plain SGD, and its declarations remain TP1/DP1 under a
separately registered tuning protocol. This separate schema/runtime surface
does not supply a release-trusted hardware signer and does not make an
industrial speculative cell runnable.

| Field | Contract |
|---|---|
| `projection_radius` | optional positive Euclidean radius |
| `additional_learning_rates` | unique increasing expert rates, ensemble only |
| `hedge_learning_rate` | positive meta rate, ensemble only |

OGD and optimistic OGD reject ensemble fields. The ensemble requires at least
two ordered expert rates. Full and LoRA remain distinct decision-coordinate
classes; averaging LoRA factors is not relabelled as averaging dense updates.
OnlineSPEC uses the same registered native layer scopes, frozen historical KV,
exact proposal distribution, cohort isolation, and one-candidate bound, but its
manifest, selection, evidence, and attestation never enter the core gate.
