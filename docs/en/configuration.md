# Configuration

[中文](../zh-CN/configuration.md) · [Home](../../README.md)

## Schema identity

Run configurations use schema version 3, are immutable after validation, and
reject unknown fields. Runtime methods are `target_only`, `static`, `tts`, and
`l0`. Recipe authority derives five scientific roles: Target-only, Static,
TTS (`tts` plus the frozen TTS authority), L0-naive (`l0` plus that authority),
and LightCone (`l0` plus an exact sealed E2 recipe receipt). An `l0` E1/E2
search declaration is an LC-candidate. The isolated OnlineSPEC comparison adds
`onlinespec_ogd`, `onlinespec_opt`, and `onlinespec_ens`; it has separate
selection and evidence identities and cannot replace a core method.

Every run binds exact target/drafter revisions, backend algorithm, context and
draft depth, pinned SGLang commit, sampling-profile SHA-256, tenant, and runtime
topology. Target-only requires `speculation_enabled=false`. Every other method
requires speculation, and its verification width must equal draft depth plus
one. Unknown schemas and retired adaptation fields fail before model loading.

These names include source capability, not a promise that a declaration is
ready to execute in a particular session. Formal mutation additionally needs
the exact root-authorized deployment/hardware policy, prepared content,
workload, compile, qualification, terminal, and capacity authorities. The
source pins only the public Ed25519 root; it deliberately carries no guessed
hardware allowlist or reusable GPU proof. DFlash, DSpark, and NEXTN adaptive
contracts and the `tp1_dp1`, `tp2_dp1`, and `tp1_dp2` topology vocabulary are
implemented in source, while their hardware-dependent paths remain
`implemented_pending_dynamic_gpu_proof` until the matching fresh proof is
verified. The adaptive runtime supports registered constant,
inverse-square-root-by-published-update, and finite-horizon cosine schedules,
plus non-negative logical publication delay. `quota_shadow` remains
declaration-valid but fails closed without its separately authorized backend
acquisition path.

Every serving run also binds the schema-v2 controlled execution policy. The
registered policy fixes context length 40,960 and server seed 1, disables radix
cache and CUDA Graph, and requires non-incremental output. The target-reference
role disables overlap scheduling; speculative roles retain it. Native
begin/reset/finalize revalidate the same policy, and reset must preserve seed 1
rather than derive a new seed from a run nonce. These values are the
experimental identity exercised by the preliminary GPU snapshot, not a claim
that they are throughput-optimal.

## Method and disabled-path contract

Target-only and Static schema configs require both `adaptation: null` and
`online_spec: null`. Target-only launches the target path without speculation;
Static describes native speculative decoding. Neither may allocate optimizer,
gradient, master, candidate, or cohort-adaptation state. Neither role becomes
executable from configuration alone. In the release-attested lane, Static
requires content-bound request, performance, aggregate speculative safety, and
signed terminal evidence. In trusted `formal_single_operator_v1`, an external
signer is not required, but the same substantive source/content/runtime, fresh
GPU qualification, capacity, terminal, and coverage gates still apply; its
result remains `trusted_single_operator_empirical_no_signature`,
`formal_measured=false`, and `UNMEASURED`.

Recipe and publication identities are orthogonal. TTS uses the frozen,
primary-source-bound recipe with the fixed barrier. L0-naive uses the same
frozen recipe authority with first-ready safe-boundary publication. LC-candidates
and LightCone use the L0 policy but, respectively, a registered E1/E2 search
recipe or the exact sealed E2 winner. Shared reconstruction, candidate
lifecycle, or evidence code does not imply identical live configs, candidates,
optimizer states, or rows. Candidate bytes are compared only in a controlled
replay whose source-state and proposal-evidence digests are identical. Exact
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

Adaptive schema configurations may use DFlash, DSpark, or NEXTN. DFlash uses
native differentiable-canvas evidence. DSpark requires the actual sampled
predecessor, W1/W2 and native confidence head plus the exact 56-candidate
selector. NEXTN requires its MTP hidden/interface, teacher rows, valid mask,
source version, and (for TP2) exact target/drafter shard authority. These are
source contracts; execution still requires their backend-specific GPU
qualification artifacts. Adaptive EAGLE remains unsupported. EAGLE3 is
available only to a post-probe compatible model/selector combination; the
release-attested lane requires a separate signed official decision, while the
trusted single-operator lane retains its unsigned compatibility authority. It
is not a generic adaptation fallback.

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

The E1/E2 **LC-candidate** optimizer registry contains `adam`, `adamw`, `sgdm`,
`nag`, `muon`, `lion`, and project-owned `chronobelief`. They make a functional
parameter/state proposal and mutate active state only on commit. TTS does not
search this registry. The TTS-Cal reconstruction registers Adam, one step,
`(beta1=.9, beta2=.999, epsilon=1e-8)`, zero weight decay, no clipping,
full-drafter/latest-update-round-only updates, request reset, side stream, the
learning-rate grid `1e-7` through `1e-3`, and strides
`{1,5,10,15,20,30,40,50}`. Its pinned DFlash loss uses float32 target-to-draft
forward KL at temperature one, valid-row masking, `exp(-(k-1)/7)` position
weights, and masked weighted normalization. The source-point value correction
is not an independent proximal term, so `lambda` is neither an input nor a
search axis. TTS and L0-naive may not inherit a schema default, historical
AdamW recipe, or E1/E2 selection.
Plain `sgd` is reserved for OnlineSPEC.

The code-owned TTS split is a post-master artifact. New content-verification
producers emit schema 2 and reject the tuning window during the master
ceremony; the window publisher then requires that exact durable master receipt
and emits schema 4 with the receipt SHA-256, verification time, and replay
reservation SHA-256. Schema-1 receipts remain decodable for historical replay,
but cannot authorize a new TTS window.

| Optimizer | Required identity | Resident moment rule |
|---|---|---|
| Adam | learning rate, betas, epsilon | FP32 first and second moments; no decay |
| AdamW | learning rate, betas, epsilon, decay | FP32 first and second moments; decoupled decay |
| SGDm | learning rate, momentum, optional decay | one FP32 momentum; coupled decay |
| NAG | learning rate, momentum, optional decay | one FP32 momentum; coupled decay |
| Lion | learning rate, betas, optional decay | one FP32 moment; decoupled decay |
| Muon | learning rate, momentum, Newton--Schulz steps, auxiliary AdamW fields | matrix momentum plus two auxiliary moments for non-matrices |
| ChronoBelief | learning rate, betas, epsilon, decoupled decay, source/safe-boundary versions | FP32 first and centered second moments; standard bias correction; age derived as exact $d_r$ |

Global `grad_clip` is positive and evidence-bound for recipes that use it; the
TTS reconstruction requires literal no-clip. Unused optimizer fields are
rejected. The schedule vocabulary includes `constant`,
`inverse_sqrt_published_update`, and `cosine_to_zero`, advancing from
published-update count rather than attempted work. Optimizer and schedule
identities enter cohort, plan, selection, and evidence digests. A skipped or
aborted ChronoBelief proposal advances neither moments, update count, nor safe
boundary; the exact GPU parity suite remains a dynamic pre-execution gate.

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
identity. Configuration validation accepts a distributed row only with the
exact source-owned `patched_two_gpu_v1` capability identity and a runtime
receipt claim. Formal dispatch then deep-verifies the corresponding dynamic
GPU proof; a caller-authored receipt or the source CPU capability alone cannot
enable execution.

`process_group_backend=gloo` is valid for the CPU collective contract. It does
not certify NCCL/CUDA behavior and must not be used as a GPU capability receipt.
The CPU contract exercises state transitions only. Production TP2/DP2 remains
`BLOCKED` until the pinned runtime emits and local control verifies the exact
GPU qualification, all-rank publication, and terminal evidence.

Prefill/decode disaggregation and two-batch overlap remain disabled. Multi-node
and more-than-two-rank configurations fail validation; there is no Kubernetes,
elastic, or automatic-failover setting.

Runtime topology and pool capacity are separate identities. The scientific
registry uses two stable logical rank slots. A strict `GpuInventory` may contain
any number of devices; the sole scheduler has explicit 1/2/4/8/16-GPU
regression coverage and freezes physical UUIDs, rank layout, ports, topology,
and whole-instance size in an `IndustrialPhysicalAssignment`. This does not
enable TP2/DP2: the exact dynamic qualification remains mandatory.

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
does not supply a release-trusted hardware signer and therefore cannot promote
an OnlineSPEC result to release-level `MEASURED`. The trusted single-operator
E0 comparison may run without that signer only after its audited source
authority and exact content/runtime, GPU qualification, capacity, terminal, and
coverage gates pass; it remains
`trusted_single_operator_empirical_no_signature`, `formal_measured=false`, and
`UNMEASURED`.

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
