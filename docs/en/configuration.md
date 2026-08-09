# Configuration

[中文](../zh-CN/configuration.md) · [Home](../../README.md)

## Schema identity

All run configurations use schema version 2 and reject unknown fields. There
are three core methods: `static`, `tts`, and `naive_async`. The registered
OnlineSPEC comparison adds `onlinespec_ogd`, `onlinespec_opt`, and
`onlinespec_ens` under a separate manifest, selection, and evidence identity.
They are not valid substitutes in the core speed study and cannot affect its
gate.

Every real run binds immutable target and drafter revisions, the pinned SGLang
commit, a sampling-profile digest, a tenant, and a runtime load. Old schemas or
retired method names fail as unknown input before model loading.

## Adaptation configuration

TTS and L0 share one identical adaptation object:

| Field | Allowed contract |
|---|---|
| `weight_update_mode` | `residual`, `lora`, or `full` |
| `parameter_scope` | `tail` or `drafter` |
| `kv_history_policy` | exactly `frozen` |
| `adaptation_scope` | exactly `cohort` |
| `adaptation_group_id` | explicit, non-empty group identity |
| `optimizer.name` | `adam`, `adamw`, `sgdm`, `nag`, `muon`, or `lion` |
| `rank` | explicit for residual/LoRA; `null` for full |
| `stride` | positive integer |
| `max_in_flight` | exactly one |

Residual is tail-only. Drafter Full/LoRA is DFlash-only and requires an
unquantized TP=DP=1 path with a canvas width equal to the speculative block
size. DSpark and EAGLE/EAGLE3 accept only tail scope. Adapted DSpark requires
verify-all execution. Adapted EAGLE/EAGLE3 requires one layer, fixed proposal
depth, top-k one, no token-map remapping, and exact full-vocabulary rejection
sampling. Target embedding, target LM head, and target model remain frozen.

Static requires `adaptation: null`. This is a semantic fast-path requirement:
no optimizer, gradient, trace, candidate, or adaptation-reserve allocation may
be created.

The formal launcher explicitly adds `--speculative-speed-study-metrics` and
exact rejection sampling to all three endpoints. Without that study flag,
native SGLang allocates no LightCone metric state. With it, DFlash, rejection
sampling, and both acceptance thresholds equal to one are mandatory; a
missing exact kernel is an error and never falls back to greedy decoding.

## Optimizer contracts

All online optimizers are functional: a side stream computes candidate
parameters and candidate state without mutating the active optimizer. The
candidate becomes active only when the TTS or L0 publication policy commits it.
The configuration is strict rather than accepting silently unused fields:

| Optimizer | Required fields | Decay and resident state |
|---|---|---|
| `adam` | learning rate, betas, epsilon | no weight decay; FP32 first and second moments |
| `adamw` | learning rate, betas, epsilon | decoupled decay; FP32 first and second moments |
| `sgdm` | learning rate, `momentum` | coupled weight decay; one FP32 momentum buffer |
| `nag` | learning rate, `momentum` | PyTorch-style Nesterov with coupled weight decay; one FP32 momentum buffer |
| `lion` | learning rate, betas | decoupled decay; one FP32 momentum buffer |
| `muon` | learning rate, `momentum`, `muon_ns_steps`, auxiliary AdamW learning rate and decay | Muon for 2-D tensors; auxiliary AdamW and two moments for non-matrix tensors |

`grad_clip` is mandatory and global across the candidate parameter list.
`momentum` is rejected for optimizers that do not use it. Muon-only fields are
rejected for every other optimizer. Adam's unused weight decay and Lion's
unused epsilon variants are likewise rejected instead of forming fake tuning
identities. Static uses no optimizer. Plain `sgd` is reserved for the registered
OnlineSPEC learners and is not a TTS/L0 optimizer choice.

The HBM ledger counts the FP32 master, every allocated moment, and the device
step scalar. Empty moments are not allocated: SGDm, NAG, and Lion therefore do
not pay for a second state tensor, while Muon pays two moments only for the
non-matrix parameters handled by auxiliary AdamW.

## Cohort identity

An update may be shared only by requests with identical target and drafter
revisions, algorithm, sampling profile, tenant, experiment group, parameter
layout, and optimizer configuration. Each active request contributes only its
latest legal supervision signal. Cancellation, epoch rollover, slot reuse, or
source-version conflict invalidates the candidate.

## Runtime rendering

`render-runtime` consumes a locked selection artifact and emits three matched
run configs plus an argv-only launch plan. The adaptation reserve and Static KV
memory fraction are mandatory hardware-preflight inputs; neither has a source
default. Generated runtime files and absolute model roots belong under the
ignored artifact directory and must not be committed.

TTS and L0 configs must be byte-equivalent after removing the `method` field.
Changing a hyperparameter, sampling profile, load, model revision, or runtime
tree requires a new selection and evidence root.

## OnlineSPEC comparison configuration

Every OnlineSPEC run has the ordinary `adaptation` object plus a required
`online_spec` object. Its optimizer must be plain `sgd`; paper-specific state
is not expressed through an Adam or momentum alias.

| Field | Contract |
|---|---|
| `projection_radius` | optional positive Euclidean radius around the initial decision |
| `additional_learning_rates` | strictly increasing expert rates, Hedge only |
| `hedge_learning_rate` | positive Hedge meta rate, Hedge only |

OGD and optimistic OGD reject ensemble fields. Hedge requires at least two
ordered expert rates, uses `weight_update_mode=full`, and rejects LoRA because
averaging factors is not equivalent to averaging the represented parameter.
DFlash supports drafter Full/LoRA for the single learners. DSpark and
EAGLE/EAGLE3 accept only the shared tail scope and retain their ordinary
backend restrictions. All OnlineSPEC methods require TP=DP=1, frozen historical
KV, cohort isolation, exact rejection sampling, and one in-flight update.

The `online_spec` identity, expert rates, projection radius, and learner method
enter configuration, layout, selection, memory, and evidence hashes. See the
[OnlineSPEC baseline](onlinespec-baseline.md) for the state transitions and
separate registered protocol.
