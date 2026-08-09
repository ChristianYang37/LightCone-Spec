# Configuration

[中文](../zh-CN/configuration.md) · [Home](../../README.md)

## Schema identity

All run configurations use schema version 2 and reject unknown fields. There
are three formal methods: `static`, `tts`, and `naive_async`. Three isolated
OnlineSpec identifiers are available only for external-baseline checks; they
are not valid substitutes in the formal speed study.

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
| `optimizer.name` | `adam` or `adamw` for TTS/L0 |
| `rank` | explicit for residual/LoRA; `null` for full |
| `stride` | positive integer |
| `max_in_flight` | exactly one |

Residual is tail-only. Drafter Full/LoRA is currently certified only for
unquantized DFlash with TP=DP=1 and a canvas width equal to the speculative
block size. Target embedding, target LM head, and target model remain frozen.

Static requires `adaptation: null`. This is a semantic fast-path requirement:
no optimizer, gradient, trace, candidate, or adaptation-reserve allocation may
be created.

The formal launcher explicitly adds `--speculative-speed-study-metrics` and
exact rejection sampling to all three endpoints. Without that study flag,
native SGLang allocates no LightCone metric state. With it, DFlash, rejection
sampling, and both acceptance thresholds equal to one are mandatory; a
missing exact kernel is an error and never falls back to greedy decoding.

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
