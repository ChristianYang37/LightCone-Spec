# Architecture

[中文](../zh-CN/architecture.md) · [README](../../README.md)

## Boundaries

LightCone-Spec separates three responsibilities. The Python package owns
configuration, experiment identity, evidence, and inference. A mail patch adds
the minimum runtime hooks to a disposable SGLang checkout. Models, datasets,
CUDA libraries, and run artifacts remain external.

The checked-in SGLang directory is never a dependency or source of truth. The
only integration identity is the exact upstream commit plus the expected final
Git tree in `patches/sglang/manifest.json`.

## Decode lifecycle

Static takes the original DFlash path and imports no adaptation module. TTS and
L0 create one homogeneous cohort runtime per server process:

1. A verification round exposes the current teacher logits, drafter hidden
   state, semantic mask, and source version.
2. At the configured stride, the latest legal signal from each active request
   is normalized into one cohort batch.
3. A low-priority CUDA stream reconstructs the current DFlash canvas, checks it
   against inference output, computes the KL gradient, and proposes an optimizer
   state without mutating active state.
4. The candidate is bound to cohort epoch, slot generation, source round, and
   source version.
5. TTS waits for the next fixed boundary; L0 checks the ready event at every
   legal boundary. Publication copies staged values into fixed-address tensors.
6. A stale, cancelled, non-finite, conflicting, or uncertified candidate is a
   no-op. Native decoding continues and evidence records the reason.

Only one candidate may be in flight. This makes publication and cancellation
semantics explicit and prevents unbounded side-stream work.

## Parameter layouts

Drafter `full` owns every floating DFlash parameter, including transformer
linear layers, norms, final norm, and `fc`; target embedding, LM head, and target
model are borrowed and frozen. Drafter `lora` selects DFlash-owned two-dimensional
`fc`, QKV, output, gate/up, and down matrices. Its A/B factors are FP32 optimizer
state; publication merges only selected matrices into existing model storage.

Tail ablations use a hidden correction followed by one target-head projection.
Residual tail correction reuses the proposal output and does not run a second
target-head projection.

## KV and versioning

Historical drafter KV is an immutable state input, not a trainable tensor. The
update-only differentiable path gathers the bounded paged history into scratch
and detaches it. Current canvas K/V stays differentiable. KV produced after a
publication uses the new weight version; telemetry stores half-open KV segments
with their source versions.

This contract avoids invalidating an entire long request. It does not claim
equivalence to rebuilding historical KV under new weights; that would be a
different algorithm.

## Memory and concurrency

Adaptation memory is allocated before KV-pool sizing. The ledger separates
active/base state, FP32 master, gradients, first and second moments, staging,
training activations, KV gather scratch, candidate scratch, graph buffers, and
telemetry. Resident adaptation state is not evictable and is never silently
offloaded or downgraded. Remaining HBM determines KV capacity and admission.

Requests share updates only when model revisions, algorithm, sampling profile,
tenant, experiment group, parameter layout, and optimizer identity match. The
batch retains only the latest legal signal per request, so skipped update rounds
do not accumulate stale supervision.

## Evidence path

Each run writes process-unique Parquet shards for run, request, round, update,
and performance records. Missing counters are errors, not zeros. Run-scope
speculative counts are never copied into generated-token buckets. The derived
`speed_study.parquet` remains under the ignored artifact root.

A formal gate needs an attestation binding the source manifest, selection,
model revisions, patched runtime tree, hardware report, and exact evidence file
digests. Without it, the status remains `UNMEASURED`.
