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

Static takes the selected backend's original path and imports no adaptation
module. TTS and L0 create one homogeneous cohort runtime per server process:

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

## OnlineSPEC comparison lifecycle

The registered OnlineSPEC baselines reuse the same cohort, supervision,
version, exactness, and fixed-address publication infrastructure, but own a
prequential learner state. A proposal is made with the current online decision;
verification supplies feedback; a transactional candidate is then committed at
the next prediction boundary. OGD stores one decision, optimistic OGD stores an
anchor and gradient hint, and Hedge stores independent decisions and cumulative
loss for every learning-rate expert.

This state is not routed through the TTS/L0 optimizer or publication policy.
The comparison has a separate manifest, selection, evidence namespace, and
attestation, and cannot affect the core speed gate. See [OnlineSPEC
baseline](onlinespec-baseline.md).

## Backend contracts

DFlash exposes its actual proposal hidden state and supports both drafter-scope
Full/LoRA and the shared tail parameterizations. Only update rounds enter the
differentiable current-canvas path; ordinary proposal rounds retain graph replay
and do not run an extra vocabulary projection.

DSpark keeps two hidden identities separate. Its Markov feature is the raw
drafter output consumed by the Markov head, while its tail feature is the exact
post-collapse, post-final-normalization input to the frozen LM head. The tail
correction is applied before the Markov bias. The stored adapter-free logits are
reconstructed at the same sampled predecessor-token path, so the actual
proposal distribution is neither corrected twice nor replaced during
verification. Adaptation requires verify-all mode; confidence-based draft
truncation would remove supervision rows and therefore fails closed.

EAGLE and EAGLE3 pin one source adapter version from `draft_extend` through the
following verification. The first proposal and every internal proposal step
use the same fixed-address tail bank. Publication is blocked while any pinned
proposal remains outstanding. The legal transition is
`verify -> candidate -> publish boundary -> draft_extend`. Filter, merge,
overlap futures, retraction, finish, and abort preserve or invalidate the
version state transactionally.
The supported online shape is deliberately narrow: single layer, fixed depth,
top-k one, no token map, and exact full-vocabulary rejection sampling.

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
active/base state, FP32 master, gradients, allocated optimizer moments,
optimizer metadata, staging, training activations, KV gather scratch, candidate
scratch, graph buffers, and telemetry. Adam/AdamW allocate two moments;
SGDm/NAG/Lion allocate one; Muon allocates one per matrix and two for each
non-matrix auxiliary-AdamW parameter. Resident adaptation state is not
evictable and is never silently offloaded or downgraded. Remaining HBM
determines KV capacity and admission.

Candidate scratch is derived from the actual master, allocated moments,
functional proposal copies, and backend merge tensors; it is not a fixed
multiple of trainable parameter count.

OnlineSPEC preflight additionally counts the shared reset/projection snapshot,
optimistic anchor/hint tensors, independent Hedge expert decisions and
gradients, cumulative losses, and weighted-merge staging. The snapshot is
reported with active/base state; learner-only tensors are `online_state` bytes.
Neither category is hidden inside the KV budget.

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
