# Architecture

[中文](../zh-CN/architecture.md) · [Home](../../README.md)

LightCone-Spec separates immutable orchestration from GPU execution. The host
layer resolves locks, manifests, parameter layouts, controllers, and evidence.
The SGLang patch exposes proposal signals and owns graph-visible buffers. The
side stream computes candidates; publication copies into a fixed-address active
bank only at a legal request boundary.

Each candidate carries `(request_epoch, slot_generation, source_version)`.
Cancellation, slot reuse, non-finite values, memory pressure, or a version
conflict discard the candidate without changing the active bank. A ready event
orders main-to-side inputs; a publish event orders side-to-main visibility.

The disabled path allocates no adaptation buffers and follows upstream SGLang.
The enabled path reserves resident adaptation memory before sizing the KV pool.
KV admission and retraction absorb load pressure; adaptation state is neither
silently evicted nor automatically moved to CPU.
