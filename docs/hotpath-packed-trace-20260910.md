# Follow-up hot-path audit: packed telemetry

Baseline: retained analytic RMS runtime v72 with local rank collector, not the
superseded global trace collector. Qwen3-8B + DFlash, TP2/c1, ChronoBelief r8,
last1, LR 1e-3, constant, S10. No formal evidence is replaced by these diagnostics.

Candidate hypothesis: fourteen per-update scalar trace writes incur avoidable
CPU dispatch and small CUDA operations. Stack detached scalar values directly
into the owned FP32 row. Preserve field order, flags, NaN entries, and ready-event
ordering; do not change optimizer, gradients, publication or reconstruction.

Validation: exact mixed-dtype/boolean trace equivalence, both-GPU fixed-state
microbenchmark, then independent Code/Math 36K 10s-warmup + 30s paired windows.
Use at most three repeats per side/domain. Microkernel gains alone do not establish
end-to-end gains. If no reproducible gain is observed, retain the v72 baseline.

Other inspected paths: TP decisions depend on cross-rank agreement, and active
BF16 reconstruction versus FP32-master gradients are not duplicate computations.
Removing either requires a separate dependency/equivalence proof. They are not
removed in this candidate.

GPU outcome: UNMEASURED. No push, paper change, full DAG, video or model download.
Collect evidence and shut down the exact instance without releasing it at end.
