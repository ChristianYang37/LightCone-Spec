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

Packed trace outcome: rejected after twelve end-to-end windows (three pairs per
domain). Code -0.405%, Math -1.100%; the roughly 39% scalar-write microbenchmark
reduction did not translate to throughput. Restore the original trace writes;
preserve candidate a3de2924 and raw results. No push, paper change, full DAG, video or model download.
Collect evidence and shut down the exact instance without releasing it at end.

Second isolated candidate: replace per-update host-created constant LR/beta
CUDA scalars with device fills. Constant schedule retains the next-publication
validity test; beta tensors retain the current default dtype. Tensor arithmetic,
clipping, bias correction, age scaling and rejected-proposal transactions are
unchanged. Test complete proposals at long-run steps and reset, then check real
GPU proposals and host-transfer activity before end-to-end comparison. Its A/B
baseline is the retained v72 runtime after reverting the unsuccessful trace
packing. Only scalar construction differs in the second end-to-end A/B.

Scalar candidate outcome: not retained after twelve windows. Code +0.480%,
Math +0.326%, below the pre-registered 1% pooled-gain rule. GPU fixed-state
proposals, moments, steps, and rejected states were exact. Five optimizer
proposals generated 15 pageable H2D transfers and 15 stream synchronizations in
the original path, versus zero of either in the candidate. Optimizer-only median
elapsed time improved from 0.4495 to 0.4022 ms on GPU0 and 0.5537 to 0.5002 ms on
GPU1. These local reductions are not a new end-to-end speed claim. Preserve
b9742476/v75 as the measured candidate; restore v72 numerical/execution code.

No final long-request suite was launched for candidates that failed the speed
screen. One updated v72 baseline timeline adds frozen-prefix replay, dense LoRA
materialization, and complete objective/backward attribution. This excluded
profile is not used for ranking, and does not redo the 480-call benchmark.

Updated baseline timeline, rank0 CUDA-event inclusive intervals: target verify
17183.26 ms, training 3305.36 ms, frozen prefix 974.99 ms, objective/backward
1642.18 ms, dense LoRA materialization 469.86 ms, optimizer 132.73 ms. Intervals
can include waits and nest/overlap: these are not additive wall-time costs or
pure kernel durations. Both ranks have valid timelines with zero dropped or
pending events. The training replay chain deserves more attention than further
scalar micro-optimizations; no new candidate is automatically adopted.

Final state: 409 CPU tests passed; SGLang patch bytes match the start-of-turn
baseline. Original formal SQLite contents are unchanged and integrity is ok.
All raw windows, timelines, microchecks and the initial harness error are kept
under local `output/hotpath-followup-20260910` and the corresponding remote
diagnostics directory. Evidence archive SHA256:
`19c3ce9b4b10ee46dad08d77f503af06ab6d688748d929904aedb3a519faa10b`.
Live API confirms instance pro-785539d54f57 is shutdown, not released. Formal
YAML, paper, GitHub and the full DAG were not modified or resumed.
