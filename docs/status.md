# Current status

Empirical status: **REVISED GATE 8 COMPLETE; GATE 9 IN PROGRESS**.

- The four-command CLI, 21-node DAG, SQLite resume, strict scientific metrics,
  ordinary SGLang diffs, and even-GPU TP2 pair pool are implemented. Gate 6 is
  complete. Qwen3.5-122B Full at TP2 `c1` remains a legitimate HBM-blocked E6
  result; it is not replaced by TP3 or relabeled as LoRA.
- Gate 7 passed three alternating donor/rebuild blocks for all six methods.
  Rebuild deltas were goodput `-0.32%..+0.50%`, p99 ITL `-0.07%..+1.02%`, and
  peak NVML HBM `0..144 MiB`; deterministic trajectories matched and rebuild
  safety counters were zero.
- The efficiency-first Gate 8 completed all ten jobs and the stage reducer.
  Candidate equality and all six runtime safety events passed. The paired
  E3a smoke then exposed a 512 MiB FlashInfer workspace limit for registered
  speculative width 16. The first conditional fix depended on a server field
  that is not initialized when the attention backend is constructed; v11
  instead gives Qwen FlashInfer a direct 768 MiB workspace.
- The revised Gate 9 no longer requires dataset splits, answer scorers,
  answer-execution sandboxes, or LLM judges. It now checks CalibrationMix composition, nine
  renderable workload pools, the E5 trace, full materialization, and measured
  result capacity. Data and trace validation passed; paired E3a, capacity, and
  post-patch performance checks remain pending.
- Future eight-GPU execution uses four independent TP2 pairs. It still requires
  eight-GPU visibility/static-load checks, four-pair interference calibration,
  and one eight-GPU preflight before the full protocol starts.

The previous acceptance run used two RTX PRO 6000 Blackwell 96 GB GPUs. Evidence is under
`lightcone-gpu-results/acceptance-v26` outside the repository. The AutoDL
instance was confirmed `shutdown`; it was not released. No full E0--E6 run was
started. Paper speed and transfer claims remain unmeasured.
