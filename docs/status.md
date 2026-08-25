# Current status

Empirical status: **GATE 7 COMPLETE; REVISED GATES 8--9 NOT YET RUN**.

- The four-command CLI, 21-node DAG, SQLite resume, strict scientific metrics,
  ordinary SGLang diffs, and even-GPU TP2 pair pool are implemented. Gate 6 is
  complete. Qwen3.5-122B Full at TP2 `c1` remains a legitimate HBM-blocked E6
  result; it is not replaced by TP3 or relabeled as LoRA.
- Gate 7 passed three alternating donor/rebuild blocks for all six methods.
  Rebuild deltas were goodput `-0.32%..+0.50%`, p99 ITL `-0.07%..+1.02%`, and
  peak NVML HBM `0..144 MiB`; deterministic trajectories matched and rebuild
  safety counters were zero.
- The earlier Gate 8 passed, but the efficiency-first protocol requires a fresh
  run with the revised implementation smoke and compressed evidence.
- The revised Gate 9 no longer requires dataset splits, answer scorers,
  answer-execution sandboxes, or LLM judges. It now checks CalibrationMix composition, nine
  renderable workload pools, the E5 trace, full materialization, and measured
  result capacity. These checks are pending.
- Future eight-GPU execution uses four independent TP2 pairs. It still requires
  eight-GPU visibility/static-load checks, four-pair interference calibration,
  and one eight-GPU preflight before the full protocol starts.

The previous acceptance run used two RTX PRO 6000 Blackwell 96 GB GPUs. Evidence is under
`lightcone-gpu-results/acceptance-v26` outside the repository. The AutoDL
instance was confirmed `shutdown`; it was not released. No full E0--E6 run was
started. Paper speed and transfer claims remain unmeasured.
