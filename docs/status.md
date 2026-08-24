# Current status

Empirical status: **GPU GATES 1--8 COMPLETE; GATE 9 BLOCKED**.

- The four-command CLI, 21-node DAG, SQLite resume, strict scientific metrics,
  ordinary SGLang diffs, and even-GPU TP2 pair pool are implemented. Gate 6 is
  complete. Qwen3.5-122B Full at TP2 `c1` remains a legitimate HBM-blocked E6
  result; it is not replaced by TP3 or relabeled as LoRA.
- Gate 7 passed three alternating donor/rebuild blocks for all six methods.
  Rebuild deltas were goodput `-0.32%..+0.50%`, p99 ITL `-0.07%..+1.02%`, and
  peak NVML HBM `0..144 MiB`; deterministic trajectories matched and rebuild
  safety counters were zero.
- Gate 8 passed on a fresh run: stage `completed`, 10/10 jobs `completed`, no
  failed or skipped jobs.
- Gate 9 is blocked. Nine formal datasets lack historical explicit split IDs;
  `bwrap` is installed but the host kernel rejects its namespace smoke; official
  MT-Bench, AlpacaEval, and Arena-Hard evaluators and judge credentials are
  absent; and 24.7 GB free space is below the measured result projection.
  TTS-Cal itself is valid at 76 tuning plus four unexecuted holdout IDs.
- Future eight-GPU execution uses four independent TP2 pairs. It still requires
  eight-GPU visibility/static-load checks, four-pair interference calibration,
  and one eight-GPU preflight before the full protocol starts.

The Gate 7--9 run used two RTX PRO 6000 Blackwell 96 GB GPUs. Evidence is under
`lightcone-gpu-results/acceptance-v26` outside the repository. The AutoDL
instance was confirmed `shutdown`; it was not released. No full E0--E6 run was
started and the paper source was not modified.
