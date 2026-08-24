# Current status

Empirical status: **GPU ACCEPTANCE INCOMPLETE**.

Implemented locally:

- four-command CLI and one-command launcher;
- fixed 21-node E0--E6 enumeration;
- two-GPU placement, clean-block lifecycle, and layout-compatible process reuse;
- SQLite WAL resume with one process/network retry;
- plain SGLang diffs preserving GPU-resident adaptation, fused optimizer work,
  fixed-address publication, backend adapters, native timing, and strict metrics;
- raw attempt outputs, CSV/Parquet summaries, and paired statistics;
- resumable E1 load, per-method width, DSpark confidence, E5 p99, and E6 load
  calibration without changing registered row counts;
- focused CPU tests and default CI;
- differentiable Qwen3.5 NEXTN current-block replay, merge-on-publication with
  LoRA scaling, ragged KV masking, and role-aware E6 capability handling.

GPU acceptance completed on 2026-08-22--23:

- Target-only, Static, TTS, L0-naive, LightCone, and OnlineSPEC smoke passed;
- TP2 DFlash publication/reset, DP2 sticky routing, and DSpark finite-update
  smoke passed;
- Qwen3.6-35B-A3B TP2 NEXTN generation passed. Its previous adaptive attempt was
  correctly rejected before publication because the old replay had no autograd
  path; the replacement replay is locally tested but has not yet been rerun on
  GPU.
- two earlier ten-cell preflight runs reached a completed stage; current-code
  preflight remains pending and those runs are not evidence for this checkout.

Still pending GPU acceptance:

- kill/restart resume exercise;
- three-block donor/rebuild performance comparison;
- 35B/122B NEXTN LoRA and Full interface checks, including TP2 and FP8
  publication;
- a fresh ten-cell preflight on the current code;
- formal dataset/evaluator availability and measured result-capacity budget;
- the registered full protocol, with unsupported E6 models retained as visible
  rejected interface rows and their dependent cells skipped.

Last read-only remote check: two idle RTX PRO 6000 Blackwell Server Edition
GPUs (97,887 MiB each), driver 580.95.05, Python 3.12.3, Torch 2.11/CUDA 12.9,
SGLang dev5, and FlashInfer 0.6.15. `nsys` and `ncu` are under
`/root/lightcone-tts-runtime/cuda-12.9/bin`. The writable root volume had about
32 GB free; the public NFS was read-only. No registered paper cell was executed.
That authorized run ended with the instance shut down; current remote state was
not rechecked in this local-only pass.
