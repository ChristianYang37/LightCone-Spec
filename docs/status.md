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
- focused CPU tests and default CI.

GPU acceptance completed on 2026-08-22:

- Target-only model-load, generation, native timing, HBM/KV, and zero
  safety-counter smoke passed;
- Static/DFlash completed generation with zero safety counters but reported 536
  committed tokens for a nominal 512-token batch; this remains unresolved;
- TTS/DFlash full-drafter startup and serialized warm-up passed after correcting
  memory planning and side-effecting health checks.

Still pending GPU acceptance:

- Static committed-token accounting, TTS scored batch, L0-naive, LightCone,
  and OnlineSPEC smoke;
- TP2 model load and publication check;
- one DSpark adaptation cell;
- one NEXTN interface cell;
- kill/restart resume exercise;
- three-block donor/rebuild performance comparison;
- then the registered full protocol.

Last read-only remote check: two idle RTX PRO 6000 Blackwell Server Edition
GPUs (97,887 MiB each), driver 580.95.05, Python 3.12.3, Torch 2.11/CUDA 12.9,
SGLang dev5, and FlashInfer 0.6.15. `nsys` and `ncu` are under
`/root/lightcone-tts-runtime/cuda-12.9/bin`. The writable root volume had about
32 GB free; the public NFS was read-only. No formal paper cell was executed.
The last retry stopped after repeated SSH timeouts, and the instance is shut down.
