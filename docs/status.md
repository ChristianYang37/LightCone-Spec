# Current status

Empirical status: **UNMEASURED**.

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

Pending manual GPU acceptance:

- Target-only, Static, TTS, L0-naive, LightCone, and OnlineSPEC smoke;
- TP2 model load and publication check;
- one DSpark adaptation cell;
- one NEXTN interface cell;
- kill/restart resume exercise;
- three-block donor/rebuild performance comparison;
- then the registered full protocol.

Last read-only remote check: two idle RTX PRO 6000 Blackwell Server Edition
GPUs (97,887 MiB each), driver 580.95.05, Python 3.12.3, Torch 2.11/CUDA 13.0,
SGLang dev5, and FlashInfer 0.6.15. `nsys` and `ncu` are under
`/root/lightcone-tts-runtime/cuda-12.9/bin`. The writable root volume had about
32 GB free; the public NFS was read-only. No GPU experiment was executed, and
the instance is shut down.
