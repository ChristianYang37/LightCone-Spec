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
- focused CPU tests and default CI.

Pending manual GPU acceptance:

- Target-only, Static, TTS, L0-naive, LightCone, and OnlineSPEC smoke;
- TP2 model load and publication check;
- one DSpark adaptation cell;
- one NEXTN interface cell;
- kill/restart resume exercise;
- three-block donor/rebuild performance comparison;
- then the registered full protocol.

No remote host was started and no GPU experiment was executed as part of the
repository rebuild.
