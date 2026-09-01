# Current status

Empirical status: **PAPER-V2 RUN ACTIVE AFTER ONLINESPEC V19 DEPLOYMENT**.

- The excluded `paper-v1-efficiency-main-v3` pilot stopped cleanly after
  preflight and 76 E3a cells. Its SQLite database passed `integrity_check` and
  its raw attempts were copied outside the repository; it cannot enter v2
  statistics.
- Preflight, E3a, TTS-Cal, E1, E2-r0 through E2-r3, and E4 are complete. The
  SQLite database passes `integrity_check`. The corrected S10 audit retained
  20 feasible r2 candidates rather than inventing an unavailable 21st row and
  reopened only ten missing r3 dependency cells. Original attempts remain
  immutable and excluded replacements are recorded explicitly.
- `paper-v2` keeps the 21-node order and now has 1,960 static jobs by running compatible conditions as measured segments on one
  resident server. Primary E3b/E5 effects use 12 blocks; secondary E4/E6/E0
  evidence uses six.
- The next deployed SGLang semantic marker is `paper-v1-nextn-shadow-v20`.
  It raises only the DFlash BF16 reconstruction KL envelope from 32 to 64
  numerical units; finite, valid-token masking, relative-RMS, and publication
  checks remain fail-closed.
- The pending soft-gate migration reinterprets ITL/TTFT SLOs as report-only,
  replaces seven affected width-4 cells, selects one hard-feasible public width,
  and reopens exactly 20 E3b-pilot plus 132 E3b-final rows. The unrelated 51
  satisfied E2 dependency rows and 36 exploratory exclusions remain skipped.
- AutoDL blocks privileged NCU counters. The raw blocked row is preserved and
  an unprivileged PyTorch/Nsys/NVML activity proxy is added without claiming
  occupancy, warp-stall, SM-issue, or hardware-bandwidth measurements.
- Full TTS remains faithful request-reset `c1`. E3b uses matched `c1` for every
  method; E5 uses TTS-LoRA-Batched, not Full TTS, for concurrency sweeps.
- Formal TTS, L0-naive, LightCone, LightCone-candidate, DSpark-LightCone, and
  TTS-LoRA-Batched rows now fail closed unless `S=10`; TTS-Cal and E4 stride
  sweeps remain exploratory. Multi-turn Full TTS/L0 preserve one adaptation
  owner across turns and reset only at the conversation boundary.
- Qwen3.5-122B Full at TP2 `c1` may remain a visible HBM-blocked E6 result. It
  is not silently replaced by LoRA or another tensor-parallel degree.
- All LightCone speed, memory, serving, transfer, table, and figure values
  remain `UNMEASURED` until the new run completes.
- E0 replaces the 236-row custom OnlineSPEC grid with three frozen
  Qwen3-8B+DFlash source-transfer validations: OGD, optimistic OGD, and Hedge
  ensemble. Their public chunk/epoch settings are recorded as provenance and
  are not reinterpreted as adaptation stride. All three passed the v19 GPU
  smoke with finite publications and zero safety-counter increments.
