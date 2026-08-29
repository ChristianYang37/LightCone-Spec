# Current status

Empirical status: **PAPER-V2 RUN PAUSED DURING E2 FOR CONCURRENCY-METRIC UPDATE**.

- The excluded `paper-v1-efficiency-main-v3` pilot stopped cleanly after
  preflight and 76 E3a cells. Its SQLite database passed `integrity_check` and
  its raw attempts were copied outside the repository; it cannot enter v2
  statistics.
- Preflight, E3a, TTS-Cal, E1, E2-r0, and E2-r1 are complete. The SQLite
  database passed `integrity_check`; the runner stopped at the E2-r2 cell
  boundary with 22 of 25 E2-r2 jobs complete and three pending, without
  changing completed attempts.
- `paper-v2` keeps the 21-node order and now has 2,185 static jobs by running compatible conditions as measured segments on one
  resident server. Primary E3b/E5 effects use 12 blocks; secondary E4/E6/E0
  evidence uses six.
- The next SGLang semantic marker is `paper-v1-nextn-shadow-v15`; cohort telemetry
  is sized for a complete measured cell rather than one request.
- Full TTS remains faithful request-reset `c1`. E3b uses matched `c1` for every
  method; E5 uses TTS-LoRA-Batched, not Full TTS, for concurrency sweeps.
- Qwen3.5-122B Full at TP2 `c1` may remain a visible HBM-blocked E6 result. It
  is not silently replaced by LoRA or another tensor-parallel degree.
- All LightCone speed, memory, serving, transfer, table, and figure values
  remain `UNMEASURED` until the new run completes.
