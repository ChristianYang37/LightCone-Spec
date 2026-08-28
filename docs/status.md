# Current status

Empirical status: **PAPER-V2 READY FOR GPU PREFLIGHT; RESULTS UNMEASURED**.

- The excluded `paper-v1-efficiency-main-v3` pilot stopped cleanly after
  preflight and 76 E3a cells. Its SQLite database passed `integrity_check` and
  its raw attempts were copied outside the repository; it cannot enter v2
  statistics.
- `paper-v2` keeps the 21-node order and reduces static materialization to
  2,317 jobs by running compatible conditions as measured segments on one
  resident server. Primary E3b/E5 effects use 12 blocks; secondary E4/E6/E0
  evidence uses six.
- The SGLang semantic marker is `paper-v1-nextn-shadow-v13`; cohort telemetry
  is sized for a complete measured cell rather than one request.
- Qwen3.5-122B Full at TP2 `c1` may remain a visible HBM-blocked E6 result. It
  is not silently replaced by LoRA or another tensor-parallel degree.
- All LightCone speed, memory, serving, transfer, table, and figure values
  remain `UNMEASURED` until the new run completes.
