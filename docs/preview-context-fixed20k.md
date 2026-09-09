# Fixed-20K context benchmark

This GitHub-only synthetic evaluation uses a **user-specified** context threshold,
not a threshold fitted from the previous Static calibration. It does not change
the formal paper protocol, its recipes, or its S=10 rule.

`preview-context-fixed20k-v1` registers four modes:

- Static DFlash;
- always-on LightCone, S=10;
- context-gated LightCone, S=10;
- context-gated LightCone, S=5.

The gate activates at **20,480 committed context tokens** (prefill plus accepted
generated tokens) at a decode-round boundary. It activates once per request;
the speculation-round clock is not reset and missed updates are not replayed.
Below the threshold, adaptation preparation and updates are skipped. Parameters,
optimizer, gate and version state reset between requests; retraction preserves
the request's state. S5 applies only to the newly registered gated mode.

Each mode runs the same frozen independent evaluation set: Chat/Code/Math,
four prompts each, ten input-length bins, **120 complete requests per mode**.
Total: **480 evaluation calls, 1,966,080 output tokens**. The old120 Static
calibration calls are retained separately and are not rerun or imported.

Configuration: Qwen3-8B+DFlash, TP2/c1, width16, BF16, fixed-reserve memory policy,
ChronoBelief/LoRA rank8/last1/LR1e-3/constant for LightCone. Every request produces
4096 tokens with ignore_eos=true and temperature1.0. Input IDs, sample seeds,
templates and dataset exclusions are unchanged. Original short input is followed
by exact4096..36864-token input bins. Logical context is40960; engine context40962
includes two native reserved slots, not two additional scientific tokens.

All four modes use the same newly recorded GPU/runtime environment. Mode order
rotates deterministically by bin; server reuse is permitted only for compatible
configuration, with a verified full request reset. No old/new environment mixing.

## Run and verify

The separate `preview-benchmark prepare` supports `--fixed-threshold 20480`.
For this version `calibrate` explicitly refuses execution; `run` consumes the
fixed gate without loading a learned calibration cache. Every mode has new IDs
prefixed with the version, and recovery only skips matching completed evidence.

Before evaluation, run nine excluded QA requests: four modes at20464 and24576
input tokens with256 outputs, plus a complete36864+4096 Static boundary test.
This covers gate crossing, initial activation, S5/S10 publication, reset and TP
consistency. Runtime/numerical errors remain fail-closed. The benchmark launcher
allows the native engine's two-slot maximum override only for benchmark jobs.

Each mode has throughput and AL3x10 tables, four sample points and sample variance
per entry. Throughput includes prefill and updates. AL counts delivered accepted
drafts plus bonus tokens per target verification call, excluding the initial
prefill-generated token. Native decode speed, memory and timing remain available.
Paired uncertainty uses twelve sample units, aggregating each prompt's ten bins,
not120 independent replicates. Compare gatedS5 with Static, alwaysS10 and gatedS10.

The report validator requires480 evaluation identities and no calibration rows,
matching candidate/manifest/environment provenance, correct stride and threshold,
and reproducible counts, timings and scores. New results remain `UNMEASURED`
until execution. On completion, verify and download the compact report/evidence,
check SQLite integrity, and shut down the instance without releasing it. Do not
resume the full DAG, videos, stride search or paper updates.
