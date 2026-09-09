# Four-adaptive context benchmark (2026-09-10)

The user requested a new 480-call evaluation: always-on LightCone S10,
always-on LightCone S64, gated LightCone S10, and gated LightCone S5, 120
calls each. The gate remains 20K = 20,480 committed context tokens.
Historical Static measurements are preserved, never overwritten or rerun.

`preview-context-fixed20k-adaptive-v2` has separate call IDs and provenance.
It retains the prior frozen evaluation token IDs, sample IDs, seeds, ten
length bins, domain balance and 4,096-token ignore-EOS output budget.
Qwen3-8B+DFlash, TP2/c1, width16, ChronoBelief, LoRA rank8/last1, LR0.001,
constant schedule and the original memory budget are unchanged.

Use the retained RMSNorm/runtime and local rank collector, not the rejected
packed-telemetry or device-scalar candidates. Historical Static uses its
original runtime attribution and is not inserted into new matched-runtime
paired statistics. Current paired comparisons use always-on S10; archive
the original Static report and its raw row digests separately.

The independent `preview-benchmark prepare --fixed-threshold 20480
--adaptive-only` mode registers this matrix. Resumption requires exact frozen
provenance and skips completed calls. No calibration or Static QA is scheduled.
Two excluded complete S64 requests (short and 36K input) verify the newly
registered stride and the 40K engine boundary before the evaluation begins.
Existing accepted S10/gate/runtime evidence is preserved as provenance.

No formal DAG, paper, video or stride search is started. Reports retain all
negative results; a correctness failure stops the controller. The authorized
run controller backs up results and SQLite and powers off without releasing
the instance at completion or failure.
