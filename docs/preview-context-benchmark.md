# Preview context benchmark v1

This GitHub-only synthetic benchmark is separate from the 96-cell preview and the
21-node paper DAG. It never resumes that DAG, records videos, or searches stride.
The four existing `lightcone-spec` commands are unchanged. A separate
`preview-benchmark` entry point provides `prepare`, `calibrate`, `run`, `report`
and six excluded short `qa` requests.

## Frozen workload

Qwen3-8B + DFlash, TP2/c1, temperature 1, non-thinking, ignore EOS, exactly 4096
output tokens. Each independent calibration/evaluation split contains Chat 4
(MT-Bench 1, AlpacaEval 1, Arena-Hard 2), Code 4 (HumanEval 1, MBPP 1,
LiveCodeBench 2), and Math 4 (GSM8K 1, MATH-500 2, AIME-2025 1).
Source IDs and prompt text are disjoint from each other, background material,
and supplied historical tuning/preview exclusion manifests. Seed 0 sampling is
fixed before measurement. No answers or repeatedly cycled tasks form background.
Dataset scope follows [DSpark §4.1](https://arxiv.org/html/2607.05147#S4.SS1),
but forced lengths/ignore-EOS are **our stress protocol**, not its reproduction.

The ten columns are original short input, then exactly 4096, 8192, …, 36864
input token IDs including template/special tokens. All produce 4096 tokens.
Calibration is 120 Static calls; evaluation is 120 calls each for Static,
always-on LightCone S10 and context-gated LightCone S10. Total: 480 calls,
1,966,080 output tokens. Calibration and evaluation IDs remain separate.
LightCone uses ChronoBelief, LoRA rank 8, last1, LR 1e-3, constant schedule.

The logical maximum is 40960 tokens. Pinned SGLang internally subtracts one
slot in `TpModelWorker.get_worker_info` and another in the scheduler's output
cap, so this benchmark launches with **40962 engine context slots**. The two
guard slots are not generated tokens: inputs/outputs and the context gate remain
bounded at 40960. The engine capacity is recorded in the environment; old
manifests lacking this field require explicit migration review, not silent reuse.

All three paths use the same memory fraction and update-reserve budget, including
an explicit benchmark-only Static reserve. Gated LightCone initializes its update
state before serving; it does not claim Static's memory footprint. Record actual
per-rank KV capacities/allocator peaks and sampled NVML peaks separately.

## Candidate threshold (not causal drift detection)

For each of 12 calibration samples and bin j relative to short-input bin 0,
compute log decode-speed and log AL ratios. For each of 18 tests (9 bins ×
2 metrics), the one-sided upper bound is
`mean + t(1 - .05/18, df=11) * sample_std / sqrt(12)`.
Both bounds must be below `log(.95)` in two consecutive bins. Freeze the first
bin's lower boundary as the request-context activation threshold. Otherwise
save `no_trigger_detected`, and gated execution remains Static.
This assumes approximately normal paired log ratios; tokens and ten bins are
not independent replications. Attention cost alone must not be called drift.

The request-local, default-off `context_gate_v1` skips adapter preparation and
submission before activation. Native committed sequence length plus its already
sampled bonus token drives activation, not reserved/padded capacity. Until the
threshold is crossed this may require a scalar host wait under overlap scheduling;
that cost stays in throughput. Activation happens once, does not reset S10's
round clock, and does not backfill missed updates. Retraction retains the gate;
request reset clears it. TP ranks must agree. Only DFlash/c1/≤40K is accepted.

## Output and score

Each method has 3×10 throughput and AL tables. Throughput is the arithmetic mean
of four per-request output-token/submission-to-completion ratios, including
prefill and updating. AL is delivered verification draft plus bonus tokens over
actual verification calls: the initial prefill-generated token is excluded.
Raw sample points/variance, native decode timestamps, ITL, setup/reset, update
counts and per-rank memory remain available. Sample variance is not run variance.
Scores are equal-cell geometric means; absolute, domain and worst-cell ratios
to matching Static accompany them. Overall paired uncertainty aggregates each
sample's ten bins first, then uses 12 paired sample units and a 95% t interval.
No performance result is implied by successful CPU tests.

## Explicit execution

Run only on the authorized server (models stay there). Prepare requires the
existing YAML, a verified environment JSON and historical exclusion manifests:

```sh
preview-benchmark prepare --config /path/benchmark.yaml --output /path/benchmark \
  --environment /path/environment.json --exclusions /path/excluded-prompts.json
preview-benchmark qa --config /path/benchmark.yaml --output /path/benchmark
preview-benchmark calibrate --config /path/benchmark.yaml --output /path/benchmark
preview-benchmark run --config /path/benchmark.yaml --output /path/benchmark
preview-benchmark report --output /path/benchmark
```

The environment includes immutable model/drafter revisions, tokenizer/template,
runtime marker, GPU UUID/name/memory/driver, TP, dtype, width, c1, and budget.
Interrupted attempts remain visible and require explicit review; they are never
silently called completed or overwritten. Completed calls with identical
provenance are reused. Setup takes the exclusive GPU-owner lock; no formal active
claims or GPU process may exist. SIGINT stops at the next request boundary.
The supervisor must back up and check SQLite, retrieve compact results, then
API power off without releasing the instance, on completion or correctness error.

## Performance PR review

Maintainers explicitly produce the GPU report; external PRs and ordinary CPU CI
never receive host credentials or start paid GPUs. Run this verification in CI
on the maintainer's supplied artifacts:

```sh
python scripts/check_preview_benchmark.py --report /path/report.json \
  --manifest /path/manifest.json --environment /path/environment.json \
  --candidate-commit FULL_CANDIDATE_SHA
```

It checks 120 calibration plus 120 evaluation calls/method, candidate/manifest/
environment consistency, per-call provenance and reproducible scores. An older
base report is reusable only with matching protocol/environment. The verifier
does **not authenticate an arbitrary JSON producer**: maintainers must inspect
the trusted GPU run/log origin before merging. Review performance manually;
there is no automatic regression cutoff. Documentation-only PRs can mark N/A.
Repository branch protection is not asserted by this local implementation.
# Two-slot boundary recovery evidence

An explicit `engine-context-recovery.json` receipt may retain the 108 completed
Static calibration calls in bins 0–8 from the pre-fix engine. It is **not** a
general permission to mix commits or machines: every input, seed, checkpoint,
GPU UUID/driver, runtime and memory budget must match; only the recorded engine
guard slots may differ. The maintainer must review the code-equivalence evidence
and the unclipped raw calls before preparing this receipt.

Imported `result.json` files keep their original commit/environment and remain
byte-for-byte source evidence. The new aggregate provenance contains their
digests and source provenance; resume and report verification reject missing,
changed or out-of-scope imports. New measurements retain the candidate commit.
The failed final-bin attempt remains separately archived. A changed GPU or
driver does not authorize reusing this calibration or silently repeating it.
