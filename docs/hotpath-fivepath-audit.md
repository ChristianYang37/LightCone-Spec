# Five-path throughput audit (excluded, 2026-09-10)

This audit does not change the formal protocol or resume the stopped benchmark.
The retained reference is SGLang v72 with analytic RMSNorm backward and the local
rank collector. Both variants use Qwen3-8B + DFlash, TP2/c1, ChronoBelief,
rank8/last1/lr=1e-3/constant/S10 and the same non-display Code/Math prompts.

## Hypotheses and correctness boundaries

| Direction | Candidate | Preserved invariant | Decision evidence |
|---|---|---|---|
| Dense LoRA construction | Evaluate `xW + (xA^T)B^T` instead of constructing `W+BA` | Must account for the existing BF16 weight-rounding boundary and gradient target | Two-rank microtest finds different forward values and gradients; not installed |
| Frozen replay / KV gathering | `index_select` for the existing NHD locations | Same locations, ordering, duplicates, bytes and detached ownership; HND branch unchanged | CPU and two-rank exact GPU gather checks; end-to-end A/B |
| Loss / reconstruction preparation | Compute valid row indexes once, select both tensors | Same valid rows, empty rejection, padding exclusion and finite/RMS/KL/top1 reductions | Exact GPU gate outputs for equal, padding-NaN, valid mismatch and empty cases; A/B |
| TP reconstruction communication | Stack the inference and active-weight local logits into one gather | Same rank concatenation and complete vocabulary; gradient-objective gather and all safety decisions retained | Actual two-rank NCCL byte comparison; A/B |
| Main-loop bookkeeping | Write the logical prefix directly when available | Same prefix value including zero, physical fallback, reset/retraction and round/version state | CPU mixed-prefix and exact GPU write checks; A/B |

These are source-pinned, opt-in transformations in `hotpath_candidates.py`, not
default production flags. Each runtime must match the reference's 55 patched
Python files plus exactly its one transformation. The KV candidate neither
caches activations across updates nor reduces the training history. The TP
candidate reduces invocation count, not data volume or safety coverage.

## Measurement contract

- Independent 10-second warmup, reset, then 30 seconds after first token;
  prefill-inclusive window throughput and decode-window speed are both retained.
- Existing EOS handling, native timestamps, cancellation/reset checks, inputs,
  seeds and scientific configuration are unchanged.
- Old/new order alternates; each domain has at most three pairs. The existing
  two-domain decision rule requires positive gains in both domains and at least
  1% geometric-mean gain after repetitions to enter independent validation.
- Model loading and runtime switching are recorded separately. Micro timings
  and profiler timings are not substituted for end-to-end throughput.
- A candidate passing screening still requires independent confirmation and
  complete-request correctness QA before it can replace the retained reference.
- Runtime failures stop the controller. Existing formal attempts and the
  historical Static measurements are not rerun or rewritten.

The local output report records all retained/rejected outcomes. A missing final
acceptance record is not permission to deploy a candidate in a formal run.

## Completed screening

All four eligible candidates completed three pairs per domain (48 performance
windows total), with matched request budgets and successful cancellation/reset.
Paired throughput gains against the retained LightCone reference were:

| Candidate | Code | Math | Outcome |
|---|---:|---:|---|
| Logical prefix write | -0.188% | -0.374% | Keep reference |
| KV index_select | +0.384% | +0.125% | Below 1% threshold |
| Single mask index | -0.211% | -0.016% | Keep reference |
| Paired logits gather | +0.122% | -0.172% | Keep reference |

No candidate qualified for independent confirmation or full-request acceptance;
therefore no additional long GPU tests were run. The production patches remain
unchanged. Source transformations and failed/no-benefit outcomes are retained so
these paths do not need to be rediscovered or retested without a new hypothesis.
