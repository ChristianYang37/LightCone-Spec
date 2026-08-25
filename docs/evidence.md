# Experiment evidence

Completed cells retain the configuration row, compressed numeric request and
cycle rows, per-request outcomes, metrics, server log, and GPU samples in one
attempt directory. Formal request rows omit generated text and token IDs.
Failed and interrupted attempts remain separate and do not enter summaries.

Stage summaries are reproducible projections from completed attempt folders:

- `summary.csv` for inspection and plotting;
- `summary.parquet` for typed analysis;
- `statistics.json` on final nodes for paired BCa intervals, paired tests, and
  Holm decisions;
- `mechanism.json` for the available mechanism, memory, topology, energy, and
  profiler-derived fields, with unsupported hardware counters left `N/A`;
- SQLite selections for downstream recipe choices and powered block counts.

Committed-token goodput, native per-token ITL, peak HBM, KV capacity, safety
counters, and distributed rank-local plus sum/max/min metrics are mandatory.
Missing fields fail the cell. E5 runs a separate 11,000-offer boundary extension
and reports p99 only when at least 10,000 requests complete.

The evidence boundary is the ordinary academic one: report model and workload
pool names, local paths, environment versions, hardware, seeds, numeric rows,
exclusion rules, block assignments, and statistical outputs. Benchmark answers
are never scored, so the evidence cannot support a target-model capability
claim. The runner does not certify the exact byte identity of local files.

Implementation checks and CPU tests are not performance evidence. Until the
registered GPU cells finish, all throughput, latency, memory, acceptance, and
transfer conclusions remain `UNMEASURED`.
