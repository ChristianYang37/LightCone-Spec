# Running the experiment

1. Copy `examples/paper.yaml` outside the checkout and fill in absolute local
   paths. The file contains only runtime paths, two GPU IDs, ports, memory
   settings, the CUDA toolkit path, the `paper-v1` preset, one process retry,
   profiler paths, and optional stage bounds.
   Convert raw datasets with `scripts/prepare_datasets.py --splits splits.csv`
   when the configured files are not already normalized. The split CSV is the
   sole source of tuning/pilot/final/holdout membership.
2. From the project checkout, run:

   ```bash
   ./run_paper.sh /root/lightcone-tts-runtime/paper.yaml
   ```

3. Re-run the same command after a runner/server interruption. Existing
   completed attempts remain complete, an interrupted attempt remains visible,
   and its job is returned to `pending` for a new attempt directory.

The first invocation stores a normalized plain `paper.yaml` in the run
directory. A later invocation with different experiment values stops instead
of mixing configurations in one SQLite run.

SIGINT and SIGTERM stop new work at the next cell boundary. A process/network
failure is retried once, including server startup. OOM, non-finite adaptation,
trajectory mismatch, missing native timing or counters, and scientific-gate
failures are not retried. Expected capacity-screen OOM is recorded as an
infeasible completed probe so higher-load screening does not abort the DAG.

Useful read-only commands:

```bash
lightcone-spec plan --config /root/lightcone-tts-runtime/paper.yaml
lightcone-spec status --run-dir /root/lightcone-results/paper-v1-main
lightcone-spec summarize --run-dir /root/lightcone-results/paper-v1-main
```

The initial SGLang patch application uses ordinary `git apply --recount
--check` followed by `git apply --recount`. Repository state is not inspected.
If a local SGLang change
overlaps a diff, resolve that ordinary patch conflict before running again.

`protocol.start_stage` and `protocol.end_stage` are optional debugging bounds.
They do not fabricate upstream selections; starting at a dependent stage still
requires its selections to exist in the same run database.
