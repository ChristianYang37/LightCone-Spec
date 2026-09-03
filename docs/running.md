# Running the experiment

1. Copy `examples/paper.yaml` outside the checkout and fill in absolute local
   paths. The file contains only runtime paths, an even list of GPU IDs, ports, memory
   settings, the CUDA toolkit path, the `paper-v2` preset, one process retry,
   profiler paths, and optional stage bounds. Consecutive IDs form TP2 pairs,
   so eight GPUs run as four independent pairs.
   Convert raw prompt pools with `scripts/prepare_datasets.py --task
   NAME=/absolute/source --output-root /absolute/pools`. Build CalibrationMix
   with `--calibration-manifest /absolute/calibration.csv`; the CSV columns are
   `source,path,count` and the counts must be 24/24/24/4 for APPS,
   OpenR1-Math, UltraChat, and controlled synthetic prompts.
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
missing native timing or counters, and scientific-gate failures are not
retried. Expected capacity-screen OOM is recorded as an
infeasible completed probe so higher-load screening does not abort the DAG.

Useful read-only commands:

```bash
lightcone-spec plan --config /root/lightcone-tts-runtime/paper.yaml
lightcone-spec status --run-dir /root/lightcone-results/paper-v2-efficiency-main
lightcone-spec summarize --run-dir /root/lightcone-results/paper-v2-efficiency-main
```

Formal attempts write compressed numeric request, cycle, GPU, and log files.
Generated text and per-token trajectories are retained only by the excluded
implementation smoke. Put the result root on the data disk.

The initial SGLang patch application uses ordinary `git apply --check`
followed by `git apply`. Repository state is not inspected.
If a local SGLang change
overlaps a diff, resolve that ordinary patch conflict before running again.

`protocol.start_stage` and `protocol.end_stage` are optional debugging bounds.
They do not fabricate upstream selections; starting at a dependent stage still
requires its selections to exist in the same run database.

## Execution resources and fixed request budgets

Registered `gpu_count` remains immutable. After excluded TP1 interference
acceptance enables the internal `tp1_resource_parallel_v1` selection, ordinary
single-server TP1/DP1 E0/E5 jobs reserve one device. TP2, two-replica DP2,
profilers and isolated controlled pairs retain their pair. Paired blocks stay
on one allocation; an already started block retains its recorded devices.
Missing/failed acceptance retains the original isolation policy. Session order
uses the original resource queue's seed, not the newly assigned GPU ID.

New attempts record `declared_gpu_count`, `execution_gpu_ids`,
`execution_gpu_count`, `execution_request_count`, and dispatcher concurrency.
The input budget is frozen from the original job before runtime load resolution
and process retries. Dispatcher capacity cannot expand the prompt sample;
TTS confirmation retains 19 prompts, explicit c128 retains 128, and E5 timed
windows/trace cycling retain their existing rules. `execution_request_count`
is the seed input-pool size for timed serving, not the total requests offered
during the window. Registered request counts and cosine horizons are unchanged.
Old attempts without these fields retain their original read-only interpretation.

Monitoring reports actual allocations separately from declared resources,
fixed input budget separately from dispatcher concurrency, and remaining leaf
cells separately from parent jobs. Only independent admitted units can overlap;
a single paired block's tail is not split to fill an idle GPU. Evaluate speedup
by completed valid cells per hour, not instantaneous GPU utilization (which is
not a measurement of model FLOP utilization).
