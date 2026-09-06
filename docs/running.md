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

Registered `gpu_count` remains immutable. With `LIGHTCONE_NUMA_ISOLATION=1`,
the runner records GPU PCI/NUMA topology, reserves at least four physical cores
or ten percent of the host for itself and the OS, and launches each SGLang
server on a disjoint physical-core set (including its SMT siblings). Memory
placement uses a preferred local node when `numactl` is available; CPU affinity
remains active when NUMA placement is unavailable. After excluded TP1 interference
acceptance enables the internal `tp1_resource_parallel_v2` selection, ordinary
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

E0-tune is the sole unconditional exception to the confirmation interference
gate: its 54 compatibility/recipe cells are exploratory, independent TP1 units
and may occupy one GPU each. E0-pilot/final still require the existing NUMA
paired-BCa acceptance before two TP1 blocks overlap. In E5 topology-transfer
rows, `closed_loop_cN` always means system concurrency; the DP2 router divides
that offered work across replicas without doubling the fixed request budget.

Monitoring reports actual allocations separately from declared resources,
fixed input budget separately from dispatcher concurrency, and remaining leaf
cells separately from parent jobs. Only independent admitted units can overlap;
a single paired block's tail is not split to fill an idle GPU. Evaluate speedup
by completed valid cells per hour, not instantaneous GPU utilization (which is
not a measurement of model FLOP utilization).

After the source-transfer migration is present, priority window v2 runs compact
E1a and E0-tune, writes a stratified 10,000-draw discrete-scheduler ETA, then
finishes every non-E5 node before resuming E5-pilot/final as the isolated tail.
It preserves the v1 audit and never substitutes default DSpark confidence
temperatures: an unavailable seven-position STS recipe disables only
DSpark-LightCone work, while independent methods remain runnable. ETA output is
stored under `stages/priority-window-v2/eta.json` and reports remaining parent
jobs, logical leaf cells, P50/P90 total time, and the E5-only tail.

## Four-block coverage rollout

The coverage extension uses marker `paper-v1-nextn-shadow-v26` on a new patched
SGLang worktree; do not apply it over the active v25 runtime. Keep the production
runner working during CPU checks and CI, then drain it at a cell boundary and
back up SQLite/WAL and active evidence before switching runtimes.

Prepare unchanged official source files with `scripts/prepare_source_coverage.py`.
Add its nine `DeepSpec-source|...` dataset keys and twelve separately verified
official checkpoint keys to the runtime configuration; do not overwrite existing
dataset/draft paths. Source block-7/TTT7 and main DFlash block-16 are distinct.

In a drained window run `scripts/gpu_acceptance.py coverage --config /absolute/paper.yaml
--output /absolute/excluded-diagnostics --phase all` (or the `gemma`, `qwen`,
`dense14`, `panels` phases). Its 41 short cells use an excluded SQLite and a
read-only formal selection snapshot. They do not enable the formal extension.
Review real capacity outcomes separately from runtime errors, and verify
nonzero updates, request reset, TP2, and deliberate valid-token safety rejection
before recording the internal `formal_coverage_runtime_v1` accepted selection.

Resume preserves old attempts. Method-specific compatibility replacements,
corrected native-teacher STS, restored E0 leaves, and E6-owned dense-14B evidence
have separate internal stage identities. Additional source/mechanism work is
exactly 1,296/48 leaves and runs before the unchanged E5-last tail. The public
21-node plan and existing six-/twelve-block evidence are not rewritten.

`stages/coverage-eta-v1/eta.json` accounts for remaining original, replacement,
and four-block leaves. It uses request-linear costs only within matched
model/backend/method, output length, load, topology and panel strata, charging
clean startup conservatively. Missing strata leave full ETA `UNMEASURED`; the
priced subset is explicitly not a total ETA. The earlier priority-v2 estimate
excludes the new source panels and must not be presented as their finish time.
Summaries and statistical reducers both consume replacement-filtered logical
cells, including completed siblings beneath superseded bundled parents.
# Coverage-only memory budget (method_peak_v1)

Only the new 1,296 source cells, 48 mechanism cells and coverage compatibility
replacements use `method_peak_v1`. Old pending jobs and started paired groups
retain `fixed_reserve_v1`; immutable job records and raw attempts are not rewritten.
The 52 GiB adaptation setting is a peak ceiling for the new policy, not an empty
allocation. Before KV allocation, the constructed TP-local adapter validates its
resident tensors and scratch categories. Additional headroom is
`max(estimated_update_peak - resident, 0)`; KV receives free memory minus this
headroom and the unchanged runtime slack. Both TP ranks must fit.

Unknown adaptive layouts stop; Static alone has zero update reserve. Full/SWA
and draft pools must hold one complete registered maximum-context request plus
verification slots. Native prefix eviction/retraction reuses slots within the
fixed pool; it does not release CUDA memory or permit silent load/context changes.
Attempts record policy, ledger, cap/headroom/slack, KV capacities and retractions.
Session reuse, paired statistics and ETA isolate the two policies.

Excluded adaptive QA reports both the side-update allocator delta and an upper
bound measured from before the first proposal to update completion, including
capture/replay and native inference temporaries. Neither is a hardware-wide
attribution of allocations to one method. Stress acceptance requires the whole-
round upper bound to fit the estimated update budget on every rank; an excessive
upper bound needs attribution review, not an automatic claim that adaptation
alone exceeded its budget. The formal async path does not enable this
instrumentation. Long-context and retraction acceptance
remain required before coverage acceptance, even if short QA passes. Estimated
versus measured values and any unmeasured components must be reported separately.
New-policy ETA cannot borrow fixed-reserve timings; missing strata keep the full
ETA `UNMEASURED`, with a separately labelled priced subset.

Run `scripts/gpu_acceptance.py memory-pressure` only in a drained excluded window,
with `--case long_tts`, `long_lightcone`, `long_gemma_lightcone`, `tp2`, or `retraction`. The first four
retain two complete 32K-output requests. The retraction case retains eight c8
requests with 8K outputs while capping its diagnostic KV pool at 41,024 slots;
the minimum full-context guard still applies. It must observe actual native KV
retractions in the server log and zero logical-prefix/safety violations, not just
complete a short request. It is separate from the 41 short QA identities, and its
timings never price the formal ETA.
