# Paper protocol

`paper-v2` keeps the original 21-node scientific order while separating
screening, primary confirmation, and secondary transfer. A registered job is
one clean-server `method × block × compatible layout`. Its `segments` run
multiple contexts, loads, traces, or workload pools on that resident server;
every segment keeps separate requests, timing, and metrics.

| Node | Max jobs | Role |
|---|---:|---|
| preflight | 10 | runtime and excluded implementation smoke |
| E3a | 140 | three-regime width, context, and capacity screen |
| TTS-Cal | 108 | 72 recipes plus up to 36 finalist confirmations |
| E1 | 100 | 68 geometry rows plus Pareto confirmation |
| E2-r0/r1/r2/r3 | 424/109/31/25 | full successive halving and four fixed roles |
| E4-screen/local/profile | 52/168/3 | systems factors, TTS update steps, six-block ablation, profiling |
| E3b-pilot/final | 20/132 | excluded pilots, 12-block primary and six-block secondary curves |
| E1a | 3 | DSpark confidence capture, source latency, native-scheduler validation (22 segments) |
| E5-pilot/final | 11/66 | concurrency curves and 12/6-block confirmation |
| E6-pilot/final | 22/60 | interface/fit and six-block two-model transfer |
| E0-tune/pilot/final | 54/88/264 | compatibility, frozen OnlineSPEC validation, bundled breadth |

The static materialization is 1,822 jobs; bounded confirmation, E1 load,
width, batching calibration, E6 load, eight TTS-S10 confirmation jobs, and 19
replacement jobs bring the maximum to about 1,917 unique runner jobs. The registered
E4-profile retry is a new attempt of an existing job. The one-time KL64
reconciliation adds seven replacement cells and one unprivileged activity
proxy to the current run without adding public DAG nodes. There is no
global `N` or `final_blocks` setting.

## Comparisons and repetition

- E3b primary: LiveCodeBench 32K generated history, five mechanism roles,
  12 paired clean-server blocks. All methods run at matched `c1`; E5 carries
  the registered serving-concurrency comparison.
- E3b secondary: MATH-500 16K generated history and four-context
  long-input/multi-turn suites, four core roles, six blocks.
- E5 primary: Target-only, DFlash Static, Full TTS, and DFlash LightCone;
  concurrency `1..256` plus one BurstGPT trace, 12 blocks. Full TTS runs only
  at `c1` because its source protocol is per-request Full adaptation.
- E5 secondary: DSpark Static/LightCone and the explicitly separate
  TTS-LoRA-Batched engineering variant, six blocks.
- E4/E6/E0 transfer uses six blocks.
- L0-naive is the publication-policy ablation. OnlineSPEC OGD, Opt, and Ens use
  frozen public-source hyperparameters and are validated and reported only for
  Qwen3-8B + DFlash. Public request chunk sizes and epochs are provenance, not
  speculation-round stride; every formal adaptive row remains at `S=10`.

Screening and tuning are exploratory. Only frozen confirmation blocks support
effect claims. Paired methods share stimuli and run on the same GPU resource in
random order. A segment failure excludes its parent job attempt from reducers;
resume keeps completed segment evidence and completes the parent only when all
segments are terminal.

Headline blocks may run concurrently only when the preflight paired-relative
95% BCa intervals for both goodput and p99 ITL lie entirely within `[-1%, +1%]`.
The intervals need not contain zero: the admission question is whether the
measured interference is within tolerance, not whether a difference is
detectable. Incomplete or non-finite calibration keeps headline execution
serial. Resume recomputes this decision from the preserved preflight evidence
and records `criterion=paired_relative_bca_within_1pct_v2`; it does not rerun
completed cells. Each final block, including all methods and bundled segments,
stays on its registered GPU resource across resume; only distinct blocks run in
parallel. TP2/DP2 jobs continue to reserve their entire GPU pair.

## TTS and DSpark

Every formal adaptive TTS, L0-naive, LightCone, LightCone-candidate,
DSpark-LightCone, TTS-LoRA-Batched, and OnlineSPEC row uses stride `S=10`. TTS-Cal retains
the source-paper stride sweep `1,5,10,15,20,30,40,50`, and E4 retains its
stride ablation, but both are exploratory and do not select the formal stride.
TTS uses Adam, one update, latest-round teacher rows, request reset, zero weight
decay, no clipping, and positional decay `exp(-1/7)`. Because its paper does
not publish a learning rate, four paired `S=10` blocks compare `3e-5` with
`1e-4` before freezing the formal recipe. Multi-turn request-scoped adaptation
keeps one owner across four turns and restores source weights only at the full
conversation boundary; zero-publication segments are rejected. The
source-policy KL is algebraically omitted for the one-step update because its
gradient is zero at the current source policy.

DSpark keeps its native checkpoint and confidence scheduler. Its adaptation
geometry is copied from the frozen DFlash LightCone recipe, with the public
confidence loss weight fixed at `1.0`. Seven sequential position temperatures
are fitted on deterministic per-domain CalibrationMix halves; the other halves
validate the calibrated native scheduler. Thresholds `0.0..0.9` are replayed
offline for accepted/rejected tokens, acceptance, Brier, ECE, and ROC-AUC only;
they do not select a production threshold. Fixed-budget source panels use
sampling temperature 1 and chain drafting.

The one-time priority window runs compact E1a, complete E5-pilot, and E0-tune
before returning to E3b-final. It is resumable and does not change `PAPER_NODES`.
TP1 E0/E5 jobs may share the two GPUs only after an excluded Qwen3-8B+DFlash
interference validation places both goodput and p99-ITL paired BCa intervals
entirely inside `[-1%,+1%]`; CPU/NUMA bindings and observed affinities are saved.

## Statistics

Three primary hypotheses share Holm correction: E3b LightCone–TTS goodput,
E3b LightCone–the faster frozen Target-only/Static baseline, and E5
LightCone–operational-baseline throughput–interactivity frontier area. E3b/E5 use paired log
ratios, BCa 95% intervals, and exact sign-flip tests over 12 blocks. E4/E6/E0
use block→request hierarchical bootstrap over six blocks. E5 p99 uses a
time-block bootstrap. E0 workload-family
results are exploratory and use BH-FDR.

Aggregate committed-token throughput and per-user generation speed are distinct
metrics. Per-user speed is the arithmetic mean of each completed request's
native decode rate, computed from committed-token timestamps. Full TTS therefore
records declared and dispatcher concurrency separately and is never divided by
a nominal load that its request-reset implementation does not execute.

## Feasibility, latency SLOs, and deployment width

Every measured row separates three notions. `hard_feasible` means that the
server and all standard requests completed without runtime, numerical,
reconstruction, safety, database, NCCL, or OOM errors. `capacity_feasible` is
defined only for explicitly registered-load or capacity experiments.
`slo_pass` and `slo_pass_rate` retain the registered TTFT and p99-ITL thresholds
but have `slo_semantics=report_only_v2`: an SLO miss is reported and plotted,
not used to skip a standard scientific row. Capacity failures, scientific
rejections, and provider-blocked profiler rows remain visible terminal outcomes
and do not stop independent workers.

DFlash reconstruction uses the same valid-token mask and BF16 numerical unit as
the differentiable loss. Its mean-KL envelope is 64 BF16 units
(`64/128^2 = 0.00390625`); finite checks and the relative-RMS threshold are
unchanged. A one-time reconciliation replaces exactly the seven affected
width-4 adaptive rows while retaining the original attempts.

One deployment width is shared by Static, Full TTS, L0-naive, and LightCone.
A candidate in `{4,8,16}` must be hard-feasible for all four methods in all
three regimes. For every method/regime, goodput is normalized by its best
common candidate; the selected width maximizes the geometric mean of these 12
normalized values. Ties prefer lower peak HBM and then the smaller width.

## Profiling evidence boundary

The original NCU row remains `BLOCKED` when the provider denies privileged GPU
performance counters. An internal `unprivileged_activity_proxy` uses the same
controlled profile window and requests a PyTorch CPU/CUDA/memory trace with
shapes, Nsys activity summaries, and 100 ms NVML utilization, memory, power,
energy, and clock samples. Some SGLang/PyTorch builds emit only CPU and memory
events in the Kineto trace; that absence is labelled rather than hidden, while
non-empty Nsys CUDA kernel activity is mandatory. Kernel/API time, launch/queue
delay, GPU gaps, stream overlap, and memcpy activity are activity-level
evidence. Occupancy, warp-stall reasons, SM issue efficiency, and hardware DRAM
bandwidth remain `N/A`; they are not inferred from the proxy.

Formal evidence is compressed numeric request/cycle data, GPU telemetry,
configuration, metrics, SQLite state, and source attempt directories. Generated
text and token trajectories appear only in the excluded implementation smoke.
