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
| E4-screen/local/profile | 48/168/3 | systems factors, six-block ablation, profiling |
| E3b-pilot/final | 20/132 | excluded pilots, 12-block primary and six-block secondary curves |
| E1a | 141 | DSpark geometry, five confidence weights, confirmation |
| E5-pilot/final | 53/160 | serving calibration, faults, 12/6-block confirmation |
| E6-pilot/final | 22/60 | interface/fit and six-block two-model transfer |
| E0-tune/pilot/final | 287/86/258 | compatibility, representative OnlineSPEC, bundled breadth |

The static materialization is 2,317 jobs; bounded TTS confirmation, E1 load,
width, and E6 load work bring the maximum to 2,385 runner jobs. There is no
global `N` or `final_blocks` setting.

## Comparisons and repetition

- E3b primary: LiveCodeBench 32K generated history, five mechanism roles,
  12 paired clean-server blocks.
- E3b secondary: MATH-500 16K generated history and four-context
  long-input/multi-turn suites, four core roles, six blocks.
- E5 primary: Target-only, Static, TTS, and LightCone; two backends; all
  registered concurrency, arrival-rate, and trace segments; 12 blocks.
- E5 topology and E4/E6/E0 transfer: six blocks.
- L0-naive is the publication-policy ablation. OnlineSPEC is tuned and reported
  only for Qwen3-8B + DFlash.

Screening and tuning are exploratory. Only frozen confirmation blocks support
effect claims. Paired methods share stimuli and run on the same GPU resource in
random order. A segment failure excludes its parent job attempt from reducers;
resume keeps completed segment evidence and completes the parent only when all
segments are terminal.

## TTS and DSpark

TTS uses Adam, one update, latest-round teacher rows, request reset, zero weight
decay, no clipping, positional decay `exp(-1/7)`, and strides
`1,5,10,15,20,30,40,50`. Its paper does not publish a learning rate, so the
nine learning rates are explicitly LightCone-Spec calibration. The source-policy
KL is algebraically omitted for the one-step update because its gradient is
zero at the current source policy.

DSpark keeps its native checkpoint and confidence scheduler. Confidence weights
`0.05,0.1,0.25,0.5,1.0` are calibration candidates; verification budget,
Brier score, ECE, and verification waste are always reported.

## Statistics

Three primary hypotheses share Holm correction: E3b LightCone–TTS goodput,
E3b LightCone–the faster frozen Target-only/Static baseline, and E5
LightCone–operational-baseline maximum feasible rate. E3b/E5 use paired log
ratios, BCa 95% intervals, and exact sign-flip tests over 12 blocks. E4/E6/E0
use block→request hierarchical bootstrap over six blocks. E5 p99 requires
10,000 completed requests and a time-block bootstrap. E0 workload-family
results are exploratory and use BH-FDR.

Formal evidence is compressed numeric request/cycle data, GPU telemetry,
configuration, metrics, SQLite state, and source attempt directories. Generated
text and token trajectories appear only in the excluded implementation smoke.
