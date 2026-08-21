# Paper protocol

`paper-v1` is a fixed scientific protocol expressed directly in
`lightcone_spec.protocol`. It has 21 nodes and does not build a second workflow
representation at runtime.

| Node | Registered rows | GPUs per cell | Purpose |
|---|---:|---:|---|
| preflight | 10 | 1 or 2 | model load, exactness, memory, HTTP, interference |
| E3a | 360 | 1 | Target-only/Static capacity surface |
| TTS-Cal | 288 | 1 | TTS numeric calibration |
| E1 | 68 | 1 | scope/rank screen with fixed roles |
| E2-r0 | 3364 | 1 | successive halving round 0 |
| E2-r1 | 844 | 1 | successive halving round 1 |
| E2-r2 | 214 | 1 | successive halving round 2 |
| E2-r3 | 57 | 1 | final recipe selection |
| E4-screen | 48 | 1 | mechanism screen |
| E4-local | 96 | 1 | local factorial |
| E4-profile | 3 | 2 | isolated profiling |
| E3b-pilot | 1920 | 1 | four excluded 480-row blocks |
| E3b-final | `480N` | 1 | long-context confirmation |
| E1a | 116 | 1 | DSpark scope/rank and verification mode |
| E5-pilot | 2064 | 2 | four 450-row pilots plus 264 failure cells |
| E5-final | `450N` | 2 | production/topology confirmation |
| E6-pilot | 242 | 2 | two interface/fit probes plus four 60-row pilots |
| E6-final | `60N` | 2 | two-model transfer confirmation |
| E0-tune | `108+239V` | 2 | Static/adaptive compatibility and independent OnlineSPEC tuning |
| E0-pilot | `64V` | 2 | four excluded breadth blocks |
| E0-final | `16VN` | 2 | breadth confirmation |

`N` is selected once from the four excluded E3b pilot blocks and must be in
12--20; E5, E6, and E0 reuse that same final-block prefix. `V` is read from completed E0 compatibility
probes. A missing selection skips only dependent nodes. Independent diagnostics
continue, and no absent row is converted into a result.

Each E0 compatibility row runs the registered Static interface, then restarts
the same model/backend as adaptive LightCone and requires a finite published
update. E3 context intervals resample blocks and requests within each selected
block before evaluating the fixed 4K/16K/32K spline. Matched width uses the
E3a winner; deployment width uses the Static E3a winner and the registered
width-16 TTS/L0/LightCone tuning configurations.

Methods in a paired block run sequentially on the same device. Independent
single-GPU blocks may run on GPUs 0 and 1 concurrently. TP2, DP2, E5, and E6
cells use both devices exclusively. DP2 uses two independent TP1 servers with
sticky cohort routing. Every independent block gets a clean server;
cells with the same model/backend, parallel layout, optimizer-state layout,
width, and profiler mode reuse it after an idle configure/reset. Incompatible
layouts restart normally. Preflight disables parallel final blocks when measured
two-GPU interference exceeds 1%.

TTS uses the fixed paper recipe: Adam, zero weight decay, no gradient clipping,
the complete drafter, one update step, latest-round-only teacher rows,
`exp(-1/7)` positional decay, and request reset. TTS and L0-naive share the
candidate update and differ only in publication policy. TTS-Cal uses 76 tuning
problems from an explicit tuning split and never executes the four explicit
holdout IDs. E2
uses the registered `(B,L)` points as closed-loop concurrency and generated
history length. E4 factors map directly to runtime arguments and its profiling
rows invoke NVTX, Nsight Systems, and Nsight Compute. E5 maps the registered
arrival traces, sticky cohorts, popularity, topology, and eleven failures to
runtime behavior; BurstGPT replays both arrival times and input/output lengths.
Every E6 role, including Target-only, runs TP2.

Required correctness checks are limited to:

- proposal version matches the published parameter version;
- TTS and L0 controlled replay produce the same staged candidate;
- greedy output token trajectories agree with Target-only;
- stochastic decoding passes the registered distributional test;
- loss, gradients, and updates are finite;
- OOM, fallback, retraction, and stale-publication events are counted;
- goodput uses committed tokens and comparisons share an HBM-feasible load;
- native token timestamps, rank-local metrics, and TP/DP sum/max/min aggregates
  are present rather than reconstructed from HTTP delivery time;
- pilot rows never enter final estimates; statistics operate on paired blocks.

Process and network failures retry at most once. OOM, numerical errors,
exactness violations, and failed scientific criteria are terminal for the cell.
