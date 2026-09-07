# Preview v1: registered formal supplement

Status: implementation in progress; all new GPU outcomes **UNMEASURED**.
The preview is not a selection search, an original-system replication, or a
replacement for the full-budget and twelve-block primary results.

| Panel | Methods / conditions | Blocks | Leaves |
|---|---|---:|---:|
| Long generation | DFlash Static / Full TTS / LightCone; LiveCodeBench + MATH-500 | 4 | 24 |
| Serving | DSpark Static / LightCone; c1, c8, c32 | 4 | 24 |
| BurstGPT | DSpark Static / LightCone; existing trace shape | 4 | 8 |

Qwen3-8B only. Seeds 0/1/2/3; deterministic randomized method order;
whole paired units remain on one GPU. The long panel has eight frozen held-out
prompts, native chat formatting, c1, normal EOS, and a 32K output limit.
Serving uses a common 32-prompt pool, existing 40,928-token context construction,
10-second warmup, 60-second measurement, and the existing output/deadline rules.
BurstGPT freezes the E5-pilot block-0 reference, exact request count, offsets,
arrivals and lengths. Missing anchors are prerequisites, not extra hidden
preview cells. Existing E5 traffic is never rewritten.

All adaptive methods retain S10 and their frozen recipes; original Full TTS
is c1 with lr=1e-4. DSpark-LightCone requires the validated seven-position STS
recipe; Static uses its native source scheduler without online adaptation.
No recipe fallback or preview-driven tuning. This panel uses the historical
fixed-reserve policy consistently, not an unregistered memory-policy change.

`formal_preview_v1.enabled` is an internal deployment selection, enabled only
after CI, backup and excluded GPU acceptance. `formal_preview_manifest_v1`
freezes recipes, prompt records, widths and trace references exactly once.
The two internal storage stages do not change the 21 public nodes, four CLI
commands or SQLite schema. Completed attempts are not repeated. After all 56
terminal outcomes, the runner returns to the existing full DAG and E5-last.

## Reporting

Run the ordinary `summarize` command to regenerate `stages/preview-v1/preview.json`.
Public export is allowlisted and excludes prompts, credentials and local paths.
AL means accepted drafts / target verification calls, excluding the bonus token.
Aggregate throughput and native per-request generation speed are distinct.
Report absolute values, all four block points, failures/unavailable cells and
paired geometric-mean ratios with Student t 95% intervals on log ratios (df=3).
This assumes independent approximately normal block log ratios. Four blocks
support effect size/consistency; two-sided exact sign-flip minimum p is 0.125.
Missing or incompatible evidence is UNMEASURED, not zero and not omitted.

## Real video (excluded from the 56 formal cells)

Use `scripts/record_preview.py` in a separately authorized isolated GPU window.
Methods: Target-only, Static EAGLE3, Static DFlash, Native DSpark,
TTS-LoRA-Batched (DFlash), LightCone (DFlash). All actually submit eight requests
at once. Video TTS is NOT original Full TTS. Each receives the same deterministic
eight 16K-token constructed long inputs, maximum 1,024 output tokens, normal EOS.
One method at a time, same GPU. GPU budgets and backend-specific widths are
recorded. Save events and requests locally; dataset text is not a public asset
unless its redistribution rights have separately been checked.

The browser displays actual received chunks, not simulated token-by-token text.
The observer is disabled during normal benchmarks. Overflow or disk failure
invalidates the recording. Native timing differs from browser/network lag.
Use a loopback SSH tunnel and `scripts/capture_preview.cjs` with Playwright.
Keep uncut originals; one take per method unless an explicit technical failure
requires a documented retry. The composite is 1x with an on-screen independent-
runs/time-aligned disclosure, never a claim of simultaneous six-way execution.

## Branch and release policy

`codex/gpu-ready-minimal-runner` remains the full experimental implementation.
`release/preview-v0.1` is the publicity target; `codex/preview-v0.1` proposes
the matching implementation via PR. No main changes or automatic merge.
Christian reviews results and merges manually. Required CI and server-side
branch protection must be verified with authenticated GitHub API access;
local workflow files alone are not proof that protection is active.

This checkout is source-available, not uniformly open source. See LICENSE and
NOTICE for new-contribution restrictions and preserved Apache/third-party rights.
