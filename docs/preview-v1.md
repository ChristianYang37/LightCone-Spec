# Preview v1: registered formal supplement

Status: implementation in progress; all new GPU outcomes **UNMEASURED**.

Scope: GitHub preview/release only. Per the 2026-09-08 instruction, this work
does not update the manuscript, Evaluation, or paper-status documents. Its
registered evidence remains auditable in the shared run without being pooled
into the paper's existing primary statistics.

## Matched-TP extension (2026-09-08)

The original 56 logical leaves are unchanged by default. A separately enabled
Qwen3.8-27B supplement adds 24 leaves: Target-only, native MTP, community DSpark,
community DFlash2, Full TTS-DFlash2 and LightCone-DFlash2, each in four clean-server
paired blocks. Combined preview: **80 logical leaves**, excluding acceptance,
anchors, replacement attempts and twelve video takes across the two models.
The new panel uses eight frozen held-out LiveCodeBench prompts, 16,384 input
tokens inside the native non-thinking chat envelope, c1, temperature 1,
normal EOS and maximum 1,024 output tokens. Seeds are 0/1/2/3. This is a separate
small-budget transfer panel, not a replacement for the original long-generation
panel or a claim of improved target task quality.

The new panel uses `method_peak_v1`, S10, TTS lr=1e-4 and the frozen LightCone
transfer recipe. Its checkpoints are frozen in `QWEN38_CHECKPOINTS`; native MTP
shares the target revision. Community draft licenses remain separate from the
LightCone source-available license. DSpark's card declares `other`; complete
license review remains required before redistribution. No checkpoint is included
in this code distribution.

Each comparison chooses a common validated TP1 or TP2. DFlash long generation,
DSpark serving plus BurstGPT, Qwen3.8 six-method results, and each six-method
video suite are separate common-TP groups. TP2 replaces every relevant TP1 row
using a new physical job identity and `replaces_job_id`; raw attempts remain.
TP2 BurstGPT needs a measured topology-matched anchor and retains the original
trace/request budget. Never mix TP, budget or execution policies in paired effects.
`formal_preview_manifest_v2` is append-only relative to the v1 audit. Qwen3.8
execution requires an exactly matching `formal_preview_qwen38_acceptance_v1`.
The old runtime has not passed Qwen3.8/DFlash2 online-update acceptance.

Video recording uses SSE `/stream`, monotonic event IDs and replay via
`Last-Event-ID`. Actual token IDs are checked against cumulative trajectories;
committed bursts appear together, with UTF-8 suffix repair and no artificial
typing timer. Browser elapsed time never supplies the final benchmark field:
the header freezes backend `aggregate_tok_s` and native per-user speed. Videos
remain excluded individual-run measurements, not four-block benchmark estimates.
Opening the HTML with `file://` explicitly reports that no live service is connected.
The recorder requires `formal_preview_video_acceptance_v2` for all six methods
at a common TP; the 27B video uses TTS-LoRA-Batched, not Full TTS.

`capture_preview.cjs` retains native CDP frame timestamps, original PNG frames,
an unchanged-speed VFR original, raw stream events and backend counts.
Capture and Chromium must run on the SSH/GPU server itself against its local
loopback service. The recorder checks the service hostname against the capture
host and rejects tunneled remote capture. Download originals only after the
take finishes; WAN latency is excluded from the recorded rendering path.
Clock round trips before/after recording bound the service-to-browser epoch offset;
clock uncertainty above 100 ms invalidates the take. Composition requires six
valid `capture.json` sidecars with the same model/TP/c8/16K/1024 configuration.
Only pre-submission lead-in is removed, starting at the earliest plausible
submission so that all prefill remains. Alignment uncertainty and frame
resolution are disclosed; neither a recording-start timestamp nor a fake
typing timer can serve as submission alignment. Technical failures retain
their frames and logs; they are not candidates for fastest-take selection.

Deployment checkpoint: remote tokenizer-failure evidence has been backed up and
the instance shut down without release. Legacy cumulative patches compile;
the documented Qwen3.8 upstream `1cf2b8c` conflicts with the current allocation
and memory patches. Porting, GPU acceptance, final videos and measured figures
are unfinished; do not mark coverage accepted from CPU tests.
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
One method at a time, same validated TP/GPU set. GPU budgets and backend-specific widths are
recorded. Save events and requests locally; dataset text is not a public asset
unless its redistribution rights have separately been checked.

The browser displays actual received chunks, not simulated token-by-token text.
The observer is disabled during normal benchmarks. Overflow or disk failure
invalidates the recording. Native timing differs from browser/network lag.
Run `scripts/capture_preview.cjs` and Playwright on the SSH server itself,
using its loopback HTTP URL; copy the recordings back only after completion.
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
