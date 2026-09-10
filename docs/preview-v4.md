# GitHub preview v4: 180 cells and selected effect videos

Implementation/acceptance are separate from measurement. All new results are
`UNMEASURED` until the matching attempts exist. This protocol supersedes preview
v3 scheduling, not its raw evidence. It does not modify the paper, full DAG,
formal recipes, public four-command CLI or SQLite schema.

## Frozen comparisons

| Panel and question | Comparison | Cells / destination |
|---|---|---|
| 8B request-local long generation: is online cost repaid inside a request? | Code/Math × Target-only, Static EAGLE3, Static DFlash, OnlineSPEC-Ensemble–DFlash transfer, LightCone–DFlash × four blocks | 40; efficiency and length trajectories |
| 8B DSpark cohort service: does adaptation help the complete serving window? | c1/c8/c32/BurstGPT × Static/LightCone × four blocks | 32; serving frontier and completion |
| Qwen3.8-27B request-local transfer: does the effect transfer to the frozen newer target? | Chat/Code/Math × Target-only, native MTP, community Static DSpark, community Static DFlash2, OnlineSPEC-Ensemble–DFlash2 transfer, LightCone–DFlash2 × four blocks | 72; model/backend efficiency |
| 8B DFlash state-lifetime intervention: does reuse repay cost, and how does switching domains affect it? | three flow orders × Static/request-reset LightCone/persistent-cohort LightCone × four blocks | 36; full-flow cumulative net time |

LightCone is ChronoBelief, LoRA rank8, last1, LR=.001, constant, S10,
**no context gate, no tuning**. OnlineSPEC retains the v3 Adam/Hedge transfer
recipe and label, not an original-engine reproduction. TTS is absent. Retain the
optimized production implementation and local scheduler collector, not rejected
teacher-reuse candidates or a global Python trace for performance collection.

Four independent clean-server paired blocks use seeds 0/1/2/3. Methods share
stimuli, per-sample seeds, topology and execution policy within a block. Freeze
method order before measurements. Select the minimum TP1/TP2 passing **all**
methods' complete registered conditions. Never upgrade one method alone, shrink
inputs, quantize, offload, or lower Static resources. Capacity failures are
reported; correctness failures stop the affected execution for diagnosis.

Long generation uses eight unseen prompts per cell, native input, c1, maximum
32768 output tokens, temperature=1, non-thinking, normal EOS. Arena-Hard Chat is
not forced to produce long answers. Source tasks are Arena-Hard, LiveCodeBench
and MATH-500. Each request restores adapter parameters, optimizer and version;
sequential submission alone is not evidence of reset.

DSpark preserves its frozen pool of 32 prompts, c1/c8/c32, ten-second warmup,
60-second measurement, request deadlines and exact BurstGPT arrivals/lengths.
Both Static and LightCone use the **same seven STS temperatures**, checkpoint,
native scheduler and width. Warmup ends with a complete reset. LightCone shares
adaptation only within the measured window. Report repeated-pool request share,
completion/timeouts and full offered-flow outcomes; repetition is not new-sample
generalization. Matching TP2 trace anchors are additional work, not part of 180.

## Data freezing and the cohort control

Freeze source revisions, sample IDs, templates and checkpoint revisions. Exclude
all prior tuning, preview, diagnostic and synthetic-benchmark samples from the
long-generation confirmation pool by source ID and exact prompt text. The 8B and
27B models share the same 32 confirmation prompts per applicable domain; do not
resample after seeing performance. Each of the four cohort blocks has 48 distinct
requests, 16 per domain; blocks are disjoint. Cohort and video may reuse historical
corpus samples, explicitly disclosed, and do not claim unseen-task generalization.
Within v4, confirmation, cohort and video pools are separate. Insufficient fresh
confirmation data is a preparation blocker, not permission to fabricate tasks.

The three orders contain exactly the same block's sample IDs and ID-bound seeds:
contiguous groups of 16; deterministic random interleaving; groups of eight per
domain repeated twice. Context/KV is never shared between unrelated requests.
Persistent adaptation receives no future domain labels, routing oracle or repeat
training requests. Output limit is 2048, c1, normal EOS. Request reset versus
cohort persistence is the only state-policy intervention.

Count from request one, including updates and measured request-reset boundaries:
`G(n) = cumulative Static wall seconds - cumulative LightCone wall seconds`.
Save all 48 points, first positive crossing, later negative crossings, final net
benefit and domain-transition costs. Separately publish loading, warmup/reset,
input preparation and tail costs, plus complete lifecycle wall time. Do not add
overlapping startup counters twice or hide them behind a steady-state-only gain.
Negative or order-sensitive effects refute broad serving-acceleration claims and
remain in the table.

## Statistics, telemetry and publication

Throughput is delivered tokens divided by the complete measurement wall time,
including online work, not throughput divided by declared concurrency. Native
per-request committed-token timestamps define per-user speed. Report output
length, EOS, completion, TTFT/ITL, updates, per-rank HBM and KV alongside speed.
AL has two explicit fields: accepted drafts / verification calls, and accepted
drafts plus **actually delivered** bonus tokens / calls. Missing bonus telemetry
stays `UNMEASURED`; one bonus per call is not assumed. Target AL is N/A.

Show all four raw block points and paired log-ratio t intervals, 95%, df=3,
with approximate-normality assumption. Requests and tokens are not independent
replicates. Pairing includes model/backend, scene, TP, memory and execution
policy, state variant and block. Never pool request-reset and persistent rows,
old/new preview revisions, or all scenes into one promotional multiplier.

Five result families: cross-model efficiency, long-generation trajectories,
cohort cumulative net time, DSpark serving curves, and memory/update cost.
Public exports exclude prompt bodies and local credentials/paths. Raw private
evidence remains immutable; missing results and all registered negative points
remain visible. Versioned jobs and group acceptance are in `preview_v4.py`;
effects and scene ranking are in `preview_v4_report.py`.

## Selected-case videos (excluded)

After four matched blocks, rank whole scenes by the geometric mean LightCone /
matched Static throughput ratio; ties use the frozen scene ID. Publish the
complete ranking. Select one scene per model, plus the best persistent-cohort
order. If no scene improves, show that honestly without retries or tuning.
Caption: **selected effect case from formal results, not average performance**.
Formal c1 evidence does not guarantee that a c8 video will improve.

Two six-method main videos use the existing accurately labelled 8B and 27B
lists (8B additionally shows Native DSpark). All use the same eight 16K inputs,
max1024 output, real c8 and minimum common validated TP, starting cold. Disclose
within-batch adapter sharing. An unsupported c8 method is unavailable, not a
substituted algorithm or serial run labelled c8.

The cohort appendix records the selected order's entire block0 48-request cold
flow for Static and persistent LightCone. There are 14 original recordings and
three 1× time-aligned composites. Each method gets one valid take; only recorded
technical failure permits retry, never slow performance. Chromium, capture and
server run on SSH loopback. Show actual arriving token chunks simultaneously;
no artificial typing queue or time compression. Retain raw events, token IDs,
timestamps, request/measurement counters and failed takes. Video measurements
are excluded single-case demonstrations, not four-block estimates.

## Deployment, execution and terminal boundary

Ruff, four CPU modules, four CLI smokes, 21-node plan, five cumulative patches
apply/compile, frontend and public-artifact checks precede CI-green deployment.
Maintain fast/preview branches and pending-review PR; no automatic merge/release.
Old periodic monitoring stays off. Prepare all code before paying for idle GPUs.

An exclusive server-local controller must run missing QA → cohort → 8B long →
DSpark → 27B → selected videos → reports/backup, refreshing each group's exact
configuration-bound acceptance before launch. QA must cover sampling, parameter
and optimizer reset, lifecycle, independent KV, all TP ranks and full budgets.
Missing proof is not a checkbox to default true. Reuse only equivalent evidence.
No full-DAG fallback is allowed, including after all 180 measurements complete.

Only complete independent units can use separate GPUs. Never split TP2/paired
blocks. Preserve attempts on interruption and resume pending identities only.
At completion—or no safe work except downloads/development—save/verify compact
evidence and API shutdown without releasing the instance. Models stay on server.

Long panels: 896 requests, 29,360,128 maximum output tokens. Cohort: 1728
requests, 3,538,944 maximum output tokens. DSpark/QA/anchors/videos are additional.
EOS reduces actual work. ETA is stratified by model, method, TP, state policy
and output length; missing strata stay unpriced. Update P50/P90 after a complete
pairing block; neither maximum token budgets nor 30-second snippets are a
measured complete-run ETA.
