# LightCone-Spec

LightCone-Spec is the experiment implementation for online drafter adaptation
in speculative decoding. It compares six distinct roles: Target-only, Static,
TTS, L0-naive, LightCone, and an independently tuned OnlineSPEC baseline.

The repository is intentionally small. The paper protocol is ordinary Python,
the run state is SQLite, and the SGLang changes are five plain unified diffs.
Repository state and source/model/data byte identity are not inspected in the
execution path.

> Formal E0--E6 status: **UNMEASURED**. CPU tests and GPU acceptance do not
> establish paper speed, capacity, or transfer results.

## One-command paper run

The intended machine is one host with an even number of GPUs; adjacent IDs form
TP2 pairs. The current acceptance host has two RTX PRO 6000 Blackwell 96 GB GPUs.
Models, datasets, the Python environment, profilers, and a local SGLang checkout
must already exist at the absolute paths in the configuration.

Raw benchmark downloads are converted once with `scripts/prepare_datasets.py`.
Formal pools need only a unique `problem_id` and a renderable `prompt` or
`turns`. They are workload stimuli: the runner does not execute answers, score
task accuracy, or call an LLM judge. Recipe selection uses a fixed 76-prompt
`CalibrationMix`: 24 APPS train, 24 OpenR1-Math train, 24 UltraChat train, and
four controlled synthetic prompts.

```bash
cp examples/paper.yaml /root/lightcone-tts-runtime/paper.yaml
# Edit only the absolute local paths and runtime settings.
./run_paper.sh /root/lightcone-tts-runtime/paper.yaml
```

On the first invocation, `run_paper.sh` checks and applies the five SGLang
diffs in lexical order, writes `paper-v1-nextn-shadow-v18` to
`.lightcone-spec-patched` in the SGLang checkout,
and starts the paper runner. Later invocations use the marker and resume the
same `run_name` from `state.sqlite`; a small import smoke confirms the required
runtime APIs. A dirty project or SGLang worktree does not
block execution, although a conflicting SGLang edit can make `git apply
--check` fail with a normal patch conflict.

The public CLI has four commands:

```bash
lightcone-spec plan --config /absolute/paper.yaml
lightcone-spec run --config /absolute/paper.yaml
lightcone-spec status --run-dir /absolute/results/run-name
lightcone-spec summarize --run-dir /absolute/results/run-name
```

`plan` is CPU-only. `run` creates or resumes a run. `status` reads SQLite, and
`summarize` rebuilds per-stage CSV and Parquet tables from completed attempts.

## Paper protocol

The fixed 21-node order is:

```text
preflight -> E3a -> TTS-Cal -> E1 -> E2-r0 -> E2-r1 -> E2-r2 -> E2-r3
-> E4-screen -> E4-local -> E4-profile -> E3b-pilot -> E3b-final -> E1a
-> E5-pilot -> E5-final -> E6-pilot -> E6-final
-> E0-tune -> E0-pilot -> E0-final
```

The `paper-v2` preset registers 2,185 static jobs and at most 2,280 unique
runner jobs after bounded finalist, load-selection, and formal-S=10
reconciliation work. The one E4-profile retry is a new attempt of its existing
job, not a new materialized job. A job is one
`method × block × compatible server layout`; its `segments` cover the
registered contexts, loads, traces, or workload pools without reloading the
same model. The per-node job counts are 10, 140, 72 (at most 108 after TTS
finalist confirmation), 68 (at most 100 after Pareto confirmation), 424, 109,
31, 25, 52, 168, 3, 20, 132, 141, 11, 66, 22, 60, 287, 86, and 258.
Primary E3b/E5 conclusions use 12 clean-server blocks; E4/E6/E0 secondary
evidence uses six. There is no global power-selected repeat count.

The runner preserves the paper gates that matter scientifically: proposal
version consistency, controlled TTS/L0 candidate replay, finite optimization
state, HBM/KV feasibility, safety events, committed-token goodput, pilot/final
block separation, paired statistics, and both Target-only and Static baselines.
Exact rejection-sampling semantics receive one excluded implementation smoke;
they are not remeasured in every formal cell.

Formal adaptive evidence fixes the publication stride at `S=10` for TTS,
L0-naive, LightCone, LightCone-candidate, DSpark-LightCone, and
TTS-LoRA-Batched. The original TTS-Cal and E4 stride grids remain visible as
exploratory ablations. A bounded four-block `S=10` comparison freezes the TTS
learning rate, and multi-turn request-scoped runs reset only after the complete
conversation.

See [docs/protocol.md](docs/protocol.md) for the grid and dependency behavior.

## Crash recovery and outputs

SQLite uses WAL mode and stores `jobs`, `attempts`, `selections`, and
`stage_state`. At startup, interrupted `running` jobs return to `pending`.
Each retry gets a separate attempt directory; only completed attempts enter
analysis. Process or network failures retry once. Numerical failures, OOM,
exactness violations, and failed scientific gates do not retry automatically.

```text
results/run-name/
├── environment.json
├── paper.yaml
├── state.sqlite
├── jobs/<readable-job-id>/attempt-01/
│   ├── config.json
│   ├── requests.jsonl.gz
│   ├── request_outcomes.jsonl.gz
│   ├── cycles.jsonl.gz
│   ├── metrics.json
│   ├── server.log.gz
│   └── gpu.csv.gz
└── stages/<node>/
    ├── summary.csv
    ├── summary.parquet
    └── statistics.json   # final nodes
```

Each independent statistical block starts a fresh server, waits for health,
warms up, measures, and shuts it down. Layout-compatible cells in that block
reuse the model process through an idle-only configure/reset endpoint; a layout,
parallelism, model, backend, width, profiler, or fault-device change starts a new
process. Paired methods stay ordered on the same GPU. Two independent single-GPU
queues may overlap; TP2, DP2, E5, and E6 cells reserve both devices. Preflight
measures paired goodput and scheduler-commit ITL interference with relative BCa intervals.
Headline blocks overlap only when both mean effects are within 1% and both
intervals include zero.

## Development checks

```bash
python -m pip install -e '.[dev]'
ruff check src tests scripts
pytest -q
lightcone-spec --help
```

Default CI runs Ruff, the four CPU test modules, and CLI smoke only. GPU smoke
is a separate manual acceptance step and requires explicit operator approval.

## Evidence boundary

The experiment package records named models and workload pools, the YAML
snapshot, software versions, seed, compressed numeric request/cycle rows,
server/GPU logs, completed attempts, block structure, and statistical outputs.
Formal rows omit generated text and token trajectories. The runner does not
establish the cryptographic identity of local files.

See [docs/running.md](docs/running.md), [docs/evidence.md](docs/evidence.md),
and [docs/status.md](docs/status.md).
