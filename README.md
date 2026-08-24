# LightCone-Spec

LightCone-Spec is the experiment implementation for online drafter adaptation
in speculative decoding. It compares six distinct roles: Target-only, Static,
TTS, L0-naive, LightCone, and an independently tuned OnlineSPEC baseline.

The repository is intentionally small. The paper protocol is ordinary Python,
the run state is SQLite, and the SGLang changes are five plain unified diffs.
Repository state and source/model/data byte identity are not inspected in the
execution path.

> Formal E0--E6 status: **UNMEASURED**. GPU acceptance is incomplete; CPU tests
> do not establish speed, capacity, or quality results.

## One-command paper run

The intended machine is one host with two RTX PRO 6000 Blackwell 96 GB GPUs.
Models, datasets, the Python environment, profilers, and a local SGLang checkout
must already exist at the absolute paths in the configuration.

Raw benchmark downloads are converted once with
`scripts/prepare_datasets.py` and an explicit `task,problem_id,split` CSV.
Code benchmarks require `bwrap` and `prlimit`. Chat benchmarks run the
task-specific official evaluator command named by `LIGHTCONE_MT_BENCH_EVALUATOR`,
`LIGHTCONE_ALPACA_EVALUATOR`, or `LIGHTCONE_ARENA_HARD_EVALUATOR`. Evaluator
credentials remain process environment variables and are never stored.

```bash
cp examples/paper.yaml /root/lightcone-tts-runtime/paper.yaml
# Edit only the absolute local paths and runtime settings.
./run_paper.sh /root/lightcone-tts-runtime/paper.yaml
```

On the first invocation, `run_paper.sh` checks and applies the five SGLang
diffs in lexical order, writes `paper-v1-nextn-shadow-v3` to
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

The registered row counts are 10, 360, 288, 68, 3364, 844, 214, 57, 48, 96,
3, 1920, `480N`, 116, 2064, `450N`, 242, `60N`, `108+239V`, `64V`, and
`16VN`, where the four excluded E3b pilot blocks select one global `N` in
12--20 and `V` is the
number of executable E0 model/backend/task combinations.

The runner preserves the paper gates that matter scientifically: proposal
version consistency, controlled candidate replay, greedy token equality,
stochastic distributional exactness, finite optimization state, HBM/KV
feasibility, safety counters, committed-token goodput, pilot/final separation,
paired block statistics, and both Target-only and Static deployment baselines.

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
│   ├── requests.jsonl
│   ├── request_outcomes.jsonl
│   ├── cycles.jsonl
│   ├── metrics.json
│   ├── server.log
│   └── gpu.csv
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

The experiment package records named models and datasets, the YAML snapshot,
software versions, seed, raw request/cycle rows, server/GPU logs, completed
attempts, block structure, and statistical outputs. It does not establish the
cryptographic identity of local files. Exact local file identity therefore
remains an operator responsibility rather than a property of this runner.

See [docs/running.md](docs/running.md), [docs/evidence.md](docs/evidence.md),
and [docs/status.md](docs/status.md).
