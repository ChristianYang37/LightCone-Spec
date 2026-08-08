# CLI

[中文](../zh-CN/cli.md) · [Home](../../README.md)

The `lightcone-spec` command exposes the following workflow:

- `doctor`: read-only host and GPU compatibility report;
- `lock`: resolve mutable model, dataset, and environment inputs;
- `prepare-models` / `prepare-datasets`: materialize and verify locked inputs;
- `serve`: launch one validated adapted server;
- `run-manifest`: execute immutable units with isolated output paths;
- `replay`: oracle replay and controller fitting;
- `exactness`: proposal/accept/reject correctness checks;
- `analyze`: protocol tables and figures;
- `validate-artifacts`: recursively verify evidence.

Use `lightcone-spec COMMAND --help` for the authoritative arguments. A missing
lock, controller, selection artifact, or runtime binding is an actionable error,
not an empty successful run. Queue exit codes distinguish success, scientific
blocking, lock contention, and resumable engineering failure.
