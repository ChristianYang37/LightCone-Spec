"""Four-command public interface for the paper experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .metrics import summarize_attempts
from .protocol import paper_plan
from .runner import PaperRunner
from .state import StateStore


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _config_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("run", help="start or resume the paper DAG")
    parser.add_argument("--config", required=True, type=_absolute)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lightcone-spec")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _config_parser(subparsers)
    plan = subparsers.add_parser("plan", help="print the registered DAG without using GPUs")
    plan.add_argument("--config", required=True, type=_absolute)
    status = subparsers.add_parser("status", help="show SQLite run progress")
    status.add_argument("--run-dir", required=True, type=_absolute)
    summarize = subparsers.add_parser("summarize", help="regenerate stage CSV and Parquet summaries")
    summarize.add_argument("--run-dir", required=True, type=_absolute)
    return parser


def _plan(config: ExperimentConfig) -> None:
    print("node\trows\tgpus\tdescription")
    for row in paper_plan(final_blocks=config.protocol.final_blocks):
        print(f"{row.name}\t{row.rows}\t{row.gpu_count}\t{row.description}")


def _status(run_dir: Path) -> None:
    state = StateStore(run_dir)
    rows = state.stage_rows()
    if not rows:
        print("No stages have been materialized.")
        return
    print("node\tstatus\tcompleted\tfailed\tskipped\ttotal")
    for row in rows:
        print(
            f"{row['node']}\t{row['status']}\t{row['completed'] or 0}\t"
            f"{row['failed'] or 0}\t{row['skipped'] or 0}\t{row['row_count']}"
        )


def _summarize(run_dir: Path) -> None:
    state = StateStore(run_dir)
    written: dict[str, int] = {}
    for row in state.stage_rows():
        node = str(row["node"])
        frame = summarize_attempts(state.completed_attempt_dirs(node), run_dir / "stages" / node)
        written[node] = len(frame)
    print(json.dumps(written, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        PaperRunner(ExperimentConfig.load(args.config)).run()
    elif args.command == "plan":
        _plan(ExperimentConfig.load(args.config))
    elif args.command == "status":
        _status(args.run_dir)
    elif args.command == "summarize":
        _summarize(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
