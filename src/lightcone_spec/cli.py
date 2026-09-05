"""Four-command public interface for the paper experiments."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

from .config import ExperimentConfig
from .metrics import summarize_metric_rows
from .protocol import (
    PAPER_NODES,
    Job,
    materialize,
    mechanism_jobs,
    paper_plan,
    segment_count,
    source_coverage_jobs,
)
from .runner import PaperRunner, _metric_rows
from .state import StateStore


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _gzip_lines(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return sum(1 for _ in stream)


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
    summarize = subparsers.add_parser(
        "summarize", help="regenerate stage CSV and Parquet summaries"
    )
    summarize.add_argument("--run-dir", required=True, type=_absolute)
    return parser


def _plan(config: ExperimentConfig) -> None:
    print("node\trows\tgpus\tdescription")
    for row in paper_plan():
        print(f"{row.name}\t{row.rows}\t{row.gpu_count}\t{row.description}")
    pairs = tuple(zip(config.gpu_ids[::2], config.gpu_ids[1::2], strict=True))
    print(f"\ngpu_pairs\t{len(pairs)}\t{pairs}")
    print(f"max_parallel_blocks\t{len(pairs)}\tone clean block per TP2 pair")
    jobs = tuple(job for node in PAPER_NODES for job in materialize(node))
    segments = sum(segment_count(job) for job in jobs)
    finite_requests = sum(
        _request_floor(job, config.server.requests_per_cell)
        for job in jobs
        if not _time_driven(job)
    )
    print("\ninternal_substage\tjobs\trequest_basis")
    if len(pairs) > 1:
        print(
            f"GPU-pair-interference\t{2 * len(pairs)}\t"
            f"{len(pairs)} isolated + {len(pairs)} concurrent TP2 blocks"
        )
    print("E1-common-load\t<=10\t7 bundled loads * (2 baselines + 2 * <=4 finalists)")
    print("E3-width-calibration\t12\t4 methods * 3 widths, each with 3 segments")
    print("E6-common-load\t10\t2 models * 5 roles, each with 9 load segments")
    print("TTS-batched-calibration\t<=4\tE1 LoRA finalists at excluded c8")
    print("TTS-S10-confirmation\t8\t2 learning rates * 4 paired blocks")
    print("S10-reconciliation\t19\t14 formal-stride + 5 masked-logit replacements")
    source, mechanism = source_coverage_jobs(), mechanism_jobs()
    print(
        f"E0-source-four-block-v1\t{len(source)}\t4 independent paired blocks; max_requests={sum(job.parameters['execution_request_count'] for job in source)}"
    )
    print(
        f"E3b-mechanism-four-block-v1\t{len(mechanism)}\t4 independent paired blocks; diagnostic timing excluded"
    )
    print(
        f"registered finite-request floor\t{finite_requests}\t"
        f"requests_per_cell={config.server.requests_per_cell}; E5 time-driven rows excluded"
    )
    print(f"registered jobs\t{len(jobs)}\tsegments={segments}")
    print(
        f"coverage added leaves\t{len(source) + len(mechanism)}\tbase_plus_coverage={segments + len(source) + len(mechanism)}; replacements separate"
    )
    print(f"maximum runner jobs\t{len(jobs) + 95}\tincluding dynamic confirmations")
    acceptance = config.results_root / "acceptance"
    if acceptance.is_dir():
        files = tuple(path for path in acceptance.rglob("*") if path.is_file())
        attempts = sum(path.name == "metrics.json" for path in files)
        request_files = tuple(
            path
            for path in files
            if path.name
            in {
                "requests.jsonl.gz",
                "request_outcomes.jsonl.gz",
                "cycles.jsonl.gz",
            }
        )
        request_rows = sum(_gzip_lines(path) for path in request_files)
        variable_bytes = sum(path.stat().st_size for path in request_files)
        fixed_bytes = sum(path.stat().st_size for path in files) - variable_bytes
        if attempts and request_rows:
            projected = (
                len(jobs) * fixed_bytes // attempts
                + finite_requests * variable_bytes // request_rows
            )
            free = shutil.disk_usage(config.results_root).free
            print(f"result capacity lower bound\t{projected}\tfree={free}; source={acceptance}")
    else:
        print(
            "result capacity lower bound\tUNMEASURED\t"
            f"place acceptance raw attempts under {acceptance}"
        )


def _time_driven(job: Job) -> bool:
    load = job.load or ""
    registered = str(job.parameters.get("registered_load", load))
    return job.node.startswith("E5") and (
        load.startswith("closed_loop_c") or registered.startswith("closed_loop_c")
    )


def _request_floor(job: Job, base: int) -> int:
    if job.node == "TTS-Cal":
        return 76
    load = job.load or ""
    if load.startswith("closed_loop_c"):
        return max(base, int(load.removeprefix("closed_loop_c")))
    if load.startswith("c") and load[1:].isdigit():
        return max(base, int(load[1:]))
    return base


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
        frame = summarize_metric_rows(_metric_rows(state, node), run_dir / "stages" / node)
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
