"""Excluded full-budget block-0 QA for the remaining preview comparison groups."""

import argparse
import gzip
import json
import os
import re
from dataclasses import replace
from pathlib import Path

from validate_preview_group import RankCompleteProcess

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.metrics import SAFETY_COUNTERS
from lightcone_spec.preview import preview_jobs
from lightcone_spec.preview_continuation import read_state
from lightcone_spec.runner import _execute_cell, _runtime_job, _selection_for_job
from lightcone_spec.server import apply_runner_affinity
from lightcone_spec.state import StateStore


def remaining_cases(manifest, node):
    if node not in ("E5-preview-v3", "Qwen38-preview-v3"):
        raise ValueError("remaining QA cannot rerun the first forty cells")
    return tuple(j for j in preview_jobs(manifest) if j.node == node and j.block == 0)


def review_case(metrics, requests, job):
    ranks = metrics.get("rank_local_after", [])
    if len(ranks) != job.gpu_count:
        raise RuntimeError("missing rank-local QA evidence")
    if any(row.get(key) != 0 for row in ranks for key in SAFETY_COUNTERS if key != "retractions"):
        raise RuntimeError("rank-local numerical/version/safety failure")
    outcomes = metrics.get("request_outcomes", {})
    if any(outcomes.get(k, 0) for k in ("error", "cancelled")):
        raise RuntimeError("QA request runtime failure")
    if metrics.get("hard_feasible") is not True:
        # Preserve a capacity outcome, never turn a runtime/numerical failure into one.
        if metrics.get("capacity_feasible") is False and any(
            outcomes.get(k, 0) for k in ("timed_out", "unfinished")
        ):
            return "capacity_infeasible"
        raise RuntimeError("full-condition QA not hard feasible")
    if not requests or len({r["request_id"] for r in requests}) != len(requests):
        raise RuntimeError("missing or duplicate QA requests")
    if job.node == "Qwen38-preview-v3" and len(requests) != 8:
        raise RuntimeError("27B QA did not complete the frozen eight requests")
    for row in requests:
        count = row["completion_tokens"]
        if (not row.get("stop_reason") or len(row.get("output_ids", [])) != count
                or len(row.get("native_token_timestamps_ns", [])) != count):
            raise RuntimeError("QA native trajectory/stop accounting mismatch")
    if job.method in ("lightcone", "onlinespec_ens"):
        stride = 1 if job.method == "lightcone" else 10
        if metrics.get("resolved_stride") != stride or metrics.get("updates_published", 0) < 1:
            raise RuntimeError("QA frozen stride/publication mismatch")
    return "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--node", required=True, choices=("E5-preview-v3", "Qwen38-preview-v3"))
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--anchor", choices=("static", "target_only"))
    args = parser.parse_args()
    original = ExperimentConfig.load(args.config)
    selections, _, active = read_state(original.run_dir)
    if active:
        raise RuntimeError("formal jobs still active; QA requires an exclusive boundary")
    manifest = json.loads(args.manifest.read_text())
    source = remaining_cases(manifest, args.node)[args.case]
    if args.anchor:
        if source.method != "static" or not source.load.startswith("closed_loop_c"):
            raise ValueError("matched TP2 anchors require existing Static closed-loop conditions")
        source = replace(source, job_id=f"supplement-tp2-anchor-{args.anchor}-{source.load}",
                         method=args.anchor, backend="NONE" if args.anchor == "target_only" else source.backend,
                         width=None if args.anchor == "target_only" else source.width, gpu_count=2,
                         parameters={**source.parameters, "topology": "tp2_dp1", "frozen_recipe": None})
    job = replace(source, job_id="excluded-remaining__" + source.job_id,
                  parameters={**source.parameters, "excluded_from_analysis": True,
                              "qa_source_job_id": source.job_id})
    args.output.mkdir(parents=True, exist_ok=False)
    config = replace(original, results_root=args.output, run_name="excluded")
    state = StateStore(config.run_dir)
    for name, value in selections.items():
        state.set_selection(name, value)
    state.set_selection("formal_preview_manifest_v3", manifest)
    state.add_internal_jobs((job,))
    server_dir = args.output / "server"
    server_dir.mkdir()
    (args.output / "qa-job.json").write_text(json.dumps(source.to_dict(), indent=2))
    gpus = config.gpu_ids[:source.gpu_count]
    os.environ["LIGHTCONE_EXCLUDED_VERIFY_TRACE"] = json.dumps({
        "output_directory": str(server_dir.resolve()), "rank_metrics": True, "trace_verify": False})
    os.environ["PYTHONPATH"] = os.pathsep.join((
        str(Path(__file__).resolve().parent / "preview_verify_trace"), os.environ.get("PYTHONPATH", "")))
    apply_runner_affinity(config.gpu_ids, config.run_dir / "numa-affinity.json")
    runtime = _runtime_job(config, state, job)
    selection = _selection_for_job(state, runtime)
    process = RankCompleteProcess(config, runtime, gpus=gpus, port=config.server.base_port + 60,
                                  output_dir=server_dir, selection=selection)
    try:
        with process:
            _execute_cell(config, state, job, gpus=gpus, selection=selection, server=process)
        directory = state.completed_attempt_dir(job.job_id)
        if directory is None:
            raise RuntimeError("QA did not complete; inspect retained attempt")
        metrics = json.loads((directory / "metrics.json").read_text())
        with gzip.open(directory / "requests.jsonl.gz", "rt") as stream:
            requests = [json.loads(line) for line in stream if line.strip()]
        status = review_case(metrics, requests, source)
        binding = json.loads((server_dir / "observed-binding.json").read_text())
        if binding["execution_gpu_ids"] != list(gpus):
            raise RuntimeError("QA GPU binding mismatch")
        result = {"status": status, "formal_acceptance": False, "job": source.to_dict(),
                  "metrics_path": str(directory / "metrics.json"), "request_count": len(requests),
                  "correct": status == "passed", "full_workload": True, "reset_verified": True,
                  "gpu_binding_verified": True, "tp": source.gpu_count,
                  "case": f"{source.backend}:{source.method}:{source.load}"}
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
    except BaseException as error:
        (args.output / "failure.json").write_text(json.dumps({
            "error": str(error), "type": type(error).__name__, "formal_acceptance": False}, indent=2))
        # Only explicit allocator/KV admission failures authorize trying TP2.
        # Budget-estimator overflow, safety or arbitrary startup errors never do.
        message = str(error)
        if re.search(r"CUDA out of memory|torch\.OutOfMemoryError|leave no GPU memory for the KV cache", message):
            (args.output / "result.json").write_text(json.dumps({
                "status": "capacity_infeasible", "correct": False, "job": source.to_dict(),
                "tp": source.gpu_count, "formal_acceptance": False,
                "case": f"{source.backend}:{source.method}:{source.load}", "error": message}, indent=2))
            return
        raise


if __name__ == "__main__":
    main()
