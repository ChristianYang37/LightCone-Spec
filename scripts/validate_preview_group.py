"""Excluded full-condition 8B long-generation QA; never opens formal acceptance."""

import argparse
import gzip
import json
import os
import sqlite3
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from validate_preview_updates import captured_rank_info

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.metrics import SAFETY_COUNTERS
from lightcone_spec.preview import preview_jobs
from lightcone_spec.runner import _execute_cell, _selection_for_job
from lightcone_spec.server import ServerProcess, apply_runner_affinity
from lightcone_spec.state import StateStore

CASES = {
    "target": ("target_only", "NONE"),
    "eagle3": ("static", "EAGLE3"),
    "dflash": ("static", "DFLASH"),
    "ensemble": ("onlinespec_ens", "DFLASH"),
    "lightcone": ("lightcone", "DFLASH"),
}


def qa_manifest(selections):
    """An excluded candidate may precede formal v3 freezing; never write it back."""
    if "formal_preview_manifest_v3" in selections:
        manifest = deepcopy(selections["formal_preview_manifest_v3"])
        if manifest.get("version") != 3:
            raise ValueError("invalid formal v3 manifest; no legacy fallback")
        return manifest, "formal_preview_manifest_v3"
    source = selections.get("formal_preview_manifest_v1")
    if not isinstance(source, dict) or source.get("version") not in (1, 2):
        raise ValueError("missing frozen preview input manifest")
    candidate = deepcopy(source)
    candidate["version"] = 3
    candidate.pop("tts_recipe", None)
    return candidate, "excluded_candidate_from_formal_preview_manifest_v1"


def full_condition_job(manifest, task, case, tp):
    """Clone a block-0 row, changing only topology and excluded provenance."""
    if manifest.get("version") != 3 or task not in ("MATH-500", "LiveCodeBench"):
        raise ValueError("requires frozen v3 long-generation prompts")
    if case not in CASES or type(tp) is not int or tp not in (1, 2):
        raise ValueError("unknown case or topology")
    frozen = deepcopy(manifest)
    frozen.setdefault("comparison_topologies", {})["dflash_long"] = tp
    method, backend = CASES[case]
    source = next(j for j in preview_jobs(frozen) if j.task == task and j.block == 0
                  and j.parameters["preview_panel"] == "long_generation"
                  and (j.method, j.backend) == (method, backend))
    assert source.parameters["execution_request_count"] == 8
    assert source.parameters["generation_tokens"] == 32768 and source.parameters["respect_eos"]
    return replace(source, job_id="excluded-full-condition__" + source.job_id,
                   parameters={**source.parameters, "excluded_from_analysis": True,
                               "qa_source_job_id": source.job_id})


class RankCompleteProcess(ServerProcess):
    """Excluded rank telemetry before the API's DP-leader filtering."""

    def configure(self, job, selection):
        client = super().configure(job, selection)
        entries = Path(f"/proc/{self.process.pid}/environ").read_bytes().split(b"\0")
        if ("CUDA_VISIBLE_DEVICES=" + ",".join(map(str, self.gpus))).encode() not in entries:
            raise RuntimeError("full-condition GPU binding mismatch")
        (self.output_dir / "observed-binding.json").write_text(json.dumps({
            "execution_gpu_ids": self.gpus, "pid": self.process.pid,
            "cpu_affinity": sorted(os.sched_getaffinity(self.process.pid))}))
        if len(self.gpus) == 2 and not getattr(client, "_excluded_rank_complete", False):
            original = client.server_info

            def info():
                started = time.time_ns()
                original()
                return captured_rank_info(self.output_dir, 2, started)

            client.server_info = info
            client._excluded_rank_complete = True
        return client


def full_condition_result(state, job):
    directory = state.completed_attempt_dir(job.job_id)
    if directory is None:
        raise RuntimeError("full-condition cell did not complete; inspect preserved attempt")
    metrics = json.loads((directory / "metrics.json").read_text())
    with gzip.open(directory / "requests.jsonl.gz", "rt") as stream:
        requests = [json.loads(line) for line in stream if line.strip()]
    if metrics.get("hard_feasible") is not True or len(requests) != 8:
        raise RuntimeError("full-condition requests/safety did not pass")
    ranks = metrics.get("rank_local_after", [])
    if len(ranks) != job.gpu_count or any(row.get(key) != 0 for row in ranks for key in SAFETY_COUNTERS):
        raise RuntimeError("full-condition rank-local safety telemetry did not pass")
    if job.method in ("lightcone", "onlinespec_ens") and metrics.get("updates_published", 0) < 1:
        raise RuntimeError("adaptive full-condition QA published no updates")
    return {"status": "passed_excluded_full_condition", "formal_acceptance": False,
            "source_job_id": job.parameters["qa_source_job_id"], "task": job.task,
            "method": job.method, "backend": job.backend, "tp": job.gpu_count,
            "request_count": len(requests), "normal_eos": True, "output_limit": 32768,
            "completion_tokens": [r["completion_tokens"] for r in requests],
            "updates_published": metrics.get("updates_published", 0),
            "metrics_path": str(directory / "metrics.json"),
            "scope": "one full-budget block-0 case, not the complete common-TP panel or formal evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=tuple(CASES))
    parser.add_argument("--task", required=True, choices=("MATH-500", "LiveCodeBench"))
    parser.add_argument("--tp", required=True, type=int, choices=(1, 2))
    args = parser.parse_args()
    original = ExperimentConfig.load(args.config)
    if len(original.gpu_ids) < args.tp:
        raise ValueError("insufficient registered GPUs")
    # Never construct StateStore on the formal run: even its initializer writes.
    with sqlite3.connect(f"file:{original.run_dir / 'state.sqlite'}?mode=ro", uri=True) as source:
        selections = {name: json.loads(value) for name, value in
                      source.execute("SELECT name,value_json FROM selections")}
    manifest, provenance = qa_manifest(selections)
    job = full_condition_job(manifest, args.task, args.case, args.tp)
    args.output.mkdir(parents=True, exist_ok=False)
    config = replace(original, results_root=args.output, run_name="excluded")
    state = StateStore(config.run_dir)
    for name, value in selections.items():
        state.set_selection(name, value)
    state.add_internal_jobs((job,))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "manifest-provenance.json").write_text(json.dumps({
        "source": provenance, "formal_manifest_written": False,
        "scope": "excluded 8B candidate; topology is tested, not frozen by this script"}, indent=2))
    (args.output / "qa-job.json").write_text(json.dumps(job.to_dict(), indent=2))
    gpus = original.gpu_ids[:args.tp]
    server_dir = args.output / "server"
    server_dir.mkdir()
    if args.tp == 2:
        os.environ["LIGHTCONE_EXCLUDED_VERIFY_TRACE"] = json.dumps({
            "output_directory": str(server_dir.resolve()), "rank_metrics": True, "trace_verify": False})
        os.environ["PYTHONPATH"] = os.pathsep.join((
            str(Path(__file__).resolve().parent / "preview_verify_trace"), os.environ.get("PYTHONPATH", "")))
    apply_runner_affinity(original.gpu_ids, config.run_dir / "numa-affinity.json")
    selection = _selection_for_job(state, job)
    process = RankCompleteProcess(config, job, gpus=gpus,
                                  port=config.server.base_port + 60,
                                  output_dir=server_dir, selection=selection)
    try:
        # _execute_cell uses the unchanged real request budget, EOS, reset,
        # safety, native metrics and process retry paths. No shortened smoke.
        with process:
            _execute_cell(config, state, job, gpus=gpus, selection=selection, server=process)
        result = full_condition_result(state, job)
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
    except BaseException as error:
        (args.output / "failure.json").write_text(json.dumps({
            "type": type(error).__name__, "error": str(error), "formal_acceptance": False}, indent=2))
        raise


if __name__ == "__main__":
    main()
