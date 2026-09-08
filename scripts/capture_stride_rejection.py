"""One excluded replay of a frozen failed audit cell; no performance acceptance."""

import argparse
import json
import os
import signal
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

from validate_preview_group import RankCompleteProcess

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.preview_continuation import continuation_lock
from lightcone_spec.runner import _execute_cell, _job_from_metric_config, _selection_for_job
from lightcone_spec.server import apply_runner_affinity
from lightcone_spec.state import StateStore


def capture_job(source):
    if not source.parameters.get("stride_audit_v1") or source.gpu_count != 2:
        raise ValueError("capture requires the frozen excluded TP2 audit row")
    return replace(source, job_id="capture__" + source.job_id, parameters={**source.parameters,
        "reconstruction_capture": True, "excluded_from_analysis": True,
        "measurement_scope": "failed_candidate_tensor_diagnosis_not_performance",
        "replays_job_id": source.job_id})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--from-round", type=int, default=4000)
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    job = capture_job(_job_from_metric_config(json.loads(args.source_job.read_text())))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "job.json").write_text(json.dumps(job.to_dict(), indent=2))
    snapshot_dir = args.output / "snapshots"
    with continuation_lock(config.run_dir / "preview-continuation.lock"):
        with sqlite3.connect(f"file:{config.run_dir / 'state.sqlite'}?mode=ro", uri=True) as db:
            if db.execute("pragma integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("formal SQLite integrity failed")
            selections = {n: json.loads(v) for n, v in db.execute("select name,value_json from selections")}
        config = replace(config, results_root=args.output, run_name="excluded")
        state = StateStore(config.run_dir)
        for name, value in selections.items():
            state.set_selection(name, value)
        state.add_internal_jobs((job,))
        server_dir = args.output / "server"
        server_dir.mkdir()
        repo = Path(__file__).resolve().parents[1]
        os.environ.update(LIGHTCONE_DFLASH_RECONSTRUCTION_DIR=str(snapshot_dir),
                          LIGHTCONE_DFLASH_CAPTURE_FROM_ROUND=str(args.from_round),
                          LIGHTCONE_TIMING_AUDIT=json.dumps({"output_directory": str(server_dir), "mode": "off"}),
                          LIGHTCONE_NUMA_ISOLATION="1",
                          PYTHONPATH=os.pathsep.join((str(repo / "scripts/timing_hooks"), str(repo / "src"))))
        apply_runner_affinity(config.gpu_ids, config.run_dir / "numa-affinity.json")
        selection = _selection_for_job(state, job)
        process = RankCompleteProcess(config, job, gpus=config.gpu_ids[:2], port=config.server.base_port + 60,
                                      output_dir=server_dir, selection=selection)
        done = threading.Event()
        captured = threading.Event()

        def watch():
            previous = None
            stable = 0
            while not done.wait(1):
                files = sorted(snapshot_dir.glob("candidate-*.pt"))
                sizes = tuple((p.name, p.stat().st_size) for p in files)
                stable = stable + 1 if sizes == previous and len(files) == 2 else 0
                previous = sizes
                # Each saved snapshot is followed by an atomic completion marker.
                if stable >= 5 and len(list(snapshot_dir.glob("candidate-*.complete"))) == 2:
                    captured.set()
                    os.kill(os.getpid(), signal.SIGINT)
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            with process:
                _execute_cell(config, state, job, gpus=config.gpu_ids[:2], selection=selection, server=process)
        except KeyboardInterrupt:
            if not captured.is_set():
                raise
        finally:
            done.set()
            watcher.join(timeout=2)
        if not captured.is_set():
            raise RuntimeError("capture did not reach a two-rank rejected candidate; inspect evidence")
        (args.output / "result.json").write_text(json.dumps({"status": "captured_rejection",
            "formal_acceptance": False, "performance_acceptance": False,
            "source_job": job.parameters["replays_job_id"], "snapshot_directory": str(snapshot_dir)}))


if __name__ == "__main__":
    main()
