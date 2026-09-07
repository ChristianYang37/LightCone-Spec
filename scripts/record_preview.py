#!/usr/bin/env python3
"""Excluded live c8 recording server. Run only in an explicitly isolated GPU window.

New contributions: LicenseRef-LightCone-Source-Available-1.0. See LICENSE.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.data import load_prompt_records
from lightcone_spec.metrics import SAFETY_COUNTERS, per_user_generation_speed
from lightcone_spec.preview import VIDEO_METHODS
from lightcone_spec.protocol import Job
from lightcone_spec.recording import StreamRecording
from lightcone_spec.runner import _fit_prompt, _runtime_job, _selection_for_job, _speed_metrics
from lightcone_spec.server import ServerProcess
from lightcone_spec.state import StateStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-index", type=int, choices=range(6), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    if args.gpu not in config.gpu_ids:
        parser.error("GPU must belong to the configured instance")
    args.output.mkdir(parents=True, exist_ok=False)
    state = StateStore(config.run_dir)
    backend, method, label = VIDEO_METHODS[args.method_index]
    job = Job(
        job_id=f"excluded-video-v1-{args.method_index}", node="excluded-video-v1", ordinal=0,
        method=method, model="Qwen/Qwen3-8B", backend=backend, task="LiveCodeBench",
        context=17408, load="c8", width=16 if backend != "NONE" else None,
        parameters={"excluded_from_analysis": True, "coverage_runtime": True,
                    "execution_request_count": 8, "stride": 10, "generation_tokens": 1024},
    )
    job = _runtime_job(config, state, job)
    selection = _selection_for_job(state, job)
    if method in {"lightcone", "tts_lora_batched"} and not selection:
        raise RuntimeError("video requires frozen adaptation recipe")
    (args.output / "server").mkdir()
    process = ServerProcess(config, job, gpus=(args.gpu,), port=config.server.base_port + 20,
                            output_dir=args.output / "server", selection=selection)
    with process as client:
        records = load_prompt_records(config.dataset_path("LiveCodeBench"), limit=8, selection_seed=0)
        tokens = [client.tokenize(row["prompt"]) for row in records]
        filler = tuple(token for row in tokens for token in row)
        prompts = [_fit_prompt(row, filler, 16384) for row in tokens]
        client.run_batch(prompts[:1], max_new_tokens=16, seed=0)
        client.reset()
        sink = StreamRecording(args.output / "stream-events.jsonl")
        status = {"status": "ready", "label": label, "dispatcher_concurrency": 8,
                  "input_tokens_per_request": 16384, "max_output_tokens": 1024,
                  "recording_scope": "excluded; independent real run; playback 1x",
                  "context_construction": "repeated workload prefix; identical across methods"}
        start_lock = threading.Lock()

        def generate():
            try:
                before = _speed_metrics(client.server_info(), "tp1_dp1")
                client.stream_observer = sink
                results, duration = client.run_batch(
                    prompts, max_new_tokens=1024, seed=0, ignore_eos=False,
                    request_id_prefix=f"video-{args.method_index}",
                )
                sink.close()
                after = _speed_metrics(client.server_info(), "tp1_dp1")
                safety = {key: after[key] - before[key] for key in SAFETY_COUNTERS}
                if any(value for key, value in safety.items() if key != "retractions"):
                    raise RuntimeError(f"video safety counters: {safety}")
                if method != "static" and method != "target_only" and after["updates_published"] <= before["updates_published"]:
                    raise RuntimeError("video adaptive run published no update")
                status.update(status="completed", duration_seconds=duration,
                              aggregate_tok_s=sum(row.completion_tokens for row in results) / duration,
                              per_user_tok_s=per_user_generation_speed(row.to_dict() for row in results), safety=safety)
                (args.output / "requests.json").write_text(json.dumps([row.to_dict() for row in results]))
            except Exception as error:
                status.update(status="failed", error=str(error))
            finally:
                client.stream_observer = None
                if not sink.closed:
                    try:
                        sink.close()
                    except RuntimeError as error:
                        status.update(status="failed", error=str(error))
                (args.output / "recording.json").write_text(json.dumps(status, indent=2))

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    payload = Path(__file__).with_name("preview_stream.html").read_bytes()
                    content_type = "text/html; charset=utf-8"
                elif parsed.path == "/events":
                    since = int(parse_qs(parsed.query).get("since", [0])[0])
                    payload = json.dumps({**status, "events": sink.snapshot(since)}).encode()
                    content_type = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                if self.path != "/start":
                    self.send_error(404)
                    return
                with start_lock:
                    if status["status"] != "ready":
                        self.send_error(409)
                        return
                    status.update(status="running", started_at=time.time())
                    threading.Thread(target=generate, daemon=True).start()
                self.send_response(202)
                self.end_headers()

        (args.output / "job.json").write_text(json.dumps(job.to_dict(), indent=2))
        print(f"Recording ready on loopback port {args.port}; connect via SSH tunnel", flush=True)
        http = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        try:
            http.serve_forever()
        finally:
            http.server_close()


if __name__ == "__main__":
    main()
