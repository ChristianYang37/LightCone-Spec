#!/usr/bin/env python3
"""Excluded live c8 recording server. Run only in an explicitly isolated GPU window.

New contributions: LicenseRef-LightCone-Source-Available-1.0. See LICENSE.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.metrics import SAFETY_COUNTERS, per_user_generation_speed
from lightcone_spec.preview import QWEN38_MODEL, QWEN38_VIDEO_METHODS, VIDEO_METHODS
from lightcone_spec.protocol import Job
from lightcone_spec.recording import StreamRecording
from lightcone_spec.runner import _cell_inputs, _runtime_job, _selection_for_job, _speed_metrics
from lightcone_spec.server import ServerProcess
from lightcone_spec.state import StateStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-index", type=int, choices=range(6), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", choices=("Qwen/Qwen3-8B", QWEN38_MODEL), default="Qwen/Qwen3-8B")
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    if args.gpu not in config.gpu_ids:
        parser.error("GPU must belong to the configured instance")
    args.output.mkdir(parents=True, exist_ok=False)
    state = StateStore(config.run_dir)
    acceptance = state.selection("formal_preview_video_acceptance_v2", {}).get(args.model, {})
    if acceptance.get("status") != "accepted" or acceptance.get("tp") != args.tp:
        raise RuntimeError("video needs full-workload six-method common-TP acceptance")
    manifest = state.selection("formal_preview_manifest_v2", None) or state.selection("formal_preview_manifest_v1", {})
    newer = args.model == QWEN38_MODEL
    backend, method, label = (QWEN38_VIDEO_METHODS if newer else VIDEO_METHODS)[args.method_index]
    prompts = manifest["qwen38"]["prompts"] if newer else manifest["prompts"]["LiveCodeBench"][:8]
    gpus = (args.gpu,) if args.tp == 1 else tuple(config.gpu_ids[:2])
    job = Job(
        job_id=f"excluded-video-v2-{'qwen38' if newer else 'qwen3'}-{args.method_index}", node="excluded-video-v2", ordinal=0,
        method=method, model=args.model, backend=backend, task="LiveCodeBench", gpu_count=args.tp,
        context=17408, load="c8", width=(4 if backend == "NEXTN" else 8 if newer else 16) if backend != "NONE" else None,
        parameters={"excluded_from_analysis": True, "coverage_runtime": True,
                    "panel": "preview_v1", "preview_panel": "excluded_video",
                    "topology": f"tp{args.tp}_dp1", "sampling_seed": 0,
                    "preview_prompt_records": prompts, "regime": "preview_constructed_chat",
                    "input_tokens": 16384, "enable_thinking": False, "respect_eos": True,
                    "execution_request_count": 8, "stride": 10, "generation_tokens": 1024,
                    "memory_budget_policy": "method_peak_v1",
                    "frozen_recipe": acceptance.get("recipes", {}).get(method)},
    )
    job = _runtime_job(config, state, job)
    selection = _selection_for_job(state, job)
    if method in {"lightcone", "tts_lora_batched"} and not selection:
        raise RuntimeError("video requires frozen adaptation recipe")
    (args.output / "server").mkdir()
    process = ServerProcess(config, job, gpus=gpus, port=config.server.base_port + 20,
                            output_dir=args.output / "server", selection=selection)
    with process as client:
        prompts, _, input_metadata = _cell_inputs(config, state, client, job)
        client.run_batch(prompts[:1], max_new_tokens=16, seed=0)
        client.reset()
        sink = StreamRecording(args.output / "stream-events.jsonl")
        status = {"status": "ready", "label": label, "model": job.model, "tp": len(process.gpus),
                  "dispatcher_concurrency": 8,
                  "input_tokens_per_request": 16384, "max_output_tokens": 1024,
                  "recording_scope": "excluded; independent real run; playback 1x",
                  "context_construction": input_metadata["context_construction"],
                  "execution_gpu_ids": list(gpus)}
        start_lock = threading.Lock()

        def generate():
            try:
                before = _speed_metrics(client.server_info(), f"tp{args.tp}_dp1")
                client.stream_observer = sink
                status["submitted_at_ns"] = time.time_ns()
                results, duration = client.run_batch(
                    prompts, max_new_tokens=1024, seed=0, ignore_eos=False,
                    temperature=1.0,
                    request_id_prefix=f"video-{args.method_index}",
                )
                sink.close()
                after = _speed_metrics(client.server_info(), f"tp{args.tp}_dp1")
                safety = {key: after[key] - before[key] for key in SAFETY_COUNTERS}
                if any(value for key, value in safety.items() if key != "retractions"):
                    raise RuntimeError(f"video safety counters: {safety}")
                if method != "static" and method != "target_only" and after["updates_published"] <= before["updates_published"]:
                    raise RuntimeError("video adaptive run published no update")
                status.update(status="completed", duration_seconds=duration,
                              committed_tokens=sum(row.completion_tokens for row in results),
                              event_count=sink.count,
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
                if parsed.path in {"/clock", "/status"}:
                    payload = json.dumps({**(dict(status) if parsed.path == "/status" else {}),
                                          "recording_hostname": socket.gethostname(),
                                          "server_epoch_ns": str(time.time_ns())}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if parsed.path == "/stream":
                    try:
                        cursor = int(self.headers.get("Last-Event-ID", "0"))
                        sink.wait_since(cursor, timeout=0)
                    except ValueError:
                        self.send_error(400, "Invalid event cursor")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    prior = None
                    try:
                        while True:
                            for event in sink.wait_since(cursor):
                                if event["sequence"] != cursor + 1:
                                    raise RuntimeError("recorded stream sequence gap")
                                cursor += 1
                                self.wfile.write((f"id: {cursor}\nevent: token\ndata: " +
                                                  json.dumps(event) + "\n\n").encode())
                            snapshot = dict(status)
                            # Completion may race the last snapshot: drain the
                            # writer's final events before sending terminal state.
                            if snapshot["status"] == "completed" and cursor != snapshot["event_count"]:
                                continue
                            serialized = json.dumps(snapshot)
                            if serialized != prior:
                                self.wfile.write(("event: state\ndata: " + serialized + "\n\n").encode())
                                prior = serialized
                            else:
                                self.wfile.write(b": heartbeat\n\n")
                            self.wfile.flush()
                            if snapshot["status"] in {"completed", "failed"}:
                                return
                    except (BrokenPipeError, ConnectionResetError):
                        # EventSource reconnects using Last-Event-ID; raw events
                        # remain in the sink and generation is unaffected.
                        return
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
