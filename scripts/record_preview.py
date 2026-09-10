#!/usr/bin/env python3
"""Excluded live c8 recording server. Run only in an explicitly isolated GPU window.

New contributions: LicenseRef-LightCone-Source-Available-1.0. See LICENSE.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.metrics import SAFETY_COUNTERS, per_user_generation_speed
from lightcone_spec.preview import QWEN38_MODEL, QWEN38_VIDEO_METHODS, VIDEO_METHODS
from lightcone_spec.preview_revision import preview_lightcone_stride, preview_recipe
from lightcone_spec.protocol import Job
from lightcone_spec.recording import StreamRecording, recording_nvml_peaks, validate_recording
from lightcone_spec.runner import (
    _cell_inputs,
    _dispatcher_concurrency,
    _runtime_job,
    _selection_for_job,
    _speed_metrics,
)
from lightcone_spec.server import GpuSampler, ServerProcess
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
    parser.add_argument("--qa", action="store_true", help="Excluded full c8 validation; never grants acceptance")
    parser.add_argument("--preview-version", type=int, choices=(3, 4), default=3)
    parser.add_argument("--cohort", action="store_true", help="V4 selected full 48-request appendix")
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    if args.gpu not in config.gpu_ids:
        parser.error("GPU must belong to the configured instance")
    args.output.mkdir(parents=True, exist_ok=False)
    from lightcone_spec.preview_continuation import read_state
    selections, _, _ = read_state(config.run_dir)
    state = StateStore(args.output / "excluded-state")
    for name, value in selections.items():
        state.set_selection(name, value)
    video_key = "cohort_appendix" if args.cohort else args.model
    acceptance = state.selection(f"formal_preview_video_acceptance_v{args.preview_version}", {}).get(video_key, {})
    if not args.qa and (acceptance.get("status") != "accepted" or acceptance.get("tp") != args.tp):
        raise RuntimeError("video needs full-workload six-method common-TP acceptance")
    manifest = state.selection(f"formal_preview_manifest_v{args.preview_version}", {})
    if manifest.get("version") != args.preview_version or (not args.qa and acceptance.get("manifest") != manifest):
        raise RuntimeError("video requires matching versioned manifest acceptance")
    if args.cohort and args.preview_version != 4:
        raise ValueError("cohort appendix is registered only in v4")
    if preview_lightcone_stride(manifest) != 10:
        raise RuntimeError("video requires restored S10 preview manifest; migrate at idle boundary first")
    newer = args.model == QWEN38_MODEL
    backend, method, label = (QWEN38_VIDEO_METHODS if newer else VIDEO_METHODS)[args.method_index]
    prompts = manifest["qwen38"]["prompts"] if newer else manifest["prompts"]["LiveCodeBench"][:8]
    gpus = (args.gpu,) if args.tp == 1 else tuple(config.gpu_ids[:2])
    job = Job(
        job_id=f"excluded-video-v3-{'qwen38' if newer else 'qwen3'}-{args.method_index}", node="excluded-video-v3", ordinal=0,
        method=method, model=args.model, backend=backend, task="LiveCodeBench", gpu_count=args.tp,
        context=17408, load="c8", width=(4 if backend == "NEXTN" else 8 if newer else 16) if backend != "NONE" else None,
        parameters={"excluded_from_analysis": True, "coverage_runtime": True,
                    "panel": "preview_v1", "preview_panel": "excluded_video", "preview_revision": 3,
                    "preview_lightcone_stride": 10,
                    "topology": f"tp{args.tp}_dp1", "sampling_seed": 0,
                    "preview_prompt_records": prompts, "regime": "preview_constructed_chat",
                    "input_tokens": 16384, "enable_thinking": False, "respect_eos": True,
                    "execution_request_count": 8, "stride": 10, "generation_tokens": 1024,
                    "memory_budget_policy": "method_peak_v1" if newer else "fixed_reserve_v1",
                    "frozen_recipe": preview_recipe(method, backend, manifest)},
    )
    if args.preview_version == 4:
        from lightcone_spec.preview_v4 import video_job
        report = json.loads((config.run_dir / "stages/preview-v4/preview.json").read_text())
        job = video_job(manifest, report, args.model, args.method_index, args.tp, cohort=args.cohort)
        if not args.qa and job.to_dict() not in acceptance.get("jobs", []):
            raise RuntimeError("selected scene/video job differs from accepted full-condition QA")
        method, backend, label = job.method, job.backend, job.parameters["method_label"]
        prompts = job.parameters["preview_prompt_records"]
        if args.qa:
            job = replace(job, parameters={**job.parameters, "preview_reset_qa": True})
    job = _runtime_job(config, state, job)
    concurrency, request_count = (1, 48) if args.cohort else (8, 8)
    output_budget = 2048 if args.cohort else 1024
    if _dispatcher_concurrency(job) != concurrency or len(prompts) != request_count:
        raise RuntimeError("video actual concurrency/request count differs from frozen display")
    selection = _selection_for_job(state, job)
    if method in {"lightcone", "onlinespec_ens"} and not selection:
        raise RuntimeError("video requires frozen adaptation recipe")
    (args.output / "server").mkdir()
    if args.tp == 2 and args.preview_version == 3:
        from validate_preview_group import RankCompleteProcess
        os.environ["LIGHTCONE_EXCLUDED_VERIFY_TRACE"] = json.dumps({
            "output_directory": str((args.output / "server").resolve()), "rank_metrics": True, "trace_verify": False})
        os.environ["PYTHONPATH"] = os.pathsep.join((
            str(Path(__file__).resolve().parent / "preview_verify_trace"), os.environ.get("PYTHONPATH", "")))
    process_class = RankCompleteProcess if args.tp == 2 and args.preview_version == 3 else ServerProcess
    process = process_class(config, job, gpus=gpus, port=config.server.base_port + 20,
                            output_dir=args.output / "server", selection=selection)
    with process as client:
        if args.tp == 2:
            client = process.configure(job, selection)
        prompts, _, input_metadata = _cell_inputs(config, state, client, job)
        client.run_batch(prompts[:1], max_new_tokens=16, seed=0)
        client.reset()
        sink = StreamRecording(args.output / "stream-events.jsonl")
        status = {"status": "ready", "label": label, "model": job.model, "tp": len(process.gpus),
                  "dispatcher_concurrency": concurrency, "request_count": request_count,
                  "input_tokens_per_request": "native" if args.cohort else 16384, "max_output_tokens": output_budget,
                  "recording_scope": "excluded; independent real run; playback 1x",
                  "video_disclosure": job.parameters.get("video_disclosure"),
                  "context_construction": input_metadata.get("context_construction", "native input"),
                  "execution_gpu_ids": list(gpus)}
        start_lock = threading.Lock()

        def generate():
            sampler = GpuSampler(gpus, args.output / "measurement-gpu.csv", interval_seconds=.1)
            try:
                before = _speed_metrics(client.server_info(), f"tp{args.tp}_dp1")
                if len(before.get("rank_local", [])) != args.tp or any(
                    row.get(key) != 0 for row in before["rank_local"] for key in (*SAFETY_COUNTERS, "updates_published")
                ):
                    raise RuntimeError("video post-warmup reset telemetry is not clean")
                client.stream_observer = sink
                sampler.start()
                status["submitted_at_ns"] = time.time_ns()
                if args.cohort:
                    results = []
                    started = time.perf_counter()
                    for index, prompt in enumerate(prompts):
                        sequence_offset = sink.count
                        def observe(event, i=index, offset=sequence_offset):
                            sink({**event, "index": i, "sequence": offset + event["sequence"],
                                  "elapsed_seconds": time.perf_counter() - started,
                                  "request_local_elapsed_seconds": event["elapsed_seconds"]})
                        client.stream_observer = observe
                        rows, _ = client.run_batch((prompt,), max_new_tokens=output_budget,
                            seed=job.parameters["preview_request_seeds"][index], ignore_eos=False,
                            temperature=1.0, request_ids=(f"video-{args.method_index}-{index:05d}",))
                        results.extend(rows)
                    duration = time.perf_counter() - started
                else:
                    results, duration = client.run_batch(
                        prompts, max_new_tokens=output_budget, seed=0, ignore_eos=False,
                        temperature=1.0,
                        request_id_prefix=f"video-{args.method_index}",
                    )
                sink.close()
                sampler.stop()
                after = _speed_metrics(client.server_info(), f"tp{args.tp}_dp1")
                accounting = validate_recording(sink.events, [r.to_dict() for r in results], duration,
                                                expected_requests=request_count)
                ranks = after.get("rank_local", [])
                if len(ranks) != args.tp or any(r.get(k) != 0 for r in ranks for k in SAFETY_COUNTERS if k != "retractions"):
                    raise RuntimeError("video requires complete clean rank-local telemetry")
                safety = {key: after[key] - before[key] for key in SAFETY_COUNTERS}
                if any(value for key, value in safety.items() if key != "retractions"):
                    raise RuntimeError(f"video safety counters: {safety}")
                if method != "static" and method != "target_only" and after["updates_published"] <= before["updates_published"]:
                    raise RuntimeError("video adaptive run published no update")
                status.update(status="completed", duration_seconds=duration,
                              committed_tokens=sum(row.completion_tokens for row in results),
                              event_count=sink.count,
                              aggregate_tok_s=sum(row.completion_tokens for row in results) / duration,
                              accepted_drafts_per_target_call=(None if method == "target_only" else
                                  (after["accepted_drafts"] - before["accepted_drafts"]) /
                                  max(1, after["target_calls"] - before["target_calls"])),
                              peak_hbm_bytes=after.get("peak_hbm_bytes"),
                              per_user_tok_s=per_user_generation_speed(row.to_dict() for row in results), safety=safety)
                status.update(accounting, rank_memory=[{k: r.get(k) for k in (
                    "peak_hbm_bytes", "peak_hbm_reserved_bytes", "memory_budget", "memory_ledger",
                    "kv_token_capacity", "measured_update_peak_bytes")} for r in ranks],
                    memory_peak_scope="since post-warmup cache reset; NVML measurement-gpu.csv is generation only",
                    full_workload=True, reset_verified=True, formal_acceptance=False,
                    updates_published=after["updates_published"]-before["updates_published"])
                status.update(recording_nvml_peaks(args.output / "measurement-gpu.csv", gpus))
                if args.preview_version == 4 and args.qa and method not in {"static", "target_only"}:
                    reset_rows = [json.loads(line) for path in (args.output / "server").glob("rank-*-reset-qa.jsonl")
                                  for line in path.read_text().splitlines() if line]
                    if any(not any(r.get("tp_rank") == rank and r.get("passed") is True for r in reset_rows)
                           for rank in range(args.tp)) or any(r.get("passed") is not True for r in reset_rows):
                        raise RuntimeError("video QA missing physical reset evidence")
                    status["parameter_optimizer_reset"] = True
                if args.preview_version == 4 and method != "target_only":
                    calls = after["delivered_verification_calls"] - before["delivered_verification_calls"]
                    drafts = after["delivered_draft_tokens"] - before["delivered_draft_tokens"]
                    bonuses = after["delivered_bonus_tokens"] - before["delivered_bonus_tokens"]
                    if calls <= 0:
                        raise RuntimeError("video missing delivered verification counts")
                    status.update(accepted_drafts_per_target_call=drafts / calls,
                                  al_with_bonus=(drafts + bonuses) / calls)
                (args.output / "requests.json").write_text(json.dumps([row.to_dict() for row in results]))
            except Exception as error:
                status.update(status="failed", error=str(error))
            finally:
                sampler.stop()
                client.stream_observer = None
                if not sink.closed:
                    try:
                        sink.close()
                    except RuntimeError as error:
                        status.update(status="failed", error=str(error))
                (args.output / "recording.json").write_text(json.dumps(status, indent=2))

        (args.output / "job.json").write_text(json.dumps(job.to_dict(), indent=2))
        if args.qa:
            status["status"] = "running"
            generate()
            if status["status"] != "completed":
                raise RuntimeError(status.get("error", "video QA failed"))
            return

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
        print(f"Recording ready on loopback port {args.port}; run capture/Chromium on this SSH host", flush=True)
        http = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        try:
            http.serve_forever()
        finally:
            http.server_close()


if __name__ == "__main__":
    main()
