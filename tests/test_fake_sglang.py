import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from lightcone_spec.client import GenerationResult, SGLangClient
from lightcone_spec.config import ExperimentConfig, ProtocolConfig, ServerConfig
from lightcone_spec.data import load_arrival_offsets, load_arrival_trace
from lightcone_spec.protocol import materialize
from lightcone_spec.runner import (
    _canonical_accuracy,
    _fault_action_passed,
    _run_request_scoped,
    _validate_committed_tokens,
)
from lightcone_spec.server import (
    ServerProcess,
    StickyReplicaClient,
    adaptation_payload,
    server_command,
)
from scripts.gpu_acceptance import _job as acceptance_job


class Handler(BaseHTTPRequestHandler):
    fail = False
    delay = 0.0
    batch_sizes = []

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path.startswith("/health"):
            self.send_response(200)
            self.end_headers()
            return
        if self.path == "/server_info":
            body = json.dumps({"speed_study_metrics": {"committed_tokens": 0}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        if Handler.fail:
            self.send_error(500)
            return
        length = int(self.headers.get("Content-Length", 0))
        if self.path == "/flush_cache?timeout=30":
            self.send_response(200)
            self.end_headers()
            return
        body = self.rfile.read(length)
        if self.path in {"/start_profile", "/stop_profile"}:
            self.send_response(200)
            self.end_headers()
            return
        request = json.loads(body)
        if self.path == "/tokenize":
            if request != {"prompt": "hello", "add_special_tokens": True}:
                self.send_error(400)
                return
            body = json.dumps({"tokens": [1, 2, 3], "count": 3}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/abort_request":
            self.send_response(200)
            self.end_headers()
            return
        if Handler.delay:
            time.sleep(Handler.delay)
        Handler.batch_sizes.append(len(request["rid"]))
        if any(
            "sampling_seed" not in params or "seed" in params
            for params in request["sampling_params"]
        ):
            self.send_error(400)
            return
        chunks = []
        for index, rid in enumerate(request["rid"]):
            chunks.append(
                "data: " + json.dumps({
                    "index": index,
                    "output_ids": [10, 11],
                    "meta_info": {
                        "id": rid, "prompt_tokens": 3, "completion_tokens": 2,
                        "finish_reason": {"type": "length"},
                        "native_token_timestamp_events": [
                            {"token_index": 0, "token_id": 10, "committed_ns": 100},
                            {"token_index": 1, "token_id": 11, "committed_ns": 100},
                        ],
                    },
                }) + "\n\n"
            )
        body = ("".join(chunks) + "data: [DONE]\n\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        Handler.fail = False
        Handler.delay = 0.0
        Handler.batch_sizes = []
        server.shutdown()
        thread.join()


def test_raw_token_output_and_failure_propagation(fake_server):
    client = SGLangClient(f"http://127.0.0.1:{fake_server}", 2)
    rows, elapsed = client.run_batch(["a", "b"], max_new_tokens=2, seed=0)
    assert elapsed > 0
    assert [row.output_ids for row in rows] == [(10, 11), (10, 11)]
    assert [row.inter_token_ms for row in rows] == [(0.0,), (0.0,)]
    Handler.fail = True
    with pytest.raises(Exception):
        client.run_batch(["a"], max_new_tokens=2, seed=0)


def test_scheduled_requests_and_trace_loading(fake_server, tmp_path: Path):
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "Timestamp,input_length,output_length\n"
        "10.0,32,8\n10.01,64,16\n10.03,128,32\n"
    )
    offsets = load_arrival_offsets(trace, limit=3)
    assert offsets == pytest.approx((0.0, 0.01, 0.03))
    _, lengths = load_arrival_trace(trace, limit=3)
    assert lengths == ((32, 8), (64, 16), (128, 32))
    client = SGLangClient(f"http://127.0.0.1:{fake_server}", 2)
    run = client.run_scheduled(
        ("a", "b", "c"), offsets, max_new_tokens=2, seed=0
    )
    assert len(run.results) == 3
    assert {row.status for row in run.outcomes} == {"completed"}
    assert run.elapsed_seconds >= offsets[-1]


def test_request_scoped_generation_is_serial(fake_server):
    client = SGLangClient(f"http://127.0.0.1:{fake_server}", 2)
    rows, _ = _run_request_scoped(client, ("a", "b", "c"), 2, 0)
    assert len(rows) == 3
    assert Handler.batch_sizes[-3:] == [1, 1, 1]


def test_scheduled_dispatcher_does_not_queue_behind_worker_pool(fake_server):
    Handler.delay = 0.05
    client = SGLangClient(f"http://127.0.0.1:{fake_server}", 2)
    run = client.run_scheduled(
        ("a", "b", "c"),
        (0.0, 0.0, 0.0),
        max_new_tokens=2,
        seed=0,
        max_in_flight=1,
    )
    assert [row.status for row in run.outcomes].count("completed") == 1
    assert [row.status for row in run.outcomes].count("unfinished") == 2
    assert sum(row.admitted_ns is not None for row in run.outcomes) == 1


def test_server_process_lifecycle(monkeypatch, tmp_path: Path):
    launched = {}

    class Process:
        pid = 12345
        returncode = None

        def poll(self): return self.returncode
        def wait(self, timeout=None): self.returncode = 0

    def launch(*args, **kwargs):
        launched.update(kwargs)
        return Process()

    monkeypatch.setattr("lightcone_spec.server.subprocess.Popen", launch)
    monkeypatch.setattr("lightcone_spec.server.GpuSampler.start", lambda self: None)
    monkeypatch.setattr("lightcone_spec.server.GpuSampler.stop", lambda self: None)
    monkeypatch.setattr("lightcone_spec.server.SGLangClient.health", lambda self: True)
    monkeypatch.setattr("lightcone_spec.server.os.killpg", lambda *a: None)
    python = tmp_path / "python"
    python.write_text("")
    model = tmp_path / "model"
    model.mkdir()
    config = ExperimentConfig(
        source=tmp_path / "paper.yaml", run_name="run", sglang_root=tmp_path,
        results_root=tmp_path, models={"Qwen/Qwen3-8B": model},
        drafts={"Qwen/Qwen3-8B|DFLASH": model},
        datasets={}, gpu_ids=(0, 1), server=ServerConfig(python=python),
        protocol=ProtocolConfig(),
    )
    output = tmp_path / "attempt"
    output.mkdir()
    process = ServerProcess(config, materialize("preflight")[0], gpus=(0, 1), port=30000, output_dir=output, selection=None)
    with process:
        assert (output / "server.pid").read_text().strip() == "12345"
        assert launched["env"]["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] == "false"
        adaptive = next(
            job
            for job in materialize("E0-tune", valid_e0=[])
            if job.model == "Qwen/Qwen3-8B" and job.backend == "DFLASH"
        )
        adaptive = adaptive.__class__(
            **{**adaptive.to_dict(), "method": "lightcone_candidate"}
        )
        process.restart_for(adaptive, None)
        assert process.adaptation is not None
    assert (output / "server.stopped").is_file()

    config.drafts["Qwen/Qwen3-8B|DSPARK"] = model
    dspark_job = next(
        job
        for job in materialize("E1a")
        if job.backend == "DSPARK"
        and job.method == "lightcone_candidate"
        and "native_heads" in job.parameters["scope"]
    )
    dspark = ServerProcess(
        config,
        dspark_job,
        gpus=(0,),
        port=30001,
        output_dir=output,
        selection=None,
    )
    with dspark:
        assert launched["env"]["SGLANG_RAGGED_VERIFY_MODE"] == "compact"
        assert "native_heads" in dspark.adaptation["parameter_scope"]


def test_onlinespec_payload_contains_independent_learner_settings():
    job = materialize(
        "E0-tune", valid_e0=[("Qwen/Qwen3-8B", "DFLASH", "LiveCodeBench")]
    )[108 + 128]
    payload = adaptation_payload(job)
    assert payload is not None
    assert payload["method"] == "onlinespec_ens"
    assert payload["online_spec"]["additional_learning_rates"]
    assert payload["online_spec"]["hedge_learning_rate"] > 0


def test_cosine_horizon_and_e1a_fixed_settings():
    job = materialize("E2-r2")[0]
    job = job.__class__(
        **{
            **job.to_dict(),
            "parameters": {
                **job.parameters,
                "schedule": "cosine_to_zero",
                "stride": 10,
                "coalescing": 2,
            },
        }
    )
    payload = adaptation_payload(job)
    assert payload["optimizer"]["schedule_total_published_updates"] == max(
        8, math.ceil((16384 - 128) / 20)
    )
    e1a = adaptation_payload(
        materialize("E1a")[0],
        {"optimizer": "adamw", "confidence_loss_weight": 0.25},
    )
    assert e1a["fixed_total_token_budget"] == 8
    assert e1a["confidence_loss_weight"] == 0.25
    assert e1a["canvas_tokens"] == 8


def test_sticky_replica_routing_is_repeatable():
    left, right = object(), object()
    client = StickyReplicaClient((left, right))
    assert client._replica("cohort-0001") is client._replica("cohort-0001")
    assert {client._replica(f"cohort-{index:04d}") for index in range(8)} == {
        left,
        right,
    }


def test_sticky_scheduled_load_is_split_between_replicas(fake_server):
    replicas = tuple(
        SGLangClient(f"http://127.0.0.1:{fake_server}", 2) for _ in range(2)
    )
    client = StickyReplicaClient(replicas)
    run = client.run_scheduled(
        ("prompt",) * 8,
        (0.0,) * 8,
        max_new_tokens=2,
        seed=0,
        routing_keys=tuple(f"cohort-{index % 4:04d}" for index in range(8)),
        max_in_flight=8,
    )
    assert len(run.results) == 8
    assert {outcome.status for outcome in run.outcomes} == {"completed"}


def test_fake_reset_tokenize_and_abort(fake_server):
    client = SGLangClient(f"http://127.0.0.1:{fake_server}", 2)
    assert client.tokenize("hello") == (1, 2, 3)
    results, _ = client.run_batch(((1, 2, 3),), max_new_tokens=2, seed=0)
    assert results[0].stop_details == {"type": "length"}
    client.reset()
    client.abort("request-id")
    client.start_profile(cuda_range=True)
    client.stop_profile()
    metrics = {
        "nonfinite_updates": 1,
        "oom_events": 0,
        "fallbacks": 0,
        "request_outcomes": {"offered": 1, "unfinished": 0},
    }
    assert _fault_action_passed("nonfinite_candidate", True, metrics)
    assert not _fault_action_passed("oom_candidate", True, metrics)


def test_committed_tokens_match_visible_outputs():
    result = GenerationResult(
        request_id="r",
        input_tokens=1,
        completion_tokens=32,
        ttft_ms=1.0,
        inter_token_ms=(0.0,) * 31,
        elapsed_seconds=1.0,
        stop_reason="length",
        output_ids=tuple(range(32)),
        output_text="",
        native_token_timestamps_ns=tuple(range(32)),
    )
    assert _validate_committed_tokens((result,), 32) == 32
    with pytest.raises(RuntimeError, match="committed 34 tokens for 32 output"):
        _validate_committed_tokens((result,), 34)


def test_target_server_keeps_overlap_and_fixed_capacity(tmp_path: Path):
    python = tmp_path / "python"
    python.write_text("")
    model = tmp_path / "model"
    model.mkdir()
    config = ExperimentConfig(
        source=tmp_path / "paper.yaml", run_name="run", sglang_root=tmp_path,
        results_root=tmp_path, models={"Qwen/Qwen3-8B": model}, drafts={},
        datasets={}, gpu_ids=(0, 1), server=ServerConfig(python=python),
        protocol=ProtocolConfig(),
    )
    output = tmp_path / "output"
    output.mkdir()
    command = server_command(
        config, materialize("preflight")[0], port=30000, output_dir=output,
        adaptation=None,
    )
    assert command[command.index("--max-running-requests") + 1] == "256"
    assert "--disable-overlap-schedule" not in command
    assert "--disable-cuda-graph" not in command
    graph_index = command.index("--cuda-graph-bs-decode")
    assert command[graph_index + 1 : graph_index + 10] == [
        "1", "2", "4", "8", "16", "32", "64", "128", "256",
    ]
    assert "--skip-server-warmup" in command


def test_code_scorer_never_executes_without_bubblewrap(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("lightcone_spec.runner.shutil.which", lambda name: None)
    result = GenerationResult(
        request_id="r",
        input_tokens=1,
        completion_tokens=1,
        ttft_ms=1.0,
        inter_token_ms=(),
        elapsed_seconds=1.0,
        stop_reason="length",
        output_ids=(1,),
        output_text="open('/root/should-not-exist', 'w')",
        native_token_timestamps_ns=(1,),
    )
    score, scorer, verdicts = _canonical_accuracy(
        "HumanEval",
        (result,),
        {"examples": ({"test_metadata": "assert True"},)},
        tmp_path / "python",
    )
    assert score is None
    assert "bubblewrap" in scorer
    assert verdicts == []


def test_dspark_loss_retains_graph_inside_inference_scheduler():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    with torch.inference_mode():
        with torch.inference_mode(False), torch.enable_grad():
            proposal = parameter @ torch.ones(2, 2)
        with torch.inference_mode(False), torch.enable_grad():
            loss = proposal.float().square().mean()
    gradient = torch.autograd.grad(loss, parameter)[0]
    assert torch.isfinite(gradient).all()


def test_dspark_server_uses_profiled_sps_table(tmp_path: Path):
    python = tmp_path / "python"
    python.write_text("")
    model = tmp_path / "model"
    model.mkdir()
    table = tmp_path / "dspark-sps.json"
    table.write_text("{}")
    config = ExperimentConfig(
        source=tmp_path / "paper.yaml",
        run_name="run",
        sglang_root=tmp_path,
        results_root=tmp_path,
        models={"Qwen/Qwen3-8B": model},
        drafts={"Qwen/Qwen3-8B|DSPARK": model},
        datasets={},
        gpu_ids=(0, 1),
        server=ServerConfig(python=python, dspark_sps_table=table),
        protocol=ProtocolConfig(),
    )
    job = next(item for item in materialize("E1a") if item.backend == "DSPARK")
    command = server_command(
        config,
        job,
        port=30000,
        output_dir=tmp_path,
        adaptation=None,
    )
    index = command.index("--speculative-dspark-sps-table-path")
    assert command[index + 1] == str(table)
    assert command[command.index("--attention-backend") + 1] == "triton"
    assert command[command.index("--mem-fraction-static") + 1] == "0.8"


def test_nextn_rejection_sampling_uses_single_branch(tmp_path: Path):
    python = tmp_path / "python"
    python.write_text("")
    model = tmp_path / "model"
    model.mkdir()
    job = next(item for item in materialize("E6-pilot") if item.backend == "NEXTN")
    config = ExperimentConfig(
        source=tmp_path / "paper.yaml",
        run_name="run",
        sglang_root=tmp_path,
        results_root=tmp_path,
        models={job.model: model},
        drafts={},
        datasets={},
        gpu_ids=(0, 1),
        server=ServerConfig(python=python, adaptation_reserve_mb=45056),
        protocol=ProtocolConfig(),
    )
    command = server_command(config, job, port=30000, output_dir=tmp_path, adaptation=None)
    maximum = command.index("--max-running-requests")
    assert command[maximum + 1] == "1"
    index = command.index("--speculative-eagle-topk")
    assert command[index + 1] == "1"
    steps = command.index("--speculative-num-steps")
    assert command[steps + 1] == "15"
    patch = (
        Path(__file__).parents[1]
        / "patches/sglang/0001-arguments-config-and-memory.diff"
    ).read_text(encoding="utf-8")
    assert '\"EAGLE\" if config.algorithm == \"NEXTN\"' in patch
    nextn_patch = (
        Path(__file__).parents[1]
        / "patches/sglang/0003-dspark-eagle3-nextn-adapters.diff"
    ).read_text(encoding="utf-8")
    assert (
        '\"EAGLE3\" if self.speculative_algorithm.is_eagle3() else \"NEXTN\"'
        in nextn_patch
    )
    assert "self._online_drafter_adapter.source_adapter_version" in nextn_patch
    acceptance = acceptance_job(
        0,
        "lightcone",
        "NEXTN",
        block=0,
        model=job.model,
        tp2=True,
    )
    assert acceptance.parameters["parameterization"] == "lora"
    assert acceptance.parameters["rank"] == 1
    adaptive_command = server_command(
        config,
        acceptance,
        port=30000,
        output_dir=tmp_path,
        adaptation=adaptation_payload(acceptance),
    )
    reserve = adaptive_command.index("--speculative-adaptation-reserve-mb")
    assert adaptive_command[reserve + 1] == "8192"
    maximum = adaptive_command.index("--max-running-requests")
    assert adaptive_command[maximum + 1] == "8"
    graphs = adaptive_command.index("--cuda-graph-bs-decode")
    assert adaptive_command[graphs + 1 : graphs + 5] == ["1", "2", "4", "8"]


def test_triton_graph_uses_ragged_layout_and_draft_width():
    patch = (
        Path(__file__).parents[1]
        / "patches/sglang/0003-dspark-eagle3-nextn-adapters.diff"
    ).read_text(encoding="utf-8")
    assert "padded_layout.qo_indptr_device" in patch
    assert 'getattr(spec_info, "draft_token_num", self.num_draft_tokens)' in patch
    assert "torch.cuda.get_device_capability(logits.device)[0] == 12" in patch
    assert "int(model_runner.server_args.speculative_num_draft_tokens or 0) >= 16" in patch
    assert "if online_adapter is not None and online_update:" in patch
    assert patch.count("online_update = online_adapter.update_due") >= 3
    assert patch.count("torch.inference_mode(not online_update)") == 5
    assert patch.count("torch.set_grad_enabled(online_update)") == 5
    assert patch.count("with torch.inference_mode(False), torch.enable_grad():") >= 6
    assert "qwen3_5_mtp.py" in patch
    assert "qwen3_5.py" in patch
    assert patch.count("-    @torch.no_grad()") == 3
    assert """-    @torch.no_grad()
     def forward(
""" in patch
    assert """+            if online_adapter is not None and online_update:
+                hidden_rows = self.draft_worker._online_hidden_rows
""" in patch
    assert """+            if online_adapter is not None and not batch.forward_mode.is_idle():
+                committed = batch_output.accept_lens.to(torch.int64)
""" in patch
    assert """+        if online_adapter is not None:
+            verified_drafts = (
""" in patch


def test_tp2_dflash_gathers_full_vocab_before_online_loss():
    patch = (
        Path(__file__).parents[1]
        / "patches/sglang/0002-side-stream-adaptation-and-publication.diff"
    ).read_text(encoding="utf-8")
    assert "class _TensorParallelAllGather" in patch
    assert "class _TensorParallelSum" in patch
    assert "local_inference_logits, target.shape[-1]" in patch
    assert "local_draft_logits, target.shape[-1]" in patch
    assert "attention_output, self.tp_group.device_group" in patch
    assert "_TensorParallelSum.apply(output, self.tp_group.device_group)" in patch


def test_eagle_relay_accepts_first_version_after_prefill():
    patch = (
        Path(__file__).parents[1]
        / "patches/sglang/0002-side-stream-adaptation-and-publication.diff"
    ).read_text(encoding="utf-8")
    assert "EAGLE adaptation cannot be enabled after relay initialization" not in patch
    assert "self.source_adapter_version_buf is None and versions is not None" in patch
    assert "EAGLE source version is missing after relay activation" in patch
