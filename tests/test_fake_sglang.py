import json
import math
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from lightcone_spec.client import GenerationResult, SGLangClient
from lightcone_spec.config import ExperimentConfig, ProtocolConfig, ServerConfig
from lightcone_spec.data import load_arrival_offsets, load_arrival_trace
from lightcone_spec.nextn import (
    MergedPublicationBank,
    PublicationSlot,
    RequestLedger,
    grad_enabled_forwards,
    torch_native_moe,
    torch_native_selected_moe,
)
from lightcone_spec.protocol import Job, materialize
from lightcone_spec.runner import (
    _canonical_accuracy,
    _check_greedy_trajectories,
    _exactness_bootstrap,
    _fault_action_passed,
    _request_scope_released,
    _run_request_scoped,
    _validate_committed_tokens,
    _validate_greedy_verify_counts,
)
from lightcone_spec.server import (
    ServerProcess,
    StickyReplicaClient,
    adaptation_payload,
    server_command,
)


def _nextn_acceptance_job(model: str) -> Job:
    return Job(
        job_id="gpu-acceptance-000-lightcone",
        node="E6-acceptance",
        ordinal=0,
        method="lightcone",
        model=model,
        backend="NEXTN",
        task="controlled_baseline",
        context=40928,
        load="c8",
        width=16,
        block=0,
        gpu_count=2,
        parameters={
            "regime": "short_input_long_generation",
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "grad_clip": 0.0,
            "parameterization": "lora",
            "rank": 1,
            "scope": "all",
            "stride": 10,
            "topology": "tp2_dp1",
        },
    )


class Handler(BaseHTTPRequestHandler):
    fail = False
    delay = 0.0
    batch_sizes = []
    sampling_params = []
    active = 0
    peak_active = 0
    active_lock = threading.Lock()

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
        with Handler.active_lock:
            Handler.active += 1
            Handler.peak_active = max(Handler.peak_active, Handler.active)
        if Handler.delay:
            time.sleep(Handler.delay)
        Handler.batch_sizes.append(len(request["rid"]))
        Handler.sampling_params.append(request["sampling_params"])
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
        with Handler.active_lock:
            Handler.active -= 1


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
        Handler.sampling_params = []
        Handler.active = 0
        Handler.peak_active = 0
        server.shutdown()
        thread.join()


def test_raw_token_output_and_failure_propagation(fake_server):
    client = SGLangClient(f"http://127.0.0.1:{fake_server}", 2)
    rows, elapsed = client.run_batch(["a", "b"], max_new_tokens=2, seed=0)
    assert elapsed > 0
    assert [row.output_ids for row in rows] == [(10, 11), (10, 11)]
    assert [row.inter_token_ms for row in rows] == [(0.0,), (0.0,)]
    assert Handler.sampling_params[0][0]["top_k"] == 1
    assert Handler.sampling_params[0][0]["top_p"] == 1.0
    client.run_batch(["a"], max_new_tokens=1, seed=0, temperature=0.8)
    assert Handler.sampling_params[1][0]["top_k"] == -1
    assert Handler.sampling_params[1][0]["top_p"] == 1.0
    Handler.fail = True
    with pytest.raises(Exception):
        client.run_batch(["a"], max_new_tokens=2, seed=0)


def test_nextn_shadow_has_finite_lora_and_full_gradients():
    torch.manual_seed(0)
    hidden = torch.randn(3, 4)
    router = torch.randn(3, 4, requires_grad=True)
    w13 = torch.randn(3, 8, 4, requires_grad=True)
    w2 = torch.randn(3, 4, 4, requires_grad=True)
    baseline = torch_native_moe(hidden, router, w13, w2, top_k=2)
    baseline.square().mean().backward()
    assert all(
        value.grad is not None
        and torch.isfinite(value.grad).all()
        and torch.count_nonzero(value.grad)
        for value in (router, w13, w2)
    )

    a = torch.randn(2, 4, requires_grad=True)
    b = torch.zeros(3, 2, requires_grad=True)
    replay = torch_native_moe(hidden, router.detach() + b @ a, w13.detach(), w2.detach(), top_k=2)
    assert torch.equal(replay, baseline.detach())
    replay.square().mean().backward()
    assert b.grad is not None and torch.isfinite(b.grad).all()
    assert torch.count_nonzero(b.grad)

    top_weights = torch.tensor([[1.0]])
    top_ids = torch.tensor([[0]])
    fp8_w13 = torch.tensor(
        [[[1.0, -1.0], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float8_e4m3fn,
    )
    fp8_w2 = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float8_e4m3fn
    )
    fp8_output = torch_native_selected_moe(
        torch.ones(1, 2),
        top_weights,
        top_ids,
        fp8_w13,
        fp8_w2,
        w13_scale=torch.full((1, 1, 1), 0.5),
        w2_scale=torch.full((1, 1, 1), 0.5),
    )
    assert fp8_output.shape == (1, 2)
    assert torch.isfinite(fp8_output).all()


def test_nextn_shadow_bypasses_model_no_grad_decorator():
    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))

        @torch.no_grad()
        def forward(self, value):
            return value @ self.weight

    draft = Draft()
    value = torch.ones(1, 2)
    assert not draft(value).requires_grad
    with grad_enabled_forwards(draft):
        loss = draft(value).square().mean()
    loss.backward()
    assert draft.weight.grad is not None
    assert torch.count_nonzero(draft.weight.grad)
    assert not draft(value).requires_grad


def test_nextn_merge_publication_and_ragged_rid_join():
    live = torch.zeros(3, 4, dtype=torch.bfloat16)
    base = torch.zeros(3, 4)
    slot = PublicationSlot("bf16", live, base, (0, 1))
    bank = MergedPublicationBank((slot,))
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    b = torch.ones(3, 2)
    address = live.data_ptr()
    bank.stage((a, b))
    bank.publish()
    assert live.data_ptr() == address
    assert torch.equal(live.float(), (b @ a).to(torch.bfloat16).float())

    fp8_live = torch.zeros(2, 2, dtype=torch.float8_e4m3fn)
    fp8_scale = torch.ones(1, 1)

    def quantize(value):
        scale = value.abs().amax().reshape(1, 1) / 448
        return (value / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale

    fp8_slot = PublicationSlot(
        "fp8",
        fp8_live,
        torch.zeros(2, 2),
        (0,),
        live_scale=fp8_scale,
        quantize=quantize,
    )
    fp8_bank = MergedPublicationBank((fp8_slot,))
    candidate = torch.tensor([[1.0, -2.0], [2.0, 4.0]])
    fp8_bank.stage((candidate,))
    fp8_bank.publish()
    assert fp8_scale.item() == pytest.approx(4 / 448)
    assert torch.allclose(fp8_live.float() * fp8_scale, candidate, atol=0.03)

    ledger = RequestLedger()
    assert ledger.begin(("a", "b"), (10, 20), (0, 2, 5))
    assert ledger.bind_verify(("b", "a"), (6, 0), (9, 2))
    assert (ledger.rows["a"].teacher_start, ledger.rows["a"].teacher_end) == (0, 2)
    assert (ledger.rows["b"].teacher_start, ledger.rows["b"].teacher_end) == (6, 9)
    assert ledger.join_accept_lens(("b", "a"), (1, 2)) == (2, 1)
    ledger.terminal("a", "eos")
    ledger.terminal("b", "cancelled")
    assert ledger.rows["a"].terminal == "eos"
    assert ledger.rows["b"].terminal == "cancelled"
    assert ledger.begin(("replacement",), (21,), (0, 2))
    assert ledger.order == ("replacement",)


def test_adapter_tensor_payload_round_trips_and_preserves_request_ownership(monkeypatch):
    import base64
    import importlib.util
    import pickle

    path = Path(__file__).parents[1] / "scripts" / "gpu_acceptance.py"
    spec = importlib.util.spec_from_file_location("gpu_acceptance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = {"layer.lora_A.weight": torch.arange(8).reshape(2, 4)}
    restored = pickle.loads(base64.b64decode(module._portable_tensor_payload(original)))
    assert torch.equal(restored["layer.lora_A.weight"], original["layer.lora_A.weight"])

    monkeypatch.setattr(
        module,
        "_post_json",
        lambda *_: [
            {"meta_info": {"id": "mixed-01"}, "output_ids": [2]},
            {"meta_info": {"id": "mixed-00"}, "output_ids": [1]},
        ],
    )
    rows, _ = module._adapter_generate("http://unused", "p", ("a", "b"), "mixed", 1, 1)
    assert [row["output_ids"] for row in rows] == [[1], [2]]


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


def test_bounded_and_closed_loop_enforce_real_concurrency(fake_server):
    Handler.delay = 0.03
    client = SGLangClient(f"http://127.0.0.1:{fake_server}", 2)
    bounded = client.run_bounded(
        ("a", "b", "c", "d"),
        max_new_tokens=2,
        seed=0,
        max_in_flight=2,
    )
    assert len(bounded.results) == 4
    assert Handler.peak_active == 2

    Handler.peak_active = 0
    closed = client.run_closed_loop(
        ("a", "b"),
        max_new_tokens=2,
        seed=0,
        max_in_flight=2,
        duration_seconds=0.07,
    )
    assert len(closed.results) >= 4
    assert Handler.peak_active == 2
    assert closed.elapsed_seconds >= 0.07


def test_request_scope_waits_for_every_rank():
    def info(*owners):
        return {
            "internal_states": [
                {
                    "speculative_adaptation_info_record": {
                        "online_adaptation": {"active_request_id": owner}
                    }
                }
                for owner in owners
            ]
        }

    assert _request_scope_released(info(None, None))
    assert not _request_scope_released(info(None, "request-1"))


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


def test_adaptive_reset_scope_tracks_multi_request_concurrency():
    tts_cal = materialize("TTS-Cal")[0]
    assert adaptation_payload(tts_cal)["reset_scope"] == "request"

    e3 = next(
        job
        for job in materialize("E3b-pilot")
        if job.method == "tts" and job.load == "common_load"
    )
    concurrent = e3.__class__(**{**e3.to_dict(), "load": "c8"})
    assert adaptation_payload(concurrent)["reset_scope"] == "cohort"

    single = concurrent.__class__(**{**concurrent.to_dict(), "load": "c1"})
    assert adaptation_payload(single)["reset_scope"] == "request"


def test_sticky_replica_routing_is_repeatable():
    left, right = object(), object()
    client = StickyReplicaClient((left, right))
    assert client._replica("cohort-0001") is client._replica("cohort-0001")
    assert client.replica_index("cohort-0001") == client.replica_index("cohort-0001")
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


def test_greedy_verify_allows_block_overshoot_but_not_mismatch():
    assert _validate_greedy_verify_counts(256, 259, 0) == {
        "unverified_prefill_tokens": 0,
        "extra_checked_tokens": 3,
    }
    with pytest.raises(RuntimeError, match="2 unverified prefill"):
        _validate_greedy_verify_counts(256, 254, 0)
    with pytest.raises(RuntimeError, match="1 mismatched"):
        _validate_greedy_verify_counts(256, 259, 1)


def test_target_server_keeps_overlap_and_fixed_capacity(tmp_path: Path):
    python = tmp_path / "python"
    python.write_text("")
    model = tmp_path / "model"
    model.mkdir()
    draft = tmp_path / "draft"
    draft.mkdir()
    config = ExperimentConfig(
        source=tmp_path / "paper.yaml", run_name="run", sglang_root=tmp_path,
        results_root=tmp_path, models={"Qwen/Qwen3-8B": model},
        drafts={"Qwen/Qwen3-8B|DFLASH": draft},
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
    assert "--enable-deterministic-inference" in command

    performance_job = materialize("preflight")[2]
    performance_command = server_command(
        config, performance_job, port=30001, output_dir=output,
        adaptation=None,
    )
    assert "--enable-deterministic-inference" not in performance_command


def test_preflight_adaptive_exactness_uses_static_deterministic_bootstrap():
    job = materialize("preflight")[1]
    assert job.parameters["deterministic_verify"] is True
    assert "deterministic_exactness" not in job.parameters
    bootstrap = _exactness_bootstrap(job)
    assert bootstrap.method == "static"
    assert bootstrap.parameters["deterministic_exactness"] is True
    assert bootstrap.parameters["controlled_replay"] is False
    assert bootstrap.parameters["exactness_bootstrap"] is True


def test_exactness_bootstrap_only_captures_single_request_graph(tmp_path: Path):
    python = tmp_path / "python"
    python.write_text("")
    model = tmp_path / "model"
    model.mkdir()
    draft = tmp_path / "draft"
    draft.mkdir()
    config = ExperimentConfig(
        source=tmp_path / "paper.yaml", run_name="run", sglang_root=tmp_path,
        results_root=tmp_path, models={"Qwen/Qwen3-8B": model},
        drafts={"Qwen/Qwen3-8B|DFLASH": draft}, datasets={}, gpu_ids=(0, 1),
        server=ServerConfig(python=python, mem_fraction_static=0.88),
        protocol=ProtocolConfig(),
    )
    output = tmp_path / "output"
    output.mkdir()
    command = server_command(
        config,
        _exactness_bootstrap(materialize("preflight")[1]),
        port=30000,
        output_dir=output,
        adaptation=None,
    )
    assert command[command.index("--max-running-requests") + 1] == "1"
    assert command[command.index("--mem-fraction-static") + 1] == "0.8"
    graph = command.index("--cuda-graph-bs-decode")
    assert command[graph + 1] == "1"
    assert command[graph + 2] == "--chunked-prefill-size"


def test_preflight_interference_only_captures_registered_request_batch(tmp_path: Path):
    python = tmp_path / "python"
    python.write_text("")
    model = tmp_path / "model"
    model.mkdir()
    draft = tmp_path / "draft"
    draft.mkdir()
    config = ExperimentConfig(
        source=tmp_path / "paper.yaml",
        run_name="run",
        sglang_root=tmp_path,
        results_root=tmp_path,
        models={"Qwen/Qwen3-8B": model},
        drafts={"Qwen/Qwen3-8B|DFLASH": draft},
        datasets={},
        gpu_ids=(0, 1),
        server=ServerConfig(python=python, requests_per_cell=16),
        protocol=ProtocolConfig(),
    )
    job = materialize("preflight")[2]
    command = server_command(config, job, port=30000, output_dir=tmp_path, adaptation=None)
    assert command[command.index("--max-running-requests") + 1] == "256"
    graph_index = command.index("--cuda-graph-bs-decode")
    assert command[graph_index + 1 : graph_index + 6] == ["1", "2", "4", "8", "16"]


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


def test_chat_score_comes_only_from_task_evaluator(monkeypatch, tmp_path: Path):
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text(
        "import json, pathlib, sys\n"
        "rows=[json.loads(x) for x in pathlib.Path(sys.argv[1]).read_text().splitlines()]\n"
        "pathlib.Path(sys.argv[2]).write_text(''.join(json.dumps({"
        "'request_id': r['request_id'], 'score': 0.75, 'evaluator': 'fastchat', "
        "'version': 'test', 'judge_model': 'judge'})+'\\n' for r in rows))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "LIGHTCONE_MT_BENCH_EVALUATOR",
        f"{sys.executable} {evaluator} {{input}} {{output}}",
    )
    result = GenerationResult(
        request_id="chat-1",
        input_tokens=1,
        completion_tokens=1,
        ttft_ms=1,
        inter_token_ms=(),
        elapsed_seconds=1,
        stop_reason="length",
        output_ids=(1,),
        output_text="answer",
        native_token_timestamps_ns=(1,),
    )
    score, scorer, verdicts = _canonical_accuracy(
        "MT-Bench",
        (result,),
        {"examples": ({"prompt": "question", "reference": None},)},
        tmp_path / "python",
    )
    assert score == 0.75
    assert scorer == "official_mt-bench_evaluator"
    assert verdicts[0]["evaluator"] == "fastchat"


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
    acceptance = _nextn_acceptance_job(job.model)
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

def test_preflight_greedy_gate_uses_aligned_controlled_requests(tmp_path):
    class State:
        run_dir = tmp_path

        def completed_attempt_dirs(self, node):
            assert node == "preflight"
            return tuple(tmp_path.iterdir())

    config = {
        "model": "m",
        "task": "controlled",
        "context": None,
        "load": None,
        "block": None,
        "parameters": {"topology": "tp2_dp1"},
    }
    for name, method, policies in (
        ("target", "target_only", (("target_only", [1, 2, 3]),)),
        (
            "adaptive",
            "l0_naive",
            (
                ("speculative_verify", [1, 2, 4]),
                ("tts", [1, 2, 3]),
                ("l0_naive", [1, 2, 3]),
            ),
        ),
    ):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps({**config, "method": method}), encoding="utf-8"
        )
        (directory / "requests.jsonl").write_text(
            json.dumps({"output_ids": [9]}) + "\n", encoding="utf-8"
        )
        (directory / "controlled.jsonl").write_text(
            "".join(
                json.dumps({"policy": policy, "output_ids": output_ids}) + "\n"
                for policy, output_ids in policies
            ),
            encoding="utf-8",
        )
    _check_greedy_trajectories(State(), "preflight")
    diagnostics = json.loads(
        (tmp_path / "stages/preflight/greedy_trajectory_diagnostics.json").read_text()
    )
    verify = next(
        row
        for row in diagnostics["comparisons"]
        if row["method"] == "speculative_verify"
    )
    assert verify["equal"] is False
    assert verify["first_mismatch"]["token_index"] == 2
    (tmp_path / "adaptive" / "controlled.jsonl").write_text(
        json.dumps({"policy": "speculative_verify", "output_ids": [1, 2, 3]})
        + "\n"
        + json.dumps({"policy": "tts", "output_ids": [1, 2, 3]})
        + "\n"
        + json.dumps({"policy": "l0_naive", "output_ids": [1, 2, 4]})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="l0_naive"):
        _check_greedy_trajectories(State(), "preflight")


def test_launcher_rejects_an_old_semantic_marker(tmp_path: Path):
    sglang = tmp_path / "sglang"
    sglang.mkdir()
    (sglang / ".lightcone-spec-patched").write_text("paper-v1-old\n")
    config = tmp_path / "paper.yaml"
    config.write_text(
        "paths:\n"
        f"  sglang_root: {sglang}\n"
        "server:\n"
        f"  python: {sys.executable}\n"
        "  cuda_home: /tmp\n"
    )
    run = subprocess.run(
        (str(Path(__file__).parents[1] / "run_paper.sh"), str(config)),
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1
    assert "restore SGLang and reapply patches" in run.stderr
