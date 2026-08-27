import ast
import json
import math
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lightcone_spec.client import GenerationResult, SGLangClient, _native_events
from lightcone_spec.config import ExperimentConfig, ProtocolConfig, ServerConfig
from lightcone_spec.data import load_arrival_offsets, load_arrival_trace
from lightcone_spec.nextn import (
    MergedPublicationBank,
    PublicationSlot,
    RequestLedger,
    anchor_replay_logits,
    flatten_attention_output,
    grad_enabled_forwards,
    gradient_leaves,
    needs_tp_gradient_sum,
    ragged_history_locations,
    ragged_kl_loss,
    ste_block_fp8_activation,
    torch_native_moe,
    torch_native_ragged_attention,
    torch_native_selected_moe,
    torch_native_shared_expert_add,
)
from lightcone_spec.protocol import Job, materialize
from lightcone_spec.runner import (
    _check_greedy_trajectories,
    _exactness_bootstrap,
    _fault_action_passed,
    _fit_prompt,
    _read_jsonl,
    _request_metrics,
    _request_scope_released,
    _run_multi_turn,
    _run_request_scoped,
    _speed_metrics,
    _trajectory_checkpoint_metrics,
    _uses_request_scope,
    _validate_committed_tokens,
    _validate_greedy_verify_counts,
    _write_jsonl,
)
from lightcone_spec.server import (
    ServerProcess,
    StickyReplicaClient,
    adaptation_payload,
    server_command,
    server_session_key,
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


def test_native_timestamps_accept_scheduler_field_names():
    output_ids = [11, 12]
    assert _native_events(
        {
            "native_token_timestamp_events": [
                {"token_index": 0, "token_id": 11, "observed_ns": 100},
                {"token_index": 1, "token_id": 12, "observed_ns": 101},
            ]
        },
        output_ids,
    ) == (100, 101)


def test_donor_metrics_preserve_unmeasured_fields():
    info = {
        "speed_study_metrics": {
            "committed_tokens": 1,
            "peak_hbm_bytes": 2,
            "kv_token_capacity": 3,
            "oom_events": 0,
            "retractions": 0,
        }
    }
    metrics = _speed_metrics(
        info,
        "tp1_dp1",
        unmeasured=(
            "peak_hbm_reserved_bytes",
            "stale_publications",
            "exactness_violations",
            "version_mismatches",
            "fallbacks",
            "nonfinite_updates",
        ),
    )
    assert metrics["stale_publications"] is None
    assert metrics["peak_hbm_reserved_bytes"] is None


class Handler(BaseHTTPRequestHandler):
    fail = False
    delay = 0.0
    batch_sizes = []
    sampling_params = []
    active = 0
    peak_active = 0
    aborted = []
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
            Handler.aborted.append(request["rid"])
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
        Handler.aborted = []
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


def test_replay_rope_preserves_non_rotary_head_suffix():
    patch = Path("patches/sglang/0002-side-stream-adaptation-and-publication.diff")
    added = []
    active = False
    for line in patch.read_text().splitlines():
        if line.startswith("diff --git "):
            active = "dflash_online_adaptation.py" in line
        elif active and line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    tree = ast.parse("\n".join(added))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_rope"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(patch), "exec"), namespace)

    value = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    cache = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    output = namespace["_rope"](value, torch.tensor([0]), cos_sin_cache=cache)
    assert torch.equal(output, torch.tensor([[[-3.0, 2.0, 1.0, 4.0, 5.0, 6.0]]]))


def test_qwen_speculative_workspace_covers_registered_width():
    patch = Path("patches/sglang/0003-dspark-eagle3-nextn-adapters.diff").read_text()
    assert "cuda_graph_config.decode.max_bs" in patch
    assert "workspace_mb = 768 if graph_max_bs >= 256 else 512" in patch
    assert "speculative_num_draft_tokens or 0" not in patch


def test_sglang_terminal_hook_preserves_abort_identity():
    patch = Path("patches/sglang/0005-nextn-shadow-replay.diff")
    lines = patch.read_text().splitlines()
    start = lines.index("+def _notify_spec_request_terminal(draft_worker, req) -> None:")
    added = []
    for line in lines[start:]:
        if line.startswith("+"):
            added.append(line[1:])
        else:
            break
    function = ast.parse("\n".join(added)).body[0]

    class BaseSpecWorker:
        pass

    class FinishAbort:
        pass

    class FinishMatchedToken:
        pass

    namespace = {
        "BaseSpecWorker": BaseSpecWorker,
        "FINISH_ABORT": FinishAbort,
        "FINISH_MATCHED_TOKEN": FinishMatchedToken,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(patch), "exec"), namespace)

    class Worker(BaseSpecWorker):
        def __init__(self):
            self.events = []

        def note_request_aborted(self, *, rid):
            self.events.append(("aborted", rid))

        def note_request_finished(self, *, rid, natural_stop):
            self.events.append(("finished", rid, natural_stop))

    worker = Worker()
    namespace["_notify_spec_request_terminal"](
        worker, SimpleNamespace(rid="cancelled", finished_reason=FinishAbort())
    )
    namespace["_notify_spec_request_terminal"](
        worker, SimpleNamespace(rid="eos", finished_reason=FinishMatchedToken())
    )
    assert worker.events == [("aborted", "cancelled"), ("finished", "eos", True)]


def test_update_is_counted_only_after_publication():
    patch = Path("patches/sglang/0002-side-stream-adaptation-and-publication.diff")
    added = []
    active = False
    for line in patch.read_text().splitlines():
        if line.startswith("diff --git "):
            active = "online_adaptation_runtime.py" in line
        elif active and line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    tree = ast.parse("\n".join(added))
    runtime = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OnlineCohortRuntime"
    )
    function = next(
        node
        for node in runtime.body
        if isinstance(node, ast.FunctionDef) and node.name == "_drain_diagnostics"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(patch), "exec"), namespace)

    trace = SimpleNamespace(diagnosed=False, buffer_index=0, status="optimized")
    fake = SimpleNamespace(
        pending=None,
        pending_timings=[],
        trace_prefix_lens=torch.tensor([[1]]),
        trace_metrics=torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]]),
        trace_optimizer_steps=torch.tensor([1]),
        trace_online_experts=torch.tensor([[[0.0]]]),
        update_traces=[trace],
        counters={"updates_published": 0},
        _record_invalid_candidate=lambda *args, **kwargs: None,
    )
    namespace["_drain_diagnostics"](fake)
    assert not trace.diagnosed
    assert fake.counters["updates_published"] == 0

    trace.status = "decision_enqueued"
    namespace["_drain_diagnostics"](fake)
    assert trace.diagnosed
    assert trace.status == "published"
    assert fake.counters["updates_published"] == 1


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


def test_nextn_shadow_uses_resident_optimizer_values_as_gradient_leaves():
    master = (torch.ones(2, 2),)
    with torch.inference_mode():
        (weight,) = gradient_leaves(master)
        with torch.inference_mode(False), torch.enable_grad():
            loss = (torch.ones(1, 2) @ weight).square().mean()
    (gradient,) = torch.autograd.grad(loss, (weight,))
    assert not weight.is_inference()
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient)


def test_nextn_replay_uses_inference_values_and_replay_gradient():
    inference = torch.tensor([[3.0, 1.0]])
    weight = torch.tensor(2.0, requires_grad=True)
    replay_scale = torch.tensor([[1e8, 4.0]])
    replay = weight * replay_scale
    anchored = anchor_replay_logits(inference, replay)
    assert torch.equal(anchored, inference)
    anchored.sum().backward()
    assert weight.grad == replay_scale.sum()


def test_nextn_operator_boundaries_keep_native_values_and_shadow_jacobian():
    weight = torch.tensor(2.0, requires_grad=True)
    hidden = anchor_replay_logits(torch.tensor([5.0]), weight * 3.0)
    logits = anchor_replay_logits(torch.tensor([11.0]), hidden * weight)
    assert torch.equal(hidden, torch.tensor([5.0]))
    assert torch.equal(logits, torch.tensor([11.0]))
    logits.sum().backward()
    assert weight.grad == 11.0


def test_nextn_attention_flattens_query_width_not_hidden_size():
    attended = torch.zeros(2, 16, 1, 256)
    assert flatten_attention_output(attended, 4096).shape == (2, 4096)


def test_nextn_fp8_activation_uses_quantized_values_and_identity_gradient():
    hidden = torch.linspace(-3, 3, 256).reshape(2, 128).requires_grad_()
    quantized = ste_block_fp8_activation(hidden)
    assert not torch.equal(quantized, hidden)
    quantized.sum().backward()
    assert torch.equal(hidden.grad, torch.ones_like(hidden))


def test_nextn_ragged_loss_keeps_gradient_inside_scheduler_inference_mode():
    ledger = RequestLedger()
    assert ledger.begin(("request",), (0,), (0, 2))
    assert ledger.bind_verify(("request",), (0,), (2,))
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    teacher = torch.zeros(2, 4)
    with torch.inference_mode():
        with torch.inference_mode(False), torch.enable_grad():
            draft = hidden @ weight
        loss = ragged_kl_loss(draft, teacher, ledger)
    (gradient,) = torch.autograd.grad(loss, (weight,))
    assert loss.requires_grad
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient)


def test_nextn_ragged_loss_flattens_without_severing_replay_graph():
    ledger = RequestLedger()
    assert ledger.begin(("request",), (0,), (0, 2))
    assert ledger.bind_verify(("request",), (0,), (2,))
    with torch.inference_mode():
        with torch.inference_mode(False), torch.enable_grad():
            weight = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
            replay = (torch.arange(6, dtype=torch.float32).reshape(2, 3) @ weight).view(
                1, 2, 4
            )
            replay = replay[:1]
        loss = ragged_kl_loss(replay, torch.zeros(2, 4), ledger)
    (gradient,) = torch.autograd.grad(loss, (weight,), allow_unused=True)
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient)


def test_nextn_shadow_shared_expert_add_keeps_gradient():
    hidden = torch.ones(2, 3)
    gate = torch.ones(3)
    shared_weight = torch.ones(3, 3, requires_grad=True)
    shared = hidden @ shared_weight
    routed = torch.zeros_like(shared)
    torch_native_shared_expert_add(hidden, gate, shared, routed)
    routed.square().mean().backward()
    assert shared_weight.grad is not None
    assert torch.count_nonzero(shared_weight.grad)


def test_nextn_ragged_attention_keeps_current_block_gradient():
    query = torch.randn(2, 4, 3, requires_grad=True)
    key = torch.randn(2, 2, 3, requires_grad=True)
    value = torch.randn(2, 2, 3, requires_grad=True)
    history_key = torch.randn(2, 3, 2, 3)
    history_value = torch.randn(2, 3, 2, 3)
    valid = torch.tensor([[True, True, True], [True, False, False]])
    output = torch_native_ragged_attention(
        query,
        key,
        value,
        history_key,
        history_value,
        valid,
        scale=3**-0.5,
    )
    output.square().mean().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad)


def test_nextn_ragged_history_excludes_current_write_slot():
    indptr = torch.tensor([0, 4, 6], dtype=torch.int32)
    indices = torch.tensor([10, 11, 12, 13, 20, 21], dtype=torch.int64)
    locations, valid = ragged_history_locations(indptr, indices)
    assert locations.tolist() == [[10, 11, 12], [20, 0, 0]]
    assert valid.tolist() == [[True, True, True], [True, False, False]]


def test_nextn_tp_reduces_only_partial_replicated_parameters():
    assert needs_tp_gradient_sum("layers.0.q_norm.weight")
    assert needs_tp_gradient_sum("layers.0.mlp.gate.weight")
    assert not needs_tp_gradient_sum("fc.weight")
    assert not needs_tp_gradient_sum("layers.0.qkv_proj.weight")


def test_nextn_merge_publication_and_ragged_rid_join():
    live = torch.zeros(3, 4, dtype=torch.bfloat16)
    base = torch.zeros(3, 4)
    slot = PublicationSlot("bf16", live, base, (0, 1), lora_scale=0.5)
    bank = MergedPublicationBank((slot,))
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    b = torch.ones(3, 2)
    address = live.data_ptr()
    bank.stage((a, b))
    bank.publish()
    assert live.data_ptr() == address
    assert torch.equal(live.float(), (0.5 * (b @ a)).to(torch.bfloat16).float())
    assert bank.matches((a, b))

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
    assert fp8_bank.matches((candidate,))

    ledger = RequestLedger()
    assert ledger.begin(("a", "b"), (10, 20), (0, 2, 5))
    assert ledger.bind_verify(("b", "a"), (6, 0), (9, 2))
    assert (ledger.rows["a"].teacher_start, ledger.rows["a"].teacher_end) == (0, 2)
    assert (ledger.rows["b"].teacher_start, ledger.rows["b"].teacher_end) == (6, 9)
    assert ledger.join_accept_lens(("b", "a"), (1, 2)) == (2, 1)
    ledger.terminal("a", "eos")
    ledger.terminal("b", "cancelled")
    ledger.terminal("b", "aborted")
    ledger.terminal("b", "finished")
    assert ledger.rows["a"].terminal == "eos"
    assert ledger.rows["b"].terminal == "aborted"
    assert ledger.terminal_states["b"] == "aborted"
    assert [row["terminal"] for row in ledger.snapshot()] == ["eos", "aborted"]
    assert ledger.begin(("replacement",), (21,), (0, 2))
    assert ledger.order == ("replacement",)
    ledger.reset()
    assert ledger.snapshot() == []
    assert ledger.terminal_states == {}


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
    dspark = module._job(0, "lightcone", "DSPARK", block=0)
    assert module._needs_publication_witness(
        dspark, {"updates_launched": 1, "updates_published": 0}
    )
    assert not module._needs_publication_witness(
        dspark, {"updates_launched": 1, "updates_published": 1}
    )


def test_gpu_smoke_rejects_cross_kernel_greedy_mismatch(monkeypatch, tmp_path: Path):
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / "gpu_acceptance.py"
    spec = importlib.util.spec_from_file_location("gpu_acceptance_exactness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Config:
        def validate_local_paths(self):
            return None

    monkeypatch.setattr(module.ExperimentConfig, "load", lambda path: Config())
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    def measure(config, job, output, *, max_new_tokens, exactness_tokens):
        del config, output, max_new_tokens, exactness_tokens
        return {
            "method": job.method,
            "backend": job.backend,
            "exactness_trajectory": [1, 2] if job.method == "target_only" else [1, 3],
        }

    monkeypatch.setattr(module, "_measure", measure)
    args = SimpleNamespace(
        config=tmp_path / "paper.yaml",
        output=tmp_path / "smoke",
        max_new_tokens=2,
        cases=("target", "static"),
    )
    with pytest.raises(RuntimeError, match="greedy trajectory differs"):
        module.smoke(args)


def test_donor_counters_are_reported_but_only_rebuild_counters_gate(tmp_path: Path):
    import importlib.util
    import json

    path = Path(__file__).parents[1] / "scripts" / "gpu_acceptance.py"
    spec = importlib.util.spec_from_file_location("gpu_acceptance_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    methods = (
        "target_only",
        "static",
        "tts",
        "l0_naive",
        "lightcone",
        "onlinespec_ogd",
    )
    donor = []
    rebuild = []
    for method in methods:
        for block in range(3):
            row = {
                "method": method,
                "block": block,
                "goodput": 1.0,
                "p99_itl_ms": 1.0,
                "peak_hbm_bytes": 1,
                "trajectories": [[1, 2]],
                "exactness_trajectory": [1, 2],
                "counters": {"retractions": 2},
            }
            donor.append(row)
            rebuild.append({**row, "counters": {"retractions": 0}})
    donor_path = tmp_path / "donor.json"
    rebuild_path = tmp_path / "rebuild.json"
    output = tmp_path / "comparison.json"
    donor_path.write_text(json.dumps(donor), encoding="utf-8")
    rebuild_path.write_text(json.dumps(rebuild), encoding="utf-8")
    args = SimpleNamespace(donor=donor_path, rebuild=rebuild_path, output=output)
    module.compare(args)
    assert json.loads(output.read_text(encoding="utf-8"))["passed"]

    rebuild[0]["counters"]["retractions"] = 1
    rebuild_path.write_text(json.dumps(rebuild), encoding="utf-8")
    with pytest.raises(SystemExit, match="rebuild has nonzero safety counters"):
        module.compare(args)

    gpu_csv = tmp_path / "gpu.csv"
    gpu_csv.write_text(
        "timestamp,index,memory_used_mb\n0,0,10\n1,0,12\n", encoding="utf-8"
    )
    assert module._nvml_peak_hbm(gpu_csv) == 12 * 1024 * 1024


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


def test_bounded_timeout_aborts_the_exact_request(fake_server):
    Handler.delay = 0.03
    client = SGLangClient(f"http://127.0.0.1:{fake_server}", 2)
    run = client.run_bounded(
        ("a",),
        max_new_tokens=2,
        seed=0,
        request_ids=("timeout-request",),
        deadline_seconds=0.005,
    )
    assert not run.results
    assert run.outcomes[0].status == "timed_out"
    assert Handler.aborted == ["timeout-request"]


def test_long_context_prompt_repeats_the_registered_workload_pool():
    prompt = _fit_prompt((7, 8), (1, 2, 3), 10)
    assert prompt == (1, 2, 3, 1, 2, 3, 1, 2, 7, 8)


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
        process.configure(adaptive, None)
        assert process.adaptation is not None
        assert process.session_key == server_session_key(adaptive)
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


def test_tts_always_resets_between_requests():
    tts_cal = materialize("TTS-Cal")[0]
    assert adaptation_payload(tts_cal)["reset_scope"] == "request"

    e3 = next(
        job
        for job in materialize("E3b-pilot")
        if job.method == "tts" and job.load == "common_load"
    )
    concurrent = e3.__class__(**{**e3.to_dict(), "load": "c8"})
    assert adaptation_payload(concurrent)["reset_scope"] == "request"
    assert _uses_request_scope(concurrent)

    single = concurrent.__class__(**{**concurrent.to_dict(), "load": "c1"})
    assert adaptation_payload(single)["reset_scope"] == "request"

    l0 = concurrent.__class__(**{**concurrent.to_dict(), "method": "l0_naive"})
    assert adaptation_payload(l0)["reset_scope"] == "request"
    assert _uses_request_scope(l0)

    lightcone = concurrent.__class__(
        **{**concurrent.to_dict(), "method": "lightcone"}
    )
    assert adaptation_payload(lightcone)["reset_scope"] == "cohort"
    assert not _uses_request_scope(lightcone)


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


def test_target_server_keeps_overlap_and_uses_registered_capacity(tmp_path: Path):
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
    assert command[command.index("--max-running-requests") + 1] == "1"
    assert "--disable-overlap-schedule" not in command
    assert "--disable-cuda-graph" not in command
    graph_index = command.index("--cuda-graph-bs-decode")
    assert command[graph_index + 1] == "1"
    assert command[graph_index + 2].startswith("--")
    assert "--skip-server-warmup" in command
    assert "--enable-deterministic-inference" in command

    performance_job = materialize("preflight")[2]
    performance_command = server_command(
        config, performance_job, port=30001, output_dir=output,
        adaptation=None,
    )
    assert "--enable-deterministic-inference" not in performance_command

    acceptance = Job(
        "acceptance-static",
        "gpu-acceptance",
        0,
        "static",
        "Qwen/Qwen3-8B",
        "DFLASH",
        "controlled_baseline",
        load="c8",
        parameters={"regime": "short_input_long_generation"},
    )
    acceptance_command = server_command(
        config, acceptance, port=30001, output_dir=output, adaptation=None
    )
    assert acceptance_command[
        acceptance_command.index("--max-running-requests") + 1
    ] == "8"

    tts = Job(
        "tts-c8",
        "E3b-final",
        0,
        "tts",
        "Qwen/Qwen3-8B",
        "DFLASH",
        "controlled_baseline",
        load="c8",
        parameters={"regime": "short_input_long_generation"},
    )
    tts_command = server_command(
        config,
        tts,
        port=30001,
        output_dir=output,
        adaptation=adaptation_payload(tts),
    )
    assert tts_command[tts_command.index("--max-running-requests") + 1] == "1"
    graph_index = tts_command.index("--cuda-graph-bs-decode")
    assert tts_command[graph_index + 1] == "1"
    assert tts_command[graph_index + 2].startswith("--")


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


def test_preflight_interference_only_captures_registered_concurrency(tmp_path: Path):
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
    assert command[command.index("--max-running-requests") + 1] == "1"
    graph_index = command.index("--cuda-graph-bs-decode")
    assert command[graph_index + 1] == "1"
    assert command[graph_index + 2].startswith("--")

    c64 = next(job for job in materialize("E3a") if job.load == "c64")
    command = server_command(
        config, c64, port=30000, output_dir=tmp_path, adaptation=None
    )
    assert command[command.index("--max-running-requests") + 1] == "64"
    graph_index = command.index("--cuda-graph-bs-decode")
    assert command[graph_index + 1 : graph_index + 8] == [
        "1",
        "2",
        "4",
        "8",
        "16",
        "32",
        "64",
    ]


def test_formal_request_evidence_omits_text_and_token_trajectory(tmp_path: Path):
    result = GenerationResult(
        request_id="r",
        input_tokens=1,
        completion_tokens=1,
        ttft_ms=1.0,
        inter_token_ms=(),
        elapsed_seconds=1.0,
        stop_reason="length",
        output_ids=(1,),
        output_text="not stored",
        native_token_timestamps_ns=(1,),
    )
    row = _request_metrics(result)
    assert "output_ids" not in row
    assert "output_text" not in row
    path = tmp_path / "requests.jsonl.gz"
    _write_jsonl(path, (row,))
    assert _read_jsonl(path) == [row]


def test_one_long_trajectory_yields_multiple_speed_checkpoints():
    result = GenerationResult(
        request_id="r",
        input_tokens=1,
        completion_tokens=4,
        ttft_ms=1.0,
        inter_token_ms=(1.0, 1.0, 1.0),
        elapsed_seconds=0.004,
        stop_reason="length",
        output_ids=(1, 2, 3, 4),
        output_text="",
        native_token_timestamps_ns=(1_000_000, 2_000_000, 3_000_000, 4_000_000),
    )
    rows = _trajectory_checkpoint_metrics((result,), (2, 4))
    assert [row["generation_tokens"] for row in rows] == [2, 4]
    assert rows[0]["goodput"] == pytest.approx(2000.0)
    assert rows[1]["itl_p99_ms"] == pytest.approx(1.0)


def test_multi_turn_itl_excludes_cross_turn_prefill_gap():
    class Client:
        turn = 0

        def run_bounded(self, prompts, *, max_new_tokens, seed, request_ids, max_in_flight):
            start = self.turn * 10_000_000_000
            self.turn += 1
            result = GenerationResult(
                request_id=request_ids[0],
                input_tokens=len(prompts[0]),
                completion_tokens=2,
                ttft_ms=5.0,
                inter_token_ms=(1.0,),
                elapsed_seconds=0.01,
                stop_reason="length",
                output_ids=(1, 2),
                output_text="",
                native_token_timestamps_ns=(start + 1_000_000, start + 2_000_000),
            )
            return SimpleNamespace(results=(result,), elapsed_seconds=0.01)

    results, _ = _run_multi_turn(
        Client(),
        ((7,),),
        8,
        0,
        request_scoped=False,
        max_in_flight=1,
    )
    assert results[0].inter_token_ms == (1.0, 1.0, 1.0, 1.0)


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
    _check_greedy_trajectories(State(), "preflight")
    diagnostics = json.loads(
        (tmp_path / "stages/preflight/greedy_trajectory_diagnostics.json").read_text()
    )
    l0 = next(
        row for row in diagnostics["comparisons"] if row["method"] == "l0_naive"
    )
    assert l0["equal"] is False


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
