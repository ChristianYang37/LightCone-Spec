import ast
import json
import math
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
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
    native_training_replay,
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
    _cell_inputs,
    _check_greedy_trajectories,
    _dispatcher_concurrency,
    _exactness_bootstrap,
    _fault_action_passed,
    _fit_prompt,
    _read_jsonl,
    _records_scientific_rejection,
    _request_count,
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
from lightcone_spec.state import StateStore


@pytest.mark.parametrize("load,method,workload,node,count", [
    ("c1", "static", "ordinary", "E0-final", 16),
    ("c16", "lightcone", "ordinary", "E0-final", 16),
    ("c128", "static", "ordinary", "E0-final", 128),
    ("c16", "tts", "ordinary", "E0-final", 16),
    ("c1", "tts", "tts_stride10_confirmation", "TTS-S10-confirmation", 19),
    ("c1", "tts", "ordinary", "TTS-Cal", 19),
    ("closed_loop_c128", "static", "ordinary", "E5-final", 128),
    ("burstgpt_shape", "static", "ordinary", "E5-final", 16),
    ("common_slo_load", "lightcone", "ordinary", "E2-r0", 16),
])
def test_execution_budget_preserves_inputs_independent_of_dispatcher(
    tmp_path, monkeypatch, load, method, workload, node, count,
):
    config = ExperimentConfig(
        source=tmp_path / "paper.yaml", run_name="test", sglang_root=tmp_path,
        results_root=tmp_path, models={}, drafts={}, datasets={"CalibrationMix": tmp_path},
        gpu_ids=(0, 1), server=ServerConfig(python=tmp_path / "python"),
        protocol=ProtocolConfig(),
    )
    state = StateStore(config.run_dir)
    job = Job(job_id="budget", ordinal=0, node=node, method=method, load=load,
              model="Qwen/Qwen3-8B", backend="DFLASH", task="MATH-500",
              context=4096, parameters={"workload": workload, "generation_tokens": 256})
    monkeypatch.setattr("lightcone_spec.runner._task_for_data", lambda *_: "CalibrationMix")
    monkeypatch.setattr("lightcone_spec.runner.load_calibration_mix",
                        lambda *_: tuple(str(i) for i in range(76)))
    monkeypatch.setattr("lightcone_spec.runner.load_prompt_records",
                        lambda *args, limit, selection_seed, **kwargs:
                        tuple({"prompt": str(selection_seed + i)} for i in range(limit)))
    monkeypatch.setattr("lightcone_spec.runner.load_prompt_pool", lambda *_: ({"prompt": "pool"},))
    config = replace(config, datasets={**config.datasets, "BurstGPT": tmp_path / "trace"})
    monkeypatch.setattr("lightcone_spec.runner.load_arrival_trace", lambda *args, limit, **kwargs:
                        (tuple(range(limit)), tuple((128 + i, 64 + i) for i in range(limit))))
    client = SimpleNamespace(tokenize=lambda prompt: tuple(prompt.encode()))
    frozen = replace(job, parameters={**job.parameters, "execution_request_count": count})
    assert _request_count(config, state, job) == count
    assert _cell_inputs(config, state, client, frozen) == _cell_inputs(config, state, client, job)
    assert _request_count(config, state, replace(frozen, load="c256")) == count
    assert "execution_request_count" not in job.parameters
    if method == "tts":
        assert _dispatcher_concurrency(job) == 1


def test_dp2_system_concurrency_does_not_double_request_budget(tmp_path):
    config = ExperimentConfig(
        source=tmp_path / "paper.yaml",
        run_name="test",
        sglang_root=tmp_path,
        results_root=tmp_path,
        models={},
        drafts={},
        datasets={},
        gpu_ids=(0, 1),
        server=ServerConfig(python=tmp_path / "python"),
        protocol=ProtocolConfig(),
    )
    state = StateStore(config.run_dir)
    job = Job(
        job_id="dp2-system-budget",
        ordinal=0,
        node="E5-final",
        method="lightcone",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="LiveCodeBench",
        context=40928,
        load="closed_loop_c128",
        gpu_count=2,
        parameters={
            "topology": "two_replica_tp1_dp2",
            "registered_concurrency_scope": "system",
        },
    )
    assert _request_count(config, state, job) == 128


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


def test_s10_scientific_rejection_is_terminal_without_stopping_siblings():
    replacement = Job(
        job_id="s10-repair__E2-r3__000023__tts",
        node="S10-reconciliation",
        ordinal=13,
        method="tts",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="CalibrationMix",
        context=40928,
        load="c1",
        width=16,
        parameters={"reconciliation_kind": "formal_stride"},
    )
    original = replace(replacement, job_id="E2-r3__000023__tts", node="E2-r3")

    assert _records_scientific_rejection(replacement)
    assert not _records_scientific_rejection(original)
    width = replace(
        replacement,
        job_id="e3-width-lightcone-4__segment-000",
        node="E3-width-calibration",
    )
    assert _records_scientific_rejection(width)


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


def _patched_logit_reconstruction_gate():
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
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_logit_reconstruction_gate"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(patch), "exec"), namespace)
    return namespace["_logit_reconstruction_gate"]


def _patched_residual_rms():
    patch = Path("patches/sglang/0002-side-stream-adaptation-and-publication.diff")
    added = []
    active = False
    for line in patch.read_text().splitlines():
        if line.startswith("diff --git "):
            active = "dflash_online_adaptation.py" in line
        elif active and line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    tree = ast.parse("\n".join(added))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_rms", "_residual_rms"}
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(functions, []), str(patch), "exec"), namespace)
    return namespace["_residual_rms"], tree


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_residual_rms_preserves_fused_fp32_sum_and_gradients(dtype):
    replay, tree = _patched_residual_rms()
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(16, 128, generator=generator).to(dtype).requires_grad_()
    residual = torch.randn(16, 128, generator=generator).to(dtype).requires_grad_()
    weight = torch.randn(128, generator=generator).to(dtype).requires_grad_()
    summed = hidden.float() + residual.float()
    expected = (
        summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
        * weight.float()
    ).to(dtype)
    output, stored = replay(hidden, residual, weight, 1e-6)
    assert torch.equal(output, expected)
    assert torch.equal(stored, summed.to(dtype))
    objective = output.float().square().mean() + stored.float().square().mean()
    expected_objective = expected.float().square().mean() + summed.to(dtype).float().square().mean()
    gradients = torch.autograd.grad(objective, (hidden, residual, weight), retain_graph=True)
    reference = torch.autograd.grad(expected_objective, (hidden, residual, weight))
    for actual, wanted in zip(gradients, reference, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    if dtype == torch.bfloat16:
        rounded = (hidden + residual).float()
        previous = (
            rounded * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + 1e-6)
            * weight.float()
        ).to(dtype)
        assert not torch.equal(previous, expected)
    adapter = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DFlashDrafterAdapter")
    for name, expected_calls in (("_layer_forward", 2), ("_surrogate_hidden", 1)):
        method = next(n for n in adapter.body if isinstance(n, ast.FunctionDef) and n.name == name)
        calls = [n for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_residual_rms"]
        assert len(calls) == expected_calls


def test_residual_rms_repeated_weight_changes_and_fresh_request():
    replay, _ = _patched_residual_rms()
    generator = torch.Generator().manual_seed(42)
    hidden = torch.randn(4, 64, generator=generator).bfloat16()
    residual = torch.randn(4, 64, generator=generator).bfloat16()
    initial = torch.randn(64, generator=generator)
    first = replay(hidden, residual, initial.bfloat16(), 1e-6)
    for step in range(300):
        weight = (initial + step * 1e-4).bfloat16()
        output, stored = replay(hidden, residual, weight, 1e-6)
        summed = hidden.float() + residual.float()
        expected = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6) * weight.float()).bfloat16()
        assert torch.equal(output, expected)
        assert torch.equal(stored, summed.bfloat16())
    reset = replay(hidden, residual, initial.bfloat16(), 1e-6)
    assert all(torch.equal(x, y) for x, y in zip(first, reset, strict=True))


def test_logit_reconstruction_ignores_only_masked_canvas_positions():
    gate = _patched_logit_reconstruction_gate()
    inference = torch.tensor([[[8.0, 0.0], [8.0, 0.0]]], dtype=torch.bfloat16)
    padded_mismatch = torch.tensor(
        [[[8.0, 0.0], [0.0, 8.0]]], dtype=torch.bfloat16
    )
    valid_mask = torch.tensor([[True, False]])
    accepted = gate(inference, padded_mismatch, valid_mask=valid_mask)
    assert bool(accepted[0])
    assert accepted[3].item() == 1.0

    valid_mismatch = torch.tensor(
        [[[0.0, 8.0], [8.0, 0.0]]], dtype=torch.bfloat16
    )
    rejected = gate(inference, valid_mismatch, valid_mask=valid_mask)
    assert not bool(rejected[0])
    assert rejected[3].item() == 0.0


def test_logit_reconstruction_uses_kl64_bf16_envelope():
    gate = _patched_logit_reconstruction_gate()
    assert gate.__kwdefaults__["kl_units"] == 64.0
    assert 0.002681 <= 64 * (1 / 128) ** 2
    inference = torch.tensor([[[8.0, 8.0]]], dtype=torch.bfloat16)
    quantized_drift = torch.tensor([[[8.0625, 7.90625]]], dtype=torch.bfloat16)
    accepted = gate(
        inference,
        quantized_drift,
        valid_mask=torch.tensor([[True]]),
    )
    assert accepted[4].item() > 32 * (1 / 128) ** 2
    assert accepted[4].item() <= 64 * (1 / 128) ** 2
    assert bool(accepted[0])


def test_logit_reconstruction_empty_mask_cannot_publish():
    gate = _patched_logit_reconstruction_gate()
    logits = torch.zeros((1, 2, 4), dtype=torch.bfloat16)
    valid, maximum, relative_rms, top1, mean_kl = gate(
        logits,
        logits,
        valid_mask=torch.zeros((1, 2), dtype=torch.bool),
    )
    assert not bool(valid)
    assert maximum.item() == 0.0
    assert math.isinf(relative_rms.item())
    assert top1.item() == 0.0
    assert math.isinf(mean_kl.item())


@pytest.mark.parametrize("selected_valid,mismatch", [(False, False), (True, False), (True, True)])
def test_dflash_microbatch_supervision_matches_loss_gate_and_publication(selected_valid, mismatch):
    # Execute the patched launch method on CPU: request 0 owns the microbatch,
    # while request 1 has supervision regardless of request 0's final canvas.
    _, tree = _patched_residual_rms()
    adapter_class = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                         and n.name == "DFlashDrafterAdapter")
    launch = next(n for n in adapter_class.body if isinstance(n, ast.FunctionDef)
                  and n.name == "maybe_launch")
    patch = Path("patches/sglang/0002-side-stream-adaptation-and-publication.diff")
    runtime_source = patch.read_text().split(
        "+++ b/python/sglang/srt/speculative/online_adaptation_runtime.py\n", 1
    )[1].split("diff --git ", 1)[0]
    runtime_tree = ast.parse("\n".join(line[1:] for line in runtime_source.splitlines()
                                      if line.startswith("+")))
    runtime_class = next(n for n in runtime_tree.body if isinstance(n, ast.ClassDef)
                         and n.name == "OnlineCohortRuntime")
    diagnose = next(n for n in runtime_class.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_record_invalid_candidate")
    optimizer_type = _patched_online_optimizer()
    optimizer = optimizer_type((torch.zeros(2, 3, 2),), _online_config("onlinespec_ogd"))
    initial = tuple(value.clone() for value in optimizer.state_tensors)
    masks = []
    submissions = []
    def loss(logits, target, mask):
        masks.append(mask)
        return (logits.square().sum(-1) * mask).sum()

    namespace = {
        "torch": torch, "UpdateTrace": object,
        "OnlineSpecOptimizer": optimizer_type,
        "_logit_reconstruction_gate": _patched_logit_reconstruction_gate(),
        "anchor_forward_value": lambda surrogate, actual: surrogate + (actual - surrogate).detach(),
        "loss_and_grad": lambda parameters, objective: (
            objective(parameters), tuple(torch.zeros_like(p) for p in parameters)
        ),
    }
    exec(compile(ast.Module([launch, diagnose], []), str(patch), "exec"), namespace)
    runtime = SimpleNamespace(
        round=740, active_version=73, disabled_reason=None, pending=None,
        counters=dict.fromkeys(("updates_discarded", "updates_skipped_no_supervision",
                                "nonfinite_updates", "exactness_violations", "fallbacks"), 0),
        update_due=lambda: True, side_update=lambda tensors: nullcontext(),
        timing=lambda name: nullcontext(),
    )
    def submit(**candidate):
        submissions.append(candidate)
        valid = candidate["finite"] & candidate["reconstruction_ok"] & candidate["supervision_nonempty"]
        optimizer.commit(candidate["proposal"], valid=valid)
        if valid:
            runtime.active_version += 1
        else:
            trace = SimpleNamespace(diagnosed=False, buffer_index=0, published_version=None)
            namespace["_record_invalid_candidate"](
                runtime, trace, finite_ok=bool(candidate["finite"]),
                reconstruction_ok=bool(candidate["reconstruction_ok"]),
                supervision_ok=bool(candidate["supervision_nonempty"]),
            )
            runtime.trace = trace
    runtime.submit = submit
    adapter = SimpleNamespace(
        request_slots=None, runtime=runtime, optimizer=optimizer, names=("weight",),
        config=SimpleNamespace(adaptation_microbatch_size=1, optimizer=SimpleNamespace(name="sgd")),
        worker=SimpleNamespace(block_size=3), _captured_input=torch.zeros(2, 3, 2),
        _captured_positions=torch.zeros(6), _captured_prefix_lens=torch.tensor([527, 379]),
        _captured_request_ids=("ending", "continuing"),
        _captured_history=SimpleNamespace(locations=torch.zeros(2), valid_mask=torch.ones(2)),
        _active_inference_parameters=lambda: {"weight": optimizer.master[0] + (8 if mismatch else 0)},
        _effective_parameters=lambda parameters: {"weight": parameters[0]},
        _surrogate_hidden=lambda values, *args: values["weight"],
        _full_vocab_logits=lambda logits, size: logits, _distillation_loss=loss,
        inference=SimpleNamespace(stage=lambda values: None),
    )
    owned = torch.tensor([[selected_valid, selected_valid], [True, True]])
    namespace["maybe_launch"](
        adapter, draft_hidden=torch.zeros(2, 3, 2), target_logits=torch.zeros(2, 3, 2),
        valid_mask=owned, lm_head_weight=torch.eye(2),
    )
    candidate = submissions[0]
    assert bool(candidate["supervision_nonempty"]) == selected_valid
    assert len(masks) == 2 and all(torch.equal(mask, owned[:1]) for mask in masks)
    if selected_valid and not mismatch:
        assert runtime.active_version == 74
        assert runtime.disabled_reason is None
    else:
        assert runtime.active_version == 73
        assert all(torch.equal(a, b) for a, b in zip(initial, optimizer.state_tensors, strict=True))
        if not selected_valid:
            assert runtime.trace.status == "no_supervision"
            assert runtime.counters["updates_skipped_no_supervision"] == 1
            assert runtime.counters["fallbacks"] == 0
            assert runtime.disabled_reason is None
        else:
            assert runtime.disabled_reason == "logit_reconstruction_mismatch"
            assert runtime.counters["fallbacks"] == 1


def test_logit_reconstruction_replays_published_model_dtype_source():
    patch = Path("patches/sglang/0002-side-stream-adaptation-and-publication.diff")
    text = patch.read_text()
    assert "+                    active_values = self._active_inference_parameters()" in text
    assert (
        "+                        member_loss, gradients = "
        "member_feedback(self.optimizer.master)"
    ) in text
    added = []
    active = False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            active = "dflash_online_adaptation.py" in line
        elif active and line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    tree = ast.parse("\n".join(added))
    adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DFlashDrafterAdapter"
    )
    function = next(
        node
        for node in adapter.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_active_inference_parameters"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(patch), "exec"),
        namespace,
    )
    source = namespace["_active_inference_parameters"]

    published = torch.tensor([1.0], dtype=torch.bfloat16)
    full = SimpleNamespace(
        config=SimpleNamespace(weight_update_mode="full"),
        base={},
        names=("weight",),
        inference=SimpleNamespace(active=(published,)),
    )
    assert source(full)["weight"] is published

    frozen = torch.tensor([2.0], dtype=torch.float32)
    lora = SimpleNamespace(
        config=SimpleNamespace(weight_update_mode="lora"),
        base={"weight": torch.tensor([1.0001]), "frozen": frozen},
        names=("weight",),
        inference=SimpleNamespace(active=(published,)),
    )
    values = source(lora)
    assert values["weight"] is published
    assert values["frozen"] is frozen


def test_request_lora_slots_are_isolated_and_generation_safe():
    patch = Path("patches/sglang/0002-side-stream-adaptation-and-publication.diff")
    text = patch.read_text()
    assert "for optimizer in self.slot_optimizers\n+                        for tensor in optimizer.master" in text
    assert "for optimizer in self.slot_optimizers\n+                            for tensor in optimizer.first" in text
    assert "for optimizer in self.slot_optimizers\n+                            for tensor in optimizer.second" in text
    assert "for optimizer in self.slot_optimizers\n+                        for tensor in optimizer.metadata_tensors" in text
    added = []
    active = False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            active = "dflash_online_adaptation.py" in line
        elif active and line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    tree = ast.parse("\n".join(added))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"RequestLoRASlot", "RequestLoRASlotBank"}
    ]
    namespace = {"dataclass": dataclass}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(patch), "exec"), namespace)
    bank = namespace["RequestLoRASlotBank"](2)
    first = bank.acquire("first")
    second = bank.acquire("second")
    assert first.index != second.index
    assert bank.acquire("first") == first
    bank.release("first")
    replacement = bank.acquire("replacement")
    assert replacement.index == first.index
    assert replacement.generation == first.generation + 1
    assert bank.get("second") == second


def test_qwen_speculative_workspace_covers_registered_width():
    patch = Path("patches/sglang/0003-dspark-eagle3-nextn-adapters.diff").read_text()
    assert "cuda_graph_config.decode.max_bs" in patch
    assert "workspace_mb = 768 if graph_max_bs >= 256 else 512" in patch
    assert "speculative_num_draft_tokens or 0" not in patch


def _patched_dspark_verification_resolver():
    patch = Path("patches/sglang/0003-dspark-eagle3-nextn-adapters.diff")
    added = []
    active = False
    for line in patch.read_text().splitlines():
        if line.startswith("diff --git "):
            active = "dspark_online_adaptation.py" in line
        elif active and line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    tree = ast.parse("\n".join(added))
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "DSparkVerificationDecision"
        )
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "resolve_dspark_verification"
        )
    ]
    namespace = {"dataclass": dataclass}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(patch), "exec"), namespace)
    return namespace["resolve_dspark_verification"]


def test_dspark_fixed_budget_scales_to_the_exact_dynamic_batch():
    resolve = _patched_dspark_verification_resolver()
    full = resolve(
        mode="fixed_budget",
        batch_size=128,
        verify_width=8,
        fixed_total_token_budget=256,
        native_total_token_budget=None,
        configured_batch_size=128,
    )
    partial = resolve(
        mode="fixed_budget",
        batch_size=1,
        verify_width=8,
        fixed_total_token_budget=256,
        native_total_token_budget=None,
        configured_batch_size=128,
    )
    assert (full.total_token_budget, full.additional_token_budget) == (256, 128)
    assert (partial.total_token_budget, partial.additional_token_budget) == (2, 1)
    with pytest.raises(ValueError, match="configured batch"):
        resolve(
            mode="fixed_budget",
            batch_size=1,
            verify_width=8,
            fixed_total_token_budget=257,
            native_total_token_budget=None,
            configured_batch_size=128,
        )


def test_online_adaptation_config_retains_registered_batch_size():
    patch = Path("patches/sglang/0001-arguments-config-and-memory.diff").read_text()
    assert "+    max_in_flight: int" in patch
    assert '+        max_in_flight = int(value.get("max_in_flight", 1))' in patch
    assert "+            max_in_flight=max_in_flight," in patch


def test_unified_dspark_model_accepts_markovless_dflash_checkpoint():
    patch = Path("patches/sglang/0003-dspark-eagle3-nextn-adapters.diff").read_text()
    assert "+        return None" in patch
    assert "-        if not dspark_config.require_markov():" in patch
    assert "-                \"DSpark draft requires markov_rank > 0, \"" in patch
    assert 'if hasattr(self.draft_model, "attach_shared_modules"):' in patch
    assert "self.draft_model.attach_shared_modules(" in patch


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
    launch_count = 0

    class Process:
        pid = 12345
        returncode = None

        def poll(self): return self.returncode
        def wait(self, timeout=None): self.returncode = 0

    def launch(*args, **kwargs):
        nonlocal launch_count
        launch_count += 1
        launched.update(kwargs)
        return Process()

    monkeypatch.setattr("lightcone_spec.server.subprocess.Popen", launch)
    monkeypatch.setattr("lightcone_spec.server.GpuSampler.start", lambda self: None)
    monkeypatch.setattr("lightcone_spec.server.GpuSampler.stop", lambda self: None)
    monkeypatch.setattr("lightcone_spec.server.SGLangClient.health", lambda self: True)
    monkeypatch.setattr("lightcone_spec.server.SGLangClient.reset", lambda self: None)
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
        and job.method == "lightcone"
        and job.parameters.get("workload") == "dspark_confidence_capture"
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
        assert dspark.adaptation["confidence_loss_weight"] == 1.0
        calibration = next(
            job
            for job in materialize("E1a")
            if job.parameters.get("workload") == "dspark_confidence_capture"
        )
        low = replace(
            calibration,
            parameters={**calibration.parameters, "confidence_threshold": 0.0},
        )
        high = replace(
            calibration,
            parameters={**calibration.parameters, "confidence_threshold": 0.9},
        )
        assert server_session_key(low) == server_session_key(high) == dspark.session_key
        launches_before_reconfigure = launch_count
        dspark.configure(low, None)
        dspark.configure(high, None)
        assert launch_count == launches_before_reconfigure
        reloaded = json.loads((output / "adaptation.json").read_text())
        assert reloaded["confidence_threshold"] == 0.9


def test_onlinespec_payload_contains_independent_learner_settings():
    job = next(
        job for job in materialize("E0-tune") if job.method == "onlinespec_ens"
    )
    payload = adaptation_payload(job)
    assert payload is not None
    assert payload["method"] == "onlinespec_ens"
    assert payload["online_spec"]["additional_learning_rates"]
    assert payload["online_spec"]["hedge_learning_rate"] > 0
    assert payload["online_spec"]["hint_momentum"] == 0.9


def _patched_online_optimizer():
    patch = Path("patches/sglang/0002-side-stream-adaptation-and-publication.diff")
    added = []
    active = False
    for line in patch.read_text().splitlines():
        if line.startswith("diff --git "):
            active = "online_adaptation_runtime.py" in line
        elif active and line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    tree = ast.parse("\n".join(added))
    wanted = {
        "_clip_fp32_gradients",
        "_project_online_parameters",
        "ParameterProposal",
        "OnlineSpecMemberProposal",
        "OnlineSpecOptimizer",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    ]
    namespace = {
        "dataclass": dataclass,
        "OnlineAdaptationConfig": object,
        "Sequence": Sequence,
        "torch": torch,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(patch), "exec"), namespace)
    return namespace["OnlineSpecOptimizer"]


def _online_config(method: str):
    return SimpleNamespace(
        method=method,
        optimizer=SimpleNamespace(learning_rate=0.1, grad_clip=1.0),
        online_spec=SimpleNamespace(
            projection_radius=None,
            additional_learning_rates=(0.2, 0.3) if method == "onlinespec_ens" else (),
            hedge_learning_rate=1.0 if method == "onlinespec_ens" else None,
            hint_momentum=0.9,
        ),
    )


def test_onlinespec_opt_uses_committed_historical_hint_and_reset_is_transactional():
    optimizer_type = _patched_online_optimizer()
    optimizer = optimizer_type((torch.tensor([0.0]),), _online_config("onlinespec_opt"))
    first = optimizer.propose((torch.tensor([1.0]),))
    assert first.parameters[0].item() == pytest.approx(-0.2)
    optimizer.commit(first)

    second = optimizer.propose((torch.tensor([0.5]),))
    assert second.second_moments[0].item() == pytest.approx(1.4)
    assert second.parameters[0].item() == pytest.approx(-0.29)
    before = tuple(value.clone() for value in optimizer.state_tensors)
    optimizer.commit(second, valid=torch.tensor(False))
    assert all(torch.equal(left, right) for left, right in zip(before, optimizer.state_tensors))

    optimizer.reset((torch.tensor([0.0]),))
    replay = optimizer.propose((torch.tensor([1.0]),))
    assert replay.parameters[0].item() == pytest.approx(first.parameters[0].item())
    assert replay.second_moments[0].item() == pytest.approx(1.0)


def test_gpu_acceptance_uses_frozen_online_spec_recipes():
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / "gpu_acceptance.py"
    spec = importlib.util.spec_from_file_location("gpu_acceptance_recipes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ogd = module._job(0, "onlinespec_ogd", "DFLASH", block=0)
    opt = module._job(1, "onlinespec_opt", "DFLASH", block=0)
    ens = module._job(2, "onlinespec_ens", "DFLASH", block=0)

    assert ogd.parameters["learning_rate"] == pytest.approx(3e-5)
    assert ogd.parameters["stride"] == 10
    assert opt.parameters["learning_rate"] == pytest.approx(1e-1)
    assert opt.parameters["hint_momentum"] == pytest.approx(0.9)
    assert ens.parameters["additional_learning_rates"] == (6e-5, 1.2e-4)
    assert ens.parameters["hedge_learning_rate"] == pytest.approx(1.0)


def test_gpu_acceptance_requires_confidence_only_for_adaptive_dspark():
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / "gpu_acceptance.py"
    spec = importlib.util.spec_from_file_location("gpu_acceptance_confidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    static = module._job(0, "static", "DSPARK", block=0, tp2=True)
    adaptive = module._job(1, "lightcone", "DSPARK", block=0, tp2=True)
    assert not module._requires_confidence_metrics(static)
    assert module._requires_confidence_metrics(adaptive)


def test_onlinespec_ensemble_experts_and_hedge_weights_remain_independent():
    optimizer_type = _patched_online_optimizer()
    optimizer = optimizer_type((torch.tensor([0.0]),), _online_config("onlinespec_ens"))
    proposal = optimizer.propose_ensemble(
        torch.tensor([0.1, 0.2, 0.3]),
        ((torch.tensor([1.0]),), (torch.tensor([1.0]),), (torch.tensor([1.0]),)),
    )
    experts = [value.item() for value in proposal.first_moments]
    assert experts == pytest.approx([-0.1, -0.2, -0.3])
    assert len(set(experts)) == 3
    optimizer.commit(proposal)
    probabilities = optimizer.expert_probabilities
    assert probabilities is not None
    assert torch.isfinite(probabilities).all()
    assert probabilities.sum().item() == pytest.approx(1.0)


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
                "registered_request_count": 3,
            },
        }
    )
    payload = adaptation_payload(job)
    assert payload["optimizer"]["schedule_total_published_updates"] == max(
        8, math.ceil(3 * job.parameters["generation_tokens"] / 20)
    )
    muon_job = job.__class__(
        **{
            **job.to_dict(),
            "parameters": {
                **job.parameters,
                "optimizer": "muon",
                "learning_rate": 3e-4,
                "weight_decay": 0.01,
                "schedule": "constant",
            },
        }
    )
    muon = adaptation_payload(muon_job)["optimizer"]
    assert muon["momentum"] == 0.95
    assert muon["muon_ns_steps"] == 5
    assert muon["muon_auxiliary_learning_rate"] == 3e-4
    assert muon["muon_auxiliary_weight_decay"] == 0.01
    calibration = next(
        item
        for item in materialize("E1a")
        if item.parameters.get("workload") == "dspark_confidence_capture"
    )
    e1a = adaptation_payload(
        materialize("E1a")[0],
        {"optimizer": "adamw", "confidence_loss_weight": 0.25},
    )
    assert e1a["fixed_total_token_budget"] == 8
    assert e1a["confidence_loss_weight"] == 1.0
    assert e1a["canvas_tokens"] == 8
    assert calibration.parameters["regime"] == "short_input_long_generation"
    assert calibration.parameters["generation_tokens"] == 8192
    assert calibration.parameters["source_transfer_recipe"] == "dflash_lightcone_recipe"
    latency = next(
        job
        for job in materialize("E1a")
        if job.parameters.get("workload") == "dspark_source_latency_panel"
    )
    latency_segment = latency.__class__(
        **{
            **latency.to_dict(),
            "context": latency.parameters["segments"][0]["context"],
            "load": latency.parameters["segments"][0]["load"],
            "parameters": {
                **latency.parameters,
                **latency.parameters["segments"][0],
            },
        }
    )
    latency_payload = adaptation_payload(latency_segment)
    assert latency_payload["max_in_flight"] == 128
    assert latency_payload["fixed_total_token_budget"] == 256
    assert latency_segment.parameters["regime"] == "long_input_short_output"
    assert latency_segment.parameters["generation_tokens"] == 256
    replacement = calibration.__class__(
        **{
            **calibration.to_dict(),
            "node": "bugfix-reconciliation-v1",
            "parameters": {
                **calibration.parameters,
                "source_node": "E1a",
                "confidence_threshold": 0.5,
                "save_confidence_outcomes": True,
            },
        }
    )
    replacement_payload = adaptation_payload(replacement)
    assert replacement_payload["confidence_loss_weight"] == calibration.parameters[
        "confidence_loss_weight"
    ]
    assert replacement_payload["confidence_threshold"] == 0.5
    assert replacement_payload["save_confidence_outcomes"] is True


def test_tts_always_resets_between_requests():
    tts_cal = materialize("TTS-Cal")[0]
    assert adaptation_payload(tts_cal)["reset_scope"] == "request"
    assert adaptation_payload(tts_cal)["telemetry_round_items"] == tts_cal.context

    e3 = next(
        job
        for job in materialize("E3b-pilot")
        if job.method == "tts" and job.load == "c1"
    )
    concurrent = e3.__class__(**{**e3.to_dict(), "load": "c8"})
    assert adaptation_payload(concurrent)["reset_scope"] == "request"
    assert _uses_request_scope(concurrent)
    assert _dispatcher_concurrency(concurrent) == 1

    single = concurrent.__class__(**{**concurrent.to_dict(), "load": "c1"})
    assert adaptation_payload(single)["reset_scope"] == "request"

    l0 = concurrent.__class__(**{**concurrent.to_dict(), "method": "l0_naive"})
    assert adaptation_payload(l0)["reset_scope"] == "request"
    assert _uses_request_scope(l0)

    lightcone = concurrent.__class__(
        **{**concurrent.to_dict(), "method": "lightcone"}
    )
    assert adaptation_payload(lightcone)["reset_scope"] == "cohort"
    assert adaptation_payload(lightcone)["telemetry_round_items"] == 3_000_000
    assert not _uses_request_scope(lightcone)
    assert _dispatcher_concurrency(lightcone) == 8


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

    c256 = replace(
        c64,
        job_id="e5-c256",
        node="E5-pilot",
        load="closed_loop_c256",
        parameters={**c64.parameters, "server_capacity": 256},
    )
    command = server_command(
        config, c256, port=30000, output_dir=tmp_path, adaptation=None
    )
    assert command[command.index("--max-running-requests") + 1] == "256"
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
    assert command[graph_index + 8].startswith("--")


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


def test_request_scoped_multi_turn_uses_one_runtime_session_per_conversation():
    class Client:
        def __init__(self):
            self.calls = []
            self.resets = 0

        def run_batch(self, prompts, *, max_new_tokens, seed, request_ids):
            self.calls.append((prompts[0], request_ids[0]))
            result = GenerationResult(
                request_id=request_ids[0],
                input_tokens=len(prompts[0]),
                completion_tokens=1,
                ttft_ms=1.0,
                inter_token_ms=(),
                elapsed_seconds=0.001,
                stop_reason="length",
                output_ids=(seed + 1,),
                output_text="",
                native_token_timestamps_ns=(seed + 1,),
            )
            return (result,), 0.001

        def reset(self):
            self.resets += 1

    client = Client()
    results, _ = _run_multi_turn(
        client,
        ((7,), (8,)),
        8,
        0,
        request_scoped=True,
        max_in_flight=1,
    )
    assert len(results) == 2
    assert client.resets == 0
    assert [request_id for _, request_id in client.calls[:4]] == [
        "multi-turn-00000::turn-0::of-4",
        "multi-turn-00000::turn-1::of-4",
        "multi-turn-00000::turn-2::of-4",
        "multi-turn-00000::turn-3::of-4",
    ]
    assert len(client.calls[0][0]) < len(client.calls[1][0]) < len(client.calls[2][0])


def test_request_scope_session_markers_reset_only_on_final_turn():
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
    functions = {
        node.name: node
        for node in runtime.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_request_scope_owner", "_request_scope_session_complete"}
    }
    namespace = {}
    for function in functions.values():
        function.decorator_list = []
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(patch), "exec"),
            namespace,
        )
    first = "multi-turn-00000::turn-0::of-4"
    final = "multi-turn-00000::turn-3::of-4"
    assert namespace["_request_scope_owner"](first) == "multi-turn-00000"
    assert not namespace["_request_scope_session_complete"](first)
    assert namespace["_request_scope_session_complete"](final)
    assert namespace["_request_scope_session_complete"]("ordinary-request")


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


def test_eagle3_rejection_sampling_uses_single_branch(tmp_path: Path):
    job = next(item for item in materialize("E0-tune") if item.backend == "EAGLE3")
    model = tmp_path / "model"
    draft = tmp_path / "draft"
    model.mkdir()
    draft.mkdir()
    config = ExperimentConfig(
        source=tmp_path / "paper.yaml",
        run_name="run",
        sglang_root=tmp_path,
        results_root=tmp_path,
        models={job.model: model},
        drafts={f"{job.model}|EAGLE3": draft},
        datasets={},
        gpu_ids=(0, 1),
        server=ServerConfig(python=tmp_path / "python"),
        protocol=ProtocolConfig(),
    )
    command = server_command(
        config,
        job,
        port=30000,
        output_dir=tmp_path,
        adaptation=None,
    )
    assert "--speculative-use-rejection-sampling" in command
    index = command.index("--speculative-eagle-topk")
    assert command[index + 1] == "1"
    steps = command.index("--speculative-num-steps")
    draft_tokens = command.index("--speculative-num-draft-tokens")
    assert command[steps + 1] == "15"
    assert command[draft_tokens + 1] == "16"

    adaptive = next(
        item
        for item in materialize("E0-tune")
        if item.backend == "EAGLE3" and item.method == "tts"
    )
    adaptive_command = server_command(
        config,
        adaptive,
        port=30000,
        output_dir=tmp_path,
        adaptation=adaptation_payload(adaptive),
    )
    steps = adaptive_command.index("--speculative-num-steps")
    assert adaptive_command[steps + 1] == "15"
    assert adaptation_payload(adaptive)["canvas_tokens"] == 16


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


@pytest.mark.parametrize("module,class_name,has_model", [
    ("gemma4_draft", "Gemma4Eagle3Model", True),
    ("gemma4_draft", "Gemma4DSparkModel", False),
    ("qwen3_draft_replay", "QwenDraftReplay", True),
    ("qwen3_draft_replay", "QwenDraftReplay", False),
])
def test_native_warmup_is_not_shadow_training(module, class_name, has_model):
    tree = _coverage_added_module(f"python/sglang/srt/models/{module}.py")
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    forward = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "forward")

    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.history = False
            if has_model:
                self.model = SimpleNamespace()

        @torch.no_grad()
        def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
            value = input_ids * self.weight
            return SimpleNamespace(next_token_logits=value, hidden_states=value)

        def _replay_hidden(self, input_ids, *args):
            if not self.history:
                raise RuntimeError("owned full-history KV snapshot required")
            value = input_ids * self.weight
            return value, value

        def _replay_logits(self, hidden):
            return hidden

    wrapper = ast.ClassDef(
        name=class_name, bases=[ast.Name(id="Native", ctx=ast.Load())],
        keywords=[], body=[forward], decorator_list=[],
    )
    namespace = {"Native": Native, "torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), "native-forward-mode", "exec"), namespace)
    model = namespace[class_name]()
    values = torch.ones(2, 4)
    assert torch.is_grad_enabled()  # CUDA Graph warm-up may enter this way.
    result = model(values, None, None)
    assert not result.hidden_states.requires_grad
    assert not getattr(model, "_reconstruction_rows", [])
    with native_training_replay(model, False):
        assert not model(values, None, None).hidden_states.requires_grad
    with pytest.raises(RuntimeError, match="owned full-history"):
        with native_training_replay(model, True):
            model(values, None, None)
    assert not hasattr(model, "_lightcone_training_replay")
    model.history = True
    with native_training_replay(model, True):
        assert model(values, None, None).hidden_states.requires_grad
    with grad_enabled_forwards(model):
        with grad_enabled_forwards(model):
            assert model._lightcone_training_replay is True
        assert model._lightcone_training_replay is True
        output = model(values, None, None)
        assert output.hidden_states.requires_grad
        output.hidden_states.sum().backward()
    assert model.weight.grad is not None and model.weight.grad.item() != 0
    assert not hasattr(model, "_lightcone_training_replay")
    assert not model(values, None, None).hidden_states.requires_grad


@pytest.mark.parametrize("tokenizer_limit", [int(1e30), 131072, -1])
def test_tokenize_reports_server_limit_not_hf_sentinel(tokenizer_limit):
    import textwrap

    patch = Path("patches/sglang/0005-nextn-shadow-replay.diff").read_text()
    chunk = patch.split(
        "diff --git a/python/sglang/srt/entrypoints/openai/serving_tokenize.py", 1
    )[1].split("\ndiff --git ", 1)[0]
    code = textwrap.dedent("\n".join(
        line[1:] for line in chunk.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ))
    tokenizer = SimpleNamespace(model_max_length=tokenizer_limit)
    namespace = {"self": SimpleNamespace(tokenizer_manager=SimpleNamespace(
        tokenizer=tokenizer, context_len=40960,
    ))}
    exec(compile(code, "tokenize-model-limit", "exec"), namespace)
    assert namespace["max_model_len"] == 40960
    assert tokenizer.model_max_length == tokenizer_limit
    assert -(2**63) <= namespace["max_model_len"] < 2**63


def test_deepspec_eagle_graph_width_uses_trained_feature_layers():
    patch = Path("patches/sglang/0005-nextn-shadow-replay.diff").read_text()
    section = patch.split(
        "diff --git a/python/sglang/srt/speculative/eagle_utils.py ", 1
    )[1].split("\ndiff --git ", 1)[0].split("\n@@ ", 1)[1]
    source = "\n".join(
        line[1:] for line in section.splitlines()[1:] if line.startswith(("+", " "))
    )
    namespace = {"ModelRunner": object}
    exec(source, namespace)
    width = namespace["get_draft_input_from_target_hidden_dim"]

    def runner(hidden, **fields):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(**fields), hidden_size=hidden, spec_hidden_size=hidden,
            ),
            spec_algorithm=SimpleNamespace(is_eagle3=lambda: True),
        )

    for hidden, layers in (
        (3840, [5, 17, 29, 41, 46]),
        (2560, [1, 9, 17, 25, 33]),
        (4096, [1, 9, 17, 25, 33]),
        (5120, [1, 10, 19, 28, 37]),
    ):
        resolved = width(runner(hidden, target_layer_ids=layers))
        assert resolved == hidden * len(layers)
        # The same buffer width is used by both draft-extend and prefill graphs.
        features = torch.zeros(8, resolved)
        projection = torch.ones(2, hidden * len(layers))
        assert torch.nn.functional.linear(features, projection).shape == (8, 2)
    assert width(runner(3840)) == 3840 * 3  # Historical EAGLE defaults unchanged.
    assert width(runner(3840, target_hidden_size=4096, target_layer_ids=[1, 2])) == 8192
    assert width(runner(3840, num_aux_hidden_states=2, target_layer_ids=[1] * 5)) == 7680
    assert width(runner(3840, eagle_aux_hidden_state_layer_ids=[1, 2], target_layer_ids=[1] * 5)) == 7680
    assert width(runner(3840, eagle_config={"eagle_aux_hidden_state_layer_ids": [1]}, target_layer_ids=[1] * 5)) == 3840
    assert width(runner(3840, eagle_config={"use_aux_hidden_state": False}, target_layer_ids=[1] * 5)) == 3840
    non_eagle = runner(3840, target_layer_ids=[1] * 5)
    non_eagle.spec_algorithm = None
    assert width(non_eagle) == 3840


def test_gemma_draft_allocator_preserves_strict_checkpoint_window():
    import textwrap

    patch = Path("patches/sglang/0005-nextn-shadow-replay.diff").read_text()
    section = patch.split(
        "diff --git a/python/sglang/srt/configs/model_config.py ", 1
    )[1].split("\ndiff --git ", 1)[0]
    hunks = section.split("\n@@ ")[1:]
    initializer = "\n".join(
        line[1:] for line in hunks[0].splitlines()[1:] if line.startswith("+")
    )
    method = "\n".join(
        line[1:] for line in hunks[1].splitlines()[1:] if line.startswith(("+", " "))
    )
    namespace = {"Optional": __import__("typing").Optional}
    exec(textwrap.dedent(method), namespace)

    class StrictGemmaText(SimpleNamespace):
        def __setattr__(self, name, value):
            if name == "sliding_window" and not isinstance(value, int):
                raise TypeError("sliding_window must retain its checkpoint integer")
            super().__setattr__(name, value)

    for architecture, is_draft, expected_window in (
        ("Gemma4DSparkModel", True, None),
        ("Gemma4Eagle3Model", True, None),
        ("Gemma4UnifiedForConditionalGeneration", False, 4096),
        ("Qwen3Eagle3Model", True, 4096),
    ):
        text = StrictGemmaText(
            sliding_window=4096, head_dim=128, v_head_dim=128,
            global_head_dim=256, attention_k_eq_v=True,
            num_key_value_heads=8, num_global_key_value_heads=2,
            layer_types=["sliding_attention", "full_attention"], num_hidden_layers=2,
        )
        config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture]),
            hf_text_config=text, is_draft_model=is_draft,
        )
        exec(textwrap.dedent(initializer), {"self": config, "is_draft_model": is_draft})
        assert text.sliding_window == 4096
        assert namespace["_get_sliding_window_size"](config) == expected_window
        if expected_window is None:
            assert text.layer_types == ["full_attention"] * 2
            assert text.head_dim == text.v_head_dim == 256
            assert text.num_key_value_heads == 2
        else:
            assert text.layer_types == ["sliding_attention", "full_attention"]
            assert text.head_dim == 128 and text.num_key_value_heads == 8


def test_gemma_shared_kv_math_and_all_norm_gradients():
    import torch
    import torch.nn.functional as F

    from lightcone_spec.gemma import (
        gemma_canvas_attention,
        gemma_residual_mlp,
        gemma_rms,
        gemma_rotary,
        gemma_softcap,
    )

    torch.manual_seed(7)
    hidden = torch.randn(2, 3, 8)
    q_weight = torch.randn(16, 8, requires_grad=True)
    k_weight = torch.randn(4, 8, requires_grad=True)
    q_norm = torch.randn(4, requires_grad=True)
    k_norm = torch.randn(4, requires_grad=True)
    query = gemma_rms(F.linear(hidden, q_weight).view(2, 3, 4, 4), q_norm, 1e-6)
    raw_kv = F.linear(hidden, k_weight).view(2, 3, 1, 4)
    key = gemma_rms(raw_kv, k_norm, 1e-6)
    value = gemma_rms(raw_kv, None, 1e-6)
    history_k, history_v = torch.randn(2, 2, 1, 4), torch.randn(2, 2, 1, 4)
    valid = torch.tensor([[True, False], [True, True]])
    output = gemma_canvas_attention(query, key, value, history_k, history_v, valid)
    all_k = torch.cat((history_k, key), dim=1).repeat_interleave(4, dim=2)
    all_v = torch.cat((history_v, value), dim=1).repeat_interleave(4, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", query, all_k)
    mask = torch.cat((valid, torch.ones(2, 3, dtype=torch.bool)), dim=1)
    scores = scores.masked_fill(~mask[:, None, None], -torch.inf)
    reference = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), all_v).flatten(-2)
    assert torch.allclose(output, reference, atol=2e-6)
    grads = torch.autograd.grad(output.square().sum(), (q_weight, k_weight, q_norm, k_norm))
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in grads)
    poisoned = history_v.clone()
    poisoned[0, 1] = 1e6
    assert torch.allclose(
        output, gemma_canvas_attention(query, key, value, history_k, poisoned, valid)
    )
    # A partial rotary cache rotates only two of four coordinates.
    positions = torch.tensor([[0, 1, 2], [0, 1, 2]])
    cache = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    rotated = gemma_rotary(value, positions, cache)
    assert torch.equal(rotated[..., 2:], value[..., 2:])
    assert torch.equal(rotated[:, 0], value[:, 0])
    parameters = [torch.randn(8, requires_grad=True) for _ in range(3)]
    parameters += [torch.randn(12, 8, requires_grad=True), torch.randn(8, 6, requires_grad=True)]
    residual = gemma_residual_mlp(
        hidden, torch.randn_like(hidden), *parameters, torch.tensor([0.8]), 1e-6
    )
    grads = torch.autograd.grad(residual.square().sum(), parameters)
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    logits = torch.tensor([-100.0, 0.0, 100.0])
    assert torch.allclose(gemma_softcap(logits, 30), 30 * torch.tanh(logits / 30))


def _coverage_added_module(relative_path):
    patch = Path("patches/sglang/0005-nextn-shadow-replay.diff").read_text()
    marker = f"diff --git a/{relative_path} b/{relative_path}\n"
    chunk = patch.split(marker, 1)[1].split("\ndiff --git ", 1)[0]
    return ast.parse(
        "\n".join(
            line[1:]
            for line in chunk.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
    )


@pytest.mark.parametrize("eagle", (False, True))
def test_official_qwen_replay_has_all_layer_gradients(monkeypatch, eagle):
    import types

    import torch.nn.functional as F

    from lightcone_spec.gemma import gemma_rms

    tree = _coverage_added_module("python/sglang/srt/models/qwen3_draft_replay.py")
    namespace = {"torch": torch, "F": F}
    exec(
        compile(
            ast.Module([item for item in tree.body if isinstance(item, ast.ClassDef)], []),
            "qwen-replay",
            "exec",
        ),
        namespace,
    )
    distributed = types.ModuleType("sglang.srt.distributed")
    distributed.get_tp_group = lambda: SimpleNamespace(world_size=1)
    wrappers = types.ModuleType("sglang.srt.models.gemma4_draft")
    wrappers._column = lambda module, value: F.linear(value, module.weight)
    wrappers._row = lambda module, value: F.linear(value, module.weight)
    wrappers._tp_copy = lambda value: value
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    monkeypatch.setitem(sys.modules, wrappers.__name__, wrappers)
    torch.manual_seed(113)
    parameters = []

    def parameter(shape):
        value = torch.nn.Parameter(torch.randn(shape) * 0.2)
        parameters.append(value)
        return value

    def linear(out_features, in_features):
        return SimpleNamespace(weight=parameter((out_features, in_features)), bias=None)

    def norm(size):
        return SimpleNamespace(weight=parameter((size,)), variance_epsilon=1e-6)

    attention = SimpleNamespace(
        total_num_kv_heads=2,
        num_kv_heads=2,
        num_heads=2,
        head_dim=4,
        q_size=8,
        kv_size=8,
        qkv_proj=linear(24, 16 if eagle else 8),
        o_proj=linear(8, 8),
        q_norm=norm(4),
        k_norm=norm(4),
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.tensor([[1.0, 1.0, 0.0, 0.0]]).repeat(8, 1)),
    )
    layer = SimpleNamespace(
        self_attn=attention,
        input_layernorm=norm(8),
        post_attention_layernorm=norm(8),
        mlp=SimpleNamespace(gate_up_proj=linear(12, 8), down_proj=linear(8, 6)),
    )
    if eagle:
        layer.hidden_norm = norm(8)
    embedding = torch.nn.Embedding(9, 8)
    backbone = SimpleNamespace(embed_tokens=embedding, layers=[layer], norm=norm(8))
    model = namespace["QwenDraftReplay"]()
    model.config = SimpleNamespace(hidden_size=8)
    if eagle:
        model.model = backbone
    else:
        model.embed_tokens, model.layers, model.norm = (
            backbone.embed_tokens,
            backbone.layers,
            backbone.norm,
        )
    count = 1 if eagle else 3
    ids = torch.arange(2 * count)
    previous = torch.randn(2 * count, 8)
    batch = SimpleNamespace(spec_info=SimpleNamespace(hidden_states=previous))
    history_k, history_v = torch.randn(2, 2, 2, 4), torch.randn(2, 2, 2, 4)
    valid = torch.tensor([[True, False], [True, True]])
    model._training_histories = [(history_k, history_v, valid)]
    actual, _ = model._replay_hidden(ids, torch.zeros(2 * count, dtype=torch.long), batch)

    def rms(module, value):
        return gemma_rms(value, module.weight, 1e-6)

    embedded = embedding(ids).detach()
    hidden = previous if eagle else embedded
    inputs = (
        torch.cat((rms(layer.input_layernorm, embedded), rms(layer.hidden_norm, hidden)), -1)
        if eagle
        else rms(layer.input_layernorm, hidden)
    )
    q, k, v = F.linear(inputs, attention.qkv_proj.weight).chunk(3, -1)
    q = rms(attention.q_norm, q.view(2, count, 2, 4))
    k = torch.cat((history_k, rms(attention.k_norm, k.view(2, count, 2, 4))), 1)
    v = torch.cat((history_v, v.view(2, count, 2, 4)), 1)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / 2
    mask = torch.cat((valid, torch.ones(2, count, dtype=torch.bool)), 1)
    scores = scores.masked_fill(~mask[:, None, None], -torch.inf)
    attended = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v).reshape(-1, 8)
    reference = hidden + F.linear(attended, attention.o_proj.weight)
    gate, up = F.linear(
        rms(layer.post_attention_layernorm, reference), layer.mlp.gate_up_proj.weight
    ).chunk(2, -1)
    reference = rms(
        backbone.norm, reference + F.linear(F.silu(gate) * up, layer.mlp.down_proj.weight)
    )
    assert torch.allclose(actual, reference, atol=2e-6)
    actual_grad = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    expected_grad = torch.autograd.grad(reference.square().sum(), parameters)
    for measured, expected in zip(actual_grad, expected_grad, strict=True):
        assert torch.isfinite(measured).all() and torch.count_nonzero(measured)
        assert torch.allclose(measured, expected, atol=2e-5, rtol=2e-4)


def _gemma_tp_gradient_worker(rank, init_path):
    """Actual two-process collectives, not forward-only simulated sharding."""
    import types

    import torch.distributed as dist
    import torch.nn.functional as functional

    from lightcone_spec.gemma import gemma_canvas_attention, gemma_rms, gemma_rotary

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        # Load the exact runtime collective autograd definitions, without importing
        # the CUDA-serving dependency tree in a CPU regression test.
        native_patch = Path("patches/sglang/0005-nextn-shadow-replay.diff").read_text().splitlines()
        start = next(
            index
            for index, line in enumerate(native_patch)
            if line.startswith("+class _TensorParallelCopy(")
        )
        native_lines = []
        for line in native_patch[start:]:
            if not line.startswith("+"):
                break
            native_lines.append(line[1:])
        native_tree = ast.parse("\n".join(native_lines))
        dflash_tree = _patched_residual_rms()[1]
        namespace = {"torch": torch, "dist": dist}
        classes = [
            item
            for tree in (native_tree, dflash_tree)
            for item in tree.body
            if isinstance(item, ast.ClassDef)
            and item.name in {"_TensorParallelCopy", "_TensorParallelSum"}
        ]
        exec(compile(ast.Module(classes, []), "runtime-collectives", "exec"), namespace)
        for module_name, symbol in (
            ("sglang.srt.speculative.native_backend_online_adaptation", "_TensorParallelCopy"),
            ("sglang.srt.speculative.dflash_online_adaptation", "_TensorParallelSum"),
        ):
            stub = types.ModuleType(module_name)
            setattr(stub, symbol, namespace[symbol])
            sys.modules[module_name] = stub
        tree = _coverage_added_module("python/sglang/srt/models/gemma4_draft.py")
        functions = [
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef)
            and item.name in {"_tp_copy", "_column", "_row", "_replay_attention"}
        ]
        namespace.update(
            F=functional,
            gemma_rms=gemma_rms,
            gemma_rotary=gemma_rotary,
            gemma_canvas_attention=gemma_canvas_attention,
            get_tp_group=lambda: SimpleNamespace(world_size=2, device_group=dist.group.WORLD),
        )
        exec(compile(ast.Module(functions, []), "gemma-replay", "exec"), namespace)
        torch.manual_seed(710)
        hidden = torch.randn(2, 8, requires_grad=True)
        q = torch.randn(16, 8, requires_grad=True)
        k = torch.randn(4, 8, requires_grad=True)
        o = torch.randn(8, 16, requires_grad=True)
        qn = torch.randn(4, requires_grad=True)
        kn = torch.randn(4, requires_grad=True)
        hist = (
            torch.randn(1, 3, 1, 4),
            torch.randn(1, 3, 1, 4),
            torch.tensor([[True, True, False]]),
        )
        positions = torch.tensor([3, 4])
        cache = torch.cat((torch.ones(8, 1), torch.zeros(8, 1)), -1)
        query = gemma_rotary(
            gemma_rms(functional.linear(hidden, q).view(1, 2, 4, 4), qn, 1e-6),
            positions.view(1, 2),
            cache,
        )
        raw_key = functional.linear(hidden, k).view(1, 2, 1, 4)
        key = gemma_rotary(gemma_rms(raw_key, kn, 1e-6), positions.view(1, 2), cache)
        value = gemma_rms(raw_key, None, 1e-6)
        reference = functional.linear(
            gemma_canvas_attention(query, key, value, *hist).view(2, 16), o
        )
        expected = torch.autograd.grad(reference.square().mean(), (hidden, q, k, o, qn, kn))
        local = [
            tensor.detach().clone().requires_grad_()
            for tensor in (
                hidden,
                q.chunk(2, 0)[rank],
                k,
                o.chunk(2, 1)[rank],
                qn,
                kn,
            )
        ]
        attention = SimpleNamespace(
            q_proj=SimpleNamespace(weight=local[1], bias=None),
            k_proj=SimpleNamespace(weight=local[2], bias=None),
            o_proj=SimpleNamespace(weight=local[3], bias=None),
            q_norm=SimpleNamespace(weight=local[4], eps=1e-6),
            k_norm=SimpleNamespace(weight=local[5], eps=1e-6),
            v_norm=SimpleNamespace(eps=1e-6),
            rotary_emb=SimpleNamespace(cos_sin_cache=cache),
            num_heads=2,
            num_kv_heads=1,
            total_num_kv_heads=1,
            head_dim=4,
            tied_kv=True,
            q_size=8,
        )
        actual = namespace["_replay_attention"](attention, local[0], positions, hist, eagle=False)
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
        gradients = torch.autograd.grad(actual.square().mean(), local)
        expected = (
            expected[0],
            expected[1].chunk(2, 0)[rank],
            expected[2],
            expected[3].chunk(2, 1)[rank],
            expected[4],
            expected[5],
        )
        for gradient, reference_gradient in zip(gradients, expected, strict=True):
            torch.testing.assert_close(gradient, reference_gradient, atol=3e-5, rtol=3e-5)
    finally:
        dist.destroy_process_group()


def test_gemma_tp2_replicated_kv_and_norm_gradients_match_unsplit_reference(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_gemma_tp_gradient_worker, args=(str(tmp_path / "gloo-init"),), nprocs=2, join=True)


def _native_lora_tp_worker(rank, init_path):
    import torch.distributed as dist
    import torch.nn.functional as F

    from lightcone_spec.native_tp import LowRankDelta, global_gradient_norm

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        for partition in ("column", "row"):
            torch.manual_seed(240)
            base = torch.randn(8, 8)
            inputs = torch.randn(5, 8)
            target = torch.randn(5, 8)
            reference = LowRankDelta(base, rank=2, seed=71, scale=2.0)
            shard = base.chunk(2, 0 if partition == "column" else 1)[rank]
            local = LowRankDelta(
                shard,
                rank=2,
                seed=71,
                scale=2.0,
                partition=partition,
                tp_rank=rank,
                tp_size=2,
                group=dist.group.WORLD,
            )
            optimizers = [
                torch.optim.AdamW(module.parameters(), lr=0.01, weight_decay=0.001)
                for module in (reference, local)
            ]
            # Three steps exercise initially-zero B, then nonzero shared A/B,
            # independent Adam moments, and a genuinely active clipping bound.
            for _ in range(3):
                for optimizer in optimizers:
                    optimizer.zero_grad()
                full_output = F.linear(inputs, reference(base))
                full_loss = (full_output - target).square().sum()
                full_loss.backward()
                if partition == "column":
                    output = F.linear(inputs, local(shard))
                    loss = (output - target.chunk(2, -1)[rank]).square().sum()
                else:
                    output = F.linear(inputs.chunk(2, -1)[rank], local(shard))
                    total = output.detach().clone()
                    dist.all_reduce(total)
                    output = total + (output - output.detach())
                    loss = (output - target).square().sum()
                loss.backward()
                params = tuple(reference.parameters())
                local_params = tuple(local.parameters())
                expected_norm = torch.stack([p.grad.square().sum() for p in params]).sum().sqrt()
                measured_norm = global_gradient_norm(
                    [p.grad for p in local_params],
                    (partition == "column", partition == "row"),
                    dist.group.WORLD,
                    2,
                )
                torch.testing.assert_close(measured_norm, expected_norm, atol=2e-5, rtol=2e-5)
                for group, norm in ((params, expected_norm), (local_params, measured_norm)):
                    for parameter in group:
                        parameter.grad.mul_(min(1.0, 0.1 / (float(norm) + 1e-12)))
                for optimizer in optimizers:
                    optimizer.step()
                for index, (full, part) in enumerate(zip(params, local_params, strict=True)):
                    sharded = (partition == "column" and index == 1) or (
                        partition == "row" and index == 0
                    )
                    axis = 0 if index == 1 else 1
                    expected = full.chunk(2, axis)[rank] if sharded else full
                    torch.testing.assert_close(part, expected, atol=2e-6, rtol=2e-5)
                    for name in ("exp_avg", "exp_avg_sq"):
                        expected = optimizers[0].state[full][name]
                        if sharded:
                            expected = expected.chunk(2, axis)[rank]
                        torch.testing.assert_close(
                            optimizers[1].state[part][name], expected, atol=2e-6, rtol=2e-5
                        )
    finally:
        dist.destroy_process_group()


def test_native_lora_tp2_three_updates_and_global_clip_match_tp1(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_native_lora_tp_worker, args=(str(tmp_path / "lora-gloo"),), nprocs=2, join=True)


def test_compact_verifier_teacher_remains_strided_and_bonus_is_not_a_draft():
    from lightcone_spec.native_tp import strided_teacher_rows

    # The compact input has [3,8] rows, but the verifier has scattered its
    # output into two fixed 8-row slots before returning to the learner.
    logits = torch.arange(16).view(16, 1)
    teacher = strided_teacher_rows(logits, 2, 7, 2)
    assert teacher[:, :, 0].tolist() == [list(range(7)), list(range(8, 15))]
    assert 7 not in teacher and 15 not in teacher
    with pytest.raises(ValueError, match="RID stride"):
        strided_teacher_rows(logits[:11], 2, 7, 2)


def test_cumulative_runtime_optimizer_uses_global_clip_norm(tmp_path):
    relative = "python/sglang/srt/speculative/online_adaptation_runtime.py"
    for patch in sorted(Path("patches/sglang").glob("*.diff")):
        subprocess.run(
            ["git", "apply", f"--include={relative}", str(patch.resolve())],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
    source = ast.parse((tmp_path / relative).read_text())
    wanted = {"_clip_fp32_gradients", "ParameterProposal", "ResidentOptimizer"}
    namespace = {"torch": torch, "math": math, "Sequence": Sequence, "dataclass": dataclass}
    exec(
        compile(
            ast.Module(
                [
                    item
                    for item in source.body
                    if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name in wanted
                ],
                [],
            ),
            "cumulative-runtime-optimizer",
            "exec",
        ),
        namespace,
    )
    config = SimpleNamespace(
        name="chronobelief",
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.99,
        epsilon=1e-8,
        weight_decay=0.0,
        grad_clip=1.0,
        schedule="constant",
    )
    optimizer = namespace["ResidentOptimizer"]((torch.zeros(1),), config)
    proposal = optimizer.propose(
        (torch.tensor([3.0]),),
        global_gradient_norm=torch.tensor(9.0),
        feedback_source_version=0,
        safe_boundary_version=0,
    )
    torch.testing.assert_close(proposal.gradient_norms, torch.tensor([9.0]))
    torch.testing.assert_close(proposal.first_moments[0], torch.tensor([1 / 30]))
