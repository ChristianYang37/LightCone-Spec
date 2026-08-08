from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

from lightcone_spec.adapters.adapter_params import AdapterShapes
from lightcone_spec.methods.base import SourceBoundCandidateBatch


def _load_tail_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "sglang"
        / "python"
        / "sglang"
        / "srt"
        / "speculative"
        / "tail_adaptation.py"
    )
    spec = importlib.util.spec_from_file_location("_lightcone_tail_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dflash_verify_batch_calls_real_batched_runtime_entry(monkeypatch):
    module = _load_tail_module()
    semantic = ModuleType("sglang.srt.speculative.dflash_info_v2")

    def semantic_mask(**kwargs):
        mask = kwargs["base_valid_mask"]
        return mask.clone(), torch.zeros(mask.shape[0], dtype=torch.bool)

    semantic.dflash_semantic_valid_mask = semantic_mask
    monkeypatch.setitem(
        sys.modules, "sglang.srt.speculative.dflash_info_v2", semantic
    )

    batch_size, depth, hidden, vocab = 3, 4, 7, 13
    request_ids = tuple(f"rid-{index}" for index in range(batch_size))

    class Runtime:
        def __init__(self):
            self.calls = []

        @staticmethod
        def wants_trace_signal(_rid):
            return False

        def launch_update_batch(self, rids, signal):
            self.calls.append((tuple(rids), signal))
            return [f"{rid}-update" for rid in rids]

        def launch_update(self, *_args, **_kwargs):
            raise AssertionError("per-request candidate entry must not be used")

    manager = object.__new__(module.LightConeTailAdaptationManager)
    manager._dflash_accept = SimpleNamespace(
        correct_len=torch.ones(batch_size, dtype=torch.int64),
        bonus=torch.zeros(batch_size, dtype=torch.bool),
    )
    manager._request_semantics = lambda _batch: (
        [100] * batch_size,
        [()] * batch_size,
        [False] * batch_size,
    )
    manager._algorithmic_censored = None
    manager._disabled = set()
    manager._capture_round = lambda _rid: True
    manager._globalize_dflash_logits = lambda logits, _ranges: logits
    manager._forward_dtype = torch.float32
    manager._r_h_dev = torch.randn(hidden, 128)
    manager._round_of = {rid: 1 for rid in request_ids}
    manager._pinned_versions = {rid: 0 for rid in request_ids}
    manager.config = SimpleNamespace(update_stride=1)
    manager.shapes = AdapterShapes(
        rank=2,
        markov_dim=0,
        vocab_size=vocab,
        weight_update_mode="tail_lora",
        hidden_size=hidden,
        draft_depth=depth,
        has_markov=False,
        has_confidence=False,
        algorithm="DFLASH",
    )
    manager.runtime = Runtime()
    manager._ExactnessViolation = RuntimeError
    manager._disable = lambda *_args, **_kwargs: None

    reqs = [SimpleNamespace(rid=rid) for rid in request_ids]
    batch = SimpleNamespace(
        reqs=reqs,
        sampling_info=SimpleNamespace(is_all_greedy=True),
    )
    proposal = SimpleNamespace(
        proposal_token_ids=torch.zeros(batch_size, depth, dtype=torch.int64),
        valid_mask=torch.ones(batch_size, depth, dtype=torch.bool),
        tail_hidden=torch.randn(batch_size, depth, hidden),
        raw_logits=torch.randn(batch_size, depth, vocab),
        proposal_logits=torch.randn(batch_size, depth, vocab),
        vocab_ranges=((0, vocab),),
    )
    target_logits = torch.randn(batch_size * depth, vocab)

    manager._launch_dflash_signals(batch, target_logits, proposal)

    assert len(manager.runtime.calls) == 1
    called_ids, compact = manager.runtime.calls[0]
    assert called_ids == request_ids
    assert isinstance(compact, SourceBoundCandidateBatch)
    assert compact.proposal_logits.shape == (batch_size, depth, vocab)
    assert compact.target_logits.shape == (batch_size, depth, vocab)
    assert compact.tail_hidden.shape == (batch_size, depth, hidden)
    # The compact online candidate does not retain another full-vocabulary raw
    # tensor; raw logits are reserved for explicitly enabled trace capture.
    assert not hasattr(compact, "base_proposal_logits")


def test_tp_consensus_preserves_request_local_health_for_batch():
    module = _load_tail_module()
    gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    actual, health = module.tp_gradient_consensus(
        gradient,
        torch.tensor([True, False]),
        tp_size=1,
    )

    assert torch.equal(health, torch.tensor([True, False]))
    assert torch.equal(actual[0], gradient[0])
    assert torch.count_nonzero(actual[1]) == 0
