"""CPU contracts for the fixed-address mixed-precision tail bank."""

from __future__ import annotations

import math

import pytest
import torch

from lightcone_spec.adapters.adapter_params import (
    AdapterShapes,
    initial_parameter_vector,
)
from lightcone_spec.exit_codes import ConfigError
from lightcone_spec.orchestration.runtime_config import _resolve_forward_dtype
from lightcone_spec.sglang_bridge.bank import (
    AdapterBank,
    estimate_adaptation_memory,
    estimate_dflash_supervision_fanout_bytes,
)


def test_lora_initialization_is_qdq_canonical_in_both_banks():
    shapes = AdapterShapes(
        rank=3,
        markov_dim=0,
        vocab_size=32,
        weight_update_mode="tail_lora",
        hidden_size=8,
        draft_depth=2,
        has_markov=False,
        has_confidence=False,
        algorithm="EAGLE3",
    )
    phi0 = initial_parameter_vector(shapes)
    bank = AdapterBank(
        num_slots=1,
        num_params=shapes.num_params(),
        forward_dtype=torch.bfloat16,
        with_optimizer=False,
    )
    slot = bank.allocate("request", "tenant")
    bank.initialize_slot(slot.slot_index, phi0)

    expected_forward = phi0.to(torch.bfloat16)
    expected_master = expected_forward.to(torch.float32)
    assert torch.equal(bank.read_forward_active(slot.slot_index), expected_forward)
    assert torch.equal(bank.read_active(slot.slot_index), expected_master)
    assert torch.equal(bank.staging[slot.slot_index], expected_master)
    # LoRA's input factor survives Q-DQ while its output factor remains an
    # exact zero, preserving the no-op forward and non-zero first gradient.
    slices = shapes.parameter_slices()
    assert torch.count_nonzero(expected_master[slices["a_h"]]) > 0
    assert torch.count_nonzero(expected_master[slices["b_h"]]) == 0


def test_publish_keeps_addresses_and_side_stream_scratch_independent():
    bank = AdapterBank(
        num_slots=2,
        num_params=6,
        forward_dtype=torch.bfloat16,
        with_optimizer=False,
    )
    slot = bank.allocate("request", "tenant")
    scratch = bank.candidate_forward_buffer()
    scratch.fill_(7)
    pointers = {
        "active": bank.active.data_ptr(),
        "staging": bank.staging.data_ptr(),
        "forward_active": bank.forward_active.data_ptr(),
        "scratch": scratch.data_ptr(),
    }
    candidate = torch.tensor(
        [1.0001, 1.003, -0.333333, 0.0001234, 19.8751, -42.1251],
        dtype=torch.float32,
    )

    bank.write_staging(slot.slot_index, slot.request_epoch, candidate)
    assert bank.publish(slot.slot_index, slot.request_epoch) == 1
    expected_forward = candidate.to(torch.bfloat16)
    expected_master = expected_forward.to(torch.float32)

    assert torch.equal(bank.read_forward_active(slot.slot_index), expected_forward)
    assert torch.equal(bank.read_active(slot.slot_index), expected_master)
    assert torch.equal(bank.staging[slot.slot_index], expected_master)
    assert torch.equal(scratch, torch.full_like(scratch, 7))
    assert pointers == {
        "active": bank.active.data_ptr(),
        "staging": bank.staging.data_ptr(),
        "forward_active": bank.forward_active.data_ptr(),
        "scratch": bank.candidate_forward_buffer().data_ptr(),
    }


def test_slot_reuse_clears_both_active_representations():
    bank = AdapterBank(
        num_slots=1,
        num_params=4,
        forward_dtype=torch.bfloat16,
        with_optimizer=False,
    )
    slot = bank.allocate("first", "tenant")
    bank.write_staging(slot.slot_index, slot.request_epoch, torch.ones(4))
    bank.publish(slot.slot_index, slot.request_epoch)
    bank.free(slot.slot_index)

    reused = bank.allocate("second", "tenant")
    assert torch.count_nonzero(bank.read_active(reused.slot_index)) == 0
    assert torch.count_nonzero(bank.read_forward_active(reused.slot_index)) == 0


def test_master_dtype_is_fail_closed():
    with pytest.raises(ValueError, match="canonical master dtype"):
        AdapterBank(
            num_slots=1,
            num_params=4,
            dtype=torch.bfloat16,
        )


def test_memory_ledger_charges_forward_bank_scratch_and_actual_row_dtype():
    args = dict(
        num_slots=3,
        max_in_flight=1,
        num_params=10,
        vocab_size=32,
        rank=2,
        markov_dim=0,
        hidden_size=8,
        draft_depth=2,
        adapter_row_capacity=7,
        with_optimizer=False,
        with_fisher=False,
        with_optimizer_preview=False,
        retain_source_signal=False,
        trace_capture=False,
        safety_factor=1.0,
        weight_update_mode="full_rank_tail",
    )
    bf16 = estimate_adaptation_memory(**args, forward_dtype_bytes=2)
    fp32 = estimate_adaptation_memory(**args, forward_dtype_bytes=4)

    assert bf16.forward_active_bytes == 3 * 10 * 2
    assert bf16.forward_candidate_scratch_bytes == 10 * 2
    assert bf16.graph_row_buffer_bytes == 7 * 10 * 2 + 7 * (8 + 2)
    assert fp32.fixed_bytes - bf16.fixed_bytes == (
        (3 + 1 + 7) * 10 * 2 + 7 * 2
    )
    assert sum(bf16.category_bytes().values()) == (
        bf16.fixed_bytes + bf16.reserve_bytes
    )


def test_dflash_qwen3_snapshot_fanout_is_charged_for_batch_1_4_8():
    # Qwen3-4B + DFlash-b16, BF16 head.  Every additional vectorized Tail-LoRA
    # row owns compact native target/corrected scores, four FP32 loss/probability
    # workspaces, one native STE gradient and hidden/u/mask features.  The
    # scheduler batch additionally retains raw and corrected BF16 proposal
    # outputs until acceptance completes.
    dims = dict(
        vocab_size=151_936,
        hidden_size=2_560,
        draft_depth=15,
        forward_dtype_bytes=2,
        output_residual=False,
        stochastic=False,
        tensor_parallel_size=1,
    )
    compact_snapshot_and_working = 50_219_535
    retained_proposal = 9_116_160
    expected = {
        1: 0,
        4: 3 * (compact_snapshot_and_working + retained_proposal),
        8: 7 * (compact_snapshot_and_working + retained_proposal),
    }
    for batch_size, expected_fanout in expected.items():
        fanout = estimate_dflash_supervision_fanout_bytes(
            batch_capacity=batch_size,
            active_capacity=batch_size,
            **dims,
        )
        assert fanout == expected_fanout

        ledger = estimate_adaptation_memory(
            num_slots=batch_size,
            max_in_flight=1,
            num_params=81_920,
            vocab_size=dims["vocab_size"],
            rank=16,
            markov_dim=0,
            hidden_size=dims["hidden_size"],
            draft_depth=dims["draft_depth"],
            adapter_row_capacity=batch_size,
            with_optimizer=True,
            with_fisher=False,
            with_optimizer_preview=False,
            retain_source_signal=False,
            trace_capture=False,
            safety_factor=1.25,
            weight_update_mode="tail_lora",
            forward_dtype_bytes=2,
            supervision_fanout_bytes=fanout,
        )
        assert ledger.supervision_fanout_bytes == expected_fanout
        assert ledger.transient_bytes == 75_057_120 + expected_fanout
        assert ledger.reserve_bytes == math.ceil(ledger.transient_bytes * 1.25)

    # The reserve includes the actual simultaneous batch backward rather than
    # pretending that request rows execute serially.
    assert math.ceil(
        (75_057_120 + expected[8]) * 1.25 / (1 << 20)
    ) == 585

    traced = estimate_dflash_supervision_fanout_bytes(
        batch_capacity=4,
        active_capacity=4,
        trace_capture=True,
        **dims,
    )
    assert traced > expected[4]


def test_preflight_dtype_resolution_is_explicit_and_fail_closed():
    assert _resolve_forward_dtype({}, None) == ("bfloat16", 2)
    assert _resolve_forward_dtype(
        {"dtype": "auto"}, {"text_config": {"torch_dtype": "float32"}}
    ) == ("float32", 4)
    assert _resolve_forward_dtype(
        {"dtype": "bf16"}, {"torch_dtype": "float32"}
    ) == ("bfloat16", 2)
    with pytest.raises(ConfigError, match="unsupported proposal forward dtype"):
        _resolve_forward_dtype({"dtype": "float16"}, None)
