"""Version state machine, double-buffer protocol and adapter-bank ABA
fail-closed behaviour (spec 3.3, 4.2-4.4, 8.3)."""

from __future__ import annotations

import math

import pytest
import torch

from lightcone_spec.exit_codes import ExactnessViolation
from lightcone_spec.runtime.double_buffer import (
    DoubleBufferStore,
    PendingUpdate,
    ReadyEvent,
)
from lightcone_spec.runtime.events import UpdateEventChain
from lightcone_spec.runtime.versions import RequestVersionState
from lightcone_spec.sglang_bridge.bank import (
    AdaptationCapacityError,
    AdapterBank,
    estimate_adaptation_memory,
)


def _vs() -> RequestVersionState:
    return RequestVersionState(
        request_id="r0", tenant_id_hash="t0", stream_id=None, request_epoch=1
    )


def test_canvas_consistency_checks():
    vs = _vs()
    vs.check_canvas_consistency(3, 3, 3)
    with pytest.raises(ExactnessViolation):
        vs.check_canvas_consistency(3, 2, 3)
    vs = _vs()
    with pytest.raises(ExactnessViolation):
        vs.check_canvas_consistency(3, 3, 4)


def test_overwrite_and_replay_checks():
    vs = _vs()
    vs.check_proposal_not_overwritten(True)
    with pytest.raises(ExactnessViolation):
        vs.check_proposal_not_overwritten(False)
    vs = _vs()
    with pytest.raises(ExactnessViolation):
        vs.check_active_not_written_during_replay(True)


def test_direct_add_requires_matching_source():
    vs = _vs()
    vs.active_version = 5
    vs.check_source_matches_active_for_direct_add(5)
    with pytest.raises(ExactnessViolation):
        vs.check_source_matches_active_for_direct_add(4)


def test_slot_ownership_aba():
    vs = _vs()
    vs.check_slot_ownership("t0", 1)
    with pytest.raises(ExactnessViolation):
        vs.check_slot_ownership("t1", 1)
    vs = _vs()
    with pytest.raises(ExactnessViolation):
        vs.check_slot_ownership("t0", 2)


def _pending(uid: str, version: int = 0) -> PendingUpdate:
    chain = UpdateEventChain(update_id=uid, source_round=0, source_version=version)
    chain.mark("snapshot")
    return PendingUpdate(
        update_id=uid,
        source_round=0,
        source_version=version,
        candidate_delta=torch.zeros(4),
        raw_gradient=torch.zeros(4),
        events=chain,
        ready=ReadyEvent(event_id=f"{uid}-done"),
    )


def test_double_buffer_publish_protocol():
    store = DoubleBufferStore(num_params=4, max_in_flight=1)
    u = _pending("u1")
    store.launch(u)
    with pytest.raises(ExactnessViolation):
        store.launch(_pending("u2"))  # exceeds max_in_flight=1
    params = torch.ones(4)
    store.write_staging(u, params)
    assert store.poll_ready() == [u]
    v = store.publish(u, params)
    assert v == 1
    assert torch.equal(store.read_active(), params)


def test_publish_before_done_event_fails():
    store = DoubleBufferStore(num_params=4)
    u = _pending("u1")
    store.launch(u)
    with pytest.raises(ExactnessViolation):
        store.publish(u, torch.ones(4))


def test_no_writes_during_graph_replay():
    store = DoubleBufferStore(num_params=4)
    u = _pending("u1")
    store.launch(u)
    store.begin_replay()
    with pytest.raises(ExactnessViolation):
        store.write_staging(u, torch.ones(4))
    store.end_replay()
    store.write_staging(u, torch.ones(4))


def test_max_in_flight_two_requires_snapshot():
    store = DoubleBufferStore(num_params=4, max_in_flight=2)
    with pytest.raises(ExactnessViolation):
        store.launch(_pending("u1"))  # missing source snapshot


def test_adapter_bank_epoch_aba():
    bank = AdapterBank(num_slots=2, num_params=4)
    slot = bank.allocate("req-a", "t0")
    old_epoch = slot.request_epoch  # capture before slot reuse mutates it
    slot_index = slot.slot_index
    bank.check_owner(slot_index, old_epoch, "t0")
    bank.free(slot_index)
    slot2 = bank.allocate("req-b", "t0")
    assert slot2.slot_index == slot_index  # reused
    assert slot2.request_epoch == old_epoch + 1
    with pytest.raises(ExactnessViolation):
        # stale epoch from the freed request (ABA)
        bank.check_owner(slot_index, old_epoch, "t0")
    with pytest.raises(ExactnessViolation):
        bank.write_staging(slot_index, old_epoch, torch.ones(4))


def test_adapter_bank_cross_tenant():
    bank = AdapterBank(num_slots=1, num_params=4)
    slot = bank.allocate("req-a", "tenant-a")
    with pytest.raises(ExactnessViolation):
        bank.check_owner(slot.slot_index, slot.request_epoch, "tenant-b")


def test_adapter_bank_fixed_addresses():
    bank = AdapterBank(
        num_slots=2,
        num_params=4,
        max_in_flight=2,
        with_optimizer=True,
        with_fisher=True,
        with_optimizer_preview=True,
    )
    ptr_before = {
        name: getattr(bank, name).data_ptr()
        for name in (
            "active",
            "staging",
            "exp_avg",
            "exp_avg_sq",
            "preview_exp_avg",
            "preview_exp_avg_sq",
            "phi_source",
            "candidate_grad",
            "candidate_delta",
            "fisher",
            "candidate_fisher",
            "candidate_health_host",
        )
    }
    slot = bank.allocate("req-a", "t0")
    assert not slot.active_has_effect
    bank.write_staging(slot.slot_index, slot.request_epoch, torch.ones(4))
    bank.publish(slot.slot_index, slot.request_epoch)
    assert slot.active_has_effect
    assert {
        name: getattr(bank, name).data_ptr() for name in ptr_before
    } == ptr_before, "publish must never reallocate any captured state"
    assert torch.equal(bank.read_active(slot.slot_index), torch.ones(4))


def test_candidate_health_lane_is_epoch_and_generation_bound():
    bank = AdapterBank(num_slots=1, num_params=4, max_in_flight=1)
    slot = bank.allocate("req-a", "t0")
    health, generation = bank.prepare_candidate_health(
        slot.slot_index, slot.request_epoch, 0
    )
    health.fill_(True)
    assert bank.read_candidate_health(
        slot.slot_index, slot.request_epoch, 0, generation
    )

    _next_health, next_generation = bank.prepare_candidate_health(
        slot.slot_index, slot.request_epoch, 0
    )
    assert next_generation == generation + 1
    with pytest.raises(ExactnessViolation, match="health lane mismatch"):
        bank.read_candidate_health(
            slot.slot_index, slot.request_epoch, 0, generation
        )

    old_epoch = slot.request_epoch
    bank.free(slot.slot_index)
    reused = bank.allocate("req-b", "t0")
    with pytest.raises(ExactnessViolation, match="health lane mismatch"):
        bank.read_candidate_health(
            reused.slot_index, old_epoch, 0, next_generation
        )


def test_adapter_bank_reuse_clears_effect_marker():
    bank = AdapterBank(num_slots=1, num_params=4)
    slot = bank.allocate("req-a", "t0")
    bank.write_staging(slot.slot_index, slot.request_epoch, torch.ones(4))
    bank.publish(slot.slot_index, slot.request_epoch)
    bank.free(slot.slot_index)

    reused = bank.allocate("req-b", "t0")
    assert reused.slot_index == slot.slot_index
    assert reused.active_version == 0
    assert not reused.active_has_effect


def test_adapter_capacity_is_fallback_not_exactness():
    bank = AdapterBank(num_slots=1, num_params=4)
    bank.allocate("req-a", "t0")
    with pytest.raises(AdaptationCapacityError):
        bank.allocate("req-b", "t0")


def test_adaptation_memory_ledger_is_monotone_and_calibration_is_floor():
    args = dict(
        num_slots=4,
        max_in_flight=1,
        num_params=4096,
        vocab_size=151936,
        rank=16,
        markov_dim=64,
        hidden_size=2560,
        draft_depth=7,
        adapter_row_capacity=48,
        with_optimizer=True,
        with_fisher=False,
        with_optimizer_preview=False,
        retain_source_signal=False,
        trace_capture=False,
        safety_factor=1.25,
    )
    base = estimate_adaptation_memory(**args)
    more_slots = estimate_adaptation_memory(**(args | {"num_slots": 8}))
    more_rows = estimate_adaptation_memory(
        **(args | {"adapter_row_capacity": args["adapter_row_capacity"] + 16})
    )
    calibrated = estimate_adaptation_memory(
        **(args | {"calibrated_reserve_mb": base.reserve_mb + 64})
    )
    assert more_slots.fixed_bytes > base.fixed_bytes
    assert more_slots.transient_bytes == base.transient_bytes
    assert more_rows.fixed_bytes > base.fixed_bytes
    # Fixed tensors are already resident before SGLang sizes KV and must not be
    # charged a second time through the explicit, future-allocation reserve.
    assert base.reserve_bytes == math.ceil(base.transient_bytes * 1.25)
    assert calibrated.reserve_mb >= base.reserve_mb + 64


def test_adaptation_memory_categories_are_mutually_exclusive_and_sum_exactly():
    ledger = estimate_adaptation_memory(
        num_slots=3,
        max_in_flight=1,
        num_params=2048,
        vocab_size=8192,
        rank=16,
        markov_dim=64,
        hidden_size=1024,
        draft_depth=7,
        adapter_row_capacity=16,
        with_optimizer=True,
        with_fisher=True,
        with_optimizer_preview=True,
        retain_source_signal=False,
        trace_capture=True,
        safety_factor=1.25,
    )

    categories = ledger.category_bytes()
    assert sum(categories.values()) == ledger.fixed_bytes + ledger.reserve_bytes
    assert categories["runtime_headroom"] == ledger.reserve_bytes
    assert "trace" not in categories
    assert "activation_reserve" not in categories

    breakdown = ledger.headroom_breakdown()
    assert breakdown["reserved_total"] == categories["runtime_headroom"]
    assert breakdown["transient_peak"] == ledger.transient_bytes
    assert breakdown["trace_within_transient"] == ledger.trace_bytes > 0
    assert (
        breakdown["supervision_fanout_within_transient"]
        == ledger.supervision_fanout_bytes
    )
    assert breakdown["safety_or_calibration_margin"] >= 0
