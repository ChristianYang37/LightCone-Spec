from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import lightcone_spec.methods as method_module
from lightcone_spec.adaptation.cohort import (
    CohortIdentity,
    CohortRuntime,
    LatestSignalBatch,
    SupervisionSignal,
)
from lightcone_spec.adaptation.kv_history import FrozenKVHistory, KVSegment
from lightcone_spec.methods import (
    CandidateUpdate,
    MethodPolicy,
    assert_candidate_equivalence,
    policy_for,
    publication_round,
)
from lightcone_spec.runtime.publication import CudaPublicationCoordinator


def test_all_methods_share_one_package_surface() -> None:
    assert method_module.__name__ == "lightcone_spec.methods"
    assert hasattr(method_module, "__path__")
    assert tuple(method_module.__all__) == (
        "CandidateTermination",
        "CandidateUpdate",
        "MethodPolicy",
        "OnlineSpecHedge",
        "OnlineSpecOGD",
        "OnlineSpecOptimistic",
        "OnlineSpecProposal",
        "PublicationDelay",
        "assert_candidate_equivalence",
        "ogd_update",
        "policy_for",
        "project_l2_ball",
        "publication_delay",
        "publication_round",
    )
    core_members = (
        "CandidateTermination",
        "CandidateUpdate",
        "MethodPolicy",
        "PublicationDelay",
        "assert_candidate_equivalence",
        "policy_for",
        "publication_delay",
        "publication_round",
    )
    onlinespec_members = (
        "OnlineSpecHedge",
        "OnlineSpecOGD",
        "OnlineSpecOptimistic",
        "OnlineSpecProposal",
        "ogd_update",
        "project_l2_ball",
    )
    assert all(
        getattr(method_module, name).__module__ == "lightcone_spec.methods.core"
        for name in core_members
    )
    assert all(
        getattr(method_module, name).__module__ == "lightcone_spec.methods.onlinespec"
        for name in onlinespec_members
    )


def candidate(**updates) -> CandidateUpdate:
    value = {
        "payload": ("tensor",),
        "source_round": 10,
        "source_version": 3,
        "cohort_epoch": 2,
        "slot_generation": 4,
        "ready_round": 12,
    }
    value.update(updates)
    return CandidateUpdate(**value)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (MethodPolicy.STATIC, None),
        (MethodPolicy.FIRST_READY_BOUNDARY, 12),
        (MethodPolicy.FIXED_BARRIER, 20),
    ],
)
def test_publication_policy(policy: MethodPolicy, expected: int | None) -> None:
    assert publication_round(policy, candidate(), 10) == expected


def test_tts_waits_only_when_ready_precedes_next_boundary() -> None:
    assert (
        publication_round(MethodPolicy.FIXED_BARRIER, candidate(ready_round=27), 10)
        == 27
    )


def test_extra_logical_delay_is_explicit_in_boundary_math() -> None:
    value = candidate()
    assert (
        publication_round(
            MethodPolicy.FIRST_READY_BOUNDARY,
            value,
            10,
            extra_logical_delay=3,
        )
        == 15
    )
    assert (
        publication_round(
            MethodPolicy.FIXED_BARRIER,
            value,
            10,
            extra_logical_delay=12,
        )
        == 24
    )


@pytest.mark.parametrize("stride", [0, -1])
def test_invalid_stride_rejected(stride: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        publication_round(MethodPolicy.FIXED_BARRIER, candidate(), stride)


def test_candidate_equivalence_is_exact() -> None:
    assert_candidate_equivalence(candidate(), candidate())
    with pytest.raises(ValueError, match="identical"):
        assert_candidate_equivalence(candidate(), candidate(ready_round=13))


@pytest.mark.parametrize("method", ["static", "tts", "l0"])
def test_policy_lookup(method: str) -> None:
    assert policy_for(method).value == method


def identity(group: str = "group-a") -> CohortIdentity:
    return CohortIdentity(
        target_revision="a" * 40,
        drafter_revision="b" * 40,
        algorithm="DFLASH",
        sampling_profile_sha256="c" * 64,
        adaptation_group_id=group,
        tenant_id="tenant-a",
        update_mode="lora",
        parameter_scope="drafter",
        parameter_layout_sha256="d" * 64,
        optimizer_identity="adamw:1e-5:0.01",
    )


def signal(
    cohort: str,
    request: str,
    sequence: int,
    *,
    version: int = 0,
    slot: int = 0,
) -> SupervisionSignal:
    return SupervisionSignal(
        cohort_sha256=cohort,
        request_id=request,
        sequence_number=sequence,
        source_version=version,
        slot_generation=slot,
        tensors=(sequence,),
        valid_positions=1,
    )


def test_cohort_identity_binds_tenant_and_group() -> None:
    assert identity("a").sha256 != identity("b").sha256
    changed = identity()
    changed_layout = CohortIdentity(
        **{**changed.__dict__, "parameter_layout_sha256": "e" * 64}
    )
    assert changed.sha256 != changed_layout.sha256


def test_latest_signal_batch_keeps_one_signal_per_request() -> None:
    cohort = identity().sha256
    batch = LatestSignalBatch(cohort)
    assert batch.offer(signal(cohort, "r0", 1))
    assert not batch.offer(signal(cohort, "r0", 1))
    assert batch.offer(signal(cohort, "r0", 2))
    assert batch.offer(signal(cohort, "r1", 1))
    rows, scale = batch.drain()
    assert [(row.request_id, row.sequence_number) for row in rows] == [
        ("r0", 2),
        ("r1", 1),
    ]
    assert scale == 0.5
    assert batch.drain() == ((), 0.0)


def test_cross_cohort_signal_rejected() -> None:
    batch = LatestSignalBatch(identity().sha256)
    with pytest.raises(ValueError, match="cross-cohort"):
        batch.offer(signal(identity("other").sha256, "r0", 1))


def test_runtime_rejects_stale_signal_and_candidate_conflicts() -> None:
    runtime = CohortRuntime(identity())
    assert not runtime.offer_signal(signal(runtime.identity.sha256, "r0", 0, version=1))
    update = runtime.begin_candidate(payload=(1,), source_round=10, ready_round=12)
    assert runtime.can_publish(
        update,
        policy=MethodPolicy.FIRST_READY_BOUNDARY,
        current_round=11,
        stride=10,
    ) == (False, "side_stream_not_ready")
    assert runtime.can_publish(
        update,
        policy=MethodPolicy.FIXED_BARRIER,
        current_round=15,
        stride=10,
    ) == (False, "waiting_fixed_boundary")
    assert runtime.can_publish(
        update,
        policy=MethodPolicy.FIRST_READY_BOUNDARY,
        current_round=12,
        stride=10,
        extra_logical_delay=1,
    ) == (False, "waiting_extra_logical_delay")
    assert runtime.can_publish(
        update,
        policy=MethodPolicy.FIRST_READY_BOUNDARY,
        current_round=13,
        stride=10,
        extra_logical_delay=1,
    ) == (True, "ready")
    assert (
        runtime.commit(
            update,
            policy=MethodPolicy.FIRST_READY_BOUNDARY,
            current_round=13,
            stride=10,
            extra_logical_delay=1,
        )
        == 1
    )


def test_commit_cannot_bypass_publication_authority() -> None:
    runtime = CohortRuntime(identity())
    update = runtime.begin_candidate(payload=(1,), source_round=10, ready_round=12)
    with pytest.raises(RuntimeError, match="side_stream_not_ready"):
        runtime.commit(
            update,
            policy=MethodPolicy.FIRST_READY_BOUNDARY,
            current_round=11,
            stride=10,
        )
    assert runtime.active_version == 0
    assert runtime.in_flight is update


def test_max_in_flight_cancel_and_aba_protection() -> None:
    runtime = CohortRuntime(identity())
    update = runtime.begin_candidate(payload=(1,), source_round=1, ready_round=2)
    with pytest.raises(RuntimeError, match="max_in_flight"):
        runtime.begin_candidate(payload=(2,), source_round=1, ready_round=2)
    runtime.cancel_request("r0")
    assert runtime.in_flight is None
    assert runtime.can_publish(
        update,
        policy=MethodPolicy.FIRST_READY_BOUNDARY,
        current_round=2,
        stride=1,
    ) == (False, "not_active_candidate")


def test_reset_and_disable_are_fail_closed() -> None:
    runtime = CohortRuntime(identity())
    runtime.disable("reconstruction_mismatch")
    assert not runtime.enabled
    with pytest.raises(RuntimeError, match="disabled"):
        runtime.begin_candidate(payload=(1,), source_round=1, ready_round=1)
    runtime.reset()
    assert runtime.enabled
    assert runtime.epoch == 1
    assert runtime.active_version == 0


def test_runtime_rejects_invalid_stride_before_boundary_math() -> None:
    runtime = CohortRuntime(identity())
    update = runtime.begin_candidate(payload=(1,), source_round=1, ready_round=1)
    with pytest.raises(ValueError, match="positive"):
        runtime.can_publish(
            update,
            policy=MethodPolicy.FIXED_BARRIER,
            current_round=1,
            stride=0,
        )


def test_frozen_kv_history_versions_only_future_tokens() -> None:
    history = FrozenKVHistory()
    history.append(8, 0)
    history.append(4, 1)
    history.append(2, 1)
    assert history.segments == [KVSegment(0, 8, 0), KVSegment(8, 14, 1)]
    assert history.version_at(7) == 0
    assert history.version_at(8) == 1
    history.retract(10)
    assert history.segments[-1] == KVSegment(8, 10, 1)
    with pytest.raises(IndexError):
        history.version_at(10)


@pytest.mark.parametrize("count", [0, -1])
def test_empty_kv_append_rejected(count: int) -> None:
    with pytest.raises(ValueError):
        FrozenKVHistory().append(count, 0)


def test_cuda_side_stream_uses_portable_lowest_priority_request() -> None:
    with (
        patch("lightcone_spec.runtime.publication.torch.cuda.Stream") as stream,
        patch("lightcone_spec.runtime.publication.torch.cuda.Event"),
    ):
        CudaPublicationCoordinator("cuda")
    stream.assert_called_once_with(
        device=torch.device("cuda"),
        priority=2**31 - 1,
    )
