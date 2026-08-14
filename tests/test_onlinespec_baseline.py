from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from lightcone_spec.config.schema import RunConfig
from lightcone_spec.methods import (
    OnlineSpecHedge,
    OnlineSpecOGD,
    OnlineSpecOptimistic,
    OnlineSpecProposal,
    ogd_update,
    project_l2_ball,
)
from lightcone_spec.sglang_bridge.config import sglang_adaptation_payload


def baseline_config(method: str = "onlinespec_ogd") -> dict:
    return {
        "schema_version": 3,
        "method": method,
        "model": {
            "key": "qwen3_8b_dflash16",
            "target": "Qwen/Qwen3-8B",
            "drafter": "z-lab/Qwen3-8B-DFlash-b16",
            "target_revision": "a" * 40,
            "drafter_revision": "b" * 40,
            "algorithm": "DFLASH",
            "max_context_length": 40960,
            "draft_depth": 15,
        },
        "runtime": {
            "sampling_profile_sha256": "c" * 64,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "speculative_num_draft_tokens": 16,
            "speculative_eagle_topk": None,
            "use_rejection_sampling": True,
            "max_running_requests": 4,
            "telemetry_detail": "headline",
        },
        "adaptation": {
            "weight_update_mode": "full",
            "parameter_scope": "all",
            "kv_history_policy": "frozen",
            "adaptation_scope": "cohort",
            "adaptation_group_id": "onlinespec-baseline",
            "optimizer": {
                "name": "sgd",
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "grad_clip": 1.0,
            },
            "rank": None,
            "stride": 10,
            "max_in_flight": 1,
            "canvas_tokens": 16,
            "loss_position_decay": 1.0,
        },
        "online_spec": {
            "projection_radius": None,
            "additional_learning_rates": [],
            "hedge_learning_rate": None,
        },
        "tenant_id": "research",
    }


def test_projected_ogd_is_transactional() -> None:
    learner = OnlineSpecOGD(
        (torch.tensor([0.0, 0.0]),),
        learning_rate=1.0,
        projection_radius=1.0,
    )
    proposal = learner.propose((torch.tensor([3.0, 4.0]),))
    torch.testing.assert_close(proposal.parameters[0], torch.tensor([-0.6, -0.8]))
    torch.testing.assert_close(learner.parameters[0], torch.zeros(2))
    learner.commit(proposal)
    torch.testing.assert_close(learner.parameters[0], torch.tensor([-0.6, -0.8]))


def test_online_learners_apply_per_learner_global_norm_clipping() -> None:
    learner = OnlineSpecOGD(
        (torch.zeros(2),),
        learning_rate=1.0,
        grad_clip=1.0,
    )
    proposal = learner.propose((torch.tensor([3.0, 4.0]),))
    torch.testing.assert_close(proposal.parameters[0], torch.tensor([-0.6, -0.8]))

    with pytest.raises(ValueError, match="grad_clip"):
        OnlineSpecOptimistic(
            (torch.zeros(1),),
            learning_rate=0.1,
            grad_clip=0.0,
        )


def test_optimistic_update_keeps_the_paper_two_state_projection() -> None:
    learner = OnlineSpecOptimistic(
        (torch.tensor([0.0]),),
        learning_rate=1.0,
        projection_radius=1.0,
    )
    first = learner.propose((torch.tensor([2.0]),))
    learner.commit(first)
    torch.testing.assert_close(learner.anchor[0], torch.tensor([-1.0]))
    torch.testing.assert_close(learner.parameters[0], torch.tensor([-1.0]))

    second = learner.propose((torch.tensor([-0.5]),))
    # The exact two-state projected update is zero. Applying the unconstrained
    # 2g_t-g_(t-1) shortcut to the published decision would incorrectly give 1.
    torch.testing.assert_close(second.parameters[0], torch.tensor([0.0]))
    torch.testing.assert_close(learner.parameters[0], torch.tensor([-1.0]))


def test_hedge_updates_independent_experts_before_combining() -> None:
    learner = OnlineSpecHedge(
        (torch.tensor([0.0]),),
        learning_rates=(0.1, 0.2),
        hedge_learning_rate=1.0,
    )
    proposal = learner.propose(
        torch.tensor([0.0, 2.0]),
        ((torch.tensor([1.0]),), (torch.tensor([-1.0]),)),
    )
    probability = torch.softmax(torch.tensor([0.0, -2.0]), dim=0)
    expected = probability[0] * -0.1 + probability[1] * 0.2
    torch.testing.assert_close(proposal.parameters[0], expected.reshape(1))
    torch.testing.assert_close(learner.experts[0][0], torch.tensor([0.0]))
    learner.commit(proposal)
    torch.testing.assert_close(learner.experts[0][0], torch.tensor([-0.1]))
    torch.testing.assert_close(learner.experts[1][0], torch.tensor([0.2]))
    torch.testing.assert_close(learner.cumulative_losses, torch.tensor([0.0, 2.0]))


def test_projection_rejects_invalid_domain() -> None:
    with pytest.raises(ValueError, match="positive"):
        project_l2_ball((torch.ones(1),), (torch.zeros(1),), 0.0)
    with pytest.raises(ValueError, match="fixed origin"):
        ogd_update(
            (torch.zeros(1),),
            (torch.ones(1),),
            0.1,
            projection_radius=1.0,
        )


def test_online_learner_commit_rejects_forged_nonfinite_state() -> None:
    learner = OnlineSpecOGD((torch.zeros(1),), learning_rate=0.1)
    forged = OnlineSpecProposal((torch.tensor([float("nan")]),), (), 1)
    with pytest.raises(ValueError, match="finite"):
        learner.commit(forged)
    torch.testing.assert_close(learner.parameters[0], torch.zeros(1))
    assert learner.step == 0


def test_stale_online_proposal_cannot_overwrite_a_committed_step() -> None:
    learner = OnlineSpecOptimistic((torch.zeros(1),), learning_rate=0.1)
    first = learner.propose((torch.tensor([1.0]),))
    stale = learner.propose((torch.tensor([-1.0]),))
    learner.commit(first)
    committed = learner.parameters[0].clone()

    with pytest.raises(ValueError, match="does not extend"):
        learner.commit(stale)

    torch.testing.assert_close(learner.parameters[0], committed)
    assert learner.step == 1


def test_hedge_uses_cumulative_not_last_round_loss() -> None:
    learner = OnlineSpecHedge(
        (torch.zeros(1),),
        learning_rates=(0.1, 0.2),
        hedge_learning_rate=1.0,
    )
    first = learner.propose(
        torch.tensor([0.0, 2.0]),
        ((torch.zeros(1),), (torch.zeros(1),)),
    )
    learner.commit(first)
    second = learner.propose(
        torch.tensor([2.0, 0.0]),
        ((torch.zeros(1),), (torch.zeros(1),)),
    )
    learner.commit(second)

    torch.testing.assert_close(learner.cumulative_losses, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(learner.probabilities, torch.tensor([0.5, 0.5]))


@pytest.mark.parametrize(
    ("method", "extra"),
    [
        ("onlinespec_ogd", {}),
        ("onlinespec_opt", {}),
        (
            "onlinespec_ens",
            {
                "additional_learning_rates": [0.03, 0.1],
                "hedge_learning_rate": 0.5,
            },
        ),
    ],
)
def test_online_baselines_are_routed_to_sglang(method: str, extra: dict) -> None:
    value = baseline_config(method)
    value["online_spec"].update(extra)
    config = RunConfig.model_validate(value)
    payload = sglang_adaptation_payload(config)
    assert payload is not None
    assert payload["method"] == method
    assert payload["optimizer"]["name"] == "sgd"
    assert payload["online_spec"] == value["online_spec"]


def test_hedge_supports_lora_decisions_and_rejects_incomplete_grid() -> None:
    value = baseline_config("onlinespec_ens")
    value["adaptation"].update(weight_update_mode="lora", rank=8, lora_alpha=8)
    value["online_spec"].update(
        additional_learning_rates=[0.03], hedge_learning_rate=0.5
    )
    config = RunConfig.model_validate(value)
    assert config.adaptation is not None
    assert config.adaptation.weight_update_mode == "lora"
    assert config.adaptation.rank == 8

    value = baseline_config("onlinespec_ens")
    with pytest.raises(ValidationError, match="multi-rate"):
        RunConfig.model_validate(value)


def test_core_method_rejects_online_baseline_state() -> None:
    value = baseline_config("tts")
    value["adaptation"]["optimizer"].update(name="adamw", learning_rate=1e-5)
    with pytest.raises(ValidationError, match="only valid for OnlineSPEC"):
        RunConfig.model_validate(value)
