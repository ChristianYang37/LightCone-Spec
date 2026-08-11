from __future__ import annotations

import pytest
import torch

from lightcone_spec.runtime.backend import (
    BackendPayload,
    BackendRegistry,
    DFlashBackendContract,
    DSparkBackendContract,
    EagleBackendContract,
    NextNBackendContract,
    ProposalEvidence,
    Reconstruction,
    dspark_composite_loss,
    dspark_conditional_survival_target,
)


def evidence() -> ProposalEvidence:
    logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    proposal = logits.softmax(dim=-1)
    return ProposalEvidence(
        backend="DSPARK",
        adapter_free_logits=logits.clone(),
        proposal_logits=logits,
        corrected_distribution=proposal,
        valid_mask=torch.ones((1, 2), dtype=torch.bool),
        teacher_rows=torch.tensor([[[0.75, 0.25], [0.25, 0.75]]]),
        predecessor_token_ids=torch.tensor([[7, 8]]),
        predecessor_embeddings=torch.randn(1, 2, 4),
        confidence=torch.zeros(1, 2),
        request_ids=("request-a",),
        cohort_sha256="a" * 64,
        source_adapter_version=3,
        payload=BackendPayload(
            schema="dspark-native-v1",
            values={
                "markov_w1_feature": torch.randn(1, 2, 4),
                "markov_w2_feature": torch.randn(1, 2, 4),
                "markov_w1_source": "inference_native",
                "markov_w2_source": "inference_native",
                "predecessor_source": "sampled_token",
                "scheduler_mode": "native_scheduler",
                "proposal_correction": "frozen_at_sampling",
            },
        ),
    )


def reconstruct(
    value: ProposalEvidence,
    delta: dict[str, torch.Tensor],
    already_applied: bool,
) -> Reconstruction:
    offset = delta.get("offset", torch.zeros_like(value.proposal_logits))
    logits = (
        value.proposal_logits if already_applied else value.proposal_logits + offset
    )
    return Reconstruction(
        proposal_logits=logits,
        corrected_distribution=logits.softmax(dim=-1),
        confidence=value.confidence,
    )


def test_dspark_contract_binds_native_features_and_reconstructs_once() -> None:
    value = evidence()
    registry = BackendRegistry((DSparkBackendContract(reconstruct),))
    result = registry.reconstruct(
        value,
        adapter_delta={"offset": torch.full_like(value.proposal_logits, 0.25)},
    )
    assert result.proposal_logits.shape == value.proposal_logits.shape
    assert bool(value.numerical_predicate())
    assert len(value.identity_sha256) == 64
    with pytest.raises(ValueError, match="double-count"):
        registry.reconstruct(
            value,
            adapter_delta={"offset": torch.ones_like(value.proposal_logits)},
            adapter_already_applied=True,
        )


def test_dspark_contract_rejects_placeholder_provenance() -> None:
    value = evidence()
    bad = ProposalEvidence(
        **{
            **value.__dict__,
            "payload": BackendPayload(
                schema="dspark-native-v1",
                values={
                    **value.payload.values,
                    "markov_w1_source": "placeholder",
                },
            ),
        }
    )
    with pytest.raises(ValueError, match="real inference"):
        DSparkBackendContract(reconstruct).validate_payload(bad)


def test_dspark_confidence_target_is_detached_and_composite_loss_is_finite() -> None:
    teacher = torch.tensor([[[0.8, 0.2], [0.3, 0.7]]])
    logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    proposal = logits.softmax(dim=-1)
    target = dspark_conditional_survival_target(teacher, proposal)
    assert not target.requires_grad
    loss = dspark_composite_loss(
        teacher_distribution=teacher,
        proposal_distribution=proposal,
        confidence_logits=torch.zeros((1, 2), requires_grad=True),
        valid_mask=torch.ones((1, 2), dtype=torch.bool),
        confidence_weight=0.25,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


@pytest.mark.parametrize(
    ("backend", "schema", "payload", "contract"),
    [
        (
            "DFLASH",
            "dflash-native-v1",
            {
                "canvas_state": torch.zeros(1),
                "proposal_correction": "frozen_at_sampling",
            },
            lambda: DFlashBackendContract(reconstruct),
        ),
        (
            "EAGLE",
            "eagle-native-v1",
            {
                "tree_state": torch.zeros(1),
                "topk": 1,
                "proposal_correction": "frozen_at_sampling",
            },
            lambda: EagleBackendContract("EAGLE", reconstruct),
        ),
        (
            "EAGLE3",
            "eagle3-native-v1",
            {
                "tree_state": torch.zeros(1),
                "topk": 1,
                "proposal_correction": "frozen_at_sampling",
            },
            lambda: EagleBackendContract("EAGLE3", reconstruct),
        ),
        (
            "NEXTN",
            "nextn-native-v1",
            {
                "mtp_hidden_state": torch.zeros(1, 2, 4),
                "interface_sha256": "f" * 64,
                "proposal_correction": "frozen_at_sampling",
            },
            lambda: NextNBackendContract(reconstruct),
        ),
    ],
)
def test_registered_native_backends_use_one_common_evidence_envelope(
    backend: str,
    schema: str,
    payload: dict,
    contract,
) -> None:
    source = evidence()
    value = ProposalEvidence(
        **{
            **source.__dict__,
            "backend": backend,
            "confidence": None,
            "payload": BackendPayload(schema=schema, values=payload),
        }
    )
    registry = BackendRegistry((contract(),))
    result = registry.reconstruct(value, adapter_delta={})
    assert result.proposal_logits.shape == source.proposal_logits.shape
