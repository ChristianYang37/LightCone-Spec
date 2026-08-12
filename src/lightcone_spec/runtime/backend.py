"""Small backend contract for exact, versioned online adaptation.

The contract deliberately describes only values shared by speculative backends.
Backend-specific tensors remain in an opaque payload whose validator and
reconstructor are owned by that backend. Static and target-only execution do
not construct :class:`ProposalEvidence`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

BackendName = Literal["DFLASH", "DSPARK", "EAGLE", "EAGLE3", "NEXTN"]


def _hash_body(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class BackendPayload:
    """Backend-owned values plus an immutable schema identity."""

    schema: str
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.schema:
            raise ValueError("backend payload schema must be non-empty")
        if not self.values:
            raise ValueError("backend payload must not be empty")

    @property
    def sha256(self) -> str:
        identity: list[dict[str, object]] = []
        for name, value in sorted(self.values.items()):
            if isinstance(value, Tensor):
                identity.append(
                    {
                        "name": name,
                        "shape": tuple(value.shape),
                        "dtype": str(value.dtype),
                        "device": str(value.device),
                    }
                )
            else:
                identity.append({"name": name, "type": type(value).__qualname__})
        return _hash_body({"schema": self.schema, "identity": identity})


@dataclass(frozen=True)
class ProposalEvidence:
    """Source-bound evidence for one exact speculative proposal.

    Construction performs structural checks only. Device-side numerical
    predicates are returned as tensors so a headline path never needs
    ``Tensor.item()``, a DtoH copy, or an implicit synchronization.
    """

    backend: BackendName
    adapter_free_logits: Tensor
    proposal_logits: Tensor
    corrected_distribution: Tensor
    valid_mask: Tensor
    teacher_rows: Tensor
    predecessor_token_ids: Tensor
    predecessor_embeddings: Tensor
    confidence: Tensor | None
    request_ids: tuple[str, ...]
    cohort_sha256: str
    source_adapter_version: int
    payload: BackendPayload

    def __post_init__(self) -> None:
        _require_sha256("cohort_sha256", self.cohort_sha256)
        if self.source_adapter_version < 0:
            raise ValueError("source_adapter_version must be non-negative")
        if not self.request_ids or len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("proposal request IDs must be non-empty and unique")
        devices = {tensor.device for tensor in self.common_tensors}
        if len(devices) != 1:
            raise ValueError("proposal evidence tensors must share one device")
        if self.adapter_free_logits.shape != self.proposal_logits.shape:
            raise ValueError("adapter-free and deployed proposal logits must align")
        if self.corrected_distribution.shape != self.proposal_logits.shape:
            raise ValueError("corrected proposal distribution must align with logits")
        if self.teacher_rows.shape != self.proposal_logits.shape:
            raise ValueError("teacher rows must align with proposal logits")
        if self.valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be boolean")
        if self.valid_mask.shape != self.proposal_logits.shape[:-1]:
            raise ValueError("valid_mask must cover every non-vocabulary row")
        if self.predecessor_token_ids.shape != self.valid_mask.shape:
            raise ValueError("predecessor-token identity must cover proposal rows")
        if self.predecessor_embeddings.shape[:-1] != self.valid_mask.shape:
            raise ValueError("predecessor embeddings must cover proposal rows")
        if (
            self.confidence is not None
            and self.confidence.shape != self.valid_mask.shape
        ):
            raise ValueError("confidence must cover proposal rows")
        if self.proposal_logits.shape[0] != len(self.request_ids):
            raise ValueError("request identity count must match the proposal batch")

    @property
    def common_tensors(self) -> tuple[Tensor, ...]:
        values = (
            self.adapter_free_logits,
            self.proposal_logits,
            self.corrected_distribution,
            self.valid_mask,
            self.teacher_rows,
            self.predecessor_token_ids,
            self.predecessor_embeddings,
        )
        return values if self.confidence is None else (*values, self.confidence)

    def numerical_predicate(self) -> Tensor:
        """Return a scalar device predicate without reading it on the host."""
        floating = tuple(
            tensor for tensor in self.common_tensors if tensor.is_floating_point()
        )
        finite = torch.stack(
            tuple(torch.isfinite(tensor).all() for tensor in floating)
        ).all()
        nonnegative = (self.corrected_distribution >= 0).all()
        row_sums = self.corrected_distribution.sum(dim=-1)
        normalised = torch.isclose(
            row_sums,
            torch.ones_like(row_sums),
            rtol=1e-5,
            atol=1e-6,
        ).all()
        return finite & nonnegative & normalised

    @property
    def identity_sha256(self) -> str:
        return _hash_body(
            {
                "backend": self.backend,
                "requests": self.request_ids,
                "cohort": self.cohort_sha256,
                "source_adapter_version": self.source_adapter_version,
                "payload": self.payload.sha256,
                "shapes": [tuple(tensor.shape) for tensor in self.common_tensors],
                "dtypes": [str(tensor.dtype) for tensor in self.common_tensors],
            }
        )


@dataclass(frozen=True)
class Reconstruction:
    proposal_logits: Tensor
    corrected_distribution: Tensor
    confidence: Tensor | None

    def numerical_predicate(self) -> Tensor:
        tensors = (self.proposal_logits, self.corrected_distribution)
        if self.confidence is not None:
            tensors = (*tensors, self.confidence)
        return torch.stack(
            tuple(torch.isfinite(tensor).all() for tensor in tensors)
        ).all()


@runtime_checkable
class BackendContract(Protocol):
    """One registered backend validator and differentiable reconstructor."""

    name: BackendName

    def validate_payload(self, evidence: ProposalEvidence) -> None: ...

    def reconstruct(
        self,
        evidence: ProposalEvidence,
        *,
        adapter_delta: Mapping[str, Tensor],
        adapter_already_applied: bool,
    ) -> Reconstruction: ...


class BackendRegistry:
    """Process-local backend contracts with duplicate registration rejected."""

    def __init__(self, contracts: Sequence[BackendContract] = ()) -> None:
        self._contracts: dict[BackendName, BackendContract] = {}
        for contract in contracts:
            self.register(contract)

    def register(self, contract: BackendContract) -> None:
        if contract.name in self._contracts:
            raise ValueError(f"duplicate backend contract {contract.name}")
        self._contracts[contract.name] = contract

    def validate(self, evidence: ProposalEvidence) -> None:
        try:
            contract = self._contracts[evidence.backend]
        except KeyError as exc:
            raise ValueError(f"unregistered backend {evidence.backend}") from exc
        contract.validate_payload(evidence)

    def reconstruct(
        self,
        evidence: ProposalEvidence,
        *,
        adapter_delta: Mapping[str, Tensor],
        adapter_already_applied: bool = False,
    ) -> Reconstruction:
        if adapter_already_applied and adapter_delta:
            raise ValueError("refusing to double-count an already-applied adapter")
        self.validate(evidence)
        return self._contracts[evidence.backend].reconstruct(
            evidence,
            adapter_delta=adapter_delta,
            adapter_already_applied=adapter_already_applied,
        )


@dataclass(frozen=True)
class FunctionalBackendContract:
    """Thin registered adapter around backend-owned validation/reconstruction."""

    name: BackendName
    payload_schema: str
    required_payload_fields: frozenset[str]
    reconstruct_fn: Callable[
        [ProposalEvidence, Mapping[str, Tensor], bool], Reconstruction
    ]

    def validate_payload(self, evidence: ProposalEvidence) -> None:
        if (
            evidence.backend != self.name
            or evidence.payload.schema != self.payload_schema
        ):
            raise ValueError("proposal evidence is bound to a different backend schema")
        missing = self.required_payload_fields - evidence.payload.values.keys()
        if missing:
            raise ValueError(f"backend payload is incomplete: {sorted(missing)}")

    def reconstruct(
        self,
        evidence: ProposalEvidence,
        *,
        adapter_delta: Mapping[str, Tensor],
        adapter_already_applied: bool,
    ) -> Reconstruction:
        result = self.reconstruct_fn(
            evidence,
            adapter_delta,
            adapter_already_applied,
        )
        if result.proposal_logits.shape != evidence.proposal_logits.shape:
            raise ValueError("backend reconstruction changed proposal-logit shape")
        if result.corrected_distribution.shape != evidence.corrected_distribution.shape:
            raise ValueError(
                "backend reconstruction changed proposal-distribution shape"
            )
        if (result.confidence is None) != (evidence.confidence is None):
            raise ValueError("backend reconstruction changed confidence availability")
        if (
            result.confidence is not None
            and result.confidence.shape != evidence.valid_mask.shape
        ):
            raise ValueError("backend reconstruction changed confidence shape")
        return result


class DFlashBackendContract(FunctionalBackendContract):
    """DFlash contract bound to the deployed differentiable-canvas state."""

    def __init__(
        self,
        reconstruct_fn: Callable[
            [ProposalEvidence, Mapping[str, Tensor], bool], Reconstruction
        ],
    ) -> None:
        super().__init__(
            name="DFLASH",
            payload_schema="dflash-native-v1",
            required_payload_fields=frozenset({"canvas_state", "proposal_correction"}),
            reconstruct_fn=reconstruct_fn,
        )

    def validate_payload(self, evidence: ProposalEvidence) -> None:
        super().validate_payload(evidence)
        if evidence.payload.values["proposal_correction"] != "frozen_at_sampling":
            raise ValueError("DFlash proposal correction must remain sampling-bound")


class EagleBackendContract(FunctionalBackendContract):
    """EAGLE-family topk=1 tree-state reconstruction contract."""

    def __init__(
        self,
        name: Literal["EAGLE", "EAGLE3"],
        reconstruct_fn: Callable[
            [ProposalEvidence, Mapping[str, Tensor], bool], Reconstruction
        ],
    ) -> None:
        schema = "eagle-native-v1" if name == "EAGLE" else "eagle3-native-v1"
        super().__init__(
            name=name,
            payload_schema=schema,
            required_payload_fields=frozenset(
                {"tree_state", "topk", "proposal_correction"}
            ),
            reconstruct_fn=reconstruct_fn,
        )

    def validate_payload(self, evidence: ProposalEvidence) -> None:
        super().validate_payload(evidence)
        values = evidence.payload.values
        if values["topk"] != 1:
            raise ValueError("adapted EAGLE reconstruction requires topk=1")
        if values["proposal_correction"] != "frozen_at_sampling":
            raise ValueError("EAGLE proposal correction must remain sampling-bound")


class NextNBackendContract(FunctionalBackendContract):
    """Native NEXTN interface with an immutable upstream interface digest."""

    def __init__(
        self,
        reconstruct_fn: Callable[
            [ProposalEvidence, Mapping[str, Tensor], bool], Reconstruction
        ],
    ) -> None:
        super().__init__(
            name="NEXTN",
            payload_schema="nextn-native-v1",
            required_payload_fields=frozenset(
                {"mtp_hidden_state", "interface_sha256", "proposal_correction"}
            ),
            reconstruct_fn=reconstruct_fn,
        )

    def validate_payload(self, evidence: ProposalEvidence) -> None:
        super().validate_payload(evidence)
        values = evidence.payload.values
        _require_sha256("NEXTN interface_sha256", values["interface_sha256"])
        hidden = values["mtp_hidden_state"]
        if not isinstance(hidden, Tensor):
            raise TypeError("NEXTN mtp_hidden_state must be a tensor")
        if hidden.device != evidence.proposal_logits.device:
            raise ValueError("NEXTN hidden state must remain device-resident")
        if hidden.shape[:-1] != evidence.valid_mask.shape:
            raise ValueError("NEXTN hidden state must cover proposal rows")
        if values["proposal_correction"] != "frozen_at_sampling":
            raise ValueError("NEXTN proposal correction must remain sampling-bound")


class DSparkBackendContract(FunctionalBackendContract):
    """DSpark envelope requiring sampled-predecessor and native Markov evidence."""

    def __init__(
        self,
        reconstruct_fn: Callable[
            [ProposalEvidence, Mapping[str, Tensor], bool], Reconstruction
        ],
    ) -> None:
        super().__init__(
            name="DSPARK",
            payload_schema="dspark-native-v1",
            required_payload_fields=frozenset(
                {
                    "markov_w1_feature",
                    "markov_w2_feature",
                    "markov_w1_source",
                    "markov_w2_source",
                    "predecessor_source",
                    "scheduler_mode",
                    "proposal_correction",
                }
            ),
            reconstruct_fn=reconstruct_fn,
        )

    def validate_payload(self, evidence: ProposalEvidence) -> None:
        super().validate_payload(evidence)
        values = evidence.payload.values
        if (
            values["markov_w1_source"] != "inference_native"
            or values["markov_w2_source"] != "inference_native"
        ):
            raise ValueError("DSpark requires real inference Markov W1/W2 features")
        if values["predecessor_source"] != "sampled_token":
            raise ValueError("DSpark requires the actual sampled predecessor")
        if values["scheduler_mode"] not in {"fixed_budget", "native_scheduler"}:
            raise ValueError("DSpark scheduler mode is not registered")
        if values["proposal_correction"] != "frozen_at_sampling":
            raise ValueError("DSpark proposal correction must remain sampling-bound")
        for name in ("markov_w1_feature", "markov_w2_feature"):
            tensor = values[name]
            if not isinstance(tensor, Tensor):
                raise TypeError(f"DSpark {name} must be a tensor")
            if tensor.device != evidence.proposal_logits.device:
                raise ValueError("DSpark Markov features must remain device-resident")
            if tensor.shape[:-1] != evidence.valid_mask.shape:
                raise ValueError("DSpark Markov features must cover proposal rows")


def dspark_conditional_survival_target(
    teacher_distribution: Tensor,
    proposal_distribution: Tensor,
) -> Tensor:
    """Stop-gradient target ``1 - TV(target, proposal)`` for confidence."""
    if teacher_distribution.shape != proposal_distribution.shape:
        raise ValueError("target and proposal distributions must align")
    total_variation = 0.5 * (
        teacher_distribution.detach() - proposal_distribution.detach()
    ).abs().sum(dim=-1)
    return (1.0 - total_variation).clamp(0.0, 1.0).detach()


def dspark_composite_loss(
    *,
    teacher_distribution: Tensor,
    proposal_distribution: Tensor,
    confidence_logits: Tensor,
    valid_mask: Tensor,
    confidence_weight: float,
) -> Tensor:
    """Proposal cross-entropy plus a tuning-locked proper confidence loss."""
    if not 0.0 <= confidence_weight < float("inf"):
        raise ValueError("confidence loss weight must be finite and non-negative")
    if teacher_distribution.shape != proposal_distribution.shape:
        raise ValueError("target and proposal distributions must align")
    if (
        valid_mask.dtype is not torch.bool
        or valid_mask.shape != confidence_logits.shape
    ):
        raise ValueError("confidence mask and logits must align")
    if teacher_distribution.shape[:-1] != valid_mask.shape:
        raise ValueError("proposal rows and confidence mask must align")
    tiny = torch.finfo(proposal_distribution.dtype).tiny
    proposal_loss = -(
        teacher_distribution.detach() * proposal_distribution.clamp_min(tiny).log()
    ).sum(dim=-1)
    survival = dspark_conditional_survival_target(
        teacher_distribution,
        proposal_distribution,
    )
    confidence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        confidence_logits,
        survival,
        reduction="none",
    )
    weights = valid_mask.to(proposal_loss.dtype)
    # An empty mask intentionally produces a non-finite candidate. The
    # device-side candidate predicate then discards it without a host read.
    return ((proposal_loss + confidence_weight * confidence_loss) * weights).sum() / (
        weights.sum()
    )
