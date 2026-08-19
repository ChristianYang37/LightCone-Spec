"""Exact-one content lineage for signed and trusted-operator execution.

The offline-root-signed branch remains a path-bound
``ContentVerificationReceipt``.  The trusted single-operator branch is a
path-bound :class:`TrustedSingleOperatorContentBundle` and deliberately has no
signature or formal authorization claim.  Consumers carry this tagged value
instead of overloading an ``*_authorization_sha256`` field with an unsigned
bundle digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Self

from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedSingleOperatorContentBundle,
    TrustedSingleOperatorContentBundleBinding,
)
from lightcone_spec.runtime.content_authorization import ContentVerificationReceipt
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

FormalContentSourceMode = Literal[
    "offline_root_signed",
    "trusted_single_operator",
]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict(value: object, *, label: str, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


@dataclass(frozen=True)
class FormalContentSourceBinding:
    """Frozen, exact-one path binding for execution content.

    ``offline_root_signed`` is intentionally only a codec/deep-reopen here;
    the existing consumer-specific ``revalidate_formal_scope`` calls remain
    responsible for signature, freshness, and scope validation.  The trusted
    branch requires a fully runtime-bound bundle before it can enter an
    execution artifact.
    """

    schema_version: Literal[1]
    kind: Literal["formal_content_source_binding"]
    mode: FormalContentSourceMode
    offline_root_signed: CanonicalJsonProofBinding | None
    trusted_single_operator: TrustedSingleOperatorContentBundleBinding | None

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_content_source_binding"
            or self.mode not in {"offline_root_signed", "trusted_single_operator"}
        ):
            raise ValueError("formal content source binding schema differs")
        present = (
            self.offline_root_signed is not None,
            self.trusted_single_operator is not None,
        )
        if sum(present) != 1 or present != (
            self.mode == "offline_root_signed",
            self.mode == "trusted_single_operator",
        ):
            raise ValueError("formal content source binding is not exact-one")
        self.reopen()

    @classmethod
    def bind_offline_root_signed(cls, path: str) -> Self:
        binding = CanonicalJsonProofBinding.bind(path)
        receipt = ContentVerificationReceipt.from_dict(binding.reopen())
        if receipt.sha256 != binding.semantic_sha256:
            raise ValueError("offline signed content receipt digest differs")
        return cls(
            schema_version=1,
            kind="formal_content_source_binding",
            mode="offline_root_signed",
            offline_root_signed=binding,
            trusted_single_operator=None,
        )

    @classmethod
    def bind_trusted_single_operator(cls, path: str) -> Self:
        binding = TrustedSingleOperatorContentBundleBinding.bind(path)
        if binding.runtime_binding_status != "BOUND":
            raise ValueError("trusted content bundle is not runtime BOUND")
        return cls(
            schema_version=1,
            kind="formal_content_source_binding",
            mode="trusted_single_operator",
            offline_root_signed=None,
            trusted_single_operator=binding,
        )

    @property
    def content_sha256(self) -> str:
        if self.mode == "offline_root_signed":
            assert self.offline_root_signed is not None
            return self.offline_root_signed.semantic_sha256
        assert self.trusted_single_operator is not None
        return self.trusted_single_operator.semantic_sha256

    def reopen(
        self,
    ) -> ContentVerificationReceipt | TrustedSingleOperatorContentBundle:
        if self.mode == "offline_root_signed":
            binding = self.offline_root_signed
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("offline signed content binding is not path-bound")
            if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
                raise RuntimeError("offline signed content receipt binding changed")
            receipt = ContentVerificationReceipt.from_dict(binding.reopen())
            if receipt.sha256 != binding.semantic_sha256:
                raise RuntimeError("offline signed content receipt identity changed")
            return receipt
        binding = self.trusted_single_operator
        if type(binding) is not TrustedSingleOperatorContentBundleBinding:
            raise TypeError("trusted content bundle binding is not exact")
        if binding.runtime_binding_status != "BOUND":
            raise ValueError("trusted content bundle is not runtime BOUND")
        bundle = binding.reopen()
        if (
            bundle.runtime_binding_status != "BOUND"
            or bundle.semantic_sha256 != binding.semantic_sha256
        ):
            raise RuntimeError("trusted content bundle identity changed")
        return bundle

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "mode": self.mode,
            "offline_root_signed": (
                None
                if self.offline_root_signed is None
                else self.offline_root_signed.to_dict()
            ),
            "trusted_single_operator": (
                None
                if self.trusted_single_operator is None
                else self.trusted_single_operator.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            value,
            label="formal content source binding",
            expected={
                "schema_version",
                "kind",
                "mode",
                "offline_root_signed",
                "trusted_single_operator",
            },
        )
        offline = row.pop("offline_root_signed")
        trusted = row.pop("trusted_single_operator")
        return cls(
            **row,
            offline_root_signed=(
                None
                if offline is None
                else CanonicalJsonProofBinding.from_dict(offline)
            ),
            trusted_single_operator=(
                None
                if trusted is None
                else TrustedSingleOperatorContentBundleBinding.from_dict(trusted)
            ),
        )  # type: ignore[arg-type]


__all__ = ["FormalContentSourceBinding", "FormalContentSourceMode"]
