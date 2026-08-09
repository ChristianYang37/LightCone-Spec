"""Translate strict public configuration to the pinned SGLang patch API."""

from __future__ import annotations

import hashlib
import json

from lightcone_spec.config.schema import RunConfig


def sglang_adaptation_payload(config: RunConfig) -> dict | None:
    if config.method == "static":
        return None
    if config.method.startswith("onlinespec_"):
        raise ValueError(
            "OnlineSpec is an isolated external baseline, not an SGLang adaptation mode"
        )
    adaptation = config.adaptation
    if adaptation is None:
        raise AssertionError("validated non-static config has no adaptation")
    return {
        "schema_version": 2,
        "method": config.method,
        "algorithm": config.model.algorithm,
        "weight_update_mode": adaptation.weight_update_mode,
        "parameter_scope": adaptation.parameter_scope,
        "kv_history_policy": adaptation.kv_history_policy,
        "adaptation_scope": adaptation.adaptation_scope,
        "adaptation_group_id": adaptation.adaptation_group_id,
        "tenant_id": config.tenant_id,
        "optimizer": adaptation.optimizer.model_dump(mode="json"),
        "rank": adaptation.rank,
        "stride": adaptation.stride,
        "max_in_flight": 1,
        "canvas_tokens": adaptation.canvas_tokens,
        "loss_position_decay": adaptation.loss_position_decay,
        "target_revision": config.model.target_revision,
        "drafter_revision": config.model.drafter_revision,
        "sampling_profile_sha256": (config.runtime.sampling_profile_sha256),
        "telemetry_detail": config.runtime.telemetry_detail,
    }


def sglang_adaptation_sha256(config: RunConfig) -> str | None:
    """Return the canonical identity reported by the patched SGLang runtime."""
    payload = sglang_adaptation_payload(config)
    if payload is None:
        return None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
