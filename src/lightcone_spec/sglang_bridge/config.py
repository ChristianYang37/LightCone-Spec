"""Translate strict public configuration to the pinned SGLang patch API."""

from __future__ import annotations

import hashlib
import json

from lightcone_spec.config.schema import RunConfig


def sglang_adaptation_payload(config: RunConfig) -> dict | None:
    # The bridge is a direct Python boundary as well as a CLI helper.  Re-run
    # the runtime gate so ``model_construct`` cannot smuggle an unregistered
    # distributed mode into the native patch payload.
    config.runtime.validate_topology()
    if config.method in {"target_only", "static"}:
        return None
    adaptation = config.adaptation
    if adaptation is None:
        raise AssertionError("validated non-static config has no adaptation")
    payload = {
        "schema_version": 3,
        "method": config.method,
        "algorithm": config.model.algorithm,
        "weight_update_mode": adaptation.weight_update_mode,
        "parameter_scope": adaptation.parameter_scope,
        "kv_history_policy": adaptation.kv_history_policy,
        "adaptation_scope": adaptation.adaptation_scope,
        "reset_scope": adaptation.reset_scope,
        "request_admission_policy": adaptation.request_admission_policy,
        "adaptation_group_id": adaptation.adaptation_group_id,
        "tenant_id": config.tenant_id,
        "optimizer": adaptation.optimizer.model_dump(mode="json"),
        "rank": adaptation.rank,
        "lora_alpha": adaptation.lora_alpha,
        "lora_matrix_policy": adaptation.lora_matrix_policy,
        "native_head_policy": adaptation.native_head_policy,
        "stride": adaptation.stride,
        "max_in_flight": 1,
        "canvas_tokens": adaptation.canvas_tokens,
        "loss_position_decay": adaptation.loss_position_decay,
        "extra_logical_delay": adaptation.extra_logical_delay,
        "teacher_row_policy": adaptation.teacher_row_policy,
        "verification_mode": adaptation.verification_mode,
        "fixed_verification_budget": adaptation.fixed_verification_budget,
        "confidence_loss_weight": adaptation.confidence_loss_weight,
        "chronobelief_release_capability_sha256": (
            adaptation.chronobelief_release_capability_sha256
        ),
        "chronobelief_gpu_proof_sha256": (adaptation.chronobelief_gpu_proof_sha256),
        "eagle3_e0_execution_authority_sha256": (
            adaptation.eagle3_e0_execution_authority_sha256
        ),
        "eagle3_compatibility_authority_sha256": (
            adaptation.eagle3_compatibility_authority_sha256
        ),
        "eagle3_model_selector_sha256": adaptation.eagle3_model_selector_sha256,
        "eagle3_native_gpu_proof_sha256": (adaptation.eagle3_native_gpu_proof_sha256),
        "eagle3_qualification_compatibility_authority_sha256": (
            adaptation.eagle3_qualification_compatibility_authority_sha256
        ),
        "eagle3_qualification_model_selector_sha256": (
            adaptation.eagle3_qualification_model_selector_sha256
        ),
        "target_revision": config.model.target_revision,
        "drafter_revision": config.model.drafter_revision,
        "sampling_profile_sha256": (config.runtime.sampling_profile_sha256),
        "telemetry_detail": config.runtime.telemetry_detail,
        "adaptation_microbatch_size": (config.runtime.adaptation_microbatch_size),
        "adaptation_publication_coalescing": (
            config.runtime.adaptation_publication_coalescing
        ),
        "adaptation_stream_priority": (config.runtime.adaptation_stream_priority),
        "topology": {
            "tensor_parallel_size": config.runtime.tensor_parallel_size,
            "data_parallel_size": config.runtime.data_parallel_size,
            "tp_rank": config.runtime.tp_rank,
            "dp_rank": config.runtime.dp_rank,
            "node_count": config.runtime.node_count,
            "node_rank": config.runtime.node_rank,
            "device_identity": config.runtime.device_identity,
            "rendezvous_identity": config.runtime.rendezvous_identity,
            "router_identity": config.runtime.router_identity,
            "clock_identity": config.runtime.clock_identity,
            "process_group_backend": config.runtime.process_group_backend,
            "distributed_runtime_capability": (
                config.runtime.distributed_runtime_capability
            ),
            "distributed_release_capability_sha256": (
                config.runtime.distributed_release_capability_sha256
            ),
            "distributed_capability_receipt_sha256": (
                config.runtime.distributed_capability_receipt_sha256
            ),
            "distributed_control_mode": config.runtime.distributed_control_mode,
            "adaptation_collective_mode": (config.runtime.adaptation_collective_mode),
        },
    }
    if config.online_spec is not None:
        payload["online_spec"] = config.online_spec.model_dump(mode="json")
    return payload


def sglang_adaptation_sha256(config: RunConfig) -> str | None:
    """Return the canonical identity reported by the patched SGLang runtime."""
    payload = sglang_adaptation_payload(config)
    if payload is None:
        return None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
