from __future__ import annotations

import json

import pytest
import torch
from pydantic import ValidationError

from lightcone_spec import PINNED_SGLANG_COMMIT
from lightcone_spec.config import load_run_config, run_config_sha256
from lightcone_spec.config.schema import RunConfig
from lightcone_spec.execution import ControlledExecutionPolicy
from lightcone_spec.sglang_bridge.config import (
    sglang_adaptation_payload,
    sglang_adaptation_sha256,
)


def config_value(method: str = "tts") -> dict:
    value = {
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
            "sglang_commit": PINNED_SGLANG_COMMIT,
            "sampling_profile_sha256": "c" * 64,
            "tensor_parallel_size": 1,
            "speculative_num_draft_tokens": 16,
            "max_running_requests": 48,
            "telemetry_detail": "headline",
        },
        "adaptation": {
            "weight_update_mode": "lora",
            "parameter_scope": "all",
            "kv_history_policy": "frozen",
            "adaptation_scope": "cohort",
            "adaptation_group_id": "confirmation-a",
            "optimizer": {
                "name": "adamw",
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "grad_clip": 1.0,
            },
            "rank": 8,
            "lora_alpha": 8,
            "stride": 10,
            "max_in_flight": 1,
            "canvas_tokens": 16,
            "loss_position_decay": 1.0,
        },
        "tenant_id": "research",
    }
    if method in {"target_only", "static"}:
        value["adaptation"] = None
    if method == "target_only":
        value["runtime"]["speculation_enabled"] = False
    return value


@pytest.mark.parametrize("method", ["target_only", "static"])
def test_disabled_methods_have_no_adaptation_payload_or_tensor_allocation(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_allocation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("disabled method attempted a tensor allocation")

    for name in ("empty", "zeros", "ones", "tensor", "randn"):
        monkeypatch.setattr(torch, name, forbidden_allocation)
    config = RunConfig.model_validate(config_value(method))
    assert sglang_adaptation_payload(config) is None
    assert sglang_adaptation_sha256(config) is None


def test_runtime_binds_registered_role_safe_execution_policy() -> None:
    config = RunConfig.model_validate(config_value("static"))
    assert config.runtime.context_length == 40960
    assert config.runtime.random_seed == 1
    assert config.runtime.disable_radix_cache is True
    assert config.runtime.disable_cuda_graph is True
    assert config.runtime.target_reference_disable_overlap_schedule is True
    assert config.runtime.speculative_disable_overlap_schedule is False
    assert config.runtime.execution_policy_sha256 == ControlledExecutionPolicy().sha256
    value = config_value("static")
    value["runtime"]["execution_policy_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="execution-policy identity mismatch"):
        RunConfig.model_validate(value)


@pytest.mark.parametrize("method", ["tts", "l0"])
def test_formal_adaptation_payload_is_schema_v3(method: str) -> None:
    config = RunConfig.model_validate(config_value(method))
    payload = sglang_adaptation_payload(config)
    assert payload is not None
    assert payload["schema_version"] == 3
    assert payload["method"] == method
    assert payload["algorithm"] == "DFLASH"
    assert payload["kv_history_policy"] == "frozen"
    assert len(sglang_adaptation_sha256(config) or "") == 64


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), 2),
        (("method",), "unsupported"),
        (("runtime", "sglang_commit"), "0" * 40),
        (("model", "target_revision"), "main"),
        (("runtime", "sampling_profile_sha256"), "A" * 64),
    ],
)
def test_legacy_or_unlocked_identity_fails(
    path: tuple[str, ...], value: object
) -> None:
    config = config_value()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        RunConfig.model_validate(config)


@pytest.mark.parametrize(
    ("mode", "scope", "rank", "alpha", "valid"),
    [
        ("lora", "last1", 8, 8, True),
        ("lora", "all", 64, 64, True),
        ("lora", "all", 8, 16, False),
        ("lora", "all", None, None, False),
        ("full", "last5", None, None, True),
        ("full", "all", 8, None, False),
    ],
)
def test_update_mode_contract(
    mode: str,
    scope: str,
    rank: int | None,
    alpha: int | None,
    valid: bool,
) -> None:
    value = config_value()
    value["adaptation"].update(
        weight_update_mode=mode,
        parameter_scope=scope,
        rank=rank,
        lora_alpha=alpha,
    )
    if valid:
        RunConfig.model_validate(value)
    else:
        with pytest.raises(ValidationError):
            RunConfig.model_validate(value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("node_count", 2, "multi-node"),
        ("speculative_num_draft_tokens", 8, r"draft_depth \+ 1"),
        ("canvas_tokens", 8, "canvas width"),
    ],
)
def test_uncertified_runtime_fails_closed(
    field: str, value: object, message: str
) -> None:
    config = config_value()
    if field == "algorithm":
        config["model"][field] = value
    elif field == "canvas_tokens":
        config["adaptation"][field] = value
    else:
        config["runtime"][field] = value
    with pytest.raises(ValidationError, match=message):
        RunConfig.model_validate(config)


def test_unimplemented_tp2_and_replica_local_dp2_topologies_fail_closed() -> None:
    tp = config_value()
    tp["runtime"].update(
        tensor_parallel_size=2,
        tp_rank=1,
        distributed_runtime_capability="patched_two_gpu_v1",
        distributed_capability_receipt_sha256="d" * 64,
    )
    with pytest.raises(ValidationError, match="does not expose TP2/DP2"):
        RunConfig.model_validate(tp)
    dp = config_value()
    dp["runtime"].update(
        data_parallel_size=2,
        dp_rank=1,
        router_identity="sticky-router-v1",
        distributed_runtime_capability="patched_two_gpu_v1",
        distributed_capability_receipt_sha256="d" * 64,
    )
    with pytest.raises(ValidationError, match="does not expose TP2/DP2"):
        RunConfig.model_validate(dp)


@pytest.mark.parametrize(
    "updates",
    [
        {"tensor_parallel_size": 2, "tp_rank": 1},
        {
            "data_parallel_size": 2,
            "dp_rank": 1,
            "router_identity": "sticky-router-v1",
        },
    ],
)
def test_two_gpu_schema_fails_closed_without_runtime_receipt(updates: dict) -> None:
    value = config_value()
    value["runtime"].update(updates)
    with pytest.raises(ValidationError, match="does not expose TP2/DP2"):
        RunConfig.model_validate(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schedule", "inverse_sqrt_published_update"),
        ("extra_logical_delay", 1),
        ("teacher_row_policy", "quota_shadow"),
    ],
)
def test_native_e2_adaptation_modes_are_schema_valid(
    field: str,
    value: object,
) -> None:
    config = config_value()
    target = (
        config["adaptation"]["optimizer"]
        if field == "schedule"
        else config["adaptation"]
    )
    target[field] = value
    parsed = RunConfig.model_validate(config)
    assert parsed.adaptation is not None
    parsed_target = (
        parsed.adaptation.optimizer if field == "schedule" else parsed.adaptation
    )
    assert getattr(parsed_target, field) == value


@pytest.mark.parametrize("algorithm", ["DSPARK", "EAGLE", "EAGLE3", "NEXTN"])
@pytest.mark.parametrize("method", ["tts", "l0"])
def test_unimplemented_backend_adaptation_fails_closed(
    algorithm: str, method: str
) -> None:
    value = config_value(method)
    value["model"]["algorithm"] = algorithm
    value["adaptation"].update(parameter_scope="last1")
    if algorithm == "DSPARK":
        value["adaptation"].update(
            parameter_scope="last3_native_heads",
            native_head_policy="full",
            confidence_loss_weight=0.25,
            verification_mode="fixed_budget",
            fixed_verification_budget=8,
        )
    if algorithm in {"EAGLE", "EAGLE3"}:
        value["runtime"]["speculative_eagle_topk"] = 1
    with pytest.raises(ValidationError, match="adaptation only for DFLASH"):
        RunConfig.model_validate(value)


@pytest.mark.parametrize("algorithm", ["DSPARK", "EAGLE", "EAGLE3", "NEXTN"])
def test_static_is_native_on_every_compatibility_backend(algorithm: str) -> None:
    value = config_value("static")
    value["model"]["algorithm"] = algorithm
    if algorithm in {"EAGLE", "EAGLE3"}:
        value["runtime"]["speculative_eagle_topk"] = 1
    config = RunConfig.model_validate(value)
    assert sglang_adaptation_payload(config) is None


@pytest.mark.parametrize(
    ("name", "updates"),
    [
        ("sgdm", {"momentum": 0.9, "weight_decay": 0.01}),
        ("nag", {"momentum": 0.9, "weight_decay": 0.01}),
        ("lion", {"beta2": 0.99, "weight_decay": 0.01}),
        (
            "muon",
            {
                "momentum": 0.95,
                "muon_ns_steps": 5,
                "muon_auxiliary_learning_rate": 1e-5,
                "muon_auxiliary_weight_decay": 0.01,
                "weight_decay": 0.01,
            },
        ),
    ],
)
def test_formal_methods_accept_extended_optimizers(
    name: str, updates: dict[str, object]
) -> None:
    value = config_value()
    value["adaptation"]["optimizer"].update(
        name=name,
        learning_rate=1e-4,
        **updates,
    )
    RunConfig.model_validate(value)


def test_optimizer_specific_fields_fail_closed() -> None:
    value = config_value()
    value["adaptation"]["optimizer"]["momentum"] = 0.9
    with pytest.raises(ValidationError, match="not a parameter"):
        RunConfig.model_validate(value)

    value = config_value()
    value["adaptation"]["optimizer"].update(
        name="sgdm", learning_rate=1e-3, momentum=0.9, beta2=0.99
    )
    with pytest.raises(ValidationError, match="must stay canonical"):
        RunConfig.model_validate(value)


def test_dflash_canvas_requires_at_least_one_proposal_position() -> None:
    config = config_value()
    config["adaptation"]["canvas_tokens"] = 1
    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        RunConfig.model_validate(config)


@pytest.mark.parametrize("optimizer", ["sgd", "none"])
def test_formal_methods_reject_non_tuning_optimizers(optimizer: str) -> None:
    value = config_value()
    value["adaptation"]["optimizer"].update(
        name=optimizer,
        learning_rate=0.0 if optimizer == "none" else 0.1,
        weight_decay=0.0,
    )
    with pytest.raises(ValidationError):
        RunConfig.model_validate(value)


def test_external_baseline_has_explicit_clean_room_runtime_state() -> None:
    value = config_value("onlinespec_ogd")
    value["adaptation"].update(
        weight_update_mode="full",
        parameter_scope="all",
        optimizer={
            "name": "sgd",
            "learning_rate": 0.1,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "grad_clip": 1.0,
        },
        rank=None,
        lora_alpha=None,
    )
    value["online_spec"] = {
        "projection_radius": None,
        "additional_learning_rates": [],
        "hedge_learning_rate": None,
    }
    config = RunConfig.model_validate(value)
    payload = sglang_adaptation_payload(config)
    assert payload is not None
    assert payload["method"] == "onlinespec_ogd"
    assert payload["online_spec"]["projection_radius"] is None


def test_loader_is_deterministic_and_strict(tmp_path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config_value()), encoding="utf-8")
    first = load_run_config(path)
    second = load_run_config(path)
    assert run_config_sha256(first) == run_config_sha256(second)
    value = config_value()
    value["unexpected"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_run_config(path)
