from __future__ import annotations

import json

import pytest
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
        "schema_version": 2,
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
            "parameter_scope": "drafter",
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
            "stride": 10,
            "max_in_flight": 1,
            "canvas_tokens": 16,
            "loss_position_decay": 1.0,
        },
        "tenant_id": "research",
    }
    if method == "static":
        value["adaptation"] = None
    return value


def test_static_has_no_adaptation_payload() -> None:
    config = RunConfig.model_validate(config_value("static"))
    assert sglang_adaptation_payload(config) is None
    assert sglang_adaptation_sha256(config) is None


def test_runtime_binds_registered_execution_policy() -> None:
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


@pytest.mark.parametrize("method", ["tts", "naive_async"])
def test_formal_adaptation_payload_is_schema_v2(method: str) -> None:
    config = RunConfig.model_validate(config_value(method))
    payload = sglang_adaptation_payload(config)
    assert payload is not None
    assert payload["schema_version"] == 2
    assert payload["method"] == method
    assert payload["algorithm"] == "DFLASH"
    assert payload["kv_history_policy"] == "frozen"
    assert len(sglang_adaptation_sha256(config) or "") == 64


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), 1),
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
    ("mode", "scope", "rank", "valid"),
    [
        ("residual", "tail", 8, True),
        ("residual", "drafter", 8, False),
        ("lora", "tail", 8, True),
        ("lora", "drafter", 8, True),
        ("lora", "drafter", None, False),
        ("full", "tail", None, True),
        ("full", "drafter", None, True),
        ("full", "drafter", 8, False),
    ],
)
def test_update_mode_contract(
    mode: str, scope: str, rank: int | None, valid: bool
) -> None:
    value = config_value()
    value["adaptation"].update(
        weight_update_mode=mode, parameter_scope=scope, rank=rank
    )
    if valid:
        RunConfig.model_validate(value)
    else:
        with pytest.raises(ValidationError):
            RunConfig.model_validate(value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tensor_parallel_size", 2, "requires TP=DP=1"),
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


@pytest.mark.parametrize("algorithm", ["DSPARK", "EAGLE", "EAGLE3"])
@pytest.mark.parametrize("method", ["tts", "naive_async"])
def test_tail_adaptation_is_available_on_linear_backends(
    algorithm: str, method: str
) -> None:
    value = config_value(method)
    value["model"]["algorithm"] = algorithm
    value["adaptation"].update(
        weight_update_mode="residual",
        parameter_scope="tail",
        rank=8,
    )
    if algorithm in {"EAGLE", "EAGLE3"}:
        value["runtime"]["speculative_eagle_topk"] = 1
    config = RunConfig.model_validate(value)
    payload = sglang_adaptation_payload(config)
    assert payload is not None
    assert payload["algorithm"] == algorithm
    assert payload["method"] == method


@pytest.mark.parametrize("algorithm", ["DSPARK", "EAGLE", "EAGLE3"])
def test_static_is_native_on_every_compatibility_backend(algorithm: str) -> None:
    value = config_value("static")
    value["model"]["algorithm"] = algorithm
    if algorithm in {"EAGLE", "EAGLE3"}:
        value["runtime"]["speculative_eagle_topk"] = 1
    config = RunConfig.model_validate(value)
    assert sglang_adaptation_payload(config) is None


@pytest.mark.parametrize("algorithm", ["DSPARK", "EAGLE", "EAGLE3"])
def test_cross_backend_drafter_scope_fails_closed(algorithm: str) -> None:
    value = config_value()
    value["model"]["algorithm"] = algorithm
    if algorithm in {"EAGLE", "EAGLE3"}:
        value["runtime"]["speculative_eagle_topk"] = 1
    with pytest.raises(ValidationError, match="parameter_scope=tail"):
        RunConfig.model_validate(value)


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
        parameter_scope="tail",
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
