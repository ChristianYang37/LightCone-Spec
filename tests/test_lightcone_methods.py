from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lightcone_spec.methods.base import ArrivalContext, DecisionKind
from lightcone_spec.methods.lightcone import LCDampMethod, LCGateMethod

from conftest import make_signal


def _arrival(phi: torch.Tensor, delay_rounds: int) -> ArrivalContext:
    return ArrivalContext(
        arrival_round=5,
        active_version=0,
        phi_active=phi,
        delay_rounds=delay_rounds,
        delay_tokens=4 * delay_rounds,
        delay_wall_us=100.0 * delay_rounds,
        delay_versions=0,
        rho_path=0.0,
        endpoint_distance=0.0,
        parameter_displacement=0.0,
    )


def test_gate_discard_all_is_true_discard_without_prediction(shapes, basis):
    artifact = SimpleNamespace(gate_discard_all=True, gate_threshold=float("-inf"))
    method = LCGateMethod(shapes, basis, SimpleNamespace(), artifact)
    candidate = SimpleNamespace(
        candidate_delta=torch.ones(shapes.num_params())
    )

    decision = method.decide(candidate, _arrival(torch.zeros(shapes.num_params()), 1))

    assert decision.kind == DecisionKind.DISCARD
    assert decision.gate_applied is False
    assert torch.count_nonzero(decision.published_delta) == 0


def test_gate_discard_all_preserves_zero_delay_parity(shapes, basis, monkeypatch):
    artifact = SimpleNamespace(gate_discard_all=True, gate_threshold=float("-inf"))
    method = LCGateMethod(shapes, basis, SimpleNamespace(), artifact)
    monkeypatch.setattr(
        method,
        "_predictions",
        lambda ctx: pytest.fail("zero-delay identity action ran the predictor"),
    )
    phi = torch.zeros(shapes.num_params())
    candidate = SimpleNamespace(candidate_delta=torch.ones_like(phi))

    decision = method.decide(candidate, _arrival(phi, 0))

    assert decision.kind == DecisionKind.APPLY
    assert torch.equal(decision.published_delta, candidate.candidate_delta)


def test_gate_constant_delay_discard_avoids_predictor_and_noop_publish(
    shapes, basis, monkeypatch
):
    artifact = SimpleNamespace(
        gate_discard_all=False,
        gate_threshold=0.2,
        extra={"gate_constant_discard_delays": [1]},
    )
    method = LCGateMethod(shapes, basis, SimpleNamespace(), artifact)
    monkeypatch.setattr(
        method,
        "_predictions",
        lambda ctx: pytest.fail("constant-delay discard ran the predictor"),
    )
    phi = torch.zeros(shapes.num_params())
    candidate = SimpleNamespace(candidate_delta=torch.ones_like(phi))

    decision = method.decide(candidate, _arrival(phi, 1))

    assert decision.kind == DecisionKind.DISCARD
    assert decision.gate_applied is False
    assert torch.count_nonzero(decision.published_delta) == 0


def test_gate_constant_delay_apply_avoids_predictor_and_reuses_delta(
    shapes, basis, monkeypatch
):
    artifact = SimpleNamespace(
        gate_discard_all=False,
        gate_threshold=0.2,
        extra={
            "gate_constant_apply_delays": [2],
            "constant_controller_profiles": {
                "2": {
                    "predicted_utility": 0.4,
                    "predicted_mismatch": 0.0,
                    "predicted_harm_probability": 0.0,
                }
            },
        },
    )
    method = LCGateMethod(shapes, basis, SimpleNamespace(), artifact)
    monkeypatch.setattr(
        method,
        "_predictions",
        lambda ctx: pytest.fail("constant-delay apply ran the predictor"),
    )
    phi = torch.zeros(shapes.num_params())
    candidate = SimpleNamespace(candidate_delta=torch.ones_like(phi))

    decision = method.decide(candidate, _arrival(phi, 2))

    assert decision.kind == DecisionKind.APPLY
    assert decision.gate_applied is True
    assert decision.published_delta is candidate.candidate_delta


def test_damp_constant_delay_profile_avoids_predictor(shapes, basis, monkeypatch):
    artifact = SimpleNamespace(
        extra={
            "constant_controller_profiles": {
                "1": {
                    "predicted_utility": 0.1,
                    "predicted_mismatch": 2.0,
                    "predicted_harm_probability": 0.3,
                    "damping_factor": 0.25,
                }
            }
        }
    )
    method = LCDampMethod(shapes, basis, SimpleNamespace(), artifact)
    monkeypatch.setattr(
        method,
        "_predictions",
        lambda ctx: pytest.fail("constant-delay damping ran the predictor"),
    )
    phi = torch.zeros(shapes.num_params())
    candidate = SimpleNamespace(candidate_delta=torch.ones_like(phi))

    decision = method.decide(candidate, _arrival(phi, 1))

    assert decision.kind == DecisionKind.DAMP
    assert decision.damping_factor == 0.25
    assert torch.equal(decision.published_delta, 0.25 * candidate.candidate_delta)


def test_damp_unit_kappa_reuses_candidate_without_multiply(shapes, basis, monkeypatch):
    artifact = SimpleNamespace(
        extra={
            "constant_controller_profiles": {
                "1": {
                    "predicted_utility": 0.4,
                    "predicted_mismatch": 0.0,
                    "predicted_harm_probability": 0.0,
                    "damping_factor": 1.0,
                }
            }
        }
    )
    method = LCDampMethod(shapes, basis, SimpleNamespace(), artifact)
    monkeypatch.setattr(
        method,
        "_predictions",
        lambda ctx: pytest.fail("constant unit-damping ran the predictor"),
    )
    phi = torch.zeros(shapes.num_params())
    candidate = SimpleNamespace(candidate_delta=torch.ones_like(phi))

    decision = method.decide(candidate, _arrival(phi, 1))

    assert decision.damping_factor == 1.0
    assert decision.published_delta is candidate.candidate_delta


@pytest.mark.integration
def test_static_gate_uses_native_folded_graph_only_while_rows_are_zero():
    from sglang.srt.speculative.dspark_components.dspark_adaptation import (
        DSparkAdaptationManager,
        DSparkStaticTelemetryManager,
    )

    manager = object.__new__(DSparkAdaptationManager)
    manager.config = SimpleNamespace(
        trace=SimpleNamespace(trace_capture_max_bytes=0), update_stride=4
    )
    manager._controller_static_fallback = True
    manager._round_of = {"r": 4}
    batch = SimpleNamespace(reqs=[SimpleNamespace(rid="r")])
    manager._has_enabled_requests = lambda batch: False

    assert manager.folded_graph_adaptation is None
    static_manager = object.__new__(DSparkStaticTelemetryManager)
    assert static_manager.folded_graph_adaptation is None
    assert manager.allow_folded(batch=batch, adapter_captured=False)

    manager._has_enabled_requests = lambda batch: True
    assert not manager.allow_folded(batch=batch, adapter_captured=False)

    manager.config.trace.trace_capture_max_bytes = 1
    assert not manager.allow_folded(batch=batch, adapter_captured=False)


@pytest.mark.integration
def test_delay_specific_constant_gate_retains_candidate_runtime(monkeypatch):
    from sglang.srt.speculative.dspark_components import dspark_adaptation as module

    config = SimpleNamespace(
        method="lc_gate",
        model=SimpleNamespace(pair_id="toy_dspark"),
        weight_update_mode="output_residual",
        async_=SimpleNamespace(logical_delay_rounds=0),
        trace=SimpleNamespace(trace_capture_max_bytes=0),
        controller=SimpleNamespace(artifact_path="controller.json"),
    )
    artifact = SimpleNamespace(
        gate_discard_all=False,
        extra={"gate_constant_discard_delays": [1]},
    )
    monkeypatch.setattr(
        "lightcone_spec.config.loader.load_adaptation_config", lambda path: config
    )
    monkeypatch.setattr(
        "lightcone_spec.controller.artifact.load_bound_controller_artifact",
        lambda *args, **kwargs: artifact,
    )
    monkeypatch.setattr(
        "lightcone_spec.methods.registry.validate_controller_artifact",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "DSparkAdaptationManager",
        lambda **kwargs: ("adaptive", kwargs),
    )
    server_args = SimpleNamespace(dspark_adaptation_config="adaptation.yaml")
    result = module.maybe_build_adaptation_manager(server_args, object())

    assert result[0] == "adaptive"
    assert result[1]["config_path"] == "adaptation.yaml"
