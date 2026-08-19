from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import lightcone_spec.experiments.formal_failure_actuator as formal_module
from lightcone_spec.experiments.failure_actuator import (
    FailureActuatorContext,
    FailureRankLaunchBinding,
    failure_phase_observation_sha256,
    failure_semantics,
)
from lightcone_spec.experiments.formal_failure_actuator import (
    FormalFailureActuatorLaunchBinding,
    execute_formal_failure_actuator_unsigned,
)
from lightcone_spec.experiments.registry import E5_FAILURES


def _sha(label: str) -> str:
    return hashlib.sha256(f"formal-failure:{label}".encode()).hexdigest()


class _CpuFailureTransport:
    def __init__(self, *, recovered: bool = True) -> None:
        self.calls: list[tuple[int, str]] = []
        self.recovered = recovered

    def invoke(self, rank, payload):  # type: ignore[no-untyped-def]
        phase = payload["phase"]
        operation = payload["operation"]
        self.calls.append((rank.rank, phase))
        semantics = failure_semantics(payload["scenario"])
        context = FailureActuatorContext(
            plan_sha256=payload["plan_sha256"],
            scenario_semantics_sha256=payload["scenario_semantics_sha256"],
            topology=rank.topology,
            rank=rank.rank,
            world_size=1 if rank.topology == "tp1_dp1" else 2,
            process_id=rank.process_id,
            process_start_monotonic_ns=rank.process_start_monotonic_ns,
            session_epoch=0,
            run_nonce_sha256=payload["run_nonce_sha256"],
        )
        response = {
            "phase": phase,
            "operation": operation,
            "monotonic_ns": {
                "arm": 10,
                "trigger": 20,
                "proof": 30,
                "recover": 40,
                "terminal": 50,
            }[phase],
            "event_count": 1,
            "observation_sha256": failure_phase_observation_sha256(
                context,
                semantics,
                phase=phase,
                operation=operation,
                event_count=1,
            ),
        }
        if phase == "terminal":
            response.update(
                counters={semantics.terminal_counter: 1},
                recovery_valid=self.recovered,
            )
        return response


def _binding(*, scenario: str, topology: str):
    gpus = (
        ("GPU-formal-0",)
        if topology == "tp1_dp1"
        else (
            "GPU-formal-0",
            "GPU-formal-1",
        )
    )
    subject = SimpleNamespace(
        assignment_sha256=_sha(f"assignment:{scenario}:{topology}"),
        inventory_sha256=_sha("inventory"),
        registry_sha256=_sha("registry"),
        serving_execution_plan_sha256=_sha(f"plan:{scenario}:{topology}"),
        materialized_cell_id=_sha(f"cell:{scenario}:{topology}"),
        scenario=scenario,
        topology=topology,
        run_nonce_sha256=_sha(f"nonce:{scenario}:{topology}"),
    )
    return SimpleNamespace(
        sha256=_sha(f"binding:{scenario}:{topology}"),
        subject=subject,
        serving_execution=SimpleNamespace(subject=SimpleNamespace(gpu_uuids=gpus)),
    )


def _launch(tmp_path: Path, binding) -> FormalFailureActuatorLaunchBinding:
    ranks = (0,) if binding.subject.topology == "tp1_dp1" else (0, 1)
    rows = []
    for rank in ranks:
        quota = tmp_path / f"quota-{rank}"
        quota.mkdir(mode=0o700)
        topology = binding.subject.topology
        group_id = 4000 if topology == "tp2_dp1" else 4000 + rank
        rows.append(
            FailureRankLaunchBinding(
                topology=topology,
                rank=rank,
                process_id=3000 + rank,
                process_group_id=group_id,
                process_start_monotonic_ns=100 + rank,
                gpu_uuid=binding.serving_execution.subject.gpu_uuids[rank],
                control_url=(
                    f"http://127.0.0.1:{19000 + rank}"
                    "/v1/lightcone-spec/failure-actuator"
                ),
                temp_quota_root=str(quota),
            )
        )
    subject = binding.subject
    return FormalFailureActuatorLaunchBinding(
        schema_version=1,
        kind="formal_e5_failure_actuator_launch_binding",
        formal_failure_execution_binding_sha256=binding.sha256,
        assignment_sha256=subject.assignment_sha256,
        inventory_sha256=subject.inventory_sha256,
        registry_sha256=subject.registry_sha256,
        serving_execution_plan_sha256=subject.serving_execution_plan_sha256,
        run_nonce_sha256=subject.run_nonce_sha256,
        topology=subject.topology,
        ranks=tuple(rows),
    )


@pytest.mark.parametrize("scenario", E5_FAILURES)
@pytest.mark.parametrize("topology", ("tp1_dp1", "tp2_dp1", "tp1_dp2"))
def test_cpu_transport_exercises_every_formal_failure_and_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    topology: str,
) -> None:
    binding = _binding(scenario=scenario, topology=topology)
    launch = _launch(tmp_path, binding)
    transport = _CpuFailureTransport()
    monkeypatch.setattr(
        formal_module,
        "require_verified_formal_failure_execution_binding",
        lambda value: value,
    )
    monkeypatch.setattr(
        formal_module,
        "SglangHttpFailureNativeControlTransport",
        _CpuFailureTransport,
    )

    proof = execute_formal_failure_actuator_unsigned(
        binding,
        launch=launch,
        transport=transport,
        raw_terminal_path=tmp_path / "terminal.json",
    )

    raw = proof.reopen()
    ranks = 1 if topology == "tp1_dp1" else 2
    assert len(transport.calls) == ranks * 5
    assert raw["formal_execution_authorized"] is False
    assert raw["recovery_receipt"]["scenario"] == scenario
    assert raw["recovery_receipt"]["topology"] == topology
    assert raw["recovery_receipt"]["recovered"] is True


def test_formal_failure_transport_type_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding(scenario=E5_FAILURES[0], topology="tp1_dp1")
    launch = _launch(tmp_path, binding)
    monkeypatch.setattr(
        formal_module,
        "require_verified_formal_failure_execution_binding",
        lambda value: value,
    )

    with pytest.raises(TypeError, match="source HTTP transport"):
        execute_formal_failure_actuator_unsigned(
            binding,
            launch=launch,
            transport=_CpuFailureTransport(),
            raw_terminal_path=tmp_path / "terminal.json",
        )


def test_formal_failure_negative_recovery_is_a_complete_fail_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding(scenario=E5_FAILURES[0], topology="tp1_dp1")
    launch = _launch(tmp_path, binding)
    transport = _CpuFailureTransport(recovered=False)
    monkeypatch.setattr(
        formal_module,
        "require_verified_formal_failure_execution_binding",
        lambda value: value,
    )
    monkeypatch.setattr(
        formal_module,
        "SglangHttpFailureNativeControlTransport",
        _CpuFailureTransport,
    )

    proof = execute_formal_failure_actuator_unsigned(
        binding,
        launch=launch,
        transport=transport,
        raw_terminal_path=tmp_path / "negative-terminal.json",
    )

    receipt = proof.reopen()["recovery_receipt"]
    assert receipt["recovered"] is False
    assert receipt["correctness_only"] is True
    assert receipt["rank_receipts"][0]["recovery_valid"] is False
