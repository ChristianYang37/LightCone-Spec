from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec.experiments.failure_actuator import (
    FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON,
    FAILURE_SCENARIO_SEMANTICS,
    FAILURE_SCENARIO_SEMANTICS_SHA256,
    RELEASE_FAILURE_ACTUATOR_CAPABILITIES,
    FailureActuatorBlocked,
    FailureActuatorContext,
    FailurePhaseObservation,
    FailureScenarioSemantics,
    FailureTerminalObservation,
    ReleaseFailureActuatorCapability,
    execute_failure_actuator_for_cpu_test,
    failure_phase_observation_sha256,
    failure_semantics,
    release_failure_actuator_capability,
    require_release_failure_actuator,
    validate_failure_recovery_receipt,
)
from lightcone_spec.experiments.failure_authority import (
    FailureInjectionAuthorityResult,
    ReleaseFailurePlan,
    bind_failure_injection_authority,
    release_failure_plan_for_cell,
    revalidate_failure_injection_authority,
)
from lightcone_spec.experiments.registry import (
    E5_FAILURES,
    ExperimentRegistry,
    build_industrial_registry,
    content_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(f"failure-actuator:{label}".encode()).hexdigest()


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return build_industrial_registry()


def _authority(
    tmp_path: Path,
    registry: ExperimentRegistry,
    scenario: str,
) -> FailureInjectionAuthorityResult:
    cell = next(
        value
        for value in registry.cells_for("E5")
        if value.identity.task == "failure_injection"
        and value.identity.arrival == f"failure:{scenario}"
    )
    plan = release_failure_plan_for_cell(registry, cell)
    path = tmp_path / f"plan-{scenario}.json"
    path.write_text(
        json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    binding = bind_failure_injection_authority(path.resolve(), registry=registry)
    return revalidate_failure_injection_authority(binding, registry=registry)


class _FakeActuator:
    actuator_id = "cpu.fake.first_party_fault_actuator.v1"
    actuator_version_sha256 = _sha("actuator-version")

    def __init__(
        self,
        *,
        wrong_proof: bool = False,
        recovery_valid: bool = True,
        trigger_count: int = 1,
        fail_rank: tuple[str, int] | None = None,
    ) -> None:
        self.wrong_proof = wrong_proof
        self.recovery_valid = recovery_valid
        self.trigger_count = trigger_count
        self.fail_rank = fail_rank

    def fresh_context(
        self,
        *,
        plan: ReleaseFailurePlan,
        semantics: FailureScenarioSemantics,
        topology: str,
        rank: int,
        run_nonce_sha256: str,
    ) -> FailureActuatorContext:
        topology_offset = 0 if topology == "tp2_dp1" else 100
        return FailureActuatorContext(
            plan_sha256=plan.sha256,
            scenario_semantics_sha256=semantics.sha256,
            topology=topology,
            rank=rank,
            world_size=2,
            process_id=2000 + topology_offset + rank,
            process_start_monotonic_ns=1,
            session_epoch=0,
            run_nonce_sha256=run_nonce_sha256,
        )

    def _observation(
        self,
        context: FailureActuatorContext,
        phase: str,
        operation: str,
        time: int,
    ) -> FailurePhaseObservation:
        return FailurePhaseObservation(
            phase=phase,
            operation=operation,
            monotonic_ns=time,
            event_count=1,
            observation_sha256=failure_phase_observation_sha256(
                context,
                failure_semantics_from_context(context),
                phase=phase,
                operation=operation,
                event_count=1,
            ),
        )

    def arm(
        self,
        context: FailureActuatorContext,
        semantics: FailureScenarioSemantics,
    ) -> FailurePhaseObservation:
        return self._observation(context, "arm", semantics.arm_operation, 10)

    def trigger(
        self,
        context: FailureActuatorContext,
        semantics: FailureScenarioSemantics,
    ) -> FailurePhaseObservation:
        if self.fail_rank == (context.topology, context.rank):
            raise RuntimeError("deterministic rank actuation failure")
        return self._observation(context, "trigger", semantics.trigger_operation, 20)

    def prove(
        self,
        context: FailureActuatorContext,
        semantics: FailureScenarioSemantics,
    ) -> FailurePhaseObservation:
        operation = "generic_self_report" if self.wrong_proof else semantics.proof_event
        return self._observation(context, "proof", operation, 30)

    def recover(
        self,
        context: FailureActuatorContext,
        semantics: FailureScenarioSemantics,
    ) -> FailurePhaseObservation:
        return self._observation(
            context,
            "recover",
            semantics.recovery_invariant,
            40,
        )

    def terminal(
        self,
        context: FailureActuatorContext,
        semantics: FailureScenarioSemantics,
    ) -> FailureTerminalObservation:
        counters = {
            "exactness_violations": 0,
            "fallbacks": 0,
            "partial_target_continuations": 0,
            "retractions": 0,
            "version_mismatches": 0,
            semantics.terminal_counter: self.trigger_count if context.rank == 0 else 0,
        }
        return FailureTerminalObservation(
            phase="terminal",
            operation=semantics.terminal_counter,
            monotonic_ns=50,
            event_count=1,
            observation_sha256=failure_phase_observation_sha256(
                context,
                semantics,
                phase="terminal",
                operation=semantics.terminal_counter,
                event_count=1,
            ),
            counters=tuple(sorted(counters.items())),
            recovery_valid=self.recovery_valid,
        )


def failure_semantics_from_context(
    context: FailureActuatorContext,
) -> FailureScenarioSemantics:
    return next(
        value
        for value in FAILURE_SCENARIO_SEMANTICS
        if value.sha256 == context.scenario_semantics_sha256
    )


def test_source_semantics_exactly_cover_all_eleven_registered_scenarios() -> None:
    assert tuple(value.scenario for value in FAILURE_SCENARIO_SEMANTICS) == E5_FAILURES
    assert len(FAILURE_SCENARIO_SEMANTICS) == 11
    assert len({value.sha256 for value in FAILURE_SCENARIO_SEMANTICS}) == 11
    assert FAILURE_SCENARIO_SEMANTICS_SHA256 == content_sha256(
        [value.to_dict() for value in FAILURE_SCENARIO_SEMANTICS]
    )
    assert all(
        len(
            {
                value.arm_operation,
                value.trigger_operation,
                value.proof_event,
                value.recovery_invariant,
                value.terminal_counter,
            }
        )
        == 5
        for value in FAILURE_SCENARIO_SEMANTICS
    )


@pytest.mark.parametrize("scenario", E5_FAILURES)
def test_cpu_fake_lifecycle_is_atomic_all_rank_and_recovered_for_each_scenario(
    tmp_path: Path,
    registry: ExperimentRegistry,
    scenario: str,
) -> None:
    authority = _authority(tmp_path, registry, scenario)
    output = (tmp_path / f"receipt-{scenario}.json").resolve()
    receipt = execute_failure_actuator_for_cpu_test(
        authority,
        _FakeActuator(),
        run_nonce_sha256=_sha(f"nonce:{scenario}"),
        receipt_path=output,
    )

    assert output.is_file()
    assert receipt.scenario == scenario
    assert receipt.scenario_semantics_sha256 == failure_semantics(scenario).sha256
    assert receipt.formal_execution_authorized is False
    assert receipt.correctness_only is True
    assert receipt.recovered is receipt.committed is True
    assert tuple((value.topology, value.rank) for value in receipt.rank_receipts) == (
        ("tp2_dp1", 0),
        ("tp2_dp1", 1),
        ("two_replica_tp1_dp2", 0),
        ("two_replica_tp1_dp2", 1),
    )
    assert all(value.session_epoch == 0 for value in receipt.rank_receipts)
    assert all(value.recovery_valid for value in receipt.rank_receipts)
    validate_failure_recovery_receipt(receipt, authority)


@pytest.mark.parametrize(
    ("actuator", "message"),
    [
        (_FakeActuator(wrong_proof=True), "source-owned scenario semantics"),
        (_FakeActuator(recovery_valid=False), "did not prove recovery"),
        (_FakeActuator(trigger_count=0), "missing from a registered topology"),
    ],
)
def test_generic_proof_missing_recovery_or_missing_counter_cannot_commit(
    tmp_path: Path,
    registry: ExperimentRegistry,
    actuator: _FakeActuator,
    message: str,
) -> None:
    authority = _authority(tmp_path, registry, "communicator_failure")
    output = (tmp_path / "must-not-exist.json").resolve()
    with pytest.raises(ValueError, match=message):
        execute_failure_actuator_for_cpu_test(
            authority,
            actuator,
            run_nonce_sha256=_sha("negative-nonce"),
            receipt_path=output,
        )
    assert not output.exists()


def test_one_rank_failure_prevents_atomic_receipt(
    tmp_path: Path,
    registry: ExperimentRegistry,
) -> None:
    authority = _authority(tmp_path, registry, "slow_rank")
    output = (tmp_path / "partial.json").resolve()
    with pytest.raises(RuntimeError, match="rank actuation failure"):
        execute_failure_actuator_for_cpu_test(
            authority,
            _FakeActuator(fail_rank=("two_replica_tp1_dp2", 1)),
            run_nonce_sha256=_sha("partial-nonce"),
            receipt_path=output,
        )
    assert not output.exists()


def test_release_capability_is_named_block_before_actuator_or_output(
    tmp_path: Path,
) -> None:
    assert RELEASE_FAILURE_ACTUATOR_CAPABILITIES == ()
    output = tmp_path / "never-created.json"
    with pytest.raises(FailureActuatorBlocked) as blocked:
        require_release_failure_actuator()
    assert blocked.value.reason == FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON
    assert not output.exists()


def test_release_registry_binds_callable_and_identity_as_one_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightcone_spec.experiments.failure_actuator as module

    def release_failure_actuator_factory():
        return _FakeActuator()

    release_failure_actuator_factory.__module__ = (
        "lightcone_spec.experiments.failure_actuator"
    )
    release_failure_actuator_factory.__qualname__ = "release_failure_actuator_factory"
    capability = ReleaseFailureActuatorCapability(
        actuator_id="release.first_party_fault_actuator.v1",
        actuator_version_sha256=_sha("release-actuator-version"),
        factory_module=release_failure_actuator_factory.__module__,
        factory_qualname=release_failure_actuator_factory.__qualname__,
        factory=release_failure_actuator_factory,
    )
    monkeypatch.setattr(
        module,
        "RELEASE_FAILURE_ACTUATOR_CAPABILITIES",
        (capability,),
    )
    assert release_failure_actuator_capability() == capability
    assert capability.sha256 == content_sha256(
        {
            "schema_version": 1,
            "kind": "release_failure_actuator_capability",
            "actuator_id": capability.actuator_id,
            "actuator_version_sha256": capability.actuator_version_sha256,
            "factory_module": capability.factory_module,
            "factory_qualname": capability.factory_qualname,
        }
    )
    release_failure_actuator_factory.__qualname__ = "replaced_factory"
    with pytest.raises(FailureActuatorBlocked) as replaced:
        release_failure_actuator_capability()
    assert replaced.value.reason == FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON

    monkeypatch.setattr(
        module,
        "RELEASE_FAILURE_ACTUATOR_CAPABILITIES",
        ((capability.actuator_id, capability.actuator_version_sha256),),
    )
    with pytest.raises(FailureActuatorBlocked) as blocked:
        release_failure_actuator_capability()
    assert blocked.value.reason == FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON


def test_cpu_diagnostic_actuator_cannot_be_registered_for_release() -> None:
    def source_factory():
        return _FakeActuator()

    source_factory.__module__ = "lightcone_spec.experiments.failure_actuator"
    source_factory.__qualname__ = "source_factory"
    with pytest.raises(ValueError, match="diagnostic actuator identities"):
        ReleaseFailureActuatorCapability(
            actuator_id=_FakeActuator.actuator_id,
            actuator_version_sha256=_FakeActuator.actuator_version_sha256,
            factory_module=source_factory.__module__,
            factory_qualname=source_factory.__qualname__,
            factory=source_factory,
        )


def test_scenario_semantics_cannot_be_posthoc_substituted(
    tmp_path: Path,
    registry: ExperimentRegistry,
) -> None:
    authority = _authority(tmp_path, registry, "disk_quota")
    wrong = replace(
        failure_semantics("disk_quota"),
        proof_event=failure_semantics("oom_candidate").proof_event,
    )
    context = _FakeActuator().fresh_context(
        plan=authority.plan,
        semantics=wrong,
        topology="tp2_dp1",
        rank=0,
        run_nonce_sha256=_sha("substitution"),
    )
    assert context.scenario_semantics_sha256 != failure_semantics("disk_quota").sha256

    class ForeignContextActuator(_FakeActuator):
        def fresh_context(self, **kwargs):  # type: ignore[no-untyped-def]
            valid = super().fresh_context(**kwargs)
            return replace(valid, scenario_semantics_sha256=wrong.sha256)

    output = (tmp_path / "foreign-context.json").resolve()
    with pytest.raises(ValueError, match="foreign execution context"):
        execute_failure_actuator_for_cpu_test(
            authority,
            ForeignContextActuator(),
            run_nonce_sha256=_sha("substitution"),
            receipt_path=output,
        )
    assert not output.exists()
