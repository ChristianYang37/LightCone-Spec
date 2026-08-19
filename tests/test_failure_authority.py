from __future__ import annotations

import base64
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from lightcone_spec.experiments.failure_actuator import (
    ReleaseFailureActuatorCapability,
)
from lightcone_spec.experiments.failure_authority import (
    FAILURE_INJECTION_EXECUTION_LIFECYCLE_UNAVAILABLE_REASON,
    FAILURE_INJECTION_FIRST_PARTY_ACTUATOR_UNAVAILABLE_REASON,
    FailureExecutionAuthorityToken,
    FailureInjectionAuthorityBinding,
    FailureInjectionAuthorityBlocked,
    bind_failure_injection_authority,
    reduce_failure_actuation_receipt,
    release_failure_plan_for_cell,
    require_failure_execution_lifecycle,
    require_failure_injection_authority,
    revalidate_failure_execution_authority,
    revalidate_failure_injection_authority,
)
from lightcone_spec.experiments.registry import (
    E5_FAILURES,
)
from lightcone_spec.experiments.registry import (
    build_legacy_industrial_registry as build_industrial_registry,
)
from lightcone_spec.runtime.attestation import TrustedAttesterPolicy


def _registry_and_cell():
    registry = build_industrial_registry()
    cell = next(
        row
        for row in registry.cells_for("E5")
        if row.identity.task == "failure_injection"
        and row.identity.arrival == "failure:communicator_failure"
    )
    return registry, cell


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _bound_plan(tmp_path: Path):
    registry, cell = _registry_and_cell()
    plan = release_failure_plan_for_cell(registry, cell)
    path = _write_json(tmp_path / "failure-plan.json", plan.to_dict())
    binding = bind_failure_injection_authority(path, registry=registry)
    return registry, plan, path, binding


def _release_capability() -> ReleaseFailureActuatorCapability:
    def release_failure_actuator_factory():
        return object()

    release_failure_actuator_factory.__module__ = (
        "lightcone_spec.experiments.failure_actuator"
    )
    release_failure_actuator_factory.__qualname__ = "release_failure_actuator_factory"
    return ReleaseFailureActuatorCapability(
        actuator_id="release.first_party_fault_actuator.v1",
        actuator_version_sha256="1" * 64,
        factory_module=release_failure_actuator_factory.__module__,
        factory_qualname=release_failure_actuator_factory.__qualname__,
        factory=release_failure_actuator_factory,
    )


def _release_policy() -> TrustedAttesterPolicy:
    public_key = b"failure-release-public-key-32b"[:32].ljust(32, b"-")
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    return TrustedAttesterPolicy(
        policy_id="release-failure-attester-policy",
        trusted_attesters=(
            ("release-failure-attester", "release-failure-key", public_key_sha256),
        ),
        public_keys=(
            (public_key_sha256, base64.b64encode(public_key).decode("ascii")),
        ),
    )


def _caller_token(plan, binding) -> FailureExecutionAuthorityToken:
    return FailureExecutionAuthorityToken(
        binding=binding,
        authority_sha256=binding.sha256,
        plan_sha256=plan.sha256,
        registry_sha256=plan.registry_sha256,
        cell_id=plan.cell_id,
        scenario=plan.scenario,
        actuator_id="caller.forged_actuator",
        actuator_version_sha256="6" * 64,
        actuator_capability_sha256="7" * 64,
    )


def _receipt(plan, binding) -> dict[str, object]:
    counter_names = [row.name for row in plan.expected_counters]
    trigger = next(row.name for row in plan.expected_counters if row.comparison == "ge")
    topologies = []
    for topology in plan.topology_targets:
        ranks = []
        for rank in topology.target_ranks:
            counters = {name: 0 for name in counter_names}
            if rank == 0:
                counters[trigger] = 1
            ranks.append(
                {
                    "rank": rank,
                    "process_id": 1000 + rank,
                    "process_start_monotonic_ns": 1,
                    "session_epoch": 0,
                    "phases": [
                        {"phase": "arm", "monotonic_ns": 10},
                        {"phase": "trigger", "monotonic_ns": 20},
                        {"phase": "recover", "monotonic_ns": 30},
                        {"phase": "terminal", "monotonic_ns": 40},
                    ],
                    "counters": counters,
                }
            )
        topologies.append({"topology": topology.topology, "rank_receipts": ranks})
    return {
        "schema_version": 1,
        "kind": "e5_atomic_failure_actuation_receipt",
        "plan_sha256": plan.sha256,
        "authority_sha256": binding.sha256,
        "run_id": "release-run-1",
        "run_nonce_sha256": "1" * 64,
        "actuator_id": "untrusted-actuator",
        "actuator_version_sha256": "2" * 64,
        "fresh_process": True,
        "topologies": topologies,
        "terminal_status": "RECOVERED",
        "committed": True,
        "attester_id": "untrusted-attester",
        "trust_domain": "hardware",
        "signature_hex": "0" * 128,
    }


def test_release_plan_is_registry_derived_and_all_rank() -> None:
    registry, cell = _registry_and_cell()
    plan = release_failure_plan_for_cell(registry, cell)
    failure_cells = tuple(
        row
        for row in registry.cells_for("E5")
        if row.identity.task == "failure_injection"
    )

    assert len(failure_cells) == 264
    assert all(
        {
            row.identity.arrival.removeprefix("failure:")
            for row in failure_cells
            if row.identity.block == block
        }
        == set(E5_FAILURES)
        for block in range(24)
    )
    assert plan.cell_id == cell.cell_id
    assert plan.scenario == "communicator_failure"
    assert tuple(row.topology for row in plan.topology_targets) == (
        "tp1_dp1",
        "tp2_dp1",
        "tp1_dp2",
    )
    assert tuple(row.target_ranks for row in plan.topology_targets) == (
        (0,),
        (0, 1),
        (0, 1),
    )
    assert plan.lifecycle.fresh_process_required
    assert plan.correctness_only


def test_bind_revalidates_raw_plan_and_blocks_before_execution(tmp_path: Path) -> None:
    registry, plan, _, binding = _bound_plan(tmp_path)

    replay = revalidate_failure_injection_authority(binding, registry=registry)
    assert replay.plan == plan
    token = require_failure_injection_authority(binding, registry=registry)
    assert token.plan_sha256 == plan.sha256
    assert token.registry_sha256 == registry.sha256


def test_failure_binding_wire_round_trip_is_path_and_content_bound(
    tmp_path: Path,
) -> None:
    registry, _, plan_path, binding = _bound_plan(tmp_path)

    restored = FailureInjectionAuthorityBinding.from_dict(binding.to_dict())

    assert restored == binding
    assert restored.plan_path == str(plan_path.resolve())
    assert (
        revalidate_failure_injection_authority(
            restored,
            registry=registry,
        ).binding.sha256
        == binding.sha256
    )
    foreign = binding.to_dict()
    foreign["cell_id"] = "0" * 64
    with pytest.raises(ValueError, match="fresh raw replay"):
        revalidate_failure_injection_authority(
            FailureInjectionAuthorityBinding.from_dict(foreign),
            registry=registry,
        )

    boolean_schema = binding.to_dict()
    boolean_schema["schema_version"] = True
    with pytest.raises(ValueError, match="schema is unsupported"):
        FailureInjectionAuthorityBinding.from_dict(boolean_schema)


def test_allowlist_token_cannot_bypass_missing_execution_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import lightcone_spec.experiments.failure_actuator as actuator_module
    import lightcone_spec.experiments.failure_authority as authority_module
    import lightcone_spec.runtime.attestation as attestation_module

    registry, plan, _, binding = _bound_plan(tmp_path)
    capability = _release_capability()
    policy = _release_policy()
    monkeypatch.setattr(
        actuator_module,
        "RELEASE_FAILURE_ACTUATOR_CAPABILITIES",
        (capability,),
    )
    monkeypatch.setattr(authority_module, "RELEASE_TRUSTED_ATTESTER_POLICY", policy)
    monkeypatch.setattr(attestation_module, "RELEASE_TRUSTED_ATTESTER_POLICY", policy)
    token = require_failure_injection_authority(binding, registry=registry)

    with pytest.raises(
        FailureInjectionAuthorityBlocked,
        match=FAILURE_INJECTION_EXECUTION_LIFECYCLE_UNAVAILABLE_REASON,
    ):
        require_failure_execution_lifecycle(
            token,
            cell=next(
                row for row in registry.cells_for("E5") if row.cell_id == plan.cell_id
            ),
            expected_registry_sha256=registry.sha256,
        )


def test_public_token_constructor_cannot_replace_source_capability(
    tmp_path: Path,
) -> None:
    import lightcone_spec.experiments.failure_authority as authority_module

    assert not hasattr(authority_module, "RELEASE_FAILURE_ACTUATORS")
    registry, plan, _, binding = _bound_plan(tmp_path)
    cell = next(row for row in registry.cells_for("E5") if row.cell_id == plan.cell_id)
    with pytest.raises(TypeError, match="first-party validation"):
        _caller_token(plan, binding)
    token = require_failure_injection_authority(binding, registry=registry)
    assert (
        revalidate_failure_execution_authority(
            token,
            cell=cell,
            expected_registry_sha256=registry.sha256,
        ).plan
        == plan
    )


def test_joint_raw_plan_and_token_rehash_cannot_change_runtime_cell(
    tmp_path: Path,
) -> None:
    registry, original_plan, plan_path, _ = _bound_plan(tmp_path)
    original_cell = next(
        row for row in registry.cells_for("E5") if row.cell_id == original_plan.cell_id
    )
    foreign_cell = next(
        row
        for row in registry.cells_for("E5")
        if row.identity.arrival == "failure:queue_saturation"
        and row.identity.block == original_cell.identity.block
    )
    foreign_plan = release_failure_plan_for_cell(registry, foreign_cell)
    _write_json(plan_path, foreign_plan.to_dict())
    foreign_binding = bind_failure_injection_authority(plan_path, registry=registry)
    foreign_token = require_failure_injection_authority(
        foreign_binding, registry=registry
    )

    with pytest.raises(ValueError, match="differs from its runtime cell"):
        revalidate_failure_execution_authority(
            foreign_token,
            cell=original_cell,
            expected_registry_sha256=registry.sha256,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(scenario="failure"), "vague or unregistered"),
        (
            lambda row: row["topology_targets"][0].update(target_ranks=[1]),
            "every rank",
        ),
        (
            lambda row: row["topology_targets"].pop(),
            "all three registered topologies",
        ),
        (
            lambda row: row.update(topology_failure_surface="f" * 64),
            "not raw authority",
        ),
    ],
)
def test_bind_rejects_caller_scenario_topology_rank_and_summary(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    registry, cell = _registry_and_cell()
    row = release_failure_plan_for_cell(registry, cell).to_dict()
    mutation(row)
    path = _write_json(tmp_path / "foreign-plan.json", row)

    with pytest.raises(ValueError, match=message):
        bind_failure_injection_authority(path, registry=registry)


def test_bind_rejects_symlink_and_revalidation_rejects_rehash(
    tmp_path: Path,
) -> None:
    registry, cell = _registry_and_cell()
    plan = release_failure_plan_for_cell(registry, cell)
    source = _write_json(tmp_path / "source.json", plan.to_dict())
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="resolved and non-symlink"):
        bind_failure_injection_authority(link, registry=registry)

    hardlink = tmp_path / "hardlink.json"
    os.link(source, hardlink)
    with pytest.raises(ValueError, match="single-link regular file"):
        bind_failure_injection_authority(hardlink, registry=registry)
    hardlink.unlink()

    binding = bind_failure_injection_authority(source, registry=registry)
    source.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="fresh raw replay"):
        revalidate_failure_injection_authority(binding, registry=registry)


def test_raw_receipt_reducer_is_structural_but_cannot_publish_surface(
    tmp_path: Path,
) -> None:
    registry, plan, _, binding = _bound_plan(tmp_path)
    receipt_path = _write_json(tmp_path / "receipt.json", _receipt(plan, binding))

    reduction = reduce_failure_actuation_receipt(
        binding, receipt_path, registry=registry
    )
    assert reduction.status == "BLOCKED"
    assert reduction.reason == FAILURE_INJECTION_FIRST_PARTY_ACTUATOR_UNAVAILABLE_REASON
    assert reduction.topology_failure_surface_sha256 is None


def test_raw_receipt_rejects_partial_rank_and_nonatomic_terminal(
    tmp_path: Path,
) -> None:
    registry, plan, _, binding = _bound_plan(tmp_path)
    partial = _receipt(plan, binding)
    partial["topologies"][1]["rank_receipts"].pop()  # type: ignore[index]
    partial_path = _write_json(tmp_path / "partial.json", partial)
    with pytest.raises(ValueError, match="partial, duplicate, or foreign ranks"):
        reduce_failure_actuation_receipt(binding, partial_path, registry=registry)

    nonterminal = _receipt(plan, binding)
    nonterminal["committed"] = False
    nonterminal_path = _write_json(tmp_path / "nonterminal.json", nonterminal)
    with pytest.raises(ValueError, match="not atomically terminal"):
        reduce_failure_actuation_receipt(binding, nonterminal_path, registry=registry)


def test_raw_receipt_rejects_counter_and_lifecycle_fabrication(
    tmp_path: Path,
) -> None:
    registry, plan, _, binding = _bound_plan(tmp_path)
    fabricated = _receipt(plan, binding)
    first_rank = fabricated["topologies"][0]["rank_receipts"][0]  # type: ignore[index]
    first_rank["counters"]["exactness_violations"] = 1
    path = _write_json(tmp_path / "fabricated.json", fabricated)
    with pytest.raises(ValueError, match="exactness_violations differs"):
        reduce_failure_actuation_receipt(binding, path, registry=registry)

    unordered = deepcopy(_receipt(plan, binding))
    phases = unordered["topologies"][0]["rank_receipts"][0]["phases"]  # type: ignore[index]
    phases[1]["phase"] = "recover"
    path = _write_json(tmp_path / "unordered.json", unordered)
    with pytest.raises(ValueError, match="incomplete or unordered"):
        reduce_failure_actuation_receipt(binding, path, registry=registry)

    expired = deepcopy(_receipt(plan, binding))
    phases = expired["topologies"][0]["rank_receipts"][0]["phases"]  # type: ignore[index]
    phases[1]["monotonic_ns"] = 30_000 * 1_000_000 + 11
    phases[2]["monotonic_ns"] = 30_000 * 1_000_000 + 12
    phases[3]["monotonic_ns"] = 30_000 * 1_000_000 + 13
    path = _write_json(tmp_path / "expired.json", expired)
    with pytest.raises(ValueError, match="exceeds its release lifecycle window"):
        reduce_failure_actuation_receipt(binding, path, registry=registry)

    resumed = deepcopy(_receipt(plan, binding))
    resumed["topologies"][0]["rank_receipts"][0]["session_epoch"] = 1  # type: ignore[index]
    path = _write_json(tmp_path / "resumed.json", resumed)
    with pytest.raises(ValueError, match="session epoch zero"):
        reduce_failure_actuation_receipt(binding, path, registry=registry)
