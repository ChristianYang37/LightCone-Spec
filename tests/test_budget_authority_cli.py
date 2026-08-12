from __future__ import annotations

import json
from pathlib import Path

from test_execution_bundle import _bundle_fixture

from lightcone_spec.cli.main import main
from lightcone_spec.experiments.budget_authority import (
    replay_budget_activation_authority,
    revalidate_budget_materialization_authority_binding,
)
from lightcone_spec.experiments.capacity_authority import bind_capacity_authority
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.planning_artifacts import (
    budget_materialization_authority_binding_from_dict,
    budget_plan_from_dict,
)


def test_bind_budget_authority_cli_publishes_replayable_raw_closure(
    tmp_path: Path,
) -> None:
    _, bundle = _bundle_fixture(tmp_path.resolve())
    output = tmp_path / "budget-authority.json"
    argv = [
        "bind-industrial-budget-authority",
        "--activation-manifest",
        bundle.activation.path,
        "--budget-policy",
        bundle.budget_policy.path,
        "--capacity-envelope",
        bundle.capacity_envelope.path,
        "--capacity-manifest",
        bundle.capacity_source_manifest.path,
        "--capacity-verification-receipt",
        bundle.capacity_verification_receipt.path,
        "--budget-plan",
        bundle.budget_plan.path,
        "--output",
        str(output),
    ]
    for source in bundle.budget_load_bindings:
        argv.extend(("--budget-load-binding", source.path))

    assert main(argv) == 0
    authority = budget_materialization_authority_binding_from_dict(
        json.loads(output.read_text(encoding="utf-8"))
    )
    capacity = bind_capacity_authority(
        bundle.capacity_source_manifest.path,
        bundle.capacity_verification_receipt.path,
    )
    assert authority.capacity_authority == capacity
    registry = replay_budget_activation_authority(authority.activation).registry
    inventory = GpuInventory.from_dict(bundle.inventory.load())
    plan = budget_plan_from_dict(bundle.budget_plan.load())
    replay = revalidate_budget_materialization_authority_binding(
        authority,
        expected_registry=registry,
        expected_inventory=plan.inventory,
        expected_plan=plan,
    )
    assert replay.budget_plan == plan
    assert inventory.sha256 == bundle.inventory.semantic_sha256
