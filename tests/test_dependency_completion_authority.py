from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_execution_bundle import _bundle_fixture

from lightcone_spec.experiments.budget_authority import (
    DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON,
    BudgetMaterializationBlockedError,
    bind_registry_stage_activation_authority,
    replay_registry_stage_activation_authority,
    require_ready_registry_stage_dependency_completions,
)
from lightcone_spec.experiments.completion_authority import DurableJsonArtifactBinding
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.planning_artifacts import (
    registry_stage_activation_authority_binding_from_dict,
    registry_stage_activation_authority_binding_to_dict,
)
from lightcone_spec.experiments.registry import ExperimentRegistry, content_sha256
from lightcone_spec.experiments.selection_authority import (
    E1_COMMON_LOAD_AUTHORITY_UNREGISTERED_REASON,
    E3A_LOCKED_OUTPUT_REDUCTION_UNREGISTERED_REASON,
    E3A_SELECTION_POLICY_UNREGISTERED_REASON,
)


def _write_bound(path: Path, value: object) -> Path:
    path = path.resolve()
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(f"{path}.sha256").write_text(
        f"{content_sha256(value)}\n",
        encoding="ascii",
    )
    return path


def test_completed_json_duplicate_keys_are_never_authority(tmp_path: Path) -> None:
    path = (tmp_path / "duplicate-completed.json").resolve()
    path.write_text('{"schema_version":4,"schema_version":4}\n', encoding="utf-8")
    Path(f"{path}.sha256").write_text(
        content_sha256({"schema_version": 4}) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        DurableJsonArtifactBinding.from_path(path)


@dataclass(frozen=True)
class _CompletionGraphFixture:
    root: Path
    manifest_path: Path
    prior_manifest_path: Path
    receipt_path: Path
    completed_path: Path
    inventory_path: Path
    inventory_source_path: Path
    output_path: Path
    registry: ExperimentRegistry
    inventory: GpuInventory
    binding: object


def _completion_spec(fixture: _CompletionGraphFixture) -> dict[str, object]:
    return {
        "receipt_artifact": str(fixture.receipt_path),
        "completed_cells_artifact": str(fixture.completed_path),
        "activation_manifest": str(fixture.prior_manifest_path),
        "inventory_artifact": str(fixture.inventory_path),
        "inventory_source_receipt": str(fixture.inventory_source_path),
        "locked_outputs": [
            {"name": "runtime_envelope", "artifact": str(fixture.output_path)}
        ],
    }


def _build_completion_graph(root: Path) -> _CompletionGraphFixture:
    root = root.resolve()
    _, bundle = _bundle_fixture(root)
    legacy = bind_registry_stage_activation_authority(bundle.activation.path)
    registry, _ = replay_registry_stage_activation_authority(legacy)
    inventory = GpuInventory.from_dict(bundle.inventory.load())
    output = next(
        artifact
        for artifact in bundle.dependency_artifacts
        if artifact.experiment == "preflight" and artifact.name == "runtime_envelope"
    )
    output_path = Path(output.source.path)
    completed = {
        "schema_version": 4,
        "kind": "industrial_completed_cells",
        "registry_sha256": registry.sha256,
        "experiment": "preflight",
        "runtime_sha256": output.source.canonical_sha256,
        "split_sha256": bundle.activation_split.canonical_sha256,
        "split_contract": {},
        "activation_binding": {},
        "inventory_sha256": inventory.sha256,
        "inventory_source_receipt_sha256": inventory.source_receipt_sha256,
        "rows": [],
    }
    completed_path = _write_bound(root / "preflight-completed-v4.json", completed)
    receipt = registry.make_receipt(
        "preflight",
        {"runtime_envelope": output.source.semantic_sha256},
        runtime_sha256=output.source.canonical_sha256,
        split_sha256=bundle.activation_split.canonical_sha256,
        completed_cells_sha256=content_sha256(completed),
    )
    receipt_path = _write_bound(
        Path(bundle.dependency_receipts[0].path), receipt.to_dict()
    )
    prior_manifest_path = _write_bound(
        root / "preflight-activation-manifest.json",
        {
            "schema_version": 2,
            "kind": "industrial_registry_stage_activation_manifest",
            "registry_artifact": bundle.registry.path,
            "experiment": "preflight",
            "runtime_artifact": output.source.path,
            "split_artifact": bundle.activation_split.path,
            "dependency_receipts": [],
            "dependency_completion_authorities": [],
        },
    )
    inventory_path = Path(bundle.inventory.path)
    inventory_source_path = Path(bundle.inventory_source_artifact.source.path)
    fixture = _CompletionGraphFixture(
        root=root,
        manifest_path=Path(bundle.activation.path),
        prior_manifest_path=prior_manifest_path,
        receipt_path=receipt_path,
        completed_path=completed_path,
        inventory_path=inventory_path,
        inventory_source_path=inventory_source_path,
        output_path=output_path,
        registry=registry,
        inventory=inventory,
        binding=None,
    )
    manifest = {
        "schema_version": 2,
        "kind": "industrial_registry_stage_activation_manifest",
        "registry_artifact": bundle.registry.path,
        "experiment": "E3a",
        "runtime_artifact": bundle.activation_runtime.path,
        "split_artifact": bundle.activation_split.path,
        "dependency_receipts": [str(receipt_path)],
        "dependency_completion_authorities": [_completion_spec(fixture)],
    }
    _write_bound(fixture.manifest_path, manifest)
    return _CompletionGraphFixture(
        **{
            **fixture.__dict__,
            "binding": bind_registry_stage_activation_authority(fixture.manifest_path),
        }
    )


@pytest.fixture(scope="module")
def completion_graph(
    tmp_path_factory: pytest.TempPathFactory,
) -> _CompletionGraphFixture:
    return _build_completion_graph(
        tmp_path_factory.mktemp("dependency-completion-authority")
    )


def _replace_bound(path: Path, value: object):
    sidecar = Path(f"{path}.sha256")
    old_body = path.read_bytes()
    old_sidecar = sidecar.read_bytes()
    _write_bound(path, value)

    def restore() -> None:
        path.write_bytes(old_body)
        sidecar.write_bytes(old_sidecar)

    return restore


def test_recursive_completion_graph_replays_but_cannot_mint_authority(
    completion_graph: _CompletionGraphFixture,
) -> None:
    fixture = completion_graph
    registry, activation = replay_registry_stage_activation_authority(fixture.binding)
    assert registry == fixture.registry
    assert activation.experiment == "E3a"
    assert len(fixture.binding.dependency_completion_authorities) == 1
    completion = fixture.binding.dependency_completion_authorities[0]
    assert completion.receipt == fixture.binding.dependency_receipts[0]
    assert completion.inventory_authority.inventory_sha256 == fixture.inventory.sha256
    wire = registry_stage_activation_authority_binding_to_dict(fixture.binding)
    assert registry_stage_activation_authority_binding_from_dict(wire) == (
        fixture.binding
    )

    with pytest.raises(ValueError, match="activation binding is missing or forged"):
        require_ready_registry_stage_dependency_completions(
            fixture.binding,
            expected_registry=fixture.registry,
            expected_gpu_inventory=fixture.inventory,
        )


def test_forged_receipt_and_swapped_completion_are_rejected(
    completion_graph: _CompletionGraphFixture,
) -> None:
    fixture = completion_graph
    receipt = json.loads(fixture.receipt_path.read_text(encoding="utf-8"))
    forged = copy.deepcopy(receipt)
    forged["outputs"][0]["content_sha256"] = content_sha256("forged-output")
    restore = _replace_bound(fixture.receipt_path, forged)
    try:
        with pytest.raises(ValueError, match="lineage, or outputs"):
            bind_registry_stage_activation_authority(fixture.manifest_path)
    finally:
        restore()

    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    manifest["dependency_completion_authorities"][0]["activation_manifest"] = str(
        fixture.manifest_path
    )
    restore = _replace_bound(fixture.manifest_path, manifest)
    try:
        with pytest.raises(ValueError, match="contain a cycle"):
            bind_registry_stage_activation_authority(fixture.manifest_path)
    finally:
        restore()

    swapped_path = _write_bound(
        fixture.root / "swapped-completed-v4.json",
        {"schema_version": 4, "kind": "industrial_completed_cells", "rows": []},
    )
    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    manifest["dependency_completion_authorities"][0]["completed_cells_artifact"] = str(
        swapped_path
    )
    restore = _replace_bound(fixture.manifest_path, manifest)
    try:
        with pytest.raises(ValueError, match="lineage, or outputs"):
            bind_registry_stage_activation_authority(fixture.manifest_path)
    finally:
        restore()


def test_sidecar_wrong_inventory_and_activation_cycle_fail_closed(
    completion_graph: _CompletionGraphFixture,
) -> None:
    fixture = completion_graph
    completed_sidecar = Path(f"{fixture.completed_path}.sha256")
    old_sidecar = completed_sidecar.read_bytes()
    completed_sidecar.write_text("0" * 64 + "\n", encoding="ascii")
    try:
        with pytest.raises(ValueError, match="sidecar"):
            replay_registry_stage_activation_authority(fixture.binding)
    finally:
        completed_sidecar.write_bytes(old_sidecar)

    inventory = json.loads(fixture.inventory_path.read_text(encoding="utf-8"))
    inventory["devices"][0]["memory_bytes"] += 1
    wrong_inventory_path = _write_bound(
        fixture.root / "wrong-inventory.json", inventory
    )
    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    manifest["dependency_completion_authorities"][0]["inventory_artifact"] = str(
        wrong_inventory_path
    )
    restore = _replace_bound(fixture.manifest_path, manifest)
    try:
        binding = bind_registry_stage_activation_authority(fixture.manifest_path)
        with pytest.raises(ValueError, match="inventory was swapped"):
            require_ready_registry_stage_dependency_completions(
                binding,
                expected_registry=fixture.registry,
                expected_gpu_inventory=fixture.inventory,
            )
    finally:
        restore()


def test_bare_receipt_prefix_has_named_missing_prior_authority_block(
    completion_graph: _CompletionGraphFixture,
) -> None:
    fixture = completion_graph
    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    manifest.pop("dependency_completion_authorities")
    manifest["schema_version"] = 1
    legacy_path = _write_bound(fixture.root / "legacy-bare-prefix.json", manifest)
    binding = bind_registry_stage_activation_authority(legacy_path)

    with pytest.raises(BudgetMaterializationBlockedError) as blocked:
        require_ready_registry_stage_dependency_completions(
            binding,
            expected_registry=fixture.registry,
            expected_gpu_inventory=fixture.inventory,
        )
    assert blocked.value.reason_code == (
        DEPENDENCY_COMPLETION_MANIFEST_AUTHORITY_MISSING_REASON
    )


def test_e3a_dependency_completion_calls_typed_locked_output_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_selection_authority import _e3a_selection_and_receipt

    from lightcone_spec.experiments import budget_authority

    registry = budget_authority.build_industrial_registry()
    _, receipt = _e3a_selection_and_receipt(registry)
    receipt_source = SimpleNamespace(role="activation_dependency_receipt")
    completed_source = SimpleNamespace(
        role="dependency_completed_cells",
        semantic_sha256=receipt.completed_cells_sha256,
    )
    inventory = object()
    prior_receipt = object()
    prior = SimpleNamespace(
        binding=object(),
        receipt=prior_receipt,
        authority=SimpleNamespace(inventory=inventory),
    )
    replay = SimpleNamespace(
        binding=object(),
        registry=registry,
        dependency_records=(prior,),
        dependency_receipts=(prior_receipt,),
        experiment="E3a",
        runtime_sha256=receipt.runtime_sha256,
        split_sha256=receipt.split_sha256,
        family_activations=(),
    )

    bound_roles: list[str] = []

    def bind_source(_path: object, *, role: str):
        bound_roles.append(role)
        return (
            receipt_source
            if role == "activation_dependency_receipt"
            else completed_source
        )

    monkeypatch.setattr(budget_authority, "bind_budget_raw_json", bind_source)
    monkeypatch.setattr(budget_authority, "load_budget_raw_json", lambda _row: {})
    monkeypatch.setattr(budget_authority, "_receipt_from_value", lambda _row: receipt)
    monkeypatch.setattr(
        budget_authority,
        "_bind_stage_activation_authority",
        lambda *_args, **_kwargs: replay,
    )
    monkeypatch.setattr(
        budget_authority,
        "_dependency_inventory_authority",
        lambda **_kwargs: (object(), inventory),
    )
    locked_outputs = [
        {"name": name, "artifact": f"/{name}.json"}
        for name in sorted(registry.definition("E3a").locked_outputs)
    ]
    spec = {
        "receipt_artifact": "/e3a-receipt.json",
        "completed_cells_artifact": "/e3a-completed.json",
        "activation_manifest": "/e3a-activation.json",
        "inventory_artifact": "/inventory.json",
        "inventory_source_receipt": "/inventory-source.json",
        "locked_outputs": locked_outputs,
    }
    with pytest.raises(BudgetMaterializationBlockedError) as blocked:
        budget_authority._bind_dependency_completion(
            spec,
            expected_receipt_source=receipt_source,
            expected_registry=registry,
            earlier_records=None,
            manifest_stack=(),
        )
    assert blocked.value.reason_code == E3A_SELECTION_POLICY_UNREGISTERED_REASON
    assert "dependency_locked_output" not in bound_roles

    duplicate = copy.deepcopy(spec)
    duplicate["locked_outputs"].append(copy.deepcopy(locked_outputs[0]))
    with pytest.raises(ValueError, match="duplicated"):
        budget_authority._bind_dependency_completion(
            duplicate,
            expected_receipt_source=receipt_source,
            expected_registry=registry,
            earlier_records=None,
            manifest_stack=(),
        )
    assert "dependency_locked_output" not in bound_roles

    from lightcone_spec.experiments import selection_authority

    monkeypatch.setattr(
        selection_authority,
        "RELEASE_E3A_SCIENTIFIC_SELECTION_POLICY",
        selection_authority.E3aScientificSelectionPolicy(
            schema_version=1,
            source_authority="test-source-policy",
            source_authority_sha256=content_sha256("test-source-policy"),
            primary_contexts=(4096,),
            reference_load_goodput_fraction=0.5,
            reference_load_statistic="maximum_width_median_static_goodput",
            reference_load_choice=("smallest_concurrency_reaching_registered_fraction"),
            width_primary_objective="maximum_worst_static_target_goodput_ratio",
            width_secondary_objective="maximum_median_static_goodput",
            width_final_tiebreak="smallest_width",
            locked_output_reducer_protocol_sha256=content_sha256(
                "test-locked-output-reducer"
            ),
        ),
    )
    with pytest.raises(BudgetMaterializationBlockedError) as blocked:
        budget_authority._bind_dependency_completion(
            spec,
            expected_receipt_source=receipt_source,
            expected_registry=registry,
            earlier_records=None,
            manifest_stack=(),
        )
    assert blocked.value.reason_code == (
        E3A_LOCKED_OUTPUT_REDUCTION_UNREGISTERED_REASON
    )
    assert "dependency_locked_output" not in bound_roles


def test_e1_dependency_completion_blocks_before_opening_opaque_common_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import budget_authority

    registry = budget_authority.build_industrial_registry()
    runtime_sha256 = content_sha256("e1-completion-runtime")
    split_sha256 = content_sha256("e1-completion-split")
    receipt = SimpleNamespace(
        experiment="E1",
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        completed_cells_sha256=content_sha256("e1-completed"),
    )
    receipt_source = SimpleNamespace(role="activation_dependency_receipt")
    completed_source = SimpleNamespace(
        role="dependency_completed_cells",
        semantic_sha256=receipt.completed_cells_sha256,
    )
    inventory = object()
    prior_receipts = (object(), object())
    prior_records = tuple(
        SimpleNamespace(
            binding=object(),
            receipt=prior_receipt,
            authority=SimpleNamespace(inventory=inventory),
        )
        for prior_receipt in prior_receipts
    )
    replay = SimpleNamespace(
        binding=object(),
        registry=registry,
        dependency_records=prior_records,
        dependency_receipts=prior_receipts,
        experiment="E1",
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        family_activations=(),
    )
    bound_roles: list[str] = []

    def bind_source(_path: object, *, role: str):
        bound_roles.append(role)
        return (
            receipt_source
            if role == "activation_dependency_receipt"
            else completed_source
        )

    monkeypatch.setattr(budget_authority, "bind_budget_raw_json", bind_source)
    monkeypatch.setattr(budget_authority, "load_budget_raw_json", lambda _row: {})
    monkeypatch.setattr(budget_authority, "_receipt_from_value", lambda _row: receipt)
    monkeypatch.setattr(
        budget_authority,
        "_bind_stage_activation_authority",
        lambda *_args, **_kwargs: replay,
    )
    monkeypatch.setattr(
        budget_authority,
        "_dependency_inventory_authority",
        lambda **_kwargs: (object(), inventory),
    )
    locked_outputs = [
        {"name": name, "artifact": f"/{name}.json"}
        for name in sorted(registry.definition("E1").locked_outputs)
    ]
    spec = {
        "receipt_artifact": "/e1-receipt.json",
        "completed_cells_artifact": "/e1-completed.json",
        "activation_manifest": "/e1-activation.json",
        "inventory_artifact": "/inventory.json",
        "inventory_source_receipt": "/inventory-source.json",
        "locked_outputs": locked_outputs,
    }
    with pytest.raises(BudgetMaterializationBlockedError) as blocked:
        budget_authority._bind_dependency_completion(
            spec,
            expected_receipt_source=receipt_source,
            expected_registry=registry,
            earlier_records=None,
            manifest_stack=(),
        )
    assert blocked.value.reason_code == E1_COMMON_LOAD_AUTHORITY_UNREGISTERED_REASON
    assert "dependency_locked_output" not in bound_roles
