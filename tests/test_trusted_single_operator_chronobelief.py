from __future__ import annotations

import hashlib
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.config import ModelPair, RunConfig, RuntimeConfig
from lightcone_spec.experiments import formal_single_operator_chronobelief as bridge
from lightcone_spec.experiments import (
    formal_single_operator_prepared_launch_producer as producer,
)
from lightcone_spec.experiments.formal_single_operator_chronobelief import (
    TRUSTED_SINGLE_OPERATOR_CHRONOBELIEF_GPU_PARITY_PROTOCOL_SHA256,
    TrustedSingleOperatorChronoBeliefGpuParityProof,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.stage_materialization import (
    E1Geometry,
    E2CandidateRecipe,
    MaterializedCell,
    default_e2_recipe_grid_authority,
)
from lightcone_spec.orchestration import formal_physical_dispatch as physical
from lightcone_spec.orchestration import live_sglang
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.readiness import (
    NATIVE_RUNTIME_QUALIFICATION_TESTS,
    NativeReadinessBlocked,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _binding(tmp_path: Path, name: str) -> CanonicalJsonProofBinding:
    path = tmp_path / f"{name}.json"
    publish_canonical_json_no_replace(
        path,
        {"schema_version": 1, "kind": "test_source", "name": name},
    )
    return CanonicalJsonProofBinding.bind(path)


def _proof(
    tmp_path: Path,
) -> tuple[
    TrustedSingleOperatorChronoBeliefGpuParityProof,
    bridge._TrustedChronoBeliefEvidenceJoin,
]:
    bindings = {
        name: _binding(tmp_path, name)
        for name in (
            "execution_source",
            "protocol_lock",
            "preflight_actual",
            "preflight_coverage",
            "exactness_result_pointer",
            "chronobelief_result_pointer",
            "chronobelief_proof_artifact",
            "dspark_result_pointer",
            "dspark_proof_artifact",
            "prerequisite_launch",
        )
    }
    join_values: dict[str, object] = {
        field.name: _sha(field.name)
        for field in fields(bridge._TrustedChronoBeliefEvidenceJoin)
        if field.name.endswith("sha256")
    }
    join_values.update(
        {
            "exactness_gpu_uuids": ("GPU-0", "GPU-1"),
            "qualified_gpu_uuids": ("GPU-1",),
            "dspark_qualification_gpu_uuids": ("GPU-0",),
            "gpu_model": "RTX-PRO-6000",
            "driver_version": "580.95.05",
            "cuda_version": "13.0",
            "patched_sglang_commit": "c" * 40,
            "patched_sglang_tree": "d" * 40,
            "dtype": "bfloat16",
            "dtype_parity_test_names": NATIVE_RUNTIME_QUALIFICATION_TESTS[
                "chronobelief_gpu_parity"
            ],
        }
    )
    join = bridge._TrustedChronoBeliefEvidenceJoin(**join_values)  # type: ignore[arg-type]
    proof = TrustedSingleOperatorChronoBeliefGpuParityProof(
        schema_version=1,
        kind="trusted_single_operator_chronobelief_gpu_parity_proof",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_CHRONOBELIEF_GPU_PARITY_PROTOCOL_SHA256
        ),
        trust_mode="trusted_single_operator_empirical_no_signature",
        formal_execution_authorized=False,
        **bindings,
        **join_values,
    )  # type: ignore[arg-type]
    return proof, join


def _inventory() -> GpuInventory:
    uuids = ("GPU-0", "GPU-1")
    return GpuInventory(
        schema_version=1,
        devices=tuple(
            GpuDevice(
                uuid=uuid,
                host_id="host",
                model="RTX-PRO-6000",
                memory_bytes=96 * 1024**3,
                compute_capability=(12, 0),
                pci_bus_id=f"0000:{index + 1:02x}:00.0",
                pci_root="root",
                numa_node=0,
                interconnects=("NVLink",),
                peer_access_class="NVLink",
                clock_policy="default",
                power_limit_watts=600.0,
                thermal_limit_celsius=83.0,
                availability=GpuAvailability.READY,
                reserved_processes=(),
                allowed_topology_groups=("pair",),
            )
            for index, uuid in enumerate(uuids)
        ),
        topology_groups=(
            GpuTopologyGroup(
                group_id="pair",
                host_id="host",
                gpu_uuids=uuids,
                fabric="NVLink",
                bandwidth_class="high",
            ),
        ),
        source_receipt_sha256=_sha("inventory"),
    )


def _e1a_cell(scope: str, verification_mode: str) -> MaterializedCell:
    return MaterializedCell(
        stage="E1a",
        method_role="LightCone-candidate",
        model="Qwen/Qwen3-8B",
        backend="DSPARK",
        task="LiveCodeBench_tuning_disjoint_from_E5",
        publication_policy="first_ready",
        recipe_sha256=_sha("chronobelief-winner"),
        dimensions=(
            ("fixed_verification_budget", 8),
            ("frozen_tts_recipe_sha256", _sha("tts")),
            ("parameterization", "full"),
            ("rank", "none"),
            ("scope", scope),
            ("verification_mode", verification_mode),
        ),
    )


def _adaptive_config(*, optimizer: str, proof_sha256: str | None) -> RunConfig:
    grid = default_e2_recipe_grid_authority()
    recipe = E2CandidateRecipe(
        geometry=E1Geometry("last1", "full", None, None),
        optimizer=optimizer,
        schedule="constant",
        learning_rate=grid.rates(optimizer=optimizer, parameterization="full")[0],
        optimizer_recipe_authority_sha256=(grid.optimizer_recipe_authority.sha256),
    )
    adaptation = grid.adaptation_config_for(
        recipe,
        canvas_tokens=8,
        adaptation_group_id=f"e1a:{optimizer}",
        chronobelief_gpu_proof_sha256=proof_sha256,
    )
    return RunConfig(
        method="l0",
        model=ModelPair(
            target_revision="1" * 40,
            drafter_revision="2" * 40,
            algorithm="DSPARK",
            draft_depth=7,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256=_sha("sampling"),
            device_identity="GPU-1",
            speculative_num_draft_tokens=8,
            max_running_requests=4,
        ),
        adaptation=adaptation,
    )


def test_empirical_proof_roundtrips_and_binds_actual_qualified_set(
    tmp_path: Path,
) -> None:
    proof, observed = _proof(tmp_path)

    bridge._validate_join(proof, observed)
    assert (
        TrustedSingleOperatorChronoBeliefGpuParityProof.from_dict(proof.to_dict())
        == proof
    )
    assert proof.qualified_gpu_uuids == ("GPU-1",)
    assert proof.trust_mode == "trusted_single_operator_empirical_no_signature"
    assert proof.formal_execution_authorized is False


@pytest.mark.parametrize(
    ("field_name", "foreign_value"),
    (
        ("qualified_gpu_uuids", ("GPU-0",)),
        ("gpu_model", "foreign-model"),
        ("driver_version", "foreign-driver"),
        ("cuda_version", "foreign-cuda"),
        ("patched_sglang_tree", "e" * 40),
        ("patch_manifest_sha256", _sha("foreign-patch-manifest")),
        ("patch_sha256", _sha("foreign-patch")),
        ("dtype", "float16"),
    ),
)
def test_deep_join_rejects_foreign_gpu_environment_or_dtype(
    tmp_path: Path,
    field_name: str,
    foreign_value: object,
) -> None:
    proof, observed = _proof(tmp_path)

    with pytest.raises(ValueError, match="deep evidence join differs"):
        bridge._validate_join(
            proof,
            replace(observed, **{field_name: foreign_value}),
        )


def test_proof_rejects_missing_or_unobserved_qualified_gpu(tmp_path: Path) -> None:
    proof, _observed = _proof(tmp_path)

    with pytest.raises(ValueError, match="qualified GPU set differs"):
        replace(proof, qualified_gpu_uuids=())
    with pytest.raises(ValueError, match="qualified GPU set differs"):
        replace(proof, qualified_gpu_uuids=("GPU-foreign",))


def test_proof_rejects_path_tamper(tmp_path: Path) -> None:
    proof, _observed = _proof(tmp_path)
    Path(proof.chronobelief_result_pointer.absolute_path).write_text(
        '{"kind":"tampered"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source changed"):
        TrustedSingleOperatorChronoBeliefGpuParityProof.from_dict(proof.to_dict())


def test_e1a_chronobelief_placement_uses_only_qualified_gpu_and_normal_balance() -> (
    None
):
    inventory = _inventory()
    cells = tuple(
        _e1a_cell(scope, verification_mode)
        for scope in ("last1", "last3", "last5", "all")
        for verification_mode in (
            "fixed_verification_budget",
            "native_scheduler",
        )
    )

    ordinary = {
        producer.deterministic_prepared_gpu_assignment(
            inventory=inventory,
            cell=cell,
        )
        for cell in cells
    }
    qualified = {
        producer._chronobelief_qualified_gpu_assignment(
            inventory=inventory,
            cell=cell,
            qualified_gpu_uuids=("GPU-1",),
        )
        for cell in cells
    }

    assert ordinary == {("GPU-0",), ("GPU-1",)}
    assert qualified == {("GPU-1",)}
    with pytest.raises(
        producer.FormalSingleOperatorPreparedLaunchBlocked,
        match="proved_gpu_assignment_missing",
    ):
        producer._chronobelief_qualified_gpu_assignment(
            inventory=inventory,
            cell=cells[0],
            qualified_gpu_uuids=("GPU-foreign",),
        )


def test_live_authority_accepts_qualified_member_and_rejects_missing_or_foreign(
    tmp_path: Path,
) -> None:
    proof, _observed = _proof(tmp_path)
    config = _adaptive_config(optimizer="chronobelief", proof_sha256=proof.sha256)

    assert live_sglang._chronobelief_execution_authority(
        config=config,
        verified_gpu_proof=None,
        expected_source_identity_sha256=None,
        inventory_sha256=proof.inventory_sha256,
        gpu_uuids=("GPU-1",),
        trusted_single_operator_proof=proof,
    ) == (proof.sha256, proof.source_identity_sha256)
    with pytest.raises(NativeReadinessBlocked, match="source_identity_unbound"):
        live_sglang._chronobelief_execution_authority(
            config=config,
            verified_gpu_proof=None,
            expected_source_identity_sha256=None,
            inventory_sha256=proof.inventory_sha256,
            gpu_uuids=("GPU-1",),
        )
    with pytest.raises(ValueError, match="differs from launch"):
        live_sglang._chronobelief_execution_authority(
            config=config,
            verified_gpu_proof=None,
            expected_source_identity_sha256=None,
            inventory_sha256=proof.inventory_sha256,
            gpu_uuids=("GPU-0",),
            trusted_single_operator_proof=proof,
        )


def test_non_chronobelief_optimizer_cannot_carry_empirical_proof(
    tmp_path: Path,
) -> None:
    proof, _observed = _proof(tmp_path)
    config = _adaptive_config(optimizer="adam", proof_sha256=None)

    with pytest.raises(ValueError, match="non-ChronoBelief path"):
        live_sglang._chronobelief_execution_authority(
            config=config,
            verified_gpu_proof=None,
            expected_source_identity_sha256=None,
            inventory_sha256=proof.inventory_sha256,
            gpu_uuids=("GPU-1",),
            trusted_single_operator_proof=proof,
        )


def test_physical_dispatch_deep_revalidates_bound_proof_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "prepared-inputs.json"
    publish_canonical_json_no_replace(
        source_path,
        {
            "schema_version": 2,
            "kind": "formal_single_operator_prepared_downstream_run_plan_inputs",
        },
    )
    source = CanonicalJsonProofBinding.bind(source_path)
    proof_binding = _binding(tmp_path, "parity-proof")
    launch = _binding(tmp_path, "prepared-launch")
    execution_source = _binding(tmp_path, "e1a-execution-source")
    inventory = _binding(tmp_path, "inventory")
    cell_id = _sha("cell")
    plan = SimpleNamespace(
        single_operator_execution_rebuild_source=source,
        launch_manifest=launch,
        materialized_cell_id=cell_id,
        inventory_sha256=inventory.semantic_sha256,
        gpu_uuids=("GPU-1",),
    )
    direct = SimpleNamespace(
        trusted_chronobelief_gpu_parity_proof=proof_binding,
        compile_launch_manifest=launch,
        materialized_cell_id=cell_id,
        inventory=inventory,
        execution_source=execution_source,
    )
    proof = SimpleNamespace(
        inventory_sha256=inventory.semantic_sha256,
        qualified_gpu_uuids=("GPU-1",),
    )
    calls = []

    def fake_revalidate_inputs(path: str, *, current_ns: int) -> object:
        assert path == source.absolute_path
        assert current_ns > 0
        return direct

    def fake_revalidate_proof(
        *, proof_path: str, execution_source_path: str, prepared_launch_path: str
    ) -> object:
        calls.append((proof_path, execution_source_path, prepared_launch_path))
        return proof

    from lightcone_spec.experiments import (
        formal_single_operator_chronobelief as bridge_module,
    )
    from lightcone_spec.experiments import (
        formal_single_operator_run_dispatch as dispatch_module,
    )

    monkeypatch.setattr(
        dispatch_module,
        "revalidate_formal_single_operator_prepared_downstream_run_plan_inputs",
        fake_revalidate_inputs,
    )
    monkeypatch.setattr(
        bridge_module,
        "revalidate_trusted_single_operator_chronobelief_for_prepared_launch",
        fake_revalidate_proof,
    )

    assert physical._trusted_single_operator_chronobelief_proof_from_plan(plan) is proof
    assert calls == [
        (
            proof_binding.absolute_path,
            execution_source.absolute_path,
            launch.absolute_path,
        )
    ]
    with pytest.raises(ValueError, match="differs from physical plan"):
        physical._trusted_single_operator_chronobelief_proof_from_plan(
            SimpleNamespace(**{**vars(plan), "gpu_uuids": ("GPU-0",)})
        )
