from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lightcone_spec.cli.main import _industrial_registry_artifact
from lightcone_spec.cli.main import main as cli_main
from lightcone_spec.experiments.capacity_authority import (
    CapacityAuthorityUnavailableError,
    bind_capacity_authority,
    build_capacity_source_manifest,
    build_capacity_verification_payload,
    capacity_source_receipt_sha256_from_paths,
    capacity_verification_receipt_template,
    revalidate_capacity_authority_binding,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.planning import (
    BUDGET_MATERIALIZATION_PROTOCOL_SHA256,
    CAPACITY_MAXIMUM_SOURCE_AGE_NS,
    CELL_CAPACITY_SIZING_PROTOCOL_SHA256,
    ZERO_MILLISECONDS,
    BudgetDisposition,
    BudgetDispositionStatus,
    BudgetJobKind,
    BudgetJobPolicy,
    BudgetPlan,
    BudgetPolicy,
    CapacityEnvelope,
    CellCapacityRequirement,
    ExpectedMaximumCount,
    ExperimentBudget,
    P99AnchorStatus,
    ScenarioMilliseconds,
    budget_inventory_identity_from_gpu_inventory,
)
from lightcone_spec.experiments.planning_artifacts import (
    budget_plan_from_dict,
    budget_plan_to_dict,
    capacity_authority_binding_from_dict,
    capacity_authority_binding_to_dict,
    capacity_envelope_to_dict,
)
from lightcone_spec.experiments.registry import (
    ExperimentRegistry,
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.stage_capacity import (
    StageCapacityGate,
    StageCapacityRetryBinding,
    StageCapacitySchedule,
    StageCapacityWaveBinding,
    materialize_stage_capacity_gate_from_raw_sources,
    revalidate_stage_capacity_gate_sources,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    attestation_message,
)


def _write_bound(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Path(f"{path}.sha256").write_text(content_sha256(value) + "\n", encoding="utf-8")
    return path.resolve()


def _scenario(value: int) -> ScenarioMilliseconds:
    return ScenarioMilliseconds(value, value, value)


def _policy() -> BudgetPolicy:
    rows = []
    for job_kind in sorted(BudgetJobKind, key=lambda row: row.value):
        rows.append(
            BudgetJobPolicy(
                job_kind=job_kind,
                startup_model_load=_scenario(10),
                compile_jit_graph_prewarm=(
                    _scenario(20)
                    if job_kind is BudgetJobKind.COMPILE
                    else ZERO_MILLISECONDS
                ),
                reset_finalization=_scenario(1),
                evidence_flush_shutdown=_scenario(1),
                retry=ZERO_MILLISECONDS,
                retry_allowance=0,
                download_compile_reservation=(
                    _scenario(20)
                    if job_kind is BudgetJobKind.DOWNLOAD
                    else ZERO_MILLISECONDS
                ),
                reserved_gpu_overhead=_scenario(1),
            )
        )
    return BudgetPolicy(
        schema_version=1,
        policy_name="capacity-authority-diagnostic-policy",
        reducer_protocol_sha256=BUDGET_MATERIALIZATION_PROTOCOL_SHA256,
        job_policies=tuple(rows),
    )


def _inventory(receipt_sha256: str) -> GpuInventory:
    group_id = "capacity-host-fabric"
    device = GpuDevice(
        uuid="GPU-capacity-authority-000",
        host_id="capacity-host",
        model="H100-SXM",
        memory_bytes=80 * 1024**3,
        compute_capability=(9, 0),
        pci_bus_id="0000:01:00.0",
        pci_root="0000:00:00.0",
        numa_node=0,
        interconnects=("NONE_SINGLE_GPU",),
        peer_access_class="single",
        clock_policy="locked",
        power_limit_watts=700.0,
        thermal_limit_celsius=83.0,
        availability=GpuAvailability.READY,
        reserved_processes=(),
        allowed_topology_groups=(group_id,),
    )
    return GpuInventory(
        schema_version=1,
        devices=(device,),
        topology_groups=(
            GpuTopologyGroup(
                group_id=group_id,
                host_id=device.host_id,
                gpu_uuids=(device.uuid,),
                fabric="none",
                bandwidth_class="single",
            ),
        ),
        source_receipt_sha256=receipt_sha256,
    )


def _inventory_receipt() -> dict[str, object]:
    content: dict[str, object] = {
        "schema_version": 1,
        "kind": "gpu_inventory_probe_receipt",
        "challenge_nonce_sha256": content_sha256("inventory-challenge"),
        "host_id": "capacity-host",
        "hostname": "capacity-host.example",
        "machine_id_sha256": content_sha256("capacity-machine"),
        "commands": {
            "gpu": {"argv": ["nvidia-smi", "--query-gpu"], "stdout": "gpu"},
            "processes": {
                "argv": ["nvidia-smi", "--query-compute-apps"],
                "stdout": "",
            },
            "topology": {
                "argv": ["nvidia-smi", "topo", "-m"],
                "stdout": "GPU0",
            },
        },
        "parsed_topology": {"gpu_rows": ["GPU0"], "pairs": []},
        "pci_locality": [
            {
                "index": 0,
                "uuid": "GPU-capacity-authority-000",
                "pci_bus_id": "0000:01:00.0",
                "pci_root": "0000:00:00.0",
                "numa_node": 0,
            }
        ],
    }
    return {**content, "receipt_sha256": content_sha256(content)}


@dataclass(frozen=True)
class _RawAuthorityFixture:
    registry: ExperimentRegistry
    cell_id: str
    inventory: GpuInventory
    envelope: CapacityEnvelope
    authority: object
    manifest_path: Path
    provider_path: Path
    verification_path: Path


def _raw_authority(root: Path, *, provider_quota_gpu_ms: int) -> _RawAuthorityFixture:
    registry = build_industrial_registry(
        gpu_uuids=("logical-capacity-slot-a", "logical-capacity-slot-b"),
        cache_root=str(root / "cache"),
        evidence_root=str(root / "evidence"),
    )
    cell = next(
        row
        for row in registry.cells_for("preflight")
        if row.identity.method == "target_only"
    )
    cell_id = cell.cell_id
    collection_nonce = content_sha256({"capacity-collection": root.name})
    captured_at_ns = time.time_ns() - 1_000_000_000

    inventory_receipt = _inventory_receipt()
    inventory_receipt_path = _write_bound(
        root / "inventory-source.json", inventory_receipt
    )
    inventory = _inventory(str(inventory_receipt["receipt_sha256"]))
    inventory_path = _write_bound(root / "inventory.json", inventory.to_dict())
    budget_inventory = budget_inventory_identity_from_gpu_inventory(inventory)

    provider = {
        "schema_version": 1,
        "kind": "industrial_provider_quota_receipt",
        "budget_inventory_sha256": budget_inventory.sha256,
        "gpu_inventory_sha256": inventory.sha256,
        "inventory_source_receipt_sha256": inventory.source_receipt_sha256,
        "provider_scope_sha256": content_sha256("provider-scope"),
        "collection_nonce_sha256": collection_nonce,
        "captured_at_ns": captured_at_ns,
        "total_quota_gpu_ms": provider_quota_gpu_ms,
        "consumed_gpu_ms": 0,
        "available_gpu_ms": provider_quota_gpu_ms,
    }
    provider_path = _write_bound(root / "provider-quota.json", provider)
    host = {
        "schema_version": 1,
        "kind": "industrial_host_capacity_receipt",
        "budget_inventory_sha256": budget_inventory.sha256,
        "gpu_inventory_sha256": inventory.sha256,
        "inventory_source_receipt_sha256": inventory.source_receipt_sha256,
        "host_sha256": budget_inventory.host_sha256,
        "filesystem_sha256": content_sha256("capacity-filesystem"),
        "collection_nonce_sha256": collection_nonce,
        "captured_at_ns": captured_at_ns,
        "host_free_bytes": 10**12,
        "host_quota_bytes": 10**12,
    }
    host_path = _write_bound(root / "host-capacity.json", host)

    provenance_specs = (
        (
            "evidence",
            "industrial_evidence_capacity_provenance",
            1_000,
        ),
        (
            "model",
            "industrial_model_staging_capacity_provenance",
            2_000,
        ),
        (
            "compile",
            "industrial_compile_overlay_capacity_provenance",
            3_000,
        ),
    )
    provenance_paths: dict[str, Path] = {}
    from lightcone_spec.experiments.capacity_authority import bind_capacity_raw_json

    for name, kind, maximum_bytes in provenance_specs:
        provenance_paths[name] = _write_bound(
            root / f"{name}-provenance.json",
            {
                "schema_version": 1,
                "kind": kind,
                "cell_id": cell_id,
                "maximum_bytes": maximum_bytes,
                "derivation_sha256": content_sha256({"capacity-provenance": name}),
            },
        )
    sizing = {
        "schema_version": 1,
        "kind": "industrial_cell_capacity_sizing_receipt",
        "registry_sha256": registry.sha256,
        "budget_inventory_sha256": budget_inventory.sha256,
        "cell_id": cell_id,
        "maximum_evidence_bytes": 1_000,
        "model_staging_bytes": 2_000,
        "compile_overlay_bytes": 3_000,
        "evidence_contract_source": bind_capacity_raw_json(
            provenance_paths["evidence"]
        ).to_dict(),
        "model_staging_source": bind_capacity_raw_json(
            provenance_paths["model"]
        ).to_dict(),
        "compile_overlay_source": bind_capacity_raw_json(
            provenance_paths["compile"]
        ).to_dict(),
        "sizing_protocol_sha256": CELL_CAPACITY_SIZING_PROTOCOL_SHA256,
    }
    sizing_path = _write_bound(root / "cell-sizing.json", sizing)
    capacity_source_sha256 = capacity_source_receipt_sha256_from_paths(
        inventory_source_receipt_path=inventory_receipt_path,
        provider_quota_receipt_path=provider_path,
        host_capacity_receipt_path=host_path,
        cell_sizing_receipt_paths=(sizing_path,),
    )
    envelope = CapacityEnvelope(
        schema_version=1,
        budget_inventory_sha256=budget_inventory.sha256,
        provider_quota_gpu_ms=provider_quota_gpu_ms,
        host_free_bytes=10**12,
        host_quota_bytes=10**12,
        cell_requirements=(
            CellCapacityRequirement(
                cell_id=cell_id,
                maximum_evidence_bytes=1_000,
                model_staging_bytes=2_000,
                compile_overlay_bytes=3_000,
            ),
        ),
        source_receipt_sha256=capacity_source_sha256,
    )
    envelope_path = _write_bound(
        root / "capacity-envelope.json", capacity_envelope_to_dict(envelope)
    )
    manifest = build_capacity_source_manifest(
        registry_sha256=registry.sha256,
        budget_inventory_sha256=budget_inventory.sha256,
        collection_nonce_sha256=collection_nonce,
        capacity_envelope_path=envelope_path,
        gpu_inventory_path=inventory_path,
        inventory_source_receipt_path=inventory_receipt_path,
        provider_quota_receipt_path=provider_path,
        host_capacity_receipt_path=host_path,
        cell_sizing_receipt_paths=(sizing_path,),
    )
    manifest_path = _write_bound(root / "capacity-manifest.json", manifest)
    challenge = AttestationChallenge.issue(
        challenge_id=f"capacity-{root.name}",
        subject_sha256=content_sha256(manifest),
        lifetime_s=60.0,
    )
    payload_sha256 = content_sha256(build_capacity_verification_payload(manifest))
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signature = private_key.sign(
        attestation_message(challenge, payload_sha256=payload_sha256)
    )
    attestation = SignedAttestation(
        schema_version=1,
        kind="lightcone_signed_attestation",
        algorithm="Ed25519",
        attester_id="capacity-lab-signer",
        key_id="capacity-lab-key",
        environment="release",
        public_key_base64=base64.b64encode(public_key).decode(),
        challenge_sha256=challenge.sha256,
        payload_sha256=payload_sha256,
        signature_base64=base64.b64encode(signature).decode(),
    )
    verification = capacity_verification_receipt_template(
        source_manifest=manifest,
        challenge=challenge,
        attestation=attestation,
    )
    verification_path = _write_bound(root / "capacity-verification.json", verification)
    authority = bind_capacity_authority(manifest_path, verification_path)
    return _RawAuthorityFixture(
        registry=registry,
        cell_id=cell_id,
        inventory=inventory,
        envelope=envelope,
        authority=authority,
        manifest_path=manifest_path,
        provider_path=provider_path,
        verification_path=verification_path,
    )


def _plan(fixture: _RawAuthorityFixture, *, authority: object | None):
    cell = next(row for row in fixture.registry.cells if row.cell_id == fixture.cell_id)
    inventory = budget_inventory_identity_from_gpu_inventory(fixture.inventory)
    wall_time = _scenario(32)
    budget = ExperimentBudget(
        schema_version=1,
        cell_id=cell.cell_id,
        experiment=cell.identity.experiment,
        method=cell.identity.method,
        workload_class=cell.resources.workload_class,
        job_kind=BudgetJobKind.COMPILE,
        startup_model_load=_scenario(10),
        compile_jit_graph_prewarm=_scenario(20),
        excluded_warmup=ZERO_MILLISECONDS,
        excluded_warmup_requests=ExpectedMaximumCount(0, 0),
        scored_arrival=ZERO_MILLISECONDS,
        request_deadline=ZERO_MILLISECONDS,
        drain=ZERO_MILLISECONDS,
        reset_finalization=_scenario(1),
        evidence_flush_shutdown=_scenario(1),
        output_tokens=ExpectedMaximumCount(0, 0),
        minimum_completed_requests=0,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=ZERO_MILLISECONDS,
        failure_injection=ZERO_MILLISECONDS,
        retry=ZERO_MILLISECONDS,
        retry_allowance=0,
        profiler=ZERO_MILLISECONDS,
        download_compile_reservation=ZERO_MILLISECONDS,
        gpu_count=1,
        topology=cell.identity.topology,
        reserved_gpu_ms=_scenario(33),
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=wall_time.scale(inventory.gpu_count),
    )
    reducer_activation_sha256s = (content_sha256("capacity-test-activation"),)
    activation_sha256 = content_sha256(
        {
            "reducer_activation_sha256s": reducer_activation_sha256s,
            "family_activation_sha256s": (),
            "family_power_reduction_sha256s": (),
        }
    )
    reason = (
        "capacity_raw_authority_missing"
        if authority is None
        else "trusted_capacity_verifier_unavailable"
    )
    return BudgetPlan(
        schema_version=2,
        registry_sha256=fixture.registry.sha256,
        activation_sha256=activation_sha256,
        reducer_activation_sha256s=reducer_activation_sha256s,
        family_activation_sha256s=(),
        family_power_reduction_sha256s=(),
        policy=_policy(),
        inventory=inventory,
        capacity_envelope=fixture.envelope,
        capacity_authority=authority,
        activated_cell_ids=(cell.cell_id,),
        budgets=(budget,),
        dispositions=(
            BudgetDisposition(
                cell_id=cell.cell_id,
                status=BudgetDispositionStatus.UNRESOLVED,
                reason_code=reason,
                source_semantics_sha256=content_sha256(
                    {"diagnostic-budget": budget.sha256, "reason": reason}
                ),
                experiment_budget_sha256=None,
            ),
        ),
        status="UNRESOLVED",
        reducer_protocol_sha256=BUDGET_MATERIALIZATION_PROTOCOL_SHA256,
    )


def test_bare_envelope_is_diagnostic_and_cannot_make_a_ready_plan(
    tmp_path: Path,
) -> None:
    fixture = _raw_authority(tmp_path / "bare", provider_quota_gpu_ms=10**12)
    plan = _plan(fixture, authority=None)

    assert plan.status == "UNRESOLVED"
    assert len(plan.budgets) == 1
    assert {row.reason_code for row in plan.dispositions} == {
        "capacity_raw_authority_missing"
    }
    with pytest.raises(ValueError, match="capacity_raw_authority_missing"):
        plan.require_ready()
    assert budget_plan_from_dict(budget_plan_to_dict(plan)) == plan


def test_valid_test_signature_cannot_unlock_source_owned_release_authority(
    tmp_path: Path,
) -> None:
    fixture = _raw_authority(tmp_path / "test-signer", provider_quota_gpu_ms=10**12)
    inventory = budget_inventory_identity_from_gpu_inventory(fixture.inventory)
    assert (
        capacity_authority_binding_from_dict(
            capacity_authority_binding_to_dict(fixture.authority)
        )
        == fixture.authority
    )

    with pytest.raises(
        CapacityAuthorityUnavailableError,
        match="trusted_capacity_verifier_unavailable",
    ):
        revalidate_capacity_authority_binding(
            fixture.authority,
            expected_registry_sha256=fixture.registry.sha256,
            expected_inventory=inventory,
            expected_envelope=fixture.envelope,
        )
    plan = _plan(fixture, authority=fixture.authority)
    assert plan.status == "UNRESOLVED"
    assert len(plan.budgets) == 1
    assert {row.reason_code for row in plan.dispositions} == {
        "trusted_capacity_verifier_unavailable"
    }

    caller_policy = replace(
        fixture.authority,
        trusted_verifier_policy_sha256=content_sha256("caller-selected-policy"),
    )
    with pytest.raises(ValueError, match="another verifier policy"):
        revalidate_capacity_authority_binding(
            caller_policy,
            expected_registry_sha256=fixture.registry.sha256,
            expected_inventory=inventory,
            expected_envelope=fixture.envelope,
        )


def test_raw_tamper_and_coordinated_rehash_with_huge_quota_never_authorize(
    tmp_path: Path,
) -> None:
    fixture = _raw_authority(tmp_path / "tamper", provider_quota_gpu_ms=10**12)
    provider = json.loads(fixture.provider_path.read_text(encoding="utf-8"))
    provider["total_quota_gpu_ms"] = 10**18
    provider["available_gpu_ms"] = 10**18
    _write_bound(fixture.provider_path, provider)
    with pytest.raises(RuntimeError, match="source or sidecar changed"):
        revalidate_capacity_authority_binding(
            fixture.authority,
            expected_registry_sha256=fixture.registry.sha256,
            expected_inventory=budget_inventory_identity_from_gpu_inventory(
                fixture.inventory
            ),
            expected_envelope=fixture.envelope,
        )

    coordinated = _raw_authority(
        tmp_path / "coordinated-rehash", provider_quota_gpu_ms=10**18
    )
    plan = _plan(coordinated, authority=coordinated.authority)
    assert coordinated.envelope.provider_quota_gpu_ms == 10**18
    assert plan.status == "UNRESOLVED"
    assert {row.reason_code for row in plan.dispositions} == {
        "trusted_capacity_verifier_unavailable"
    }
    with pytest.raises(ValueError, match="trusted_capacity_verifier_unavailable"):
        plan.require_ready()


def test_formal_revalidation_rejects_an_expired_signed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _raw_authority(tmp_path / "stale", provider_quota_gpu_ms=10**12)
    verification = json.loads(fixture.verification_path.read_text(encoding="utf-8"))
    stale_now_ns = verification["challenge"]["expires_ns"] + 1
    monkeypatch.setattr(
        "lightcone_spec.experiments.capacity_authority.time.time_ns",
        lambda: stale_now_ns,
    )

    with pytest.raises(ValueError, match="expired"):
        revalidate_capacity_authority_binding(
            fixture.authority,
            expected_registry_sha256=fixture.registry.sha256,
            expected_inventory=budget_inventory_identity_from_gpu_inventory(
                fixture.inventory
            ),
            expected_envelope=fixture.envelope,
        )


def _raw_stage_schedule(fixture: _RawAuthorityFixture) -> StageCapacitySchedule:
    return StageCapacitySchedule(
        schema_version=1,
        kind="industrial_stage_capacity_schedule",
        registry_sha256=fixture.registry.sha256,
        experiment="preflight",
        activated_cell_ids=(fixture.cell_id,),
        gpu_inventory_sha256=fixture.inventory.sha256,
        dispatch_plan_sha256=content_sha256("raw-capacity-dispatch"),
        budget_plan_sha256=content_sha256("raw-capacity-budget"),
        capacity_envelope_sha256=fixture.envelope.sha256,
        capacity_authority_sha256=fixture.authority.sha256,
        waves=(
            StageCapacityWaveBinding(
                wave_index=0,
                cell_ids=(fixture.cell_id,),
                topology_sha256=content_sha256("raw-capacity-wave"),
            ),
        ),
        retries=(
            StageCapacityRetryBinding(
                cell_id=fixture.cell_id,
                experiment_budget_sha256=content_sha256("raw-capacity-cell-budget"),
                retry_allowance=2,
            ),
        ),
    )


def test_dynamic_stage_gate_is_derived_from_reopened_raw_bytes_and_schedule(
    tmp_path: Path,
) -> None:
    fixture = _raw_authority(tmp_path / "dynamic-stage", provider_quota_gpu_ms=10**12)
    schedule = _raw_stage_schedule(fixture)
    now_ns = time.time_ns()

    gate = materialize_stage_capacity_gate_from_raw_sources(
        fixture.registry,
        experiment="preflight",
        activated_cell_ids=(fixture.cell_id,),
        source_manifest_path=str(fixture.manifest_path),
        schedule=StageCapacitySchedule.from_dict(schedule.to_dict()),
        now_ns=now_ns,
    )

    assert gate.schema_version == 3
    assert gate.status == "AVAILABLE"
    assert gate.capacity_verification_receipt_sha256 is None
    assert gate.capacity_source_authority is not None
    assert gate.retained_evidence_bytes == 3_000
    assert gate.maximum_concurrent_transient_bytes == 5_000
    assert type(gate).from_dict(gate.to_dict()) == gate
    replay = revalidate_stage_capacity_gate_sources(
        fixture.registry,
        gate,
        schedule=schedule,
        now_ns=now_ns,
    )
    assert replay.source_manifest.path == str(fixture.manifest_path)
    assert replay.capacity_envelope == fixture.envelope

    provider = json.loads(fixture.provider_path.read_text(encoding="utf-8"))
    provider["host_capacity_injection"] = True
    _write_bound(fixture.provider_path, provider)
    with pytest.raises((RuntimeError, ValueError), match="changed|fields differ"):
        revalidate_stage_capacity_gate_sources(
            fixture.registry,
            gate,
            schedule=schedule,
            now_ns=now_ns,
        )


def test_dynamic_stage_gate_rejects_stale_raw_observation_and_wrong_schedule(
    tmp_path: Path,
) -> None:
    fixture = _raw_authority(tmp_path / "dynamic-stale", provider_quota_gpu_ms=10**12)
    schedule = _raw_stage_schedule(fixture)
    wrong_schedule = replace(
        schedule,
        capacity_envelope_sha256=content_sha256("foreign-capacity-envelope"),
    )
    with pytest.raises(ValueError, match="differs from the exact stage schedule"):
        materialize_stage_capacity_gate_from_raw_sources(
            fixture.registry,
            experiment="preflight",
            activated_cell_ids=(fixture.cell_id,),
            source_manifest_path=str(fixture.manifest_path),
            schedule=wrong_schedule,
            now_ns=time.time_ns(),
        )

    provider = json.loads(fixture.provider_path.read_text(encoding="utf-8"))
    stale_now_ns = int(provider["captured_at_ns"]) + CAPACITY_MAXIMUM_SOURCE_AGE_NS + 1
    with pytest.raises(ValueError, match="observations are stale"):
        materialize_stage_capacity_gate_from_raw_sources(
            fixture.registry,
            experiment="preflight",
            activated_cell_ids=(fixture.cell_id,),
            source_manifest_path=str(fixture.manifest_path),
            schedule=schedule,
            now_ns=stale_now_ns,
        )


def test_materialize_stage_capacity_gate_cli_reopens_all_inputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dynamic-cli"
    fixture = _raw_authority(root, provider_quota_gpu_ms=10**12)
    schedule = _raw_stage_schedule(fixture)
    registry_path = _write_bound(
        root / "registry.json",
        _industrial_registry_artifact(
            fixture.registry,
            base_port=24_000,
            cache_root=str(root / "cache"),
            evidence_root=str(root / "evidence"),
            seed=20_260_811,
        ),
    )
    schedule_path = _write_bound(root / "schedule.json", schedule.to_dict())
    output_path = root / "stage-capacity-gate.json"

    assert (
        cli_main(
            [
                "materialize-stage-capacity-gate",
                "--registry",
                str(registry_path),
                "--capacity-source-manifest",
                str(fixture.manifest_path),
                "--stage-schedule",
                str(schedule_path),
                "--now-ns",
                str(time.time_ns()),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    gate = StageCapacityGate.from_dict(
        json.loads(output_path.read_text(encoding="utf-8"))
    )
    assert gate.schema_version == 3
    assert gate.status == "AVAILABLE"
    assert gate.capacity_source_authority is not None
