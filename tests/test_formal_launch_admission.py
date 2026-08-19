from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_control_attestation import _root_binding
from test_formal_gpu_hour_registry import (
    _bundle,
    _control,
    _deployment,
    _reservation_for_challenges,
)

from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.stage_capacity import (
    StageCapacityRetryBinding,
    StageCapacitySchedule,
    StageCapacityWaveBinding,
)
from lightcone_spec.orchestration import formal_launch_admission as admission
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return content_sha256({"formal-launch-admission-test": label})


def _binding(root: Path, name: str, value: object) -> CanonicalJsonProofBinding:
    path = (root / name).resolve()
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _stage_schedule_binding(
    root: Path,
    *,
    cells: tuple[str, ...],
    waves: tuple[tuple[str, ...], ...],
    inventory_sha256: str,
    capacity_envelope_sha256: str,
) -> CanonicalJsonProofBinding:
    schedule = StageCapacitySchedule(
        schema_version=1,
        kind="industrial_stage_capacity_schedule",
        registry_sha256=build_industrial_registry().sha256,
        experiment="E3a",
        activated_cell_ids=cells,
        gpu_inventory_sha256=inventory_sha256,
        dispatch_plan_sha256=_sha("dispatch"),
        budget_plan_sha256=_sha("budget"),
        capacity_envelope_sha256=capacity_envelope_sha256,
        capacity_authority_sha256=_sha("capacity-authority"),
        waves=tuple(
            StageCapacityWaveBinding(
                wave_index=index,
                cell_ids=wave,
                topology_sha256=_sha(f"wave:{index}:{','.join(wave)}"),
            )
            for index, wave in enumerate(waves)
        ),
        retries=tuple(
            StageCapacityRetryBinding(
                cell_id=cell_id,
                experiment_budget_sha256=_sha(f"budget:{cell_id}"),
                retry_allowance=1,
            )
            for cell_id in cells
        ),
    )
    path = root / f"stage-capacity-schedule-{schedule.sha256}.json"
    if path.exists():
        binding = CanonicalJsonProofBinding.bind(path)
        assert StageCapacitySchedule.from_dict(binding.reopen()) == schedule
        return binding
    return _binding(root, path.name, schedule.to_dict())


def _schedule(
    root: Path,
    *,
    cell_id: str,
    cells: tuple[str, ...],
    wave: tuple[str, ...],
    topology: str,
    gpu_uuids: tuple[str, ...],
    provider_reserved_gpu_count: int,
    suffix: str,
) -> admission.FormalStageCapacitySchedule:
    inventory_sha256 = _sha("inventory")
    capacity_envelope_sha256 = _sha("capacity-envelope")
    stage_waves = (
        (wave,) if set(wave) == set(cells) else tuple((current,) for current in cells)
    )
    stage_schedule = _stage_schedule_binding(
        root,
        cells=cells,
        waves=stage_waves,
        inventory_sha256=inventory_sha256,
        capacity_envelope_sha256=capacity_envelope_sha256,
    )
    return admission.FormalStageCapacitySchedule(
        schema_version=1,
        kind="lightcone_formal_stage_capacity_schedule",
        protocol_sha256=admission.FORMAL_STAGE_CAPACITY_PROTOCOL_SHA256,
        stage="E3a",
        protocol_lock_sha256=_sha("protocol-lock"),
        runtime_authority_manifest_sha256=_sha("runtime"),
        registry_sha256=build_industrial_registry().sha256,
        materialization=_binding(
            root,
            f"materialization-{suffix}.json",
            {"kind": "test-materialization", "cells": list(cells)},
        ),
        materialization_receipt_sha256=_sha("materialization"),
        stage_capacity_schedule=stage_schedule,
        stage_capacity_schedule_sha256=stage_schedule.semantic_sha256,
        activated_cell_ids=cells,
        materialized_cell_id=cell_id,
        execution_binding_sha256=_sha(f"execution-binding:{suffix}"),
        execution_plan_sha256=_sha(f"execution-plan:{suffix}"),
        run_plan=_binding(
            root,
            f"run-plan-{suffix}.json",
            {"kind": "test-run-plan", "cell": cell_id},
        ),
        run_plan_sha256=_sha(f"run-plan:{suffix}"),
        topology_mode=topology,
        inventory_sha256=inventory_sha256,
        gpu_uuids=gpu_uuids,
        wave_index=stage_waves.index(wave),
        wave_cell_ids=wave,
        provider_inventory_gpu_count=2,
        provider_reserved_gpu_count=provider_reserved_gpu_count,
        retry_allowance=1,
        capacity_source_manifest=_binding(
            root,
            f"capacity-source-{suffix}.json",
            {"kind": "test-capacity-source", "cell": cell_id},
        ),
        capacity_envelope_sha256=capacity_envelope_sha256,
        budget_inventory_sha256=_sha("budget-inventory"),
        capacity_captured_at_ns=1,
    )


def _gate(schedule: admission.FormalStageCapacitySchedule):
    return admission.FormalStageCapacityGate(
        schema_version=3,
        kind="lightcone_formal_stage_capacity_gate",
        protocol_sha256=admission.FORMAL_STAGE_CAPACITY_PROTOCOL_SHA256,
        stage=schedule.stage,
        materialization_receipt_sha256=(schedule.materialization_receipt_sha256),
        materialized_cell_id=schedule.materialized_cell_id,
        schedule_sha256=schedule.sha256,
        inventory_sha256=schedule.inventory_sha256,
        status="AVAILABLE",
        reason_code="formal_stage_capacity_verified",
        observed_free_bytes=3,
        retained_evidence_bytes=1,
        maximum_concurrent_transient_bytes=1,
        high_water_bytes=2,
        safety_margin_bytes=1,
        required_free_bytes=3,
        capacity_source_replay_sha256=_sha("capacity-replay"),
    )


def _limits(
    root: Path,
    *,
    cells: tuple[str, ...],
    gpu_counts: tuple[int, ...],
    timeout_by_cell: tuple[int, ...],
    provider_counts: tuple[int, ...] | None = None,
    provider_timeout_by_cell: tuple[int, ...] | None = None,
) -> admission._FormalStageBudgetLimits:
    replay = (root / "replay").resolve()
    replay.mkdir(mode=0o700)
    if provider_counts is None:
        provider_counts = tuple(2 for _cell in cells)
    if provider_timeout_by_cell is None:
        provider_timeout_by_cell = timeout_by_cell
    return admission._FormalStageBudgetLimits(
        mode="available_stage_gpu_hour",
        authority_sha256=_sha("authority"),
        authority_reservation_sha256=_sha("authority-reservation"),
        control_replay_root=str(replay),
        ledger_parent=str((root / "budget-consumption").resolve()),
        stage="E3a",
        materialization_receipt_sha256=_sha("materialization"),
        cell_ids=cells,
        gpu_count_by_cell=tuple(zip(cells, gpu_counts, strict=True)),
        provider_reserved_gpu_count_cap_by_cell=tuple(
            zip(cells, provider_counts, strict=True)
        ),
        allowed_attempts_per_cell=2,
        hard_timeout_ns_by_cell=tuple(zip(cells, timeout_by_cell, strict=True)),
        provider_wave_hard_timeout_ns_by_cell=tuple(
            zip(cells, provider_timeout_by_cell, strict=True)
        ),
        compute_charge_gpu_ns_by_cell=tuple(
            (cell_id, timeout * gpu_count)
            for cell_id, timeout, gpu_count in zip(
                cells, timeout_by_cell, gpu_counts, strict=True
            )
        ),
        reserved_charge_gpu_ns_by_cell=tuple(
            (cell_id, timeout * provider_count)
            for cell_id, timeout, provider_count in zip(
                cells,
                provider_timeout_by_cell,
                provider_counts,
                strict=True,
            )
        ),
        hard_timeout_derivation_sha256=_sha("timeout-derivation"),
        maximum_compute_gpu_ns=10**15,
        maximum_reserved_gpu_ns=10**15,
    )


def _capacity_control(
    *,
    schedule: admission.FormalStageCapacitySchedule,
    gate: admission.FormalStageCapacityGate,
    nonce: int,
):
    root_private = Ed25519PrivateKey.generate()
    control_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    bundle = _bundle(control_private)
    authorization = _deployment(
        root_private,
        root_binding=root_binding,
        bundle=bundle,
        inventory_sha256=schedule.inventory_sha256,
        nonce=90 + nonce,
    )
    return _control(
        control_private,
        root_binding=root_binding,
        bundle=bundle,
        authorization=authorization,
        artifact_type="capacity",
        artifact_sha256=gate.sha256,
        protocol_sha256=admission.FORMAL_STAGE_CAPACITY_PROTOCOL_SHA256,
        lineage_sha256=admission.formal_stage_capacity_control_lineage_sha256(
            schedule=schedule,
            gate=gate,
        ),
        nonce=100 + nonce,
    )


def test_budget_ledger_separates_compute_from_provider_reservation(
    tmp_path: Path,
) -> None:
    first, second, tp2 = tuple(_sha(label) for label in ("first", "second", "tp2"))

    isolated_root = tmp_path / "isolated"
    isolated_root.mkdir()
    isolated = _schedule(
        isolated_root,
        cell_id=first,
        cells=(first,),
        wave=(first,),
        topology="tp1_dp1",
        gpu_uuids=("GPU-0",),
        provider_reserved_gpu_count=2,
        suffix="isolated",
    )
    isolated_gate = _gate(isolated)
    isolated_entry = admission.FormalStageBudgetConsumption.from_dict(
        admission._reserve_stage_budget_consumption(
            limits=_limits(
                isolated_root,
                cells=(first,),
                gpu_counts=(1,),
                timeout_by_cell=(10,),
            ),
            schedule=isolated,
            gate=isolated_gate,
            capacity_control=_capacity_control(
                schedule=isolated, gate=isolated_gate, nonce=1
            ),
            reserved_ns=1,
        ).reopen()
    )
    assert isolated_entry.compute_charge_gpu_ns == 10
    assert isolated_entry.reserved_charge_gpu_ns == 20

    pair_root = tmp_path / "pair"
    pair_root.mkdir()
    pair_limits = _limits(
        pair_root,
        cells=tuple(sorted((first, second))),
        gpu_counts=(1, 1),
        timeout_by_cell=(10, 10),
        provider_counts=(1, 1),
    )
    pair_entries = []
    for index, cell_id in enumerate(tuple(sorted((first, second)))):
        schedule = _schedule(
            pair_root,
            cell_id=cell_id,
            cells=tuple(sorted((first, second))),
            wave=tuple(sorted((first, second))),
            topology="tp1_dp1",
            gpu_uuids=(f"GPU-{index}",),
            provider_reserved_gpu_count=1,
            suffix=f"pair-{index}",
        )
        gate = _gate(schedule)
        pair_entries.append(
            admission.FormalStageBudgetConsumption.from_dict(
                admission._reserve_stage_budget_consumption(
                    limits=pair_limits,
                    schedule=schedule,
                    gate=gate,
                    capacity_control=_capacity_control(
                        schedule=schedule, gate=gate, nonce=10 + index
                    ),
                    reserved_ns=2 + index,
                ).reopen()
            )
        )
    assert sum(row.compute_charge_gpu_ns for row in pair_entries) == 20
    assert sum(row.reserved_charge_gpu_ns for row in pair_entries) == 20

    tp2_root = tmp_path / "tp2"
    tp2_root.mkdir()
    distributed = _schedule(
        tp2_root,
        cell_id=tp2,
        cells=(tp2,),
        wave=(tp2,),
        topology="tp2_dp1",
        gpu_uuids=("GPU-0", "GPU-1"),
        provider_reserved_gpu_count=2,
        suffix="tp2",
    )
    distributed_gate = _gate(distributed)
    distributed_entry = admission.FormalStageBudgetConsumption.from_dict(
        admission._reserve_stage_budget_consumption(
            limits=_limits(
                tp2_root,
                cells=(tp2,),
                gpu_counts=(2,),
                timeout_by_cell=(10,),
            ),
            schedule=distributed,
            gate=distributed_gate,
            capacity_control=_capacity_control(
                schedule=distributed, gate=distributed_gate, nonce=20
            ),
            reserved_ns=4,
        ).reopen()
    )
    assert distributed_entry.compute_charge_gpu_ns == 20
    assert distributed_entry.reserved_charge_gpu_ns == 20


def test_per_cell_timeout_map_is_not_a_stage_average(tmp_path: Path) -> None:
    cells = tuple(sorted((_sha("short"), _sha("long"))))
    root = tmp_path / "timeouts"
    root.mkdir()
    limits = _limits(
        root,
        cells=cells,
        gpu_counts=(1, 1),
        timeout_by_cell=(7, 19),
    )
    observed = {}
    for index, cell_id in enumerate(cells):
        schedule = _schedule(
            root,
            cell_id=cell_id,
            cells=cells,
            wave=(cell_id,),
            topology="tp1_dp1",
            gpu_uuids=(f"GPU-{index}",),
            provider_reserved_gpu_count=2,
            suffix=f"timeout-{index}",
        )
        gate = _gate(schedule)
        entry = admission.FormalStageBudgetConsumption.from_dict(
            admission._reserve_stage_budget_consumption(
                limits=limits,
                schedule=schedule,
                gate=gate,
                capacity_control=_capacity_control(
                    schedule=schedule, gate=gate, nonce=30 + index
                ),
                reserved_ns=10 + index,
            ).reopen()
        )
        observed[cell_id] = entry.hard_timeout_ns
    assert observed == dict(zip(cells, (7, 19), strict=True))


def test_stage_average_timeout_fallback_is_not_exposed() -> None:
    assert not hasattr(admission, "_full_stage_hard_timeout_authority")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("hard_timeout_ns", 9),
        ("provider_wave_hard_timeout_ns", 11),
        ("protocol_lock_sha256", _sha("foreign-lock")),
        ("runtime_authority_manifest_sha256", _sha("foreign-runtime")),
        ("root_manifest_sha256", _sha("foreign-root")),
        ("registry_sha256", _sha("foreign-registry")),
    ),
)
def test_rehashed_admission_field_tamper_differs_from_deterministic_rebuild(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    cell_id = _sha("rebuild-cell")
    schedule = _schedule(
        tmp_path,
        cell_id=cell_id,
        cells=(cell_id,),
        wave=(cell_id,),
        topology="tp1_dp1",
        gpu_uuids=("GPU-0",),
        provider_reserved_gpu_count=2,
        suffix="rebuild",
    )
    gate = _gate(schedule)
    control = _capacity_control(schedule=schedule, gate=gate, nonce=50)
    limits = _limits(
        tmp_path,
        cells=(cell_id,),
        gpu_counts=(1,),
        timeout_by_cell=(10,),
    )
    budget_consumption = admission._reserve_stage_budget_consumption(
        limits=limits,
        schedule=schedule,
        gate=gate,
        capacity_control=control,
        reserved_ns=20,
    )
    reservation = _reservation_for_challenges(
        tmp_path,
        label="rebuild-reservation",
        challenges=(_sha("reservation-challenge"),),
        reserved_ns=20,
    )
    expected = admission._rebuild_expected_admission(
        plan=SimpleNamespace(private_output_root=str(tmp_path.resolve())),
        schedule_binding=_binding(tmp_path, "formal-capacity.json", schedule.to_dict()),
        schedule=schedule,
        gate_binding=_binding(tmp_path, "formal-capacity-gate.json", gate.to_dict()),
        capacity_control=control,
        root_manifest_sha256=_sha("release-root"),
        budget_mode="available_stage_gpu_hour",
        stage_gpu_hour_receipt=_binding(
            tmp_path, "stage-gpu-hour.json", {"kind": "test-stage-budget"}
        ),
        pilot_launch_budget=None,
        pilot_budget_verification_receipt=None,
        reservation=reservation,
        budget_consumption=budget_consumption,
        hard_timeout_ns=10,
        provider_wave_hard_timeout_ns=10,
    )
    tampered = replace(expected, **{field: value})
    reopened = admission.FormalStageLaunchAdmission.from_dict(tampered.to_dict())
    with pytest.raises(ValueError, match="deterministic rebuild"):
        admission._require_expected_admission(
            reopened,
            expected,
            label="formal launch admission",
        )


def test_rehashed_admission_consumption_path_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    cell_id = _sha("path-cell")
    schedule = _schedule(
        tmp_path,
        cell_id=cell_id,
        cells=(cell_id,),
        wave=(cell_id,),
        topology="tp1_dp1",
        gpu_uuids=("GPU-0",),
        provider_reserved_gpu_count=2,
        suffix="path",
    )
    gate = _gate(schedule)
    control = _capacity_control(schedule=schedule, gate=gate, nonce=60)
    limits = _limits(
        tmp_path,
        cells=(cell_id,),
        gpu_counts=(1,),
        timeout_by_cell=(10,),
    )
    expected = admission._rebuild_expected_admission(
        plan=SimpleNamespace(private_output_root=str(tmp_path.resolve())),
        schedule_binding=_binding(tmp_path, "path-capacity.json", schedule.to_dict()),
        schedule=schedule,
        gate_binding=_binding(tmp_path, "path-gate.json", gate.to_dict()),
        capacity_control=control,
        root_manifest_sha256=_sha("release-root"),
        budget_mode="available_stage_gpu_hour",
        stage_gpu_hour_receipt=_binding(
            tmp_path, "path-budget.json", {"kind": "test-stage-budget"}
        ),
        pilot_launch_budget=None,
        pilot_budget_verification_receipt=None,
        reservation=_reservation_for_challenges(
            tmp_path,
            label="path-reservation",
            challenges=(_sha("path-challenge"),),
            reserved_ns=20,
        ),
        budget_consumption=admission._reserve_stage_budget_consumption(
            limits=limits,
            schedule=schedule,
            gate=gate,
            capacity_control=control,
            reserved_ns=20,
        ),
        hard_timeout_ns=10,
        provider_wave_hard_timeout_ns=10,
    )
    foreign = tmp_path / "foreign" / "formal-stage-launch-consumed.json"
    tampered = replace(expected, consumption_path=str(foreign.resolve()))
    with pytest.raises(ValueError, match="deterministic rebuild"):
        admission._require_expected_admission(
            admission.FormalStageLaunchAdmission.from_dict(tampered.to_dict()),
            expected,
            label="formal launch admission",
        )
