from __future__ import annotations

from types import SimpleNamespace

import pytest

from lightcone_spec.experiments import stage_capacity
from lightcone_spec.experiments.capacity_authority import (
    CapacityAuthorityUnavailableError,
)
from lightcone_spec.experiments.planning import (
    CapacityEnvelope,
    CellCapacityRequirement,
)
from lightcone_spec.experiments.registry import (
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.stage_capacity import (
    LEGACY_MINIMUM_FREE_BYTES,
    STAGE_CAPACITY_MINIMUM_SAFETY_MARGIN_BYTES,
    StageCapacityRetryBinding,
    StageCapacitySchedule,
    StageCapacityWaveBinding,
    evaluate_stage_capacity,
)


def _sha(label: str) -> str:
    return content_sha256({"stage-capacity-test": label})


def _stage_cells(registry) -> tuple[str, ...]:
    return tuple(sorted(cell.cell_id for cell in registry.cells_for("preflight")))


def _envelope(cell_ids: tuple[str, ...], *, host_bytes: int) -> CapacityEnvelope:
    return CapacityEnvelope(
        schema_version=1,
        budget_inventory_sha256=_sha("inventory"),
        provider_quota_gpu_ms=10**12,
        host_free_bytes=host_bytes,
        host_quota_bytes=host_bytes,
        cell_requirements=tuple(
            CellCapacityRequirement(
                cell_id=cell_id,
                maximum_evidence_bytes=100_000_000,
                model_staging_bytes=200_000_000,
                compile_overlay_bytes=300_000_000,
            )
            for cell_id in cell_ids
        ),
        source_receipt_sha256=_sha("sources"),
    )


def _schedule(
    registry,
    cells: tuple[str, ...],
    *,
    retries: int = 1,
) -> StageCapacitySchedule:
    midpoint = len(cells) // 2
    waves = (cells[:midpoint], cells[midpoint:])
    return StageCapacitySchedule(
        schema_version=1,
        kind="industrial_stage_capacity_schedule",
        registry_sha256=registry.sha256,
        experiment="preflight",
        activated_cell_ids=cells,
        gpu_inventory_sha256=_sha("gpu-inventory"),
        dispatch_plan_sha256=_sha("dispatch-plan"),
        budget_plan_sha256=_sha("budget-plan"),
        capacity_envelope_sha256=_envelope(cells, host_bytes=20_000_000_000).sha256,
        capacity_authority_sha256=_sha("capacity-authority"),
        waves=tuple(
            StageCapacityWaveBinding(
                wave_index=index,
                cell_ids=wave,
                topology_sha256=_sha(f"topology-{index}"),
            )
            for index, wave in enumerate(waves)
        ),
        retries=tuple(
            StageCapacityRetryBinding(
                cell_id=cell_id,
                experiment_budget_sha256=_sha(f"budget-{cell_id}"),
                retry_allowance=retries,
            )
            for cell_id in cells
        ),
    )


def test_stage_capacity_uses_explicit_100gb_fallback_only_without_authority() -> None:
    registry = build_industrial_registry()
    cells = _stage_cells(registry)

    blocked = evaluate_stage_capacity(
        registry,
        experiment="preflight",
        activated_cell_ids=cells,
        legacy_host_free_bytes=LEGACY_MINIMUM_FREE_BYTES - 1,
    )
    ready = evaluate_stage_capacity(
        registry,
        experiment="preflight",
        activated_cell_ids=cells,
        legacy_host_free_bytes=LEGACY_MINIMUM_FREE_BYTES,
    )

    assert blocked.mode == ready.mode == "LEGACY_100GB_FALLBACK"
    assert blocked.status == "BLOCKED"
    assert ready.status == "AVAILABLE"
    assert ready.required_free_bytes == 100_000_000_000
    assert ready.capacity_envelope_sha256 is None


def test_fresh_signed_stage_envelope_replaces_global_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = build_industrial_registry()
    cells = _stage_cells(registry)
    envelope = _envelope(cells, host_bytes=20_000_000_000)
    schedule = _schedule(registry, cells)
    result = SimpleNamespace(
        registry_sha256=registry.sha256,
        capacity_envelope=envelope,
        verification_receipt_sha256=_sha("verified"),
        gpu_inventory=SimpleNamespace(sha256=schedule.gpu_inventory_sha256),
    )
    monkeypatch.setattr(
        stage_capacity,
        "revalidate_capacity_authority_binding",
        lambda *_args, **_kwargs: result,
    )

    gate = evaluate_stage_capacity(
        registry,
        experiment="preflight",
        activated_cell_ids=cells,
        capacity_authority=object(),  # type: ignore[arg-type]
        capacity_inventory=object(),  # type: ignore[arg-type]
        capacity_envelope=envelope,
        schedule=schedule,
        legacy_host_free_bytes=0,
    )

    assert gate.mode == "SIGNED_STAGE_ENVELOPE"
    assert gate.status == "AVAILABLE"
    assert gate.retained_evidence_bytes == len(cells) * 200_000_000
    assert gate.maximum_concurrent_transient_bytes == (
        max(len(wave.cell_ids) for wave in schedule.waves) * 500_000_000
    )
    assert gate.high_water_bytes == (
        gate.retained_evidence_bytes + gate.maximum_concurrent_transient_bytes
    )
    assert gate.safety_margin_bytes >= STAGE_CAPACITY_MINIMUM_SAFETY_MARGIN_BYTES
    assert gate.required_free_bytes < LEGACY_MINIMUM_FREE_BYTES
    assert gate.capacity_envelope_sha256 == envelope.sha256
    assert type(gate).from_dict(gate.to_dict()) == gate
    tampered = gate.to_dict()
    tampered["maximum_concurrent_transient_bytes"] += 1
    with pytest.raises(ValueError):
        type(gate).from_dict(tampered)


def test_unavailable_signature_falls_back_but_tamper_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = build_industrial_registry()
    cells = _stage_cells(registry)
    envelope = _envelope(cells, host_bytes=20_000_000_000)
    schedule = _schedule(registry, cells)

    def unavailable(*_args, **_kwargs):
        raise CapacityAuthorityUnavailableError()

    monkeypatch.setattr(
        stage_capacity,
        "revalidate_capacity_authority_binding",
        unavailable,
    )
    gate = evaluate_stage_capacity(
        registry,
        experiment="preflight",
        activated_cell_ids=cells,
        capacity_authority=object(),  # type: ignore[arg-type]
        capacity_inventory=object(),  # type: ignore[arg-type]
        capacity_envelope=envelope,
        schedule=schedule,
        legacy_host_free_bytes=LEGACY_MINIMUM_FREE_BYTES,
    )
    assert gate.mode == "LEGACY_100GB_FALLBACK"

    def tampered(*_args, **_kwargs):
        raise ValueError("capacity source was tampered")

    monkeypatch.setattr(
        stage_capacity,
        "revalidate_capacity_authority_binding",
        tampered,
    )
    with pytest.raises(ValueError, match="tampered"):
        evaluate_stage_capacity(
            registry,
            experiment="preflight",
            activated_cell_ids=cells,
            capacity_authority=object(),  # type: ignore[arg-type]
            capacity_inventory=object(),  # type: ignore[arg-type]
            capacity_envelope=envelope,
            schedule=schedule,
            legacy_host_free_bytes=LEGACY_MINIMUM_FREE_BYTES,
        )


def test_stage_capacity_rejects_partial_authority_and_foreign_cells() -> None:
    registry = build_industrial_registry()
    cells = _stage_cells(registry)
    with pytest.raises(ValueError, match="supplied together"):
        evaluate_stage_capacity(
            registry,
            experiment="preflight",
            activated_cell_ids=cells,
            capacity_authority=object(),  # type: ignore[arg-type]
            legacy_host_free_bytes=LEGACY_MINIMUM_FREE_BYTES,
        )
    foreign = registry.cells_for("E3a")[0].cell_id
    with pytest.raises(ValueError, match="another experiment"):
        evaluate_stage_capacity(
            registry,
            experiment="preflight",
            activated_cell_ids=tuple(sorted((*cells[:-1], foreign))),
            legacy_host_free_bytes=LEGACY_MINIMUM_FREE_BYTES,
        )
