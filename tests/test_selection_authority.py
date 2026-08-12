from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.industrial_analysis import (
    BoundArtifact,
    IndustrialCellEvidence,
    RawE1ParetoEvidenceManifest,
    RawE3aSelectionEvidenceManifest,
    _LoadedCell,
)
from lightcone_spec.experiments.planning import (
    E1GeometryIdentity,
    SealedE3aSelection,
    reduce_e1_activation,
)
from lightcone_spec.experiments.registry import (
    E1_OPTIMIZER_ANCHORS,
    ExperimentCell,
    ExperimentReceipt,
    ExperimentRegistry,
    LockedOutput,
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.selection_authority import (
    SelectionReductionAuthorityUnavailableError,
    bind_e1_pareto_reduction_authority,
    bind_e3a_selection_reduction_authority,
    reduce_e1_pareto_from_raw,
    reduce_e3a_selection_from_raw,
)
from lightcone_spec.experiments.statistics import HardwareEnvelope
from lightcone_spec.telemetry.records import OUTPUT_HASH_FORMAT

_SAFETY_COUNTERS = (
    "exactness_violations",
    "version_mismatches",
    "fallbacks",
    "nonfinite_updates",
    "oom_events",
    "retractions",
    "communicator_failures",
    "evidence_dropped_rows",
)


def _sha(label: str) -> str:
    return content_sha256({"test-selection-authority": label})


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return build_industrial_registry()


@pytest.fixture(scope="module")
def inventory() -> GpuInventory:
    device = GpuDevice(
        uuid="GPU-selection-000",
        host_id="selection-host",
        model="H100-SXM",
        memory_bytes=80 * 1024**3,
        compute_capability=(9, 0),
        pci_bus_id="0000:01:00.0",
        pci_root="selection-root",
        numa_node=0,
        interconnects=("NVLink4",),
        peer_access_class="NVSwitch",
        clock_policy="locked-1980MHz",
        power_limit_watts=700.0,
        thermal_limit_celsius=83.0,
        availability=GpuAvailability.READY,
        reserved_processes=(),
        allowed_topology_groups=("selection-group",),
    )
    return GpuInventory(
        schema_version=1,
        devices=(device,),
        topology_groups=(
            GpuTopologyGroup(
                group_id="selection-group",
                host_id="selection-host",
                gpu_uuids=(device.uuid,),
                fabric="NVLink",
                bandwidth_class="high",
            ),
        ),
        source_receipt_sha256=_sha("inventory-source"),
    )


@pytest.fixture(scope="module")
def hardware_envelope() -> HardwareEnvelope:
    return HardwareEnvelope(
        gpu_clock_mhz_min=1900.0,
        gpu_clock_mhz_max=2000.0,
        memory_clock_mhz_min=1500.0,
        memory_clock_mhz_max=1600.0,
        temperature_c_max=80.0,
        power_watts_min=500.0,
        power_watts_max=700.0,
        power_state="P0",
    )


def _reference(tmp_path: Path, cell_id: str) -> IndustrialCellEvidence:
    def bound(label: str) -> BoundArtifact:
        return BoundArtifact(
            path=(tmp_path / f"{cell_id}-{label}.json").resolve(),
            sha256=_sha(f"{cell_id}-{label}"),
        )

    return IndustrialCellEvidence(
        cell_id=cell_id,
        terminal_receipts=(bound("terminal"),),
        hardware_receipt=bound("hardware"),
        budget_observation=bound("budget"),
        completion_contract=bound("schema-v4-completion"),
    )


def _request_row(*, goodput: float, token_ids: tuple[int, ...] = (101, 102)) -> dict:
    duration_ns = round(len(token_ids) / goodput * 1_000_000_000)
    assert duration_ns > 0
    token_timestamps = [0, duration_ns]
    token_body = json.dumps(list(token_ids), separators=(",", ":"))
    token_sha256 = hashlib.sha256(token_body.encode()).hexdigest()
    return {
        "request_id": "request-000",
        "input_tokens": 1,
        "arrival_ns": 0,
        "admitted_ns": 0,
        "first_token_ns": 0,
        "completed_ns": duration_ns,
        "ttft_ms": 0.0,
        "inter_token_ms": json.dumps([duration_ns / 1_000_000.0]),
        "token_timestamps_ns": json.dumps(token_timestamps),
        "token_timing_coverage": 1.0,
        "coalesced_intervals": 0,
        "output_tokens": len(token_ids),
        "outcome_status": "completed",
        "finished": True,
        "output_hash_format": OUTPUT_HASH_FORMAT,
        "output_token_ids": token_body,
        "output_token_ids_sha256": token_sha256,
        "output_sha256": token_sha256,
    }


def _loaded(
    cell: ExperimentCell,
    *,
    runtime_sha256: str,
    split_sha256: str,
    index: int,
    goodput: float,
    peak_hbm_bytes: int = 1_000,
    exposed_update_ms: float = 1.0,
    unsafe: bool = False,
    published: bool = True,
    token_ids: tuple[int, ...] = (101, 102),
) -> _LoadedCell:
    adapted = cell.identity.method in {"tts", "l0"}
    update_rows = (
        (
            {
                "candidate_status": "published",
                "exposed_update_ms": exposed_update_ms,
            },
        )
        if adapted and published
        else ()
    )
    performance = {
        "offered_requests": 1,
        "peak_hbm_bytes": peak_hbm_bytes,
        "updates_launched": len(update_rows),
        "updates_published": len(update_rows),
        "exposed_update_ms": exposed_update_ms if update_rows else None,
        **{counter: 0 for counter in _SAFETY_COUNTERS},
    }
    if unsafe:
        performance["exactness_violations"] = 1
    common = {
        "model_pair": "selection-model-pair",
        "runtime_sha256": runtime_sha256,
        "split_sha256": split_sha256,
        "corpus_sha256": _sha("common-corpus"),
        "arrival_trace_sha256": _sha("common-arrival"),
        "request_ids_sha256": _sha("common-requests"),
        "sampling_profile_sha256": _sha("common-sampling"),
        "model_lock_sha256": _sha("common-model-lock"),
        "patched_sglang_tree": PINNED_SGLANG_TREE,
        "experiment_budget_sha256": _sha(f"budget-{index}"),
        "topology_sha256": _sha(f"topology-{index}"),
        "config_sha256": _sha(f"config-{index}"),
        "rank_config_sha256": _sha(f"rank-config-{index}"),
        "run_id": f"selection-run-{index:04d}",
        "run_nonce_sha256": _sha(f"nonce-{index}"),
    }
    return _LoadedCell(
        cell=cell,
        observation_source_cell_id=cell.cell_id,
        evidence_alias_reduction_sha256=None,
        run_rows=(common,),
        request_rows=(_request_row(goodput=goodput, token_ids=token_ids),),
        performance_rows_by_rank=((performance,),),
        update_rows_by_rank=(update_rows,),
        terminal_receipt_sha256s=(_sha(f"terminal-{index}"),),
        hardware_receipt_sha256=_sha(f"hardware-{index}"),
        physical_gpu_uuids=("GPU-selection-000",),
        experiment_budget_sha256=common["experiment_budget_sha256"],
        inventory_sha256=_sha("unused-inventory"),
        inventory_source_receipt_sha256=_sha("inventory-source"),
        fixed_instance_gpu_count=1,
        physical_host_id="selection-host",
        budget_observation_sha256=_sha(f"observation-{index}"),
        hardware_validity=((f"hardware-{index}", "VALID", ()),),
    )


def _patch_loader(
    monkeypatch: pytest.MonkeyPatch,
    loaded: dict[str, _LoadedCell],
    *,
    trust_native: bool = True,
):
    from lightcone_spec.experiments import industrial_analysis, selection_authority

    def load(reference: IndustrialCellEvidence, **_kwargs):
        return loaded[reference.cell_id]

    monkeypatch.setattr(industrial_analysis, "_load_cell", load)
    monkeypatch.setattr(
        industrial_analysis,
        "validate_raw_evidence_manifest_sidecars",
        lambda _manifest: None,
    )
    if trust_native:
        monkeypatch.setattr(
            selection_authority,
            "_validate_native_terminal_authority",
            lambda *_args, **_kwargs: (),
        )


def _e3a_goodput(cell: ExperimentCell) -> float:
    if cell.identity.method == "target_only":
        return 200.0
    concurrency = cell.identity.concurrency
    width = cell.identity.width
    best = {
        1: 100.0,
        2: 160.0,
        4: 190.0,
        8: 200.0,
        16: 199.0,
        32: 180.0,
        64: 150.0,
    }[concurrency]
    if width == 4:
        return best - 10.0
    if width == 16 and concurrency != 4:
        return best - 5.0
    return best


def _e3a_fixture(
    tmp_path: Path,
    registry: ExperimentRegistry,
) -> tuple[RawE3aSelectionEvidenceManifest, dict[str, _LoadedCell], str, str]:
    runtime_sha256 = _sha("e3a-runtime")
    split_sha256 = _sha("e3a-split")
    cells = tuple(sorted(registry.cells_for("E3a"), key=lambda row: row.cell_id))
    loaded = {
        cell.cell_id: _loaded(
            cell,
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
            index=index,
            goodput=_e3a_goodput(cell),
        )
        for index, cell in enumerate(cells)
    }
    manifest = RawE3aSelectionEvidenceManifest(
        schema_version=2,
        cells=tuple(_reference(tmp_path, cell.cell_id) for cell in cells),
    )
    return manifest, loaded, runtime_sha256, split_sha256


def test_e3a_reducer_recomputes_90pct_load_and_width_tiebreak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry: ExperimentRegistry,
    inventory: GpuInventory,
    hardware_envelope: HardwareEnvelope,
) -> None:
    manifest, loaded, runtime_sha256, split_sha256 = _e3a_fixture(tmp_path, registry)
    _patch_loader(monkeypatch, loaded)
    selection = reduce_e3a_selection_from_raw(
        registry=registry,
        manifest=manifest,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        confirmation_data_visible=False,
    )
    assert selection.concurrency == 4
    assert selection.width == 8
    authority = bind_e3a_selection_reduction_authority(
        registry=registry,
        manifest=manifest,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    assert authority.revalidate() == selection
    assert authority.selection_sha256 == selection.sha256


def test_e3a_formal_reduction_blocks_without_release_trusted_native_attester(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry: ExperimentRegistry,
    inventory: GpuInventory,
    hardware_envelope: HardwareEnvelope,
) -> None:
    manifest, loaded, runtime_sha256, split_sha256 = _e3a_fixture(tmp_path, registry)
    _patch_loader(monkeypatch, loaded, trust_native=False)
    with pytest.raises(
        SelectionReductionAuthorityUnavailableError,
        match="trusted_hardware_attester_unavailable",
    ):
        reduce_e3a_selection_from_raw(
            registry=registry,
            manifest=manifest,
            hardware_envelope=hardware_envelope,
            inventory=inventory,
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
            confirmation_data_visible=False,
        )


def test_native_terminal_authority_rejects_parquet_output_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry: ExperimentRegistry,
) -> None:
    from lightcone_spec.experiments import selection_authority

    runtime_sha256 = _sha("native-runtime")
    split_sha256 = _sha("native-split")
    cell = next(
        row for row in registry.cells_for("E3a") if row.identity.method == "static"
    )
    loaded = _loaded(
        cell,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        index=0,
        goodput=100.0,
    )
    run = loaded.run_rows[0]
    policy_sha256 = _sha("native-policy")
    terminal_sha256 = _sha("native-terminal-semantic")
    native_path = (tmp_path / "selection.native-terminal.json").resolve()
    native_body = b"{}\n"
    native_raw_sha256 = hashlib.sha256(native_body).hexdigest()
    native_path.write_bytes(native_body)
    Path(f"{native_path}.sha256").write_text(f"{native_raw_sha256}\n", encoding="utf-8")
    terminal = {
        "run_id": run["run_id"],
        "rank": 0,
        "native_terminal_artifact": {
            "path": native_path.name,
            "size": len(native_body),
            "raw_sha256": native_raw_sha256,
            "terminal_sha256": terminal_sha256,
            "trusted_attester_policy_sha256": policy_sha256,
        },
    }
    terminal_body = json.dumps(terminal, sort_keys=True).encode("utf-8")
    terminal_path = (tmp_path / "selection.complete.json").resolve()
    terminal_path.write_bytes(terminal_body)
    terminal_raw_sha256 = hashlib.sha256(terminal_body).hexdigest()
    Path(f"{terminal_path}.sha256").write_text(
        f"{terminal_raw_sha256}\n", encoding="utf-8"
    )
    reference = _reference(tmp_path, cell.cell_id)
    reference = replace(
        reference,
        terminal_receipts=(
            BoundArtifact(path=terminal_path, sha256=terminal_raw_sha256),
        ),
    )
    bound_run = {
        **run,
        "native_terminal_artifact_path": native_path.name,
        "native_terminal_artifact_size": len(native_body),
        "native_terminal_raw_sha256": native_raw_sha256,
        "native_terminal_sha256": terminal_sha256,
        "trusted_attester_policy_sha256": policy_sha256,
    }
    exact_expectation = SimpleNamespace(
        request_id="request-000",
        input_token_ids=(7,),
        output_token_ids=(101, 102),
        terminal_status="completed",
        submitted_to_server=True,
    )
    binding = SimpleNamespace(
        run_id=run["run_id"],
        run_nonce_sha256=run["run_nonce_sha256"],
        execution_plan_sha256=runtime_sha256,
        rank_config_sha256=run["rank_config_sha256"],
        method=cell.identity.method,
        scored_request_ids=("request-000",),
    )
    validated = SimpleNamespace(
        binding=binding,
        requests=(exact_expectation,),
        trusted_attestation=True,
        terminal_sha256=terminal_sha256,
    )
    monkeypatch.setattr(
        selection_authority,
        "RELEASE_TRUSTED_ATTESTER_POLICY",
        SimpleNamespace(release_ready=True, sha256=policy_sha256),
    )
    monkeypatch.setattr(
        selection_authority,
        "validate_native_terminal_artifact",
        lambda *_args, **_kwargs: validated,
    )
    exact_loaded = replace(loaded, run_rows=(bound_run,))
    assert (
        selection_authority._validate_native_terminal_authority(
            {cell.cell_id: exact_loaded}, references=(reference,)
        )[0]["raw_sha256"]
        == native_raw_sha256
    )

    tampered_loaded = replace(
        exact_loaded,
        request_rows=(_request_row(goodput=100.0, token_ids=(101, 999)),),
    )
    with pytest.raises(ValueError, match="outcome differs from Parquet"):
        selection_authority._validate_native_terminal_authority(
            {cell.cell_id: tampered_loaded}, references=(reference,)
        )


def test_e3a_reducer_rejects_coverage_confirmation_and_token_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry: ExperimentRegistry,
    inventory: GpuInventory,
    hardware_envelope: HardwareEnvelope,
) -> None:
    manifest, loaded, runtime_sha256, split_sha256 = _e3a_fixture(tmp_path, registry)
    _patch_loader(monkeypatch, loaded)
    kwargs = {
        "registry": registry,
        "manifest": manifest,
        "hardware_envelope": hardware_envelope,
        "inventory": inventory,
        "runtime_sha256": runtime_sha256,
        "split_sha256": split_sha256,
    }
    with pytest.raises(ValueError, match="confirmation"):
        reduce_e3a_selection_from_raw(**kwargs, confirmation_data_visible=True)
    incomplete = RawE3aSelectionEvidenceManifest(
        schema_version=2,
        cells=manifest.cells[:-1],
    )
    with pytest.raises(ValueError, match="exactly cover"):
        reduce_e3a_selection_from_raw(
            **{**kwargs, "manifest": incomplete},
            confirmation_data_visible=False,
        )

    static = next(
        cell for cell in registry.cells_for("E3a") if cell.identity.method == "static"
    )
    original = loaded[static.cell_id]
    loaded[static.cell_id] = replace(
        original,
        request_rows=(
            _request_row(
                goodput=_e3a_goodput(static),
                token_ids=(101, 999),
            ),
        ),
    )
    with pytest.raises(ValueError, match="token trajectories"):
        reduce_e3a_selection_from_raw(**kwargs, confirmation_data_visible=False)


def _e3a_selection_and_receipt(
    registry: ExperimentRegistry,
) -> tuple[SealedE3aSelection, ExperimentReceipt]:
    selection = SealedE3aSelection(
        schema_version=1,
        registry_sha256=registry.sha256,
        runtime_sha256=_sha("e3a-source-runtime"),
        split_sha256=_sha("e3a-source-split"),
        width=8,
        concurrency=4,
        reducer_evidence_sha256=_sha("e3a-source-evidence"),
    )
    outputs = {
        name: _sha(f"E3a-{name}") for name in registry.definition("E3a").locked_outputs
    }
    outputs["matched_width"] = selection.matched_width_output_sha256
    outputs["e1_reference_load"] = selection.reference_load_output_sha256
    receipt = ExperimentReceipt(
        experiment="E3a",
        registry_sha256=registry.sha256,
        runtime_sha256=selection.runtime_sha256,
        split_sha256=selection.split_sha256,
        completed_cells_sha256=_sha("e3a-completed"),
        dependency_receipts=(
            LockedOutput(
                name="preflight",
                content_sha256=_sha("preflight-receipt"),
            ),
        ),
        outputs=tuple(
            LockedOutput(name=name, content_sha256=outputs[name])
            for name in sorted(outputs)
        ),
    )
    return selection, receipt


def _e1_fixture(
    tmp_path: Path,
    registry: ExperimentRegistry,
) -> tuple[
    object,
    SealedE3aSelection,
    ExperimentReceipt,
    RawE1ParetoEvidenceManifest,
    dict[str, _LoadedCell],
    str,
]:
    selection, receipt = _e3a_selection_and_receipt(registry)
    activation = reduce_e1_activation(
        registry,
        e3a_receipt=receipt,
        selection=selection,
    )
    cells_by_id = {cell.cell_id: cell for cell in registry.cells_for("E1")}
    active = tuple(
        sorted(
            (cells_by_id[cell_id] for cell_id in activation.plan.activated_cell_ids),
            key=lambda row: row.cell_id,
        )
    )
    geometries = sorted(
        {
            E1GeometryIdentity.from_cell(cell).sha256: E1GeometryIdentity.from_cell(
                cell
            )
            for cell in active
            if cell.identity.method in {"tts", "l0"}
        }.values(),
        key=lambda row: row.sha256,
    )
    winner_sha256 = geometries[0].sha256
    loaded: dict[str, _LoadedCell] = {}
    for index, cell in enumerate(active):
        if cell.identity.method in {"target_only", "static"}:
            goodput = 100.0
            hbm = 500
            exposed = 0.0
        else:
            geometry = E1GeometryIdentity.from_cell(cell)
            winner = geometry.sha256 == winner_sha256
            goodput = 130.0 if winner else 110.0
            hbm = 100 if winner else 200
            exposed = 0.5 if winner else 1.0
        loaded[cell.cell_id] = _loaded(
            cell,
            runtime_sha256=activation.plan.runtime_sha256,
            split_sha256=activation.plan.split_sha256,
            index=index,
            goodput=goodput,
            peak_hbm_bytes=hbm,
            exposed_update_ms=exposed,
        )
    manifest = RawE1ParetoEvidenceManifest(
        schema_version=2,
        cells=tuple(_reference(tmp_path, cell.cell_id) for cell in active),
    )
    return (
        activation,
        selection,
        receipt,
        manifest,
        loaded,
        winner_sha256,
    )


def test_e1_reducer_replays_activation_and_computes_four_objective_pareto(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry: ExperimentRegistry,
    inventory: GpuInventory,
    hardware_envelope: HardwareEnvelope,
) -> None:
    activation, selection, receipt, manifest, loaded, winner_sha256 = _e1_fixture(
        tmp_path, registry
    )
    _patch_loader(monkeypatch, loaded)
    source_authority = _sha("source-activation-authority")
    artifact = reduce_e1_pareto_from_raw(
        registry=registry,
        activation=activation,
        e3a_receipt=receipt,
        e3a_selection=selection,
        source_activation_authority_sha256=source_authority,
        manifest=manifest,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        confirmation_data_visible=False,
    )
    assert tuple(row.sha256 for row in artifact.surviving_geometries) == (
        winner_sha256,
    )
    assert artifact.e1_activation_sha256 == activation.sha256
    authority = bind_e1_pareto_reduction_authority(
        registry=registry,
        activation=activation,
        e3a_receipt=receipt,
        e3a_selection=selection,
        source_activation_authority_sha256=source_authority,
        manifest=manifest,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
    )
    assert authority.revalidate() == artifact
    assert authority.pareto_sha256 == artifact.sha256


def test_e1_unsafe_adaptive_geometry_is_negative_not_global_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry: ExperimentRegistry,
    inventory: GpuInventory,
    hardware_envelope: HardwareEnvelope,
) -> None:
    activation, selection, receipt, manifest, loaded, winner_sha256 = _e1_fixture(
        tmp_path, registry
    )
    winner_cell = next(
        cell
        for cell in registry.cells_for("E1")
        if cell.cell_id in loaded
        and cell.identity.method in {"tts", "l0"}
        and E1GeometryIdentity.from_cell(cell).sha256 == winner_sha256
        and cell.identity.optimizer == E1_OPTIMIZER_ANCHORS[0]
    )
    current = loaded[winner_cell.cell_id]
    performance = dict(current.performance_rows_by_rank[0][0])
    performance["exactness_violations"] = 1
    loaded[winner_cell.cell_id] = replace(
        current,
        performance_rows_by_rank=((performance,),),
    )
    _patch_loader(monkeypatch, loaded)
    artifact = reduce_e1_pareto_from_raw(
        registry=registry,
        activation=activation,
        e3a_receipt=receipt,
        e3a_selection=selection,
        source_activation_authority_sha256=_sha("source-activation-authority"),
        manifest=manifest,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        confirmation_data_visible=False,
    )
    assert winner_sha256 not in {row.sha256 for row in artifact.surviving_geometries}
    assert artifact.surviving_geometries


def test_e1_reducer_rejects_forged_activation_and_downstream_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry: ExperimentRegistry,
    inventory: GpuInventory,
    hardware_envelope: HardwareEnvelope,
) -> None:
    activation, selection, receipt, manifest, loaded, _ = _e1_fixture(
        tmp_path, registry
    )
    _patch_loader(monkeypatch, loaded)
    kwargs = {
        "registry": registry,
        "e3a_receipt": receipt,
        "e3a_selection": selection,
        "source_activation_authority_sha256": _sha("source-activation-authority"),
        "manifest": manifest,
        "hardware_envelope": hardware_envelope,
        "inventory": inventory,
    }
    with pytest.raises(ValueError, match="downstream"):
        reduce_e1_pareto_from_raw(
            **kwargs,
            activation=activation,
            confirmation_data_visible=True,
        )
    forged = replace(activation, reducer_protocol_sha256=_sha("forged-reducer"))
    with pytest.raises(ValueError, match="raw-authority replay"):
        reduce_e1_pareto_from_raw(
            **kwargs,
            activation=forged,
            confirmation_data_visible=False,
        )
