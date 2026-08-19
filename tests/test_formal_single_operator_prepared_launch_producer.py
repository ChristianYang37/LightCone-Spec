from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.config import (
    AdaptationConfig,
    ModelPair,
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
)
from lightcone_spec.experiments import (
    formal_single_operator_prepared_launch_producer as producer,
)
from lightcone_spec.experiments.formal_content_source import (
    FormalContentSourceBinding,
)
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedSingleOperatorContentBundle,
    TrustedSingleOperatorContentBundleBinding,
)
from lightcone_spec.experiments.formal_single_operator_prepared_launch import (
    FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256,
    TRUSTED_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256,
    FormalSingleOperatorPreparedLaunchBundle,
    FormalSingleOperatorPreparedLaunchEntry,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorExecutionSource,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.stage_materialization import MaterializedCell
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.formal_sharded_artifact import (
    load_formal_canonical_sequence_shard_index,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _inventory(size: int = 2) -> GpuInventory:
    uuids = tuple(f"GPU-{index}" for index in range(size))
    group = GpuTopologyGroup(
        group_id="pair",
        host_id="host",
        gpu_uuids=uuids,
        fabric="NVLink",
        bandwidth_class="high",
    )
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
        topology_groups=(group,),
        source_receipt_sha256=_sha("inventory"),
    )


def _e5_cell(*, role: str, block: int, topology: str = "tp1_dp1") -> MaterializedCell:
    return MaterializedCell(
        stage="E5",
        method_role=role,
        model="Qwen/Qwen3-8B",
        backend="NONE" if role == "Target-only" else "DFLASH",
        task="production_slo_power_prefix",
        publication_policy=(
            "fixed_barrier"
            if role == "TTS"
            else "first_ready"
            if role in {"L0-naive", "LightCone"}
            else "none"
        ),
        recipe_sha256=_sha("recipe")
        if role in {"TTS", "L0-naive", "LightCone"}
        else None,
        dimensions=tuple(
            sorted(
                {
                    "backend_authority": "DFLASH",
                    "block": block,
                    "block_phase": "excluded_pilot",
                    "concurrency": 4,
                    "family": "closed_loop",
                    "family_id": "closed_loop_c4",
                    "topology": topology,
                    **(
                        {"tts_l0_pair_id": _sha(f"pair:{block}")}
                        if role in {"TTS", "L0-naive"}
                        else {}
                    ),
                }.items()
            )
        ),
    )


def test_single_gpu_placement_preserves_pairs_and_balances_blocks() -> None:
    inventory = _inventory()
    assignments = {
        block: {
            producer.deterministic_prepared_gpu_assignment(
                inventory=inventory,
                cell=_e5_cell(role=role, block=block),
            )
            for role in ("Target-only", "Static", "TTS", "L0-naive", "LightCone")
        }
        for block in range(16)
    }

    assert all(len(values) == 1 for values in assignments.values())
    used = {next(iter(values)) for values in assignments.values()}
    assert used == {("GPU-0",), ("GPU-1",)}
    assert all(
        next(iter(assignments[block])) == (f"GPU-{block % 2}",) for block in assignments
    )


def test_two_gpu_placement_is_atomic_and_inventory_ordered() -> None:
    assignment = producer.deterministic_prepared_gpu_assignment(
        inventory=_inventory(),
        cell=_e5_cell(role="LightCone", block=0, topology="tp2_dp1"),
    )
    assert assignment == ("GPU-0", "GPU-1")

    with pytest.raises(
        producer.FormalSingleOperatorPreparedLaunchBlocked,
        match="two_gpu_topology_unavailable",
    ):
        producer.deterministic_prepared_gpu_assignment(
            inventory=_inventory(1),
            cell=_e5_cell(role="LightCone", block=0, topology="tp2_dp1"),
        )


def test_target_only_e5_uses_backend_authority_not_none() -> None:
    cell = _e5_cell(role="Target-only", block=0)
    assert producer._cell_backend(cell) == "DFLASH"


def _source(node: str) -> FormalSingleOperatorExecutionSource:
    # Config derivation needs only the already-validated source node after the
    # recipe-context reducer is patched to its typed return below.
    value = object.__new__(FormalSingleOperatorExecutionSource)
    object.__setattr__(value, "node", node)
    return value


def _base_config(*, method: str = "static") -> RunConfig:
    return RunConfig(
        method=method,  # type: ignore[arg-type]
        model=ModelPair(
            target="Qwen/Qwen3-8B",
            drafter="z-lab/Qwen3-8B-DFlash-b16",
            target_revision="1" * 40,
            drafter_revision="2" * 40,
            algorithm="DFLASH",
            draft_depth=15,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256=_sha("sampling"),
            device_identity="GPU-old",
            speculative_num_draft_tokens=16,
            max_running_requests=4,
        ),
    )


def _recipe_context() -> SimpleNamespace:
    return SimpleNamespace(
        matched_width=8,
        common_load=4,
        frozen_tts_recipe_sha256=_sha("tts"),
        tts_learning_rate=1e-5,
        tts_stride=10,
        lightcone_recipe=SimpleNamespace(optimizer="adam", sha256=_sha("lc")),
        dspark_selected_configuration=None,
        dspark_selected_recipe_sha256=None,
        e0_selected_recipes=(),
    )


def _e3b_cell(*, role: str) -> MaterializedCell:
    recipe = _sha("tts") if role in {"TTS", "L0-naive"} else None
    return MaterializedCell(
        stage="E3b",
        method_role=role,
        model="Qwen/Qwen3-8B",
        backend="NONE" if role == "Target-only" else "DFLASH",
        task="heldout_long_context_confirmation",
        publication_policy=(
            "fixed_barrier"
            if role == "TTS"
            else "first_ready"
            if role == "L0-naive"
            else "none"
        ),
        recipe_sha256=recipe,
        dimensions=tuple(
            sorted(
                {
                    "block": 0,
                    "block_phase": "excluded_pilot",
                    "context": 4096,
                    "load": "common_load",
                    "regime": "short_input_long_generation",
                    "width_panel": "matched",
                    **(
                        {"tts_l0_pair_id": _sha("tts-l0")}
                        if role in {"TTS", "L0-naive"}
                        else {}
                    ),
                }.items()
            )
        ),
    )


def test_run_config_derives_target_static_and_tts_l0_without_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _recipe_context()
    monkeypatch.setattr(
        producer, "_trusted_chain_recipe_context", lambda _source: context
    )
    source = _source("e3b_pilot")

    target = producer.derive_prepared_run_config(
        source=source,
        cell=_e3b_cell(role="Target-only"),
        prerequisite=_base_config(),
        gpu_uuids=("GPU-0",),
    )
    static = producer.derive_prepared_run_config(
        source=source,
        cell=_e3b_cell(role="Static"),
        prerequisite=_base_config(),
        gpu_uuids=("GPU-0",),
    )
    tts = producer.derive_prepared_run_config(
        source=source,
        cell=_e3b_cell(role="TTS"),
        prerequisite=_base_config(),
        gpu_uuids=("GPU-0",),
    )
    l0 = producer.derive_prepared_run_config(
        source=source,
        cell=_e3b_cell(role="L0-naive"),
        prerequisite=_base_config(),
        gpu_uuids=("GPU-0",),
    )

    assert target.method == "target_only"
    assert target.runtime.speculation_enabled is False
    assert target.adaptation is None
    assert static.method == "static" and static.adaptation is None
    assert tts.method == "tts" and l0.method == "l0"
    assert tts.adaptation == l0.adaptation
    assert tts.runtime.max_running_requests == 4
    assert tts.runtime.speculative_num_draft_tokens == 8
    assert tts.runtime.device_identity == "GPU-0"


def test_config_rejects_foreign_backend_prerequisite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        producer, "_trusted_chain_recipe_context", lambda _source: _recipe_context()
    )
    foreign = _base_config().model_copy(
        update={
            "model": _base_config().model.model_copy(update={"algorithm": "DSPARK"})
        }
    )
    with pytest.raises(
        producer.FormalSingleOperatorPreparedLaunchBlocked,
        match="compatible_model_backend_topology_prerequisite_missing",
    ):
        producer.derive_prepared_run_config(
            source=_source("e3b_pilot"),
            cell=_e3b_cell(role="Static"),
            prerequisite=RunConfig.model_validate(foreign.model_dump(mode="json")),
            gpu_uuids=("GPU-0",),
        )


def test_profiler_config_is_exact_selected_local_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        producer, "_trusted_chain_recipe_context", lambda _source: _recipe_context()
    )
    selected = RunConfig(
        method="l0",
        model=_base_config().model,
        runtime=_base_config().runtime.model_copy(
            update={
                "adaptation_microbatch_size": 8,
                "adaptation_publication_coalescing": 8,
                "adaptation_stream_priority": "high",
            }
        ),
        adaptation=AdaptationConfig(
            weight_update_mode="full",
            parameter_scope="all",
            adaptation_group_id="selected-headline",
            optimizer=OptimizerConfig(name="adam", learning_rate=1e-5),
            stride=50,
            canvas_tokens=16,
        ),
    )
    selected = RunConfig.model_validate(selected.model_dump(mode="json"))
    cell = MaterializedCell(
        stage="E4",
        method_role="LightCone",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="mechanism_profile_only",
        publication_policy="diagnostic_only",
        recipe_sha256=_sha("lc"),
        dimensions=tuple(
            sorted(
                {
                    "profiler": "nsight_systems",
                    "selected_configuration_sha256": _sha("selected"),
                }.items()
            )
        ),
    )

    actual = producer.derive_prepared_run_config(
        source=_source("e4_profiler"),
        cell=cell,
        prerequisite=selected,
        gpu_uuids=("GPU-0",),
    )
    expected = selected.model_copy(
        update={
            "runtime": selected.runtime.model_copy(
                update={"telemetry_detail": "profile"}
            ),
            "adaptation": selected.adaptation.model_copy(
                update={
                    "adaptation_group_id": (f"formal-single-e4-{cell.cell_id[:24]}")
                }
            ),
        }
    )
    assert actual == RunConfig.model_validate(expected.model_dump(mode="json"))


def test_schedule_phase_derives_e0_path_and_non_circular_identities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundle,
    )
    from lightcone_spec.experiments.formal_single_operator_e0_workloads import (
        E0TaskNativeSourceAuthority,
    )
    from lightcone_spec.orchestration import formal_physical_dispatch

    cell = MaterializedCell(
        stage="E0",
        method_role="Static",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="GSM8K",
        publication_policy="none",
        recipe_sha256=None,
        dimensions=tuple(
            sorted(
                {
                    "block": 0,
                    "block_phase": "excluded_pilot",
                    "compatibility_decision_id": _sha("decision"),
                    "load": "concurrency_one",
                }.items()
            )
        ),
    )
    workload_path = (tmp_path / "e0-authority.json").resolve()
    workload_path.write_text("{}", encoding="utf-8")
    descriptor = SimpleNamespace(
        task="GSM8K",
        source=SimpleNamespace(absolute_path=str(workload_path)),
    )
    content = object.__new__(TrustedSingleOperatorContentBundle)
    object.__setattr__(content, "locked_workloads", ())
    object.__setattr__(content, "e0_task_native_descriptors", (descriptor,))
    content_binding = SimpleNamespace(reopen=lambda: content)
    entry = SimpleNamespace(
        materialized_cell_id=cell.cell_id,
        physical_kind="serving",
        compile_launch_manifest=SimpleNamespace(
            absolute_path=str((tmp_path / "launch.json").resolve())
        ),
    )
    draft = SimpleNamespace(
        entries=(entry,),
        materialization=SimpleNamespace(reopen=dict),
        content_source_binding=content_binding,
        execution_source=SimpleNamespace(
            absolute_path=str((tmp_path / "source.json").resolve())
        ),
    )
    workload = object.__new__(E0TaskNativeSourceAuthority)
    object.__setattr__(workload, "task", "GSM8K")
    object.__setattr__(workload, "support_status", "READY")
    execution_id, subject_id = _sha("execution"), _sha("subject")
    captured: dict[str, object] = {}

    monkeypatch.setattr(producer, "load_prepared_launch_draft", lambda _path: draft)
    monkeypatch.setattr(
        producer,
        "_prepared_draft_entries",
        lambda _draft, **_kwargs: (entry,),
    )
    monkeypatch.setattr(
        producer,
        "stage_materialization_receipt_from_dict",
        lambda _value: SimpleNamespace(cells=(cell,)),
    )
    monkeypatch.setattr(
        producer,
        "expected_schedule_identities",
        lambda **_kwargs: (execution_id, subject_id),
    )
    monkeypatch.setattr(
        formal_physical_dispatch, "_workload_id_for_cell", lambda _cell: "GSM8K"
    )
    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_single_operator_e0_workloads."
        "load_e0_task_native_source_authority",
        lambda _path: workload,
    )

    receipt = SimpleNamespace(
        materialized_cell_id=cell.cell_id,
        execution_binding_sha256=execution_id,
        subject_sha256=subject_id,
        content_source_binding=content_binding,
    )

    def materialize(**kwargs: object) -> object:
        captured.update(kwargs)
        return receipt

    monkeypatch.setattr(
        formal_physical_dispatch,
        "materialize_trusted_single_operator_request_schedule",
        materialize,
    )

    actual = producer.materialize_prepared_request_schedule(
        draft_path=(tmp_path / "draft.json").resolve(),
        materialized_cell_id=cell.cell_id,
        private_output_root=tmp_path.resolve(),
    )

    assert actual is receipt
    assert captured["workload_source_path"] == workload_path
    assert captured["execution_binding_sha256"] == execution_id
    assert captured["subject_sha256"] == subject_id
    assert captured["e5_arrival_plan_path"] is None
    assert captured["context_filler_artifact_path"] is None


def test_context_filler_is_published_once_per_draft_tokenizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lightcone_spec.experiments import formal_single_operator_context_artifact

    draft_path = (tmp_path / "prepared-launch-draft.json").resolve()
    publish_canonical_json_no_replace(draft_path, {"draft": True})
    launch_path = (tmp_path / "compile-launch.json").resolve()
    publish_canonical_json_no_replace(launch_path, {"launch": True})
    member_id = _sha("context-tokenizer-member")
    launch = SimpleNamespace(
        tokenizer_content_member_id=member_id,
        tokenizer_model_id="Qwen/Qwen3-8B",
        tokenizer_revision="1" * 40,
    )
    row = SimpleNamespace(
        compile_launch_manifest=SimpleNamespace(absolute_path=str(launch_path))
    )
    content_source = object()
    draft = SimpleNamespace(content_source_binding=content_source)
    published: list[Path] = []
    loaded: list[Path] = []
    monkeypatch.setattr(CompileLaunchManifest, "load", lambda _path: launch)

    def materialize(**kwargs: object) -> CanonicalJsonProofBinding:
        root = Path(kwargs["output_directory"])
        artifact = root / "context-filler-authority.json"
        publish_canonical_json_no_replace(artifact, {"filler": True})
        published.append(artifact)
        return CanonicalJsonProofBinding.bind(artifact)

    def load(path: str, **kwargs: object) -> object:
        assert kwargs["content_source_binding"] is content_source
        loaded.append(Path(path))
        return object()

    monkeypatch.setattr(
        formal_single_operator_context_artifact,
        "materialize_trusted_context_filler_artifact",
        materialize,
    )
    monkeypatch.setattr(
        formal_single_operator_context_artifact,
        "load_trusted_context_filler_artifact",
        load,
    )

    first = producer._materialize_context_filler_for_cell(
        draft_path=draft_path,
        draft=draft,
        row=row,
        cell=_e3b_cell(role="Static"),
    )
    second = producer._materialize_context_filler_for_cell(
        draft_path=draft_path,
        draft=draft,
        row=row,
        cell=_e3b_cell(role="Static"),
    )

    expected = (
        tmp_path
        / "context-filler-artifacts"
        / member_id
        / "context-filler-authority.json"
    ).resolve()
    assert first == second == CanonicalJsonProofBinding.bind(expected)
    assert published == [expected]
    assert loaded == [expected, expected]


def test_controlled_context_schedule_receives_draft_owned_filler_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lightcone_spec.experiments import workload_authority
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundle,
    )
    from lightcone_spec.orchestration import formal_physical_dispatch

    cell = _e3b_cell(role="Static")
    raw_path = (tmp_path / "locked-raw.jsonl").resolve()
    raw_path.write_text("{}\n", encoding="utf-8")
    member = SimpleNamespace(
        workload_id="livecodebench_v6_hard",
        raw_source_path=str(raw_path),
        authority_sha256=_sha("workload-authority"),
    )
    content = object.__new__(TrustedSingleOperatorContentBundle)
    object.__setattr__(content, "locked_workloads", (member,))
    object.__setattr__(content, "e0_task_native_descriptors", ())
    content_binding = SimpleNamespace(reopen=lambda: content)
    launch_path = (tmp_path / "launch.json").resolve()
    launch_path.write_text("{}", encoding="utf-8")
    entry = SimpleNamespace(
        materialized_cell_id=cell.cell_id,
        physical_kind="serving",
        compile_launch_manifest=SimpleNamespace(absolute_path=str(launch_path)),
    )
    draft = SimpleNamespace(
        entries=(entry,),
        materialization=SimpleNamespace(reopen=dict),
        content_source_binding=content_binding,
        execution_source=SimpleNamespace(
            absolute_path=str((tmp_path / "source.json").resolve())
        ),
    )
    authority = SimpleNamespace(sha256=member.authority_sha256)
    filler_path = (tmp_path / "context-filler.json").resolve()
    publish_canonical_json_no_replace(filler_path, {"filler": True})
    filler_binding = CanonicalJsonProofBinding.bind(filler_path)
    execution_id, subject_id = _sha("execution"), _sha("subject")
    captured: dict[str, object] = {}
    monkeypatch.setattr(producer, "load_prepared_launch_draft", lambda _path: draft)
    monkeypatch.setattr(
        producer, "_prepared_draft_entries", lambda _draft, **_kwargs: (entry,)
    )
    monkeypatch.setattr(
        producer,
        "stage_materialization_receipt_from_dict",
        lambda _value: SimpleNamespace(cells=(cell,)),
    )
    monkeypatch.setattr(
        producer,
        "expected_schedule_identities",
        lambda **_kwargs: (execution_id, subject_id),
    )
    monkeypatch.setattr(
        producer,
        "_materialize_context_filler_for_cell",
        lambda **_kwargs: filler_binding,
    )
    monkeypatch.setattr(
        formal_physical_dispatch,
        "_workload_id_for_cell",
        lambda _cell: "livecodebench_v6_hard",
    )
    monkeypatch.setattr(
        workload_authority,
        "bind_formal_workload_authority",
        lambda _workload, _path: authority,
    )
    monkeypatch.setattr(
        workload_authority,
        "formal_workload_authority_cli_artifact",
        lambda _authority: {"workload": True},
    )
    receipt = SimpleNamespace(
        materialized_cell_id=cell.cell_id,
        execution_binding_sha256=execution_id,
        subject_sha256=subject_id,
        content_source_binding=content_binding,
    )

    def materialize(**kwargs: object) -> object:
        captured.update(kwargs)
        return receipt

    monkeypatch.setattr(
        formal_physical_dispatch,
        "materialize_trusted_single_operator_request_schedule",
        materialize,
    )

    assert (
        producer.materialize_prepared_request_schedule(
            draft_path=(tmp_path / "draft.json").resolve(),
            materialized_cell_id=cell.cell_id,
            private_output_root=tmp_path.resolve(),
        )
        is receipt
    )
    assert captured["context_filler_artifact_path"] == str(filler_path)


def test_sharded_bundle_publisher_scales_to_ten_thousand_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_path = (tmp_path / "fixture.json").resolve()
    publish_canonical_json_no_replace(fixture_path, {"fixture": True})
    fixture_binding = CanonicalJsonProofBinding.bind(fixture_path)

    content_path = (tmp_path / "content.json").resolve()
    publish_canonical_json_no_replace(content_path, {"fixture": "content"})
    content_semantic = _sha("trusted-content")
    fake_content = object.__new__(TrustedSingleOperatorContentBundle)
    object.__setattr__(fake_content, "runtime_binding_status", "BOUND")
    object.__setattr__(fake_content, "semantic_sha256", content_semantic)
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: fake_content,
    )
    trusted_binding = TrustedSingleOperatorContentBundleBinding(
        absolute_path=str(content_path),
        size=content_path.stat().st_size,
        raw_sha256=hashlib.sha256(content_path.read_bytes()).hexdigest(),
        semantic_sha256=content_semantic,
        runtime_binding_status="BOUND",
    )
    content_source = FormalContentSourceBinding(
        schema_version=1,
        kind="formal_content_source_binding",
        mode="trusted_single_operator",
        offline_root_signed=None,
        trusted_single_operator=trusted_binding,
    )

    cell_ids = tuple(sorted(_sha(f"scale-cell-{index:05d}") for index in range(10_000)))
    entries = tuple(
        FormalSingleOperatorPreparedLaunchEntry(
            schema_version=1,
            kind="formal_single_operator_prepared_launch_entry",
            protocol_sha256=(
                FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256
            ),
            materialized_cell_id=cell_id,
            physical_kind="serving",
            run_config=fixture_binding,
            compile_launch_manifest=fixture_binding,
            request_schedule_receipt=fixture_binding,
            launch_compatibility_key_sha256=_sha(f"compatibility:{cell_id}"),
            target_content_member_id="target-member",
            drafter_content_member_id="drafter-member",
            tokenizer_content_member_id="tokenizer-member",
            inventory_sha256=_sha("inventory"),
            topology_mode="tp1_dp1",
            gpu_uuids=("GPU-0",),
            server_argv_sha256=_sha(f"argv:{cell_id}"),
            profiler_subject=None,
        )
        for cell_id in cell_ids
    )
    bundle = FormalSingleOperatorPreparedLaunchBundle(
        schema_version=2,
        kind="formal_single_operator_prepared_launch_bundle",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_PREPARED_LAUNCH_BUNDLE_PROTOCOL_SHA256
        ),
        node="e5_final",
        stage="E5",
        phase="final",
        execution_source=fixture_binding,
        execution_source_sha256=_sha("execution-source"),
        protocol_lock_sha256=_sha("protocol-lock"),
        materialization_sha256=_sha("materialization"),
        materialization_source_decision_sha256=_sha("source-decision"),
        inventory=fixture_binding,
        content_verification_receipt=None,
        entries=entries,
        content_source_binding=content_source,
    )
    output = (tmp_path / "prepared-launch.json").resolve()

    sharded = producer.publish_sharded_prepared_launch_bundle(
        bundle=bundle,
        output_path=output,
    )

    assert sharded.schema_version == 3
    assert sharded.entries == ()
    assert sharded.entries_shard_index is not None
    assert output.stat().st_size < 2 * 1024 * 1024
    assert sharded.entries_shard_index.size < 2 * 1024 * 1024
    index = load_formal_canonical_sequence_shard_index(
        sharded.entries_shard_index.absolute_path
    )
    assert index.total_rows == 10_000
    assert index.shard_count > 1
    assert all(row.binding.size < 2 * 1024 * 1024 for row in index.shards)

    calls: list[int] = []
    reference_type = type(index.shards[0])
    original_reopen = reference_type.reopen

    def counted_reopen(self: object, **kwargs: object) -> object:
        calls.append(self.shard_ordinal)  # type: ignore[attr-defined]
        return original_reopen(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(reference_type, "reopen", counted_reopen)
    selected = FormalSingleOperatorPreparedLaunchEntry.from_dict(index.row_at(9_999))
    assert selected.materialized_cell_id == cell_ids[9_999]
    assert len(calls) == 1
    assert index.revalidate().total_rows == 10_000

    with pytest.raises(FileExistsError, match="already exists"):
        producer.publish_sharded_prepared_launch_bundle(
            bundle=bundle,
            output_path=output,
        )


def test_protocol_digest_and_draft_schema_are_stable() -> None:
    assert producer.FORMAL_SINGLE_OPERATOR_PREPARED_LAUNCH_PRODUCER_PROTOCOL_SHA256 == (
        "ccd141d20bd004397ebc6d5753a4e48e5c4e62b03739b78bfc9ea6987b46d92e"
    )
    assert set(producer.PreparedLaunchDraftEntry.__dataclass_fields__) >= {
        "run_config",
        "compile_cache_plan",
        "prewarm_manifest",
        "sampling_profile",
        "compile_launch_manifest",
        "schedule_state",
    }
