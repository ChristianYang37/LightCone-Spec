from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments import formal_failure_execution as failure
from lightcone_spec.experiments import formal_registry as registry
from lightcone_spec.experiments import (
    formal_single_operator_e0_compatibility as e0_compatibility,
)
from lightcone_spec.experiments import (
    formal_single_operator_prepared_launch as prepared_launch,
)
from lightcone_spec.experiments import (
    formal_single_operator_prepared_launch_producer as prepared_producer,
)
from lightcone_spec.experiments import (
    formal_single_operator_prerequisite_launch_producer as prerequisite,
)
from lightcone_spec.experiments import (
    formal_single_operator_run_dispatch as run_dispatch,
)
from lightcone_spec.experiments import formal_single_operator_stages as stages
from lightcone_spec.experiments.stage_materialization import MaterializedCell
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


class _ReachedTypedBoundary(RuntimeError):
    pass


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _canonical(tmp_path: Path, name: str, value: object) -> CanonicalJsonProofBinding:
    path = (tmp_path / f"{name}.json").resolve()
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _formal(tmp_path: Path, name: str, value: object) -> object:
    return stages.publish_formal_single_operator_json_artifact(
        (tmp_path / f"{name}.json").resolve(),
        value,
    )


def test_real_formal_json_bindings_reach_every_prepared_execution_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the production call sites that require semantic reopen labels.

    Expensive scientific producers are stopped immediately after each typed
    execution-source materialization/ProtocolLock replay.  The path bindings
    and the production entry points themselves are real; a regression to a
    naked ``FormalSingleOperatorJsonBinding.reopen()`` fails before any marker.
    """

    materialization_binding = _formal(
        tmp_path,
        "materialization",
        {"kind": "typed-materialization-source"},
    )
    protocol_binding = _formal(
        tmp_path,
        "protocol-lock",
        {"kind": "typed-protocol-lock-source"},
    )
    auxiliary_binding = _formal(
        tmp_path,
        "e0-compatibility",
        {"kind": "typed-e0-compatibility-source"},
    )
    source_sha256 = _sha("execution-source")
    execution_binding = _canonical(
        tmp_path,
        "execution-source",
        {"execution_source_sha256": source_sha256},
    )
    content_binding = _canonical(
        tmp_path,
        "trusted-content",
        {"kind": "trusted-content"},
    )
    content_source = SimpleNamespace(
        absolute_path=content_binding.absolute_path,
        content_sha256=_sha("trusted-content"),
    )
    protocol_lock = SimpleNamespace(
        sha256=_sha("protocol-lock"),
        content_source_mode="trusted_single_operator",
        trusted_single_operator_content_bundle_sha256=content_source.content_sha256,
    )
    materialization = SimpleNamespace(cells=(), sha256=_sha("materialization"))
    source = SimpleNamespace(
        sha256=source_sha256,
        node="e5_final",
        stage="E5",
        phase="final",
        materialization_source=materialization_binding,
        protocol_lock_source=protocol_binding,
        content_source_binding=content_source,
        predecessor_completion_source=SimpleNamespace(
            absolute_path="/typed/predecessor.json"
        ),
        runtime_authority_manifest_sha256=_sha("runtime"),
    )

    monkeypatch.setattr(
        prerequisite,
        "load_formal_single_operator_execution_source",
        lambda _path: source,
    )
    monkeypatch.setattr(
        prerequisite,
        "stage_materialization_receipt_from_dict",
        lambda _value: materialization,
    )
    monkeypatch.setattr(
        prerequisite,
        "protocol_lock_from_dict",
        lambda _value: protocol_lock,
    )
    monkeypatch.setattr(
        prerequisite,
        "_runtime_inputs",
        lambda _source: (_ for _ in ()).throw(
            _ReachedTypedBoundary("prerequisite-publication")
        ),
    )
    prerequisite_root = (tmp_path / "prerequisite-output").resolve()
    prerequisite_root.mkdir()
    repository = (tmp_path / "repository").resolve()
    repository.mkdir()
    base_launch = _canonical(tmp_path, "base-launch", {"kind": "base-launch"})
    with pytest.raises(_ReachedTypedBoundary, match="prerequisite-publication"):
        prerequisite.publish_formal_single_operator_prerequisite_launch_index(
            execution_source_path=execution_binding.absolute_path,
            base_environment_launch_manifest_path=base_launch.absolute_path,
            repository_root=repository,
            private_output_root=prerequisite_root,
        )

    monkeypatch.setattr(
        prepared_producer,
        "load_formal_single_operator_execution_source",
        lambda _path: source,
    )
    monkeypatch.setattr(
        prepared_producer,
        "stage_materialization_receipt_from_dict",
        lambda _value: materialization,
    )
    monkeypatch.setattr(
        prepared_producer,
        "protocol_lock_from_dict",
        lambda _value: protocol_lock,
    )
    monkeypatch.setattr(
        prepared_producer.FormalContentSourceBinding,
        "bind_trusted_single_operator",
        classmethod(lambda _cls, _path: content_source),
    )
    monkeypatch.setattr(
        prepared_producer,
        "_runtime_inputs",
        lambda _content: (_ for _ in ()).throw(_ReachedTypedBoundary("prepared-draft")),
    )
    prepared_root = (tmp_path / "prepared-output").resolve()
    prepared_root.mkdir()
    with pytest.raises(_ReachedTypedBoundary, match="prepared-draft"):
        prepared_producer.prepare_launch_draft(
            execution_source_path=execution_binding.absolute_path,
            content_source_path=content_binding.absolute_path,
            prerequisite_launch_manifest_paths=(base_launch.absolute_path,),
            private_output_root=prepared_root,
        )

    chain_row = SimpleNamespace(decision=SimpleNamespace(payload={}))
    monkeypatch.setattr(
        prepared_launch,
        "_trusted_completion_chain",
        lambda _source: {
            "e3a": chain_row,
            "tts_cal": chain_row,
            "e2_r3": chain_row,
        },
    )
    monkeypatch.setattr(
        registry,
        "protocol_lock_from_dict",
        lambda _value: (_ for _ in ()).throw(
            _ReachedTypedBoundary("prepared-recipe-context")
        ),
    )
    with pytest.raises(_ReachedTypedBoundary, match="prepared-recipe-context"):
        prepared_launch._trusted_chain_recipe_context(source)

    e0_cell = MaterializedCell(
        stage="E0",
        method_role="Compatibility",
        model="Qwen/Qwen3-4B",
        backend="DFLASH",
        task="compatibility_decision",
        publication_policy="decision_only",
        recipe_sha256=None,
        dimensions=(),
    )
    e0_materialization = SimpleNamespace(cells=(e0_cell,))
    e0_source = SimpleNamespace(
        **{
            **vars(source),
            "node": "e0_tuning",
            "stage": "E0",
            "phase": "tuning",
            "auxiliary_source_binding": lambda _kind: auxiliary_binding,
            "reopen_auxiliary_source": lambda _kind: auxiliary_binding.reopen(
                label="test E0 auxiliary"
            ),
        }
    )
    monkeypatch.setattr(
        run_dispatch,
        "route_formal_single_operator_cell",
        lambda **_kwargs: (
            e0_source,
            e0_cell,
            SimpleNamespace(physical_kind="e0_compatibility_decision"),
        ),
    )
    monkeypatch.setattr(
        run_dispatch,
        "stage_materialization_receipt_from_dict",
        lambda _value: e0_materialization,
    )
    monkeypatch.setattr(
        stages,
        "rebuild_formal_single_operator_stage_completion",
        lambda _path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        registry, "protocol_lock_from_dict", lambda _value: protocol_lock
    )
    monkeypatch.setattr(
        stages,
        "FormalSingleOperatorE0CompatibilityActualValidator",
        lambda **_kwargs: (_ for _ in ()).throw(
            _ReachedTypedBoundary("E0-run-dispatch")
        ),
    )
    with pytest.raises(_ReachedTypedBoundary, match="E0-run-dispatch"):
        run_dispatch.revalidate_formal_single_operator_e0_compatibility_decision(
            execution_source_path=execution_binding.absolute_path,
            materialized_cell_id=e0_cell.cell_id,
        )

    failure_cell = SimpleNamespace(cell_id=_sha("failure-cell"))
    failure_entry = SimpleNamespace(
        physical_kind="e5_failure",
        request_schedule_receipt=object(),
    )
    validated = SimpleNamespace(entry=lambda _cell_id: failure_entry)
    failure_source = SimpleNamespace(**{**vars(source), "node": "e5_final"})
    monkeypatch.setattr(
        prepared_launch,
        "revalidate_formal_single_operator_prepared_launch_bundle",
        lambda **_kwargs: validated,
    )
    monkeypatch.setattr(
        run_dispatch,
        "route_formal_single_operator_cell",
        lambda **_kwargs: (
            failure_source,
            failure_cell,
            SimpleNamespace(physical_kind="e5_failure"),
        ),
    )
    monkeypatch.setattr(
        registry,
        "stage_materialization_receipt_from_dict",
        lambda _value: materialization,
    )
    monkeypatch.setattr(
        registry, "protocol_lock_from_dict", lambda _value: protocol_lock
    )
    monkeypatch.setattr(
        failure,
        "_require_exact_e5_final_failure_materialization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            _ReachedTypedBoundary("E5-failure")
        ),
    )
    failure_root = (tmp_path / "failure-output").resolve()
    failure_root.mkdir()
    with pytest.raises(_ReachedTypedBoundary, match="E5-failure"):
        failure.materialize_formal_single_operator_e5_failure_execution_descriptor(
            execution_source_path=execution_binding.absolute_path,
            materialized_cell_id=failure_cell.cell_id,
            prepared_launch_bundle_path=base_launch.absolute_path,
            repository_root=repository,
            private_output_root=failure_root,
            current_ns=1,
        )

    monkeypatch.setattr(
        stages,
        "load_formal_single_operator_execution_source",
        lambda _path: e0_source,
    )
    monkeypatch.setattr(
        registry,
        "stage_materialization_receipt_from_dict",
        lambda _value: e0_materialization,
    )
    monkeypatch.setattr(
        e0_compatibility,
        "revalidate_trusted_e0_compatibility_bundle_value",
        lambda _value: (_ for _ in ()).throw(_ReachedTypedBoundary("E0-authority")),
    )
    with pytest.raises(_ReachedTypedBoundary, match="E0-authority"):
        e0_compatibility.derive_trusted_single_operator_eagle3_execution_authority(
            execution_source_path=execution_binding.absolute_path,
            materialized_cell_id=e0_cell.cell_id,
            compile_launch_manifest_path=base_launch.absolute_path,
        )
