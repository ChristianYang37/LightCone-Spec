from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec.config import RunConfig, run_config_sha256
from lightcone_spec.orchestration import (
    formal_serving_session_group_launch as launch_module,
)
from lightcone_spec.orchestration.formal_serving_session_group import (
    build_formal_serving_session_group_spec,
    formal_serving_session_reuse_exclusion_reason,
    partition_formal_serving_session_groups,
)
from lightcone_spec.orchestration.formal_serving_session_group_launch import (
    publish_formal_serving_resident_group_launch_authority,
    revalidate_formal_serving_resident_group_launch_authority,
)
from lightcone_spec.orchestration.formal_serving_session_group_worker import (
    FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256,
    FormalServingSessionGroupExecutionSpec,
    revalidate_formal_serving_session_group_execution,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_FIXTURE_PATH = Path(__file__).with_name("test_formal_serving_session_group.py")
_SPEC = importlib.util.spec_from_file_location("_group_launch_fixture", _FIXTURE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_GROUP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GROUP)


def _publish(path: Path, value: object) -> CanonicalJsonProofBinding:
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _adaptive_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "lightcone_spec.runtime.compile_runner.CompileLaunchManifest.child_environment",
        lambda self: {
            "PATH": ":".join(self.path_entries),
            "LD_LIBRARY_PATH": ":".join(self.library_path_entries),
            "CUDA_HOME": self.cuda_home,
            "CUDA_PATH": self.cuda_home,
            "CUDA_VISIBLE_DEVICES": ",".join(self.gpu_uuids),
            "LANG": "C",
            "LC_ALL": "C",
        },
    )
    authority_binding, authority = _GROUP._published_authority(
        tmp_path / "authority",
        monkeypatch,
        method_family="lightcone",
    )
    configs = tuple(
        _GROUP._config(
            label=f"adaptive-{index}",
            method="l0",
            adaptation_group_id=f"cell-namespace-{index}",
        )
        for index in range(2)
    )
    launches = tuple(
        _GROUP._producer_generated_launch(
            tmp_path,
            label=f"adaptive-{index}",
            config=config,
            port=29_000 + index,
        )
        for index, config in enumerate(configs)
    )
    specs = tuple(
        _GROUP._group_spec(
            tmp_path,
            index=900 + index,
            config=config,
            launch=launch,
            method_family="lightcone",
            source_snapshot_sha256=authority.source_snapshot_sha256,
            protocol_lock_sha256=authority.protocol_lock_sha256,
            inventory_sha256=authority.inventory_sha256,
        )
        for index, (config, launch) in enumerate(zip(configs, launches, strict=True))
    )
    plans = partition_formal_serving_session_groups(
        specs,
        reset_authorities=(authority_binding,),
        max_member_count=2,
        max_estimated_duration_seconds=60.0,
    )
    assert len(plans) == 1 and plans[0].execution_mode == "shared_session_tp1"
    plan_path = (tmp_path / "adaptive-group-plan.json").resolve()
    plan_binding = _publish(plan_path, plans[0].to_dict())
    execution_spec = FormalServingSessionGroupExecutionSpec(
        schema_version=1,
        kind="formal_serving_session_group_execution_spec",
        protocol_sha256=FORMAL_SERVING_SESSION_GROUP_EXECUTION_PROTOCOL_SHA256,
        group_plan_path=plan_binding.absolute_path,
        reset_authority_path=authority_binding.absolute_path,
        output_directory=str((tmp_path / "adaptive-execution").resolve()),
        formal_measured=False,
    )
    spec_path = (tmp_path / "adaptive-execution-spec.json").resolve()
    _publish(spec_path, execution_spec.to_dict())
    execution = revalidate_formal_serving_session_group_execution(spec_path)
    launch_bindings = tuple(
        _publish(
            (tmp_path / f"source-launch-{index}.json").resolve(),
            {"kind": "source-launch-fixture", "index": index},
        )
        for index, launch in enumerate(launches)
    )
    registered = tuple(
        (binding, launch, config)
        for binding, launch, config in zip(
            launch_bindings, launches, configs, strict=True
        )
    )
    monkeypatch.setattr(
        launch_module,
        "_registered_launches",
        lambda plan: registered if plan == execution.plan else (),
    )
    return execution, configs, launches


def test_adaptive_group_launch_rebinds_every_cell_namespace_and_deep_reopens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution, configs, launches = _adaptive_execution(tmp_path, monkeypatch)
    result = publish_formal_serving_resident_group_launch_authority(
        execution=execution,
        output_root=(tmp_path / "group-launch").resolve(),
    )
    reopened = revalidate_formal_serving_resident_group_launch_authority(
        result.binding.absolute_path
    )

    assert reopened == result
    assert reopened.run_config.adaptation is not None
    assert reopened.run_config.adaptation.adaptation_group_id == (
        execution.plan.session_adaptation_group_id
    )
    assert reopened.authority.actual_server_argv != launches[0].server_argv
    rendered = "\n".join(
        (
            *reopened.authority.actual_server_argv,
            *(
                f"{key}={value}"
                for key, value in reopened.authority.actual_child_environment
            ),
        )
    )
    assert execution.plan.group_id in rendered
    assert launches[0].run_config_path not in rendered
    assert launches[1].run_config_path not in rendered
    for config in configs:
        assert config.adaptation is not None
        assert config.adaptation.adaptation_group_id not in rendered
    assert reopened.authority.group_run_config.absolute_path in rendered
    assert reopened.authority.group_adaptation_config is not None
    assert reopened.authority.group_adaptation_config.absolute_path in rendered

    Path(reopened.authority.group_run_config.absolute_path).write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed"):
        revalidate_formal_serving_resident_group_launch_authority(
            result.binding.absolute_path
        )


def test_adaptive_backend_without_group_config_support_is_forced_fresh(
    tmp_path: Path,
) -> None:
    config = _GROUP._config(label="eagle3", method="l0")
    unsupported = RunConfig.model_construct(
        **{
            **config.__dict__,
            "model": config.model.model_copy(update={"algorithm": "EAGLE3"}),
        }
    )
    launch = _GROUP._producer_generated_launch(
        tmp_path,
        label="eagle3",
        config=config,
        port=29_100,
    )
    assert (
        formal_serving_session_reuse_exclusion_reason(
            physical_kind="serving",
            launch=launch,
            config=unsupported,
        )
        == "adaptive_runtime_authority_requires_fresh_process"
    )
    launch = replace(
        launch,
        run_config_semantic_sha256=run_config_sha256(unsupported),
    )
    spec = build_formal_serving_session_group_spec(
        node="e3b_final",
        stage="E3b",
        phase="final",
        materialized_cell_id=_GROUP._sha("eagle3-cell"),
        attempt=1,
        physical_kind="serving",
        method_family="lightcone",
        protocol_lock_sha256=_GROUP._sha("protocol"),
        source_snapshot_sha256=_GROUP._sha("source"),
        inventory_sha256=launch.inventory_sha256,
        run_plan=_publish((tmp_path / "eagle3-plan.json").resolve(), {"cell": 1}),
        prepared_launch_entry_sha256=_GROUP._sha("entry"),
        compile_launch_manifest_sha256=_GROUP._sha("launch"),
        request_schedule_sha256=_GROUP._sha("schedule"),
        launch=launch,
        config=unsupported,
        output_directory=str((tmp_path / "eagle3-output").resolve()),
        estimated_duration_seconds=1.0,
        dispatch_order_key=("0001",),
    )
    assert spec.normalized_process_key is None
    assert (
        spec.reuse_exclusion_reason
        == "adaptive_runtime_authority_requires_fresh_process"
    )
