from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_formal_single_operator_stages import _protocol_lock, _trusted_protocol_lock

from lightcone_spec.experiments import (
    formal_single_operator_protocol_lock as lock_module,
)
from lightcone_spec.experiments.formal_protocol import (
    TrustedSingleOperatorProtocolSourceBinding,
    TrustedSingleOperatorProtocolSourceBindings,
)
from lightcone_spec.experiments.formal_registry import (
    protocol_lock_from_dict,
    protocol_lock_to_dict,
)
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedSingleOperatorContentBundleBinding,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    materialize_formal_single_operator_node,
    publish_formal_single_operator_json_artifact,
)
from lightcone_spec.orchestration.experiment_operator import ExperimentOperatorStore
from lightcone_spec.orchestration.formal_single_operator_dag_driver import (
    DriverFileBinding,
    PathBoundFormalDagDriverConfig,
    ProductionFormalDagCallbackBuilder,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def test_legacy_schema4_lock_codec_omits_trusted_source_bindings() -> None:
    lock = _protocol_lock()
    encoded = protocol_lock_to_dict(lock)

    assert "trusted_single_operator_source_bindings" not in encoded
    assert protocol_lock_from_dict(encoded) == lock


def test_trusted_lock_builder_reopens_tts_before_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts_path = (tmp_path / "tts-source.json").resolve()
    publish_canonical_json_no_replace(tts_path, {"kind": "test-tts-source"})
    calls: list[str] = []

    def load_tts(_path: str) -> SimpleNamespace:
        calls.append("tts")
        return SimpleNamespace()

    class ContentBindObserved(RuntimeError):
        pass

    def bind_content(
        _cls: type[TrustedSingleOperatorContentBundleBinding],
        path: str | Path,
    ) -> None:
        calls.append("content")
        raise ContentBindObserved(str(path))

    monkeypatch.setattr(
        lock_module,
        "load_tts_calibration_authority_artifact",
        load_tts,
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "bind",
        classmethod(bind_content),
    )

    with pytest.raises(ContentBindObserved):
        lock_module.build_trusted_single_operator_protocol_lock(
            protocol_id="tts-before-content-order-test",
            trusted_content_bundle_path=tmp_path / "unused-content.json",
            formal_runtime_authority_manifest_path=tmp_path / "unused-runtime.json",
            tts_calibration_authority_path=tts_path,
            chronobelief_authority_path=tmp_path / "unused-chronobelief.json",
            e1_recipe_anchor_authority_path=tmp_path / "unused-e1.json",
        )

    assert calls == ["tts", "content"]


def test_lock_builder_forwards_identity_only_runtime_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import (
        formal_single_operator_model_registry as models,
    )

    tts_path = (tmp_path / "tts-source.json").resolve()
    publish_canonical_json_no_replace(tts_path, {"kind": "test-tts-source"})
    bundle = SimpleNamespace()
    observed: list[tuple[object, bool, bool]] = []

    monkeypatch.setattr(
        lock_module,
        "load_tts_calibration_authority_artifact",
        lambda _path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "bind",
        classmethod(
            lambda _cls, _path: SimpleNamespace(
                runtime_binding_status="BOUND",
                reopen=lambda: bundle,
            )
        ),
    )

    def require_bound(
        value,
        *,
        require_capacity_available=True,
        revalidate_runtime_observations=True,
    ) -> None:
        observed.append(
            (
                value,
                require_capacity_available,
                revalidate_runtime_observations,
            )
        )
        raise RuntimeError("observed runtime policy")

    monkeypatch.setattr(
        models, "require_formal_v03_bound_content_bundle", require_bound
    )

    with pytest.raises(RuntimeError, match="observed runtime policy"):
        lock_module.build_trusted_single_operator_protocol_lock(
            protocol_id="identity-only-policy-test",
            trusted_content_bundle_path=tmp_path / "unused-content.json",
            formal_runtime_authority_manifest_path=tmp_path / "unused-runtime.json",
            tts_calibration_authority_path=tts_path,
            chronobelief_authority_path=tmp_path / "unused-chronobelief.json",
            e1_recipe_anchor_authority_path=tmp_path / "unused-e1.json",
            require_capacity_available=False,
            revalidate_runtime_observations=False,
        )

    assert observed == [(bundle, False, False)]


def test_lock_builder_rejects_tts_authority_bound_to_foreign_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import (
        formal_single_operator_model_registry as models,
    )

    def content_binding(label: str) -> TrustedSingleOperatorContentBundleBinding:
        path = (tmp_path / f"{label}.json").resolve()
        publish_canonical_json_no_replace(path, {"kind": label})
        raw = CanonicalJsonProofBinding.bind(path)
        return TrustedSingleOperatorContentBundleBinding(
            absolute_path=raw.absolute_path,
            size=raw.size,
            raw_sha256=raw.raw_sha256,
            semantic_sha256=raw.semantic_sha256,
            runtime_binding_status="BOUND",
        )

    current = content_binding("current-content")
    foreign = content_binding("foreign-content")
    tts_path = (tmp_path / "tts-source.json").resolve()
    publish_canonical_json_no_replace(tts_path, {"kind": "test-tts-source"})
    monkeypatch.setattr(
        lock_module,
        "load_tts_calibration_authority_artifact",
        lambda _path: SimpleNamespace(
            schema_version=4,
            trusted_content_bundle_source=foreign,
        ),
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "bind",
        classmethod(lambda _cls, _path: current),
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: SimpleNamespace(),
    )
    monkeypatch.setattr(
        models,
        "require_formal_v03_bound_content_bundle",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ValueError, match="binds another content bundle"):
        lock_module.build_trusted_single_operator_protocol_lock(
            protocol_id="foreign-tts-content-test",
            trusted_content_bundle_path=current.absolute_path,
            formal_runtime_authority_manifest_path=tmp_path / "unused-runtime.json",
            tts_calibration_authority_path=tts_path,
            chronobelief_authority_path=tmp_path / "unused-chronobelief.json",
            e1_recipe_anchor_authority_path=tmp_path / "unused-e1.json",
        )


@pytest.mark.parametrize("legacy", (False, True))
def test_lock_builder_rejects_foreign_or_legacy_e1_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy: bool,
) -> None:
    from lightcone_spec.experiments import (
        formal_single_operator_model_registry as models,
    )

    def content_binding(label: str) -> TrustedSingleOperatorContentBundleBinding:
        path = (tmp_path / f"{label}.json").resolve()
        publish_canonical_json_no_replace(path, {"kind": label})
        raw = CanonicalJsonProofBinding.bind(path)
        return TrustedSingleOperatorContentBundleBinding(
            absolute_path=raw.absolute_path,
            size=raw.size,
            raw_sha256=raw.raw_sha256,
            semantic_sha256=raw.semantic_sha256,
            runtime_binding_status="BOUND",
        )

    current = content_binding("current-content")
    foreign = content_binding("foreign-content")
    tts_path = (tmp_path / "tts-source.json").resolve()
    e1_path = (tmp_path / "e1-source.json").resolve()
    publish_canonical_json_no_replace(tts_path, {"kind": "test-tts-source"})
    publish_canonical_json_no_replace(e1_path, {"kind": "test-e1-source"})
    monkeypatch.setattr(
        lock_module,
        "load_tts_calibration_authority_artifact",
        lambda _path: SimpleNamespace(
            schema_version=4,
            trusted_content_bundle_source=current,
        ),
    )
    monkeypatch.setattr(
        lock_module,
        "load_e1_recipe_anchor_authority_artifact",
        lambda _path: SimpleNamespace(
            schema_version=2 if legacy else 3,
            trusted_content_bundle_source=None if legacy else foreign,
        ),
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "bind",
        classmethod(lambda _cls, _path: current),
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: SimpleNamespace(),
    )
    monkeypatch.setattr(
        models,
        "require_formal_v03_bound_content_bundle",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ValueError, match="E1 authority binds another content bundle"):
        lock_module.build_trusted_single_operator_protocol_lock(
            protocol_id="foreign-e1-content-test",
            trusted_content_bundle_path=current.absolute_path,
            formal_runtime_authority_manifest_path=tmp_path / "unused-runtime.json",
            tts_calibration_authority_path=tts_path,
            chronobelief_authority_path=tmp_path / "unused-chronobelief.json",
            e1_recipe_anchor_authority_path=e1_path,
        )


def _lock_with_live_structural_sources(
    tmp_path: Path,
) -> tuple[object, TrustedSingleOperatorContentBundleBinding]:
    def canonical(label: str) -> CanonicalJsonProofBinding:
        path = (tmp_path / f"{label}.json").resolve()
        publish_canonical_json_no_replace(path, {"kind": label})
        return CanonicalJsonProofBinding.bind(path)

    content = canonical("content")
    runtime = canonical("runtime")
    tts = canonical("tts")
    chrono = canonical("chrono")
    e1 = canonical("e1")

    def source(binding: CanonicalJsonProofBinding):
        return TrustedSingleOperatorProtocolSourceBinding(
            absolute_path=binding.absolute_path,
            raw_sha256=binding.raw_sha256,
            semantic_sha256=binding.semantic_sha256,
            size=binding.size,
        )

    content_binding = TrustedSingleOperatorContentBundleBinding(
        absolute_path=content.absolute_path,
        size=content.size,
        raw_sha256=content.raw_sha256,
        semantic_sha256=content.semantic_sha256,
        runtime_binding_status="BOUND",
    )
    sources = TrustedSingleOperatorProtocolSourceBindings(
        trusted_content_bundle_source=source(content),
        formal_runtime_authority_manifest_source=source(runtime),
        tts_calibration_authority_source=source(tts),
        chronobelief_authority_source=source(chrono),
        e1_recipe_anchor_authority_source=source(e1),
    )
    return (
        replace(
            _trusted_protocol_lock(),
            trusted_single_operator_content_bundle_sha256=content.semantic_sha256,
            trusted_single_operator_source_bindings=sources,
        ),
        content_binding,
    )


def test_lock_revalidator_forwards_identity_only_runtime_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, content_binding = _lock_with_live_structural_sources(tmp_path)
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        lock_module,
        "load_tts_calibration_authority_artifact",
        lambda _path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "bind",
        classmethod(lambda _cls, _path: content_binding),
    )

    def build(**kwargs):
        observed.update(kwargs)
        return lock

    monkeypatch.setattr(
        lock_module,
        "build_trusted_single_operator_protocol_lock",
        build,
    )

    assert (
        lock_module.revalidate_trusted_single_operator_protocol_lock(
            lock,
            require_capacity_available=False,
            revalidate_runtime_observations=False,
        )
        == lock
    )
    assert observed["require_capacity_available"] is False
    assert observed["revalidate_runtime_observations"] is False


def test_publisher_and_root_materializer_require_dynamic_runtime_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content_path = (tmp_path / "content.json").resolve()
    lock = _trusted_protocol_lock(content_path=content_path)
    policies: list[tuple[bool, bool]] = []

    def revalidate(value, **kwargs):
        policies.append(
            (
                kwargs["require_capacity_available"],
                kwargs["revalidate_runtime_observations"],
            )
        )
        return value

    monkeypatch.setattr(
        lock_module,
        "revalidate_trusted_single_operator_protocol_lock",
        revalidate,
    )
    lock_module.publish_trusted_single_operator_protocol_lock(
        lock,
        tmp_path / "published-lock.json",
    )
    assert policies == [(True, True), (True, True)]

    lock_path = (tmp_path / "root-lock.json").resolve()
    publish_formal_single_operator_json_artifact(
        lock_path,
        protocol_lock_to_dict(lock),
    )
    policies.clear()

    class RootReplayObserved(RuntimeError):
        pass

    def observe_root(_value, **kwargs):
        policies.append(
            (
                kwargs["require_capacity_available"],
                kwargs["revalidate_runtime_observations"],
            )
        )
        raise RootReplayObserved

    monkeypatch.setattr(
        lock_module,
        "revalidate_trusted_single_operator_protocol_lock",
        observe_root,
    )
    with pytest.raises(RootReplayObserved):
        materialize_formal_single_operator_node(
            node="preflight",
            predecessor_completion_path=None,
            protocol_lock_path=lock_path,
            content_source_path=content_path,
            materialization_output_path=tmp_path / "unused-materialization.json",
            node_materialization_output_path=tmp_path / "unused-node.json",
            created_ns=10,
        )
    assert policies == [(True, True)]


def test_dag_identity_consumer_uses_identity_only_runtime_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import formal_single_operator_content as content

    content_path = (tmp_path / "content.json").resolve()
    lock = _trusted_protocol_lock(content_path=content_path)
    lock_path = (tmp_path / "lock.json").resolve()
    publish_formal_single_operator_json_artifact(
        lock_path,
        protocol_lock_to_dict(lock),
    )
    observed: dict[str, object] = {}

    def revalidate(value, **kwargs):
        observed.update(kwargs)
        return value

    monkeypatch.setattr(
        lock_module,
        "revalidate_trusted_single_operator_protocol_lock",
        revalidate,
    )
    monkeypatch.setattr(
        content,
        "load_trusted_single_operator_content_bundle",
        lambda _path: SimpleNamespace(
            semantic_sha256=lock.trusted_single_operator_content_bundle_sha256,
            runtime_binding_status="BOUND",
            runtime_observations=SimpleNamespace(),
            source_snapshot=SimpleNamespace(repository_root=str(tmp_path)),
        ),
    )
    builder = object.__new__(ProductionFormalDagCallbackBuilder)
    builder.config = SimpleNamespace(
        protocol_lock=SimpleNamespace(absolute_path=str(lock_path)),
        content_source=SimpleNamespace(absolute_path=str(content_path)),
        repository_root=str(tmp_path),
    )

    builder._validate_identity_inputs()

    assert observed["require_capacity_available"] is False
    assert observed["revalidate_runtime_observations"] is False


@pytest.mark.parametrize(
    "fresh_runtime_failure",
    (RuntimeError("low free space"), OSError("statvfs failed")),
)
def test_driver_materializer_keeps_dynamic_runtime_out_of_source_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_runtime_failure: Exception,
) -> None:
    from lightcone_spec.experiments import formal_single_operator_stages as stages

    observed: dict[str, object] = {}

    def materialize(**kwargs):
        observed.update(kwargs)
        if (
            kwargs["require_capacity_available"]
            or kwargs["revalidate_runtime_observations"]
        ):
            raise fresh_runtime_failure
        publish_canonical_json_no_replace(
            kwargs["materialization_output_path"],
            {"kind": "identity-only-materialization"},
        )
        publish_canonical_json_no_replace(
            kwargs["node_materialization_output_path"],
            {"kind": "identity-only-node"},
        )

    monkeypatch.setattr(stages, "materialize_formal_single_operator_node", materialize)
    monkeypatch.setattr(
        stages,
        "rebuild_formal_single_operator_node_materialization",
        lambda _path: SimpleNamespace(
            artifact=SimpleNamespace(
                node="preflight",
                auxiliary_sources=(),
                predecessor_source=None,
            ),
            materialization=SimpleNamespace(cells=()),
        ),
    )
    builder = object.__new__(ProductionFormalDagCallbackBuilder)
    builder.nodes_root = (tmp_path / "nodes").resolve()
    builder.config = SimpleNamespace(
        protocol_lock=SimpleNamespace(absolute_path="/unused/protocol-lock.json"),
        content_source=SimpleNamespace(absolute_path="/unused/content.json"),
    )
    builder.clock_ns = lambda: 10

    rebuilt = builder.materialize("preflight", None)

    assert rebuilt.expected_cell_ids == ()
    assert observed["require_capacity_available"] is False
    assert observed["revalidate_runtime_observations"] is False


def _driver_capacity_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from test_formal_single_operator_capacity import _publish_fixture_authority

    repository = (tmp_path / "repository").resolve()
    repository.mkdir()
    authority_path, authority, run_root = _publish_fixture_authority(
        tmp_path,
        monkeypatch,
        repository_root=repository,
    )
    authority_binding = CanonicalJsonProofBinding.bind(authority_path)
    doctor_path = (tmp_path / "driver-doctor.json").resolve()
    publish_canonical_json_no_replace(
        doctor_path,
        {
            "stage_capacity": {
                "authority": authority_binding.to_dict(),
                "authority_sha256": authority.sha256,
            }
        },
    )
    inputs: dict[str, Path] = {"doctor": doctor_path}
    for name in (
        "protocol-lock",
        "content",
        "runtime-authority",
        "inventory",
        "preflight-workload",
    ):
        path = (tmp_path / f"driver-{name}.json").resolve()
        publish_canonical_json_no_replace(path, {"kind": name})
        inputs[name] = path
    catalog = (tmp_path / "prerequisite-catalog").resolve()
    catalog.mkdir()
    config = PathBoundFormalDagDriverConfig(
        schema_version=1,
        kind="formal_single_operator_dag_driver_config",
        repository_root=str(repository),
        run_root=str(run_root),
        protocol_lock=DriverFileBinding.bind(inputs["protocol-lock"]),
        content_source=DriverFileBinding.bind(inputs["content"]),
        runtime_authority_manifest=DriverFileBinding.bind(inputs["runtime-authority"]),
        inventory=DriverFileBinding.bind(inputs["inventory"]),
        doctor_report=DriverFileBinding.bind(inputs["doctor"]),
        preflight_workload_authority=DriverFileBinding.bind(
            inputs["preflight-workload"]
        ),
        profiler_tools=(),
        prerequisite_index_catalog_directory=str(catalog),
    )
    return config, authority_path, authority, authority_binding


def test_running_restart_constructs_between_safety_and_new_wave_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_formal_single_operator_capacity import (
        _bypass_deep_revalidation,
        _command,
        _initial_decision,
    )

    from lightcone_spec import doctor as doctor_module
    from lightcone_spec.experiments import formal_single_operator_capacity as capacity

    config, authority_path, authority, authority_binding = _driver_capacity_fixture(
        tmp_path,
        monkeypatch,
    )
    assert authority.schema_version == 4
    _bypass_deep_revalidation(
        monkeypatch,
        authority_path=authority_path,
        authority=authority,
    )
    free_bytes = 20 * 1024**3
    assert authority.safety_margin_bytes <= free_bytes < authority.required_free_bytes
    monkeypatch.setattr(capacity, "_free_bytes", lambda _path: free_bytes)
    monkeypatch.setattr(
        ProductionFormalDagCallbackBuilder,
        "_validate_identity_inputs",
        lambda _self: None,
    )
    monkeypatch.setattr(
        doctor_module,
        "revalidate_trusted_single_operator_doctor_report",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        capacity,
        "trusted_single_operator_capacity_authority_from_doctor",
        lambda *_args, **_kwargs: (
            authority_binding,
            authority,
            _initial_decision(authority),
        ),
    )
    running = _command(Path(config.run_root), 0)
    monkeypatch.setattr(
        ExperimentOperatorStore,
        "physical_commands",
        lambda _self, *, status="PENDING": (running,) if status == "RUNNING" else (),
    )

    with ExperimentOperatorStore(
        tmp_path / "restart.sqlite3", run_id="restart"
    ) as store:
        builder = ProductionFormalDagCallbackBuilder(config=config, store=store)

        assert builder.capacity_authority_path == str(authority_path)
        assert builder.nodes_root.is_dir()
        assert store.dispatch_control() == ("RUN", None)


@pytest.mark.parametrize("running", (False, True))
def test_dynamic_capacity_probe_error_durably_stops_without_new_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    running: bool,
) -> None:
    from test_formal_single_operator_capacity import _command

    from lightcone_spec import doctor as doctor_module
    from lightcone_spec.experiments import formal_single_operator_capacity as capacity

    config, authority_path, _authority, _binding = _driver_capacity_fixture(
        tmp_path,
        monkeypatch,
    )
    nodes_root = Path(config.run_root) / "formal-dag-nodes"
    if running:
        nodes_root.mkdir()
    command = _command(Path(config.run_root), 0)
    monkeypatch.setattr(
        ProductionFormalDagCallbackBuilder,
        "_validate_identity_inputs",
        lambda _self: None,
    )
    monkeypatch.setattr(
        doctor_module,
        "revalidate_trusted_single_operator_doctor_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("statvfs failed")),
    )
    monkeypatch.setattr(
        capacity,
        "trusted_single_operator_capacity_authority_from_doctor",
        lambda *_args, **_kwargs: pytest.fail(
            "dynamic authority ran after doctor error"
        ),
    )
    monkeypatch.setattr(
        ExperimentOperatorStore,
        "physical_commands",
        lambda _self, *, status="PENDING": (
            (command,) if running and status == "RUNNING" else ()
        ),
    )

    with ExperimentOperatorStore(
        tmp_path / f"probe-error-{running}.sqlite3",
        run_id=f"probe-error-{running}",
    ) as store:
        builder = ProductionFormalDagCallbackBuilder(config=config, store=store)

        assert builder.capacity_authority_path == str(authority_path)
        assert store.dispatch_control() == (
            "STOP",
            "trusted_restart_capacity_probe_failed:OSError",
        )
        assert nodes_root.exists() is running


def test_dag_identity_consumer_rejects_forged_schema5_lock(
    tmp_path: Path,
) -> None:
    content_path = (tmp_path / "forged-content.json").resolve()
    lock_path = (tmp_path / "forged-lock.json").resolve()
    publish_formal_single_operator_json_artifact(
        lock_path,
        protocol_lock_to_dict(_trusted_protocol_lock(content_path=content_path)),
    )
    builder = object.__new__(ProductionFormalDagCallbackBuilder)
    builder.config = SimpleNamespace(
        protocol_lock=SimpleNamespace(absolute_path=str(lock_path)),
        content_source=SimpleNamespace(absolute_path=str(content_path)),
        repository_root=str(tmp_path),
    )

    with pytest.raises(ValueError, match="GPU proof"):
        builder._validate_identity_inputs()
