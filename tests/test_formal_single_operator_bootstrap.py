from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.orchestration import formal_single_operator_bootstrap as boot


def _file(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def test_bootstrap_config_cli_surface_is_path_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _file(tmp_path / "driver.json", {"kind": "driver"})
    authority = _file(tmp_path / "onlinespec.json", {"kind": "authority"})
    output = (tmp_path / "bootstrap.json").resolve()
    monkeypatch.setattr(
        boot,
        "load_path_bound_formal_dag_driver_config",
        lambda path: SimpleNamespace(path=path),
    )

    config = boot.publish_path_bound_formal_bootstrap_config(
        driver_config_path=driver,
        onlinespec_source_authority_path=authority,
        output_path=output,
    )

    assert boot.load_path_bound_formal_bootstrap_config(output) == config
    assert set(config.to_dict()) == {
        "schema_version",
        "kind",
        "driver_config",
        "onlinespec_source_authority",
    }
    with pytest.raises(FileExistsError):
        boot.publish_path_bound_formal_bootstrap_config(
            driver_config_path=driver,
            onlinespec_source_authority_path=authority,
            output_path=output,
        )


def test_e6_bootstrap_discovers_fresh_tp2_then_binds_exact_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import (
        formal_single_operator_e6_launch_producer as e6,
    )

    predecessor = _file(tmp_path / "e5-completion.json", {"kind": "completion"})
    tp2 = _file(tmp_path / "tp2-launch.json", {"kind": "launch"})
    launch_a = _file(tmp_path / "e6-a.json", {"model": "a"})
    launch_b = _file(tmp_path / "e6-b.json", {"model": "b"})
    root = (tmp_path / "run" / "formal-dag-bootstrap").resolve()
    catalog = (tmp_path / "catalog").resolve()
    root.mkdir(parents=True)
    catalog.mkdir()
    observed: dict[str, object] = {}
    supervisor = object.__new__(boot.FormalSingleOperatorBootstrapSupervisor)
    supervisor.root = root
    supervisor.catalog = catalog
    supervisor.config = SimpleNamespace(onlinespec_source_authority=None)
    supervisor.driver_config = SimpleNamespace(
        protocol_lock=SimpleNamespace(absolute_path="/run/protocol-lock.json"),
        content_source=SimpleNamespace(absolute_path="/run/content.json"),
    )
    monkeypatch.setattr(
        supervisor,
        "_preflight_launches",
        lambda path: {"tp2_dp1": str(tp2)},
    )

    def publish_index(**kwargs):
        observed["index"] = kwargs
        Path(kwargs["output_root"]).mkdir()
        return SimpleNamespace(
            predecessor_completion=SimpleNamespace(absolute_path=str(predecessor)),
            launch_manifest_paths={"model-a": str(launch_a), "model-b": str(launch_b)},
        )

    def bind(**kwargs):
        observed["binding"] = kwargs

    monkeypatch.setattr(
        e6,
        "publish_formal_single_operator_e6_builtin_mtp_launch_index",
        publish_index,
    )
    monkeypatch.setattr(boot, "publish_auxiliary_input_catalog_binding", bind)

    supervisor._publish_e6_auxiliary(str(predecessor))

    assert observed["index"]["base_environment_launch_manifest_path"] == str(tp2)  # type: ignore[index]
    assert set(observed["binding"]["input_paths"]) == {  # type: ignore[index]
        str(launch_a),
        str(launch_b),
    }
    assert observed["binding"]["node"] == "e6_pilot"  # type: ignore[index]


def test_prerequisite_bootstrap_recovers_source_published_before_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import (
        formal_single_operator_prerequisite_launch_producer as producer,
    )
    from lightcone_spec.experiments.formal_single_operator_stages import (
        formal_single_operator_node_spec,
    )

    node = "e4_profiler"
    spec = formal_single_operator_node_spec(node)
    run_root = (tmp_path / "run").resolve()
    source_path = (
        run_root
        / "formal-dag-nodes"
        / f"{spec.ordinal:02d}-{spec.node}"
        / "execution"
        / "execution-source.json"
    )
    source_path.parent.mkdir(parents=True)
    _file(source_path, {"kind": "source-published-before-block"})
    predecessor = _file(tmp_path / "predecessor.json", {"kind": "completion"})
    tp2 = _file(tmp_path / "tp2.json", {"kind": "launch"})
    source_sha256 = "a" * 64
    source = SimpleNamespace(
        node=node,
        sha256=source_sha256,
        predecessor_completion_source=SimpleNamespace(absolute_path=str(predecessor)),
    )
    rows = (
        {
            "node": node,
            "state": "BLOCKED",
            "execution_source_path": None,
        },
    )
    supervisor = object.__new__(boot.FormalSingleOperatorBootstrapSupervisor)
    supervisor.root = (run_root / "formal-dag-bootstrap").resolve()
    supervisor.root.mkdir(parents=True)
    supervisor.catalog = (tmp_path / "catalog").resolve()
    supervisor.catalog.mkdir()
    supervisor.driver_config = SimpleNamespace(
        run_root=str(run_root),
        repository_root=str(tmp_path.resolve()),
    )
    supervisor.driver = SimpleNamespace(
        store=SimpleNamespace(controller_nodes=lambda: rows)
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_single_operator_stages."
        "load_formal_single_operator_execution_source",
        lambda path: source if path == str(source_path) else None,
    )
    monkeypatch.setattr(
        supervisor,
        "_prerequisite_exists",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        supervisor,
        "_preflight_launches",
        lambda path: {"tp2_dp1": str(tp2)} if path == str(predecessor) else {},
    )

    def publish_index(**kwargs):
        observed["producer"] = kwargs
        index_path = (
            Path(kwargs["private_output_root"]) / "prerequisite-launch-index.json"
        )
        _file(index_path, {"kind": "index"})
        return SimpleNamespace(execution_source_sha256=source_sha256)

    def publish_binding(**kwargs):
        observed["binding"] = kwargs
        _file(Path(kwargs["output_path"]), {"kind": "binding"})

    monkeypatch.setattr(
        producer,
        "publish_formal_single_operator_prerequisite_launch_index",
        publish_index,
    )
    monkeypatch.setattr(
        producer,
        "execution_source_prerequisite_launch_demands",
        lambda _source: (object(),),
    )
    monkeypatch.setattr(
        boot,
        "publish_prerequisite_index_catalog_binding",
        publish_binding,
    )

    assert supervisor._publish_prerequisites() == (node,)
    assert observed["producer"]["execution_source_path"] == str(source_path)  # type: ignore[index]
    assert observed["producer"]["base_environment_launch_manifest_path"] == str(tp2)  # type: ignore[index]
    assert observed["binding"]["node"] == node  # type: ignore[index]


def test_e0_all_na_bootstrap_skips_three_prerequisite_publications_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import (
        formal_single_operator_prerequisite_launch_producer as producer,
    )
    from lightcone_spec.experiments import (
        formal_single_operator_stages as stages,
    )

    nodes = ("e0_tuning", "e0_pilot", "e0_final")
    sources = {
        node: SimpleNamespace(node=node, sha256=character * 64)
        for node, character in zip(nodes, "abc", strict=True)
    }
    source_paths = {
        node: str(_file(tmp_path / f"{node}-source.json", {"node": node}))
        for node in nodes
    }
    rows = tuple(
        {
            "node": node,
            "state": "MATERIALIZED",
            "execution_source_path": source_paths[node],
        }
        for node in nodes
    )
    supervisor = object.__new__(boot.FormalSingleOperatorBootstrapSupervisor)
    supervisor.root = (tmp_path / "bootstrap").resolve()
    supervisor.root.mkdir()
    supervisor.catalog = (tmp_path / "catalog").resolve()
    supervisor.catalog.mkdir()
    supervisor.driver = SimpleNamespace(
        store=SimpleNamespace(controller_nodes=lambda: rows)
    )
    monkeypatch.setattr(
        stages,
        "load_formal_single_operator_execution_source",
        lambda path: sources[
            next(node for node in nodes if source_paths[node] == path)
        ],
    )
    monkeypatch.setattr(
        producer,
        "execution_source_prerequisite_launch_demands",
        lambda source: () if source in sources.values() else (object(),),
    )
    monkeypatch.setattr(
        producer,
        "publish_formal_single_operator_prerequisite_launch_index",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("ALL_NA must not publish a prerequisite index")
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_prerequisite_exists",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("ALL_NA has no prerequisite catalog")
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_preflight_launches",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("ALL_NA needs no base launch")
        ),
    )

    assert supervisor._publish_prerequisites() == ()
    assert tuple(supervisor.root.iterdir()) == ()

    blocked_rows = tuple(
        {
            "node": node,
            "state": "BLOCKED",
            "blocker_reason": (
                f"{node}: exact prerequisite index binding is unavailable"
            ),
            "execution_source_path": source_paths[node],
        }
        for node in nodes
    )
    supervisor.driver = SimpleNamespace(
        store=SimpleNamespace(controller_nodes=lambda: blocked_rows)
    )
    assert supervisor._catalog_ready_blocked_nodes() == tuple(sorted(nodes))


def test_e0_bootstrap_binds_exact_twelve_source_owned_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import (
        formal_single_operator_e0_interface_producer as producer,
    )

    assert tuple(
        inspect.signature(
            producer.publish_formal_single_operator_e0_preprobe_interface_index
        ).parameters
    ) == (
        "protocol_lock_path",
        "predecessor_completion_path",
        "trusted_content_bundle_path",
        "output_root",
    )

    predecessor = _file(tmp_path / "e6-completion.json", {"kind": "completion"})
    descriptors = tuple(
        _file(tmp_path / f"descriptor-{index:02d}.json", {"index": index})
        for index in range(12)
    )
    supervisor = object.__new__(boot.FormalSingleOperatorBootstrapSupervisor)
    supervisor.root = tmp_path.resolve()
    supervisor.catalog = (tmp_path / "catalog").resolve()
    supervisor.catalog.mkdir()
    supervisor.config = SimpleNamespace(onlinespec_source_authority=None)
    supervisor.driver_config = SimpleNamespace(
        protocol_lock=SimpleNamespace(absolute_path="/run/protocol-lock.json"),
        content_source=SimpleNamespace(absolute_path="/run/content.json"),
    )
    observed: dict[str, object] = {}

    def publish_index(**kwargs):
        observed["index"] = kwargs
        Path(kwargs["output_root"]).mkdir()
        return SimpleNamespace(
            interface_descriptor_paths={
                f"model-{index // 3}|backend-{index % 3}": str(path)
                for index, path in enumerate(descriptors)
            }
        )

    def bind(**kwargs):
        observed["binding"] = kwargs

    monkeypatch.setattr(
        producer,
        "publish_formal_single_operator_e0_preprobe_interface_index",
        publish_index,
    )
    monkeypatch.setattr(boot, "publish_auxiliary_input_catalog_binding", bind)

    supervisor._publish_e0_auxiliary(str(predecessor))

    assert set(observed["binding"]["input_paths"]) == {  # type: ignore[index]
        str(path) for path in descriptors
    }
    assert observed["binding"]["node"] == "e0_tuning"  # type: ignore[index]
    assert observed["binding"]["predecessor_completion_path"] == str(predecessor)  # type: ignore[index]


def test_e0_bootstrap_reenters_publisher_after_retained_partial_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import (
        formal_single_operator_e0_interface_producer as producer,
    )

    predecessor = _file(tmp_path / "e6-completion.json", {"kind": "completion"})
    descriptors = tuple(
        _file(tmp_path / f"descriptor-{index:02d}.json", {"index": index})
        for index in range(12)
    )
    supervisor = object.__new__(boot.FormalSingleOperatorBootstrapSupervisor)
    supervisor.root = tmp_path.resolve()
    supervisor.catalog = (tmp_path / "catalog").resolve()
    supervisor.catalog.mkdir()
    supervisor.config = SimpleNamespace(onlinespec_source_authority=None)
    supervisor.driver_config = SimpleNamespace(
        protocol_lock=SimpleNamespace(absolute_path="/run/protocol-lock.json"),
        content_source=SimpleNamespace(absolute_path="/run/content.json"),
    )
    calls = 0

    def publish_index(**kwargs):
        nonlocal calls
        calls += 1
        root = Path(kwargs["output_root"])
        root.mkdir(mode=0o700, exist_ok=True)
        if calls == 1:
            partial = root / "pair-00"
            partial.mkdir(mode=0o700)
            _file(partial / "interrupted.json", {"retained": True})
            raise RuntimeError("simulated crash")
        _file(root / "e0-preprobe-interface-index.json", {"kind": "index"})
        return SimpleNamespace(
            interface_descriptor_paths={
                f"model-{index // 3}|backend-{index % 3}": str(path)
                for index, path in enumerate(descriptors)
            }
        )

    monkeypatch.setattr(
        producer,
        "publish_formal_single_operator_e0_preprobe_interface_index",
        publish_index,
    )
    monkeypatch.setattr(
        boot,
        "publish_auxiliary_input_catalog_binding",
        lambda **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="simulated crash"):
        supervisor._publish_e0_auxiliary(str(predecessor))
    partial = supervisor.root / "auxiliary/e0-exact-twelve/pair-00/interrupted.json"
    assert partial.is_file()

    supervisor._publish_e0_auxiliary(str(predecessor))

    assert calls == 2
    assert partial.is_file()


def test_e0_bootstrap_tampered_index_is_revalidated_not_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import (
        formal_single_operator_e0_interface_producer as producer,
    )

    predecessor = _file(tmp_path / "e6-completion.json", {"kind": "completion"})
    supervisor = object.__new__(boot.FormalSingleOperatorBootstrapSupervisor)
    supervisor.root = tmp_path.resolve()
    supervisor.catalog = (tmp_path / "catalog").resolve()
    supervisor.catalog.mkdir()
    supervisor.config = SimpleNamespace(onlinespec_source_authority=None)
    supervisor.driver_config = SimpleNamespace(
        protocol_lock=SimpleNamespace(absolute_path="/run/protocol-lock.json"),
        content_source=SimpleNamespace(absolute_path="/run/content.json"),
    )
    root = supervisor.root / "auxiliary/e0-exact-twelve"
    root.mkdir(mode=0o700, parents=True)
    index = _file(root / "e0-preprobe-interface-index.json", {"tampered": True})
    monkeypatch.setattr(
        producer,
        "revalidate_formal_single_operator_e0_preprobe_interface_index",
        lambda path: (_ for _ in ()).throw(ValueError(f"tampered index: {path}")),
    )
    monkeypatch.setattr(
        producer,
        "publish_formal_single_operator_e0_preprobe_interface_index",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("tampered index must not be replaced")
        ),
    )

    with pytest.raises(ValueError, match="tampered index"):
        supervisor._publish_e0_auxiliary(str(predecessor))

    assert index.read_text(encoding="utf-8") == '{"tampered":true}\n'


def test_synchronize_resumes_only_nodes_receiving_new_source_owned_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resumed: list[tuple[str, str]] = []

    class Store:
        @staticmethod
        def controller_node(node: str) -> dict[str, str]:
            return {"state": "BLOCKED" if node == "e6_pilot" else "PLANNED"}

    driver = SimpleNamespace(
        store=Store(),
        resume_node=lambda **kwargs: resumed.append((kwargs["node"], kwargs["reason"])),
    )
    supervisor = object.__new__(boot.FormalSingleOperatorBootstrapSupervisor)
    supervisor.driver = driver
    monkeypatch.setattr(
        supervisor,
        "_publish_auxiliary",
        lambda: ("e6_pilot",),
    )
    monkeypatch.setattr(
        supervisor,
        "_publish_prerequisites",
        lambda: ("e4_profiler",),
    )
    monkeypatch.setattr(
        supervisor,
        "_catalog_ready_blocked_nodes",
        lambda: (),
    )

    prerequisites, auxiliary, resumed_nodes = supervisor._synchronize()

    assert prerequisites == ("e4_profiler",)
    assert auxiliary == ("e6_pilot",)
    assert resumed_nodes == ("e6_pilot",)
    assert resumed == [("e6_pilot", "source_owned_bootstrap_input_published")]


def test_synchronize_recovers_catalog_published_before_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resumed: list[str] = []

    class Store:
        @staticmethod
        def controller_node(node: str) -> dict[str, str]:
            assert node == "e6_pilot"
            return {"state": "BLOCKED"}

    supervisor = object.__new__(boot.FormalSingleOperatorBootstrapSupervisor)
    supervisor.driver = SimpleNamespace(
        store=Store(),
        resume_node=lambda **kwargs: resumed.append(kwargs["node"]),
    )
    monkeypatch.setattr(supervisor, "_publish_auxiliary", lambda: ())
    monkeypatch.setattr(supervisor, "_publish_prerequisites", lambda: ())
    monkeypatch.setattr(
        supervisor,
        "_catalog_ready_blocked_nodes",
        lambda: ("e6_pilot",),
    )

    prerequisites, auxiliary, resumed_nodes = supervisor._synchronize()

    assert prerequisites == auxiliary == ()
    assert resumed_nodes == ("e6_pilot",)
    assert resumed == ["e6_pilot"]
