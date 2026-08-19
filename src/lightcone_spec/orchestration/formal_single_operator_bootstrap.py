"""Path-only bootstrap supervisor for the trusted v03 experiment DAG.

The production DAG intentionally resolves prerequisite and auxiliary catalogs
from append-only directories.  This module closes the publication loop without
inventing JSON: it observes only deeply bound controller paths, invokes the
source-owned publishers, installs their catalog bindings, and resumes exactly
the node that had been waiting for the newly published input.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from lightcone_spec.orchestration.formal_single_operator_dag_driver import (
    AuxiliaryInputCatalogBinding,
    DriverFileBinding,
    FormalSingleOperatorDagDriver,
    PathBoundFormalDagDriverConfig,
    PrerequisiteIndexCatalogBinding,
    _publish_no_replace,
    _read_canonical_json,
    build_production_formal_dag_driver,
    load_path_bound_formal_dag_driver_config,
    publish_auxiliary_input_catalog_binding,
    publish_prerequisite_index_catalog_binding,
)

_CONFIG_KIND = "formal_single_operator_bootstrap_config"
_PREPARED_NODES = frozenset(
    {
        "e4_profiler",
        "e3b_pilot",
        "e3b_final",
        "e1a",
        "e5_pilot",
        "e5_final",
        "e6_pilot",
        "e6_final",
        "e0_tuning",
        "e0_pilot",
        "e0_final",
    }
)


class FormalSingleOperatorBootstrapBlocked(RuntimeError):
    """A required source-owned bootstrap producer or input is unavailable."""


@dataclass(frozen=True)
class PathBoundFormalBootstrapConfig:
    """All bootstrap inputs are existing file bindings, never derived hashes."""

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_bootstrap_config"]
    driver_config: DriverFileBinding
    onlinespec_source_authority: DriverFileBinding | None

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != _CONFIG_KIND:
            raise ValueError("formal bootstrap config schema differs")
        if type(self.driver_config) is not DriverFileBinding:
            raise TypeError("formal bootstrap driver config is not path-bound")
        if (
            self.onlinespec_source_authority is not None
            and type(self.onlinespec_source_authority) is not DriverFileBinding
        ):
            raise TypeError("formal bootstrap OnlineSPEC authority is not path-bound")
        load_path_bound_formal_dag_driver_config(self.driver_config.absolute_path)

    @property
    def sha256(self) -> str:
        from lightcone_spec.experiments.formal_protocol import content_sha256

        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "driver_config": self.driver_config.to_dict(),
            "onlinespec_source_authority": (
                None
                if self.onlinespec_source_authority is None
                else self.onlinespec_source_authority.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> PathBoundFormalBootstrapConfig:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("formal bootstrap config fields differ")
        row = dict(value)
        driver = row.pop("driver_config")
        authority = row.pop("onlinespec_source_authority")
        return cls(
            **row,
            driver_config=DriverFileBinding.from_dict(driver),
            onlinespec_source_authority=(
                None if authority is None else DriverFileBinding.from_dict(authority)
            ),
        )  # type: ignore[arg-type]


def publish_path_bound_formal_bootstrap_config(
    *,
    driver_config_path: str | Path,
    output_path: str | Path,
    onlinespec_source_authority_path: str | Path | None = None,
) -> PathBoundFormalBootstrapConfig:
    """Publish one no-replace supervisor config from file paths only."""

    value = PathBoundFormalBootstrapConfig(
        schema_version=1,
        kind=_CONFIG_KIND,
        driver_config=DriverFileBinding.bind(driver_config_path),
        onlinespec_source_authority=(
            None
            if onlinespec_source_authority_path is None
            else DriverFileBinding.bind(onlinespec_source_authority_path)
        ),
    )
    _publish_no_replace(output_path, value.to_dict())
    if load_path_bound_formal_bootstrap_config(output_path) != value:
        raise RuntimeError("formal bootstrap config changed during publication")
    return value


def load_path_bound_formal_bootstrap_config(
    path: str | Path,
) -> PathBoundFormalBootstrapConfig:
    return PathBoundFormalBootstrapConfig.from_dict(
        _read_canonical_json(path, label="formal bootstrap config")
    )


@dataclass(frozen=True)
class FormalBootstrapCycle:
    schema_version: Literal[1]
    published_prerequisite_nodes: tuple[str, ...]
    published_auxiliary_nodes: tuple[str, ...]
    resumed_nodes: tuple[str, ...]
    controller_action: str
    controller_node: str | None
    controller_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FormalSingleOperatorBootstrapSupervisor:
    """Single-process producer/scheduler supervisor for one exact driver."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        driver: FormalSingleOperatorDagDriver | None = None,
    ) -> None:
        self.config = load_path_bound_formal_bootstrap_config(config_path)
        self.driver_config: PathBoundFormalDagDriverConfig = (
            load_path_bound_formal_dag_driver_config(
                self.config.driver_config.absolute_path
            )
        )
        self.driver = driver or build_production_formal_dag_driver(
            self.config.driver_config.absolute_path
        )
        self.catalog = Path(self.driver_config.prerequisite_index_catalog_directory)
        self.root = Path(self.driver_config.run_root) / "formal-dag-bootstrap"
        self.root.mkdir(mode=0o700, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("formal bootstrap root is unavailable")

    def close(self) -> None:
        self.driver.close()

    def _catalog_rows(self) -> tuple[object, ...]:
        rows = []
        for path in sorted(self.catalog.glob("*.json")):
            try:
                raw = _read_canonical_json(path, label="bootstrap catalog row")
            except (OSError, TypeError, ValueError):
                continue
            for cls in (PrerequisiteIndexCatalogBinding, AuxiliaryInputCatalogBinding):
                try:
                    rows.append(cls.from_dict(raw))
                    break
                except (TypeError, ValueError):
                    continue
        return tuple(rows)

    def _prerequisite_exists(self, *, node: str, source_sha256: str) -> bool:
        matches = tuple(
            row
            for row in self._catalog_rows()
            if type(row) is PrerequisiteIndexCatalogBinding
            and row.node == node
            and row.execution_source_sha256 == source_sha256
        )
        if len(matches) > 1:
            raise ValueError("bootstrap prerequisite catalog is ambiguous")
        if matches:
            from lightcone_spec.experiments.formal_single_operator_prerequisite_launch_producer import (
                load_formal_single_operator_prerequisite_launch_index,
            )

            load_formal_single_operator_prerequisite_launch_index(
                matches[0].prerequisite_index_path
            )
        return bool(matches)

    def _auxiliary_exists(self, *, node: str, predecessor_path: str) -> bool:
        predecessor = DriverFileBinding.bind(predecessor_path)
        matches = tuple(
            row
            for row in self._catalog_rows()
            if type(row) is AuxiliaryInputCatalogBinding
            and row.node == node
            and row.predecessor_completion.absolute_path == predecessor.absolute_path
            and row.predecessor_completion.raw_sha256 == predecessor.raw_sha256
        )
        if len(matches) > 1:
            raise ValueError("bootstrap auxiliary catalog is ambiguous")
        return bool(matches)

    @staticmethod
    def _preflight_launches(completion_path: str) -> dict[str, str]:
        from lightcone_spec.experiments.formal_single_operator_prerequisite_launch_producer import (
            trusted_preflight_qualification_launch_paths_from_completion,
        )

        return trusted_preflight_qualification_launch_paths_from_completion(
            completion_path
        )

    def _published_execution_source_path(
        self,
        *,
        node: str,
        controller_row: Mapping[str, object],
    ) -> str | None:
        """Recover the source published immediately before a prerequisite block.

        The controller records an execution source only after ``plan`` returns.
        A prepared planner necessarily publishes that source first and can then
        block while resolving its prerequisite catalog.  In that state the
        SQLite row has no source path yet, so use the driver's code-owned node
        layout and deep validation below instead of requiring a hand-authored
        catalog or a second planning path.
        """

        recorded = controller_row.get("execution_source_path")
        if recorded:
            if type(recorded) is not str:
                raise TypeError("bootstrap execution source path is not text")
            return recorded
        if controller_row.get("state") not in {"MATERIALIZED", "BLOCKED"}:
            return None
        from lightcone_spec.experiments.formal_single_operator_stages import (
            formal_single_operator_node_spec,
        )

        spec = formal_single_operator_node_spec(node)
        candidate = (
            Path(self.driver_config.run_root)
            / "formal-dag-nodes"
            / f"{spec.ordinal:02d}-{spec.node}"
            / "execution"
            / "execution-source.json"
        )
        if not candidate.exists():
            return None
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("bootstrap execution source is not a regular file")
        return str(candidate)

    def _publish_prerequisites(self) -> tuple[str, ...]:
        from lightcone_spec.experiments.formal_single_operator_prerequisite_launch_producer import (
            publish_formal_single_operator_prerequisite_launch_index,
        )
        from lightcone_spec.experiments.formal_single_operator_stages import (
            load_formal_single_operator_execution_source,
        )

        published = []
        for row in self.driver.store.controller_nodes():
            node = str(row["node"])
            source_path = self._published_execution_source_path(
                node=node,
                controller_row=row,
            )
            if node not in _PREPARED_NODES or not source_path:
                continue
            source = load_formal_single_operator_execution_source(source_path)
            if self._prerequisite_exists(
                node=node,
                source_sha256=source.sha256,
            ):
                continue
            predecessor = source.predecessor_completion_source
            if predecessor is None:
                raise ValueError(f"{node}: prepared source lacks predecessor")
            base = self._preflight_launches(predecessor.absolute_path)["tp2_dp1"]
            publication_root = (
                self.root / "prerequisites" / f"{node}-{source.sha256[:20]}"
            )
            publication_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            index_path = publication_root / "prerequisite-launch-index.json"
            if publication_root.exists():
                if not index_path.is_file() or index_path.is_symlink():
                    raise RuntimeError("incomplete prerequisite publication retained")
                from lightcone_spec.experiments.formal_single_operator_prerequisite_launch_producer import (
                    load_formal_single_operator_prerequisite_launch_index,
                )

                index = load_formal_single_operator_prerequisite_launch_index(
                    index_path
                )
            else:
                publication_root.mkdir(mode=0o700)
                index = publish_formal_single_operator_prerequisite_launch_index(
                    execution_source_path=source_path,
                    base_environment_launch_manifest_path=base,
                    repository_root=self.driver_config.repository_root,
                    private_output_root=publication_root,
                )
            binding_path = self.catalog / (
                f"prerequisite-{node}-{source.sha256[:20]}.json"
            )
            publish_prerequisite_index_catalog_binding(
                node=node,
                execution_source_path=source_path,
                prerequisite_index_path=(
                    publication_root / "prerequisite-launch-index.json"
                ),
                output_path=binding_path,
            )
            if index.execution_source_sha256 != source.sha256:
                raise RuntimeError("bootstrap prerequisite source changed")
            published.append(node)
        return tuple(published)

    def _publish_e6_auxiliary(self, predecessor_path: str) -> None:
        from lightcone_spec.experiments.formal_single_operator_e6_launch_producer import (
            publish_formal_single_operator_e6_builtin_mtp_launch_index,
            revalidate_formal_single_operator_e6_builtin_mtp_launch_index,
        )

        launches = self._preflight_launches(predecessor_path)
        output_root = self.root / "auxiliary" / "e6-exact-two"
        output_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        index_path = output_root / "e6-built-in-mtp-launch-index.json"
        if output_root.exists():
            if not index_path.is_file() or index_path.is_symlink():
                raise RuntimeError("incomplete E6 exact-two publication retained")
            index = revalidate_formal_single_operator_e6_builtin_mtp_launch_index(
                index_path
            )
        else:
            index = publish_formal_single_operator_e6_builtin_mtp_launch_index(
                protocol_lock_path=self.driver_config.protocol_lock.absolute_path,
                predecessor_completion_path=predecessor_path,
                trusted_content_bundle_path=(
                    self.driver_config.content_source.absolute_path
                ),
                base_environment_launch_manifest_path=launches["tp2_dp1"],
                output_root=output_root,
            )
        if index.predecessor_completion.absolute_path != predecessor_path:
            raise ValueError("retained E6 exact-two predecessor differs")
        publish_auxiliary_input_catalog_binding(
            node="e6_pilot",
            predecessor_completion_path=predecessor_path,
            input_paths=tuple(index.launch_manifest_paths.values()),
            output_path=self.catalog / "auxiliary-e6-pilot.json",
        )

    def _publish_e0_auxiliary(self, predecessor_path: str) -> None:
        try:
            from lightcone_spec.experiments.formal_single_operator_e0_interface_producer import (
                publish_formal_single_operator_e0_preprobe_interface_index,
                revalidate_formal_single_operator_e0_preprobe_interface_index,
            )
        except ModuleNotFoundError as error:
            if error.name != (
                "lightcone_spec.experiments."
                "formal_single_operator_e0_interface_producer"
            ):
                raise
            raise FormalSingleOperatorBootstrapBlocked(
                "e0_source_owned_preprobe_interface_producer_unavailable"
            ) from error

        output_root = self.root / "auxiliary" / "e0-exact-twelve"
        output_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        index_path = output_root / "e0-preprobe-interface-index.json"
        if index_path.is_file() and not index_path.is_symlink():
            index = revalidate_formal_single_operator_e0_preprobe_interface_index(
                index_path
            )
        else:
            if os.path.lexists(index_path):
                raise RuntimeError("invalid E0 exact-twelve index retained")
            index = publish_formal_single_operator_e0_preprobe_interface_index(
                protocol_lock_path=self.driver_config.protocol_lock.absolute_path,
                predecessor_completion_path=predecessor_path,
                trusted_content_bundle_path=(
                    self.driver_config.content_source.absolute_path
                ),
                output_root=output_root,
            )
        publish_auxiliary_input_catalog_binding(
            node="e0_tuning",
            predecessor_completion_path=predecessor_path,
            input_paths=tuple(index.interface_descriptor_paths.values()),
            onlinespec_source_authority_path=(
                None
                if self.config.onlinespec_source_authority is None
                else self.config.onlinespec_source_authority.absolute_path
            ),
            output_path=self.catalog / "auxiliary-e0-tuning.json",
        )

    def _publish_auxiliary(self) -> tuple[str, ...]:
        published = []
        for node, predecessor_node, publisher in (
            ("e6_pilot", "e5_final", self._publish_e6_auxiliary),
            ("e0_tuning", "e6_final", self._publish_e0_auxiliary),
        ):
            target = self.driver.store.controller_node(node)
            predecessor = self.driver.store.controller_node(predecessor_node)
            predecessor_path = predecessor.get("completion_path")
            if target["state"] == "REDUCED" or not predecessor_path:
                continue
            if self._auxiliary_exists(
                node=node,
                predecessor_path=predecessor_path,
            ):
                continue
            publisher(predecessor_path)
            published.append(node)
        return tuple(published)

    def _catalog_ready_blocked_nodes(self) -> tuple[str, ...]:
        """Recover the narrow publish-before-resume crash window."""

        from lightcone_spec.experiments.formal_single_operator_stages import (
            load_formal_single_operator_execution_source,
        )

        ready = []
        rows = {str(row["node"]): row for row in self.driver.store.controller_nodes()}
        for node, row in rows.items():
            if row.get("state") != "BLOCKED":
                continue
            reason = row.get("blocker_reason")
            if (
                node in _PREPARED_NODES
                and reason == f"{node}: exact prerequisite index binding is unavailable"
            ):
                source_path = self._published_execution_source_path(
                    node=node,
                    controller_row=row,
                )
                if source_path is None:
                    continue
                source = load_formal_single_operator_execution_source(source_path)
                if self._prerequisite_exists(
                    node=node,
                    source_sha256=source.sha256,
                ):
                    ready.append(node)
                continue
            if (
                node in {"e6_pilot", "e0_tuning"}
                and reason
                == f"{node}: exact predecessor-bound auxiliary inputs are unavailable"
            ):
                predecessor_node = "e5_final" if node == "e6_pilot" else "e6_final"
                predecessor_path = rows[predecessor_node].get("completion_path")
                if type(predecessor_path) is str and self._auxiliary_exists(
                    node=node,
                    predecessor_path=predecessor_path,
                ):
                    ready.append(node)
        return tuple(sorted(ready))

    def _synchronize(self) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        auxiliary = self._publish_auxiliary()
        prerequisites = self._publish_prerequisites()
        published = set(auxiliary) | set(prerequisites)
        resumable = published | set(self._catalog_ready_blocked_nodes())
        resumed = []
        for node in sorted(resumable):
            if self.driver.store.controller_node(node)["state"] == "BLOCKED":
                self.driver.resume_node(
                    node=node,
                    reason="source_owned_bootstrap_input_published",
                )
                resumed.append(node)
        return prerequisites, auxiliary, tuple(resumed)

    def run_once(self) -> FormalBootstrapCycle:
        before_prerequisites, before_auxiliary, before_resumed = self._synchronize()
        cycle = self.driver.run_once()
        after_prerequisites, after_auxiliary, after_resumed = self._synchronize()
        controller = cycle.controller
        return FormalBootstrapCycle(
            schema_version=1,
            published_prerequisite_nodes=tuple(
                dict.fromkeys((*before_prerequisites, *after_prerequisites))
            ),
            published_auxiliary_nodes=tuple(
                dict.fromkeys((*before_auxiliary, *after_auxiliary))
            ),
            resumed_nodes=tuple(dict.fromkeys((*before_resumed, *after_resumed))),
            controller_action=controller.action,
            controller_node=controller.node,
            controller_reason=controller.detail,
        )

    def run_until_event(self) -> FormalBootstrapCycle:
        """Run without LLM polling until completion or a true unresolved block."""

        while True:
            cycle = self.run_once()
            if cycle.controller_action in {"COMPLETE", "BLOCKED"}:
                if cycle.resumed_nodes:
                    continue
                return cycle
            time.sleep(1.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    operations = parser.add_subparsers(dest="operation", required=True)
    write = operations.add_parser("write-config", allow_abbrev=False)
    write.add_argument("--driver-config", required=True)
    write.add_argument("--onlinespec-source-authority")
    write.add_argument("--output", required=True)
    for name in ("once", "run"):
        command = operations.add_parser(name, allow_abbrev=False)
        command.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "write-config":
        config = publish_path_bound_formal_bootstrap_config(
            driver_config_path=args.driver_config,
            onlinespec_source_authority_path=args.onlinespec_source_authority,
            output_path=args.output,
        )
        print(json.dumps({"config_sha256": config.sha256}, sort_keys=True))
        return 0
    supervisor = FormalSingleOperatorBootstrapSupervisor(args.config)
    try:
        cycle = (
            supervisor.run_once()
            if args.operation == "once"
            else supervisor.run_until_event()
        )
        print(json.dumps(cycle.to_dict(), sort_keys=True))
        if cycle.controller_action == "COMPLETE":
            return 43
        return 42 if cycle.controller_action == "BLOCKED" else 0
    finally:
        supervisor.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FormalBootstrapCycle",
    "FormalSingleOperatorBootstrapBlocked",
    "FormalSingleOperatorBootstrapSupervisor",
    "PathBoundFormalBootstrapConfig",
    "load_path_bound_formal_bootstrap_config",
    "main",
    "publish_path_bound_formal_bootstrap_config",
]
