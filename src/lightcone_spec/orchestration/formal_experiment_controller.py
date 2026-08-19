"""Restart-safe non-LLM driver for the formal single-operator DAG.

The controller owns ordering and durable state, while source-owned callbacks own
scientific materialization, physical-plan production, actual-result discovery,
and reduction.  One ``run_once`` call performs at most one durable DAG
transition.  It never invents a future cell and it never treats a directory
name as evidence of completion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from lightcone_spec.orchestration.experiment_operator import (
    AuxiliaryCellAdoption,
    AuxiliaryGroupTerminal,
    AuxiliaryPhysicalGroupSpec,
    CellAttemptSpec,
    ControllerArtifactBinding,
    ExperimentOperatorError,
    ExperimentOperatorStore,
    PhysicalAttemptGroupMemberSpec,
    QueuedCommandSpec,
    SpawnedProcess,
)

_PRE_MATERIALIZATION_AUXILIARY = {
    "e6_pilot": ("e6_interface_fit", 2),
    "e0_tuning": ("e0_compatibility", 108),
}

DagControllerAction = Literal[
    "MATERIALIZED",
    "PLANNED",
    "WAITING",
    "REDUCED",
    "BLOCKED",
    "COMPLETE",
]


class FormalExperimentDagBlocked(RuntimeError):
    """A source-owned prerequisite is absent or scientifically invalid."""

    def __init__(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip() or "\x00" in reason:
            raise ValueError("DAG blocker reason must be non-empty text")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class DagMaterialization:
    materialization: ControllerArtifactBinding
    node_materialization: ControllerArtifactBinding
    expected_cell_ids: tuple[str, ...]
    auxiliary_sources: tuple[tuple[str, ControllerArtifactBinding], ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.materialization) is not ControllerArtifactBinding
            or type(self.node_materialization) is not ControllerArtifactBinding
        ):
            raise TypeError("DAG materialization requires exact artifact bindings")
        if self.expected_cell_ids != tuple(sorted(set(self.expected_cell_ids))):
            raise ValueError("DAG expected cell IDs must be uniquely sorted")
        if any(
            not isinstance(cell_id, str) or not cell_id
            for cell_id in self.expected_cell_ids
        ):
            raise ValueError("DAG expected cell IDs must be non-empty strings")
        prior: str | None = None
        for kind, binding in self.auxiliary_sources:
            if (
                not isinstance(kind, str)
                or not kind
                or (prior is not None and kind <= prior)
            ):
                raise ValueError("DAG auxiliary sources must be uniquely sorted")
            if type(binding) is not ControllerArtifactBinding:
                raise TypeError("DAG auxiliary source requires an exact binding")
            prior = kind


@dataclass(frozen=True)
class DagCellLaunch:
    attempt: CellAttemptSpec
    command: QueuedCommandSpec

    def __post_init__(self) -> None:
        if (
            type(self.attempt) is not CellAttemptSpec
            or type(self.command) is not QueuedCommandSpec
        ):
            raise TypeError("DAG cell launch requires exact operator specs")
        if (self.attempt.cell_id, self.attempt.attempt) != (
            self.command.cell_id,
            self.command.attempt,
        ):
            raise ValueError("DAG attempt and command identities differ")
        if self.attempt.command_sha256 != self.command.command_sha256:
            raise ValueError("DAG attempt and command hashes differ")


@dataclass(frozen=True)
class DagPhysicalAttemptGroup:
    """One registered shared parent represented by logical attempts."""

    group_id: str
    members: tuple[PhysicalAttemptGroupMemberSpec, ...]
    leader_cell_id: str
    group_kind: Literal["preflight_exact_ten", "tp1_serving_session"] = (
        "preflight_exact_ten"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("DAG physical group ID must be non-empty text")
        if not isinstance(self.leader_cell_id, str) or not self.leader_cell_id:
            raise ValueError("DAG physical group leader must be non-empty text")
        if type(self.members) is not tuple or any(
            type(member) is not PhysicalAttemptGroupMemberSpec
            for member in self.members
        ):
            raise TypeError("DAG physical group requires exact member specs")
        if self.group_kind not in {
            "preflight_exact_ten",
            "tp1_serving_session",
        }:
            raise ValueError("DAG physical group kind is not registered")
        identities = tuple(
            (member.attempt.cell_id, member.attempt.attempt) for member in self.members
        )
        if len(set(identities)) != len(identities):
            raise ValueError("DAG physical group attempts must be unique")
        kinds = tuple(member.logical_kind for member in self.members)
        if self.group_kind == "preflight_exact_ten":
            if len(self.members) != 10 or identities != tuple(sorted(identities)):
                raise ValueError("DAG preflight group must contain ten sorted attempts")
            if (
                kinds.count("compile") != 1
                or kinds.count("exactness") != 1
                or kinds.count("interference") != 8
            ):
                raise ValueError(
                    "DAG preflight group must contain exact 1+1+8 coverage"
                )
        elif not (2 <= len(self.members) <= 32 and set(kinds) == {"serving"}):
            raise ValueError(
                "DAG TP1 serving session group must contain 2-32 serving attempts"
            )
        if self.leader_cell_id != self.members[0].attempt.cell_id:
            raise ValueError("DAG physical group leader is not canonical")


@dataclass(frozen=True)
class DagExecutionPlan:
    execution_source: ControllerArtifactBinding
    prepared_launch: ControllerArtifactBinding | None
    launches: tuple[DagCellLaunch, ...]
    physical_attempt_groups: tuple[DagPhysicalAttemptGroup, ...] = ()

    def __post_init__(self) -> None:
        if type(self.execution_source) is not ControllerArtifactBinding:
            raise TypeError("DAG plan requires an exact execution-source binding")
        if (
            self.prepared_launch is not None
            and type(self.prepared_launch) is not ControllerArtifactBinding
        ):
            raise TypeError("DAG prepared launch requires an exact binding")
        launch_identities = tuple(
            (launch.attempt.cell_id, launch.attempt.attempt) for launch in self.launches
        )
        if launch_identities != tuple(sorted(launch_identities)):
            raise ValueError("DAG standalone launches must be sorted")
        identities = launch_identities + tuple(
            (member.attempt.cell_id, member.attempt.attempt)
            for group in self.physical_attempt_groups
            for member in group.members
        )
        if len(identities) != len(set(identities)):
            raise ValueError("DAG launches must be unique by attempt identity")
        group_ids = tuple(group.group_id for group in self.physical_attempt_groups)
        if group_ids != tuple(sorted(set(group_ids))):
            raise ValueError("DAG physical groups must be uniquely sorted")


@dataclass(frozen=True)
class DagReduction:
    decision: ControllerArtifactBinding
    completion: ControllerArtifactBinding

    def __post_init__(self) -> None:
        if (
            type(self.decision) is not ControllerArtifactBinding
            or type(self.completion) is not ControllerArtifactBinding
        ):
            raise TypeError("DAG reduction requires exact artifact bindings")


@dataclass(frozen=True)
class DagControllerCallbacks:
    materialize: Callable[[str, ControllerArtifactBinding | None], DagMaterialization]
    plan: Callable[[str, ControllerArtifactBinding], DagExecutionPlan]
    actual_results: Callable[[str, tuple[dict[str, Any], ...]], Mapping[str, str]]
    reduce: Callable[[str, ControllerArtifactBinding, Mapping[str, str]], DagReduction]
    auxiliary_plan: (
        Callable[
            [str, ControllerArtifactBinding | None], AuxiliaryPhysicalGroupSpec | None
        ]
        | None
    ) = None
    auxiliary_launch: Callable[[AuxiliaryPhysicalGroupSpec], SpawnedProcess] | None = (
        None
    )
    auxiliary_terminal: (
        Callable[
            [AuxiliaryPhysicalGroupSpec, Mapping[str, Any]],
            AuxiliaryGroupTerminal | None,
        ]
        | None
    ) = None
    materialize_with_auxiliary: (
        Callable[
            [
                str,
                ControllerArtifactBinding | None,
                Mapping[str, ControllerArtifactBinding],
            ],
            DagMaterialization,
        ]
        | None
    ) = None
    auxiliary_adoptions: (
        Callable[
            [str, ControllerArtifactBinding, AuxiliaryPhysicalGroupSpec],
            tuple[AuxiliaryCellAdoption, ...],
        ]
        | None
    ) = None


@dataclass(frozen=True)
class DagControllerStep:
    node: str | None
    action: DagControllerAction
    detail: str


class FormalExperimentDagController:
    """Advance the first non-reduced node through one durable transition."""

    def __init__(
        self,
        *,
        store: ExperimentOperatorStore,
        callbacks: DagControllerCallbacks,
    ) -> None:
        if type(store) is not ExperimentOperatorStore:
            raise TypeError("DAG controller requires an exact operator store")
        if type(callbacks) is not DagControllerCallbacks:
            raise TypeError("DAG controller requires exact callbacks")
        self._store = store
        self._callbacks = callbacks

    def run_once(self) -> DagControllerStep:
        nodes = self._store.controller_nodes()
        active = next((row for row in nodes if row["state"] != "REDUCED"), None)
        if active is None:
            return DagControllerStep(None, "COMPLETE", "all DAG nodes are reduced")
        node = str(active["node"])
        if any(
            row["state"] != "UNMATERIALIZED"
            for row in nodes[int(active["ordinal"]) + 1 :]
        ):
            raise ExperimentOperatorError(
                "a downstream controller node advanced ahead of its predecessor"
            )
        if active["state"] == "BLOCKED":
            return DagControllerStep(
                node,
                "BLOCKED",
                str(active["blocker_reason"] or "controller node is blocked"),
            )
        try:
            if active["state"] == "UNMATERIALIZED":
                return self._materialize(active, nodes)
            if active["state"] == "MATERIALIZED":
                return self._plan(active)
            if active["state"] == "PLANNED":
                return self._reduce_or_wait(active)
        except FormalExperimentDagBlocked as error:
            self._store.mark_controller_blocked(node=node, reason=error.reason)
            return DagControllerStep(node, "BLOCKED", error.reason)
        raise AssertionError("controller node state is not registered")

    def run_until_wait(self, *, maximum_transitions: int = 64) -> DagControllerStep:
        if (
            isinstance(maximum_transitions, bool)
            or not isinstance(maximum_transitions, int)
            or maximum_transitions < 1
        ):
            raise ValueError("maximum transitions must be a positive integer")
        step: DagControllerStep | None = None
        for _ in range(maximum_transitions):
            step = self.run_once()
            if step.action in {"WAITING", "BLOCKED", "COMPLETE"}:
                return step
        assert step is not None
        raise RuntimeError("DAG controller transition limit was exhausted")

    def _materialize(
        self,
        active: Mapping[str, Any],
        nodes: Sequence[Mapping[str, Any]],
    ) -> DagControllerStep:
        ordinal = int(active["ordinal"])
        predecessor = (
            None
            if ordinal == 0
            else ControllerArtifactBinding(
                str(nodes[ordinal - 1]["completion_path"]),
                str(nodes[ordinal - 1]["completion_sha256"]),
            )
        )
        node = str(active["node"])
        auxiliary_spec: AuxiliaryPhysicalGroupSpec | None = None
        auxiliary_sources: dict[str, ControllerArtifactBinding] = {}
        if node in _PRE_MATERIALIZATION_AUXILIARY:
            auxiliary_spec, step = self._advance_auxiliary(node, predecessor)
            if step is not None:
                return step
            assert auxiliary_spec is not None
            latest = self._store.latest_controller_auxiliary_group(node)
            if latest is None or latest["status"] != "COMPLETE":
                raise AssertionError("completed auxiliary group disappeared")
            auxiliary_sources[auxiliary_spec.source_kind] = ControllerArtifactBinding(
                str(latest["publication_path"]),
                str(latest["publication_sha256"]),
            )
        if auxiliary_sources:
            if self._callbacks.materialize_with_auxiliary is None:
                raise FormalExperimentDagBlocked(
                    f"{node} auxiliary-aware materializer is unavailable"
                )
            result = self._callbacks.materialize_with_auxiliary(
                node,
                predecessor,
                auxiliary_sources,
            )
        else:
            result = self._callbacks.materialize(node, predecessor)
        if type(result) is not DagMaterialization:
            raise TypeError("DAG materializer returned another result type")
        if dict(result.auxiliary_sources) != auxiliary_sources:
            raise ValueError(
                "DAG materialization auxiliary sources differ from completed work"
            )
        self._store.record_controller_materialization(
            node=node,
            materialization=result.materialization,
            node_materialization=result.node_materialization,
            expected_cell_ids=result.expected_cell_ids,
            auxiliary_sources=dict(result.auxiliary_sources),
        )
        if auxiliary_spec is not None:
            self._adopt_auxiliary(
                node=node,
                node_materialization=result.node_materialization,
                auxiliary_spec=auxiliary_spec,
                expected_cell_ids=result.expected_cell_ids,
            )
        return DagControllerStep(
            node,
            "MATERIALIZED",
            f"materialized {len(result.expected_cell_ids)} exact cells",
        )

    def _plan(self, active: Mapping[str, Any]) -> DagControllerStep:
        node = str(active["node"])
        node_materialization = ControllerArtifactBinding(
            str(active["node_materialization_path"]),
            str(active["node_materialization_sha256"]),
        )
        if node in _PRE_MATERIALIZATION_AUXILIARY:
            auxiliary_spec = self._rebuilt_auxiliary_spec(node)
            rebuilt = self._rebuild_auxiliary_materialization(active, auxiliary_spec)
            self._adopt_auxiliary(
                node=node,
                node_materialization=node_materialization,
                auxiliary_spec=auxiliary_spec,
                expected_cell_ids=rebuilt.expected_cell_ids,
            )
        result = self._callbacks.plan(node, node_materialization)
        if type(result) is not DagExecutionPlan:
            raise TypeError("DAG planner returned another result type")
        launch_ids = tuple(
            launch.attempt.cell_id for launch in result.launches
        ) + tuple(
            member.attempt.cell_id
            for group in result.physical_attempt_groups
            for member in group.members
        )
        adopted_ids = self._store.controller_auxiliary_adopted_cell_ids(node)
        if len(launch_ids) + len(adopted_ids) != len(set(launch_ids + adopted_ids)):
            raise ValueError("DAG plan must contain one initial launch per cell")
        all_ids = tuple(sorted(launch_ids + adopted_ids))
        expected_digest = hashlib.sha256(
            _canonical_json(all_ids).encode("utf-8")
        ).hexdigest()
        if (
            len(all_ids) != int(active["expected_cell_count"])
            or expected_digest != active["expected_cell_ids_sha256"]
        ):
            raise ValueError("DAG plan cell universe differs from materialization")
        for launch in result.launches:
            self._ensure_launch_registered(launch)
        for group in result.physical_attempt_groups:
            self._store.materialize_physical_attempt_group(
                group_id=group.group_id,
                members=group.members,
                leader_cell_id=group.leader_cell_id,
                group_kind=group.group_kind,
            )
        self._store.record_controller_execution_plan(
            node=node,
            execution_source=result.execution_source,
            prepared_launch=result.prepared_launch,
        )
        return DagControllerStep(node, "PLANNED", f"queued {len(launch_ids)} cells")

    def _advance_auxiliary(
        self,
        node: str,
        predecessor: ControllerArtifactBinding | None,
    ) -> tuple[AuxiliaryPhysicalGroupSpec | None, DagControllerStep | None]:
        callbacks = self._callbacks
        if (
            callbacks.auxiliary_plan is None
            or callbacks.auxiliary_launch is None
            or callbacks.auxiliary_terminal is None
            or callbacks.auxiliary_adoptions is None
        ):
            raise FormalExperimentDagBlocked(
                f"{node} pre-materialization auxiliary runner is unavailable"
            )
        spec = callbacks.auxiliary_plan(node, predecessor)
        self._validate_auxiliary_spec(node, spec)
        assert spec is not None
        registered = self._store.register_controller_auxiliary_group(spec)
        if registered:
            return spec, DagControllerStep(
                node,
                "WAITING",
                f"registered auxiliary {spec.source_kind} attempt {spec.attempt}",
            )
        latest = self._store.latest_controller_auxiliary_group(node)
        if latest is None or (latest["group_id"], int(latest["attempt"])) != (
            spec.group_id,
            spec.attempt,
        ):
            raise ExperimentOperatorError(
                "durable auxiliary attempt differs from rebuilt controller plan"
            )
        if latest["status"] == "PENDING":
            try:
                self._store.start_controller_auxiliary_group_with_launcher(
                    spec,
                    launcher=lambda: callbacks.auxiliary_launch(spec),
                )
            except Exception as error:  # noqa: BLE001 - injected launch boundary
                latest_after_error = self._store.latest_controller_auxiliary_group(node)
                if (
                    latest_after_error is not None
                    and latest_after_error["status"] == "RUNNING"
                ):
                    self._store.fail_controller_auxiliary_spawn(
                        spec,
                        exception_type=type(error).__name__,
                    )
                return spec, DagControllerStep(
                    node,
                    "WAITING",
                    f"retained auxiliary spawn failure attempt {spec.attempt}",
                )
            return spec, DagControllerStep(
                node,
                "WAITING",
                f"started auxiliary {spec.source_kind} attempt {spec.attempt}",
            )
        if latest["status"] == "RUNNING":
            terminal = callbacks.auxiliary_terminal(spec, latest)
            if terminal is None:
                return spec, DagControllerStep(
                    node,
                    "WAITING",
                    f"auxiliary {spec.source_kind} attempt {spec.attempt} is active",
                )
            if type(terminal) is not AuxiliaryGroupTerminal:
                raise TypeError("auxiliary terminal callback returned another type")
            self._store.finish_controller_auxiliary_group(spec, terminal)
            latest = self._store.latest_controller_auxiliary_group(node)
            assert latest is not None
            if latest["status"] != "COMPLETE":
                if (
                    latest["failure_class"] == "INFRASTRUCTURE"
                    and int(latest["attempt"]) < 3
                ):
                    return spec, DagControllerStep(
                        node,
                        "WAITING",
                        (
                            f"retained auxiliary infrastructure failure attempt "
                            f"{spec.attempt}; retry eligible"
                        ),
                    )
                raise FormalExperimentDagBlocked(
                    f"{node} auxiliary attempt {spec.attempt} failed"
                )
            return spec, DagControllerStep(
                node,
                "WAITING",
                f"completed auxiliary {spec.source_kind} attempt {spec.attempt}",
            )
        if latest["status"] == "FAILED":
            if (
                latest["failure_class"] == "INFRASTRUCTURE"
                and int(latest["attempt"]) < 3
            ):
                return spec, DagControllerStep(
                    node,
                    "WAITING",
                    f"auxiliary {spec.source_kind} is eligible for infrastructure retry",
                )
            raise FormalExperimentDagBlocked(
                f"{node} auxiliary attempt {spec.attempt} is FAILED"
            )
        if latest["status"] != "COMPLETE":
            raise AssertionError("controller auxiliary state is not registered")
        return spec, None

    def _rebuilt_auxiliary_spec(self, node: str) -> AuxiliaryPhysicalGroupSpec:
        if self._callbacks.auxiliary_plan is None:
            raise FormalExperimentDagBlocked(
                f"{node} auxiliary plan callback is unavailable"
            )
        nodes = self._store.controller_nodes()
        active = next(row for row in nodes if row["node"] == node)
        ordinal = int(active["ordinal"])
        predecessor = (
            None
            if ordinal == 0
            else ControllerArtifactBinding(
                str(nodes[ordinal - 1]["completion_path"]),
                str(nodes[ordinal - 1]["completion_sha256"]),
            )
        )
        spec = self._callbacks.auxiliary_plan(node, predecessor)
        self._validate_auxiliary_spec(node, spec)
        assert spec is not None
        latest = self._store.latest_controller_auxiliary_group(node)
        if (
            latest is None
            or latest["status"] != "COMPLETE"
            or (latest["group_id"], int(latest["attempt"]))
            != (spec.group_id, spec.attempt)
        ):
            raise FormalExperimentDagBlocked(
                f"{node} completed auxiliary source is unavailable"
            )
        return spec

    def _rebuild_auxiliary_materialization(
        self,
        active: Mapping[str, Any],
        auxiliary_spec: AuxiliaryPhysicalGroupSpec,
    ) -> DagMaterialization:
        callback = self._callbacks.materialize_with_auxiliary
        if callback is None:
            raise FormalExperimentDagBlocked(
                f"{active['node']} auxiliary-aware materializer is unavailable"
            )
        nodes = self._store.controller_nodes()
        ordinal = int(active["ordinal"])
        predecessor = (
            None
            if ordinal == 0
            else ControllerArtifactBinding(
                str(nodes[ordinal - 1]["completion_path"]),
                str(nodes[ordinal - 1]["completion_sha256"]),
            )
        )
        latest = self._store.latest_controller_auxiliary_group(str(active["node"]))
        if latest is None or latest["status"] != "COMPLETE":
            raise FormalExperimentDagBlocked(
                f"{active['node']} completed auxiliary source is unavailable"
            )
        source = ControllerArtifactBinding(
            str(latest["publication_path"]),
            str(latest["publication_sha256"]),
        )
        rebuilt = callback(
            str(active["node"]),
            predecessor,
            {auxiliary_spec.source_kind: source},
        )
        if type(rebuilt) is not DagMaterialization:
            raise TypeError("auxiliary-aware materializer returned another type")
        expected_digest = hashlib.sha256(
            _canonical_json(rebuilt.expected_cell_ids).encode("utf-8")
        ).hexdigest()
        if (
            rebuilt.materialization.absolute_path != active["materialization_path"]
            or rebuilt.materialization.sha256 != active["materialization_sha256"]
            or rebuilt.node_materialization.absolute_path
            != active["node_materialization_path"]
            or rebuilt.node_materialization.sha256
            != active["node_materialization_sha256"]
            or len(rebuilt.expected_cell_ids) != int(active["expected_cell_count"])
            or expected_digest != active["expected_cell_ids_sha256"]
            or dict(rebuilt.auxiliary_sources) != {auxiliary_spec.source_kind: source}
        ):
            raise ExperimentOperatorError(
                "rebuilt auxiliary materialization differs from durable node"
            )
        return rebuilt

    def _validate_auxiliary_spec(
        self,
        node: str,
        spec: AuxiliaryPhysicalGroupSpec | None,
    ) -> None:
        expected_kind, expected_count = _PRE_MATERIALIZATION_AUXILIARY[node]
        if type(spec) is not AuxiliaryPhysicalGroupSpec:
            raise FormalExperimentDagBlocked(f"{node} auxiliary plan is unavailable")
        if (
            spec.node != node
            or spec.source_kind != expected_kind
            or len(spec.jobs) != expected_count
        ):
            raise ValueError(
                f"{node} auxiliary plan differs from exact {expected_kind} coverage"
            )

    def _adopt_auxiliary(
        self,
        *,
        node: str,
        node_materialization: ControllerArtifactBinding,
        auxiliary_spec: AuxiliaryPhysicalGroupSpec,
        expected_cell_ids: tuple[str, ...],
    ) -> None:
        callback = self._callbacks.auxiliary_adoptions
        if callback is None:
            raise FormalExperimentDagBlocked(
                f"{node} auxiliary adoption callback is unavailable"
            )
        adoptions = callback(node, node_materialization, auxiliary_spec)
        if type(adoptions) is not tuple or any(
            type(value) is not AuxiliaryCellAdoption for value in adoptions
        ):
            raise TypeError("auxiliary adoption callback returned another type")
        adoption_ids = tuple(value.attempt.cell_id for value in adoptions)
        if len(adoptions) != len(auxiliary_spec.jobs) or len(set(adoption_ids)) != len(
            adoption_ids
        ):
            raise ValueError("auxiliary adoption cell coverage differs")
        if not set(adoption_ids).issubset(expected_cell_ids):
            raise ValueError("auxiliary adoption contains a non-materialized cell")
        self._store.adopt_controller_auxiliary_jobs(
            node=node,
            group_id=auxiliary_spec.group_id,
            group_attempt=auxiliary_spec.attempt,
            adoptions=adoptions,
        )

    def _ensure_launch_registered(self, launch: DagCellLaunch) -> None:
        spec = launch.attempt
        existing = self._store.latest_attempt(spec.cell_id)
        if existing is None:
            self._store.materialize_attempt(spec)
        else:
            _require_attempt_matches(existing, spec)
        queued = self._store.command_for_attempt(spec.cell_id, spec.attempt)
        if queued is None:
            self._store.enqueue_command(launch.command)
        elif queued != launch.command:
            raise ExperimentOperatorError(
                "durable queued command differs from the rebuilt DAG plan"
            )

    def _reduce_or_wait(self, active: Mapping[str, Any]) -> DagControllerStep:
        node = str(active["node"])
        attempts = self._store.latest_stage_attempts(node)
        expected = int(active["expected_cell_count"])
        if len(attempts) != expected:
            raise ExperimentOperatorError(
                "planned DAG node does not have exact latest-attempt coverage"
            )
        statuses = {str(row["status"]) for row in attempts}
        if statuses & {"PENDING", "RUNNING"}:
            return DagControllerStep(node, "WAITING", "cell attempts are still active")
        failures = tuple(
            str(row["cell_id"]) for row in attempts if row["status"] != "COMPLETE"
        )
        if failures:
            raise FormalExperimentDagBlocked(
                "terminal non-COMPLETE cells: " + ",".join(failures)
            )
        actuals = dict(self._callbacks.actual_results(node, attempts))
        expected_ids = tuple(str(row["cell_id"]) for row in attempts)
        if tuple(sorted(actuals)) != expected_ids:
            raise ValueError("DAG actual-result mapping differs from completed cells")
        node_materialization = ControllerArtifactBinding(
            str(active["node_materialization_path"]),
            str(active["node_materialization_sha256"]),
        )
        reduction = self._callbacks.reduce(node, node_materialization, actuals)
        if type(reduction) is not DagReduction:
            raise TypeError("DAG reducer returned another result type")
        self._store.record_controller_reduction(
            node=node,
            decision=reduction.decision,
            completion=reduction.completion,
        )
        return DagControllerStep(node, "REDUCED", "exact node coverage was reduced")


def _require_attempt_matches(
    actual: Mapping[str, Any], expected: CellAttemptSpec
) -> None:
    projection = {
        "cell_id": actual["cell_id"],
        "attempt": actual["attempt"],
        "stage": actual["stage"],
        "phase": actual["phase"],
        "block": actual["block_id"],
        "seed": actual["seed"],
        "scientific_axes": actual["scientific_axes"],
        "identity": actual["identity"],
        "command_sha256": actual["command_sha256"],
        "scientific_command_sha256": actual["scientific_command_sha256"],
        "output_directory": actual["output_directory"],
    }
    expected_projection = {
        "cell_id": expected.cell_id,
        "attempt": expected.attempt,
        "stage": expected.stage,
        "phase": expected.phase,
        "block": expected.block,
        "seed": expected.seed,
        "scientific_axes": dict(expected.scientific_axes),
        "identity": dict(expected.identity),
        "command_sha256": expected.command_sha256,
        "scientific_command_sha256": expected.scientific_command_sha256,
        "output_directory": expected.output_directory,
    }
    if projection != expected_projection:
        raise ExperimentOperatorError(
            "durable attempt differs from the rebuilt DAG launch"
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


__all__ = [
    "DagCellLaunch",
    "DagControllerCallbacks",
    "DagControllerStep",
    "DagExecutionPlan",
    "DagMaterialization",
    "DagPhysicalAttemptGroup",
    "DagReduction",
    "FormalExperimentDagBlocked",
    "FormalExperimentDagController",
]
