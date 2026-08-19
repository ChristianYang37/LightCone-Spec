"""Path-bound group-scoped launch authority for resident ordinary TP1 serving.

The normalized process key proves that members may share one process.  This
module performs the separate, necessary step of materializing the *actual*
group launch: one RunConfig, one adaptive namespace, one telemetry path, one
argv/environment, and one port.  No caller-supplied digest or in-memory token
can authorize reuse; the returned object is reconstructed from its path.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.config import RunConfig, load_run_config, run_config_sha256
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.orchestration.formal_serving_session_group import (
    FormalServingSessionGroupPlan,
    normalized_formal_serving_process_key,
)
from lightcone_spec.orchestration.formal_serving_session_group_worker import (
    RevalidatedFormalServingSessionGroupExecution,
)
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.sglang_bridge.config import sglang_adaptation_payload

FORMAL_SERVING_RESIDENT_GROUP_LAUNCH_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_serving_resident_group_launch_authority",
        "input": "deep_reopened_group_plan_and_every_registered_compile_launch",
        "run_config": "group_scoped_canonical_RunConfig_with_bound_sidecar",
        "adaptation": (
            "schema3_payload_with_group_session_adaptation_namespace_or_absent"
        ),
        "process": "exact_actual_argv_sorted_environment_host_port_and_checkout",
        "backend_scope": "ordinary_tp1_DFlash_or_DSpark_adaptive_else_frozen",
        "claim": "trusted_single_operator_empirical_no_signature",
        "formal_measured": False,
    }
)

_KIND = "formal_serving_resident_group_launch_authority"
_EVIDENCE_LEVEL = "trusted_single_operator_empirical_no_signature"
_GROUP_ENVIRONMENT_KEYS = (
    "LIGHTCONE_RESIDENT_GROUP_ID",
    "LIGHTCONE_RESIDENT_SESSION_ADAPTATION_GROUP_ID",
)


def _absolute_path(label: str, value: object) -> Path:
    if type(value) is not str:
        raise TypeError(f"{label} must be a path string")
    path = Path(value)
    if (
        not path.is_absolute()
        or path != path.resolve(strict=False)
        or path == Path(path.anchor)
    ):
        raise ValueError(f"{label} must be an absolute normalized non-root path")
    return path


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _text(label: str, value: object) -> str:
    if type(value) is not str or not value or "\n" in value or "\x00" in value:
        raise ValueError(f"{label} must be non-empty single-line text")
    return value


def _write_text_no_replace(path: Path, value: str) -> EvidenceFileBinding:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        payload = value.encode("utf-8")
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return EvidenceFileBinding.bind(path, label="resident group launch sidecar")


def _reopen_text(binding: EvidenceFileBinding, *, label: str) -> str:
    binding.reopen(label=label)
    return Path(binding.absolute_path).read_text(encoding="utf-8")


def _replace_flag(
    argv: tuple[str, ...], flag: str, value: str, *, required: bool = True
) -> tuple[str, ...]:
    result = list(argv)
    positions = tuple(index for index, item in enumerate(result) if item == flag)
    if len(positions) != (1 if required else 0):
        if not required and not positions:
            return argv
        raise ValueError(f"resident group launch requires exactly one {flag}")
    position = positions[0]
    if position + 1 >= len(result):
        raise ValueError(f"resident group launch {flag} lacks a value")
    result[position + 1] = value
    return tuple(result)


def _registered_launches(
    plan: FormalServingSessionGroupPlan,
) -> tuple[tuple[CanonicalJsonProofBinding, CompileLaunchManifest, RunConfig], ...]:
    from lightcone_spec.orchestration.formal_physical_dispatch import (
        FormalServingRunPlan,
    )

    rows = []
    for member in plan.members:
        run_plan = FormalServingRunPlan.from_dict(member.run_plan.reopen())
        launch_binding = run_plan.launch_manifest
        if (
            CanonicalJsonProofBinding.bind(launch_binding.absolute_path)
            != launch_binding
        ):
            raise ValueError("resident registered launch binding changed")
        launch = CompileLaunchManifest.load(launch_binding.absolute_path)
        config = load_run_config(launch.run_config_path)
        if (
            launch.sha256 != launch_binding.semantic_sha256
            or member.compile_launch_manifest_sha256 != launch.sha256
            or member.run_config_sha256 != run_config_sha256(config)
            or member.materialized_cell_id != run_plan.materialized_cell_id
        ):
            raise ValueError("resident registered launch leaves its member")
        key = normalized_formal_serving_process_key(launch=launch, config=config)
        if plan.normalized_process_key is None or key != plan.normalized_process_key:
            raise ValueError("resident registered launch leaves the group process key")
        rows.append((launch_binding, launch, config))
    return tuple(rows)


def _group_run_config(
    *, plan: FormalServingSessionGroupPlan, source: RunConfig
) -> RunConfig:
    if plan.normalized_process_key is None:
        raise ValueError("resident group launch lacks a normalized process key")
    adaptation = source.adaptation
    if plan.normalized_process_key.adaptive:
        if (
            adaptation is None
            or plan.session_adaptation_group_id is None
            or source.model.algorithm not in {"DFLASH", "DSPARK"}
        ):
            raise ValueError("adaptive resident backend lacks group config support")
        original_namespaces = {
            member.original_adaptation_group_id for member in plan.members
        }
        if None in original_namespaces or plan.session_adaptation_group_id in (
            original_namespaces
        ):
            raise ValueError("resident group namespace is not distinct from cells")
        adaptation = adaptation.model_copy(
            update={"adaptation_group_id": plan.session_adaptation_group_id},
            deep=True,
        )
    elif adaptation is not None or plan.session_adaptation_group_id is not None:
        raise ValueError("non-adaptive resident group carries adaptation state")
    return RunConfig.model_validate(
        source.model_copy(update={"adaptation": adaptation}, deep=True).model_dump(
            mode="json"
        )
    )


def _actual_argv(
    *,
    launch: CompileLaunchManifest,
    config: RunConfig,
    run_config_path: Path,
    adaptation_path: Path | None,
    telemetry_path: Path | None,
    port: int,
) -> tuple[str, ...]:
    result = tuple(launch.server_argv)
    result = _replace_flag(result, "--run-config", str(run_config_path))
    result = _replace_flag(result, "--run-config-sha256", run_config_sha256(config))
    result = _replace_flag(result, "--host", "127.0.0.1")
    result = _replace_flag(result, "--port", str(port))
    if config.adaptation is None:
        if any(
            flag in result
            for flag in (
                "--speculative-adaptation-config",
                "--speculative-adaptation-telemetry-path",
            )
        ):
            raise ValueError("non-adaptive resident argv carries adaptation paths")
    else:
        if adaptation_path is None or telemetry_path is None:
            raise ValueError("adaptive resident argv lacks group paths")
        result = _replace_flag(
            result, "--speculative-adaptation-config", str(adaptation_path)
        )
        result = _replace_flag(
            result,
            "--speculative-adaptation-telemetry-path",
            str(telemetry_path),
        )
    return result


def _actual_environment(
    *, launch: CompileLaunchManifest, plan: FormalServingSessionGroupPlan
) -> tuple[tuple[str, str], ...]:
    environment = launch.child_environment()
    environment[_GROUP_ENVIRONMENT_KEYS[0]] = plan.group_id
    environment[_GROUP_ENVIRONMENT_KEYS[1]] = plan.session_adaptation_group_id or "NONE"
    return tuple(sorted(environment.items()))


@dataclass(frozen=True)
class FormalServingResidentGroupLaunchAuthority:
    schema_version: Literal[1]
    kind: Literal["formal_serving_resident_group_launch_authority"]
    protocol_sha256: str
    group_plan: CanonicalJsonProofBinding
    group_id: str
    normalized_process_key_sha256: str
    source_launch_manifests: tuple[CanonicalJsonProofBinding, ...]
    group_run_config: CanonicalJsonProofBinding
    group_run_config_sidecar: EvidenceFileBinding
    group_run_config_sha256: str
    group_adaptation_config: CanonicalJsonProofBinding | None
    group_adaptation_config_sidecar: EvidenceFileBinding | None
    group_adaptation_config_sha256: str | None
    session_adaptation_group_id: str | None
    adaptation_telemetry_path: str | None
    patched_sglang_checkout: str
    host: Literal["127.0.0.1"]
    port: int
    actual_server_argv: tuple[str, ...]
    actual_server_argv_sha256: str
    actual_child_environment: tuple[tuple[str, str], ...]
    actual_child_environment_sha256: str
    evidence_level: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != _KIND
            or self.protocol_sha256
            != FORMAL_SERVING_RESIDENT_GROUP_LAUNCH_PROTOCOL_SHA256
            or self.host != "127.0.0.1"
            or self.evidence_level != _EVIDENCE_LEVEL
            or self.formal_measured is not False
        ):
            raise ValueError("resident group launch authority identity differs")
        for label, value in (
            ("group", self.group_id),
            ("process key", self.normalized_process_key_sha256),
            ("RunConfig", self.group_run_config_sha256),
            ("argv", self.actual_server_argv_sha256),
            ("environment", self.actual_child_environment_sha256),
        ):
            _sha256(f"resident group launch {label}", value)
        if (
            type(self.group_plan) is not CanonicalJsonProofBinding
            or type(self.source_launch_manifests) is not tuple
            or len(self.source_launch_manifests) < 2
            or any(
                type(item) is not CanonicalJsonProofBinding
                for item in self.source_launch_manifests
            )
            or type(self.group_run_config) is not CanonicalJsonProofBinding
            or type(self.group_run_config_sidecar) is not EvidenceFileBinding
        ):
            raise TypeError("resident group launch source/config bindings differ")
        adaptive = self.session_adaptation_group_id is not None
        if adaptive:
            _text(
                "resident group adaptation namespace",
                self.session_adaptation_group_id,
            )
            _sha256(
                "resident group adaptation config",
                self.group_adaptation_config_sha256,
            )
            if (
                type(self.group_adaptation_config) is not CanonicalJsonProofBinding
                or type(self.group_adaptation_config_sidecar) is not EvidenceFileBinding
                or self.adaptation_telemetry_path is None
            ):
                raise TypeError("adaptive resident group launch bindings differ")
            _absolute_path(
                "resident group adaptation telemetry",
                self.adaptation_telemetry_path,
            )
        elif any(
            value is not None
            for value in (
                self.group_adaptation_config,
                self.group_adaptation_config_sidecar,
                self.group_adaptation_config_sha256,
                self.adaptation_telemetry_path,
            )
        ):
            raise ValueError("non-adaptive resident group carries adaptation evidence")
        _absolute_path("resident group checkout", self.patched_sglang_checkout)
        if type(self.port) is not int or not 1024 <= self.port <= 65535:
            raise ValueError("resident group launch port differs")
        if (
            type(self.actual_server_argv) is not tuple
            or not self.actual_server_argv
            or content_sha256({"argv": list(self.actual_server_argv)})
            != self.actual_server_argv_sha256
            or type(self.actual_child_environment) is not tuple
            or tuple(sorted(self.actual_child_environment))
            != self.actual_child_environment
            or len(dict(self.actual_child_environment))
            != len(self.actual_child_environment)
            or content_sha256(
                {"environment": [list(item) for item in self.actual_child_environment]}
            )
            != self.actual_child_environment_sha256
        ):
            raise ValueError("resident group actual process identity differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "group_plan": self.group_plan.to_dict(),
            "source_launch_manifests": [
                item.to_dict() for item in self.source_launch_manifests
            ],
            "group_run_config": self.group_run_config.to_dict(),
            "group_run_config_sidecar": self.group_run_config_sidecar.to_dict(),
            "group_adaptation_config": (
                None
                if self.group_adaptation_config is None
                else self.group_adaptation_config.to_dict()
            ),
            "group_adaptation_config_sidecar": (
                None
                if self.group_adaptation_config_sidecar is None
                else self.group_adaptation_config_sidecar.to_dict()
            ),
            "actual_server_argv": list(self.actual_server_argv),
            "actual_child_environment": [
                list(item) for item in self.actual_child_environment
            ],
        }
        if include_sha256:
            value["authority_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "authority_sha256",
        }:
            raise ValueError("resident group launch authority fields differ")
        row = dict(value)
        declared = _sha256(
            "resident group launch authority", row.pop("authority_sha256")
        )
        row["group_plan"] = CanonicalJsonProofBinding.from_dict(row["group_plan"])
        row["source_launch_manifests"] = tuple(
            CanonicalJsonProofBinding.from_dict(item)
            for item in row["source_launch_manifests"]
        )
        row["group_run_config"] = CanonicalJsonProofBinding.from_dict(
            row["group_run_config"]
        )
        row["group_run_config_sidecar"] = EvidenceFileBinding.from_dict(
            row["group_run_config_sidecar"],
            label="resident group launch RunConfig sidecar",
        )
        if row["group_adaptation_config"] is not None:
            row["group_adaptation_config"] = CanonicalJsonProofBinding.from_dict(
                row["group_adaptation_config"]
            )
        if row["group_adaptation_config_sidecar"] is not None:
            row["group_adaptation_config_sidecar"] = EvidenceFileBinding.from_dict(
                row["group_adaptation_config_sidecar"],
                label="resident group launch adaptation sidecar",
            )
        row["actual_server_argv"] = tuple(row["actual_server_argv"])
        row["actual_child_environment"] = tuple(
            tuple(item) for item in row["actual_child_environment"]
        )
        result = cls(**row)  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("resident group launch authority digest differs")
        return result


@dataclass(frozen=True)
class RevalidatedFormalServingResidentGroupLaunch:
    binding: CanonicalJsonProofBinding
    authority: FormalServingResidentGroupLaunchAuthority
    plan: FormalServingSessionGroupPlan
    run_config: RunConfig
    source_launch: CompileLaunchManifest


def _validate_authority(
    binding: CanonicalJsonProofBinding,
    authority: FormalServingResidentGroupLaunchAuthority,
) -> RevalidatedFormalServingResidentGroupLaunch:
    canonical_members = (
        authority.group_plan,
        *authority.source_launch_manifests,
        authority.group_run_config,
        *(
            ()
            if authority.group_adaptation_config is None
            else (authority.group_adaptation_config,)
        ),
    )
    if any(
        CanonicalJsonProofBinding.bind(item.absolute_path) != item
        for item in canonical_members
    ):
        raise ValueError("resident group launch canonical member changed")
    plan = FormalServingSessionGroupPlan.from_dict(authority.group_plan.reopen())
    if (
        plan.execution_mode != "shared_session_tp1"
        or plan.group_id != authority.group_id
        or plan.normalized_process_key is None
        or plan.normalized_process_key.sha256 != authority.normalized_process_key_sha256
        or plan.session_adaptation_group_id != authority.session_adaptation_group_id
    ):
        raise ValueError("resident group launch leaves its group plan")
    registered = _registered_launches(plan)
    if tuple(item[0] for item in registered) != authority.source_launch_manifests:
        raise ValueError("resident group launch source manifest coverage differs")
    source_launch = registered[0][1]
    expected_config = _group_run_config(plan=plan, source=registered[0][2])
    actual_config = load_run_config(authority.group_run_config.absolute_path)
    if (
        actual_config != expected_config
        or authority.group_run_config.semantic_sha256
        != run_config_sha256(expected_config)
        or authority.group_run_config_sha256 != run_config_sha256(expected_config)
        or _reopen_text(
            authority.group_run_config_sidecar,
            label="resident group launch RunConfig sidecar",
        )
        != f"{run_config_sha256(expected_config)}\n"
    ):
        raise ValueError("resident group RunConfig evidence differs")
    adaptation_payload = sglang_adaptation_payload(expected_config)
    if adaptation_payload is None:
        adaptation_path = None
        telemetry_path = None
        if authority.group_adaptation_config is not None:
            raise ValueError("non-adaptive resident group has an adaptation artifact")
    else:
        assert authority.group_adaptation_config is not None
        assert authority.group_adaptation_config_sidecar is not None
        if (
            authority.group_adaptation_config.reopen() != adaptation_payload
            or authority.group_adaptation_config.semantic_sha256
            != content_sha256(adaptation_payload)
            or authority.group_adaptation_config_sha256
            != content_sha256(adaptation_payload)
            or _reopen_text(
                authority.group_adaptation_config_sidecar,
                label="resident group launch adaptation sidecar",
            )
            != f"{content_sha256(adaptation_payload)}\n"
        ):
            raise ValueError("resident group adaptation config evidence differs")
        adaptation_path = Path(authority.group_adaptation_config.absolute_path)
        assert authority.adaptation_telemetry_path is not None
        telemetry_path = Path(authority.adaptation_telemetry_path)
    expected_argv = _actual_argv(
        launch=source_launch,
        config=expected_config,
        run_config_path=Path(authority.group_run_config.absolute_path),
        adaptation_path=adaptation_path,
        telemetry_path=telemetry_path,
        port=authority.port,
    )
    expected_environment = _actual_environment(launch=source_launch, plan=plan)
    if (
        authority.patched_sglang_checkout != source_launch.patched_sglang_checkout
        or authority.port != source_launch.localhost_port
        or authority.actual_server_argv != expected_argv
        or authority.actual_child_environment != expected_environment
    ):
        raise ValueError("resident group actual process differs from its sources")
    return RevalidatedFormalServingResidentGroupLaunch(
        binding=binding,
        authority=authority,
        plan=plan,
        run_config=expected_config,
        source_launch=source_launch,
    )


def publish_formal_serving_resident_group_launch_authority(
    *,
    execution: RevalidatedFormalServingSessionGroupExecution,
    output_root: str | Path,
) -> RevalidatedFormalServingResidentGroupLaunch:
    if type(execution) is not RevalidatedFormalServingSessionGroupExecution:
        raise TypeError("resident group launch requires deep execution evidence")
    root = _absolute_path("resident group launch root", str(output_root))
    if root.exists():
        if root.is_symlink() or not root.is_dir() or any(root.iterdir()):
            raise FileExistsError("resident group launch root must be absent or empty")
    else:
        root.mkdir(parents=True, mode=0o700)
    registered = _registered_launches(execution.plan)
    source_launch = registered[0][1]
    config = _group_run_config(plan=execution.plan, source=registered[0][2])
    config_path = root / "run-config.json"
    publish_canonical_json_no_replace(config_path, config.model_dump(mode="json"))
    config_binding = CanonicalJsonProofBinding.bind(config_path)
    config_sha = run_config_sha256(config)
    config_sidecar = _write_text_no_replace(
        Path(f"{config_path}.sha256"), f"{config_sha}\n"
    )
    payload = sglang_adaptation_payload(config)
    adaptation_binding: CanonicalJsonProofBinding | None = None
    adaptation_sidecar: EvidenceFileBinding | None = None
    adaptation_sha: str | None = None
    telemetry_path: Path | None = None
    if payload is not None:
        adaptation_path = root / "adaptation-config.json"
        publish_canonical_json_no_replace(adaptation_path, payload)
        adaptation_binding = CanonicalJsonProofBinding.bind(adaptation_path)
        adaptation_sha = content_sha256(payload)
        adaptation_sidecar = _write_text_no_replace(
            Path(f"{adaptation_path}.sha256"), f"{adaptation_sha}\n"
        )
        telemetry_path = root / "adaptation-telemetry.json"
    port = source_launch.localhost_port
    argv = _actual_argv(
        launch=source_launch,
        config=config,
        run_config_path=config_path,
        adaptation_path=(
            None
            if adaptation_binding is None
            else Path(adaptation_binding.absolute_path)
        ),
        telemetry_path=telemetry_path,
        port=port,
    )
    environment = _actual_environment(launch=source_launch, plan=execution.plan)
    authority = FormalServingResidentGroupLaunchAuthority(
        schema_version=1,
        kind=_KIND,
        protocol_sha256=FORMAL_SERVING_RESIDENT_GROUP_LAUNCH_PROTOCOL_SHA256,
        group_plan=execution.plan_binding,
        group_id=execution.plan.group_id,
        normalized_process_key_sha256=execution.plan.normalized_process_key.sha256,
        source_launch_manifests=tuple(item[0] for item in registered),
        group_run_config=config_binding,
        group_run_config_sidecar=config_sidecar,
        group_run_config_sha256=config_sha,
        group_adaptation_config=adaptation_binding,
        group_adaptation_config_sidecar=adaptation_sidecar,
        group_adaptation_config_sha256=adaptation_sha,
        session_adaptation_group_id=execution.plan.session_adaptation_group_id,
        adaptation_telemetry_path=(
            None if telemetry_path is None else str(telemetry_path)
        ),
        patched_sglang_checkout=source_launch.patched_sglang_checkout,
        host="127.0.0.1",
        port=port,
        actual_server_argv=argv,
        actual_server_argv_sha256=content_sha256({"argv": list(argv)}),
        actual_child_environment=environment,
        actual_child_environment_sha256=content_sha256(
            {"environment": [list(item) for item in environment]}
        ),
        evidence_level=_EVIDENCE_LEVEL,
        formal_measured=False,
    )
    manifest_path = root / "group-launch-authority.json"
    publish_canonical_json_no_replace(manifest_path, authority.to_dict())
    return revalidate_formal_serving_resident_group_launch_authority(manifest_path)


def revalidate_formal_serving_resident_group_launch_authority(
    path: str | Path,
) -> RevalidatedFormalServingResidentGroupLaunch:
    binding = CanonicalJsonProofBinding.bind(path)
    authority = FormalServingResidentGroupLaunchAuthority.from_dict(binding.reopen())
    return _validate_authority(binding, authority)


__all__ = (
    "FORMAL_SERVING_RESIDENT_GROUP_LAUNCH_PROTOCOL_SHA256",
    "FormalServingResidentGroupLaunchAuthority",
    "RevalidatedFormalServingResidentGroupLaunch",
    "publish_formal_serving_resident_group_launch_authority",
    "revalidate_formal_serving_resident_group_launch_authority",
)
