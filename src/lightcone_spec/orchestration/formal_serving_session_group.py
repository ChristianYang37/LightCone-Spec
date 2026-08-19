"""Deterministic TP1 formal-serving session grouping.

The current formal DAG launches one process per cell.  This module freezes the
process-compatibility and grouping interfaces needed to amortize starts without
changing scientific cell identity.  It does not launch a server or mutate the
operator ledger; those integrations consume the plans defined here.

Only trusted, schema-v2, single-GPU serving launches can share a process.
Distributed, NEXTN, profiler, failure-injection, unqualified, and singleton
work always receives an explicit fresh-process plan.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.config import RunConfig, run_config_sha256
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_single_operator_session_reset import (
    TrustedEmpiricalTp1SessionResetAuthority,
    revalidate_trusted_empirical_tp1_session_reset_authority,
)
from lightcone_spec.runtime.compile_cache import CompileCacheLaunchPlan
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

FormalServingPhysicalKind = Literal["serving", "profiler", "e5_failure"]
FormalServingSessionExecutionMode = Literal[
    "shared_session_tp1",
    "fresh_process_per_cell",
]

FORMAL_SERVING_NORMALIZED_PROCESS_KEY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_serving_normalized_process_key",
        "source": "deep_revalidated_trusted_schema2_compile_launch_and_RunConfig",
        "normalization": (
            "cell_paths_port_run_config_digest_device_rendezvous_router_and_"
            "adaptation_group_id_replaced_by_process_semantics"
        ),
        "preserved": (
            "unknown_argv_flags_model_content_backend_method_recipe_runtime_"
            "cache_graph_context_tp_dp_and_environment"
        ),
        "scope": "tp1_non_NEXTN_serving_only",
    }
)
FORMAL_SERVING_SESSION_GROUP_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_serving_session_group_plan",
        "member": (
            "exact_cell_attempt_run_plan_prepared_entry_config_launch_schedule_"
            "identity_gpu_output_and_duration"
        ),
        "partition": (
            "same_materialized_node_stage_phase_gpu_normalized_process_key_and_"
            "trusted_empirical_reset_scope_bounded_by_count_and_duration"
        ),
        "shared_gate": "exact_empirical_tp1_authority_or_fresh_process",
        "adaptive_namespace": "one_group_scoped_namespace_derived_from_group_id",
        "excluded": "distributed_NEXTN_profiler_failure_and_singleton",
        "formal_measured": False,
    }
)

_METHODS = {
    "target_only",
    "static",
    "tts",
    "l0",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
}
_METHOD_FAMILIES = {*_METHODS, "lightcone"}
_REUSABLE_BACKENDS = {"DFLASH", "DSPARK", "EAGLE3"}
# A backend being able to reset a frozen session is not, by itself, evidence
# that its adaptive launch can be rebound to one group-scoped state namespace.
# DFlash and DSpark consume the schema-3 adaptation payload rendered below;
# EAGLE3 remains fresh until that native namespace path is qualified.
_ADAPTIVE_GROUP_CONFIG_BACKENDS = {"DFLASH", "DSPARK"}
_FRESH_REASONS = {
    "profiler_requires_fresh_process",
    "failure_injection_requires_fresh_process",
    "distributed_topology_requires_fresh_process",
    "nextn_requires_fresh_process",
    "backend_session_reset_gate_unsupported",
    "adaptive_group_config_backend_unsupported",
    "trusted_prepared_launch_required",
    "session_reset_gate_missing",
    "session_group_singleton",
    "predicted_duration_exceeds_session_bound",
}
_NORMALIZED_FLAG_VALUES = {
    "--checkout": "<PATCHED_SGLANG_CHECKOUT>",
    "--compile-cache-plan": "<COMPILE_CACHE_PLAN>",
    "--run-config": "<GROUP_RUN_CONFIG>",
    "--model-path": "<TARGET_SNAPSHOT>",
    "--speculative-draft-model-path": "<DRAFTER_SNAPSHOT>",
    "--host": "<GROUP_HOST>",
    "--port": "<GROUP_PORT>",
    "--speculative-adaptation-config": "<GROUP_ADAPTATION_CONFIG>",
    "--speculative-adaptation-telemetry-path": "<GROUP_TELEMETRY_PATH>",
}


class FormalServingSessionReuseExcluded(ValueError):
    """A launch is valid but deliberately outside the shared-session surface."""

    def __init__(self, reason_code: str) -> None:
        if reason_code not in _FRESH_REASONS:
            raise ValueError("formal serving exclusion reason is unsupported")
        self.reason_code = reason_code
        super().__init__(reason_code)


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_git_object_id(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase Git object id")
    return value


def _require_text(label: str, value: object) -> str:
    if type(value) is not str or not value or "\n" in value or "\x00" in value:
        raise ValueError(f"{label} must be non-empty single-line text")
    return value


def _absolute_normalized_path(label: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a path string")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError(f"{label} must be absolute and normalized")
    return value


def _binding(binding: object, *, label: str) -> CanonicalJsonProofBinding:
    if type(binding) is not CanonicalJsonProofBinding:
        raise TypeError(f"{label} must be an exact canonical JSON binding")
    return binding


def _config_process_projection(config: RunConfig) -> dict[str, object]:
    runtime = config.runtime.model_dump(mode="json")
    for field in (
        "sampling_profile_sha256",
        "device_identity",
        "rendezvous_identity",
        "router_identity",
        "tp_rank",
        "dp_rank",
        "node_rank",
    ):
        runtime.pop(field)
    adaptation = (
        None if config.adaptation is None else config.adaptation.model_dump(mode="json")
    )
    if adaptation is not None:
        adaptation.pop("adaptation_group_id")
    return {
        "schema_version": 1,
        "method": config.method,
        "model": config.model.model_dump(mode="json"),
        "runtime": runtime,
        "adaptation_recipe_without_cell_namespace": adaptation,
        "online_spec": (
            None
            if config.online_spec is None
            else config.online_spec.model_dump(mode="json")
        ),
        "tenant_id": config.tenant_id,
        "session_adaptation_namespace_mode": (
            "none"
            if config.adaptation is None
            else "group_scoped_source_owned_all_reset_v1"
        ),
    }


def _argv_value(argv: Sequence[str], flag: str) -> str:
    positions = tuple(index for index, value in enumerate(argv) if value == flag)
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"formal serving argv requires exactly one {flag}")
    return argv[positions[0] + 1]


def _normalize_server_argv(
    *,
    launch: CompileLaunchManifest,
    config: RunConfig,
    process_config_sha256: str,
    compile_cache_process_sha256: str,
) -> tuple[str, ...]:
    argv = list(launch.server_argv)
    if not argv or any(type(value) is not str or not value for value in argv):
        raise ValueError("formal serving server argv is empty or malformed")
    expected = {
        "--checkout": launch.patched_sglang_checkout,
        "--compile-cache-plan": launch.compile_cache_plan_path,
        "--compile-cache-plan-sha256": launch.compile_cache_plan_sha256,
        "--run-config": launch.run_config_path,
        "--run-config-sha256": launch.run_config_semantic_sha256,
        "--model-path": launch.target_snapshot_path,
        "--host": "127.0.0.1",
        "--port": str(launch.localhost_port),
    }
    if config.method != "target_only" and config.model.nextn_mtp_mode != (
        "built_in_mtp"
    ):
        if launch.drafter_snapshot_path is None:
            raise ValueError("speculative launch lacks its drafter snapshot")
        expected["--speculative-draft-model-path"] = launch.drafter_snapshot_path
    if config.adaptation is not None:
        for flag in (
            "--speculative-adaptation-config",
            "--speculative-adaptation-telemetry-path",
        ):
            _argv_value(argv, flag)
    else:
        for flag in (
            "--speculative-adaptation-config",
            "--speculative-adaptation-telemetry-path",
        ):
            if flag in argv:
                raise ValueError("non-adaptive launch carries adaptation paths")
    for flag, value in expected.items():
        if _argv_value(argv, flag) != value:
            raise ValueError(f"formal serving argv differs at {flag}")
    for flag, replacement in _NORMALIZED_FLAG_VALUES.items():
        if flag not in argv:
            continue
        position = argv.index(flag)
        argv[position + 1] = replacement
    run_config_position = argv.index("--run-config-sha256")
    argv[run_config_position + 1] = process_config_sha256
    cache_plan_position = argv.index("--compile-cache-plan-sha256")
    argv[cache_plan_position + 1] = compile_cache_process_sha256
    return tuple(argv)


def _compile_cache_key_sha256(launch: CompileLaunchManifest) -> str:
    value = _argv_value(launch.server_argv, "--compile-cache-key-sha256")
    return _require_sha256("formal serving compile-cache key", value)


@dataclass(frozen=True)
class FormalServingNormalizedProcessKey:
    """Process-affecting projection with per-cell identities normalized away."""

    schema_version: Literal[1]
    kind: Literal["formal_serving_normalized_process_key"]
    protocol_sha256: str
    patched_sglang_commit: str
    patched_sglang_tree: str
    target_identity_sha256: str
    drafter_identity_sha256: str | None
    tokenizer_identity_sha256: str
    method: str
    backend: str
    topology_mode: Literal["tp1_dp1"]
    server_context_limit: int
    max_running_requests: int
    compile_cache_process_sha256: str
    compile_cache_mode: Literal["build", "reuse"]
    compile_cache_key_sha256: str
    process_run_config_sha256: str
    process_environment_sha256: str
    normalized_server_argv: tuple[str, ...]
    normalized_server_argv_sha256: str
    adaptive: bool

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_serving_normalized_process_key"
            or self.protocol_sha256
            != FORMAL_SERVING_NORMALIZED_PROCESS_KEY_PROTOCOL_SHA256
            or self.patched_sglang_commit != PINNED_SGLANG_COMMIT
            or self.patched_sglang_tree != PINNED_SGLANG_TREE
            or self.method not in _METHODS
            or self.backend not in _REUSABLE_BACKENDS
            or self.topology_mode != "tp1_dp1"
        ):
            raise ValueError("formal serving normalized process key schema differs")
        for label, value in (
            ("target identity", self.target_identity_sha256),
            ("tokenizer identity", self.tokenizer_identity_sha256),
            ("compile-cache process", self.compile_cache_process_sha256),
            ("compile-cache key", self.compile_cache_key_sha256),
            ("process RunConfig", self.process_run_config_sha256),
            ("process environment", self.process_environment_sha256),
            ("normalized server argv", self.normalized_server_argv_sha256),
        ):
            _require_sha256(f"formal serving {label}", value)
        _require_git_object_id(
            "formal serving patched SGLang tree", self.patched_sglang_tree
        )
        if self.drafter_identity_sha256 is not None:
            _require_sha256(
                "formal serving drafter identity", self.drafter_identity_sha256
            )
        if self.compile_cache_mode not in {"build", "reuse"}:
            raise ValueError("formal serving compile-cache mode differs")
        if type(self.server_context_limit) is not int or self.server_context_limit < 1:
            raise ValueError("formal serving context limit is invalid")
        if type(self.max_running_requests) is not int or self.max_running_requests < 1:
            raise ValueError("formal serving max-running-requests is invalid")
        if (
            type(self.normalized_server_argv) is not tuple
            or not self.normalized_server_argv
            or any(
                type(value) is not str or not value or "\x00" in value
                for value in self.normalized_server_argv
            )
            or self.normalized_server_argv_sha256
            != content_sha256({"argv": list(self.normalized_server_argv)})
        ):
            raise ValueError("formal serving normalized argv identity differs")
        for flag, replacement in _NORMALIZED_FLAG_VALUES.items():
            if (
                flag in self.normalized_server_argv
                and _argv_value(self.normalized_server_argv, flag) != replacement
            ):
                raise ValueError(f"formal serving normalized argv leaked {flag}")
        _require_sha256(
            "formal serving normalized RunConfig argv",
            _argv_value(self.normalized_server_argv, "--run-config-sha256"),
        )
        if (
            _argv_value(self.normalized_server_argv, "--run-config-sha256")
            != self.process_run_config_sha256
            or _argv_value(
                self.normalized_server_argv,
                "--compile-cache-plan-sha256",
            )
            != self.compile_cache_process_sha256
        ):
            raise ValueError("normalized argv and process projections differ")
        if self.adaptive != (
            "--speculative-adaptation-config" in self.normalized_server_argv
        ):
            raise ValueError("normalized argv adaptation mode differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = asdict(self)
        value["normalized_server_argv"] = list(self.normalized_server_argv)
        if include_sha256:
            value["process_key_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "process_key_sha256",
        }:
            raise ValueError("formal serving process key fields differ")
        row = dict(value)
        declared = _require_sha256(
            "formal serving process key", row.pop("process_key_sha256")
        )
        raw_argv = row.pop("normalized_server_argv")
        if type(raw_argv) is not list or any(
            type(item) is not str for item in raw_argv
        ):
            raise TypeError("formal serving normalized argv must be an array")
        result = cls(**row, normalized_server_argv=tuple(raw_argv))  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("formal serving process key digest differs")
        return result


def normalized_formal_serving_process_key(
    *,
    launch: CompileLaunchManifest,
    config: RunConfig,
) -> FormalServingNormalizedProcessKey:
    """Project one already-revalidated trusted launch onto process semantics."""

    if type(launch) is not CompileLaunchManifest or type(config) is not RunConfig:
        raise TypeError("formal serving process key requires exact launch/config")
    # Re-run Pydantic validation without reopening launch-owned files.  The
    # prepared-launch authority remains responsible for path-bound deep reopen.
    if RunConfig.model_validate(config.model_dump(mode="json")) != config:
        raise ValueError("formal serving RunConfig failed exact revalidation")
    if config.runtime.topology_mode != "tp1_dp1" or len(launch.gpu_uuids) != 1:
        raise FormalServingSessionReuseExcluded(
            "distributed_topology_requires_fresh_process"
        )
    if config.model.algorithm == "NEXTN" or launch.schema_version == 3:
        raise FormalServingSessionReuseExcluded("nextn_requires_fresh_process")
    if config.model.algorithm not in _REUSABLE_BACKENDS:
        raise FormalServingSessionReuseExcluded(
            "backend_session_reset_gate_unsupported"
        )
    if (
        config.adaptation is not None
        and config.model.algorithm not in _ADAPTIVE_GROUP_CONFIG_BACKENDS
    ):
        raise FormalServingSessionReuseExcluded(
            "adaptive_group_config_backend_unsupported"
        )
    if launch.schema_version != 2:
        raise FormalServingSessionReuseExcluded("trusted_prepared_launch_required")
    if (
        launch.patched_sglang_commit != PINNED_SGLANG_COMMIT
        or launch.patched_sglang_tree != PINNED_SGLANG_TREE
        or launch.run_config_semantic_sha256 != run_config_sha256(config)
        or launch.target_model_id != config.model.target
        or launch.target_revision != config.model.target_revision
        or launch.sampling_profile_sha256 != config.runtime.sampling_profile_sha256
        or launch.server_argv_sha256
        != content_sha256({"argv": list(launch.server_argv)})
        or launch.localhost_port < 1024
        or launch.localhost_port > 65535
    ):
        raise ValueError("formal serving launch/config identity differs")
    if config.method == "target_only":
        if launch.drafter_model_id is not None:
            raise ValueError("target-only launch unexpectedly carries a drafter")
    elif (
        launch.drafter_model_id != config.model.drafter
        or launch.drafter_revision != config.model.drafter_revision
    ):
        raise ValueError("formal serving drafter identity differs")

    projection = _config_process_projection(config)
    process_config_sha = content_sha256(projection)
    cache_plan = CompileCacheLaunchPlan.load(launch.compile_cache_plan_path)
    if (
        cache_plan.sha256 != launch.compile_cache_plan_sha256
        or cache_plan.key.sha256 != _compile_cache_key_sha256(launch)
    ):
        raise ValueError("formal serving compile-cache plan binding differs")
    cache_process_sha = content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_serving_compile_cache_process_projection",
            "key_sha256": cache_plan.key.sha256,
            "cache_mode": cache_plan.cache_mode,
            "builder_id": cache_plan.builder_id,
            "base_receipt_sha256": cache_plan.base_receipt_sha256,
        }
    )
    normalized_argv = _normalize_server_argv(
        launch=launch,
        config=config,
        process_config_sha256=process_config_sha,
        compile_cache_process_sha256=cache_process_sha,
    )
    target_identity = content_sha256(
        {
            "member_id": launch.target_content_member_id,
            "model_id": launch.target_model_id,
            "revision": launch.target_revision,
            "snapshot_sha256": launch.target_snapshot_sha256,
        }
    )
    drafter_identity = (
        None
        if launch.drafter_model_id is None
        else content_sha256(
            {
                "member_id": launch.drafter_content_member_id,
                "model_id": launch.drafter_model_id,
                "revision": launch.drafter_revision,
            }
        )
    )
    tokenizer_identity = content_sha256(
        {
            "member_id": launch.tokenizer_content_member_id,
            "model_id": launch.tokenizer_model_id,
            "revision": launch.tokenizer_revision,
        }
    )
    environment_sha = content_sha256(
        {
            "PATH": list(launch.path_entries),
            "LD_LIBRARY_PATH": list(launch.library_path_entries),
            "CUDA_HOME": launch.cuda_home,
        }
    )
    return FormalServingNormalizedProcessKey(
        schema_version=1,
        kind="formal_serving_normalized_process_key",
        protocol_sha256=(FORMAL_SERVING_NORMALIZED_PROCESS_KEY_PROTOCOL_SHA256),
        patched_sglang_commit=launch.patched_sglang_commit,
        patched_sglang_tree=launch.patched_sglang_tree,
        target_identity_sha256=target_identity,
        drafter_identity_sha256=drafter_identity,
        tokenizer_identity_sha256=tokenizer_identity,
        method=config.method,
        backend=config.model.algorithm,
        topology_mode="tp1_dp1",
        server_context_limit=config.runtime.context_length,
        max_running_requests=config.runtime.max_running_requests,
        compile_cache_process_sha256=cache_process_sha,
        compile_cache_mode=cache_plan.cache_mode,  # type: ignore[arg-type]
        compile_cache_key_sha256=_compile_cache_key_sha256(launch),
        process_run_config_sha256=process_config_sha,
        process_environment_sha256=environment_sha,
        normalized_server_argv=normalized_argv,
        normalized_server_argv_sha256=content_sha256({"argv": list(normalized_argv)}),
        adaptive=config.adaptation is not None,
    )


def formal_serving_session_reuse_exclusion_reason(
    *,
    physical_kind: str,
    launch: CompileLaunchManifest,
    config: RunConfig,
) -> str | None:
    if physical_kind == "profiler":
        return "profiler_requires_fresh_process"
    if physical_kind == "e5_failure":
        return "failure_injection_requires_fresh_process"
    if physical_kind != "serving":
        raise ValueError("formal serving physical kind is unsupported")
    if config.runtime.topology_mode != "tp1_dp1" or len(launch.gpu_uuids) != 1:
        return "distributed_topology_requires_fresh_process"
    if config.model.algorithm == "NEXTN" or launch.schema_version == 3:
        return "nextn_requires_fresh_process"
    if config.model.algorithm not in _REUSABLE_BACKENDS:
        return "backend_session_reset_gate_unsupported"
    if (
        config.adaptation is not None
        and config.model.algorithm not in _ADAPTIVE_GROUP_CONFIG_BACKENDS
    ):
        return "adaptive_group_config_backend_unsupported"
    if launch.schema_version != 2:
        return "trusted_prepared_launch_required"
    return None


@dataclass(frozen=True)
class FormalServingSessionGroupSpec:
    """Exact one-cell input to deterministic session partitioning."""

    schema_version: Literal[1]
    kind: Literal["formal_serving_session_group_spec"]
    protocol_sha256: str
    node: str
    stage: str
    phase: str
    materialized_cell_id: str
    attempt: int
    physical_kind: FormalServingPhysicalKind
    protocol_lock_sha256: str
    source_snapshot_sha256: str
    inventory_sha256: str
    assigned_gpu_uuids: tuple[str, ...]
    method: str
    method_family: str
    backend: str
    topology_mode: str
    run_plan: CanonicalJsonProofBinding
    prepared_launch_entry_sha256: str
    run_config_sha256: str
    compile_launch_manifest_sha256: str
    request_schedule_sha256: str | None
    output_directory: str
    estimated_duration_seconds: float
    dispatch_order_key: tuple[str, ...]
    normalized_process_key: FormalServingNormalizedProcessKey | None
    reuse_exclusion_reason: str | None
    original_adaptation_group_id: str | None
    formal_measured: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_serving_session_group_spec"
            or self.protocol_sha256 != FORMAL_SERVING_SESSION_GROUP_PROTOCOL_SHA256
            or self.physical_kind not in {"serving", "profiler", "e5_failure"}
            or self.method not in _METHODS
            or self.method_family not in _METHOD_FAMILIES
            or self.formal_measured is not False
        ):
            raise ValueError("formal serving group spec schema differs")
        for label, value in (
            ("node", self.node),
            ("stage", self.stage),
            ("phase", self.phase),
            ("backend", self.backend),
            ("topology", self.topology_mode),
        ):
            _require_text(f"formal serving group {label}", value)
        for label, value in (
            ("cell", self.materialized_cell_id),
            ("ProtocolLock", self.protocol_lock_sha256),
            ("source snapshot", self.source_snapshot_sha256),
            ("inventory", self.inventory_sha256),
            ("prepared launch entry", self.prepared_launch_entry_sha256),
            ("RunConfig", self.run_config_sha256),
            ("compile launch", self.compile_launch_manifest_sha256),
        ):
            _require_sha256(f"formal serving group {label}", value)
        if self.request_schedule_sha256 is not None:
            _require_sha256(
                "formal serving group request schedule",
                self.request_schedule_sha256,
            )
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("formal serving group attempt is invalid")
        if (
            type(self.assigned_gpu_uuids) is not tuple
            or not self.assigned_gpu_uuids
            or len(set(self.assigned_gpu_uuids)) != len(self.assigned_gpu_uuids)
        ):
            raise ValueError("formal serving group GPU assignment differs")
        for gpu_uuid in self.assigned_gpu_uuids:
            _require_text("formal serving group GPU UUID", gpu_uuid)
        _binding(self.run_plan, label="formal serving group run plan")
        _absolute_normalized_path(
            "formal serving group output directory", self.output_directory
        )
        if (
            type(self.estimated_duration_seconds) is not float
            or not math.isfinite(self.estimated_duration_seconds)
            or self.estimated_duration_seconds <= 0
        ):
            raise ValueError("formal serving group duration estimate is invalid")
        if (
            type(self.dispatch_order_key) is not tuple
            or not self.dispatch_order_key
            or any(
                type(value) is not str or not value or "\n" in value
                for value in self.dispatch_order_key
            )
        ):
            raise ValueError("formal serving dispatch order key is invalid")
        reusable = self.reuse_exclusion_reason is None
        if reusable != (
            type(self.normalized_process_key) is FormalServingNormalizedProcessKey
        ):
            raise ValueError("formal serving reuse eligibility is ambiguous")
        if reusable:
            assert self.normalized_process_key is not None
            if (
                self.physical_kind != "serving"
                or self.topology_mode != "tp1_dp1"
                or len(self.assigned_gpu_uuids) != 1
                or self.backend != self.normalized_process_key.backend
                or self.method != self.normalized_process_key.method
                or self.normalized_process_key.topology_mode != "tp1_dp1"
            ):
                raise ValueError("formal serving reusable spec differs from its key")
        elif self.reuse_exclusion_reason not in _FRESH_REASONS:
            raise ValueError("formal serving fresh-process reason is unsupported")
        adaptive = self.original_adaptation_group_id is not None
        if (
            self.normalized_process_key is not None
            and self.normalized_process_key.adaptive != adaptive
        ):
            raise ValueError("formal serving adaptation namespace binding differs")
        if self.original_adaptation_group_id is not None:
            _require_text(
                "formal serving original adaptation group",
                self.original_adaptation_group_id,
            )
        expected_runtime_methods = {
            "lightcone": {"l0"},
            "l0": {"l0"},
        }.get(self.method_family, {self.method_family})
        if self.method not in expected_runtime_methods:
            raise ValueError("scientific method family differs from runtime method")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    @property
    def sort_key(self) -> tuple[tuple[str, ...], str, int]:
        return (self.dispatch_order_key, self.materialized_cell_id, self.attempt)

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "assigned_gpu_uuids": list(self.assigned_gpu_uuids),
            "run_plan": self.run_plan.to_dict(),
            "dispatch_order_key": list(self.dispatch_order_key),
            "normalized_process_key": (
                None
                if self.normalized_process_key is None
                else self.normalized_process_key.to_dict()
            ),
        }
        if include_sha256:
            value["spec_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "spec_sha256",
        }:
            raise ValueError("formal serving group spec fields differ")
        row = dict(value)
        declared = _require_sha256("formal serving group spec", row.pop("spec_sha256"))
        raw_gpus = row.pop("assigned_gpu_uuids")
        raw_order = row.pop("dispatch_order_key")
        raw_process_key = row.pop("normalized_process_key")
        raw_run_plan = row.pop("run_plan")
        if type(raw_gpus) is not list or type(raw_order) is not list:
            raise TypeError("formal serving group spec arrays differ")
        result = cls(
            **row,
            assigned_gpu_uuids=tuple(raw_gpus),
            run_plan=CanonicalJsonProofBinding.from_dict(raw_run_plan),
            dispatch_order_key=tuple(raw_order),
            normalized_process_key=(
                None
                if raw_process_key is None
                else FormalServingNormalizedProcessKey.from_dict(raw_process_key)
            ),
        )  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("formal serving group spec digest differs")
        return result


def build_formal_serving_session_group_spec(
    *,
    node: str,
    stage: str,
    phase: str,
    materialized_cell_id: str,
    attempt: int,
    physical_kind: FormalServingPhysicalKind,
    method_family: str,
    protocol_lock_sha256: str,
    source_snapshot_sha256: str,
    inventory_sha256: str,
    run_plan: CanonicalJsonProofBinding,
    prepared_launch_entry_sha256: str,
    compile_launch_manifest_sha256: str,
    request_schedule_sha256: str | None,
    launch: CompileLaunchManifest,
    config: RunConfig,
    output_directory: str,
    estimated_duration_seconds: float,
    dispatch_order_key: tuple[str, ...],
) -> FormalServingSessionGroupSpec:
    """Build a cell spec after prepared-launch deep revalidation."""

    if type(launch) is not CompileLaunchManifest or type(config) is not RunConfig:
        raise TypeError("formal serving group spec requires exact launch/config")
    if (
        launch.run_config_semantic_sha256 != run_config_sha256(config)
        or launch.gpu_uuids == ()
    ):
        raise ValueError("formal serving group launch/config identity differs")
    exclusion = formal_serving_session_reuse_exclusion_reason(
        physical_kind=physical_kind,
        launch=launch,
        config=config,
    )
    process_key = (
        None
        if exclusion is not None
        else normalized_formal_serving_process_key(launch=launch, config=config)
    )
    return FormalServingSessionGroupSpec(
        schema_version=1,
        kind="formal_serving_session_group_spec",
        protocol_sha256=FORMAL_SERVING_SESSION_GROUP_PROTOCOL_SHA256,
        node=node,
        stage=stage,
        phase=phase,
        materialized_cell_id=materialized_cell_id,
        attempt=attempt,
        physical_kind=physical_kind,
        protocol_lock_sha256=protocol_lock_sha256,
        source_snapshot_sha256=source_snapshot_sha256,
        inventory_sha256=inventory_sha256,
        assigned_gpu_uuids=launch.gpu_uuids,
        method=config.method,
        method_family=method_family,
        backend=config.model.algorithm,
        topology_mode=config.runtime.topology_mode,
        run_plan=run_plan,
        prepared_launch_entry_sha256=prepared_launch_entry_sha256,
        run_config_sha256=run_config_sha256(config),
        compile_launch_manifest_sha256=compile_launch_manifest_sha256,
        request_schedule_sha256=request_schedule_sha256,
        output_directory=output_directory,
        estimated_duration_seconds=estimated_duration_seconds,
        dispatch_order_key=dispatch_order_key,
        normalized_process_key=process_key,
        reuse_exclusion_reason=exclusion,
        original_adaptation_group_id=(
            None if config.adaptation is None else config.adaptation.adaptation_group_id
        ),
        formal_measured=False,
    )


@dataclass(frozen=True)
class FormalServingSessionGroupPlan:
    """One bounded physical process lifetime or one explicit fresh fallback."""

    schema_version: Literal[1]
    kind: Literal["formal_serving_session_group_plan"]
    protocol_sha256: str
    group_id: str
    execution_mode: FormalServingSessionExecutionMode
    reason_code: str
    node: str
    stage: str
    phase: str
    assigned_gpu_uuids: tuple[str, ...]
    normalized_process_key: FormalServingNormalizedProcessKey | None
    reset_authority_sha256: str | None
    session_adaptation_group_id: str | None
    max_member_count: int
    max_estimated_duration_seconds: float
    members: tuple[FormalServingSessionGroupSpec, ...]
    formal_measured: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_serving_session_group_plan"
            or self.protocol_sha256 != FORMAL_SERVING_SESSION_GROUP_PROTOCOL_SHA256
            or self.execution_mode
            not in {"shared_session_tp1", "fresh_process_per_cell"}
            or self.formal_measured is not False
        ):
            raise ValueError("formal serving group plan schema differs")
        _require_sha256("formal serving group ID", self.group_id)
        for label, value in (
            ("node", self.node),
            ("stage", self.stage),
            ("phase", self.phase),
            ("reason", self.reason_code),
        ):
            _require_text(f"formal serving plan {label}", value)
        if (
            type(self.members) is not tuple
            or not self.members
            or any(
                type(row) is not FormalServingSessionGroupSpec for row in self.members
            )
            or tuple(row.sort_key for row in self.members)
            != tuple(sorted(row.sort_key for row in self.members))
            or len({(row.materialized_cell_id, row.attempt) for row in self.members})
            != len(self.members)
        ):
            raise ValueError("formal serving plan members are not canonical")
        if any(
            (row.node, row.stage, row.phase) != (self.node, self.stage, self.phase)
            or row.assigned_gpu_uuids != self.assigned_gpu_uuids
            for row in self.members
        ):
            raise ValueError("formal serving plan member scope differs")
        if (
            type(self.max_member_count) is not int
            or self.max_member_count < 2
            or len(self.members) > self.max_member_count
            or type(self.max_estimated_duration_seconds) is not float
            or not math.isfinite(self.max_estimated_duration_seconds)
            or self.max_estimated_duration_seconds <= 0
        ):
            raise ValueError("formal serving plan bounds are invalid")
        total_duration = sum(row.estimated_duration_seconds for row in self.members)
        if total_duration > self.max_estimated_duration_seconds and not (
            len(self.members) == 1
            and self.execution_mode == "fresh_process_per_cell"
            and self.reason_code == "predicted_duration_exceeds_session_bound"
        ):
            raise ValueError("formal serving plan exceeds its duration bound")
        if self.execution_mode == "shared_session_tp1":
            if (
                len(self.members) < 2
                or len(self.assigned_gpu_uuids) != 1
                or type(self.normalized_process_key)
                is not FormalServingNormalizedProcessKey
                or self.reset_authority_sha256 is None
                or self.reason_code != "trusted_empirical_tp1_reset_gate_passed"
                or any(
                    row.normalized_process_key != self.normalized_process_key
                    or row.reuse_exclusion_reason is not None
                    for row in self.members
                )
            ):
                raise ValueError("shared formal serving plan lacks exact reset scope")
            _require_sha256(
                "formal serving reset authority", self.reset_authority_sha256
            )
            expected_namespace = (
                f"formal-session-{self.group_id[:32]}"
                if self.normalized_process_key.adaptive
                else None
            )
            if self.session_adaptation_group_id != expected_namespace:
                raise ValueError("shared adaptation namespace differs from group")
        elif (
            len(self.members) != 1
            or self.reset_authority_sha256 is not None
            or self.session_adaptation_group_id is not None
            or self.reason_code not in _FRESH_REASONS
        ):
            raise ValueError("fresh formal serving plan is not one explicit fallback")
        if self.group_id != _formal_serving_group_id(
            execution_mode=self.execution_mode,
            reason_code=self.reason_code,
            members=self.members,
            normalized_process_key=self.normalized_process_key,
            reset_authority_sha256=self.reset_authority_sha256,
        ):
            raise ValueError("formal serving group ID differs from its members")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    @property
    def session_plan_sha256(self) -> str:
        return self.sha256

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "assigned_gpu_uuids": list(self.assigned_gpu_uuids),
            "normalized_process_key": (
                None
                if self.normalized_process_key is None
                else self.normalized_process_key.to_dict()
            ),
            "members": [row.to_dict() for row in self.members],
        }
        if include_sha256:
            value["plan_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "plan_sha256",
        }:
            raise ValueError("formal serving group plan fields differ")
        row = dict(value)
        declared = _require_sha256("formal serving group plan", row.pop("plan_sha256"))
        raw_gpus = row.pop("assigned_gpu_uuids")
        raw_key = row.pop("normalized_process_key")
        raw_members = row.pop("members")
        if type(raw_gpus) is not list or type(raw_members) is not list:
            raise TypeError("formal serving group plan arrays differ")
        result = cls(
            **row,
            assigned_gpu_uuids=tuple(raw_gpus),
            normalized_process_key=(
                None
                if raw_key is None
                else FormalServingNormalizedProcessKey.from_dict(raw_key)
            ),
            members=tuple(
                FormalServingSessionGroupSpec.from_dict(item) for item in raw_members
            ),
        )  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("formal serving group plan digest differs")
        return result


def _formal_serving_group_id(
    *,
    execution_mode: FormalServingSessionExecutionMode,
    reason_code: str,
    members: tuple[FormalServingSessionGroupSpec, ...],
    normalized_process_key: FormalServingNormalizedProcessKey | None,
    reset_authority_sha256: str | None,
) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_serving_session_group_id",
            "protocol_sha256": FORMAL_SERVING_SESSION_GROUP_PROTOCOL_SHA256,
            "execution_mode": execution_mode,
            "reason_code": reason_code,
            "member_sha256s": [row.sha256 for row in members],
            "normalized_process_key_sha256": (
                None
                if normalized_process_key is None
                else normalized_process_key.sha256
            ),
            "reset_authority_sha256": reset_authority_sha256,
        }
    )


def _make_plan(
    *,
    members: tuple[FormalServingSessionGroupSpec, ...],
    execution_mode: FormalServingSessionExecutionMode,
    reason_code: str,
    normalized_process_key: FormalServingNormalizedProcessKey | None,
    reset_authority_sha256: str | None,
    max_member_count: int,
    max_estimated_duration_seconds: float,
) -> FormalServingSessionGroupPlan:
    first = members[0]
    group_id = _formal_serving_group_id(
        execution_mode=execution_mode,
        reason_code=reason_code,
        members=members,
        normalized_process_key=normalized_process_key,
        reset_authority_sha256=reset_authority_sha256,
    )
    return FormalServingSessionGroupPlan(
        schema_version=1,
        kind="formal_serving_session_group_plan",
        protocol_sha256=FORMAL_SERVING_SESSION_GROUP_PROTOCOL_SHA256,
        group_id=group_id,
        execution_mode=execution_mode,
        reason_code=reason_code,
        node=first.node,
        stage=first.stage,
        phase=first.phase,
        assigned_gpu_uuids=first.assigned_gpu_uuids,
        normalized_process_key=normalized_process_key,
        reset_authority_sha256=reset_authority_sha256,
        session_adaptation_group_id=(
            f"formal-session-{group_id[:32]}"
            if execution_mode == "shared_session_tp1"
            and normalized_process_key is not None
            and normalized_process_key.adaptive
            else None
        ),
        max_member_count=max_member_count,
        max_estimated_duration_seconds=max_estimated_duration_seconds,
        members=members,
        formal_measured=False,
    )


def _fresh_plan(
    spec: FormalServingSessionGroupSpec,
    *,
    reason_code: str,
    max_member_count: int,
    max_estimated_duration_seconds: float,
) -> FormalServingSessionGroupPlan:
    return _make_plan(
        members=(spec,),
        execution_mode="fresh_process_per_cell",
        reason_code=reason_code,
        normalized_process_key=spec.normalized_process_key,
        reset_authority_sha256=None,
        max_member_count=max_member_count,
        max_estimated_duration_seconds=max_estimated_duration_seconds,
    )


def _authority_by_scope(
    authorities: Sequence[CanonicalJsonProofBinding],
) -> Mapping[
    tuple[str, str, str, str, str, str, str],
    TrustedEmpiricalTp1SessionResetAuthority,
]:
    result: dict[
        tuple[str, str, str, str, str, str, str],
        TrustedEmpiricalTp1SessionResetAuthority,
    ] = {}
    for binding in authorities:
        if type(binding) is not CanonicalJsonProofBinding:
            raise TypeError(
                "formal serving partition requires path-bound reset authorities"
            )
        rebound, authority = revalidate_trusted_empirical_tp1_session_reset_authority(
            binding.absolute_path
        )
        if rebound != binding:
            raise ValueError("formal serving reset authority binding changed")
        if authority.scope_key in result:
            raise ValueError("formal serving reset authority scope is ambiguous")
        result[authority.scope_key] = authority
    return result


def partition_formal_serving_session_groups(
    specs: Sequence[FormalServingSessionGroupSpec],
    *,
    reset_authorities: Sequence[CanonicalJsonProofBinding] = (),
    max_member_count: int = 32,
    max_estimated_duration_seconds: float = 3600.0,
) -> tuple[FormalServingSessionGroupPlan, ...]:
    """Partition current-node cells; every unsafe case becomes fresh fallback."""

    values = tuple(specs)
    if not values:
        return ()
    if any(type(row) is not FormalServingSessionGroupSpec for row in values):
        raise TypeError("formal serving partition requires exact group specs")
    identities = tuple((row.materialized_cell_id, row.attempt) for row in values)
    if len(set(identities)) != len(identities):
        raise ValueError("formal serving partition cell attempts are duplicated")
    if len({row.output_directory for row in values}) != len(values):
        raise ValueError("formal serving partition output directories collide")
    if type(max_member_count) is not int or max_member_count < 2:
        raise ValueError("formal serving session group requires at least two slots")
    if (
        type(max_estimated_duration_seconds) is not float
        or not math.isfinite(max_estimated_duration_seconds)
        or max_estimated_duration_seconds <= 0
    ):
        raise ValueError("formal serving session duration bound is invalid")
    authorities = _authority_by_scope(reset_authorities)
    fresh: list[FormalServingSessionGroupPlan] = []
    eligible: dict[
        tuple[str, str, str, tuple[str, ...], str, str],
        tuple[
            TrustedEmpiricalTp1SessionResetAuthority,
            list[FormalServingSessionGroupSpec],
        ],
    ] = {}
    for spec in sorted(values, key=lambda row: row.sort_key):
        if spec.reuse_exclusion_reason is not None:
            fresh.append(
                _fresh_plan(
                    spec,
                    reason_code=spec.reuse_exclusion_reason,
                    max_member_count=max_member_count,
                    max_estimated_duration_seconds=max_estimated_duration_seconds,
                )
            )
            continue
        assert spec.normalized_process_key is not None
        authority_scope = (
            spec.protocol_lock_sha256,
            spec.source_snapshot_sha256,
            spec.normalized_process_key.patched_sglang_tree,
            spec.inventory_sha256,
            spec.assigned_gpu_uuids[0],
            spec.backend,
            spec.method_family,
        )
        authority = authorities.get(authority_scope)
        if authority is None:
            fresh.append(
                _fresh_plan(
                    spec,
                    reason_code="session_reset_gate_missing",
                    max_member_count=max_member_count,
                    max_estimated_duration_seconds=max_estimated_duration_seconds,
                )
            )
            continue
        bucket_key = (
            spec.node,
            spec.stage,
            spec.phase,
            spec.assigned_gpu_uuids,
            spec.normalized_process_key.sha256,
            authority.sha256,
        )
        bucket = eligible.setdefault(bucket_key, (authority, []))
        bucket[1].append(spec)

    shared_or_singleton: list[FormalServingSessionGroupPlan] = []
    for bucket_key in sorted(eligible):
        authority, rows = eligible[bucket_key]
        ordered = sorted(rows, key=lambda row: row.sort_key)
        chunk: list[FormalServingSessionGroupSpec] = []
        duration = 0.0

        def flush(*, authority_sha256: str = authority.sha256) -> None:
            nonlocal chunk, duration
            if not chunk:
                return
            members = tuple(chunk)
            if len(members) == 1:
                reason = (
                    "predicted_duration_exceeds_session_bound"
                    if duration > max_estimated_duration_seconds
                    else "session_group_singleton"
                )
                shared_or_singleton.append(
                    _fresh_plan(
                        members[0],
                        reason_code=reason,
                        max_member_count=max_member_count,
                        max_estimated_duration_seconds=(max_estimated_duration_seconds),
                    )
                )
            else:
                key = members[0].normalized_process_key
                assert key is not None
                shared_or_singleton.append(
                    _make_plan(
                        members=members,
                        execution_mode="shared_session_tp1",
                        reason_code=("trusted_empirical_tp1_reset_gate_passed"),
                        normalized_process_key=key,
                        reset_authority_sha256=authority_sha256,
                        max_member_count=max_member_count,
                        max_estimated_duration_seconds=(max_estimated_duration_seconds),
                    )
                )
            chunk = []
            duration = 0.0

        for row in ordered:
            would_exceed = len(chunk) >= max_member_count or (
                bool(chunk)
                and duration + row.estimated_duration_seconds
                > max_estimated_duration_seconds
            )
            if would_exceed:
                flush()
            chunk.append(row)
            duration += row.estimated_duration_seconds
            if row.estimated_duration_seconds > max_estimated_duration_seconds:
                flush()
        flush()

    plans = fresh + shared_or_singleton
    return tuple(
        sorted(
            plans,
            key=lambda plan: (
                plan.members[0].sort_key,
                plan.execution_mode,
                plan.group_id,
            ),
        )
    )


__all__ = (
    "FORMAL_SERVING_NORMALIZED_PROCESS_KEY_PROTOCOL_SHA256",
    "FORMAL_SERVING_SESSION_GROUP_PROTOCOL_SHA256",
    "FormalServingNormalizedProcessKey",
    "FormalServingSessionExecutionMode",
    "FormalServingSessionGroupPlan",
    "FormalServingSessionGroupSpec",
    "FormalServingSessionReuseExcluded",
    "build_formal_serving_session_group_spec",
    "formal_serving_session_reuse_exclusion_reason",
    "normalized_formal_serving_process_key",
    "partition_formal_serving_session_groups",
)
