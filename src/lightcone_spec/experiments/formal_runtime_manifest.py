"""Source-owned builder for the formal runtime-authority manifest.

Each member binds two deliberately separate identities.  The protocol, runner,
and test-set fields are stable semantic identities exported by their verifier
or derived from this closed layout.  ``source_sha256`` additionally commits to
the exact implementation and test bytes.  This prevents a source edit from
silently preserving a ProtocolLock without corrupting the semantic identities
that downstream reducers compare against their own constants.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from lightcone_spec.experiments.formal_protocol import (
    FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS,
    FormalRuntimeAuthorityManifest,
    FormalRuntimeAuthorityMember,
    content_sha256,
)

FORMAL_RUNTIME_AUTHORITY_ID = "lightcone-formal-runtime-source-authority-v2"
_MAXIMUM_SOURCE_BYTES = 16 * 1024 * 1024
_PROTOCOL_SOURCE = "src/lightcone_spec/experiments/formal_protocol.py"


@dataclass(frozen=True)
class FormalRuntimeSourceLayout:
    member_id: str
    runner_sources: tuple[str, ...]
    test_nodes: tuple[str, ...]


def _layout(
    member_id: str,
    runner_source: str,
    test_source: str,
) -> FormalRuntimeSourceLayout:
    return FormalRuntimeSourceLayout(
        member_id=member_id,
        runner_sources=(runner_source,),
        test_nodes=(test_source,),
    )


def _slo_layout(
    member_id: str,
    runner_source: str,
    test_source: str,
) -> FormalRuntimeSourceLayout:
    """Bind one serving reducer to the shared exact SLO-goodput authority."""

    return FormalRuntimeSourceLayout(
        member_id=member_id,
        runner_sources=(
            runner_source,
            "src/lightcone_spec/experiments/formal_slo_metrics.py",
        ),
        test_nodes=(test_source, "tests/test_formal_slo_metrics.py"),
    )


FORMAL_RUNTIME_SOURCE_LAYOUT = (
    FormalRuntimeSourceLayout(
        member_id="all_stage_execution_mapper",
        runner_sources=(
            "src/lightcone_spec/experiments/formal_stage_execution.py",
            "src/lightcone_spec/experiments/formal_single_operator_e4_execution.py",
        ),
        test_nodes=(
            "tests/test_formal_stage_execution.py",
            "tests/test_formal_single_operator_e4_execution.py",
        ),
    ),
    _layout(
        "download_completion_reducer",
        "src/lightcone_spec/runtime/download_runner.py",
        "tests/test_download_runner.py",
    ),
    _layout(
        "e0_compatibility_reducer",
        "src/lightcone_spec/experiments/e0_stage_authority.py",
        "tests/test_e0_stage_authority.py",
    ),
    _layout(
        "e0_fdr_reducer",
        "src/lightcone_spec/experiments/breadth_fdr_authority.py",
        "tests/test_breadth_fdr_authority.py",
    ),
    _slo_layout(
        "e0_power_prefix_reducer",
        "src/lightcone_spec/experiments/e0_stage_authority.py",
        "tests/test_e0_stage_authority.py",
    ),
    FormalRuntimeSourceLayout(
        member_id="e1_pareto_reducer",
        runner_sources=(
            "src/lightcone_spec/experiments/e1_stage_authority.py",
            "src/lightcone_spec/orchestration/formal_serving_lift.py",
        ),
        test_nodes=(
            "tests/test_formal_stage_execution.py",
            "tests/test_stage_itl_proof.py",
        ),
    ),
    _layout(
        "e1a_verification_reducer",
        "src/lightcone_spec/experiments/downstream_stage_authority.py",
        "tests/test_e0_authority_artifact.py",
    ),
    _layout(
        "e2_successive_halving_reducer",
        "src/lightcone_spec/experiments/e2_stage_authority.py",
        "tests/test_e2_stage_authority.py",
    ),
    _layout(
        "e3a_selection_reducer",
        "src/lightcone_spec/experiments/e3a_stage_authority.py",
        "tests/test_e3a_stage_authority.py",
    ),
    _slo_layout(
        "e3b_confirmation_reducer",
        "src/lightcone_spec/experiments/downstream_stage_authority.py",
        "tests/test_e0_authority_artifact.py",
    ),
    _slo_layout(
        "e3b_power_prefix_reducer",
        "src/lightcone_spec/experiments/downstream_stage_authority.py",
        "tests/test_formal_registry_power_sources.py",
    ),
    _layout(
        "e4_local_factorial_reducer",
        "src/lightcone_spec/experiments/e4_stage_authority.py",
        "tests/test_formal_registry_integration.py",
    ),
    _layout(
        "e4_strength2_screen_reducer",
        "src/lightcone_spec/experiments/e4_stage_authority.py",
        "tests/test_formal_registry_integration.py",
    ),
    _layout(
        "e4_winner_neighborhood_reducer",
        "src/lightcone_spec/experiments/e4_stage_authority.py",
        "tests/test_formal_stage_results.py",
    ),
    _slo_layout(
        "e5_anchor_selection_reducer",
        "src/lightcone_spec/experiments/downstream_stage_authority.py",
        "tests/test_formal_registry_power_sources.py",
    ),
    _slo_layout(
        "e5_confirmation_reducer",
        "src/lightcone_spec/experiments/downstream_stage_authority.py",
        "tests/test_e0_authority_artifact.py",
    ),
    _layout(
        "e5_failure_reducer",
        "src/lightcone_spec/experiments/formal_failure_execution.py",
        "tests/test_formal_failure_execution.py",
    ),
    _slo_layout(
        "e5_power_prefix_reducer",
        "src/lightcone_spec/experiments/downstream_stage_authority.py",
        "tests/test_formal_registry_power_sources.py",
    ),
    _slo_layout(
        "e6_confirmation_reducer",
        "src/lightcone_spec/experiments/e6_stage_authority.py",
        "tests/test_e6_stage_authority.py",
    ),
    _layout(
        "e6_model_compatibility_reducer",
        "src/lightcone_spec/experiments/e6_stage_authority.py",
        "tests/test_e6_stage_authority.py",
    ),
    _slo_layout(
        "e6_power_prefix_reducer",
        "src/lightcone_spec/experiments/e6_stage_authority.py",
        "tests/test_e6_stage_authority.py",
    ),
    _layout(
        "failure_actuator",
        "src/lightcone_spec/experiments/formal_failure_actuator.py",
        "tests/test_formal_failure_actuator.py",
    ),
    FormalRuntimeSourceLayout(
        member_id="gpu_hour_budget_reducer",
        runner_sources=(
            "src/lightcone_spec/experiments/gpu_hour_authority.py",
            "src/lightcone_spec/experiments/formal_gpu_hour_registry.py",
            "src/lightcone_spec/experiments/formal_gpu_hour_proof.py",
            "src/lightcone_spec/runtime/scientific_source_validation.py",
        ),
        test_nodes=(
            "tests/test_gpu_hour_authority.py",
            "tests/test_formal_gpu_hour_registry.py",
            "tests/test_formal_gpu_hour_proof.py",
            "tests/test_gpu_hour_operator_cli.py",
            "tests/test_offline_scientific_signing.py",
        ),
    ),
    _layout(
        "onlinespec_learner",
        "src/lightcone_spec/experiments/onlinespec.py",
        "tests/test_onlinespec_protocol.py",
    ),
    _slo_layout(
        "onlinespec_tuning_reducer",
        "src/lightcone_spec/experiments/e0_stage_authority.py",
        "tests/test_e0_stage_authority.py",
    ),
    _layout(
        "power_energy_sampler",
        "src/lightcone_spec/experiments/runtime_metrics.py",
        "tests/test_runtime_metrics_authority.py",
    ),
    _layout(
        "profiler_runner",
        "src/lightcone_spec/experiments/profiler_authority.py",
        "tests/test_profiler_authority.py",
    ),
    FormalRuntimeSourceLayout(
        member_id="stage_coverage_reducer",
        runner_sources=(
            "src/lightcone_spec/experiments/formal_downstream_prefix.py",
            "src/lightcone_spec/experiments/formal_stage_coverage.py",
            "src/lightcone_spec/experiments/formal_stage_coverage_portable.py",
            "src/lightcone_spec/experiments/formal_materialization_shards.py",
            "src/lightcone_spec/runtime/scientific_source_validation.py",
        ),
        test_nodes=(
            "tests/test_formal_stage_coverage.py",
            "tests/test_formal_stage_coverage_portable.py",
            "tests/test_offline_scientific_signing.py",
            "tests/test_tts_calibration_authority.py",
        ),
    ),
    _layout(
        "stage_materialization_reducer",
        "src/lightcone_spec/experiments/stage_materialization.py",
        "tests/test_signed_stage_materialization.py",
    ),
    _layout(
        "tts_calibration_reducer",
        "src/lightcone_spec/experiments/tts_calibration_authority.py",
        "tests/test_tts_calibration_authority.py",
    ),
)


def _repository_root(value: str | Path) -> Path:
    root = Path(value)
    if not root.is_absolute() or Path(os.path.abspath(root)) != root:
        raise ValueError("formal runtime source root must be absolute and normalized")
    status = root.lstat()
    if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
        raise ValueError("formal runtime source root must be a real directory")
    if not (root / "pyproject.toml").is_file():
        raise ValueError("formal runtime source root is not the project checkout")
    return root


def _source_file_sha256(root: Path, relative_path: str) -> str:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("formal runtime source path is not repository-relative")
    current = root
    for component in relative.parts[:-1]:
        current /= component
        status = current.lstat()
        if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
            raise ValueError("formal runtime source path has an unsafe ancestor")
    path = root / relative
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= _MAXIMUM_SOURCE_BYTES
        ):
            raise ValueError("formal runtime source is not one bounded regular file")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("formal runtime source ended while read")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError("formal runtime source changed while read")
        if os.read(descriptor, 1):
            raise RuntimeError("formal runtime source grew while read")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _default_semantic_sha256(
    *,
    layout: FormalRuntimeSourceLayout,
    identity_kind: str,
) -> str:
    if identity_kind not in {"protocol", "runner", "test_set"}:
        raise ValueError("formal runtime semantic identity kind is invalid")
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": f"lightcone_formal_runtime_{identity_kind}_semantic_identity",
        "member_id": layout.member_id,
    }
    if identity_kind == "runner":
        payload["entrypoint_sources"] = layout.runner_sources
    elif identity_kind == "test_set":
        payload["pytest_nodes"] = layout.test_nodes
    return content_sha256(payload)


def _semantic_identity_overrides() -> dict[str, tuple[str, str | None, str | None]]:
    """Return exact verifier-owned identities for every protocol with one."""

    from lightcone_spec.experiments.breadth_fdr_authority import (
        E0_BREADTH_FDR_PROTOCOL_SHA256,
    )
    from lightcone_spec.experiments.downstream_stage_authority import (
        E1A_VERIFICATION_PROTOCOL_SHA256,
        E3B_CONFIRMATION_PROTOCOL_SHA256,
        E3B_POWER_PREFIX_PROTOCOL_SHA256,
        E5_CONFIRMATION_PROTOCOL_SHA256,
        E5_POWER_AND_ANCHOR_PROTOCOL_SHA256,
    )
    from lightcone_spec.experiments.e1_stage_authority import (
        E1_STAGED_PARETO_PROTOCOL_SHA256,
    )
    from lightcone_spec.experiments.e2_stage_authority import (
        E2_STAGED_HALVING_PROTOCOL_SHA256,
    )
    from lightcone_spec.experiments.e3a_stage_authority import (
        E3A_STAGED_REDUCTION_PROTOCOL_SHA256,
        E3A_STAGED_REDUCTION_RUNNER_SHA256,
        E3A_STAGED_REDUCTION_TEST_SET_SHA256,
    )
    from lightcone_spec.experiments.e4_stage_authority import (
        E4_SELECTION_PROTOCOL_SHA256,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        E6_CONFIRMATION_PROTOCOL_SHA256,
        E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256,
        E6_POWER_PREFIX_PROTOCOL_SHA256,
    )
    from lightcone_spec.experiments.failure_actuator import (
        FAILURE_ACTUATOR_PROTOCOL_SHA256,
    )
    from lightcone_spec.experiments.formal_stage_coverage import (
        FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256,
        FORMAL_STAGE_COVERAGE_RUNNER_SHA256,
        FORMAL_STAGE_COVERAGE_TEST_SET_SHA256,
    )
    from lightcone_spec.experiments.formal_stage_execution import (
        FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256,
        FORMAL_SERVING_EXECUTION_RUNNER_SHA256,
        FORMAL_SERVING_EXECUTION_TEST_SET_SHA256,
    )
    from lightcone_spec.experiments.gpu_hour_authority import (
        FORMAL_GPU_HOUR_BUDGET_PROTOCOL_SHA256,
        FORMAL_GPU_HOUR_BUDGET_RUNNER_SHA256,
        FORMAL_GPU_HOUR_BUDGET_TEST_SET_SHA256,
    )
    from lightcone_spec.experiments.profiler_authority import (
        PROFILER_AUTHORITY_PROTOCOL_SHA256,
    )
    from lightcone_spec.experiments.runtime_metrics import (
        RUNTIME_METRICS_REDUCER_PROTOCOL_SHA256,
    )
    from lightcone_spec.experiments.stage_materialization import (
        E5_FAILURE_DIAGNOSTIC_PROTOCOL_SHA256,
    )
    from lightcone_spec.runtime.download_runner import (
        DOWNLOAD_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
    )

    return {
        "all_stage_execution_mapper": (
            FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256,
            FORMAL_SERVING_EXECUTION_RUNNER_SHA256,
            FORMAL_SERVING_EXECUTION_TEST_SET_SHA256,
        ),
        "download_completion_reducer": (
            DOWNLOAD_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
            None,
            None,
        ),
        "e0_fdr_reducer": (E0_BREADTH_FDR_PROTOCOL_SHA256, None, None),
        "e1_pareto_reducer": (E1_STAGED_PARETO_PROTOCOL_SHA256, None, None),
        "e1a_verification_reducer": (
            E1A_VERIFICATION_PROTOCOL_SHA256,
            None,
            None,
        ),
        "e2_successive_halving_reducer": (
            E2_STAGED_HALVING_PROTOCOL_SHA256,
            None,
            None,
        ),
        "e3a_selection_reducer": (
            E3A_STAGED_REDUCTION_PROTOCOL_SHA256,
            E3A_STAGED_REDUCTION_RUNNER_SHA256,
            E3A_STAGED_REDUCTION_TEST_SET_SHA256,
        ),
        "e3b_confirmation_reducer": (
            E3B_CONFIRMATION_PROTOCOL_SHA256,
            None,
            None,
        ),
        "e3b_power_prefix_reducer": (
            E3B_POWER_PREFIX_PROTOCOL_SHA256,
            None,
            None,
        ),
        "e4_local_factorial_reducer": (E4_SELECTION_PROTOCOL_SHA256, None, None),
        "e4_strength2_screen_reducer": (E4_SELECTION_PROTOCOL_SHA256, None, None),
        "e4_winner_neighborhood_reducer": (
            E4_SELECTION_PROTOCOL_SHA256,
            None,
            None,
        ),
        "e5_anchor_selection_reducer": (
            E5_POWER_AND_ANCHOR_PROTOCOL_SHA256,
            None,
            None,
        ),
        "e5_confirmation_reducer": (E5_CONFIRMATION_PROTOCOL_SHA256, None, None),
        "e5_failure_reducer": (
            E5_FAILURE_DIAGNOSTIC_PROTOCOL_SHA256,
            None,
            None,
        ),
        "e5_power_prefix_reducer": (
            E5_POWER_AND_ANCHOR_PROTOCOL_SHA256,
            None,
            None,
        ),
        "e6_confirmation_reducer": (E6_CONFIRMATION_PROTOCOL_SHA256, None, None),
        "e6_model_compatibility_reducer": (
            E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256,
            None,
            None,
        ),
        "e6_power_prefix_reducer": (
            E6_POWER_PREFIX_PROTOCOL_SHA256,
            None,
            None,
        ),
        "failure_actuator": (FAILURE_ACTUATOR_PROTOCOL_SHA256, None, None),
        "gpu_hour_budget_reducer": (
            FORMAL_GPU_HOUR_BUDGET_PROTOCOL_SHA256,
            FORMAL_GPU_HOUR_BUDGET_RUNNER_SHA256,
            FORMAL_GPU_HOUR_BUDGET_TEST_SET_SHA256,
        ),
        "power_energy_sampler": (
            RUNTIME_METRICS_REDUCER_PROTOCOL_SHA256,
            None,
            None,
        ),
        "profiler_runner": (PROFILER_AUTHORITY_PROTOCOL_SHA256, None, None),
        "stage_coverage_reducer": (
            FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256,
            FORMAL_STAGE_COVERAGE_RUNNER_SHA256,
            FORMAL_STAGE_COVERAGE_TEST_SET_SHA256,
        ),
    }


def build_source_formal_runtime_authority_manifest(
    repository_root: str | Path,
) -> FormalRuntimeAuthorityManifest:
    """Build all authority members from the closed source layout."""

    root = _repository_root(repository_root)
    if tuple(row.member_id for row in FORMAL_RUNTIME_SOURCE_LAYOUT) != (
        FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS
    ):
        raise RuntimeError("formal runtime source layout differs from member allowlist")
    overrides = _semantic_identity_overrides()
    if not set(overrides).issubset(FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS):
        raise RuntimeError("formal runtime semantic overrides are unregistered")
    members = []
    for layout in FORMAL_RUNTIME_SOURCE_LAYOUT:
        test_paths = tuple(
            sorted({node.partition("::")[0] for node in layout.test_nodes})
        )
        if any(not node or node.endswith("::") for node in layout.test_nodes):
            raise ValueError("formal runtime pytest node is invalid")
        default_protocol = _default_semantic_sha256(
            layout=layout,
            identity_kind="protocol",
        )
        default_runner = _default_semantic_sha256(
            layout=layout,
            identity_kind="runner",
        )
        default_test_set = _default_semantic_sha256(
            layout=layout,
            identity_kind="test_set",
        )
        protocol, runner, test_set = overrides.get(
            layout.member_id,
            (default_protocol, default_runner, default_test_set),
        )
        members.append(
            FormalRuntimeAuthorityMember(
                member_id=layout.member_id,
                protocol_sha256=protocol,
                runner_sha256=runner or default_runner,
                test_set_sha256=test_set or default_test_set,
                source_sha256=content_sha256(
                    {
                        "schema_version": 2,
                        "kind": "lightcone_formal_runtime_member_source_commitment",
                        "member_id": layout.member_id,
                        "nodes": layout.test_nodes,
                        "files": tuple(
                            (path, _source_file_sha256(root, path))
                            for path in tuple(
                                sorted(
                                    {
                                        _PROTOCOL_SOURCE,
                                        *layout.runner_sources,
                                        *test_paths,
                                    }
                                )
                            )
                        ),
                    }
                ),
            )
        )
    return FormalRuntimeAuthorityManifest(
        schema_version=2,
        authority_id=FORMAL_RUNTIME_AUTHORITY_ID,
        members=tuple(members),
    )


__all__ = [
    "FORMAL_RUNTIME_AUTHORITY_ID",
    "FORMAL_RUNTIME_SOURCE_LAYOUT",
    "FormalRuntimeSourceLayout",
    "build_source_formal_runtime_authority_manifest",
]
