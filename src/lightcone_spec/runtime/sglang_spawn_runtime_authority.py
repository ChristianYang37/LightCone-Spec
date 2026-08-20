"""Spawn-safe revalidation of SGLang runtime GPU authority.

SGLang uses the ``spawn`` multiprocessing start method.  Process-local proof
provider closures therefore cannot be an authority boundary: each worker must
deep-reopen the inherited, path-bound source and reconstruct exact proof
objects before the patched runtime may install its providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from lightcone_spec.experiments.formal_protocol import content_sha256

RankPublicationMode = Literal[
    "none",
    "formal_nccl_v1",
    "qualification_nccl_v1",
]

_VERIFIED_SPAWN_RUNTIME_BRIDGE_SENTINEL = object()
_ADAPTATION_CONFIG_SHA256_ENVIRONMENT = "LIGHTCONE_FORMAL_ADAPTATION_CONFIG_SHA256"


@dataclass(frozen=True, init=False)
class VerifiedSglangSpawnRuntimeAuthorityBridge:
    """Opaque bridge issued only after child-process source replay."""

    schema_version: Literal[1]
    kind: Literal["verified_sglang_spawn_runtime_authority_bridge"]
    source_environment_sha256: str
    proofs_by_role: tuple[tuple[Literal["distributed", "native"], object], ...]
    rank_publication_mode: RankPublicationMode

    def __init__(
        self,
        *,
        source_environment_sha256: str,
        proofs_by_role: tuple[tuple[Literal["distributed", "native"], object], ...],
        rank_publication_mode: RankPublicationMode,
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _VERIFIED_SPAWN_RUNTIME_BRIDGE_SENTINEL:
            raise TypeError("spawn runtime bridge requires source revalidation")
        roles = tuple(row[0] for row in proofs_by_role)
        if (
            len(source_environment_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source_environment_sha256
            )
            or type(proofs_by_role) is not tuple
            or any(type(row) is not tuple or len(row) != 2 for row in proofs_by_role)
            or roles != tuple(sorted(set(roles)))
            or any(role not in {"distributed", "native"} for role in roles)
            or rank_publication_mode
            not in {"none", "formal_nccl_v1", "qualification_nccl_v1"}
            or (rank_publication_mode != "none" and "distributed" not in roles)
        ):
            raise ValueError("spawn runtime bridge identity differs")
        object.__setattr__(self, "schema_version", 1)
        object.__setattr__(
            self, "kind", "verified_sglang_spawn_runtime_authority_bridge"
        )
        object.__setattr__(self, "source_environment_sha256", source_environment_sha256)
        object.__setattr__(self, "proofs_by_role", proofs_by_role)
        object.__setattr__(self, "rank_publication_mode", rank_publication_mode)


def _issue_bridge(
    *,
    source_environment_sha256: str,
    proofs_by_role: tuple[tuple[Literal["distributed", "native"], object], ...],
    rank_publication_mode: RankPublicationMode,
) -> VerifiedSglangSpawnRuntimeAuthorityBridge:
    return VerifiedSglangSpawnRuntimeAuthorityBridge(
        source_environment_sha256=source_environment_sha256,
        proofs_by_role=proofs_by_role,
        rank_publication_mode=rank_publication_mode,
        _verification_tag=_VERIFIED_SPAWN_RUNTIME_BRIDGE_SENTINEL,
    )


def _revalidate_spawn_adaptation_payload(*, launch: object, config: object) -> None:
    from lightcone_spec.sglang_bridge.launch import (
        _bind_runtime_adaptation_config,
    )

    argv = getattr(launch, "server_argv", None)
    if type(argv) is not tuple or argv.count("--") != 1:
        raise ValueError("spawn launch wrapper argv differs")
    inner = list(argv[argv.index("--") + 1 :])
    binding = _bind_runtime_adaptation_config(config, inner)
    expected = os.environ.get(_ADAPTATION_CONFIG_SHA256_ENVIRONMENT)
    if binding is None:
        if expected is not None:
            raise ValueError("allocation-free spawn carries adaptation authority")
    elif binding.semantic_sha256 != expected:
        raise ValueError("spawn adaptation payload authority differs")


def _qualification_bridge() -> VerifiedSglangSpawnRuntimeAuthorityBridge:
    from sglang.srt.speculative.native_runtime_release import (
        NativeRuntimeQualificationBootstrap,
    )

    from lightcone_spec.config import load_run_config
    from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
    from lightcone_spec.runtime.distributed import (
        DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES,
    )
    from lightcone_spec.runtime.readiness import (
        NATIVE_RUNTIME_RELEASE_CAPABILITY,
        NATIVE_RUNTIME_SUITE_CAPABILITIES,
    )
    from lightcone_spec.sglang_bridge.launch import (
        _load_qualification_runtime_bridge_environment,
    )

    assignment, dispatch_sha256 = _load_qualification_runtime_bridge_environment()
    launch = CompileLaunchManifest.load(assignment.launch_manifest.absolute_path)
    config = load_run_config(launch.run_config_path)
    _revalidate_spawn_adaptation_payload(launch=launch, config=config)
    topology_mode = config.runtime.topology_mode
    if (
        topology_mode not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}
        or launch.gpu_uuids != assignment.gpu_uuids
        or tuple(config.runtime.device_identity.split(",")) != assignment.gpu_uuids
    ):
        raise ValueError("spawn qualification topology differs from assignment")
    common = {
        "assignment_sha256": assignment.sha256,
        "dispatch_sha256": dispatch_sha256,
        "suite_id": assignment.suite_id,
        "source_identity_sha256": assignment.source_identity_sha256,
        "inventory_sha256": assignment.inventory_sha256,
        "hardware_envelope_sha256": assignment.hardware_envelope_sha256,
        "topology_mode": topology_mode,
        "topology_sha256": assignment.topology_sha256,
        "gpu_uuids": assignment.gpu_uuids,
        "eagle3_selector_status": assignment.eagle3_selector_status,
        "eagle3_compatibility_authority_sha256": (
            assignment.eagle3_compatibility_authority_sha256
        ),
        "eagle3_model_selector_sha256": assignment.eagle3_model_selector_sha256,
    }
    proofs: list[tuple[Literal["distributed", "native"], object]] = []
    if topology_mode != "tp1_dp1":
        capability = DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES[topology_mode]
        if config.runtime.distributed_release_capability_sha256 != capability.sha256:
            raise ValueError("spawn qualification distributed capability differs")
        proofs.append(
            (
                "distributed",
                NativeRuntimeQualificationBootstrap(
                    **common,
                    source_capability_sha256=capability.sha256,
                    backend_capabilities=(),
                ),
            )
        )
    if config.model.algorithm in {"DSPARK", "NEXTN", "EAGLE3"}:
        proofs.append(
            (
                "native",
                NativeRuntimeQualificationBootstrap(
                    **common,
                    source_capability_sha256=NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256,
                    backend_capabilities=tuple(
                        NATIVE_RUNTIME_SUITE_CAPABILITIES.get(assignment.suite_id, ())
                    ),
                ),
            )
        )
    if not proofs:
        raise ValueError("spawn qualification supplied no runtime authority")
    source_environment_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "qualification_spawn_runtime_authority_environment",
            "assignment_sha256": assignment.sha256,
            "dispatch_sha256": dispatch_sha256,
            "launch_manifest_sha256": launch.sha256,
        }
    )
    return _issue_bridge(
        source_environment_sha256=source_environment_sha256,
        proofs_by_role=tuple(proofs),
        rank_publication_mode=(
            "qualification_nccl_v1" if topology_mode == "tp2_dp1" else "none"
        ),
    )


def revalidate_sglang_spawn_runtime_authority_environment() -> (
    VerifiedSglangSpawnRuntimeAuthorityBridge | None
):
    """Rebuild trusted or qualification authority in the current worker."""

    qualification = os.environ.get("LIGHTCONE_NATIVE_QUALIFICATION_MODE") == "1"
    from lightcone_spec.runtime.trusted_single_operator_runtime import (
        bind_trusted_single_operator_runtime_authority_environment,
        verify_trusted_single_operator_runtime_authority_source,
    )

    trusted_binding = bind_trusted_single_operator_runtime_authority_environment(
        os.environ
    )
    if qualification and trusted_binding is not None:
        raise ValueError("spawn runtime authority lanes are mutually exclusive")
    if qualification:
        return _qualification_bridge()
    if trusted_binding is None:
        return None
    source, tokens = verify_trusted_single_operator_runtime_authority_source(
        trusted_binding.absolute_path,
        expected_source_binding=trusted_binding,
    )
    from lightcone_spec.config import load_run_config
    from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

    launch = CompileLaunchManifest.load(source.launch_manifest.absolute_path)
    _revalidate_spawn_adaptation_payload(
        launch=launch,
        config=load_run_config(launch.run_config_path),
    )
    proofs_by_role = tuple((token.role, token) for token in tokens)
    if tuple(row[0] for row in proofs_by_role) != tuple(
        role.role for role in source.roles
    ):
        raise ValueError("spawn trusted runtime roles differ from source")
    return _issue_bridge(
        source_environment_sha256=content_sha256(
            {
                "schema_version": 1,
                "kind": "trusted_spawn_runtime_authority_environment",
                "source_binding": trusted_binding.to_dict(),
                "source_sha256": source.sha256,
            }
        ),
        proofs_by_role=proofs_by_role,
        rank_publication_mode=(
            "formal_nccl_v1" if source.topology_mode == "tp2_dp1" else "none"
        ),
    )


__all__ = [
    "VerifiedSglangSpawnRuntimeAuthorityBridge",
    "revalidate_sglang_spawn_runtime_authority_environment",
]
