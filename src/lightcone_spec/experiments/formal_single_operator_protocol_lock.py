"""Source-owned ProtocolLock builder for trusted single-operator evidence."""

from __future__ import annotations

from pathlib import Path

from lightcone_spec.experiments.formal_method_authority import (
    load_chronobelief_authority_artifact,
    load_tts_calibration_authority_artifact,
)
from lightcone_spec.experiments.formal_protocol import (
    ProtocolLock,
    code_owned_qualification_source_identities,
)
from lightcone_spec.experiments.formal_registry import (
    formal_runtime_authority_manifest_from_dict,
    protocol_lock_to_dict,
)
from lightcone_spec.experiments.formal_runtime_manifest import (
    build_source_formal_runtime_authority_manifest,
)
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedSingleOperatorContentBundleBinding,
    bind_trusted_source_snapshot,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorJsonBinding,
    publish_formal_single_operator_json_artifact,
)
from lightcone_spec.experiments.formal_stage_execution import (
    load_e1_recipe_anchor_authority_artifact,
)
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.stage_materialization import (
    default_e2_recipe_grid_authority,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

_ENGLISH_PROTOCOL = "docs/en/experiment-protocol.md"
_CHINESE_PROTOCOL = "docs/zh-CN/experiment-protocol.md"


def _tracked_sha256(bundle: object, relative_path: str) -> str:
    source = bundle.source_snapshot
    matches = tuple(row for row in source.files if row.relative_path == relative_path)
    if len(matches) != 1:
        raise ValueError(f"trusted source lacks exact {relative_path}")
    return matches[0].sha256


def build_trusted_single_operator_protocol_lock(
    *,
    protocol_id: str,
    trusted_content_bundle_path: str | Path,
    formal_runtime_authority_manifest_path: str | Path,
    tts_calibration_authority_path: str | Path,
    chronobelief_authority_path: str | Path,
    e1_recipe_anchor_authority_path: str | Path,
) -> ProtocolLock:
    """Rebuild a schema-5 lock from paths and code-owned authorities only."""

    content_binding = TrustedSingleOperatorContentBundleBinding.bind(
        trusted_content_bundle_path
    )
    if content_binding.runtime_binding_status != "BOUND":
        raise ValueError("trusted ProtocolLock requires a runtime-BOUND content bundle")
    bundle = content_binding.reopen()
    source = bind_trusted_source_snapshot(bundle.source_snapshot.repository_root)
    if source != bundle.source_snapshot:
        raise RuntimeError("trusted ProtocolLock source checkout changed")

    runtime_binding = CanonicalJsonProofBinding.bind(
        formal_runtime_authority_manifest_path
    )
    runtime = formal_runtime_authority_manifest_from_dict(runtime_binding.reopen())
    rebuilt_runtime = build_source_formal_runtime_authority_manifest(
        source.repository_root
    )
    if runtime != rebuilt_runtime:
        raise ValueError("trusted ProtocolLock runtime authority differs from source")

    tts = load_tts_calibration_authority_artifact(tts_calibration_authority_path)
    chronobelief = load_chronobelief_authority_artifact(chronobelief_authority_path)
    e1 = load_e1_recipe_anchor_authority_artifact(e1_recipe_anchor_authority_path)
    qualifications = code_owned_qualification_source_identities()
    native = qualifications["native_runtime"]
    compile_identity = qualifications["compile"]
    exactness = qualifications["exactness"]
    lock = ProtocolLock(
        schema_version=5,
        protocol_id=protocol_id,
        code_git_head=source.git_head,
        code_git_tree=source.git_tree,
        # Compile/readiness contracts register the canonical manifest object,
        # while the source snapshot separately retains its exact raw bytes.
        patch_manifest_sha256=source.patch_manifest_semantic_sha256,
        registry_sha256=build_industrial_registry().sha256,
        english_protocol_sha256=_tracked_sha256(bundle, _ENGLISH_PROTOCOL),
        chinese_protocol_sha256=_tracked_sha256(bundle, _CHINESE_PROTOCOL),
        tts_calibration_authority_sha256=tts.authority.sha256,
        chronobelief_authority_sha256=chronobelief.authority.sha256,
        e1_recipe_anchor_authority_sha256=e1.authority.sha256,
        e2_recipe_grid_authority_sha256=default_e2_recipe_grid_authority().sha256,
        formal_runtime_authority_manifest_sha256=runtime.sha256,
        offline_release_trust_root_sha256=None,
        prepared_model_content_authorization_sha256=None,
        formal_workload_e3a_authorization_sha256=None,
        formal_workload_e0_authorization_sha256=None,
        burstgpt_shape_authorization_sha256=None,
        native_runtime_qualification_protocol_sha256=native[0],
        native_runtime_qualification_runner_sha256=native[1],
        native_runtime_qualification_test_set_sha256=native[2],
        compile_qualification_protocol_sha256=compile_identity[0],
        compile_qualification_runner_sha256=compile_identity[1],
        compile_qualification_test_set_sha256=compile_identity[2],
        exactness_qualification_protocol_sha256=exactness[0],
        exactness_qualification_runner_sha256=exactness[1],
        exactness_qualification_test_set_sha256=exactness[2],
        content_source_mode="trusted_single_operator",
        trusted_single_operator_content_bundle_sha256=(content_binding.semantic_sha256),
    )
    # Close the read window over every external method/runtime input.
    if (
        TrustedSingleOperatorContentBundleBinding.bind(trusted_content_bundle_path)
        != content_binding
        or CanonicalJsonProofBinding.bind(formal_runtime_authority_manifest_path)
        != runtime_binding
        or load_tts_calibration_authority_artifact(tts_calibration_authority_path)
        != tts
        or load_chronobelief_authority_artifact(chronobelief_authority_path)
        != chronobelief
        or load_e1_recipe_anchor_authority_artifact(e1_recipe_anchor_authority_path)
        != e1
        or bind_trusted_source_snapshot(source.repository_root) != source
    ):
        raise RuntimeError("trusted ProtocolLock input changed while being built")
    return lock


def publish_trusted_single_operator_protocol_lock(
    lock: ProtocolLock,
    output_path: str | Path,
) -> FormalSingleOperatorJsonBinding:
    if type(lock) is not ProtocolLock or lock.schema_version != 5:
        raise TypeError("trusted ProtocolLock publisher requires schema 5")
    binding = publish_formal_single_operator_json_artifact(
        output_path,
        protocol_lock_to_dict(lock),
    )
    if binding.semantic_sha256 != lock.sha256:
        raise RuntimeError("trusted ProtocolLock publication digest differs")
    return binding


__all__ = [
    "build_trusted_single_operator_protocol_lock",
    "publish_trusted_single_operator_protocol_lock",
]
