"""Runtime exactness and CUDA publication helpers."""

from .attestation import (
    NO_TRUSTED_ATTESTERS,
    RELEASE_TRUSTED_ATTESTER_POLICY,
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
    attestation_message,
    require_release_trusted_attester_policy,
    verify_attestation_signature,
)
from .attester_bundle import (
    SOURCE_RELEASE_TRUSTED_ATTESTER_ANCHOR,
    AttestationNoncePolicy,
    LoadedTrustedAttesterPolicyBundle,
    TrustedAttesterAnchorDescriptor,
    TrustedAttesterPolicyBundle,
    TrustedAttesterPolicyBundleBinding,
    load_source_release_trusted_attester_policy_bundle,
    load_trusted_attester_policy_bundle,
)
from .compile_cache import (
    COMPILE_CACHE_ENVIRONMENT_VARIABLES,
    PINNED_SGLANG_COMPILE_SOURCE_SHA256,
    CompileCacheAttemptReceipt,
    CompileCacheCorruptionError,
    CompileCacheFile,
    CompileCacheForeignIdentityError,
    CompileCacheIncompleteError,
    CompileCacheKey,
    CompileCacheLaunchPlan,
    CompileCacheLaunchSession,
    CompileCacheOverlay,
    CompileCacheReceipt,
    ImmutableCompileCache,
    preflight_compile_cache_launch,
    start_compile_cache_launch,
)

__all__ = [
    "COMPILE_CACHE_ENVIRONMENT_VARIABLES",
    "NO_TRUSTED_ATTESTERS",
    "PINNED_SGLANG_COMPILE_SOURCE_SHA256",
    "RELEASE_TRUSTED_ATTESTER_POLICY",
    "SOURCE_RELEASE_TRUSTED_ATTESTER_ANCHOR",
    "AllRankPublicationCoordinator",
    "AttestationChallenge",
    "AttestationNoncePolicy",
    "BackendContract",
    "BackendPayload",
    "BackendRegistry",
    "CanvasReconstruction",
    "CohortRouteIdentity",
    "CompileCacheAttemptReceipt",
    "CompileCacheCorruptionError",
    "CompileCacheFile",
    "CompileCacheForeignIdentityError",
    "CompileCacheIncompleteError",
    "CompileCacheKey",
    "CompileCacheLaunchPlan",
    "CompileCacheLaunchSession",
    "CompileCacheOverlay",
    "CompileCacheReceipt",
    "CudaPublicationCoordinator",
    "DFlashBackendContract",
    "DSparkBackendContract",
    "DifferentiableCanvasContract",
    "EagleBackendContract",
    "FunctionalBackendContract",
    "GlooPublicationTransport",
    "ImmutableCompileCache",
    "InferenceParameterOwnership",
    "LoadedTrustedAttesterPolicyBundle",
    "NextNBackendContract",
    "ParameterOwnership",
    "PrepareDisposition",
    "PreparedPublication",
    "ProposalEvidence",
    "PublicationCandidate",
    "PublicationDecision",
    "PublicationOutcome",
    "RankDecisionReceipt",
    "RankPrepare",
    "RankTopologyReceipt",
    "Reconstruction",
    "ReplicaLocalRouter",
    "SignedAttestation",
    "TopologyIdentity",
    "TopologyReceiptSet",
    "TrustedAttesterAnchorDescriptor",
    "TrustedAttesterPolicy",
    "TrustedAttesterPolicyBundle",
    "TrustedAttesterPolicyBundleBinding",
    "UpdateIdentity",
    "attestation_message",
    "dspark_composite_loss",
    "dspark_conditional_survival_target",
    "load_source_release_trusted_attester_policy_bundle",
    "load_trusted_attester_policy_bundle",
    "position_weighted_kl",
    "preflight_compile_cache_launch",
    "rejection_sample",
    "require_release_trusted_attester_policy",
    "start_compile_cache_launch",
    "validate_decision_receipts",
    "verify_attestation_signature",
]


def __getattr__(name: str) -> object:
    backend_exports = {
        "BackendContract",
        "BackendPayload",
        "BackendRegistry",
        "DFlashBackendContract",
        "DSparkBackendContract",
        "EagleBackendContract",
        "FunctionalBackendContract",
        "NextNBackendContract",
        "ProposalEvidence",
        "Reconstruction",
        "dspark_composite_loss",
        "dspark_conditional_survival_target",
    }
    canvas_exports = {
        "CanvasReconstruction",
        "DifferentiableCanvasContract",
        "position_weighted_kl",
    }
    distributed_exports = {
        "AllRankPublicationCoordinator",
        "CohortRouteIdentity",
        "GlooPublicationTransport",
        "InferenceParameterOwnership",
        "ParameterOwnership",
        "PrepareDisposition",
        "PreparedPublication",
        "PublicationCandidate",
        "PublicationDecision",
        "PublicationOutcome",
        "RankDecisionReceipt",
        "RankPrepare",
        "RankTopologyReceipt",
        "ReplicaLocalRouter",
        "TopologyIdentity",
        "TopologyReceiptSet",
        "UpdateIdentity",
        "validate_decision_receipts",
    }
    if name in backend_exports:
        from . import backend

        return getattr(backend, name)
    if name in canvas_exports:
        from . import dflash_canvas

        return getattr(dflash_canvas, name)
    if name in distributed_exports:
        from . import distributed

        return getattr(distributed, name)
    if name == "rejection_sample":
        from .exactness import rejection_sample

        return rejection_sample
    if name == "CudaPublicationCoordinator":
        from .publication import CudaPublicationCoordinator

        return CudaPublicationCoordinator
    raise AttributeError(name)
