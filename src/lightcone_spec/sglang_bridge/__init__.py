"""HTTP and runtime configuration boundary for patched SGLang.

Keep package import lightweight: the verified launcher must establish its CUDA
allocator contract before any optional client/config export imports Torch.
"""

from .checkout import verify_patched_checkout

__all__ = [
    "GenerationResult",
    "MethodRun",
    "SGLangHTTPClient",
    "ServerSnapshot",
    "independent_method_run",
    "sglang_adaptation_payload",
    "sglang_adaptation_sha256",
    "verify_patched_checkout",
]


def __getattr__(name: str) -> object:
    if name in {
        "GenerationResult",
        "MethodRun",
        "ServerSnapshot",
        "SGLangHTTPClient",
        "independent_method_run",
    }:
        from . import client

        return getattr(client, name)
    if name in {"sglang_adaptation_payload", "sglang_adaptation_sha256"}:
        from . import config

        return getattr(config, name)
    raise AttributeError(name)
