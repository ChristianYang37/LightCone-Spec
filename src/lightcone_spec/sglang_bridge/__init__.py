"""HTTP and runtime configuration boundary for patched SGLang."""

from .checkout import verify_patched_checkout
from .client import (
    GenerationResult,
    MethodRun,
    ServerSnapshot,
    SGLangHTTPClient,
    independent_method_run,
)
from .config import sglang_adaptation_payload, sglang_adaptation_sha256

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
