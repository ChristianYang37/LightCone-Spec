"""GPU-resident cohort adaptation primitives."""

from .cohort import CohortIdentity, CohortRuntime, SupervisionSignal
from .kv_history import FrozenKVHistory, KVSegment
from .memory import AdaptationMemoryLedger
from .optimizer import FixedAddressBank, GPUOptimizer
from .parameters import DFlashParameterPlan, LoRAFactors

__all__ = [
    "AdaptationMemoryLedger",
    "CohortIdentity",
    "CohortRuntime",
    "DFlashParameterPlan",
    "FixedAddressBank",
    "FrozenKVHistory",
    "GPUOptimizer",
    "KVSegment",
    "LoRAFactors",
    "SupervisionSignal",
]
