"""Static/TTS/L0 speed-study protocol."""

from .data import ControlledWindow, LongContinuationAdapter
from .evidence import GpuEvidenceAttestation, evidence_files_sha256
from .onlinespec import (
    OnlineSpecCandidate,
    OnlineSpecGpuAttestation,
    OnlineSpecManifest,
    OnlineSpecSelection,
    OnlineSpecTuningMeasurement,
    compare_onlinespec,
    onlinespec_candidates,
    select_onlinespec,
)
from .protocol import (
    ConfirmationBlock,
    TuningCandidate,
    confirmation_blocks,
    tuning_candidates,
)
from .sampling import SamplingProfile
from .selection import (
    CandidateMeasurement,
    SelectionArtifact,
    select_shared_config,
)
from .statistics import SpeedGate, evaluate_speed_gate

__all__ = [
    "CandidateMeasurement",
    "ConfirmationBlock",
    "ControlledWindow",
    "GpuEvidenceAttestation",
    "LongContinuationAdapter",
    "OnlineSpecCandidate",
    "OnlineSpecGpuAttestation",
    "OnlineSpecManifest",
    "OnlineSpecSelection",
    "OnlineSpecTuningMeasurement",
    "SamplingProfile",
    "SelectionArtifact",
    "SpeedGate",
    "TuningCandidate",
    "compare_onlinespec",
    "confirmation_blocks",
    "evaluate_speed_gate",
    "evidence_files_sha256",
    "onlinespec_candidates",
    "select_onlinespec",
    "select_shared_config",
    "tuning_candidates",
]
