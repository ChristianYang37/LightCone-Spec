"""Static/TTS/L0 speed-study protocol."""

from lightcone_spec.execution import ControlledExecutionPolicy

from .data import ControlledWindow, LongContinuationAdapter
from .evidence import (
    GpuEvidenceAttestation,
    GreedyTargetReference,
    TargetOutput,
    evidence_files_sha256,
)
from .onlinespec import (
    ONLINE_SPEC_TUNING_STAGES,
    OnlineSpecCandidate,
    OnlineSpecGpuAttestation,
    OnlineSpecManifest,
    OnlineSpecSelection,
    OnlineSpecTuningMeasurement,
    compare_onlinespec,
    onlinespec_candidates,
    onlinespec_tuning_stage,
    select_onlinespec,
    select_onlinespec_heldout_anchor,
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
    select_heldout_anchor,
    select_shared_config,
)
from .statistics import PairwiseSpeedGate, SpeedGate, evaluate_speed_gate

__all__ = [
    "ONLINE_SPEC_TUNING_STAGES",
    "CandidateMeasurement",
    "ConfirmationBlock",
    "ControlledExecutionPolicy",
    "ControlledWindow",
    "GpuEvidenceAttestation",
    "GreedyTargetReference",
    "LongContinuationAdapter",
    "OnlineSpecCandidate",
    "OnlineSpecGpuAttestation",
    "OnlineSpecManifest",
    "OnlineSpecSelection",
    "OnlineSpecTuningMeasurement",
    "PairwiseSpeedGate",
    "SamplingProfile",
    "SelectionArtifact",
    "SpeedGate",
    "TargetOutput",
    "TuningCandidate",
    "compare_onlinespec",
    "confirmation_blocks",
    "evaluate_speed_gate",
    "evidence_files_sha256",
    "onlinespec_candidates",
    "onlinespec_tuning_stage",
    "select_heldout_anchor",
    "select_onlinespec",
    "select_onlinespec_heldout_anchor",
    "select_shared_config",
    "tuning_candidates",
]
