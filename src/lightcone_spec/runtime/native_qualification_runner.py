"""Source-owned live GPU qualification for native runtime/backend suites.

The remote machine receives no release signing key.  This runner executes one
exact test file from the pinned patched SGLang checkout, requires clean GPU
process boundaries, and publishes a live observation, exact JUnit, and an
unsigned :class:`NativeRuntimeGpuProofReceipt`.  A local verifier must later
bind that receipt with an external release control and atomic replay record.

CPU tests can validate the state machine and failure behavior.  They cannot
turn any suite into GPU-verified evidence.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.config import load_run_config
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    VerifiedControlArtifact,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.distributed import (
    DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS,
    DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES,
    DISTRIBUTED_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S,
    DistributedRuntimeGpuProofReceipt,
    VerifiedDistributedRuntimeGpuProof,
    build_distributed_runtime_gpu_proof_artifact,
    validate_distributed_runtime_gpu_proof_artifact,
    verify_distributed_runtime_gpu_proof,
)
from lightcone_spec.runtime.preflight_runner import (
    EvidenceFileBinding,
    ExactnessPreflightAssignment,
    ExactnessPreflightResultPointer,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.readiness import (
    NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256,
    NATIVE_RUNTIME_QUALIFICATION_TESTS,
    NATIVE_RUNTIME_RELEASE_CAPABILITY,
    NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S,
    NativeRuntimeGpuProofReceipt,
    VerifiedNativeRuntimeGpuProof,
    build_native_runtime_gpu_proof_artifact,
    validate_native_runtime_gpu_proof_artifact,
    verify_native_runtime_gpu_proof,
)

SourceOwnedNativeQualificationSuite = Literal[
    "chronobelief_gpu_parity",
    "dspark_tp1",
    "dspark_tp2",
    "dspark_dp2",
    "eagle3_tp1",
    "nextn_tp1",
    "nextn_tp2",
    "native_hot_path_tp1",
]
SourceOwnedDistributedQualificationSuite = Literal["tp2_dp1", "tp1_dp2"]
SourceOwnedRuntimeQualificationSuite = (
    SourceOwnedNativeQualificationSuite | SourceOwnedDistributedQualificationSuite
)

NATIVE_RUNTIME_GPU_TEST_FILES: dict[SourceOwnedRuntimeQualificationSuite, str] = {
    "chronobelief_gpu_parity": (
        "test/registered/unit/spec/test_chronobelief_gpu_parity_qualification.py"
    ),
    "dspark_tp1": ("test/registered/unit/spec/test_dspark_live_gpu_qualification.py"),
    "dspark_tp2": (
        "test/registered/unit/spec/test_dspark_tp2_live_gpu_qualification.py"
    ),
    "dspark_dp2": (
        "test/registered/unit/spec/test_dspark_dp2_live_gpu_qualification.py"
    ),
    "eagle3_tp1": ("test/registered/unit/spec/test_eagle3_live_gpu_qualification.py"),
    "nextn_tp1": ("test/registered/unit/spec/test_nextn_tp1_live_gpu_qualification.py"),
    "nextn_tp2": ("test/registered/unit/spec/test_nextn_tp2_live_gpu_qualification.py"),
    "native_hot_path_tp1": (
        "test/registered/unit/spec/test_native_hot_path_live_gpu_qualification.py"
    ),
    "tp2_dp1": ("test/registered/unit/spec/test_tp2_dp1_live_gpu_qualification.py"),
    "tp1_dp2": ("test/registered/unit/spec/test_tp1_dp2_live_gpu_qualification.py"),
}

NATIVE_RUNTIME_GPU_TEST_NAMES: dict[
    SourceOwnedRuntimeQualificationSuite, tuple[str, ...]
] = {
    suite_id: tuple(
        f"test_{name}"
        for name in (
            DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS[suite_id]
            if suite_id in DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS
            else NATIVE_RUNTIME_QUALIFICATION_TESTS[suite_id]
        )
    )
    for suite_id in NATIVE_RUNTIME_GPU_TEST_FILES
}

_SUITE_ALGORITHM: dict[SourceOwnedRuntimeQualificationSuite, str] = {
    "chronobelief_gpu_parity": "DFLASH",
    "dspark_tp1": "DSPARK",
    "dspark_tp2": "DSPARK",
    "dspark_dp2": "DSPARK",
    "eagle3_tp1": "EAGLE3",
    "nextn_tp1": "NEXTN",
    "nextn_tp2": "NEXTN",
    "native_hot_path_tp1": "DFLASH",
    "tp2_dp1": "DFLASH",
    "tp1_dp2": "DFLASH",
}
_SHA256 = frozenset("0123456789abcdef")
_MAX_TIMEOUT_SECONDS = 3_600.0
_PROCESS_GROUP_TERM_SECONDS = 30.0
_GPU_QUERY_TIMEOUT_SECONDS = 30.0


def _runner_protocol_sha256(suite_id: SourceOwnedRuntimeQualificationSuite) -> str:
    if suite_id in DISTRIBUTED_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S:
        return DISTRIBUTED_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[suite_id]
    return NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[suite_id]


def _source_capability_sha256(
    suite_id: SourceOwnedRuntimeQualificationSuite,
) -> str:
    if suite_id in DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES:
        return DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES[suite_id].sha256
    return NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256


def _expected_topology(
    suite_id: SourceOwnedRuntimeQualificationSuite,
) -> tuple[int, int, int]:
    if suite_id in {"nextn_tp2", "dspark_tp2", "tp2_dp1"}:
        return 2, 1, 2
    if suite_id in {"dspark_dp2", "tp1_dp2"}:
        return 1, 2, 2
    return 1, 1, 1


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or "\n" in value
        or "\r" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be canonical single-line text")
    return value


def _absolute_path(label: str, value: object) -> Path:
    path = Path(_require_text(label, value))
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError(f"{label} must be absolute and normalized")
    if path == Path(path.anchor):
        raise ValueError(f"{label} cannot be a filesystem root")
    return path


def _require_child_path(label: str, path: Path, parent: Path) -> None:
    if path.parent != parent:
        raise ValueError(f"{label} must be a direct child of the evidence directory")


def _csv_rows(value: str, *, columns: int, label: str) -> tuple[tuple[str, ...], ...]:
    if not value.strip():
        return ()
    rows = tuple(
        tuple(field.strip() for field in row) for row in csv.reader(io.StringIO(value))
    )
    if any(len(row) != columns or any(not field for field in row) for row in rows):
        raise ValueError(f"{label} output is malformed")
    return rows


@dataclass(frozen=True)
class NativeRuntimeQualificationAssignment:
    """Immutable launch identity for one suite-specific live GPU qualification."""

    schema_version: Literal[1, 2]
    kind: Literal["formal_native_runtime_gpu_qualification_assignment"]
    suite_id: SourceOwnedRuntimeQualificationSuite
    runner_protocol_sha256: str
    registry_sha256: str
    runtime_sha256: str
    topology_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    run_nonce_sha256: str
    gpu_uuids: tuple[str, ...]
    gpu_models: tuple[str, ...]
    launch_manifest: CanonicalJsonProofBinding
    base_exactness_result_pointer: CanonicalJsonProofBinding | None
    eagle3_selector_status: Literal["COMPATIBLE"] | None
    eagle3_compatibility_authority_sha256: str | None
    eagle3_model_selector_sha256: str | None
    python_executable: str
    python_executable_raw_sha256: str
    python_executable_size: int
    nvidia_smi_executable: str
    nvidia_smi_raw_sha256: str
    nvidia_smi_size: int
    evidence_directory: str
    trusted_single_operator_authority: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {1, 2}
            or self.kind != "formal_native_runtime_gpu_qualification_assignment"
            or self.suite_id not in NATIVE_RUNTIME_GPU_TEST_FILES
            or self.runner_protocol_sha256 != _runner_protocol_sha256(self.suite_id)
        ):
            raise ValueError("native qualification assignment schema is unsupported")
        if self.schema_version == 1:
            if self.trusted_single_operator_authority is not None:
                raise ValueError(
                    "signed native qualification cannot carry trusted authority"
                )
        else:
            if self.suite_id not in {
                "chronobelief_gpu_parity",
                "dspark_tp1",
                "dspark_tp2",
                "dspark_dp2",
                "tp2_dp1",
                "tp1_dp2",
            }:
                raise ValueError(
                    "trusted exact-ten qualification suite is not registered"
                )
            if type(self.trusted_single_operator_authority) is not (
                CanonicalJsonProofBinding
            ):
                raise TypeError("trusted qualification requires a path-bound authority")
            trusted = self.trusted_single_operator_authority.reopen()
            if (
                type(trusted) is not dict
                or trusted.get("kind")
                != "formal_single_operator_preflight_qualification_plan"
                or trusted.get("suite_id") != self.suite_id
                or trusted.get("trust_mode") != "trusted_single_operator_no_signature"
                or trusted.get("formal_measurement") is not False
                or trusted.get("launch_manifest") != self.launch_manifest.to_dict()
            ):
                raise ValueError(
                    "trusted qualification authority does not bind this launch"
                )
        for label, digest in (
            ("runner protocol", self.runner_protocol_sha256),
            ("registry", self.registry_sha256),
            ("runtime", self.runtime_sha256),
            ("topology", self.topology_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
            ("run nonce", self.run_nonce_sha256),
            ("Python executable", self.python_executable_raw_sha256),
            ("nvidia-smi executable", self.nvidia_smi_raw_sha256),
        ):
            _require_sha256(f"native qualification {label}", digest)
        expected_tp, expected_dp, expected_gpu_count = _expected_topology(self.suite_id)
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != expected_gpu_count
            or len(set(self.gpu_uuids)) != expected_gpu_count
            or any(not uuid.startswith("GPU-") for uuid in self.gpu_uuids)
            or type(self.gpu_models) is not tuple
            or len(self.gpu_models) != expected_gpu_count
            or any(
                not _require_text("native qualification GPU model", row)
                for row in self.gpu_models
            )
        ):
            raise ValueError("native qualification GPU assignment is invalid")
        if type(self.launch_manifest) is not CanonicalJsonProofBinding:
            raise TypeError(
                "native qualification requires a path-bound launch manifest"
            )
        launch = CompileLaunchManifest.load(self.launch_manifest.absolute_path)
        run_config = load_run_config(launch.run_config_path)
        if (
            launch.sha256 != self.launch_manifest.semantic_sha256
            or launch.patched_sglang_commit != PINNED_SGLANG_COMMIT
            or launch.patched_sglang_tree != PINNED_SGLANG_TREE
            or launch.inventory_sha256 != self.inventory_sha256
            or launch.gpu_uuids != self.gpu_uuids
            or run_config.model.algorithm != _SUITE_ALGORITHM[self.suite_id]
            or run_config.runtime.tensor_parallel_size != expected_tp
            or run_config.runtime.data_parallel_size != expected_dp
            or not run_config.runtime.speculation_enabled
        ):
            raise ValueError("native qualification launch identity differs")
        eagle3_binding = (
            self.eagle3_selector_status,
            self.eagle3_compatibility_authority_sha256,
            self.eagle3_model_selector_sha256,
        )
        if self.suite_id == "eagle3_tp1":
            adaptation = run_config.adaptation
            if (
                self.eagle3_selector_status != "COMPATIBLE"
                or any(value is None for value in eagle3_binding[1:])
                or adaptation is None
                or adaptation.eagle3_qualification_compatibility_authority_sha256
                != self.eagle3_compatibility_authority_sha256
                or adaptation.eagle3_qualification_model_selector_sha256
                != self.eagle3_model_selector_sha256
                or any(
                    value is not None
                    for value in (
                        adaptation.eagle3_e0_execution_authority_sha256,
                        adaptation.eagle3_compatibility_authority_sha256,
                        adaptation.eagle3_model_selector_sha256,
                        adaptation.eagle3_native_gpu_proof_sha256,
                    )
                )
            ):
                raise ValueError(
                    "EAGLE3 qualification requires a COMPATIBLE official selector"
                )
            _require_sha256(
                "EAGLE3 qualification compatibility",
                self.eagle3_compatibility_authority_sha256,
            )
            _require_sha256(
                "EAGLE3 qualification model selector",
                self.eagle3_model_selector_sha256,
            )
        elif any(value is not None for value in eagle3_binding):
            raise ValueError(
                "EAGLE3 selector authority is restricted to eagle3_tp1 qualification"
            )
        if self.suite_id in DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS:
            if (
                type(self.base_exactness_result_pointer)
                is not CanonicalJsonProofBinding
            ):
                raise TypeError(
                    "distributed qualification requires path-bound base exactness"
                )
            base = ExactnessPreflightResultPointer.load(
                self.base_exactness_result_pointer.absolute_path
            )
            base_assignment = ExactnessPreflightAssignment.load(
                base.assignment.absolute_path
            )
            if (
                base.schema_version != (4 if self.schema_version == 1 else 2)
                or base.sha256 != self.base_exactness_result_pointer.semantic_sha256
                or base_assignment.registry_sha256 != self.registry_sha256
                or base_assignment.runtime_sha256 != self.runtime_sha256
                or base_assignment.inventory_sha256 != self.inventory_sha256
                or base_assignment.hardware_envelope_sha256
                != self.hardware_envelope_sha256
                or base_assignment.gpu_uuids != self.gpu_uuids
                or base.junit_xml is None
                or (
                    self.schema_version == 1
                    and base.qualification_proof_artifact is None
                )
            ):
                raise ValueError(
                    "distributed qualification base exactness identity differs"
                )
        elif self.base_exactness_result_pointer is not None:
            raise ValueError(
                "single-suite qualification cannot carry base exactness evidence"
            )
        evidence = _absolute_path(
            "native qualification evidence directory", self.evidence_directory
        )
        if not evidence.is_dir() or evidence.is_symlink():
            raise ValueError("native qualification evidence directory is unavailable")
        for label, path, digest, size in (
            (
                "Python executable",
                self.python_executable,
                self.python_executable_raw_sha256,
                self.python_executable_size,
            ),
            (
                "nvidia-smi executable",
                self.nvidia_smi_executable,
                self.nvidia_smi_raw_sha256,
                self.nvidia_smi_size,
            ),
        ):
            binding = EvidenceFileBinding(
                absolute_path=str(
                    _absolute_path(f"native qualification {label}", path)
                ),
                raw_sha256=digest,
                size=size,
            )
            binding.reopen(label=f"native qualification {label}")
        test_file = (
            Path(launch.patched_sglang_checkout)
            / (NATIVE_RUNTIME_GPU_TEST_FILES[self.suite_id])
        )
        if (
            test_file.is_symlink()
            or not test_file.is_file()
            or test_file.resolve(strict=True).parent == Path(test_file.anchor)
        ):
            raise ValueError("native qualification source-owned test file is absent")

    def to_dict(self) -> dict[str, object]:
        value = {
            **asdict(self),
            "gpu_uuids": list(self.gpu_uuids),
            "gpu_models": list(self.gpu_models),
            "launch_manifest": self.launch_manifest.to_dict(),
            "base_exactness_result_pointer": (
                None
                if self.base_exactness_result_pointer is None
                else self.base_exactness_result_pointer.to_dict()
            ),
        }
        if self.schema_version == 1:
            value.pop("trusted_single_operator_authority")
        else:
            assert self.trusted_single_operator_authority is not None
            value["trusted_single_operator_authority"] = (
                self.trusted_single_operator_authority.to_dict()
            )
        return value

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @cached_property
    def source_identity_sha256(self) -> str:
        return _sha(
            {
                "schema_version": 1,
                "kind": "native_runtime_gpu_qualification_source_identity",
                "assignment_sha256": self.sha256,
                "suite_id": self.suite_id,
                "runner_protocol_sha256": self.runner_protocol_sha256,
                "registry_sha256": self.registry_sha256,
                "runtime_sha256": self.runtime_sha256,
                "launch_manifest_sha256": self.launch_manifest.semantic_sha256,
                "base_exactness_result_pointer_sha256": (
                    None
                    if self.base_exactness_result_pointer is None
                    else self.base_exactness_result_pointer.semantic_sha256
                ),
                "source_capability_sha256": _source_capability_sha256(self.suite_id),
            }
        )

    @cached_property
    def dispatch_lineage_sha256(self) -> str:
        return _sha(
            {
                "schema_version": 1,
                "kind": "native_runtime_gpu_qualification_dispatch_lineage",
                "assignment_sha256": self.sha256,
                "source_identity_sha256": self.source_identity_sha256,
                "inventory_sha256": self.inventory_sha256,
                "hardware_envelope_sha256": self.hardware_envelope_sha256,
                "gpu_uuids": list(self.gpu_uuids),
            }
        )

    @cached_property
    def artifact_stem(self) -> str:
        return f"native-{self.suite_id}-{self.sha256[:16]}"

    def evidence_path(self, suffix: str) -> Path:
        if not suffix or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-."
            for character in suffix
        ):
            raise ValueError("native qualification evidence suffix is invalid")
        return Path(self.evidence_directory) / f"{self.artifact_stem}.{suffix}"

    def write(self, path: str | Path) -> CanonicalJsonProofBinding:
        destination = _absolute_path("native qualification assignment", str(path))
        publish_canonical_json_no_replace(destination, self.to_dict())
        return CanonicalJsonProofBinding.bind(destination, semantic_sha256=self.sha256)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        schema_version = value.get("schema_version") if type(value) is dict else None
        expected = {
            field.name
            for field in fields(cls)
            if field.name != "trusted_single_operator_authority"
        }
        if schema_version == 2:
            expected.add("trusted_single_operator_authority")
        if type(value) is not dict or set(value) != expected:
            raise ValueError("native qualification assignment fields differ")
        row = dict(value)
        row["launch_manifest"] = CanonicalJsonProofBinding.from_dict(
            row["launch_manifest"]
        )
        base = row["base_exactness_result_pointer"]
        row["base_exactness_result_pointer"] = (
            None if base is None else CanonicalJsonProofBinding.from_dict(base)
        )
        trusted = row.pop("trusted_single_operator_authority", None)
        row["trusted_single_operator_authority"] = (
            None if trusted is None else CanonicalJsonProofBinding.from_dict(trusted)
        )
        for name in ("gpu_uuids", "gpu_models"):
            values = row[name]
            if type(values) is not list:
                raise TypeError(f"native qualification {name} must be an array")
            row[name] = tuple(values)
        return cls(**row)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        binding = CanonicalJsonProofBinding.bind(path)
        assignment = cls.from_dict(binding.reopen())
        if assignment.sha256 != binding.semantic_sha256:
            raise ValueError("native qualification assignment identity changed")
        return assignment


@dataclass(frozen=True)
class NativeRuntimeQualificationDispatchAuthority:
    schema_version: Literal[1]
    kind: Literal["formal_native_runtime_gpu_qualification_dispatch_authority"]
    assignment_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    verified_ns: int
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_native_runtime_gpu_qualification_dispatch_authority"
        ):
            raise ValueError("native qualification dispatch schema is unsupported")
        for label, value in (
            ("assignment", self.assignment_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
        ):
            _require_sha256(f"native qualification dispatch {label}", value)
        if type(self.verified_ns) is not int or self.verified_ns < 0:
            raise ValueError("native qualification dispatch time is invalid")
        if type(self.control_attestation) is not ControlArtifactAttestation:
            raise TypeError("native qualification dispatch requires a control envelope")
        if type(self.replay_reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("native qualification dispatch requires replay evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "control_attestation": self.control_attestation.to_dict(),
            "replay_reservation": self.replay_reservation.to_dict(),
        }

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = {field.name for field in fields(cls)}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("native qualification dispatch fields differ")
        row = dict(value)
        row["control_attestation"] = ControlArtifactAttestation.from_dict(
            row["control_attestation"]
        )
        row["replay_reservation"] = ChallengeReplayReservationBinding.from_dict(
            row["replay_reservation"]
        )
        return cls(**row)

    def revalidate(
        self, *, assignment: NativeRuntimeQualificationAssignment
    ) -> VerifiedControlArtifact:
        if (
            self.assignment_sha256 != assignment.sha256
            or self.inventory_sha256 != assignment.inventory_sha256
            or self.hardware_envelope_sha256 != assignment.hardware_envelope_sha256
        ):
            raise ValueError(
                "native qualification dispatch belongs to another assignment"
            )
        subject = self.control_attestation.subject
        if (
            subject.artifact_type != "non_serving_terminal"
            or subject.artifact_sha256 != assignment.sha256
            or subject.protocol_sha256 != assignment.runner_protocol_sha256
            or subject.registry_sha256 != assignment.registry_sha256
            or subject.lineage_sha256 != assignment.dispatch_lineage_sha256
            or self.control_attestation.hardware_envelope_sha256
            != assignment.hardware_envelope_sha256
        ):
            raise ValueError("native qualification dispatch control is not exact")
        verified = verify_release_control_artifact_attestation(
            self.control_attestation,
            expected_inventory_sha256=assignment.inventory_sha256,
            now_ns=self.verified_ns,
            consumed_challenge_sha256s=(),
        )
        challenges = self.replay_reservation.revalidate()
        expected_challenges = tuple(
            sorted(
                {
                    verified.challenge_sha256,
                    verified.deployment_policy_challenge_sha256,
                }
            )
        )
        reservation_sha256 = control_challenge_reservation_sha256(
            (verified,), reserved_ns=self.verified_ns
        )
        if (
            challenges != expected_challenges
            or self.replay_reservation.reservation_sha256 != reservation_sha256
            or self.replay_reservation.reserved_ns != self.verified_ns
        ):
            raise ValueError("native qualification dispatch reservation differs")
        return verified

    def write(self, path: str | Path) -> CanonicalJsonProofBinding:
        publish_canonical_json_no_replace(path, self.to_dict())
        return CanonicalJsonProofBinding.bind(path, semantic_sha256=self.sha256)


def authorize_native_runtime_qualification_dispatch(
    *,
    assignment: NativeRuntimeQualificationAssignment,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
) -> NativeRuntimeQualificationDispatchAuthority:
    """Reserve a root-authorized dispatch before any GPU child is launched."""

    subject = control_attestation.subject
    if (
        subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != assignment.sha256
        or subject.protocol_sha256 != assignment.runner_protocol_sha256
        or subject.registry_sha256 != assignment.registry_sha256
        or subject.lineage_sha256 != assignment.dispatch_lineage_sha256
        or control_attestation.hardware_envelope_sha256
        != assignment.hardware_envelope_sha256
    ):
        raise ValueError("native qualification dispatch control is not exact")
    verified = verify_and_reserve_release_control_artifact_attestations(
        (control_attestation,),
        expected_inventory_sha256=assignment.inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
    )[0]
    reservation_sha256 = control_challenge_reservation_sha256(
        (verified,), reserved_ns=now_ns
    )
    authority = NativeRuntimeQualificationDispatchAuthority(
        schema_version=1,
        kind="formal_native_runtime_gpu_qualification_dispatch_authority",
        assignment_sha256=assignment.sha256,
        inventory_sha256=assignment.inventory_sha256,
        hardware_envelope_sha256=assignment.hardware_envelope_sha256,
        verified_ns=now_ns,
        control_attestation=control_attestation,
        replay_reservation=replay_store.bind_reservation(reservation_sha256),
    )
    authority.revalidate(assignment=assignment)
    return authority


@dataclass(frozen=True)
class NativeRuntimeQualificationObservation:
    """First-party live-server observation emitted by the patched suite."""

    schema_version: Literal[1]
    kind: Literal["source_owned_native_runtime_live_observation"]
    suite_id: SourceOwnedRuntimeQualificationSuite
    runner_protocol_sha256: str
    assignment_sha256: str
    source_capability_sha256: str
    launch_manifest_sha256: str
    prepared_model_content_manifest_sha256: str
    run_config_sha256: str
    inventory_sha256: str
    gpu_uuids: tuple[str, ...]
    completed_test_names: tuple[str, ...]
    server_process_ids: tuple[int, ...]
    rank_terminal_sha256s: tuple[str, ...]
    live_server_receipt_sha256: str
    native_terminal_sha256: str
    native_itl_pointer_sha256: str
    graph_observation_sha256: str
    worker_hook_observation_sha256: str
    scored_request_inputs_sha256: str
    completed_request_count: int
    worker_hook_invocation_count: int
    graph_replay_count: int
    native_timestamp_count: int
    started_ns: int
    finished_ns: int
    actual_sglang_server: Literal[True]
    component_only: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "source_owned_native_runtime_live_observation"
            or self.suite_id not in NATIVE_RUNTIME_GPU_TEST_FILES
            or self.runner_protocol_sha256 != _runner_protocol_sha256(self.suite_id)
            or self.source_capability_sha256 != _source_capability_sha256(self.suite_id)
            or self.completed_test_names != NATIVE_RUNTIME_GPU_TEST_NAMES[self.suite_id]
            or self.actual_sglang_server is not True
            or self.component_only is not False
        ):
            raise ValueError("native qualification live observation is incomplete")
        for label, digest in (
            ("assignment", self.assignment_sha256),
            ("source capability", self.source_capability_sha256),
            ("launch manifest", self.launch_manifest_sha256),
            ("prepared content manifest", self.prepared_model_content_manifest_sha256),
            ("run config", self.run_config_sha256),
            ("inventory", self.inventory_sha256),
            ("live server receipt", self.live_server_receipt_sha256),
            ("native terminal", self.native_terminal_sha256),
            ("native ITL pointer", self.native_itl_pointer_sha256),
            ("graph observation", self.graph_observation_sha256),
            ("worker hook observation", self.worker_hook_observation_sha256),
            ("scored request inputs", self.scored_request_inputs_sha256),
        ):
            _require_sha256(f"native qualification observation {label}", digest)
        _expected_tp, _expected_dp, expected_ranks = _expected_topology(self.suite_id)
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != expected_ranks
            or len(set(self.gpu_uuids)) != expected_ranks
            or type(self.rank_terminal_sha256s) is not tuple
            or len(self.rank_terminal_sha256s) != expected_ranks
            or len(set(self.rank_terminal_sha256s)) != expected_ranks
        ):
            raise ValueError("native qualification observation rank coverage differs")
        for digest in self.rank_terminal_sha256s:
            _require_sha256("native qualification rank terminal", digest)
        if (
            type(self.server_process_ids) is not tuple
            or not self.server_process_ids
            or len(set(self.server_process_ids)) != len(self.server_process_ids)
            or any(type(pid) is not int or pid < 1 for pid in self.server_process_ids)
        ):
            raise ValueError("native qualification server process coverage is invalid")
        for label, value in (
            ("completed request count", self.completed_request_count),
            ("worker hook invocation count", self.worker_hook_invocation_count),
            ("graph replay count", self.graph_replay_count),
            ("native timestamp count", self.native_timestamp_count),
            ("started time", self.started_ns),
            ("finished time", self.finished_ns),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"native qualification {label} must be positive")
        if self.finished_ns < self.started_ns:
            raise ValueError("native qualification observation time moved backwards")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "gpu_uuids": list(self.gpu_uuids),
            "completed_test_names": list(self.completed_test_names),
            "server_process_ids": list(self.server_process_ids),
            "rank_terminal_sha256s": list(self.rank_terminal_sha256s),
        }

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = {field.name for field in fields(cls)}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("native qualification observation fields differ")
        row = dict(value)
        for name in (
            "gpu_uuids",
            "completed_test_names",
            "server_process_ids",
            "rank_terminal_sha256s",
        ):
            values = row[name]
            if type(values) is not list:
                raise TypeError(
                    f"native qualification observation {name} must be an array"
                )
            row[name] = tuple(values)
        return cls(**row)

    def validate_assignment(
        self, assignment: NativeRuntimeQualificationAssignment
    ) -> None:
        launch = CompileLaunchManifest.load(assignment.launch_manifest.absolute_path)
        if (
            self.suite_id != assignment.suite_id
            or self.assignment_sha256 != assignment.sha256
            or self.launch_manifest_sha256 != assignment.launch_manifest.semantic_sha256
            or self.prepared_model_content_manifest_sha256
            != launch.prepared_model_content_manifest_sha256
            or self.run_config_sha256 != launch.run_config_semantic_sha256
            or self.inventory_sha256 != assignment.inventory_sha256
            or self.gpu_uuids != assignment.gpu_uuids
        ):
            raise ValueError(
                "native qualification observation belongs to another launch"
            )


@dataclass(frozen=True)
class NativeRuntimeQualificationResultPointer:
    schema_version: Literal[1]
    kind: Literal["formal_native_runtime_gpu_qualification_result_pointer"]
    assignment: CanonicalJsonProofBinding
    dispatch_authority: CanonicalJsonProofBinding
    before_gpu_snapshot: CanonicalJsonProofBinding
    after_gpu_snapshot: CanonicalJsonProofBinding
    junit_xml: EvidenceFileBinding
    log: EvidenceFileBinding
    live_observation: CanonicalJsonProofBinding
    runner_terminal: CanonicalJsonProofBinding
    unsigned_gpu_proof: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_native_runtime_gpu_qualification_result_pointer"
        ):
            raise ValueError(
                "native qualification result pointer schema is unsupported"
            )
        for value in (
            self.assignment,
            self.dispatch_authority,
            self.before_gpu_snapshot,
            self.after_gpu_snapshot,
            self.live_observation,
            self.runner_terminal,
            self.unsigned_gpu_proof,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError(
                    "native qualification result requires canonical bindings"
                )
        for value in (self.junit_xml, self.log):
            if type(value) is not EvidenceFileBinding:
                raise TypeError("native qualification result requires file bindings")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "assignment": self.assignment.to_dict(),
            "dispatch_authority": self.dispatch_authority.to_dict(),
            "before_gpu_snapshot": self.before_gpu_snapshot.to_dict(),
            "after_gpu_snapshot": self.after_gpu_snapshot.to_dict(),
            "junit_xml": self.junit_xml.to_dict(),
            "log": self.log.to_dict(),
            "live_observation": self.live_observation.to_dict(),
            "runner_terminal": self.runner_terminal.to_dict(),
            "unsigned_gpu_proof": self.unsigned_gpu_proof.to_dict(),
        }

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = {field.name for field in fields(cls)}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("native qualification result pointer fields differ")
        row = dict(value)
        for name in (
            "assignment",
            "dispatch_authority",
            "before_gpu_snapshot",
            "after_gpu_snapshot",
            "live_observation",
            "runner_terminal",
            "unsigned_gpu_proof",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        row["junit_xml"] = EvidenceFileBinding.from_dict(
            row["junit_xml"], label="native qualification JUnit"
        )
        row["log"] = EvidenceFileBinding.from_dict(
            row["log"], label="native qualification log"
        )
        return cls(**row)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        binding = CanonicalJsonProofBinding.bind(path)
        result = cls.from_dict(binding.reopen())
        if result.sha256 != binding.semantic_sha256:
            raise ValueError("native qualification result pointer changed")
        result.validate()
        return result

    def validate(self) -> None:
        assignment = NativeRuntimeQualificationAssignment.from_dict(
            self.assignment.reopen()
        )
        if assignment.sha256 != self.assignment.semantic_sha256:
            raise ValueError("native qualification result assignment changed")
        dispatch = NativeRuntimeQualificationDispatchAuthority.from_dict(
            self.dispatch_authority.reopen()
        )
        dispatch.revalidate(assignment=assignment)
        before = _validate_gpu_snapshot(
            self.before_gpu_snapshot.reopen(), assignment=assignment, phase="before"
        )
        after = _validate_gpu_snapshot(
            self.after_gpu_snapshot.reopen(), assignment=assignment, phase="after"
        )
        if before["status"] != "AVAILABLE" or after["status"] != "AVAILABLE":
            raise ValueError("native qualification GPU snapshots are unavailable")
        if before["compute_process_rows"] or after["compute_process_rows"]:
            raise ValueError("native qualification GPU boundary is not clean")
        self.junit_xml.reopen(label="native qualification JUnit")
        self.log.reopen(label="native qualification log")
        expected_names = tuple(
            sorted(NATIVE_RUNTIME_GPU_TEST_NAMES[assignment.suite_id])
        )
        if _junit_summary(Path(self.junit_xml.absolute_path)) != (
            expected_names,
            8,
            8,
            0,
            0,
            0,
        ):
            raise ValueError("native qualification JUnit is not exact 8/8 zero-skip")
        observation = NativeRuntimeQualificationObservation.from_dict(
            self.live_observation.reopen()
        )
        observation.validate_assignment(assignment)
        terminal = self.runner_terminal.reopen()
        expected_terminal_fields = {
            "schema_version",
            "kind",
            "runner_protocol_sha256",
            "assignment_sha256",
            "suite_id",
            "status",
            "runner_process_id",
            "started_ns",
            "finished_ns",
            "exit_code",
            "process_group_empty",
            "before_gpu_snapshot_sha256",
            "after_gpu_snapshot_sha256",
            "junit_xml_sha256",
            "log_sha256",
            "live_observation_sha256",
            "unsigned_gpu_proof_sha256",
        }
        if (
            set(terminal) != expected_terminal_fields
            or terminal["schema_version"] != 1
            or terminal["kind"] != "formal_native_runtime_gpu_qualification_terminal"
            or terminal["runner_protocol_sha256"] != assignment.runner_protocol_sha256
            or terminal["assignment_sha256"] != assignment.sha256
            or terminal["suite_id"] != assignment.suite_id
            or terminal["status"] != "PASSED"
            or type(terminal["runner_process_id"]) is not int
            or terminal["runner_process_id"] < 1
            or type(terminal["started_ns"]) is not int
            or type(terminal["finished_ns"]) is not int
            or terminal["started_ns"] < 1
            or terminal["finished_ns"] < terminal["started_ns"]
            or terminal["exit_code"] != 0
            or terminal["process_group_empty"] is not True
            or terminal["before_gpu_snapshot_sha256"]
            != self.before_gpu_snapshot.raw_sha256
            or terminal["after_gpu_snapshot_sha256"]
            != self.after_gpu_snapshot.raw_sha256
            or terminal["junit_xml_sha256"] != self.junit_xml.raw_sha256
            or terminal["log_sha256"] != self.log.raw_sha256
            or terminal["live_observation_sha256"] != observation.sha256
            or terminal["unsigned_gpu_proof_sha256"]
            != self.unsigned_gpu_proof.raw_sha256
        ):
            raise ValueError("native qualification runner terminal differs")
        raw_receipt = self.unsigned_gpu_proof.reopen()
        if assignment.suite_id in DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS:
            receipt = DistributedRuntimeGpuProofReceipt.from_dict(raw_receipt)
            base = assignment.base_exactness_result_pointer
            if base is None:
                raise AssertionError("distributed result lost base exactness")
            base_pointer = ExactnessPreflightResultPointer.load(base.absolute_path)
            if (
                receipt.sha256 != self.unsigned_gpu_proof.semantic_sha256
                or receipt.topology_mode != assignment.suite_id
                or receipt.runner_protocol_sha256 != assignment.runner_protocol_sha256
                or receipt.assignment_sha256 != assignment.sha256
                or receipt.qualification_observation_sha256 != observation.sha256
                or receipt.base_exactness_result_pointer_sha256 != base.semantic_sha256
                or receipt.topology_sha256 != assignment.topology_sha256
                or receipt.source_identity_sha256 != assignment.source_identity_sha256
                or receipt.inventory_sha256 != assignment.inventory_sha256
                or receipt.gpu_uuids != assignment.gpu_uuids
                or receipt.hardware_envelope_sha256
                != assignment.hardware_envelope_sha256
                or receipt.run_nonce_sha256 != assignment.run_nonce_sha256
                or base_pointer.junit_xml is None
                or receipt.junit_xml_sha256 != base_pointer.junit_xml.raw_sha256
                or receipt.qualification_junit_xml_sha256 != self.junit_xml.raw_sha256
            ):
                raise ValueError("distributed qualification unsigned proof differs")
        else:
            receipt = NativeRuntimeGpuProofReceipt.from_dict(raw_receipt)
            if (
                receipt.sha256 != self.unsigned_gpu_proof.semantic_sha256
                or receipt.suite_id != assignment.suite_id
                or receipt.runner_protocol_sha256 != assignment.runner_protocol_sha256
                or receipt.assignment_sha256 != assignment.sha256
                or receipt.qualification_observation_sha256 != observation.sha256
                or receipt.topology_sha256 != assignment.topology_sha256
                or receipt.source_identity_sha256 != assignment.source_identity_sha256
                or receipt.inventory_sha256 != assignment.inventory_sha256
                or receipt.gpu_uuids != assignment.gpu_uuids
                or receipt.hardware_envelope_sha256
                != assignment.hardware_envelope_sha256
                or receipt.run_nonce_sha256 != assignment.run_nonce_sha256
                or receipt.junit_xml_sha256 != self.junit_xml.raw_sha256
            ):
                raise ValueError("native qualification unsigned proof differs")


@dataclass(frozen=True)
class NativeRuntimeQualificationFailurePointer:
    schema_version: Literal[1]
    kind: Literal["formal_native_runtime_gpu_qualification_failure_pointer"]
    assignment: CanonicalJsonProofBinding
    dispatch_authority: CanonicalJsonProofBinding
    before_gpu_snapshot: CanonicalJsonProofBinding
    after_gpu_snapshot: CanonicalJsonProofBinding
    log: EvidenceFileBinding
    fatal_terminal: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_native_runtime_gpu_qualification_failure_pointer"
        ):
            raise ValueError(
                "native qualification failure pointer schema is unsupported"
            )
        for value in (
            self.assignment,
            self.dispatch_authority,
            self.before_gpu_snapshot,
            self.after_gpu_snapshot,
            self.fatal_terminal,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError(
                    "native qualification failure requires canonical bindings"
                )
        if type(self.log) is not EvidenceFileBinding:
            raise TypeError("native qualification failure requires a log binding")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "assignment": self.assignment.to_dict(),
            "dispatch_authority": self.dispatch_authority.to_dict(),
            "before_gpu_snapshot": self.before_gpu_snapshot.to_dict(),
            "after_gpu_snapshot": self.after_gpu_snapshot.to_dict(),
            "log": self.log.to_dict(),
            "fatal_terminal": self.fatal_terminal.to_dict(),
        }

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = {field.name for field in fields(cls)}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("native qualification failure pointer fields differ")
        row = dict(value)
        for name in (
            "assignment",
            "dispatch_authority",
            "before_gpu_snapshot",
            "after_gpu_snapshot",
            "fatal_terminal",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        row["log"] = EvidenceFileBinding.from_dict(
            row["log"], label="native qualification failure log"
        )
        return cls(**row)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        binding = CanonicalJsonProofBinding.bind(path)
        pointer = cls.from_dict(binding.reopen())
        if pointer.sha256 != binding.semantic_sha256:
            raise ValueError("native qualification failure pointer changed")
        pointer.validate()
        return pointer

    def validate(self) -> None:
        assignment = NativeRuntimeQualificationAssignment.from_dict(
            self.assignment.reopen()
        )
        if assignment.sha256 != self.assignment.semantic_sha256:
            raise ValueError("native qualification failure assignment changed")
        NativeRuntimeQualificationDispatchAuthority.from_dict(
            self.dispatch_authority.reopen()
        ).revalidate(assignment=assignment)
        _validate_gpu_snapshot(
            self.before_gpu_snapshot.reopen(), assignment=assignment, phase="before"
        )
        _validate_gpu_snapshot(
            self.after_gpu_snapshot.reopen(), assignment=assignment, phase="after"
        )
        self.log.reopen(label="native qualification failure log")
        fatal = self.fatal_terminal.reopen()
        expected = {
            "schema_version",
            "kind",
            "runner_protocol_sha256",
            "assignment_sha256",
            "suite_id",
            "status",
            "error_code",
            "error_type",
            "started_ns",
            "finished_ns",
            "before_gpu_snapshot_sha256",
            "after_gpu_snapshot_sha256",
            "log_sha256",
        }
        if (
            set(fatal) != expected
            or fatal["schema_version"] != 1
            or fatal["kind"] != "formal_native_runtime_gpu_qualification_fatal_terminal"
            or fatal["runner_protocol_sha256"] != assignment.runner_protocol_sha256
            or fatal["assignment_sha256"] != assignment.sha256
            or fatal["suite_id"] != assignment.suite_id
            or fatal["status"] != "ERROR"
            or type(fatal["error_code"]) is not str
            or not fatal["error_code"]
            or type(fatal["error_type"]) is not str
            or not fatal["error_type"]
            or type(fatal["started_ns"]) is not int
            or type(fatal["finished_ns"]) is not int
            or fatal["started_ns"] < 1
            or fatal["finished_ns"] < fatal["started_ns"]
            or fatal["before_gpu_snapshot_sha256"]
            != self.before_gpu_snapshot.raw_sha256
            or fatal["after_gpu_snapshot_sha256"] != self.after_gpu_snapshot.raw_sha256
            or fatal["log_sha256"] != self.log.raw_sha256
        ):
            raise ValueError("native qualification fatal terminal differs")


def _capture_tool_output(command: tuple[str, ...]) -> str:
    result = subprocess.run(
        command,
        env={"LANG": "C", "LC_ALL": "C"},
        check=True,
        capture_output=True,
        text=True,
        timeout=_GPU_QUERY_TIMEOUT_SECONDS,
    )
    if result.stderr or len(result.stdout) > 1024 * 1024:
        raise RuntimeError("native qualification nvidia-smi query is invalid")
    return result.stdout


def _gpu_snapshot(
    assignment: NativeRuntimeQualificationAssignment,
    *,
    phase: Literal["before", "after"],
) -> dict[str, object]:
    base = {
        "schema_version": 1,
        "kind": "formal_native_runtime_gpu_qualification_snapshot",
        "runner_protocol_sha256": assignment.runner_protocol_sha256,
        "assignment_sha256": assignment.sha256,
        "inventory_sha256": assignment.inventory_sha256,
        "phase": phase,
        "captured_ns": time.monotonic_ns(),
        "gpu_uuids": list(assignment.gpu_uuids),
    }
    try:
        gpu_output = _capture_tool_output(
            (
                assignment.nvidia_smi_executable,
                "--query-gpu=uuid,name,memory.used",
                "--format=csv,noheader,nounits",
            )
        )
        gpu_by_uuid: dict[str, dict[str, object]] = {}
        for uuid, name, memory in _csv_rows(
            gpu_output, columns=3, label="native qualification GPU"
        ):
            if uuid in gpu_by_uuid:
                raise ValueError("native qualification GPU UUID is duplicated")
            gpu_by_uuid[uuid] = {
                "uuid": uuid,
                "name": name,
                "memory_used_mib": int(memory),
            }
        gpu_rows = tuple(gpu_by_uuid.get(uuid) for uuid in assignment.gpu_uuids)
        if (
            any(row is None for row in gpu_rows)
            or tuple(row["name"] for row in gpu_rows if row is not None)
            != assignment.gpu_models
            or any(
                type(row["memory_used_mib"]) is not int or row["memory_used_mib"] < 0
                for row in gpu_rows
                if row is not None
            )
        ):
            raise ValueError("native qualification GPU inventory differs")
        process_output = _capture_tool_output(
            (
                assignment.nvidia_smi_executable,
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            )
        )
        process_rows = []
        observed: set[tuple[str, int]] = set()
        for uuid, raw_pid, raw_memory in _csv_rows(
            process_output, columns=3, label="native qualification compute process"
        ):
            if uuid not in assignment.gpu_uuids:
                continue
            pid = int(raw_pid)
            memory = int(raw_memory)
            if pid < 1 or memory < 0 or (uuid, pid) in observed:
                raise ValueError("native qualification compute process is invalid")
            observed.add((uuid, pid))
            process_rows.append(
                {"gpu_uuid": uuid, "pid": pid, "used_gpu_memory_mib": memory}
            )
        process_rows.sort(key=lambda row: (row["gpu_uuid"], row["pid"]))
        return {
            **base,
            "status": "AVAILABLE",
            "gpu_rows": list(gpu_rows),
            "compute_process_rows": process_rows,
            "error_code": None,
        }
    except (OSError, subprocess.SubprocessError, TypeError, ValueError):
        return {
            **base,
            "status": "ERROR",
            "gpu_rows": [],
            "compute_process_rows": [],
            "error_code": "nvidia_smi_snapshot_unavailable",
        }


def _validate_gpu_snapshot(
    value: object,
    *,
    assignment: NativeRuntimeQualificationAssignment,
    phase: Literal["before", "after"],
) -> dict[str, object]:
    expected = {
        "schema_version",
        "kind",
        "runner_protocol_sha256",
        "assignment_sha256",
        "inventory_sha256",
        "phase",
        "captured_ns",
        "gpu_uuids",
        "status",
        "gpu_rows",
        "compute_process_rows",
        "error_code",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("native qualification GPU snapshot fields differ")
    row = dict(value)
    if (
        row["schema_version"] != 1
        or row["kind"] != "formal_native_runtime_gpu_qualification_snapshot"
        or row["runner_protocol_sha256"] != assignment.runner_protocol_sha256
        or row["assignment_sha256"] != assignment.sha256
        or row["inventory_sha256"] != assignment.inventory_sha256
        or row["phase"] != phase
        or type(row["captured_ns"]) is not int
        or row["captured_ns"] < 1
        or row["gpu_uuids"] != list(assignment.gpu_uuids)
        or row["status"] not in {"AVAILABLE", "ERROR"}
        or type(row["gpu_rows"]) is not list
        or type(row["compute_process_rows"]) is not list
    ):
        raise ValueError("native qualification GPU snapshot identity differs")
    if row["status"] == "ERROR":
        if (
            row["gpu_rows"]
            or row["compute_process_rows"]
            or row["error_code"] != "nvidia_smi_snapshot_unavailable"
        ):
            raise ValueError("native qualification failed GPU snapshot is malformed")
        return row
    if row["error_code"] is not None or len(row["gpu_rows"]) != len(
        assignment.gpu_uuids
    ):
        raise ValueError("native qualification available GPU snapshot is malformed")
    for index, gpu in enumerate(row["gpu_rows"]):
        if (
            type(gpu) is not dict
            or set(gpu) != {"uuid", "name", "memory_used_mib"}
            or gpu["uuid"] != assignment.gpu_uuids[index]
            or gpu["name"] != assignment.gpu_models[index]
            or type(gpu["memory_used_mib"]) is not int
            or gpu["memory_used_mib"] < 0
        ):
            raise ValueError("native qualification GPU row differs")
    observed: set[tuple[str, int]] = set()
    for process in row["compute_process_rows"]:
        if (
            type(process) is not dict
            or set(process) != {"gpu_uuid", "pid", "used_gpu_memory_mib"}
            or process["gpu_uuid"] not in assignment.gpu_uuids
            or type(process["pid"]) is not int
            or process["pid"] < 1
            or (process["gpu_uuid"], process["pid"]) in observed
            or type(process["used_gpu_memory_mib"]) is not int
            or process["used_gpu_memory_mib"] < 0
        ):
            raise ValueError("native qualification compute-process row differs")
        observed.add((process["gpu_uuid"], process["pid"]))
    return row


def _junit_summary(path: Path) -> tuple[tuple[str, ...], int, int, int, int, int]:
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    names = tuple(sorted(case.attrib.get("name", "") for case in cases))
    failed = sum(case.find("failure") is not None for case in cases)
    errored = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    return (
        names,
        len(cases),
        len(cases) - failed - errored - skipped,
        failed,
        errored,
        skipped,
    )


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(process: subprocess.Popen[bytes]) -> tuple[bool, bool]:
    """Return ``(empty, required_sigkill)`` after bounded cleanup."""

    if not _process_group_exists(process.pid):
        return True, False
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=_PROCESS_GROUP_TERM_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30.0)
        return not _process_group_exists(process.pid), True
    return not _process_group_exists(process.pid), False


def _publish_snapshot(
    assignment: NativeRuntimeQualificationAssignment,
    *,
    phase: Literal["before", "after"],
) -> CanonicalJsonProofBinding:
    path = assignment.evidence_path(f"{phase}.json")
    value = _gpu_snapshot(assignment, phase=phase)
    publish_canonical_json_no_replace(path, value)
    binding = CanonicalJsonProofBinding.bind(path)
    _validate_gpu_snapshot(binding.reopen(), assignment=assignment, phase=phase)
    return binding


def _publish_failure(
    *,
    assignment: NativeRuntimeQualificationAssignment,
    assignment_binding: CanonicalJsonProofBinding,
    dispatch_binding: CanonicalJsonProofBinding,
    before_binding: CanonicalJsonProofBinding,
    after_binding: CanonicalJsonProofBinding,
    log_binding: EvidenceFileBinding,
    error_code: str,
    error: Exception,
    started_ns: int,
    finished_ns: int,
) -> CanonicalJsonProofBinding:
    fatal = {
        "schema_version": 1,
        "kind": "formal_native_runtime_gpu_qualification_fatal_terminal",
        "runner_protocol_sha256": assignment.runner_protocol_sha256,
        "assignment_sha256": assignment.sha256,
        "suite_id": assignment.suite_id,
        "status": "ERROR",
        "error_code": _require_text("native qualification error code", error_code),
        "error_type": type(error).__name__,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "before_gpu_snapshot_sha256": before_binding.raw_sha256,
        "after_gpu_snapshot_sha256": after_binding.raw_sha256,
        "log_sha256": log_binding.raw_sha256,
    }
    fatal_path = assignment.evidence_path("fatal.json")
    publish_canonical_json_no_replace(fatal_path, fatal)
    fatal_binding = CanonicalJsonProofBinding.bind(fatal_path)
    pointer = NativeRuntimeQualificationFailurePointer(
        schema_version=1,
        kind="formal_native_runtime_gpu_qualification_failure_pointer",
        assignment=assignment_binding,
        dispatch_authority=dispatch_binding,
        before_gpu_snapshot=before_binding,
        after_gpu_snapshot=after_binding,
        log=log_binding,
        fatal_terminal=fatal_binding,
    )
    pointer_path = assignment.evidence_path("failure-pointer.json")
    publish_canonical_json_no_replace(pointer_path, pointer.to_dict())
    return CanonicalJsonProofBinding.bind(pointer_path, semantic_sha256=pointer.sha256)


class NativeRuntimeQualificationError(RuntimeError):
    def __init__(
        self,
        reason_code: str,
        *,
        failure_pointer: CanonicalJsonProofBinding,
    ) -> None:
        super().__init__(f"native runtime GPU qualification failed: {reason_code}")
        self.reason_code = reason_code
        self.failure_pointer = failure_pointer


_SOURCE_OWNED_QUALIFICATION_TIMEOUT_SECONDS = 1_800.0


def execute_native_runtime_qualification(
    assignment_path: str | Path,
    dispatch_authority_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Execute one exact patched live suite and publish immutable raw evidence."""

    timeout_seconds = _SOURCE_OWNED_QUALIFICATION_TIMEOUT_SECONDS
    if not 1.0 <= timeout_seconds <= _MAX_TIMEOUT_SECONDS:
        raise RuntimeError("source-owned native qualification timeout is invalid")
    assignment_binding = CanonicalJsonProofBinding.bind(assignment_path)
    assignment = NativeRuntimeQualificationAssignment.from_dict(
        assignment_binding.reopen()
    )
    if assignment.sha256 != assignment_binding.semantic_sha256:
        raise ValueError("native qualification assignment identity changed")
    dispatch_binding = CanonicalJsonProofBinding.bind(dispatch_authority_path)
    dispatch = NativeRuntimeQualificationDispatchAuthority.from_dict(
        dispatch_binding.reopen()
    )
    dispatch.revalidate(assignment=assignment)
    evidence = Path(assignment.evidence_directory)
    _require_child_path(
        "native qualification assignment",
        Path(assignment_binding.absolute_path),
        evidence,
    )
    _require_child_path(
        "native qualification dispatch", Path(dispatch_binding.absolute_path), evidence
    )

    before_binding = _publish_snapshot(assignment, phase="before")
    log_path = assignment.evidence_path("runner.log")
    junit_path = assignment.evidence_path("junit.xml")
    observation_path = assignment.evidence_path("live-observation.json")
    receipt_path = assignment.evidence_path("unsigned-proof.json")
    terminal_path = assignment.evidence_path("terminal.json")
    result_path = assignment.evidence_path("result-pointer.json")
    started_ns = time.monotonic_ns()
    process: subprocess.Popen[bytes] | None = None
    execution_error: Exception | None = None
    error_code = "qualification_runner_failed"
    after_binding: CanonicalJsonProofBinding | None = None

    before = _validate_gpu_snapshot(
        before_binding.reopen(), assignment=assignment, phase="before"
    )
    with log_path.open("xb") as log_file:
        log_file.write(b"source-owned native runtime GPU qualification\n")
        if before["status"] != "AVAILABLE" or before["compute_process_rows"]:
            execution_error = RuntimeError("native qualification GPU is not clean")
            error_code = "gpu_precondition_not_clean"
        else:
            launch = CompileLaunchManifest.load(
                assignment.launch_manifest.absolute_path
            )
            test_file = (
                Path(launch.patched_sglang_checkout)
                / (NATIVE_RUNTIME_GPU_TEST_FILES[assignment.suite_id])
            )
            node_ids = tuple(
                f"{test_file}::{name}"
                for name in NATIVE_RUNTIME_GPU_TEST_NAMES[assignment.suite_id]
            )
            command = (
                assignment.python_executable,
                "-m",
                "pytest",
                "-q",
                *node_ids,
                f"--junitxml={junit_path}",
            )
            environment = launch.child_environment()
            environment.update(
                {
                    "PYTHONPATH": launch.patched_sglang_checkout,
                    "LIGHTCONE_NATIVE_QUALIFICATION_ASSIGNMENT_PATH": (
                        assignment_binding.absolute_path
                    ),
                    "LIGHTCONE_NATIVE_QUALIFICATION_ASSIGNMENT_SHA256": (
                        assignment.sha256
                    ),
                    "LIGHTCONE_NATIVE_QUALIFICATION_RUNNER_PROTOCOL_SHA256": (
                        assignment.runner_protocol_sha256
                    ),
                    "LIGHTCONE_NATIVE_QUALIFICATION_SOURCE_CAPABILITY_SHA256": (
                        _source_capability_sha256(assignment.suite_id)
                    ),
                    "LIGHTCONE_NATIVE_QUALIFICATION_DISPATCH_PATH": (
                        dispatch_binding.absolute_path
                    ),
                    "LIGHTCONE_NATIVE_QUALIFICATION_DISPATCH_SHA256": (
                        dispatch_binding.semantic_sha256
                    ),
                    "LIGHTCONE_NATIVE_QUALIFICATION_OBSERVATION_PATH": str(
                        observation_path
                    ),
                    "LIGHTCONE_COMPILE_LAUNCH_MANIFEST_PATH": (
                        assignment.launch_manifest.absolute_path
                    ),
                    "LIGHTCONE_COMPILE_LAUNCH_MANIFEST_SHA256": (
                        assignment.launch_manifest.semantic_sha256
                    ),
                }
            )
            try:
                process = subprocess.Popen(
                    command,
                    cwd=launch.patched_sglang_checkout,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
                try:
                    process.wait(timeout=float(timeout_seconds))
                except subprocess.TimeoutExpired as error:
                    execution_error = error
                    error_code = "qualification_timeout"
                if process.poll() is None or _process_group_exists(process.pid):
                    group_empty, required_sigkill = _terminate_process_group(process)
                    if execution_error is None:
                        execution_error = RuntimeError(
                            "native qualification left a child process group"
                        )
                        error_code = (
                            "qualification_cleanup_required_sigkill"
                            if required_sigkill
                            else "qualification_cleanup_required_sigterm"
                        )
                    if not group_empty:
                        execution_error = RuntimeError(
                            "native qualification process group survived cleanup"
                        )
                        error_code = "qualification_cleanup_incomplete"
                if execution_error is None and process.returncode != 0:
                    execution_error = RuntimeError(
                        "native qualification pytest returned nonzero"
                    )
                    error_code = "qualification_pytest_failed"
            except (OSError, subprocess.SubprocessError) as error:
                execution_error = error
                error_code = "qualification_process_failed"
            finally:
                log_file.flush()
                os.fsync(log_file.fileno())

    after_binding = _publish_snapshot(assignment, phase="after")
    after = _validate_gpu_snapshot(
        after_binding.reopen(), assignment=assignment, phase="after"
    )
    if (
        after["status"] != "AVAILABLE" or after["compute_process_rows"]
    ) and execution_error is None:
        execution_error = RuntimeError("native qualification GPU cleanup is not clean")
        error_code = "gpu_postcondition_not_clean"
    log_binding = EvidenceFileBinding.bind(log_path, label="native qualification log")

    if execution_error is None:
        try:
            expected_names = tuple(
                sorted(NATIVE_RUNTIME_GPU_TEST_NAMES[assignment.suite_id])
            )
            if _junit_summary(junit_path) != (expected_names, 8, 8, 0, 0, 0):
                raise ValueError(
                    "native qualification JUnit is not exact 8/8 zero-skip"
                )
            junit_binding = EvidenceFileBinding.bind(
                junit_path, label="native qualification JUnit"
            )
            observation_binding = CanonicalJsonProofBinding.bind(observation_path)
            observation = NativeRuntimeQualificationObservation.from_dict(
                observation_binding.reopen()
            )
            if observation.sha256 != observation_binding.semantic_sha256:
                raise ValueError("native qualification observation identity changed")
            observation.validate_assignment(assignment)
            if assignment.suite_id in DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS:
                base = assignment.base_exactness_result_pointer
                if base is None:
                    raise AssertionError(
                        "distributed qualification lost base exactness"
                    )
                base_pointer = ExactnessPreflightResultPointer.load(base.absolute_path)
                if base_pointer.junit_xml is None:
                    raise ValueError("distributed qualification base JUnit is absent")
                capability = DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES[
                    assignment.suite_id
                ]
                receipt: (
                    DistributedRuntimeGpuProofReceipt | NativeRuntimeGpuProofReceipt
                ) = DistributedRuntimeGpuProofReceipt(
                    schema_version=1,
                    kind="lightcone_distributed_runtime_gpu_proof",
                    topology_mode=assignment.suite_id,
                    topology_sha256=assignment.topology_sha256,
                    runner_protocol_sha256=assignment.runner_protocol_sha256,
                    assignment_sha256=assignment.sha256,
                    qualification_observation_sha256=observation.sha256,
                    base_exactness_result_pointer_sha256=base.semantic_sha256,
                    source_capability_sha256=capability.sha256,
                    pinned_sglang_commit=capability.pinned_sglang_commit,
                    patched_sglang_tree=capability.patched_sglang_tree,
                    semantic_patch_sha256=capability.semantic_patch_sha256,
                    run_nonce_sha256=assignment.run_nonce_sha256,
                    qualification_authority_sha256=(
                        NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256
                    ),
                    source_identity_sha256=assignment.source_identity_sha256,
                    inventory_sha256=assignment.inventory_sha256,
                    gpu_uuids=assignment.gpu_uuids,
                    hardware_envelope_sha256=assignment.hardware_envelope_sha256,
                    junit_xml_sha256=base_pointer.junit_xml.raw_sha256,
                    tests_collected=8,
                    tests_passed=8,
                    tests_failed=0,
                    tests_errored=0,
                    tests_skipped=0,
                    qualification_junit_xml_sha256=junit_binding.raw_sha256,
                    qualification_test_names=(
                        DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS[assignment.suite_id]
                    ),
                    qualification_tests_collected=8,
                    qualification_tests_passed=8,
                    qualification_tests_failed=0,
                    qualification_tests_errored=0,
                    qualification_tests_skipped=0,
                )
            else:
                topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"] = {
                    "nextn_tp2": "tp2_dp1",
                    "dspark_tp2": "tp2_dp1",
                    "dspark_dp2": "tp1_dp2",
                }.get(assignment.suite_id, "tp1_dp1")
                receipt = NativeRuntimeGpuProofReceipt(
                    schema_version=1,
                    kind="lightcone_native_runtime_gpu_proof",
                    suite_id=assignment.suite_id,
                    topology_mode=topology_mode,
                    topology_sha256=assignment.topology_sha256,
                    runner_protocol_sha256=assignment.runner_protocol_sha256,
                    assignment_sha256=assignment.sha256,
                    qualification_observation_sha256=observation.sha256,
                    source_capability_sha256=NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256,
                    pinned_sglang_commit=PINNED_SGLANG_COMMIT,
                    patched_sglang_tree=PINNED_SGLANG_TREE,
                    semantic_patch_sha256=(
                        NATIVE_RUNTIME_RELEASE_CAPABILITY.semantic_patch_sha256
                    ),
                    run_nonce_sha256=assignment.run_nonce_sha256,
                    qualification_authority_sha256=(
                        NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256
                    ),
                    source_identity_sha256=assignment.source_identity_sha256,
                    inventory_sha256=assignment.inventory_sha256,
                    gpu_uuids=assignment.gpu_uuids,
                    hardware_envelope_sha256=assignment.hardware_envelope_sha256,
                    junit_xml_sha256=junit_binding.raw_sha256,
                    test_names=NATIVE_RUNTIME_QUALIFICATION_TESTS[assignment.suite_id],
                    tests_collected=8,
                    tests_passed=8,
                    tests_failed=0,
                    tests_errored=0,
                    tests_skipped=0,
                )
            receipt_binding = receipt.write_unsigned(str(receipt_path))
            finished_ns = time.monotonic_ns()
            terminal = {
                "schema_version": 1,
                "kind": "formal_native_runtime_gpu_qualification_terminal",
                "runner_protocol_sha256": assignment.runner_protocol_sha256,
                "assignment_sha256": assignment.sha256,
                "suite_id": assignment.suite_id,
                "status": "PASSED",
                "runner_process_id": process.pid if process is not None else 0,
                "started_ns": started_ns,
                "finished_ns": finished_ns,
                "exit_code": process.returncode if process is not None else None,
                "process_group_empty": (
                    process is not None and not _process_group_exists(process.pid)
                ),
                "before_gpu_snapshot_sha256": before_binding.raw_sha256,
                "after_gpu_snapshot_sha256": after_binding.raw_sha256,
                "junit_xml_sha256": junit_binding.raw_sha256,
                "log_sha256": log_binding.raw_sha256,
                "live_observation_sha256": observation.sha256,
                "unsigned_gpu_proof_sha256": receipt_binding.raw_sha256,
            }
            publish_canonical_json_no_replace(terminal_path, terminal)
            terminal_binding = CanonicalJsonProofBinding.bind(terminal_path)
            pointer = NativeRuntimeQualificationResultPointer(
                schema_version=1,
                kind="formal_native_runtime_gpu_qualification_result_pointer",
                assignment=assignment_binding,
                dispatch_authority=dispatch_binding,
                before_gpu_snapshot=before_binding,
                after_gpu_snapshot=after_binding,
                junit_xml=junit_binding,
                log=log_binding,
                live_observation=observation_binding,
                runner_terminal=terminal_binding,
                unsigned_gpu_proof=receipt_binding,
            )
            publish_canonical_json_no_replace(result_path, pointer.to_dict())
            result_binding = CanonicalJsonProofBinding.bind(
                result_path, semantic_sha256=pointer.sha256
            )
            NativeRuntimeQualificationResultPointer.load(result_path)
            return result_binding
        except Exception as error:  # noqa: BLE001 - preserve post-run evidence
            execution_error = error
            error_code = "qualification_evidence_validation_failed"

    assert execution_error is not None
    failure_binding = _publish_failure(
        assignment=assignment,
        assignment_binding=assignment_binding,
        dispatch_binding=dispatch_binding,
        before_binding=before_binding,
        after_binding=after_binding,
        log_binding=log_binding,
        error_code=error_code,
        error=execution_error,
        started_ns=started_ns,
        finished_ns=time.monotonic_ns(),
    )
    raise NativeRuntimeQualificationError(
        error_code, failure_pointer=failure_binding
    ) from execution_error


def finalize_native_runtime_qualification(
    result_pointer_path: str | Path,
    *,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
    proof_artifact_path: str | Path,
    expected_root_manifest_sha256: str,
) -> tuple[
    CanonicalJsonProofBinding,
    VerifiedNativeRuntimeGpuProof | VerifiedDistributedRuntimeGpuProof,
]:
    """Lift one pulled unsigned result with a local external release control."""

    result = NativeRuntimeQualificationResultPointer.load(result_pointer_path)
    assignment = NativeRuntimeQualificationAssignment.from_dict(
        result.assignment.reopen()
    )
    observation = NativeRuntimeQualificationObservation.from_dict(
        result.live_observation.reopen()
    )
    if assignment.suite_id in DISTRIBUTED_RUNTIME_QUALIFICATION_TESTS:
        base = assignment.base_exactness_result_pointer
        if base is None:
            raise AssertionError("distributed qualification lost base exactness")
        capability = DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES[assignment.suite_id]
        verified = verify_distributed_runtime_gpu_proof(
            result.unsigned_gpu_proof.absolute_path,
            control_attestation=control_attestation,
            replay_store=replay_store,
            expected_topology_mode=assignment.suite_id,
            expected_topology_sha256=assignment.topology_sha256,
            expected_source_capability_sha256=capability.sha256,
            expected_source_identity_sha256=assignment.source_identity_sha256,
            expected_inventory_sha256=assignment.inventory_sha256,
            expected_gpu_uuids=assignment.gpu_uuids,
            expected_hardware_envelope_sha256=assignment.hardware_envelope_sha256,
            expected_run_nonce_sha256=assignment.run_nonce_sha256,
            now_ns=now_ns,
        )
        artifact = build_distributed_runtime_gpu_proof_artifact(
            receipt_path=result.unsigned_gpu_proof.absolute_path,
            control_attestation=control_attestation,
            replay_store=replay_store,
            verified_proof=verified,
        )
        publish_canonical_json_no_replace(proof_artifact_path, artifact.to_dict())
        artifact_binding = CanonicalJsonProofBinding.bind(
            proof_artifact_path, semantic_sha256=artifact.sha256
        )
        revalidated = validate_distributed_runtime_gpu_proof_artifact(
            artifact_binding.absolute_path,
            expected_topology_mode=assignment.suite_id,
            expected_topology_sha256=assignment.topology_sha256,
            expected_source_identity_sha256=assignment.source_identity_sha256,
            expected_inventory_sha256=assignment.inventory_sha256,
            expected_gpu_uuids=assignment.gpu_uuids,
            expected_hardware_envelope_sha256=assignment.hardware_envelope_sha256,
            expected_assignment_sha256=assignment.sha256,
            expected_qualification_observation_sha256=observation.sha256,
            expected_base_exactness_result_pointer_sha256=base.semantic_sha256,
            expected_root_manifest_sha256=expected_root_manifest_sha256,
            now_ns=now_ns,
        )
    else:
        verified = verify_native_runtime_gpu_proof(
            result.unsigned_gpu_proof.absolute_path,
            control_attestation=control_attestation,
            replay_store=replay_store,
            expected_suite_id=assignment.suite_id,
            expected_topology_sha256=assignment.topology_sha256,
            expected_source_identity_sha256=assignment.source_identity_sha256,
            expected_inventory_sha256=assignment.inventory_sha256,
            expected_gpu_uuids=assignment.gpu_uuids,
            expected_hardware_envelope_sha256=assignment.hardware_envelope_sha256,
            expected_run_nonce_sha256=assignment.run_nonce_sha256,
            now_ns=now_ns,
        )
        artifact = build_native_runtime_gpu_proof_artifact(
            receipt_path=result.unsigned_gpu_proof.absolute_path,
            control_attestation=control_attestation,
            replay_store=replay_store,
            verified_proof=verified,
        )
        publish_canonical_json_no_replace(proof_artifact_path, artifact.to_dict())
        artifact_binding = CanonicalJsonProofBinding.bind(
            proof_artifact_path, semantic_sha256=artifact.sha256
        )
        revalidated = validate_native_runtime_gpu_proof_artifact(
            artifact_binding.absolute_path,
            expected_suite_id=assignment.suite_id,
            expected_topology_sha256=assignment.topology_sha256,
            expected_source_identity_sha256=assignment.source_identity_sha256,
            expected_inventory_sha256=assignment.inventory_sha256,
            expected_gpu_uuids=assignment.gpu_uuids,
            expected_hardware_envelope_sha256=assignment.hardware_envelope_sha256,
            expected_assignment_sha256=assignment.sha256,
            expected_qualification_observation_sha256=observation.sha256,
            expected_root_manifest_sha256=expected_root_manifest_sha256,
            now_ns=now_ns,
        )
    return artifact_binding, revalidated


__all__ = [
    "NATIVE_RUNTIME_GPU_TEST_FILES",
    "NATIVE_RUNTIME_GPU_TEST_NAMES",
    "NativeRuntimeQualificationAssignment",
    "NativeRuntimeQualificationDispatchAuthority",
    "NativeRuntimeQualificationError",
    "NativeRuntimeQualificationFailurePointer",
    "NativeRuntimeQualificationObservation",
    "NativeRuntimeQualificationResultPointer",
    "SourceOwnedDistributedQualificationSuite",
    "SourceOwnedNativeQualificationSuite",
    "SourceOwnedRuntimeQualificationSuite",
    "authorize_native_runtime_qualification_dispatch",
    "execute_native_runtime_qualification",
    "finalize_native_runtime_qualification",
]
