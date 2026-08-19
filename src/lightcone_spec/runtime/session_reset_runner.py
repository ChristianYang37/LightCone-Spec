"""Single-GPU source-owned session-reset qualification.

The remote host receives no signing key.  It executes an exact manifest-bound
adapted TP1/DP1 server twice (cold reference, then reused-session process),
publishes JUnit, a first-party raw rank terminal, and an unsigned native proof.
Only a later local root-authorized control envelope may lift that receipt into
a durable :class:`NativeRuntimeGpuProofArtifact`.
"""

from __future__ import annotations

import hashlib
import http.client
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
from urllib.parse import urlsplit

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
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.readiness import (
    NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256,
    NATIVE_RUNTIME_QUALIFICATION_TESTS,
    NATIVE_RUNTIME_RELEASE_CAPABILITY,
    NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S,
    NativeRuntimeGpuProofArtifact,
    NativeRuntimeGpuProofReceipt,
    VerifiedNativeRuntimeGpuProof,
    build_native_runtime_gpu_proof_artifact,
    verify_native_runtime_gpu_proof,
)

_SHA256 = frozenset("0123456789abcdef")
SESSION_RESET_GPU_TEST_FILE = (
    "test/registered/unit/spec/test_session_reset_gpu_qualification.py"
)
SESSION_RESET_GPU_TEST_NAMES = tuple(
    f"test_{name}" for name in NATIVE_RUNTIME_QUALIFICATION_TESTS["session_reset_tp1"]
)
SESSION_RESET_RUNNER_PROTOCOL_SHA256 = NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[
    "session_reset_tp1"
]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


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


def _raw_file_sha256(path: Path, *, label: str) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    body = path.read_bytes()
    if not body:
        raise ValueError(f"{label} cannot be empty")
    return hashlib.sha256(body).hexdigest(), len(body)


def _argument_value(argv: tuple[str, ...], flag: str) -> str | None:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if not positions:
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"session launch has malformed {flag}")
    return argv[positions[0] + 1]


@dataclass(frozen=True)
class SessionResetQualificationAssignment:
    schema_version: Literal[1]
    kind: Literal["formal_session_reset_tp1_assignment"]
    protocol_sha256: str
    registry_sha256: str
    runtime_sha256: str
    topology_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    gpu_uuid: str
    gpu_model: str
    run_nonce_sha256: str
    launch_manifest: CanonicalJsonProofBinding
    nvidia_smi_executable: str
    nvidia_smi_raw_sha256: str
    evidence_directory: str
    method: Literal["tts", "l0"]
    input_token_ids: tuple[int, ...]
    max_new_tokens: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_session_reset_tp1_assignment"
            or self.protocol_sha256 != SESSION_RESET_RUNNER_PROTOCOL_SHA256
        ):
            raise ValueError("session reset assignment schema/protocol is unsupported")
        for label, digest in (
            ("registry", self.registry_sha256),
            ("runtime", self.runtime_sha256),
            ("topology", self.topology_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
            ("run nonce", self.run_nonce_sha256),
            ("nvidia-smi", self.nvidia_smi_raw_sha256),
        ):
            _require_sha256(f"session reset {label}", digest)
        if not self.gpu_uuid.startswith("GPU-"):
            raise ValueError("session reset assignment requires an inventory GPU UUID")
        _require_text("session reset GPU model", self.gpu_model)
        if self.method not in {"tts", "l0"}:
            raise ValueError("session reset assignment must exercise TTS or L0")
        if (
            type(self.input_token_ids) is not tuple
            or not self.input_token_ids
            or any(
                type(token) is not int or token < 0 for token in self.input_token_ids
            )
        ):
            raise ValueError("session reset prompt token IDs are invalid")
        if type(self.max_new_tokens) is not int or not 2 <= self.max_new_tokens <= 128:
            raise ValueError("session reset output length must be in [2, 128]")
        if type(self.launch_manifest) is not CanonicalJsonProofBinding:
            raise TypeError("session reset requires a path-bound launch manifest")
        launch = CompileLaunchManifest.load(self.launch_manifest.path)
        if (
            launch.sha256 != self.launch_manifest.semantic_sha256
            or launch.patched_sglang_commit != PINNED_SGLANG_COMMIT
            or launch.patched_sglang_tree != PINNED_SGLANG_TREE
            or launch.inventory_sha256 != self.inventory_sha256
            or launch.gpu_uuids != (self.gpu_uuid,)
        ):
            raise ValueError("session reset launch manifest identity differs")
        run_config = load_run_config(launch.run_config_path)
        if (
            run_config.method != self.method
            or run_config.model.algorithm != "DFLASH"
            or run_config.runtime.tensor_parallel_size != 1
            or run_config.runtime.data_parallel_size != 1
            or not run_config.runtime.speculation_enabled
        ):
            raise ValueError("session reset requires an adapted DFlash TP1/DP1 run")
        argv = launch.server_argv
        if (
            "--speculative-speed-study-metrics" not in argv
            or _argument_value(argv, "--speculative-algorithm") != "DFLASH"
            or _argument_value(argv, "--speculative-adaptation-config") is None
        ):
            raise ValueError("session reset launch lacks source-owned adaptation flags")
        evidence = _absolute_path(
            "session reset evidence directory", self.evidence_directory
        )
        executable = _absolute_path(
            "session reset nvidia-smi", self.nvidia_smi_executable
        )
        if not evidence.is_dir() or evidence.is_symlink():
            raise ValueError("session reset evidence directory is unavailable")
        actual, size = _raw_file_sha256(executable, label="session reset nvidia-smi")
        if size < 1 or actual != self.nvidia_smi_raw_sha256:
            raise ValueError("session reset nvidia-smi executable changed")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "launch_manifest": self.launch_manifest.to_dict(),
            "input_token_ids": list(self.input_token_ids),
        }

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @cached_property
    def source_identity_sha256(self) -> str:
        return _sha(
            {
                "schema_version": 1,
                "kind": "session_reset_qualification_source_identity",
                "assignment_sha256": self.sha256,
                "registry_sha256": self.registry_sha256,
                "runtime_sha256": self.runtime_sha256,
                "launch_manifest_sha256": self.launch_manifest.semantic_sha256,
            }
        )

    @cached_property
    def dispatch_lineage_sha256(self) -> str:
        return _sha(
            {
                "schema_version": 1,
                "kind": "session_reset_qualification_dispatch_lineage",
                "assignment_sha256": self.sha256,
                "source_identity_sha256": self.source_identity_sha256,
                "topology_sha256": self.topology_sha256,
                "inventory_sha256": self.inventory_sha256,
                "hardware_envelope_sha256": self.hardware_envelope_sha256,
            }
        )

    def write(self, path: str | Path) -> CanonicalJsonProofBinding:
        destination = _absolute_path("session reset assignment", str(path))
        publish_canonical_json_no_replace(str(destination), self.to_dict())
        return CanonicalJsonProofBinding.bind(
            str(destination), semantic_sha256=self.sha256
        )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = {field.name for field in fields(cls)}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("session reset assignment fields differ")
        row = dict(value)
        row["launch_manifest"] = CanonicalJsonProofBinding.from_dict(
            row["launch_manifest"]
        )
        tokens = row["input_token_ids"]
        if type(tokens) is not list:
            raise TypeError("session reset prompt tokens must be an array")
        row["input_token_ids"] = tuple(tokens)
        return cls(**row)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        binding = CanonicalJsonProofBinding.bind(str(path))
        value = cls.from_dict(binding.reopen())
        if binding.semantic_sha256 != value.sha256:
            raise ValueError("session reset assignment semantic identity changed")
        return value


@dataclass(frozen=True)
class SessionResetDispatchAuthority:
    schema_version: Literal[1]
    kind: Literal["formal_session_reset_dispatch_authority"]
    assignment_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    verified_ns: int
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_session_reset_dispatch_authority"
        ):
            raise ValueError("session reset dispatch authority schema is unsupported")
        for label, digest in (
            ("assignment", self.assignment_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
        ):
            _require_sha256(f"session reset dispatch {label}", digest)
        if type(self.verified_ns) is not int or self.verified_ns < 0:
            raise ValueError("session reset dispatch verification time is invalid")
        if type(self.control_attestation) is not ControlArtifactAttestation:
            raise TypeError("session reset dispatch requires an exact control envelope")
        if type(self.replay_reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("session reset dispatch requires a replay reservation")

    def revalidate(
        self, *, assignment: SessionResetQualificationAssignment
    ) -> VerifiedControlArtifact:
        if (
            assignment.sha256 != self.assignment_sha256
            or assignment.inventory_sha256 != self.inventory_sha256
            or assignment.hardware_envelope_sha256 != self.hardware_envelope_sha256
        ):
            raise ValueError("session reset dispatch belongs to another assignment")
        subject = self.control_attestation.subject
        if (
            subject.artifact_type != "non_serving_terminal"
            or subject.artifact_sha256 != assignment.sha256
            or subject.protocol_sha256 != SESSION_RESET_RUNNER_PROTOCOL_SHA256
            or subject.registry_sha256 != assignment.registry_sha256
            or subject.lineage_sha256 != assignment.dispatch_lineage_sha256
            or self.control_attestation.hardware_envelope_sha256
            != assignment.hardware_envelope_sha256
        ):
            raise ValueError("session reset dispatch control subject is not exact")
        verified = verify_release_control_artifact_attestation(
            self.control_attestation,
            expected_inventory_sha256=self.inventory_sha256,
            now_ns=self.verified_ns,
            consumed_challenge_sha256s=(),
        )
        challenges = self.replay_reservation.revalidate()
        expected = tuple(
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
            challenges != expected
            or self.replay_reservation.reservation_sha256 != reservation_sha256
            or self.replay_reservation.reserved_ns != self.verified_ns
        ):
            raise ValueError("session reset dispatch reservation differs")
        return verified

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
            raise ValueError("session reset dispatch authority fields differ")
        row = dict(value)
        row["control_attestation"] = ControlArtifactAttestation.from_dict(
            row["control_attestation"]
        )
        row["replay_reservation"] = ChallengeReplayReservationBinding.from_dict(
            row["replay_reservation"]
        )
        return cls(**row)


def authorize_session_reset_dispatch(
    *,
    assignment: SessionResetQualificationAssignment,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
) -> SessionResetDispatchAuthority:
    """Verify and reserve dispatch before either server process is launched."""

    subject = control_attestation.subject
    if (
        subject.artifact_type != "non_serving_terminal"
        or subject.artifact_sha256 != assignment.sha256
        or subject.protocol_sha256 != SESSION_RESET_RUNNER_PROTOCOL_SHA256
        or subject.registry_sha256 != assignment.registry_sha256
        or subject.lineage_sha256 != assignment.dispatch_lineage_sha256
        or control_attestation.hardware_envelope_sha256
        != assignment.hardware_envelope_sha256
    ):
        raise ValueError("session reset dispatch control subject is not exact")
    verified = verify_and_reserve_release_control_artifact_attestations(
        (control_attestation,),
        expected_inventory_sha256=assignment.inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
    )[0]
    reservation_sha256 = control_challenge_reservation_sha256(
        (verified,), reserved_ns=now_ns
    )
    authority = SessionResetDispatchAuthority(
        schema_version=1,
        kind="formal_session_reset_dispatch_authority",
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
class SessionResetRawRankTerminal:
    schema_version: Literal[1]
    kind: Literal["formal_session_reset_raw_rank_terminal"]
    runner_protocol_sha256: str
    assignment_sha256: str
    global_rank: Literal[0]
    gpu_uuid: str
    status: Literal["PASSED"]
    started_ns: int
    finished_ns: int
    process_id: int
    completed_test_names: tuple[str, ...]
    native_terminal_sha256: str
    reset_receipt_sha256: str
    close_receipt_sha256: str
    cold_output_ids_sha256: str
    reused_output_ids_sha256: str
    observation_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_session_reset_raw_rank_terminal"
            or self.runner_protocol_sha256 != SESSION_RESET_RUNNER_PROTOCOL_SHA256
            or self.global_rank != 0
            or self.status != "PASSED"
            or self.completed_test_names != SESSION_RESET_GPU_TEST_NAMES
        ):
            raise ValueError("session reset raw rank terminal is incomplete")
        for label, digest in (
            ("assignment", self.assignment_sha256),
            ("native terminal", self.native_terminal_sha256),
            ("reset receipt", self.reset_receipt_sha256),
            ("close receipt", self.close_receipt_sha256),
            ("cold trajectory", self.cold_output_ids_sha256),
            ("reused trajectory", self.reused_output_ids_sha256),
            ("observation", self.observation_sha256),
        ):
            _require_sha256(f"session reset rank {label}", digest)
        if (
            not self.gpu_uuid.startswith("GPU-")
            or type(self.started_ns) is not int
            or type(self.finished_ns) is not int
            or self.started_ns < 1
            or self.finished_ns < self.started_ns
            or type(self.process_id) is not int
            or self.process_id < 1
            or self.cold_output_ids_sha256 != self.reused_output_ids_sha256
        ):
            raise ValueError("session reset raw rank observation is invalid")

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "completed_test_names": list(self.completed_test_names)}

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = {field.name for field in fields(cls)}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("session reset raw rank terminal fields differ")
        row = dict(value)
        names = row["completed_test_names"]
        if type(names) is not list:
            raise TypeError("session reset completed tests must be an array")
        row["completed_test_names"] = tuple(names)
        return cls(**row)


@dataclass(frozen=True)
class SessionResetQualificationResultPointer:
    schema_version: Literal[1]
    kind: Literal["formal_session_reset_qualification_result_pointer"]
    assignment: CanonicalJsonProofBinding
    dispatch_authority: CanonicalJsonProofBinding
    before_gpu_snapshot: CanonicalJsonProofBinding
    after_gpu_snapshot: CanonicalJsonProofBinding
    cold_reference: CanonicalJsonProofBinding
    junit_xml: EvidenceFileBinding
    log: EvidenceFileBinding
    rank_terminal: CanonicalJsonProofBinding
    unsigned_gpu_proof: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_session_reset_qualification_result_pointer"
        ):
            raise ValueError("session reset result pointer schema is unsupported")
        for binding in (
            self.assignment,
            self.dispatch_authority,
            self.before_gpu_snapshot,
            self.after_gpu_snapshot,
            self.cold_reference,
            self.rank_terminal,
            self.unsigned_gpu_proof,
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("session reset pointer requires canonical bindings")
            binding.__post_init__()
        for binding in (self.junit_xml, self.log):
            if type(binding) is not EvidenceFileBinding:
                raise TypeError("session reset pointer requires exact file bindings")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "assignment": self.assignment.to_dict(),
            "dispatch_authority": self.dispatch_authority.to_dict(),
            "before_gpu_snapshot": self.before_gpu_snapshot.to_dict(),
            "after_gpu_snapshot": self.after_gpu_snapshot.to_dict(),
            "cold_reference": self.cold_reference.to_dict(),
            "junit_xml": self.junit_xml.to_dict(),
            "log": self.log.to_dict(),
            "rank_terminal": self.rank_terminal.to_dict(),
            "unsigned_gpu_proof": self.unsigned_gpu_proof.to_dict(),
        }

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = {field.name for field in fields(cls)}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("session reset result pointer fields differ")
        row = dict(value)
        for name in (
            "assignment",
            "dispatch_authority",
            "before_gpu_snapshot",
            "after_gpu_snapshot",
            "cold_reference",
            "rank_terminal",
            "unsigned_gpu_proof",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        row["junit_xml"] = EvidenceFileBinding.from_dict(
            row["junit_xml"], label="session reset JUnit"
        )
        row["log"] = EvidenceFileBinding.from_dict(
            row["log"], label="session reset log"
        )
        return cls(**row)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        binding = CanonicalJsonProofBinding.bind(str(path))
        pointer = cls.from_dict(binding.reopen())
        pointer.validate()
        return pointer

    def validate(self) -> None:
        assignment = SessionResetQualificationAssignment.from_dict(
            self.assignment.reopen()
        )
        if assignment.sha256 != self.assignment.semantic_sha256:
            raise ValueError("session reset result assignment changed")
        dispatch = SessionResetDispatchAuthority.from_dict(
            self.dispatch_authority.reopen()
        )
        dispatch.revalidate(assignment=assignment)
        before = _validate_gpu_snapshot(
            self.before_gpu_snapshot.reopen(), assignment=assignment
        )
        after = _validate_gpu_snapshot(
            self.after_gpu_snapshot.reopen(), assignment=assignment
        )
        if (
            before["status"] != "AVAILABLE"
            or after["status"] != "AVAILABLE"
            or before["compute_process_rows"]
            or after["compute_process_rows"]
            or before["gpu"]["uuid"] != after["gpu"]["uuid"]
            or before["gpu"]["name"] != after["gpu"]["name"]
        ):
            raise ValueError("session reset qualification lacks clean GPU boundaries")
        cold = self.cold_reference.reopen()
        if (
            set(cold)
            != {
                "schema_version",
                "kind",
                "assignment_sha256",
                "server_argv_sha256",
                "input_token_ids",
                "max_new_tokens",
                "output_token_ids",
                "server_startup_duration_ns",
            }
            or cold["schema_version"] != 1
            or cold["kind"] != "formal_session_reset_cold_reference"
            or cold["assignment_sha256"] != assignment.sha256
            or cold["input_token_ids"] != list(assignment.input_token_ids)
            or cold["max_new_tokens"] != assignment.max_new_tokens
        ):
            raise ValueError("session reset cold reference differs")
        rank = SessionResetRawRankTerminal.from_dict(self.rank_terminal.reopen())
        if (
            rank.assignment_sha256 != assignment.sha256
            or rank.gpu_uuid != assignment.gpu_uuid
        ):
            raise ValueError("session reset rank terminal differs from assignment")
        names, collected, passed, failed, errored, skipped = _junit_summary(
            Path(self.junit_xml.absolute_path)
        )
        self.junit_xml.reopen(label="session reset JUnit")
        self.log.reopen(label="session reset log")
        if names != tuple(sorted(SESSION_RESET_GPU_TEST_NAMES)) or (
            collected,
            passed,
            failed,
            errored,
            skipped,
        ) != (8, 8, 0, 0, 0):
            raise ValueError("session reset JUnit is not exact 8/8 zero-skip")
        receipt = NativeRuntimeGpuProofReceipt.from_dict(
            self.unsigned_gpu_proof.reopen()
        )
        if (
            receipt.sha256 != self.unsigned_gpu_proof.semantic_sha256
            or receipt.suite_id != "session_reset_tp1"
            or receipt.topology_sha256 != assignment.topology_sha256
            or receipt.source_identity_sha256 != assignment.source_identity_sha256
            or receipt.inventory_sha256 != assignment.inventory_sha256
            or receipt.gpu_uuids != (assignment.gpu_uuid,)
            or receipt.hardware_envelope_sha256 != assignment.hardware_envelope_sha256
            or receipt.run_nonce_sha256 != assignment.run_nonce_sha256
            or receipt.junit_xml_sha256 != self.junit_xml.raw_sha256
        ):
            raise ValueError("session reset unsigned proof differs from result")


@dataclass(frozen=True)
class SessionResetQualificationFailurePointer:
    """Durable non-authorizing evidence for any precondition or runner failure."""

    schema_version: Literal[1]
    kind: Literal["formal_session_reset_qualification_failure_pointer"]
    assignment: CanonicalJsonProofBinding
    dispatch_authority: CanonicalJsonProofBinding
    before_gpu_snapshot: CanonicalJsonProofBinding
    after_gpu_snapshot: CanonicalJsonProofBinding
    log: EvidenceFileBinding
    fatal_terminal: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_session_reset_qualification_failure_pointer"
        ):
            raise ValueError("session reset failure pointer schema is unsupported")

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
            raise ValueError("session reset failure pointer fields differ")
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
            row["log"], label="session reset failure log"
        )
        return cls(**row)

    def validate(self) -> None:
        assignment = SessionResetQualificationAssignment.from_dict(
            self.assignment.reopen()
        )
        if assignment.sha256 != self.assignment.semantic_sha256:
            raise ValueError("session reset failure assignment changed")
        SessionResetDispatchAuthority.from_dict(
            self.dispatch_authority.reopen()
        ).revalidate(assignment=assignment)
        _validate_gpu_snapshot(self.before_gpu_snapshot.reopen(), assignment=assignment)
        _validate_gpu_snapshot(self.after_gpu_snapshot.reopen(), assignment=assignment)
        fatal = self.fatal_terminal.reopen()
        if (
            set(fatal)
            != {
                "schema_version",
                "kind",
                "assignment_sha256",
                "status",
                "error_code",
                "error_type",
                "started_ns",
                "finished_ns",
            }
            or fatal["schema_version"] != 1
            or fatal["kind"] != "formal_session_reset_qualification_fatal_terminal"
            or fatal["assignment_sha256"] != assignment.sha256
            or fatal["status"] != "ERROR"
        ):
            raise ValueError("session reset fatal terminal differs")
        self.log.reopen(label="session reset failure log")


@dataclass(frozen=True)
class SessionResetQualificationProofPointer:
    schema_version: Literal[1]
    kind: Literal["formal_session_reset_qualification_proof_pointer"]
    result_pointer: CanonicalJsonProofBinding
    gpu_proof_artifact: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_session_reset_qualification_proof_pointer"
        ):
            raise ValueError("session reset proof pointer schema is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "result_pointer": self.result_pointer.to_dict(),
            "gpu_proof_artifact": self.gpu_proof_artifact.to_dict(),
        }

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = {field.name for field in fields(cls)}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("session reset proof pointer fields differ")
        row = dict(value)
        row["result_pointer"] = CanonicalJsonProofBinding.from_dict(
            row["result_pointer"]
        )
        row["gpu_proof_artifact"] = CanonicalJsonProofBinding.from_dict(
            row["gpu_proof_artifact"]
        )
        return cls(**row)

    def revalidate(self, *, now_ns: int) -> VerifiedNativeRuntimeGpuProof:
        result = SessionResetQualificationResultPointer.from_dict(
            self.result_pointer.reopen()
        )
        result.validate()
        artifact = NativeRuntimeGpuProofArtifact.from_dict(
            self.gpu_proof_artifact.reopen()
        )
        verified = artifact.revalidate(now_ns=now_ns)
        if verified.receipt_raw_sha256 != result.unsigned_gpu_proof.raw_sha256:
            raise ValueError("session reset proof artifact belongs to another result")
        return verified


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


def _gpu_snapshot(
    assignment: SessionResetQualificationAssignment,
) -> dict[str, object]:
    """Query the exact assigned UUID and all compute processes source-side."""

    base = {
        "schema_version": 1,
        "kind": "formal_session_reset_gpu_snapshot",
        "assignment_sha256": assignment.sha256,
        "inventory_sha256": assignment.inventory_sha256,
        "captured_ns": time.monotonic_ns(),
    }
    environment = {
        "PATH": str(Path(assignment.nvidia_smi_executable).parent),
        "LANG": "C",
        "LC_ALL": "C",
    }
    try:
        gpu = subprocess.run(
            (
                assignment.nvidia_smi_executable,
                "--query-gpu=uuid,name,memory.used",
                "--format=csv,noheader,nounits",
            ),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        gpu_rows = []
        for line in gpu.stdout.splitlines():
            values = tuple(value.strip() for value in line.split(","))
            if len(values) != 3:
                raise ValueError("nvidia-smi GPU row is malformed")
            if values[0] == assignment.gpu_uuid:
                gpu_rows.append(
                    {
                        "uuid": values[0],
                        "name": _require_text("session reset GPU name", values[1]),
                        "memory_used_mib": int(values[2]),
                    }
                )
        if (
            len(gpu_rows) != 1
            or gpu_rows[0]["name"] != assignment.gpu_model
            or gpu_rows[0]["memory_used_mib"] < 0
        ):
            raise ValueError("assigned GPU UUID is absent or duplicated")
        processes = subprocess.run(
            (
                assignment.nvidia_smi_executable,
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        process_rows = []
        for line in processes.stdout.splitlines():
            values = tuple(value.strip() for value in line.split(","))
            if not values or values == ("",):
                continue
            if len(values) != 3:
                raise ValueError("nvidia-smi compute-process row is malformed")
            if values[0] != assignment.gpu_uuid:
                continue
            process_rows.append(
                {
                    "gpu_uuid": values[0],
                    "pid": int(values[1]),
                    "used_gpu_memory_mib": int(values[2]),
                }
            )
        process_rows.sort(key=lambda row: (row["pid"], row["used_gpu_memory_mib"]))
        return {
            **base,
            "status": "AVAILABLE",
            "gpu": gpu_rows[0],
            "compute_process_rows": process_rows,
            "error_code": None,
        }
    except (OSError, subprocess.SubprocessError, TypeError, ValueError):
        return {
            **base,
            "status": "ERROR",
            "gpu": None,
            "compute_process_rows": [],
            "error_code": "nvidia_smi_snapshot_unavailable",
        }


def _validate_gpu_snapshot(
    value: object, *, assignment: SessionResetQualificationAssignment
) -> dict[str, object]:
    expected = {
        "schema_version",
        "kind",
        "assignment_sha256",
        "inventory_sha256",
        "captured_ns",
        "status",
        "gpu",
        "compute_process_rows",
        "error_code",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("session reset GPU snapshot fields differ")
    row = dict(value)
    if (
        row["schema_version"] != 1
        or row["kind"] != "formal_session_reset_gpu_snapshot"
        or row["assignment_sha256"] != assignment.sha256
        or row["inventory_sha256"] != assignment.inventory_sha256
        or type(row["captured_ns"]) is not int
        or row["captured_ns"] < 1
        or row["status"] not in {"AVAILABLE", "ERROR"}
        or type(row["compute_process_rows"]) is not list
    ):
        raise ValueError("session reset GPU snapshot identity is invalid")
    if row["status"] == "ERROR":
        if row["gpu"] is not None or row["error_code"] != (
            "nvidia_smi_snapshot_unavailable"
        ):
            raise ValueError("session reset failed snapshot is malformed")
        return row
    if row["error_code"] is not None or type(row["gpu"]) is not dict:
        raise ValueError("session reset available snapshot is malformed")
    gpu = row["gpu"]
    if (
        set(gpu) != {"uuid", "name", "memory_used_mib"}
        or gpu["uuid"] != assignment.gpu_uuid
        or gpu["name"] != assignment.gpu_model
        or type(gpu["memory_used_mib"]) is not int
        or gpu["memory_used_mib"] < 0
    ):
        raise ValueError("session reset assigned GPU row is invalid")
    observed_pids: set[int] = set()
    for process in row["compute_process_rows"]:
        if (
            type(process) is not dict
            or set(process) != {"gpu_uuid", "pid", "used_gpu_memory_mib"}
            or process["gpu_uuid"] != assignment.gpu_uuid
            or type(process["pid"]) is not int
            or process["pid"] < 1
            or process["pid"] in observed_pids
            or type(process["used_gpu_memory_mib"]) is not int
            or process["used_gpu_memory_mib"] < 0
        ):
            raise ValueError("session reset compute-process row is invalid")
        observed_pids.add(process["pid"])
    return row


def _post_generate(
    *, base_url: str, input_token_ids: tuple[int, ...], max_new_tokens: int
) -> tuple[int, ...]:
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=120)
    payload = {
        "input_ids": list(input_token_ids),
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
        "rid": "session-cold-reference",
        "stream": False,
    }
    connection.request(
        "POST",
        "/generate",
        body=_canonical_bytes(payload),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    body = response.read()
    connection.close()
    if response.status != 200:
        raise RuntimeError("session reset cold generate failed")
    value = json.loads(body)
    output = value.get("output_ids") if isinstance(value, dict) else None
    if not isinstance(output, list) or not output:
        raise RuntimeError("session reset cold generate lacks output token IDs")
    return tuple(int(token) for token in output)


def _wait_ready(
    process: subprocess.Popen[bytes], *, base_url: str, timeout_s: float
) -> int:
    started = time.monotonic_ns()
    deadline = time.monotonic() + timeout_s
    parsed = urlsplit(base_url)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("session reset server exited before readiness")
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
        try:
            connection.request("GET", "/health_generate")
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                return time.monotonic_ns() - started
        except OSError:
            pass
        finally:
            connection.close()
        time.sleep(0.25)
    raise TimeoutError("session reset server readiness timed out")


def _stop_process_group(
    process: subprocess.Popen[bytes], *, timeout_s: float = 30
) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=timeout_s)


def _launch_server(
    launch: CompileLaunchManifest, *, log_file
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        launch.server_argv,
        cwd=launch.patched_sglang_checkout,
        env=launch.child_environment(),
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


_SOURCE_OWNED_SESSION_RESET_TIMEOUT_SECONDS = 1_200


def execute_session_reset_qualification(
    *,
    assignment_path: str | Path,
    dispatch_control: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
) -> SessionResetQualificationResultPointer | SessionResetQualificationFailurePointer:
    """Run both processes and publish an unsigned, still non-authorizing result."""

    timeout_seconds = _SOURCE_OWNED_SESSION_RESET_TIMEOUT_SECONDS
    assignment_source = _absolute_path("session reset assignment", str(assignment_path))
    assignment_binding = CanonicalJsonProofBinding.bind(str(assignment_source))
    assignment = SessionResetQualificationAssignment.from_dict(
        assignment_binding.reopen()
    )
    if assignment.sha256 != assignment_binding.semantic_sha256:
        raise ValueError("session reset assignment identity changed")
    authority = authorize_session_reset_dispatch(
        assignment=assignment,
        control_attestation=dispatch_control,
        replay_store=replay_store,
        now_ns=now_ns,
    )
    evidence = Path(assignment.evidence_directory)
    prefix = f"session-reset-{assignment.sha256}"
    paths = {
        name: evidence / f"{prefix}.{suffix}"
        for name, suffix in {
            "dispatch": "dispatch.json",
            "before": "gpu-before.json",
            "after": "gpu-after.json",
            "cold": "cold.json",
            "junit": "junit.xml",
            "log": "log",
            "rank": "rank0.json",
            "receipt": "unsigned-proof.json",
            "fatal": "fatal.json",
            "pointer": "pointer.json",
        }.items()
    }
    for path in paths.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"session reset evidence already exists: {path.name}")
    publish_canonical_json_no_replace(str(paths["dispatch"]), authority.to_dict())
    log_file = paths["log"].open("xb", buffering=0)
    log_file.write(b"session-reset qualification started\n")
    cold_process: subprocess.Popen[bytes] | None = None
    reused_process: subprocess.Popen[bytes] | None = None
    started_ns = time.monotonic_ns()
    execution_error: Exception | None = None
    error_code: str | None = None
    cold_binding: CanonicalJsonProofBinding | None = None
    try:
        before = _gpu_snapshot(assignment)
        publish_canonical_json_no_replace(str(paths["before"]), before)
        validated_before = _validate_gpu_snapshot(before, assignment=assignment)
        if (
            validated_before["status"] != "AVAILABLE"
            or validated_before["compute_process_rows"]
        ):
            error_code = "gpu_precondition_not_clean"
            raise RuntimeError("session reset GPU precondition is not clean")
        launch = CompileLaunchManifest.load(assignment.launch_manifest.path)
        base_url = f"http://127.0.0.1:{launch.localhost_port}"
        cold_process = _launch_server(launch, log_file=log_file)
        cold_startup_ns = _wait_ready(
            cold_process, base_url=base_url, timeout_s=min(timeout_seconds, 600)
        )
        cold_output = _post_generate(
            base_url=base_url,
            input_token_ids=assignment.input_token_ids,
            max_new_tokens=assignment.max_new_tokens,
        )
        _stop_process_group(cold_process)
        cold_process = None
        cold = {
            "schema_version": 1,
            "kind": "formal_session_reset_cold_reference",
            "assignment_sha256": assignment.sha256,
            "server_argv_sha256": launch.server_argv_sha256,
            "input_token_ids": list(assignment.input_token_ids),
            "max_new_tokens": assignment.max_new_tokens,
            "output_token_ids": list(cold_output),
            "server_startup_duration_ns": cold_startup_ns,
        }
        publish_canonical_json_no_replace(str(paths["cold"]), cold)
        cold_binding = CanonicalJsonProofBinding.bind(str(paths["cold"]))

        reused_process = _launch_server(launch, log_file=log_file)
        reused_startup_ns = _wait_ready(
            reused_process, base_url=base_url, timeout_s=min(timeout_seconds, 600)
        )
        python = launch.server_argv[0]
        if not Path(python).is_absolute():
            raise ValueError("session reset server argv must use an absolute Python")
        command = (
            python,
            "-m",
            "pytest",
            "-q",
            f"--junitxml={paths['junit']}",
            *(
                f"{SESSION_RESET_GPU_TEST_FILE}::{name}"
                for name in SESSION_RESET_GPU_TEST_NAMES
            ),
        )
        environment = {
            **launch.child_environment(),
            "LIGHTCONE_SESSION_BASE_URL": base_url,
            "LIGHTCONE_SESSION_ASSIGNMENT_SHA256": assignment.sha256,
            "LIGHTCONE_SESSION_RUNNER_PROTOCOL_SHA256": (
                SESSION_RESET_RUNNER_PROTOCOL_SHA256
            ),
            "LIGHTCONE_SESSION_GPU_UUID": assignment.gpu_uuid,
            "LIGHTCONE_SESSION_METHOD": assignment.method,
            "LIGHTCONE_SESSION_COLD_REFERENCE_PATH": str(paths["cold"]),
            "LIGHTCONE_SESSION_COLD_REFERENCE_RAW_SHA256": (cold_binding.raw_sha256),
            "LIGHTCONE_SESSION_REUSED_STARTUP_DURATION_NS": str(reused_startup_ns),
            "LIGHTCONE_SESSION_RANK0_TERMINAL_PATH": str(paths["rank"]),
        }
        completed = subprocess.run(
            command,
            cwd=launch.patched_sglang_checkout,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        _stop_process_group(reused_process)
        reused_process = None
        log_file.flush()
        os.fsync(log_file.fileno())
        if completed.returncode != 0:
            raise RuntimeError("session reset qualification pytest failed")
    except Exception as error:  # noqa: BLE001 - retain timeout/failure evidence
        execution_error = error
        if error_code is None:
            error_code = "qualification_execution_failed"
    finally:
        cleanup_failed = False
        for process in (cold_process, reused_process):
            if process is not None:
                try:
                    _stop_process_group(process)
                except BaseException:  # noqa: BLE001 - preserve cleanup evidence
                    cleanup_failed = True
        if cleanup_failed and execution_error is None:
            execution_error = RuntimeError("session reset process cleanup failed")
            error_code = "process_group_cleanup_failed"
        after = _gpu_snapshot(assignment)
        try:
            publish_canonical_json_no_replace(str(paths["after"]), after)
        except Exception as error:  # noqa: BLE001 - retain post-state evidence
            if execution_error is None:
                execution_error = error
                error_code = "after_snapshot_publication_failed"
        else:
            validated_after = _validate_gpu_snapshot(after, assignment=assignment)
            if (
                validated_after["status"] != "AVAILABLE"
                or validated_after["compute_process_rows"]
            ) and execution_error is None:
                execution_error = RuntimeError(
                    "session reset GPU postcondition is not clean"
                )
                error_code = "gpu_postcondition_not_clean"
        log_file.flush()
        os.fsync(log_file.fileno())
        log_file.close()

    before_binding = CanonicalJsonProofBinding.bind(str(paths["before"]))
    after_binding = CanonicalJsonProofBinding.bind(str(paths["after"]))
    dispatch_binding = CanonicalJsonProofBinding.bind(str(paths["dispatch"]))
    log_binding = EvidenceFileBinding.bind(paths["log"], label="session reset log")
    if execution_error is None:
        try:
            names, collected, passed, failed, errored, skipped = _junit_summary(
                paths["junit"]
            )
            if names != tuple(sorted(SESSION_RESET_GPU_TEST_NAMES)) or (
                collected,
                passed,
                failed,
                errored,
                skipped,
            ) != (8, 8, 0, 0, 0):
                raise RuntimeError("session reset qualification JUnit is not exact 8/8")
            rank_binding = CanonicalJsonProofBinding.bind(str(paths["rank"]))
            rank = SessionResetRawRankTerminal.from_dict(rank_binding.reopen())
            if (
                rank.assignment_sha256 != assignment.sha256
                or rank.gpu_uuid != assignment.gpu_uuid
            ):
                raise ValueError("session reset raw terminal differs from assignment")
            junit_binding = EvidenceFileBinding.bind(
                paths["junit"], label="session reset JUnit"
            )
            receipt = NativeRuntimeGpuProofReceipt(
                schema_version=1,
                kind="lightcone_native_runtime_gpu_proof",
                suite_id="session_reset_tp1",
                topology_mode="tp1_dp1",
                topology_sha256=assignment.topology_sha256,
                runner_protocol_sha256=SESSION_RESET_RUNNER_PROTOCOL_SHA256,
                assignment_sha256=assignment.sha256,
                qualification_observation_sha256=rank.sha256,
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
                gpu_uuids=(assignment.gpu_uuid,),
                hardware_envelope_sha256=assignment.hardware_envelope_sha256,
                junit_xml_sha256=junit_binding.raw_sha256,
                test_names=NATIVE_RUNTIME_QUALIFICATION_TESTS["session_reset_tp1"],
                tests_collected=8,
                tests_passed=8,
                tests_failed=0,
                tests_errored=0,
                tests_skipped=0,
            )
            receipt_binding = receipt.write_unsigned(str(paths["receipt"]))
            if cold_binding is None:
                raise RuntimeError("session reset cold reference binding is absent")
            pointer = SessionResetQualificationResultPointer(
                schema_version=1,
                kind="formal_session_reset_qualification_result_pointer",
                assignment=assignment_binding,
                dispatch_authority=dispatch_binding,
                before_gpu_snapshot=before_binding,
                after_gpu_snapshot=after_binding,
                cold_reference=cold_binding,
                junit_xml=junit_binding,
                log=log_binding,
                rank_terminal=rank_binding,
                unsigned_gpu_proof=receipt_binding,
            )
            pointer.validate()
            publish_canonical_json_no_replace(str(paths["pointer"]), pointer.to_dict())
            return SessionResetQualificationResultPointer.load(paths["pointer"])
        except Exception as error:  # noqa: BLE001 - retain invalid evidence
            execution_error = error
            error_code = "qualification_evidence_validation_failed"

    fatal = {
        "schema_version": 1,
        "kind": "formal_session_reset_qualification_fatal_terminal",
        "assignment_sha256": assignment.sha256,
        "status": "ERROR",
        "error_code": error_code or "qualification_unknown_error",
        "error_type": type(execution_error).__name__,
        "started_ns": started_ns,
        "finished_ns": time.monotonic_ns(),
    }
    publish_canonical_json_no_replace(str(paths["fatal"]), fatal)
    failure = SessionResetQualificationFailurePointer(
        schema_version=1,
        kind="formal_session_reset_qualification_failure_pointer",
        assignment=assignment_binding,
        dispatch_authority=dispatch_binding,
        before_gpu_snapshot=before_binding,
        after_gpu_snapshot=after_binding,
        log=log_binding,
        fatal_terminal=CanonicalJsonProofBinding.bind(str(paths["fatal"])),
    )
    failure.validate()
    publish_canonical_json_no_replace(str(paths["pointer"]), failure.to_dict())
    return SessionResetQualificationFailurePointer.from_dict(
        CanonicalJsonProofBinding.bind(str(paths["pointer"])).reopen()
    )


def finalize_session_reset_qualification(
    *,
    result_pointer_path: str | Path,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
    proof_artifact_path: str | Path,
    output_pointer_path: str | Path,
) -> SessionResetQualificationProofPointer:
    """Locally trust-lift one pulled unsigned result without another GPU run."""

    result_binding = CanonicalJsonProofBinding.bind(str(result_pointer_path))
    result = SessionResetQualificationResultPointer.from_dict(result_binding.reopen())
    result.validate()
    assignment = SessionResetQualificationAssignment.from_dict(
        result.assignment.reopen()
    )
    verified = verify_native_runtime_gpu_proof(
        result.unsigned_gpu_proof.path,
        control_attestation=control_attestation,
        replay_store=replay_store,
        expected_suite_id="session_reset_tp1",
        expected_topology_sha256=assignment.topology_sha256,
        expected_source_identity_sha256=assignment.source_identity_sha256,
        expected_inventory_sha256=assignment.inventory_sha256,
        expected_gpu_uuids=(assignment.gpu_uuid,),
        expected_hardware_envelope_sha256=assignment.hardware_envelope_sha256,
        expected_run_nonce_sha256=assignment.run_nonce_sha256,
        now_ns=now_ns,
    )
    artifact = build_native_runtime_gpu_proof_artifact(
        receipt_path=result.unsigned_gpu_proof.path,
        control_attestation=control_attestation,
        replay_store=replay_store,
        verified_proof=verified,
    )
    artifact_path = _absolute_path(
        "session reset proof artifact", str(proof_artifact_path)
    )
    publish_canonical_json_no_replace(str(artifact_path), artifact.to_dict())
    artifact_binding = CanonicalJsonProofBinding.bind(
        str(artifact_path), semantic_sha256=artifact.sha256
    )
    pointer = SessionResetQualificationProofPointer(
        schema_version=1,
        kind="formal_session_reset_qualification_proof_pointer",
        result_pointer=result_binding,
        gpu_proof_artifact=artifact_binding,
    )
    pointer.revalidate(now_ns=now_ns)
    output = _absolute_path(
        "session reset qualification proof pointer", str(output_pointer_path)
    )
    publish_canonical_json_no_replace(str(output), pointer.to_dict())
    return SessionResetQualificationProofPointer.from_dict(
        CanonicalJsonProofBinding.bind(str(output)).reopen()
    )


__all__ = [
    "SESSION_RESET_GPU_TEST_FILE",
    "SESSION_RESET_GPU_TEST_NAMES",
    "SESSION_RESET_RUNNER_PROTOCOL_SHA256",
    "SessionResetDispatchAuthority",
    "SessionResetQualificationAssignment",
    "SessionResetQualificationFailurePointer",
    "SessionResetQualificationProofPointer",
    "SessionResetQualificationResultPointer",
    "SessionResetRawRankTerminal",
    "authorize_session_reset_dispatch",
    "execute_session_reset_qualification",
    "finalize_session_reset_qualification",
]
