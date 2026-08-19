"""Trusted single-operator authority for empirical TP1 session reset.

This authority is deliberately narrower than the release-signing GPU proof.
It permits one exact trusted operator run to amortize compatible TP1 server
starts after an eight-test GPU qualification, while keeping every scientific
claim ``formal_measured=False``.  It cannot be converted into, or stand in for,
``VerifiedNativeRuntimeGpuProof``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry import protocol_lock_from_dict
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedSingleOperatorContentBundleBinding,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorJsonBinding,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
    relocated_evidence_path,
)

TrustedSessionResetBackend = Literal["DFLASH", "DSPARK", "EAGLE3"]
TrustedSessionResetMethodFamily = Literal[
    "target_only",
    "static",
    "tts",
    "l0",
    "lightcone",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
]

TRUSTED_EMPIRICAL_TP1_SESSION_RESET_SUITE = "session_reset_tp1_empirical_v1"
TRUSTED_EMPIRICAL_TP1_SESSION_RESET_TESTS = (
    "same_server_process_identity",
    "native_session_epoch_lineage",
    "exact_output_token_trajectory",
    "request_queue_empty_after_trace",
    "optimizer_candidate_and_adaptation_state_reset",
    "registered_cache_policy_restored",
    "terminal_writer_fully_flushed",
    "hbm_returns_without_monotonic_growth",
)
TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "trusted_empirical_tp1_session_reset_authority",
        "scope": ("one_protocol_lock_source_tree_inventory_gpu_backend_method_family"),
        "topology": "tp1_dp1_only",
        "qualification": TRUSTED_EMPIRICAL_TP1_SESSION_RESET_TESTS,
        "physical_trace_count": 2,
        "pass_gate": "exactly_8_collected_8_passed_zero_fail_error_skip",
        "evidence": ("junit_raw_terminal_native_lifecycle_reset_state_and_hbm_sha256"),
        "claim": "trusted_single_operator_empirical_no_signature",
        "formal_measured": False,
        "fallback": "fresh_process_per_cell",
    }
)

_BACKENDS = {"DFLASH", "DSPARK", "EAGLE3"}
_METHOD_FAMILIES = {
    "target_only",
    "static",
    "tts",
    "l0",
    "lightcone",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
}

TRUSTED_EMPIRICAL_TP1_SESSION_RESET_QUALIFICATION_SPEC_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "trusted_empirical_tp1_session_reset_qualification_spec",
        "input": (
            "paths_only_protocol_lock_content_inventory_junit_terminal_"
            "native_lifecycle_reset_state_and_hbm"
        ),
        "scientific_scope": "gpu_backend_method_family_and_run_id_in_spec",
        "publisher": "derive_every_digest_by_deep_reopen",
    }
)


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_git_object_id(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase Git object id")
    return value


def _require_text(label: str, value: object) -> str:
    if type(value) is not str or not value or "\n" in value or "\x00" in value:
        raise ValueError(f"{label} must be non-empty single-line text")
    return value


def _absolute_path(label: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a path string")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError(f"{label} must be absolute and normalized")
    return value


@dataclass(frozen=True)
class TrustedEmpiricalTp1SessionResetQualificationSpec:
    """Path-only input emitted by the TP1 qualification producer."""

    schema_version: Literal[1]
    kind: Literal["trusted_empirical_tp1_session_reset_qualification_spec"]
    protocol_sha256: str
    topology_mode: Literal["tp1_dp1"]
    suite_id: Literal["session_reset_tp1_empirical_v1"]
    gpu_uuid: str
    backend: TrustedSessionResetBackend
    method_family: TrustedSessionResetMethodFamily
    qualification_run_id: str
    protocol_lock_path: str
    content_bundle_path: str
    inventory_path: str
    junit_xml_path: str
    raw_terminal_path: str
    native_lifecycle_path: str
    reset_state_evidence_path: str
    hbm_evidence_path: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "trusted_empirical_tp1_session_reset_qualification_spec"
            or self.protocol_sha256
            != TRUSTED_EMPIRICAL_TP1_SESSION_RESET_QUALIFICATION_SPEC_PROTOCOL_SHA256
            or self.topology_mode != "tp1_dp1"
            or self.suite_id != TRUSTED_EMPIRICAL_TP1_SESSION_RESET_SUITE
            or self.backend not in _BACKENDS
            or self.method_family not in _METHOD_FAMILIES
        ):
            raise ValueError("trusted empirical reset qualification spec differs")
        _require_text("trusted empirical reset GPU UUID", self.gpu_uuid)
        if not self.gpu_uuid.startswith("GPU-"):
            raise ValueError("trusted empirical reset qualification lacks a GPU UUID")
        _require_sha256(
            "trusted empirical reset qualification run",
            self.qualification_run_id,
        )
        for label, value in (
            ("ProtocolLock", self.protocol_lock_path),
            ("content bundle", self.content_bundle_path),
            ("inventory", self.inventory_path),
            ("JUnit", self.junit_xml_path),
            ("raw terminal", self.raw_terminal_path),
            ("native lifecycle", self.native_lifecycle_path),
            ("reset-state evidence", self.reset_state_evidence_path),
            ("HBM evidence", self.hbm_evidence_path),
        ):
            _absolute_path(f"trusted empirical reset {label}", value)
        paths = (
            self.protocol_lock_path,
            self.content_bundle_path,
            self.inventory_path,
            self.junit_xml_path,
            self.raw_terminal_path,
            self.native_lifecycle_path,
            self.reset_state_evidence_path,
            self.hbm_evidence_path,
        )
        if len(set(paths)) != len(paths):
            raise ValueError("trusted empirical reset qualification aliases inputs")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("trusted empirical reset qualification fields differ")
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class TrustedEmpiricalTp1SessionResetAuthority:
    """One unsigned but content-bound operational reset qualification.

    Instances represent PASS only.  Failed or missing qualification evidence is
    retained by the preflight ledger but cannot be constructed as an authority;
    the session partitioner consequently emits fresh-process singleton plans.
    """

    schema_version: Literal[1]
    kind: Literal["trusted_empirical_tp1_session_reset_authority"]
    protocol_sha256: str
    evidence_level: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured: Literal[False]
    status: Literal["PASS"]
    topology_mode: Literal["tp1_dp1"]
    suite_id: Literal["session_reset_tp1_empirical_v1"]
    protocol_lock_sha256: str
    source_snapshot_sha256: str
    patched_sglang_tree: str
    inventory_sha256: str
    gpu_uuid: str
    backend: TrustedSessionResetBackend
    method_family: TrustedSessionResetMethodFamily
    qualification_run_id: str
    qualification_spec: CanonicalJsonProofBinding
    protocol_lock: FormalSingleOperatorJsonBinding
    content_bundle: TrustedSingleOperatorContentBundleBinding
    inventory: CanonicalJsonProofBinding
    test_names: tuple[str, ...]
    tests_collected: int
    tests_passed: int
    tests_failed: int
    tests_errored: int
    tests_skipped: int
    junit_xml_sha256: str
    raw_terminal_sha256: str
    native_lifecycle_sha256: str
    reset_state_evidence_sha256: str
    hbm_evidence_sha256: str
    junit_xml: EvidenceFileBinding
    raw_terminal: EvidenceFileBinding
    native_lifecycle: CanonicalJsonProofBinding
    reset_state_evidence: CanonicalJsonProofBinding
    hbm_evidence: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "trusted_empirical_tp1_session_reset_authority"
            or self.protocol_sha256
            != TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PROTOCOL_SHA256
            or self.evidence_level != "trusted_single_operator_empirical_no_signature"
            or self.formal_measured is not False
            or self.status != "PASS"
            or self.topology_mode != "tp1_dp1"
            or self.suite_id != TRUSTED_EMPIRICAL_TP1_SESSION_RESET_SUITE
            or self.patched_sglang_tree != PINNED_SGLANG_TREE
        ):
            raise ValueError("trusted empirical TP1 reset authority schema differs")
        for label, value in (
            ("ProtocolLock", self.protocol_lock_sha256),
            ("source snapshot", self.source_snapshot_sha256),
            ("inventory", self.inventory_sha256),
            ("qualification run", self.qualification_run_id),
            ("JUnit", self.junit_xml_sha256),
            ("raw terminal", self.raw_terminal_sha256),
            ("native lifecycle", self.native_lifecycle_sha256),
            ("reset-state evidence", self.reset_state_evidence_sha256),
            ("HBM evidence", self.hbm_evidence_sha256),
        ):
            _require_sha256(f"trusted empirical reset {label}", value)
        _require_git_object_id(
            "trusted empirical reset patched SGLang tree",
            self.patched_sglang_tree,
        )
        _require_text("trusted empirical reset GPU UUID", self.gpu_uuid)
        if not self.gpu_uuid.startswith("GPU-"):
            raise ValueError("trusted empirical reset requires one GPU UUID")
        if self.backend not in _BACKENDS or self.method_family not in (
            _METHOD_FAMILIES
        ):
            raise ValueError("trusted empirical reset backend/method is unsupported")
        if type(self.qualification_spec) is not CanonicalJsonProofBinding:
            raise TypeError("trusted empirical reset qualification is not path-bound")
        if type(self.protocol_lock) is not FormalSingleOperatorJsonBinding:
            raise TypeError("trusted empirical reset ProtocolLock is not path-bound")
        if type(self.content_bundle) is not TrustedSingleOperatorContentBundleBinding:
            raise TypeError("trusted empirical reset content is not path-bound")
        if type(self.inventory) is not CanonicalJsonProofBinding:
            raise TypeError("trusted empirical reset inventory is not path-bound")
        for label, binding in (
            ("JUnit", self.junit_xml),
            ("raw terminal", self.raw_terminal),
        ):
            if type(binding) is not EvidenceFileBinding:
                raise TypeError(f"trusted empirical reset {label} is not path-bound")
        for label, binding in (
            ("native lifecycle", self.native_lifecycle),
            ("reset-state evidence", self.reset_state_evidence),
            ("HBM evidence", self.hbm_evidence),
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError(f"trusted empirical reset {label} is not path-bound")
        if (
            self.protocol_lock.semantic_sha256 != self.protocol_lock_sha256
            or self.content_bundle.runtime_binding_status != "BOUND"
            or self.inventory.semantic_sha256 != self.inventory_sha256
            or self.junit_xml.raw_sha256 != self.junit_xml_sha256
            or self.raw_terminal.raw_sha256 != self.raw_terminal_sha256
            or self.native_lifecycle.semantic_sha256 != self.native_lifecycle_sha256
            or self.reset_state_evidence.semantic_sha256
            != self.reset_state_evidence_sha256
            or self.hbm_evidence.semantic_sha256 != self.hbm_evidence_sha256
        ):
            raise ValueError("trusted empirical reset path/digest binding differs")
        if self.test_names != TRUSTED_EMPIRICAL_TP1_SESSION_RESET_TESTS:
            raise ValueError("trusted empirical reset test coverage differs")
        counts = (
            self.tests_collected,
            self.tests_passed,
            self.tests_failed,
            self.tests_errored,
            self.tests_skipped,
        )
        if any(type(value) is not int or value < 0 for value in counts) or counts != (
            8,
            8,
            0,
            0,
            0,
        ):
            raise ValueError("trusted empirical reset requires exact 8/8 PASS")

    @cached_property
    def scope_key(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.protocol_lock_sha256,
            self.source_snapshot_sha256,
            self.patched_sglang_tree,
            self.inventory_sha256,
            self.gpu_uuid,
            self.backend,
            self.method_family,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    @property
    def operational_reuse_allowed(self) -> bool:
        """Return an operational permission, never a formal measurement claim."""

        return True

    def matches(
        self,
        *,
        protocol_lock_sha256: str,
        source_snapshot_sha256: str,
        patched_sglang_tree: str,
        inventory_sha256: str,
        gpu_uuid: str,
        backend: str,
        method_family: str,
    ) -> bool:
        for label, value in (
            ("ProtocolLock", protocol_lock_sha256),
            ("source snapshot", source_snapshot_sha256),
            ("inventory", inventory_sha256),
        ):
            _require_sha256(f"trusted empirical reset match {label}", value)
        _require_git_object_id(
            "trusted empirical reset match patched SGLang tree",
            patched_sglang_tree,
        )
        _require_text("trusted empirical reset match GPU UUID", gpu_uuid)
        _require_text("trusted empirical reset match backend", backend)
        _require_text("trusted empirical reset match method", method_family)
        return self.scope_key == (
            protocol_lock_sha256,
            source_snapshot_sha256,
            patched_sglang_tree,
            inventory_sha256,
            gpu_uuid,
            backend,
            method_family,
        )

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = asdict(self)
        value["test_names"] = list(self.test_names)
        value["qualification_spec"] = self.qualification_spec.to_dict()
        value["protocol_lock"] = self.protocol_lock.to_dict()
        value["content_bundle"] = self.content_bundle.to_dict()
        value["inventory"] = self.inventory.to_dict()
        value["junit_xml"] = self.junit_xml.to_dict()
        value["raw_terminal"] = self.raw_terminal.to_dict()
        value["native_lifecycle"] = self.native_lifecycle.to_dict()
        value["reset_state_evidence"] = self.reset_state_evidence.to_dict()
        value["hbm_evidence"] = self.hbm_evidence.to_dict()
        if include_sha256:
            value["authority_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            *cls.__dataclass_fields__,
            "authority_sha256",
        }:
            raise ValueError("trusted empirical reset authority fields differ")
        row = dict(value)
        declared = _require_sha256(
            "trusted empirical reset authority",
            row.pop("authority_sha256"),
        )
        raw_tests = row.pop("test_names")
        if type(raw_tests) is not list or any(
            type(item) is not str for item in raw_tests
        ):
            raise TypeError("trusted empirical reset tests must be an array")
        row["qualification_spec"] = CanonicalJsonProofBinding.from_dict(
            row["qualification_spec"]
        )
        row["protocol_lock"] = FormalSingleOperatorJsonBinding.from_dict(
            row["protocol_lock"]
        )
        row["content_bundle"] = TrustedSingleOperatorContentBundleBinding.from_dict(
            row["content_bundle"]
        )
        row["inventory"] = CanonicalJsonProofBinding.from_dict(row["inventory"])
        for name in ("junit_xml", "raw_terminal"):
            row[name] = EvidenceFileBinding.from_dict(
                row[name], label=f"trusted empirical reset {name}"
            )
        for name in (
            "native_lifecycle",
            "reset_state_evidence",
            "hbm_evidence",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        result = cls(**row, test_names=tuple(raw_tests))  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("trusted empirical reset authority digest differs")
        return result


def _strict_evidence_object(
    value: object,
    *,
    label: str,
    fields: set[str],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"trusted empirical reset {label} fields differ")
    return dict(value)


def _validate_common_evidence_scope(
    row: dict[str, object],
    *,
    spec: TrustedEmpiricalTp1SessionResetQualificationSpec,
    kind: str,
) -> None:
    if (
        row["schema_version"] != 1
        or row["kind"] != kind
        or row["suite_id"] != spec.suite_id
        or row["topology_mode"] != "tp1_dp1"
        or row["gpu_uuid"] != spec.gpu_uuid
        or row["backend"] != spec.backend
        or row["method_family"] != spec.method_family
        or row["qualification_run_id"] != spec.qualification_run_id
    ):
        raise ValueError("trusted empirical reset evidence scope differs")


def _validate_native_lifecycle(
    value: object,
    *,
    spec: TrustedEmpiricalTp1SessionResetQualificationSpec,
) -> None:
    row = _strict_evidence_object(
        value,
        label="native lifecycle",
        fields={
            "schema_version",
            "kind",
            "suite_id",
            "topology_mode",
            "gpu_uuid",
            "backend",
            "method_family",
            "qualification_run_id",
            "server_pid",
            "session_epochs",
            "execution_plan_sha256s",
            "exact_output_token_trajectory",
            "native_timestamp_coverage",
        },
    )
    _validate_common_evidence_scope(
        row,
        spec=spec,
        kind="trusted_empirical_tp1_session_reset_native_lifecycle",
    )
    epochs = row["session_epochs"]
    plans = row["execution_plan_sha256s"]
    if (
        type(row["server_pid"]) is not int
        or row["server_pid"] < 1
        or type(epochs) is not list
        or epochs != [1, 2]
        or type(plans) is not list
        or len(plans) != 2
        or len(set(plans)) != 2
        or any(
            type(plan) is not str
            or len(plan) != 64
            or any(character not in "0123456789abcdef" for character in plan)
            for plan in plans
        )
        or row["exact_output_token_trajectory"] is not True
        or row["native_timestamp_coverage"] is not True
    ):
        raise ValueError("trusted empirical reset native lifecycle did not pass")


def _validate_reset_state(
    value: object,
    *,
    spec: TrustedEmpiricalTp1SessionResetQualificationSpec,
) -> None:
    row = _strict_evidence_object(
        value,
        label="reset state",
        fields={
            "schema_version",
            "kind",
            "suite_id",
            "topology_mode",
            "gpu_uuid",
            "backend",
            "method_family",
            "qualification_run_id",
            "reset_boundary_count",
            "request_queue_empty",
            "optimizer_state_reset",
            "candidate_state_reset",
            "adaptation_state_reset",
            "registered_cache_policy_restored",
            "terminal_writer_flushed",
            "previous_requests_fully_terminal",
        },
    )
    _validate_common_evidence_scope(
        row,
        spec=spec,
        kind="trusted_empirical_tp1_session_reset_state_evidence",
    )
    gates = (
        "request_queue_empty",
        "optimizer_state_reset",
        "candidate_state_reset",
        "adaptation_state_reset",
        "registered_cache_policy_restored",
        "terminal_writer_flushed",
        "previous_requests_fully_terminal",
    )
    if row["reset_boundary_count"] != 2 or any(row[name] is not True for name in gates):
        raise ValueError("trusted empirical reset state did not pass")


def _validate_hbm_evidence(
    value: object,
    *,
    spec: TrustedEmpiricalTp1SessionResetQualificationSpec,
) -> None:
    row = _strict_evidence_object(
        value,
        label="HBM evidence",
        fields={
            "schema_version",
            "kind",
            "suite_id",
            "topology_mode",
            "gpu_uuid",
            "backend",
            "method_family",
            "qualification_run_id",
            "initial_memory_bytes",
            "memory_after_reset_bytes",
            "allowed_growth_bytes",
            "monotonic_growth_detected",
        },
    )
    _validate_common_evidence_scope(
        row,
        spec=spec,
        kind="trusted_empirical_tp1_session_reset_hbm_evidence",
    )
    initial = row["initial_memory_bytes"]
    after = row["memory_after_reset_bytes"]
    allowed = row["allowed_growth_bytes"]
    if (
        type(initial) is not int
        or initial < 0
        or type(after) is not list
        or len(after) != 2
        or any(type(item) is not int or item < 0 for item in after)
        or type(allowed) is not int
        or allowed < 0
        or row["monotonic_growth_detected"] is not False
        or any(after_value > initial + allowed for after_value in after)
    ):
        raise ValueError("trusted empirical reset HBM gate did not pass")


def _validate_junit(
    binding: EvidenceFileBinding,
) -> None:
    binding.reopen(label="trusted empirical reset JUnit")
    body = Path(relocated_evidence_path(binding.absolute_path)).read_bytes()
    try:
        root = ET.fromstring(body)
    except ET.ParseError as error:
        raise ValueError("trusted empirical reset JUnit is malformed") from error
    cases = tuple(root.iter("testcase"))
    names = tuple(case.attrib.get("name") for case in cases)
    if names != TRUSTED_EMPIRICAL_TP1_SESSION_RESET_TESTS or any(
        any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
        for case in cases
    ):
        raise ValueError("trusted empirical reset JUnit requires exact 8/8 PASS")
    suites = (root,) if root.tag == "testsuite" else tuple(root.iter("testsuite"))
    if not suites:
        raise ValueError("trusted empirical reset JUnit lacks a test suite")

    def declared(name: str) -> int:
        total = 0
        for suite in suites:
            raw = suite.attrib.get(name, "0")
            if not raw.isdecimal():
                raise ValueError("trusted empirical reset JUnit count is invalid")
            total += int(raw)
        return total

    if (
        declared("tests") != 8
        or declared("failures") != 0
        or declared("errors") != 0
        or declared("skipped") != 0
    ):
        raise ValueError("trusted empirical reset JUnit counts differ")
    if (
        EvidenceFileBinding.bind(
            Path(relocated_evidence_path(binding.absolute_path)),
            label="trusted empirical reset JUnit",
        ).raw_sha256
        != binding.raw_sha256
    ):
        raise RuntimeError("trusted empirical reset JUnit changed while validating")


def _derive_trusted_empirical_tp1_session_reset_authority(
    qualification_spec: CanonicalJsonProofBinding,
) -> TrustedEmpiricalTp1SessionResetAuthority:
    if type(qualification_spec) is not CanonicalJsonProofBinding:
        raise TypeError("trusted empirical reset requires a path-bound qualification")
    rebound_spec = CanonicalJsonProofBinding.bind(qualification_spec.absolute_path)
    if rebound_spec != qualification_spec:
        raise ValueError("trusted empirical reset qualification changed")
    spec = TrustedEmpiricalTp1SessionResetQualificationSpec.from_dict(
        qualification_spec.reopen()
    )
    protocol_binding = FormalSingleOperatorJsonBinding.bind(
        spec.protocol_lock_path,
        label="trusted empirical reset ProtocolLock",
    )
    lock = protocol_lock_from_dict(
        protocol_binding.reopen(label="trusted empirical reset ProtocolLock")
    )
    content_binding = TrustedSingleOperatorContentBundleBinding.bind(
        spec.content_bundle_path
    )
    content = content_binding.reopen()
    inventory_binding = CanonicalJsonProofBinding.bind(spec.inventory_path)
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    inventory.device(spec.gpu_uuid)
    if (
        lock.schema_version != 5
        or lock.content_source_mode != "trusted_single_operator"
        or lock.trusted_single_operator_content_bundle_sha256
        != content_binding.semantic_sha256
        or content_binding.runtime_binding_status != "BOUND"
        or content.runtime_binding_status != "BOUND"
        or content.source_snapshot.patched_sglang_tree != PINNED_SGLANG_TREE
        or inventory.sha256 != inventory_binding.semantic_sha256
    ):
        raise ValueError("trusted empirical reset upstream identity differs")

    junit = EvidenceFileBinding.bind(
        Path(spec.junit_xml_path), label="trusted empirical reset JUnit"
    )
    raw_terminal = EvidenceFileBinding.bind(
        Path(spec.raw_terminal_path), label="trusted empirical reset raw terminal"
    )
    native_lifecycle = CanonicalJsonProofBinding.bind(spec.native_lifecycle_path)
    reset_state = CanonicalJsonProofBinding.bind(spec.reset_state_evidence_path)
    hbm = CanonicalJsonProofBinding.bind(spec.hbm_evidence_path)
    _validate_junit(junit)
    raw_terminal.reopen(label="trusted empirical reset raw terminal")
    _validate_native_lifecycle(native_lifecycle.reopen(), spec=spec)
    _validate_reset_state(reset_state.reopen(), spec=spec)
    _validate_hbm_evidence(hbm.reopen(), spec=spec)

    authority = TrustedEmpiricalTp1SessionResetAuthority(
        schema_version=1,
        kind="trusted_empirical_tp1_session_reset_authority",
        protocol_sha256=TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PROTOCOL_SHA256,
        evidence_level="trusted_single_operator_empirical_no_signature",
        formal_measured=False,
        status="PASS",
        topology_mode="tp1_dp1",
        suite_id=TRUSTED_EMPIRICAL_TP1_SESSION_RESET_SUITE,
        protocol_lock_sha256=lock.sha256,
        source_snapshot_sha256=content.source_snapshot.source_snapshot_sha256,
        patched_sglang_tree=content.source_snapshot.patched_sglang_tree,
        inventory_sha256=inventory.sha256,
        gpu_uuid=spec.gpu_uuid,
        backend=spec.backend,
        method_family=spec.method_family,
        qualification_run_id=spec.qualification_run_id,
        qualification_spec=qualification_spec,
        protocol_lock=protocol_binding,
        content_bundle=content_binding,
        inventory=inventory_binding,
        test_names=TRUSTED_EMPIRICAL_TP1_SESSION_RESET_TESTS,
        tests_collected=8,
        tests_passed=8,
        tests_failed=0,
        tests_errored=0,
        tests_skipped=0,
        junit_xml_sha256=junit.raw_sha256,
        raw_terminal_sha256=raw_terminal.raw_sha256,
        native_lifecycle_sha256=native_lifecycle.semantic_sha256,
        reset_state_evidence_sha256=reset_state.semantic_sha256,
        hbm_evidence_sha256=hbm.semantic_sha256,
        junit_xml=junit,
        raw_terminal=raw_terminal,
        native_lifecycle=native_lifecycle,
        reset_state_evidence=reset_state,
        hbm_evidence=hbm,
    )
    if CanonicalJsonProofBinding.bind(qualification_spec.absolute_path) != (
        qualification_spec
    ):
        raise RuntimeError("trusted empirical reset inputs changed while deriving")
    return authority


def publish_trusted_empirical_tp1_session_reset_authority(
    *,
    qualification_spec_path: str | Path,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish from paths only; no caller-authored digest can enter the result."""

    spec_binding = CanonicalJsonProofBinding.bind(qualification_spec_path)
    authority = _derive_trusted_empirical_tp1_session_reset_authority(spec_binding)
    publish_canonical_json_no_replace(output_path, authority.to_dict())
    binding, reopened = revalidate_trusted_empirical_tp1_session_reset_authority(
        output_path
    )
    if reopened != authority:
        raise RuntimeError("trusted empirical reset authority changed on publication")
    return binding


def revalidate_trusted_empirical_tp1_session_reset_authority(
    path: str | Path,
) -> tuple[CanonicalJsonProofBinding, TrustedEmpiricalTp1SessionResetAuthority]:
    """Deep-reopen the authority and every upstream/raw evidence member."""

    binding = CanonicalJsonProofBinding.bind(path)
    authority = TrustedEmpiricalTp1SessionResetAuthority.from_dict(binding.reopen())
    expected = _derive_trusted_empirical_tp1_session_reset_authority(
        authority.qualification_spec
    )
    if authority != expected:
        raise ValueError("trusted empirical reset authority replay differs")
    return binding, authority


__all__ = (
    "TRUSTED_EMPIRICAL_TP1_SESSION_RESET_PROTOCOL_SHA256",
    "TRUSTED_EMPIRICAL_TP1_SESSION_RESET_QUALIFICATION_SPEC_PROTOCOL_SHA256",
    "TRUSTED_EMPIRICAL_TP1_SESSION_RESET_SUITE",
    "TRUSTED_EMPIRICAL_TP1_SESSION_RESET_TESTS",
    "TrustedEmpiricalTp1SessionResetAuthority",
    "TrustedEmpiricalTp1SessionResetQualificationSpec",
    "TrustedSessionResetBackend",
    "TrustedSessionResetMethodFamily",
    "publish_trusted_empirical_tp1_session_reset_authority",
    "revalidate_trusted_empirical_tp1_session_reset_authority",
)
