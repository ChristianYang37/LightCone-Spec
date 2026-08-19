"""Trusted single-operator ChronoBelief GPU-parity evidence.

The legacy release path authorizes ChronoBelief with a root-signed
``VerifiedNativeRuntimeGpuProof``.  A trusted single operator does not possess
that release authority, but it still needs to bind a ChronoBelief winner to
the *actual* fresh preflight evidence before constructing DSpark E1a runs.

This module deliberately creates a different, explicitly empirical token.  It
deep-replays the current preflight coverage graph, the two-GPU exactness
pointer, the ChronoBelief mixed-precision qualification, and the independent
DSpark TP1 qualification.  The resulting token is path-bound to the exact
DSpark prerequisite launch and can only be consumed by a prepared E1a launch
whose GPU belongs to the parity-qualified UUID set and whose GPU model,
driver/CUDA stack, patched SGLang tree, content bundle, inventory epoch, and
ProtocolLock still match.  It is never a signed or formal-MEASURED authority.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.config import load_run_config
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry import protocol_lock_from_dict
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorPreflightActualReceipt,
    load_formal_single_operator_execution_source,
    rebuild_formal_single_operator_stage_completion,
)
from lightcone_spec.runtime.compile_cache import CompileCacheLaunchPlan
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.native_qualification_runner import (
    NativeRuntimeQualificationAssignment,
    NativeRuntimeQualificationResultPointer,
)
from lightcone_spec.runtime.preflight_runner import (
    ExactnessPreflightAssignment,
    ExactnessPreflightResultPointer,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.readiness import NATIVE_RUNTIME_QUALIFICATION_TESTS

_SHA256 = frozenset("0123456789abcdef")
_ENVIRONMENT_KEY_FIELDS = (
    "patched_sglang_tree",
    "patch_manifest_sha256",
    "patch_sha256",
    "source_sha256",
    "python_version",
    "torch_version",
    "triton_version",
    "cuda_version",
    "driver_version",
    "sm_architecture",
    "gpu_model",
    "dtype",
    "allocator",
    "build_flags",
)

TRUSTED_SINGLE_OPERATOR_CHRONOBELIEF_GPU_PARITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "trusted_single_operator_chronobelief_gpu_parity_protocol",
        "trust_mode": "trusted_single_operator_empirical_no_signature",
        "formal_execution_authorized": False,
        "source": (
            "current_E1a_execution_source_and_exact_preflight_completion_deep_replay"
        ),
        "join": (
            "two_gpu_exactness_identity_plus_actual_chronobelief_qualified_GPU_"
            "UUID_set_plus_independent_dspark_tp1_qualification"
        ),
        "runtime": (
            "prepared_GPU_membership_in_qualified_set_plus_same_GPU_model_driver_"
            "CUDA_patched_tree_patch_manifest_patch_bytes_content_inventory_"
            "runtime_and_ProtocolLock"
        ),
        "raw_evidence": "terminal_JUnit_log_and_unsigned_receipt_SHA256",
        "consumer": "ChronoBelief_E1a_DSpark_TP1_prepared_launch_only",
        "forbidden": (
            "DFLASH_prerequisite_proof_clone",
            "GPU_outside_qualified_set_or_different_model_driver_CUDA_patch_dtype",
            "non_ChronoBelief_RunConfig",
            "signed_or_formal_MEASURED_claim",
        ),
    }
)


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError(f"{label} must be canonical non-empty text")
    return value


def _strict(value: object, expected: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return dict(value)


@dataclass(frozen=True)
class _TrustedChronoBeliefEvidenceJoin:
    execution_source_sha256: str
    protocol_lock_sha256: str
    content_source_binding_sha256: str
    content_bundle_sha256: str
    runtime_authority_manifest_sha256: str
    preflight_runtime_sha256: str
    inventory_sha256: str
    exactness_gpu_uuids: tuple[str, str]
    qualified_gpu_uuids: tuple[str, ...]
    dspark_qualification_gpu_uuids: tuple[str, ...]
    gpu_model: str
    driver_version: str
    cuda_version: str
    patched_sglang_commit: str
    patched_sglang_tree: str
    patch_manifest_sha256: str
    patch_sha256: str
    dtype: str
    dtype_parity_test_names: tuple[str, ...]
    exactness_terminal_raw_sha256: str
    exactness_junit_raw_sha256: str
    exactness_log_raw_sha256: str
    chronobelief_terminal_raw_sha256: str
    chronobelief_junit_raw_sha256: str
    chronobelief_log_raw_sha256: str
    chronobelief_unsigned_receipt_raw_sha256: str
    dspark_terminal_raw_sha256: str
    dspark_junit_raw_sha256: str
    dspark_log_raw_sha256: str
    dspark_unsigned_receipt_raw_sha256: str


@dataclass(frozen=True)
class TrustedSingleOperatorChronoBeliefGpuParityProof:
    """Path-bound empirical parity proof; explicitly not release authority."""

    schema_version: Literal[1]
    kind: Literal["trusted_single_operator_chronobelief_gpu_parity_proof"]
    protocol_sha256: str
    trust_mode: Literal["trusted_single_operator_empirical_no_signature"]
    formal_execution_authorized: Literal[False]
    execution_source: CanonicalJsonProofBinding
    protocol_lock: CanonicalJsonProofBinding
    preflight_actual: CanonicalJsonProofBinding
    preflight_coverage: CanonicalJsonProofBinding
    exactness_result_pointer: CanonicalJsonProofBinding
    chronobelief_result_pointer: CanonicalJsonProofBinding
    chronobelief_proof_artifact: CanonicalJsonProofBinding
    dspark_result_pointer: CanonicalJsonProofBinding
    dspark_proof_artifact: CanonicalJsonProofBinding
    prerequisite_launch: CanonicalJsonProofBinding
    execution_source_sha256: str
    protocol_lock_sha256: str
    content_source_binding_sha256: str
    content_bundle_sha256: str
    runtime_authority_manifest_sha256: str
    preflight_runtime_sha256: str
    inventory_sha256: str
    exactness_gpu_uuids: tuple[str, str]
    qualified_gpu_uuids: tuple[str, ...]
    dspark_qualification_gpu_uuids: tuple[str, ...]
    gpu_model: str
    driver_version: str
    cuda_version: str
    patched_sglang_commit: str
    patched_sglang_tree: str
    patch_manifest_sha256: str
    patch_sha256: str
    dtype: str
    dtype_parity_test_names: tuple[str, ...]
    exactness_terminal_raw_sha256: str
    exactness_junit_raw_sha256: str
    exactness_log_raw_sha256: str
    chronobelief_terminal_raw_sha256: str
    chronobelief_junit_raw_sha256: str
    chronobelief_log_raw_sha256: str
    chronobelief_unsigned_receipt_raw_sha256: str
    dspark_terminal_raw_sha256: str
    dspark_junit_raw_sha256: str
    dspark_log_raw_sha256: str
    dspark_unsigned_receipt_raw_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "trusted_single_operator_chronobelief_gpu_parity_proof"
            or self.protocol_sha256
            != TRUSTED_SINGLE_OPERATOR_CHRONOBELIEF_GPU_PARITY_PROTOCOL_SHA256
            or self.trust_mode != "trusted_single_operator_empirical_no_signature"
            or self.formal_execution_authorized is not False
        ):
            raise ValueError("trusted ChronoBelief proof schema differs")
        for field in (
            self.execution_source,
            self.protocol_lock,
            self.preflight_actual,
            self.preflight_coverage,
            self.exactness_result_pointer,
            self.chronobelief_result_pointer,
            self.chronobelief_proof_artifact,
            self.dspark_result_pointer,
            self.dspark_proof_artifact,
            self.prerequisite_launch,
        ):
            if type(field) is not CanonicalJsonProofBinding:
                raise TypeError("trusted ChronoBelief proof source is not path-bound")
            if CanonicalJsonProofBinding.bind(field.absolute_path) != field:
                raise ValueError("trusted ChronoBelief proof source changed")
        for item in fields(_TrustedChronoBeliefEvidenceJoin):
            value = getattr(self, item.name)
            if item.name.endswith("sha256"):
                _require_sha256(f"trusted ChronoBelief {item.name}", value)
        for label, value in (
            ("GPU model", self.gpu_model),
            ("driver", self.driver_version),
            ("CUDA", self.cuda_version),
            ("patched commit", self.patched_sglang_commit),
            ("patched tree", self.patched_sglang_tree),
            ("dtype", self.dtype),
        ):
            _require_text(f"trusted ChronoBelief {label}", value)
        if (
            type(self.exactness_gpu_uuids) is not tuple
            or len(self.exactness_gpu_uuids) != 2
            or len(set(self.exactness_gpu_uuids)) != 2
            or type(self.qualified_gpu_uuids) is not tuple
            or not self.qualified_gpu_uuids
            or self.qualified_gpu_uuids != tuple(sorted(self.qualified_gpu_uuids))
            or len(set(self.qualified_gpu_uuids)) != len(self.qualified_gpu_uuids)
            or not set(self.qualified_gpu_uuids).issubset(self.exactness_gpu_uuids)
            or type(self.dspark_qualification_gpu_uuids) is not tuple
            or not self.dspark_qualification_gpu_uuids
            or not set(self.dspark_qualification_gpu_uuids).issubset(
                self.exactness_gpu_uuids
            )
        ):
            raise ValueError("trusted ChronoBelief qualified GPU set differs")
        for gpu_uuid in (
            *self.exactness_gpu_uuids,
            *self.qualified_gpu_uuids,
            *self.dspark_qualification_gpu_uuids,
        ):
            _require_text("trusted ChronoBelief GPU UUID", gpu_uuid)
        if (
            self.dtype_parity_test_names
            != NATIVE_RUNTIME_QUALIFICATION_TESTS["chronobelief_gpu_parity"]
        ):
            raise ValueError("trusted ChronoBelief dtype parity suite differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    @property
    def source_identity_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "trusted_single_operator_chronobelief_source_identity",
                "proof_sha256": self.sha256,
                "execution_source_sha256": self.execution_source_sha256,
                "protocol_lock_sha256": self.protocol_lock_sha256,
                "content_source_binding_sha256": (self.content_source_binding_sha256),
                "runtime_authority_manifest_sha256": (
                    self.runtime_authority_manifest_sha256
                ),
                "preflight_runtime_sha256": self.preflight_runtime_sha256,
                "qualified_gpu_uuids": self.qualified_gpu_uuids,
            }
        )

    def evidence_join(self) -> _TrustedChronoBeliefEvidenceJoin:
        return _TrustedChronoBeliefEvidenceJoin(
            **{
                field.name: getattr(self, field.name)
                for field in fields(_TrustedChronoBeliefEvidenceJoin)
            }
        )

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name
            not in {
                "execution_source",
                "protocol_lock",
                "preflight_actual",
                "preflight_coverage",
                "exactness_result_pointer",
                "chronobelief_result_pointer",
                "chronobelief_proof_artifact",
                "dspark_result_pointer",
                "dspark_proof_artifact",
                "prerequisite_launch",
                "dtype_parity_test_names",
                "exactness_gpu_uuids",
                "qualified_gpu_uuids",
                "dspark_qualification_gpu_uuids",
            }
        }
        for name in (
            "execution_source",
            "protocol_lock",
            "preflight_actual",
            "preflight_coverage",
            "exactness_result_pointer",
            "chronobelief_result_pointer",
            "chronobelief_proof_artifact",
            "dspark_result_pointer",
            "dspark_proof_artifact",
            "prerequisite_launch",
        ):
            value[name] = getattr(self, name).to_dict()
        value["dtype_parity_test_names"] = list(self.dtype_parity_test_names)
        value["exactness_gpu_uuids"] = list(self.exactness_gpu_uuids)
        value["qualified_gpu_uuids"] = list(self.qualified_gpu_uuids)
        value["dspark_qualification_gpu_uuids"] = list(
            self.dspark_qualification_gpu_uuids
        )
        if include_sha256:
            value["proof_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            value,
            set(cls.__dataclass_fields__) | {"proof_sha256"},
            label="trusted ChronoBelief proof",
        )
        declared = _require_sha256(
            "trusted ChronoBelief proof", row.pop("proof_sha256")
        )
        for name in (
            "execution_source",
            "protocol_lock",
            "preflight_actual",
            "preflight_coverage",
            "exactness_result_pointer",
            "chronobelief_result_pointer",
            "chronobelief_proof_artifact",
            "dspark_result_pointer",
            "dspark_proof_artifact",
            "prerequisite_launch",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        raw_tests = row.pop("dtype_parity_test_names")
        raw_exactness_gpu_uuids = row.pop("exactness_gpu_uuids")
        raw_qualified_gpu_uuids = row.pop("qualified_gpu_uuids")
        raw_dspark_gpu_uuids = row.pop("dspark_qualification_gpu_uuids")
        if any(
            type(raw) is not list
            for raw in (
                raw_tests,
                raw_exactness_gpu_uuids,
                raw_qualified_gpu_uuids,
                raw_dspark_gpu_uuids,
            )
        ):
            raise TypeError("trusted ChronoBelief tuple field is not an array")
        result = cls(
            dtype_parity_test_names=tuple(raw_tests),
            exactness_gpu_uuids=tuple(raw_exactness_gpu_uuids),
            qualified_gpu_uuids=tuple(raw_qualified_gpu_uuids),
            dspark_qualification_gpu_uuids=tuple(raw_dspark_gpu_uuids),
            **row,
        )  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("trusted ChronoBelief proof digest differs")
        return result


def _one_preflight_actual(source: object) -> tuple[CanonicalJsonProofBinding, object]:
    predecessor_source = getattr(source, "predecessor_completion_source", None)
    if predecessor_source is None:
        raise ValueError("trusted ChronoBelief proof lacks predecessor completion")
    current = rebuild_formal_single_operator_stage_completion(
        predecessor_source.absolute_path
    )
    matches = []
    while current is not None:
        if current.artifact.node == "preflight":
            matches.append(current)
        current = current.predecessor
    if len(matches) != 1:
        raise ValueError("trusted ChronoBelief proof lacks one fresh preflight")
    actual_sources = {
        row.source.absolute_path: row.source
        for row in matches[0].artifact.actual_results
    }
    if len(actual_sources) != 1:
        raise ValueError("trusted ChronoBelief preflight source is not unique")
    actual_source = next(iter(actual_sources.values()))
    binding = CanonicalJsonProofBinding.bind(actual_source.absolute_path)
    raw = actual_source.reopen(label="trusted ChronoBelief preflight actual")
    if (
        type(raw) is dict
        and raw.get("kind") == "formal_single_operator_exact_ten_preflight_completion"
    ):
        from lightcone_spec.experiments.formal_preflight_inputs import (
            FormalSingleOperatorPreflightCompletion,
            revalidate_formal_single_operator_preflight_completion,
        )

        serialized = FormalSingleOperatorPreflightCompletion.from_dict(raw)
        return binding, revalidate_formal_single_operator_preflight_completion(
            binding.absolute_path,
            current_ns=serialized.finished_ns,
        )
    return binding, FormalSingleOperatorPreflightActualReceipt.from_dict(raw)


def _qualification_source(coverage: object, suite_id: str) -> object:
    matches = tuple(
        row for row in coverage.qualification_proof_sources if row.suite_id == suite_id
    )
    if len(matches) != 1:
        raise ValueError(f"trusted ChronoBelief proof lacks exact {suite_id}")
    return matches[0]


def _same_environment(*plans: CompileCacheLaunchPlan) -> bool:
    first = plans[0].key
    return all(
        all(
            getattr(plan.key, name) == getattr(first, name)
            for name in _ENVIRONMENT_KEY_FIELDS
        )
        and plan.cache_root == plans[0].cache_root
        for plan in plans[1:]
    )


def _collect_deep_evidence(
    *,
    execution_source_path: str | Path,
    prerequisite_launch_path: str | Path,
) -> tuple[
    _TrustedChronoBeliefEvidenceJoin,
    dict[str, CanonicalJsonProofBinding],
]:
    """Deep-replay the source graph and flatten only immutable join claims."""

    source_binding = CanonicalJsonProofBinding.bind(execution_source_path)
    source = load_formal_single_operator_execution_source(source_binding.absolute_path)
    if (
        source.stage != "E1a"
        or source.schema_version != 3
        or source.content_source_binding is None
    ):
        raise ValueError("trusted ChronoBelief proof is restricted to trusted E1a")
    protocol_binding = CanonicalJsonProofBinding.bind(
        source.protocol_lock_source.absolute_path
    )
    protocol_lock = protocol_lock_from_dict(protocol_binding.reopen())
    if (
        protocol_lock.schema_version != 5
        or protocol_lock.content_source_mode != "trusted_single_operator"
        or protocol_lock.sha256 != source.protocol_lock_sha256
        or protocol_lock.trusted_single_operator_content_bundle_sha256
        != source.content_source_binding.content_sha256
    ):
        raise ValueError("trusted ChronoBelief ProtocolLock/content differs")

    preflight_binding, preflight = _one_preflight_actual(source)
    from lightcone_spec.experiments.formal_preflight_inputs import (
        FormalPreflightExecutionInputs,
        FormalSingleOperatorPreflightCompletion,
    )

    trusted_completion = type(preflight) is FormalSingleOperatorPreflightCompletion
    if trusted_completion:
        from lightcone_spec.experiments.formal_single_operator_preflight_qualification import (
            load_formal_single_operator_preflight_qualification_plan,
            load_formal_single_operator_preflight_qualification_plan_index,
            revalidate_formal_single_operator_preflight_qualification_result,
        )

        coverage_binding = preflight.execution_inputs
        inputs = FormalPreflightExecutionInputs.from_dict(coverage_binding.reopen())
        if inputs.schema_version != 4 or inputs.qualification_plan_index is None:
            raise ValueError("trusted ChronoBelief preflight qualifications are absent")
        index = load_formal_single_operator_preflight_qualification_plan_index(
            inputs.qualification_plan_index.absolute_path
        )
        plans_by_suite = {
            plan.suite_id: (binding, plan)
            for binding in index.plans
            for plan in (
                load_formal_single_operator_preflight_qualification_plan(
                    binding.absolute_path
                ),
            )
        }
        chrono_plan_binding, chrono_plan = plans_by_suite["chronobelief_gpu_parity"]
        dspark_plan_binding, dspark_plan = plans_by_suite["dspark_tp1"]
        chrono_binding = CanonicalJsonProofBinding.bind(chrono_plan.result_path)
        dspark_binding = CanonicalJsonProofBinding.bind(dspark_plan.result_path)
        chrono_pointer = (
            revalidate_formal_single_operator_preflight_qualification_result(
                chrono_binding.absolute_path
            )
        )
        dspark_pointer = (
            revalidate_formal_single_operator_preflight_qualification_result(
                dspark_binding.absolute_path
            )
        )
        chrono_assignment = NativeRuntimeQualificationAssignment.load(
            chrono_pointer.assignment.absolute_path
        )
        dspark_assignment = NativeRuntimeQualificationAssignment.load(
            dspark_pointer.assignment.absolute_path
        )
        chrono_launch = CompileLaunchManifest.load(
            chrono_assignment.launch_manifest.absolute_path
        )
        dspark_launch = CompileLaunchManifest.load(
            dspark_assignment.launch_manifest.absolute_path
        )
        exactness_binding = preflight.exactness_result
        exactness = ExactnessPreflightResultPointer.load(
            exactness_binding.absolute_path
        )
        runtime_sha256 = protocol_lock.formal_runtime_authority_manifest_sha256
        chrono_proof_binding = chrono_pointer.empirical_proof
        dspark_proof_binding = dspark_pointer.empirical_proof
        chrono_terminal_raw = chrono_pointer.runner_terminal.raw_sha256
        chrono_junit_raw = chrono_pointer.junit_xml.raw_sha256
        chrono_log_raw = chrono_pointer.stdout.raw_sha256
        chrono_receipt_raw = chrono_pointer.empirical_proof.raw_sha256
        dspark_terminal_raw = dspark_pointer.runner_terminal.raw_sha256
        dspark_junit_raw = dspark_pointer.junit_xml.raw_sha256
        dspark_log_raw = dspark_pointer.stdout.raw_sha256
        dspark_receipt_raw = dspark_pointer.empirical_proof.raw_sha256
        _ = (chrono_plan_binding, dspark_plan_binding)
    else:
        coverage_binding = CanonicalJsonProofBinding.bind(
            preflight.final_evidence_source.absolute_path
        )
        from lightcone_spec.experiments.formal_preflight_coverage import (
            FormalPreflightStageCoverageProofArtifact,
            revalidate_formal_preflight_stage_coverage_proof_artifact,
        )

        coverage = FormalPreflightStageCoverageProofArtifact.from_dict(
            coverage_binding.reopen()
        )
        revalidate_formal_preflight_stage_coverage_proof_artifact(
            coverage_binding.absolute_path,
            now_ns=max(time.time_ns(), preflight.verified_ns, coverage.verified_ns),
        )
        if (
            preflight.protocol_lock_sha256 != protocol_lock.sha256
            or coverage.protocol_lock_sha256 != protocol_lock.sha256
            or coverage_binding.semantic_sha256 != content_sha256(coverage.to_dict())
        ):
            raise ValueError("trusted ChronoBelief preflight lineage differs")
        exactness_binding = CanonicalJsonProofBinding.bind(
            coverage.exactness_result_source.absolute_path
        )
        exactness = ExactnessPreflightResultPointer.load(
            exactness_binding.absolute_path
        )
        chrono_source = _qualification_source(coverage, "chronobelief_gpu_parity")
        chrono_binding = CanonicalJsonProofBinding.bind(
            chrono_source.result_pointer.absolute_path
        )
        chrono_pointer = NativeRuntimeQualificationResultPointer.load(
            chrono_binding.absolute_path
        )
        chrono_assignment = NativeRuntimeQualificationAssignment.load(
            chrono_pointer.assignment.absolute_path
        )
        chrono_launch = CompileLaunchManifest.load(
            chrono_assignment.launch_manifest.absolute_path
        )
        dspark_source = _qualification_source(coverage, "dspark_tp1")
        dspark_binding = CanonicalJsonProofBinding.bind(
            dspark_source.result_pointer.absolute_path
        )
        dspark_pointer = NativeRuntimeQualificationResultPointer.load(
            dspark_binding.absolute_path
        )
        dspark_assignment = NativeRuntimeQualificationAssignment.load(
            dspark_pointer.assignment.absolute_path
        )
        dspark_launch = CompileLaunchManifest.load(
            dspark_assignment.launch_manifest.absolute_path
        )
        runtime_sha256 = coverage.runtime_sha256
        inventory_sha256 = coverage.inventory_sha256
        chrono_proof_binding = CanonicalJsonProofBinding.bind(
            chrono_source.proof_artifact.absolute_path
        )
        dspark_proof_binding = CanonicalJsonProofBinding.bind(
            dspark_source.proof_artifact.absolute_path
        )
        chrono_terminal_raw = chrono_pointer.runner_terminal.raw_sha256
        chrono_junit_raw = chrono_pointer.junit_xml.raw_sha256
        chrono_log_raw = chrono_pointer.log.raw_sha256
        chrono_receipt_raw = chrono_pointer.unsigned_gpu_proof.raw_sha256
        dspark_terminal_raw = dspark_pointer.runner_terminal.raw_sha256
        dspark_junit_raw = dspark_pointer.junit_xml.raw_sha256
        dspark_log_raw = dspark_pointer.log.raw_sha256
        dspark_receipt_raw = dspark_pointer.unsigned_gpu_proof.raw_sha256
    if exactness.junit_xml is None:
        raise ValueError("trusted ChronoBelief exactness lacks JUnit")
    exactness_assignment = ExactnessPreflightAssignment.load(
        exactness.assignment.absolute_path
    )
    if trusted_completion:
        inventory_sha256 = exactness_assignment.inventory_sha256

    prerequisite_binding = CanonicalJsonProofBinding.bind(prerequisite_launch_path)
    prerequisite = CompileLaunchManifest.load(prerequisite_binding.absolute_path)
    prerequisite_config = load_run_config(prerequisite.run_config_path)
    chrono_config = load_run_config(chrono_launch.run_config_path)
    dspark_config = load_run_config(dspark_launch.run_config_path)
    plans = tuple(
        CompileCacheLaunchPlan.load(launch.compile_cache_plan_path)
        for launch in (chrono_launch, dspark_launch, prerequisite)
    )
    environment = plans[0].key
    identities = {
        (
            launch.target_model_id,
            launch.target_revision,
            launch.drafter_model_id,
            launch.drafter_revision,
            launch.tokenizer_model_id,
            launch.tokenizer_revision,
        )
        for launch in (chrono_launch, dspark_launch, prerequisite)
    }
    content = source.content_source_binding
    if (
        chrono_assignment.suite_id != "chronobelief_gpu_parity"
        or dspark_assignment.suite_id != "dspark_tp1"
        or chrono_config.model.algorithm != "DFLASH"
        or dspark_config.model.algorithm != "DSPARK"
        or prerequisite_config.model.algorithm != "DSPARK"
        or prerequisite.formal_stage != "E1a"
        or prerequisite.sha256 != prerequisite_binding.semantic_sha256
        or len(identities) != 1
        or not _same_environment(*plans)
        or exactness_assignment.driver_version != environment.driver_version
        or exactness_assignment.cuda_version != environment.cuda_version
        or exactness_assignment.patched_sglang_commit
        != prerequisite.patched_sglang_commit
        or exactness_assignment.patched_sglang_tree != environment.patched_sglang_tree
        or prerequisite.patched_sglang_tree != environment.patched_sglang_tree
        or protocol_lock.patch_manifest_sha256 != environment.patch_manifest_sha256
        or exactness_assignment.runtime_sha256 != runtime_sha256
        or chrono_assignment.runtime_sha256 != runtime_sha256
        or dspark_assignment.runtime_sha256 != runtime_sha256
        or exactness_assignment.inventory_sha256 != inventory_sha256
        or chrono_assignment.inventory_sha256 != inventory_sha256
        or dspark_assignment.inventory_sha256 != inventory_sha256
        or prerequisite.inventory_sha256 != inventory_sha256
        or len(chrono_assignment.gpu_uuids) != 1
        or not set(chrono_assignment.gpu_uuids).issubset(exactness_assignment.gpu_uuids)
        or not set(dspark_assignment.gpu_uuids).issubset(exactness_assignment.gpu_uuids)
        or exactness_assignment.gpu_model != environment.gpu_model
        or any(model != environment.gpu_model for model in chrono_assignment.gpu_models)
        or any(model != environment.gpu_model for model in dspark_assignment.gpu_models)
        or chrono_assignment.hardware_envelope_sha256
        != exactness_assignment.hardware_envelope_sha256
        or dspark_assignment.hardware_envelope_sha256
        != exactness_assignment.hardware_envelope_sha256
        or exactness_assignment.input_locks.content_source_binding != content
        or chrono_launch.content_source_binding != content
        or dspark_launch.content_source_binding != content
        or prerequisite.content_source_binding != content
    ):
        raise ValueError(
            "ChronoBelief/DSpark prerequisite is not one exact preflight runtime"
        )

    join = _TrustedChronoBeliefEvidenceJoin(
        execution_source_sha256=source.sha256,
        protocol_lock_sha256=protocol_lock.sha256,
        content_source_binding_sha256=content.sha256,
        content_bundle_sha256=content.content_sha256,
        runtime_authority_manifest_sha256=source.runtime_authority_manifest_sha256,
        preflight_runtime_sha256=runtime_sha256,
        inventory_sha256=inventory_sha256,
        exactness_gpu_uuids=exactness_assignment.gpu_uuids,
        qualified_gpu_uuids=tuple(sorted(chrono_assignment.gpu_uuids)),
        dspark_qualification_gpu_uuids=dspark_assignment.gpu_uuids,
        gpu_model=environment.gpu_model,
        driver_version=environment.driver_version,
        cuda_version=environment.cuda_version,
        patched_sglang_commit=prerequisite.patched_sglang_commit,
        patched_sglang_tree=environment.patched_sglang_tree,
        patch_manifest_sha256=environment.patch_manifest_sha256,
        patch_sha256=environment.patch_sha256,
        dtype=environment.dtype,
        dtype_parity_test_names=NATIVE_RUNTIME_QUALIFICATION_TESTS[
            "chronobelief_gpu_parity"
        ],
        exactness_terminal_raw_sha256=exactness.terminal.raw_sha256,
        exactness_junit_raw_sha256=exactness.junit_xml.raw_sha256,
        exactness_log_raw_sha256=exactness.log.raw_sha256,
        chronobelief_terminal_raw_sha256=chrono_terminal_raw,
        chronobelief_junit_raw_sha256=chrono_junit_raw,
        chronobelief_log_raw_sha256=chrono_log_raw,
        chronobelief_unsigned_receipt_raw_sha256=chrono_receipt_raw,
        dspark_terminal_raw_sha256=dspark_terminal_raw,
        dspark_junit_raw_sha256=dspark_junit_raw,
        dspark_log_raw_sha256=dspark_log_raw,
        dspark_unsigned_receipt_raw_sha256=dspark_receipt_raw,
    )
    bindings = {
        "execution_source": source_binding,
        "protocol_lock": protocol_binding,
        "preflight_actual": preflight_binding,
        "preflight_coverage": coverage_binding,
        "exactness_result_pointer": exactness_binding,
        "chronobelief_result_pointer": chrono_binding,
        "chronobelief_proof_artifact": chrono_proof_binding,
        "dspark_result_pointer": dspark_binding,
        "dspark_proof_artifact": dspark_proof_binding,
        "prerequisite_launch": prerequisite_binding,
    }
    return join, bindings


def _validate_join(
    proof: TrustedSingleOperatorChronoBeliefGpuParityProof,
    observed: _TrustedChronoBeliefEvidenceJoin,
) -> None:
    if proof.evidence_join() != observed:
        raise ValueError("trusted ChronoBelief deep evidence join differs")


def build_trusted_single_operator_chronobelief_gpu_parity_proof(
    *,
    execution_source_path: str | Path,
    prerequisite_launch_path: str | Path,
) -> TrustedSingleOperatorChronoBeliefGpuParityProof:
    join, bindings = _collect_deep_evidence(
        execution_source_path=execution_source_path,
        prerequisite_launch_path=prerequisite_launch_path,
    )
    proof = TrustedSingleOperatorChronoBeliefGpuParityProof(
        schema_version=1,
        kind="trusted_single_operator_chronobelief_gpu_parity_proof",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_CHRONOBELIEF_GPU_PARITY_PROTOCOL_SHA256
        ),
        trust_mode="trusted_single_operator_empirical_no_signature",
        formal_execution_authorized=False,
        **bindings,  # type: ignore[arg-type]
        **{
            field.name: getattr(join, field.name)
            for field in fields(_TrustedChronoBeliefEvidenceJoin)
        },
    )
    _validate_join(proof, join)
    return proof


def publish_trusted_single_operator_chronobelief_gpu_parity_proof(
    *,
    execution_source_path: str | Path,
    prerequisite_launch_path: str | Path,
    output_path: str | Path,
) -> TrustedSingleOperatorChronoBeliefGpuParityProof:
    proof = build_trusted_single_operator_chronobelief_gpu_parity_proof(
        execution_source_path=execution_source_path,
        prerequisite_launch_path=prerequisite_launch_path,
    )
    publish_canonical_json_no_replace(output_path, proof.to_dict())
    rebound = load_trusted_single_operator_chronobelief_gpu_parity_proof(output_path)
    if rebound != proof:
        raise RuntimeError("trusted ChronoBelief proof changed during publication")
    return rebound


def load_trusted_single_operator_chronobelief_gpu_parity_proof(
    path: str | Path,
) -> TrustedSingleOperatorChronoBeliefGpuParityProof:
    binding = CanonicalJsonProofBinding.bind(path)
    proof = TrustedSingleOperatorChronoBeliefGpuParityProof.from_dict(binding.reopen())
    if binding.semantic_sha256 != content_sha256(proof.to_dict()):
        raise ValueError("trusted ChronoBelief proof binding differs")
    observed, bindings = _collect_deep_evidence(
        execution_source_path=proof.execution_source.absolute_path,
        prerequisite_launch_path=proof.prerequisite_launch.absolute_path,
    )
    _validate_join(proof, observed)
    for name, expected in bindings.items():
        if getattr(proof, name) != expected:
            raise ValueError("trusted ChronoBelief evidence path binding differs")
    return proof


def revalidate_trusted_single_operator_chronobelief_for_prepared_launch(
    *,
    proof_path: str | Path,
    execution_source_path: str | Path,
    prepared_launch_path: str | Path,
) -> TrustedSingleOperatorChronoBeliefGpuParityProof:
    """Deep-replay immediately before a prepared TP1 server allocation."""

    proof = load_trusted_single_operator_chronobelief_gpu_parity_proof(proof_path)
    if CanonicalJsonProofBinding.bind(execution_source_path) != proof.execution_source:
        raise ValueError(
            "trusted ChronoBelief proof belongs to another execution source"
        )
    prepared = CompileLaunchManifest.load(prepared_launch_path)
    prerequisite = CompileLaunchManifest.load(proof.prerequisite_launch.absolute_path)
    prepared_config = load_run_config(prepared.run_config_path)
    prepared_plan = CompileCacheLaunchPlan.load(prepared.compile_cache_plan_path)
    prerequisite_plan = CompileCacheLaunchPlan.load(
        prerequisite.compile_cache_plan_path
    )
    adaptation = prepared_config.adaptation
    if (
        prepared.schema_version != 2
        or prepared.formal_stage != "E1a"
        or prepared_config.model.algorithm != "DSPARK"
        or prepared_config.runtime.topology_mode != "tp1_dp1"
        or adaptation is None
        or adaptation.optimizer.name != "chronobelief"
        or adaptation.chronobelief_gpu_proof_sha256 != proof.sha256
        or prepared.inventory_sha256 != proof.inventory_sha256
        or len(prepared.gpu_uuids) != 1
        or prepared.gpu_uuids[0] not in proof.qualified_gpu_uuids
        or prepared.content_source_binding != prerequisite.content_source_binding
        or prepared.patched_sglang_commit != proof.patched_sglang_commit
        or prepared.patched_sglang_tree != proof.patched_sglang_tree
        or any(
            getattr(prepared_plan.key, name) != getattr(prerequisite_plan.key, name)
            for name in _ENVIRONMENT_KEY_FIELDS
        )
        or prepared_plan.key.driver_version != proof.driver_version
        or prepared_plan.key.cuda_version != proof.cuda_version
        or prepared_plan.key.patch_manifest_sha256 != proof.patch_manifest_sha256
        or prepared_plan.key.patch_sha256 != proof.patch_sha256
        or prepared_plan.key.dtype != proof.dtype
        or prepared_plan.key.gpu_model != proof.gpu_model
        or (
            prepared.target_model_id,
            prepared.target_revision,
            prepared.drafter_model_id,
            prepared.drafter_revision,
            prepared.tokenizer_model_id,
            prepared.tokenizer_revision,
        )
        != (
            prerequisite.target_model_id,
            prerequisite.target_revision,
            prerequisite.drafter_model_id,
            prerequisite.drafter_revision,
            prerequisite.tokenizer_model_id,
            prerequisite.tokenizer_revision,
        )
    ):
        raise ValueError("prepared E1a ChronoBelief launch differs from GPU proof")
    return proof


__all__ = (
    "TRUSTED_SINGLE_OPERATOR_CHRONOBELIEF_GPU_PARITY_PROTOCOL_SHA256",
    "TrustedSingleOperatorChronoBeliefGpuParityProof",
    "build_trusted_single_operator_chronobelief_gpu_parity_proof",
    "load_trusted_single_operator_chronobelief_gpu_parity_proof",
    "publish_trusted_single_operator_chronobelief_gpu_parity_proof",
    "revalidate_trusted_single_operator_chronobelief_for_prepared_launch",
)
