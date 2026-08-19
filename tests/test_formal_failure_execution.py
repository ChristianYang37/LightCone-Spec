from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec.experiments.formal_failure_execution import (
    FORMAL_FAILURE_EXECUTION_BINDING_PROTOCOL_SHA256,
    FORMAL_SINGLE_OPERATOR_E5_FAILURE_EXECUTION_PROTOCOL_SHA256,
    FormalFailureExecutionRebuildInput,
    FormalFailureExecutionSubject,
    FormalSingleOperatorE5FailureExecutionDescriptor,
    VerifiedFormalFailureExecutionBinding,
    current_formal_failure_execution_binding_sha256,
    formal_single_operator_e5_failure_native_identities,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorJsonBinding,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return content_sha256({"label": label})


def _subject() -> FormalFailureExecutionSubject:
    values = {
        "materialized_cell_id": _sha("cell"),
        "serving_execution_binding_sha256": _sha("serving-binding"),
        "serving_execution_plan_sha256": _sha("serving-plan"),
        "scenario": "duplicate_retry",
        "backend": "DFLASH",
        "topology": "tp1_dp1",
        "cohort_count": 4,
        "inventory_sha256": _sha("inventory"),
        "run_nonce_sha256": _sha("run-nonce"),
    }
    assignment = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_e5_failure_assignment",
            "protocol_sha256": FORMAL_FAILURE_EXECUTION_BINDING_PROTOCOL_SHA256,
            **values,
        }
    )
    return FormalFailureExecutionSubject(
        schema_version=1,
        protocol_lock_sha256=_sha("protocol-lock"),
        formal_runtime_authority_manifest_sha256=_sha("runtime-manifest"),
        materialization_receipt_sha256=_sha("materialization"),
        materialized_cell_id=values["materialized_cell_id"],
        serving_execution_binding_sha256=values["serving_execution_binding_sha256"],
        serving_execution_plan_sha256=values["serving_execution_plan_sha256"],
        serving_rank_config_sha256=_sha("rank-config"),
        assignment_sha256=assignment,
        inventory_sha256=values["inventory_sha256"],
        registry_sha256=_sha("registry"),
        backend=values["backend"],
        topology=values["topology"],
        scenario=values["scenario"],
        cohort_count=values["cohort_count"],
        run_nonce_sha256=values["run_nonce_sha256"],
        failure_actuator_authority_sha256=_sha("actuator"),
        failure_reducer_authority_sha256=_sha("reducer"),
        correctness_only=True,
    )


def _binding(tmp_path: Path, name: str) -> CanonicalJsonProofBinding:
    path = (tmp_path / f"{name}.json").resolve()
    publish_canonical_json_no_replace(path, {"kind": "test", "name": name})
    return CanonicalJsonProofBinding.bind(path)


def test_formal_failure_assignment_is_distinct_from_serving_plan() -> None:
    subject = _subject()

    assert subject.assignment_sha256 != subject.serving_execution_plan_sha256
    assert subject.correctness_only is True
    with pytest.raises(ValueError, match="not canonical"):
        replace(subject, scenario="cancellation")
    with pytest.raises(ValueError, match="outside the 264-row matrix"):
        replace(subject, topology="two_replica_tp1_dp2")


def test_formal_failure_verified_binding_cannot_be_directly_constructed() -> None:
    with pytest.raises(TypeError, match="verifier-constructed"):
        VerifiedFormalFailureExecutionBinding(
            subject=_subject(),
            serving_execution=object(),  # type: ignore[arg-type]
            _construction_seal=object(),
        )


def test_formal_failure_rebuild_descriptor_round_trip_and_tamper_rejection() -> None:
    descriptor = FormalFailureExecutionRebuildInput(
        schema_version=1,
        kind="formal_failure_execution_rebuild_input",
        protocol_sha256=FORMAL_FAILURE_EXECUTION_BINDING_PROTOCOL_SHA256,
        subject=_subject(),
        serving_execution_rebuild_input_sha256=_sha("serving-rebuild"),
        expected_failure_execution_binding_sha256=_sha("failure-binding"),
    )
    assert (
        FormalFailureExecutionRebuildInput.from_dict(descriptor.to_dict()) == descriptor
    )

    foreign = descriptor.to_dict()
    foreign["expected_failure_execution_binding_sha256"] = _sha("foreign")
    with pytest.raises(ValueError, match="digest differs"):
        FormalFailureExecutionRebuildInput.from_dict(foreign)

    digest_only = {"rebuild_input_sha256": descriptor.sha256}
    with pytest.raises(ValueError, match="fields differ"):
        FormalFailureExecutionRebuildInput.from_dict(digest_only)


def test_current_e5_failure_descriptor_is_public_one_shot_and_path_bound(
    tmp_path: Path,
) -> None:
    subject = _subject()
    bindings = {
        name: _binding(tmp_path, name)
        for name in (
            "execution-source",
            "prepared-bundle",
            "runtime",
            "inventory",
            "launch",
            "schedule",
        )
    }
    materialization_path = (tmp_path / "materialization.json").resolve()
    publish_canonical_json_no_replace(
        materialization_path,
        {"kind": "test", "name": "materialization"},
    )
    materialization = FormalSingleOperatorJsonBinding.bind(
        materialization_path,
        label="test materialization",
    )
    descriptor = FormalSingleOperatorE5FailureExecutionDescriptor(
        schema_version=1,
        kind="formal_single_operator_e5_failure_execution_descriptor",
        protocol_sha256=(FORMAL_SINGLE_OPERATOR_E5_FAILURE_EXECUTION_PROTOCOL_SHA256),
        execution_source=bindings["execution-source"],
        execution_source_sha256=_sha("execution-source"),
        prepared_launch_bundle=bindings["prepared-bundle"],
        prepared_launch_bundle_sha256=_sha("prepared-bundle"),
        prepared_launch_entry_sha256=_sha("prepared-entry"),
        runtime_authority_manifest=bindings["runtime"],
        materialization=materialization,
        materialization_sha256=_sha("materialization"),
        inventory=bindings["inventory"],
        compile_launch_manifest=bindings["launch"],
        request_schedule_receipt=bindings["schedule"],
        execution_binding_sha256=subject.serving_execution_binding_sha256,
        subject_sha256=_sha("serving-subject"),
        failure_subject=subject,
        expected_failure_execution_binding_sha256=(
            current_formal_failure_execution_binding_sha256(subject)
        ),
        gpu_uuids=("GPU-test-0",),
        attempt_id="attempt-0",
        retry_allowance=0,
        exclusive_timing=True,
        private_output_root=str(tmp_path.resolve()),
    )
    assert (
        FormalSingleOperatorE5FailureExecutionDescriptor.from_dict(descriptor.to_dict())
        == descriptor
    )
    assert descriptor.retry_allowance == 0
    assert descriptor.exclusive_timing is True

    foreign = descriptor.to_dict()
    foreign["gpu_uuids"] = ["GPU-foreign"]
    with pytest.raises(ValueError, match="digest differs"):
        FormalSingleOperatorE5FailureExecutionDescriptor.from_dict(foreign)

    Path(bindings["schedule"].absolute_path).write_text(
        '{"kind":"mutated"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="changed"):
        FormalSingleOperatorE5FailureExecutionDescriptor.from_dict(descriptor.to_dict())


def test_current_e5_failure_native_identities_are_attempt_zero_deterministic() -> None:
    kwargs = {
        "prepared_launch_bundle_sha256": _sha("bundle"),
        "prepared_launch_entry_sha256": _sha("entry"),
        "compile_launch_manifest_sha256": _sha("launch"),
        "request_schedule_sha256": _sha("schedule"),
        "topology_mode": "tp2_dp1",
        "gpu_uuids": ("GPU-0", "GPU-1"),
    }
    first = formal_single_operator_e5_failure_native_identities(**kwargs)
    assert first == formal_single_operator_e5_failure_native_identities(**kwargs)
    assert len(set(first)) == 3
    changed = formal_single_operator_e5_failure_native_identities(
        **{**kwargs, "prepared_launch_entry_sha256": _sha("foreign-entry")}
    )
    assert changed[0] != first[0]
    assert changed[1:] == first[1:]
