from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_industrial_completion_contract import (
    _rebind_generic_activation,
    _serving_bundle,
    _write_bound,
)
from test_native_terminal_provider import _release_policy

from lightcone_spec.experiments.completion_authority import (
    CompletedCellAuthority,
    CompletionAuthorityUnavailableError,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuDispatchExecutionContext,
    InterferenceEnvelope,
)
from lightcone_spec.experiments.registry import content_sha256
from lightcone_spec.runtime.attestation import NO_TRUSTED_ATTESTERS


def _authority(bundle: dict[str, object]) -> CompletedCellAuthority:
    completed_path = bundle["completed_path"]
    assert isinstance(completed_path, Path)
    return CompletedCellAuthority.from_path(
        completed_path,
        registry=bundle["registry"],
        inventory=bundle["inventory"],
        trusted_attester_policy=NO_TRUSTED_ATTESTERS,
        direct_dependency_receipt=bundle["direct_dependency_receipt"],
        activation_artifact=bundle["activation"],
    )


def test_cpu_measured_rows_cannot_mint_completed_ids(tmp_path: Path) -> None:
    authority = _authority(_serving_bundle(tmp_path))

    with pytest.raises(
        CompletionAuthorityUnavailableError,
        match="trusted_hardware_attester_unavailable",
    ):
        authority.derive_completed_cell_ids()


def test_legacy_missing_writer_policy_cannot_mint_completed_ids(
    tmp_path: Path,
) -> None:
    bundle = _serving_bundle(tmp_path)
    completed = copy.deepcopy(bundle["completed"])
    measured = next(row for row in completed["rows"] if row["status"] == "MEASURED")
    root = Path(measured["evidence_root"])
    terminal_path = root / f"{measured['run_id']}.rank0.complete.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    prepared_path = root / terminal["prepared_receipt_name"]
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    checkpoint_path = root / prepared["checkpoint"]["name"]
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    def write_canonical(path: Path, value: dict[str, object]) -> bytes:
        body = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        path.write_bytes(body)
        return body

    for value in (checkpoint, prepared, terminal):
        value.pop("writer_policy", None)
        value.pop("writer_policy_sha256", None)
    checkpoint_body = write_canonical(checkpoint_path, checkpoint)
    prepared["checkpoint"]["sha256"] = hashlib.sha256(checkpoint_body).hexdigest()
    prepared["checkpoint"]["size"] = len(checkpoint_body)
    terminal["checkpoint"] = copy.deepcopy(prepared["checkpoint"])
    prepared_body = write_canonical(prepared_path, prepared)
    prepared_sha256 = hashlib.sha256(prepared_body).hexdigest()
    terminal["prepared_receipt_sha256"] = prepared_sha256
    terminal["prepared_receipt_size"] = len(prepared_body)
    observation_path = Path(measured["budget_observation_path"])
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    observation["terminal_evidence_sha256"] = prepared_sha256
    observation_sha256 = content_sha256(
        {
            "schema_version": observation["schema_version"],
            "budget": observation["budget"],
            "observed_component_ms": observation["observed_component_ms"],
            "measured_gpu_ms": observation["measured_gpu_ms"],
            "fixed_instance_billed_gpu_ms": observation["fixed_instance_billed_gpu_ms"],
            "terminal_evidence_sha256": prepared_sha256,
        }
    )
    observation["budget_observation_sha256"] = observation_sha256
    observation_body = write_canonical(observation_path, observation)
    observation_sidecar = Path(f"{observation_path}.sha256")
    sidecar_body = f"{observation_sha256}\n".encode("ascii")
    observation_sidecar.write_bytes(sidecar_body)
    terminal["budget_observation"].update(
        {
            "receipt_sha256": hashlib.sha256(observation_body).hexdigest(),
            "receipt_size": len(observation_body),
            "sidecar_sha256": hashlib.sha256(sidecar_body).hexdigest(),
            "sidecar_size": len(sidecar_body),
            "budget_observation_sha256": observation_sha256,
        }
    )
    terminal_body = write_canonical(terminal_path, terminal)
    measured["terminal_receipt_sha256"] = hashlib.sha256(terminal_body).hexdigest()
    measured["budget_observation_sha256"] = observation_sha256
    completed_path = bundle["completed_path"]
    assert isinstance(completed_path, Path)
    _write_bound(completed_path, completed)

    with pytest.raises(ValueError, match="release EvidenceWriterPolicy"):
        _authority(bundle).derive_completed_cell_ids()


def test_caller_selected_release_ready_policy_is_not_a_trust_root(
    tmp_path: Path,
) -> None:
    bundle = _serving_bundle(tmp_path)
    completed_path = bundle["completed_path"]
    assert isinstance(completed_path, Path)
    caller_policy = _release_policy(Ed25519PrivateKey.generate())

    with pytest.raises(ValueError, match="caller-selected trust roots"):
        CompletedCellAuthority.from_path(
            completed_path,
            registry=bundle["registry"],
            inventory=bundle["inventory"],
            trusted_attester_policy=caller_policy,
            direct_dependency_receipt=bundle["direct_dependency_receipt"],
            activation_artifact=bundle["activation"],
        )


def test_execution_context_requires_budget_plan_before_completion_replay(
    tmp_path: Path,
) -> None:
    bundle = _serving_bundle(tmp_path)
    authority = _authority(bundle)
    with pytest.raises(TypeError, match="budget_plan"):
        GpuDispatchExecutionContext(
            registry=bundle["registry"],
            inventory=bundle["inventory"],
            interference_envelope=InterferenceEnvelope.serial(
                source_receipt_sha256=content_sha256("completion-authority-test")
            ),
            budgets=(),
            completion_authorities=(authority,),
        )


def test_completed_artifact_and_sidecar_are_reopened_on_every_derive(
    tmp_path: Path,
) -> None:
    bundle = _serving_bundle(tmp_path)
    authority = _authority(bundle)
    completed_path = bundle["completed_path"]
    assert isinstance(completed_path, Path)

    completed = copy.deepcopy(bundle["completed"])
    completed["rows"][0]["status"] = "BLOCKED"
    _write_bound(completed_path, completed)

    with pytest.raises(RuntimeError, match="artifact or sidecar changed"):
        authority.derive_completed_cell_ids()


@pytest.mark.parametrize(
    "field",
    (
        "budget_plan_sha256",
        "capacity_authority_sha256",
        "budget_materialization_authority_sha256",
    ),
)
def test_completion_rejects_tampered_launch_budget_authority(
    tmp_path: Path,
    field: str,
) -> None:
    bundle = _serving_bundle(tmp_path)
    completed_path = bundle["completed_path"]
    assert isinstance(completed_path, Path)
    tampered = copy.deepcopy(bundle["completed"])
    measured_cell_id = next(
        row["cell_id"] for row in tampered["rows"] if row["status"] == "MEASURED"
    )
    contract = next(
        row
        for row in tampered["split_contract"]["cells"]
        if row["cell_id"] == measured_cell_id
    )
    contract["physical_assignment"][field] = content_sha256(
        {"tampered-launch-authority": field}
    )
    _write_bound(completed_path, tampered)

    with pytest.raises(ValueError, match="locked-split identity mismatch"):
        _authority(bundle).derive_completed_cell_ids()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "fields differ from schema"),
        ("legacy", "identity mismatch"),
        ("billing", "billing mismatch"),
    ),
)
def test_raw_completion_parser_requires_physical_assignment_schema3(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    bundle = _serving_bundle(tmp_path)
    tampered = copy.deepcopy(bundle["completed"])
    split = tampered["split_contract"]
    assignment = split["cells"][0]["physical_assignment"]
    if mutation == "missing":
        assignment.pop("budget_materialization_authority_sha256")
    elif mutation == "legacy":
        assignment["schema_version"] = 2
    else:
        assignment["fixed_instance_billing_semantics"] = "per_assigned_gpu"
    tampered["split_sha256"] = content_sha256(split)
    activation = _rebind_generic_activation(bundle, tampered)
    completed_path = tmp_path / f"raw-completion-physical-{mutation}.json"
    _write_bound(completed_path, tampered)
    bundle["completed_path"] = completed_path
    bundle["activation"] = activation

    with pytest.raises(ValueError, match=message):
        _authority(bundle).derive_completed_cell_ids()


def test_forged_e5_stage_and_terminal_receipt_swap_fail_before_trust(
    tmp_path: Path,
) -> None:
    forged_bundle = _serving_bundle(tmp_path / "forged-stage")
    forged = copy.deepcopy(forged_bundle["completed"])
    forged["experiment"] = "E5"
    forged_path = tmp_path / "forged-stage.json"
    _write_bound(forged_path, forged)
    forged_bundle["completed_path"] = forged_path
    with pytest.raises(ValueError, match="activation identity/lineage mismatch"):
        _authority(forged_bundle).derive_completed_cell_ids()

    swapped_bundle = _serving_bundle(tmp_path / "receipt-swap")
    swapped = copy.deepcopy(swapped_bundle["completed"])
    measured = next(row for row in swapped["rows"] if row["status"] == "MEASURED")
    measured["terminal_receipt_sha256"] = "0" * 64
    swapped_path = tmp_path / "receipt-swap.json"
    _write_bound(swapped_path, swapped)
    swapped_bundle["completed_path"] = swapped_path
    with pytest.raises(ValueError, match="final receipt digest mismatch"):
        _authority(swapped_bundle).derive_completed_cell_ids()


def test_native_terminal_tamper_is_detected_from_raw_file(tmp_path: Path) -> None:
    bundle = _serving_bundle(tmp_path)
    authority = _authority(bundle)
    completed = bundle["completed"]
    assert isinstance(completed, dict)
    measured = next(row for row in completed["rows"] if row["status"] == "MEASURED")
    evidence_root = Path(measured["evidence_root"])
    (native_path,) = tuple(evidence_root.glob("*.native-terminal.json"))
    value = json.loads(native_path.read_text(encoding="utf-8"))
    value["terminal_sha256"] = "f" * 64
    native_path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="native terminal artifact content binding"):
        authority.derive_completed_cell_ids()
