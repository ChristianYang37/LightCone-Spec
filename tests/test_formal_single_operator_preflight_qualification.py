from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from lightcone_spec.experiments.formal_preflight_inputs import _run_config
from lightcone_spec.experiments.formal_single_operator_preflight_qualification import (
    TRUSTED_PREFLIGHT_QUALIFICATION_SUITES,
    TrustedQualificationDispatchAuthority,
    TrustedQualificationLaunchEntry,
    _qualification_config,
    materialize_formal_single_operator_preflight_qualification_plans,
    publish_formal_single_operator_preflight_qualification_launch_index,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding


def _binding(path: Path, kind: str) -> CanonicalJsonProofBinding:
    path.write_text(
        json.dumps({"kind": kind}, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return CanonicalJsonProofBinding.bind(path.resolve())


def _base_config():
    return _run_config(
        sampling_profile=SamplingProfile(),
        target_revision="1" * 40,
        drafter_revision="2" * 40,
        gpu_uuids=("GPU-0",),
        runtime_qualification_sha256="a" * 64,
    )


def test_qualification_launch_producer_and_plan_materializer_are_path_only() -> None:
    assert tuple(
        inspect.signature(
            publish_formal_single_operator_preflight_qualification_launch_index
        ).parameters
    ) == (
        "protocol_lock_path",
        "content_source_path",
        "inventory_path",
        "doctor_report_path",
        "exactness_assignment_path",
        "base_tp1_launch_path",
        "base_tp2_launch_path",
        "output_root",
    )
    assert tuple(
        inspect.signature(
            materialize_formal_single_operator_preflight_qualification_plans
        ).parameters
    ) == ("qualification_launch_index_path", "output_root")
    assert TRUSTED_PREFLIGHT_QUALIFICATION_SUITES == (
        "chronobelief_gpu_parity",
        "dspark_dp2",
        "dspark_tp1",
        "dspark_tp2",
        "tp1_dp2",
        "tp2_dp1",
    )


def test_dspark_config_uses_its_typed_drafter_and_dispatch_authority() -> None:
    base = _base_config()
    receipt = "b" * 64
    config = _qualification_config(
        base=base,
        suite_id="dspark_tp2",
        gpu_uuids=("GPU-0", "GPU-1"),
        distributed_receipt_sha256=receipt,
        drafter_model_id="source-owned/dspark-drafter",
        drafter_revision="3" * 40,
        draft_depth=7,
    )

    assert config.model.algorithm == "DSPARK"
    assert config.model.drafter == "source-owned/dspark-drafter"
    assert config.model.drafter_revision == "3" * 40
    assert config.model.draft_depth == 7
    assert config.runtime.tensor_parallel_size == 2
    assert config.runtime.data_parallel_size == 1
    assert config.runtime.distributed_capability_receipt_sha256 == receipt
    assert base.model.algorithm == "DFLASH"
    assert base.model.drafter != config.model.drafter


def test_single_rank_qualification_does_not_smuggle_distributed_receipt() -> None:
    config = _qualification_config(
        base=_base_config(),
        suite_id="dspark_tp1",
        gpu_uuids=("GPU-0",),
        distributed_receipt_sha256="c" * 64,
        drafter_model_id="source-owned/dspark-drafter",
        drafter_revision="3" * 40,
        draft_depth=7,
    )

    assert config.runtime.topology_mode == "tp1_dp1"
    assert config.runtime.distributed_capability_receipt_sha256 is None


def test_launch_entry_rejects_dflash_identity_for_dspark_suite(
    tmp_path: Path,
) -> None:
    dispatch = _binding(tmp_path / "dispatch.json", "dispatch")
    launch = _binding(tmp_path / "launch.json", "launch")
    valid = TrustedQualificationLaunchEntry(
        suite_id="dspark_tp1",
        backend="DSPARK",
        topology_mode="tp1_dp1",
        gpu_uuids=("GPU-0",),
        dispatch_authority=dispatch,
        launch_manifest=launch,
    )
    assert TrustedQualificationLaunchEntry.from_dict(valid.to_dict()) == valid
    with pytest.raises(ValueError, match="launch entry differs"):
        TrustedQualificationLaunchEntry(
            suite_id="dspark_tp1",
            backend="DFLASH",
            topology_mode="tp1_dp1",
            gpu_uuids=("GPU-0",),
            dispatch_authority=dispatch,
            launch_manifest=launch,
        )


def test_dispatch_schema_rejects_legacy_none_authority_field() -> None:
    value = {
        name: None
        for name in TrustedQualificationDispatchAuthority.__dataclass_fields__
    }
    value["native_runtime_qualification_authority_sha256"] = None
    with pytest.raises(ValueError, match="dispatch fields differ"):
        TrustedQualificationDispatchAuthority.from_dict(value)
