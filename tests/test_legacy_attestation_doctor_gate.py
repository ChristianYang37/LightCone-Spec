from __future__ import annotations

import argparse
import copy
import importlib
import json

import pytest

from lightcone_spec import (
    PINNED_SGLANG_COMMIT,
    PINNED_SGLANG_PATCH_COUNT,
    PINNED_SGLANG_TREE,
)
from lightcone_spec.experiments.onlinespec import OnlineSpecManifest
from lightcone_spec.orchestration import PreliminarySpeedStudyManifest

cli = importlib.import_module("lightcone_spec.cli.main")


def _passing_doctor() -> dict:
    manifest_sha256 = "a" * 64
    checks = {name: {"status": "PASS"} for name in cli._ATTESTATION_DOCTOR_CHECKS}
    return {
        "schema_version": 1,
        "status": "PASS",
        "readiness": {
            "status": "PASS",
            "pass_count": len(checks),
            "fail_count": 0,
            "unknown_count": 0,
        },
        "runtime_manifest": {
            "valid": True,
            "sha256": manifest_sha256,
            "sidecar_sha256": manifest_sha256,
            "error": None,
        },
        "checks": checks,
        "roots": {
            "project": "/runtime/lightcone-spec",
            "patched_sglang": "/runtime/sglang",
            "distinct": True,
        },
        "source_tree": {
            "path": "/runtime/sglang",
            "is_git_checkout": True,
            "root_matches_toplevel": True,
            "head": "b" * 40,
            "tree": PINNED_SGLANG_TREE,
            "dirty": False,
            "pinned_ancestor": True,
            "patch_commits": PINNED_SGLANG_PATCH_COUNT,
        },
        "commands": {"nvidia_smi": "two exact GPU inventory rows"},
        "gpu": {"gpu_pool_visible": True, "visible_gpu_count": 2},
        "compatibility": {
            "status": "PASS",
            "python_supported": True,
            "single_node_only": True,
            "multi_node_supported": False,
            "sglang_commit": PINNED_SGLANG_COMMIT,
            "sglang_tree": PINNED_SGLANG_TREE,
            "patch_count": PINNED_SGLANG_PATCH_COUNT,
            "manifest_sha256": manifest_sha256,
        },
    }


def _replace(report: dict, path: tuple[str, ...], value: object) -> dict:
    changed = copy.deepcopy(report)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return changed


def test_complete_pass_doctor_is_accepted_for_legacy_attestation() -> None:
    report = _passing_doctor()
    assert cli._validate_attestation_doctor(report, label="GPU") is report


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("schema_version",), 2, "schema-v1"),
        (("status",), "FAIL", "top-level"),
        (("readiness", "status"), "UNKNOWN", "readiness.status"),
        (("compatibility", "status"), "FAIL", "compatibility.status"),
        (("runtime_manifest", "valid"), False, "manifest is not valid"),
        (("runtime_manifest", "sha256"), "A" * 64, "digests"),
        (("runtime_manifest", "sidecar_sha256"), "c" * 64, "digests"),
        (("compatibility", "manifest_sha256"), "c" * 64, "digests"),
        (("readiness", "unknown_count"), 1, "counters"),
        (("roots", "distinct"), False, "roots must be distinct"),
        (("source_tree", "head"), "B" * 40, "source-tree identity"),
        (("source_tree", "tree"), "c" * 40, "source-tree identity"),
        (("source_tree", "dirty"), True, "source-tree identity"),
        (("commands", "nvidia_smi"), "", "nvidia-smi"),
        (("gpu", "gpu_pool_visible"), False, "GPU-pool"),
        (("gpu", "visible_gpu_count"), 0, "GPU-pool"),
    ),
)
def test_legacy_attestation_rejects_incomplete_doctor_contract(
    path: tuple[str, ...], value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cli._validate_attestation_doctor(
            _replace(_passing_doctor(), path, value), label="GPU"
        )


def test_legacy_attestation_requires_every_named_doctor_check() -> None:
    report = _passing_doctor()
    del report["checks"]["driver"]
    report["readiness"]["pass_count"] -= 1
    with pytest.raises(ValueError, match="checks are missing: driver"):
        cli._validate_attestation_doctor(report, label="GPU")


@pytest.mark.parametrize("status", ("FAIL", "UNKNOWN"))
def test_legacy_attestation_rejects_any_nonpass_doctor_check(status: str) -> None:
    report = _passing_doctor()
    report["checks"]["gpu_topology"]["status"] = status
    report["readiness"]["pass_count"] -= 1
    report["readiness"][f"{status.lower()}_count"] = 1
    with pytest.raises(ValueError, match="every doctor check must be PASS"):
        cli._validate_attestation_doctor(report, label="GPU")


def _preliminary_manifest(tmp_path) -> str:
    path = tmp_path / "preliminary-manifest.json"
    PreliminarySpeedStudyManifest.default().write(path)
    return str(path)


def test_speed_attester_is_categorically_preliminary(capsys, tmp_path) -> None:
    output = tmp_path / "attestation-decision.json"
    args = argparse.Namespace(
        manifest=_preliminary_manifest(tmp_path),
        output=str(output),
    )
    assert cli._attest(args) == 42
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["status"] == "PRELIMINARY_DIAGNOSTIC_ONLY"
    assert decision["formal_execution_authorized"] is False
    assert decision["industrial_evidence_receipt"] is None
    assert capsys.readouterr().out.strip() == cli._canonical_sha256(decision)


def test_matching_cpu_doctor_cannot_mint_legacy_measured_attestation(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(cli, "_TRUSTED_HARDWARE_ATTESTER_ID", "future-attester")
    output = tmp_path / "attestation.json"
    args = argparse.Namespace(
        manifest=_preliminary_manifest(tmp_path),
        output=str(output),
    )
    assert cli._attest(args) == 42
    assert "MEASURED" not in output.read_text(encoding="utf-8")


def test_hand_authored_legacy_attestation_cannot_enter_analysis(
    tmp_path,
) -> None:
    output = tmp_path / "analysis.json"
    args = argparse.Namespace(
        manifest=_preliminary_manifest(tmp_path),
        attestation="hand-authored-attestation.json",
        output=str(output),
    )
    with pytest.raises(ValueError, match="cannot consume any attestation"):
        cli._analyze(args)
    assert not output.exists()


def _onlinespec_manifest(tmp_path) -> str:
    path = tmp_path / "onlinespec-manifest.json"
    OnlineSpecManifest.default().write(path)
    return str(path)


def test_onlinespec_attester_is_categorically_diagnostic(tmp_path) -> None:
    output = tmp_path / "attestation.json"
    args = argparse.Namespace(
        manifest=_onlinespec_manifest(tmp_path),
        output=str(output),
    )
    assert cli._attest_onlinespec(args) == 42
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["status"] == "PRELIMINARY_DIAGNOSTIC_ONLY"
    assert decision["formal_execution_authorized"] is False


def test_matching_cpu_doctor_cannot_mint_onlinespec_measured_attestation(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(cli, "_TRUSTED_HARDWARE_ATTESTER_ID", "future-attester")
    output = tmp_path / "attestation.json"
    args = argparse.Namespace(
        manifest=_onlinespec_manifest(tmp_path),
        output=str(output),
    )
    assert cli._attest_onlinespec(args) == 42
    assert "MEASURED" not in output.read_text(encoding="utf-8")


def test_hand_authored_onlinespec_attestation_cannot_enter_analysis(tmp_path) -> None:
    output = tmp_path / "analysis.json"
    args = argparse.Namespace(
        manifest=_onlinespec_manifest(tmp_path),
        attestation="hand-authored-attestation.json",
        output=str(output),
    )
    with pytest.raises(ValueError, match="cannot consume attestation"):
        cli._analyze_onlinespec(args)
    assert not output.exists()
