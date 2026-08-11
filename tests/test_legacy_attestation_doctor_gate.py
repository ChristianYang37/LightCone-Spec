from __future__ import annotations

import argparse
import copy
import importlib
import json
from types import SimpleNamespace

import pytest

from lightcone_spec import (
    PINNED_SGLANG_COMMIT,
    PINNED_SGLANG_PATCH_COUNT,
    PINNED_SGLANG_TREE,
)

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
        "gpu": {"two_gpu_visible": True},
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
        (("gpu", "two_gpu_visible"), False, "two-GPU"),
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


def _patch_common_attestation_inputs(monkeypatch) -> None:
    manifest = SimpleNamespace(
        sha256="1" * 64,
        methods=("static", "tts", "l0"),
        confirmation_repetitions=8,
        formal_context_start=16384,
        safe_context_limit=40960,
    )
    selection = SimpleNamespace(sha256="2" * 64, model_lock_sha256="3" * 64)
    lock = SimpleNamespace(
        sha256="3" * 64,
        models=(
            SimpleNamespace(model_id="Qwen/Qwen3-8B", revision="4" * 40),
            SimpleNamespace(model_id="z-lab/Qwen3-8B-DFlash-b16", revision="5" * 40),
        ),
    )
    monkeypatch.setattr(
        cli, "SpeedStudyManifest", SimpleNamespace(load=lambda _path: manifest)
    )
    monkeypatch.setattr(
        cli, "SelectionArtifact", SimpleNamespace(load=lambda _path: selection)
    )
    monkeypatch.setattr(cli, "ModelLock", SimpleNamespace(load=lambda _path: lock))
    monkeypatch.setattr(cli, "_assert_selection_study", lambda *_args: None)
    monkeypatch.setattr(cli, "_load_formal_table", lambda *_args, **_kwargs: None)


def test_speed_attester_uses_complete_doctor_gate(monkeypatch, tmp_path) -> None:
    _patch_common_attestation_inputs(monkeypatch)
    doctor = tmp_path / "doctor.json"
    doctor.write_text(
        json.dumps(_replace(_passing_doctor(), ("status",), "FAIL")),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        manifest="manifest.json",
        selection="selection.json",
        model_lock="model-lock.json",
        performance="performance.parquet",
        doctor_json=str(doctor),
        output=str(tmp_path / "attestation.json"),
    )
    with pytest.raises(ValueError, match="top-level readiness is not PASS"):
        cli._attest(args)


def test_matching_cpu_doctor_cannot_mint_legacy_measured_attestation(
    monkeypatch, tmp_path
) -> None:
    _patch_common_attestation_inputs(monkeypatch)
    doctor = tmp_path / "doctor.json"
    doctor.write_text(json.dumps(_passing_doctor()), encoding="utf-8")
    output = tmp_path / "attestation.json"
    args = argparse.Namespace(
        manifest="manifest.json",
        selection="selection.json",
        model_lock="model-lock.json",
        performance="performance.parquet",
        doctor_json=str(doctor),
        output=str(output),
    )
    with pytest.raises(RuntimeError, match="trusted_hardware_attester_unavailable"):
        cli._attest(args)
    assert not output.exists()


def test_hand_authored_legacy_attestation_cannot_enter_analysis(
    monkeypatch, tmp_path
) -> None:
    _patch_common_attestation_inputs(monkeypatch)
    monkeypatch.setattr(
        cli.GpuEvidenceAttestation,
        "load",
        lambda _path: pytest.fail("untrusted attestation loader was reached"),
    )
    output = tmp_path / "analysis.json"
    args = argparse.Namespace(
        manifest="manifest.json",
        selection="selection.json",
        model_lock="model-lock.json",
        performance="performance.parquet",
        attestation="hand-authored-attestation.json",
        bootstrap_seed=1,
        output=str(output),
    )
    with pytest.raises(RuntimeError, match="trusted_hardware_attester_unavailable"):
        cli._analyze(args)
    assert not output.exists()


def test_onlinespec_attester_uses_complete_doctor_gate(monkeypatch, tmp_path) -> None:
    manifest = SimpleNamespace(
        sha256="1" * 64,
        methods=("static", "online_lr", "opt_hydra", "ens_eagle"),
        confirmation_repetitions=8,
    )
    selection = SimpleNamespace(sha256="2" * 64)
    lock = SimpleNamespace(sha256="3" * 64)
    monkeypatch.setattr(
        cli, "OnlineSpecManifest", SimpleNamespace(load=lambda _path: manifest)
    )
    monkeypatch.setattr(
        cli, "OnlineSpecSelection", SimpleNamespace(load=lambda _path: selection)
    )
    monkeypatch.setattr(cli, "ModelLock", SimpleNamespace(load=lambda _path: lock))
    monkeypatch.setattr(cli, "_assert_onlinespec_study", lambda *_args: None)
    monkeypatch.setattr(cli, "_onlinespec_table", lambda *_args, **_kwargs: None)
    doctor = tmp_path / "doctor.json"
    doctor.write_text(
        json.dumps(_replace(_passing_doctor(), ("compatibility", "status"), "FAIL")),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        manifest="manifest.json",
        selection="selection.json",
        model_lock="model-lock.json",
        performance="performance.parquet",
        doctor_json=str(doctor),
        output=str(tmp_path / "attestation.json"),
    )
    with pytest.raises(ValueError, match="compatibility.status is not PASS"):
        cli._attest_onlinespec(args)


def test_matching_cpu_doctor_cannot_mint_onlinespec_measured_attestation(
    monkeypatch, tmp_path
) -> None:
    manifest = SimpleNamespace(
        sha256="1" * 64,
        methods=("static", "online_lr", "opt_hydra", "ens_eagle"),
        confirmation_repetitions=8,
    )
    selection = SimpleNamespace(sha256="2" * 64)
    lock = SimpleNamespace(sha256="3" * 64)
    monkeypatch.setattr(
        cli, "OnlineSpecManifest", SimpleNamespace(load=lambda _path: manifest)
    )
    monkeypatch.setattr(
        cli, "OnlineSpecSelection", SimpleNamespace(load=lambda _path: selection)
    )
    monkeypatch.setattr(cli, "ModelLock", SimpleNamespace(load=lambda _path: lock))
    monkeypatch.setattr(cli, "_assert_onlinespec_study", lambda *_args: None)
    monkeypatch.setattr(cli, "_onlinespec_table", lambda *_args, **_kwargs: None)
    doctor = tmp_path / "doctor.json"
    doctor.write_text(json.dumps(_passing_doctor()), encoding="utf-8")
    output = tmp_path / "attestation.json"
    args = argparse.Namespace(
        manifest="manifest.json",
        selection="selection.json",
        model_lock="model-lock.json",
        performance="performance.parquet",
        doctor_json=str(doctor),
        output=str(output),
    )
    with pytest.raises(RuntimeError, match="trusted_hardware_attester_unavailable"):
        cli._attest_onlinespec(args)
    assert not output.exists()


def test_hand_authored_onlinespec_attestation_cannot_enter_analysis(
    monkeypatch, tmp_path
) -> None:
    manifest = SimpleNamespace(sha256="1" * 64)
    selection = SimpleNamespace(sha256="2" * 64)
    lock = SimpleNamespace(sha256="3" * 64)
    monkeypatch.setattr(
        cli, "OnlineSpecManifest", SimpleNamespace(load=lambda _path: manifest)
    )
    monkeypatch.setattr(
        cli, "OnlineSpecSelection", SimpleNamespace(load=lambda _path: selection)
    )
    monkeypatch.setattr(cli, "ModelLock", SimpleNamespace(load=lambda _path: lock))
    monkeypatch.setattr(cli, "_assert_onlinespec_study", lambda *_args: None)
    monkeypatch.setattr(cli, "_onlinespec_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli.OnlineSpecGpuAttestation,
        "load",
        lambda _path: pytest.fail("untrusted attestation loader was reached"),
    )
    output = tmp_path / "analysis.json"
    args = argparse.Namespace(
        manifest="manifest.json",
        selection="selection.json",
        model_lock="model-lock.json",
        performance="performance.parquet",
        attestation="hand-authored-attestation.json",
        bootstrap_seed=1,
        output=str(output),
    )
    with pytest.raises(RuntimeError, match="trusted_hardware_attester_unavailable"):
        cli._analyze_onlinespec(args)
    assert not output.exists()
