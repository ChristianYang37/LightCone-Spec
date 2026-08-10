from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE, __version__
from lightcone_spec.cli.main import main
from lightcone_spec.doctor import _command, doctor_report
from lightcone_spec.experiments.onlinespec import OnlineSpecManifest
from lightcone_spec.orchestration import SpeedStudyManifest
from lightcone_spec.sglang_bridge import verify_patched_checkout
from lightcone_spec.sglang_bridge.launch import _bind_interpreter_tools

ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / "patches" / "sglang"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_package_and_schema_version_are_focused_release() -> None:
    assert __version__ == "0.2.0"


def test_cli_exposes_content_bound_onlinespec_source_verifier(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["verify-onlinespec-source", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--checkout" in output
    assert "--audit" in output
    assert "--output" in output


def test_tracked_speed_manifest_matches_the_registered_protocol() -> None:
    path = ROOT / "manifests" / "speed-study" / "static_tts_l0_v2.json"
    manifest = SpeedStudyManifest.load(path)
    assert manifest == SpeedStudyManifest.default()
    assert Path(f"{path}.sha256").read_text().strip() == manifest.sha256


def test_tracked_onlinespec_manifest_matches_registered_protocol() -> None:
    path = ROOT / "manifests" / "speed-study" / "onlinespec_baseline_v2.json"
    manifest = OnlineSpecManifest.load(path)
    assert manifest == OnlineSpecManifest.default()
    assert Path(f"{path}.sha256").read_text().strip() == manifest.sha256


def test_tracked_onlinespec_source_audit_is_content_bound() -> None:
    from lightcone_spec.experiments.onlinespec import (
        ONLINE_SPEC_CLAIM_SCOPE,
        ONLINE_SPEC_COMMIT,
        ONLINE_SPEC_SOURCE_AUDIT_SHA256,
        ONLINE_SPEC_TREE,
    )

    path = ROOT / "manifests" / "provenance" / "onlinespec_source_audit_v2.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema_version"] == 2
    assert value["commit"] == ONLINE_SPEC_COMMIT
    assert value["tree"] == ONLINE_SPEC_TREE
    assert value["claim_scope"] == ONLINE_SPEC_CLAIM_SCOPE
    assert value["license_status"] == "no-license-file-present-at-audited-commit"
    assert value["license_files"] == []
    assert {row["name"] for row in value["instantiations"]} == {
        "online_lr",
        "opt_hydra",
        "ens_eagle",
    }
    assert {
        "EAGLE/script/EAGLE/eagle-ens.sh",
        "EAGLE/script/EAGLE-3/eagle3-ens.sh",
        "Hydra/script/run.sh",
        "LR/script/reproduce.sh",
    } <= set(value["key_files"])
    assert value["observed_source_mismatches"] == [
        "README_and_shell_entrypoints_invoke_last_chunk_reset_hedge_not_cumulative_ens",
        "README_calls_momentum_optimistic_while_the_paper_uses_a_historical_gradient_hint",
    ]
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == ONLINE_SPEC_SOURCE_AUDIT_SHA256
    assert Path(f"{path}.sha256").read_text().strip() == ONLINE_SPEC_SOURCE_AUDIT_SHA256


def test_patch_manifest_binds_series_files_and_tree() -> None:
    manifest = json.loads((PATCH_ROOT / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["upstream"]["commit"] == PINNED_SGLANG_COMMIT
    assert manifest["expected_tree"] == PINNED_SGLANG_TREE
    series = [
        line.strip()
        for line in (PATCH_ROOT / "series").read_text().splitlines()
        if line.strip()
    ]
    assert series == [entry["file"] for entry in manifest["patches"]]
    assert len(series) == 8
    for entry in manifest["patches"]:
        patch = PATCH_ROOT / entry["file"]
        assert patch.is_file()
        assert sha256(patch) == entry["sha256"]
        assert entry["files"] == sorted(set(entry["files"]))


def test_patch_apply_script_requires_exact_clean_upstream() -> None:
    text = (PATCH_ROOT / "apply.sh").read_text()
    assert PINNED_SGLANG_COMMIT in text
    assert PINNED_SGLANG_TREE in text
    assert "status --porcelain=v1" in text
    assert 'git -C "$CHECKOUT" am' in text


def test_runtime_launcher_rejects_unpatched_upstream_checkout(tmp_path) -> None:
    checkout = tmp_path / "upstream"
    package = checkout / "python" / "sglang"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "unpatched"],
        cwd=checkout,
        check=True,
    )
    with pytest.raises(ValueError, match="tree mismatch"):
        verify_patched_checkout(checkout)


def test_runtime_launcher_exposes_tools_from_its_interpreter(
    monkeypatch, tmp_path
) -> None:
    interpreter_bin = tmp_path / "runtime" / "bin"
    interpreter_bin.mkdir(parents=True)
    interpreter = interpreter_bin / "python"
    interpreter.symlink_to(Path(sys.executable).resolve())
    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(("/usr/bin", str(interpreter_bin), "/bin")),
    )

    _bind_interpreter_tools()
    _bind_interpreter_tools()

    entries = os.environ["PATH"].split(os.pathsep)
    assert entries[0] == str(interpreter_bin.resolve())
    assert entries.count(str(interpreter_bin.resolve())) == 1


def test_doctor_records_actual_source_tree_identity(tmp_path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=checkout,
        check=True,
    )
    (checkout / "source.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=checkout,
        check=True,
    )
    source = doctor_report(checkout)["source_tree"]
    assert source["is_git_checkout"] is True
    assert (
        source["head"]
        == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
    )
    assert source["dirty"] is False
    assert source["pinned_ancestor"] is False
    assert source["patch_commits"] is None


def test_doctor_rejects_failed_command_output(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/bin/fake")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: type(
            "Completed", (), {"returncode": 1, "stdout": "driver error"}
        )(),
    )
    assert _command(["nvidia-smi"]) is None


@pytest.mark.integration
def test_patchset_applies_only_to_explicit_checkout() -> None:
    checkout_value = os.environ.get("LIGHTCONE_SGLANG_UPSTREAM")
    if not checkout_value:
        pytest.skip("set LIGHTCONE_SGLANG_UPSTREAM to an explicit clean checkout")
    checkout = Path(checkout_value).resolve()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_sglang_patchset.py"),
            "--upstream-checkout",
            str(checkout),
            "--compile-only",
        ],
        check=True,
    )


def test_cli_builds_and_validates_immutable_manifest(tmp_path, capsys) -> None:
    output = tmp_path / "study.json"
    assert main(["build-speed-study", "--output", str(output)]) == 0
    manifest = SpeedStudyManifest.load(output)
    assert manifest.gpu_evidence == "UNMEASURED"
    assert manifest.methods == ("static", "tts", "naive_async")
    assert capsys.readouterr().out.strip() == manifest.sha256


def test_cli_help_contains_only_focused_workflow(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "run-confirmation" in output
    assert "collect-speed-study" in output
    assert "run-controlled-slice" in output
    assert "render-static-load-runtime" in output
    assert "advance-tuning-stage" in output
    assert "run-natural-slice" in output
    assert "build-profiler-plan" in output
    assert "select-speed-config" in output
    assert "select-anchor-config" in output
    assert "attest-speed-study" in output
    assert "run-onlinespec-tuning-slice" in output
    assert "advance-onlinespec-tuning-stage" in output
    assert "run-onlinespec-confirmation" in output
    assert "attest-onlinespec-study" in output
