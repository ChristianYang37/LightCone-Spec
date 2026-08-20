from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lightcone_spec import (
    PINNED_SGLANG_COMMIT,
    PINNED_SGLANG_PATCH_COUNT,
    PINNED_SGLANG_TREE,
    __version__,
)
from lightcone_spec.cli.main import main
from lightcone_spec.doctor import _command, doctor_report
from lightcone_spec.experiments.onlinespec import OnlineSpecManifest
from lightcone_spec.orchestration import PreliminarySpeedStudyManifest
from lightcone_spec.sglang_bridge import verify_patched_checkout
from lightcone_spec.sglang_bridge.launch import _bind_interpreter_tools

ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / "patches" / "sglang"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_package_and_schema_version_are_focused_release() -> None:
    assert __version__ == "0.3.0"


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
    manifest = PreliminarySpeedStudyManifest.load(path)
    assert manifest == PreliminarySpeedStudyManifest.default()
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
    assert len(series) == PINNED_SGLANG_PATCH_COUNT
    assert sorted(path.name for path in PATCH_ROOT.glob("*.patch")) == series
    for entry in manifest["patches"]:
        patch = PATCH_ROOT / entry["file"]
        assert patch.is_file()
        assert sha256(patch) == entry["sha256"]
        assert entry["files"] == sorted(set(entry["files"]))
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    assert (PATCH_ROOT / "manifest.json.sha256").read_text() == (
        f"{hashlib.sha256(canonical).hexdigest()}\n"
    )


def test_latest_patch_binds_request_scoped_source_point_reset() -> None:
    manifest = json.loads((PATCH_ROOT / "manifest.json").read_text())
    latest = manifest["patches"][-1]
    assert latest == {
        "file": "0008-fix-spec-isolate-request-scoped-adaptation-state.patch",
        "sha256": ("0c4db4f8798645c0ba65e97031030fb5e891d15f63cd75105fc1e1656c1a2874"),
        "files": [
            "python/sglang/srt/entrypoints/http_server.py",
            "python/sglang/srt/managers/scheduler.py",
            "python/sglang/srt/managers/scheduler_components/batch_result_processor.py",
            "python/sglang/srt/speculative/dflash_online_adaptation.py",
            "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py",
            "python/sglang/srt/speculative/formal_gang_serving.py",
            "python/sglang/srt/speculative/native_backend_online_adaptation.py",
            "python/sglang/srt/speculative/native_runtime_release.py",
            "python/sglang/srt/speculative/online_adaptation_config.py",
            "python/sglang/srt/speculative/online_adaptation_runtime.py",
            "python/sglang/srt/speculative/terminal_speculative_evidence.py",
            "test/registered/unit/managers/test_scheduler_request_scoped_admission.py",
            "test/registered/unit/spec/test_dspark_online_adaptation_contract.py",
            "test/registered/unit/spec/test_eagle3_online_adaptation_contract.py",
            "test/registered/unit/spec/test_formal_gang_serving.py",
            "test/registered/unit/spec/test_native_runtime_release.py",
            "test/registered/unit/spec/test_online_adaptation_protocol.py",
            "test/registered/unit/spec/test_terminal_speculative_evidence.py",
        ],
    }
    patch = (PATCH_ROOT / latest["file"]).read_text()
    assert '"serialized_native_scheduler_v1"' in patch
    assert "REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256" in patch
    assert "test_finish_and_abort_restore_identical_numeric_source_point" in patch
    assert "test_eleven_thousand_request_archives_reuse_slots_without_drop" in patch
    assert "test_real_spawn_reimports_and_restores_worker_proof_and_hook" in patch
    assert "test_chunked_abort_resets_exactly_once_before_release" in patch
    assert "test_decode_oom_sticky_disables_and_resets_exactly_once" in patch

    metrics_entry = next(
        entry
        for entry in manifest["patches"]
        if entry["file"] == "0006-fix-metrics-handle-target-only-draft-width.patch"
    )
    metrics = (PATCH_ROOT / metrics_entry["file"]).read_text()
    assert "def _terminal_verified_drafts(self, target_calls: int) -> int:" in metrics
    assert "test_speculative_metrics_reject_a_missing_draft_width" in metrics

    verifier = (ROOT / "scripts" / "verify_sglang_patchset.py").read_text()
    assert "test_terminal_target_metrics.py" in verifier
    assert "reverse removal retained the patch-0008 contract" in verifier
    assert "_verify_gpu_qualification_collection" in verifier
    assert "len(node_ids) != 96 or len(set(node_ids)) != 96" in verifier


def test_patch_exports_exact_official_sglang_output_token_ids() -> None:
    manifest = json.loads((PATCH_ROOT / "manifest.json").read_text())
    files = manifest["patches"][0]["files"]
    assert "python/sglang/benchmark/serving.py" in files
    assert "test/registered/unit/benchmark/test_serving_output_token_ids.py" in files
    patch = (PATCH_ROOT / manifest["patches"][0]["file"]).read_text()
    assert "generated_token_ids: Optional[List[int]] = None" in patch
    assert "_merge_sglang_generated_token_ids" in patch
    assert 'data["output_ids"]' in patch


def test_patch_binds_terminal_accounting_seed_and_role_publication_contracts() -> None:
    manifest = json.loads((PATCH_ROOT / "manifest.json").read_text())
    patch = (PATCH_ROOT / manifest["patches"][0]["file"]).read_text()
    assert '"enabled": bool(server_args.speculative_speed_study_metrics)' in patch
    assert 'allocation_free = method in {"target_only", "static"}' in patch
    assert "seed = int(self.server_args.random_seed)" in patch
    assert "int(run_nonce_sha256[:8], 16)" not in patch
    assert "ready_now = ready.query()" in patch
    assert 'if self.config.method == "l0":\n+            if not ready_now' in patch
    assert 'with self.timing("barrier"):' in patch
    assert "main.wait_event(ready)" in patch
    assert "TTS_FIXED_BOUNDARY_WAIT_TIMEOUT_S = 30.0" in patch
    assert "while not ready.query():" in patch
    assert "if time.monotonic() >= deadline:" in patch
    assert "TTS fixed-boundary candidate readiness timed out" in patch
    boundary = patch.split("+    def boundary(self) -> bool:", 1)[1].split(
        "+    def begin_round(", 1
    )[0]
    assert ".synchronize(" not in boundary
    assert ".item(" not in boundary


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


def test_runtime_launcher_uses_registered_patch_count(monkeypatch, tmp_path) -> None:
    import lightcone_spec.sglang_bridge.checkout as checkout_module

    checkout = tmp_path / "patched"
    package = checkout / "python" / "sglang"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")

    def fake_git(_checkout: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return PINNED_SGLANG_TREE
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if arguments == (
            "rev-list",
            "--count",
            f"{PINNED_SGLANG_COMMIT}..HEAD",
        ):
            return str(PINNED_SGLANG_PATCH_COUNT)
        raise AssertionError(arguments)

    monkeypatch.setattr(checkout_module, "_git", fake_git)
    monkeypatch.setattr(
        checkout_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    assert checkout_module.verify_patched_checkout(checkout) == checkout.resolve()


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
    assert main(["build-preliminary-speed-study", "--output", str(output)]) == 0
    manifest = PreliminarySpeedStudyManifest.load(output)
    assert manifest.gpu_evidence == "PRELIMINARY_DIAGNOSTIC_ONLY"
    assert manifest.formal_execution_authorized is False
    assert manifest.methods == ("static", "tts", "l0")
    assert capsys.readouterr().out.strip() == manifest.sha256


def test_cli_help_contains_only_focused_workflow(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "run-preliminary-confirmation" in output
    assert "collect-preliminary-speed-study" in output
    assert "run-preliminary-controlled-slice" in output
    assert "render-preliminary-static-load-runtime" in output
    assert "advance-preliminary-tuning-stage" in output
    assert "run-preliminary-natural-slice" in output
    assert "build-preliminary-profiler-plan" in output
    assert "select-preliminary-speed-config" in output
    assert "select-preliminary-anchor-config" in output
    assert "attest-preliminary-speed-study" in output
    assert "run-onlinespec-tuning-slice" in output
    assert "advance-onlinespec-tuning-stage" in output
    assert "select-onlinespec-anchor-config" in output
    assert "run-onlinespec-confirmation" in output
    assert "attest-onlinespec-study" in output
