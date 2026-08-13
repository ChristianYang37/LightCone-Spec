from __future__ import annotations

import ast
import hashlib
import inspect
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import lightcone_spec.experiments.runner as legacy_runner
from lightcone_spec.cli.main import (
    _execute_dispatch_wave,
    _load_industrial_analysis_manifest,
    _parser,
    _preliminary_gate_statistics,
    _preliminary_method_statistics,
)
from lightcone_spec.cli.main import (
    main as cli_main,
)
from lightcone_spec.experiments.onlinespec import OnlineSpecManifest
from lightcone_spec.orchestration import PreliminarySpeedStudyManifest
from lightcone_spec.orchestration.execution_bundle import (
    execute_dispatch_wave_bundles,
)
from lightcone_spec.orchestration.manifest import (
    PRELIMINARY_DIAGNOSTIC_ONLY,
    PRELIMINARY_SPEED_STUDY_MANIFEST_KIND,
)


def _canonical_sha256(value: object) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _write_bound_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    Path(f"{path}.sha256").write_text(
        _canonical_sha256(value) + "\n",
        encoding="utf-8",
    )


def _write_raw_bound_json(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    parsed = json.loads(body)
    Path(f"{path}.sha256").write_text(
        _canonical_sha256(parsed) + "\n",
        encoding="ascii",
    )


def test_preliminary_manifest_cannot_be_relabelled_as_formal() -> None:
    manifest = PreliminarySpeedStudyManifest.default()
    assert manifest.kind == PRELIMINARY_SPEED_STUDY_MANIFEST_KIND
    assert manifest.evidence_scope == PRELIMINARY_DIAGNOSTIC_ONLY
    assert manifest.gpu_evidence == PRELIMINARY_DIAGNOSTIC_ONLY
    assert manifest.formal_execution_authorized is False
    assert manifest.industrial_authority_consumption == "FORBIDDEN"
    with pytest.raises(ValueError, match="cannot authorize formal execution"):
        replace(manifest, formal_execution_authorized=True).validate()
    with pytest.raises(ValueError, match="cannot consume industrial authorities"):
        replace(manifest, industrial_authority_consumption="ALLOWED").validate()
    with pytest.raises(ValueError, match="cannot contain formal GPU claims"):
        replace(manifest, gpu_evidence="MEASURED").validate()


def test_historical_manifest_is_readable_but_forced_to_preliminary_scope(
    tmp_path: Path,
) -> None:
    current = PreliminarySpeedStudyManifest.default()
    historical = current.to_dict()
    for field in (
        "kind",
        "evidence_scope",
        "formal_execution_authorized",
        "industrial_authority_consumption",
    ):
        historical.pop(field)
    historical["name"] = "static-tts-l0-speed-study"
    historical["formal_context_start"] = historical.pop("diagnostic_context_start")
    historical["gpu_evidence"] = "UNMEASURED"
    path = tmp_path / "historical.json"
    _write_bound_json(path, historical)

    loaded = PreliminarySpeedStudyManifest.load(path)
    assert loaded.sha256 == _canonical_sha256(historical)
    assert loaded.evidence_scope == PRELIMINARY_DIAGNOSTIC_ONLY
    assert loaded.formal_execution_authorized is False
    with pytest.raises(ValueError, match="historical unscoped manifests are read-only"):
        loaded.write(tmp_path / "rewritten.json")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", 2.0),
        ("schema_version", True),
        ("formal_execution_authorized", 0),
        ("diagnostic_context_start", True),
        ("concurrency_grid", [True, 2, 4, 8, 16, 32, 48]),
        (
            "tuning_stages",
            [[2, 4096.0], [4, 8192], [8, 16384], [16, 32768]],
        ),
    ),
)
def test_preliminary_manifest_json_rejects_numeric_type_spoofs(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = PreliminarySpeedStudyManifest.default().to_dict()
    payload[field] = value
    path = tmp_path / f"preliminary-{field}.json"
    _write_bound_json(path, payload)

    with pytest.raises((TypeError, ValueError)):
        PreliminarySpeedStudyManifest.load(path)


def test_historical_manifest_schema_float_cannot_enter_compatibility_path(
    tmp_path: Path,
) -> None:
    historical = PreliminarySpeedStudyManifest.default().to_dict()
    for field in (
        "kind",
        "evidence_scope",
        "formal_execution_authorized",
        "industrial_authority_consumption",
    ):
        historical.pop(field)
    historical["schema_version"] = 2.0
    historical["name"] = "static-tts-l0-speed-study"
    historical["formal_context_start"] = historical.pop("diagnostic_context_start")
    historical["gpu_evidence"] = "UNMEASURED"
    path = tmp_path / "historical-float-schema.json"
    _write_bound_json(path, historical)

    with pytest.raises((TypeError, ValueError)):
        PreliminarySpeedStudyManifest.load(path)


def test_preliminary_manifest_rejects_duplicate_nonfinite_and_missing_fields(
    tmp_path: Path,
) -> None:
    payload = PreliminarySpeedStudyManifest.default().to_dict()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    duplicate = canonical.replace(
        '"schema_version":2',
        '"schema_version":false,"schema_version":2',
        1,
    )
    assert duplicate != canonical
    duplicate_path = tmp_path / "preliminary-duplicate.json"
    _write_raw_bound_json(duplicate_path, duplicate)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        PreliminarySpeedStudyManifest.load(duplicate_path)

    nonfinite = canonical.replace(
        '"confirmation_repetitions":8',
        '"confirmation_repetitions":1e999',
        1,
    )
    assert nonfinite != canonical
    nonfinite_path = tmp_path / "preliminary-nonfinite.json"
    _write_raw_bound_json(nonfinite_path, nonfinite)
    with pytest.raises(ValueError, match="non-finite JSON number"):
        PreliminarySpeedStudyManifest.load(nonfinite_path)

    malformed = payload.copy()
    malformed["model_pair"] = 7
    malformed_path = tmp_path / "preliminary-model-pair.json"
    _write_bound_json(malformed_path, malformed)
    with pytest.raises(ValueError, match="model pair mismatch"):
        PreliminarySpeedStudyManifest.load(malformed_path)

    historical = payload.copy()
    for field in (
        "kind",
        "evidence_scope",
        "formal_execution_authorized",
        "industrial_authority_consumption",
        "model_pair",
    ):
        historical.pop(field)
    historical["name"] = "static-tts-l0-speed-study"
    historical["formal_context_start"] = historical.pop("diagnostic_context_start")
    historical["gpu_evidence"] = "UNMEASURED"
    historical_path = tmp_path / "historical-missing-model-pair.json"
    _write_bound_json(historical_path, historical)
    with pytest.raises(ValueError, match="fields do not match schema"):
        PreliminarySpeedStudyManifest.load(historical_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", 2.0),
        ("schema_version", True),
        ("formal_execution_authorized", 0),
        (
            "tuning_stages",
            [[2, 16384], [4, 24576], [8, 32768], [16, True]],
        ),
    ),
)
def test_onlinespec_manifest_json_rejects_numeric_type_spoofs(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = OnlineSpecManifest.default()._payload()
    payload[field] = value
    path = tmp_path / f"onlinespec-{field}.json"
    _write_bound_json(path, payload)

    with pytest.raises((TypeError, ValueError)):
        OnlineSpecManifest.load(path)


def test_historical_onlinespec_manifest_is_forced_preliminary_and_read_only(
    tmp_path: Path,
) -> None:
    historical = OnlineSpecManifest.default()._payload()
    for field in (
        "kind",
        "evidence_scope",
        "formal_execution_authorized",
        "industrial_authority_consumption",
    ):
        historical.pop(field)
    historical["name"] = "onlinespec-clean-room-baseline"
    historical["formal_context_start"] = historical.pop("diagnostic_context_start")
    historical["gpu_evidence"] = "UNMEASURED"
    path = tmp_path / "historical-onlinespec.json"
    _write_bound_json(path, historical)

    loaded = OnlineSpecManifest.load(path)
    assert loaded.sha256 == _canonical_sha256(historical)
    assert loaded.evidence_scope == PRELIMINARY_DIAGNOSTIC_ONLY
    assert loaded.formal_execution_authorized is False
    with pytest.raises(ValueError, match="read-only preliminary evidence"):
        loaded.write(tmp_path / "rewritten-onlinespec.json")

    historical["schema_version"] = 2.0
    spoofed = tmp_path / "historical-onlinespec-float-schema.json"
    _write_bound_json(spoofed, historical)
    with pytest.raises((TypeError, ValueError)):
        OnlineSpecManifest.load(spoofed)


def test_onlinespec_manifest_rejects_duplicate_nonfinite_and_unknown_fields(
    tmp_path: Path,
) -> None:
    payload = OnlineSpecManifest.default()._payload()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    duplicate = canonical.replace(
        '"schema_version":2',
        '"schema_version":null,"schema_version":2',
        1,
    )
    assert duplicate != canonical
    duplicate_path = tmp_path / "onlinespec-duplicate.json"
    _write_raw_bound_json(duplicate_path, duplicate)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        OnlineSpecManifest.load(duplicate_path)

    nonfinite = canonical.replace(
        '"confirmation_repetitions":8',
        '"confirmation_repetitions":1e999',
        1,
    )
    assert nonfinite != canonical
    nonfinite_path = tmp_path / "onlinespec-nonfinite.json"
    _write_raw_bound_json(nonfinite_path, nonfinite)
    with pytest.raises(ValueError, match="non-finite JSON number"):
        OnlineSpecManifest.load(nonfinite_path)

    payload["_historical_source_sha256"] = "a" * 64
    unknown_path = tmp_path / "onlinespec-private-field.json"
    _write_bound_json(unknown_path, payload)
    with pytest.raises(ValueError, match="fields do not match schema"):
        OnlineSpecManifest.load(unknown_path)


@pytest.mark.parametrize(
    ("manifest", "loader"),
    (
        (
            PreliminarySpeedStudyManifest.default().to_dict(),
            PreliminarySpeedStudyManifest.load,
        ),
        (OnlineSpecManifest.default()._payload(), OnlineSpecManifest.load),
    ),
)
def test_preliminary_manifest_sidecars_are_exact_single_lines(
    tmp_path: Path,
    manifest: dict[str, object],
    loader,
) -> None:
    path = tmp_path / "manifest.json"
    _write_bound_json(path, manifest)
    sidecar = Path(f"{path}.sha256")
    sidecar.write_text(f" {_canonical_sha256(manifest)}\n", encoding="ascii")
    with pytest.raises(ValueError, match="sidecar"):
        loader(path)


def test_public_runner_entrypoints_require_concrete_manifests_without_sha_or_kwargs() -> (
    None
):
    public = {
        name: value
        for name, value in inspect.getmembers(legacy_runner, inspect.isfunction)
        if value.__module__ == legacy_runner.__name__
        and not name.startswith("_")
        and name.startswith(("run_", "measure_", "collect_"))
    }
    assert set(public) == {
        "collect_onlinespec_performance",
        "collect_preliminary_confirmation_performance",
        "measure_onlinespec_controlled_slice",
        "measure_preliminary_controlled_slice",
        "run_onlinespec_confirmation_slice",
        "run_preliminary_confirmation_slice",
        "run_preliminary_greedy_target_reference",
        "run_preliminary_natural_replication_slice",
    }
    for name, function in public.items():
        signature = inspect.signature(function)
        assert "manifest_sha256" not in signature.parameters
        assert all(
            parameter.kind is not inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        required_manifest = (
            "onlinespec_manifest" if "onlinespec" in name else "preliminary_manifest"
        )
        assert required_manifest in signature.parameters


def test_every_public_runner_entrypoint_consumes_its_exact_manifest_gate() -> None:
    public = {
        name: value
        for name, value in inspect.getmembers(legacy_runner, inspect.isfunction)
        if value.__module__ == legacy_runner.__name__
        and not name.startswith("_")
        and name.startswith(("run_", "measure_", "collect_"))
    }
    for name, function in public.items():
        helper = (
            "_onlinespec_manifest_sha256"
            if "onlinespec" in name
            else "_preliminary_manifest_sha256"
        )
        tree = ast.parse(inspect.getsource(function))
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert helper in called_names, f"{name} bypasses {helper}"


def test_exact_manifest_type_gate_precedes_runner_core(monkeypatch) -> None:
    monkeypatch.setattr(
        legacy_runner,
        "_measure_controlled_slice",
        lambda **_kwargs: pytest.fail("runner core was reached"),
    )
    common = {
        "client": object(),
        "method": "static",
        "samples": (),
        "phase": "static_load_screen",
        "stage": -1,
        "candidate_id": None,
        "config_sha256": "a" * 64,
        "model_lock_sha256": "b" * 64,
        "adaptation_config_sha256": None,
        "sampling_profile": object(),
        "context_limit": 4096,
        "concurrency": 1,
        "adaptation_group_id": "diagnostic",
        "warmup": False,
    }
    with pytest.raises(TypeError, match="exact PreliminarySpeedStudyManifest"):
        legacy_runner.measure_preliminary_controlled_slice(
            preliminary_manifest=OnlineSpecManifest.default(),
            **common,
        )
    with pytest.raises(TypeError, match="exact OnlineSpecManifest"):
        legacy_runner.measure_onlinespec_controlled_slice(
            onlinespec_manifest=PreliminarySpeedStudyManifest.default(),
            **common,
        )


def test_manifest_object_cannot_smuggle_a_bare_historical_sha() -> None:
    preliminary = replace(
        PreliminarySpeedStudyManifest.default(),
        _historical_source_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="historical manifest source identity"):
        legacy_runner._preliminary_manifest_sha256(preliminary)

    onlinespec = replace(
        OnlineSpecManifest.default(),
        _historical_source_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="historical OnlineSPEC source identity"):
        legacy_runner._onlinespec_manifest_sha256(onlinespec)


def test_industrial_analyzer_rejects_legacy_manifest_before_evidence_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "preliminary.json"
    PreliminarySpeedStudyManifest.default().write(manifest)
    monkeypatch.setattr(
        "lightcone_spec.cli.main._analysis_bound_json_path",
        lambda *_args, **_kwargs: pytest.fail("industrial evidence path was reached"),
    )
    with pytest.raises(ValueError, match="cannot enter industrial analysis"):
        _load_industrial_analysis_manifest(manifest)

    output = tmp_path / "industrial-output.json"
    with pytest.raises(ValueError, match="cannot enter industrial analysis"):
        cli_main(
            [
                "analyze-industrial",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_industrial_analyzer_rejects_float_schema_before_evidence_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "industrial-float-schema.json"
    _write_bound_json(
        manifest,
        {
            "schema_version": 3.0,
            "kind": "industrial_analysis_manifest",
            "registry_artifact": None,
            "pilot_activation": None,
            "final_activation": None,
            "confirmation_power_manifest": None,
            "gpu_inventory": None,
            "evidence_alias_manifests": [],
            "evidence_dependence_map": None,
            "gpu_attestation": None,
            "doctor_report": None,
            "hardware_envelope": {},
            "bootstrap": {},
            "blocks": [],
        },
    )
    monkeypatch.setattr(
        "lightcone_spec.cli.main._analysis_bound_json_path",
        lambda *_args, **_kwargs: pytest.fail("industrial evidence path was reached"),
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        _load_industrial_analysis_manifest(manifest)


def test_formal_execution_command_has_one_bundle_to_executor_path() -> None:
    assert (
        _execute_dispatch_wave.__globals__["execute_dispatch_wave_bundles"]
        is execute_dispatch_wave_bundles
    )
    formal_executor = execute_dispatch_wave_bundles.__globals__[
        "execute_industrial_plan"
    ]
    assert formal_executor.__module__ == "lightcone_spec.orchestration.executor"
    assert all(
        function is not formal_executor
        for _, function in inspect.getmembers(legacy_runner, inspect.isfunction)
    )


def test_preliminary_reducers_whitelist_out_future_formal_claim_fields() -> None:
    method = {
        "method": "tts",
        "mean_speedup": 0.1,
        "ci_lower": 0.01,
        "ci_upper": 0.2,
        "safety_pass": True,
        "acceleration_pass": True,
        "status": "MEASURED",
        "formal_ready": True,
    }

    @dataclass(frozen=True)
    class FutureGate:
        status: str
        tts: dict[str, object]
        l0: dict[str, object]
        l0_vs_tts: dict[str, object]
        gpu_evidence: str
        evidence_sha256: str
        formal_ready: bool

    gate = FutureGate(
        status="MEASURED",
        tts=method,
        l0={**method, "method": "l0"},
        l0_vs_tts={
            "numerator_method": "l0",
            "denominator_method": "tts",
            "mean_speedup": 0.01,
            "ci_lower": 0.0,
            "ci_upper": 0.02,
            "no_worse_pass": True,
            "status": "MEASURED",
            "formal_ready": True,
        },
        gpu_evidence="MEASURED",
        evidence_sha256="a" * 64,
        formal_ready=True,
    )
    reduced = _preliminary_gate_statistics(gate)
    encoded = json.dumps(reduced, sort_keys=True)
    assert "MEASURED" not in encoded
    assert "formal_ready" not in encoded
    assert reduced["status"] == PRELIMINARY_DIAGNOSTIC_ONLY
    assert reduced["evidence_sha256"] is None

    method_reduced = _preliminary_method_statistics(method)
    assert "status" not in method_reduced
    assert "formal_ready" not in method_reduced


def test_legacy_cli_names_are_unambiguously_preliminary() -> None:
    choices = _parser()._subparsers._group_actions[0].choices
    migration = {
        "build-speed-study": "build-preliminary-speed-study",
        "select-speed-config": "select-preliminary-speed-config",
        "select-anchor-config": "select-preliminary-anchor-config",
        "render-runtime": "render-preliminary-runtime",
        "render-static-load-runtime": "render-preliminary-static-load-runtime",
        "render-target-only-runtime": "render-preliminary-target-only-runtime",
        "render-tuning-runtime": "render-preliminary-tuning-runtime",
        "render-replication-runtime": "render-preliminary-replication-runtime",
        "list-tuning-candidates": "list-preliminary-tuning-candidates",
        "run-controlled-slice": "run-preliminary-controlled-slice",
        "run-natural-slice": "run-preliminary-natural-slice",
        "build-profiler-plan": "build-preliminary-profiler-plan",
        "collect-static-load-screen": "collect-preliminary-static-load-screen",
        "advance-tuning-stage": "advance-preliminary-tuning-stage",
        "run-confirmation": "run-preliminary-confirmation",
        "run-target-reference": "run-preliminary-target-reference",
        "collect-speed-study": "collect-preliminary-speed-study",
        "build-confirmation-queue": "build-preliminary-confirmation-queue",
        "attest-speed-study": "attest-preliminary-speed-study",
        "analyze-speed-study": "analyze-preliminary-speed-study",
    }
    assert set(migration.values()) <= set(choices)
    assert set(migration).isdisjoint(choices)
