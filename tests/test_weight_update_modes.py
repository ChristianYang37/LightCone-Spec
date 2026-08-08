from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.cli.main import build_parser
from lightcone_spec.config.loader import validate_adaptation_config_dict
from lightcone_spec.config.schema import MODEL_PAIRS, effective_proposal_depth
from lightcone_spec.controller.artifact import (
    ControllerArtifact,
    controller_artifact_filename,
    resolve_controller_artifact,
)
from lightcone_spec.exit_codes import ConfigError
from lightcone_spec.orchestration.catalog import (
    p1_manifest,
    p5_cross_backend_tail_manifest,
    p5_cross_backend_trace_manifest,
    p5_priority_dflash_0_40k_manifest,
    p5_priority_dflash_l3_evaluation_manifest,
    p5_priority_dflash_paired_trace_manifest,
    p5_priority_dflash_smoke_manifest,
    p5_priority_dflash_stride_screen_manifest,
)
from lightcone_spec.orchestration.manifest import ExperimentManifest
from lightcone_spec.orchestration.runtime_config import (
    _preflight_adaptation_reserve_mb,
)
from lightcone_spec.orchestration.units import RunUnit
from lightcone_spec.sglang_bridge.client import _pair_server_args, _server_args


def _config(scope: str) -> dict:
    return {
        "schema_version": 1,
        "method": "tts",
        "trainable_scope": scope,
        "optimizer": "adamw",
        "async": {"enabled": False, "max_in_flight": 1},
        "trace": {"artifact_root": "/tmp/lightcone-test"},
        "model": {"pair_id": "toy_tail"},
        "dataset": {"adapter": "toy_tail"},
    }


@pytest.mark.parametrize(
    ("source", "layout", "public", "rank"),
    [
        ("adapter", "output_residual", "residual", 16),
        ("residual", "output_residual", "residual", 16),
        ("output-residual", "output_residual", "residual", 16),
        ("output_residual", "output_residual", "residual", 16),
        ("lora", "tail_lora", "lora", 16),
        ("tail_lora", "tail_lora", "lora", 16),
        ("full", "full_rank_tail", "full", None),
        ("full_rank_tail", "full_rank_tail", "full", None),
    ],
)
def test_schema_v1_weight_update_aliases_resolve_canonically(
    source, layout, public, rank
):
    config = validate_adaptation_config_dict(_config(source))
    assert config.trainable_scope == layout
    assert config.tail_layout_mode == layout
    assert config.weight_update_mode == public
    assert config.parameter_scope == "tail"
    assert config.effective_adapter_rank == rank
    assert config.model_dump(mode="json")["trainable_scope"] == layout


def test_public_mode_and_parameter_scope_are_independent_schema_inputs():
    raw = _config("adapter")
    raw.pop("trainable_scope")
    raw.update(
        weight_update_mode="lora",
        parameter_scope="allowlist",
        parameter_allowlist=["model.layers.31.*", "lm_head"],
    )
    config = validate_adaptation_config_dict(raw)
    assert config.weight_update_mode == "lora"
    assert config.tail_layout_mode == "tail_lora"
    assert config.parameter_scope == "allowlist"
    assert config.parameter_allowlist == ("model.layers.31.*", "lm_head")


def test_public_and_schema_v1_mode_conflict_fails_closed():
    raw = _config("tail_lora")
    raw["weight_update_mode"] = "full"
    with pytest.raises(ConfigError, match="conflicts with deprecated"):
        validate_adaptation_config_dict(raw)


@pytest.mark.parametrize(
    "patch, message",
    [
        (
            {"weight_update_mode": "lora", "parameter_scope": "allowlist"},
            "requires a non-empty",
        ),
        (
            {
                "weight_update_mode": "lora",
                "parameter_scope": "tail",
                "parameter_allowlist": ["x"],
            },
            "only valid",
        ),
        (
            {"weight_update_mode": "residual", "parameter_scope": "all"},
            "requires parameter_scope=tail",
        ),
    ],
)
def test_invalid_mode_scope_combinations_fail_closed(patch, message):
    raw = _config("adapter")
    raw.pop("trainable_scope")
    raw.update(patch)
    with pytest.raises(ConfigError, match=message):
        validate_adaptation_config_dict(raw)


def _unit(method: str, scope: str = "adapter", rank: int = 16) -> RunUnit:
    return RunUnit(
        phase="mode-test",
        model_pair="toy_tail",
        method=method,
        dataset="toy_tail",
        prompt_subset="full",
        seed=0,
        lifecycle="stream",
        sampling_profile="greedy_t0",
        trainable_scope=scope,
        stride=4,
        logical_delay=0,
        concurrency=1,
        contention_condition="none",
        adapter_rank=rank,
    )


def test_manifest_override_changes_only_non_static_and_recomputes_hashes():
    original = ExperimentManifest(
        name="mode-test",
        phase="mode-test",
        description="mode override",
        units=[_unit("static"), _unit("tts")],
    )
    original_hash = original.content_sha256()
    static_id, adapted_id = [unit.unit_id for unit in original.units]

    resolved = original.with_weight_update_mode("lora")

    assert original.units[1].trainable_scope == "adapter"
    assert resolved.units[0].trainable_scope == "output_residual"
    assert resolved.units[0].unit_id == static_id
    assert resolved.units[1].trainable_scope == "tail_lora"
    assert resolved.units[1].weight_update_mode == "lora"
    assert resolved.units[1].unit_id != adapted_id
    assert resolved.content_sha256() != original_hash
    assert original.with_weight_update_mode(None) is original


def test_manifest_method_subset_is_immutable_hashed_and_fail_closed():
    original = ExperimentManifest(
        name="method-subset",
        phase="mode-test",
        description="staged execution overlay",
        units=[_unit("static"), _unit("tts"), _unit("naive_async")],
    )
    original_hash = original.content_sha256()

    selected = original.with_methods(["static", "tts"])

    assert [unit.method for unit in original.units] == [
        "static",
        "tts",
        "naive_async",
    ]
    assert [unit.method for unit in selected.units] == ["static", "tts"]
    assert selected.content_sha256() != original_hash
    assert original.with_methods(None) is original
    with pytest.raises(ConfigError, match="absent from this manifest"):
        original.with_methods(["not-a-method"])


def test_manifest_lifecycle_subset_is_immutable_hashed_and_fail_closed():
    original = ExperimentManifest(
        name="lifecycle-subset",
        phase="mode-test",
        description="lifecycle execution overlay",
        units=[_unit("tts"), replace(_unit("naive_async"), lifecycle="request")],
    )
    original_hash = original.content_sha256()

    selected = original.with_lifecycles(["stream", "stream"])

    assert [unit.lifecycle for unit in original.units] == ["stream", "request"]
    assert [unit.lifecycle for unit in selected.units] == ["stream"]
    assert selected.content_sha256() != original_hash
    assert selected.expected_units()[0]["expected_manifest_sha256"] == (
        selected.content_sha256()
    )
    assert original.with_lifecycles(None) is original
    with pytest.raises(ConfigError, match="requires at least one"):
        original.with_lifecycles([])
    with pytest.raises(ConfigError, match="absent from this manifest"):
        original.with_lifecycles(["not-a-lifecycle"])


def test_manifest_learning_rate_overlay_is_immutable_and_hash_bound():
    original = ExperimentManifest(
        name="learning-rate-overlay",
        phase="mode-test",
        description="engine parameter overlay",
        units=[_unit("tts")],
        engine_params={"lr": 1e-3, "max_rounds": 4},
    )
    original_expected = original.expected_units()

    resolved = original.with_learning_rate(2e-4)

    assert original.engine_params["lr"] == 1e-3
    assert resolved.engine_params["lr"] == pytest.approx(2e-4)
    assert resolved.content_sha256() != original.content_sha256()
    assert resolved.expected_units() != original_expected
    assert resolved.expected_units()[0]["expected_manifest_sha256"] == (
        resolved.content_sha256()
    )
    assert original.with_learning_rate(None) is original


@pytest.mark.parametrize(
    "learning_rate", [0.0, -1e-4, float("nan"), float("inf"), True, "1e-4"]
)
def test_manifest_learning_rate_overlay_rejects_invalid_values(learning_rate):
    manifest = ExperimentManifest(
        name="invalid-learning-rate",
        phase="mode-test",
        description="fail closed",
        units=[_unit("tts")],
    )
    with pytest.raises(ConfigError, match="positive finite"):
        manifest.with_learning_rate(learning_rate)


def test_gpu_manifest_preflight_deduplicates_controller_contracts_and_scopes(
    monkeypatch,
):
    from lightcone_spec.orchestration import runtime_config

    first = replace(
        _unit("lc_gate"),
        model_pair="qwen3_4b_dspark7",
        dataset="math500",
    )
    same_contract = replace(first, dataset="mt_bench", prompt_subset="other")
    calls = []

    def materialize(unit, engine_params, run_dir):
        calls.append((unit, dict(engine_params), run_dir))
        return {}

    monkeypatch.setattr(runtime_config, "materialize_gpu_runtime", materialize)
    evidence = runtime_config.preflight_gpu_manifest_inputs(
        [first, same_contract], {"controller_root": "/controller"}
    )
    assert evidence["controller_contracts_checked"] == 1
    assert evidence["representative_unit_ids"] == [first.unit_id]
    assert len(calls) == 1

    with pytest.raises(ConfigError, match="parameter_scope=tail only"):
        runtime_config.preflight_gpu_manifest_inputs(
            [replace(first, parameter_scope="all")], {}
        )


def test_run_manifest_cli_applies_overlay_in_memory_only(tmp_path, monkeypatch):
    from lightcone_spec.cli.main import cmd_run_manifest
    from lightcone_spec.orchestration import executor

    source = ExperimentManifest(
        name="cli-mode-test",
        phase="mode-test",
        description="source remains immutable",
        units=[_unit("static"), _unit("tts")],
    )
    path = tmp_path / "manifest.json"
    source.write(path)
    source_bytes = path.read_bytes()
    captured = []

    def execute(manifest, *_args, **_kwargs):
        captured.append(manifest)
        return SimpleNamespace(ok=True, counts=lambda: {"complete_valid": 2})

    monkeypatch.setattr(executor, "execute_manifest", execute)
    args = SimpleNamespace(
        manifest=str(path),
        artifact_root=str(tmp_path / "artifacts"),
        lockfile=None,
        runtime_root=str(tmp_path / "runtime"),
        model_roots=None,
        controller_root=None,
        peak_tflops_per_gpu=None,
        no_resume=False,
        weight_update_mode="lora",
        lifecycles=["stream"],
        learning_rate=2e-4,
    )
    assert cmd_run_manifest(args) == 0
    assert path.read_bytes() == source_bytes
    assert captured[0].units[0].trainable_scope == "output_residual"
    assert captured[0].units[1].trainable_scope == "tail_lora"
    assert captured[0].engine_params["weight_update_mode_override"] == "lora"
    assert captured[0].engine_params["lr"] == pytest.approx(2e-4)


def test_run_manifest_holds_root_lock_across_receipt_and_execution(
    tmp_path, monkeypatch
):
    from lightcone_spec.cli import main as cli_main
    from lightcone_spec.locking import lockfile as lockfile_module
    from lightcone_spec.orchestration import executor, runtime_config

    source = ExperimentManifest(
        name="locked-publication",
        phase="mode-test",
        description="receipt and resume scan share one critical section",
        units=[
            replace(
                _unit("static"),
                model_pair="qwen3_4b_dspark7",
                dataset="math500",
            )
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    source.write(manifest_path)
    events = []

    class FakeLock:
        datasets = ()

        @staticmethod
        def content_sha256():
            return "a" * 64

    @contextmanager
    def locked(root):
        events.append(("lock-enter", Path(root)))
        yield Path(root).resolve()
        events.append(("lock-exit", Path(root)))

    def write_receipt(receipt, output):
        events.append(("receipt", Path(output)))
        return "b" * 64

    def execute(manifest, root, **kwargs):
        events.append(("execute", Path(root)))
        assert manifest.engine_params["dataset_preflight_sha256"] == "b" * 64
        implementation = manifest.engine_params["runtime_implementation_fingerprint"]
        assert len(implementation["sha256"]) == 64
        assert {
            "lightcone_spec/orchestration/executor.py",
            "lightcone_spec/sglang_bridge/client.py",
            "sglang/srt/speculative/dflash_worker_v2.py",
            "sglang/srt/speculative/tail_adaptation.py",
        } <= set(implementation["files"])
        return SimpleNamespace(ok=True, counts=lambda: {"complete_valid": 1})

    monkeypatch.setattr(cli_main, "_artifact_root_lock", locked)
    monkeypatch.setattr(cli_main, "_prepare_locked_datasets", lambda *_a, **_k: {})
    monkeypatch.setattr(cli_main, "_write_dataset_receipt", write_receipt)
    monkeypatch.setattr(lockfile_module, "load_lockfile", lambda _path: FakeLock())
    monkeypatch.setattr(
        __import__("lightcone_spec.doctor", fromlist=["configure_runtime_cuda_toolkit"]),
        "configure_runtime_cuda_toolkit",
        lambda _root: {"supported": True, "root": "/cuda", "version": "12.9"},
    )
    monkeypatch.setattr(
        runtime_config, "preflight_gpu_manifest_inputs", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(executor, "execute_manifest", execute)

    artifact_root = tmp_path / "runs"
    assert cli_main.cmd_run_manifest(
        SimpleNamespace(
            manifest=str(manifest_path),
            artifact_root=str(artifact_root),
            lockfile=str(tmp_path / "lock.json"),
            runtime_root=str(tmp_path / "runtime"),
            model_roots=str(tmp_path / "model-roots.json"),
            controller_root=None,
            peak_tflops_per_gpu=None,
            no_resume=False,
            weight_update_mode=None,
            methods=None,
        )
    ) == 0
    assert [event[0] for event in events] == [
        "lock-enter",
        "receipt",
        "execute",
        "lock-exit",
    ]
    assert events[1][1] == artifact_root.resolve() / "dataset-preflight.json"


def test_artifact_root_lock_is_cross_process(tmp_path):
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH")))
    )
    program = """
import sys
from lightcone_spec.cli.main import _artifact_root_lock
with _artifact_root_lock(sys.argv[1]):
    print("locked", flush=True)
    sys.stdin.readline()
"""
    processes = []
    try:
        first = subprocess.Popen(
            [sys.executable, "-c", program, str(tmp_path)],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(first)
        assert first.stdout is not None
        assert select.select([first.stdout], [], [], 5)[0]
        assert first.stdout.readline().strip() == "locked"

        second = subprocess.Popen(
            [sys.executable, "-c", program, str(tmp_path)],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(second)
        assert second.stdout is not None
        assert not select.select([second.stdout], [], [], 0.25)[0]

        assert first.stdin is not None
        first.stdin.write("\n")
        first.stdin.flush()
        assert select.select([second.stdout], [], [], 5)[0]
        assert second.stdout.readline().strip() == "locked"
        assert second.stdin is not None
        second.stdin.write("\n")
        second.stdin.flush()
        assert first.wait(timeout=5) == 0
        assert second.wait(timeout=5) == 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


def test_full_tail_identity_ignores_inapplicable_adapter_rank_and_rejects_collision():
    first = _unit("tts", rank=8)
    second = replace(first, adapter_rank=32)
    manifest = ExperimentManifest(
        name="collision",
        phase="mode-test",
        description="rank must not make a full-tail pseudo experiment",
        units=[first, second],
    )
    with pytest.raises(ConfigError, match="duplicate identity"):
        manifest.with_weight_update_mode("full")


def test_legacy_full_and_canonical_full_tail_share_effective_identity():
    legacy = _unit("tts", scope="full", rank=8)
    canonical = _unit("tts", scope="full_rank_tail", rank=32)
    assert legacy.identity_dict()["trainable_scope"] == "full_rank_tail"
    assert legacy.identity_dict()["adapter_rank"] is None
    assert legacy.unit_id == canonical.unit_id


def test_parameter_scope_is_independent_and_tail_preserves_frozen_unit_id():
    legacy = _unit("tts", scope="tail_lora", rank=16)
    explicit_tail = replace(legacy, parameter_scope="tail")
    all_parameters = replace(legacy, parameter_scope="all")
    allowlisted = replace(
        legacy,
        parameter_scope="allowlist",
        parameter_allowlist=("model.layers.31.*",),
    )

    assert legacy.unit_id == explicit_tail.unit_id
    assert all_parameters.unit_id != legacy.unit_id
    assert allowlisted.unit_id not in {legacy.unit_id, all_parameters.unit_id}
    resolved = allowlisted.to_manifest_dict()
    assert resolved["weight_update_mode"] == "lora"
    assert resolved["parameter_scope"] == "allowlist"
    assert resolved["parameter_allowlist"] == ["model.layers.31.*"]


def test_public_only_manifest_unit_is_readable():
    source = _unit("tts", scope="lora").to_manifest_dict()
    source.pop("trainable_scope")
    source.pop("unit_id")
    loaded = RunUnit.from_dict(source)
    assert loaded.trainable_scope == "tail_lora"
    assert loaded.weight_update_mode == "lora"
    assert loaded.parameter_scope == "tail"


@pytest.mark.parametrize(
    ("legacy_scope", "canonical_scope"),
    [
        ("adapter", "output_residual"),
        ("output-residual", "output_residual"),
        ("lora", "tail_lora"),
        ("tail-lora", "tail_lora"),
        ("full", "full_rank_tail"),
    ],
)
def test_all_schema_v1_aliases_share_canonical_run_identity(
    legacy_scope, canonical_scope
):
    legacy = _unit("tts", scope=legacy_scope)
    canonical = _unit("tts", scope=canonical_scope)
    assert legacy.identity_dict()["trainable_scope"] == canonical_scope
    assert legacy.unit_id == canonical.unit_id


@pytest.mark.parametrize(
    ("legacy_scope", "canonical_scope"),
    [("adapter", "output_residual"), ("lora", "tail_lora")],
)
def test_legacy_alias_manifest_id_loads_canonical_in_memory_and_output(
    tmp_path, legacy_scope, canonical_scope
):
    from lightcone_spec.locking.hashing import sha256_json

    legacy = _unit("tts", scope=legacy_scope)
    source_identity = legacy.key_dict()
    unit = {
        **source_identity,
        "unit_id": sha256_json(source_identity),
        "required": True,
        "allow_resource_skip": False,
    }
    body = {
        "schema_version": 1,
        "name": f"old-{legacy_scope}",
        "phase": "mode-test",
        "description": "schema-v1 alias provenance",
        "profile": "cpu_reference",
        "lockfile_sha256": None,
        "engine_params": {},
        "units": [unit],
    }
    path = tmp_path / f"old-{legacy_scope}.json"
    path.write_text(json.dumps(body))
    loaded = ExperimentManifest.load(path)
    assert loaded.units[0].trainable_scope == canonical_scope
    assert loaded.to_dict()["units"][0]["trainable_scope"] == canonical_scope


@pytest.mark.parametrize(
    ("legacy_scope", "canonical_scope", "legacy_rank", "canonical_rank"),
    [
        ("adapter", "output_residual", 16, 16),
        ("lora", "tail_lora", 16, 16),
        ("full", "full_rank_tail", 8, 32),
    ],
)
def test_manifest_load_rejects_duplicate_effective_alias_identity(
    tmp_path, legacy_scope, canonical_scope, legacy_rank, canonical_rank
):
    legacy = _unit("tts", scope=legacy_scope, rank=legacy_rank)
    canonical = _unit("tts", scope=canonical_scope, rank=canonical_rank)
    body = ExperimentManifest(
        name="duplicate-full-tail",
        phase="mode-test",
        description="aliases must not form pseudo experiments",
        units=[legacy, canonical],
    ).to_dict()
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ConfigError, match="duplicate identity"):
        ExperimentManifest.load(path)


def test_legacy_full_manifest_id_loads_as_canonical_effective_identity(tmp_path):
    legacy = _unit("tts", scope="full", rank=8)
    old_identity = legacy.key_dict()
    from lightcone_spec.locking.hashing import sha256_json

    unit = {
        **old_identity,
        "unit_id": sha256_json(old_identity),
        "required": True,
        "allow_resource_skip": False,
    }
    body = {
        "schema_version": 1,
        "name": "old-full",
        "phase": "mode-test",
        "description": "pre-canonical full identity",
        "profile": "cpu_reference",
        "lockfile_sha256": None,
        "engine_params": {},
        "units": [unit],
    }
    path = tmp_path / "old-full.json"
    path.write_text(json.dumps(body))
    loaded = ExperimentManifest.load(path)
    resolved = loaded.to_dict()["units"][0]
    assert loaded.units[0].trainable_scope == "full_rank_tail"
    assert loaded.units[0].unit_id != unit["unit_id"]
    assert resolved["trainable_scope"] == "full_rank_tail"
    assert resolved["adapter_rank"] is None
    assert resolved["unit_id"] == loaded.units[0].unit_id


def test_legacy_exact_byte_sidecar_with_trailing_newline_remains_loadable(tmp_path):
    from lightcone_spec.locking.hashing import sha256_bytes

    manifest = ExperimentManifest(
        name="legacy-newline",
        phase="mode-test",
        description="historical exact-byte sidecar",
        units=[_unit("tts")],
    )
    path = tmp_path / "legacy-newline.json"
    source = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    path.write_text(source)
    path.with_suffix(".json.sha256").write_text(
        sha256_bytes(source.encode("utf-8")) + "\n"
    )
    assert ExperimentManifest.load(path).units[0].unit_id == manifest.units[0].unit_id


@pytest.mark.parametrize(
    "command", ["serve", "run-manifest", "analyze", "validate-artifacts"]
)
def test_cli_exposes_one_three_tier_weight_update_option(command):
    parser = build_parser()
    required = {
        "serve": ["--config", "x", "--lockfile", "l", "--model-roots", "r"],
        "run-manifest": ["--manifest", "m", "--artifact-root", "a"],
        "analyze": [
            "--manifest",
            "m",
            "--artifact-root",
            "a",
            "--output-dir",
            "o",
        ],
        "validate-artifacts": ["--manifest", "m", "--artifact-root", "a"],
    }[command]
    args = parser.parse_args(
        [command, *required, "--weight-update-mode", "residual"]
    )
    assert args.weight_update_mode == "residual"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [command, *required, "--weight-update-mode", "output-residual"]
        )


@pytest.mark.parametrize(
    "command", ["run-manifest", "analyze", "validate-artifacts"]
)
def test_manifest_commands_expose_lifecycle_and_learning_rate_overlays(command):
    parser = build_parser()
    required = {
        "run-manifest": ["--manifest", "m", "--artifact-root", "a"],
        "analyze": [
            "--manifest",
            "m",
            "--artifact-root",
            "a",
            "--output-dir",
            "o",
        ],
        "validate-artifacts": ["--manifest", "m", "--artifact-root", "a"],
    }[command]
    args = parser.parse_args(
        [
            command,
            *required,
            "--lifecycles",
            "stream",
            "request",
            "--learning-rate",
            "0.0002",
        ]
    )
    assert args.lifecycles == ["stream", "request"]
    assert args.learning_rate == pytest.approx(2e-4)


@pytest.mark.parametrize(
    "overlays, message",
    [
        ({"lifecycles": []}, "requires at least one"),
        ({"lifecycles": ["unknown"]}, "absent from this manifest"),
        ({"learning_rate": 0.0}, "positive finite"),
        ({"learning_rate": float("nan")}, "positive finite"),
        ({"learning_rate": float("inf")}, "positive finite"),
    ],
)
@pytest.mark.parametrize(
    "command", ["run-manifest", "analyze", "validate-artifacts"]
)
def test_manifest_commands_fail_closed_on_invalid_derived_overlays(
    tmp_path, command, overlays, message
):
    from lightcone_spec.cli.main import (
        cmd_analyze,
        cmd_run_manifest,
        cmd_validate_artifacts,
    )

    manifest = ExperimentManifest(
        name="invalid-overlay-cli",
        phase="mode-test",
        description="all manifest commands share overlay validation",
        units=[_unit("tts")],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.write(manifest_path)
    artifact_root = tmp_path / "runs"
    artifact_root.mkdir()
    common = {
        "manifest": str(manifest_path),
        "artifact_root": str(artifact_root),
        "methods": None,
        "lifecycles": ["stream"],
        "weight_update_mode": None,
        "learning_rate": None,
    }
    common.update(overlays)
    if command == "run-manifest":
        args = SimpleNamespace(
            **common,
            lockfile=None,
            runtime_root=str(tmp_path / "runtime"),
            model_roots=None,
            controller_root=None,
            peak_tflops_per_gpu=None,
            no_resume=False,
        )
        function = cmd_run_manifest
    elif command == "analyze":
        args = SimpleNamespace(
            **common,
            output_dir=str(tmp_path / "analysis"),
            baseline="static",
            itl_slo_ms=50.0,
        )
        function = cmd_analyze
    else:
        args = SimpleNamespace(**common, coverage_output=None)
        function = cmd_validate_artifacts
    with pytest.raises(ConfigError, match=message):
        function(args)


@pytest.mark.parametrize("command", ["analyze", "validate-artifacts"])
@pytest.mark.parametrize(
    "overlay", [{"lifecycles": ["stream"]}, {"learning_rate": 1e-4}]
)
def test_read_commands_reject_derived_overlay_without_manifest(
    tmp_path, command, overlay
):
    from lightcone_spec.cli.main import cmd_analyze, cmd_validate_artifacts

    artifact_root = tmp_path / "runs"
    artifact_root.mkdir()
    common = {
        "artifact_root": str(artifact_root),
        "manifest": None,
        "methods": None,
        "lifecycles": None,
        "weight_update_mode": None,
        "learning_rate": None,
    }
    common.update(overlay)
    if command == "analyze":
        args = SimpleNamespace(
            **common,
            output_dir=str(tmp_path / "analysis"),
            baseline="static",
            itl_slo_ms=50.0,
        )
        function = cmd_analyze
    else:
        args = SimpleNamespace(**common, coverage_output=None)
        function = cmd_validate_artifacts
    with pytest.raises(ConfigError, match="requires --manifest"):
        function(args)


def test_model_pairs_declare_backend_capabilities_without_expanding_old_p1():
    expected = {
        "qwen3_8b_dspark7": "DSPARK",
        "qwen3_8b_dflash16": "DFLASH",
        "qwen3_8b_eagle3": "EAGLE3",
        "llama2_7b_eagle": "EAGLE",
    }
    for pair_id, algorithm in expected.items():
        pair = MODEL_PAIRS[pair_id]
        assert pair["speculative_algorithm"] == algorithm
        assert pair["capabilities"]["tail_adaptation"] is True
        assert pair["capabilities"]["multi_layer"] is False
    assert {unit.model_pair for unit in p1_manifest().units}.isdisjoint(
        {"qwen3_8b_dflash16", "qwen3_8b_eagle3", "llama2_7b_eagle"}
    )


def test_backend_server_args_are_capability_driven_and_fail_closed_to_linear_eagle():
    common = dict(
        target_path="target",
        drafter_path="draft",
        adaptation_config_path="adapt.yaml",
        num_draft_tokens=8,
        tensor_parallel_size=1,
        random_seed=0,
    )
    dflash = _pair_server_args(MODEL_PAIRS["qwen3_8b_dflash16"], **common)
    assert dflash["speculative_algorithm"] == "DFLASH"
    assert dflash["speculative_accept_threshold_single"] == 1.0
    assert "dspark_adaptation_config" not in dflash
    assert dflash["speculative_adaptation_config"] == "adapt.yaml"

    eagle = _pair_server_args(MODEL_PAIRS["qwen3_8b_eagle3"], **common)
    assert eagle["speculative_algorithm"] == "EAGLE3"
    assert eagle["speculative_eagle_topk"] == 1
    assert eagle["speculative_num_draft_tokens"] == 8
    assert eagle["speculative_num_steps"] == 7
    assert eagle["enable_multi_layer_eagle"] is False
    assert eagle["speculative_use_rejection_sampling"] is True


def test_eagle_desired_depth_is_the_only_steps_and_preflight_source(tmp_path):
    target = tmp_path / "target"
    drafter = tmp_path / "drafter"
    target.mkdir()
    drafter.mkdir()
    (target / "config.json").write_text(
        json.dumps({"hidden_size": 512, "vocab_size": 4096})
    )
    (drafter / "config.json").write_text("{}")
    roots = {
        MODEL_PAIRS["qwen3_8b_eagle3"]["target"]: str(target),
        MODEL_PAIRS["qwen3_8b_eagle3"]["drafter"]: str(drafter),
    }
    unit = replace(
        _unit("naive_async", "output_residual"),
        model_pair="qwen3_8b_eagle3",
    )

    args = _pair_server_args(
        MODEL_PAIRS["qwen3_8b_eagle3"],
        target_path="target",
        drafter_path="draft",
        adaptation_config_path="adapt.yaml",
        num_draft_tokens=5,
        tensor_parallel_size=1,
        random_seed=0,
    )
    assert args["speculative_num_draft_tokens"] == 5
    assert args["speculative_num_steps"] == 4

    reserve_depth_5 = _preflight_adaptation_reserve_mb(
        unit, {"speculative_num_draft_tokens": 5}, roots
    )
    reserve_depth_8 = _preflight_adaptation_reserve_mb(
        unit, {"speculative_num_draft_tokens": 8}, roots
    )
    assert reserve_depth_8 > reserve_depth_5 > 0


def test_pair_default_draft_window_and_cross_backend_catalog():
    dflash_unit = replace(
        _unit("tts", "output_residual"), model_pair="qwen3_8b_dflash16"
    )
    args = _server_args(dflash_unit, {}, "adapt.yaml")
    assert args["speculative_algorithm"] == "DFLASH"
    assert args["speculative_num_draft_tokens"] == 16

    manifest = p5_cross_backend_tail_manifest()
    assert len(manifest.units) == 3 * 3 * 2 * 5
    assert {unit.model_pair for unit in manifest.units} == {
        "qwen3_8b_dspark7",
        "qwen3_8b_dflash16",
        "qwen3_8b_eagle3",
    }
    assert manifest.engine_params["p5_context_lengths"] == [4096, 16384, 32768]
    assert len({unit.unit_id for unit in manifest.units}) == len(manifest.units)


def test_server_graph_cap_uses_the_preflight_adapter_row_capacity():
    unit = replace(
        _unit("tts", "output_residual"),
        model_pair="qwen3_4b_dflash16",
        concurrency=4,
    )
    args = _server_args(
        unit,
        {"max_running_requests": 20, "adapter_row_capacity": 20},
        "adapt.yaml",
    )
    assert args["max_running_requests"] == 20
    assert args["cuda_graph_max_bs_decode"] == 20

    explicit = _server_args(
        unit,
        {
            "max_running_requests": 20,
            "cuda_graph_max_bs_decode": 32,
            "adapter_row_capacity": 32,
        },
        "adapt.yaml",
    )
    assert explicit["cuda_graph_max_bs_decode"] == 32

    with pytest.raises(ValueError, match="adapter_row_capacity"):
        _server_args(
            unit,
            {"max_running_requests": 20, "adapter_row_capacity": 256},
            "adapt.yaml",
        )


def test_effective_proposal_depth_matches_backend_window_semantics():
    dspark = MODEL_PAIRS["qwen3_4b_dspark7"]
    dflash = MODEL_PAIRS["qwen3_4b_dflash16"]
    eagle = MODEL_PAIRS["qwen3_8b_eagle3"]

    assert effective_proposal_depth(dspark, 8) == 7
    assert effective_proposal_depth(dspark, 5) == 4
    assert effective_proposal_depth(dflash, 16) == 15
    assert effective_proposal_depth(dflash, 8) == 7
    assert effective_proposal_depth(eagle, 8) == 8
    assert effective_proposal_depth(eagle, 5) == 5
    with pytest.raises(ValueError, match="must be >= 2"):
        effective_proposal_depth(dflash, 1)
    with pytest.raises(ValueError, match="must be >= 1"):
        effective_proposal_depth(eagle, 0)


def test_cross_backend_trace_producer_pairs_l0_and_tts_with_bounded_quota():
    manifest = p5_cross_backend_trace_manifest()
    assert len(manifest.units) == 2 * 3
    assert manifest.engine_params["trace_capture_max_bytes"] == 6 * (1 << 30)
    assert manifest.engine_params["trace_capture_max_records_per_request"] == 1
    assert manifest.engine_params["prompt_limit"] == 96
    assert manifest.engine_params["trace_producer_methods"] == [
        "naive_async",
        "tts",
    ]
    for pair_id in {
        "qwen3_8b_dspark7",
        "qwen3_8b_dflash16",
        "qwen3_8b_eagle3",
    }:
        pair_units = [unit for unit in manifest.units if unit.model_pair == pair_id]
        assert {unit.method for unit in pair_units} == {"naive_async", "tts"}
        assert {
            (unit.dataset, unit.prompt_subset, unit.seed, unit.lifecycle)
            for unit in pair_units
        } == {("livecodebench", "p5_ctx_16384", 0, "stream")}
    for mode in ("output-residual", "lora", "full"):
        resolved = manifest.with_weight_update_mode(mode)
        assert len({unit.unit_id for unit in resolved.units}) == len(resolved.units)


def test_priority_dflash_long_context_manifest_has_frozen_lora_identity():
    manifest = p5_priority_dflash_0_40k_manifest()
    assert manifest.name == "p5_priority_dflash_0_40k_v1"
    assert manifest.engine_params == {
        "prompt_limit": 16,
        "benchmark_repetitions": 3,
        "max_new_tokens": 128,
        "ignore_eos": True,
        "max_running_requests": 8,
        "max_total_tokens": 400000,
        "p5_context_lengths": [512, 4096, 16384, 40000],
        "peak_tflops_per_gpu": 500.0,
        "peak_tflops_basis": (
            "nvidia_official_1pflops_bf16_sparse_dense_inferred_half_v1"
        ),
        "lr": 3e-5,
        "warmup_prompts": 4,
        "request_timeout_s": 1800,
    }
    assert 40000 + manifest.engine_params["max_new_tokens"] <= 40960
    assert "not a literal zero-context" in manifest.description
    assert len(manifest.units) == 3 * 6
    assert len({unit.unit_id for unit in manifest.units}) == len(manifest.units)
    assert {unit.method for unit in manifest.units} == {
        "static",
        "tts",
        "naive_async",
        "lc_gate",
        "lc_damp",
        "lc_transport",
    }
    assert {unit.concurrency for unit in manifest.units} == {1, 4, 8}
    assert {
        (
            unit.model_pair,
            unit.lifecycle,
            unit.sampling_profile,
            unit.trainable_scope,
            unit.adapter_rank,
            unit.stride,
            unit.logical_delay,
            unit.prompt_subset,
        )
        for unit in manifest.units
    } == {
        (
            "qwen3_4b_dflash16",
            "stream",
            "greedy_t0",
            "tail_lora",
            16,
            4,
            0,
            "p5_ctx_512-40000",
        )
    }


def test_priority_dflash_stride_screen_has_nine_paired_non_claim_units():
    manifest = p5_priority_dflash_stride_screen_manifest()
    expected_engine_params = {
        "prompt_limit": 40,
        "prompt_offset": 0,
        "benchmark_repetitions": 3,
        "max_new_tokens": 512,
        "ignore_eos": True,
        "max_running_requests": 20,
        "max_total_tokens": 400000,
        "p5_context_lengths": [4096, 16384],
        "p5_context_timing_contract": "independent_exact_context_group_v1",
        "pytorch_cuda_alloc_conf": "backend:native,expandable_segments:True",
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "warmup_prompts": 20,
        "trace_level": "light",
        "claim_scope": "candidate_screen_only_no_ci",
        "p5_stride_screen_required_safety_columns": [
            "adaptation_fallback_count"
        ],
        "request_timeout_s": 1800,
    }

    assert manifest.name == "p5_priority_dflash_stride_screen_v1"
    assert manifest.phase == "p5_priority_dflash_stride_screen_v1"
    assert "Non-claim" in manifest.description
    assert manifest.engine_params == expected_engine_params
    assert len(manifest.units) == 9
    assert len({unit.unit_id for unit in manifest.units}) == 9
    assert [unit.method for unit in manifest.units].count("static") == 1
    assert [unit.method for unit in manifest.units].count("tts") == 4
    assert [unit.method for unit in manifest.units].count("naive_async") == 4
    assert [unit.method for unit in manifest.units] == [
        "static",
        *("tts" for _ in range(4)),
        *("naive_async" for _ in range(4)),
    ]
    assert {
        (unit.method, unit.stride, unit.contention_condition)
        for unit in manifest.units
    } == {
        ("static", 1, "none"),
        *(("tts", stride, "none") for stride in (1, 4, 8, 16)),
        *(
            ("naive_async", stride, "realistic_async")
            for stride in (1, 4, 8, 16)
        ),
    }
    assert {
        (
            unit.model_pair,
            unit.dataset,
            unit.prompt_subset,
            unit.lifecycle,
            unit.sampling_profile,
            unit.trainable_scope,
            unit.adapter_rank,
            unit.logical_delay,
            unit.concurrency,
        )
        for unit in manifest.units
    } == {
        (
            "qwen3_4b_dflash16",
            "livecodebench",
            "p5_ctx_4096-16384",
            "stream",
            "greedy_t0",
            "tail_lora",
            16,
            0,
            20,
        )
    }
    required_slots = 20 * (
        16384
        + expected_engine_params["max_new_tokens"]
        + MODEL_PAIRS["qwen3_4b_dflash16"]["default_num_draft_tokens"]
    )
    assert required_slots <= expected_engine_params["max_total_tokens"]

    manifest_path = (
        Path(__file__).parents[1]
        / "manifests/p5/p5_priority_dflash_stride_screen_v1.json"
    )
    written = ExperimentManifest.load(manifest_path)
    assert written.to_dict() == manifest.to_dict()
    assert [unit.unit_id for unit in written.units] == [
        unit.unit_id for unit in manifest.units
    ]


def test_priority_dflash_trace_uses_natural_delay_for_both_methods():
    manifest = p5_priority_dflash_paired_trace_manifest()
    assert manifest.name == "p5_priority_dflash_paired_trace_v1"
    assert len(manifest.units) == 4
    assert {unit.method for unit in manifest.units} == {"naive_async", "tts"}
    assert {unit.logical_delay for unit in manifest.units} == {0}
    assert {
        (
            unit.model_pair,
            unit.lifecycle,
            unit.sampling_profile,
            unit.trainable_scope,
            unit.adapter_rank,
            unit.stride,
            unit.concurrency,
        )
        for unit in manifest.units
    } == {
        ("qwen3_4b_dflash16", "stream", "greedy_t0", "tail_lora", 16, 4, 1),
        ("qwen3_4b_dflash16", "stream", "greedy_t0", "tail_lora", 16, 4, 4),
    }
    assert manifest.engine_params["lr"] == 3e-5
    assert manifest.engine_params["max_new_tokens"] == 512
    assert manifest.engine_params["ignore_eos"] is True
    assert manifest.engine_params["p5_context_lengths"] == [4096, 16384, 40000]
    assert manifest.engine_params["prompt_limit"] == 48
    assert manifest.engine_params["max_running_requests"] == 4
    assert manifest.engine_params["max_total_tokens"] == 400000
    assert manifest.engine_params["trace_capture_max_bytes"] == 6 * (1 << 30)
    assert manifest.engine_params["trace_capture_max_records_per_request"] == 3
    assert manifest.engine_params["trace_capture_sampling"] == "staged"
    assert manifest.engine_params["trace_producer_methods"] == [
        "naive_async",
        "tts",
    ]


def test_priority_dflash_l3_evaluation_exactly_mirrors_phase1_tts_cells():
    phase1 = p5_priority_dflash_paired_trace_manifest()
    phase2 = p5_priority_dflash_l3_evaluation_manifest()
    tts_units = [unit for unit in phase1.units if unit.method == "tts"]

    def paired_cell(unit):
        return (
            unit.model_pair,
            unit.dataset,
            unit.prompt_subset,
            unit.seed,
            unit.lifecycle,
            unit.sampling_profile,
            unit.trainable_scope,
            unit.adapter_rank,
            unit.stride,
            unit.logical_delay,
            unit.concurrency,
        )

    assert {paired_cell(unit) for unit in phase2.units} == {
        paired_cell(unit) for unit in tts_units
    }
    assert {unit.method for unit in phase2.units} == {"lc_transport"}
    for key in (
        "prompt_limit",
        "benchmark_repetitions",
        "max_new_tokens",
        "ignore_eos",
        "max_running_requests",
        "max_total_tokens",
        "p5_context_lengths",
        "lr",
        "trace_level",
        "trace_capture_max_bytes",
        "trace_capture_max_records_per_request",
        "trace_capture_sampling",
        "warmup_prompts",
        "request_timeout_s",
    ):
        assert phase2.engine_params[key] == phase1.engine_params[key]
    assert phase2.engine_params["l3_evaluation_only"] is True
    assert phase2.engine_params["trace_producer_methods"] == ["lc_transport"]
    assert phase2.engine_params["trace_capture_max_bytes"] > 0


def test_priority_dflash_smoke_is_minimal_greedy_adaptation_run():
    manifest = p5_priority_dflash_smoke_manifest()
    assert manifest.name == "p5_priority_dflash_smoke_v1"
    assert manifest.engine_params["ignore_eos"] is True
    assert len(manifest.units) == 3
    assert {unit.method for unit in manifest.units} == {
        "static",
        "tts",
        "naive_async",
    }
    assert len({unit.unit_id for unit in manifest.units}) == len(manifest.units)
    assert {
        (
            unit.model_pair,
            unit.prompt_subset,
            unit.lifecycle,
            unit.sampling_profile,
            unit.trainable_scope,
            unit.adapter_rank,
            unit.concurrency,
            unit.logical_delay,
        )
        for unit in manifest.units
    } == {
        (
            "qwen3_4b_dflash16",
            "p5_ctx_512",
            "stream",
            "greedy_t0",
            "tail_lora",
            16,
            1,
            0,
        )
    }
    assert manifest.engine_params["prompt_limit"] == 2
    assert manifest.engine_params["benchmark_repetitions"] == 1
    assert manifest.engine_params["max_new_tokens"] == 32
    assert manifest.engine_params["max_running_requests"] == 1
    assert manifest.engine_params["p5_context_lengths"] == [512]
    assert manifest.engine_params["lr"] == 3e-5


def test_controller_artifact_resolution_binds_pair_mode_and_layout(
    tmp_path, monkeypatch
):
    pair = "qwen3_8b_eagle3"
    mode = "tail_lora"
    layout_sha = "a" * 64
    name = controller_artifact_filename(pair, mode, layout_sha)
    path = tmp_path / name
    path.touch()
    artifact = SimpleNamespace(
        model_pair_id=pair,
        extra={
            "parameter_layout_sha256": layout_sha,
            "controller_runtime_identity": {
                "candidate": {"weight_update_mode": mode}
            },
        },
    )
    monkeypatch.setattr(
        ControllerArtifact,
        "load",
        staticmethod(lambda _path: artifact),
    )

    resolved_path, resolved_artifact = resolve_controller_artifact(
        tmp_path,
        model_pair_id=pair,
        weight_update_mode="lora",
    )
    assert resolved_path == path
    assert resolved_artifact is artifact


def test_controller_artifact_resolution_fails_closed_before_model_load(tmp_path):
    with pytest.raises(ConfigError, match="bounded p5_cross_backend_trace producer"):
        resolve_controller_artifact(
            tmp_path,
            model_pair_id="qwen3_8b_dflash16",
            weight_update_mode="full",
        )
