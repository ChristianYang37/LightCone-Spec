from __future__ import annotations

import json
import asyncio
import math
import runpy
import sys
import time
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from lightcone_spec.adapters.projections import (
    load_projection_artifact,
    save_projection_artifact,
)
from lightcone_spec.doctor import (
    collect_doctor_report,
    configure_runtime_cuda_toolkit,
)
from lightcone_spec.exit_codes import (
    ConfigError,
    ExactnessViolation,
    LockError,
    ResourceSkip,
    RuntimeGpuFailure,
)
from lightcone_spec.locking.lockfile import (
    LockedEnvironment,
    LockedFile,
    LockedHFSnapshot,
    Lockfile,
)
from lightcone_spec.locking.download import write_model_roots
from lightcone_spec.locking.hashing import sha256_file
from lightcone_spec.locking.verify import verify_lockfile_offline
from lightcone_spec.orchestration import runtime_config
from lightcone_spec.orchestration.manifest import ExperimentManifest
from lightcone_spec.orchestration.runtime_config import (
    _preflight_adaptation_reserve_mb,
    materialize_gpu_runtime,
)
from lightcone_spec.orchestration.units import RunUnit
from lightcone_spec.replay.real import (
    _l3_transport_gate,
    _scan_exactness_evidence,
    fit_real_replay,
)
from lightcone_spec.replay.splits import split_of_group
from lightcone_spec.sglang_bridge.client import (
    _GpuSystemSampler,
    _paired_sampling_seed,
    _run_streaming_chunk,
    _run_streaming_pool,
    _runtime_request_id,
    _server_args,
    _telemetry_to_rows,
)
from lightcone_spec.sglang_bridge import client as bridge_client
from lightcone_spec.sglang_bridge.runtime import _sequence_group_from_request_id
from lightcone_spec.sglang_bridge.telemetry import (
    RoundTelemetry,
    TelemetrySink,
    UpdateTelemetry,
)


def _unit(method: str = "static") -> RunUnit:
    return RunUnit(
        phase="smoke_gpu",
        model_pair="qwen3_4b_dspark7",
        method=method,
        dataset="alpaca",
        prompt_subset="smoke32",
        seed=0,
        lifecycle="request",
        sampling_profile="greedy_t0",
        trainable_scope="adapter",
        stride=4,
        logical_delay=0,
        concurrency=4,
        contention_condition="none",
        adapter_rank=16,
    )


def test_load_prompts_forwards_the_declared_deterministic_offset(monkeypatch):
    calls = []

    class Adapter:
        def load_samples(self, lock, *, limit, offset):
            calls.append((lock, limit, offset))
            return [types.SimpleNamespace(sample_id="heldout", prompt="prompt")]

    monkeypatch.setattr(
        "lightcone_spec.benchmarks.registry.get_adapter", lambda _key: Adapter()
    )
    lock = object()
    prompts = bridge_client._load_prompts(
        _unit(),
        {
            "prompt_limit": 48,
            "prompt_offset": 40,
            "enable_thinking": False,
        },
        lock,
    )
    assert calls == [(lock, 48, 40)]
    assert prompts == [{"sample_id": "heldout", "prompt": "prompt"}]


def test_request_sampling_seed_is_paired_and_stochastic_mode_is_deterministic():
    seed = _paired_sampling_seed(
        sample_id="prompt-a:ctx-16384", repetition=2, experiment_seed=17
    )
    assert seed == _paired_sampling_seed(
        sample_id="prompt-a:ctx-16384", repetition=2, experiment_seed=17
    )
    assert seed != _paired_sampling_seed(
        sample_id="prompt-a:ctx-16384", repetition=3, experiment_seed=17
    )
    assert seed != _paired_sampling_seed(
        sample_id="prompt-b:ctx-16384", repetition=2, experiment_seed=17
    )
    stochastic = replace(_unit(), sampling_profile="main_t1_p1")
    args = _server_args(stochastic, {}, "adaptation.yaml")
    assert args["enable_deterministic_inference"] is True
    assert "enable_deterministic_inference" not in _server_args(
        _unit(), {}, "adaptation.yaml"
    )


def test_scheduler_moments_report_realized_batch_without_row_bias():
    before = {
        "flops": 10.0,
        "read_bytes": 20.0,
        "write_bytes": 30.0,
        "decode_moments": [5.0, 10.0, 100.0, 22.0, 500.0, 7.0],
        "prefill_tokens": 100,
        "prefill_busy_us": 200,
        "fallbacks": 0,
        "retractions": 0,
        "peak_running": 4,
        "peak_queue": 0,
    }
    # Two measured scheduler steps, with (B, span_us)=(2,100),(4,300).
    after = {
        **before,
        "flops": 2e12 + 10.0,
        "decode_moments": [7.0, 16.0, 500.0, 42.0, 1900.0, 27.0],
        "prefill_tokens": 132,
        "prefill_busy_us": 1200,
    }
    evidence = bridge_client._performance_evidence(
        before, after, wall_s=2.0, peak_tflops_per_gpu=2.0,
        offered_concurrency=4,
    )
    assert evidence["decode_step_count"] == 2
    assert evidence["decode_batch_size_step_mean"] == pytest.approx(3.0)
    assert evidence["decode_batch_size_time_mean"] == pytest.approx(3.5)
    assert evidence["decode_batch_size_std"] == pytest.approx(1.0)
    assert evidence["decode_batch_fill_ratio"] == pytest.approx(0.875)
    assert evidence["decode_scheduler_span_s"] == pytest.approx(0.0004)
    assert evidence["decode_generated_tps_scheduler_span"] == pytest.approx(50000)
    assert evidence["prefill_uncached_tokens"] == 32
    assert evidence["prefill_busy_s"] == pytest.approx(0.001)
    assert evidence["estimated_mfu"] == pytest.approx(0.5)


def test_configured_mfu_denominator_requires_positive_scheduler_evidence():
    evidence = {
        "estimated_tflops_per_gpu": 0.0,
        "estimated_mfu": 0.0,
        "decode_step_count": 0,
    }
    with pytest.raises(RuntimeGpuFailure, match="requires positive target-model"):
        bridge_client._require_configured_mfu_evidence(
            evidence,
            peak_tflops_per_gpu=500.0,
            location="confirmation",
        )
    bridge_client._require_configured_mfu_evidence(
        evidence,
        peak_tflops_per_gpu=None,
        location="non-headline screen",
    )


def _round_record(**overrides) -> dict:
    record = {
        "kind": "round",
        "request_id": "runtime-rid",
        "round_id": 0,
        "active_version": 0,
        "proposal_version": 0,
        "draft_tokens": 8,
        "accepted_drafts": 6,
        "committed_per_verify": 7,
        "target_calls": 1,
        "draft_cuda_us": 10.0,
        "verify_cuda_us": 20.0,
        "accept_cuda_us": 3.0,
        "draft_cpu_us": 2.0,
        "verify_cpu_us": 2.0,
        "rng_substream_id": "rng-0",
        "version_canary_ok": True,
        "prefix_pos_before": 10,
        "prefix_pos_after": 17,
        "prefix_len_before": 10,
        "verify_len": 9,
        "batch_size": 4,
        "offered_concurrency": 4,
        "round_wall_us": 1000.0,
        "prefix_feature_exact": True,
        "algorithmic_censored": False,
    }
    record.update(overrides)
    return record


def _system_samples(hbm: int = 123) -> list[dict]:
    return [
        {
            "timestamp_us": 1.0,
            "gpu_index": 0,
            "hbm_used_bytes": hbm,
            "sm_occupancy": None,
            "gpu_utilization": 50.0,
            "power_watts": 100.0,
            "energy_joules_delta": 1.0,
            "main_stream_active": True,
            "side_stream_active": True,
            "stream_contention_class": "realistic_async",
            "sync_us_delta": 0.0,
            "sample_source": "runtime_instrumentation",
            "activity_provenance": "observed",
            "contention_provenance": "observed",
            "sync_provenance": "observed",
        }
    ]


def test_nvml_sampler_does_not_fabricate_stream_or_sync_observations(monkeypatch):
    fake_nvml = types.ModuleType("pynvml")
    fake_nvml.nvmlInit = lambda: None
    fake_nvml.nvmlShutdown = lambda: None
    fake_nvml.nvmlDeviceGetHandleByIndex = lambda _index: object()
    fake_nvml.nvmlDeviceGetMemoryInfo = lambda _handle: types.SimpleNamespace(
        used=1024
    )
    fake_nvml.nvmlDeviceGetUtilizationRates = (
        lambda _handle: types.SimpleNamespace(gpu=75)
    )
    fake_nvml.nvmlDeviceGetPowerUsage = lambda _handle: 125_000
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    sampler = _GpuSystemSampler(
        replace(
            _unit(method="naive_async"),
            contention_condition="realistic_async",
        ),
        interval_s=0.001,
    )
    deadline = time.monotonic() + 1.0
    while not sampler.samples and time.monotonic() < deadline:
        time.sleep(0.001)
    samples = sampler.stop()

    assert samples
    sample = samples[0]
    assert sample["sample_source"] == "nvml"
    assert sample["main_stream_active"] is None
    assert sample["side_stream_active"] is None
    assert sample["activity_provenance"] == "not_observed"
    assert sample["sync_us_delta"] is None
    assert sample["sync_provenance"] == "not_observed"
    assert sample["stream_contention_class"] == "realistic_async"
    assert sample["contention_provenance"] == "declared_not_observed"


def test_l3_gate_reports_cluster_bootstrapped_relative_drop_reduction():
    test_groups = []
    candidate = 0
    while len(test_groups) < 2:
        group = f"transport-test-{candidate}"
        if split_of_group(group) == "test":
            test_groups.append(group)
        candidate += 1

    records = [
        types.SimpleNamespace(
            row=types.SimpleNamespace(sequence_id=test_groups[0]),
            delta_g=np.asarray([1.0]),
            delta_z=np.asarray([0.0]),
        ),
        types.SimpleNamespace(
            row=types.SimpleNamespace(sequence_id=test_groups[1]),
            delta_g=np.asarray([2.0]),
            delta_z=np.asarray([2.0]),
        ),
    ]
    transport = types.SimpleNamespace(state_correction=lambda delta_z: delta_z)
    gate = _l3_transport_gate(records, transport, bootstrap_b=100)

    assert gate["relative_error_reduction_vs_drop"] == pytest.approx(0.8)
    assert gate["baseline_mse"] == pytest.approx(2.5)
    assert gate["transport_mse"] == pytest.approx(0.5)
    assert gate["mean_squared_error_reduction_vs_drop"] == pytest.approx(2.0)
    assert gate["n_test_groups"] == 2
    assert gate["enabled"] is False
    assert gate["evidence_insufficient"] is True
    assert gate["transport_fit_diagnostic"]["used_for_enable"] is False
    assert "transported acceptance" in gate["disabled_reason"]


def test_server_args_forward_explicit_graph_backend_without_disabling_decode():
    args = bridge_client._server_args(
        _unit("naive_async"),
        {
            "model_roots": {},
            "cuda_graph_backend_prefill": "disabled",
            "cuda_graph_backend_decode": "full",
        },
        "/tmp/adaptation.yaml",
    )
    assert args["cuda_graph_backend_prefill"] == "disabled"
    assert args["cuda_graph_backend_decode"] == "full"
    assert "disable_cuda_graph" not in args


def _write_minimal_lock(tmp_path: Path) -> tuple[Path, Path]:
    repos = ("Qwen/Qwen3-4B", "deepseek-ai/dspark_qwen3_4b_block7")
    lock = Lockfile(
        created_utc="2026-07-31T00:00:00Z",
        hf_snapshots=[
            LockedHFSnapshot(
                repo_id=repo,
                snapshot_sha=("a" if i == 0 else "b") * 40,
                role="target" if i == 0 else "drafter",
            )
            for i, repo in enumerate(repos)
        ],
        environment=LockedEnvironment(
            python_version="3.12",
            torch_version="2.11.0",
        ),
    )
    lock_path = tmp_path / "lightcone.lock.json"
    lock.write(lock_path)
    roots = {}
    for i, repo in enumerate(repos):
        root = tmp_path / f"model-{i}"
        root.mkdir()
        roots[repo] = str(root)
    roots_path = tmp_path / "model-roots.json"
    write_model_roots(roots, roots_path)
    return lock_path, roots_path


def test_native_installer_wheelhouse_is_content_addressed(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "install_native.py"
    namespace = runpy.run_path(str(script))
    metadata = namespace["PINNED_CU129_WHEELS"]
    resolve = namespace["_resolve_pinned_wheel"]
    filename = next(iter(metadata))
    source, artifact = resolve(filename, wheelhouse=None)
    assert source == metadata[filename]["url"]
    assert artifact["sha256"] == metadata[filename]["sha256"]
    path = tmp_path / filename
    path.write_bytes(b"wrong wheel")
    with pytest.raises(SystemExit, match="wheel hash drift"):
        resolve(filename, wheelhouse=tmp_path)
    path.unlink()
    with pytest.raises(SystemExit, match="wheel is missing"):
        resolve(filename, wheelhouse=tmp_path)


def test_native_installer_closes_cuda12_dependency_intersection():
    script = Path(__file__).parents[1] / "scripts" / "install_native.py"
    namespace = runpy.run_path(str(script))
    pins = set(namespace["CU12_COMPATIBILITY_PINS"])
    assert "nvidia-cutlass-dsl==4.6.0" in pins
    assert "numpy==2.3.5" in pins
    assert "fsspec==2026.6.0" in pins
    assert "setuptools==81.0.0" in pins
    assert "protobuf==6.33.5" in pins
    assert "grpcio-health-checking==1.82.0" in pins
    assert "grpcio-reflection==1.82.0" in pins


def test_offline_lock_verifies_git_blob_and_rejects_unknown_hash(tmp_path):
    import hashlib

    root = tmp_path / "model"
    root.mkdir()
    content = b"locked config\n"
    (root / "config.json").write_bytes(content)
    blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    snapshot = LockedHFSnapshot(
        repo_id="org/model",
        snapshot_sha="a" * 40,
        role="target",
        files=[
            LockedFile(
                relpath="config.json",
                size_bytes=len(content),
                sha256=f"gitblob:{blob}",
            )
        ],
    )
    lock = Lockfile(
        created_utc="2026-07-31T00:00:00Z",
        hf_snapshots=[snapshot],
        environment=LockedEnvironment(python_version="3.12", torch_version="2.11"),
    )
    assert verify_lockfile_offline(lock, {"org/model": root}) == ["org/model"]
    (root / "unexpected.py").write_text("raise RuntimeError('not locked')\n")
    with pytest.raises(LockError, match="unexpected files"):
        verify_lockfile_offline(lock, {"org/model": root})
    (root / "unexpected.py").unlink()
    snapshot.files[0].sha256 = "unknown"
    with pytest.raises(LockError, match="no content hash"):
        verify_lockfile_offline(lock, {"org/model": root})


def test_runtime_config_is_materialized_before_gpu_load(tmp_path):
    lock_path, roots_path = _write_minimal_lock(tmp_path)
    overlay = materialize_gpu_runtime(
        _unit(),
        {
            "lockfile_path": str(lock_path),
            "model_roots_path": str(roots_path),
            "runtime_root": str(tmp_path / "runtime"),
            "max_new_tokens": 64,
            "ignore_eos": True,
        },
        tmp_path / "run",
    )
    assert Path(overlay["adaptation_config_path"]).is_file()
    assert overlay["locked_target_revision"] == "a" * 40
    assert "adaptation-telemetry-*.jsonl" in overlay["telemetry_glob"]
    body = Path(overlay["adaptation_config_path"]).read_text()
    assert "logical_delay_rounds: 0" in body
    assert "projection_artifact_path:" in body
    assert "ignore_eos: true" in body
    assert overlay["memory_calibration_sha256"]
    assert overlay["calibrated_reserve_mb"] == 0


def test_runtime_config_materializes_adamw_weight_decay_only_for_adaptation(
    tmp_path,
):
    import yaml

    lock_path, roots_path = _write_minimal_lock(tmp_path)
    common = {
        "lockfile_path": str(lock_path),
        "model_roots_path": str(roots_path),
        "runtime_root": str(tmp_path / "runtime"),
        "weight_decay": 1e-2,
    }

    adaptive = materialize_gpu_runtime(
        _unit("naive_async"), common, tmp_path / "run-adaptive"
    )
    static = materialize_gpu_runtime(
        _unit("static"), common, tmp_path / "run-static"
    )
    adaptive_config = yaml.safe_load(
        Path(adaptive["adaptation_config_path"]).read_text()
    )
    static_config = yaml.safe_load(
        Path(static["adaptation_config_path"]).read_text()
    )

    assert adaptive_config["optimizer"] == "adamw"
    assert adaptive_config["weight_decay"] == pytest.approx(1e-2)
    assert static_config["optimizer"] == "none"
    assert static_config["weight_decay"] == 0.0


def test_runtime_config_materializes_explicit_l3_evaluation_mode(
    tmp_path, monkeypatch
):
    import yaml

    from lightcone_spec.controller import artifact as artifact_module
    from lightcone_spec.methods import registry as registry_module

    lock_path, roots_path = _write_minimal_lock(tmp_path)
    artifact_path = tmp_path / "controller.json"
    artifact_path.write_text("{}")
    sentinel = object()
    validated = []
    monkeypatch.setattr(
        artifact_module,
        "resolve_controller_artifact",
        lambda *_args, **_kwargs: (artifact_path, sentinel),
    )
    monkeypatch.setattr(
        registry_module,
        "validate_controller_artifact",
        lambda cfg, artifact: validated.append((cfg, artifact)),
    )

    overlay = materialize_gpu_runtime(
        _unit("lc_transport"),
        {
            "lockfile_path": str(lock_path),
            "model_roots_path": str(roots_path),
            "runtime_root": str(tmp_path / "runtime"),
            "controller_root": str(tmp_path),
            "trace_capture_max_bytes": 1 << 20,
            "l3_evaluation_only": True,
        },
        tmp_path / "run-l3-evaluation",
    )

    config = yaml.safe_load(Path(overlay["adaptation_config_path"]).read_text())
    assert config["trace"]["l3_evaluation_only"] is True
    assert config["trace"]["trace_capture_max_bytes"] == 1 << 20
    assert validated and validated[0][0].trace.l3_evaluation_only is True
    assert validated[0][1] is sentinel


def test_controller_artifact_is_required_before_runtime_or_model_load(tmp_path):
    lock_path, roots_path = _write_minimal_lock(tmp_path)
    controller_root = tmp_path / "controllers"
    controller_root.mkdir()
    run_dir = tmp_path / "run-controller-missing"
    with pytest.raises(ConfigError, match="bounded p5_cross_backend_trace producer"):
        materialize_gpu_runtime(
            _unit("lc_gate"),
            {
                "lockfile_path": str(lock_path),
                "model_roots_path": str(roots_path),
                "runtime_root": str(tmp_path / "runtime"),
                "controller_root": str(controller_root),
            },
            run_dir,
        )
    assert not (run_dir / "adaptation.runtime.yaml").exists()


def test_preflight_adaptation_reserve_is_sized_before_kv(tmp_path):
    target = tmp_path / "target"
    drafter = tmp_path / "drafter"
    target.mkdir()
    drafter.mkdir()
    (target / "config.json").write_text(
        json.dumps({"hidden_size": 512, "vocab_size": 4096})
    )
    (drafter / "config.json").write_text(
        json.dumps({"markov_rank": 64})
    )
    roots = {
        "Qwen/Qwen3-4B": str(target),
        "deepseek-ai/dspark_qwen3_4b_block7": str(drafter),
    }
    adaptive = _unit("naive_async")
    output_residual = _preflight_adaptation_reserve_mb(adaptive, {}, roots)
    full_tail = _preflight_adaptation_reserve_mb(
        replace(adaptive, trainable_scope="full"), {}, roots
    )

    assert output_residual > 0
    assert full_tail > output_residual
    assert _preflight_adaptation_reserve_mb(_unit("static"), {}, roots) == 0


def test_materialized_capacity_binds_preflight_runtime_and_calibration(tmp_path):
    lock_path, roots_path = _write_minimal_lock(tmp_path)
    overlay = materialize_gpu_runtime(
        _unit("naive_async"),
        {
            "lockfile_path": str(lock_path),
            "model_roots_path": str(roots_path),
            "runtime_root": str(tmp_path / "runtime"),
            "max_running_requests": 20,
        },
        tmp_path / "run",
    )
    config = yaml.safe_load(Path(overlay["adaptation_config_path"]).read_text())
    identity = overlay["memory_calibration_identity"]

    assert overlay["adapter_row_capacity"] == 20
    assert config["runtime"]["adapter_row_capacity"] == 20
    assert identity["schema_version"] == 2
    assert identity["adapter_row_capacity"] == 20
    assert identity["memory_estimator_schema_version"] >= 2
    assert identity["dflash_supervision_fanout_schema_version"] >= 2
    assert len(identity["runtime_implementation_sha256"]) == 64
    server_args = _server_args(
        _unit("naive_async"), overlay, overlay["adaptation_config_path"]
    )
    assert server_args["cuda_graph_max_bs_decode"] == 20


def test_dflash_preflight_reserves_qwen3_batch_snapshot_fanout(tmp_path):
    target = tmp_path / "target"
    drafter = tmp_path / "drafter"
    target.mkdir()
    drafter.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 2_560,
                "vocab_size": 151_936,
                "torch_dtype": "bfloat16",
            }
        )
    )
    (drafter / "config.json").write_text("{}")
    roots = {
        "Qwen/Qwen3-4B": str(target),
        "z-lab/Qwen3-4B-DFlash-b16": str(drafter),
    }
    base = replace(
        _unit("naive_async"),
        model_pair="qwen3_4b_dflash16",
        trainable_scope="tail_lora",
    )
    reserves = [
        _preflight_adaptation_reserve_mb(
            replace(base, concurrency=batch_size),
            {"max_running_requests": batch_size},
            roots,
        )
        for batch_size in (1, 4, 8)
    ]
    # Vectorized candidate rows execute simultaneously: the compact snapshot
    # removes raw/FP32-source clones, while preflight must now reserve each
    # additional row's actual loss/probability workspace.  Under-reserving the
    # old serialized peak would let KV consume memory needed by the batch.
    assert reserves == [90, 302, 585]


def test_runtime_model_hash_verification_is_reused_until_tree_changes(
    tmp_path, monkeypatch
):
    lock_path, roots_path = _write_minimal_lock(tmp_path)
    params = {
        "lockfile_path": str(lock_path),
        "model_roots_path": str(roots_path),
        "runtime_root": str(tmp_path / "runtime"),
    }
    calls = []
    original = runtime_config.verify_lockfile_offline

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    runtime_config._VERIFIED_MODEL_ROOT_STATES.clear()
    monkeypatch.setattr(runtime_config, "verify_lockfile_offline", counted)
    materialize_gpu_runtime(_unit(), params, tmp_path / "run-1")
    materialize_gpu_runtime(_unit(), params, tmp_path / "run-2")
    assert len(calls) == 1

    roots = json.loads(roots_path.read_text())
    Path(roots["Qwen/Qwen3-4B"]).joinpath("unexpected.bin").write_bytes(b"drift")
    with pytest.raises(LockError, match="unexpected files"):
        materialize_gpu_runtime(_unit(), params, tmp_path / "run-3")
    assert len(calls) == 2
    runtime_config._VERIFIED_MODEL_ROOT_STATES.clear()


def test_memory_calibration_is_signature_locked_and_loaded_before_engine(tmp_path):
    lock_path, roots_path = _write_minimal_lock(tmp_path)
    params = {
        "lockfile_path": str(lock_path),
        "model_roots_path": str(roots_path),
        "runtime_root": str(tmp_path / "runtime"),
    }
    first = materialize_gpu_runtime(_unit(), params, tmp_path / "run-1")
    calibration = Path(first["memory_calibration_path"])
    calibration.parent.mkdir(parents=True)
    calibration.write_text(
        json.dumps(
                {
                    "schema_version": 2,
                "identity_sha256": first["memory_calibration_sha256"],
                "identity": first["memory_calibration_identity"],
                "recommended_reserve_mb": 321,
            }
        )
    )
    second = materialize_gpu_runtime(_unit(), params, tmp_path / "run-2")
    assert second["calibrated_reserve_mb"] == 321
    assert "calibrated_reserve_mb: 321" in Path(
        second["adaptation_config_path"]
    ).read_text()

    body = json.loads(calibration.read_text())
    body["identity"]["concurrency"] = 999
    calibration.write_text(json.dumps(body))
    with pytest.raises(ConfigError, match="identity mismatch"):
        materialize_gpu_runtime(_unit(), params, tmp_path / "run-3")


def test_memory_calibration_rejects_legacy_record_schema(tmp_path):
    lock_path, roots_path = _write_minimal_lock(tmp_path)
    params = {
        "lockfile_path": str(lock_path),
        "model_roots_path": str(roots_path),
        "runtime_root": str(tmp_path / "runtime"),
    }
    first = materialize_gpu_runtime(_unit(), params, tmp_path / "run-1")
    calibration = Path(first["memory_calibration_path"])
    calibration.parent.mkdir(parents=True)
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity_sha256": first["memory_calibration_sha256"],
                "identity": first["memory_calibration_identity"],
                "recommended_reserve_mb": 9999,
            }
        )
    )

    with pytest.raises(ConfigError, match="refusing stale reserve"):
        materialize_gpu_runtime(_unit(), params, tmp_path / "run-2")


def test_runtime_config_fails_before_model_load_without_lock(tmp_path):
    with pytest.raises(LockError, match="--lockfile"):
        materialize_gpu_runtime(_unit(), {}, tmp_path / "run")


def test_sglang_source_root_accepts_checkout_or_package_path(tmp_path):
    checkout = tmp_path / "sglang-checkout"
    package = checkout / "python" / "sglang"
    (package / "srt").mkdir(parents=True)

    assert runtime_config._sglang_package_root(checkout) == package.resolve()
    assert runtime_config._sglang_package_root(package) == package.resolve()


def test_runtime_implementation_fingerprint_binds_explicit_execution_surface(
    tmp_path,
):
    roots = {
        "lightcone_spec": tmp_path / "lightcone_spec",
        "sglang": tmp_path / "sglang",
    }
    for component, relative_paths in runtime_config._RUNTIME_IMPLEMENTATION_FILES.items():
        for relative in relative_paths:
            path = roots[component] / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{component}/{relative}\n")

    first = runtime_config.runtime_implementation_fingerprint(
        lightcone_root=roots["lightcone_spec"],
        sglang_root=roots["sglang"],
        locked_reference={"sglang_fork_commit": "deadbeef"},
    )
    critical_files = {
        "lightcone_spec/adapters/losses.py",
        "lightcone_spec/adapters/projections.py",
        "lightcone_spec/controller/artifact.py",
        "lightcone_spec/controller/damping.py",
        "lightcone_spec/controller/gate.py",
        "lightcone_spec/methods/base.py",
        "lightcone_spec/methods/lightcone.py",
        "lightcone_spec/methods/optim.py",
        "lightcone_spec/methods/registry.py",
        "lightcone_spec/methods/simple.py",
        "lightcone_spec/orchestration/executor.py",
        "lightcone_spec/sglang_bridge/hooks.py",
        "lightcone_spec/sglang_bridge/client.py",
        "lightcone_spec/sglang_bridge/static_observer.py",
        "lightcone_spec/transport/apply.py",
        "sglang/srt/entrypoints/engine.py",
        "sglang/srt/speculative/dflash_worker_v2.py",
        "sglang/srt/speculative/dspark_components/dspark_adaptation.py",
        "sglang/srt/speculative/eagle_worker_v2.py",
        "sglang/srt/speculative/tail_adaptation.py",
    }
    assert critical_files <= set(first["files"])
    assert first["locked_reference"] == {"sglang_fork_commit": "deadbeef"}
    assert len(first["sha256"]) == 64

    changed = roots["sglang"] / "srt/speculative/tail_adaptation.py"
    changed.write_text("changed implementation\n")
    second = runtime_config.runtime_implementation_fingerprint(
        lightcone_root=roots["lightcone_spec"],
        sglang_root=roots["sglang"],
        locked_reference={"sglang_fork_commit": "deadbeef"},
    )
    assert second["sha256"] != first["sha256"]
    assert (
        second["files"]["sglang/srt/speculative/tail_adaptation.py"]["sha256"]
        != first["files"]["sglang/srt/speculative/tail_adaptation.py"]["sha256"]
    )


@pytest.mark.parametrize(
    "relative",
    (
        "methods/base.py",
        "methods/simple.py",  # TTS and L0
        "methods/lightcone.py",  # L1, L2 and L3
        "methods/registry.py",
        "methods/optim.py",
    ),
)
def test_runtime_fingerprint_changes_with_method_semantics(tmp_path, relative):
    roots = {
        "lightcone_spec": tmp_path / "lightcone_spec",
        "sglang": tmp_path / "sglang",
    }
    for component, relative_paths in runtime_config._RUNTIME_IMPLEMENTATION_FILES.items():
        for path_relative in relative_paths:
            path = roots[component] / path_relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{component}/{path_relative}\n")

    first = runtime_config.runtime_implementation_fingerprint(
        lightcone_root=roots["lightcone_spec"],
        sglang_root=roots["sglang"],
    )
    changed = roots["lightcone_spec"] / relative
    changed.write_text(changed.read_text() + "# semantic drift\n")
    second = runtime_config.runtime_implementation_fingerprint(
        lightcone_root=roots["lightcone_spec"],
        sglang_root=roots["sglang"],
    )

    key = f"lightcone_spec/{relative}"
    assert second["sha256"] != first["sha256"]
    assert second["files"][key] != first["files"][key]


def test_runtime_fingerprint_fails_closed_when_method_source_is_missing(tmp_path):
    roots = {
        "lightcone_spec": tmp_path / "lightcone_spec",
        "sglang": tmp_path / "sglang",
    }
    missing = "methods/lightcone.py"
    for component, relative_paths in runtime_config._RUNTIME_IMPLEMENTATION_FILES.items():
        for relative in relative_paths:
            if component == "lightcone_spec" and relative == missing:
                continue
            path = roots[component] / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{component}/{relative}\n")

    with pytest.raises(
        ConfigError,
        match=r"runtime implementation file is missing: "
        r"lightcone_spec/methods/lightcone\.py",
    ):
        runtime_config.runtime_implementation_fingerprint(
            lightcone_root=roots["lightcone_spec"],
            sglang_root=roots["sglang"],
        )


def test_missing_dataset_fails_before_engine_construction(tmp_path, monkeypatch):
    lock_path, _roots_path = _write_minimal_lock(tmp_path)
    constructed = []

    class FakeEngine:
        def __init__(self, **_kwargs):
            constructed.append(True)

    fake_sglang = types.ModuleType("sglang")
    fake_srt = types.ModuleType("sglang.srt")
    fake_entrypoints = types.ModuleType("sglang.srt.entrypoints")
    fake_engine = types.ModuleType("sglang.srt.entrypoints.engine")
    fake_engine.Engine = FakeEngine
    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", fake_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints", fake_entrypoints)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.engine", fake_engine)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        bridge_client,
        "_load_prompts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            LockError("dataset is not locked")
        ),
    )

    with pytest.raises(LockError, match="dataset is not locked"):
        bridge_client.run_unit_via_sglang(
            _unit(),
            {
                "lockfile_path": str(lock_path),
                "adaptation_config_path": str(tmp_path / "adaptation.yaml"),
            },
            "run",
        )
    assert not constructed


def test_unknown_engine_startup_exception_is_runtime_failure(tmp_path, monkeypatch):
    lock_path, _roots_path = _write_minimal_lock(tmp_path)

    class FakeEngine:
        def __init__(self, **_kwargs):
            raise RuntimeError("startup exploded")

    fake_engine = types.ModuleType("sglang.srt.entrypoints.engine")
    fake_engine.Engine = FakeEngine
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(
        sys.modules, "sglang.srt.entrypoints", types.ModuleType("sglang.srt.entrypoints")
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.engine", fake_engine)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bridge_client, "_load_prompts", lambda *_args: [])

    with pytest.raises(RuntimeGpuFailure, match="RuntimeError.*startup exploded"):
        bridge_client.run_unit_via_sglang(
            _unit(),
            {
                "lockfile_path": str(lock_path),
                "adaptation_config_path": str(tmp_path / "adaptation.yaml"),
            },
            "run",
        )


def test_unknown_request_exception_is_runtime_failure_and_shuts_down(
    tmp_path, monkeypatch
):
    lock_path, _roots_path = _write_minimal_lock(tmp_path)
    events = []

    class FakeEngine:
        def __init__(self, **_kwargs):
            pass

        def get_server_info(self):
            return {"internal_states": []}

        def shutdown(self):
            events.append("shutdown")

    fake_engine = types.ModuleType("sglang.srt.entrypoints.engine")
    fake_engine.Engine = FakeEngine
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(
        sys.modules, "sglang.srt.entrypoints", types.ModuleType("sglang.srt.entrypoints")
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.engine", fake_engine)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        bridge_client,
        "_load_prompts",
        lambda *_args: [
            {"sample_id": "sample", "prompt": "prompt", "input_ids": None}
        ],
    )
    monkeypatch.setattr(
        bridge_client,
        "_run_streaming_pool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("request RPC exploded")
        ),
    )

    with pytest.raises(RuntimeGpuFailure, match="ValueError.*request RPC exploded"):
        bridge_client.run_unit_via_sglang(
            _unit(),
            {
                "lockfile_path": str(lock_path),
                "adaptation_config_path": str(tmp_path / "adaptation.yaml"),
            },
            "run",
        )
    assert events == ["shutdown"]


@pytest.mark.parametrize(
    "error",
    (ResourceSkip("capacity"), ExactnessViolation("version mismatch")),
)
def test_declared_lightcone_outcomes_are_not_normalized(tmp_path, monkeypatch, error):
    lock_path, _roots_path = _write_minimal_lock(tmp_path)

    class FakeEngine:
        pass

    fake_engine = types.ModuleType("sglang.srt.entrypoints.engine")
    fake_engine.Engine = FakeEngine
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(
        sys.modules, "sglang.srt.entrypoints", types.ModuleType("sglang.srt.entrypoints")
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.engine", fake_engine)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        bridge_client,
        "_load_prompts",
        lambda *_args: (_ for _ in ()).throw(error),
    )

    with pytest.raises(type(error)) as raised:
        bridge_client.run_unit_via_sglang(
            _unit(),
            {
                "lockfile_path": str(lock_path),
                "adaptation_config_path": str(tmp_path / "adaptation.yaml"),
            },
            "run",
        )
    assert raised.value is error


def test_target_only_adaptation_fallback_fails_before_first_prompt(
    tmp_path, monkeypatch
):
    lock_path, _roots_path = _write_minimal_lock(tmp_path)
    calls = []
    reason = "dflash_exact_stochastic_sampling_kernel_unavailable"

    class FakeEngine:
        def __init__(self, **_kwargs):
            self.server_args = types.SimpleNamespace(
                speculative_adaptation_fallback_reason=reason
            )

        def get_server_info(self):
            return {
                "speculative_adaptation_fallback_reason": reason,
                "internal_states": [],
            }

        def shutdown(self):
            calls.append("shutdown")

    fake_sglang = types.ModuleType("sglang")
    fake_srt = types.ModuleType("sglang.srt")
    fake_entrypoints = types.ModuleType("sglang.srt.entrypoints")
    fake_engine = types.ModuleType("sglang.srt.entrypoints.engine")
    fake_engine.Engine = FakeEngine
    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", fake_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints", fake_entrypoints)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.engine", fake_engine)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bridge_client, "_load_prompts", lambda *_args: [])
    monkeypatch.setattr(
        bridge_client,
        "_run_streaming_pool",
        lambda *_args, **_kwargs: calls.append("prompt"),
    )
    adaptation_path = tmp_path / "run" / "adaptation.yaml"

    with pytest.raises(RuntimeGpuFailure, match="target-only.*fallback_reason"):
        bridge_client.run_unit_via_sglang(
            _unit(),
            {
                "lockfile_path": str(lock_path),
                "adaptation_config_path": str(adaptation_path),
            },
            "run",
        )

    assert calls == ["shutdown"]
    evidence = json.loads((adaptation_path.parent / "server-info.json").read_text())
    assert evidence["speculative_adaptation_fallback_reason"] == reason


def test_runtime_request_group_excludes_run_delay_and_repetition():
    first = _runtime_request_id(
        run_id="delay-0", repetition=0, index=3, sample_id="alpaca-19"
    )
    second = _runtime_request_id(
        run_id="delay-5", repetition=4, index=99, sample_id="alpaca-19"
    )
    assert first != second
    assert _sequence_group_from_request_id(first) == _sequence_group_from_request_id(
        second
    )
    legacy = f"lightcone-g{'a' * 64}-profile-run-0-0"
    assert _sequence_group_from_request_id(legacy) == "a" * 64
    assert _sequence_group_from_request_id("external-request") == "external-request"


def test_runtime_request_group_pairs_p5_context_checkpoints():
    short = _runtime_request_id(
        run_id="p5", repetition=0, index=0, sample_id="task-19:ctx-4096"
    )
    long = _runtime_request_id(
        run_id="p5", repetition=0, index=0, sample_id="task-19:ctx-16384"
    )
    assert short != long
    assert _sequence_group_from_request_id(short) == _sequence_group_from_request_id(
        long
    )


def test_measured_requests_use_one_full_pool_per_context_across_repetitions(
    tmp_path, monkeypatch
):
    lock_path, _roots_path = _write_minimal_lock(tmp_path)
    prompts = [
        {"sample_id": f"{sample}:ctx-{context}", "prompt": None, "input_ids": [1]}
        for context in (512, 16384)
        for sample in ("task-a", "task-b")
    ]
    pool_calls = []
    engine_args = []
    captured_summaries = []

    class FakeEngine:
        def __init__(self, **kwargs):
            engine_args.append(kwargs)

        def get_server_info(self):
            return {"internal_states": []}

        def shutdown(self):
            pass

    class FakeSampler:
        def stop(self):
            return []

    def fake_pool(
        _engine,
        jobs,
        _sampling_params,
        *,
        run_id,
        concurrency,
        request_kind="measured",
        **_kwargs,
    ):
        pool_calls.append(
            {
                "jobs": list(jobs),
                "run_id": run_id,
                "concurrency": concurrency,
                "request_kind": request_kind,
            }
        )
        return [
            {
                "output": {
                    "text": "ok",
                    "meta_info": {
                        "id": _runtime_request_id(
                            run_id=run_id,
                            repetition=job["repetition"],
                            index=index,
                            sample_id=job["prompt"]["sample_id"],
                        ),
                        "completion_tokens": 2,
                    },
                },
                "itl_ms": [1.0],
                "request_wall_s": 0.01,
                "ttft_ms": 1.0,
                **job,
            }
            for index, job in enumerate(jobs)
        ]

    def capture_rows(_unit, _run_id, _paths, summaries, *_args, **_kwargs):
        captured_summaries.extend(summaries)
        return {"summaries": summaries}

    fake_engine_module = types.ModuleType("sglang.srt.entrypoints.engine")
    fake_engine_module.Engine = FakeEngine
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(
        sys.modules, "sglang.srt.entrypoints", types.ModuleType("sglang.srt.entrypoints")
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.entrypoints.engine", fake_engine_module
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(bridge_client, "_load_prompts", lambda *_args: prompts)
    monkeypatch.setattr(bridge_client, "_run_streaming_pool", fake_pool)
    monkeypatch.setattr(
        bridge_client, "_GpuSystemSampler", lambda *_args, **_kwargs: FakeSampler()
    )
    monkeypatch.setattr(bridge_client, "_telemetry_to_rows", capture_rows)

    unit = replace(_unit(), seed=17, concurrency=3)
    bridge_client.run_unit_via_sglang(
        unit,
        {
            "lockfile_path": str(lock_path),
            "adaptation_config_path": str(tmp_path / "adaptation.yaml"),
            "benchmark_repetitions": 3,
            "warmup_prompts": 1,
        },
        "paired-run",
    )

    measured = [call for call in pool_calls if call["request_kind"] == "measured"]
    assert len(measured) == 2
    assert [call["concurrency"] for call in measured] == [3, 3]
    assert [call["run_id"] for call in measured] == [
        "paired-run-cg0",
        "paired-run-cg1",
    ]
    for call, context in zip(measured, (512, 16384)):
        assert [
            (job["prompt"]["sample_id"], job["repetition"])
            for job in call["jobs"]
        ] == [
            (f"{sample}:ctx-{context}", repetition)
            for repetition in range(3)
            for sample in ("task-a", "task-b")
        ]
    assert engine_args[0]["random_seed"] == 17
    measured_seeds = [
        job["sampling_seed"] for call in measured for job in call["jobs"]
    ]
    assert len(measured_seeds) == len(set(measured_seeds))
    assert len(captured_summaries) == 12
    assert len({row["runtime_request_id"] for row in captured_summaries}) == 12
    assert {row["run_total_tokens"] for row in captured_summaries} == {12}
    assert all("performance_evidence" in row for row in captured_summaries)

    groups = {
        (
            row["sample_id"].split(":ctx-", 1)[0],
            int(row["sample_id"].split(":ctx-", 1)[1].split(":repeat-", 1)[0]),
            row["repetition"],
        ): _sequence_group_from_request_id(row["runtime_request_id"])
        for row in captured_summaries
    }
    for sample in ("task-a", "task-b"):
        assert {
            groups[(sample, context, repetition)]
            for context in (512, 16384)
            for repetition in range(3)
        } == {groups[(sample, 512, 0)]}
    assert groups[("task-a", 512, 0)] != groups[("task-b", 512, 0)]


def test_deferred_telemetry_reports_materialization_errors(tmp_path):
    class BadEvent:
        @staticmethod
        def synchronize():
            raise RuntimeError("event failed")

    sink = TelemetrySink(tmp_path / "telemetry.jsonl")
    record = RoundTelemetry(
        request_id="rid",
        round_id=0,
        active_version=0,
        proposal_version=0,
        draft_tokens=1,
        accepted_drafts=0,
        committed_per_verify=1,
        target_calls=1,
        draft_cuda_us=0,
        verify_cuda_us=0,
        accept_cuda_us=0,
        draft_cpu_us=1,
        verify_cpu_us=1,
        rng_substream_id="rid:0",
        version_canary_ok=True,
    )
    sink.emit_round_deferred(record, {"accept_end": BadEvent()})
    assert sink.flush(timeout_s=1)
    assert sink.health() == {
        "error_count": 1,
        "last_error": "RuntimeError: event failed",
    }
    sink.close()


def test_telemetry_uses_runtime_request_id_and_never_zero_fills(tmp_path):
    telemetry = tmp_path / "adaptation-telemetry-p1-r0.jsonl"
    telemetry.write_text(
        json.dumps(
            _round_record(
                request_id="warmup-rid",
                accepted_drafts=1,
                committed_per_verify=2,
                draft_cuda_us=1.0,
                verify_cuda_us=1.0,
                accept_cuda_us=1.0,
                draft_cpu_us=1.0,
                verify_cpu_us=1.0,
                rng_substream_id="warmup",
                prefix_pos_after=12,
                round_wall_us=99_000.0,
            )
        )
        + "\n"
        +
        json.dumps(
            _round_record(
                cache_policy="frozen_old_kv",
                proposal_weight_version=7,
                kv_version_min=2,
                kv_version_max=7,
                kv_append_version=7,
                cache_version_canary_ok=True,
            )
        )
        + "\n"
        + json.dumps(
            {
                "kind": "update",
                "request_id": "runtime-rid",
                "update_id": "runtime-rid-u0",
                "source_round": 0,
                "source_version": 0,
                "source_training_loss": 1.75,
                "source_expected_accepted_prefix": 3.25,
                "source_prefix_len": 16384,
                "snapshot_ts_us": 1.0,
                "teacher_ts_us": 2.0,
                "launch_ts_us": 3.0,
                "done_ts_us": 4.0,
                "commit_ts_us": 5.0,
                "exposure_ts_us": 5.0,
                "active_version_at_arrival": 0,
                "staging_version": 1,
                "published_version": 1,
                "decision": "discard_noop_publish",
                "damping_factor": 0.0,
                "effective_delay_rounds": 1,
                "delay_tokens": 7,
                "delay_wall_us": 321.0,
                "delay_versions": 0,
                "rho_path": 1.25,
                "endpoint_distance": 0.75,
                "parameter_displacement": 0.125,
                "threshold": 0.2,
                "grad_clip_scale": 0.5,
                "optimizer_step": 3,
                "candidate_cuda_us": 30.0,
                "candidate_batch_size": 8,
                "backward_cuda_us": 20.0,
                "optimizer_cuda_us": 5.0,
                "publish_cuda_us": 2.5,
                "gradient_weight_version": 7,
                "gradient_kv_version_min": 2,
                "gradient_kv_version_max": 7,
                "gradient_version_canary_ok": True,
            }
        )
        + "\n"
    )
    summaries = [
        {
            "sample_id": "dataset-id",
            "runtime_request_id": "runtime-rid",
            "meta": {"completion_tokens": 7},
            "batch_wall_s": 0.01,
        }
    ]
    with pytest.raises(RuntimeGpuFailure, match="system telemetry is missing"):
        _telemetry_to_rows(
            _unit(),
            "run-missing-system-telemetry",
            [telemetry],
            summaries,
            0.01,
            123,
        )
    rows = _telemetry_to_rows(
        _unit(),
        "run",
        [telemetry],
        summaries,
        0.01,
        123,
        system_samples=_system_samples(),
    )
    assert len(rows["rounds"]) == 1
    round_row = rows["rounds"][0]
    assert round_row["prefix_len_before"] == 10
    assert round_row["verify_len"] == 9
    assert round_row["batch_size"] == 4
    assert round_row["offered_concurrency"] == 4
    assert round_row["round_wall_us"] == 1000.0
    assert round_row["prefix_feature_exact"] is True
    assert round_row["cache_policy"] == "frozen_old_kv"
    assert round_row["proposal_weight_version"] == 7
    assert round_row["kv_version_min"] == 2
    assert round_row["kv_version_max"] == 7
    assert round_row["kv_append_version"] == 7
    assert round_row["cache_version_canary_ok"] is True
    summary = rows["request_summary"][0]
    assert rows["system_samples"][0]["hbm_used_bytes"] == 123
    assert summary["mean_accepted_drafts"] == 6
    assert summary["target_calls_per_output_token"] == pytest.approx(1 / 7)
    assert summary["p99_round_ms"] == 1.0
    assert summary["decode_tps"] > 0
    assert summary["estimated_perf_scope"] == "target_model_only"
    assert rows["decisions"][0]["decision"] == "discard_noop_publish"
    assert rows["decisions"][0]["damping_factor"] == 0.0
    assert rows["decisions"][0]["rho_path"] == 1.25
    assert rows["decisions"][0]["threshold"] == 0.2
    assert rows["updates"][0]["delay_tokens"] == 7
    assert rows["updates"][0]["delay_wall_us"] == 321.0
    assert rows["updates"][0]["grad_clip_scale"] == 0.5
    assert rows["updates"][0]["optimizer_step"] == 3
    assert rows["updates"][0]["backward_cuda_us"] == 20.0
    assert rows["updates"][0]["optimizer_cuda_us"] == 5.0
    assert rows["updates"][0]["candidate_batch_size"] == 8
    assert rows["updates"][0]["publish_cuda_us"] == 2.5
    assert rows["updates"][0]["gradient_weight_version"] == 7
    assert rows["updates"][0]["gradient_kv_version_min"] == 2
    assert rows["updates"][0]["gradient_kv_version_max"] == 7
    assert rows["updates"][0]["gradient_version_canary_ok"] is True
    assert rows["updates"][0]["source_training_loss"] == 1.75
    assert rows["updates"][0]["source_expected_accepted_prefix"] == 3.25
    assert rows["updates"][0]["source_prefix_len"] == 16384

    from lightcone_spec.artifacts.rundir import RunDirectory

    rd = RunDirectory(tmp_path, "telemetry-roundtrip")
    rd.create({"unit_id": _unit().unit_id})
    rd.write_table("rounds", rows["rounds"])
    rd.write_table("updates", rows["updates"])
    parquet_round = rd.read_table("rounds").to_pylist()[0]
    parquet_update = rd.read_table("updates").to_pylist()[0]
    for field_name in (
        "prefix_len_before",
        "verify_len",
        "batch_size",
        "offered_concurrency",
        "round_wall_us",
        "prefix_feature_exact",
        "algorithmic_censored",
        "cache_policy",
        "proposal_weight_version",
        "kv_version_min",
        "kv_version_max",
        "kv_append_version",
        "cache_version_canary_ok",
    ):
        assert parquet_round[field_name] == round_row[field_name]
    for field_name in (
        "backward_cuda_us",
        "optimizer_cuda_us",
        "candidate_batch_size",
        "publish_cuda_us",
        "source_training_loss",
        "source_expected_accepted_prefix",
        "source_prefix_len",
        "gradient_weight_version",
        "gradient_kv_version_min",
        "gradient_kv_version_max",
        "gradient_version_canary_ok",
    ):
        assert parquet_update[field_name] == rows["updates"][0][field_name]

    fallback_rows = _telemetry_to_rows(
        _unit(method="lc_gate"),
        "run-gate-fallback",
        [telemetry],
        summaries,
        0.01,
        123,
        system_samples=[
            {
                "timestamp_us": 1.0,
                "gpu_index": 0,
                "hbm_used_bytes": 123,
                "sm_occupancy": None,
                "gpu_utilization": 50.0,
                "power_watts": 100.0,
                "energy_joules_delta": 1.0,
                "main_stream_active": True,
                "side_stream_active": True,
                "stream_contention_class": "realistic_async",
                "sync_us_delta": 0.0,
            }
        ],
        server_info={
            "internal_states": [
                {
                    "dspark_info_record": {
                        "adaptation": {"controller_static_fallback": True}
                    }
                }
            ]
        },
    )
    # A controller fallback must not rewrite historical sampler observations.
    assert fallback_rows["system_samples"][0]["side_stream_active"] is True
    assert (
        fallback_rows["system_samples"][0]["stream_contention_class"]
        == "realistic_async"
    )
    assert (
        fallback_rows["system_samples"][0]["activity_provenance"]
        == "legacy_unspecified"
    )

    text = telemetry.read_text().replace(
        '"version_canary_ok": true', '"version_canary_ok": false'
    )
    telemetry.write_text(text)
    failed = _telemetry_to_rows(
        _unit(),
        "run-failed",
        [telemetry],
        summaries,
        0.01,
        123,
        system_samples=_system_samples(),
    )
    assert failed["_status"] == "failed_exactness"
    assert failed["request_summary"][0]["decode_tps"] == 0.0

    records = [json.loads(line) for line in telemetry.read_text().splitlines()]
    for record in records:
        if record.get("kind") == "round":
            record["version_canary_ok"] = True
        elif record.get("kind") == "update":
            record["failure_reason"] = "non_finite_candidate"
    telemetry.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    numerical_failure = _telemetry_to_rows(
        _unit(),
        "run-numerical-failure",
        [telemetry],
        summaries,
        0.01,
        123,
        system_samples=_system_samples(),
    )
    assert numerical_failure["_status"] == "failed_runtime"
    assert numerical_failure["request_summary"][0]["decode_tps"] == 0.0

    for record in records:
        if record.get("kind") == "round":
            record["prefix_feature_exact"] = False
        elif record.get("kind") == "update":
            record["failure_reason"] = None
    telemetry.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    prefix_failure = _telemetry_to_rows(
        _unit(),
        "run-prefix-failure",
        [telemetry],
        summaries,
        0.01,
        123,
        system_samples=_system_samples(),
    )
    assert prefix_failure["_status"] == "failed_exactness"
    assert prefix_failure["request_summary"][0]["decode_tps"] == 0.0


def test_empty_or_zero_token_telemetry_fails_closed(tmp_path):
    with pytest.raises(RuntimeGpuFailure, match="missing or empty"):
        _telemetry_to_rows(_unit(), "run", [], [], 1.0, 0)

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            _round_record(
                request_id="r",
                draft_tokens=1,
                accepted_drafts=0,
                committed_per_verify=1,
                prefix_pos_after=11,
                verify_len=2,
            )
        )
        + "\n"
    )
    with pytest.raises(RuntimeGpuFailure, match="completion_tokens"):
        _telemetry_to_rows(
            _unit(),
            "run",
            [telemetry],
            [{"sample_id": "s", "meta": {"completion_tokens": 0}}],
            1.0,
            1,
            system_samples=_system_samples(1),
        )


def test_real_zero_acceptance_is_valid_but_missing_or_impossible_evidence_fails(
    tmp_path,
):
    telemetry = tmp_path / "adaptation-telemetry-zero.jsonl"
    telemetry.write_text(
        json.dumps(
            _round_record(
                accepted_drafts=0,
                committed_per_verify=1,
                prefix_pos_after=11,
            )
        )
        + "\n"
    )
    summaries = [
        {
            "sample_id": "hard-prompt",
            "runtime_request_id": "runtime-rid",
            "meta": {"completion_tokens": 1},
            "batch_wall_s": 0.01,
        }
    ]
    rows = _telemetry_to_rows(
        _unit(),
        "run-zero-acceptance",
        [telemetry],
        summaries,
        0.01,
        123,
        system_samples=_system_samples(),
    )
    assert rows["rounds"][0]["accepted_drafts"] == 0
    assert rows["request_summary"][0]["mean_accepted_drafts"] == 0

    exactness_fallback = _telemetry_to_rows(
        _unit("lc_gate"),
        "run-exactness-fallback",
        [telemetry],
        summaries,
        0.01,
        123,
        system_samples=_system_samples(),
        server_info={
            "internal_states": [
                {
                    "speculative_adaptation_info_record": {
                        "adaptation": {"fallback_counts": {"exactness": 1}}
                    }
                }
            ]
        },
    )
    assert exactness_fallback["_status"] == "failed_exactness"

    runtime_fallback = _telemetry_to_rows(
        _unit("lc_gate"),
        "run-runtime-fallback",
        [telemetry],
        summaries,
        0.01,
        123,
        system_samples=_system_samples(),
        server_info={
            "internal_states": [
                {
                    "speculative_adaptation_info_record": {
                        "adaptation": {
                            "fallback_counts": {"adapter_slots_exhausted": 1}
                        }
                    }
                }
            ]
        },
    )
    assert runtime_fallback["_status"] == "failed_runtime"

    telemetry.write_text(
        json.dumps(_round_record())
        + "\n"
        + json.dumps(
            {
                "kind": "update",
                "request_id": "runtime-rid",
                "update_id": "u-missing-timing",
                "source_round": 0,
                "source_version": 0,
                "snapshot_ts_us": 1.0,
                "launch_ts_us": 2.0,
                "done_ts_us": 3.0,
                "candidate_cuda_us": 1.0,
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeGpuFailure, match="missing backward_cuda_us"):
        _telemetry_to_rows(
            _unit("lc_gate"),
            "run-missing-component-timing",
            [telemetry],
            summaries,
            0.01,
            123,
        )

    missing = _round_record()
    missing.pop("committed_per_verify")
    telemetry.write_text(json.dumps(missing) + "\n")
    with pytest.raises(RuntimeGpuFailure, match="missing required fields"):
        _telemetry_to_rows(
            _unit(), "run-missing", [telemetry], summaries, 0.01, 123
        )

    missing = _round_record()
    missing.pop("algorithmic_censored")
    telemetry.write_text(json.dumps(missing) + "\n")
    with pytest.raises(RuntimeGpuFailure, match="algorithmic_censored"):
        _telemetry_to_rows(
            _unit(), "run-missing-censor", [telemetry], summaries, 0.01, 123
        )

    telemetry.write_text(
        json.dumps(_round_record(algorithmic_censored=0)) + "\n"
    )
    with pytest.raises(RuntimeGpuFailure, match="explicit booleans"):
        _telemetry_to_rows(
            _unit(), "run-invalid-censor", [telemetry], summaries, 0.01, 123
        )

    telemetry.write_text(
        json.dumps(
            _round_record(
                accepted_drafts=8,
                committed_per_verify=9,
                verify_len=2,
            )
        )
        + "\n"
    )
    with pytest.raises(RuntimeGpuFailure, match="exceeds the verified/drafted bound"):
        _telemetry_to_rows(
            _unit(), "run-impossible", [telemetry], summaries, 0.01, 123
        )


@pytest.mark.parametrize(
    ("algorithm", "model_pair"),
    (("DFLASH", "qwen3_4b_dflash16"), ("EAGLE", "llama2_7b_eagle")),
)
def test_tail_info_record_reports_memory_to_request_summary(
    tmp_path, algorithm, model_pair
):
    telemetry = tmp_path / f"adaptation-telemetry-{algorithm.lower()}.jsonl"
    telemetry.write_text(json.dumps(_round_record()) + "\n")
    summaries = [
        {
            "sample_id": "memory-prompt",
            "runtime_request_id": "runtime-rid",
            "meta": {"completion_tokens": 7},
            "batch_wall_s": 0.01,
        }
    ]
    unit = replace(_unit("tts"), model_pair=model_pair)
    rows = _telemetry_to_rows(
        unit,
        f"run-{algorithm.lower()}",
        [telemetry],
        summaries,
        0.01,
        123,
        system_samples=_system_samples(),
        server_info={
            "internal_states": [
                {
                    "speculative_adaptation_info_record": {
                        "adaptation": {
                            "algorithm": algorithm,
                            "memory": {
                                "fixed_bytes": 44_894_720,
                                "reserve_bytes": 94_371_840,
                            },
                        }
                    }
                }
            ]
        },
    )

    summary = rows["request_summary"][0]
    assert summary["adaptation_fixed_bytes"] == 44_894_720
    assert summary["adaptation_reserve_bytes"] == 94_371_840


def test_static_info_record_keeps_adaptation_memory_zero(tmp_path):
    telemetry = tmp_path / "adaptation-telemetry-static.jsonl"
    telemetry.write_text(json.dumps(_round_record()) + "\n")
    rows = _telemetry_to_rows(
        replace(_unit(), model_pair="qwen3_4b_dflash16"),
        "run-static",
        [telemetry],
        [
            {
                "sample_id": "static-prompt",
                "runtime_request_id": "runtime-rid",
                "meta": {"completion_tokens": 7},
                "batch_wall_s": 0.01,
            }
        ],
        0.01,
        123,
        system_samples=_system_samples(),
        server_info={
            "internal_states": [
                {
                    "speculative_adaptation_info_record": {
                        "adaptation": {
                            "static_observer": True,
                            "memory": {
                                "fixed_bytes": 0,
                                "reserve_bytes": 0,
                            },
                        }
                    }
                }
            ]
        },
    )

    summary = rows["request_summary"][0]
    assert summary["adaptation_fixed_bytes"] == 0
    assert summary["adaptation_reserve_bytes"] == 0


def test_streaming_chunk_issues_concurrent_requests_and_records_itl():
    class Engine:
        def __init__(self):
            self.loop = asyncio.new_event_loop()
            self.active = 0
            self.peak = 0
            self.rids = []
            self.sampling_params = []

        async def async_generate(self, *, rid, sampling_params, **_kwargs):
            self.rids.append(rid)
            self.sampling_params.append(sampling_params)
            async def stream():
                self.active += 1
                self.peak = max(self.peak, self.active)
                await asyncio.sleep(0.002)
                yield {"text": "a", "meta_info": {"id": rid, "completion_tokens": 1}}
                await asyncio.sleep(0.002)
                yield {"text": "ab", "meta_info": {"id": rid, "completion_tokens": 2}}
                self.active -= 1

            return stream()

    engine = Engine()
    try:
        result = _run_streaming_chunk(
            engine,
            [
                {"sample_id": "a", "prompt": "one"},
                {"sample_id": "b", "prompt": "two"},
            ],
            {"max_new_tokens": 2},
            repetition=0,
            run_id="run",
        )
        measured_rids = list(engine.rids)
        _run_streaming_pool(
            engine,
            [
                {
                    "prompt": {"sample_id": "a", "prompt": "one"},
                    "repetition": 0,
                    "sampling_seed": 123,
                }
            ],
            {"max_new_tokens": 2},
            run_id="run",
            concurrency=1,
            request_kind="warmup",
        )
    finally:
        engine.loop.close()
    assert engine.peak == 2
    assert all(rid.startswith("lightcone-g") for rid in measured_rids)
    assert engine.rids[-1].startswith("lightcone-warmup-")
    assert engine.sampling_params[-1]["sampling_seed"] == 123
    assert [item["output"]["text"] for item in result] == ["ab", "ab"]
    assert all(item["itl_ms"] and item["itl_ms"][0] > 0 for item in result)


def test_streaming_itl_preserves_speculative_chunk_arrival_semantics():
    class Engine:
        def __init__(self):
            self.loop = asyncio.new_event_loop()

        async def async_generate(self, **_kwargs):
            async def stream():
                yield {
                    "text": "abc",
                    "meta_info": {"completion_tokens": 3},
                }
                await asyncio.sleep(0.002)
                yield {
                    "text": "abcde",
                    "meta_info": {"completion_tokens": 5},
                }

            return stream()

    engine = Engine()
    try:
        result = _run_streaming_pool(
            engine,
            [
                {
                    "prompt": {"sample_id": "burst", "prompt": "one"},
                    "repetition": 0,
                    "sampling_seed": 7,
                }
            ],
            {"max_new_tokens": 5},
            run_id="burst-run",
            concurrency=1,
        )[0]
    finally:
        engine.loop.close()

    # Five emitted tokens have four inter-token intervals.  Tokens co-emitted
    # in the first and second chunks have zero observable delay; only the first
    # token in the second chunk waits for the inter-chunk interval.
    assert len(result["itl_ms"]) == 4
    assert sum(value == 0.0 for value in result["itl_ms"]) == 3
    assert sum(value > 0.0 for value in result["itl_ms"]) == 1


def test_streaming_chunk_timeout_fails_closed_and_cancels_tasks():
    class Engine:
        def __init__(self):
            self.loop = asyncio.new_event_loop()
            self.cancelled = False

        async def async_generate(self, **_kwargs):
            owner = self

            async def stream():
                try:
                    await asyncio.sleep(60)
                    yield {"text": "late", "meta_info": {"completion_tokens": 1}}
                finally:
                    owner.cancelled = True

            return stream()

    engine = Engine()
    try:
        with pytest.raises(RuntimeGpuFailure, match="exceeded"):
            _run_streaming_chunk(
                engine,
                [{"sample_id": "a", "prompt": "one"}],
                {"max_new_tokens": 1},
                repetition=0,
                run_id="run",
                timeout_s=0.01,
            )
        assert engine.cancelled
    finally:
        engine.loop.close()


def test_telemetry_flush_drains_deferred_work(tmp_path):
    sink = TelemetrySink(tmp_path / "telemetry.jsonl")
    marker = tmp_path / "deferred.done"
    sink.defer(lambda: marker.write_text("done"))
    assert sink.flush(timeout_s=2.0)
    sink.close()
    assert marker.read_text() == "done"


def test_exactness_evidence_is_positive_and_fail_closed(tmp_path):
    path = tmp_path / "nested" / "adaptation-telemetry-p1-r0.jsonl"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "kind": "round",
                "request_id": "r",
                "version_canary_ok": True,
                "prefix_feature_exact": True,
            }
        )
        + "\n"
    )
    assert _scan_exactness_evidence(tmp_path)["verified"] is True
    path.write_text(
        json.dumps(
            {
                "kind": "round",
                "request_id": "r",
                "version_canary_ok": False,
                "prefix_feature_exact": True,
            }
        )
        + "\n"
    )
    report = _scan_exactness_evidence(tmp_path)
    assert report["verified"] is False
    assert report["violation_count"] == 1

    path.write_text(
        json.dumps(
            {
                "kind": "round",
                "request_id": "r",
                "version_canary_ok": True,
                "prefix_feature_exact": False,
            }
        )
        + "\n"
    )
    report = _scan_exactness_evidence(tmp_path)
    assert report["verified"] is False
    assert report["violations"][0]["reason"] == "prefix_feature_inexact"


def test_gpu_manifests_and_hashes_are_self_consistent():
    manifest_root = Path(__file__).parents[1] / "manifests" / "smoke"
    for name in (
        "smoke_gpu_qwen3_4b.json",
        "smoke_gpu_qwen3_4b_controller.json",
        "trace_gpu_qwen3_4b.json",
        "trace_gpu_qwen3_4b_paired.json",
        "benchmark_gpu_qwen3_4b.json",
        "profile_gpu_qwen3_4b.json",
    ):
        manifest = ExperimentManifest.load(manifest_root / name)
        assert manifest.units
        assert all(unit.model_pair == "qwen3_4b_dspark7" for unit in manifest.units)
        if name == "trace_gpu_qwen3_4b_paired.json":
            assert {unit.method for unit in manifest.units} == {
                "naive_async",
                "tts",
            }
            assert manifest.engine_params["trace_producer_methods"] == [
                "naive_async",
                "tts",
            ]
    tune = ExperimentManifest.load(
        Path(__file__).parents[1]
        / "manifests/load_tune/load_tune_gpu_qwen3_4b.json"
    )
    assert {unit.concurrency for unit in tune.units} == {1, 2, 4, 8, 16, 32, 48}
    assert tune.engine_params["benchmark_repetitions"] == 5
    p5 = ExperimentManifest.load(
        Path(__file__).parents[1]
        / "manifests/p5/p5_long_context_acceptance_engine_reuse.json"
    )
    assert len(p5.units) == 75
    assert {unit.method for unit in p5.units} == {
        "static", "tts", "naive_async", "lc_gate", "lc_damp"
    }
    assert "lc_transport" not in {unit.method for unit in p5.units}
    assert p5.engine_params["benchmark_repetitions"] == 5
    assert set(p5.engine_params["p5_context_lengths"]) == {
        512, 1024, 2048, 4096, 8192, 16384, 32768
    }
    assert {unit.concurrency for unit in p5.units} == {1, 4, 16}
    assert sum(unit.allow_resource_skip for unit in p5.units) == 15
    continuous = ExperimentManifest.load(
        Path(__file__).parents[1]
        / "manifests/p5/p5_priority_dflash_continuous40k_calibration_v1.json"
    )
    assert {unit.method for unit in continuous.units} == {
        "static",
        "tts",
        "naive_async",
    }
    assert {unit.stride for unit in continuous.units} == {1}
    assert {unit.model_pair for unit in continuous.units} == {
        "qwen3_4b_dflash16"
    }
    assert continuous.engine_params["p5_context_lengths"] == [128]
    assert continuous.engine_params["p5_continuous_prefix_windows"] == [
        [128, 4096],
        [4096, 10240],
        [10240, 20480],
        [20480, 30720],
        [30720, 40944],
    ]
    assert continuous.engine_params["prompt_limit"] == 2
    assert continuous.engine_params["benchmark_repetitions"] == 5
    assert (
        continuous.engine_params["p5_context_lengths"][0]
        + continuous.engine_params["max_new_tokens"]
        == continuous.engine_params["p5_continuous_prefix_windows"][-1][1]
    )
    p5_14b = ExperimentManifest.load(
        Path(__file__).parents[1]
        / "manifests/p5/p5_long_context_screen_qwen3_14b_40k_c1_lr1e2.json"
    )
    assert {unit.method for unit in p5_14b.units} == {
        "static",
        "tts",
        "naive_async",
    }
    assert {unit.model_pair for unit in p5_14b.units} == {
        "qwen3_14b_dspark7"
    }
    assert p5_14b.lockfile_sha256 is not None


def test_doctor_no_network_is_read_only(tmp_path):
    report = collect_doctor_report(tmp_path / "runtime", min_free_gib=0, check_network=False)
    assert report["network"] == {"skipped": True}
    assert report["runtime_root"].endswith("runtime")
    assert "nvidia_smi" in report["commands"]
    assert "rustc" in report["commands"]
    assert "driver_max" in report["cuda_compatibility"]


def test_runtime_cuda_toolkit_overrides_stale_system_nvcc(tmp_path, monkeypatch):
    toolkit = tmp_path / "cuda-12.9" / "bin"
    toolkit.mkdir(parents=True)
    nvcc = toolkit / "nvcc"
    nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 12.9, V12.9.86'\n")
    nvcc.chmod(0o755)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "0")
    selected = configure_runtime_cuda_toolkit(tmp_path)
    assert selected["version"] == "12.9"
    assert selected["root"] == str((tmp_path / "cuda-12.9").resolve())
    assert __import__("os").environ["CUDA_HOME"] == selected["root"]
    assert int(__import__("os").environ["OMP_NUM_THREADS"]) >= 1
    assert selected["omp_num_threads"] >= 1


def test_projection_artifact_hashes_file_and_each_array(tmp_path):
    path = tmp_path / "projection.npz"
    save_projection_artifact(
        path,
        {"hidden_projection": np.eye(3), "output_basis": np.ones((4, 2))},
        {"pair_id": "qwen3_4b_dspark7", "target_revision": "locked-a"},
    )
    arrays, metadata = load_projection_artifact(path)
    assert arrays["output_basis"].dtype == np.float32
    assert metadata["pair_id"] == "qwen3_4b_dspark7"

    # Updating the sidecar's file digest alone must not make a modified array
    # valid: the per-array content hashes are independently checked.
    with np.load(path) as cached:
        hidden = cached["hidden_projection"].copy()
        output = cached["output_basis"].copy()
    output[0, 0] = 9.0
    np.savez_compressed(path, hidden_projection=hidden, output_basis=output)
    meta_path = Path(str(path) + ".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["file_sha256"] = sha256_file(path)
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ConfigError, match="array hash drift"):
        load_projection_artifact(path)


def test_deferred_cuda_timing_flushes_before_close(tmp_path):
    class Event:
        def __init__(self, ms):
            self.ms = ms

        def synchronize(self):
            return None

        def elapsed_time(self, other):
            return other.ms - self.ms

    sink = TelemetrySink(tmp_path / "deferred.jsonl")
    record = RoundTelemetry(
        request_id="r",
        round_id=0,
        active_version=0,
        proposal_version=0,
        draft_tokens=7,
        accepted_drafts=0,
        committed_per_verify=0,
        target_calls=1,
        draft_cuda_us=0.0,
        verify_cuda_us=0.0,
        accept_cuda_us=0.0,
        draft_cpu_us=1.0,
        verify_cpu_us=2.0,
        rng_substream_id="rng",
        version_canary_ok=True,
    )
    sink.emit_round_deferred(
        record,
        {
            "draft_start": Event(0.0),
            "draft_end": Event(1.0),
            "verify_end": Event(3.0),
            "accept_end": Event(3.5),
            "request_index": 0,
            "prefix_lens_cpu": [37],
            "commit_lens_cpu": [5],
            "new_seq_lens_cpu": [42],
            "verify_lens_cpu": [6],
        },
    )
    sink.close()
    row = json.loads((tmp_path / "deferred.jsonl").read_text())
    assert row["draft_cuda_us"] == 1000.0
    assert row["verify_cuda_us"] == 2000.0
    assert row["accept_cuda_us"] == 500.0
    assert row["accepted_drafts"] == 4
    assert row["prefix_len_before"] == 37
    assert row["prefix_pos_before"] == 37
    assert row["prefix_pos_after"] == 42
    assert row["verify_len"] == 6


def test_deferred_round_rejects_host_and_pinned_count_disagreement(tmp_path):
    class Event:
        def __init__(self, ms):
            self.ms = ms

        def synchronize(self):
            return None

        def elapsed_time(self, other):
            return other.ms - self.ms

    sink = TelemetrySink(tmp_path / "mismatch.jsonl")
    record = RoundTelemetry(
        request_id="r",
        round_id=0,
        active_version=0,
        proposal_version=0,
        draft_tokens=3,
        accepted_drafts=3,
        committed_per_verify=4,
        target_calls=1,
        draft_cuda_us=0.0,
        verify_cuda_us=0.0,
        accept_cuda_us=0.0,
        draft_cpu_us=1.0,
        verify_cpu_us=1.0,
        rng_substream_id="r:0",
        version_canary_ok=True,
        prefix_pos_before=512,
        prefix_pos_after=516,
        prefix_len_before=512,
        verify_len=4,
    )
    sink.emit_round_deferred(
        record,
        {
            "draft_start": Event(0.0),
            "draft_end": Event(1.0),
            "verify_end": Event(2.0),
            "accept_end": Event(3.0),
            "request_index": 0,
            "commit_lens_cpu": [3],
            "new_seq_lens_cpu": [515],
            "expected_committed_per_verify": 4,
            "expected_prefix_pos_after": 516,
        },
    )
    assert sink.flush(timeout_s=1.0)
    health = sink.health()
    sink.close()
    assert health["error_count"] == 1
    assert "committed count disagrees" in health["last_error"]
    # The exact host-derived count remains in evidence; the benchmark fails
    # through telemetry health instead of accepting the conflicting copy.
    row = json.loads((tmp_path / "mismatch.jsonl").read_text())
    assert row["committed_per_verify"] == 4
    assert row["prefix_pos_after"] == 516


def test_deferred_round_releases_preallocated_lane_after_materialization(tmp_path):
    class Event:
        def __init__(self, ms):
            self.ms = ms

        def synchronize(self):
            return None

        def elapsed_time(self, other):
            return other.ms - self.ms

    released = []
    sink = TelemetrySink(tmp_path / "release.jsonl")
    record = RoundTelemetry(
        request_id="r",
        round_id=0,
        active_version=0,
        proposal_version=0,
        draft_tokens=3,
        accepted_drafts=0,
        committed_per_verify=0,
        target_calls=1,
        draft_cuda_us=0.0,
        verify_cuda_us=0.0,
        accept_cuda_us=0.0,
        draft_cpu_us=0.0,
        verify_cpu_us=0.0,
        rng_substream_id="r:0",
        version_canary_ok=True,
        batch_size=2,
    )
    sink.emit_round_deferred(
        record,
        {
            "draft_start": Event(0.0),
            "draft_end": Event(1.0),
            "verify_end": Event(2.0),
            "accept_end": Event(3.0),
            "signal_prep_start": Event(3.0),
            "signal_prep_end": Event(5.0),
            "telemetry_ready": Event(5.0),
            "request_index": 0,
            "commit_lens_cpu": [2],
            "new_seq_lens_cpu": [42],
            "release": lambda: released.append(True),
        },
    )
    sink.close()
    assert released == [True]
    row = json.loads((tmp_path / "release.jsonl").read_text())
    assert row["signal_prep_cuda_us"] == pytest.approx(1000.0)


def test_deferred_update_metrics_fail_closed_without_decode_thread_sync(tmp_path):
    class Event:
        def synchronize(self):
            return None

    released = []
    sink = TelemetrySink(tmp_path / "update.jsonl")
    sink.emit_update_deferred(
        UpdateTelemetry(
            request_id="r",
            update_id="u",
            source_round=1,
            source_version=0,
            snapshot_ts_us=1.0,
            launch_ts_us=2.0,
            source_prefix_len=12,
            published_version=None,
            decision="discard",
        ),
        {
            "event": Event(),
            "numerical_ok": False,
            "grad_norm": 0.0,
            "candidate_delta_norm": 0.0,
            "source_training_loss": float("nan"),
            "source_expected_accepted_prefix": float("nan"),
            "release": lambda: released.append(True),
        },
    )
    sink.close()
    assert released == [True]
    row = json.loads((tmp_path / "update.jsonl").read_text())
    assert row["failure_reason"] == "non_finite_candidate"
    assert row["decision"] == "discard"
    assert row["done_ts_us"] >= row["launch_ts_us"]
    assert row["source_prefix_len"] == 12
    assert math.isnan(row["source_training_loss"])
    assert math.isnan(row["source_expected_accepted_prefix"])


def test_deferred_update_materializes_backward_and_optimizer_cuda_time(tmp_path):
    class Event:
        def __init__(self, ms):
            self.ms = ms

        def synchronize(self):
            return None

        def elapsed_time(self, other):
            return other.ms - self.ms

    ready = Event(4.0)

    class DeferredScalar:
        def __init__(self, value):
            self.value = value

        def __float__(self):
            assert ready.synchronized
            return float(self.value)

    ready.synchronized = False
    original_synchronize = ready.synchronize

    def synchronize():
        original_synchronize()
        ready.synchronized = True

    ready.synchronize = synchronize
    sink = TelemetrySink(tmp_path / "update-components.jsonl")
    sink.emit_update_deferred(
        UpdateTelemetry(
            request_id="r",
            update_id="u",
            source_round=1,
            source_version=0,
            snapshot_ts_us=1.0,
            launch_ts_us=2.0,
            source_prefix_len=4096,
            decision="apply",
        ),
        {
            "event": ready,
            "numerical_ok": True,
            "grad_norm": 1.0,
            "candidate_delta_norm": 0.5,
            "source_training_loss": DeferredScalar(2.25),
            "source_expected_accepted_prefix": DeferredScalar(4.5),
            "backward_start": Event(1.0),
            "backward_end": Event(2.5),
            "optimizer_end": Event(3.25),
        },
    )
    sink.close()
    row = json.loads((tmp_path / "update-components.jsonl").read_text())
    assert row["backward_cuda_us"] == pytest.approx(1500.0)
    assert row["optimizer_cuda_us"] == pytest.approx(750.0)
    assert row["source_training_loss"] == pytest.approx(2.25)
    assert row["source_expected_accepted_prefix"] == pytest.approx(4.5)
    assert row["source_prefix_len"] == 4096


def test_deferred_nonfinite_publish_is_telemetry_health_error(tmp_path):
    class Event:
        def synchronize(self):
            return None

    sink = TelemetrySink(tmp_path / "invalid-publish.jsonl")
    sink.emit_update_deferred(
        UpdateTelemetry(
            request_id="r",
            update_id="u",
            source_round=1,
            source_version=0,
            snapshot_ts_us=1.0,
            source_prefix_len=8,
            published_version=1,
            decision="apply",
        ),
        {
            "event": Event(),
            "numerical_ok": False,
            "grad_norm": 0.0,
            "candidate_delta_norm": 0.0,
            "source_training_loss": float("nan"),
            "source_expected_accepted_prefix": float("nan"),
        },
    )
    sink.close()
    assert sink.health()["error_count"] == 1
    assert "non-finite candidate was reported as published" in sink.health()[
        "last_error"
    ]


def test_real_replay_fits_train_only_normalizer_and_data_hash(tmp_path):
    groups = {}
    i = 0
    while set(groups) != {"train", "calibration", "test"}:
        group = f"request-{i}"
        groups.setdefault(split_of_group(group), group)
        i += 1
    replay = tmp_path / "nested" / "real-replay"
    replay.mkdir(parents=True)
    (tmp_path / "nested" / "adaptation.runtime.yaml").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "naive_async",
                "lifecycle": "stream",
                "optimizer": "adamw",
                "update_stride": 4,
                "lr": 1e-4,
                "async": {
                    "enabled": True,
                    "logical_delay_rounds": 0,
                    "max_in_flight": 1,
                },
                "trace": {
                    "artifact_root": str(tmp_path / "nested"),
                    "trace_capture_max_bytes": 1024,
                },
                "model": {"pair_id": "qwen3_4b_dspark7"},
                "dataset": {"adapter": "livecodebench"},
            }
        )
    )
    index_lines = []
    for j, split in enumerate(("train", "calibration", "test")):
        group = groups[split]
        source = torch.arange(387, dtype=torch.float32) + j
        arrival = source + float(j + 1)
        payload = {
            "schema_version": 2,
            "provenance_method": "naive_async",
            "controller_label_source": "full_candidate_utility",
            "utility_metric": "survival_weighted_accepted_prefix_v1",
            "sequence_id": group,
            "update_id": f"u-{j}",
            "source_round": 1,
                "arrival_round": 2,
                "candidate_arrival_round": 2,
                "actual_arrival_round": 2,
                "paired_tts_barrier": False,
                "prefix_feature_exact": True,
            "round_delay": 1.0,
            "token_delay": 2.0,
            "wall_us": 3.0,
            "endpoint_distance": 0.1,
            "rho_path": 0.2,
            "parameter_displacement": 0.3,
            "source_prefix_len": 4096.0 * (j + 1),
            "source_acceptance": 1.0 + 0.1 * j,
            "source_training_loss": 0.5 + 0.1 * j,
            "source_grad_norm": 0.25 + 0.1 * j,
            "actual_published_utility": 1.0 if j != 2 else -1.0,
            "full_candidate_utility": 1.0 if j != 2 else -1.0,
            "training_loss_gain": 0.5 if j != 2 else -0.25,
            "relative_gradient_mismatch": 0.2,
            "cosine": 0.5,
            "harmful": int(j == 2),
            "delta_g": torch.tensor([0.1, 0.2, 0.3, 0.4]) * (j + 1),
            "delta_z": arrival - source,
            "source_z_raw": source,
            "arrival_z_raw": arrival,
        }
        full_utility = float(payload["full_candidate_utility"])
        for provenance in ("naive_async", "tts"):
            update_id = f"{provenance}-u-{j}"
            paired = provenance == "tts"
            method_payload = {
                **payload,
                "provenance_method": provenance,
                "update_id": update_id,
                "actual_arrival_round": 4 if paired else 2,
                "paired_tts_barrier": paired,
                "oracle_l1_utility": max(full_utility, 0.0),
                "oracle_l2_utility": max(full_utility, 0.0),
                "oracle_l2_kappa": 1.0 if full_utility >= 0.0 else 0.0,
                "utility_by_kappa": {
                    "0.0": 0.0,
                    "1.0": full_utility,
                },
            }
            path = replay / f"p1-{update_id}.pt"
            torch.save(method_payload, path)
            index_lines.append(
                json.dumps(
                    {
                        "path": path.name,
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                        "sequence_id": group,
                        "update_id": update_id,
                        "parameter_layout_sha256": "1" * 64,
                    }
                )
            )
    (replay / "index-p1.jsonl").write_text("\n".join(index_lines) + "\n")
    telemetry = tmp_path / "nested" / "adaptation-telemetry-p1-r0.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "kind": "round",
                "version_canary_ok": True,
                "prefix_feature_exact": True,
            }
        )
        + "\n"
    )
    # A final replay root may contain phase-1 controller traces together with
    # a failed phase-2 L3 evaluation.  L3's failure must close only the L3
    # gate; it is not evidence against the phase-1 L1/L2 controller traces.
    l3_evaluation = tmp_path / "l3-evaluation"
    l3_evaluation.mkdir()
    (l3_evaluation / "adaptation.runtime.yaml").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "lc_transport",
                "lifecycle": "stream",
                "optimizer": "adamw",
                "update_stride": 4,
                "lr": 1e-4,
                "async": {
                    "enabled": True,
                    "logical_delay_rounds": 0,
                    "max_in_flight": 1,
                },
                "controller": {"artifact_path": "/unused/controller.json"},
                "transport": {"basis_path": "/unused/transport.json"},
                "trace": {
                    "artifact_root": str(l3_evaluation),
                    "trace_capture_max_bytes": 1 << 20,
                    "l3_evaluation_only": True,
                },
                "model": {"pair_id": "qwen3_4b_dspark7"},
                "dataset": {"adapter": "livecodebench"},
            }
        )
    )
    (l3_evaluation / "adaptation-telemetry-p2-r0.jsonl").write_text(
        json.dumps(
            {
                "kind": "round",
                "version_canary_ok": False,
                "prefix_feature_exact": True,
            }
        )
        + "\n"
    )
    result = fit_real_replay(
        tmp_path,
        model_pair_id="qwen3_4b_dspark7",
        transport_rank=4,
    )
    artifact = result.artifact
    assert artifact.zvectorizer.mean is not None
    assert artifact.zvectorizer.std is not None
    assert np.all(artifact.zvectorizer.std > 0)
    assert len(artifact.extra["real_replay_data_sha256"]) == 64
    assert len(artifact.extra["exactness_evidence_sha256"]) == 64
    assert len(artifact.extra["runtime_config_evidence_sha256"]) == 64
    assert len(artifact.extra["test_group_hash"]) == 64
    assert artifact.extra["split_seed"] == 0
    assert (
        artifact.extra["controller_utility_metric"]
        == "survival_weighted_accepted_prefix_v1"
    )
    assert artifact.extra["utility_diagnostics"]["mean_training_loss_gain"] == 0.25
    assert len(artifact.utility_predictor.coef) == 5
    assert len(artifact.extra["controller_runtime_identity_sha256"]) == 64
    assert artifact.extra["parameter_layout_sha256"] == "1" * 64
    assert artifact.extra["controller_runtime_identity"]["candidate"]["lr"] == 1e-4
    # Phase-1 TTS/L0 exactness makes the frozen map evaluation-ready, but it
    # cannot stand in for an lc_transport evaluation run's exactness evidence.
    assert artifact.extra["l3_gate"]["evaluation_ready"] is True
    assert artifact.extra["l3_gate"]["exactness"]["verified"] is False
    assert artifact.extra["l3_gate"]["exactness"]["violation_count"] == 1
    assert artifact.extra["trace_exactness"]["verified"] is True
    assert artifact.extra["trace_exactness"]["owner_methods"] == [
        "naive_async",
        "tts",
    ]
    assert artifact.extra["trace_exactness"]["l3_evaluation_only"] is False
    assert artifact.extra["trace_producer_contract"][
        "observed_provenance_methods"
    ] == ["naive_async", "tts"]

    from lightcone_spec.config.loader import validate_adaptation_config_dict
    from lightcone_spec.methods.registry import validate_controller_artifact

    controller_cfg = validate_adaptation_config_dict(
        {
            "schema_version": 1,
            "method": "lc_damp",
            "lifecycle": "stream",
            "optimizer": "adamw",
            "update_stride": 4,
            "lr": 1e-4,
            "async": {
                "enabled": True,
                "logical_delay_rounds": 2,
                "max_in_flight": 1,
            },
            "controller": {"artifact_path": "/unused/controller.json"},
            "trace": {"artifact_root": str(tmp_path)},
            "model": {"pair_id": "qwen3_4b_dspark7"},
            "dataset": {"adapter": "math500"},
        }
    )
    assert artifact.extra["oracle_replay_gate"]["complete"] is False
    artifact.extra["oracle_replay_gate"] = {
        "complete": True,
        "l1_eligible": True,
        "l2_eligible": True,
    }
    artifact.extra["tts_paired_gate"] = {
        "complete": True,
        "l1_eligible": True,
        "l2_eligible": True,
    }
    artifact.extra["learned_policy_gate"] = {
        "complete": True,
        "l1_eligible": True,
        "l2_eligible": True,
    }
    validate_controller_artifact(controller_cfg, artifact)
    trace_exactness = artifact.extra.pop("trace_exactness")
    with pytest.raises(ConfigError, match="trace exactness"):
        validate_controller_artifact(controller_cfg, artifact)
    artifact.extra["trace_exactness"] = trace_exactness
    tts_gate = artifact.extra.pop("tts_paired_gate")
    with pytest.raises(ConfigError, match="paired real-TTS barrier"):
        validate_controller_artifact(controller_cfg, artifact)
    artifact.extra["tts_paired_gate"] = tts_gate
    learned_gate = artifact.extra.pop("learned_policy_gate")
    with pytest.raises(ConfigError, match="fitted policy"):
        validate_controller_artifact(controller_cfg, artifact)
    artifact.extra["learned_policy_gate"] = learned_gate
    artifact.extra["controller_utility_metric"] = "training_loss_gain_v1"
    with pytest.raises(ConfigError, match="accepted-prefix utility"):
        validate_controller_artifact(controller_cfg, artifact)
    artifact.extra["controller_utility_metric"] = (
        "survival_weighted_accepted_prefix_v1"
    )
    l3_eval_cfg = validate_adaptation_config_dict(
        {
            "schema_version": 1,
            "method": "lc_transport",
            "lifecycle": "stream",
            "optimizer": "adamw",
            "update_stride": 4,
            "lr": 1e-4,
            "async": {
                "enabled": True,
                "logical_delay_rounds": 2,
                "max_in_flight": 1,
            },
            "controller": {"artifact_path": "/unused/controller.json"},
            "transport": {"basis_path": "/unused/transport.json"},
            "trace": {
                "artifact_root": str(tmp_path),
                "trace_capture_max_bytes": 1 << 20,
                "l3_evaluation_only": True,
            },
            "model": {"pair_id": "qwen3_4b_dspark7"},
            "dataset": {"adapter": "math500"},
        }
    )
    validate_controller_artifact(l3_eval_cfg, artifact)
    l3_production_cfg = l3_eval_cfg.model_copy(deep=True)
    l3_production_cfg.trace.l3_evaluation_only = False
    with pytest.raises(ConfigError, match="paired TTS/L2"):
        validate_controller_artifact(l3_production_cfg, artifact)

    controller_cfg.lr = 1e-2
    with pytest.raises(ConfigError, match="runtime identity"):
        validate_controller_artifact(controller_cfg, artifact)
