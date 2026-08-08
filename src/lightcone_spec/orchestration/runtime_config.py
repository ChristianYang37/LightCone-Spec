"""Materialize host-specific runtime configuration outside immutable manifests."""

from __future__ import annotations

import json
import math
import os
import tempfile
from importlib.util import find_spec
from pathlib import Path

import yaml

from lightcone_spec.config.loader import validate_adaptation_config_dict
from lightcone_spec.config.schema import (
    CONTROLLER_METHODS,
    MODEL_PAIRS,
    canonical_tail_layout_mode,
    canonical_weight_update_mode,
    effective_proposal_depth,
    pair_thinking_config,
)
from lightcone_spec.exit_codes import ConfigError, LockError
from lightcone_spec.locking.hashing import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from lightcone_spec.locking.lockfile import load_lockfile
from lightcone_spec.locking.download import load_model_roots
from lightcone_spec.locking.verify import verify_lockfile_offline
from lightcone_spec.orchestration.units import RunUnit


# A manifest may materialize dozens of units against the same immutable model
# pair. Re-hashing every multi-GB weight file for every unit wastes host I/O
# while the GPU is idle. Cache only within this executor process and include a
# cheap full-tree stat signature in the key, so additions, removals or ordinary
# file mutations invalidate the successful verification.
_VERIFIED_MODEL_ROOT_STATES: set[tuple] = set()


def _sglang_package_root(path: str | Path) -> Path:
    """Normalize either an SGLang checkout or import-package path."""
    root = Path(path).expanduser().resolve()
    if (root / "srt").is_dir():
        return root
    checkout_package = root / "python" / "sglang"
    if (checkout_package / "srt").is_dir():
        return checkout_package
    return root


def _default_sglang_package_root() -> Path:
    explicit = os.environ.get("LIGHTCONE_SGLANG_SOURCE_ROOT")
    if explicit:
        return _sglang_package_root(explicit)
    spec = find_spec("sglang")
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            candidate = _sglang_package_root(location)
            if (candidate / "srt").is_dir():
                return candidate
    return _sglang_package_root(Path(__file__).resolve().parents[3] / "sglang")


# This is deliberately an explicit evidence surface, not a repository walk.
# The LightCone entries include the complete import closure reached by the
# SGLang tail-adaptation manager (candidate math, controller/artifact loading,
# trajectory features, transport, events and hooks), plus the small outer
# execution/validation boundary.  Adding a new runtime dependency must add its
# source here; absence fails closed below before model loading.
_RUNTIME_IMPLEMENTATION_FILES = {
    "lightcone_spec": (
        "adapters/adapter_params.py",
        "adapters/losses.py",
        "adapters/projections.py",
        "artifacts/rundir.py",
        "artifacts/validator.py",
        "cli/main.py",
        "config/loader.py",
        "config/schema.py",
        "controller/artifact.py",
        "controller/damping.py",
        "controller/gate.py",
        "exit_codes.py",
        "locking/hashing.py",
        "locking/lockfile.py",
        "methods/base.py",
        "methods/lightcone.py",
        "methods/onlinespec.py",
        "methods/optim.py",
        "methods/registry.py",
        "methods/simple.py",
        "orchestration/executor.py",
        "orchestration/manifest.py",
        "orchestration/runtime_config.py",
        "orchestration/units.py",
        "runtime/events.py",
        "sglang_bridge/bank.py",
        "sglang_bridge/client.py",
        "sglang_bridge/hooks.py",
        "sglang_bridge/runtime.py",
        "sglang_bridge/static_observer.py",
        "sglang_bridge/telemetry.py",
        "trajectory/distance.py",
        "trajectory/features.py",
        "trajectory/predictors.py",
        "trajectory/state.py",
        "trajectory/zvector.py",
        "transport/apply.py",
        "transport/fisher.py",
        "transport/fit.py",
    ),
    "sglang": (
        "srt/arg_groups/speculative_hook.py",
        "srt/entrypoints/engine.py",
        "srt/managers/overlap_utils.py",
        "srt/managers/scheduler.py",
        "srt/managers/scheduler_components/batch_result_processor.py",
        "srt/managers/scheduler_components/metrics_reporter.py",
        "srt/managers/tokenizer_manager.py",
        "srt/managers/tp_worker.py",
        "srt/mem_cache/kv_cache_configurator.py",
        "srt/model_executor/model_runner_components/kv_pool_runtime.py",
        "srt/observability/cpu_monitor.py",
        "srt/observability/metrics_collector.py",
        "srt/server_args.py",
        "srt/speculative/dflash_info_v2.py",
        "srt/speculative/dflash_utils.py",
        "srt/speculative/dflash_worker_v2.py",
        "srt/speculative/dspark_components/dspark_adaptation.py",
        "srt/speculative/dspark_components/dspark_draft.py",
        "srt/speculative/dspark_components/dspark_worker_v2.py",
        "srt/speculative/eagle_draft_cuda_graph_runner.py",
        "srt/speculative/eagle_info.py",
        "srt/speculative/eagle_worker_common.py",
        "srt/speculative/eagle_worker_v2.py",
        "srt/speculative/multi_layer_eagle_worker_v2.py",
        "srt/speculative/tail_adaptation.py",
        "srt/utils/common.py",
        "srt/utils/watchdog.py",
    ),
}


def _verify_bound_model_roots(path: Path, engine_params: dict) -> None:
    """Enforce an immutable manifest's model-roots content binding."""

    expected = engine_params.get("model_roots_sha256")
    if expected is None:
        return
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(char not in "0123456789abcdef" for char in expected)
    ):
        raise LockError("manifest model_roots_sha256 is not a lowercase SHA-256")
    if not path.is_file() or sha256_file(path) != expected:
        raise LockError(
            "--model-roots does not match the immutable manifest binding"
        )


def runtime_implementation_fingerprint(
    *,
    lightcone_root: str | Path | None = None,
    sglang_root: str | Path | None = None,
    locked_reference: dict[str, str] | None = None,
) -> dict:
    """Hash the explicit GPU execution surface without walking either repo."""
    roots = {
        "lightcone_spec": (
            Path(lightcone_root).resolve()
            if lightcone_root is not None
            else Path(__file__).resolve().parents[1]
        ),
        "sglang": (
            Path(sglang_root).resolve()
            if sglang_root is not None
            else _default_sglang_package_root()
        ),
    }
    files = {}
    for component, relative_paths in _RUNTIME_IMPLEMENTATION_FILES.items():
        root = roots[component]
        for relative in relative_paths:
            path = root / relative
            if not path.is_file():
                raise ConfigError(
                    f"runtime implementation file is missing: {component}/{relative}"
                )
            files[f"{component}/{relative}"] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    body = {
        "schema_version": 1,
        "files": files,
        "locked_reference": dict(sorted((locked_reference or {}).items())),
    }
    return {**body, "sha256": sha256_json(body)}


def _read_locked_model_config(
    roots: dict[str, str], repo_id: str
) -> dict | None:
    path = Path(roots[repo_id]) / "config.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid locked model config {path}: {exc}") from exc
    return value if isinstance(value, dict) else None


def _resolve_forward_dtype(
    engine_params: dict, target_config: dict | None
) -> tuple[str, int]:
    """Resolve the certified proposal-head dtype before KV sizing.

    An explicit engine dtype wins. ``auto`` follows the immutable target
    config; fixtures and legacy configs without dtype metadata use the
    adaptation schema's BF16 default.  FP16 is intentionally rejected until
    its reconstruction/exactness path is certified rather than silently
    borrowing the BF16 identity merely because both occupy two bytes.
    """

    value = engine_params.get("dtype")
    if value is None or str(value).lower() == "auto":
        candidates: list[object] = []
        if target_config is not None:
            target_text = target_config.get("text_config")
            candidates.extend(
                target_config.get(key) for key in ("torch_dtype", "dtype")
            )
            if isinstance(target_text, dict):
                candidates.extend(
                    target_text.get(key) for key in ("torch_dtype", "dtype")
                )
        value = next((candidate for candidate in candidates if candidate), None)
    normalized = str(value or "bfloat16").lower().removeprefix("torch.")
    aliases = {
        "bfloat16": ("bfloat16", 2),
        "bf16": ("bfloat16", 2),
        "float32": ("float32", 4),
        "fp32": ("float32", 4),
        "float": ("float32", 4),
    }
    if normalized not in aliases:
        raise ConfigError(
            f"unsupported proposal forward dtype {value!r}; the certified "
            "mixed-precision contract currently supports bfloat16 or float32"
        )
    return aliases[normalized]


def _model_root_state(lock, roots: dict[str, str]) -> tuple:
    snapshots = []
    for snap in lock.hf_snapshots:
        root_value = roots.get(snap.repo_id)
        if root_value is None:
            continue
        root = Path(root_value).resolve()
        files = []
        if root.is_dir():
            for path in root.rglob("*"):
                relative = path.relative_to(root)
                if not path.is_file() or ".cache" in relative.parts:
                    continue
                stat = path.stat()
                files.append(
                    (
                        relative.as_posix(),
                        int(stat.st_size),
                        int(stat.st_mtime_ns),
                        int(stat.st_ino),
                    )
                )
        snapshots.append(
            (
                snap.repo_id,
                snap.snapshot_sha,
                str(root),
                tuple(sorted(files)),
            )
        )
    return lock.content_sha256(), tuple(sorted(snapshots))


def _verify_model_roots_once(lock, roots: dict[str, str]) -> None:
    state = _model_root_state(lock, roots)
    if state in _VERIFIED_MODEL_ROOT_STATES:
        return
    verify_lockfile_offline(lock, roots, require_all=False)
    _VERIFIED_MODEL_ROOT_STATES.add(state)


def _memory_calibration_identity(
    unit: RunUnit,
    engine_params: dict,
    target_revision: str,
    drafter_revision: str,
    forward_dtype: str,
    adapter_row_capacity: int,
    runtime_implementation_sha256: str,
) -> dict:
    """Stable key for warmup peaks that are safe to reuse across runs."""
    gpu = {"name": "cuda-unavailable", "total_memory": 0, "capability": None}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            gpu = {
                "name": props.name,
                "total_memory": int(props.total_memory),
                "capability": [int(props.major), int(props.minor)],
            }
    except Exception:
        pass
    weight_update_mode = canonical_weight_update_mode(unit.trainable_scope)
    tail_layout_mode = canonical_tail_layout_mode(unit.trainable_scope)
    from lightcone_spec.sglang_bridge.bank import (
        ADAPTATION_MEMORY_ESTIMATOR_SCHEMA_VERSION,
        DFLASH_SUPERVISION_FANOUT_SCHEMA_VERSION,
    )

    return {
        "schema_version": 2,
        "gpu": gpu,
        "model_pair": unit.model_pair,
        "target_revision": target_revision,
        "drafter_revision": drafter_revision,
        "forward_dtype": forward_dtype,
        "method": unit.method,
        "speculative_algorithm": MODEL_PAIRS[unit.model_pair][
            "speculative_algorithm"
        ],
        "weight_update_mode": weight_update_mode,
        "parameter_scope": unit.parameter_scope,
        "tail_layout_mode": tail_layout_mode,
        "adapter_rank": (
            None if weight_update_mode == "full" else unit.adapter_rank
        ),
        "concurrency": unit.concurrency,
        "adaptation_slots": int(
            engine_params.get("adaptation_slots", unit.concurrency)
        ),
        "adapter_row_capacity": int(adapter_row_capacity),
        "memory_estimator_schema_version": (
            ADAPTATION_MEMORY_ESTIMATOR_SCHEMA_VERSION
        ),
        "dflash_supervision_fanout_schema_version": (
            DFLASH_SUPERVISION_FANOUT_SCHEMA_VERSION
        ),
        "runtime_implementation_sha256": runtime_implementation_sha256,
        "max_in_flight": int(engine_params.get("max_in_flight", 1)),
        "tensor_parallel_size": int(engine_params.get("tensor_parallel_size", 1)),
        "speculative_num_draft_tokens": int(
            engine_params.get(
                "speculative_num_draft_tokens",
                MODEL_PAIRS[unit.model_pair]["default_num_draft_tokens"],
            )
        ),
        "trace_capture_max_bytes": int(
            engine_params.get("trace_capture_max_bytes", 0)
        ),
        "trace_capture_max_records_per_request": int(
            engine_params.get("trace_capture_max_records_per_request", 4)
        ),
    }


def _load_memory_calibration(
    runtime_root: Path,
    identity: dict,
) -> tuple[str, Path, int]:
    digest = sha256_bytes(canonical_json(identity).encode("utf-8"))
    path = runtime_root / "memory-calibration" / f"{digest}.json"
    if not path.is_file():
        return digest, path, 0
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid memory calibration record {path}: {exc}") from exc
    if record.get("schema_version") != 2:
        raise ConfigError(
            f"unsupported memory calibration schema at {path}; refusing stale reserve"
        )
    if record.get("identity_sha256") != digest or record.get("identity") != identity:
        raise ConfigError(
            f"memory calibration identity mismatch at {path}; refusing stale reserve"
        )
    reserve = int(record.get("recommended_reserve_mb", 0))
    if reserve < 0:
        raise ConfigError(f"negative calibrated reserve in {path}")
    return digest, path, reserve


def _optimizer(method: str) -> str:
    if method == "static":
        return "none"
    if method in ("onlinespec_ogd", "onlinespec_opt", "onlinespec_ens"):
        return "sgd"
    return "adamw"


def _resolved_adapter_row_capacity(unit: RunUnit, engine_params: dict) -> int:
    """Resolve the capacity that both SGLang graph capture and tail rows use."""

    from lightcone_spec.sglang_bridge.bank import resolve_adapter_row_capacity

    max_running_requests = max(
        int(unit.concurrency),
        int(engine_params.get("max_running_requests", 48)),
    )
    graph_max = engine_params.get("cuda_graph_max_bs_decode")
    capacity = resolve_adapter_row_capacity(
        max_running_requests=max_running_requests,
        cuda_graph_max_bs_decode=(
            None if graph_max is None else int(graph_max)
        ),
    )
    declared = engine_params.get("adapter_row_capacity")
    if declared is not None and int(declared) != capacity:
        raise ConfigError(
            "adapter_row_capacity disagrees with max_running_requests/decode "
            f"graph capacity: declared {declared}, resolved {capacity}"
        )
    return capacity


def _preflight_adaptation_reserve_mb(
    unit: RunUnit,
    engine_params: dict,
    roots: dict[str, str],
) -> int:
    """Size resident tail state from locked configs before SGLang sizes KV."""
    if unit.method == "static":
        return 0
    pair = MODEL_PAIRS[unit.model_pair]
    target = _read_locked_model_config(roots, pair["target"])
    drafter = _read_locked_model_config(roots, pair["drafter"])
    if target is None or drafter is None:
        # Minimal unit fixtures may omit model metadata. Real lockfiles created
        # by prepare-models contain config.json; runtime construction remains a
        # second fail-closed check if a third-party lock omits it.
        return 0

    target_text = target.get("text_config", target)
    if not isinstance(target_text, dict):
        target_text = target

    def first_int(mapping: dict, names: tuple[str, ...]) -> int | None:
        for name in names:
            value = mapping.get(name)
            if value is not None:
                return int(value)
        return None

    hidden_size = first_int(target_text, ("hidden_size", "d_model", "n_embd"))
    vocab_size = first_int(target_text, ("vocab_size",)) or first_int(
        target, ("vocab_size",)
    )
    if hidden_size is None or vocab_size is None:
        raise ConfigError(
            "locked target config lacks hidden_size/vocab_size required for "
            "pre-KV adaptation sizing"
        )
    algorithm = pair["speculative_algorithm"]
    markov_dim = 0
    if algorithm == "DSPARK":
        drafter_text = drafter.get("text_config", drafter)
        dspark = drafter.get("dspark_config", {})
        if not isinstance(drafter_text, dict):
            drafter_text = drafter
        if not isinstance(dspark, dict):
            dspark = {}
        markov_dim = (
            first_int(drafter, ("dspark_markov_rank", "markov_rank"))
            or first_int(dspark, ("markov_rank",))
            or first_int(drafter_text, ("markov_rank",))
            or 0
        )
        if markov_dim <= 0:
            raise ConfigError(
                "locked DSpark config lacks a positive markov rank required "
                "for adaptation sizing"
            )

    from lightcone_spec.adapters.adapter_params import AdapterShapes
    from lightcone_spec.sglang_bridge.bank import (
        estimate_adaptation_memory,
        estimate_dflash_supervision_fanout_bytes,
    )

    mode = canonical_tail_layout_mode(unit.trainable_scope)
    _forward_dtype, forward_dtype_bytes = _resolve_forward_dtype(
        engine_params, target
    )
    draft_depth = effective_proposal_depth(
        pair,
        int(
            engine_params.get(
                "speculative_num_draft_tokens",
                pair["default_num_draft_tokens"],
            )
        ),
    )
    shapes = AdapterShapes(
        rank=unit.adapter_rank,
        markov_dim=markov_dim,
        vocab_size=vocab_size,
        weight_update_mode=mode,
        hidden_size=hidden_size,
        draft_depth=draft_depth,
        has_markov=bool(pair["capabilities"]["markov_features"]),
        has_confidence=bool(pair["capabilities"]["confidence_head"]),
        algorithm=algorithm,
    )
    num_slots = int(engine_params.get("adaptation_slots", unit.concurrency))
    row_capacity = _resolved_adapter_row_capacity(unit, engine_params)
    supervision_fanout_bytes = 0
    if algorithm == "DFLASH":
        supervision_fanout_bytes = estimate_dflash_supervision_fanout_bytes(
            batch_capacity=row_capacity,
            active_capacity=min(num_slots, row_capacity),
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            draft_depth=draft_depth,
            forward_dtype_bytes=forward_dtype_bytes,
            output_residual=mode == "output_residual",
            stochastic=unit.sampling_profile != "greedy_t0",
            tensor_parallel_size=int(
                engine_params.get("tensor_parallel_size", 1)
            ),
            trace_capture=(
                int(engine_params.get("trace_capture_max_bytes", 0)) > 0
            ),
        )
    ledger = estimate_adaptation_memory(
        num_slots=num_slots,
        max_in_flight=(
            int(engine_params.get("max_in_flight", 1))
            if unit.method == "lc_transport"
            else 1
        ),
        num_params=shapes.num_params(),
        vocab_size=vocab_size,
        rank=unit.adapter_rank,
        markov_dim=markov_dim,
        hidden_size=hidden_size,
        draft_depth=draft_depth,
        adapter_row_capacity=row_capacity,
        with_optimizer=_optimizer(unit.method) == "adamw",
        with_fisher=unit.method == "lc_transport",
        with_optimizer_preview=unit.method == "lc_transport",
        retain_source_signal=unit.method == "onlinespec_ens",
        trace_capture=int(engine_params.get("trace_capture_max_bytes", 0)) > 0,
        safety_factor=float(engine_params.get("memory_safety_factor", 1.25)),
        enabled=True,
        weight_update_mode=mode,
        forward_dtype_bytes=forward_dtype_bytes,
        supervision_fanout_bytes=supervision_fanout_bytes,
    )
    # Speculative workers construct their adaptation manager before SGLang
    # profiles and allocates the KV pool.  The fixed bank/graph rows are thus
    # already reflected in available GPU memory and must not be subtracted a
    # second time.  Only later side-stream allocations need explicit headroom.
    return math.ceil(ledger.reserve_bytes / (1 << 20))


def _materialize_sampling_section(unit: RunUnit, engine_params: dict) -> dict:
    thinking = pair_thinking_config(MODEL_PAIRS[unit.model_pair])
    if "enable_thinking" in engine_params:
        enable_thinking = bool(engine_params["enable_thinking"])
    else:
        enable_thinking = bool(thinking["enable_thinking"])
    section = {
        "temperature": 0.0 if unit.sampling_profile == "greedy_t0" else 1.0,
        "top_p": 1.0,
        "max_new_tokens": int(engine_params.get("max_new_tokens", 32768)),
        "ignore_eos": bool(engine_params.get("ignore_eos", False)),
        "enable_thinking": enable_thinking,
    }
    if enable_thinking:
        section["reasoning_parser"] = (
            engine_params.get("reasoning_parser") or thinking["reasoning_parser"]
        )
    else:
        section["reasoning_parser"] = None
    return section


def materialize_gpu_runtime(
    unit: RunUnit,
    engine_params: dict,
    run_dir: str | Path,
) -> dict:
    """Return a per-unit engine overlay and write its validated YAML.

    Machine-local paths are intentionally excluded from the immutable source
    manifest.  Their resolved values and the generated config digest are
    recorded in the run manifest instead.
    """
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    pair = MODEL_PAIRS[unit.model_pair]
    weight_update_mode = canonical_weight_update_mode(unit.trainable_scope)
    tail_layout_mode = canonical_tail_layout_mode(unit.trainable_scope)
    if unit.method != "static" and unit.parameter_scope != "tail":
        raise ConfigError(
            "SGLang online adaptation currently supports only "
            "parameter_scope=tail; all/allowlist updates require cache "
            "rebuild and are available only in the DFlash reference harness"
        )
    lock_path = engine_params.get("lockfile_path")
    if not lock_path:
        raise LockError(
            "real GPU units require run-manifest --lockfile before model load"
        )
    lock = load_lockfile(lock_path)
    target_lock = lock.find_snapshot(pair["target"])
    drafter_lock = lock.find_snapshot(pair["drafter"])

    roots_path = engine_params.get("model_roots_path")
    if not roots_path:
        raise LockError(
            "real GPU units require --model-roots from prepare-models"
        )
    _verify_bound_model_roots(Path(roots_path), engine_params)
    roots = load_model_roots(roots_path)
    missing_roots = [repo for repo in (pair["target"], pair["drafter"]) if repo not in roots]
    if missing_roots:
        raise LockError(f"verified model roots missing: {missing_roots}")
    for repo in (pair["target"], pair["drafter"]):
        if not Path(roots[repo]).is_dir():
            raise LockError(f"verified model root no longer exists: {repo} -> {roots[repo]}")
    _verify_model_roots_once(
        lock,
        {repo: roots[repo] for repo in (pair["target"], pair["drafter"])},
    )
    target_config = _read_locked_model_config(roots, pair["target"])
    forward_dtype, _forward_dtype_bytes = _resolve_forward_dtype(
        engine_params, target_config
    )
    adapter_row_capacity = _resolved_adapter_row_capacity(unit, engine_params)
    runtime_fingerprint = engine_params.get("runtime_implementation_fingerprint")
    if runtime_fingerprint is None:
        runtime_fingerprint = runtime_implementation_fingerprint()
    runtime_implementation_sha256 = (
        runtime_fingerprint.get("sha256")
        if isinstance(runtime_fingerprint, dict)
        else None
    )
    if (
        not isinstance(runtime_implementation_sha256, str)
        or len(runtime_implementation_sha256) != 64
        or any(
            char not in "0123456789abcdef"
            for char in runtime_implementation_sha256
        )
    ):
        raise ConfigError(
            "runtime_implementation_fingerprint must contain a lowercase SHA-256"
        )

    artifact_path = None
    controller_artifact = None
    if unit.method in CONTROLLER_METHODS or unit.method in (
        "lc_gate",
        "lc_damp",
        "lc_transport",
    ):
        from lightcone_spec.controller.artifact import (
            load_bound_controller_artifact,
            resolve_controller_artifact,
        )

        explicit = engine_params.get("controller_artifact_path")
        controller_root = engine_params.get("controller_root")
        if explicit:
            artifact_path = str(Path(explicit).resolve())
            controller_artifact = load_bound_controller_artifact(
                artifact_path,
                model_pair_id=unit.model_pair,
                weight_update_mode=tail_layout_mode,
            )
        elif controller_root:
            resolved_path, controller_artifact = resolve_controller_artifact(
                controller_root,
                model_pair_id=unit.model_pair,
                weight_update_mode=tail_layout_mode,
            )
            artifact_path = str(resolved_path.resolve())
        else:
            raise ConfigError(
                f"{unit.method} requires a frozen controller artifact for "
                f"{unit.model_pair}/{weight_update_mode}; pass --controller-root "
                "after running the bounded trace producer and replay fit"
            )

    runtime_root = Path(
        engine_params.get("runtime_root", "~/lightcone-tts-runtime")
    ).expanduser().resolve()
    calibration_identity = _memory_calibration_identity(
        unit,
        engine_params,
        target_lock.snapshot_sha,
        drafter_lock.snapshot_sha,
        forward_dtype,
        adapter_row_capacity,
        runtime_implementation_sha256,
    )
    calibration_sha, calibration_path, cached_reserve_mb = (
        _load_memory_calibration(runtime_root, calibration_identity)
    )
    preflight_reserve_mb = _preflight_adaptation_reserve_mb(
        unit, engine_params, roots
    )
    runtime_reserve_mb = max(
        int(engine_params.get("calibrated_reserve_mb", 0)),
        cached_reserve_mb,
        preflight_reserve_mb,
    )
    projection_path = None
    if weight_update_mode == "residual":
        projection_path = (
            runtime_root
            / "projections"
            / f"{unit.model_pair}-rank{unit.adapter_rank}-tp{{rank}}.npz"
        )
    trace_root = run_dir / "runtime"
    cfg_dict = {
        "schema_version": 1,
        "method": unit.method,
        "lifecycle": unit.lifecycle,
        # Retain the schema-v1 key for frozen controller/layout compatibility,
        # while emitting the public representation and independent scope.
        "trainable_scope": tail_layout_mode,
        "weight_update_mode": weight_update_mode,
        "parameter_scope": unit.parameter_scope,
        "parameter_allowlist": list(unit.parameter_allowlist),
        "update_stride": unit.stride,
        "optimizer": _optimizer(unit.method),
        "lr": float(engine_params.get("lr", 1e-4)),
        "weight_decay": (
            0.0
            if unit.method == "static"
            else float(engine_params.get("weight_decay", 0.0))
        ),
        "lambda_prox": float(engine_params.get("lambda_prox", 0.1)),
        "confidence_loss_weight": float(
            engine_params.get("confidence_loss_weight", 1.0)
        ),
        "grad_clip": float(engine_params.get("grad_clip", 1.0)),
        "trust_region_radius": float(
            engine_params.get("trust_region_radius", 1.0)
        ),
        "adapter_rank": unit.adapter_rank,
        "async": {
            "enabled": unit.method not in ("static", "sync_fresh"),
            "logical_delay_rounds": unit.logical_delay,
            "max_in_flight": (
                int(engine_params.get("max_in_flight", 1))
                if unit.method == "lc_transport"
                else 1
            ),
            "stream_priority": int(engine_params.get("stream_priority", 0)),
        },
        "controller": {"artifact_path": artifact_path},
        "transport": {
            "rank": unit.adapter_rank,
            "basis_path": artifact_path if unit.method == "lc_transport" else None,
        },
        "trace": {
            "level": engine_params.get("trace_level", "light"),
            "privacy_mode": engine_params.get("privacy_mode", "benchmark"),
            "artifact_root": str(trace_root),
            "trace_capture_max_bytes": int(
                engine_params.get("trace_capture_max_bytes", 0)
            ),
            "trace_capture_max_records_per_request": int(
                engine_params.get("trace_capture_max_records_per_request", 4)
            ),
            "trace_capture_sampling": engine_params.get(
                "trace_capture_sampling", "first"
            ),
            "l3_evaluation_only": bool(
                engine_params.get("l3_evaluation_only", False)
            ),
        },
        "model": {
            "pair_id": unit.model_pair,
            "target_revision": target_lock.snapshot_sha,
            "drafter_revision": drafter_lock.snapshot_sha,
            "tokenizer_revision": target_lock.snapshot_sha,
            "dtype": forward_dtype,
            "projection_artifact_path": (
                str(projection_path) if projection_path is not None else None
            ),
        },
        "dataset": {"adapter": unit.dataset},
        "sampling": _materialize_sampling_section(unit, engine_params),
        "runtime": {
            "seed": unit.seed,
            "concurrency": unit.concurrency,
            "adaptation_slots": int(
                engine_params.get("adaptation_slots", unit.concurrency)
            ),
            "adapter_row_capacity": adapter_row_capacity,
            "memory_safety_factor": float(
                engine_params.get("memory_safety_factor", 1.25)
            ),
            "calibrated_reserve_mb": int(runtime_reserve_mb),
            "tensor_parallel_size": int(
                engine_params.get("tensor_parallel_size", 1)
            ),
            "speculative_num_draft_tokens": int(
                engine_params.get(
                    "speculative_num_draft_tokens",
                    pair["default_num_draft_tokens"],
                )
            ),
        },
    }
    cfg = validate_adaptation_config_dict(cfg_dict)
    if artifact_path is not None:
        from lightcone_spec.methods.registry import validate_controller_artifact

        validate_controller_artifact(cfg, controller_artifact)
    config_path = run_dir / "adaptation.runtime.yaml"
    resolved_cfg = cfg.model_dump(by_alias=True, mode="json")
    resolved_cfg["weight_update_mode"] = cfg.weight_update_mode
    body = yaml.safe_dump(resolved_cfg, sort_keys=True)
    config_path.write_text(body)
    overlay = dict(engine_params)
    overlay.update(
        {
            "adaptation_config_path": str(config_path),
            "model_roots": roots,
            "telemetry_glob": str(trace_root / "adaptation-telemetry-*.jsonl"),
            "runtime_config_sha256": sha256_bytes(body.encode("utf-8")),
            "locked_target_revision": target_lock.snapshot_sha,
            "locked_drafter_revision": drafter_lock.snapshot_sha,
            "speculative_algorithm": pair["speculative_algorithm"],
            "speculative_capabilities": pair["capabilities"],
            "weight_update_mode": weight_update_mode,
            "parameter_scope": unit.parameter_scope,
            "tail_layout_mode": tail_layout_mode,
            "effective_adapter_rank": (
                None
                if weight_update_mode == "full"
                else unit.adapter_rank
            ),
            "memory_calibration_identity": calibration_identity,
            "memory_calibration_sha256": calibration_sha,
            "memory_calibration_path": str(calibration_path),
            "calibrated_reserve_mb": runtime_reserve_mb,
            "preflight_adaptation_reserve_mb": preflight_reserve_mb,
            "adapter_row_capacity": adapter_row_capacity,
            "forward_dtype": forward_dtype,
        }
    )
    if int(engine_params.get("profile_steps", 0)) > 0:
        # Keep profiler traces inside the immutable run evidence directory;
        # machine-local absolute paths never enter the source manifest.
        overlay["profile_output_dir"] = str(run_dir / "profiles")
    return overlay


def preflight_gpu_manifest_inputs(
    units: list[RunUnit] | tuple[RunUnit, ...], engine_params: dict
) -> dict:
    """Validate every distinct live adaptation/controller contract up front.

    Unit execution materializes the same config again inside its immutable run
    directory.  This earlier pass exists so an unsupported parameter scope or
    a missing/mismatched controller aborts before the first model is loaded,
    rather than after Static/TTS units have already consumed GPU time.
    """
    live_units = [unit for unit in units if unit.method != "static"]
    unsupported = sorted(
        {
            (unit.model_pair, unit.parameter_scope)
            for unit in live_units
            if unit.parameter_scope != "tail"
        }
    )
    if unsupported:
        raise ConfigError(
            "SGLang online adaptation supports parameter_scope=tail only; "
            f"unsupported manifest scopes: {unsupported}"
        )

    # Dataset, prompt bucket and logical delay are observed controller inputs,
    # not artifact-identity fields.  Concurrency remains in the key because it
    # changes the pre-KV resident-memory reservation.
    representatives: dict[tuple, RunUnit] = {}
    for unit in live_units:
        if unit.method not in CONTROLLER_METHODS:
            continue
        key = (
            unit.model_pair,
            unit.method,
            unit.weight_update_mode,
            unit.parameter_scope,
            tuple(unit.parameter_allowlist),
            unit.adapter_rank if unit.weight_update_mode != "full" else None,
            unit.stride,
            unit.lifecycle,
            unit.sampling_profile,
            unit.concurrency,
            unit.transport_variant,
        )
        representatives.setdefault(key, unit)

    checked = []
    if representatives:
        with tempfile.TemporaryDirectory(
            prefix="lightcone-runtime-preflight-"
        ) as temporary_root:
            root = Path(temporary_root)
            for index, unit in enumerate(representatives.values()):
                materialize_gpu_runtime(unit, engine_params, root / str(index))
                checked.append(unit.unit_id)
    return {
        "controller_contracts_checked": len(checked),
        "representative_unit_ids": checked,
    }
