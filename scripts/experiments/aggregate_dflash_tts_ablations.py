#!/usr/bin/env python3
"""Aggregate auditable DFlash/TTS calibration runs into ablation tables.

Each input directory must contain the ``summary.json`` and ``rounds.jsonl``
written by ``dflash_tts_reference.py``.  Rows are bucketed by the *observed*
``prefix_length_before``; a requested/final context length is never used as a
surrogate.  Run-level HBM is repeated with an explicit scope label because the
reference harness records a run peak, not a per-round allocator sample.

The output deliberately remains descriptive.  It preserves sample/seed and
source identities so a later paired bootstrap can select a learning rate or
rank without silently treating context buckets as independent replicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
FULLY_VERIFIED_IDENTITY = "fully_verified_content_sha256_v1"
ADAPTED_MODES = frozenset(
    {
        "full-drafter",
        "drafter-lora",
        "full-rank-tail",
        "tail-lora",
        "output-residual",
    }
)
LORA_MODES = frozenset({"drafter-lora", "tail-lora"})
PAIR_PARAMETER_KEYS = (
    "seed",
    "temperature",
    "block_size",
    "mask_token_id",
    "stop_token_ids",
    "max_new_tokens",
    "draft_cache_policy",
    "dtype",
    "enable_thinking",
)
PAIR_OPTIONAL_PARAMETER_KEYS = (
    "device",
    "prompt_field",
    "messages_field",
    "turns_field",
    "canonical_greedy_verifier",
)
ABLATION_CONTROL_KEYS = (
    "mode",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "rank",
    "adapter_seed",
    "update_stride",
    "proximal_lambda",
    "position_weighting",
    "position_decay_gamma",
    "loss_reduction",
    "adam_betas",
    "adam_eps",
    "draft_cache_policy",
    "audit_cuda_timing",
    "parameter_audit_stride",
)
MEMORY_KEYS = (
    "forward_parameter_bytes",
    "master_parameter_bytes",
    "forward_gradient_bytes",
    "master_gradient_bytes",
    "optimizer_moment_bytes",
    "persistent_bytes",
    "estimated_update_peak_bytes",
    "total_bytes",
    "parameter_audit_cpu_snapshot_bytes",
)
_MISSING = object()
_SHA256_HEX = frozenset("0123456789abcdef")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"{path}: cannot read JSONL: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _get(value: dict[str, Any], *path: str, default: Any = _MISSING) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _required(value: dict[str, Any], run_root: Path, *path: str) -> Any:
    result = _get(value, *path)
    if result is _MISSING or result is None or result == "":
        raise ValueError(
            f"{run_root}: missing core identity field {'.'.join(path)}"
        )
    return result


def _coalesce(
    label: str,
    candidates: Iterable[tuple[str, Any]],
    *,
    allow_none: bool = True,
) -> Any:
    present = [(name, value) for name, value in candidates if value is not _MISSING]
    nonnull = [(name, value) for name, value in present if value is not None]
    if not nonnull:
        return None if allow_none else _MISSING
    first_name, first = nonnull[0]
    first_json = _canonical_json(first)
    for name, value in nonnull[1:]:
        if _canonical_json(value) != first_json:
            raise ValueError(
                f"conflicting {label}: {first_name}={first!r}, {name}={value!r}"
            )
    return first


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    return parsed


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _parameter_alias(parameters: dict[str, Any], *names: str) -> Any:
    return _coalesce(
        "/".join(names),
        ((name, parameters.get(name, _MISSING)) for name in names),
    )


def _optimizer_memory(summary: dict[str, Any], mode: str) -> dict[str, int | None]:
    payload = _coalesce(
        "optimizer_memory_bytes",
        (
            ("generation.optimizer_memory_bytes", _get(summary, "generation", "optimizer_memory_bytes")),
            ("parameters.optimizer_memory_bytes", _get(summary, "parameters", "optimizer_memory_bytes")),
            ("optimizer_memory_bytes", summary.get("optimizer_memory_bytes", _MISSING)),
        ),
    )
    if payload is None:
        return {key: 0 if mode == "static" else None for key in MEMORY_KEYS}
    if not isinstance(payload, dict):
        raise ValueError("optimizer_memory_bytes must be an object")
    output: dict[str, int | None] = {}
    for key in MEMORY_KEYS:
        value = payload.get(key)
        if value is None:
            output[key] = None
            continue
        parsed = _as_int(value, f"optimizer_memory_bytes.{key}")
        if parsed < 0:
            raise ValueError(f"optimizer_memory_bytes.{key} must be non-negative")
        output[key] = parsed
    if all(output[key] is not None for key in MEMORY_KEYS[:5]):
        expected_persistent = sum(
            int(output[key])
            for key in (
                "forward_parameter_bytes",
                "master_parameter_bytes",
                "optimizer_moment_bytes",
            )
        )
        persistent = output["persistent_bytes"]
        if persistent is not None and persistent != expected_persistent:
            raise ValueError(
                "optimizer memory identity failed: persistent_bytes != "
                "forward parameters + master parameters + moments"
            )
        forward_gradient = output["forward_gradient_bytes"]
        master_gradient = output["master_gradient_bytes"]
        update_peak = output["estimated_update_peak_bytes"]
        if persistent is not None and update_peak is not None:
            expected_peak = persistent + int(forward_gradient) + int(master_gradient)
            if update_peak != expected_peak:
                raise ValueError(
                    "optimizer memory identity failed: estimated update peak mismatch"
                )
        total = output["total_bytes"]
        if total is not None and update_peak is not None and total != update_peak:
            raise ValueError(
                "optimizer memory identity failed: total_bytes != estimated update peak"
            )
    return output


def _hbm_summary(summary: dict[str, Any]) -> tuple[int | None, int | None]:
    hbm = _get(summary, "generation", "hbm_bytes")
    if hbm is _MISSING:
        # Compatibility for a short-lived draft schema where the final CUDA
        # snapshot was top-level.  In v2, top-level hbm_bytes instead contains
        # phase snapshots and must not be compared with generation.hbm_bytes.
        top = summary.get("hbm_bytes")
        hbm = (
            top
            if isinstance(top, dict) and "running_peak_allocated" in top
            else None
        )
    if hbm is not None and not isinstance(hbm, dict):
        raise ValueError("hbm_bytes must be an object")
    hbm = hbm or {}
    value = _coalesce(
        "peak_hbm_bytes",
        (
            ("generation.peak_hbm_bytes", _get(summary, "generation", "peak_hbm_bytes")),
            ("peak_hbm_bytes", summary.get("peak_hbm_bytes", _MISSING)),
            ("memory.peak_hbm_bytes", _get(summary, "memory", "peak_hbm_bytes")),
            ("hbm_bytes.running_peak_allocated", hbm.get("running_peak_allocated", _MISSING)),
        ),
    )
    reserved = _coalesce(
        "peak_hbm_reserved_bytes",
        (
            ("generation.peak_hbm_reserved_bytes", _get(summary, "generation", "peak_hbm_reserved_bytes")),
            ("peak_hbm_reserved_bytes", summary.get("peak_hbm_reserved_bytes", _MISSING)),
            ("hbm_bytes.running_peak_reserved", hbm.get("running_peak_reserved", _MISSING)),
        ),
    )
    parsed: list[int | None] = []
    for label, item in (("peak_hbm_bytes", value), ("peak_hbm_reserved_bytes", reserved)):
        if item is None:
            parsed.append(None)
            continue
        number = _as_int(item, label)
        if number < 0:
            raise ValueError(f"{label} must be non-negative")
        parsed.append(number)
    return parsed[0], parsed[1]


def _hbm_phases(summary: dict[str, Any]) -> dict[str, int | None]:
    payload = summary.get("hbm_bytes")
    phases = ("after_model_load", "after_adapter", "after_optimizer", "after_run")
    metrics = ("allocated_end", "reserved_end", "running_peak_allocated", "running_peak_reserved")
    output: dict[str, int | None] = {
        f"hbm_{phase}_{metric}_bytes": None
        for phase in phases
        for metric in metrics
    }
    output.update(
        {
            f"hbm_{label}_{metric}_delta_bytes": None
            for label in ("adapter", "optimizer", "run")
            for metric in ("allocated_end", "reserved_end")
        }
    )
    output.update(
        {
            f"whole_process_peak_{metric}_bytes": None
            for metric in ("running_peak_allocated", "running_peak_reserved")
        }
    )
    if payload is None:
        return output
    if not isinstance(payload, dict):
        raise ValueError("top-level hbm_bytes must be an object or null")
    if "running_peak_allocated" in payload:
        # Compatibility final snapshot, not phase evidence.
        return output
    for phase in phases:
        snapshot = payload.get(phase)
        if snapshot is None:
            continue
        if not isinstance(snapshot, dict):
            raise ValueError(f"hbm_bytes.{phase} must be an object or null")
        for metric in metrics:
            value = snapshot.get(metric)
            if value is None:
                continue
            parsed = _as_int(value, f"hbm_bytes.{phase}.{metric}")
            if parsed < 0:
                raise ValueError(
                    f"hbm_bytes.{phase}.{metric} must be non-negative"
                )
            output[f"hbm_{phase}_{metric}_bytes"] = parsed
    for label, left, right in (
        ("adapter", "after_model_load", "after_adapter"),
        ("optimizer", "after_adapter", "after_optimizer"),
        ("run", "after_optimizer", "after_run"),
    ):
        for metric in ("allocated_end", "reserved_end"):
            before = output[f"hbm_{left}_{metric}_bytes"]
            after = output[f"hbm_{right}_{metric}_bytes"]
            output[f"hbm_{label}_{metric}_delta_bytes"] = (
                None if before is None or after is None else after - before
            )
    for metric in ("running_peak_allocated", "running_peak_reserved"):
        phase_peaks = [
            output[f"hbm_{phase}_{metric}_bytes"]
            for phase in ("after_optimizer", "after_run")
            if output[f"hbm_{phase}_{metric}_bytes"] is not None
        ]
        output[f"whole_process_peak_{metric}_bytes"] = (
            max(phase_peaks) if len(phase_peaks) == 2 else None
        )
    return output


def _trainable_layout(
    summary: dict[str, Any], mode: str
) -> tuple[int | None, int | None, str | None, str | None]:
    layout = summary.get("trainable_layout")
    generation = summary.get("generation", {})
    if not isinstance(layout, dict):
        layout = {}
    count = generation.get("trainable_parameter_count")
    if count is None:
        count = summary.get("trainable_parameter_count")
    if count is None:
        count = layout.get("parameter_count")
    if count is None and isinstance(layout.get("parameters"), list):
        numels = [entry.get("numel") for entry in layout["parameters"] if isinstance(entry, dict)]
        if numels and all(value is not None for value in numels):
            count = sum(_as_int(value, "trainable_layout.parameters[].numel") for value in numels)
    tensors = layout.get("parameter_tensors")
    if tensors is None and isinstance(layout.get("parameters"), list):
        tensors = len(layout["parameters"])
    # These hashes intentionally bind different schemas in the reference
    # harness.  ``trainable_layout`` includes master/forward dtype metadata;
    # the runtime hash binds the exact parameter list used by the update loop.
    # They must be retained separately rather than falsely required to match.
    layout_hash = layout.get("layout_sha256")
    runtime_layout_hash = generation.get("parameter_layout_sha256")
    if mode == "static":
        count = 0 if count is None else count
        tensors = 0 if tensors is None else tensors
    return (
        None if count is None else _as_int(count, "trainable_parameter_count"),
        None if tensors is None else _as_int(tensors, "trainable_parameter_tensors"),
        None if layout_hash is None else str(layout_hash),
        None if runtime_layout_hash is None else str(runtime_layout_hash),
    )


def _round_prefix(row: dict[str, Any]) -> int:
    """Read the v1/v2+ real-prefix field without accepting disagreement."""

    prefix = _coalesce(
        "prefix length",
        (
            ("prefix_length_before", row.get("prefix_length_before", _MISSING)),
            ("prefix_len_before", row.get("prefix_len_before", _MISSING)),
        ),
        allow_none=False,
    )
    if prefix is _MISSING:
        raise ValueError("round lacks true prefix length")
    return _as_int(prefix, "prefix_length_before")


def _effective_output_by_round(
    summary: dict[str, Any], rounds: Sequence[dict[str, Any]]
) -> dict[int, int]:
    """Clip the terminal verification block to tokens present in final output."""

    generation = summary.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("summary generation must be an object")
    output_tokens = _as_int(
        generation.get("num_output_tokens"), "generation.num_output_tokens"
    )
    remaining = output_tokens
    effective: dict[int, int] = {}
    for row in rounds:
        index = _as_int(row.get("round_index"), "round_index")
        recorded = _as_int(row.get("acceptance_length"), "acceptance_length")
        effective[index] = min(recorded, max(remaining, 0))
        remaining -= effective[index]
    # A stop-token run can retain the final target bonus that would otherwise
    # seed the next proposal round.  The harness counts it in output token IDs,
    # not in acceptance_length.  Any gap larger than one is an invalid trace.
    if remaining == 1 and rounds:
        final_index = _as_int(rounds[-1].get("round_index"), "round_index")
        effective[final_index] += 1
        remaining = 0
    if remaining != 0:
        raise ValueError(
            "round trace cannot account for "
            f"{output_tokens} output tokens (remaining={remaining})"
        )
    return effective


def _projection_artifact_sha256(summary: dict[str, Any]) -> str | None:
    value = _get(
        summary,
        "trainable_layout",
        "projection_identity",
        "artifact_file_sha256",
        default=None,
    )
    if value is None:
        return None
    if not _is_sha256(value):
        raise ValueError(
            "trainable_layout.projection_identity.artifact_file_sha256 must "
            "be a lowercase SHA256"
        )
    return str(value)


def _layout_family(
    summary: dict[str, Any], mode: str, rank: int | None
) -> tuple[str | None, str]:
    """Hash rank-invariant parameterization recipe and module coverage.

    Full layout hashes deliberately cannot identify a rank ablation family: the
    A/B tensor shapes change on the ablated axis.  This identity preserves the
    non-rank dimensions, names, dtypes, initialization recipe, and adapter seed
    while replacing only dimensions equal to the declared rank.
    """

    if mode not in LORA_MODES:
        return None, "not_applicable"
    layout = summary.get("trainable_layout")
    if not isinstance(layout, dict):
        return None, "unavailable_missing_trainable_layout"
    parameters = layout.get("parameters")
    if not isinstance(parameters, list) or not parameters:
        return None, "unavailable_missing_parameter_coverage"
    if rank is None or rank <= 0:
        return None, "unavailable_missing_rank"
    normalized_parameters: list[dict[str, Any]] = []
    for index, entry in enumerate(parameters):
        if not isinstance(entry, dict):
            return None, f"unavailable_invalid_parameter_entry_{index}"
        name = entry.get("name")
        shape = entry.get("shape")
        if not isinstance(name, str) or not name:
            return None, f"unavailable_missing_parameter_name_{index}"
        if not isinstance(shape, list) or not shape:
            return None, f"unavailable_missing_parameter_shape_{index}"
        normalized_shape: list[int | str] = []
        for dimension in shape:
            parsed = _as_int(dimension, f"trainable_layout.parameters[{index}].shape")
            normalized_shape.append("$rank" if parsed == rank else parsed)
        normalized_parameters.append(
            {
                "name": name,
                "shape": normalized_shape,
                "forward_dtype": entry.get("forward_dtype", entry.get("dtype")),
                "master_dtype": entry.get("master_dtype"),
            }
        )
    parameters = summary.get("parameters")
    adapter_seed = parameters.get("adapter_seed") if isinstance(parameters, dict) else None
    initialization = layout.get("initialization")
    family = {
        "schema_version": 1,
        "algorithm": layout.get("algorithm", "DFLASH"),
        "mode": mode,
        "has_markov": layout.get("has_markov", False),
        "has_confidence": layout.get("has_confidence", False),
        "adapter_seed": layout.get("adapter_seed", adapter_seed),
        "initialization_recipe": initialization,
        "module_parameters": normalized_parameters,
    }
    return _sha256_json(family), "available_rank_invariant_layout_family_v1"


def _runtime_fingerprint(
    summary: dict[str, Any],
) -> tuple[str | None, str, str | None]:
    """Return a canonical environment hash and strict HBM-comparability status."""

    payload = summary.get("runtime_fingerprint")
    if payload is None:
        return None, "missing", None
    if not isinstance(payload, dict):
        raise ValueError("runtime_fingerprint must be an object or null")
    required_top = (
        "schema_version",
        "python_version",
        "python_implementation",
        "platform",
        "torch_version",
        "cuda_runtime_version",
        "cuda_driver_version",
        "attention_implementation",
        "dtype",
        "device",
        "resolved_device",
        "resolved_device",
        "allocator_config",
        "cuda_visible_devices",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "allow_tf32",
        "float32_matmul_precision",
        "cudnn_benchmark",
    )
    missing = [key for key in required_top if key not in payload]
    gpu = payload.get("gpu")
    if not isinstance(gpu, dict):
        missing.append("gpu")
    else:
        for key in ("name", "total_memory_bytes", "compute_capability"):
            if key not in gpu:
                missing.append(f"gpu.{key}")
    if payload.get("schema_version") != 1:
        missing.append("schema_version=1")
    if not isinstance(payload.get("allocator_config"), dict):
        missing.append("allocator_config.object")
    allow_tf32 = payload.get("allow_tf32")
    if not isinstance(allow_tf32, dict) or not {"matmul", "cudnn"}.issubset(
        allow_tf32
    ):
        missing.append("allow_tf32.matmul+cudnn")
    device_type = str(
        payload.get("resolved_device", payload.get("device", ""))
    ).partition(":")[0]
    if device_type == "cuda":
        for key in ("cuda_runtime_version", "cuda_driver_version"):
            if payload.get(key) is None:
                missing.append(key + "_null")
    fingerprint = _sha256_json(payload)
    if missing:
        return (
            fingerprint,
            "incomplete_missing_" + ",".join(sorted(missing)),
            _canonical_json(payload),
        )
    return fingerprint, "complete_hbm_comparable_v1", _canonical_json(payload)


def _numeric_runtime_identity(summary: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Bind settings that can alter proposal/verification numerics.

    Allocator policy, total HBM, Python/platform text, and device index are
    intentionally excluded: they matter for memory comparability, not the
    mathematical paired-AL identity.
    """

    payload = summary.get("runtime_fingerprint")
    if payload is None:
        return None, "legacy_missing_runtime_fingerprint"
    if not isinstance(payload, dict):
        raise ValueError("runtime_fingerprint must be an object or null")
    required = (
        "torch_version",
        "cuda_runtime_version",
        "cuda_driver_version",
        "attention_implementation",
        "dtype",
        "device",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "allow_tf32",
        "float32_matmul_precision",
        "cudnn_benchmark",
        "gpu",
    )
    missing = [key for key in required if key not in payload]
    missing.extend(
        key
        for key in (
            "torch_version",
            "cuda_runtime_version",
            "cuda_driver_version",
            "attention_implementation",
            "dtype",
            "device",
        )
        if key in payload and payload[key] is None
    )
    if missing:
        return None, "incomplete_numeric_runtime_missing_" + ",".join(missing)
    gpu = payload["gpu"]
    gpu_identity = None
    if gpu is not None:
        if not isinstance(gpu, dict):
            raise ValueError("runtime_fingerprint.gpu must be an object or null")
        missing_gpu = [
            key for key in ("name", "compute_capability") if key not in gpu
        ]
        if missing_gpu:
            return (
                None,
                "incomplete_numeric_runtime_missing_"
                + ",".join(f"gpu.{key}" for key in missing_gpu),
            )
        gpu_identity = {
            "name": gpu.get("name"),
            "compute_capability": gpu.get("compute_capability"),
        }
    device = payload["device"]
    device_type = (
        None if device is None else str(device).partition(":")[0].lower()
    )
    identity = {
        "schema_version": 1,
        "torch_version": payload["torch_version"],
        "cuda_runtime_version": payload["cuda_runtime_version"],
        "cuda_driver_version": payload["cuda_driver_version"],
        "attention_implementation": payload["attention_implementation"],
        "dtype": payload["dtype"],
        "device_type": device_type,
        "deterministic_algorithms": payload["deterministic_algorithms"],
        "deterministic_warn_only": payload["deterministic_warn_only"],
        "allow_tf32": payload["allow_tf32"],
        "float32_matmul_precision": payload["float32_matmul_precision"],
        "cudnn_benchmark": payload["cudnn_benchmark"],
        "gpu": gpu_identity,
    }
    return identity, "complete_numeric_runtime_v1"


def _round_scalar(row: dict[str, Any], metric: str) -> tuple[float | None, str | None]:
    aliases = {
        "loss": (
            ("update.loss", _get(row, "update", "loss")),
            ("update.total_loss", _get(row, "update", "total_loss")),
            ("loss", row.get("loss", _MISSING)),
            ("tts_loss", row.get("tts_loss", _MISSING)),
            ("adaptation_loss", row.get("adaptation_loss", _MISSING)),
        ),
        "distillation_kl": (
            ("update.distillation_kl", _get(row, "update", "distillation_kl")),
            ("distillation_kl", row.get("distillation_kl", _MISSING)),
        ),
        "proximal_kl": (
            ("update.proximal_kl", _get(row, "update", "proximal_kl")),
            ("proximal_kl", row.get("proximal_kl", _MISSING)),
        ),
        "grad_norm": (
            ("update.grad_norm", _get(row, "update", "grad_norm")),
            ("grad_norm", row.get("grad_norm", _MISSING)),
        ),
        "backward_cuda_us": (
            ("update.backward_cuda_us", _get(row, "update", "backward_cuda_us")),
            ("backward_cuda_us", row.get("backward_cuda_us", _MISSING)),
        ),
        "optimizer_cuda_us": (
            ("update.optimizer_cuda_us", _get(row, "update", "optimizer_cuda_us")),
            ("optimizer_cuda_us", row.get("optimizer_cuda_us", _MISSING)),
        ),
        "update_cuda_us": (
            ("update.update_cuda_us", _get(row, "update", "update_cuda_us")),
            ("update_cuda_us", row.get("update_cuda_us", _MISSING)),
        ),
        "parameter_delta_l2": (
            ("update.parameter_delta_l2", _get(row, "update", "parameter_delta_l2")),
            ("parameter_delta_l2", row.get("parameter_delta_l2", _MISSING)),
        ),
        "parameter_displacement_l2": (
            ("update.parameter_displacement_l2", _get(row, "update", "parameter_displacement_l2")),
            ("parameter_displacement_l2", row.get("parameter_displacement_l2", _MISSING)),
        ),
        "parameter_l2": (
            ("update.parameter_l2", _get(row, "update", "parameter_l2")),
            ("parameter_l2", row.get("parameter_l2", _MISSING)),
        ),
        "relative_parameter_delta": (
            ("update.relative_parameter_delta", _get(row, "update", "relative_parameter_delta")),
            ("relative_parameter_delta", row.get("relative_parameter_delta", _MISSING)),
        ),
        "parameter_audit_interval_steps": (
            ("update.parameter_audit_interval_steps", _get(row, "update", "parameter_audit_interval_steps")),
            ("parameter_audit_interval_steps", row.get("parameter_audit_interval_steps", _MISSING)),
        ),
    }
    present = [(name, value) for name, value in aliases[metric] if value is not _MISSING and value is not None]
    if not present:
        return None, None
    first_name, first_value = present[0]
    first = _as_float(first_value, first_name)
    for name, value in present[1:]:
        other = _as_float(value, name)
        if not (
            first == other
            or (math.isnan(first) and math.isnan(other))
        ):
            raise ValueError(f"conflicting {metric}: {first_name}={first}, {name}={other}")
    return first, first_name


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _content_inventory(
    value: Any, *, label: str, require_sha256: bool
) -> list[dict[str, Any]] | None:
    if value is None:
        if require_sha256:
            raise ValueError(f"{label} must be a non-empty file inventory")
        return None
    if not isinstance(value, list) or (require_sha256 and not value):
        raise ValueError(f"{label} must be a non-empty file inventory")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(f"{label}[{index}] must be an object")
        name = entry.get("name")
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"{label}[{index}].name is missing or duplicated")
        seen.add(name)
        parsed_size = _as_int(size, f"{label}[{index}].bytes")
        if parsed_size < 0:
            raise ValueError(f"{label}[{index}].bytes must be non-negative")
        if require_sha256 and not _is_sha256(digest):
            raise ValueError(f"{label}[{index}].sha256 must be a lowercase SHA256")
        output.append(
            {
                "name": name,
                "bytes": parsed_size,
                "sha256": str(digest) if digest is not None else None,
            }
        )
    return sorted(output, key=lambda entry: str(entry["name"]))


def _paired_identity(summary: dict[str, Any], run_root: Path) -> dict[str, Any]:
    parameters = _required(summary, run_root, "parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{run_root}: parameters must be an object")
    schema_version = _as_int(
        _required(summary, run_root, "schema_version"), "schema_version"
    )
    verification_status = _get(
        summary, "artifact_identity", "verification_status", default=None
    )
    fully_verified = (
        schema_version >= 3 and verification_status == FULLY_VERIFIED_IDENTITY
    )
    numeric_runtime, numeric_runtime_status = _numeric_runtime_identity(summary)

    def model_identity(kind: str) -> dict[str, Any]:
        model = _required(summary, run_root, "models", kind)
        if not isinstance(model, dict):
            raise ValueError(f"{run_root}: models.{kind} must be an object")
        hashes = {
            key: value
            for key, value in model.items()
            if (
                key.endswith("_sha256") or key.endswith(".json_sha256")
            )
            # ``dflash_tts_reference`` historically flattened the live module
            # layout next to immutable checkpoint hashes.  Drafter-LoRA wraps
            # Linear modules and therefore changes this runtime-only digest,
            # even though config/index/weight content is the same checkpoint.
            # Keep old artifacts pairable without weakening any content hash.
            and key != "layout_sha256"
        }
        return {
            "declared_revision": str(
                _required(summary, run_root, "models", kind, "declared_revision")
            ),
            "hashes": hashes,
            "weight_files": _content_inventory(
                model.get("weight_files"),
                label=f"models.{kind}.weight_files",
                require_sha256=fully_verified,
            ),
        }

    target_identity = model_identity("target")
    draft_identity = model_identity("draft")
    tokenizer = summary.get("tokenizer")
    tokenizer_identity: dict[str, Any] | None = None
    if tokenizer is not None:
        if not isinstance(tokenizer, dict):
            raise ValueError(f"{run_root}: tokenizer must be an object")
        tokenizer_identity = {
            "files": _content_inventory(
                tokenizer.get("files"),
                label="tokenizer.files",
                require_sha256=fully_verified,
            ),
            "content_identity_sha256": tokenizer.get(
                "content_identity_sha256"
            ),
        }
    elif fully_verified:
        raise ValueError(f"{run_root}: verified schema requires tokenizer identity")
    rendered_input = _get(
        summary, "dataset", "rendered_input_token_ids", default=None
    )
    if rendered_input is not None and not isinstance(rendered_input, dict):
        raise ValueError(
            f"{run_root}: dataset.rendered_input_token_ids must be an object"
        )
    if fully_verified:
        if not isinstance(rendered_input, dict):
            raise ValueError(
                f"{run_root}: verified schema requires rendered input token identity"
            )
        if rendered_input.get("serialization") != "int64_le_c_order_v1":
            raise ValueError(f"{run_root}: unsupported rendered input serialization")
        shape = rendered_input.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or _as_int(shape[0], "rendered input batch") != 1
            or _as_int(shape[1], "rendered input length") <= 0
        ):
            raise ValueError(f"{run_root}: invalid rendered input token shape")
        if not _is_sha256(rendered_input.get("sha256")):
            raise ValueError(
                f"{run_root}: rendered input token identity lacks SHA256"
            )
    identity = {
        "artifact_schema_version": schema_version,
        "identity_verification_status": (
            FULLY_VERIFIED_IDENTITY if fully_verified else "legacy_unverified"
        ),
        "numeric_runtime_identity": numeric_runtime,
        "numeric_runtime_status": numeric_runtime_status,
        "reference_source_sha256": str(_required(summary, run_root, "reference", "source_sha256")),
        "harness_source_sha256": _coalesce(
            "harness_source_sha256",
            (
                ("harness.source_sha256", _get(summary, "harness", "source_sha256")),
                ("harness_source_sha256", summary.get("harness_source_sha256", _MISSING)),
            ),
        ),
        "reference_transformers_version": _get(
            summary, "reference", "transformers_version", default=None
        ),
        "target_revision": target_identity["declared_revision"],
        "draft_revision": draft_identity["declared_revision"],
        "target_artifact_identity": target_identity,
        "draft_artifact_identity": draft_identity,
        "tokenizer_artifact_identity": tokenizer_identity,
        "rendered_input_token_ids": rendered_input,
        "dataset_sha256": str(_required(summary, run_root, "dataset", "sha256")),
        "dataset_revision": str(_required(summary, run_root, "dataset", "declared_revision")),
        "sample_index": _as_int(_required(summary, run_root, "dataset", "sample_index"), "sample_index"),
        "sample_id": str(_required(summary, run_root, "dataset", "sample_id")),
        "input_format": _get(summary, "dataset", "input_format", default=None),
        "thinking_effective_via_chat_template": _get(
            summary,
            "dataset",
            "thinking_effective_via_chat_template",
            default=None,
        ),
    }
    for key in PAIR_PARAMETER_KEYS:
        if key not in parameters:
            raise ValueError(f"{run_root}: missing core identity field parameters.{key}")
        identity[key] = parameters[key]
    for key in PAIR_OPTIONAL_PARAMETER_KEYS:
        identity[key] = parameters.get(key)
    return identity


def _validate_run(
    run_root: Path,
    summary: dict[str, Any],
    rounds: list[dict[str, Any]],
) -> None:
    schema_version = _as_int(_required(summary, run_root, "schema_version"), "schema_version")
    if schema_version not in {1, 2, 3}:
        raise ValueError(f"{run_root}: unsupported schema_version {schema_version}")
    if _required(summary, run_root, "status") != "complete_reference_run":
        raise ValueError(f"{run_root}: run is not complete_reference_run")
    mode = str(_required(summary, run_root, "mode"))
    if mode not in ADAPTED_MODES | {"static"}:
        raise ValueError(f"{run_root}: unsupported mode {mode!r}")
    _required(summary, run_root, "trainable_scope")
    identity = _paired_identity(summary, run_root)
    generation = _required(summary, run_root, "generation")
    if not isinstance(generation, dict):
        raise ValueError(f"{run_root}: generation must be an object")
    parameters = _required(summary, run_root, "parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{run_root}: parameters must be an object")
    declared_rounds = _as_int(_required(summary, run_root, "generation", "rounds"), "generation.rounds")
    if not rounds:
        raise ValueError(f"{run_root}: empty rounds.jsonl")
    if len(rounds) != declared_rounds:
        raise ValueError(
            f"{run_root}: rounds.jsonl has {len(rounds)} rows, summary declares {declared_rounds}"
        )
    expected_prefix = _as_int(
        _required(summary, run_root, "generation", "num_input_tokens"),
        "generation.num_input_tokens",
    )
    block_size = _as_int(identity["block_size"], "parameters.block_size")
    expected_trainable_count, trainable_tensors, _layout, runtime_layout = _trainable_layout(
        summary, mode
    )
    summary_optimizer_steps = _as_int(
        _required(summary, run_root, "generation", "optimizer_steps"),
        "generation.optimizer_steps",
    )
    if mode == "static":
        if summary_optimizer_steps != 0:
            raise ValueError(f"{run_root}: static run has optimizer steps")
        if expected_trainable_count not in {None, 0} or trainable_tensors not in {
            None,
            0,
        }:
            raise ValueError(f"{run_root}: static run declares trainable parameters")
        if parameters.get("optimizer") is not None:
            raise ValueError(f"{run_root}: static run declares an optimizer")
    elif expected_trainable_count is not None and expected_trainable_count <= 0:
        raise ValueError(f"{run_root}: adapted run has no trainable parameters")
    if schema_version >= 2:
        if identity["harness_source_sha256"] is None:
            raise ValueError(f"{run_root}: schema v2 requires harness.source_sha256")
        harness_schema = _as_int(
            _required(summary, run_root, "harness", "artifact_schema_version"),
            "harness.artifact_schema_version",
        )
        if harness_schema != schema_version:
            raise ValueError(f"{run_root}: harness artifact schema mismatch")
        declared_rounds = str(
            _required(summary, run_root, "output", "rounds_jsonl")
        )
        if Path(declared_rounds).is_absolute() or declared_rounds != "rounds.jsonl":
            raise ValueError(
                f"{run_root}: schema v2 output.rounds_jsonl must be rounds.jsonl"
            )
        if expected_trainable_count is None or runtime_layout is None:
            raise ValueError(
                f"{run_root}: schema v2 requires trainable count and runtime layout"
            )
    if schema_version >= 3:
        if identity["identity_verification_status"] != FULLY_VERIFIED_IDENTITY:
            raise ValueError(
                f"{run_root}: schema v3 requires {FULLY_VERIFIED_IDENTITY}"
            )
        rendered_shape = identity["rendered_input_token_ids"]["shape"]
        if _as_int(rendered_shape[1], "rendered input length") != expected_prefix:
            raise ValueError(
                f"{run_root}: rendered input length does not match num_input_tokens"
            )
    applied_update_count = 0
    for index, row in enumerate(rounds):
        row_schema = row.get("schema_version", schema_version)
        if _as_int(row_schema, "round.schema_version") != schema_version:
            raise ValueError(f"{run_root}: schema drift at round {index}")
        round_index = _as_int(_required(row, run_root, "round_index"), "round_index")
        if round_index != index:
            raise ValueError(f"{run_root}: non-contiguous round_index at {index}")
        try:
            prefix = _round_prefix(row)
        except ValueError as exc:
            raise ValueError(f"{run_root}: round {index}: {exc}") from exc
        if prefix != expected_prefix:
            raise ValueError(f"{run_root}: broken prefix chain at round {index}")
        acceptance = _as_int(_required(row, run_root, "acceptance_length"), "acceptance_length")
        accepted = _as_int(_required(row, run_root, "accepted_draft_tokens"), "accepted_draft_tokens")
        committed = _required(row, run_root, "committed_token_ids")
        if not isinstance(committed, list) or len(committed) != acceptance:
            raise ValueError(f"{run_root}: invalid committed token count at round {index}")
        if acceptance != accepted + 1 or accepted < 0 or acceptance > block_size:
            raise ValueError(f"{run_root}: invalid DFlash acceptance at round {index}")
        if row.get("sample_id", identity["sample_id"]) != identity["sample_id"]:
            raise ValueError(f"{run_root}: sample_id drift at round {index}")
        if row.get("mode", mode) != mode:
            raise ValueError(f"{run_root}: mode drift at round {index}")
        if schema_version >= 2:
            provenance = _required(row, run_root, "provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"{run_root}: round {index} provenance must be an object")
            expected_provenance = {
                "reference_source_sha256": identity["reference_source_sha256"],
                "target_declared_revision": identity["target_revision"],
                "draft_declared_revision": identity["draft_revision"],
                "dataset_declared_revision": identity["dataset_revision"],
                "dataset_sha256": identity["dataset_sha256"],
                "harness_source_sha256": identity["harness_source_sha256"],
            }
            for key, expected in expected_provenance.items():
                if provenance.get(key) != expected:
                    raise ValueError(
                        f"{run_root}: provenance {key} drift at round {index}"
                    )
            if _as_int(
                _required(row, run_root, "trainable_parameter_count"),
                "round.trainable_parameter_count",
            ) != expected_trainable_count:
                raise ValueError(
                    f"{run_root}: trainable parameter count drift at round {index}"
                )
            if str(_required(row, run_root, "parameter_layout_sha256")) != runtime_layout:
                raise ValueError(
                    f"{run_root}: runtime parameter layout drift at round {index}"
                )
            if row.get("draft_cache_policy") != identity["draft_cache_policy"]:
                raise ValueError(f"{run_root}: cache policy drift at round {index}")
        loss, _source = _round_scalar(row, "loss")
        applied = bool(_get(row, "update", "applied", default=loss is not None))
        applied_update_count += int(applied)
        optimizer_step = _get(row, "update", "optimizer_step")
        parameters_with_grad = _get(row, "update", "parameters_with_grad")
        if mode == "static" and (
            applied
            or loss is not None
            or optimizer_step not in {_MISSING, None}
            or parameters_with_grad not in {_MISSING, None, 0}
        ):
            raise ValueError(f"{run_root}: static update evidence at round {index}")
        if mode != "static" and applied and loss is None:
            raise ValueError(f"{run_root}: applied update without loss at round {index}")
        if mode != "static" and applied:
            if optimizer_step in {_MISSING, None}:
                raise ValueError(
                    f"{run_root}: applied update without optimizer_step at round {index}"
                )
            if parameters_with_grad in {_MISSING, None} or _as_int(
                parameters_with_grad, "parameters_with_grad"
            ) <= 0:
                raise ValueError(
                    f"{run_root}: applied update without gradients at round {index}"
                )
        expected_prefix += acceptance
    if applied_update_count != summary_optimizer_steps:
        raise ValueError(
            f"{run_root}: optimizer step count {summary_optimizer_steps} does not "
            f"match {applied_update_count} applied rounds"
        )


def _load_run(run_root: Path) -> dict[str, Any]:
    root = run_root.expanduser().resolve()
    summary_path = root / "summary.json"
    rounds_path = root / "rounds.jsonl"
    if not summary_path.is_file() or not rounds_path.is_file():
        raise ValueError(f"{root}: expected summary.json and rounds.jsonl")
    summary = _read_json(summary_path)
    rounds = _read_jsonl(rounds_path)
    _validate_run(root, summary, rounds)
    identity = _paired_identity(summary, root)
    rounds_sha256 = _sha256_file(rounds_path)
    declared_rounds_sha256 = _coalesce(
        "rounds_sha256",
        (
            ("rounds_sha256", summary.get("rounds_sha256", _MISSING)),
            ("output.rounds_sha256", _get(summary, "output", "rounds_sha256")),
        ),
    )
    if declared_rounds_sha256 is not None and str(declared_rounds_sha256) != rounds_sha256:
        raise ValueError(
            f"{root}: rounds_sha256 mismatch: declared {declared_rounds_sha256}, actual {rounds_sha256}"
        )
    return {
        "root": root,
        "summary": summary,
        "rounds": rounds,
        "summary_sha256": _sha256_file(summary_path),
        "rounds_sha256": rounds_sha256,
        "pair_identity": identity,
        "pair_key": _sha256_json(identity),
    }


def _bucket_rows(run: dict[str, Any], bucket_size: int) -> list[dict[str, Any]]:
    summary = run["summary"]
    rounds = run["rounds"]
    mode = str(summary["mode"])
    parameters = summary["parameters"]
    generation = summary["generation"]
    identity = run["pair_identity"]
    optimizer = _parameter_alias(parameters, "optimizer")
    lr = _parameter_alias(parameters, "lr", "learning_rate")
    weight_decay = _parameter_alias(parameters, "weight_decay", "adam_weight_decay")
    rank = _parameter_alias(parameters, "rank", "adapter_rank")
    if mode in LORA_MODES | {"output-residual"} and rank is None:
        raise ValueError(f"{run['root']}: {mode} requires a recorded rank")
    parsed_rank = None if rank is None else _as_int(rank, "rank")
    if mode != "static" and (optimizer is None or lr is None or weight_decay is None):
        raise ValueError(
            f"{run['root']}: adapted run requires optimizer, lr, and weight decay"
        )
    optimizer_memory = _optimizer_memory(summary, mode)
    def memory_sum(*keys: str) -> int | None:
        values = [optimizer_memory[key] for key in keys]
        return (
            None
            if any(value is None for value in values)
            else sum(int(value) for value in values)
        )

    optimizer_persistent = optimizer_memory["persistent_bytes"]
    if optimizer_persistent is None:
        optimizer_persistent = memory_sum(
            "forward_parameter_bytes",
            "master_parameter_bytes",
            "optimizer_moment_bytes",
        )
    optimizer_update_peak = optimizer_memory["estimated_update_peak_bytes"]
    memory_evidence = (
        "static_zero"
        if mode == "static"
        else "exact_declared_optimizer_tensor_ledger"
    )
    if optimizer_update_peak is None:
        optimizer_update_peak = memory_sum(
            "forward_parameter_bytes",
            "master_parameter_bytes",
            "master_gradient_bytes",
            "optimizer_moment_bytes",
        )
        memory_evidence = (
            "static_zero"
            if mode == "static"
            else (
                "legacy_lower_bound_missing_forward_gradient"
                if optimizer_update_peak is not None
                else "missing"
            )
        )
    incremental_optimizer_resident = memory_sum(
        "master_parameter_bytes", "optimizer_moment_bytes"
    )
    if mode not in {"static", "full-drafter"}:
        forward_bytes = optimizer_memory["forward_parameter_bytes"]
        incremental_optimizer_resident = (
            None
            if incremental_optimizer_resident is None or forward_bytes is None
            else incremental_optimizer_resident + int(forward_bytes)
        )
    (
        trainable_count,
        trainable_tensors,
        layout_hash,
        runtime_layout_hash,
    ) = _trainable_layout(summary, mode)
    peak_hbm, peak_hbm_reserved = _hbm_summary(summary)
    hbm_phases = _hbm_phases(summary)
    whole_process_peak_hbm = hbm_phases[
        "whole_process_peak_running_peak_allocated_bytes"
    ]
    whole_process_peak_hbm_reserved = hbm_phases[
        "whole_process_peak_running_peak_reserved_bytes"
    ]
    projection_artifact_sha256 = _projection_artifact_sha256(summary)
    layout_family_sha256, layout_family_status = _layout_family(
        summary, mode, parsed_rank
    )
    (
        runtime_fingerprint_sha256,
        runtime_fingerprint_status,
        runtime_fingerprint_json,
    ) = _runtime_fingerprint(summary)
    target = summary["models"]["target"]
    draft = summary["models"]["draft"]
    source_identity = {
        "reference_source_sha256": identity["reference_source_sha256"],
        "dataset_sha256": identity["dataset_sha256"],
        "dataset_revision": identity["dataset_revision"],
        "target_revision": identity["target_revision"],
        "draft_revision": identity["draft_revision"],
        "target_config_sha256": target.get("config.json_sha256"),
        "draft_config_sha256": draft.get("config.json_sha256"),
        "target_artifact_identity_sha256": _sha256_json(
            identity["target_artifact_identity"]
        ),
        "draft_artifact_identity_sha256": _sha256_json(
            identity["draft_artifact_identity"]
        ),
        "parameter_layout_sha256": layout_hash,
        "runtime_parameter_layout_sha256": runtime_layout_hash,
        "projection_artifact_sha256": projection_artifact_sha256,
        "layout_family_sha256": layout_family_sha256,
        "layout_family_status": layout_family_status,
        "identity_verification_status": identity[
            "identity_verification_status"
        ],
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "runtime_fingerprint_status": runtime_fingerprint_status,
        "numeric_runtime_identity_sha256": (
            None
            if identity["numeric_runtime_identity"] is None
            else _sha256_json(identity["numeric_runtime_identity"])
        ),
        "numeric_runtime_status": identity["numeric_runtime_status"],
        "harness_source_sha256": identity["harness_source_sha256"],
    }
    output_tokens = _as_int(
        generation["num_output_tokens"], "generation.num_output_tokens"
    )
    target_calls = generation.get("target_calls")
    physical_target_calls: int | None = None
    canonical_audit_target_calls: int | None = None
    if target_calls is not None:
        if not isinstance(target_calls, dict):
            raise ValueError("generation.target_calls must be an object")
        physical_target_calls = _as_int(
            _required(target_calls, run["root"], "physical_total"),
            "generation.target_calls.physical_total",
        )
        canonical_audit_target_calls = _as_int(
            target_calls.get("canonical_prefill", 0),
            "generation.target_calls.canonical_prefill",
        ) + _as_int(
            target_calls.get("canonical_commit_verify_decode", 0),
            "generation.target_calls.canonical_commit_verify_decode",
        )
        if physical_target_calls < canonical_audit_target_calls:
            raise ValueError(
                "generation target-call ledger has canonical calls above total"
            )
    try:
        effective_output_by_round = _effective_output_by_round(summary, rounds)
    except ValueError as exc:
        raise ValueError(f"{run['root']}: {exc}") from exc

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rounds:
        prefix_int = _round_prefix(row)
        grouped[(prefix_int // bucket_size) * bucket_size].append(row)

    run_acceptance = [_as_int(row["acceptance_length"], "acceptance_length") for row in rounds]
    run_accepted = [_as_int(row["accepted_draft_tokens"], "accepted_draft_tokens") for row in rounds]
    decode_seconds = generation.get("decode_seconds")
    output: list[dict[str, Any]] = []
    for start in sorted(grouped):
        selected = grouped[start]
        acceptance = [_as_int(row["acceptance_length"], "acceptance_length") for row in selected]
        accepted = [_as_int(row["accepted_draft_tokens"], "accepted_draft_tokens") for row in selected]
        algorithmic_committed = sum(acceptance)
        committed = sum(
            effective_output_by_round[_as_int(row["round_index"], "round_index")]
            for row in selected
        )
        if committed <= 0:
            raise ValueError(f"{run['root']}: bucket contains no effective output tokens")
        bucket_physical_calls: int | None = 0
        bucket_canonical_calls: int | None = 0
        for selected_row in selected:
            calls = selected_row.get("target_calls")
            if not isinstance(calls, dict):
                bucket_physical_calls = None
                bucket_canonical_calls = None
                break
            assert bucket_physical_calls is not None
            assert bucket_canonical_calls is not None
            bucket_physical_calls += _as_int(
                calls.get("physical_total"), "round.target_calls.physical_total"
            )
            bucket_canonical_calls += _as_int(
                calls.get("canonical_commit_verify", 0),
                "round.target_calls.canonical_commit_verify",
            )
        scalar_values: dict[str, list[float]] = defaultdict(list)
        scalar_sources: dict[str, set[str]] = defaultdict(set)
        observed_counts: dict[str, int] = defaultdict(int)
        nonfinite_counts: dict[str, int] = defaultdict(int)
        applied_updates = 0
        gradient_tensor_counts: list[int] = []
        optimizer_steps: list[int] = []
        prefixes: list[int] = []
        round_hbm: dict[str, list[int]] = defaultdict(list)
        for row in selected:
            prefixes.append(
                _as_int(
                    _round_prefix(row), "prefix_length_before"
                )
            )
            loss, _loss_source = _round_scalar(row, "loss")
            applied = bool(_get(row, "update", "applied", default=loss is not None))
            applied_updates += int(applied)
            for metric in (
                "loss",
                "distillation_kl",
                "proximal_kl",
                "grad_norm",
                "backward_cuda_us",
                "optimizer_cuda_us",
                "update_cuda_us",
                "parameter_delta_l2",
                "parameter_displacement_l2",
                "parameter_l2",
                "relative_parameter_delta",
                "parameter_audit_interval_steps",
            ):
                value, source = _round_scalar(row, metric)
                if value is None:
                    continue
                observed_counts[metric] += 1
                if source is not None:
                    scalar_sources[metric].add(source)
                if math.isfinite(value):
                    scalar_values[metric].append(value)
                else:
                    nonfinite_counts[metric] += 1
            parameter_tensors = _get(row, "update", "parameters_with_grad")
            if parameter_tensors is not _MISSING and parameter_tensors is not None and applied:
                gradient_tensor_counts.append(_as_int(parameter_tensors, "parameters_with_grad"))
            step = _get(row, "update", "optimizer_step")
            if step is not _MISSING and step is not None:
                optimizer_steps.append(_as_int(step, "optimizer_step"))
            hbm = row.get("hbm_bytes")
            if hbm is not None:
                if not isinstance(hbm, dict):
                    raise ValueError("round hbm_bytes must be an object or null")
                for key in (
                    "allocated_end",
                    "reserved_end",
                    "running_peak_allocated",
                    "running_peak_reserved",
                ):
                    value = hbm.get(key)
                    if value is None:
                        continue
                    parsed = _as_int(value, f"hbm_bytes.{key}")
                    if parsed < 0:
                        raise ValueError(f"hbm_bytes.{key} must be non-negative")
                    round_hbm[key].append(parsed)

        row = {
            "schema_version": SCHEMA_VERSION,
            "run_root": str(run["root"]),
            "pair_key": run["pair_key"],
            "sample_id": identity["sample_id"],
            "sample_index": identity["sample_index"],
            "seed": identity["seed"],
            "mode": mode,
            "trainable_scope": summary["trainable_scope"],
            "optimizer": None if optimizer is None else str(optimizer).lower(),
            "learning_rate": None if lr is None else _as_float(lr, "learning_rate"),
            "weight_decay": None if weight_decay is None else _as_float(weight_decay, "weight_decay"),
            "rank": parsed_rank,
            "adapter_seed": parameters.get("adapter_seed"),
            "update_stride": parameters.get("update_stride"),
            "proximal_lambda": parameters.get("proximal_lambda"),
            "position_weighting": parameters.get("position_weighting"),
            "position_decay_gamma": parameters.get("position_decay_gamma"),
            "loss_reduction": parameters.get("loss_reduction"),
            "adam_betas": parameters.get("adam_betas"),
            "adam_eps": parameters.get("adam_eps"),
            "audit_cuda_timing": parameters.get("audit_cuda_timing", False),
            "parameter_audit_stride": parameters.get("parameter_audit_stride", 0),
            "draft_cache_policy": identity["draft_cache_policy"],
            "runtime_fingerprint_json": runtime_fingerprint_json,
            "gradient_semantics": _get(summary, "reconstruction_status", "gradient_semantics", default=None),
            "prefix_bucket_size": bucket_size,
            "prefix_bucket_start": start,
            "prefix_bucket_end_exclusive": start + bucket_size,
            "prefix_len_observed_min": min(prefixes),
            "prefix_len_observed_max": max(prefixes),
            "prefix_len_observed_mean": statistics.fmean(prefixes),
            "prefix_len_observed_median": statistics.median(prefixes),
            "verification_calls": len(selected),
            "committed_output_tokens": committed,
            "algorithmic_committed_tokens": algorithmic_committed,
            "paper_acceptance_length": statistics.fmean(acceptance),
            "accepted_drafts_per_verify": statistics.fmean(accepted),
            "committed_tokens_per_verify": committed / len(selected),
            "algorithmic_committed_tokens_per_verify": algorithmic_committed
            / len(selected),
            "target_calls_per_output_token": len(selected) / committed,
            "logical_block_target_calls_per_output_token": (
                len(selected) / committed
            ),
            "physical_target_calls_per_output_token": (
                None
                if bucket_physical_calls is None
                else bucket_physical_calls / committed
            ),
            "canonical_audit_target_calls_per_output_token": (
                None
                if bucket_canonical_calls is None
                else bucket_canonical_calls / committed
            ),
            "target_call_metric_scope": (
                "headline_algorithmic_metric_is_logical_block_verify_calls; "
                "physical and canonical calls are reference-audit overhead"
            ),
            "bucket_gain_status": "descriptive_round_composition_not_prefix_matched",
            "updates_applied": applied_updates,
            "loss_mean": _mean(scalar_values["loss"]),
            "loss_median": _median(scalar_values["loss"]),
            "loss_observed_count": observed_counts["loss"],
            "loss_finite_count": len(scalar_values["loss"]),
            "loss_nonfinite_count": nonfinite_counts["loss"],
            "loss_source_fields": ",".join(sorted(scalar_sources["loss"])),
            "distillation_kl_mean": _mean(scalar_values["distillation_kl"]),
            "proximal_kl_mean": _mean(scalar_values["proximal_kl"]),
            "grad_norm_mean": _mean(scalar_values["grad_norm"]),
            "backward_cuda_us_mean": _mean(scalar_values["backward_cuda_us"]),
            "optimizer_cuda_us_mean": _mean(scalar_values["optimizer_cuda_us"]),
            "update_cuda_us_mean": _mean(scalar_values["update_cuda_us"]),
            "parameter_audit_count": len(scalar_values["parameter_delta_l2"]),
            "parameter_delta_l2_mean": _mean(scalar_values["parameter_delta_l2"]),
            "parameter_delta_l2_median": _median(scalar_values["parameter_delta_l2"]),
            "parameter_delta_l2_max": max(scalar_values["parameter_delta_l2"], default=None),
            "parameter_displacement_l2_last": scalar_values["parameter_displacement_l2"][-1] if scalar_values["parameter_displacement_l2"] else None,
            "parameter_displacement_l2_max": max(scalar_values["parameter_displacement_l2"], default=None),
            "parameter_l2_last": scalar_values["parameter_l2"][-1] if scalar_values["parameter_l2"] else None,
            "relative_parameter_delta_mean": _mean(scalar_values["relative_parameter_delta"]),
            "relative_parameter_delta_max": max(scalar_values["relative_parameter_delta"], default=None),
            "parameter_audit_interval_steps_sum": (
                sum(scalar_values["parameter_audit_interval_steps"])
                if scalar_values["parameter_audit_interval_steps"]
                else None
            ),
            "optimizer_step_min": min(optimizer_steps) if optimizer_steps else None,
            "optimizer_step_max": max(optimizer_steps) if optimizer_steps else None,
            "trainable_parameter_count": trainable_count,
            "trainable_parameter_tensors": trainable_tensors,
            "gradient_parameter_tensors_min": min(gradient_tensor_counts) if gradient_tensor_counts else None,
            "gradient_parameter_tensors_max": max(gradient_tensor_counts) if gradient_tensor_counts else None,
            "peak_hbm_bytes": peak_hbm,
            "peak_hbm_gib": None if peak_hbm is None else peak_hbm / (1 << 30),
            "peak_hbm_reserved_bytes": peak_hbm_reserved,
            "peak_hbm_reserved_gib": None if peak_hbm_reserved is None else peak_hbm_reserved / (1 << 30),
            "peak_hbm_scope": "run_peak_repeated_across_prefix_buckets",
            "peak_hbm_measurement_scope": (
                "decode_window_after_optimizer_peak_reset; includes resident "
                "models and optimizer but excludes earlier load peak"
            ),
            "whole_process_peak_hbm_bytes": whole_process_peak_hbm,
            "whole_process_peak_hbm_gib": (
                None
                if whole_process_peak_hbm is None
                else whole_process_peak_hbm / (1 << 30)
            ),
            "whole_process_peak_hbm_reserved_bytes": (
                whole_process_peak_hbm_reserved
            ),
            "whole_process_peak_hbm_scope": (
                "max(pre_decode_running_peak, post_reset_decode_running_peak)"
            ),
            **hbm_phases,
            "hbm_round_samples": len(round_hbm["allocated_end"]),
            "hbm_allocated_end_max_bytes": max(round_hbm["allocated_end"], default=None),
            "hbm_reserved_end_max_bytes": max(round_hbm["reserved_end"], default=None),
            "hbm_running_peak_allocated_max_bytes": max(round_hbm["running_peak_allocated"], default=None),
            "hbm_running_peak_reserved_max_bytes": max(round_hbm["running_peak_reserved"], default=None),
            "hbm_round_scope": "max_observed_in_true_prefix_bucket",
            **optimizer_memory,
            "optimizer_memory_evidence": memory_evidence,
            "optimizer_resident_bytes": optimizer_persistent,
            "optimizer_resident_gib": None if optimizer_persistent is None else optimizer_persistent / (1 << 30),
            "optimizer_update_peak_bytes": optimizer_update_peak,
            "optimizer_update_peak_gib": None if optimizer_update_peak is None else optimizer_update_peak / (1 << 30),
            "optimizer_bytes_per_trainable_parameter": (
                None
                if optimizer_persistent is None or not trainable_count
                else optimizer_persistent / trainable_count
            ),
            "declared_incremental_optimizer_resident_bytes": (
                incremental_optimizer_resident
            ),
            "optimizer_ledger_excludes": (
                "autograd_activations,source_logit_clone,optimizer_workspace"
            ),
            "run_rounds": len(rounds),
            "run_num_input_tokens": _as_int(generation["num_input_tokens"], "num_input_tokens"),
            "run_num_output_tokens": output_tokens,
            "run_max_prefix_len_before": max(
                _round_prefix(row_)
                for row_ in rounds
            ),
            "run_paper_acceptance_length": statistics.fmean(run_acceptance),
            "run_accepted_drafts_per_verify": statistics.fmean(run_accepted),
            "run_target_calls_per_output_token": len(rounds) / max(output_tokens, 1),
            "run_logical_block_target_calls_per_output_token": (
                len(rounds) / max(output_tokens, 1)
            ),
            "run_physical_target_calls_per_output_token": (
                None
                if physical_target_calls is None
                else physical_target_calls / max(output_tokens, 1)
            ),
            "run_canonical_audit_target_calls_per_output_token": (
                None
                if canonical_audit_target_calls is None
                else canonical_audit_target_calls / max(output_tokens, 1)
            ),
            "reference_decode_seconds": None if decode_seconds is None else _as_float(decode_seconds, "decode_seconds"),
            "reference_tokens_per_second": (
                None
                if decode_seconds is None or _as_float(decode_seconds, "decode_seconds") <= 0
                else output_tokens / _as_float(decode_seconds, "decode_seconds")
            ),
            "summary_sha256": run["summary_sha256"],
            "rounds_sha256": run["rounds_sha256"],
            "artifact_set_sha256": _sha256_json(
                {"summary": run["summary_sha256"], "rounds": run["rounds_sha256"]}
            ),
            **source_identity,
            "source_identity_sha256": _sha256_json(source_identity),
            "exact_output_token_match_static": None,
            "paired_static_summary_sha256": None,
            "static_paper_acceptance_length": None,
            "static_accepted_drafts_per_verify": None,
            "static_target_calls_per_output_token": None,
            "paper_acceptance_length_gain_vs_static": None,
            "accepted_draft_gain_vs_static": None,
            "target_calls_per_output_token_change_vs_static": None,
            "peak_hbm_over_static_bytes": None,
            "whole_process_peak_hbm_over_static_bytes": None,
            "hbm_pairing_status": "unavailable_no_static_baseline",
            "static_runtime_fingerprint_sha256": None,
            "static_run_paper_acceptance_length": None,
            "static_run_accepted_drafts_per_verify": None,
            "static_run_target_calls_per_output_token": None,
            "run_paper_acceptance_length_gain_vs_static": None,
            "run_accepted_draft_gain_vs_static": None,
            "run_target_calls_per_output_token_change_vs_static": None,
            "run_gain_status": "unavailable_no_static_baseline",
        }
        output.append(row)
    return output


def build_long_table(run_roots: Sequence[Path], *, bucket_size: int = 4096) -> list[dict[str, Any]]:
    """Load, validate, pair, and bucket DFlash reference artifacts."""

    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    if not run_roots:
        raise ValueError("at least one run root is required")
    resolved = [Path(path).expanduser().resolve() for path in run_roots]
    if len(resolved) != len(set(resolved)):
        raise ValueError("duplicate run roots are not allowed")
    runs = [_load_run(path) for path in resolved]
    static_by_pair: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run["summary"]["mode"] != "static":
            continue
        pair_identity = run["pair_identity"]
        if (
            pair_identity["identity_verification_status"]
            == FULLY_VERIFIED_IDENTITY
            and pair_identity["numeric_runtime_status"]
            != "complete_numeric_runtime_v1"
        ):
            # Keep the artifact readable, but unknown numerical state cannot
            # certify a formal Static/adaptation pair.
            continue
        pair_key = run["pair_key"]
        if pair_key in static_by_pair:
            raise ValueError(f"ambiguous static baseline for pair_key {pair_key}")
        static_by_pair[pair_key] = run

    rows_by_run = {str(run["root"]): _bucket_rows(run, bucket_size) for run in runs}
    static_bucket: dict[tuple[str, int], dict[str, Any]] = {}
    for pair_key, run in static_by_pair.items():
        for row in rows_by_run[str(run["root"])]:
            static_bucket[(pair_key, int(row["prefix_bucket_start"]))] = row

    output: list[dict[str, Any]] = []
    for run in runs:
        summary = run["summary"]
        baseline = static_by_pair.get(run["pair_key"])
        baseline_run_row = (
            None
            if baseline is None
            else rows_by_run[str(baseline["root"])][0]
        )
        exact: bool | None = None
        if baseline is not None:
            candidate_tokens = _get(summary, "output", "token_ids")
            baseline_tokens = _get(baseline["summary"], "output", "token_ids")
            if candidate_tokens is not _MISSING and baseline_tokens is not _MISSING:
                exact = candidate_tokens == baseline_tokens
        for row in rows_by_run[str(run["root"])]:
            row = dict(row)
            row["exact_output_token_match_static"] = exact
            if baseline is not None:
                row["paired_static_summary_sha256"] = baseline["summary_sha256"]
            if baseline_run_row is not None:
                row["static_runtime_fingerprint_sha256"] = baseline_run_row[
                    "runtime_fingerprint_sha256"
                ]
                row["static_run_paper_acceptance_length"] = baseline_run_row[
                    "run_paper_acceptance_length"
                ]
                row["static_run_accepted_drafts_per_verify"] = baseline_run_row[
                    "run_accepted_drafts_per_verify"
                ]
                row["static_run_target_calls_per_output_token"] = baseline_run_row[
                    "run_target_calls_per_output_token"
                ]
                if exact is True:
                    row["run_paper_acceptance_length_gain_vs_static"] = (
                        row["run_paper_acceptance_length"]
                        - baseline_run_row["run_paper_acceptance_length"]
                    )
                    row["run_accepted_draft_gain_vs_static"] = (
                        row["run_accepted_drafts_per_verify"]
                        - baseline_run_row["run_accepted_drafts_per_verify"]
                    )
                    row["run_target_calls_per_output_token_change_vs_static"] = (
                        row["run_target_calls_per_output_token"]
                        - baseline_run_row["run_target_calls_per_output_token"]
                    )
                    row["run_gain_status"] = (
                        "run_level_exact_output_pair_fully_verified"
                        if row["identity_verification_status"]
                        == FULLY_VERIFIED_IDENTITY
                        else "pilot_descriptive_legacy_identity"
                    )
                else:
                    row["run_gain_status"] = (
                        "unavailable_output_token_ids_not_exact"
                    )
                candidate_runtime = row["runtime_fingerprint_sha256"]
                static_runtime = baseline_run_row[
                    "runtime_fingerprint_sha256"
                ]
                if (
                    row["runtime_fingerprint_status"]
                    != "complete_hbm_comparable_v1"
                    or baseline_run_row["runtime_fingerprint_status"]
                    != "complete_hbm_comparable_v1"
                ):
                    row["hbm_pairing_status"] = (
                        "pilot_descriptive_missing_or_incomplete_runtime_fingerprint"
                    )
                elif candidate_runtime != static_runtime:
                    row["hbm_pairing_status"] = (
                        "unavailable_runtime_fingerprint_mismatch"
                    )
                else:
                    row["hbm_pairing_status"] = (
                        "eligible_runtime_fingerprint_match"
                    )
            base = static_bucket.get((run["pair_key"], int(row["prefix_bucket_start"])))
            if base is not None:
                row["static_paper_acceptance_length"] = base["paper_acceptance_length"]
                row["static_accepted_drafts_per_verify"] = base["accepted_drafts_per_verify"]
                row["static_target_calls_per_output_token"] = base["target_calls_per_output_token"]
                if exact is True:
                    row["bucket_gain_status"] = (
                        "descriptive_exact_output_pair_round_composition_not_"
                        "prefix_matched"
                    )
                    row["paper_acceptance_length_gain_vs_static"] = (
                        row["paper_acceptance_length"]
                        - base["paper_acceptance_length"]
                    )
                    row["accepted_draft_gain_vs_static"] = (
                        row["accepted_drafts_per_verify"]
                        - base["accepted_drafts_per_verify"]
                    )
                    row["target_calls_per_output_token_change_vs_static"] = (
                        row["target_calls_per_output_token"]
                        - base["target_calls_per_output_token"]
                    )
                    if (
                        row["hbm_pairing_status"]
                        == "eligible_runtime_fingerprint_match"
                        and
                        row["peak_hbm_bytes"] is not None
                        and base["peak_hbm_bytes"] is not None
                    ):
                        row["peak_hbm_over_static_bytes"] = (
                            row["peak_hbm_bytes"] - base["peak_hbm_bytes"]
                        )
                    if (
                        row["hbm_pairing_status"]
                        == "eligible_runtime_fingerprint_match"
                        and row["whole_process_peak_hbm_bytes"] is not None
                        and base["whole_process_peak_hbm_bytes"] is not None
                    ):
                        row["whole_process_peak_hbm_over_static_bytes"] = (
                            row["whole_process_peak_hbm_bytes"]
                            - base["whole_process_peak_hbm_bytes"]
                        )
                else:
                    row["bucket_gain_status"] = (
                        "unavailable_output_token_ids_not_exact"
                    )
            output.append(row)
    return sorted(
        output,
        key=lambda row: (
            str(row["sample_id"]),
            int(row["seed"]),
            str(row["draft_cache_policy"]),
            str(row["mode"]),
            -1 if row["rank"] is None else int(row["rank"]),
            -1.0 if row["learning_rate"] is None else float(row["learning_rate"]),
            float(row["weight_decay"] or 0.0),
            int(row["prefix_bucket_start"]),
            str(row["run_root"]),
        ),
    )


def _pareto_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    # HBM is a run peak, so a formal memory/AL frontier must also use one
    # run-level acceptance point.  Coarse context buckets have different round
    # compositions under Static and adaptation; they remain descriptive only.
    candidates_by_run: dict[str, dict[str, Any]] = {}
    for source in rows:
        if source["mode"] == "static":
            continue
        run_root = str(source["run_root"])
        if run_root in candidates_by_run:
            continue
        row = dict(source)
        row.update(
            {
                "prefix_bucket_start": None,
                "prefix_bucket_end_exclusive": None,
                "paper_acceptance_length": row["run_paper_acceptance_length"],
                "accepted_drafts_per_verify": row[
                    "run_accepted_drafts_per_verify"
                ],
                "target_calls_per_output_token": row[
                    "run_target_calls_per_output_token"
                ],
                "paper_acceptance_length_gain_vs_static": row[
                    "run_paper_acceptance_length_gain_vs_static"
                ],
                "accepted_draft_gain_vs_static": row[
                    "run_accepted_draft_gain_vs_static"
                ],
                "target_calls_per_output_token_change_vs_static": row[
                    "run_target_calls_per_output_token_change_vs_static"
                ],
                "bucket_gain_status": "not_used_for_formal_pareto",
            }
        )
        candidates_by_run[run_root] = row
    candidates = list(candidates_by_run.values())
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[str(row["pair_key"])].append(row)
    for group in groups.values():
        for row in group:
            gain = row["run_accepted_draft_gain_vs_static"]
            row["pareto_scope"] = "run_level_exact_output_pair"
            row["pareto_acceptance_basis"] = (
                "run_level_exact_accepted_draft_gain_vs_static"
            )
            row["pareto_acceptance_value"] = gain
            row["pareto_memory_axis"] = "whole_process_peak_hbm_bytes"
            unavailable: list[str] = []
            if row["identity_verification_status"] != FULLY_VERIFIED_IDENTITY:
                unavailable.append("identity_not_fully_verified")
            if row["exact_output_token_match_static"] is not True:
                unavailable.append("output_tokens_not_exact")
            if gain is None:
                unavailable.append("missing_run_level_gain")
            if row["whole_process_peak_hbm_bytes"] is None:
                unavailable.append("missing_whole_process_peak_hbm")
            if (
                row["hbm_pairing_status"]
                != "eligible_runtime_fingerprint_match"
            ):
                unavailable.append("runtime_fingerprint_not_comparable")
            row["pareto_eligible"] = not unavailable
            row["pareto_status"] = (
                "eligible_run_level_exact_pair"
                if not unavailable
                else "unavailable_" + ",".join(unavailable)
            )
            row["pareto_optimal"] = None
        eligible = [row for row in group if row["pareto_eligible"]]
        for row in eligible:
            memory = int(row["whole_process_peak_hbm_bytes"])
            acceptance = float(row["pareto_acceptance_value"])
            dominated = any(
                int(other["whole_process_peak_hbm_bytes"]) <= memory
                and float(other["pareto_acceptance_value"]) >= acceptance
                and (
                    int(other["whole_process_peak_hbm_bytes"]) < memory
                    or float(other["pareto_acceptance_value"]) > acceptance
                )
                for other in eligible
                if other is not row
            )
            row["pareto_optimal"] = not dominated
    return sorted(
        candidates,
        key=lambda row: (
            str(row["pair_key"]),
            str(row["sample_id"]),
            int(row["seed"]),
            row["pareto_optimal"] is not True,
            (
                math.inf
                if row["whole_process_peak_hbm_bytes"] is None
                else int(row["whole_process_peak_hbm_bytes"])
            ),
            str(row["mode"]),
            -1 if row["rank"] is None else int(row["rank"]),
            -1.0 if row["learning_rate"] is None else float(row["learning_rate"]),
            str(row["run_root"]),
        ),
    )


def _stable_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["pair_key"]),
        str(row["sample_id"]),
        int(row["seed"]),
        str(row["mode"]),
        -1 if row["rank"] is None else int(row["rank"]),
        -1.0 if row["learning_rate"] is None else float(row["learning_rate"]),
        float(row["weight_decay"] or 0.0),
        int(row["prefix_bucket_start"]),
        str(row["run_root"]),
    )


def _ablation_comparison_key(
    row: dict[str, Any], *, axis: str
) -> tuple[str | None, str]:
    if axis not in {"learning_rate", "rank"}:
        raise ValueError(f"unsupported ablation axis {axis}")
    controls = {
        key: row.get(key)
        for key in ABLATION_CONTROL_KEYS
        if key != axis
    }
    identity: dict[str, Any] = {
        "pair_key": row["pair_key"],
        "prefix_bucket_start": row["prefix_bucket_start"],
        "controls_except_axis": controls,
    }
    if axis == "learning_rate":
        required = (
            "parameter_layout_sha256",
            "runtime_parameter_layout_sha256",
        )
        missing = [key for key in required if not row.get(key)]
        if row.get("mode") == "output-residual" and not row.get(
            "projection_artifact_sha256"
        ):
            missing.append("projection_artifact_sha256")
        if missing:
            return None, "unavailable_missing_" + ",".join(missing)
        identity["parameterization_identity"] = {
            "trainable_layout_sha256": row["parameter_layout_sha256"],
            "runtime_parameter_layout_sha256": row[
                "runtime_parameter_layout_sha256"
            ],
            "projection_artifact_sha256": row.get(
                "projection_artifact_sha256"
            ),
        }
        status = "comparable_exact_parameter_layout"
    else:
        family = row.get("layout_family_sha256")
        if not family:
            return None, str(
                row.get(
                    "layout_family_status",
                    "unavailable_missing_layout_family_sha256",
                )
            )
        identity["rank_invariant_parameterization_identity"] = family
        status = "comparable_rank_invariant_layout_family"
    return _sha256_json(identity), status


def build_ablation_tables(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return deterministic descriptive views without pooling prompt clusters."""

    ordered = sorted((dict(row) for row in rows), key=_stable_row_key)
    lr: list[dict[str, Any]] = []
    for row in ordered:
        if row["mode"] == "static":
            continue
        key, status = _ablation_comparison_key(row, axis="learning_rate")
        lr.append(
            dict(
                row,
                ablation_family="learning_rate",
                learning_rate_comparison_key=key,
                learning_rate_comparison_status=status,
            )
        )
    rank: list[dict[str, Any]] = []
    for row in ordered:
        if row["mode"] not in LORA_MODES:
            continue
        key, status = _ablation_comparison_key(row, axis="rank")
        rank.append(
            dict(
                row,
                ablation_family="lora_rank",
                rank_comparison_key=key,
                rank_comparison_status=status,
            )
        )
    pareto = [
        dict(row, ablation_family="memory_al_pareto")
        for row in _pareto_table(ordered)
    ]
    return {
        "long": ordered,
        "lr_ablation": lr,
        "memory_al_pareto": pareto,
        "lora_rank_ablation": rank,
    }


def _column_order(rows: Sequence[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    preferred = [
        "schema_version",
        "ablation_family",
        "run_root",
        "sample_id",
        "sample_index",
        "seed",
        "mode",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "rank",
        "draft_cache_policy",
        "prefix_bucket_start",
        "prefix_bucket_end_exclusive",
        "paper_acceptance_length",
        "paper_acceptance_length_gain_vs_static",
        "accepted_drafts_per_verify",
        "accepted_draft_gain_vs_static",
        "logical_block_target_calls_per_output_token",
        "physical_target_calls_per_output_token",
        "canonical_audit_target_calls_per_output_token",
        "target_calls_per_output_token",
        "loss_mean",
        "loss_median",
        "loss_finite_count",
        "trainable_parameter_count",
        "whole_process_peak_hbm_bytes",
        "whole_process_peak_hbm_over_static_bytes",
        "peak_hbm_bytes",
        "optimizer_resident_bytes",
        "pareto_optimal",
        "exact_output_token_match_static",
    ]
    keys = set().union(*(row.keys() for row in rows))
    return [key for key in preferred if key in keys] + sorted(keys.difference(preferred))


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _write_table(
    output_dir: Path,
    name: str,
    rows: Sequence[dict[str, Any]],
    *,
    bucket_size: int,
    parquet: bool,
) -> list[Path]:
    columns = _column_order(rows)
    csv_path = output_dir / f"dflash_tts_{name}.csv"
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(_json_safe(dict(row)) for row in rows)
    csv_tmp.replace(csv_path)
    json_path = output_dir / f"dflash_tts_{name}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "table": name,
        "bucket_size": bucket_size,
        "rows": [_json_safe(dict(row)) for row in rows],
    }
    _atomic_text(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    paths = [csv_path, json_path]
    if parquet:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - project dependency
            raise RuntimeError("--parquet requires the project pandas/pyarrow dependencies") from exc
        parquet_path = output_dir / f"dflash_tts_{name}.parquet"
        temporary = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
        pd.DataFrame([_json_safe(dict(row)) for row in rows], columns=columns).to_parquet(
            temporary, index=False
        )
        temporary.replace(parquet_path)
        paths.append(parquet_path)
    return paths


def write_ablation_tables(
    output_dir: Path,
    tables: dict[str, list[dict[str, Any]]],
    *,
    bucket_size: int,
    parquet: bool = False,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in ("long", "lr_ablation", "memory_al_pareto", "lora_rank_ablation"):
        written.extend(
            _write_table(
                output,
                name,
                tables.get(name, []),
                bucket_size=bucket_size,
                parquet=parquet,
            )
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "classification": "descriptive_paired_ablation_inputs_not_independent_context_replicates",
        "bucket_semantics": "observed_prefix_length_before",
        "bucket_gain_semantics": (
            "descriptive_round_composition_not_prefix_matched"
        ),
        "pareto_semantics": (
            "run_level_exact_output_pair_only; unavailable without fully "
            "verified artifact identity and matching complete runtime fingerprint"
        ),
        "peak_hbm_scope": "run_peak_repeated_across_prefix_buckets",
        "pareto_memory_axis": "whole_process_peak_hbm_bytes",
        "target_call_semantics": {
            "headline": "logical_block_target_calls_per_output_token",
            "legacy_alias": "target_calls_per_output_token",
            "reference_audit": [
                "physical_target_calls_per_output_token",
                "canonical_audit_target_calls_per_output_token",
            ],
            "interpretation": (
                "physical and canonical calls measure exact reference-audit "
                "overhead and are not deployment throughput metrics"
            ),
        },
        "bucket_size": bucket_size,
        "tables": {
            name: {"rows": len(rows)} for name, rows in sorted(tables.items())
        },
        "files": {
            path.name: _sha256_file(path) for path in sorted(written)
        },
        "input_artifacts": sorted(
            {
                (row["run_root"], row["summary_sha256"], row["rounds_sha256"])
                for row in tables.get("long", [])
            }
        ),
    }
    manifest_path = output / "dflash_tts_ablation_manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"output_dir": str(output), "manifest": str(manifest_path), **manifest}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket-size", type=int, default=4096)
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="also write Parquet using the project's pandas/pyarrow dependencies",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_long_table(args.run_roots, bucket_size=args.bucket_size)
    tables = build_ablation_tables(rows)
    manifest = write_ablation_tables(
        args.output_dir,
        tables,
        bucket_size=args.bucket_size,
        parquet=args.parquet,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
