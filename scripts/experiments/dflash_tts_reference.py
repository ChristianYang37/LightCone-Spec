#!/usr/bin/env python3
"""Offline, auditable DFlash TTS adaptation reference harness.

The harness keeps the six experiment labels deliberately disjoint:
``static``, full-drafter TTS, drafter-wide LoRA, and the three cache-safe
LightCone tail layouts.  Full-drafter and drafter-LoRA retain historical draft
KV only when ``--draft-cache-policy stale`` is selected; the artifact labels
that approximation explicitly.  Tail updates never mutate the draft backbone
and therefore keep historical draft KV mathematically valid.

The generation loop follows ``z-lab/dflash`` commit ``94e4abc``:

* a verifier-produced token seeds ``[seed, MASK, ..., MASK]``;
* the draft model consumes selected target hidden states and target embeddings;
* draft hidden states are projected by the frozen target LM head;
* the target verifies the whole block;
* the consecutive greedy/sample match prefix is committed, followed by the
  target bonus token that seeds the next round.

The upstream ``dflash_generate`` function is inference-only and exposes no
round hook.  Consequently, the update loop here is an explicitly labelled
reconstruction.  Learning rate, proximal weight, position weighting, update
stride, and draft-cache policy are required CLI choices and are written to the
evidence artifact.  No network access is permitted: models, dataset, and the
official DFlash checkout must all be local paths.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA_VERSION = 3
HIDDEN_PROJECTION_DIM = 128
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
COMMAND_SHA256_SCHEME = "canonical_json_harness_argv_without_digest_v1"
ADAPTATION_MODES = (
    "static",
    "full-drafter",
    "drafter-lora",
    "full-rank-tail",
    "tail-lora",
    "output-residual",
)
TAIL_MODES = frozenset({"full-rank-tail", "tail-lora", "output-residual"})
DRAFTER_MUTATING_MODES = frozenset({"full-drafter", "drafter-lora"})
TRAINABLE_SCOPE_NAMES = {
    "static": "none_static",
    "full-drafter": "full_drafter_all_parameters",
    "drafter-lora": "drafter_wide_lora_fc_attention_mlp",
    "full-rank-tail": "full_rank_tail",
    "tail-lora": "tail_lora",
    "output-residual": "output_residual",
}
OFFICIAL_REFERENCE_REVISION = "94e4abc"
OFFICIAL_REFERENCE_SOURCE_SHA256 = (
    "6dfb12d0eeebe085d78d23fc816a481a8b25257fd76891e939a41423c29c697b"
)
EXPECTED_GENERATE_PARAMETERS = (
    "model",
    "target",
    "input_ids",
    "max_new_tokens",
    "stop_token_ids",
    "temperature",
    "block_size",
    "mask_token_id",
    "return_stats",
)
ROUND_PROVENANCE_FIELDS = frozenset(
    {
        "reference_revision",
        "reference_source_sha256",
        "target_declared_revision",
        "draft_declared_revision",
        "dataset_declared_revision",
        "dataset_sha256",
        "harness_source_sha256",
    }
)

TOKENIZER_ARTIFACT_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "added_tokens.json",
    "tokenizer.model",
    "spiece.model",
)


@dataclass(frozen=True)
class UpdateEvidence:
    applied: bool
    optimizer_step: int | None
    loss: float | None
    distillation_kl: float | None
    proximal_kl: float | None
    grad_norm: float | None
    parameters_with_grad: int
    parameters_without_grad: tuple[str, ...]
    backward_cuda_us: float | None = None
    optimizer_cuda_us: float | None = None
    update_cuda_us: float | None = None
    parameter_delta_l2: float | None = None
    parameter_displacement_l2: float | None = None
    parameter_l2: float | None = None
    relative_parameter_delta: float | None = None
    parameter_audit_interval_steps: int | None = None


def _normal_fan_in(
    shape: tuple[int, ...], *, fan_in: int, seed_generator: torch.Generator
) -> torch.Tensor:
    return torch.randn(shape, generator=seed_generator, dtype=torch.float32).mul_(
        fan_in**-0.5
    )


class DrafterLoRALinear(torch.nn.Module):
    """Zero-effect LoRA wrapper for one frozen drafter ``nn.Linear``.

    The orientation follows ``linear(linear(x, A), B)``.  A uses locked
    fan-in random initialization, B starts at zero, and ``alpha / rank = 1``.
    Forward parameters stay in model dtype while the optimizer owns separate
    FP32 masters and moments.
    """

    def __init__(
        self,
        base: torch.nn.Linear,
        *,
        rank: int,
        generator: torch.Generator,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = int(rank)
        self.scaling = float(self.alpha / self.rank)
        a = _normal_fan_in(
            (self.rank, base.in_features),
            fan_in=base.in_features,
            seed_generator=generator,
        ).to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_a = torch.nn.Parameter(a)
        self.lora_b = torch.nn.Parameter(
            torch.zeros(
                base.out_features,
                self.rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.base.bias

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        delta = F.linear(F.linear(value, self.lora_a), self.lora_b)
        return base + delta.mul(self.scaling)


def _is_drafter_lora_linear(name: str, module: torch.nn.Module) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    components = name.lower().split(".")
    return (
        "fc" in components
        or "mlp" in components
        or "self_attn" in components
        or "attention" in components
        or "attn" in components
    )


def _replace_submodule(
    root: torch.nn.Module, qualified_name: str, replacement: torch.nn.Module
) -> None:
    parent_name, _, child_name = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if isinstance(parent, (torch.nn.ModuleDict, torch.nn.ModuleList)):
        parent[child_name] = replacement
    else:
        setattr(parent, child_name, replacement)


def inject_drafter_lora(
    draft_model: torch.nn.Module, *, rank: int, adapter_seed: int
) -> tuple[torch.nn.Parameter, ...]:
    """Inject LoRA into ``fc`` and every attention/MLP linear, fail closed."""

    if any(
        isinstance(module, DrafterLoRALinear)
        for module in draft_model.modules()
    ):
        raise RuntimeError("drafter-lora is already installed")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(adapter_seed))
    selected = [
        (name, module)
        for name, module in draft_model.named_modules()
        if _is_drafter_lora_linear(name, module)
    ]
    if not selected:
        raise RuntimeError(
            "drafter-lora found no fc or attention/MLP nn.Linear modules"
        )
    adapters: list[DrafterLoRALinear] = []
    for name, module in selected:
        assert isinstance(module, torch.nn.Linear)
        adapter = DrafterLoRALinear(module, rank=rank, generator=generator)
        _replace_submodule(draft_model, name, adapter)
        adapters.append(adapter)
    return tuple(
        parameter
        for adapter in adapters
        for parameter in (adapter.lora_a, adapter.lora_b)
    )


def _build_hidden_projection(hidden_size: int, seed: int) -> torch.Tensor:
    if hidden_size < HIDDEN_PROJECTION_DIM:
        raise ValueError(
            f"hidden size {hidden_size} is below the locked projection width "
            f"{HIDDEN_PROJECTION_DIM}"
        )
    # Import lazily so static/full-drafter remains a self-contained reference
    # even when only the DFlash checkout is on PYTHONPATH.  Output-residual is
    # required to use the exact LightCone artifact builder, not a lookalike RNG.
    try:
        from lightcone_spec.adapters.projections import build_hidden_projection
    except ImportError as exc:
        raise RuntimeError(
            "output-residual requires this repository's src directory on PYTHONPATH so the "
            "canonical projection artifact builder is available"
        ) from exc
    return torch.from_numpy(
        build_hidden_projection(
            hidden_size, HIDDEN_PROJECTION_DIM, seed=int(seed)
        )
    )


def _build_output_basis(
    head_weight: torch.Tensor, hidden_projection: torch.Tensor, rank: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    if rank <= 0:
        raise ValueError("adapter rank must be positive")
    if rank > HIDDEN_PROJECTION_DIM:
        raise ValueError(
            f"adapter rank {rank} exceeds output basis width "
            f"{HIDDEN_PROJECTION_DIM}"
        )
    try:
        from lightcone_spec.adapters.projections import build_output_basis
    except ImportError as exc:
        raise RuntimeError(
            "output-residual requires the canonical LightCone output-basis builder"
        ) from exc
    weight_numpy = (
        head_weight.detach().to(device="cpu", dtype=torch.float32).numpy()
    )
    projection_numpy = (
        hidden_projection.detach().to(device="cpu", dtype=torch.float32).numpy()
    )
    # DFlash has no Markov head; an empty Vx0 matrix gives the exact runtime SVD
    # recipe over colnorm(W_lm R_h) without inventing a DSpark-only component.
    artifact = build_output_basis(
        weight_numpy,
        np.empty((weight_numpy.shape[0], 0), dtype=np.float32),
        projection_numpy,
        rank,
    )
    hidden_sha256 = hashlib.sha256(
        np.ascontiguousarray(projection_numpy, dtype=np.float32).tobytes()
    ).hexdigest()
    identity = {
        "builder": "lightcone_spec.adapters.projections",
        "target_head_shape": list(head_weight.shape),
        "target_head_dtype": str(head_weight.dtype),
        "hidden_projection_recipe": "pcg64_float64_qr_signfix_v1",
        "hidden_projection_seed": None,
        "hidden_projection_sha256": hidden_sha256,
        "output_basis_recipe": "dflash_colnorm_wlm_rh_float64_svd_signfix_v1",
        "output_basis_sha256": artifact.sha256(),
        "output_basis_input_sha256": artifact.input_sha256,
        "output_basis_rank": artifact.rank,
        "dropped_columns": artifact.dropped_columns,
    }
    return torch.from_numpy(artifact.basis), identity


def _projection_artifact_paths(path: str | Path) -> tuple[Path, Path]:
    requested = Path(path).expanduser().resolve()
    payload = (
        requested
        if str(requested).endswith(".npz")
        else Path(str(requested) + ".npz")
    )
    return payload, Path(str(payload) + ".meta.json")


def _canonical_json_value(value: Any, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc


def _projection_expected_metadata(
    *,
    head_weight: torch.Tensor,
    rank: int,
    adapter_seed: int,
    projection_binding: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_schema_version": 1,
        "kind": "dflash_output_residual_projections",
        "algorithm": "DFLASH",
        "rank": int(rank),
        "seed": int(adapter_seed),
        "target_head": {
            "shape": list(head_weight.shape),
            "dtype": str(head_weight.dtype),
        },
        "binding": _canonical_json_value(
            projection_binding, "projection artifact binding"
        ),
        "hidden_projection_recipe": "pcg64_float64_qr_signfix_v1",
        "output_basis_recipe": "dflash_colnorm_wlm_rh_float64_svd_signfix_v1",
    }


def _validate_projection_arrays(
    arrays: dict[str, np.ndarray],
    *,
    head_weight: torch.Tensor,
    rank: int,
) -> None:
    if set(arrays) != {"hidden_projection", "output_basis"}:
        raise ValueError(
            "projection artifact arrays must be exactly hidden_projection and "
            "output_basis"
        )
    expected_shapes = {
        "hidden_projection": (
            int(head_weight.shape[1]),
            HIDDEN_PROJECTION_DIM,
        ),
        "output_basis": (int(head_weight.shape[0]), int(rank)),
    }
    for name, expected_shape in expected_shapes.items():
        value = arrays[name]
        if value.dtype != np.float32 or value.shape != expected_shape:
            raise ValueError(
                f"projection artifact {name} has dtype/shape "
                f"{value.dtype}/{value.shape}, expected float32/{expected_shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"projection artifact {name} contains non-finite values")


def _load_bound_projection_artifact(
    path: str | Path,
    *,
    head_weight: torch.Tensor,
    rank: int,
    expected_metadata: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    try:
        from lightcone_spec.adapters.projections import load_projection_artifact
    except ImportError as exc:
        raise RuntimeError(
            "projection artifact loading requires this repository's src directory on PYTHONPATH"
        ) from exc
    arrays, metadata = load_projection_artifact(path)
    for key in (
        "rank",
        "seed",
        "target_head",
        "binding",
        "artifact_schema_version",
        "kind",
        "algorithm",
        "hidden_projection_recipe",
        "output_basis_recipe",
    ):
        if metadata.get(key) != expected_metadata[key]:
            raise ValueError(
                f"projection artifact {key} mismatch: expected "
                f"{expected_metadata[key]!r}, got {metadata.get(key)!r}"
            )
    _validate_projection_arrays(arrays, head_weight=head_weight, rank=rank)
    arrays_sha256 = metadata.get("arrays_sha256")
    if not isinstance(arrays_sha256, dict):
        raise ValueError("projection artifact is missing array hashes")
    required_dynamic = ("output_basis_input_sha256", "dropped_columns")
    if any(key not in metadata for key in required_dynamic):
        raise ValueError("projection artifact is missing output-basis provenance")
    identity = {
        "builder": "lightcone_spec.adapters.projections",
        "storage": "artifact",
        "artifact_file_sha256": metadata["file_sha256"],
        "target_head_shape": list(head_weight.shape),
        "target_head_dtype": str(head_weight.dtype),
        "hidden_projection_recipe": metadata["hidden_projection_recipe"],
        "hidden_projection_seed": metadata["seed"],
        "hidden_projection_sha256": arrays_sha256["hidden_projection"],
        "output_basis_recipe": metadata["output_basis_recipe"],
        "output_basis_sha256": arrays_sha256["output_basis"],
        "output_basis_input_sha256": metadata["output_basis_input_sha256"],
        "output_basis_rank": metadata["rank"],
        "dropped_columns": metadata["dropped_columns"],
        "binding": metadata["binding"],
    }
    return (
        torch.from_numpy(arrays["hidden_projection"]),
        torch.from_numpy(arrays["output_basis"]),
        identity,
    )


def resolve_output_residual_projections(
    *,
    head_weight: torch.Tensor,
    rank: int,
    adapter_seed: int,
    projection_artifact: str | Path | None,
    projection_binding: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build once or load a strictly bound canonical projection artifact."""

    binding = {} if projection_binding is None else projection_binding
    if projection_artifact is not None:
        target_head_artifact = binding.get("target_head_artifact")
        target_weight_files = (
            target_head_artifact.get("weight_files")
            if isinstance(target_head_artifact, dict)
            else None
        )
        if not isinstance(target_weight_files, list) or not target_weight_files:
            raise ValueError(
                "output-residual projection artifact must bind target LM-head "
                "weight shards"
            )
        if any(
            not isinstance(item, dict)
            or not _is_sha256(item.get("sha256"))
            for item in target_weight_files
        ):
            raise ValueError(
                "output-residual projection artifact lacks target shard hashes"
            )
    expected_metadata = _projection_expected_metadata(
        head_weight=head_weight,
        rank=rank,
        adapter_seed=adapter_seed,
        projection_binding=binding,
    )
    if projection_artifact is not None:
        if projection_binding is None:
            raise ValueError(
                "projection_artifact requires target/draft/head projection_binding"
            )
        payload, meta_path = _projection_artifact_paths(projection_artifact)
        if payload.exists() != meta_path.exists():
            raise ValueError(
                "projection artifact is incomplete; payload and metadata sidecar "
                "must either both exist or both be absent"
            )
        if payload.exists():
            return _load_bound_projection_artifact(
                projection_artifact,
                head_weight=head_weight,
                rank=rank,
                expected_metadata=expected_metadata,
            )

    hidden_projection = _build_hidden_projection(
        int(head_weight.shape[1]), adapter_seed
    )
    output_basis, identity = _build_output_basis(
        head_weight, hidden_projection, rank
    )
    identity["hidden_projection_seed"] = int(adapter_seed)
    identity["binding"] = expected_metadata["binding"]
    if projection_artifact is None:
        identity["storage"] = "ephemeral"
        return hidden_projection, output_basis, identity

    try:
        from lightcone_spec.adapters.projections import save_projection_artifact
    except ImportError as exc:
        raise RuntimeError(
            "projection artifact saving requires this repository's src directory on PYTHONPATH"
        ) from exc
    save_projection_artifact(
        projection_artifact,
        {
            "hidden_projection": hidden_projection.numpy(),
            "output_basis": output_basis.numpy(),
        },
        {
            **expected_metadata,
            "output_basis_input_sha256": identity[
                "output_basis_input_sha256"
            ],
            "dropped_columns": identity["dropped_columns"],
        },
    )
    # Always re-open through the validating path: a completed run never trusts
    # a partially persisted or hash-drifted artifact.
    return _load_bound_projection_artifact(
        projection_artifact,
        head_weight=head_weight,
        rank=rank,
        expected_metadata=expected_metadata,
    )


class TailAdapter(torch.nn.Module):
    """DFlash cache-safe tail parameterization matching ``AdapterShapes``.

    DFlash has neither DSpark Markov terms nor a confidence head.  Thus the
    three layouts reduce to ``B A_h R_h^T h``, ``W (h A_h) B_h``, and
    ``W h D_h`` respectively.
    """

    def __init__(
        self,
        *,
        mode: str,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        forward_dtype: torch.dtype,
        device: torch.device,
        adapter_seed: int,
        hidden_projection: torch.Tensor | None = None,
        output_basis: torch.Tensor | None = None,
        projection_identity: dict[str, Any] | None = None,
    ):
        super().__init__()
        if mode not in TAIL_MODES:
            raise ValueError(f"unknown tail mode {mode!r}")
        if rank <= 0:
            raise ValueError("adapter rank must be positive")
        self.mode = mode
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        self.adapter_seed = int(adapter_seed)
        self.projection_identity = projection_identity
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.adapter_seed)
        if mode == "output-residual":
            if hidden_projection is None or output_basis is None:
                raise ValueError(
                    "output-residual requires frozen projection and output basis"
                )
            self.register_buffer(
                "hidden_projection",
                hidden_projection.to(device=device, dtype=forward_dtype),
            )
            self.register_buffer(
                "output_basis", output_basis.to(device=device, dtype=forward_dtype)
            )
            self.a_h = torch.nn.Parameter(
                torch.zeros(
                    self.rank,
                    HIDDEN_PROJECTION_DIM,
                    device=device,
                    dtype=forward_dtype,
                )
            )
        elif mode == "tail-lora":
            self.a_h = torch.nn.Parameter(
                _normal_fan_in(
                    (self.hidden_size, self.rank),
                    fan_in=self.hidden_size,
                    seed_generator=generator,
                ).to(device=device, dtype=forward_dtype)
            )
            self.b_h = torch.nn.Parameter(
                torch.zeros(
                    self.rank,
                    self.hidden_size,
                    device=device,
                    dtype=forward_dtype,
                )
            )
        else:
            self.d_h = torch.nn.Parameter(
                torch.zeros(
                    self.hidden_size,
                    self.hidden_size,
                    device=device,
                    dtype=forward_dtype,
                )
            )

    @classmethod
    def from_target_head(
        cls,
        *,
        mode: str,
        target_head: torch.nn.Module,
        rank: int,
        adapter_seed: int,
        projection_artifact: str | Path | None = None,
        projection_binding: dict[str, Any] | None = None,
    ) -> "TailAdapter":
        weight = getattr(target_head, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise RuntimeError("tail adaptation requires a 2D frozen target head")
        vocab_size, hidden_size = map(int, weight.shape)
        hidden_projection = None
        output_basis = None
        projection_identity = None
        if mode == "output-residual":
            (
                hidden_projection,
                output_basis,
                projection_identity,
            ) = resolve_output_residual_projections(
                head_weight=weight,
                rank=rank,
                adapter_seed=adapter_seed,
                projection_artifact=projection_artifact,
                projection_binding=projection_binding,
            )
        elif projection_artifact is not None:
            raise ValueError(
                "projection_artifact is only valid for output-residual"
            )
        return cls(
            mode=mode,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            rank=rank,
            forward_dtype=weight.dtype,
            device=weight.device,
            adapter_seed=adapter_seed,
            hidden_projection=hidden_projection,
            output_basis=output_basis,
            projection_identity=projection_identity,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        base_logits: torch.Tensor,
        target_head: torch.nn.Module,
    ) -> torch.Tensor:
        if self.mode == "output-residual":
            projected = hidden @ self.hidden_projection
            delta_logits = (projected @ self.a_h.T) @ self.output_basis.T
        else:
            if self.mode == "tail-lora":
                delta_hidden = (hidden @ self.a_h) @ self.b_h
            else:
                delta_hidden = hidden @ self.d_h
            weight = getattr(target_head, "weight", None)
            if not isinstance(weight, torch.Tensor):
                raise RuntimeError("target head lost its projection weight")
            delta_logits = F.linear(delta_hidden, weight, bias=None)
        return base_logits + delta_logits.to(base_logits.dtype)

    def layout(self) -> dict[str, Any]:
        parameters = [
            {
                "name": name,
                "shape": list(parameter.shape),
                "forward_dtype": str(parameter.dtype),
                "master_dtype": "torch.float32",
                "numel": int(parameter.numel()),
            }
            for name, parameter in self.named_parameters()
        ]
        identity = {
            "schema_version": 2,
            "algorithm": "DFLASH",
            "mode": self.mode,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "rank": None if self.mode == "full-rank-tail" else self.rank,
            "has_markov": False,
            "has_confidence": False,
            "initialization": (
                {
                    "scheme": "normal_fan_in_input_zero_output",
                    "seed": self.adapter_seed,
                    "alpha_over_rank": 1.0,
                }
                if self.mode == "tail-lora"
                else {"scheme": "zero"}
            ),
            "parameters": parameters,
            "parameter_tensors": len(parameters),
            "parameter_count": sum(
                int(parameter["numel"]) for parameter in parameters
            ),
            "projection_identity": self.projection_identity,
        }
        identity["layout_sha256"] = _json_sha256(identity)
        return identity


class FP32MasterOptimizer:
    """Persistent FP32 masters/gradients/moments for model-dtype forwards."""

    def __init__(
        self,
        named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
        *,
        optimizer_name: str,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        parameter_audit_enabled: bool = False,
    ):
        self.named_forward_parameters = tuple(named_parameters)
        if not self.named_forward_parameters:
            raise ValueError("adaptation optimizer requires trainable parameters")
        ids = [id(parameter) for _name, parameter in self.named_forward_parameters]
        if len(ids) != len(set(ids)):
            raise ValueError("adaptation optimizer received duplicate parameters")
        self.optimizer_name = optimizer_name.lower()
        if self.optimizer_name not in {"adam", "adamw"}:
            raise ValueError("optimizer must be adam or adamw")
        self.master_parameters = tuple(
            torch.nn.Parameter(
                parameter.detach().to(dtype=torch.float32).clone(),
                requires_grad=True,
            )
            for _name, parameter in self.named_forward_parameters
        )
        optimizer_class = (
            torch.optim.Adam
            if self.optimizer_name == "adam"
            else torch.optim.AdamW
        )
        self.inner_optimizer = optimizer_class(
            self.master_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.weight_decay = float(weight_decay)
        self._parameter_audit_initial: tuple[torch.Tensor, ...] | None = None
        self._parameter_audit_previous: tuple[torch.Tensor, ...] | None = None
        self._parameter_audit_previous_step = 0
        if parameter_audit_enabled:
            initial = tuple(
                parameter.detach().to(device="cpu", dtype=torch.float32).clone()
                for parameter in self.master_parameters
            )
            self._parameter_audit_initial = initial
            self._parameter_audit_previous = tuple(
                parameter.clone() for parameter in initial
            )

    @property
    def owned_forward_parameter_ids(self) -> frozenset[int]:
        return frozenset(
            id(parameter) for _name, parameter in self.named_forward_parameters
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        for _name, parameter in self.named_forward_parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)
        self.inner_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self) -> None:
        for (_name, forward), master in zip(
            self.named_forward_parameters, self.master_parameters, strict=True
        ):
            master.grad = (
                None
                if forward.grad is None
                else forward.grad.detach().to(dtype=torch.float32)
            )
        self.inner_optimizer.step()
        for (_name, forward), master in zip(
            self.named_forward_parameters, self.master_parameters, strict=True
        ):
            forward.copy_(master.to(dtype=forward.dtype))

    @torch.no_grad()
    def audit_parameters(self, *, optimizer_step: int) -> dict[str, float | int]:
        """Return opt-in master-parameter drift without consuming extra HBM.

        Snapshots intentionally live on CPU and are created only when
        ``parameter_audit_enabled`` is requested.  ``parameter_delta_l2`` spans
        the interval since the previous audited step, which can exceed one
        optimizer step when the caller samples sparsely.
        """

        initial = self._parameter_audit_initial
        previous = self._parameter_audit_previous
        if initial is None or previous is None:
            raise RuntimeError("parameter audit was not enabled for this optimizer")
        if optimizer_step <= self._parameter_audit_previous_step:
            raise RuntimeError("parameter audit steps must be strictly increasing")

        delta_sq = 0.0
        displacement_sq = 0.0
        current_sq = 0.0
        previous_sq = 0.0
        for master, initial_value, previous_value in zip(
            self.master_parameters, initial, previous, strict=True
        ):
            current = master.detach().to(device="cpu", dtype=torch.float32)
            delta_sq += float(
                torch.sum((current - previous_value).double().square()).item()
            )
            displacement_sq += float(
                torch.sum((current - initial_value).double().square()).item()
            )
            current_sq += float(torch.sum(current.double().square()).item())
            previous_sq += float(
                torch.sum(previous_value.double().square()).item()
            )
            previous_value.copy_(current)

        delta_l2 = math.sqrt(delta_sq)
        previous_l2 = math.sqrt(previous_sq)
        interval = optimizer_step - self._parameter_audit_previous_step
        self._parameter_audit_previous_step = optimizer_step
        return {
            "parameter_delta_l2": delta_l2,
            "parameter_displacement_l2": math.sqrt(displacement_sq),
            "parameter_l2": math.sqrt(current_sq),
            "relative_parameter_delta": delta_l2 / max(previous_l2, 1e-30),
            "parameter_audit_interval_steps": interval,
        }

    def memory_accounting(self) -> dict[str, int]:
        forward_bytes = sum(
            parameter.numel() * parameter.element_size()
            for _name, parameter in self.named_forward_parameters
        )
        master_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.master_parameters
        )
        moment_bytes = 0
        for state in self.inner_optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if isinstance(value, torch.Tensor):
                    moment_bytes += value.numel() * value.element_size()
        forward_gradient_bytes = forward_bytes
        master_gradient_bytes = master_bytes
        persistent_bytes = forward_bytes + master_bytes + moment_bytes
        estimated_update_peak_bytes = (
            persistent_bytes + forward_gradient_bytes + master_gradient_bytes
        )
        audit_cpu_bytes = 0
        for snapshot in (
            self._parameter_audit_initial,
            self._parameter_audit_previous,
        ):
            if snapshot is not None:
                audit_cpu_bytes += sum(
                    value.numel() * value.element_size() for value in snapshot
                )
        return {
            "forward_parameter_bytes": int(forward_bytes),
            "master_parameter_bytes": int(master_bytes),
            "master_gradient_bytes": int(master_gradient_bytes),
            "optimizer_moment_bytes": int(moment_bytes),
            "forward_gradient_bytes": int(forward_gradient_bytes),
            "persistent_bytes": int(persistent_bytes),
            "estimated_update_peak_bytes": int(estimated_update_peak_bytes),
            "total_bytes": int(estimated_update_peak_bytes),
            "parameter_audit_cpu_snapshot_bytes": int(audit_cpu_bytes),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_sha256(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _argv_sha256(value: Sequence[str]) -> str:
    body = json.dumps(
        list(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _command_attestation(
    raw_argv: Sequence[str],
    *,
    run_identity_sha256: str | None,
    command_sha256: str | None,
) -> dict[str, Any]:
    """Verify the runner's ordered harness argv, or label a direct run."""

    if run_identity_sha256 is None and command_sha256 is None:
        return {
            "status": "direct_unbound",
            "scheme": None,
            "run_identity_sha256": None,
            "command_sha256": None,
        }
    if run_identity_sha256 is None or command_sha256 is None:
        raise ValueError(
            "--run-identity-sha256 and --command-sha256 must be provided together"
        )
    positions = [
        index
        for index, value in enumerate(raw_argv)
        if value == "--command-sha256"
    ]
    if len(positions) != 1:
        raise ValueError("--command-sha256 must appear exactly once")
    position = positions[0]
    if position + 1 >= len(raw_argv):
        raise ValueError("--command-sha256 requires a value")
    if raw_argv[position + 1] != command_sha256:
        raise ValueError("parsed command sha256 differs from raw argv")
    unsigned_argv = [
        str(Path(__file__).resolve()),
        *raw_argv[:position],
        *raw_argv[position + 2 :],
    ]
    observed = _argv_sha256(unsigned_argv)
    if observed != command_sha256:
        raise ValueError(
            "command sha256 mismatch: "
            f"expected {command_sha256}, observed {observed}"
        )
    return {
        "status": "runner_bound",
        "scheme": COMMAND_SHA256_SCHEME,
        "run_identity_sha256": run_identity_sha256,
        "command_sha256": command_sha256,
    }


def _require_local_dir(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{label} must be an existing local directory: {path}")
    return path


def _require_local_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} must be an existing local file: {path}")
    return path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def load_reference_module(root: Path, module_name: str) -> ModuleType:
    """Import the pinned local DFlash source and verify its public contract."""

    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module(module_name)
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass

    source = inspect.getsourcefile(module)
    if source is None:
        raise RuntimeError(
            f"cannot resolve source for reference module {module_name!r}"
        )
    source_path = Path(source).resolve()
    if not _is_relative_to(source_path, root):
        raise RuntimeError(
            f"reference module resolved outside --reference-root: {source_path}"
        )
    validate_reference_api(module)
    return module


def validate_reference_api(module: ModuleType) -> None:
    draft_cls = getattr(module, "DFlashDraftModel", None)
    generate = getattr(module, "dflash_generate", None)
    extract = getattr(module, "extract_context_feature", None)
    if not inspect.isclass(draft_cls):
        raise RuntimeError("reference module does not expose DFlashDraftModel")
    if not callable(generate):
        raise RuntimeError("reference module does not expose dflash_generate")
    if not callable(extract):
        raise RuntimeError("reference module does not expose extract_context_feature")
    parameters = tuple(inspect.signature(generate).parameters)
    if parameters != EXPECTED_GENERATE_PARAMETERS:
        raise RuntimeError(
            "dflash_generate signature drift: "
            f"expected {EXPECTED_GENERATE_PARAMETERS}, got {parameters}"
        )


def sample_tokens(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """Match the official reference sampler, including its greedy draft use."""

    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    batch, seq_len, vocab = logits.shape
    probabilities = torch.softmax(logits.reshape(-1, vocab) / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).view(batch, seq_len)


def consecutive_acceptance_length(
    block_token_ids: torch.Tensor, posterior_token_ids: torch.Tensor
) -> int:
    if block_token_ids.ndim != 2 or block_token_ids.shape[0] != 1:
        raise ValueError("the official DFlash reference supports batch size one")
    if posterior_token_ids.shape != block_token_ids.shape:
        raise ValueError("posterior token shape must match the DFlash block")
    matches = block_token_ids[:, 1:] == posterior_token_ids[:, :-1]
    return int(matches.to(torch.int64).cumprod(dim=1).sum(dim=1)[0].item())


def build_position_weights(
    length: int,
    scheme: str,
    decay_gamma: float | None,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return unnormalised paper-style position weights.

    ``linear`` decreases from one to zero.  ``exponential`` follows the public
    SpecForge/DFlash training recipe, ``weight[k] = exp(-k / gamma)``.  Loss
    reduction remains a separate explicit choice.
    """

    if length <= 0:
        raise ValueError("position-weight length must be positive")
    if scheme == "uniform":
        if decay_gamma is not None:
            raise ValueError(
                "--position-decay-gamma is not used with uniform weights"
            )
        return torch.ones(length, dtype=torch.float32, device=device)
    if scheme == "linear":
        if decay_gamma is not None:
            raise ValueError(
                "--position-decay-gamma is not used with linear weights"
            )
        return torch.linspace(1.0, 0.0, length, dtype=torch.float32, device=device)
    if scheme == "exponential":
        if decay_gamma is None or decay_gamma <= 0.0:
            raise ValueError(
                "positive --position-decay-gamma is required for exponential weights"
            )
        positions = torch.arange(length, dtype=torch.float32, device=device)
        return torch.exp(-positions / float(decay_gamma))
    raise ValueError(f"unknown position weighting {scheme!r}")


def tts_kl_objective(
    proposal_logits: torch.Tensor,
    target_logits: torch.Tensor,
    source_proposal_logits: torch.Tensor,
    position_weights: torch.Tensor,
    proximal_lambda: float,
    reduction: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-KL distillation plus source-proposal proximal KL."""

    if proposal_logits.shape != target_logits.shape:
        raise ValueError("proposal and target logits must have identical shapes")
    if source_proposal_logits.shape != proposal_logits.shape:
        raise ValueError("source proposal logits must match proposal logits")
    if proposal_logits.ndim != 3:
        raise ValueError("TTS logits must be [batch, position, vocabulary]")
    if tuple(position_weights.shape) != (proposal_logits.shape[1],):
        raise ValueError("one position weight is required per draft proposal")
    if proximal_lambda < 0.0:
        raise ValueError("proximal lambda must be non-negative")

    q_log = F.log_softmax(proposal_logits.float(), dim=-1)
    p_log = F.log_softmax(target_logits.detach().float(), dim=-1)
    p = p_log.exp()
    source_log = F.log_softmax(source_proposal_logits.detach().float(), dim=-1)
    source = source_log.exp()
    distillation_per_position = (p * (p_log - q_log)).sum(dim=-1)
    proximal_per_position = (source * (source_log - q_log)).sum(dim=-1)
    weights = position_weights.to(q_log.device, dtype=q_log.dtype).unsqueeze(0)
    distillation = (weights * distillation_per_position).sum()
    proximal = (weights * proximal_per_position).sum()
    if reduction == "weighted-mean":
        denominator = weights.sum() * proposal_logits.shape[0]
        distillation = distillation / denominator
        proximal = proximal / denominator
    elif reduction != "sum":
        raise ValueError("loss reduction must be weighted-mean or sum")
    return distillation + proximal_lambda * proximal, distillation, proximal


def _parameter_layout(model: torch.nn.Module) -> dict[str, Any]:
    entries = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": int(parameter.numel()),
        }
        for name, parameter in model.named_parameters()
    ]
    return {
        "parameter_tensors": len(entries),
        "parameter_count": sum(entry["numel"] for entry in entries),
        "layout_sha256": _json_sha256(entries),
    }


def _trainable_layout(
    mode: str,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    rank: int,
    adapter_seed: int,
    tail_adapter: TailAdapter | None,
) -> dict[str, Any]:
    if tail_adapter is not None:
        return tail_adapter.layout()
    entries = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "forward_dtype": str(parameter.dtype),
            "master_dtype": "torch.float32",
        }
        for name, parameter in named_parameters
    ]
    return {
        "schema_version": 2,
        "algorithm": "DFLASH",
        "mode": mode,
        "rank": rank if mode == "drafter-lora" else None,
        "adapter_seed": adapter_seed if mode == "drafter-lora" else None,
        "has_markov": False,
        "has_confidence": False,
        "parameters": entries,
        "parameter_tensors": len(entries),
        "parameter_count": sum(entry["numel"] for entry in entries),
        "layout_sha256": _json_sha256(entries),
    }


def configure_trainable_scope(
    draft_model: torch.nn.Module,
    mode: str,
    *,
    rank: int = 16,
    adapter_seed: int = 0,
) -> tuple[torch.nn.Parameter, ...]:
    parameters = tuple(draft_model.parameters())
    if not parameters:
        raise RuntimeError("DFlashDraftModel exposes no parameters")
    if mode not in ADAPTATION_MODES:
        raise ValueError(f"unknown mode {mode!r}")
    for parameter in parameters:
        parameter.requires_grad_(False)
    selected: tuple[torch.nn.Parameter, ...]
    if mode == "full-drafter":
        for parameter in parameters:
            if not (parameter.is_floating_point() or parameter.is_complex()):
                raise RuntimeError(
                    "full-drafter optimizer requires floating parameters; "
                    f"found {parameter.dtype}"
                )
            parameter.requires_grad_(True)
        selected = parameters
    elif mode == "drafter-lora":
        selected = inject_drafter_lora(
            draft_model, rank=rank, adapter_seed=adapter_seed
        )
    else:
        # Static and all tail layouts keep the drafter backbone frozen.  Tail
        # parameters are constructed separately because the official model has
        # no proposal-head module of its own: it shares the target LM head.
        selected = ()
    # Online adaptation needs autograd, not training-mode stochasticity.  Keep
    # dropout and other train/eval-dependent proposal behavior disabled so a
    # checkpoint with non-zero dropout remains comparable to static decoding.
    draft_model.eval()
    return selected


def _trainable_named_parameters(
    draft_model: torch.nn.Module,
    tail_adapter: TailAdapter | None = None,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    named = tuple(
        (f"draft.{name}", parameter)
        for name, parameter in draft_model.named_parameters()
        if parameter.requires_grad
    )
    if tail_adapter is not None:
        named += tuple(
            (f"tail.{name}", parameter)
            for name, parameter in tail_adapter.named_parameters()
            if parameter.requires_grad
        )
    return named


def _trainable_parameters(
    draft_model: torch.nn.Module,
    tail_adapter: TailAdapter | None = None,
) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter
        for _name, parameter in _trainable_named_parameters(
            draft_model, tail_adapter
        )
    )


def freeze_target(target: torch.nn.Module) -> None:
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)


def _global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    gradients = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    devices = {gradient.device for gradient in gradients}
    if len(devices) != 1:
        raise RuntimeError("gradient norm expects all drafter gradients on one device")
    # Accumulate every per-tensor norm and the final norm on device.  The sole
    # DtoH synchronization is the final scalar .item(), once per update round.
    per_tensor_norms = torch.stack(
        [
            torch.linalg.vector_norm(gradient, ord=2, dtype=torch.float32)
            for gradient in gradients
        ]
    )
    return float(torch.linalg.vector_norm(per_tensor_norms, ord=2).item())


def _optimizer_owned_parameter_ids(
    optimizer: torch.optim.Optimizer | FP32MasterOptimizer,
) -> frozenset[int]:
    if isinstance(optimizer, FP32MasterOptimizer):
        return optimizer.owned_forward_parameter_ids
    return frozenset(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def apply_tts_optimizer_step(
    *,
    trainable_named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer | FP32MasterOptimizer,
    proposal_logits: torch.Tensor,
    target_logits: torch.Tensor,
    position_weights: torch.Tensor,
    proximal_lambda: float,
    loss_reduction: str,
    optimizer_step: int,
    scope_name: str = "declared_adaptation_layout",
    audit_cuda_timing: bool = False,
    audit_parameter_stats: bool = False,
) -> UpdateEvidence:
    """Apply one auditable TTS update to exactly the declared parameter set."""

    named_parameters = tuple(trainable_named_parameters)
    if not named_parameters:
        raise RuntimeError("TTS update has no trainable parameters")
    expected_ids = {id(parameter) for _name, parameter in named_parameters}
    optimizer_ids = _optimizer_owned_parameter_ids(optimizer)
    if optimizer_ids != expected_ids:
        raise RuntimeError(
            "optimizer parameter set does not exactly match the declared layout"
        )
    if not proposal_logits.requires_grad:
        raise RuntimeError(
            "proposal logits are detached; refusing to label this a TTS update"
        )

    optimizer.zero_grad(set_to_none=True)
    cuda_events: dict[str, torch.cuda.Event] | None = None
    if audit_cuda_timing and proposal_logits.device.type == "cuda":
        cuda_events = {
            name: torch.cuda.Event(enable_timing=True)
            for name in (
                "update_start",
                "backward_start",
                "backward_end",
                "optimizer_start",
                "optimizer_end",
            )
        }
        cuda_events["update_start"].record()
    source_proposal_logits = proposal_logits.detach().clone()
    loss, distillation, proximal = tts_kl_objective(
        proposal_logits,
        target_logits,
        source_proposal_logits,
        position_weights,
        proximal_lambda,
        loss_reduction,
    )
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite TTS loss")
    if cuda_events is not None:
        cuda_events["backward_start"].record()
    loss.backward()
    if cuda_events is not None:
        cuda_events["backward_end"].record()
    missing = tuple(
        name for name, parameter in named_parameters if parameter.grad is None
    )
    if missing:
        raise RuntimeError(
            f"TTS backward did not reach every parameter in scope {scope_name}: "
            + ", ".join(missing[:16])
        )
    grad_norm = _global_grad_norm(parameter for _name, parameter in named_parameters)
    if not math.isfinite(grad_norm):
        raise RuntimeError("non-finite adaptation gradient norm")
    if cuda_events is not None:
        cuda_events["optimizer_start"].record()
    optimizer.step()
    if cuda_events is not None:
        cuda_events["optimizer_end"].record()
        cuda_events["optimizer_end"].synchronize()
    parameter_stats: dict[str, float | int] = {}
    if audit_parameter_stats:
        if not isinstance(optimizer, FP32MasterOptimizer):
            raise RuntimeError(
                "parameter audit requires the FP32 master optimizer"
            )
        parameter_stats = optimizer.audit_parameters(optimizer_step=optimizer_step)
    loss_value, distillation_value, proximal_value = torch.stack(
        (loss.detach(), distillation.detach(), proximal.detach())
    ).tolist()
    backward_cuda_us = None
    optimizer_cuda_us = None
    update_cuda_us = None
    if cuda_events is not None:
        backward_cuda_us = 1000.0 * cuda_events["backward_start"].elapsed_time(
            cuda_events["backward_end"]
        )
        optimizer_cuda_us = 1000.0 * cuda_events["optimizer_start"].elapsed_time(
            cuda_events["optimizer_end"]
        )
        update_cuda_us = 1000.0 * cuda_events["update_start"].elapsed_time(
            cuda_events["optimizer_end"]
        )
    evidence = UpdateEvidence(
        applied=True,
        optimizer_step=optimizer_step,
        loss=float(loss_value),
        distillation_kl=float(distillation_value),
        proximal_kl=float(proximal_value),
        grad_norm=grad_norm,
        parameters_with_grad=len(named_parameters),
        parameters_without_grad=(),
        backward_cuda_us=backward_cuda_us,
        optimizer_cuda_us=optimizer_cuda_us,
        update_cuda_us=update_cuda_us,
        parameter_delta_l2=parameter_stats.get("parameter_delta_l2"),
        parameter_displacement_l2=parameter_stats.get(
            "parameter_displacement_l2"
        ),
        parameter_l2=parameter_stats.get("parameter_l2"),
        relative_parameter_delta=parameter_stats.get(
            "relative_parameter_delta"
        ),
        parameter_audit_interval_steps=parameter_stats.get(
            "parameter_audit_interval_steps"
        ),
    )
    # Gradients can be model-sized; retain only scalar evidence between rounds.
    optimizer.zero_grad(set_to_none=True)
    return evidence


def apply_full_drafter_adam_step(
    *,
    draft_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | FP32MasterOptimizer,
    proposal_logits: torch.Tensor,
    target_logits: torch.Tensor,
    position_weights: torch.Tensor,
    proximal_lambda: float,
    loss_reduction: str,
    optimizer_step: int,
) -> UpdateEvidence:
    """Backward-compatible full-drafter entry point used by earlier artifacts."""

    return apply_tts_optimizer_step(
        trainable_named_parameters=tuple(draft_model.named_parameters()),
        optimizer=optimizer,
        proposal_logits=proposal_logits,
        target_logits=target_logits,
        position_weights=position_weights,
        proximal_lambda=proximal_lambda,
        loss_reduction=loss_reduction,
        optimizer_step=optimizer_step,
        scope_name=TRAINABLE_SCOPE_NAMES["full-drafter"],
    )


def _no_update_evidence(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> UpdateEvidence:
    return UpdateEvidence(
        applied=False,
        optimizer_step=None,
        loss=None,
        distillation_kl=None,
        proximal_kl=None,
        grad_norm=None,
        parameters_with_grad=0,
        parameters_without_grad=tuple(
            name for name, _ in named_parameters
        ),
    )


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError(
            "model has no parameters from which to resolve device"
        ) from exc


def _detach_dynamic_cache(cache: Any) -> int:
    """Detach pinned-Transformers DynamicCache tensors after an Adam step.

    The stale policy intentionally keeps pre-update KV values, but it must not
    retain their autograd graph into the next round.  Transformers 4.57 uses
    ``cache.layers[*].keys/values``; the legacy-list fallback makes the helper
    easy to exercise with adjacent cache implementations without silently
    changing values.
    """

    detached = 0
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            for attribute in ("keys", "values"):
                value = getattr(layer, attribute, None)
                if isinstance(value, torch.Tensor):
                    setattr(layer, attribute, value.detach())
                    detached += 1
        if int(cache.get_seq_length()) > 0 and not detached:
            raise RuntimeError(
                "non-empty DynamicCache exposes no detachable layer KV tensors"
            )
        return detached

    for attribute in ("key_cache", "value_cache"):
        values = getattr(cache, attribute, None)
        if isinstance(values, list):
            for index, value in enumerate(values):
                if isinstance(value, torch.Tensor):
                    values[index] = value.detach()
                    detached += 1
    return detached


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_memory_snapshot(device: torch.device) -> dict[str, int] | None:
    """Read allocator counters without synchronizing the device."""

    if device.type != "cuda":
        return None
    return {
        "allocated_end": int(torch.cuda.memory_allocated(device)),
        "reserved_end": int(torch.cuda.memory_reserved(device)),
        "running_peak_allocated": int(torch.cuda.max_memory_allocated(device)),
        "running_peak_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def _cuda_driver_version() -> int | None:
    """Read the CUDA driver ABI version without device synchronization."""

    torch_query = getattr(torch._C, "_cuda_getDriverVersion", None)
    if callable(torch_query):
        try:
            return int(torch_query())
        except (RuntimeError, TypeError, ValueError):
            pass
    library_name = ctypes.util.find_library("cuda")
    candidates = [library_name, "libcuda.so.1", "nvcuda.dll"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            driver = ctypes.CDLL(candidate)
            if driver.cuInit(0) != 0:
                continue
            version = ctypes.c_int()
            if driver.cuDriverGetVersion(ctypes.byref(version)) == 0:
                return int(version.value)
        except (AttributeError, OSError, TypeError, ValueError):
            continue
    return None


def determinism_contract(enabled: bool) -> dict[str, Any]:
    """Return the exact numerical-runtime settings promised by the CLI.

    This object is deliberately JSON-native so the runner can freeze the same
    values in its run identity and the harness can record the observed values
    in ``summary.json``.
    """

    return {
        "enabled": bool(enabled),
        "cublas_workspace_config": (
            DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG if enabled else None
        ),
        "torch_deterministic_algorithms": bool(enabled),
        "torch_deterministic_warn_only": False,
        "cuda_matmul_allow_tf32": not enabled,
        "cudnn_allow_tf32": not enabled,
        "float32_matmul_precision": "highest" if enabled else "high",
        "cudnn_benchmark": not enabled,
        "cudnn_deterministic": bool(enabled),
        "sdpa_backends": {
            "flash": not enabled,
            "memory_efficient": not enabled,
            "math": True,
            "cudnn": not enabled,
        },
    }


def configure_determinism(enabled: bool) -> dict[str, Any]:
    """Apply the numerical contract before any CUDA work and verify it.

    ``CUBLAS_WORKSPACE_CONFIG`` is process-global and must be present before
    CUDA initialization.  A direct harness invocation may set it here because
    importing torch does not initialize CUDA.  Deterministic mode rejects every
    already-initialized CUDA process because the harness cannot prove when an
    apparently-correct environment value was installed.  The frozen-sweep
    runner exports it before starting Python, which is the formal path.
    """

    expected = determinism_contract(enabled)
    expected_workspace = expected["cublas_workspace_config"]
    observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if enabled and torch.cuda.is_initialized():
        raise RuntimeError(
            "deterministic mode must be configured before CUDA initialization"
        )
    if torch.cuda.is_initialized() and observed_workspace != expected_workspace:
        raise RuntimeError(
            "numerical runtime must be configured before CUDA initialization: "
            "CUBLAS_WORKSPACE_CONFIG mismatch"
        )
    if expected_workspace is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected_workspace

    torch.use_deterministic_algorithms(bool(enabled), warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = not enabled
    torch.backends.cudnn.allow_tf32 = not enabled
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = bool(enabled)
    torch.set_float32_matmul_precision("highest" if enabled else "high")
    torch.backends.cuda.enable_flash_sdp(not enabled)
    torch.backends.cuda.enable_mem_efficient_sdp(not enabled)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(not enabled)

    warn_only_query = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None
    )
    observed = {
        "enabled": bool(enabled),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only": (
            bool(warn_only_query()) if callable(warn_only_query) else False
        ),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "sdpa_backends": {
            "flash": bool(torch.backends.cuda.flash_sdp_enabled()),
            "memory_efficient": bool(
                torch.backends.cuda.mem_efficient_sdp_enabled()
            ),
            "math": bool(torch.backends.cuda.math_sdp_enabled()),
            "cudnn": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        },
    }
    if observed != expected:
        raise RuntimeError(
            "failed to apply deterministic runtime contract: "
            f"expected={expected!r}, observed={observed!r}"
        )
    return observed


def build_runtime_fingerprint(
    *,
    device: torch.device,
    dtype: str,
    attention_implementation: str,
    requested_device: str | None = None,
    deterministic: bool | None = None,
) -> dict[str, Any]:
    """Capture numerical and allocator environment without a CUDA barrier."""

    gpu: dict[str, Any] | None = None
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "device_index": int(index),
        }
    allow_tf32 = {
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    warn_only_query = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None
    )
    fingerprint = {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda,
        "cuda_driver_version": (
            _cuda_driver_version() if device.type == "cuda" else None
        ),
        "attention_implementation": attention_implementation,
        "dtype": dtype,
        "device": str(device) if requested_device is None else requested_device,
        "resolved_device": str(device),
        "allocator_config": {
            name: os.environ.get(name)
            for name in (
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_ALLOC_CONF",
            )
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": (
            bool(warn_only_query()) if callable(warn_only_query) else None
        ),
        "allow_tf32": allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "sdpa_backends": {
            "flash": bool(torch.backends.cuda.flash_sdp_enabled()),
            "memory_efficient": bool(
                torch.backends.cuda.mem_efficient_sdp_enabled()
            ),
            "math": bool(torch.backends.cuda.math_sdp_enabled()),
            "cudnn": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        },
        "gpu": gpu,
    }
    if deterministic is not None:
        expected = determinism_contract(deterministic)
        observed = {
            "enabled": bool(deterministic),
            "cublas_workspace_config": fingerprint[
                "cublas_workspace_config"
            ],
            "torch_deterministic_algorithms": fingerprint[
                "deterministic_algorithms"
            ],
            "torch_deterministic_warn_only": (
                fingerprint["deterministic_warn_only"]
                if fingerprint["deterministic_warn_only"] is not None
                else False
            ),
            "cuda_matmul_allow_tf32": fingerprint["allow_tf32"]["matmul"],
            "cudnn_allow_tf32": fingerprint["allow_tf32"]["cudnn"],
            "float32_matmul_precision": fingerprint[
                "float32_matmul_precision"
            ],
            "cudnn_benchmark": fingerprint["cudnn_benchmark"],
            "cudnn_deterministic": fingerprint["cudnn_deterministic"],
            "sdpa_backends": fingerprint["sdpa_backends"],
        }
        if observed != expected:
            raise RuntimeError(
                "runtime fingerprint violates deterministic contract: "
                f"expected={expected!r}, observed={observed!r}"
            )
        fingerprint["determinism_contract"] = observed
    return fingerprint


def _timed_call(
    device: torch.device, function: Callable[[], Any]
) -> tuple[Any, float]:
    _sync(device)
    started = time.perf_counter()
    value = function()
    _sync(device)
    return value, time.perf_counter() - started


def _target_embedding(target: torch.nn.Module) -> torch.nn.Module:
    model = getattr(target, "model", None)
    embedding = getattr(model, "embed_tokens", None)
    if embedding is None:
        embedding = target.get_input_embeddings()
    if embedding is None:
        raise RuntimeError("target does not expose an input embedding")
    return embedding


def _target_head(target: torch.nn.Module) -> torch.nn.Module:
    head = getattr(target, "lm_head", None)
    if head is None:
        head = target.get_output_embeddings()
    if head is None:
        raise RuntimeError("target does not expose an LM head")
    return head


def _extract_hidden(
    extract_context_feature: Callable[
        [Sequence[torch.Tensor], Sequence[int]], torch.Tensor
    ],
    output: Any,
    layer_ids: Sequence[int],
) -> torch.Tensor:
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("target output did not include hidden_states")
    return extract_context_feature(hidden_states, list(layer_ids)).detach()


def run_reference_sequence(
    *,
    draft_model: torch.nn.Module,
    target: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: Sequence[int] | None,
    temperature: float,
    block_size: int,
    mask_token_id: int,
    mode: str,
    update_stride: int,
    position_weighting: str,
    position_decay_gamma: float | None,
    loss_reduction: str,
    proximal_lambda: float,
    optimizer: torch.optim.Optimizer | FP32MasterOptimizer | None,
    tail_adapter: TailAdapter | None = None,
    draft_cache_policy: str,
    cache_factory: Callable[[], Any],
    extract_context_feature: Callable[
        [Sequence[torch.Tensor], Sequence[int]], torch.Tensor
    ],
    seed: int,
    sample_id: str,
    provenance: dict[str, str],
    audit_cuda_timing: bool = False,
    parameter_audit_stride: int = 0,
    canonical_greedy_verifier: bool = True,
    round_observer: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    """Run one batch-one request with the official DFlash round semantics."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("DFlash reference harness requires input_ids shape [1, N]")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if block_size < 2:
        raise ValueError("TTS reference requires block_size >= 2")
    if update_stride <= 0:
        raise ValueError("update_stride must be positive")
    if parameter_audit_stride < 0:
        raise ValueError("parameter_audit_stride must be non-negative")
    if canonical_greedy_verifier and temperature >= 1e-5:
        raise ValueError(
            "canonical commit verification currently supports greedy decoding "
            "only; stochastic runs must use the non-exact diagnostic path"
        )
    if draft_cache_policy not in {"rebuild", "stale"}:
        raise ValueError("draft_cache_policy must be rebuild or stale")
    if mode not in ADAPTATION_MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "static" and optimizer is not None:
        raise ValueError("static mode must not construct an optimizer")
    if mode != "static" and optimizer is None:
        raise ValueError(f"{mode} mode requires an optimizer")
    if parameter_audit_stride > 0 and not isinstance(
        optimizer, FP32MasterOptimizer
    ):
        raise ValueError("parameter audit requires the FP32 master optimizer")
    if mode in TAIL_MODES and (
        tail_adapter is None or tail_adapter.mode != mode
    ):
        raise ValueError(f"{mode} requires its matching TailAdapter")
    if mode not in TAIL_MODES and tail_adapter is not None:
        raise ValueError(f"{mode} must not install a TailAdapter")
    if any(parameter.requires_grad for parameter in target.parameters()):
        raise RuntimeError("target must be frozen in both reference modes")
    draft_parameters = tuple(draft_model.parameters())
    if mode == "static" and any(
        parameter.requires_grad for parameter in draft_parameters
    ):
        raise RuntimeError("static mode requires every drafter parameter frozen")
    if mode == "full-drafter" and not all(
        parameter.requires_grad for parameter in draft_parameters
    ):
        raise RuntimeError(
            "full-drafter mode requires every DFlashDraftModel parameter trainable"
        )
    if mode in TAIL_MODES and any(
        parameter.requires_grad for parameter in draft_parameters
    ):
        raise RuntimeError(f"{mode} requires the drafter backbone frozen")
    if mode == "drafter-lora":
        adapters = tuple(
            module
            for module in draft_model.modules()
            if isinstance(module, DrafterLoRALinear)
        )
        expected_lora_ids = {
            id(parameter)
            for adapter in adapters
            for parameter in (adapter.lora_a, adapter.lora_b)
        }
        actual_trainable_ids = {
            id(parameter)
            for parameter in draft_parameters
            if parameter.requires_grad
        }
        if not adapters or actual_trainable_ids != expected_lora_ids:
            raise RuntimeError(
                "drafter-lora scope must contain exactly A/B for fc and all "
                "attention/MLP Linear wrappers"
            )
    trainable_named_parameters = _trainable_named_parameters(
        draft_model, tail_adapter
    )
    if mode != "static" and not trainable_named_parameters:
        raise RuntimeError(f"{mode} resolved to an empty trainable layout")
    if mode == "static" and trainable_named_parameters:
        raise RuntimeError("static mode resolved unexpected trainable parameters")
    runtime_layout = (
        tail_adapter.layout()
        if tail_adapter is not None
        else {
            "mode": mode,
            "parameters": [
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                }
                for name, parameter in trainable_named_parameters
            ],
        }
    )
    parameter_layout_sha256 = runtime_layout.get(
        "layout_sha256", _json_sha256(runtime_layout)
    )
    trainable_parameter_count = sum(
        parameter.numel() for _name, parameter in trainable_named_parameters
    )
    optimizer_name = (
        None
        if optimizer is None
        else (
            optimizer.optimizer_name
            if isinstance(optimizer, FP32MasterOptimizer)
            else optimizer.__class__.__name__.lower()
        )
    )
    if draft_model.training:
        raise RuntimeError(
            "drafter must stay in eval mode; full-drafter autograd does not require "
            "dropout-enabled train mode"
        )
    missing_provenance = ROUND_PROVENANCE_FIELDS.difference(provenance)
    if missing_provenance:
        raise ValueError(
            "round provenance is missing: "
            + ", ".join(sorted(missing_provenance))
        )

    device = _model_device(target)
    input_ids = input_ids.to(device=device, dtype=torch.long)
    num_input_tokens = int(input_ids.shape[1])
    max_length = num_input_tokens + int(max_new_tokens)
    output_ids = torch.full(
        (1, max_length + block_size),
        int(mask_token_id),
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    # The block verifier reproduces DFlash's target call and supplies the full
    # proposal-branch logits used by TTS distillation.  Greedy commit decisions
    # use a separate one-token cache so changing proposal acceptance cannot
    # change target tokens merely by changing target query chunk boundaries.
    block_target_cache = cache_factory()
    canonical_target_cache = (
        cache_factory() if canonical_greedy_verifier else block_target_cache
    )
    persistent_draft_cache = cache_factory()
    target_embedding = _target_embedding(target)
    target_head = _target_head(target)
    layer_ids = tuple(int(value) for value in draft_model.target_layer_ids)
    position_weights = build_position_weights(
        block_size - 1,
        position_weighting,
        position_decay_gamma,
        device=device,
    )

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    def target_prefill(target_cache):
        return target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=target_cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

    with torch.no_grad():
        prefill, canonical_prefill_seconds = _timed_call(
            device, lambda: target_prefill(canonical_target_cache)
        )
        block_prefill_seconds = 0.0
        if canonical_greedy_verifier:
            block_prefill, block_prefill_seconds = _timed_call(
                device, lambda: target_prefill(block_target_cache)
            )
            del block_prefill
        else:
            block_prefill_seconds = canonical_prefill_seconds
            canonical_prefill_seconds = 0.0
        prefill_seconds = canonical_prefill_seconds + block_prefill_seconds
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample_tokens(
            prefill.logits, temperature
        )
        initial_target_hidden = _extract_hidden(
            extract_context_feature, prefill, layer_ids
        )
        del prefill

    incremental_target_hidden = initial_target_hidden
    # Rebuild deliberately recomputes from the full accepted prefix.  Stale
    # only needs the incremental hidden slice and must not pay an O(L^2)
    # torch.cat allocation/bandwidth cost over a long generation.
    target_hidden_history: torch.Tensor | None = (
        initial_target_hidden if draft_cache_policy == "rebuild" else None
    )
    records: list[dict[str, Any]] = []
    start = num_input_tokens
    round_index = 0
    optimizer_steps = 0
    parameter_version = 0
    decode_started = time.perf_counter()

    while start < max_length:
        round_started = time.perf_counter()
        proposal_parameter_version = parameter_version
        block_ids = output_ids[:, start : start + block_size].clone()
        block_positions = position_ids[:, start : start + block_size]
        update_due = mode != "static" and (
            (round_index + 1) % update_stride == 0
        )
        if draft_cache_policy == "stale":
            draft_cache = persistent_draft_cache
            target_hidden = incremental_target_hidden
        else:
            draft_cache = cache_factory()
            if target_hidden_history is None:
                raise RuntimeError("rebuild policy requires full target hidden history")
            target_hidden = target_hidden_history

        draft_cache_len_before = int(draft_cache.get_seq_length())
        with torch.no_grad():
            noise_embedding = target_embedding(block_ids).detach()

        def draft_forward():
            def backbone_forward():
                return draft_model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :, draft_cache.get_seq_length() : start + block_size
                    ],
                    past_key_values=draft_cache,
                    use_cache=True,
                    is_causal=False,
                )[:, 1 - block_size :, :]

            if tail_adapter is not None:
                # Cache-safe tail training must not retain a backbone graph or
                # accidentally create gradients for the frozen target head.
                with torch.no_grad():
                    hidden = backbone_forward().detach()
                    base_logits = target_head(hidden).detach()
                return tail_adapter(hidden, base_logits, target_head)
            hidden = backbone_forward()
            return target_head(hidden)

        if update_due:
            draft_logits, draft_seconds = _timed_call(device, draft_forward)
        else:
            with torch.no_grad():
                draft_logits, draft_seconds = _timed_call(device, draft_forward)
        draft_cache_len_after_forward = int(draft_cache.get_seq_length())
        draft_cache.crop(start)
        draft_cache_len_after_crop = int(draft_cache.get_seq_length())
        block_ids[:, 1:] = sample_tokens(draft_logits.detach())

        def target_verify():
            return target(
                block_ids,
                position_ids=block_positions,
                past_key_values=block_target_cache,
                use_cache=True,
                output_hidden_states=True,
            )

        with torch.no_grad():
            target_output, verify_seconds = _timed_call(device, target_verify)
            block_posterior = sample_tokens(target_output.logits, temperature)
            block_accepted_draft_tokens = consecutive_acceptance_length(
                block_ids, block_posterior
            )

            canonical_verifier_seconds = 0.0
            canonical_verifier_token_ids: list[int] = []
            if canonical_greedy_verifier:

                def canonical_commit_verify():
                    hidden_slices: list[torch.Tensor] = []
                    for block_index in range(block_size):
                        one_token_output = target(
                            block_ids[:, block_index : block_index + 1],
                            position_ids=block_positions[
                                :, block_index : block_index + 1
                            ],
                            past_key_values=canonical_target_cache,
                            use_cache=True,
                            output_hidden_states=True,
                        )
                        next_token = sample_tokens(one_token_output.logits, 0.0)
                        next_token_id = int(next_token[0, 0].item())
                        canonical_verifier_token_ids.append(next_token_id)
                        hidden_slices.append(
                            _extract_hidden(
                                extract_context_feature,
                                one_token_output,
                                layer_ids,
                            )
                        )
                        if block_index == block_size - 1:
                            return (
                                block_size - 1,
                                next_token[:, 0],
                                torch.cat(hidden_slices, dim=1),
                            )
                        proposed_next = int(
                            block_ids[0, block_index + 1].item()
                        )
                        if proposed_next != next_token_id:
                            return (
                                block_index,
                                next_token[:, 0],
                                torch.cat(hidden_slices, dim=1),
                            )
                    raise AssertionError("unreachable canonical verifier state")

                (
                    (
                        accepted_draft_tokens,
                        bonus_token,
                        verified_hidden,
                    ),
                    canonical_verifier_seconds,
                ) = _timed_call(device, canonical_commit_verify)
            else:
                accepted_draft_tokens = block_accepted_draft_tokens
                bonus_token = block_posterior[:, accepted_draft_tokens]
                verified_hidden = _extract_hidden(
                    extract_context_feature, target_output, layer_ids
                )[:, : accepted_draft_tokens + 1, :]

        update_started = time.perf_counter()
        detached_cache_tensors = 0
        if update_due:
            optimizer_steps += 1
            update_evidence = apply_tts_optimizer_step(
                trainable_named_parameters=trainable_named_parameters,
                optimizer=optimizer,
                proposal_logits=draft_logits,
                target_logits=target_output.logits[:, : block_size - 1, :],
                position_weights=position_weights,
                proximal_lambda=proximal_lambda,
                loss_reduction=loss_reduction,
                optimizer_step=optimizer_steps,
                scope_name=TRAINABLE_SCOPE_NAMES[mode],
                audit_cuda_timing=audit_cuda_timing,
                audit_parameter_stats=(
                    parameter_audit_stride > 0
                    and optimizer_steps % parameter_audit_stride == 0
                ),
            )
            parameter_version += 1
            if (
                draft_cache_policy == "stale"
                and mode in DRAFTER_MUTATING_MODES
            ):
                detached_cache_tensors = _detach_dynamic_cache(draft_cache)
            _sync(device)
        else:
            update_evidence = _no_update_evidence(
                trainable_named_parameters
                if mode != "static"
                else tuple(draft_model.named_parameters())
            )
        update_seconds = time.perf_counter() - update_started

        official_acceptance_length = accepted_draft_tokens + 1
        output_ids[
            :, start : start + official_acceptance_length
        ] = block_ids[:, :official_acceptance_length]
        output_ids[:, start + official_acceptance_length] = bonus_token
        start += official_acceptance_length
        block_target_cache.crop(start)
        canonical_target_cache.crop(start)
        if int(canonical_target_cache.get_seq_length()) != start:
            raise RuntimeError(
                "canonical target cache length does not match committed prefix"
            )

        incremental_target_hidden = verified_hidden
        if draft_cache_policy == "rebuild":
            if target_hidden_history is None:
                raise RuntimeError("rebuild policy lost target hidden history")
            target_hidden_history = torch.cat(
                (target_hidden_history, verified_hidden), dim=1
            )

        draft_block_token_ids = [int(value) for value in block_ids[0].tolist()]
        target_posterior_token_ids = [
            int(value) for value in block_posterior[0].tolist()
        ]
        committed_token_ids = draft_block_token_ids[
            :official_acceptance_length
        ]
        bonus_token_id = int(bonus_token[0].item())

        record = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "round_index": round_index,
            "seed": seed,
            "provenance": dict(provenance),
            "mode": mode,
            "trainable_scope": TRAINABLE_SCOPE_NAMES[mode],
            "trainable_parameter_count": trainable_parameter_count,
            "parameter_layout_sha256": parameter_layout_sha256,
            "optimizer": optimizer_name,
            "draft_module_training": bool(draft_model.training),
            "draft_cache_policy": draft_cache_policy,
            "gradient_history": (
                "not_applicable_static"
                if mode == "static"
                else (
                    "detached_truncated"
                    if mode in DRAFTER_MUTATING_MODES
                    and draft_cache_policy == "stale"
                    else (
                        "full_prefix_recomputed_current_parameters"
                        if mode in DRAFTER_MUTATING_MODES
                        else "current_round_cache_safe_tail_only"
                    )
                )
            ),
            "proposal_cache_version": (
                "static_consistent"
                if mode == "static"
                else (
                    "hybrid_pre_update_history_plus_current_round"
                    if mode in DRAFTER_MUTATING_MODES
                    and draft_cache_policy == "stale"
                    else (
                        "current_parameters_full_prefix"
                        if mode in DRAFTER_MUTATING_MODES
                        else "cache_safe_frozen_drafter_history"
                    )
                )
            ),
            "proposal_parameter_version": proposal_parameter_version,
            "parameter_version_after_update": parameter_version,
            "prefix_len_before": start - official_acceptance_length,
            "accepted_draft_tokens": accepted_draft_tokens,
            "block_verifier_accepted_draft_tokens": (
                block_accepted_draft_tokens
            ),
            "acceptance_length": official_acceptance_length,
            "committed_token_ids": committed_token_ids,
            "bonus_token_id": bonus_token_id,
            "draft_block_token_ids": draft_block_token_ids,
            "proposal_token_ids": draft_block_token_ids[1:],
            "target_posterior_token_ids": target_posterior_token_ids,
            "canonical_verifier_token_ids": canonical_verifier_token_ids,
            "commit_verifier": (
                "canonical_greedy_q_len_1"
                if canonical_greedy_verifier
                else "block_non_exact_diagnostic"
            ),
            "target_calls": {
                "block_verify": 1,
                "canonical_commit_verify": len(
                    canonical_verifier_token_ids
                ),
                "physical_total": 1 + len(canonical_verifier_token_ids),
            },
            "timing_seconds": {
                "draft_forward": draft_seconds,
                "target_verify": verify_seconds,
                "target_block_verify": verify_seconds,
                "target_canonical_commit_verify": (
                    canonical_verifier_seconds
                ),
                "target_physical_total": (
                    verify_seconds + canonical_verifier_seconds
                ),
                "update": update_seconds,
                "round_total": time.perf_counter() - round_started,
                "round_total_scope": (
                    "includes mandatory device-to-host evidence extraction; "
                    "excludes round-observer JSON serialization and flush"
                ),
            },
            "update": asdict(update_evidence),
            "cache_lengths": {
                "draft_before": draft_cache_len_before,
                "draft_after_forward": draft_cache_len_after_forward,
                "draft_after_crop": draft_cache_len_after_crop,
                "target_after_crop": int(
                    canonical_target_cache.get_seq_length()
                ),
                "block_target_after_crop": int(
                    block_target_cache.get_seq_length()
                ),
                "canonical_target_after_crop": int(
                    canonical_target_cache.get_seq_length()
                ),
                "draft_tensors_detached_after_update": detached_cache_tensors,
            },
            "hbm_bytes": _cuda_memory_snapshot(device),
        }
        records.append(record)
        if round_observer is not None:
            round_observer(record)
        round_index += 1

        if stop_token_ids is not None:
            stop_set = {int(stop_id) for stop_id in stop_token_ids}
            if bonus_token_id in stop_set or any(
                token_id in stop_set for token_id in committed_token_ids
            ):
                break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_ids is not None:
        stops = torch.tensor(tuple(stop_token_ids), device=device)
        positions = torch.isin(output_ids[0, num_input_tokens:], stops).nonzero(
            as_tuple=True
        )[0]
        if positions.numel() > 0:
            output_ids = output_ids[
                :, : num_input_tokens + int(positions[0].item()) + 1
            ]
    decode_seconds = time.perf_counter() - decode_started
    decode_block_verify_seconds = sum(
        float(record["timing_seconds"]["target_block_verify"])
        for record in records
    )
    decode_canonical_verify_seconds = sum(
        float(record["timing_seconds"]["target_canonical_commit_verify"])
        for record in records
    )
    canonical_commit_calls = sum(
        int(record["target_calls"]["canonical_commit_verify"])
        for record in records
    )
    block_verify_calls = len(records)
    summary = {
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": int(output_ids.shape[1]) - num_input_tokens,
        "rounds": len(records),
        "optimizer_steps": optimizer_steps,
        "final_parameter_version": parameter_version,
        "mode": mode,
        "trainable_scope": TRAINABLE_SCOPE_NAMES[mode],
        "trainable_parameter_count": trainable_parameter_count,
        "parameter_layout_sha256": parameter_layout_sha256,
        "optimizer": optimizer_name,
        "time_to_first_token_seconds": prefill_seconds,
        "target_timing_seconds": {
            "block_prefill": block_prefill_seconds,
            "canonical_prefill": canonical_prefill_seconds,
            "block_verify_decode": decode_block_verify_seconds,
            "canonical_commit_verify_decode": (
                decode_canonical_verify_seconds
            ),
            "physical_total": (
                block_prefill_seconds
                + canonical_prefill_seconds
                + decode_block_verify_seconds
                + decode_canonical_verify_seconds
            ),
        },
        "target_calls": {
            "block_prefill": 1,
            "canonical_prefill": int(canonical_greedy_verifier),
            "block_verify_decode": block_verify_calls,
            "canonical_commit_verify_decode": canonical_commit_calls,
            "physical_total": (
                1
                + int(canonical_greedy_verifier)
                + block_verify_calls
                + canonical_commit_calls
            ),
        },
        "exactness": {
            "classification": (
                "canonical_greedy_q_len_1_reference_audit"
                if canonical_greedy_verifier
                else "non_exact_diagnostic"
            ),
            "selection_eligible": bool(canonical_greedy_verifier),
            "canonical_commit_verifier": bool(canonical_greedy_verifier),
            "block_verifier_role": (
                "proposal_branch_distillation_and_diagnostic_only"
                if canonical_greedy_verifier
                else "commit_and_distillation_non_exact_diagnostic"
            ),
        },
        "decode_seconds": decode_seconds,
        "time_per_output_token_seconds": (
            decode_seconds / max(int(output_ids.shape[1]) - num_input_tokens, 1)
        ),
        "acceptance_lengths": [record["acceptance_length"] for record in records],
        "block_verifier_acceptance_lengths": [
            int(record["block_verifier_accepted_draft_tokens"]) + 1
            for record in records
        ],
    }
    if device.type == "cuda":
        hbm_bytes = _cuda_memory_snapshot(device)
        if hbm_bytes is None:
            raise RuntimeError("CUDA memory snapshot unexpectedly unavailable")
        summary["peak_hbm_bytes"] = hbm_bytes["running_peak_allocated"]
        summary["peak_hbm_reserved_bytes"] = hbm_bytes[
            "running_peak_reserved"
        ]
        summary["hbm_bytes"] = hbm_bytes
    if isinstance(optimizer, FP32MasterOptimizer):
        summary["optimizer_memory_bytes"] = optimizer.memory_accounting()
    return output_ids, records, summary


def _load_dataset_record(path: Path, sample_index: int) -> tuple[dict[str, Any], str]:
    if sample_index < 0:
        raise ValueError("sample-index must be non-negative")
    if path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    else:
        value = json.loads(path.read_text())
        if isinstance(value, dict) and isinstance(value.get("data"), list):
            records = value["data"]
        elif isinstance(value, list):
            records = value
        else:
            raise ValueError("dataset JSON must be a list or contain a data list")
    if sample_index >= len(records):
        raise ValueError(
            f"sample-index {sample_index} is outside dataset size {len(records)}"
        )
    record = records[sample_index]
    if not isinstance(record, dict):
        raise ValueError("dataset record must be a JSON object")
    sample_id = str(record.get("id", record.get("sample_id", sample_index)))
    return record, sample_id


def _tokenize_record(
    tokenizer: Any,
    record: dict[str, Any],
    *,
    prompt_field: str,
    messages_field: str,
    turns_field: str,
    enable_thinking: bool,
) -> torch.Tensor:
    if messages_field in record:
        messages = record[messages_field]
        if not isinstance(messages, list):
            raise ValueError(f"dataset field {messages_field!r} must be a list")
        return tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    if turns_field in record:
        turns = record[turns_field]
        if not isinstance(turns, list) or not turns:
            raise ValueError(
                f"dataset field {turns_field!r} must be a non-empty list"
            )
        if not all(isinstance(turn, str) for turn in turns):
            raise ValueError(
                f"every entry in dataset field {turns_field!r} must be a string"
            )
        messages = [{"role": "user", "content": turn} for turn in turns]
        return tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    if prompt_field not in record:
        raise ValueError(
            "dataset record has none of "
            f"{messages_field!r}, {turns_field!r}, or {prompt_field!r}"
        )
    prompt = record[prompt_field]
    if not isinstance(prompt, str):
        raise ValueError(f"dataset field {prompt_field!r} must be a string")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _record_input_format(
    record: dict[str, Any],
    *,
    messages_field: str,
    turns_field: str,
    prompt_field: str,
) -> dict[str, Any]:
    if messages_field in record:
        return {"format": "messages", "field": messages_field}
    if turns_field in record:
        return {
            "format": "turns_as_user_messages",
            "field": turns_field,
            "turn_count": len(record[turns_field]),
        }
    return {
        "format": "prompt_as_user_message",
        "field": prompt_field,
    }


def _parse_stop_ids(value: str | None, tokenizer: Any) -> list[int] | None:
    if value is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        return None if eos is None else [int(eos)]
    if value.lower() == "none":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _hashed_file_identity(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _model_artifact_identity(path: Path) -> dict[str, Any]:
    """Bind model metadata and every local safetensors shard by content."""

    identity: dict[str, Any] = {"path": str(path)}
    for name in ("config.json", "model.safetensors.index.json"):
        candidate = path / name
        if candidate.is_file():
            identity[f"{name}_sha256"] = _sha256_file(candidate)
    identity["weight_files"] = [
        _hashed_file_identity(item) for item in sorted(path.glob("*.safetensors"))
    ]
    if not identity["weight_files"]:
        raise ValueError(f"model has no local safetensors weights: {path}")
    identity["content_identity_sha256"] = _json_sha256(
        {
            key: value
            for key, value in identity.items()
            if key != "path"
        }
    )
    return identity


def _tokenizer_artifact_identity(path: Path) -> dict[str, Any]:
    files = [
        _hashed_file_identity(path / name)
        for name in TOKENIZER_ARTIFACT_FILES
        if (path / name).is_file()
    ]
    if not files:
        raise ValueError(f"target model has no recognized tokenizer files: {path}")
    identity = {"path": str(path), "files": files}
    identity["content_identity_sha256"] = _json_sha256(files)
    return identity


def _identity_stat_cache(
    *, target_path: Path, draft_path: Path
) -> dict[str, list[dict[str, Any]]]:
    def collect(root: Path, names: Iterable[str]) -> list[dict[str, Any]]:
        rows = []
        for name in names:
            candidate = root / name
            if candidate.is_file():
                stat = candidate.stat()
                rows.append(
                    {
                        "name": name,
                        "bytes": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    }
                )
        return rows

    return {
        "target": collect(
            target_path,
            [
                "config.json",
                "model.safetensors.index.json",
                *(item.name for item in sorted(target_path.glob("*.safetensors"))),
            ],
        ),
        "draft": collect(
            draft_path,
            [
                "config.json",
                "model.safetensors.index.json",
                *(item.name for item in sorted(draft_path.glob("*.safetensors"))),
            ],
        ),
        "tokenizer": collect(target_path, TOKENIZER_ARTIFACT_FILES),
    }


def _load_artifact_identity_lock(
    path: Path,
    *,
    expected_sha256: str,
    target_path: Path,
    draft_path: Path,
    target_revision: str,
    draft_revision: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Trust content hashes once, while checking immutable-file stat guards."""

    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "artifact identity lock sha256 mismatch: expected "
            f"{expected_sha256}, got {observed_sha256}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid artifact identity lock {path}: {exc}") from exc
    expected_header = {
        "schema_version": 1,
        "kind": "dflash_tts_artifact_identity_lock",
    }
    for key, value in expected_header.items():
        if payload.get(key) != value:
            raise ValueError(f"artifact identity lock {key} mismatch")
    target = payload.get("target")
    draft = payload.get("draft")
    tokenizer = payload.get("tokenizer")
    if not all(isinstance(value, dict) for value in (target, draft, tokenizer)):
        raise ValueError("artifact identity lock lacks target/draft/tokenizer identity")
    assert isinstance(target, dict) and isinstance(draft, dict)
    if target.get("path") != str(target_path) or target.get("revision") != target_revision:
        raise ValueError("artifact identity lock target mismatch")
    if draft.get("path") != str(draft_path) or draft.get("revision") != draft_revision:
        raise ValueError("artifact identity lock draft mismatch")
    if tokenizer.get("path") != str(target_path):
        raise ValueError("artifact identity lock tokenizer path mismatch")
    stats = payload.get("file_stats")
    current_stats = _identity_stat_cache(
        target_path=target_path, draft_path=draft_path
    )
    if stats != current_stats:
        raise ValueError(
            "artifact files changed since the content identity lock was created"
        )
    for role, identity in (("target", target), ("draft", draft)):
        weights = identity.get("weight_files")
        if not isinstance(weights, list) or not weights:
            raise ValueError(f"artifact identity lock {role} has no weight shards")
        if any(
            not isinstance(item, dict)
            or set(("name", "bytes", "sha256")) - set(item)
            or not _is_sha256(item.get("sha256"))
            for item in weights
        ):
            raise ValueError(
                f"artifact identity lock {role} lacks shard content hashes"
            )
        expected_content_sha256 = _json_sha256(
            {
                key: value
                for key, value in identity.items()
                if key not in {"path", "revision", "content_identity_sha256"}
            }
        )
        if identity.get("content_identity_sha256") != expected_content_sha256:
            raise ValueError(
                f"artifact identity lock {role} content identity mismatch"
            )
    tokenizer_files = tokenizer.get("files")
    if not isinstance(tokenizer_files, list) or not tokenizer_files:
        raise ValueError("artifact identity lock has no tokenizer content hashes")
    if any(
        not isinstance(item, dict)
        or set(("name", "bytes", "sha256")) - set(item)
        or not _is_sha256(item.get("sha256"))
        for item in tokenizer_files
    ):
        raise ValueError("artifact identity lock tokenizer identity is incomplete")
    if tokenizer.get("content_identity_sha256") != _json_sha256(tokenizer_files):
        raise ValueError("artifact identity lock tokenizer content identity mismatch")
    return target, draft, tokenizer, {
        "path": str(path),
        "sha256": observed_sha256,
        "verification": "content_sha256_with_immutable_stat_guard_v1",
    }


def _rendered_input_token_ids_identity(input_ids: torch.Tensor) -> dict[str, Any]:
    values = np.ascontiguousarray(
        input_ids.detach().to(device="cpu", dtype=torch.int64).numpy(),
        dtype="<i8",
    )
    return {
        "serialization": "int64_le_c_order_v1",
        "shape": list(values.shape),
        "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


def _resolve_dtype(name: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return values[name]


def _config_context_limit(config: Any) -> int | None:
    candidates: list[int] = []
    for name in (
        "max_position_embeddings",
        "max_sequence_length",
        "max_seq_len",
        "model_max_length",
    ):
        value = getattr(config, name, None)
        if isinstance(value, int) and 0 < value < 10**9:
            candidates.append(int(value))
    for name in ("text_config", "model_config", "draft_config"):
        nested = getattr(config, name, None)
        if nested is not None and nested is not config:
            value = _config_context_limit(nested)
            if value is not None:
                candidates.append(value)
    return min(candidates) if candidates else None


def resolve_checkpoint_context_limit(
    target: torch.nn.Module, draft_model: torch.nn.Module
) -> int:
    limits = [
        value
        for value in (
            _config_context_limit(getattr(target, "config", None)),
            _config_context_limit(getattr(draft_model, "config", None)),
        )
        if value is not None
    ]
    if not limits:
        raise ValueError(
            "target and draft checkpoints expose no finite context limit; "
            "refusing an unaudited long-context run"
        )
    return min(limits)


def validate_total_context_limit(
    *,
    input_tokens: int,
    max_new_tokens: int,
    block_size: int,
    checkpoint_limit: int,
) -> int:
    """Fail before decode if the last prefix plus pending block cannot fit."""

    # ``start`` is the prefix length before a proposal.  The last possible
    # proposal starts one token before the requested generation ceiling.
    required = int(input_tokens) + int(max_new_tokens) + int(block_size) - 1
    if required > int(checkpoint_limit):
        raise ValueError(
            "requested prefix plus pending DFlash block exceeds checkpoint "
            f"context limit: required={required}, limit={checkpoint_limit}, "
            f"input={input_tokens}, max_new_tokens={max_new_tokens}, "
            f"block_size={block_size}"
        )
    return required


def _assert_official_parity(
    *,
    module: ModuleType,
    draft_model: torch.nn.Module,
    target: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: Sequence[int] | None,
    temperature: float,
    block_size: int,
    mask_token_id: int,
    cache_factory: Callable[[], Any],
    seed: int,
    provenance: dict[str, str],
) -> dict[str, Any]:
    """Certify the local stale-cache block path against untouched DFlash.

    This is a reconstruction check for the proposal/block branch, not the
    exact target-output gate.  Untouched DFlash commits the block verifier's
    result, whereas selection-eligible runs commit an independent ``q_len=1``
    canonical target token.  Near-tied logits can differ across those query
    chunkings, so asking the canonical commit path to equal the official block
    trajectory would incorrectly reject the exactness repair itself.

    ``rebuild`` is a counterfactual cache policy: it recomputes historical
    entries with a different GEMM shape instead of retaining upstream's old
    draft KV.  Deterministic CUDA execution does not make those BF16 reduction
    paths bitwise identical.  Rebuild is therefore paired against its own
    Static baseline in the cache-policy diagnostic, not certified as an exact
    reconstruction of upstream DFlash here.
    """

    was_training = draft_model.training
    prior_requires_grad = [
        parameter.requires_grad for parameter in draft_model.parameters()
    ]
    for parameter in draft_model.parameters():
        parameter.requires_grad_(False)
    draft_model.eval()
    device = _model_device(target)
    results: dict[str, Any] = {}
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    official = module.dflash_generate(
        draft_model,
        target=target,
        input_ids=input_ids.to(device),
        max_new_tokens=max_new_tokens,
        stop_token_ids=(None if stop_token_ids is None else list(stop_token_ids)),
        temperature=temperature,
        block_size=block_size,
        mask_token_id=mask_token_id,
        return_stats=True,
    )
    official_ids = official.output_ids.detach().cpu()
    official_lengths = [int(value) for value in official.acceptance_lengths]
    for policy in ("stale",):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        output_ids, _records, summary = run_reference_sequence(
            draft_model=draft_model,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            block_size=block_size,
            mask_token_id=mask_token_id,
            mode="static",
            update_stride=1,
            position_weighting="uniform",
            position_decay_gamma=None,
            loss_reduction="sum",
            proximal_lambda=0.0,
            optimizer=None,
            draft_cache_policy=policy,
            cache_factory=cache_factory,
            extract_context_feature=module.extract_context_feature,
            seed=seed,
            sample_id="static-parity-preflight",
            provenance=provenance,
            canonical_greedy_verifier=False,
        )
        ids_match = torch.equal(output_ids.detach().cpu(), official_ids)
        acceptance_match = summary["acceptance_lengths"] == official_lengths
        results[policy] = {
            "output_ids_match": ids_match,
            "acceptance_lengths_match": acceptance_match,
            "acceptance_lengths": summary["acceptance_lengths"],
        }
        if not ids_match or not acceptance_match:
            raise RuntimeError(
                f"static {policy} path does not match official dflash_generate"
            )
    draft_model.train(was_training)
    for parameter, requires_grad in zip(
        draft_model.parameters(), prior_requires_grad, strict=True
    ):
        parameter.requires_grad_(requires_grad)
    return {
        "max_new_tokens": max_new_tokens,
        "classification": "official_stale_cache_block_verifier_reconstruction",
        "official_policy": "stale",
        "official_acceptance_lengths": official_lengths,
        "policies": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=ADAPTATION_MODES, required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--reference-module", default="dflash.model")
    parser.add_argument("--reference-revision", default=OFFICIAL_REFERENCE_REVISION)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--draft-revision", required=True)
    parser.add_argument(
        "--artifact-identity-lock",
        help=(
            "immutable runner-generated target/draft/tokenizer content-hash lock; "
            "avoids re-reading weight shards for every run"
        ),
    )
    parser.add_argument("--artifact-identity-lock-sha256")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument(
        "--sample-index",
        type=int,
        required=True,
        help="explicit original dataset index; selected paper-like sample is 419",
    )
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument(
        "--turns-field",
        default="turns",
        help="benchmark list-of-strings field, converted to ordered user messages",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-identity-sha256",
        help="runner identity digest echoed into the completed artifact",
    )
    parser.add_argument(
        "--command-sha256",
        help=(
            "digest of the ordered harness argv excluding this flag/value; "
            "must be paired with --run-identity-sha256"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "enable the formal deterministic numerical contract; use "
            "--no-deterministic only for explicitly labelled performance runs"
        ),
    )
    parser.add_argument(
        "--canonical-greedy-verifier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use a proposal-boundary-invariant q_len=1 target cache for greedy "
            "accept/commit; disabling it produces a non-exact diagnostic run"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--stop-token-ids", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--proximal-lambda", type=float, required=True)
    parser.add_argument("--update-stride", type=int, required=True)
    parser.add_argument(
        "--position-weighting",
        choices=("uniform", "linear", "exponential"),
        required=True,
    )
    parser.add_argument("--position-decay-gamma", type=float, default=None)
    parser.add_argument(
        "--loss-reduction", choices=("weighted-mean", "sum"), required=True
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--audit-cuda-timing",
        action="store_true",
        help=(
            "record synchronized CUDA-event backward/optimizer/update timings; "
            "disabled by default because reading elapsed events synchronizes"
        ),
    )
    parser.add_argument(
        "--parameter-audit-stride",
        type=int,
        default=0,
        help=(
            "record CPU-snapshotted parameter delta/displacement every N "
            "optimizer steps; 0 disables the expensive full-parameter scan"
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="drafter-LoRA, tail-LoRA, and output-residual rank",
    )
    parser.add_argument(
        "--adapter-seed",
        type=int,
        default=0,
        help="locked LoRA/projection initialization seed",
    )
    parser.add_argument(
        "--projection-artifact",
        default=None,
        help=(
            "optional canonical .npz cache for output-residual projection/basis; "
            "other modes reject this flag"
        ),
    )
    parser.add_argument(
        "--draft-cache-policy", choices=("rebuild", "stale"), required=True
    )
    parser.add_argument("--parity-max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--skip-static-parity-preflight",
        action="store_true",
        help=(
            "explicitly skip official static token/AL parity "
            "(recorded as uncertified)"
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.proximal_lambda < 0.0:
        raise ValueError("--proximal-lambda must be non-negative")
    if args.update_stride <= 0:
        raise ValueError("--update-stride must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be non-negative")
    if getattr(args, "canonical_greedy_verifier", True) and args.temperature >= 1e-5:
        raise ValueError(
            "--canonical-greedy-verifier supports temperature=0 greedy "
            "selection only; use --no-canonical-greedy-verifier for an "
            "explicit non-exact stochastic diagnostic"
        )
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if getattr(args, "parameter_audit_stride", 0) < 0:
        raise ValueError("--parameter-audit-stride must be non-negative")
    if args.rank <= 0:
        raise ValueError("--rank must be positive")
    if args.rank > HIDDEN_PROJECTION_DIM and args.mode == "output-residual":
        raise ValueError(
            f"--rank cannot exceed {HIDDEN_PROJECTION_DIM} for output-residual"
        )
    if args.projection_artifact is not None and args.mode != "output-residual":
        raise ValueError(
            "--projection-artifact is only valid with --mode output-residual"
        )
    artifact_identity_lock = getattr(args, "artifact_identity_lock", None)
    artifact_identity_lock_sha256 = getattr(
        args, "artifact_identity_lock_sha256", None
    )
    if (artifact_identity_lock is None) != (
        artifact_identity_lock_sha256 is None
    ):
        raise ValueError(
            "--artifact-identity-lock and --artifact-identity-lock-sha256 "
            "must be provided together"
        )
    if artifact_identity_lock_sha256 is not None and (
        len(artifact_identity_lock_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in artifact_identity_lock_sha256
        )
    ):
        raise ValueError("--artifact-identity-lock-sha256 must be lowercase hex")
    run_identity_sha256 = getattr(args, "run_identity_sha256", None)
    command_sha256 = getattr(args, "command_sha256", None)
    if (run_identity_sha256 is None) != (command_sha256 is None):
        raise ValueError(
            "--run-identity-sha256 and --command-sha256 must be provided together"
        )
    for value, label in (
        (run_identity_sha256, "--run-identity-sha256"),
        (command_sha256, "--command-sha256"),
    ):
        if value is not None and not _is_sha256(value):
            raise ValueError(f"{label} must be lowercase hex")
    build_position_weights(
        2, args.position_weighting, args.position_decay_gamma, device="cpu"
    )


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    _validate_args(args)
    run_attestation = _command_attestation(
        raw_argv,
        run_identity_sha256=args.run_identity_sha256,
        command_sha256=args.command_sha256,
    )
    # This must precede model loading or any other operation that can create a
    # CUDA context.  The formal runner additionally sets the cuBLAS variable in
    # the child environment before Python starts.
    configure_determinism(args.deterministic)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    harness_source_path = Path(__file__).resolve()
    harness_source_sha256 = _sha256_file(harness_source_path)

    # Set offline guards before importing Transformers or the reference module.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    reference_root = _require_local_dir(args.reference_root, "reference root")
    target_path = _require_local_dir(args.target_model, "target model")
    draft_path = _require_local_dir(args.draft_model, "draft model")
    dataset_path = _require_local_file(args.dataset, "dataset")
    dataset_sha256 = _sha256_file(dataset_path)
    if args.artifact_identity_lock is not None:
        identity_lock_path = _require_local_file(
            args.artifact_identity_lock, "artifact identity lock"
        )
        (
            target_artifact_identity,
            draft_artifact_identity,
            tokenizer_artifact_identity,
            artifact_identity_lock,
        ) = _load_artifact_identity_lock(
            identity_lock_path,
            expected_sha256=args.artifact_identity_lock_sha256,
            target_path=target_path,
            draft_path=draft_path,
            target_revision=args.target_revision,
            draft_revision=args.draft_revision,
        )
        # The lock includes declared revisions for runner planning.  Summary
        # model objects keep the historical shape and store revision once.
        target_artifact_identity = dict(target_artifact_identity)
        draft_artifact_identity = dict(draft_artifact_identity)
        target_artifact_identity.pop("revision", None)
        draft_artifact_identity.pop("revision", None)
    else:
        target_artifact_identity = _model_artifact_identity(target_path)
        draft_artifact_identity = _model_artifact_identity(draft_path)
        tokenizer_artifact_identity = _tokenizer_artifact_identity(target_path)
        artifact_identity_lock = {
            "path": None,
            "sha256": None,
            "verification": "direct_content_sha256_v1",
        }
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    module = load_reference_module(reference_root, args.reference_module)
    source_path = Path(inspect.getsourcefile(module) or "").resolve()
    source_sha256 = _sha256_file(source_path)
    if args.reference_revision != OFFICIAL_REFERENCE_REVISION:
        raise ValueError(
            "this harness is audited only for DFlash reference revision "
            f"{OFFICIAL_REFERENCE_REVISION}; got {args.reference_revision}"
        )
    if source_sha256 != OFFICIAL_REFERENCE_SOURCE_SHA256:
        raise ValueError(
            "official DFlash model.py source drift: "
            f"expected {OFFICIAL_REFERENCE_SOURCE_SHA256}, got {source_sha256}"
        )
    round_provenance = {
        "reference_revision": args.reference_revision,
        "reference_source_sha256": source_sha256,
        "target_declared_revision": args.target_revision,
        "draft_declared_revision": args.draft_revision,
        "dataset_declared_revision": args.dataset_revision,
        "dataset_sha256": dataset_sha256,
        "harness_source_sha256": harness_source_sha256,
    }
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    except ImportError as exc:
        raise RuntimeError(
            "the offline reference harness requires transformers==4.57.1 "
            "and accelerate, as pinned by z-lab/dflash"
        ) from exc
    if transformers.__version__ != "4.57.1":
        raise RuntimeError(
            "z-lab/dflash 94e4abc pins transformers==4.57.1; got "
            f"{transformers.__version__}"
        )

    dtype = _resolve_dtype(args.dtype)
    target = AutoModelForCausalLM.from_pretrained(
        target_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    draft_model = module.DFlashDraftModel.from_pretrained(
        draft_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    if not isinstance(draft_model, module.DFlashDraftModel):
        raise RuntimeError("loaded draft is not the official DFlashDraftModel")
    if draft_model.__class__.__module__.startswith("sglang."):
        raise RuntimeError(
            "SGLang DFlashDraftModel is inference-only here; use the official "
            "Transformers reference class for full-drafter backward"
        )
    # Snapshot the checkpoint module before LoRA wrappers add runtime-only
    # parameters.  Pair identity must describe immutable model content, while
    # the post-install layout remains separate adaptation evidence.
    draft_checkpoint_parameter_layout = _parameter_layout(draft_model)
    tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    freeze_target(target)
    device = _model_device(target)
    runtime_fingerprint = build_runtime_fingerprint(
        device=device,
        dtype=args.dtype,
        attention_implementation=args.attn_implementation,
        requested_device=args.device,
        deterministic=args.deterministic,
    )
    hbm_after_model_load = _cuda_memory_snapshot(device)

    record, sample_id = _load_dataset_record(dataset_path, args.sample_index)
    input_ids = _tokenize_record(
        tokenizer,
        record,
        prompt_field=args.prompt_field,
        messages_field=args.messages_field,
        turns_field=args.turns_field,
        enable_thinking=args.enable_thinking,
    )
    rendered_input_identity = _rendered_input_token_ids_identity(input_ids)
    round_provenance.update(
        {
            "tokenizer_content_identity_sha256": tokenizer_artifact_identity[
                "content_identity_sha256"
            ],
            "rendered_input_token_ids_sha256": rendered_input_identity["sha256"],
        }
    )
    input_format = _record_input_format(
        record,
        messages_field=args.messages_field,
        turns_field=args.turns_field,
        prompt_field=args.prompt_field,
    )
    stop_token_ids = _parse_stop_ids(args.stop_token_ids, tokenizer)
    block_size = int(args.block_size or draft_model.block_size)
    mask_token_id = args.mask_token_id
    if mask_token_id is None:
        mask_token_id = getattr(draft_model, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError(
            "mask token id is absent from the draft checkpoint; pass --mask-token-id"
        )
    checkpoint_context_limit = resolve_checkpoint_context_limit(target, draft_model)
    required_context = validate_total_context_limit(
        input_tokens=int(input_ids.shape[1]),
        max_new_tokens=args.max_new_tokens,
        block_size=block_size,
        checkpoint_limit=checkpoint_context_limit,
    )

    parity: dict[str, Any]
    if args.skip_static_parity_preflight:
        parity = {"status": "skipped_by_explicit_cli"}
    else:
        parity = {
            "status": "passed",
            **_assert_official_parity(
                module=module,
                draft_model=draft_model,
                target=target,
                input_ids=input_ids,
                max_new_tokens=min(
                    args.max_new_tokens, args.parity_max_new_tokens
                ),
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
                block_size=block_size,
                mask_token_id=int(mask_token_id),
                cache_factory=DynamicCache,
                seed=args.seed,
                provenance=round_provenance,
            ),
        }

    configure_trainable_scope(
        draft_model,
        args.mode,
        rank=args.rank,
        adapter_seed=args.adapter_seed,
    )
    tail_adapter: TailAdapter | None = None
    if args.mode in TAIL_MODES:
        projection_binding = {
            "target_revision": args.target_revision,
            "draft_revision": args.draft_revision,
            "reference_revision": args.reference_revision,
            "reference_source_sha256": source_sha256,
            # The frozen LM head is loaded from these exact target shards.
            # A shape/dtype match alone is not an artifact identity.
            "target_head_artifact": target_artifact_identity,
            "draft_artifact": draft_artifact_identity,
        }
        tail_adapter = TailAdapter.from_target_head(
            mode=args.mode,
            target_head=_target_head(target),
            rank=args.rank,
            adapter_seed=args.adapter_seed,
            projection_artifact=args.projection_artifact,
            projection_binding=projection_binding,
        )
        tail_adapter.eval()
    hbm_after_adapter = _cuda_memory_snapshot(device)
    trainable_named_parameters = _trainable_named_parameters(
        draft_model, tail_adapter
    )
    optimizer: FP32MasterOptimizer | None = None
    if args.mode != "static":
        optimizer = FP32MasterOptimizer(
            trainable_named_parameters,
            optimizer_name=args.optimizer,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            parameter_audit_enabled=args.parameter_audit_stride > 0,
        )
    hbm_after_optimizer = _cuda_memory_snapshot(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rounds_path = output_dir / "rounds.jsonl"
    with rounds_path.open("w", encoding="utf-8") as rounds_file:

        def observe(row: dict[str, Any]) -> None:
            rounds_file.write(json.dumps(row, sort_keys=True) + "\n")
            rounds_file.flush()

        output_ids, _rounds, generation = run_reference_sequence(
            draft_model=draft_model,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=args.temperature,
            block_size=block_size,
            mask_token_id=int(mask_token_id),
            mode=args.mode,
            update_stride=args.update_stride,
            position_weighting=args.position_weighting,
            position_decay_gamma=args.position_decay_gamma,
            loss_reduction=args.loss_reduction,
            proximal_lambda=args.proximal_lambda,
            optimizer=optimizer,
            tail_adapter=tail_adapter,
            draft_cache_policy=args.draft_cache_policy,
            cache_factory=DynamicCache,
            extract_context_feature=module.extract_context_feature,
            seed=args.seed,
            sample_id=sample_id,
            provenance=round_provenance,
            audit_cuda_timing=args.audit_cuda_timing,
            parameter_audit_stride=args.parameter_audit_stride,
            canonical_greedy_verifier=args.canonical_greedy_verifier,
            round_observer=observe,
        )
    rounds_sha256 = _sha256_file(rounds_path)
    hbm_after_run = _cuda_memory_snapshot(device)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete_reference_run"
            if args.canonical_greedy_verifier
            else "complete_non_exact_diagnostic_run"
        ),
        "harness": {
            "source_sha256": harness_source_sha256,
            "artifact_schema_version": SCHEMA_VERSION,
        },
        "run_attestation": run_attestation,
        "artifact_identity": {
            "verification_status": "fully_verified_content_sha256_v1",
            "lock": artifact_identity_lock,
        },
        "runtime_fingerprint": runtime_fingerprint,
        "method": {
            "static": "static_dflash",
            "full-drafter": "tts_full_drafter_reconstruction",
            "drafter-lora": "tts_drafter_wide_lora_reconstruction",
            "full-rank-tail": "tts_lightcone_full_rank_tail_reference",
            "tail-lora": "tts_lightcone_tail_lora_reference",
            "output-residual": "tts_lightcone_output_residual_reference",
        }[args.mode],
        "mode": args.mode,
        "trainable_scope": TRAINABLE_SCOPE_NAMES[args.mode],
        "trainable_layout": _trainable_layout(
            args.mode,
            trainable_named_parameters,
            rank=args.rank,
            adapter_seed=args.adapter_seed,
            tail_adapter=tail_adapter,
        ),
        "explicit_non_equivalence": {
            "is_full_drafter": args.mode == "full-drafter",
            "is_drafter_wide_lora": args.mode == "drafter-lora",
            "is_lightcone_cache_safe_tail": args.mode in TAIL_MODES,
            "note": (
                "full-drafter means every official DFlashDraftModel parameter; "
                "drafter-lora means fc plus all attention/MLP Linear modules; "
                "the three tail modes mutate only proposal-head outputs and are "
                "not aliases for either drafter-wide scope"
            ),
        },
        "reconstruction_status": {
            "paper_lr_public": False,
            "paper_position_weighting_public": False,
            "paper_proximal_lambda_public": False,
            "paper_optimizer_and_weight_decay_public": False,
            "one_step_proximal_effect": (
                "source-point KL has zero first-step gradient; retained only as "
                "an explicit auditable reconstruction parameter"
            ),
            "paper_stride_is_cli_reconstruction": True,
            "cache_policy_is_cli_reconstruction": True,
            "cache_policy": args.draft_cache_policy,
            "cache_policy_note": (
                "for full-drafter/drafter-lora, rebuild recomputes the full prefix "
                "and stale treats historical KV as frozen. Tail modes leave the "
                "backbone unchanged, so both policies remain cache-safe. None is "
                "claimed as the paper's undisclosed implementation."
            ),
            "gradient_semantics": (
                "detached_truncated_hybrid_cache"
                if args.mode in DRAFTER_MUTATING_MODES
                and args.draft_cache_policy == "stale"
                else (
                    "full_prefix_recomputed_current_parameters"
                    if args.mode in DRAFTER_MUTATING_MODES
                    else (
                        "current_round_cache_safe_tail_only"
                        if args.mode in TAIL_MODES
                        else "not_applicable_static"
                    )
                )
            ),
            "parameter_scope_vs_gradient_note": (
                "scope describes optimizer ownership; stale full/drafter-LoRA "
                "does not backpropagate through historical KV, while tail modes "
                "intentionally need only current proposal hidden states"
            ),
        },
        "timing_status": {
            "classification": "instrumented_reference_non_headline",
            "cuda_synchronization": (
                "synchronize before and after draft/verify calls and after updates"
            ),
            "note": (
                "round timings prioritize auditability and phase attribution; "
                "mandatory acceptance/token/update scalar host transfers and "
                "synchronous JSONL flushes plus the canonical q_len=1 commit "
                "verifier make wall time and phase sums unsuitable for DFlash/"
                "SGLang headline throughput comparisons"
            ),
            "long_context_cache_limit": (
                "the pinned official Transformers DynamicCache grows with "
                "torch.cat and crop views; stale avoids full target-hidden "
                "history but remains an O(L^2)-copy reference cache, not an "
                "optimized 8K serving cache"
            ),
            "checkpoint_context_limit": checkpoint_context_limit,
            "required_prefix_plus_block": required_context,
            "audit_cuda_timing": args.audit_cuda_timing,
            "cuda_event_fields": (
                ["backward_cuda_us", "optimizer_cuda_us", "update_cuda_us"]
                if args.audit_cuda_timing
                else []
            ),
        },
        "reference": {
            "declared_revision": args.reference_revision,
            "expected_revision": OFFICIAL_REFERENCE_REVISION,
            "module": args.reference_module,
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "transformers_version": transformers.__version__,
            "official_static_parity": parity,
        },
        "models": {
            "target": {
                **target_artifact_identity,
                "declared_revision": args.target_revision,
                "frozen": True,
            },
            "draft": {
                **draft_artifact_identity,
                "declared_revision": args.draft_revision,
                **draft_checkpoint_parameter_layout,
                "runtime_parameter_layout": _parameter_layout(draft_model),
                "all_parameters_in_optimizer": args.mode == "full-drafter",
                "drafter_lora_modules": [
                    name
                    for name, module_ in draft_model.named_modules()
                    if isinstance(module_, DrafterLoRALinear)
                ],
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for name, parameter in trainable_named_parameters
                    if name.startswith("draft.")
                ),
                "module_training": bool(draft_model.training),
                "autograd_enabled_for_all_parameters": (
                    args.mode == "full-drafter"
                ),
                "dropout_policy": "disabled_by_eval_mode",
                "dropout_modules": [
                    {"name": name, "p": float(module_.p)}
                    for name, module_ in draft_model.named_modules()
                    if isinstance(module_, torch.nn.modules.dropout._DropoutNd)
                ],
                "dropout_config": {
                    key: value
                    for key, value in (
                        draft_model.config.to_dict().items()
                        if hasattr(draft_model, "config")
                        else ()
                    )
                    if "dropout" in key.lower()
                },
            },
        },
        "tokenizer": tokenizer_artifact_identity,
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "declared_revision": args.dataset_revision,
            "sample_index": args.sample_index,
            "sample_id": sample_id,
            "input_format": input_format,
            "thinking_requested": args.enable_thinking,
            "thinking_effective_via_chat_template": (
                args.enable_thinking
            ),
            "rendered_input_token_ids": rendered_input_identity,
        },
        "parameters": {
            "seed": args.seed,
            "deterministic": args.deterministic,
            "canonical_greedy_verifier": (
                args.canonical_greedy_verifier
            ),
            "temperature": args.temperature,
            "block_size": block_size,
            "mask_token_id": int(mask_token_id),
            "stop_token_ids": stop_token_ids,
            "max_new_tokens": args.max_new_tokens,
            "checkpoint_context_limit": checkpoint_context_limit,
            "required_prefix_plus_block": required_context,
            "lr": args.lr,
            "proximal_lambda": args.proximal_lambda,
            "update_stride": args.update_stride,
            "position_weighting": args.position_weighting,
            "position_decay_gamma": args.position_decay_gamma,
            "loss_reduction": args.loss_reduction,
            "optimizer": args.optimizer.upper() if optimizer is not None else None,
            "optimizer_steps_per_selected_round": (
                1 if optimizer is not None else 0
            ),
            "adam_betas": [args.adam_beta1, args.adam_beta2],
            "adam_eps": args.adam_eps,
            "weight_decay": args.weight_decay,
            "audit_cuda_timing": args.audit_cuda_timing,
            "parameter_audit_stride": args.parameter_audit_stride,
            "parity_max_new_tokens": args.parity_max_new_tokens,
            "skip_static_parity_preflight": (
                args.skip_static_parity_preflight
            ),
            "rank": (
                args.rank
                if args.mode
                in {"drafter-lora", "tail-lora", "output-residual"}
                else None
            ),
            "adapter_seed": (
                args.adapter_seed
                if args.mode
                in {"drafter-lora", "tail-lora", "output-residual"}
                else None
            ),
            "projection_artifact": (
                None
                if args.projection_artifact is None
                else str(_projection_artifact_paths(args.projection_artifact)[0])
            ),
            "precision_contract": {
                "forward_parameters": args.dtype,
                "master_parameters": "float32" if optimizer is not None else None,
                "master_gradients": "float32" if optimizer is not None else None,
                "optimizer_moments": "float32" if optimizer is not None else None,
            },
            "optimizer_memory_bytes": (
                optimizer.memory_accounting() if optimizer is not None else None
            ),
            "draft_cache_policy": args.draft_cache_policy,
            "dtype": args.dtype,
            "device": args.device,
            "enable_thinking": args.enable_thinking,
            "prompt_field": args.prompt_field,
            "messages_field": args.messages_field,
            "turns_field": args.turns_field,
        },
        "generation": generation,
        "exactness": generation["exactness"],
        "hbm_bytes": {
            "after_model_load": hbm_after_model_load,
            "after_adapter": hbm_after_adapter,
            "after_optimizer": hbm_after_optimizer,
            "after_run": hbm_after_run,
        },
        "output": {
            "token_ids": [int(value) for value in output_ids[0].tolist()],
            "text": tokenizer.decode(output_ids[0], skip_special_tokens=False),
            "rounds_jsonl": rounds_path.name,
            "rounds_sha256": rounds_sha256,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"summary": str(summary_path), "generation": generation}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
