"""Backend-native trainable plans and merge-only LoRA state."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

LORA_RANKS = (1, 2, 4, 8, 16, 32, 64)
LAYER_SCOPES = ("last1", "last3", "last5", "all")
DSPARK_HYBRID_SCOPES = (
    "last1_native_heads",
    "last3_native_heads",
    "last5_native_heads",
)

_BORROWED_COMPONENTS = (
    "embed_tokens",
    "lm_head",
    "target_model",
    "target_embedding",
)
_LORA_LINEAR = re.compile(
    r"(?:^|\.)(?:fc|q_proj|k_proj|v_proj|qkv_proj|o_proj|"
    r"gate_proj|gate_up_proj|up_proj|down_proj)(?:\.|$)"
)
_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _is_owned(name: str) -> bool:
    components = frozenset(name.split("."))
    return components.isdisjoint(_BORROWED_COMPONENTS)


def _dtype_bytes(dtype: str) -> int:
    sizes = {
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.float32": 4,
        "torch.float64": 8,
    }
    try:
        return sizes[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported trainable dtype {dtype}") from exc


def _numel(shape: tuple[int, ...]) -> int:
    value = 1
    for dimension in shape:
        value *= dimension
    return value


@dataclass(frozen=True)
class ParameterEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str
    parameterization: str = "full"
    ownership: str = "sharded"

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.shape
            or any(dimension < 1 for dimension in self.shape)
        ):
            raise ValueError("parameter entries require a name and positive shape")
        if self.parameterization not in {"full", "lora"}:
            raise ValueError("parameterization must be full or lora")
        if self.ownership not in {"sharded", "replicated"}:
            raise ValueError("ownership must be sharded or replicated")


@dataclass(frozen=True)
class PlanMemoryPrediction:
    active_merged: int
    masters: int
    gradients: int
    optimizer_first: int
    optimizer_second: int
    candidate: int
    staging: int
    merge_scratch: int

    @property
    def resident_bytes(self) -> int:
        return (
            self.active_merged
            + self.masters
            + self.optimizer_first
            + self.optimizer_second
            + self.staging
        )

    @property
    def peak_bytes(self) -> int:
        return (
            self.resident_bytes + self.gradients + self.candidate + self.merge_scratch
        )


@dataclass(frozen=True)
class TrainablePlan:
    backend: str
    mode: str
    scope: str
    rank: int | None
    lora_alpha: int | None
    entries: tuple[ParameterEntry, ...]
    frozen_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.backend not in {"DFLASH", "DSPARK", "EAGLE", "EAGLE3", "NEXTN"}:
            raise ValueError("unknown proposal backend")
        if self.mode not in {"lora", "full"}:
            raise ValueError("only Full and LoRA parameterizations are supported")
        if self.mode == "full" and (
            self.rank is not None or self.lora_alpha is not None
        ):
            raise ValueError("Full plans require rank and alpha to be null")
        if self.mode == "lora" and (
            self.rank not in LORA_RANKS or self.lora_alpha != self.rank
        ):
            raise ValueError("LoRA requires a registered rank and alpha/r=1")
        if not self.entries:
            raise ValueError("trainable plan must not be empty")
        names = tuple(entry.name for entry in self.entries)
        if len(names) != len(set(names)):
            raise ValueError("trainable plan contains duplicate parameters")
        if set(names) & set(self.frozen_names):
            raise ValueError("a parameter cannot be both trainable and frozen")

    @property
    def sha256(self) -> str:
        body = {
            "backend": self.backend,
            "mode": self.mode,
            "scope": self.scope,
            "rank": self.rank,
            "lora_alpha": self.lora_alpha,
            "entries": [entry.__dict__ for entry in self.entries],
            "frozen_names": self.frozen_names,
            "state_layout": self.state_layout,
        }
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def trainable_parameter_count(self) -> int:
        count = 0
        for entry in self.entries:
            if entry.parameterization == "lora":
                if self.rank is None or len(entry.shape) != 2:
                    raise AssertionError("LoRA layout is not a ranked matrix")
                count += self.rank * (entry.shape[0] + entry.shape[1])
            else:
                count += _numel(entry.shape)
        return count

    @property
    def state_layout(self) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for entry in self.entries:
            if entry.parameterization == "lora":
                assert self.rank is not None
                shapes = ((self.rank, entry.shape[1]), (entry.shape[0], self.rank))
            else:
                shapes = (entry.shape,)
            rows.append(
                {
                    "name": entry.name,
                    "parameterization": entry.parameterization,
                    "ownership": entry.ownership,
                    "state_shapes": shapes,
                }
            )
        return tuple(rows)

    def predict_memory(self, optimizer: str) -> PlanMemoryPrediction:
        """Generate state bytes from this exact plan, never a parallel estimator."""
        if optimizer not in {"adam", "adamw", "sgdm", "nag", "muon", "lion"}:
            raise ValueError("optimizer has no implemented state layout")
        trainable = self.trainable_parameter_count
        masters = 4 * trainable
        gradients = 4 * trainable
        first = 4 * trainable
        second = 4 * trainable if optimizer in {"adam", "adamw"} else 0
        if optimizer == "muon":
            second = sum(
                8 * _numel(entry.shape)
                for entry in self.entries
                if entry.parameterization == "full" and len(entry.shape) != 2
            )
        # Inference and staging banks always contain the full selected native
        # tensors.  A LoRA factorization changes trainable state, not the size
        # of the fixed-address merged weights consumed by inference.
        active = sum(
            _numel(entry.shape) * _dtype_bytes(entry.dtype) for entry in self.entries
        )
        # ``LoRAFactors.merged`` materializes FP32 base, product, and merged
        # output tensors.  Charge all three full-matrix temporaries rather than
        # treating factor-only state as the merge peak.
        merge_scratch = sum(
            3 * 4 * _numel(entry.shape)
            for entry in self.entries
            if entry.parameterization == "lora"
        )
        return PlanMemoryPrediction(
            active_merged=active,
            masters=masters,
            gradients=gradients,
            optimizer_first=first,
            optimizer_second=second,
            candidate=masters + first + second,
            staging=active,
            merge_scratch=merge_scratch,
        )


def _normalise_items(
    named_parameters: Mapping[str, Tensor] | Iterable[tuple[str, Tensor]],
) -> tuple[tuple[str, Tensor], ...]:
    items = (
        tuple(named_parameters.items())
        if isinstance(named_parameters, Mapping)
        else tuple(named_parameters)
    )
    names = tuple(name for name, _ in items)
    if len(names) != len(set(names)):
        raise ValueError("named parameter input contains duplicate names")
    return items


def _selected_layers(items: tuple[tuple[str, Tensor], ...], scope: str) -> set[int]:
    indices = {
        int(match.group(1))
        for name, _ in items
        if (match := _LAYER_INDEX.search(name)) is not None
    }
    if not indices:
        raise ValueError("backend selector found no drafter layers")
    ordered = sorted(indices)
    if ordered != list(range(ordered[-1] + 1)):
        raise ValueError("drafter layer indices must be contiguous from zero")
    if scope == "all":
        return set(ordered)
    try:
        count = int(scope.removeprefix("last"))
    except ValueError as exc:
        raise ValueError(f"unknown layer scope {scope}") from exc
    if count > len(ordered):
        raise ValueError(f"scope {scope} exceeds the backend layer count")
    return set(ordered[-count:])


def _entry(
    name: str,
    parameter: Tensor,
    *,
    parameterization: str,
    ownership: str,
) -> ParameterEntry:
    return ParameterEntry(
        name=name,
        shape=tuple(parameter.shape),
        dtype=str(parameter.dtype),
        parameterization=parameterization,
        ownership=ownership,
    )


class DFlashParameterPlan(TrainablePlan):
    @classmethod
    def build(
        cls,
        named_parameters: Mapping[str, Tensor] | Iterable[tuple[str, Tensor]],
        *,
        mode: str,
        scope: str,
        rank: int | None = None,
        replicated_names: Iterable[str] = (),
    ) -> DFlashParameterPlan:
        if mode not in {"lora", "full"}:
            raise ValueError("DFlash supports only Full and LoRA")
        if scope not in LAYER_SCOPES:
            raise ValueError("DFlash scope must be last1, last3, last5, or all")
        items = _normalise_items(named_parameters)
        layers = _selected_layers(items, scope)
        replicated = frozenset(replicated_names)
        selected: list[ParameterEntry] = []
        frozen: list[str] = []
        for name, parameter in items:
            match = _LAYER_INDEX.search(name)
            eligible = (
                _is_owned(name)
                and parameter.is_floating_point()
                and match is not None
                and int(match.group(1)) in layers
            )
            parameterization = "full"
            if mode == "lora":
                eligible &= (
                    parameter.ndim == 2 and _LORA_LINEAR.search(name) is not None
                )
                parameterization = "lora"
            if eligible:
                selected.append(
                    _entry(
                        name,
                        parameter,
                        parameterization=parameterization,
                        ownership="replicated" if name in replicated else "sharded",
                    )
                )
            else:
                frozen.append(name)
        return cls(
            backend="DFLASH",
            mode=mode,
            scope=scope,
            rank=rank,
            lora_alpha=rank if mode == "lora" else None,
            entries=tuple(selected),
            frozen_names=tuple(sorted(frozen)),
        )


class NativeLayerParameterPlan(TrainablePlan):
    """Layer-native selector for EAGLE-family and native NEXTN backends."""

    @classmethod
    def build(
        cls,
        named_parameters: Mapping[str, Tensor] | Iterable[tuple[str, Tensor]],
        *,
        backend: str,
        mode: str,
        scope: str,
        rank: int | None = None,
        replicated_names: Iterable[str] = (),
    ) -> NativeLayerParameterPlan:
        if backend not in {"EAGLE", "EAGLE3", "NEXTN"}:
            raise ValueError("native layer plan supports EAGLE/EAGLE3/NEXTN only")
        if mode not in {"lora", "full"}:
            raise ValueError("native layer plans support only Full and LoRA")
        if scope not in LAYER_SCOPES:
            raise ValueError("native scope must be last1, last3, last5, or all")
        items = _normalise_items(named_parameters)
        layers = _selected_layers(items, scope)
        replicated = frozenset(replicated_names)
        selected: list[ParameterEntry] = []
        frozen: list[str] = []
        for name, parameter in items:
            match = _LAYER_INDEX.search(name)
            eligible = (
                _is_owned(name)
                and parameter.is_floating_point()
                and match is not None
                and int(match.group(1)) in layers
            )
            parameterization = "full"
            if mode == "lora":
                eligible &= (
                    parameter.ndim == 2 and _LORA_LINEAR.search(name) is not None
                )
                parameterization = "lora"
            if eligible:
                selected.append(
                    _entry(
                        name,
                        parameter,
                        parameterization=parameterization,
                        ownership="replicated" if name in replicated else "sharded",
                    )
                )
            else:
                frozen.append(name)
        return cls(
            backend=backend,
            mode=mode,
            scope=scope,
            rank=rank,
            lora_alpha=rank if mode == "lora" else None,
            entries=tuple(selected),
            frozen_names=tuple(sorted(frozen)),
        )


class DSparkParameterPlan(TrainablePlan):
    @classmethod
    def build(
        cls,
        named_parameters: Mapping[str, Tensor] | Iterable[tuple[str, Tensor]],
        *,
        mode: str,
        scope: str,
        rank: int | None = None,
        w1_name: str,
        w2_name: str,
        acceptance_name: str,
        replicated_names: Iterable[str] = (),
    ) -> DSparkParameterPlan:
        if mode not in {"lora", "full"}:
            raise ValueError("DSpark supports only Full and LoRA")
        if scope not in {*LAYER_SCOPES, *DSPARK_HYBRID_SCOPES}:
            raise ValueError("unregistered DSpark scope")
        hybrid = scope in DSPARK_HYBRID_SCOPES
        layer_scope = scope.removesuffix("_native_heads")
        if hybrid and layer_scope == "all":
            raise ValueError("DSpark has no all-layer native-head hybrid cell")
        items = _normalise_items(named_parameters)
        by_name = dict(items)
        head_names = (w1_name, w2_name, acceptance_name)
        if len(set(head_names)) != 3 or any(name not in by_name for name in head_names):
            raise ValueError("DSpark W1, W2, and acceptance names must resolve exactly")
        if by_name[acceptance_name].ndim > 1 or by_name[acceptance_name].numel() != 1:
            raise ValueError("DSpark acceptance projection must be scalar")
        layers = _selected_layers(items, layer_scope)
        replicated = frozenset((*replicated_names, *head_names))
        selected: list[ParameterEntry] = []
        frozen: list[str] = []
        for name, parameter in items:
            match = _LAYER_INDEX.search(name)
            in_layer = match is not None and int(match.group(1)) in layers
            is_head = name in head_names
            eligible = _is_owned(name) and parameter.is_floating_point() and in_layer
            parameterization = "full"
            if mode == "lora":
                eligible &= (
                    parameter.ndim == 2 and _LORA_LINEAR.search(name) is not None
                )
                parameterization = "lora"
            if hybrid and is_head:
                eligible = True
                parameterization = "full"
            if eligible:
                selected.append(
                    _entry(
                        name,
                        parameter,
                        parameterization=parameterization,
                        ownership="replicated" if name in replicated else "sharded",
                    )
                )
            else:
                frozen.append(name)
        selected_names = {entry.name for entry in selected}
        if hybrid and not set(head_names) <= selected_names:
            raise AssertionError("DSpark hybrid plan lost a native head")
        if not hybrid and set(head_names) & selected_names:
            raise AssertionError("layer-only DSpark plan must freeze native heads")
        return cls(
            backend="DSPARK",
            mode=mode,
            scope=scope,
            rank=rank,
            lora_alpha=rank if mode == "lora" else None,
            entries=tuple(selected),
            frozen_names=tuple(sorted(frozen)),
        )


@dataclass
class LoRAFactors:
    """Two trainable factors with an exactly zero functional initial delta."""

    a: Tensor
    b: Tensor
    rank: int
    alpha: int

    @classmethod
    def initialize(
        cls,
        weight: Tensor,
        rank: int,
        *,
        seed: int,
        alpha: int | None = None,
    ) -> LoRAFactors:
        if weight.ndim != 2:
            raise ValueError("LoRA requires a matrix")
        if rank not in LORA_RANKS or rank > min(weight.shape):
            raise ValueError("invalid registered LoRA rank")
        resolved_alpha = rank if alpha is None else alpha
        if resolved_alpha != rank:
            raise ValueError("registered LoRA requires alpha/r=1")
        generator = torch.Generator(device=weight.device)
        generator.manual_seed(seed)
        a = torch.empty(
            (rank, weight.shape[1]),
            device=weight.device,
            dtype=torch.float32,
        )
        torch.nn.init.kaiming_uniform_(a, a=5**0.5, generator=generator)
        b = torch.zeros(
            (weight.shape[0], rank),
            device=weight.device,
            dtype=torch.float32,
        )
        return cls(a=a, b=b, rank=rank, alpha=resolved_alpha)

    def merged(self, base: Tensor) -> Tensor:
        if base.shape != (self.b.shape[0], self.a.shape[1]):
            raise ValueError("LoRA factors do not match the base matrix")
        return base.to(torch.float32) + (self.alpha / self.rank) * (self.b @ self.a)
