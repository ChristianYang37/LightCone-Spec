"""Immutable tuning grid and independent confirmation scheduling."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass

from lightcone_spec.config.schema import RunConfig

# DFlash trains block-16 drafters with
#   w_k = exp(-(k - 1) / gamma), gamma = 7.
# The runtime loss uses ``decay ** arange(depth)``, so this is the exact base
# that reproduces the checkpoint's position weighting.  Keep it protocol-bound
# rather than exposing an unsupported result-derived tuning dimension.
DFLASH_BLOCK_SIZE = 16
DFLASH_LOSS_DECAY_GAMMA = 7.0
DFLASH_LOSS_POSITION_DECAY = math.exp(-1.0 / DFLASH_LOSS_DECAY_GAMMA)

TUNING_STAGES = (
    (2, 4096),
    (4, 8192),
    (8, 16384),
    (16, 40960),
)


def tuning_stage(stage: int) -> tuple[int, int]:
    if stage not in range(len(TUNING_STAGES)):
        raise ValueError("tuning stage must be in [0, 4)")
    return TUNING_STAGES[stage]


@dataclass(frozen=True)
class TuningCandidate:
    weight_update_mode: str
    parameter_scope: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    rank: int | None
    stride: int
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.999
    momentum: float | None = None
    muon_ns_steps: int | None = None
    muon_auxiliary_learning_rate: float | None = None
    muon_auxiliary_weight_decay: float | None = None

    @property
    def candidate_id(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(body).hexdigest()


def tuning_candidates() -> tuple[TuningCandidate, ...]:
    candidates: list[TuningCandidate] = []
    for stride in (1, 5, 10, 20, 40, 80):
        for optimizer, weight_decay, momentum in (
            ("adam", 0.0, None),
            ("adamw", 0.0, None),
            ("adamw", 0.01, None),
            ("sgdm", 0.0, 0.9),
            ("nag", 0.0, 0.9),
            ("lion", 0.01, None),
            ("muon", 0.01, 0.95),
        ):
            beta2 = 0.99 if optimizer == "lion" else 0.999
            lora_learning_rates = {
                "sgdm": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
                "nag": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
                "lion": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
                "muon": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
            }.get(optimizer, (1e-5, 3e-5, 1e-4, 3e-4, 1e-3))
            full_learning_rates = {
                "sgdm": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
                "nag": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
                "lion": (1e-8, 3e-8, 1e-7, 3e-7, 1e-6),
                "muon": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
            }.get(optimizer, (1e-7, 3e-7, 1e-6, 3e-6, 1e-5))
            for rank in (4, 8, 16, 32):
                for learning_rate in lora_learning_rates:
                    candidates.append(
                        TuningCandidate(
                            "lora",
                            "drafter",
                            optimizer,
                            learning_rate,
                            weight_decay,
                            rank,
                            stride,
                            beta2=beta2,
                            momentum=momentum,
                            muon_ns_steps=(5 if optimizer == "muon" else None),
                            muon_auxiliary_learning_rate=(
                                1e-5 if optimizer == "muon" else None
                            ),
                            muon_auxiliary_weight_decay=(
                                0.01 if optimizer == "muon" else None
                            ),
                        )
                    )
            for learning_rate in full_learning_rates:
                candidates.append(
                    TuningCandidate(
                        "full",
                        "drafter",
                        optimizer,
                        learning_rate,
                        weight_decay,
                        None,
                        stride,
                        beta2=beta2,
                        momentum=momentum,
                        muon_ns_steps=(5 if optimizer == "muon" else None),
                        muon_auxiliary_learning_rate=(
                            1e-5 if optimizer == "muon" else None
                        ),
                        muon_auxiliary_weight_decay=(
                            0.01 if optimizer == "muon" else None
                        ),
                    )
                )
    identities = [candidate.candidate_id for candidate in candidates]
    if len(identities) != len(set(identities)):
        raise AssertionError("tuning grid contains duplicate identities")
    return tuple(candidates)


def successive_halving(
    candidate_ids: tuple[str, ...],
    scores: dict[str, float],
    *,
    keep_fraction: float = 0.25,
) -> tuple[str, ...]:
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    if set(candidate_ids) - scores.keys():
        raise ValueError("every candidate needs a tuning-only score")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate identities must be unique")
    if any(not math.isfinite(float(scores[key])) for key in candidate_ids):
        raise ValueError("candidate scores must be finite")
    keep = max(1, int(len(candidate_ids) * keep_fraction))
    return tuple(sorted(candidate_ids, key=lambda key: (-scores[key], key))[:keep])


@dataclass(frozen=True)
class ConfirmationBlock:
    block: int
    method_order: tuple[str, ...]
    reset_cohort_before_each: bool = True


def paired_blocks(
    seed: int,
    methods: tuple[str, ...],
    count: int = 8,
) -> tuple[ConfirmationBlock, ...]:
    if count < 1:
        raise ValueError("at least one confirmation block is required")
    if len(methods) < 2 or len(set(methods)) != len(methods):
        raise ValueError("paired blocks require unique comparison methods")
    randomizer = random.Random(seed)
    blocks: list[ConfirmationBlock] = []
    for block in range(count):
        order = list(methods)
        randomizer.shuffle(order)
        blocks.append(
            ConfirmationBlock(
                block=block,
                method_order=tuple(order),
            )
        )
    return tuple(blocks)


def confirmation_blocks(seed: int, count: int = 8) -> tuple[ConfirmationBlock, ...]:
    return paired_blocks(seed, ("static", "tts", "naive_async"), count)


def onlinespec_blocks(seed: int, count: int = 8) -> tuple[ConfirmationBlock, ...]:
    return paired_blocks(
        seed,
        (
            "static",
            "onlinespec_ogd",
            "onlinespec_opt",
            "onlinespec_ens",
        ),
        count,
    )


def select_static_load(
    rows: list[dict],
    *,
    required_context_limit: int,
) -> int:
    """Select maximum-goodput safe concurrency from the fixed scan."""
    if required_context_limit < 1:
        raise ValueError("required context limit must be positive")
    expected = {1, 2, 4, 8, 16, 32, 48}
    observed = {int(row["concurrency"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("static load screen must cover the complete grid")
    for row in rows:
        goodput = float(row["decode_goodput_tps"])
        p99 = float(row["itl_p99_ms"])
        oom = int(row["oom_events"])
        retractions = int(row["retractions"])
        capacity = int(row["kv_token_capacity"])
        if not math.isfinite(goodput) or goodput <= 0:
            raise ValueError("static goodput must be finite and positive")
        if not math.isfinite(p99) or p99 < 0:
            raise ValueError("static p99 ITL must be finite and non-negative")
        if oom < 0 or retractions < 0 or capacity < 1:
            raise ValueError("static safety counters cannot be negative")
    c1 = next(row for row in rows if int(row["concurrency"]) == 1)
    threshold = 2.0 * float(c1["itl_p99_ms"])
    eligible = [
        row
        for row in rows
        if int(row["oom_events"]) == 0
        and int(row["retractions"]) == 0
        and int(row["kv_token_capacity"])
        >= int(row["concurrency"]) * required_context_limit
        and float(row["itl_p99_ms"]) <= threshold
    ]
    if not eligible:
        raise ValueError("no concurrency satisfies the static safety envelope")
    winner = max(
        eligible,
        key=lambda row: (
            float(row["decode_goodput_tps"]),
            -int(row["concurrency"]),
        ),
    )
    return int(winner["concurrency"])


def assert_matched_confirmation_configs(
    configs: dict[str, RunConfig],
    *,
    selected_candidate: TuningCandidate,
    selected_concurrency: int,
) -> None:
    """Ensure TTS/L0 differ only in publication policy and match selection."""
    if set(configs) != {"static", "tts", "naive_async"}:
        raise ValueError("confirmation requires exactly three method configs")
    for method, config in configs.items():
        assert_confirmation_slice_config(
            config,
            method=method,
            selected_candidate=selected_candidate,
            selected_concurrency=selected_concurrency,
        )
    reference_model = configs["static"].model
    reference_runtime = configs["static"].runtime
    for config in configs.values():
        if config.model != reference_model:
            raise ValueError("formal methods must use one immutable model pair")
        runtime = config.runtime.model_dump()
        expected_runtime = reference_runtime.model_dump()
        if runtime != expected_runtime:
            raise ValueError("formal methods must use one runtime configuration")
    tts = configs["tts"].adaptation
    asynchronous = configs["naive_async"].adaptation
    if tts is None or asynchronous is None or tts != asynchronous:
        raise ValueError("TTS and L0 must use an identical adaptation config")
    selected = _candidate_fields(selected_candidate)
    actual = _adaptation_fields(tts)
    if actual != selected:
        raise ValueError("confirmation config does not match the tuning winner")


def _candidate_fields(candidate: TuningCandidate) -> dict[str, object]:
    return {
        "weight_update_mode": candidate.weight_update_mode,
        "parameter_scope": candidate.parameter_scope,
        "optimizer": candidate.optimizer,
        "learning_rate": candidate.learning_rate,
        "weight_decay": candidate.weight_decay,
        "rank": candidate.rank,
        "stride": candidate.stride,
        "grad_clip": candidate.grad_clip,
        "beta1": candidate.beta1,
        "beta2": candidate.beta2,
        "momentum": candidate.momentum,
        "muon_ns_steps": candidate.muon_ns_steps,
        "muon_auxiliary_learning_rate": (candidate.muon_auxiliary_learning_rate),
        "muon_auxiliary_weight_decay": (candidate.muon_auxiliary_weight_decay),
    }


def _adaptation_fields(adaptation: object) -> dict[str, object]:
    return {
        "weight_update_mode": adaptation.weight_update_mode,
        "parameter_scope": adaptation.parameter_scope,
        "optimizer": adaptation.optimizer.name,
        "learning_rate": adaptation.optimizer.learning_rate,
        "weight_decay": adaptation.optimizer.weight_decay,
        "rank": adaptation.rank,
        "stride": adaptation.stride,
        "grad_clip": adaptation.optimizer.grad_clip,
        "beta1": adaptation.optimizer.beta1,
        "beta2": adaptation.optimizer.beta2,
        "momentum": adaptation.optimizer.momentum,
        "muon_ns_steps": adaptation.optimizer.muon_ns_steps,
        "muon_auxiliary_learning_rate": (
            adaptation.optimizer.muon_auxiliary_learning_rate
        ),
        "muon_auxiliary_weight_decay": (
            adaptation.optimizer.muon_auxiliary_weight_decay
        ),
    }


def assert_confirmation_slice_config(
    config: RunConfig,
    *,
    method: str,
    selected_candidate: TuningCandidate,
    selected_concurrency: int,
) -> None:
    """Validate one exclusive-device slice before launching a measurement."""
    if method not in {"static", "tts", "naive_async"}:
        raise ValueError("unknown formal confirmation method")
    if config.method != method:
        raise ValueError("method config is bound to the wrong endpoint")
    if config.runtime.max_running_requests < selected_concurrency:
        raise ValueError("selected load exceeds an endpoint admission limit")
    if method == "static":
        if config.adaptation is not None:
            raise ValueError("Static must not carry adaptation state")
        return
    adaptation = config.adaptation
    if adaptation is None:
        raise ValueError("adapted slice lacks adaptation configuration")
    if _adaptation_fields(adaptation) != _candidate_fields(selected_candidate):
        raise ValueError("confirmation config does not match the tuning winner")
