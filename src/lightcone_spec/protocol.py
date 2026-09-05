"""Plain Python representation of the paper-v2 experiment DAG."""

from __future__ import annotations

import itertools
import math
import random
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

PAPER_NODES = (
    "preflight",
    "E3a",
    "TTS-Cal",
    "E1",
    "E2-r0",
    "E2-r1",
    "E2-r2",
    "E2-r3",
    "E4-screen",
    "E4-local",
    "E4-profile",
    "E3b-pilot",
    "E3b-final",
    "E1a",
    "E5-pilot",
    "E5-final",
    "E6-pilot",
    "E6-final",
    "E0-tune",
    "E0-pilot",
    "E0-final",
)

CONTEXTS = (1024, 2048, 4096, 8192, 16384, 24576, 32768, 40928)
LONG_CONTEXTS = (4096, 16384, 32768, 40928)
CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)
WIDTHS = (4, 8, 16)
ROLES = ("target_only", "static", "tts", "l0_naive", "lightcone")
CORE_ROLES = ("target_only", "static", "tts", "lightcone")
E5_PRIMARY_METHODS = ("target_only", "static", "lightcone", "tts")
E5_SECONDARY_METHODS = ("static", "lightcone", "tts_lora_batched")
SCOPES = ("last1", "last3", "last5", "all")
RANKS = (1, 2, 4, 8, 16, 32, 64)
OPTIMIZERS = ("adam", "adamw", "sgdm", "nag", "muon", "lion", "chronobelief")
SCHEDULES = ("constant", "inverse_sqrt_published_update", "cosine_to_zero")
LEARNING_RATES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
TTS_LEARNING_RATES = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
TTS_STRIDES = (1, 5, 10, 15, 20, 30, 40, 50)
FORMAL_ADAPTATION_STRIDE = 10
FORMAL_ADAPTIVE_METHODS = {
    "tts",
    "tts_lora_batched",
    "l0_naive",
    "lightcone",
    "lightcone_candidate",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
}
EXPLORATORY_STRIDE_WORKLOADS = {
    "tts_calibration_screen",
    "tts_calibration_confirmation",
    "systems_screen",
    "systems_local_factorial",
}
DSPARK_CONFIDENCE_LOSS_WEIGHT = 1.0
DSPARK_CONFIDENCE_POSITIONS = 7
DSPARK_CONFIDENCE_THRESHOLDS = tuple(round(value / 10, 1) for value in range(10))
GENERATION_CHECKPOINTS = (1024, 2048, 4096, 8192, 16384, 24576, 32768)
TTS_GENERATION_TOKENS = 4096
GEOMETRY_GENERATION_TOKENS = 8192
E2_GENERATION_TOKENS = (2048, 4096, 8192, 16384)
E6_GENERATION_TOKENS = 16384
MAX_E2_GEOMETRIES = 4
PILOT_BLOCKS = (0, 1, 2, 3)
PRIMARY_BLOCKS = tuple(range(12))
SECONDARY_BLOCKS = tuple(range(6))
E1_REFERENCE_LOAD = "c1"
E4_LOADS = ("low", "moderate", "saturation")
E4_TRAFFIC = ("pure_decode", "mixed_prefill_decode")
E5_CONCURRENCY = (1, 2, 4, 8, 16, 32, 64, 128, 256)
E5_WARMUP_SECONDS = 10
E5_HEADLINE_SECONDS = 60
E5_REQUEST_DEADLINE_SECONDS = 120
E5_DRAIN_SECONDS = 180
E6_MODELS = ("Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.5-122B-A10B-FP8")
E0_MODELS = ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Gemma4-12B")
E0_BACKENDS = ("EAGLE3", "DFLASH", "DSPARK")
E0_TASKS = (
    "GSM8K",
    "MATH-500",
    "AIME-2025",
    "MBPP",
    "HumanEval",
    "LiveCodeBench",
    "MT-Bench",
    "AlpacaEval",
    "Arena-Hard",
)
TTS_SOURCE_TASKS = (
    "AIME-2024",
    "AIME-2025",
    "MATH-500",
    "OlympiadBench-Math",
    "OlympiadBench-Physics",
    "GPQA-Diamond",
    "TheoremQA",
    "LiveCodeBench",
)
SOURCE_PAIRED_SEEDS = (980406, 980407, 980408, 980409)
SOURCE_EVALUATION_TASKS = (
    ("GSM8K", "gsm8k", 500), ("MATH-500", "math500", 500),
    ("AIME-2025", "aime25", 30), ("HumanEval", "humaneval", 164),
    ("MBPP", "mbpp", 256), ("LiveCodeBench", "livecodebench", 500),
    ("MT-Bench", "mt-bench", 80), ("AlpacaEval", "alpaca", 500),
    ("Arena-Hard", "arena-hard-v2", 500),
)
SOURCE_METHODS = ("static", "tts", "lightcone")
SOURCE_COVERAGE_NODE = "E0-source-four-block-v1"
MECHANISM_NODE = "E3b-mechanism-four-block-v1"


def source_checkpoint_id(model: str, backend: str) -> str:
    """DeepSpec source weights are separate from the main DFlash b16 recipe."""
    model_name = {
        "Qwen/Qwen3-4B": "qwen3_4b", "Qwen/Qwen3-8B": "qwen3_8b",
        "Qwen/Qwen3-14B": "qwen3_14b", "Gemma4-12B": "gemma4_12b",
    }[model]
    if backend not in E0_BACKENDS:
        raise ValueError(f"unsupported source backend {backend}")
    suffix = "ttt7" if backend == "EAGLE3" else "block7"
    return f"deepseek-ai/{backend.lower()}_{model_name}_{suffix}"


def source_coverage_jobs() -> tuple[Job, ...]:
    """Immutable extra evidence; do not rewrite already materialized E0 jobs."""
    rows = []
    for model, backend in itertools.product(E0_MODELS, E0_BACKENDS):
        for block, seed in enumerate(SOURCE_PAIRED_SEEDS):
            for task, dataset, count in SOURCE_EVALUATION_TASKS:
                methods = list(SOURCE_METHODS)
                random.Random(f"source-v1|{model}|{backend}|{task}|{seed}").shuffle(methods)
                for method in methods:
                    dense = model == "Qwen/Qwen3-14B"
                    rows.append(dict(
                        model=model, backend=backend, method=method, task=task,
                        block=block, context=40960, load="c1", width=8,
                        gpu_count=2 if dense else 1,
                        topology="tp2_dp1" if dense else "tp1_dp1",
                        evidence_owner="E6" if dense else "E0",
                        panel="dense_14b_source_transfer" if dense else "source_transfer",
                        workload="dspark_complete_source_four_block",
                        dataset_key=f"DeepSpec-source|{dataset}",
                        source_dataset=dataset, source_max_samples=count,
                        source_checkpoint=source_checkpoint_id(model, backend),
                        checkpoint_family="deepspec_next_token_v1",
                        draft_key=f"DeepSpec-source|{model}|{backend}",
                        execution_request_count=count, sampling_seed=seed,
                        generation_tokens=2048, regime="source_native_prompt",
                        temperature=1.0, enable_thinking=False, respect_eos=True,
                        verification="fixed_budget", proposal_budget=8,
                        draft_positions=7, target_bonus_tokens=1, drafting="chain",
                        confidence_threshold=0.0, stride=FORMAL_ADAPTATION_STRIDE,
                        clean_server_per_cell=True, requires_isolation=True,
                        statistical_unit="independent_clean_server_paired_block",
                        pairing_key=f"source-v1|{model}|{backend}|{task}",
                    ))
    return _jobs(SOURCE_COVERAGE_NODE, rows)


def mechanism_jobs() -> tuple[Job, ...]:
    rows = []
    tasks = ("AIME-2025", "MATH-500", "OlympiadBench-Math", "LiveCodeBench")
    for task, block in itertools.product(tasks, range(4)):
        methods = list(SOURCE_METHODS)
        random.Random(f"mechanism-v1|{task}|{block}").shuffle(methods)
        for method in methods:
            rows.append(dict(
                model="Qwen/Qwen3-8B", backend="DFLASH", method=method,
                task=task, block=block, context=40960, load="c1", width=16,
                execution_request_count=30 if task == "AIME-2025" else 32,
                generation_tokens=32768, sampling_seed=block,
                stimulus_selection_seed=0, regime="mechanism_native_prompt",
                temperature=1.0, respect_eos=True, enable_thinking=False,
                workload="long_generation_mechanism_four_block",
                mechanism_telemetry=True, generation_bin_tokens=2048,
                clean_server_per_cell=True, requires_isolation=True,
                exclude_from_headline_performance=True,
                stride=FORMAL_ADAPTATION_STRIDE,
                statistical_unit="independent_clean_server_paired_block",
                pairing_key=f"mechanism-v1|{task}",
            ))
    return _jobs(MECHANISM_NODE, rows)


E0_ONLINESPEC_METHODS = (
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
)
E0_ONLINESPEC_RECIPES: dict[str, dict[str, Any]] = {
    "onlinespec_ogd": {
        "parameterization": "full",
        "scope": "all",
        "rank": None,
        "learning_rate": 3e-5,
        "stride": FORMAL_ADAPTATION_STRIDE,
        "grad_clip": 1.0,
        "source_backend": "EAGLE",
        "source_chunk_size": 40,
        "source_epochs": 5,
        "source_transfer": "qwen3_dflash",
    },
    "onlinespec_opt": {
        "parameterization": "full",
        "scope": "all",
        "rank": None,
        "learning_rate": 1e-1,
        "stride": FORMAL_ADAPTATION_STRIDE,
        "grad_clip": 1.0,
        "hint_momentum": 0.9,
        "source_backend": "Hydra",
        "source_chunk_size": 80,
        "source_epochs": 3,
        "source_transfer": "qwen3_dflash",
    },
    "onlinespec_ens": {
        "parameterization": "full",
        "scope": "all",
        "rank": None,
        "learning_rate": 3e-5,
        "stride": FORMAL_ADAPTATION_STRIDE,
        "grad_clip": 1.0,
        "additional_learning_rates": (6e-5, 1.2e-4),
        "hedge_learning_rate": 1.0,
        "source_backend": "EAGLE",
        "source_chunk_size": 40,
        "source_epochs": 5,
        "source_transfer": "qwen3_dflash",
    },
}


@dataclass(frozen=True)
class Job:
    job_id: str
    node: str
    ordinal: int
    method: str
    model: str
    backend: str
    task: str
    context: int | None = None
    load: str | None = None
    width: int | None = None
    block: int | None = None
    gpu_count: int = 1
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NodePlan:
    name: str
    rows: str
    gpu_count: int
    description: str


def uses_formal_adaptation_stride(job: Job) -> bool:
    """Return whether a job belongs to the frozen S=10 scientific protocol.

    TTS-Cal and the E4 systems factorial deliberately retain their registered
    stride sweeps as exploratory evidence.  Every other adaptive paper job is
    evaluated at the common formal stride.
    """

    return (
        job.method in FORMAL_ADAPTIVE_METHODS
        and job.parameters.get("workload") not in EXPLORATORY_STRIDE_WORKLOADS
    )


def _slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")[:48] or "cell"


def _jobs(node: str, rows: Iterable[dict[str, Any]]) -> tuple[Job, ...]:
    result = []
    for ordinal, source in enumerate(rows):
        row = dict(source)
        label = "__".join(
            _slug(row.get(name))
            for name in ("method", "model", "backend", "task", "block", "_job_label")
            if row.get(name) is not None
        )
        row.pop("_job_label", None)
        result.append(
            Job(
                job_id=f"{node}__{ordinal:06d}__{label}",
                node=node,
                ordinal=ordinal,
                method=str(row.pop("method")),
                model=str(row.pop("model")),
                backend=str(row.pop("backend")),
                task=str(row.pop("task")),
                context=row.pop("context", None),
                load=row.pop("load", None),
                width=row.pop("width", None),
                block=row.pop("block", None),
                gpu_count=int(row.pop("gpu_count", 1)),
                parameters=row,
            )
        )
    return tuple(result)


def _segments(*rows: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def paper_plan(
    *, valid_e0: int | None = None, e1_safe: int = MAX_E2_GEOMETRIES
) -> tuple[NodePlan, ...]:
    v_text = str(valid_e0) if valid_e0 is not None else "V (0-12 pairs)"
    e2 = [105 * e1_safe]
    for _ in range(3):
        e2.append(max(math.ceil(e2[-1] / 4), 21))
    return (
        NodePlan("preflight", "10", 2, "runtime, implementation smoke, memory, and interference"),
        NodePlan("E3a", "140", 1, "bundled width, context, capacity, and short baseline screen"),
        NodePlan("TTS-Cal", "<=108", 1, "72-cell TTS screen plus finalist confirmation"),
        NodePlan("E1", "<=100", 1, "complete geometry screen plus Pareto confirmation"),
        *tuple(
            NodePlan(f"E2-r{i}", str(rows + 4), 1, "successive halving plus four fixed roles")
            for i, rows in enumerate(e2)
        ),
        NodePlan("E4-screen", "52", 1, "systems screen and TTS update-step ablation"),
        NodePlan("E4-local", "168", 1, "local factorial and six-block cumulative ablation"),
        NodePlan("E4-profile", "3", 2, "isolated profilers"),
        NodePlan("E3b-pilot", "20", 1, "excluded bundled trajectory pilots"),
        NodePlan("E3b-final", "132", 1, "12-block primary and six-block secondary confirmation"),
        NodePlan("E1a", "3", 1, "DSpark source capture, latency, and native-scheduler validation"),
        NodePlan(
            "E5-pilot",
            "19",
            2,
            "serving curve pilot plus DFlash/DSpark TP2 and DP2 transfer",
        ),
        NodePlan(
            "E5-final",
            "114",
            2,
            "12-block TP1 primary and six-block dual-GPU secondary transfer",
        ),
        NodePlan("E6-pilot", "22", 2, "interface, fit, and bundled pilots"),
        NodePlan("E6-final", "60", 2, "six-block native-MTP transfer"),
        NodePlan("E0-tune", "54", 2, "compatibility and frozen OnlineSPEC validation"),
        NodePlan("E0-pilot", "88", 2, f"two-block bundled breadth pilot; {v_text}"),
        NodePlan("E0-final", "264", 2, "six-block bundled cross-workload transfer"),
    )


def _parameter_geometries() -> tuple[dict[str, Any], ...]:
    rows = []
    for scope in SCOPES:
        rows.append({"scope": scope, "parameterization": "full", "rank": None})
        rows.extend({"scope": scope, "parameterization": "lora", "rank": rank} for rank in RANKS)
    return tuple(rows)


def _preflight() -> Iterator[dict[str, Any]]:
    yield dict(
        method="target_only",
        model="Qwen/Qwen3-8B",
        backend="NONE",
        task="controlled_baseline",
        gpu_count=2,
        topology="tp2_dp1",
        preflight_kind="runtime_load",
        controlled_pair_baseline=True,
        deterministic_exactness=True,
    )
    yield dict(
        method="l0_naive",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="controlled_baseline",
        gpu_count=2,
        topology="tp2_dp1",
        controlled_replay=True,
        preflight_kind="implementation_smoke",
        deterministic_verify=True,
    )
    for block, mode, gpu in itertools.product(range(2), ("isolated", "concurrent"), range(2)):
        long_context = block == 1
        yield dict(
            method="static",
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="MATH-500" if long_context else "controlled_baseline",
            context=40928 if long_context else 4096,
            block=block,
            mode=mode,
            gpu_index=gpu,
            load="c1",
            regime="multi_turn_shared_prefix" if long_context else "long_input_short_output",
            workload="interference",
        )


def _e3a() -> Iterator[dict[str, Any]]:
    for context, concurrency in itertools.product(LONG_CONTEXTS, CONCURRENCY):
        common = _segments(
            {"task": "MATH-500", "context": context, "regime": "long_input_short_output"},
            {
                "task": "controlled_baseline",
                "context": context,
                "regime": "multi_turn_shared_prefix",
            },
        )
        yield dict(
            method="target_only",
            model="Qwen/Qwen3-8B",
            backend="NONE",
            task="MATH-500",
            context=context,
            load=f"c{concurrency}",
            segments=common,
            workload="e3a_context_capacity",
        )
        for width in WIDTHS:
            yield dict(
                method="static",
                model="Qwen/Qwen3-8B",
                backend="DFLASH",
                task="MATH-500",
                context=context,
                load=f"c{concurrency}",
                width=width,
                segments=common,
                workload="e3a_context_capacity",
            )
    for concurrency in CONCURRENCY:
        segment = {
            "task": "LiveCodeBench",
            "context": 40928,
            "regime": "short_input_long_generation",
            "generation_tokens": 4096,
            "generation_checkpoints": (1024, 2048, 4096),
        }
        yield dict(
            method="target_only",
            model="Qwen/Qwen3-8B",
            backend="NONE",
            task="LiveCodeBench",
            context=40928,
            load=f"c{concurrency}",
            segments=_segments(segment),
            workload="e3a_short_baseline",
        )
        for width in WIDTHS:
            yield dict(
                method="static",
                model="Qwen/Qwen3-8B",
                backend="DFLASH",
                task="LiveCodeBench",
                context=40928,
                load=f"c{concurrency}",
                width=width,
                segments=_segments(segment),
                workload="e3a_short_baseline",
            )


def _tts_cal() -> Iterator[dict[str, Any]]:
    for lr, stride in itertools.product(TTS_LEARNING_RATES, TTS_STRIDES):
        yield dict(
            method="tts",
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="CalibrationMix",
            context=40928,
            width=16,
            block=0,
            learning_rate=lr,
            stride=stride,
            optimizer="adam",
            parameterization="full",
            scope="all",
            regime="short_input_long_generation",
            generation_tokens=TTS_GENERATION_TOKENS,
            workload="tts_calibration_screen",
        )


def _e1() -> Iterator[dict[str, Any]]:
    for role in ("target_only", "static", "tts", "l0_naive"):
        yield dict(
            method=role,
            model="Qwen/Qwen3-8B",
            backend="NONE" if role == "target_only" else "DFLASH",
            task="CalibrationMix",
            context=40928,
            width=16,
            fixed_role=True,
            regime="short_input_long_generation",
            generation_tokens=GEOMETRY_GENERATION_TOKENS,
        )
    for geometry, optimizer in itertools.product(_parameter_geometries(), ("adamw", "sgdm")):
        yield dict(
            method="lightcone_candidate",
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="CalibrationMix",
            context=40928,
            width=16,
            optimizer=optimizer,
            regime="short_input_long_generation",
            generation_tokens=GEOMETRY_GENERATION_TOKENS,
            **geometry,
        )


def e2_candidates(geometries: Iterable[dict[str, Any]] | None = None) -> tuple[dict[str, Any], ...]:
    selected = tuple(geometries) if geometries is not None else _parameter_geometries()
    return tuple(
        {**geometry, "optimizer": optimizer, "learning_rate": lr, "schedule": schedule}
        for geometry, optimizer, lr, schedule in itertools.product(
            selected, OPTIMIZERS, LEARNING_RATES, SCHEDULES
        )
    )


def _e2(round_index: int, candidates: Iterable[dict[str, Any]] | None) -> Iterator[dict[str, Any]]:
    limits = (420, 105, 27, 21)
    rows = (
        list(candidates)
        if candidates is not None
        else list(e2_candidates(_parameter_geometries()[:MAX_E2_GEOMETRIES]))[: limits[round_index]]
    )
    for candidate in rows:
        recipe = {
            key: candidate[key]
            for key in (
                "parameterization",
                "rank",
                "scope",
                "optimizer",
                "learning_rate",
                "schedule",
            )
            if key in candidate
        }
        yield dict(
            method="lightcone_candidate",
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="CalibrationMix",
            context=(4096, 8192, 16384, 40928)[round_index],
            load=f"c{(2, 4, 8, 16)[round_index]}",
            width=16,
            round=round_index,
            minimum_updates=(2, 4, 8, 16)[round_index],
            regime="short_input_long_generation",
            generation_tokens=E2_GENERATION_TOKENS[round_index],
            **recipe,
        )
    for role in ("target_only", "static", "tts", "l0_naive"):
        yield dict(
            method=role,
            model="Qwen/Qwen3-8B",
            backend="NONE" if role == "target_only" else "DFLASH",
            task="CalibrationMix",
            context=(4096, 8192, 16384, 40928)[round_index],
            load=f"c{(2, 4, 8, 16)[round_index]}",
            width=16,
            fixed_role=True,
            regime="short_input_long_generation",
            generation_tokens=E2_GENERATION_TOKENS[round_index],
        )


E4_SCREEN_LEVELS = {
    "stride": (1, 50),
    "microbatch": (1, 8),
    "coalescing": (1, 8),
    "stream_priority": ("default", "high"),
}


def _e4_screen() -> Iterator[dict[str, Any]]:
    for row, (a, b, c) in enumerate(itertools.product((0, 1), repeat=3)):
        factors = {
            "stride": E4_SCREEN_LEVELS["stride"][a],
            "microbatch": E4_SCREEN_LEVELS["microbatch"][b],
            "coalescing": E4_SCREEN_LEVELS["coalescing"][c],
            "stream_priority": E4_SCREEN_LEVELS["stream_priority"][a ^ b ^ c],
        }
        for load, traffic in itertools.product(E4_LOADS, E4_TRAFFIC):
            yield dict(
                method="lightcone",
                model="Qwen/Qwen3-8B",
                backend="DFLASH",
                task="controlled_baseline",
                context=40928,
                width=16,
                load=load,
                screen_row=row,
                traffic=traffic,
                workload="systems_screen",
                **factors,
            )
    for steps in (1, 2, 4, 8):
        yield dict(
            method="tts",
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="LiveCodeBench",
            context=32768,
            width=16,
            load="c1",
            update_steps=steps,
            workload="tts_update_steps",
        )


def _e4_local(neighborhoods: dict[str, tuple[object, object]] | None) -> Iterator[dict[str, Any]]:
    values = neighborhoods or {
        "stride": (1, 5),
        "microbatch": (1, 2),
        "coalescing": (1, 2),
        "stream_priority": ("default", "high"),
    }
    names = tuple(values)
    for row, levels in enumerate(itertools.product(*(values[name] for name in names))):
        for load, traffic in itertools.product(E4_LOADS, E4_TRAFFIC):
            yield dict(
                method="lightcone",
                model="Qwen/Qwen3-8B",
                backend="DFLASH",
                task="controlled_baseline",
                context=40928,
                width=16,
                load=load,
                local_row=row,
                traffic=traffic,
                workload="systems_local_factorial",
                **dict(zip(names, levels, strict=True)),
            )
    variants = ("base", "side_stream", "staging", "publication", "coalescing", "full")
    for block, variant, traffic in itertools.product(SECONDARY_BLOCKS, variants, E4_TRAFFIC):
        yield dict(
            method="lightcone",
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="controlled_baseline",
            context=40928,
            width=16,
            load="moderate",
            block=block,
            traffic=traffic,
            cumulative_variant=variant,
            workload="systems_cumulative_ablation",
        )


def _e4_profiles() -> Iterator[dict[str, Any]]:
    for profiler in ("nvtx", "nsys", "ncu"):
        yield dict(
            method="lightcone",
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="controlled_baseline",
            context=40928,
            width=16,
            gpu_count=2,
            profiler=profiler,
            workload="isolated_profile",
        )


def _e3b_pilot() -> Iterator[dict[str, Any]]:
    segments = _segments(
        *(
            {
                "task": "LiveCodeBench",
                "context": context,
                "regime": "short_input_long_generation"
                if context == 32768
                else "long_input_short_output",
                "generation_tokens": 32768 if context == 32768 else 256,
                "generation_checkpoints": GENERATION_CHECKPOINTS if context == 32768 else (),
            }
            for context in (4096, 16384, 32768, 40928)
        )
    )
    for block, role in itertools.product(PILOT_BLOCKS, ROLES):
        yield dict(
            method=role,
            model="Qwen/Qwen3-8B",
            backend="NONE" if role == "target_only" else "DFLASH",
            comparison_backend="DFLASH",
            task="LiveCodeBench",
            context=32768,
            load="c1",
            block=block,
            segments=segments,
            workload="excluded_trajectory_pilot",
        )


def _e3b_final() -> Iterator[dict[str, Any]]:
    primary = _segments(
        {
            "task": "LiveCodeBench",
            "context": 32768,
            "regime": "short_input_long_generation",
            "generation_tokens": 32768,
            "generation_checkpoints": GENERATION_CHECKPOINTS,
        }
    )
    for block, role in itertools.product(PRIMARY_BLOCKS, ROLES):
        yield dict(
            method=role,
            model="Qwen/Qwen3-8B",
            backend="NONE" if role == "target_only" else "DFLASH",
            comparison_backend="DFLASH",
            task="LiveCodeBench",
            context=32768,
            load="c1",
            block=block,
            segments=primary,
            workload="primary_long_history",
        )
    secondary = _segments(
        *(
            {
                "task": task,
                "context": 16384,
                "regime": "short_input_long_generation",
                "generation_tokens": 16384,
                "generation_checkpoints": tuple(
                    x for x in GENERATION_CHECKPOINTS if x <= 16384
                ),
                "source_panel": "tts_long_history",
            }
            for task in ("AIME-2025", "MATH-500", "OlympiadBench-Math", "LiveCodeBench")
        ),
        *(
            {
                "task": task,
                "context": 4096,
                "regime": "short_input_long_generation",
                "generation_tokens": 4096,
                "generation_checkpoints": (1024, 2048, 4096),
                "source_panel": "tts_breadth",
            }
            for task in TTS_SOURCE_TASKS
        ),
    )
    for block, role in itertools.product(SECONDARY_BLOCKS, CORE_ROLES):
        yield dict(
            method=role,
            model="Qwen/Qwen3-8B",
            backend="NONE" if role == "target_only" else "DFLASH",
            comparison_backend="DFLASH",
            task="MATH-500",
            context=16384,
            load="c1",
            block=block,
            segments=secondary,
            workload="secondary_long_history",
        )
    contexts = _segments(
        *(
            {"task": "MATH-500", "context": context, "regime": "long_input_short_output"}
            for context in LONG_CONTEXTS
        )
    )
    turns = _segments(
        *(
            {
                "task": "controlled_baseline",
                "context": context,
                "regime": "multi_turn_shared_prefix",
            }
            for context in LONG_CONTEXTS
        )
    )
    for suite, segments in (("long_input", contexts), ("multi_turn", turns)):
        for block, role in itertools.product(SECONDARY_BLOCKS, CORE_ROLES):
            yield dict(
                method=role,
                model="Qwen/Qwen3-8B",
                backend="NONE" if role == "target_only" else "DFLASH",
                comparison_backend="DFLASH",
                task=str(segments[0]["task"]),
                context=40928,
                load="c1",
                block=block,
                segments=segments,
                workload=f"secondary_{suite}",
            )


def _e1a() -> Iterator[dict[str, Any]]:
    common = {
        "method": "lightcone",
        "model": "Qwen/Qwen3-8B",
        "backend": "DSPARK",
        "task": "CalibrationMix",
        "context": 40928,
        "width": 8,
        "regime": "short_input_long_generation",
        "generation_tokens": GEOMETRY_GENERATION_TOKENS,
        "temperature": 1.0,
        "confidence_loss_weight": DSPARK_CONFIDENCE_LOSS_WEIGHT,
        "source_transfer_recipe": "dflash_lightcone_recipe",
    }
    yield {
        **common,
        "verification": "fixed_budget",
        "workload": "dspark_confidence_capture",
        "segments": _segments(
            *(
                {
                    "domain": domain,
                    "confidence_threshold": 0.0,
                    "save_confidence_outcomes": True,
                    "calibration_split_seed": 0,
                    "calibration_split": "fit",
                    "execution_request_count": 12,
                    "proposal_budget": 8,
                }
                for domain in ("math", "code", "chat")
            )
        ),
    }
    yield {
        **common,
        "verification": "fixed_budget",
        "workload": "dspark_source_latency_panel",
        "segments": _segments(
            *(
                {
                    "proposal_budget": budget,
                    "load": "c128",
                    "batch_size": 128,
                    "context": context,
                    "latency_context": context,
                    "regime": "long_input_short_output",
                    "generation_tokens": 256,
                    "execution_request_count": 128,
                }
                for budget, context in itertools.product(
                    (2, 4, 6, 8), (512, 1024, 2048, 4096)
                )
            )
        ),
    }
    yield {
        **common,
        "verification": "native_scheduler",
        "workload": "dspark_native_scheduler_validation",
        "segments": _segments(
            *(
                {
                    "domain": domain,
                    "confidence_threshold": None,
                    "save_confidence_outcomes": True,
                    "calibration_split": "validation",
                    "calibration_split_seed": 0,
                    "execution_request_count": 12,
                }
                for domain in ("math", "code", "chat")
            )
        ),
    }


def _e5_serving_segments() -> list[dict[str, Any]]:
    return _segments(
        *(
            {"load": f"closed_loop_c{value}", "registered_load": f"closed_loop_c{value}"}
            for value in E5_CONCURRENCY
        ),
        {
            "load": "burstgpt_shape",
            "registered_load": "burstgpt_shape",
            "arrival_trace": "BurstGPT",
        },
    )


def _e5_method_segments(method: str) -> list[dict[str, Any]]:
    if method == "tts":
        return _segments(
            {"load": "closed_loop_c1", "registered_load": "closed_loop_c1"},
            {
                "load": "burstgpt_shape",
                "registered_load": "burstgpt_shape",
                "arrival_trace": "BurstGPT",
            },
        )
    return _e5_serving_segments()


def _e5_topology_pilot_segments() -> list[dict[str, Any]]:
    """Small dual-GPU transfer panel; concurrency is system-wide."""
    return _segments(
        *(
            {
                "load": f"closed_loop_c{value}",
                "registered_load": f"closed_loop_c{value}",
                "registered_concurrency_scope": "system",
            }
            for value in (1, 32, 128)
        ),
        {
            "load": "burstgpt_shape",
            "registered_load": "burstgpt_shape",
            "arrival_trace": "BurstGPT",
            "registered_concurrency_scope": "system",
        },
    )


def _e5_topology_final_segments() -> list[dict[str, Any]]:
    """Confirmatory dual-GPU transfer loads after the pilot."""
    return _segments(
        *(
            {
                "load": f"closed_loop_c{value}",
                "registered_load": f"closed_loop_c{value}",
                "registered_concurrency_scope": "system",
            }
            for value in (32, 128)
        ),
        {
            "load": "burstgpt_shape",
            "registered_load": "burstgpt_shape",
            "arrival_trace": "BurstGPT",
            "registered_concurrency_scope": "system",
        },
    )


def _e5_pilot() -> Iterator[dict[str, Any]]:
    methods = (
        ("NONE", "target_only"),
        ("DFLASH", "static"),
        ("DFLASH", "lightcone"),
        ("DFLASH", "tts"),
        ("DSPARK", "static"),
        ("DSPARK", "lightcone"),
        ("DFLASH", "tts_lora_batched"),
    )
    for backend, role in methods:
        yield dict(
            method=role,
            model="Qwen/Qwen3-8B",
            backend=backend,
            task="LiveCodeBench",
            context=40928,
            load="closed_loop_c1",
            block=0,
            gpu_count=2,
            comparison_backend="DFLASH" if backend == "NONE" else backend,
            segments=_e5_method_segments(role),
            workload="serving_pilot",
        )
    for backend, topology in itertools.product(
        ("DFLASH", "DSPARK"), ("tp2_dp1", "two_replica_tp1_dp2")
    ):
        yield dict(
            method="lightcone",
            model="Qwen/Qwen3-8B",
            backend=backend,
            task="controlled_baseline",
            context=40928,
            load="c1",
            gpu_count=2,
            topology=topology,
            workload="topology_compatibility",
        )
    # Append-only: the first eleven registered parents above retain their
    # historical identities and configurations on resume.
    for backend, role, topology in itertools.product(
        ("DFLASH", "DSPARK"),
        ("static", "lightcone"),
        ("tp2_dp1", "two_replica_tp1_dp2"),
    ):
        yield dict(
            method=role,
            model="Qwen/Qwen3-8B",
            backend=backend,
            task="LiveCodeBench",
            context=40928,
            load="closed_loop_c1",
            block=0,
            gpu_count=2,
            topology=topology,
            registered_concurrency_scope="system",
            comparison_backend=backend,
            segments=_e5_topology_pilot_segments(),
            workload="multigpu_serving_transfer",
            _job_label=f"topology-transfer-{topology}",
        )


def _e5_final() -> Iterator[dict[str, Any]]:
    for block, role in itertools.product(PRIMARY_BLOCKS, E5_PRIMARY_METHODS):
        backend = "NONE" if role == "target_only" else "DFLASH"
        yield dict(
            method=role,
            model="Qwen/Qwen3-8B",
            backend=backend,
            task="LiveCodeBench",
            context=40928,
            load="closed_loop_c1",
            block=block,
            gpu_count=2,
            comparison_backend="DFLASH",
            segments=_e5_method_segments(role),
            workload="primary_serving_frontier",
        )
    for block, role in itertools.product(SECONDARY_BLOCKS, E5_SECONDARY_METHODS):
        backend = "DSPARK" if role != "tts_lora_batched" else "DFLASH"
        yield dict(
            method=role,
            model="Qwen/Qwen3-8B",
            backend=backend,
            task="LiveCodeBench",
            context=40928,
            load="closed_loop_c1",
            block=block,
            gpu_count=2,
            comparison_backend=backend,
            segments=_e5_method_segments(role),
            workload="secondary_serving_frontier",
            )
    # Secondary six-block transfer. These rows never enter the TP1 H3
    # frontier and use system-wide concurrency for both TP2 and DP2.
    for backend, role, topology, block in itertools.product(
        ("DFLASH", "DSPARK"),
        ("static", "lightcone"),
        ("tp2_dp1", "two_replica_tp1_dp2"),
        SECONDARY_BLOCKS,
    ):
        yield dict(
            method=role,
            model="Qwen/Qwen3-8B",
            backend=backend,
            task="LiveCodeBench",
            context=40928,
            load="closed_loop_c32",
            block=block,
            gpu_count=2,
            topology=topology,
            registered_concurrency_scope="system",
            comparison_backend=backend,
            segments=_e5_topology_final_segments(),
            workload="multigpu_serving_transfer",
            _job_label=f"topology-transfer-{topology}",
        )


def _e6_segments() -> list[dict[str, Any]]:
    return _segments(
        *(
            {"task": task, "context": context, "regime": "long_input_short_output"}
            for task, context in itertools.product(
                ("LiveCodeBench", "MATH-500"), (4096, 16384, 32768)
            )
        ),
        {
            "task": "LiveCodeBench",
            "context": 40928,
            "regime": "short_input_long_generation",
            "generation_tokens": E6_GENERATION_TOKENS,
            "generation_checkpoints": tuple(
                x for x in GENERATION_CHECKPOINTS if x <= E6_GENERATION_TOKENS
            ),
        },
    )


def _e6_pilot() -> Iterator[dict[str, Any]]:
    for model in E6_MODELS:
        yield dict(
            method="lightcone",
            model=model,
            backend="NEXTN",
            task="LiveCodeBench",
            load="c1",
            gpu_count=2,
            interface_fit=True,
            minimum_updates=1,
        )
    for model, role in itertools.product(E6_MODELS, ROLES):
        for block in (0, 1):
            yield dict(
                method=role,
                model=model,
                backend="NONE" if role == "target_only" else "NEXTN",
                comparison_backend="NEXTN",
                task="LiveCodeBench",
                context=40928,
                load="common_slo_load",
                block=block,
                gpu_count=2,
                segments=_e6_segments(),
                workload="native_mtp_pilot",
            )


def _e6_final() -> Iterator[dict[str, Any]]:
    for model, role, block in itertools.product(E6_MODELS, ROLES, SECONDARY_BLOCKS):
        yield dict(
            method=role,
            model=model,
            backend="NONE" if role == "target_only" else "NEXTN",
            comparison_backend="NEXTN",
            task="LiveCodeBench",
            context=40928,
            load="common_slo_load",
            block=block,
            gpu_count=2,
            segments=_e6_segments(),
            workload="native_mtp_transfer",
        )


def _e0_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(itertools.product(E0_MODELS, E0_BACKENDS))


def _onlinespec_source_recipes() -> tuple[dict[str, Any], ...]:
    """Return the three frozen source-transfer recipes used by E0.

    The source chunk size and epoch count are provenance only.  They describe
    request-level training in the public OnlineSPEC implementations and are
    deliberately not mapped to our in-process speculation-round stride.
    """

    return tuple(
        {"method": method, **recipe}
        for method, recipe in E0_ONLINESPEC_RECIPES.items()
    )


def _e0_methods(
    valid_pairs: set[tuple[str, str]] | None = None,
    e0_recipes: dict[str, dict[str, Any]] | None = None,
) -> tuple[tuple[str, str, str], ...]:
    pairs = set(_e0_pairs()) if valid_pairs is None else valid_pairs
    models = {model for model, _ in pairs}
    rows = [(model, "NONE", "target_only") for model in E0_MODELS if model in models]
    rows.extend(
        (model, backend, method)
        for model, backend in sorted(pairs)
        for method in ("static", "tts", "lightcone")
    )
    representative = ("Qwen/Qwen3-8B", "DFLASH")
    if representative in pairs:
        rows.append((*representative, "l0_naive"))
        feasible = (
            E0_ONLINESPEC_METHODS
            if e0_recipes is None
            else tuple(
                method
                for method in E0_ONLINESPEC_METHODS
                if f"{representative[0]}|{representative[1]}|{method}" in e0_recipes
            )
        )
        rows.extend((*representative, method) for method in feasible)
    if valid_pairs is None and e0_recipes is None and len(rows) != 44:
        raise AssertionError("E0 bundled method surface must contain 44 rows")
    return tuple(rows)


def _e0_tune() -> Iterator[dict[str, Any]]:
    for model, backend in _e0_pairs():
        yield dict(
            method="static",
            model=model,
            backend=backend,
            task="CalibrationMix",
            probe=True,
            adaptive_probe=True,
            gpu_count=2,
        )
    for candidate in _onlinespec_source_recipes():
        yield dict(
            **candidate,
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="CalibrationMix",
            recipe_validation=True,
            _job_label="source-transfer",
            gpu_count=2,
        )
    for method in ("static", "tts", "l0_naive"):
        yield dict(
            method=method,
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="CalibrationMix",
            source_reference=True,
            _job_label="source-reference",
            gpu_count=2,
        )
    for model, backend, method in itertools.product(
        E0_MODELS, E0_BACKENDS, ("static", "tts", "lightcone")
    ):
        yield dict(
            method=method,
            model=model,
            backend=backend,
            task="CalibrationMix",
            pair_calibration=True,
            _job_label="pair-calibration",
            gpu_count=2,
        )


def _e0_blocks(
    blocks: Iterable[int],
    valid_pairs: set[tuple[str, str]] | None = None,
    e0_recipes: dict[str, dict[str, Any]] | None = None,
    *,
    source_panel: bool = False,
) -> Iterator[dict[str, Any]]:
    for model, backend, method in _e0_methods(valid_pairs, e0_recipes):
        segments = _segments(
            *(
                {
                    "task": task,
                    "context": 40928,
                    "workload": "cross_workload_efficiency",
                }
                for task in E0_TASKS
            )
        )
        if source_panel:
            segments.extend(
                {
                    "task": task,
                    "context": 4096,
                    "workload": "tts_source_panel",
                    "temperature": 1.0,
                }
                for task in TTS_SOURCE_TASKS
            )
            if backend == "DSPARK":
                segments.extend(
                    {
                        "task": task,
                        "context": 4096,
                        "workload": "dspark_source_panel",
                        "temperature": 1.0,
                        "drafting": "chain",
                        "verification": "fixed_budget",
                        "proposal_budget": 8,
                    }
                    for task in E0_TASKS
                )
        for block in blocks:
            yield dict(
                method=method,
                model=model,
                backend=backend,
                comparison_backend=backend if backend != "NONE" else "shared_target",
                task=E0_TASKS[0],
                context=40928,
                load="common_slo_load",
                block=block,
                gpu_count=2,
                segments=segments,
                workload="cross_workload_efficiency",
            )


def materialize(
    node: str,
    *,
    e2_rows: Iterable[dict[str, Any]] | None = None,
    e4_neighborhoods: dict[str, tuple[object, object]] | None = None,
    valid_e0: Iterable[tuple[str, str, str]] | None = None,
    e0_recipes: dict[str, dict[str, Any]] | None = None,
    **_legacy: object,
) -> tuple[Job, ...]:
    if node not in PAPER_NODES:
        raise ValueError(f"unknown paper node {node}")
    if node == "preflight":
        rows = _preflight()
    elif node == "E3a":
        rows = _e3a()
    elif node == "TTS-Cal":
        rows = _tts_cal()
    elif node == "E1":
        rows = _e1()
    elif node.startswith("E2-r"):
        rows = _e2(int(node[-1]), e2_rows)
    elif node == "E4-screen":
        rows = _e4_screen()
    elif node == "E4-local":
        rows = _e4_local(e4_neighborhoods)
    elif node == "E4-profile":
        rows = _e4_profiles()
    elif node == "E3b-pilot":
        rows = _e3b_pilot()
    elif node == "E3b-final":
        rows = _e3b_final()
    elif node == "E1a":
        rows = _e1a()
    elif node == "E5-pilot":
        rows = _e5_pilot()
    elif node == "E5-final":
        rows = _e5_final()
    elif node == "E6-pilot":
        rows = _e6_pilot()
    elif node == "E6-final":
        rows = _e6_final()
    elif node == "E0-tune":
        rows = _e0_tune()
    elif node == "E0-pilot":
        pairs = None if valid_e0 is None else {(model, backend) for model, backend, _ in valid_e0}
        rows = _e0_blocks((0, 1), pairs, e0_recipes, source_panel=True)
    else:
        pairs = None if valid_e0 is None else {(model, backend) for model, backend, _ in valid_e0}
        rows = _e0_blocks(SECONDARY_BLOCKS, pairs, e0_recipes)
    return _jobs(node, rows)


def segment_count(job: Job) -> int:
    segments = job.parameters.get("segments")
    return len(segments) if isinstance(segments, list) else 1


def default_row_counts() -> dict[str, int]:
    return {node: len(materialize(node)) for node in PAPER_NODES}
