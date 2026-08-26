"""Plain Python representation of the paper-v2 experiment DAG."""

from __future__ import annotations

import itertools
import math
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
SCOPES = ("last1", "last3", "last5", "all")
RANKS = (1, 2, 4, 8, 16, 32, 64)
OPTIMIZERS = ("adam", "adamw", "sgdm", "nag", "muon", "lion", "chronobelief")
SCHEDULES = ("constant", "inverse_sqrt_published_update", "cosine_to_zero")
LEARNING_RATES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
TTS_LEARNING_RATES = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
TTS_STRIDES = (1, 5, 10, 15, 20, 30, 40, 50)
CONFIDENCE_WEIGHTS = (0.05, 0.1, 0.25, 0.5, 1.0)
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
E5_LOADS = (0.25, 0.50, 0.75, 0.90, 1.00, 1.10, 1.25)
E5_TRACES = (
    "immediate_burst",
    "burstgpt_shape",
    "moderate_soak",
    "saturation_soak",
    "overload_soak",
)
E5_TOPOLOGIES = ("tp1_dp1", "tp2_dp1", "two_replica_tp1_dp2")
E5_COHORTS = (1, 16, 64)
E5_POPULARITY = ("uniform", "zipf")
E5_FAILURES = (
    "queue_saturation",
    "cancellation",
    "duplicate_retry",
    "nonfinite_candidate",
    "oom_candidate",
    "telemetry_backpressure",
    "disk_quota",
    "slow_rank",
    "communicator_failure",
    "replica_drain",
    "replica_restart",
)
E5_WARMUP_SECONDS = 10
E5_HEADLINE_SECONDS = 60
E5_SOAK_SECONDS = 300
E5_REQUEST_DEADLINE_SECONDS = 120
E5_DRAIN_SECONDS = 180
E5_P99_MIN_COMPLETED = 10_000
E5_P99_EXTENSION_REQUESTS = 11_000
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


def _slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")[:48] or "cell"


def _jobs(node: str, rows: Iterable[dict[str, Any]]) -> tuple[Job, ...]:
    result = []
    for ordinal, source in enumerate(rows):
        row = dict(source)
        label = "__".join(
            _slug(row.get(name))
            for name in ("method", "model", "backend", "task", "block")
            if row.get(name) is not None
        )
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
    v_text = str(valid_e0) if valid_e0 is not None else "V (0-108)"
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
        NodePlan("E4-screen", "48", 1, "systems mechanism screen"),
        NodePlan("E4-local", "168", 1, "local factorial and six-block cumulative ablation"),
        NodePlan("E4-profile", "3", 2, "isolated profilers"),
        NodePlan("E3b-pilot", "20", 1, "excluded bundled trajectory pilots"),
        NodePlan("E3b-final", "132", 1, "12-block primary and six-block secondary confirmation"),
        NodePlan("E1a", "141", 1, "DSpark screen, confidence calibration, and confirmation"),
        NodePlan("E5-pilot", "53", 2, "serving calibration and one-shot fault diagnostics"),
        NodePlan("E5-final", "160", 2, "12-block serving and six-block topology confirmation"),
        NodePlan("E6-pilot", "22", 2, "interface, fit, and bundled pilots"),
        NodePlan("E6-final", "60", 2, "six-block native-MTP transfer"),
        NodePlan("E0-tune", "287", 2, "compatibility and representative OnlineSPEC tuning"),
        NodePlan("E0-pilot", "86", 2, f"two-block bundled breadth pilot; {v_text}"),
        NodePlan("E0-final", "258", 2, "six-block bundled cross-workload transfer"),
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
            key: value
            for key, value in candidate.items()
            if key not in {"round", "fixed_role", "design_index", "probe", "tuning_index"}
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
            load="common_load",
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
            load="common_load",
            block=block,
            segments=primary,
            workload="primary_long_history",
        )
    secondary = _segments(
        {
            "task": "MATH-500",
            "context": 16384,
            "regime": "short_input_long_generation",
            "generation_tokens": 16384,
            "generation_checkpoints": tuple(x for x in GENERATION_CHECKPOINTS if x <= 16384),
        }
    )
    for block, role in itertools.product(SECONDARY_BLOCKS, CORE_ROLES):
        yield dict(
            method=role,
            model="Qwen/Qwen3-8B",
            backend="NONE" if role == "target_only" else "DFLASH",
            comparison_backend="DFLASH",
            task="MATH-500",
            context=16384,
            load="common_load",
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
                load="common_load",
                block=block,
                segments=segments,
                workload=f"secondary_{suite}",
            )


def _e1a() -> Iterator[dict[str, Any]]:
    configs = list(_parameter_geometries())
    for depth in ("last1", "last3", "last5"):
        configs.append({"scope": f"{depth}_native_heads", "parameterization": "full", "rank": None})
        configs.extend(
            {"scope": f"{depth}_native_heads", "parameterization": "lora", "rank": rank}
            for rank in RANKS
        )
    configs.extend(
        (
            {"scope": "none", "parameterization": "none", "rank": None, "baseline": "target_only"},
            {"scope": "none", "parameterization": "none", "rank": None, "baseline": "static"},
        )
    )
    for configuration, verification in itertools.product(
        configs, ("fixed_budget", "native_scheduler")
    ):
        baseline = configuration.get("baseline")
        yield dict(
            method=baseline or "lightcone_candidate",
            model="Qwen/Qwen3-8B",
            backend="NONE" if baseline == "target_only" else "DSPARK",
            task="CalibrationMix",
            context=40928,
            width=16,
            verification=verification,
            regime="short_input_long_generation",
            generation_tokens=GEOMETRY_GENERATION_TOKENS,
            **configuration,
        )
    for weight in CONFIDENCE_WEIGHTS:
        yield dict(
            method="lightcone_candidate",
            model="Qwen/Qwen3-8B",
            backend="DSPARK",
            task="CalibrationMix",
            context=40928,
            width=16,
            confidence_loss_weight=weight,
            verification="native_scheduler",
            workload="confidence_calibration",
        )
    for slot, block in itertools.product(range(4), range(5)):
        yield dict(
            method="lightcone_candidate",
            model="Qwen/Qwen3-8B",
            backend="DSPARK",
            task="CalibrationMix",
            context=40928,
            width=16,
            finalist_slot=slot,
            block=block,
            verification="native_scheduler",
            workload="dspark_finalist_confirmation",
        )


def _e5_serving_segments() -> list[dict[str, Any]]:
    return _segments(
        *(
            {"load": f"closed_loop_c{value}", "registered_load": f"closed_loop_c{value}"}
            for value in E5_CONCURRENCY
        ),
        *({"load": f"lambda_{value}", "registered_load": f"lambda_{value}"} for value in E5_LOADS),
        *({"load": trace, "registered_load": trace} for trace in E5_TRACES),
    )


def _e5_pilot() -> Iterator[dict[str, Any]]:
    for backend, role in itertools.product(("DFLASH", "DSPARK"), CORE_ROLES):
        yield dict(
            method=role,
            model="Qwen/Qwen3-8B",
            backend=backend,
            task="LiveCodeBench",
            context=40928,
            load="closed_loop_c1",
            block=0,
            gpu_count=2,
            segments=_e5_serving_segments(),
            workload="serving_pilot",
        )
    for backend, role in itertools.product(("DFLASH", "DSPARK"), CORE_ROLES):
        for replicate in range(2):
            yield dict(
                method=role,
                model="Qwen/Qwen3-8B",
                backend=backend,
                task="LiveCodeBench",
                context=40928,
                load="saturation",
                block=replicate,
                gpu_count=2,
                segments=_segments(
                    *(
                        {
                            "topology": topology,
                            "cohorts": cohort,
                            "popularity": popularity,
                            "load": "saturation",
                        }
                        for topology, cohort, popularity in itertools.product(
                            E5_TOPOLOGIES, E5_COHORTS, E5_POPULARITY
                        )
                    )
                ),
                workload="topology_pilot",
            )
    for backend, failure in itertools.product(("DFLASH", "DSPARK"), E5_FAILURES):
        yield dict(
            method="lightcone",
            model="Qwen/Qwen3-8B",
            backend=backend,
            task="controlled_baseline",
            context=40928,
            load="c256" if failure == "queue_saturation" else "c1",
            gpu_count=2,
            failure=failure,
            topology="tp2_dp1",
            cohorts=1,
            workload="failure_injection",
        )
    for index in range(7):
        yield dict(
            method="lightcone",
            model="Qwen/Qwen3-8B",
            backend="DFLASH" if index < 4 else "DSPARK",
            task="LiveCodeBench",
            context=40928,
            load="saturation",
            block=index,
            gpu_count=2,
            calibration_slot=index,
            workload="serving_calibration",
        )


def _e5_final() -> Iterator[dict[str, Any]]:
    for backend, block, role in itertools.product(("DFLASH", "DSPARK"), PRIMARY_BLOCKS, CORE_ROLES):
        yield dict(
            method=role,
            model="Qwen/Qwen3-8B",
            backend=backend,
            task="LiveCodeBench",
            context=40928,
            load="closed_loop_c1",
            block=block,
            gpu_count=2,
            segments=_e5_serving_segments(),
            workload="production_crossover",
        )
    topology_segments = _segments(
        *(
            {
                "topology": topology,
                "cohorts": cohort,
                "popularity": popularity,
                "load": "saturation",
            }
            for topology, cohort, popularity in itertools.product(
                E5_TOPOLOGIES, E5_COHORTS, E5_POPULARITY
            )
        )
    )
    for backend, block, role in itertools.product(
        ("DFLASH", "DSPARK"), SECONDARY_BLOCKS, CORE_ROLES
    ):
        yield dict(
            method=role,
            model="Qwen/Qwen3-8B",
            backend=backend,
            task="LiveCodeBench",
            context=40928,
            load="saturation",
            block=block,
            gpu_count=2,
            segments=topology_segments,
            workload="topology_confirmation",
        )
    for backend, method in itertools.product(("DFLASH", "DSPARK"), ("static", "lightcone")):
        yield dict(
            method=method,
            model="Qwen/Qwen3-8B",
            backend=backend,
            task="LiveCodeBench",
            context=40928,
            load="saturation",
            gpu_count=2,
            p99_extension=True,
            workload="p99_extension",
        )
    for block in SECONDARY_BLOCKS:
        for method in ("static", "lightcone"):
            yield dict(
                method=method,
                model="Qwen/Qwen3-8B",
                backend="DFLASH",
                task="LiveCodeBench",
                context=40928,
                load="saturation",
                block=block,
                gpu_count=2,
                matched_throughput=True,
                workload="matched_throughput",
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


def _onlinespec_candidates() -> tuple[dict[str, Any], ...]:
    rows = []
    for method in ("onlinespec_ogd", "onlinespec_opt"):
        for stride, parameterization, rank, lr in itertools.product(
            (20, 40, 80, 160), ("full", "lora"), (None, 8, 16, 32), (1e-4, 1e-3, 1e-2, 1e-1)
        ):
            if (parameterization == "full") == (rank is None):
                rows.append(
                    dict(
                        method=method,
                        parameterization=parameterization,
                        scope="all",
                        rank=rank,
                        learning_rate=lr,
                        stride=stride,
                        grad_clip=1.0,
                    )
                )
    for stride, (parameterization, rank), lr, hedge in itertools.product(
        (40, 80, 160),
        (("full", None), ("lora", 8), ("lora", 16), ("lora", 32)),
        (1e-4, 1e-3, 1e-2),
        (0.1, 0.5, 1.0),
    ):
        rows.append(
            dict(
                method="onlinespec_ens",
                parameterization=parameterization,
                scope="all",
                rank=rank,
                learning_rate=lr,
                stride=stride,
                grad_clip=1.0,
                additional_learning_rates=(lr * 3, lr * 10),
                hedge_learning_rate=hedge,
            )
        )
    if len(rows) != 236:
        raise AssertionError("OnlineSPEC grid must contain 236 candidates")
    return tuple(rows)


def _e0_methods(
    valid_pairs: set[tuple[str, str]] | None = None,
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
        rows.extend((*representative, method) for method in ("onlinespec_ogd", "onlinespec_opt"))
    if valid_pairs is None and len(rows) != 43:
        raise AssertionError("E0 bundled method surface must contain 43 rows")
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
    for index, candidate in enumerate(_onlinespec_candidates()):
        yield dict(
            **candidate,
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="CalibrationMix",
            tuning_index=index,
            gpu_count=2,
        )
    for index, method in enumerate(("static", "tts", "l0_naive")):
        yield dict(
            method=method,
            model="Qwen/Qwen3-8B",
            backend="DFLASH",
            task="CalibrationMix",
            tuning_index=236 + index,
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
            gpu_count=2,
        )


def _e0_blocks(
    blocks: Iterable[int], valid_pairs: set[tuple[str, str]] | None = None
) -> Iterator[dict[str, Any]]:
    segments = _segments(
        *(
            {"task": task, "context": 40928, "workload": "cross_workload_efficiency"}
            for task in E0_TASKS
        )
    )
    for model, backend, method in _e0_methods(valid_pairs):
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
        rows = _e0_blocks((0, 1), pairs)
    else:
        pairs = None if valid_e0 is None else {(model, backend) for model, backend, _ in valid_e0}
        rows = _e0_blocks(SECONDARY_BLOCKS, pairs)
    return _jobs(node, rows)


def segment_count(job: Job) -> int:
    segments = job.parameters.get("segments")
    return len(segments) if isinstance(segments, list) else 1


def default_row_counts() -> dict[str, int]:
    return {node: len(materialize(node)) for node in PAPER_NODES}
