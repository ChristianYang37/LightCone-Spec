"""Plain Python representation of the registered paper experiment DAG."""

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
REGIMES = (
    "long_input_short_output",
    "short_input_long_generation",
    "multi_turn_shared_prefix",
)
E3_STRATA = ("controlled_baseline", "LiveCodeBench", "MATH-500")
CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)
WIDTHS = (4, 8, 16)
ROLES = ("target_only", "static", "tts", "l0_naive", "lightcone")
E0_ROLES = (*ROLES, "onlinespec_ogd", "onlinespec_opt", "onlinespec_ens")
SCOPES = ("last1", "last3", "last5", "all")
RANKS = (1, 2, 4, 8, 16, 32, 64)
OPTIMIZERS = ("adam", "adamw", "sgdm", "nag", "muon", "lion", "chronobelief")
SCHEDULES = ("constant", "inverse_sqrt_published_update", "cosine_to_zero")
LEARNING_RATES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
TTS_LEARNING_RATES = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
TTS_STRIDES = (1, 5, 10, 15, 20, 30, 40, 50)
PILOT_BLOCKS = (0, 1, 2, 3)
E5_CONCURRENCY = (1, 2, 4, 8, 16, 32, 64, 128, 256)
E5_LOADS = (0.25, 0.50, 0.75, 0.90, 1.00, 1.10, 1.25)
E5_TRACES = ("immediate_burst", "burstgpt_shape", "moderate_soak", "saturation_soak", "overload_soak")
E5_TOPOLOGIES = ("tp1_dp1", "tp2_dp1", "two_replica_tp1_dp2")
E5_COHORTS = (1, 4, 16, 64)
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
E4_LOADS = ("low", "moderate", "saturation")
E4_TRAFFIC = ("pure_decode", "mixed_prefill_decode")
E4_SCREEN_LEVELS = {
    "stride": (1, 50),
    "microbatch": (1, 8),
    "coalescing": (1, 8),
    "stream_priority": ("default", "high"),
}
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
    "Alpaca",
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
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return text[:48] or "cell"


def _jobs(node: str, rows: Iterable[dict[str, Any]]) -> tuple[Job, ...]:
    result: list[Job] = []
    for ordinal, row in enumerate(rows):
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


def paper_plan(final_blocks: int | None = None, valid_e0: int | None = None, e1_safe: int = 32) -> tuple[NodePlan, ...]:
    n_text = str(final_blocks) if final_blocks is not None else "N (12-20)"
    v_text = str(valid_e0) if valid_e0 is not None else "V (0-108)"
    e2 = [105 * e1_safe]
    for _ in range(3):
        e2.append(max(math.ceil(e2[-1] / 4), 21))
    return (
        NodePlan("preflight", "10", 2, "runtime, exactness, memory, and interference"),
        NodePlan("E3a", "360", 1, "Target-only/Static context-load-width surface"),
        NodePlan("TTS-Cal", "288", 1, "frozen TTS numeric calibration"),
        NodePlan("E1", "68", 1, "32 geometries, two anchors, four fixed roles"),
        *tuple(NodePlan(f"E2-r{i}", str(rows + 4), 1, "successive-halving recipes plus fixed roles") for i, rows in enumerate(e2)),
        NodePlan("E4-screen", "48", 1, "systems mechanism screen"),
        NodePlan("E4-local", "96", 1, "local factorial"),
        NodePlan("E4-profile", "3", 2, "isolated profilers"),
        NodePlan("E3b-pilot", "1920", 1, "four excluded 480-row blocks"),
        NodePlan("E3b-final", f"480*{n_text}", 1, "held-out long-context confirmation"),
        NodePlan("E1a", "116", 1, "58 DSpark configurations by two verification modes"),
        NodePlan("E5-pilot", "2064", 2, "four 450-row pilots plus 264 fault diagnostics"),
        NodePlan("E5-final", f"450*{n_text}", 2, "production and topology confirmation"),
        NodePlan("E6-pilot", "242", 2, "two fit rows plus four 60-row pilots"),
        NodePlan("E6-final", f"60*{n_text}", 2, "two-point native-MTP confirmation"),
        NodePlan("E0-tune", f"108+239*{v_text}", 2, "compatibility probes and OnlineSPEC tuning"),
        NodePlan("E0-pilot", f"64*{v_text}", 2, "four excluded breadth blocks"),
        NodePlan("E0-final", f"16*{v_text}*{n_text}", 2, "breadth confirmation"),
    )


def _parameter_geometries() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        rows.append({"scope": scope, "parameterization": "full", "rank": None})
        rows.extend({"scope": scope, "parameterization": "lora", "rank": rank} for rank in RANKS)
    return tuple(rows)


def _preflight() -> Iterator[dict[str, Any]]:
    yield dict(method="target_only", model="Qwen/Qwen3-8B", backend="NONE", task="controlled_baseline", gpu_count=2, topology="tp2_dp1", preflight_kind="runtime_load", distribution_check=True, controlled_pair_baseline=True)
    yield dict(method="l0_naive", model="Qwen/Qwen3-8B", backend="DFLASH", task="controlled_baseline", gpu_count=2, topology="tp2_dp1", controlled_replay=True, distribution_check=True, preflight_kind="exactness_memory")
    for block, mode, gpu in itertools.product(range(2), ("isolated", "concurrent"), range(2)):
        yield dict(method="static", model="Qwen/Qwen3-8B", backend="DFLASH", task="controlled_baseline", context=4096, block=block, mode=mode, gpu_index=gpu, workload="interference")


def _e3a() -> Iterator[dict[str, Any]]:
    for context_index, context in enumerate(
        value for value in CONTEXTS if value not in LONG_CONTEXTS
    ):
        for regime_index, (regime, method) in enumerate(
            itertools.product(REGIMES, ("target_only", "static"))
        ):
            task = E3_STRATA[(context_index + regime_index // 2) % len(E3_STRATA)]
            yield dict(method=method, model="Qwen/Qwen3-8B", backend="NONE" if method == "target_only" else "DFLASH", task=task, context=context, load="c1", width=None if method == "target_only" else 8, regime=regime)
    for condition_index, (context, regime, concurrency) in enumerate(
        itertools.product(LONG_CONTEXTS, REGIMES, CONCURRENCY)
    ):
        task = E3_STRATA[condition_index % len(E3_STRATA)]
        yield dict(method="target_only", model="Qwen/Qwen3-8B", backend="NONE", task=task, context=context, load=f"c{concurrency}", regime=regime)
        for width in WIDTHS:
            yield dict(method="static", model="Qwen/Qwen3-8B", backend="DFLASH", task=task, context=context, load=f"c{concurrency}", width=width, regime=regime)


def _tts_cal() -> Iterator[dict[str, Any]]:
    for lr, stride, block in itertools.product(TTS_LEARNING_RATES, TTS_STRIDES, PILOT_BLOCKS):
        yield dict(method="tts", model="Qwen/Qwen3-8B", backend="DFLASH", task="TTS-Cal", context=40928, width=16, block=block, learning_rate=lr, stride=stride, optimizer="adam", parameterization="full", scope="all", workload="tts_calibration")


def _e1() -> Iterator[dict[str, Any]]:
    for role in ("target_only", "static", "tts", "l0_naive"):
        yield dict(method=role, model="Qwen/Qwen3-8B", backend="NONE" if role == "target_only" else "DFLASH", task="LiveCodeBench", context=40928, width=16, fixed_role=True)
    for geometry, optimizer in itertools.product(_parameter_geometries(), ("adamw", "sgdm")):
        yield dict(method="lightcone_candidate", model="Qwen/Qwen3-8B", backend="DFLASH", task="LiveCodeBench", context=40928, width=16, optimizer=optimizer, **geometry)


def e2_candidates(geometries: Iterable[dict[str, Any]] | None = None) -> tuple[dict[str, Any], ...]:
    selected = tuple(geometries) if geometries is not None else _parameter_geometries()
    return tuple({**geometry, "optimizer": optimizer, "learning_rate": lr, "schedule": schedule} for geometry, optimizer, lr, schedule in itertools.product(selected, OPTIMIZERS, LEARNING_RATES, SCHEDULES))


def _e2(round_index: int, candidates: Iterable[dict[str, Any]] | None) -> Iterator[dict[str, Any]]:
    rows = list(candidates if candidates is not None else e2_candidates())
    target = [3360, 840, 210, 53][round_index]
    for candidate in rows[:target]:
        recipe = {
            key: value
            for key, value in candidate.items()
            if key not in {"round", "fixed_role", "design_index", "probe", "tuning_index"}
        }
        yield dict(method="lightcone_candidate", model="Qwen/Qwen3-8B", backend="DFLASH", task="LiveCodeBench", context=(4096, 8192, 16384, 40928)[round_index], load=f"c{(2, 4, 8, 16)[round_index]}", width=16, round=round_index, minimum_updates=(2, 4, 8, 16)[round_index], regime="short_input_long_generation", **recipe)
    for role in ("target_only", "static", "tts", "l0_naive"):
        yield dict(method=role, model="Qwen/Qwen3-8B", backend="NONE" if role == "target_only" else "DFLASH", task="LiveCodeBench", context=(4096, 8192, 16384, 40928)[round_index], load=f"c{(2, 4, 8, 16)[round_index]}", width=16, fixed_role=True, regime="short_input_long_generation")


def _generic(node: str, count: int, task: str, gpu_count: int = 1) -> Iterator[dict[str, Any]]:
    for index in range(count):
        yield dict(method="lightcone", model="Qwen/Qwen3-8B", backend="DFLASH", task=task, context=40928, width=16, gpu_count=gpu_count, design_index=index)


def _e4_screen() -> Iterator[dict[str, Any]]:
    levels = E4_SCREEN_LEVELS
    for row, (a, b, c) in enumerate(itertools.product((0, 1), repeat=3)):
        factors = {
            "stride": levels["stride"][a],
            "microbatch": levels["microbatch"][b],
            "coalescing": levels["coalescing"][c],
            "stream_priority": levels["stream_priority"][a ^ b ^ c],
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
                chunked_prefill=traffic == "mixed_prefill_decode",
                prefix_reuse=traffic == "mixed_prefill_decode",
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
    names = ("stride", "microbatch", "coalescing", "stream_priority")
    for row, levels in enumerate(itertools.product(*(values[name] for name in names))):
        factors = dict(zip(names, levels, strict=True))
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
                chunked_prefill=traffic == "mixed_prefill_decode",
                prefix_reuse=traffic == "mixed_prefill_decode",
                graph_replay=True,
                workload="systems_local_factorial",
                **factors,
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
            graph_replay=True,
            workload="isolated_profile",
        )


def _e3b(blocks: Iterable[int]) -> Iterator[dict[str, Any]]:
    conditions = tuple(
        itertools.product(
            CONTEXTS,
            REGIMES,
            ("concurrency_one", "common_load"),
            ("matched", "deployment_optimal"),
        )
    )
    for block in blocks:
        for condition_index, (context, regime, load, panel) in enumerate(conditions):
            task = E3_STRATA[condition_index % len(E3_STRATA)]
            for role in ROLES:
                yield dict(method=role, model="Qwen/Qwen3-8B", backend="NONE" if role == "target_only" else "DFLASH", task=task, context=context, load=load, block=block, regime=regime, width_panel=panel)


def _e1a() -> Iterator[dict[str, Any]]:
    configs = list(_parameter_geometries())
    for depth in ("last1", "last3", "last5"):
        configs.append({"scope": f"{depth}_native_heads", "parameterization": "full", "rank": None})
        configs.extend({"scope": f"{depth}_native_heads", "parameterization": "lora", "rank": rank} for rank in RANKS)
    configs.extend(({"scope": "none", "parameterization": "none", "rank": None, "baseline": "target_only"}, {"scope": "none", "parameterization": "none", "rank": None, "baseline": "static"}))
    for configuration, verification in itertools.product(configs, ("fixed_budget", "native_scheduler")):
        baseline = configuration.get("baseline")
        yield dict(method=baseline or "lightcone_candidate", model="Qwen/Qwen3-8B", backend="NONE" if baseline == "target_only" else "DSPARK", task="LiveCodeBench", context=40928, width=16, verification=verification, **configuration)


def _e5_blocks(blocks: Iterable[int]) -> Iterator[dict[str, Any]]:
    for block, backend in itertools.product(blocks, ("DFLASH", "DSPARK")):
        for concurrency, role in itertools.product(E5_CONCURRENCY, ROLES):
            yield dict(method=role, model="Qwen/Qwen3-8B", backend=backend, task="LiveCodeBench", context=40928, load=f"closed_loop_c{concurrency}", block=block, gpu_count=2, workload="production_crossover")
        for factor, role in itertools.product(E5_LOADS, ROLES):
            yield dict(method=role, model="Qwen/Qwen3-8B", backend=backend, task="LiveCodeBench", context=40928, load=f"lambda_{factor}", block=block, gpu_count=2, workload="production_crossover")
        for trace, role in itertools.product(E5_TRACES, ROLES):
            yield dict(method=role, model="Qwen/Qwen3-8B", backend=backend, task="LiveCodeBench", context=40928, load=trace, block=block, gpu_count=2, workload="production_crossover")
        for topology, cohorts, popularity, role in itertools.product(
            E5_TOPOLOGIES, E5_COHORTS, E5_POPULARITY, ROLES
        ):
            yield dict(method=role, model="Qwen/Qwen3-8B", backend=backend, task="LiveCodeBench", context=40928, load="saturation", block=block, topology=topology, cohorts=cohorts, popularity=popularity, gpu_count=2, workload="topology_cohort_capacity")


def _e5_failures() -> Iterator[dict[str, Any]]:
    for failure, backend, topology, cohorts in itertools.product(E5_FAILURES, ("DFLASH", "DSPARK"), E5_TOPOLOGIES, E5_COHORTS):
        yield dict(method="lightcone", model="Qwen/Qwen3-8B", backend=backend, task="controlled_baseline", context=40928, load="c256" if failure == "queue_saturation" else "c1", gpu_count=2, failure=failure, topology=topology, cohorts=cohorts, workload="failure_injection")


def _e6(blocks: Iterable[int]) -> Iterator[dict[str, Any]]:
    for block, model, role, task, context in itertools.product(blocks, E6_MODELS, ROLES, ("LiveCodeBench", "MATH-500"), (4096, 16384, 32768)):
        yield dict(method=role, model=model, backend="NONE" if role == "target_only" else "NEXTN", task=task, context=context, load="common_slo_load", block=block, gpu_count=2)


def _e0_probes() -> tuple[tuple[str, str, str], ...]:
    return tuple(itertools.product(E0_MODELS, E0_BACKENDS, E0_TASKS))


def _onlinespec_candidates() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for method in ("onlinespec_ogd", "onlinespec_opt"):
        for stride, parameterization, rank, learning_rate in itertools.product(
            (20, 40, 80, 160),
            ("full", "lora"),
            (None, 8, 16, 32),
            (1e-4, 1e-3, 1e-2, 1e-1),
        ):
            if (parameterization == "full") != (rank is None):
                continue
            rows.append(
                dict(
                    method=method,
                    parameterization=parameterization,
                    scope="all",
                    rank=rank,
                    learning_rate=learning_rate,
                    stride=stride,
                    grad_clip=1.0,
                )
            )
    for stride, (parameterization, rank), learning_rate, hedge_rate in itertools.product(
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
                learning_rate=learning_rate,
                stride=stride,
                grad_clip=1.0,
                additional_learning_rates=(learning_rate * 3, learning_rate * 10),
                hedge_learning_rate=hedge_rate,
            )
        )
    if len(rows) != 236:
        raise AssertionError("OnlineSPEC grid must contain 236 candidates")
    return tuple(rows)


def materialize(
    node: str,
    *,
    final_blocks: int = 12,
    valid_e0: Iterable[tuple[str, str, str]] | None = None,
    e2_rows: Iterable[dict[str, Any]] | None = None,
    e0_recipes: dict[str, dict[str, Any]] | None = None,
    e4_neighborhoods: dict[str, tuple[object, object]] | None = None,
) -> tuple[Job, ...]:
    if node not in PAPER_NODES:
        raise ValueError(f"unknown paper node {node}")
    final = range(4, 4 + final_blocks)
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
        rows = _e3b(PILOT_BLOCKS)
    elif node == "E3b-final":
        rows = _e3b(final)
    elif node == "E1a":
        rows = _e1a()
    elif node == "E5-pilot":
        rows = itertools.chain(_e5_blocks(PILOT_BLOCKS), _e5_failures())
    elif node == "E5-final":
        rows = _e5_blocks(final)
    elif node == "E6-pilot":
        rows = itertools.chain(
            (
                dict(
                    method="lightcone",
                    model=model,
                    backend="NEXTN",
                    task="LiveCodeBench",
                    load="c1",
                    gpu_count=2,
                    interface_fit=True,
                    minimum_updates=1,
                )
                for model in E6_MODELS
            ),
            _e6(PILOT_BLOCKS),
        )
    elif node == "E6-final":
        rows = _e6(final)
    else:
        valid = tuple(valid_e0) if valid_e0 is not None else _e0_probes()
        if node == "E0-tune":
            candidates = _onlinespec_candidates()
            rows = itertools.chain(
                (
                    dict(
                        method="static",
                        model=model,
                        backend=backend,
                        task=task,
                        probe=True,
                        adaptive_probe=True,
                        gpu_count=2,
                    )
                    for model, backend, task in _e0_probes()
                ),
                (
                    dict(
                        **candidate,
                        model=model,
                        backend=backend,
                        task=task,
                        tuning_index=index,
                        gpu_count=2,
                    )
                    for model, backend, task in valid
                    for index, candidate in enumerate(candidates)
                ),
                (
                    dict(
                        method=method,
                        model=model,
                        backend=backend,
                        task=task,
                        tuning_index=236 + index,
                        gpu_count=2,
                    )
                    for model, backend, task in valid
                    for index, method in enumerate(("static", "tts", "l0_naive"))
                ),
            )
        else:
            blocks = PILOT_BLOCKS if node == "E0-pilot" else final
            recipes = e0_recipes or {}
            rows = (
                dict(
                    method=role,
                    model=model,
                    backend=backend,
                    task=task,
                    load=load,
                    block=block,
                    gpu_count=2,
                    **recipes.get(f"{model}|{backend}|{task}|{role}", {}),
                )
                for model, backend, task in valid
                for block, role, load in itertools.product(
                    blocks, E0_ROLES, ("concurrency_one", "common_slo_load")
                )
            )
    return _jobs(node, rows)


def default_row_counts(final_blocks: int = 12) -> dict[str, int]:
    return {node: len(materialize(node, final_blocks=final_blocks)) for node in PAPER_NODES}
