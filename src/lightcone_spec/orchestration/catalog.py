"""P0-P5 experiment manifest catalog (spec 13 and P5 proposal).

Every phase's run units are generated deterministically here and written
as immutable manifest JSON files. GPU phases target the reference
profiles; P0 and the smoke manifest run on the CPU reference engine.
"""

from __future__ import annotations

from pathlib import Path

from lightcone_spec.config.schema import (
    DSPARK_MODEL_PAIR_IDS,
    METHOD_KEYS,
)
from lightcone_spec.orchestration.manifest import ExperimentManifest
from lightcone_spec.orchestration.units import RunUnit

MAIN_SEEDS = (0, 1, 2)
P2_MAIN_TASKS = ("livecodebench", "aime25", "math500", "mt_bench")
P4_BREADTH_TASKS = (
    "gsm8k",
    "mbpp",
    "humaneval",
    "aime24",
    "olympiadbench_math",
    "olympiadbench_physics",
    "gpqa_diamond",
    "theoremqa",
    "alpaca",
    "arena_hard_v2",
)
P5_TASKS = ("livecodebench", "math500", "mt_bench")
P5_CONTEXT_LENGTHS = (512, 1024, 2048, 4096, 8192, 16384, 32768)
P5_METHODS = ("static", "tts", "naive_async", "lc_gate", "lc_damp")
P5_CROSS_BACKEND_PAIRS = (
    "qwen3_8b_dspark7",
    "qwen3_8b_dflash16",
    "qwen3_8b_eagle3",
)
P5_CROSS_BACKEND_CONTEXTS = (4096, 16384, 32768)
P5_PRIORITY_DFLASH_CONTEXTS = (512, 4096, 16384, 40000)
P5_PRIORITY_FINAL_CONTEXTS = (
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    40000,
)
P5_PRIORITY_STRIDE_CANDIDATES = (1, 4, 8, 16)
P5_PRIORITY_CONFIRMATION_MIN_PROMPT_CLUSTERS = 32
P5_PRIORITY_PYTORCH_CUDA_ALLOC_CONF = (
    "backend:native,expandable_segments:True"
)
# NVIDIA publishes 1 PFLOP/s BF16 tensor throughput for this Blackwell server
# GPU with structured sparsity.  P5 binds the dense denominator explicitly as
# half that value; this is an inferred normalization constant, not a claim that
# the workload executes sparse kernels.
P5_RTX_PRO_6000_SERVER_DENSE_BF16_TFLOPS_PER_GPU = 500.0
P5_RTX_PRO_6000_SERVER_DENSE_BF16_PEAK_BASIS = (
    "nvidia_official_1pflops_bf16_sparse_dense_inferred_half_v1"
)
P5_PRIORITY_CONFIRMATION_LOADS = (
    ("p5_ctx_4096-16384", 8),
    ("p5_ctx_4096", 48),
    ("p5_ctx_16384", 20),
)
P5_PRIORITY_METHODS = (
    "static",
    "tts",
    "naive_async",
    "lc_gate",
    "lc_damp",
    "lc_transport",
)
TOY_METHODS_P0 = (
    "static",
    "sync_fresh",
    "tts",
    "naive_async",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
    "lc_gate",
    "lc_damp",
    "lc_transport",
    "oracle_current",
)


def _unit(**kw) -> RunUnit:
    defaults = dict(
        prompt_subset="full",
        lifecycle="request",
        sampling_profile="main_t1_p1",
        trainable_scope="adapter",
        stride=10,
        logical_delay=0,
        concurrency=1,
        contention_condition="none",
        adapter_rank=16,
        transport_variant=None,
        required=True,
        allow_resource_skip=False,
    )
    defaults.update(kw)
    return RunUnit(**defaults)


# ---------------------------------------------------------------------------
# P0: toy diagnostics (CPU reference engine)
# ---------------------------------------------------------------------------


def p0_manifest() -> ExperimentManifest:
    units = []
    for method in TOY_METHODS_P0:
        for dataset in ("markov4_world", "phase_switch", "aba_recurrence"):
            for delay in (0, 2, 5):
                if method in ("static", "sync_fresh") and delay > 0:
                    continue
                units.append(
                    _unit(
                        phase="p0",
                        model_pair="toy_markov4",
                        method=method,
                        dataset=dataset,
                        seed=0,
                        stride=4,
                        logical_delay=delay,
                    )
                )
    # Twin diagnostics (idle / wall-only / state-only).
    for dataset in ("idle_insertion_twins", "wall_only_twins", "state_only_twins"):
        for method in ("naive_async", "lc_damp"):
            units.append(
                _unit(
                    phase="p0",
                    model_pair="toy_markov4",
                    method=method,
                    dataset=dataset,
                    seed=0,
                    stride=4,
                    logical_delay=3,
                )
            )
    return ExperimentManifest(
        name="p0_toy_diagnostics",
        phase="p0",
        description="P0 toy diagnostics: sampling exactness world, "
        "optimization toy conditions, twins; CPU reference engine",
        units=units,
        engine_params={
            "num_requests": 4,
            "max_rounds": 24,
            "max_new_tokens": 48,
            "lr": 1e-3,
            # Produced by `lightcone-spec replay --output-dir
            # artifacts/controller` before running P0 (fail closed if
            # missing); resolved relative to the working directory.
            "controller_artifact_path": "artifacts/controller/toy_markov4.controller.json",
        },
        profile="cpu_reference",
    )


# ---------------------------------------------------------------------------
# P1: replay label/predictor phase (GPU traces; toy replay runs locally)
# ---------------------------------------------------------------------------


def p1_manifest() -> ExperimentManifest:
    units = []
    for pair_id in DSPARK_MODEL_PAIR_IDS:
        for dataset in P2_MAIN_TASKS:
            for delay in (1, 3, 5, 10, 20):
                units.append(
                    _unit(
                        phase="p1",
                        model_pair=pair_id,
                        method="naive_async",
                        dataset=dataset,
                        prompt_subset="replay_pool",
                        seed=0,
                        logical_delay=delay,
                        contention_condition="staleness_only",
                    )
                )
    return ExperimentManifest(
        name="p1_replay_labels",
        phase="p1",
        description="P1 counterfactual replay traces and stale-update "
        "labels across injected delays; controller artifacts are fitted "
        "from these traces and frozen per model pair",
        units=units,
        engine_params={"prompt_limit": 96, "max_new_tokens": 8192},
        profile="reference_8x80gb",
    )


# ---------------------------------------------------------------------------
# P2: main system matrix
# ---------------------------------------------------------------------------


def p2_manifest() -> ExperimentManifest:
    units = []
    for pair_id in DSPARK_MODEL_PAIR_IDS:
        for method in METHOD_KEYS:
            for dataset in P2_MAIN_TASKS:
                for seed in MAIN_SEEDS:
                    for lifecycle in ("request", "stream"):
                        units.append(
                            _unit(
                                phase="p2",
                                model_pair=pair_id,
                                method=method,
                                dataset=dataset,
                                seed=seed,
                                lifecycle=lifecycle,
                                contention_condition="realistic_async",
                            )
                        )
    # Greedy parity side table (temperature 0), request lifecycle only.
    for pair_id in DSPARK_MODEL_PAIR_IDS:
        for method in ("static", "naive_async", "lc_gate", "lc_damp"):
            for dataset in ("math500",):
                units.append(
                    _unit(
                        phase="p2",
                        model_pair=pair_id,
                        method=method,
                        dataset=dataset,
                        seed=0,
                        sampling_profile="greedy_t0",
                    )
                )
    return ExperimentManifest(
        name="p2_main_matrix",
        phase="p2",
        description="P2 main table: 4 pairs x 11 methods x 4 tasks x 3 "
        "seeds x {request, stream} + greedy parity side table",
        units=units,
        engine_params={"prompt_limit": 128, "max_new_tokens": 32768},
        profile="reference_8x80gb",
    )


# ---------------------------------------------------------------------------
# P3: mechanism separation
# ---------------------------------------------------------------------------


def p3_manifest() -> ExperimentManifest:
    units = []
    pair_id = "qwen3_8b_dspark7"
    # Delay sweep under the three contention conditions.
    for cond in ("control", "staleness_only", "contention_only", "realistic_async"):
        for delay in (0, 1, 3, 5, 10, 20):
            for method in ("naive_async", "lc_gate", "lc_damp", "lc_transport"):
                if cond == "contention_only" and delay > 0:
                    continue
                units.append(
                    _unit(
                        phase="p3",
                        model_pair=pair_id,
                        method=method,
                        dataset="livecodebench",
                        seed=0,
                        logical_delay=delay,
                        contention_condition=cond,
                    )
                )
    # Diagnostic negative controls.
    for method in (
        "round_discard",
        "wall_damp",
        "endpoint_gate",
        "parameter_only",
        "random_transport",
    ):
        for delay in (3, 10):
            units.append(
                _unit(
                    phase="p3",
                    model_pair=pair_id,
                    method=method,
                    dataset="livecodebench",
                    seed=0,
                    logical_delay=delay,
                    contention_condition="staleness_only",
                )
            )
    # L3 parameter-staleness with max_in_flight=2 (transport variants).
    for variant in ("joint", "parameter_only", "state_only", "random", "discard",
                    "l2_no_transport"):
        units.append(
            _unit(
                phase="p3",
                model_pair=pair_id,
                method="lc_transport",
                dataset="livecodebench",
                seed=0,
                logical_delay=5,
                contention_condition="staleness_only",
                transport_variant=variant,
            )
        )
    # Cache-safe full-rank tail capacity check.  This does not mutate drafter
    # backbone weights and therefore remains valid with stream lifecycle/KV.
    for pair, dataset in (
        ("qwen3_4b_dspark7", "livecodebench"),
        ("qwen3_8b_dspark7", "aime25"),
    ):
        units.append(
            _unit(
                phase="p3",
                model_pair=pair,
                method="naive_async",
                dataset=dataset,
                seed=0,
                lifecycle="stream",
                trainable_scope="full_rank_tail",
                allow_resource_skip=True,
                contention_condition="staleness_only",
            )
        )
    return ExperimentManifest(
        name="p3_mechanism_tail",
        phase="p3",
        description="P3 mechanism separation: staleness vs contention vs "
        "delay sweeps, diagnostic controls, transport variants and a "
        "full-rank tail capacity check",
        units=units,
        engine_params={"prompt_limit": 96, "max_new_tokens": 16384},
        profile="reference_8x80gb",
    )


# ---------------------------------------------------------------------------
# P4: breadth + streaming
# ---------------------------------------------------------------------------


def p4_manifest() -> ExperimentManifest:
    units = []
    for pair_id in ("qwen3_8b_dspark7", "gemma4_12b_dspark7"):
        for dataset in P4_BREADTH_TASKS:
            for method in ("static", "naive_async", "lc_gate", "lc_damp"):
                units.append(
                    _unit(
                        phase="p4",
                        model_pair=pair_id,
                        method=method,
                        dataset=dataset,
                        seed=0,
                        contention_condition="realistic_async",
                    )
                )
    # Streaming with concurrency sweep.
    for concurrency in (1, 4, 8, 16):
        for method in ("static", "naive_async", "lc_damp"):
            units.append(
                _unit(
                    phase="p4",
                    model_pair="qwen3_8b_dspark7",
                    method=method,
                    dataset="livecodebench",
                    seed=0,
                    lifecycle="stream",
                    concurrency=concurrency,
                    contention_condition="realistic_async",
                )
            )
    return ExperimentManifest(
        name="p4_breadth_streaming",
        phase="p4",
        description="P4 breadth tasks and streaming concurrency sweep",
        units=units,
        engine_params={"prompt_limit": 64, "max_new_tokens": 16384},
        profile="reference_8x80gb",
    )


# ---------------------------------------------------------------------------
# P5: long-context survival-weighted acceptance
# ---------------------------------------------------------------------------


def p5_manifest() -> ExperimentManifest:
    units = []
    for task in P5_TASKS:
        # Reuse one SGLang Engine across compatible context buckets.  The
        # 32K/c16 bucket remains isolated because it exceeds the calibrated
        # single-card KV capacity and may be resource-skipped independently.
        stream_groups = (
            (1, "p5_ctx_512-32768", True, False),
            (4, "p5_ctx_512-32768", True, False),
            (16, "p5_ctx_512-16384", True, False),
            (16, "p5_ctx_32768", False, True),
        )
        for concurrency, prompt_subset, required, allow_skip in stream_groups:
            for method in P5_METHODS:
                units.append(
                    _unit(
                        phase="p5",
                        model_pair="qwen3_4b_dspark7",
                        method=method,
                        dataset=task,
                        prompt_subset=prompt_subset,
                        seed=0,
                        lifecycle="stream",
                        stride=4,
                        concurrency=concurrency,
                        contention_condition=(
                            "none"
                            if method in ("static", "tts")
                            else "realistic_async"
                        ),
                        required=required,
                        allow_resource_skip=allow_skip,
                    )
                )
        # Request-reset is a deliberately small side table, not mixed with
        # the streaming headline or its bootstrap clusters.
        for method in P5_METHODS:
            units.append(
                _unit(
                    phase="p5",
                    model_pair="qwen3_4b_dspark7",
                    method=method,
                    dataset=task,
                    prompt_subset="p5_ctx_request_side",
                    seed=0,
                    lifecycle="request",
                    stride=4,
                    concurrency=1,
                    contention_condition=(
                        "none"
                        if method in ("static", "tts")
                        else "realistic_async"
                    ),
                )
            )
    return ExperimentManifest(
        name="p5_long_context_acceptance_engine_reuse",
        phase="p5",
        description=(
            "Exact-token prefix checkpoints with engine reuse across compatible "
            "context buckets; survival-weighted acceptance, elasticity/curvature "
            "and engineering cost gates. L3 is omitted until its existing "
            "held-out enable gate passes."
        ),
        profile="local_1x96gb",
        engine_params={
            "prompt_limit": 32,
            "benchmark_repetitions": 5,
            "max_new_tokens": 256,
            "max_running_requests": 48,
            "max_total_tokens": 400000,
            "p5_context_lengths": list(P5_CONTEXT_LENGTHS),
            "p5_request_context_lengths": [4096, 16384, 32768],
            "speculative_num_draft_tokens": 8,
            "warmup_prompts": 4,
            "request_timeout_s": 1800,
        },
        units=units,
    )


def p5_cross_backend_tail_manifest() -> ExperimentManifest:
    """Paired Qwen3-8B DSpark/DFlash/EAGLE3 tail-capacity matrix.

    One source manifest is executed three times with ``--weight-update-mode``.
    Static units retain identical IDs; non-static unit and manifest identities
    are recomputed for each effective tier.
    """
    units = []
    for pair_id in P5_CROSS_BACKEND_PAIRS:
        for task in P5_TASKS:
            for concurrency in (1, 4):
                for method in P5_METHODS:
                    units.append(
                        _unit(
                            phase="p5_cross_backend",
                            model_pair=pair_id,
                            method=method,
                            dataset=task,
                            prompt_subset="p5_ctx_4096-32768",
                            seed=0,
                            lifecycle="stream",
                            trainable_scope="output_residual",
                            stride=4,
                            concurrency=concurrency,
                            contention_condition=(
                                "none"
                                if method in ("static", "tts")
                                else "realistic_async"
                            ),
                        )
                    )
    return ExperimentManifest(
        name="p5_cross_backend_tail",
        phase="p5_cross_backend",
        description=(
            "Paired Qwen3-8B DSpark7, DFlash-b16 and EAGLE3 long-context "
            "tail-update comparison at 4K/16K/32K. Execute once per explicit "
            "weight-update mode; L3 remains gated."
        ),
        profile="local_1x96gb",
        engine_params={
            "prompt_limit": 32,
            "benchmark_repetitions": 5,
            "max_new_tokens": 256,
            "max_running_requests": 48,
            "max_total_tokens": 400000,
            "p5_context_lengths": list(P5_CROSS_BACKEND_CONTEXTS),
            "warmup_prompts": 4,
            "request_timeout_s": 1800,
        },
        units=units,
    )


def p5_cross_backend_trace_manifest() -> ExperimentManifest:
    """Historical injected-delay producer per backend and tail tier.

    Execute this source manifest once per ``--weight-update-mode`` into a
    separate artifact root.  Its L0 units deliberately retain the immutable
    ``logical_delay=2`` design, so their timing is an injected-staleness study,
    not evidence of naturally occurring GPU arrival delay.  Use
    :func:`p5_priority_dflash_paired_trace_manifest` for the new natural-delay
    L0/TTS pairing.
    """
    units = [
        _unit(
            phase="p5_cross_backend_trace",
            model_pair=pair_id,
            method=method,
            dataset="livecodebench",
            prompt_subset="p5_ctx_16384",
            seed=0,
            lifecycle="stream",
            trainable_scope="output_residual",
            stride=4,
            logical_delay=2 if method == "naive_async" else 0,
            concurrency=1,
            contention_condition=(
                "realistic_async" if method == "naive_async" else "none"
            ),
        )
        for pair_id in P5_CROSS_BACKEND_PAIRS
        for method in ("naive_async", "tts")
    ]
    return ExperimentManifest(
        name="p5_cross_backend_trace",
        phase="p5_cross_backend_trace",
        description=(
            "Paired bounded controller trace producer for Qwen3-8B DSpark7, "
            "DFlash-b16 and EAGLE3. Run once per explicit tail mode; L0 emits "
            "full-candidate labels and TTS emits paired barrier evidence."
        ),
        profile="local_1x96gb",
        engine_params={
            "prompt_limit": 96,
            "benchmark_repetitions": 1,
            "max_new_tokens": 128,
            "max_running_requests": 2,
            "max_total_tokens": 100000,
            "p5_context_lengths": [16384],
            "trace_level": "full",
            # Full-rank 8B tails are roughly 64--70 MiB per replay record.  One
            # record/request and 6 GiB/run retain enough independent groups for
            # the train/calibration/test split without exhausting a 200 GiB disk.
            "trace_capture_max_bytes": 6 * (1 << 30),
            "trace_capture_max_records_per_request": 1,
            "trace_producer_methods": ["naive_async", "tts"],
            "warmup_prompts": 1,
            "request_timeout_s": 1800,
        },
        units=units,
    )


def p5_priority_dflash_0_40k_manifest() -> ExperimentManifest:
    """Held-out DFlash4B long-context comparison at the frozen safe tier."""
    units = [
        _unit(
            phase="p5_priority_dflash_0_40k_v1",
            model_pair="qwen3_4b_dflash16",
            method=method,
            dataset="livecodebench",
            prompt_subset="p5_ctx_512-40000",
            seed=0,
            lifecycle="stream",
            sampling_profile="greedy_t0",
            trainable_scope="tail_lora",
            stride=4,
            logical_delay=0,
            concurrency=concurrency,
            contention_condition=(
                "none" if method in ("static", "tts") else "realistic_async"
            ),
        )
        for concurrency in (1, 4, 8)
        for method in P5_PRIORITY_METHODS
    ]
    return ExperimentManifest(
        name="p5_priority_dflash_0_40k_v1",
        phase="p5_priority_dflash_0_40k_v1",
        description=(
            "Paired Qwen3-4B DFlash-b16 tail-LoRA evaluation from the shortest "
            "measured 512-token prefix through 40,000 tokens. The 512-token "
            "bucket is a short-context proxy, not a literal zero-context "
            "measurement; all asynchronous methods use natural GPU arrival "
            "timing with no injected logical delay."
        ),
        profile="local_1x96gb",
        engine_params={
            "prompt_limit": 16,
            "benchmark_repetitions": 3,
            "max_new_tokens": 128,
            "ignore_eos": True,
            "max_running_requests": 8,
            "max_total_tokens": 400000,
            "p5_context_lengths": list(P5_PRIORITY_DFLASH_CONTEXTS),
            "peak_tflops_per_gpu": (
                P5_RTX_PRO_6000_SERVER_DENSE_BF16_TFLOPS_PER_GPU
            ),
            "peak_tflops_basis": (
                P5_RTX_PRO_6000_SERVER_DENSE_BF16_PEAK_BASIS
            ),
            "lr": 3e-5,
            "warmup_prompts": 4,
            "request_timeout_s": 1800,
        },
        units=units,
    )


def p5_priority_dflash_final_manifest(
    *,
    name: str,
    model_pair: str,
    trainable_scope: str,
    adapter_rank: int,
    methods: tuple[str, ...],
    load_groups: tuple[tuple[str, int], ...],
    tts_stride: int | None,
    adaptation_stride: int,
    lr: float,
    weight_decay: float,
    lockfile_sha256: str,
    model_roots_sha256: str,
    locked_model_revisions: dict[str, str],
    runtime_implementation_fingerprint: dict,
    controller_identity_sha256: str,
    claim_scope: str,
    tts_role_strides: dict[str, int] | None = None,
) -> ExperimentManifest:
    """Build one evidence-bound final P5 load profile.

    The checked-in ``p5_priority_dflash_0_40k_v1`` manifest remains a legacy
    calibration input.  Final evidence must instead call this parameterized
    builder after stride confirmation/controller gating, so optimizer and
    publication identities cannot silently fall back to its old lr/stride.
    """

    allowed_methods = {
        "static",
        "tts",
        "naive_async",
        "lc_gate",
        "lc_damp",
        "lc_transport",
    }
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("final P5 methods must be non-empty and unique")
    unknown = sorted(set(methods) - allowed_methods)
    if unknown:
        raise ValueError(f"unsupported final P5 methods: {unknown}")
    if not {"static", "tts", "naive_async"}.issubset(methods):
        raise ValueError("final P5 always requires Static, TTS and L0")
    if not load_groups or any(concurrency <= 0 for _, concurrency in load_groups):
        raise ValueError("final P5 load groups must have positive concurrency")

    role_names = (
        "acceptance_best",
        "engineering_best",
        "same_stride",
    )
    if tts_role_strides is None:
        if isinstance(tts_stride, bool) or not isinstance(tts_stride, int) or tts_stride <= 0:
            raise ValueError("final P5 requires a positive TTS stride")
        resolved_tts_roles = {name: tts_stride for name in role_names}
    else:
        if set(tts_role_strides) != set(role_names):
            raise ValueError("final P5 TTS role strides are incomplete")
        resolved_tts_roles = dict(tts_role_strides)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in resolved_tts_roles.values()
        ):
            raise ValueError("final P5 TTS role strides must be positive integers")
        if tts_stride is not None and tts_stride != resolved_tts_roles["acceptance_best"]:
            raise ValueError("legacy TTS stride conflicts with acceptance-best role")
    unique_tts_strides = tuple(dict.fromkeys(resolved_tts_roles.values()))

    units = []
    for prompt_subset, concurrency in load_groups:
        method_strides = []
        for method in methods:
            if method == "static":
                method_strides.append((method, 1))
            elif method == "tts":
                method_strides.extend((method, stride) for stride in unique_tts_strides)
            else:
                method_strides.append((method, adaptation_stride))
        for method, stride in method_strides:
            units.append(
                _unit(
                    phase=name,
                    model_pair=model_pair,
                    method=method,
                    dataset="livecodebench",
                    prompt_subset=prompt_subset,
                    seed=0,
                    lifecycle="stream",
                    sampling_profile="greedy_t0",
                    trainable_scope=trainable_scope,
                    stride=stride,
                    logical_delay=0,
                    concurrency=concurrency,
                    adapter_rank=adapter_rank,
                    contention_condition=(
                        "none"
                        if method in ("static", "tts")
                        else "realistic_async"
                    ),
                )
            )

    checkpoint_limit = 40960
    max_new_tokens = 512
    return ExperimentManifest(
        name=name,
        phase=name,
        description=(
            "Evidence-bound DFlash final comparison over exact prefix "
            "checkpoints 512/1K/2K/4K/8K/16K/32K/40K. Each context owns an "
            "independent measurement clock; 512 is the short-context proxy, "
            "not literal zero context. The 40K checkpoint plus 512 output "
            "tokens remains within the declared 40,960-token checkpoint limit."
        ),
        profile="local_1x96gb",
        lockfile_sha256=lockfile_sha256,
        engine_params={
            "prompt_limit": 48,
            "prompt_offset": 184,
            "benchmark_repetitions": 5,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
            "max_running_requests": max(
                concurrency for _, concurrency in load_groups
            ),
            "max_total_tokens": 400000,
            "p5_context_lengths": list(P5_PRIORITY_FINAL_CONTEXTS),
            "p5_context_timing_contract": "independent_exact_context_group_v1",
            "p5_load_groups": [
                {
                    "prompt_subset": prompt_subset,
                    "concurrency": concurrency,
                }
                for prompt_subset, concurrency in load_groups
            ],
            "checkpoint_max_context_length": checkpoint_limit,
            "checkpoint_max_new_tokens": max_new_tokens,
            "pytorch_cuda_alloc_conf": P5_PRIORITY_PYTORCH_CUDA_ALLOC_CONF,
            "peak_tflops_per_gpu": (
                P5_RTX_PRO_6000_SERVER_DENSE_BF16_TFLOPS_PER_GPU
            ),
            "peak_tflops_basis": (
                P5_RTX_PRO_6000_SERVER_DENSE_BF16_PEAK_BASIS
            ),
            "optimizer": "adamw",
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "warmup_prompts": 20,
            "trace_level": "light",
            "claim_scope": claim_scope,
            "model_roots_sha256": model_roots_sha256,
            "locked_model_revisions": locked_model_revisions,
            "runtime_implementation_fingerprint": (
                runtime_implementation_fingerprint
            ),
            "matched_controller_identity_sha256": (
                controller_identity_sha256
            ),
            "tts_role_strides": resolved_tts_roles,
            "request_timeout_s": 1800,
        },
        units=units,
    )


def p5_priority_dflash_stride_screen_manifest() -> ExperimentManifest:
    """Non-claim high-load screen for the TTS/L0 publication stride."""
    common = {
        "phase": "p5_priority_dflash_stride_screen_v1",
        "model_pair": "qwen3_4b_dflash16",
        "dataset": "livecodebench",
        "prompt_subset": "p5_ctx_4096-16384",
        "seed": 0,
        "lifecycle": "stream",
        "sampling_profile": "greedy_t0",
        "trainable_scope": "tail_lora",
        "logical_delay": 0,
        "concurrency": 20,
    }
    units = [
        _unit(
            **common,
            method="static",
            stride=1,
            contention_condition="none",
        )
    ]
    units.extend(
        _unit(
            **common,
            method=method,
            stride=stride,
            contention_condition=(
                "none" if method == "tts" else "realistic_async"
            ),
        )
        for method in ("tts", "naive_async")
        for stride in (1, 4, 8, 16)
    )
    return ExperimentManifest(
        name="p5_priority_dflash_stride_screen_v1",
        phase="p5_priority_dflash_stride_screen_v1",
        description=(
            "Non-claim Qwen3-4B DFlash-b16 candidate screen at grouped 4K/16K "
            "prefixes, prompt slice [0,40), and concurrency 20. It compares "
            "one Static baseline with "
            "paired TTS/L0 strides 1/4/8/16 using the frozen safe tail-LoRA "
            "optimizer tier; results select a confirmation design and are not "
            "standalone evidence of superiority."
        ),
        profile="local_1x96gb",
        engine_params={
            "prompt_limit": 40,
            "prompt_offset": 0,
            "benchmark_repetitions": 3,
            "max_new_tokens": 512,
            "ignore_eos": True,
            "max_running_requests": 20,
            "max_total_tokens": 400000,
            "p5_context_lengths": [4096, 16384],
            "p5_context_timing_contract": (
                "independent_exact_context_group_v1"
            ),
            "pytorch_cuda_alloc_conf": P5_PRIORITY_PYTORCH_CUDA_ALLOC_CONF,
            "lr": 1e-4,
            "weight_decay": 1e-2,
            "warmup_prompts": 20,
            "trace_level": "light",
            "claim_scope": "candidate_screen_only_no_ci",
            "p5_stride_screen_required_safety_columns": [
                "adaptation_fallback_count"
            ],
            "request_timeout_s": 1800,
        },
        units=units,
    )


def p5_priority_dflash_stride_confirmation_manifest(
    *,
    tts_acceptance_stride: int,
    tts_engineering_stride: int,
    l0_stride: int,
    lockfile_sha256: str | None = None,
) -> ExperimentManifest:
    """Paired confirmation for stride-screen winners at three useful loads.

    The builder is deliberately parameterized instead of being registered in
    ``ALL_MANIFESTS``: its three stride choices are evidence selected and must be
    bound to that selector receipt by the handoff script.  One grouped c8 unit
    covers both 4K and 16K; the c48 and c20 units isolate the respective
    high-load 4K and 16K cells.  If both winners use the same stride, the
    TTS roles with equal strides share one physical run unit.
    """

    for name, value in (
        ("tts_acceptance_stride", tts_acceptance_stride),
        ("tts_engineering_stride", tts_engineering_stride),
        ("l0_stride", l0_stride),
    ):
        if isinstance(value, bool) or value not in P5_PRIORITY_STRIDE_CANDIDATES:
            raise ValueError(
                f"{name} must be one of {P5_PRIORITY_STRIDE_CANDIDATES}, got "
                f"{value!r}"
            )

    phase = "p5_priority_dflash_stride_confirmation_v1"
    role_specs = (
        ("static", 1, "none"),
        ("tts", tts_acceptance_stride, "none"),
        ("tts", tts_engineering_stride, "none"),
        ("tts", l0_stride, "none"),
        ("naive_async", l0_stride, "realistic_async"),
    )
    units = []
    seen: set[tuple[str, int, str, int]] = set()
    for method, stride, contention in role_specs:
        for prompt_subset, concurrency in P5_PRIORITY_CONFIRMATION_LOADS:
            identity = (method, stride)
            cell = (*identity, prompt_subset, concurrency)
            if cell in seen:
                continue
            seen.add(cell)
            units.append(
                _unit(
                    phase=phase,
                    model_pair="qwen3_4b_dflash16",
                    method=method,
                    dataset="livecodebench",
                    prompt_subset=prompt_subset,
                    seed=0,
                    lifecycle="stream",
                    sampling_profile="greedy_t0",
                    trainable_scope="tail_lora",
                    stride=stride,
                    logical_delay=0,
                    concurrency=concurrency,
                    contention_condition=contention,
                )
            )
    return ExperimentManifest(
        name=phase,
        phase=phase,
        description=(
            "Receipt-selected Qwen3-4B DFlash-b16 paired confirmation at "
            f"4K/16K: acceptance-best TTS stride {tts_acceptance_stride}, "
            f"engineering-best TTS stride {tts_engineering_stride}, L0-best "
            f"stride {l0_stride}, and same-stride TTS on the disjoint prompt "
            "slice [40,88) under c8 plus high-load c48/c20."
        ),
        profile="local_1x96gb",
        lockfile_sha256=lockfile_sha256,
        engine_params={
            "prompt_limit": 48,
            "prompt_offset": 40,
            "benchmark_repetitions": 5,
            "max_new_tokens": 512,
            "ignore_eos": True,
            "max_running_requests": 48,
            "max_total_tokens": 400000,
            "p5_context_lengths": [4096, 16384],
            "p5_context_timing_contract": (
                "independent_exact_context_group_v1"
            ),
            "pytorch_cuda_alloc_conf": P5_PRIORITY_PYTORCH_CUDA_ALLOC_CONF,
            "peak_tflops_per_gpu": (
                P5_RTX_PRO_6000_SERVER_DENSE_BF16_TFLOPS_PER_GPU
            ),
            "peak_tflops_basis": (
                P5_RTX_PRO_6000_SERVER_DENSE_BF16_PEAK_BASIS
            ),
            "lr": 1e-4,
            "weight_decay": 1e-2,
            "warmup_prompts": 20,
            "trace_level": "light",
            "claim_scope": "paired_stride_confirmation",
            "p5_confirmation_min_paired_prompt_clusters": (
                P5_PRIORITY_CONFIRMATION_MIN_PROMPT_CLUSTERS
            ),
            "request_timeout_s": 1800,
        },
        units=units,
    )


def p5_priority_dflash_paired_trace_manifest() -> ExperimentManifest:
    """Natural-arrival L0/TTS trace pair for DFlash4B tail-LoRA controllers."""
    units = [
        _unit(
            phase="p5_priority_dflash_paired_trace_v1",
            model_pair="qwen3_4b_dflash16",
            method=method,
            dataset="livecodebench",
            prompt_subset="p5_ctx_4096-40000",
            seed=0,
            lifecycle="stream",
            sampling_profile="greedy_t0",
            trainable_scope="tail_lora",
            stride=4,
            logical_delay=0,
            concurrency=concurrency,
            contention_condition=(
                "realistic_async" if method == "naive_async" else "none"
            ),
        )
        for method in ("naive_async", "tts")
        for concurrency in (1, 4)
    ]
    return ExperimentManifest(
        name="p5_priority_dflash_paired_trace_v1",
        phase="p5_priority_dflash_paired_trace_v1",
        description=(
            "Bounded paired DFlash4B tail-LoRA controller traces at 4K, 16K "
            "and 40K with concurrency 1/4. Three staged labels per request "
            "cover early, middle and late generation; L0 and TTS both use "
            "logical delay zero so staleness comes only from real GPU "
            "execution and publication timing."
        ),
        profile="local_1x96gb",
        engine_params={
            "prompt_limit": 48,
            "benchmark_repetitions": 1,
            # One trace label needs the first stride-4 candidate, the later TTS
            # barrier, and eight complete utility rounds after each arrival.
            # A 128-token DFlash request can finish before that evidence closes.
            "max_new_tokens": 512,
            "ignore_eos": True,
            "max_running_requests": 4,
            "max_total_tokens": 400000,
            "p5_context_lengths": [4096, 16384, 40000],
            "peak_tflops_per_gpu": (
                P5_RTX_PRO_6000_SERVER_DENSE_BF16_TFLOPS_PER_GPU
            ),
            "peak_tflops_basis": (
                P5_RTX_PRO_6000_SERVER_DENSE_BF16_PEAK_BASIS
            ),
            "lr": 3e-5,
            "trace_level": "full",
            "trace_capture_max_bytes": 6 * (1 << 30),
            "trace_capture_max_records_per_request": 3,
            "trace_capture_sampling": "staged",
            "trace_producer_methods": ["naive_async", "tts"],
            "warmup_prompts": 1,
            "request_timeout_s": 1800,
        },
        units=units,
    )


def p5_priority_dflash_l3_evaluation_manifest() -> ExperimentManifest:
    """Bounded phase-2 L3 labels against the frozen phase-1 transport map.

    The unit cells intentionally mirror the TTS half of
    :func:`p5_priority_dflash_paired_trace_manifest`.  The only method-level
    difference is ``lc_transport`` itself; the explicit evaluation-only flag
    lets the runtime use a phase-1 map without opening the production L3 gate.
    """
    units = [
        _unit(
            phase="p5_priority_dflash_l3_evaluation_v1",
            model_pair="qwen3_4b_dflash16",
            method="lc_transport",
            dataset="livecodebench",
            prompt_subset="p5_ctx_4096-40000",
            seed=0,
            lifecycle="stream",
            sampling_profile="greedy_t0",
            trainable_scope="tail_lora",
            stride=4,
            logical_delay=0,
            concurrency=concurrency,
            contention_condition="realistic_async",
        )
        for concurrency in (1, 4)
    ]
    return ExperimentManifest(
        name="p5_priority_dflash_l3_evaluation_v1",
        phase="p5_priority_dflash_l3_evaluation_v1",
        description=(
            "Benchmark-only phase-2 DFlash4B L3 evaluation at the exact "
            "LiveCodeBench 4K/16K/40K, concurrency 1/4, prompt, seed and "
            "staged-trace cells used by phase-1 TTS. It consumes a frozen "
            "phase-1 controller/map and cannot enable production L3 by itself."
        ),
        profile="local_1x96gb",
        engine_params={
            "prompt_limit": 48,
            "benchmark_repetitions": 1,
            "max_new_tokens": 512,
            "ignore_eos": True,
            "max_running_requests": 4,
            "max_total_tokens": 400000,
            "p5_context_lengths": [4096, 16384, 40000],
            "peak_tflops_per_gpu": (
                P5_RTX_PRO_6000_SERVER_DENSE_BF16_TFLOPS_PER_GPU
            ),
            "peak_tflops_basis": (
                P5_RTX_PRO_6000_SERVER_DENSE_BF16_PEAK_BASIS
            ),
            "lr": 3e-5,
            "trace_level": "full",
            "trace_capture_max_bytes": 6 * (1 << 30),
            "trace_capture_max_records_per_request": 3,
            "trace_capture_sampling": "staged",
            "trace_producer_methods": ["lc_transport"],
            "l3_evaluation_only": True,
            "warmup_prompts": 1,
            "request_timeout_s": 1800,
        },
        units=units,
    )


def p5_priority_dflash_smoke_manifest() -> ExperimentManifest:
    """Minimal DFlash4B orchestration and online-adaptation GPU smoke."""
    units = [
        _unit(
            phase="p5_priority_dflash_smoke_v1",
            model_pair="qwen3_4b_dflash16",
            method=method,
            dataset="livecodebench",
            prompt_subset="p5_ctx_512",
            seed=0,
            lifecycle="stream",
            sampling_profile="greedy_t0",
            trainable_scope="tail_lora",
            stride=4,
            logical_delay=0,
            concurrency=1,
            contention_condition=(
                "none" if method in ("static", "tts") else "realistic_async"
            ),
        )
        for method in ("static", "tts", "naive_async")
    ]
    return ExperimentManifest(
        name="p5_priority_dflash_smoke_v1",
        phase="p5_priority_dflash_smoke_v1",
        description=(
            "Minimal greedy Qwen3-4B DFlash-b16 tail-LoRA smoke at a "
            "512-token prefix, run before the priority experiment queue."
        ),
        profile="local_1x96gb",
        engine_params={
            "prompt_limit": 2,
            "benchmark_repetitions": 1,
            "max_new_tokens": 32,
            "ignore_eos": True,
            "max_running_requests": 1,
            "max_total_tokens": 4096,
            "p5_context_lengths": [512],
            "lr": 3e-5,
            "warmup_prompts": 0,
            "request_timeout_s": 900,
        },
        units=units,
    )


def classic_eagle_compat_manifest() -> ExperimentManifest:
    """Classic EAGLE smoke/compatibility evidence, not a Qwen3 comparison."""
    units = [
        _unit(
            phase="eagle_compat",
            model_pair="llama2_7b_eagle",
            method=method,
            dataset=dataset,
            prompt_subset="compat32",
            seed=0,
            lifecycle="stream",
            trainable_scope="output_residual",
            stride=4,
            concurrency=1,
            contention_condition=(
                "none" if method in ("static", "tts") else "realistic_async"
            ),
        )
        for dataset in ("math500", "mt_bench")
        for method in P5_METHODS
    ]
    return ExperimentManifest(
        name="classic_eagle_tail_compat",
        phase="eagle_compat",
        description=(
            "Classic Llama-2 EAGLE single-chain compatibility side table; "
            "reported separately from the shared-target Qwen3 matrix."
        ),
        profile="local_1x96gb",
        engine_params={
            "prompt_limit": 32,
            "benchmark_repetitions": 5,
            "max_new_tokens": 256,
            "max_running_requests": 16,
            "warmup_prompts": 4,
            "request_timeout_s": 900,
        },
        units=units,
    )


def load_tune_manifest() -> ExperimentManifest:
    """Qwen3-4B saturation sweep used by the dual-profile selector."""
    units = [
        _unit(
            phase="load_tune",
            model_pair="qwen3_4b_dspark7",
            method=method,
            dataset="alpaca",
            prompt_subset="load_tune128",
            seed=0,
            stride=4,
            concurrency=concurrency,
            contention_condition=(
                "none" if method == "static" else "realistic_async"
            ),
        )
        for concurrency in (1, 2, 4, 8, 16, 32, 48)
        for method in ("static", "tts", "naive_async", "lc_gate")
    ]
    return ExperimentManifest(
        name="load_tune_gpu_qwen3_4b",
        phase="load_tune",
        description=(
            "Continuous-load Qwen3-4B DSpark saturation sweep for "
            "throughput-first and latency-SLO-first profile selection"
        ),
        units=units,
        engine_params={
            "prompt_limit": 128,
            "max_new_tokens": 256,
            "warmup_prompts": 8,
            "benchmark_repetitions": 5,
            "max_running_requests": 48,
            "speculative_num_draft_tokens": 8,
            "tensor_parallel_size": 1,
            "trace_level": "light",
        },
        profile="auto_gpu_preflight",
    )


# ---------------------------------------------------------------------------
# Local CPU smoke manifest
# ---------------------------------------------------------------------------


def smoke_manifest(controller_artifact_path: str | None = None) -> ExperimentManifest:
    methods = ["static", "sync_fresh", "tts", "naive_async", "onlinespec_ogd",
               "onlinespec_opt", "onlinespec_ens", "oracle_current"]
    if controller_artifact_path:
        methods += ["lc_gate", "lc_damp", "lc_transport"]
    units = [
        _unit(
            phase="smoke",
            model_pair="toy_markov4",
            method=m,
            dataset="markov4_world",
            seed=0,
            stride=4,
            logical_delay=2 if m not in ("static", "sync_fresh") else 0,
        )
        for m in methods
    ]
    return ExperimentManifest(
        name="smoke_cpu",
        phase="smoke",
        description="local CPU smoke: all runnable methods on the toy "
        "Markov world through the full artifact chain",
        units=units,
        engine_params={
            "num_requests": 3,
            "max_rounds": 16,
            "max_new_tokens": 32,
            "lr": 1e-3,
            "controller_artifact_path": controller_artifact_path,
        },
        profile="cpu_reference",
    )


ALL_MANIFESTS = {
    "p0": p0_manifest,
    "p1": p1_manifest,
    "p2": p2_manifest,
    "p3": p3_manifest,
    "p4": p4_manifest,
    "p5": p5_manifest,
    "p5_cross_backend": p5_cross_backend_tail_manifest,
    "p5_cross_backend_trace": p5_cross_backend_trace_manifest,
    "p5_priority_dflash_0_40k": p5_priority_dflash_0_40k_manifest,
    "p5_priority_dflash_stride_screen": (
        p5_priority_dflash_stride_screen_manifest
    ),
    "p5_priority_dflash_paired_trace": (
        p5_priority_dflash_paired_trace_manifest
    ),
    "p5_priority_dflash_l3_evaluation": (
        p5_priority_dflash_l3_evaluation_manifest
    ),
    "p5_priority_dflash_smoke": p5_priority_dflash_smoke_manifest,
    "eagle_compat": classic_eagle_compat_manifest,
    "load_tune": load_tune_manifest,
}


def write_all(manifest_dir: str | Path) -> dict[str, str]:
    out = {}
    manifest_dir = Path(manifest_dir)
    for name, builder in ALL_MANIFESTS.items():
        manifest = builder()
        path = manifest_dir / name / f"{manifest.name}.json"
        digest = manifest.write(path)
        out[name] = digest
    return out
