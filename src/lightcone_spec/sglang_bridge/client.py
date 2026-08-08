"""GPU unit driver (spec 8, 9.2 serve).

Launches the pinned SGLang fork engine with the backend-neutral
speculative-adaptation config, streams benchmark prompts through it,
collects fork telemetry and converts it into the spec-11 artifact rows.
Runs only on declared CUDA hardware with the fork installed; it never
fabricates results elsewhere (the executor guards this).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path

from lightcone_spec.exit_codes import LightconeError, ResourceSkip, RuntimeGpuFailure
from lightcone_spec.locking.lockfile import load_lockfile
from lightcone_spec.orchestration.units import RunUnit
from lightcone_spec.config.schema import MODEL_PAIRS, pair_thinking_config
from lightcone_spec.sglang_bridge.bank import resolve_adapter_row_capacity


class _GpuSystemSampler:
    """Low-rate NVML sampler; it never touches or synchronizes CUDA streams."""

    def __init__(
        self,
        unit: RunUnit,
        device_indices: list[int] | None = None,
        interval_s: float = 0.5,
    ):
        self.unit = unit
        self.device_indices = device_indices or [0]
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._done = threading.Event()
        self._thread = None
        self._pynvml = None
        self._handles: list[tuple[int, object]] = []
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
            for device_index in self.device_indices:
                token = visible[device_index] if device_index < len(visible) else None
                if token and not token.isdigit():
                    handle = pynvml.nvmlDeviceGetHandleByUUID(token)
                else:
                    physical_index = (
                        int(token) if token and token.isdigit() else device_index
                    )
                    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
                self._handles.append((device_index, handle))
            self._thread = threading.Thread(
                target=self._run, name="lightcone-nvml", daemon=True
            )
            self._thread.start()
        except Exception:
            self._pynvml = None

    def _run(self) -> None:
        last = time.monotonic()
        while not self._done.is_set():
            now = time.monotonic()
            for device_index, handle in self._handles:
                try:
                    mem = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                    try:
                        power = self._pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except Exception:
                        power = 0.0
                    self.samples.append(
                        {
                            "timestamp_us": time.time() * 1e6,
                            "gpu_index": device_index,
                            "hbm_used_bytes": int(mem.used),
                            "sm_occupancy": None,
                            "gpu_utilization": float(util.gpu),
                            "power_watts": float(power),
                            "energy_joules_delta": float(
                                power * max(now - last, 0.0)
                            ),
                            "main_stream_active": None,
                            "side_stream_active": None,
                            "stream_contention_class": self.unit.contention_condition,
                            "sync_us_delta": None,
                            "sample_source": "nvml",
                            "activity_provenance": "not_observed",
                            "contention_provenance": "declared_not_observed",
                            "sync_provenance": "not_observed",
                        }
                    )
                except Exception:
                    pass
            last = now
            self._done.wait(self.interval_s)

    def stop(self) -> list[dict]:
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        return self.samples


def _runtime_request_id(
    *,
    run_id: str,
    repetition: int,
    index: int,
    sample_id: str,
    sampling_seed: int | None = None,
) -> str:
    """Encode both the exact checkpoint and its stable source-prompt group."""
    sample = str(sample_id)
    source = re.sub(r":ctx-\d+$", "", sample)
    checkpoint = hashlib.sha256(sample.encode("utf-8")).hexdigest()
    prompt_group = hashlib.sha256(source.encode("utf-8")).hexdigest()
    seed_component = (
        "" if sampling_seed is None else f"-s{int(sampling_seed):016x}"
    )
    return (
        f"lightcone-g{checkpoint}-p{prompt_group}-"
        f"{run_id}{seed_component}-{repetition}-{index}"
    )


def _paired_sampling_seed(
    *, sample_id: str, repetition: int, experiment_seed: int
) -> int:
    """Stable request seed shared by every method for one paired sample."""

    material = (
        f"lightcone-sampling-v1\0{int(experiment_seed)}\0"
        f"{str(sample_id)}\0{int(repetition)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _run_cancellation_probe(engine, prompt: str, sampling_params: dict, run_id: str) -> None:
    """Start, abort and drain one real streaming request before measurement."""
    rid = f"lightcone-cancel-{run_id}"

    async def probe() -> None:
        generator = await engine.async_generate(
            prompt=prompt,
            sampling_params={
                **sampling_params,
                "max_new_tokens": 512,
                "sampling_seed": _paired_sampling_seed(
                    sample_id=rid, repetition=-2, experiment_seed=0
                ),
            },
            stream=True,
            rid=rid,
        )
        try:
            await asyncio.wait_for(generator.__anext__(), timeout=60)
        except Exception as exc:
            raise RuntimeGpuFailure(
                f"cancellation probe did not start a streaming request: {exc}"
            ) from exc
        engine.tokenizer_manager.abort_request(rid=rid)
        try:
            while True:
                await asyncio.wait_for(generator.__anext__(), timeout=30)
        except (StopAsyncIteration, ValueError):
            return
        except asyncio.TimeoutError as exc:
            raise RuntimeGpuFailure(
                "cancellation probe was not drained within 30 seconds"
            ) from exc

    engine.loop.run_until_complete(probe())


def _run_streaming_pool(
    engine,
    jobs: list[dict],
    sampling_params: dict,
    *,
    run_id: str,
    concurrency: int,
    timeout_s: float = 600.0,
    request_kind: str = "measured",
) -> list[dict]:
    """Keep a bounded request pool full without chunk-boundary bubbles."""

    if timeout_s <= 0:
        raise RuntimeGpuFailure("request_timeout_s must be positive")
    if concurrency <= 0:
        raise RuntimeGpuFailure("concurrency must be positive")
    if request_kind not in ("measured", "warmup"):
        raise RuntimeGpuFailure(f"unsupported request kind: {request_kind}")

    async def one(index: int, job: dict) -> dict:
        prompt = job["prompt"]
        repetition = int(job["repetition"])
        sampling_seed = job.get("sampling_seed")
        rid = (
            f"lightcone-warmup-{run_id}-{index}"
            if request_kind == "warmup"
            else _runtime_request_id(
                run_id=run_id,
                repetition=repetition,
                index=index,
                sample_id=prompt["sample_id"],
                sampling_seed=(
                    int(sampling_seed) if sampling_seed is not None else None
                ),
            )
        )
        request_start = time.perf_counter()
        request_input = (
            {"input_ids": prompt["input_ids"]}
            if prompt.get("input_ids") is not None
            else {"prompt": prompt["prompt"]}
        )
        request_sampling_params = dict(sampling_params)
        if sampling_seed is not None:
            request_sampling_params["sampling_seed"] = int(sampling_seed)
        generator = await engine.async_generate(
            **request_input,
            sampling_params=request_sampling_params,
            stream=True,
            rid=rid,
        )
        previous_tokens = 0
        previous_time = None
        first_token_time = None
        itl_ms: list[float] = []
        final = None
        async for output in generator:
            now = time.perf_counter()
            final = output
            meta = output.get("meta_info", {}) if isinstance(output, dict) else {}
            completion_tokens = int(meta.get("completion_tokens", 0) or 0)
            new_tokens = max(completion_tokens - previous_tokens, 0)
            if new_tokens > 0:
                if first_token_time is None:
                    first_token_time = now
                    # The first emitted token is accounted for by TTFT.  Any
                    # additional tokens in the same speculative burst arrive
                    # at the same observable timestamp and therefore have
                    # zero inter-token latency.
                    itl_ms.extend([0.0] * max(new_tokens - 1, 0))
                else:
                    # Preserve the actual streaming arrival process.  The
                    # first token in a later chunk waited for the inter-chunk
                    # interval; co-emitted tokens did not.  Averaging the
                    # interval across a speculative burst fabricates token
                    # timestamps and suppresses tail latency.
                    if previous_time is None:  # defensive invariant
                        raise RuntimeGpuFailure(
                            "streaming ITL state lost its previous timestamp"
                        )
                    itl_ms.append(1000.0 * (now - previous_time))
                    itl_ms.extend([0.0] * max(new_tokens - 1, 0))
                previous_time = now
                previous_tokens = completion_tokens
        if final is None:
            raise RuntimeGpuFailure(f"streaming request {rid} returned no output")
        request_end = time.perf_counter()
        return {
            "output": final,
            "itl_ms": itl_ms,
            "request_wall_s": request_end - request_start,
            "ttft_ms": (
                1000.0 * (first_token_time - request_start)
                if first_token_time is not None
                else None
            ),
            "repetition": repetition,
            "prompt": prompt,
        }

    async def run_all():
        next_index = 0
        results: list[dict | None] = [None] * len(jobs)

        async def worker() -> None:
            nonlocal next_index
            while next_index < len(jobs):
                index = next_index
                next_index += 1
                try:
                    results[index] = await asyncio.wait_for(
                        one(index, jobs[index]), timeout=timeout_s
                    )
                except asyncio.TimeoutError as exc:
                    raise RuntimeGpuFailure(
                        f"streaming request {index} exceeded {timeout_s:g} seconds"
                    ) from exc

        workers = [
            asyncio.create_task(worker())
            for _ in range(min(concurrency, max(len(jobs), 1)))
        ]
        try:
            await asyncio.gather(*workers)
        except BaseException:
            for task in workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        return [item for item in results if item is not None]

    return engine.loop.run_until_complete(run_all())


def _run_streaming_chunk(
    engine,
    chunk: list[dict],
    sampling_params: dict,
    *,
    repetition: int,
    run_id: str,
    timeout_s: float = 600.0,
) -> list[dict]:
    """Compatibility wrapper for tests and callers using one bounded chunk."""
    return _run_streaming_pool(
        engine,
        [{"prompt": prompt, "repetition": repetition} for prompt in chunk],
        sampling_params,
        run_id=run_id,
        concurrency=max(len(chunk), 1),
        timeout_s=timeout_s,
    )


def _pair_server_args(
    pair: dict,
    *,
    target_path: str,
    drafter_path: str,
    adaptation_config_path: str | None,
    num_draft_tokens: int,
    tensor_parallel_size: int,
    random_seed: int,
    adaptation_reserve_mb: int = 0,
) -> dict:
    """Translate a declared model-pair capability into SGLang arguments."""
    algorithm = pair["speculative_algorithm"]
    args = {
        "model_path": target_path,
        "speculative_algorithm": algorithm,
        "speculative_draft_model_path": drafter_path,
        "speculative_num_draft_tokens": int(num_draft_tokens),
        "tp_size": int(tensor_parallel_size),
        "random_seed": int(random_seed),
    }
    if adaptation_config_path is not None:
        args["speculative_adaptation_config"] = adaptation_config_path
        args["speculative_adaptation_reserve_mb"] = int(adaptation_reserve_mb)
    if algorithm in ("EAGLE", "EAGLE3"):
        # The first implementation is a single linear chain.  Tree/multi-layer
        # EAGLE needs a different proposal-q/version contract and fails closed
        # by construction because these values are not caller-overridable here.
        # SGLang's topk=1 invariant is
        # ``num_draft_tokens == num_steps + 1``: draft-extend samples step zero,
        # then the recurrent draft loop samples the remaining steps.  Keep one
        # desired-depth source of truth so manifests, memory sizing and runtime
        # telemetry cannot silently disagree by one token.
        if int(num_draft_tokens) < 2:
            raise ValueError("linear EAGLE requires at least two draft tokens")
        args["speculative_eagle_topk"] = 1
        args["speculative_num_steps"] = int(num_draft_tokens) - 1
        args["enable_multi_layer_eagle"] = False
        args["speculative_use_rejection_sampling"] = True
    elif algorithm == "DFLASH":
        # DFlash proposes a deterministic chain.  Thresholds below one alter
        # its stochastic target distribution and are forbidden for LightCone.
        args["speculative_accept_threshold_single"] = 1.0
        args["speculative_accept_threshold_acc"] = 1.0
    thinking = pair_thinking_config(pair)
    if thinking["enable_thinking"]:
        # Bake thinking into the server defaults so chat-template and generate
        # paths agree even when a caller forgets per-request kwargs.
        if thinking["reasoning_parser"]:
            args["reasoning_parser"] = thinking["reasoning_parser"]
        args["default_chat_template_kwargs"] = dict(
            thinking["chat_template_kwargs"] or {"enable_thinking": True}
        )
    return args


def _server_args(unit: RunUnit, engine_params: dict, adaptation_config_path: str):
    pair = MODEL_PAIRS[unit.model_pair]
    roots = engine_params.get("model_roots", {})
    args = _pair_server_args(
        pair,
        target_path=roots.get(pair["target"], pair["target"]),
        drafter_path=roots.get(pair["drafter"], pair["drafter"]),
        adaptation_config_path=adaptation_config_path,
        num_draft_tokens=engine_params.get(
            "speculative_num_draft_tokens",
            pair["default_num_draft_tokens"],
        ),
        tensor_parallel_size=engine_params.get("tensor_parallel_size", 1),
        random_seed=unit.seed,
        adaptation_reserve_mb=int(
            engine_params.get("calibrated_reserve_mb", 0)
        ),
    )
    # engine_params may force thinking on/off for a single unit; otherwise the
    # model-pair default from _pair_server_args already applied.
    if "enable_thinking" in engine_params:
        if bool(engine_params["enable_thinking"]):
            thinking = pair_thinking_config(pair)
            args["reasoning_parser"] = (
                engine_params.get("reasoning_parser")
                or thinking["reasoning_parser"]
            )
            args["default_chat_template_kwargs"] = {"enable_thinking": True}
        else:
            args.pop("reasoning_parser", None)
            args.pop("default_chat_template_kwargs", None)
    if unit.sampling_profile != "greedy_t0":
        # SamplingParams.sampling_seed is consumed only when SGLang builds
        # deterministic request RNG state; setting the field alone is not a
        # stochastic exactness guarantee.
        args["enable_deterministic_inference"] = True
    max_running_requests = max(
        int(unit.concurrency),
        int(engine_params.get("max_running_requests", 48)),
    )
    explicit_graph_max = engine_params.get("cuda_graph_max_bs_decode")
    adapter_row_capacity = resolve_adapter_row_capacity(
        max_running_requests=max_running_requests,
        cuda_graph_max_bs_decode=(
            None if explicit_graph_max is None else int(explicit_graph_max)
        ),
    )
    declared_capacity = engine_params.get("adapter_row_capacity")
    if (
        declared_capacity is not None
        and int(declared_capacity) != adapter_row_capacity
    ):
        raise ValueError(
            "materialized adapter_row_capacity no longer matches the server "
            f"request/graph contract: {declared_capacity} != {adapter_row_capacity}"
        )
    args.update({
        "max_running_requests": max_running_requests,
        # Do not let SGLang expand an otherwise-unspecified graph cap to its
        # hardware-wide default after preflight has sized a smaller tail bank.
        "cuda_graph_max_bs_decode": (
            adapter_row_capacity
            if explicit_graph_max is None
            else int(explicit_graph_max)
        ),
        "enable_metrics": bool(engine_params.get("enable_metrics", True)),
        "enable_mfu_metrics": bool(engine_params.get("enable_mfu_metrics", True)),
        "enable_metrics_for_all_schedulers": bool(
            engine_params.get("enable_metrics_for_all_schedulers", True)
        ),
        "enable_layerwise_nvtx_marker": bool(
            engine_params.get("profile_steps", 0)
            and engine_params.get("enable_layerwise_nvtx_marker", True)
        ),
    })
    for name in (
        "max_total_tokens",
        "mem_fraction_static",
        "attention_backend",
        "cuda_graph_backend_decode",
        "cuda_graph_backend_prefill",
        "cuda_graph_max_bs_decode",
        "cuda_graph_max_bs_prefill",
        "disable_cuda_graph",
    ):
        if engine_params.get(name) is not None:
            args[name] = engine_params[name]
    return args


def _adaptation_info_record(state: dict) -> dict:
    """Read the generic scheduler payload, with schema-v1 DSpark fallback."""
    return (
        state.get("speculative_adaptation_info_record")
        or state.get("dspark_info_record")
        or {}
    )


def _write_memory_calibration(
    engine_params: dict,
    *,
    baseline_bytes: int,
    peak_bytes: int,
) -> None:
    """Atomically publish one GPU/model-signature warmup calibration."""
    path_value = engine_params.get("memory_calibration_path")
    identity = engine_params.get("memory_calibration_identity")
    identity_sha = engine_params.get("memory_calibration_sha256")
    if not path_value or not identity or not identity_sha:
        return
    observed = max(int(peak_bytes) - int(baseline_bytes), 0)
    observed_mb = math.ceil(observed / (1 << 20))
    recommended_mb = max(
        int(engine_params.get("calibrated_reserve_mb", 0)),
        math.ceil(observed_mb * float(engine_params.get("calibration_safety_factor", 1.2))),
    )
    record = {
        "schema_version": 2,
        "identity_sha256": identity_sha,
        "identity": identity,
        "baseline_allocated_bytes": int(baseline_bytes),
        "warmup_peak_allocated_bytes": int(peak_bytes),
        "observed_peak_delta_bytes": observed,
        "recommended_reserve_mb": recommended_mb,
        "updated_unix_s": time.time(),
    }
    body = json.dumps(record, sort_keys=True, indent=2) + "\n"
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(body)
    os.replace(tmp, path)
    run_copy = Path(engine_params["adaptation_config_path"]).parent / "memory-calibration.json"
    run_copy.write_text(body)


def _server_perf_snapshot(engine) -> tuple[dict, dict]:
    """Read cumulative per-GPU work counters outside the measured hot path."""
    try:
        info = engine.get_server_info()
    except Exception:
        return {}, {
            "flops": 0.0,
            "read_bytes": 0.0,
            "write_bytes": 0.0,
            "decode_moments": [0.0] * 6,
            "prefill_tokens": 0,
            "prefill_busy_us": 0,
            "fallbacks": 0,
            "retractions": 0,
            "peak_running": 0,
            "peak_queue": 0,
        }
    states = info.get("internal_states") or []
    counters = []
    for state in states:
        perf = state.get("estimated_perf_cumulative")
        if perf:
            counters.append(perf)
    n = max(len(counters), 1)
    fallbacks = retractions = peak_running = peak_queue = 0
    decode_moments = [0.0] * 6
    prefill_tokens = prefill_busy_us = 0
    for state in states:
        record = _adaptation_info_record(state)
        adaptation = record.get("adaptation") or {}
        fallbacks += sum(
            int(value)
            for value in (adaptation.get("fallback_counts") or {}).values()
        )
        load = state.get("load_cumulative") or {}
        retractions += int(load.get("retracted_requests", 0) or 0)
        peak_running = max(peak_running, int(load.get("peak_running_requests", 0) or 0))
        peak_queue = max(peak_queue, int(load.get("peak_queue_requests", 0) or 0))
        moments = load.get("decode_moments") or ()
        if len(moments) == len(decode_moments):
            for index, value in enumerate(moments):
                decode_moments[index] += float(value)
        prefill_tokens += int(load.get("total_prefill_uncached_tokens", 0) or 0)
        prefill_busy_us += int(load.get("total_prefill_busy_us", 0) or 0)
    return info, {
        "flops": sum(float(x.get("flops_per_gpu", 0.0)) for x in counters) / n,
        "read_bytes": sum(
            float(x.get("read_bytes_per_gpu", 0.0)) for x in counters
        )
        / n,
        "write_bytes": sum(
            float(x.get("write_bytes_per_gpu", 0.0)) for x in counters
        )
        / n,
        "decode_moments": decode_moments,
        "prefill_tokens": prefill_tokens,
        "prefill_busy_us": prefill_busy_us,
        "fallbacks": fallbacks,
        "retractions": retractions,
        "peak_running": peak_running,
        "peak_queue": peak_queue,
    }


def _performance_evidence(
    before: dict,
    after: dict,
    wall_s: float,
    peak_tflops_per_gpu: float | None,
    offered_concurrency: int,
) -> dict:
    elapsed = max(float(wall_s), 1e-9)
    flops = max(float(after["flops"]) - float(before["flops"]), 0.0)
    tflops = flops / elapsed / 1e12
    peak = float(peak_tflops_per_gpu or 0.0)
    before_moments = before.get("decode_moments") or [0.0] * 6
    after_moments = after.get("decode_moments") or [0.0] * 6
    if len(before_moments) != 6 or len(after_moments) != 6:
        decode = [0.0] * 6
    else:
        decode = [
            max(float(end) - float(start), 0.0)
            for start, end in zip(before_moments, after_moments)
        ]
    steps, batch_sum, scheduler_span_us, batch_sq_sum, batch_time_sum, generated = decode
    step_mean = batch_sum / steps if steps > 0.0 else None
    time_mean = (
        batch_time_sum / scheduler_span_us if scheduler_span_us > 0.0 else None
    )
    batch_variance = (
        max(batch_sq_sum / steps - step_mean * step_mean, 0.0)
        if steps > 0.0 and step_mean is not None
        else None
    )
    return {
        "estimated_tflops_per_gpu": tflops,
        "estimated_mfu": tflops / peak if peak > 0.0 else None,
        "estimated_read_gbps_per_gpu": max(
            float(after["read_bytes"]) - float(before["read_bytes"]), 0.0
        )
        / elapsed
        / 1e9,
        "estimated_write_gbps_per_gpu": max(
            float(after["write_bytes"]) - float(before["write_bytes"]), 0.0
        )
        / elapsed
        / 1e9,
        "peak_tflops_per_gpu": peak if peak > 0.0 else None,
        "decode_step_count": int(steps),
        "decode_batch_size_step_mean": step_mean,
        "decode_batch_size_time_mean": time_mean,
        "decode_batch_size_std": (
            math.sqrt(batch_variance) if batch_variance is not None else None
        ),
        "decode_batch_fill_ratio": (
            min(time_mean / max(int(offered_concurrency), 1), 1.0)
            if time_mean is not None
            else None
        ),
        # This is the interval between scheduler decode launches, not CUDA
        # device-busy time.  The explicit name prevents it from being
        # misreported as kernel utilization.
        "decode_scheduler_span_s": scheduler_span_us / 1e6,
        "decode_generated_tps_scheduler_span": (
            generated / (scheduler_span_us / 1e6)
            if scheduler_span_us > 0.0
            else None
        ),
        "prefill_uncached_tokens": max(
            int(after.get("prefill_tokens", 0))
            - int(before.get("prefill_tokens", 0)),
            0,
        ),
        "prefill_busy_s": max(
            int(after.get("prefill_busy_us", 0))
            - int(before.get("prefill_busy_us", 0)),
            0,
        )
        / 1e6,
        "adaptation_fallback_count": max(
            int(after["fallbacks"]) - int(before["fallbacks"]), 0
        ),
        "kv_retracted_requests": max(
            int(after["retractions"]) - int(before["retractions"]), 0
        ),
        "peak_running_requests": int(after["peak_running"]),
        "peak_queue_requests": int(after["peak_queue"]),
    }


def _require_configured_mfu_evidence(
    evidence: dict, *, peak_tflops_per_gpu: float | None, location: str
) -> None:
    """Fail closed when a run explicitly claims an MFU denominator.

    Runs without a configured denominator remain valid and report ``null``
    MFU.  Once a denominator is bound, however, missing scheduler FLOP/decode
    counters must not silently become a plausible zero-utilization result.
    """

    peak = float(peak_tflops_per_gpu or 0.0)
    if peak <= 0.0:
        return
    tflops = evidence.get("estimated_tflops_per_gpu")
    mfu = evidence.get("estimated_mfu")
    if (
        tflops is None
        or mfu is None
        or not math.isfinite(float(tflops))
        or not math.isfinite(float(mfu))
        or float(tflops) <= 0.0
        or float(mfu) <= 0.0
        or int(evidence.get("decode_step_count") or 0) <= 0
    ):
        raise RuntimeGpuFailure(
            f"{location}: configured peak_tflops_per_gpu={peak:g} requires "
            "positive target-model FLOP and decode-step evidence"
        )


def run_unit_via_sglang(unit: RunUnit, engine_params: dict, run_id: str) -> dict:
    """Normalize unexpected engine/RPC errors into immutable runtime failures.

    Declared LightCone outcomes retain their stable type/exit code.  Process
    control exceptions inherit from ``BaseException`` and are deliberately not
    intercepted.
    """
    try:
        return _run_unit_via_sglang_impl(unit, engine_params, run_id)
    except LightconeError:
        raise
    except Exception as exc:
        raise RuntimeGpuFailure(
            f"unexpected SGLang engine/request failure ({type(exc).__name__}): {exc}"
        ) from exc


def _run_unit_via_sglang_impl(unit: RunUnit, engine_params: dict, run_id: str) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeGpuFailure("CUDA unavailable for real model-pair unit")
    try:
        import sglang
        from sglang.srt.entrypoints.engine import Engine
    except ImportError as exc:
        raise RuntimeGpuFailure(f"pinned SGLang fork not importable: {exc}") from exc

    lock_path = engine_params.get("lockfile_path")
    if not lock_path:
        raise RuntimeGpuFailure("GPU units require lockfile_path in engine_params")
    lock = load_lockfile(lock_path)
    pair = MODEL_PAIRS[unit.model_pair]
    lock.find_snapshot(pair["target"])
    lock.find_snapshot(pair["drafter"])

    adaptation_config_path = engine_params.get("adaptation_config_path")
    if not adaptation_config_path:
        raise RuntimeGpuFailure(
            "GPU units require adaptation_config_path in engine_params"
        )

    # Resolve and load the locked dataset before allocating model weights or
    # KV cache. Missing datasets must fail without paying for an Engine start.
    prompts = _load_prompts(unit, engine_params, lock)
    args = _server_args(unit, engine_params, adaptation_config_path)
    engine = Engine(**args)
    system_sampler = None
    system_samples: list[dict] = []
    profile_steps = max(0, int(engine_params.get("profile_steps", 0)))
    profile_output_dir = (
        Path(engine_params["profile_output_dir"])
        if profile_steps and engine_params.get("profile_output_dir")
        else None
    )
    try:
        startup_server_info = engine.get_server_info()
        fallback_reason = startup_server_info.get(
            "speculative_adaptation_fallback_reason"
        ) or getattr(
            getattr(engine, "server_args", None),
            "speculative_adaptation_fallback_reason",
            None,
        )
        if fallback_reason:
            # Preserve the startup evidence before raising. The executor will
            # mark this unit failed_runtime/unsupported and emit empty tables;
            # no prompt is sent and no empty telemetry can become "complete".
            evidence_path = Path(adaptation_config_path).parent / "server-info.json"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(
                json.dumps(startup_server_info, indent=2, sort_keys=True) + "\n"
            )
            raise RuntimeGpuFailure(
                "speculative adaptation is unsupported by this SGLang build; "
                f"the engine started target-only (fallback_reason={fallback_reason})"
            )
        sampling_params = {
            "temperature": 0.0 if unit.sampling_profile == "greedy_t0" else 1.0,
            "top_p": 1.0,
            "max_new_tokens": engine_params.get("max_new_tokens", 32768),
            "ignore_eos": bool(engine_params.get("ignore_eos", False)),
        }
        warmup_sampling_params = sampling_params
        if engine_params.get("p5_continuous_prefix_windows"):
            warmup_tokens = int(
                engine_params.get("continuous_warmup_new_tokens", 128)
            )
            if warmup_tokens <= 0:
                raise RuntimeGpuFailure(
                    "continuous_warmup_new_tokens must be positive"
                )
            warmup_sampling_params = {
                **sampling_params,
                "max_new_tokens": min(
                    warmup_tokens, int(sampling_params["max_new_tokens"])
                ),
            }
        concurrency = max(1, int(unit.concurrency))
        prompt_groups: dict[int | None, list[dict]] = {}
        for prompt in prompts:
            context_length = _p5_prompt_context_length(prompt["sample_id"])
            prompt_groups.setdefault(context_length, []).append(prompt)
        # At least one real request is required to calibrate transient tensors
        # and warm CUDA graphs before any reported measurement. Grouped P5
        # units warm every exact context bucket at the offered load; otherwise
        # the first repetition at each new length would include cold setup.
        warmup_count = max(
            1,
            int(engine_params.get("warmup_prompts", 1)),
            concurrency,
        )
        if warmup_count:
            for group_index, warmup_group in enumerate(prompt_groups.values()):
                warmup = warmup_group[: min(warmup_count, len(warmup_group))]
                generated = _run_streaming_pool(
                    engine,
                    [
                        {
                            "prompt": prompt,
                            "repetition": 0,
                            "sampling_seed": _paired_sampling_seed(
                                sample_id=prompt["sample_id"],
                                repetition=-1,
                                experiment_seed=unit.seed,
                            ),
                        }
                        for prompt in warmup
                    ],
                    warmup_sampling_params,
                    run_id=f"{run_id}-wg{group_index}",
                    concurrency=min(concurrency, len(warmup)),
                    timeout_s=float(engine_params.get("request_timeout_s", 600.0)),
                    request_kind="warmup",
                )
                if len(generated) != len(warmup):
                    raise RuntimeGpuFailure(
                        "SGLang warmup returned a different result count: "
                        f"{len(generated)} != {len(warmup)}"
                    )
            torch.cuda.synchronize()
        if engine_params.get("cancellation_smoke", False):
            _run_cancellation_probe(
                engine,
                "Continue emitting numbered tokens until explicitly stopped.",
                sampling_params,
                run_id,
            )
        torch.cuda.reset_peak_memory_stats()
        system_sampler = _GpuSystemSampler(
            unit, device_indices=list(range(torch.cuda.device_count()))
        )
        if profile_steps:
            if profile_output_dir is None:
                raise RuntimeGpuFailure(
                    "profile_steps requires a run-local profile_output_dir"
                )
            profile_output_dir.mkdir(parents=True, exist_ok=True)
            engine.start_profile(
                output_dir=str(profile_output_dir),
                num_steps=profile_steps,
                activities=list(engine_params.get("profile_activities", ["CPU", "GPU"])),
                with_stack=bool(engine_params.get("profile_with_stack", False)),
                record_shapes=bool(engine_params.get("profile_record_shapes", True)),
                profile_prefix=f"lightcone-{unit.method}-{run_id}",
            )
        summaries = []
        _server_info_before, perf_before = _server_perf_snapshot(engine)
        t0 = time.perf_counter()
        repetitions = max(1, int(engine_params.get("benchmark_repetitions", 1)))
        # Throughput and MFU must be context-resolved.  Keep one bounded pool
        # continuously full across prompts and repetitions *within* each exact
        # context bucket, then allow the single scientifically necessary idle
        # boundary before changing prefix length.  Mixing 4K and 16K in one
        # wall-clock denominator would copy a blended goodput/MFU into both
        # cells and make the P5 comparison invalid.
        streamed = []
        for group_index, prompt_group in enumerate(prompt_groups.values()):
            jobs = [
                {
                    "prompt": prompt,
                    "repetition": repetition,
                    "sampling_seed": _paired_sampling_seed(
                        sample_id=prompt["sample_id"],
                        repetition=repetition,
                        experiment_seed=unit.seed,
                    ),
                }
                for repetition in range(repetitions)
                for prompt in prompt_group
            ]
            _group_info_before, group_perf_before = _server_perf_snapshot(engine)
            group_t0 = time.perf_counter()
            group_streamed = _run_streaming_pool(
                engine,
                jobs,
                sampling_params,
                run_id=f"{run_id}-cg{group_index}",
                concurrency=concurrency,
                timeout_s=float(engine_params.get("request_timeout_s", 600.0)),
            )
            group_wall = time.perf_counter() - group_t0
            _group_info_after, group_perf_after = _server_perf_snapshot(engine)
            if len(group_streamed) != len(jobs):
                raise RuntimeGpuFailure(
                    "SGLang context request pool returned a different result "
                    f"count: {len(group_streamed)} != {len(jobs)}"
                )
            group_tokens = sum(
                int(
                    (
                        item["output"].get("meta_info", {})
                        if isinstance(item["output"], dict)
                        else {}
                    ).get("completion_tokens", 0)
                    or 0
                )
                for item in group_streamed
            )
            group_evidence = _performance_evidence(
                group_perf_before,
                group_perf_after,
                group_wall,
                engine_params.get("peak_tflops_per_gpu"),
                concurrency,
            )
            _require_configured_mfu_evidence(
                group_evidence,
                peak_tflops_per_gpu=engine_params.get("peak_tflops_per_gpu"),
                location=f"context group {group_index}",
            )
            for item in group_streamed:
                item["measurement_wall_s"] = group_wall
                item["measurement_total_tokens"] = group_tokens
                item["performance_evidence"] = group_evidence
            streamed.extend(group_streamed)
        wall = time.perf_counter() - t0
        server_info_after, perf_after = _server_perf_snapshot(engine)
        for state in server_info_after.get("internal_states") or []:
            record = _adaptation_info_record(state)
            adaptation = record.get("adaptation") or {}
            if adaptation and not adaptation.get("telemetry_flushed", False):
                raise RuntimeGpuFailure(
                    "adaptation telemetry did not flush within 120 seconds"
                )
            if int(adaptation.get("telemetry_error_count", 0) or 0):
                raise RuntimeGpuFailure(
                    "adaptation telemetry materialization failed: "
                    f"{adaptation.get('telemetry_last_error') or 'unknown error'}"
                )
        perf_evidence = _performance_evidence(
            perf_before,
            perf_after,
            wall,
            engine_params.get("peak_tflops_per_gpu"),
            concurrency,
        )
        _require_configured_mfu_evidence(
            perf_evidence,
            peak_tflops_per_gpu=engine_params.get("peak_tflops_per_gpu"),
            location="complete measured unit",
        )
        (Path(adaptation_config_path).parent / "server-info.json").write_text(
            json.dumps(server_info_after, indent=2, sort_keys=True) + "\n"
        )
        for stream_result in streamed:
            prompt = stream_result["prompt"]
            repetition = int(stream_result["repetition"])
            out = stream_result["output"]
            meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
            summaries.append(
                {
                    "sample_id": (
                        f"{prompt['sample_id']}:repeat-{repetition}"
                        if repetitions > 1
                        else prompt["sample_id"]
                    ),
                    "runtime_request_id": meta.get("id"),
                    "output": out["text"] if isinstance(out, dict) else str(out),
                    "meta": meta,
                    "request_wall_s": stream_result["request_wall_s"],
                    "run_wall_s": stream_result["measurement_wall_s"],
                    "run_total_tokens": stream_result[
                        "measurement_total_tokens"
                    ],
                    "performance_evidence": stream_result[
                        "performance_evidence"
                    ],
                    "offered_concurrency": concurrency,
                    "repetition": repetition,
                    "ttft_ms": stream_result["ttft_ms"],
                    "itl_ms": stream_result["itl_ms"],
                }
            )
        peak_hbm = int(torch.cuda.max_memory_allocated())
    finally:
        if system_sampler is not None:
            system_samples = system_sampler.stop()
        engine.shutdown()

    server_reported_hbm = max(
        (
            int(
                sum(
                    float(value)
                    for key, value in (state.get("memory_usage") or {}).items()
                    if key in ("weight", "kvcache", "graph")
                )
                * (1 << 30)
            )
            for state in server_info_after.get("internal_states") or []
        ),
        default=0,
    )
    peak_hbm = max(
        peak_hbm,
        server_reported_hbm,
        max((int(sample["hbm_used_bytes"]) for sample in system_samples), default=0),
    )
    allocator_growth = 0
    for state in server_info_after.get("internal_states") or []:
        record = _adaptation_info_record(state)
        memory = (record.get("adaptation") or {}).get("memory") or {}
        allocator_growth = max(
            allocator_growth, int(memory.get("allocator_growth_bytes", 0) or 0)
        )
    _write_memory_calibration(
        engine_params,
        baseline_bytes=0,
        peak_bytes=allocator_growth,
    )

    if profile_steps:
        profile_files = [
            path
            for path in profile_output_dir.rglob("*")
            if path.is_file() and path.stat().st_size > 0
        ]
        if not profile_files:
            raise RuntimeGpuFailure(
                "SGLang profiler produced no non-empty trace files; refusing "
                "to report a profiling run as complete"
            )

    telemetry_glob = engine_params.get("telemetry_glob")
    if telemetry_glob:
        telemetry_paths = sorted(Path(p) for p in __import__("glob").glob(telemetry_glob))
    else:
        telemetry_paths = [
            Path(engine_params.get("telemetry_path", f"/tmp/{run_id}-telemetry.jsonl"))
        ]
    rows = _telemetry_to_rows(
        unit,
        run_id,
        telemetry_paths,
        summaries,
        wall,
        peak_hbm,
        system_samples=system_samples,
        performance_evidence=perf_evidence,
        server_info=server_info_after,
    )
    return rows


_P5_CONTEXT_SUBSET = re.compile(r"^p5_ctx_(\d+)$")
_P5_CONTEXT_RANGE_SUBSET = re.compile(r"^p5_ctx_(\d+)-(\d+)$")
_P5_SAMPLE_CONTEXT = re.compile(r":ctx-(\d+)$")


def _p5_prompt_context_length(sample_id: str) -> int | None:
    match = _P5_SAMPLE_CONTEXT.search(str(sample_id))
    return int(match.group(1)) if match else None


def _p5_context_lengths(unit: RunUnit, engine_params: dict) -> tuple[int, ...]:
    match = _P5_CONTEXT_SUBSET.fullmatch(unit.prompt_subset)
    if match:
        return (int(match.group(1)),)
    if unit.prompt_subset == "p5_ctx_request_side":
        return tuple(int(v) for v in engine_params.get("p5_request_context_lengths", ()))
    match = _P5_CONTEXT_RANGE_SUBSET.fullmatch(unit.prompt_subset)
    if not match:
        return ()
    low, high = (int(match.group(1)), int(match.group(2)))
    return tuple(
        int(value)
        for value in engine_params.get("p5_context_lengths", ())
        if low <= int(value) <= high
    )


def _coerce_token_ids(raw) -> list[int]:
    """Normalize tokenizer / chat-template outputs to a flat int id list."""
    if hasattr(raw, "input_ids"):
        raw = raw["input_ids"]
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, tuple):
        raw = list(raw)
    if not isinstance(raw, list) or not raw:
        raise RuntimeGpuFailure("tokenizer returned empty token ids")
    if isinstance(raw[0], (list, tuple)):
        raw = list(raw[0])
    return [int(token) for token in raw]


def _encode_prompt_token_ids(tokenizer, prompt: str, *, enable_thinking: bool) -> list[int]:
    """Encode a benchmark prompt, optionally via thinking-enabled chat template."""
    if not enable_thinking:
        return _coerce_token_ids(
            tokenizer.encode(prompt, add_special_tokens=False)
        )
    messages = [{"role": "user", "content": prompt}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=True,
        )
    except TypeError:
        # Some templates expose thinking only through chat_template_kwargs.
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            chat_template_kwargs={"enable_thinking": True},
        )
    return _coerce_token_ids(encoded)


def _p5_prefix_prompts(
    unit: RunUnit,
    engine_params: dict,
    samples,
    context_lengths: int | tuple[int, ...],
) -> list[dict]:
    """Build paired, exact-token prefix checkpoints before Engine startup.

    Each task prompt remains at the suffix.  Its deterministic carrier is
    assembled from other prompts in the same locked dataset, so every method,
    seed and load condition receives identical input ids without a
    decode/re-tokenize round trip.
    """
    from transformers import AutoTokenizer

    pair = MODEL_PAIRS[unit.model_pair]
    thinking = pair_thinking_config(pair)
    if "enable_thinking" in engine_params:
        enable_thinking = bool(engine_params["enable_thinking"])
    else:
        enable_thinking = bool(thinking["enable_thinking"])
    roots = engine_params.get("model_roots", {})
    target_root = roots.get(pair["target"])
    if not target_root:
        raise RuntimeGpuFailure(
            "P5 exact prefix checkpoints require a verified target model root"
        )
    if isinstance(context_lengths, int):
        context_lengths = (context_lengths,)
    if not context_lengths:
        raise RuntimeGpuFailure("P5 context group resolved to no context lengths")
    max_new = int(engine_params.get("max_new_tokens", 0) or 0)
    declared_limit = pair.get("max_context_length")
    if declared_limit is not None:
        oversized = [
            int(length)
            for length in context_lengths
            if int(length) + max_new > int(declared_limit)
        ]
        if oversized:
            raise RuntimeGpuFailure(
                f"{unit.model_pair} declares max context {declared_limit}, but "
                f"P5 prefixes {oversized} plus max_new_tokens={max_new} exceed "
                "it; refusing unsupported long-context run"
            )
    capacity = int(engine_params.get("max_total_tokens", 0) or 0)
    draft_tokens = int(
        engine_params.get(
            "speculative_num_draft_tokens", pair["default_num_draft_tokens"]
        )
    )
    runnable_contexts = []
    resource_skips = []
    for context_length in context_lengths:
        required = int(unit.concurrency) * (
            context_length + max_new + draft_tokens
        )
        if capacity and required > capacity:
            resource_skips.append(
                {
                    "context_length": context_length,
                    "required_kv_token_slots": required,
                    "max_total_tokens": capacity,
                }
            )
        else:
            runnable_contexts.append(context_length)
    if not runnable_contexts:
        skipped = resource_skips[0]
        raise ResourceSkip(
            "P5 paired prefixes require at least "
            f"{skipped['required_kv_token_slots']} KV token slots for "
            f"context={skipped['context_length']}, concurrency={unit.concurrency}, "
            f"but max_total_tokens={capacity}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        target_root, local_files_only=True, trust_remote_code=False
    )
    encoded = [
        _encode_prompt_token_ids(
            tokenizer, sample.prompt, enable_thinking=enable_thinking
        )
        for sample in samples
    ]
    separator = tokenizer.encode(
        "\n\n--- next context document ---\n\n", add_special_tokens=False
    )
    if not separator:
        separator = [int(tokenizer.eos_token_id or 0)]

    output = []
    checkpoint_rows = []
    for context_length in runnable_contexts:
        for index, (sample, suffix) in enumerate(zip(samples, encoded)):
            truncated = len(suffix) > context_length
            if truncated:
                input_ids = suffix[-context_length:]
            else:
                needed = context_length - len(suffix)
                carrier: list[int] = []
                cursor = 1
                while len(carrier) < needed:
                    source = encoded[(index + cursor) % len(encoded)] if encoded else []
                    carrier.extend(source)
                    carrier.extend(separator)
                    cursor += 1
                input_ids = carrier[:needed] + suffix
            if len(input_ids) != context_length:
                raise RuntimeGpuFailure(
                    f"P5 checkpoint length drift: {len(input_ids)} != {context_length}"
                )
            sample_id = f"{sample.sample_id}:ctx-{context_length}"
            token_sha = hashlib.sha256(
                json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            output.append(
                {"sample_id": sample_id, "prompt": None, "input_ids": input_ids}
            )
            checkpoint_rows.append(
                {
                    "sample_id": sample_id,
                    "source_sample_id": sample.sample_id,
                    "context_length": context_length,
                    "input_ids_sha256": token_sha,
                    "source_prompt_truncated": truncated,
                    "enable_thinking": enable_thinking,
                }
            )

    evidence = {
        "schema_version": 1,
        "dataset": unit.dataset,
        "prompt_subset": unit.prompt_subset,
        "tokenizer_root": str(target_root),
        "enable_thinking": enable_thinking,
        "reasoning_parser": thinking["reasoning_parser"] if enable_thinking else None,
        "resource_skips": resource_skips,
        "checkpoints": checkpoint_rows,
    }
    evidence_path = (
        Path(engine_params["adaptation_config_path"]).parent
        / "prefix-checkpoints.json"
    )
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return output


def _load_prompts(unit: RunUnit, engine_params: dict, lock) -> list[dict]:
    from lightcone_spec.benchmarks.registry import get_adapter

    adapter = get_adapter(unit.dataset)
    samples = adapter.load_samples(
        lock,
        limit=engine_params.get("prompt_limit", 128),
        offset=engine_params.get("prompt_offset", 0),
    )
    context_lengths = _p5_context_lengths(unit, engine_params)
    if context_lengths:
        return _p5_prefix_prompts(unit, engine_params, samples, context_lengths)

    pair = MODEL_PAIRS[unit.model_pair]
    thinking = pair_thinking_config(pair)
    if "enable_thinking" in engine_params:
        enable_thinking = bool(engine_params["enable_thinking"])
    else:
        enable_thinking = bool(thinking["enable_thinking"])
    if not enable_thinking:
        return [{"sample_id": s.sample_id, "prompt": s.prompt} for s in samples]

    # Non-P5 path: materialize thinking-enabled chat-template ids so the
    # generate API cannot silently run without the think channel.
    from transformers import AutoTokenizer

    roots = engine_params.get("model_roots", {})
    target_root = roots.get(pair["target"])
    if not target_root:
        raise RuntimeGpuFailure(
            "thinking-enabled prompts require a verified target model root"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        target_root, local_files_only=True, trust_remote_code=False
    )
    return [
        {
            "sample_id": sample.sample_id,
            "prompt": None,
            "input_ids": _encode_prompt_token_ids(
                tokenizer, sample.prompt, enable_thinking=True
            ),
        }
        for sample in samples
    ]


_REQUIRED_ROUND_TELEMETRY_FIELDS = frozenset(
    {
        "request_id",
        "round_id",
        "active_version",
        "proposal_version",
        "draft_tokens",
        "accepted_drafts",
        "committed_per_verify",
        "target_calls",
        "draft_cuda_us",
        "verify_cuda_us",
        "accept_cuda_us",
        "draft_cpu_us",
        "verify_cpu_us",
        "rng_substream_id",
        "version_canary_ok",
        "prefix_pos_before",
        "prefix_pos_after",
        "prefix_len_before",
        "verify_len",
        "batch_size",
        "offered_concurrency",
        "round_wall_us",
        "prefix_feature_exact",
        "algorithmic_censored",
    }
)


def _validate_round_telemetry_record(
    rec: dict, *, telemetry_path: Path, line_number: int
) -> None:
    """Reject absent/default evidence without rejecting a real zero acceptance.

    A hard prompt can legitimately accept no draft token.  What makes a round
    usable evidence is that it attempted and verified draft work, committed the
    mandatory target token, and carries finite component timings and load/prefix
    coordinates.  Validate that contract before any ``dict.get(..., 0)`` can
    turn a missing producer field into plausible-looking telemetry.
    """

    location = f"{telemetry_path}:{line_number}"
    missing = sorted(_REQUIRED_ROUND_TELEMETRY_FIELDS.difference(rec))
    if missing:
        raise RuntimeGpuFailure(
            f"{location}: round telemetry is missing required fields: "
            + ", ".join(missing)
        )

    try:
        request_id = str(rec["request_id"])
        rng_substream_id = str(rec["rng_substream_id"])
        round_id = int(rec["round_id"])
        active_version = int(rec["active_version"])
        proposal_version = int(rec["proposal_version"])
        draft_tokens = int(rec["draft_tokens"])
        accepted_drafts = int(rec["accepted_drafts"])
        committed = int(rec["committed_per_verify"])
        target_calls = int(rec["target_calls"])
        verify_len = int(rec["verify_len"])
        prefix_before = int(rec["prefix_pos_before"])
        prefix_len_before = int(rec["prefix_len_before"])
        prefix_after = int(rec["prefix_pos_after"])
        batch_size = int(rec["batch_size"])
        offered_concurrency = int(rec["offered_concurrency"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeGpuFailure(
            f"{location}: round telemetry contains a non-integral counter: {exc}"
        ) from exc

    if not request_id or not rng_substream_id:
        raise RuntimeGpuFailure(
            f"{location}: round telemetry request/rng identity is empty"
        )
    if min(round_id, active_version, proposal_version) < 0:
        raise RuntimeGpuFailure(
            f"{location}: round/version identifiers must be non-negative"
        )
    if draft_tokens <= 0 or target_calls <= 0:
        raise RuntimeGpuFailure(
            f"{location}: round must contain positive draft_tokens and target_calls"
        )
    if not 1 <= verify_len <= draft_tokens + 1:
        raise RuntimeGpuFailure(
            f"{location}: verify_len={verify_len} is outside [1, {draft_tokens + 1}]"
        )
    verified_drafts = max(verify_len - 1, 0)
    if not 0 <= accepted_drafts <= min(draft_tokens, verified_drafts):
        raise RuntimeGpuFailure(
            f"{location}: accepted_drafts={accepted_drafts} exceeds the "
            f"verified/drafted bound {min(draft_tokens, verified_drafts)}"
        )
    if committed < accepted_drafts + 1:
        raise RuntimeGpuFailure(
            f"{location}: committed_per_verify={committed} does not include "
            "the accepted prefix plus the mandatory target token"
        )
    if batch_size <= 0 or offered_concurrency <= 0:
        raise RuntimeGpuFailure(
            f"{location}: batch/load fields must be positive"
        )
    if min(prefix_before, prefix_len_before, prefix_after) < 0:
        raise RuntimeGpuFailure(
            f"{location}: prefix positions must be non-negative"
        )
    if prefix_before != prefix_len_before:
        raise RuntimeGpuFailure(
            f"{location}: prefix_pos_before and prefix_len_before disagree"
        )
    if prefix_after < prefix_before:
        raise RuntimeGpuFailure(
            f"{location}: prefix_pos_after precedes prefix_len_before"
        )

    for field_name in (
        "draft_cuda_us",
        "verify_cuda_us",
        "accept_cuda_us",
        "draft_cpu_us",
        "verify_cpu_us",
        "round_wall_us",
    ):
        try:
            value = float(rec[field_name])
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeGpuFailure(
                f"{location}: {field_name} is not numeric"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeGpuFailure(
                f"{location}: {field_name} must be finite and non-negative"
            )

    signal_prep = rec.get("signal_prep_cuda_us")
    if signal_prep is not None:
        try:
            signal_prep = float(signal_prep)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeGpuFailure(
                f"{location}: signal_prep_cuda_us is not numeric"
            ) from exc
        if not math.isfinite(signal_prep) or signal_prep < 0.0:
            raise RuntimeGpuFailure(
                f"{location}: signal_prep_cuda_us must be finite and non-negative"
            )

    if (
        not isinstance(rec["version_canary_ok"], bool)
        or not isinstance(rec["prefix_feature_exact"], bool)
        or not isinstance(rec["algorithmic_censored"], bool)
    ):
        raise RuntimeGpuFailure(
            f"{location}: exactness fields must be explicit booleans"
        )


def _telemetry_to_rows(
    unit: RunUnit,
    run_id: str,
    telemetry_paths: list[Path],
    summaries: list[dict],
    wall_s: float,
    peak_hbm_bytes: int,
    system_samples: list[dict] | None = None,
    performance_evidence: dict | None = None,
    server_info: dict | None = None,
) -> dict:
    common = {
        "schema_version": 1,
        "run_id": run_id,
        "unit_id": unit.unit_id,
        "stream_id": None,
        "tenant_id_hash": "tenant-0",
        "model_pair_id": unit.model_pair,
        "method": unit.method,
        "dataset": unit.dataset,
        "seed": unit.seed,
        "lifecycle": unit.lifecycle,
    }
    rows: dict = {
        "rounds": [],
        "updates": [],
        "decisions": [],
        "system_samples": [],
        "request_summary": [],
    }
    exactness_failure = False
    runtime_failure = False
    measured_runtime_ids = {
        str(s["runtime_request_id"])
        for s in summaries
        if s.get("runtime_request_id")
    }
    round_wall_by_request: dict[str, list[float]] = {}
    target_calls_by_request: dict[str, int] = {}
    accepted_by_request: dict[str, list[int]] = {}
    committed_by_request: dict[str, list[int]] = {}
    mismatch_by_request: dict[str, int] = {}
    existing_paths = [p for p in telemetry_paths if p.is_file() and p.stat().st_size]
    if not existing_paths:
        raise RuntimeGpuFailure(
            "adaptation telemetry is missing or empty; refusing complete_valid"
        )
    for telemetry_path in existing_paths:
        for line_number, line in enumerate(
            telemetry_path.read_text().splitlines(), start=1
        ):
            rec = json.loads(line)
            kind = rec.pop("kind", None)
            # Warmup is deliberately run through the real model path but is
            # excluded from benchmark artifacts whenever SGLang exposes its
            # request id.  If that id is absent, retain all rows and use the
            # aggregate fallback below rather than silently dropping data.
            if (
                measured_runtime_ids
                and rec.get("request_id")
                and str(rec["request_id"]) not in measured_runtime_ids
            ):
                continue
            if kind == "round":
                _validate_round_telemetry_record(
                    rec,
                    telemetry_path=telemetry_path,
                    line_number=line_number,
                )
                rid = rec["request_id"]
                # Preserve the measured wall time in the normative round row;
                # it is also used locally for request-summary percentiles.
                wall_us = float(rec.get("round_wall_us", 0.0))
                round_wall_by_request.setdefault(rid, []).append(wall_us)
                target_calls_by_request[rid] = target_calls_by_request.get(rid, 0) + int(
                    rec.get("target_calls", 0)
                )
                accepted_by_request.setdefault(rid, []).append(
                    int(rec.get("accepted_drafts", 0))
                )
                committed_by_request.setdefault(rid, []).append(
                    int(rec.get("committed_per_verify", 0))
                )
                if not rec.get("version_canary_ok", False) or not rec.get(
                    "prefix_feature_exact", False
                ):
                    exactness_failure = True
                    mismatch_by_request[rid] = mismatch_by_request.get(rid, 0) + 1
                rows["rounds"].append(
                    {
                        **common,
                        "request_id": rid,
                        **rec,
                        "target_topk_token_ids": [],
                        "target_topk_probs": [],
                        "target_other_mass": 0.0,
                        "proposal_topk_token_ids": [],
                        "proposal_topk_probs": [],
                        "proposal_other_mass": 0.0,
                        "hidden_proj": [],
                        "event_sketch": [],
                        "endpoint_from_previous": 0.0,
                    }
                )
            elif kind == "update":
                rid = rec["request_id"]
                source_training_loss = rec.get("source_training_loss")
                source_expected_accepted_prefix = rec.get(
                    "source_expected_accepted_prefix"
                )
                source_prefix_len = rec.get("source_prefix_len")
                if rec.get("launch_ts_us") is not None:
                    for field_name in (
                        "candidate_cuda_us",
                        "backward_cuda_us",
                        "optimizer_cuda_us",
                    ):
                        if field_name not in rec or rec[field_name] is None:
                            raise RuntimeGpuFailure(
                                f"{telemetry_path}:{line_number}: launched update "
                                f"is missing {field_name} evidence"
                            )
                        try:
                            component_us = float(rec[field_name])
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise RuntimeGpuFailure(
                                f"{telemetry_path}:{line_number}: {field_name} "
                                "is not numeric"
                            ) from exc
                        if not math.isfinite(component_us) or component_us < 0.0:
                            raise RuntimeGpuFailure(
                                f"{telemetry_path}:{line_number}: {field_name} "
                                "must be finite and non-negative"
                            )
                    for field_name, value in (
                        ("source_training_loss", source_training_loss),
                        (
                            "source_expected_accepted_prefix",
                            source_expected_accepted_prefix,
                        ),
                        ("source_prefix_len", source_prefix_len),
                    ):
                        if value is None:
                            raise RuntimeGpuFailure(
                                f"{telemetry_path}:{line_number}: launched update "
                                f"is missing {field_name} evidence"
                            )
                    try:
                        source_training_loss = float(source_training_loss)
                        source_expected_accepted_prefix = float(
                            source_expected_accepted_prefix
                        )
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise RuntimeGpuFailure(
                            f"{telemetry_path}:{line_number}: source loss evidence "
                            "is not numeric"
                        ) from exc
                    if isinstance(source_prefix_len, bool) or not isinstance(
                        source_prefix_len, int
                    ):
                        raise RuntimeGpuFailure(
                            f"{telemetry_path}:{line_number}: source_prefix_len "
                            "must be an integer"
                        )
                    if source_prefix_len < 0:
                        raise RuntimeGpuFailure(
                            f"{telemetry_path}:{line_number}: source_prefix_len "
                            "must be non-negative"
                        )
                else:
                    if source_training_loss is not None:
                        source_training_loss = float(source_training_loss)
                    if source_expected_accepted_prefix is not None:
                        source_expected_accepted_prefix = float(
                            source_expected_accepted_prefix
                        )
                    if source_prefix_len is not None:
                        source_prefix_len = int(source_prefix_len)
                candidate_batch_size = rec.get("candidate_batch_size")
                if candidate_batch_size is not None:
                    if isinstance(candidate_batch_size, bool):
                        raise RuntimeGpuFailure(
                            f"{telemetry_path}:{line_number}: "
                            "candidate_batch_size must be a positive integer"
                        )
                    try:
                        candidate_batch_size = int(candidate_batch_size)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise RuntimeGpuFailure(
                            f"{telemetry_path}:{line_number}: "
                            "candidate_batch_size must be a positive integer"
                        ) from exc
                    if candidate_batch_size <= 0:
                        raise RuntimeGpuFailure(
                            f"{telemetry_path}:{line_number}: "
                            "candidate_batch_size must be a positive integer"
                        )
                reason = rec.get("failure_reason")
                if isinstance(reason, str) and (
                    reason.startswith("exactness:")
                    or reason.startswith("fallback:exactness:")
                ):
                    exactness_failure = True
                    mismatch_by_request[rid] = mismatch_by_request.get(rid, 0) + 1
                elif reason == "non_finite_candidate" or (
                    isinstance(reason, str) and reason.startswith("fallback:")
                ):
                    # Serving continues with the no-op adapter publication,
                    # but numerical corruption invalidates this benchmark.
                    runtime_failure = True
                published = rec.get("published_version")
                decision = rec.get("decision") or (
                    "apply" if published is not None else "discard"
                )
                rows["updates"].append(
                    {
                        **common,
                        "request_id": rid,
                        "update_id": rec["update_id"],
                        "source_round": rec["source_round"],
                        "apply_round": (
                            rec.get("source_round", 0)
                            + rec.get("effective_delay_rounds", 0)
                            if published is not None
                            else None
                        ),
                        "exposure_round": (
                            rec.get("source_round", 0)
                            + rec.get("effective_delay_rounds", 0)
                            if published is not None
                            else None
                        ),
                        "source_version": rec["source_version"],
                        "source_training_loss": source_training_loss,
                        "source_expected_accepted_prefix": (
                            source_expected_accepted_prefix
                        ),
                        "source_prefix_len": source_prefix_len,
                        "active_version_at_arrival": int(
                            rec.get("active_version_at_arrival")
                            if rec.get("active_version_at_arrival") is not None
                            else max(
                                int(published or rec["source_version"])
                                - (1 if published else 0),
                                0,
                            )
                        ),
                        "staging_version": int(
                            rec.get("staging_version")
                            if rec.get("staging_version") is not None
                            else (published or rec["source_version"] + 1)
                        ),
                        "published_version": published,
                        "delay_rounds": rec.get("effective_delay_rounds", 0),
                        "delay_tokens": int(rec.get("delay_tokens", 0)),
                        "delay_wall_us": float(
                            rec.get("delay_wall_us")
                            if rec.get("delay_wall_us") is not None
                            else max(
                                0.0,
                                float(rec.get("commit_ts_us") or rec.get("done_ts_us") or 0.0)
                                - float(rec.get("snapshot_ts_us") or 0.0),
                            )
                        ),
                        "delay_versions": int(
                            rec.get("delay_versions")
                            if rec.get("delay_versions") is not None
                            else max(
                                int(
                                    rec.get("active_version_at_arrival")
                                    if rec.get("active_version_at_arrival") is not None
                                    else rec["source_version"]
                                )
                                - rec["source_version"],
                                0,
                            )
                        ),
                        "snapshot_ts_us": float(rec.get("snapshot_ts_us") or 0.0),
                        "teacher_ts_us": rec.get("teacher_ts_us"),
                        "launch_ts_us": rec.get("launch_ts_us"),
                        "done_ts_us": rec.get("done_ts_us"),
                        "commit_ts_us": rec.get("commit_ts_us"),
                        "exposure_ts_us": rec.get("exposure_ts_us"),
                        "launch_event_id": rec["update_id"] + "-launch",
                        "done_event_id": rec["update_id"] + "-done",
                        "commit_event_id": (
                            rec["update_id"] + "-commit" if published is not None else None
                        ),
                        "grad_norm": float(rec.get("grad_norm", 0.0)),
                        "grad_clip_scale": float(rec.get("grad_clip_scale", 1.0)),
                        "grad_sketch": [],
                        "candidate_delta_norm": float(
                            rec.get("candidate_delta_norm", 0.0)
                        ),
                        "side_queue_cuda_us": float(
                            rec.get("side_queue_cuda_us", 0.0)
                        ),
                        "candidate_cuda_us": float(
                            rec.get("candidate_cuda_us", 0.0)
                        ),
                        "candidate_batch_size": (
                            None
                            if candidate_batch_size is None
                            else candidate_batch_size
                        ),
                        "backward_cuda_us": (
                            None
                            if rec.get("backward_cuda_us") is None
                            else float(rec["backward_cuda_us"])
                        ),
                        "optimizer_cuda_us": (
                            None
                            if rec.get("optimizer_cuda_us") is None
                            else float(rec["optimizer_cuda_us"])
                        ),
                        "gradient_weight_version": rec.get(
                            "gradient_weight_version"
                        ),
                        "gradient_kv_version_min": rec.get(
                            "gradient_kv_version_min"
                        ),
                        "gradient_kv_version_max": rec.get(
                            "gradient_kv_version_max"
                        ),
                        "gradient_version_canary_ok": rec.get(
                            "gradient_version_canary_ok"
                        ),
                        "barrier_wait_cpu_us": float(
                            rec.get("barrier_wait_cpu_us", 0.0)
                        ),
                        "publish_cuda_us": float(
                            rec.get("publish_cuda_us", 0.0)
                        ),
                        "optimizer_step": int(rec.get("optimizer_step", 0)),
                        "numerical_ok": reason != "non_finite_candidate",
                        "failure_reason": reason,
                    }
                )
                rows["decisions"].append(
                    {
                        **common,
                        "request_id": rid,
                        "update_id": rec["update_id"],
                        "rho_path": float(rec.get("rho_path", 0.0)),
                        "endpoint_distance": float(
                            rec.get("endpoint_distance", 0.0)
                        ),
                        "parameter_displacement": float(
                            rec.get("parameter_displacement", 0.0)
                        ),
                        "predicted_utility": rec.get("predicted_utility"),
                        "predicted_mismatch": rec.get("predicted_mismatch"),
                        "predicted_harm_probability": rec.get(
                            "predicted_harm_probability"
                        ),
                        "threshold": rec.get("threshold"),
                        "decision": decision,
                        "damping_factor": float(
                            1.0
                            if rec.get("damping_factor") is None
                            else rec["damping_factor"]
                        ),
                        "transport_rank": None,
                        "parameter_comp_norm": None,
                        "state_transport_norm": None,
                        "random_transport": False,
                        "controller_cpu_us": float(
                            rec.get("controller_cpu_us", 0.0)
                        ),
                        "controller_cuda_us": float(
                            rec.get("controller_cuda_us", 0.0)
                        ),
                    }
                )
            elif kind == "system":
                sample = {**common, "request_id": "", **rec}
                sample.setdefault("sample_source", "legacy_unspecified")
                sample.setdefault("activity_provenance", "legacy_unspecified")
                sample.setdefault("contention_provenance", "legacy_unspecified")
                sample.setdefault("sync_provenance", "legacy_unspecified")
                rows["system_samples"].append(sample)
    if not rows["rounds"]:
        raise RuntimeGpuFailure(
            "telemetry contains no decode rounds; refusing complete_valid"
        )
    if sum(int(row["draft_tokens"]) for row in rows["rounds"]) <= 0:
        raise RuntimeGpuFailure("telemetry contains zero draft tokens")
    if sum(int(row["target_calls"]) for row in rows["rounds"]) <= 0:
        raise RuntimeGpuFailure("telemetry contains zero target calls")
    if not any(
        float(row["draft_cuda_us"])
        + float(row["verify_cuda_us"])
        + float(row["accept_cuda_us"])
        > 0.0
        for row in rows["rounds"]
    ):
        raise RuntimeGpuFailure(
            "CUDA event timing is absent or zero for every decode round"
        )
    if peak_hbm_bytes <= 0:
        raise RuntimeGpuFailure("peak CUDA memory is zero; HBM evidence missing")
    samples = list(system_samples or [])
    if not samples:
        raise RuntimeGpuFailure(
            "GPU system telemetry is missing; refusing to synthesize zero "
            "utilization/power evidence for a completed run"
        )
    normalized_samples: list[dict] = []
    for index, raw_sample in enumerate(samples):
        sample = dict(raw_sample)
        # Schema-v1 runtime producers did not identify how stream/sync fields
        # were obtained.  Preserve their values for compatibility, but label
        # them as legacy-unknown rather than upgrading them to observations.
        sample.setdefault("sample_source", "legacy_unspecified")
        sample.setdefault("activity_provenance", "legacy_unspecified")
        sample.setdefault("contention_provenance", "legacy_unspecified")
        sample.setdefault("sync_provenance", "legacy_unspecified")
        allowed_provenance = {
            "activity_provenance": {
                "observed",
                "simulated",
                "not_observed",
                "legacy_unspecified",
            },
            "contention_provenance": {
                "observed",
                "simulated",
                "declared_not_observed",
                "legacy_unspecified",
            },
            "sync_provenance": {
                "observed",
                "simulated",
                "not_observed",
                "legacy_unspecified",
            },
        }
        for field_name, allowed in allowed_provenance.items():
            if sample[field_name] not in allowed:
                raise RuntimeGpuFailure(
                    f"GPU system telemetry sample {index} has invalid "
                    f"{field_name}"
                )
        try:
            hbm = int(sample["hbm_used_bytes"])
            utilization = float(sample["gpu_utilization"])
            power = float(sample["power_watts"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeGpuFailure(
                f"GPU system telemetry sample {index} is incomplete"
            ) from exc
        sync_raw = sample.get("sync_us_delta")
        try:
            sync_us = None if sync_raw is None else float(sync_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeGpuFailure(
                f"GPU system telemetry sample {index} has invalid sync timing"
            ) from exc
        invalid_sync = sync_us is not None and (
            not math.isfinite(sync_us) or sync_us < 0.0
        )
        if (
            hbm <= 0
            or not all(
                math.isfinite(value) and value >= 0.0
                for value in (utilization, power)
            )
            or invalid_sync
        ):
            raise RuntimeGpuFailure(
                f"GPU system telemetry sample {index} has invalid HBM/timing values"
            )
        if sync_us is None and sample["sync_provenance"] not in {
            "not_observed",
            "legacy_unspecified",
        }:
            raise RuntimeGpuFailure(
                f"GPU system telemetry sample {index} declares sync provenance "
                "without a value"
            )
        if sample["activity_provenance"] == "not_observed" and any(
            sample.get(name) is not None
            for name in ("main_stream_active", "side_stream_active")
        ):
            raise RuntimeGpuFailure(
                f"GPU system telemetry sample {index} contains unobserved "
                "stream activity values"
            )
        if sample["activity_provenance"] in {"observed", "simulated"} and any(
            not isinstance(sample.get(name), bool)
            for name in ("main_stream_active", "side_stream_active")
        ):
            raise RuntimeGpuFailure(
                f"GPU system telemetry sample {index} is missing declared "
                "stream activity observations"
            )
        if (
            sample["sync_provenance"] in {"observed", "simulated"}
            and sync_us is None
        ):
            raise RuntimeGpuFailure(
                f"GPU system telemetry sample {index} is missing declared "
                "sync timing evidence"
            )
        normalized_samples.append(sample)
    samples = normalized_samples
    rows["system_samples"].extend(
        {**common, "request_id": "", **sample} for sample in samples
    )
    sampled_energy_j = sum(
        max(0.0, float(sample.get("energy_joules_delta", 0.0) or 0.0))
        for sample in samples
    )
    performance_evidence = performance_evidence or {}
    internal_states = (server_info or {}).get("internal_states") or []
    fallback_count = 0
    kv_retracted_requests = 0
    peak_running_requests = 0
    peak_queue_requests = 0
    model_weight_hbm_bytes = 0
    kv_cache_hbm_bytes = 0
    cuda_graph_hbm_bytes = 0
    kv_token_capacity = 0
    adaptation_fixed_bytes = 0
    adaptation_reserve_bytes = 0
    controller_static_fallback = False
    for state in internal_states:
        record = _adaptation_info_record(state)
        adaptation = record.get("adaptation") or {}
        memory = adaptation.get("memory") or {}
        adaptation_fixed_bytes += int(memory.get("fixed_bytes", 0) or 0)
        adaptation_reserve_bytes += int(memory.get("reserve_bytes", 0) or 0)
        controller_static_fallback = controller_static_fallback or bool(
            adaptation.get("controller_static_fallback", False)
        )
        fallback_counts = {
            str(reason): int(value)
            for reason, value in (adaptation.get("fallback_counts") or {}).items()
        }
        fallback_count += sum(fallback_counts.values())
        if fallback_counts.get("exactness", 0) > 0:
            exactness_failure = True
        if any(
            count > 0 and reason != "exactness"
            for reason, count in fallback_counts.items()
        ):
            runtime_failure = True
        load = state.get("load_cumulative") or {}
        kv_retracted_requests += int(load.get("retracted_requests", 0) or 0)
        peak_running_requests = max(
            peak_running_requests, int(load.get("peak_running_requests", 0) or 0)
        )
        peak_queue_requests = max(
            peak_queue_requests, int(load.get("peak_queue_requests", 0) or 0)
        )
        server_memory = state.get("memory_usage") or {}
        model_weight_hbm_bytes += int(
            float(server_memory.get("weight", 0.0) or 0.0) * (1 << 30)
        )
        kv_cache_hbm_bytes += int(
            float(server_memory.get("kvcache", 0.0) or 0.0) * (1 << 30)
        )
        cuda_graph_hbm_bytes += int(
            float(server_memory.get("graph", 0.0) or 0.0) * (1 << 30)
        )
        kv_token_capacity += int(server_memory.get("token_capacity", 0) or 0)
    if controller_static_fallback:
        # The fallback state is recorded in request/run diagnostics.  Do not
        # rewrite independently sampled stream fields: NVML cannot observe
        # stream activity, and an after-the-fact fallback is not evidence for
        # the state at each sampling timestamp.
        pass
    fallback_count = int(
        performance_evidence.get("adaptation_fallback_count", fallback_count)
    )
    if fallback_count > 0 and not exactness_failure:
        # The cumulative delta is the only available evidence when the final
        # diagnostics RPC itself failed.  A per-request adaptation fallback is
        # safe for serving but invalidates an adaptation benchmark.
        runtime_failure = True
    kv_retracted_requests = int(
        performance_evidence.get("kv_retracted_requests", kv_retracted_requests)
    )
    peak_running_requests = int(
        performance_evidence.get("peak_running_requests", peak_running_requests)
    )
    peak_queue_requests = int(
        performance_evidence.get("peak_queue_requests", peak_queue_requests)
    )

    def percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        pos = (len(ordered) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)

    gpu_utilization = [float(sample["gpu_utilization"]) for sample in samples]
    nvml_gpu_utilization_mean = sum(gpu_utilization) / len(gpu_utilization)
    nvml_gpu_busy_fraction_90 = sum(
        value >= 90.0 for value in gpu_utilization
    ) / len(gpu_utilization)

    all_round_ms = [
        value / 1000.0
        for values in round_wall_by_request.values()
        for value in values
    ]
    all_accepted = [value for values in accepted_by_request.values() for value in values]
    all_committed = [value for values in committed_by_request.values() for value in values]
    total_target_calls = sum(target_calls_by_request.values())
    total_output_tokens = sum(
        max(0, int(s.get("meta", {}).get("completion_tokens", 0) or 0))
        for s in summaries
    )
    if total_output_tokens <= 0:
        raise RuntimeGpuFailure(
            "SGLang returned no positive completion_tokens; refusing a zero-filled "
            "performance artifact"
        )

    for s in summaries:
        meta = s.get("meta", {})
        row_performance_evidence = s.get("performance_evidence") or performance_evidence
        tokens = int(meta.get("completion_tokens", 0) or 0)
        runtime_rid = str(s.get("runtime_request_id") or "")
        matched = runtime_rid in round_wall_by_request
        round_ms = (
            [x / 1000.0 for x in round_wall_by_request[runtime_rid]]
            if matched
            else all_round_ms
        )
        itl_ms = [float(value) for value in s.get("itl_ms", [])]
        accepted = accepted_by_request.get(runtime_rid, all_accepted)
        committed = committed_by_request.get(runtime_rid, all_committed)
        target_calls = (
            target_calls_by_request.get(runtime_rid, 0)
            if matched
            else total_target_calls * tokens / max(total_output_tokens, 1)
        )
        mismatch_count = (
            mismatch_by_request.get(runtime_rid, 0)
            if matched
            else sum(mismatch_by_request.values())
        )
        request_wall = float(s.get("request_wall_s", wall_s))
        run_wall = float(s.get("run_wall_s", wall_s))
        run_tokens = int(s.get("run_total_tokens", total_output_tokens))
        valid_goodput = (
            0.0
            if exactness_failure or runtime_failure
            else run_tokens / max(run_wall, 1e-9)
        )
        valid_request_tps = (
            0.0
            if exactness_failure or runtime_failure
            else tokens / max(request_wall, 1e-9)
        )
        request_status = (
            "failed_exactness"
            if exactness_failure
            else ("failed_runtime" if runtime_failure else "complete_valid")
        )
        rows["request_summary"].append(
            {
                **common,
                "request_id": s["sample_id"],
                "prompt_id_hash": s["sample_id"],
                "task_type": unit.dataset,
                "output_tokens": tokens,
                "quality_metric_name": "deferred_scoring",
                "quality_value": None,
                "decode_wall_s": request_wall,
                "e2e_wall_s": request_wall,
                "decode_tps": valid_goodput,
                "e2e_tps": valid_request_tps,
                "goodput_tps": valid_goodput,
                "offered_concurrency": int(
                    s.get("offered_concurrency", unit.concurrency)
                ),
                "ttft_ms": s.get("ttft_ms"),
                "queue_ms": (
                    float(meta["waiting_time"] * 1000.0)
                    if meta.get("waiting_time") is not None
                    else None
                ),
                "estimated_perf_scope": "target_model_only",
                "estimated_tflops_per_gpu": row_performance_evidence.get(
                    "estimated_tflops_per_gpu"
                ),
                "estimated_mfu": row_performance_evidence.get("estimated_mfu"),
                "estimated_read_gbps_per_gpu": row_performance_evidence.get(
                    "estimated_read_gbps_per_gpu"
                ),
                "estimated_write_gbps_per_gpu": row_performance_evidence.get(
                    "estimated_write_gbps_per_gpu"
                ),
                "peak_tflops_per_gpu": row_performance_evidence.get(
                    "peak_tflops_per_gpu"
                ),
                "decode_step_count": row_performance_evidence.get(
                    "decode_step_count"
                ),
                "decode_batch_size_step_mean": row_performance_evidence.get(
                    "decode_batch_size_step_mean"
                ),
                "decode_batch_size_time_mean": row_performance_evidence.get(
                    "decode_batch_size_time_mean"
                ),
                "decode_batch_size_std": row_performance_evidence.get(
                    "decode_batch_size_std"
                ),
                "decode_batch_fill_ratio": row_performance_evidence.get(
                    "decode_batch_fill_ratio"
                ),
                "decode_scheduler_span_s": row_performance_evidence.get(
                    "decode_scheduler_span_s"
                ),
                "decode_generated_tps_scheduler_span": row_performance_evidence.get(
                    "decode_generated_tps_scheduler_span"
                ),
                "prefill_uncached_tokens": row_performance_evidence.get(
                    "prefill_uncached_tokens"
                ),
                "prefill_busy_s": row_performance_evidence.get("prefill_busy_s"),
                "nvml_gpu_utilization_mean": nvml_gpu_utilization_mean,
                "nvml_gpu_utilization_p10": percentile(gpu_utilization, 0.10),
                "nvml_gpu_utilization_p90": percentile(gpu_utilization, 0.90),
                "nvml_gpu_busy_fraction_90": nvml_gpu_busy_fraction_90,
                "adaptation_fallback_count": fallback_count,
                "kv_retracted_requests": kv_retracted_requests,
                "peak_running_requests": int(
                    row_performance_evidence.get(
                        "peak_running_requests", peak_running_requests
                    )
                ),
                "peak_queue_requests": int(
                    row_performance_evidence.get(
                        "peak_queue_requests", peak_queue_requests
                    )
                ),
                "model_weight_hbm_bytes": model_weight_hbm_bytes,
                "kv_cache_hbm_bytes": kv_cache_hbm_bytes,
                "cuda_graph_hbm_bytes": cuda_graph_hbm_bytes,
                "kv_token_capacity": kv_token_capacity,
                "adaptation_fixed_bytes": adaptation_fixed_bytes,
                "adaptation_reserve_bytes": adaptation_reserve_bytes,
                "mean_accepted_drafts": (
                    sum(accepted) / len(accepted) if accepted else 0.0
                ),
                "mean_committed_per_verify": (
                    sum(committed) / len(committed) if committed else 0.0
                ),
                "target_calls_per_output_token": target_calls / max(tokens, 1),
                "p50_round_ms": percentile(round_ms, 0.50),
                "p95_round_ms": percentile(round_ms, 0.95),
                "p99_round_ms": percentile(round_ms, 0.99),
                "p50_itl_ms": percentile(itl_ms, 0.50),
                "p95_itl_ms": percentile(itl_ms, 0.95),
                "p99_itl_ms": percentile(itl_ms, 0.99),
                "energy_per_token_j": (
                    sampled_energy_j / total_output_tokens
                    if sampled_energy_j > 0.0
                    else None
                ),
                "peak_hbm_bytes": peak_hbm_bytes,
                "version_mismatch_count": mismatch_count,
                "status": request_status,
            }
        )
    rows["_status"] = (
        "failed_exactness"
        if exactness_failure
        else ("failed_runtime" if runtime_failure else "complete_valid")
    )
    return rows
