"""Thin SGLang HTTP client that retains token trajectories and stream timing."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    input_tokens: int
    completion_tokens: int
    ttft_ms: float
    inter_token_ms: tuple[float, ...]
    elapsed_seconds: float
    stop_reason: str | None
    output_ids: tuple[int, ...]
    output_text: str
    native_token_timestamps_ns: tuple[int, ...]
    stop_details: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["inter_token_ms"] = list(self.inter_token_ms)
        value["output_ids"] = list(self.output_ids)
        value["native_token_timestamps_ns"] = list(self.native_token_timestamps_ns)
        return value


@dataclass(frozen=True)
class RequestOutcome:
    request_id: str
    status: str
    offered_ns: int
    admitted_ns: int | None
    finished_ns: int | None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScheduledRun:
    results: tuple[GenerationResult, ...]
    outcomes: tuple[RequestOutcome, ...]
    elapsed_seconds: float


def _request_error(error: Exception) -> str:
    if isinstance(error, TimeoutError):
        return "timed_out"
    if isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError):
        return "timed_out"
    return "error"


def _run_bounded(
    client,
    prompts: Iterable[str | Sequence[int]],
    *,
    max_new_tokens: int | Sequence[int],
    seed: int,
    temperature: float,
    routing_keys: Iterable[str] | None,
    request_ids: Sequence[str] | None,
    max_in_flight: int,
    deadline_seconds: float,
) -> ScheduledRun:
    prompt_rows = tuple(prompts)
    if not prompt_rows or max_in_flight < 1:
        raise ValueError("bounded requests need prompts and positive concurrency")
    keys = tuple(routing_keys) if routing_keys is not None else (None,) * len(prompt_rows)
    ids = (
        tuple(request_ids)
        if request_ids is not None
        else tuple(f"bounded-{index:05d}" for index in range(len(prompt_rows)))
    )
    budgets = (
        (max_new_tokens,) * len(prompt_rows)
        if isinstance(max_new_tokens, int)
        else tuple(max_new_tokens)
    )
    if not (len(keys) == len(ids) == len(budgets) == len(prompt_rows)):
        raise ValueError("bounded requests need one key, ID, and budget per prompt")
    if len(set(ids)) != len(ids):
        raise ValueError("bounded request IDs must be unique")

    started = time.perf_counter()
    results: dict[int, GenerationResult] = {}
    outcomes: list[RequestOutcome | None] = [None] * len(prompt_rows)

    def submit(index: int) -> GenerationResult:
        rows, _ = client.run_batch(
            (prompt_rows[index],),
            max_new_tokens=budgets[index],
            seed=seed + index,
            temperature=temperature,
            routing_key=keys[index],
            request_ids=(ids[index],),
            timeout_seconds=deadline_seconds,
        )
        return rows[0]

    active: dict[Future[GenerationResult], tuple[int, int, int]] = {}
    next_index = 0
    with ThreadPoolExecutor(max_workers=max_in_flight) as pool:
        while next_index < len(prompt_rows) or active:
            while next_index < len(prompt_rows) and len(active) < max_in_flight:
                offered_ns = time.monotonic_ns()
                admitted_ns = time.monotonic_ns()
                active[pool.submit(submit, next_index)] = (
                    next_index,
                    offered_ns,
                    admitted_ns,
                )
                next_index += 1
            done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in done:
                index, offered_ns, admitted_ns = active.pop(future)
                finished_ns = time.monotonic_ns()
                try:
                    results[index] = future.result()
                    outcomes[index] = RequestOutcome(
                        ids[index], "completed", offered_ns, admitted_ns, finished_ns
                    )
                except Exception as error:
                    client.abort(ids[index])
                    outcomes[index] = RequestOutcome(
                        ids[index],
                        _request_error(error),
                        offered_ns,
                        admitted_ns,
                        finished_ns,
                        f"{type(error).__name__}: {error}",
                    )
    return ScheduledRun(
        tuple(results[index] for index in sorted(results)),
        tuple(outcome for outcome in outcomes if outcome is not None),
        time.perf_counter() - started,
    )


def _run_closed_loop(
    client,
    prompts: Iterable[str | Sequence[int]],
    *,
    max_new_tokens: int | Sequence[int],
    seed: int,
    temperature: float,
    routing_keys: Iterable[str] | None,
    max_in_flight: int,
    duration_seconds: float,
    deadline_seconds: float,
    request_id_prefix: str,
) -> ScheduledRun:
    prompt_rows = tuple(prompts)
    if not prompt_rows or max_in_flight < 1 or duration_seconds <= 0:
        raise ValueError("closed-loop requests need prompts, concurrency, and duration")
    keys = tuple(routing_keys) if routing_keys is not None else (None,) * len(prompt_rows)
    budgets = (
        (max_new_tokens,) * len(prompt_rows)
        if isinstance(max_new_tokens, int)
        else tuple(max_new_tokens)
    )
    if len(keys) != len(prompt_rows) or len(budgets) != len(prompt_rows):
        raise ValueError("closed-loop corpus needs one key and budget per prompt")

    started = time.perf_counter()
    stop_at = started + duration_seconds
    lock = threading.Lock()
    failed = threading.Event()
    sequence = 0
    results: dict[int, GenerationResult] = {}
    outcomes: dict[int, RequestOutcome] = {}

    def worker() -> None:
        nonlocal sequence
        while not failed.is_set() and time.perf_counter() < stop_at:
            with lock:
                index = sequence
                sequence += 1
            corpus_index = index % len(prompt_rows)
            request_id = f"{request_id_prefix}-{index:06d}"
            offered_ns = time.monotonic_ns()
            admitted_ns = time.monotonic_ns()
            try:
                rows, _ = client.run_batch(
                    (prompt_rows[corpus_index],),
                    max_new_tokens=budgets[corpus_index],
                    seed=seed + index,
                    temperature=temperature,
                    routing_key=keys[corpus_index],
                    request_ids=(request_id,),
                    timeout_seconds=deadline_seconds,
                )
                result = rows[0]
                outcome = RequestOutcome(
                    request_id,
                    "completed",
                    offered_ns,
                    admitted_ns,
                    time.monotonic_ns(),
                )
                with lock:
                    results[index] = result
                    outcomes[index] = outcome
            except Exception as error:
                status = _request_error(error)
                if status == "error":
                    failed.set()
                with lock:
                    outcomes[index] = RequestOutcome(
                        request_id,
                        status,
                        offered_ns,
                        admitted_ns,
                        time.monotonic_ns(),
                        f"{type(error).__name__}: {error}",
                    )

    with ThreadPoolExecutor(max_workers=max_in_flight) as pool:
        futures = tuple(pool.submit(worker) for _ in range(max_in_flight))
        for future in futures:
            future.result()
    return ScheduledRun(
        tuple(results[index] for index in sorted(results)),
        tuple(outcomes[index] for index in sorted(outcomes)),
        time.perf_counter() - started,
    )


def _native_events(meta: dict, output_ids: list[int]) -> tuple[int, ...]:
    events = meta.get("native_token_timestamp_events")
    if not isinstance(events, list) or len(events) != len(output_ids):
        raise RuntimeError("final response lacks complete native token timestamps")
    timestamps: list[int] = []
    for index, (event, token_id) in enumerate(zip(events, output_ids, strict=True)):
        if not isinstance(event, dict):
            raise RuntimeError("native token timestamp event is malformed")
        if event["token_index"] != index or event["token_id"] != token_id:
            raise RuntimeError("native token timestamps changed token trajectory")
        observed = event.get("committed_ns", event.get("observed_ns"))
        if not isinstance(observed, int) or observed < 0 or (timestamps and observed < timestamps[-1]):
            raise RuntimeError("native token timestamps are not monotone")
        timestamps.append(observed)
    return tuple(timestamps)


def _consume_stream(response, request_ids: tuple[str, ...], started: float) -> tuple[GenerationResult, ...]:
    count = len(request_ids)
    prior_arrival: list[float | None] = [None] * count
    token_counts = [0] * count
    first_token: list[float | None] = [None] * count
    intervals: list[list[float]] = [[] for _ in request_ids]
    finals: list[dict | None] = [None] * count
    finished: list[float | None] = [None] * count
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        meta = chunk.get("meta_info")
        if not isinstance(meta, dict):
            raise RuntimeError("stream response is missing metadata")
        index = chunk.get("index", 0)
        if not isinstance(index, int) or not 0 <= index < count:
            raise RuntimeError("stream response has an invalid batch index")
        if count > 1 and meta.get("id") != request_ids[index]:
            raise RuntimeError("stream response request identity changed")
        arrived = time.perf_counter()
        completion = int(meta.get("completion_tokens", token_counts[index]))
        new_tokens = completion - token_counts[index]
        if new_tokens < 0:
            raise RuntimeError("stream completion count regressed")
        if new_tokens:
            if prior_arrival[index] is None:
                first_token[index] = (arrived - started) * 1000
                intervals[index].extend([0.0] * max(0, new_tokens - 1))
            else:
                intervals[index].append((arrived - prior_arrival[index]) * 1000)
                intervals[index].extend([0.0] * max(0, new_tokens - 1))
            prior_arrival[index] = arrived
            token_counts[index] = completion
        finals[index] = chunk
        if meta.get("finish_reason") is not None:
            finished[index] = arrived
    results: list[GenerationResult] = []
    for index, request_id in enumerate(request_ids):
        final = finals[index]
        if final is None or first_token[index] is None or finished[index] is None:
            raise RuntimeError("stream did not return a complete result")
        output_ids = final.get("output_ids")
        if not isinstance(output_ids, list) or len(output_ids) != token_counts[index]:
            raise RuntimeError("final response lacks the full output-token trajectory")
        meta = final.get("meta_info", {})
        prompt_tokens = meta.get("prompt_tokens")
        if not isinstance(prompt_tokens, int) or prompt_tokens < 1:
            raise RuntimeError("final response lacks prompt-token count")
        reason = meta.get("finish_reason")
        stop_details = dict(reason) if isinstance(reason, dict) else None
        if stop_details is not None:
            reason = stop_details.get("type")
        native_timestamps = _native_events(meta, output_ids)
        native_intervals = tuple(
            (right - left) / 1_000_000
            for left, right in zip(native_timestamps, native_timestamps[1:])
        )
        results.append(
            GenerationResult(
                request_id=request_id,
                input_tokens=prompt_tokens,
                completion_tokens=token_counts[index],
                ttft_ms=float(first_token[index]),
                inter_token_ms=native_intervals,
                elapsed_seconds=float(finished[index] - started),
                stop_reason=None if reason is None else str(reason),
                output_ids=tuple(int(value) for value in output_ids),
                output_text=str(final.get("text", "")),
                native_token_timestamps_ns=native_timestamps,
                stop_details=stop_details,
            )
        )
    return tuple(results)


class SGLangClient:
    def __init__(self, base_url: str, timeout_seconds: int):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=5) as response:
                return response.status == 200
        except Exception:
            pass
        return False

    def reset(self) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/flush_cache?timeout=30", data=b"", method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            if response.status != 200:
                raise RuntimeError("SGLang cache reset failed")

    def start_profile(
        self, *, output_dir: str | None = None, cuda_range: bool = False
    ) -> None:
        body: dict[str, object] = {
            "activities": ["CUDA_PROFILER"] if cuda_range else ["CPU", "GPU"],
            "with_stack": False,
            "record_shapes": False,
        }
        if output_dir is not None:
            body["output_dir"] = output_dir
        request = urllib.request.Request(
            f"{self.base_url}/start_profile",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            if response.status != 200:
                raise RuntimeError("SGLang profiler start failed")

    def stop_profile(self) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/stop_profile", data=b"", method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            if response.status != 200:
                raise RuntimeError("SGLang profiler stop failed")

    def server_info(self) -> dict[str, object]:
        with urllib.request.urlopen(f"{self.base_url}/server_info", timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise RuntimeError("SGLang server info is malformed")
        return payload

    def tokenize(self, text: str) -> tuple[int, ...]:
        request = urllib.request.Request(
            f"{self.base_url}/tokenize",
            data=json.dumps({"prompt": text, "add_special_tokens": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read())
        values = payload.get("tokens") if isinstance(payload, dict) else None
        if not isinstance(values, list) or any(not isinstance(value, int) for value in values):
            raise RuntimeError("SGLang tokenize response lacks input_ids")
        return tuple(values)

    def run_batch(
        self,
        prompts: Iterable[str | Sequence[int]],
        *,
        max_new_tokens: int,
        seed: int,
        temperature: float = 0.0,
        request_id_prefix: str = "request",
        routing_key: str | None = None,
        request_ids: Sequence[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[tuple[GenerationResult, ...], float]:
        prompt_rows = tuple(prompts)
        if not prompt_rows:
            raise ValueError("a request batch cannot be empty")
        request_ids = (
            tuple(request_ids)
            if request_ids is not None
            else tuple(
                f"{request_id_prefix}-{index:05d}"
                for index in range(len(prompt_rows))
            )
        )
        if len(request_ids) != len(prompt_rows) or len(set(request_ids)) != len(request_ids):
            raise ValueError("request IDs must be unique within a batch")
        are_text = all(isinstance(row, str) for row in prompt_rows)
        are_ids = all(not isinstance(row, str) for row in prompt_rows)
        if not (are_text or are_ids):
            raise ValueError("a request batch cannot mix text and token IDs")
        body = {
            "rid": list(request_ids),
            "sampling_params": [
                {
                    "temperature": temperature,
                    "top_k": 1 if temperature == 0 else -1,
                    "top_p": 1.0,
                    "min_p": 0.0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                    "sampling_seed": seed + index,
                }
                for index in range(len(prompt_rows))
            ],
            "stream": True,
            "return_native_token_timestamps": True,
        }
        body["text" if are_text else "input_ids"] = [
            row if isinstance(row, str) else list(row) for row in prompt_rows
        ]
        headers = {"Content-Type": "application/json"}
        if routing_key is not None:
            headers["X-SMG-Routing-Key"] = routing_key
        request = urllib.request.Request(
            f"{self.base_url}/generate",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(
            request, timeout=timeout_seconds or self.timeout_seconds
        ) as response:
            results = _consume_stream(response, request_ids, started)
        return results, time.perf_counter() - started

    def abort(self, request_id: str) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/abort_request",
            data=json.dumps({"rid": request_id}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            if response.status != 200:
                raise RuntimeError("SGLang request cancellation failed")

    def run_bounded(
        self,
        prompts: Iterable[str | Sequence[int]],
        *,
        max_new_tokens: int | Sequence[int],
        seed: int,
        temperature: float = 0.0,
        routing_keys: Iterable[str] | None = None,
        request_ids: Sequence[str] | None = None,
        max_in_flight: int = 1,
        deadline_seconds: float = 120.0,
    ) -> ScheduledRun:
        return _run_bounded(
            self,
            prompts,
            max_new_tokens=max_new_tokens,
            seed=seed,
            temperature=temperature,
            routing_keys=routing_keys,
            request_ids=request_ids,
            max_in_flight=max_in_flight,
            deadline_seconds=deadline_seconds,
        )

    def run_closed_loop(
        self,
        prompts: Iterable[str | Sequence[int]],
        *,
        max_new_tokens: int | Sequence[int],
        seed: int,
        temperature: float = 0.0,
        routing_keys: Iterable[str] | None = None,
        max_in_flight: int = 1,
        duration_seconds: float = 60.0,
        deadline_seconds: float = 120.0,
        request_id_prefix: str = "closed-loop",
    ) -> ScheduledRun:
        return _run_closed_loop(
            self,
            prompts,
            max_new_tokens=max_new_tokens,
            seed=seed,
            temperature=temperature,
            routing_keys=routing_keys,
            max_in_flight=max_in_flight,
            duration_seconds=duration_seconds,
            deadline_seconds=deadline_seconds,
            request_id_prefix=request_id_prefix,
        )

    def run_scheduled(
        self,
        prompts: Iterable[str | Sequence[int]],
        arrival_offsets: Iterable[float],
        *,
        max_new_tokens: int | Sequence[int],
        seed: int,
        temperature: float = 0.0,
        routing_keys: Iterable[str] | None = None,
        request_ids: Sequence[str] | None = None,
        max_in_flight: int = 256,
        deadline_seconds: float = 120.0,
        drain_seconds: float = 180.0,
    ) -> ScheduledRun:
        prompt_rows = tuple(prompts)
        offsets = tuple(float(value) for value in arrival_offsets)
        if len(prompt_rows) != len(offsets) or not prompt_rows:
            raise ValueError("scheduled requests need one arrival offset per prompt")
        if offsets[0] < 0 or any(right < left for left, right in zip(offsets, offsets[1:])):
            raise ValueError("arrival offsets must be non-negative and monotone")
        keys = tuple(routing_keys) if routing_keys is not None else (None,) * len(prompt_rows)
        if len(keys) != len(prompt_rows):
            raise ValueError("scheduled requests need one routing key per prompt")
        ids = (
            tuple(request_ids)
            if request_ids is not None
            else tuple(f"scheduled-{index:05d}" for index in range(len(prompt_rows)))
        )
        if len(ids) != len(prompt_rows) or len(set(ids)) != len(ids):
            raise ValueError("scheduled requests need unique request IDs")
        budgets = (
            (max_new_tokens,) * len(prompt_rows)
            if isinstance(max_new_tokens, int)
            else tuple(max_new_tokens)
        )
        if len(budgets) != len(prompt_rows):
            raise ValueError("scheduled requests need one output length per prompt")
        if max_in_flight < 1 or deadline_seconds <= 0 or drain_seconds < 0:
            raise ValueError("scheduled request limits must be positive")
        started = time.perf_counter()
        outcomes: list[RequestOutcome | None] = [None] * len(prompt_rows)
        results: dict[int, GenerationResult] = {}

        def submit(index: int, request_id: str) -> GenerationResult:
            rows, _ = self.run_batch(
                (prompt_rows[index],),
                max_new_tokens=budgets[index],
                seed=seed + index,
                temperature=temperature,
                routing_key=keys[index],
                request_ids=(request_id,),
                timeout_seconds=deadline_seconds,
            )
            return rows[0]

        active: dict[Future[GenerationResult], tuple[int, str, int, int]] = {}

        def finish(future: Future[GenerationResult]) -> None:
            index, request_id, offered_ns, admitted_ns = active.pop(future)
            finished_ns = time.monotonic_ns()
            try:
                results[index] = future.result()
                outcomes[index] = RequestOutcome(
                    request_id, "completed", offered_ns, admitted_ns, finished_ns
                )
            except TimeoutError as error:
                outcomes[index] = RequestOutcome(
                    request_id,
                    "timed_out",
                    offered_ns,
                    admitted_ns,
                    finished_ns,
                    f"{type(error).__name__}: {error}",
                )
            except urllib.error.URLError as error:
                status = "timed_out" if isinstance(error.reason, TimeoutError) else "error"
                outcomes[index] = RequestOutcome(
                    request_id,
                    status,
                    offered_ns,
                    admitted_ns,
                    finished_ns,
                    f"{type(error).__name__}: {error}",
                )
            except Exception as error:
                outcomes[index] = RequestOutcome(
                    request_id,
                    "error",
                    offered_ns,
                    admitted_ns,
                    finished_ns,
                    f"{type(error).__name__}: {error}",
                )

        with ThreadPoolExecutor(max_workers=max_in_flight) as pool:
            for index, offset in enumerate(offsets):
                delay = started + offset - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                for future in tuple(active):
                    if future.done():
                        finish(future)
                request_id = ids[index]
                offered_ns = time.monotonic_ns()
                if len(active) >= max_in_flight:
                    outcomes[index] = RequestOutcome(
                        request_id, "unfinished", offered_ns, None, None, "admission limit"
                    )
                    continue
                admitted_ns = time.monotonic_ns()
                future = pool.submit(submit, index, request_id)
                active[future] = (index, request_id, offered_ns, admitted_ns)

            drain_deadline = time.perf_counter() + drain_seconds
            while active and time.perf_counter() < drain_deadline:
                done, _ = wait(
                    tuple(active),
                    timeout=min(0.1, max(0.0, drain_deadline - time.perf_counter())),
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    finish(future)
            for future, (index, request_id, offered_ns, admitted_ns) in tuple(
                active.items()
            ):
                future.cancel()
                try:
                    self.abort(request_id)
                except Exception:
                    pass
                outcomes[index] = RequestOutcome(
                    request_id,
                    "unfinished",
                    offered_ns,
                    admitted_ns,
                    time.monotonic_ns(),
                    "drain deadline",
                )
                active.pop(future)
        return ScheduledRun(
            tuple(results[index] for index in sorted(results)),
            tuple(outcome for outcome in outcomes if outcome is not None),
            time.perf_counter() - started,
        )
