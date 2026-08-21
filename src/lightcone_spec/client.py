"""Thin SGLang HTTP client that retains token trajectories and stream timing."""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
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

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["inter_token_ms"] = list(self.inter_token_ms)
        value["output_ids"] = list(self.output_ids)
        value["native_token_timestamps_ns"] = list(self.native_token_timestamps_ns)
        return value


def _native_events(meta: dict, output_ids: list[int]) -> tuple[int, ...]:
    events = meta.get("native_token_timestamp_events")
    if not isinstance(events, list) or len(events) != len(output_ids):
        raise RuntimeError("final response lacks complete native token timestamps")
    timestamps: list[int] = []
    for index, (event, token_id) in enumerate(zip(events, output_ids, strict=True)):
        if not isinstance(event, dict) or set(event) != {"token_index", "token_id", "observed_ns"}:
            raise RuntimeError("native token timestamp event is malformed")
        if event["token_index"] != index or event["token_id"] != token_id:
            raise RuntimeError("native token timestamps changed token trajectory")
        observed = event["observed_ns"]
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
        if isinstance(reason, dict):
            reason = reason.get("type")
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
            )
        )
    return tuple(results)


class SGLangClient:
    def __init__(self, base_url: str, timeout_seconds: int):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def health(self) -> bool:
        for endpoint in ("/health_generate", "/health"):
            try:
                with urllib.request.urlopen(self.base_url + endpoint, timeout=5) as response:
                    if response.status == 200:
                        return True
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

    def server_info(self) -> dict[str, object]:
        with urllib.request.urlopen(f"{self.base_url}/server_info", timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise RuntimeError("SGLang server info is malformed")
        return payload

    def tokenize(self, text: str) -> tuple[int, ...]:
        request = urllib.request.Request(
            f"{self.base_url}/tokenize",
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read())
        values = payload.get("input_ids") if isinstance(payload, dict) else None
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
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                    "seed": seed + index,
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
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
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

    def run_scheduled(
        self,
        prompts: Iterable[str | Sequence[int]],
        arrival_offsets: Iterable[float],
        *,
        max_new_tokens: int | Sequence[int],
        seed: int,
        temperature: float = 0.0,
        routing_keys: Iterable[str] | None = None,
    ) -> tuple[tuple[GenerationResult, ...], float]:
        prompt_rows = tuple(prompts)
        offsets = tuple(float(value) for value in arrival_offsets)
        if len(prompt_rows) != len(offsets) or not prompt_rows:
            raise ValueError("scheduled requests need one arrival offset per prompt")
        if offsets[0] < 0 or any(right < left for left, right in zip(offsets, offsets[1:])):
            raise ValueError("arrival offsets must be non-negative and monotone")
        keys = tuple(routing_keys) if routing_keys is not None else (None,) * len(prompt_rows)
        if len(keys) != len(prompt_rows):
            raise ValueError("scheduled requests need one routing key per prompt")
        budgets = (
            (max_new_tokens,) * len(prompt_rows)
            if isinstance(max_new_tokens, int)
            else tuple(max_new_tokens)
        )
        if len(budgets) != len(prompt_rows):
            raise ValueError("scheduled requests need one output length per prompt")
        started = time.perf_counter()

        def submit(index: int) -> GenerationResult:
            delay = started + offsets[index] - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            rows, _ = self.run_batch(
                (prompt_rows[index],),
                max_new_tokens=budgets[index],
                seed=seed + index,
                temperature=temperature,
                request_id_prefix=f"scheduled-{index:05d}",
                routing_key=keys[index],
            )
            return rows[0]

        with ThreadPoolExecutor(max_workers=min(256, len(prompt_rows))) as pool:
            results = tuple(pool.map(submit, range(len(prompt_rows))))
        return results, time.perf_counter() - started
