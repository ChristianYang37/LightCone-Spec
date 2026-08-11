"""High-load streaming client with true independent measurement intervals."""

from __future__ import annotations

import json
import math
import time
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass

from lightcone_spec.execution import ControlledExecutionPolicy

_MEASURED_METHODS = {
    "static",
    "tts",
    "naive_async",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
}


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    input_tokens: int
    completion_tokens: int
    ttft_ms: float
    inter_token_ms: tuple[float, ...]
    token_arrival_ms: tuple[float, ...]
    elapsed_s: float
    stop_reason: str | None
    response: dict


@dataclass(frozen=True)
class ServerSnapshot:
    target_calls: int
    accepted_drafts: int
    committed_tokens: int
    verified_drafts: int
    verification_waste: int
    oom_events: int
    retractions: int
    peak_hbm_bytes: int
    kv_bytes: int
    kv_token_capacity: int
    batch_fill: float
    queue_occupancy: float
    graph_replay_hit_rate: float
    adaptation: dict | None

    @classmethod
    def parse(cls, payload: dict) -> ServerSnapshot:
        states = payload.get("internal_states")
        if states is not None:
            if (
                not isinstance(states, list)
                or len(states) != 1
                or not isinstance(states[0], dict)
            ):
                raise RuntimeError(
                    "speed-study collection requires exactly one SGLang DP state"
                )
            state = states[0]
        else:
            # Current SGLang exposes a per-DP list.  The singular envelope is
            # retained for older patchsets and small protocol fixtures.
            state = payload.get("internal_state", payload)
        if not isinstance(state, dict):
            raise RuntimeError(  # noqa: TRY004 - malformed remote response
                "SGLang server state is not an object"
            )
        metrics = state.get("speed_study_metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError(  # noqa: TRY004 - malformed remote response
                "patched SGLang speed-study metrics are missing"
            )
        required = {
            "target_calls",
            "accepted_drafts",
            "committed_tokens",
            "verified_drafts",
            "verification_waste",
            "oom_events",
            "retractions",
            "peak_hbm_bytes",
            "kv_bytes",
            "kv_token_capacity",
            "batch_fill",
            "queue_occupancy",
            "graph_replay_hit_rate",
        }
        if not required <= set(metrics):
            missing = sorted(required - set(metrics))
            raise RuntimeError(
                "patched SGLang speed metrics are incomplete: "
                + ", ".join(missing)
            )
        record = state.get("speculative_adaptation_info_record")
        adaptation = None
        if record is not None:
            if not isinstance(record, dict) or not isinstance(
                record.get("online_adaptation"), dict
            ):
                raise RuntimeError("adaptation diagnostics are malformed")
            adaptation = record["online_adaptation"]
        snapshot = cls(
            target_calls=int(metrics["target_calls"]),
            accepted_drafts=int(metrics["accepted_drafts"]),
            committed_tokens=int(metrics["committed_tokens"]),
            verified_drafts=int(metrics["verified_drafts"]),
            verification_waste=int(metrics["verification_waste"]),
            oom_events=int(metrics["oom_events"]),
            retractions=int(metrics["retractions"]),
            peak_hbm_bytes=int(metrics["peak_hbm_bytes"]),
            kv_bytes=int(metrics["kv_bytes"]),
            kv_token_capacity=int(metrics["kv_token_capacity"]),
            batch_fill=float(metrics["batch_fill"]),
            queue_occupancy=float(metrics["queue_occupancy"]),
            graph_replay_hit_rate=float(metrics["graph_replay_hit_rate"]),
            adaptation=adaptation,
        )
        counts = (
            snapshot.target_calls,
            snapshot.accepted_drafts,
            snapshot.committed_tokens,
            snapshot.verified_drafts,
            snapshot.verification_waste,
            snapshot.oom_events,
            snapshot.retractions,
            snapshot.peak_hbm_bytes,
            snapshot.kv_bytes,
            snapshot.kv_token_capacity,
        )
        if any(value < 0 for value in counts):
            raise RuntimeError("SGLang speed counters cannot be negative")
        if (
            snapshot.committed_tokens
            != snapshot.accepted_drafts + snapshot.target_calls
            or snapshot.verification_waste
            != snapshot.verified_drafts - snapshot.accepted_drafts
        ):
            raise RuntimeError("SGLang speculative counters are inconsistent")
        load_values = (
            snapshot.batch_fill,
            snapshot.queue_occupancy,
            snapshot.graph_replay_hit_rate,
        )
        if not all(math.isfinite(value) for value in load_values):
            raise RuntimeError("SGLang load metrics must be finite")
        if snapshot.batch_fill < 0 or snapshot.queue_occupancy < 0:
            raise RuntimeError("SGLang load metrics cannot be negative")
        if not 0.0 <= snapshot.graph_replay_hit_rate <= 1.0:
            raise RuntimeError("SGLang graph replay rate is outside [0, 1]")
        return snapshot


@dataclass(frozen=True)
class MethodRun:
    results: tuple[GenerationResult, ...]
    elapsed_s: float
    before: ServerSnapshot
    after: ServerSnapshot


def _consume_generation_stream(
    response,
    *,
    request_ids: tuple[str, ...],
    started: float,
    require_batch_identity: bool,
) -> tuple[GenerationResult, ...]:
    """Decode one SSE connection into request-local arrival timelines."""
    count = len(request_ids)
    previous_arrivals: list[float | None] = [None] * count
    previous_tokens = [0] * count
    ttft_ms: list[float | None] = [None] * count
    intervals: list[list[float]] = [[] for _ in request_ids]
    arrivals: list[list[float]] = [[] for _ in request_ids]
    finals: list[dict | None] = [None] * count
    finished_at: list[float | None] = [None] * count
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
            raise TypeError("stream response metadata is malformed")
        if require_batch_identity:
            index = chunk.get("index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < count
                or meta.get("id") != request_ids[index]
            ):
                raise RuntimeError("batch stream response identity is malformed")
        else:
            if count != 1 or meta.get("id", request_ids[0]) != request_ids[0]:
                raise RuntimeError("single stream response identity is malformed")
            index = 0
        arrived = time.perf_counter()
        completion_tokens = int(meta.get("completion_tokens", previous_tokens[index]))
        new_tokens = completion_tokens - previous_tokens[index]
        if new_tokens < 0:
            raise RuntimeError("stream completion token count regressed")
        if new_tokens:
            arrivals[index].extend([arrived * 1000.0] * new_tokens)
            previous = previous_arrivals[index]
            if previous is None:
                ttft_ms[index] = (arrived - started) * 1000.0
                intervals[index].extend([0.0] * max(0, new_tokens - 1))
            else:
                intervals[index].append((arrived - previous) * 1000.0)
                intervals[index].extend([0.0] * max(0, new_tokens - 1))
            previous_arrivals[index] = arrived
            previous_tokens[index] = completion_tokens
        finals[index] = chunk
        if meta.get("finish_reason") is not None:
            finished_at[index] = arrived

    results: list[GenerationResult] = []
    for index, request_id in enumerate(request_ids):
        final = finals[index]
        first_token_ms = ttft_ms[index]
        finish = finished_at[index]
        if (
            final is None
            or previous_tokens[index] < 1
            or first_token_ms is None
            or finish is None
        ):
            raise RuntimeError("stream result coverage is incomplete")
        if len(intervals[index]) != previous_tokens[index] - 1:
            raise RuntimeError("ITL sample count does not match completion tokens")
        output_ids = final.get("output_ids")
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != previous_tokens[index]
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in output_ids
            )
        ):
            raise RuntimeError(
                "final stream chunk lacks the complete output-token trajectory"
            )
        meta = final.get("meta_info")
        if not isinstance(meta, dict) or "prompt_tokens" not in meta:
            raise RuntimeError("final stream chunk lacks prompt token telemetry")
        input_tokens = int(meta["prompt_tokens"])
        if input_tokens < 1:
            raise RuntimeError("prompt token count must be positive")
        finish_reason = meta.get("finish_reason")
        if isinstance(finish_reason, dict):
            stop_reason = str(finish_reason.get("type") or "unknown")
        elif finish_reason is None:
            stop_reason = None
        else:
            stop_reason = str(finish_reason)
        results.append(
            GenerationResult(
                request_id=request_id,
                input_tokens=input_tokens,
                completion_tokens=previous_tokens[index],
                ttft_ms=first_token_ms,
                inter_token_ms=tuple(intervals[index]),
                token_arrival_ms=tuple(arrivals[index]),
                elapsed_s=finish - started,
                stop_reason=stop_reason,
                response=final,
            )
        )
    return tuple(results)


class SGLangHTTPClient:
    def __init__(self, base_url: str, timeout_s: float = 3600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def reset_engine(self) -> None:
        # SGLang performs a tiny asynchronous startup warmup after its HTTP
        # endpoint becomes reachable.  Ask the scheduler to defer this flush
        # until it is truly idle instead of racing that request.
        request = urllib.request.Request(
            f"{self.base_url}/flush_cache?timeout=30",
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = response.read().decode("utf-8")
        if "Cache flushed" not in payload:
            raise RuntimeError("SGLang did not acknowledge engine reset")

    def server_info(self) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/server_info", method="GET"
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise RuntimeError(  # noqa: TRY004 - malformed remote response
                "SGLang server_info response is not an object"
            )
        return payload

    def tokenize_prompts(self, prompts: Iterable[str]) -> tuple[tuple[int, ...], int]:
        values = tuple(prompts)
        if not values or any(not isinstance(prompt, str) or not prompt for prompt in values):
            raise ValueError("tokenization requires non-empty prompt strings")
        request = urllib.request.Request(
            f"{self.base_url}/tokenize",
            data=json.dumps({"prompt": list(values)}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise RuntimeError(  # noqa: TRY004 - malformed remote response
                "SGLang tokenize response is not an object"
            )
        counts = payload.get("count")
        max_model_len = payload.get("max_model_len")
        if (
            not isinstance(counts, list)
            or len(counts) != len(values)
            or isinstance(max_model_len, bool)
            or not isinstance(max_model_len, int)
        ):
            raise RuntimeError("SGLang tokenize response is incomplete")
        resolved = tuple(int(count) for count in counts)
        if any(count < 1 for count in resolved) or max_model_len < 1:
            raise RuntimeError("SGLang returned invalid tokenization limits")
        return resolved, max_model_len

    def stream_generate(self, payload: dict) -> GenerationResult:
        request_id = str(payload["rid"])
        body = dict(payload)
        body["stream"] = True
        request = urllib.request.Request(
            f"{self.base_url}/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            results = _consume_generation_stream(
                response,
                request_ids=(request_id,),
                started=started,
                require_batch_identity=False,
            )
        return results[0]

    def stream_generate_batch(
        self, payloads: Iterable[dict]
    ) -> tuple[tuple[GenerationResult, ...], float]:
        """Submit one ordered native batch and retain per-request stream timing.

        One HTTP batch gives SGLang a deterministic admission order while its
        own ``max_running_requests`` remains the sole concurrency limiter.
        This avoids host-thread races changing long greedy trajectories and
        also removes one client connection/thread per queued request.
        """
        requests = tuple(payloads)
        if not requests:
            raise ValueError("a non-empty batch is required")
        allowed = {"rid", "text", "sampling_params"}
        if any(set(payload) != allowed for payload in requests):
            raise ValueError("formal batch payload fields are not canonical")
        request_ids = tuple(str(payload["rid"]) for payload in requests)
        if any(not request_id for request_id in request_ids) or len(
            set(request_ids)
        ) != len(request_ids):
            raise ValueError("batch request IDs must be non-empty and unique")
        if any(
            not isinstance(payload["text"], str)
            or not payload["text"]
            or not isinstance(payload["sampling_params"], dict)
            for payload in requests
        ):
            raise ValueError("batch prompts and sampling parameters are malformed")

        body = {
            "rid": list(request_ids),
            "text": [payload["text"] for payload in requests],
            "sampling_params": [payload["sampling_params"] for payload in requests],
            "stream": True,
        }
        request = urllib.request.Request(
            f"{self.base_url}/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            results = _consume_generation_stream(
                response,
                request_ids=request_ids,
                started=started,
                require_batch_identity=True,
            )
        ended = time.perf_counter()
        return results, ended - started

    def run_loaded_batch(
        self,
        payloads: Iterable[dict],
        *,
        concurrency: int,
    ) -> tuple[tuple[GenerationResult, ...], float]:
        """Submit one complete queue; SGLang owns the locked admission limit."""
        requests = tuple(payloads)
        if concurrency < 1 or not requests:
            raise ValueError("a non-empty positive-concurrency batch is required")
        if len(requests) < concurrency:
            raise ValueError(
                "the request queue cannot reach the registered concurrency"
            )
        results, elapsed = self.stream_generate_batch(requests)
        if {result.request_id for result in results} != {
            str(payload["rid"]) for payload in requests
        }:
            raise RuntimeError("native batch response coverage is incomplete")
        return tuple(sorted(results, key=lambda row: row.request_id)), elapsed


def _require_complete_output_stream(server_info: dict) -> None:
    try:
        ControlledExecutionPolicy().validate_server_info(
            server_info, role="speculative"
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "formal evidence requires the registered controlled execution policy "
            "and non-incremental streaming"
        ) from exc


def independent_method_run(
    client: SGLangHTTPClient,
    *,
    method: str,
    payloads: Iterable[dict],
    concurrency: int,
    adaptation_group_id: str | None,
) -> MethodRun:
    """One measured method owns one clean cohort and registered timing unit."""
    if method not in _MEASURED_METHODS:
        raise ValueError("unknown measured method")
    if method != "static" and not adaptation_group_id:
        raise ValueError("adapted methods require a cohort identity")
    client.reset_engine()
    before_info = client.server_info()
    _require_complete_output_stream(before_info)
    before = ServerSnapshot.parse(before_info)
    if before.target_calls != 0:
        raise RuntimeError("engine reset did not clear target-call counters")
    results, elapsed_s = client.run_loaded_batch(
        payloads,
        concurrency=concurrency,
    )
    after_info = client.server_info()
    _require_complete_output_stream(after_info)
    after = ServerSnapshot.parse(after_info)
    if after.target_calls < 1:
        raise RuntimeError("completed run reported no target calls")
    if method == "static" and after.adaptation is not None:
        raise RuntimeError("Static unexpectedly allocated adaptation state")
    if method != "static" and after.adaptation is None:
        raise RuntimeError("adapted run lacks adaptation diagnostics")
    return MethodRun(results, elapsed_s, before, after)
