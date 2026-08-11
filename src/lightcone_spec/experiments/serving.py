"""Thin bindings to the pinned SGLang ``bench_serving`` HTTP primitive.

The industrial executor owns arrival, admission, cancellation, and evidence
semantics.  This module deliberately reuses SGLang's official asynchronous
``async_request_sglang_generate`` implementation for the network request.  It
does not reproduce the upstream HTTP client or its benchmark metric reducer.

Upstream's request result reports ITLs after distributing one SSE chunk gap
over every token in that chunk.  Those values are unsuitable for a p99 claim.
The adapter therefore retains only a conservative coalesced chunk observation;
it never turns the upstream distributed values into token timestamps.
"""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from lightcone_spec.experiments.load import (
    FrozenSamplingParameters,
    ImmutableRequest,
    TokenChunkTiming,
)
from lightcone_spec.experiments.registry import content_sha256
from lightcone_spec.sglang_bridge.checkout import verify_patched_checkout


@dataclass(frozen=True)
class BoundServingRequest:
    """One immutable request plus its scheduler-owned route assignment."""

    request_id: str
    namespace: str
    split: str
    ordinal: int
    input_token_ids: tuple[int, ...]
    requested_output_tokens: int
    arrival_us: int
    cancellation_offset_us: int | None
    cohort_id: str
    cohort_sha256: str
    route_id: str
    sampling: FrozenSamplingParameters

    @classmethod
    def create(
        cls,
        request: ImmutableRequest,
        *,
        route_id: str,
    ) -> BoundServingRequest:
        request.validate()
        if not isinstance(route_id, str) or not route_id or "\n" in route_id:
            raise ValueError("route_id must be non-empty single-line text")
        value = cls(
            request_id=request.request_id,
            namespace=request.namespace,
            split=request.split,
            ordinal=request.ordinal,
            input_token_ids=request.input_token_ids,
            requested_output_tokens=request.requested_output_tokens,
            arrival_us=request.arrival_us,
            cancellation_offset_us=request.cancellation_offset_us,
            cohort_id=request.cohort_id,
            cohort_sha256=content_sha256(request.cohort_id),
            route_id=route_id,
            sampling=request.sampling,
        )
        value.validate(request)
        return value

    def validate(self, request: ImmutableRequest | None = None) -> None:
        if (
            not self.request_id
            or not self.namespace
            or not self.split
            or not self.route_id
            or not self.cohort_id
        ):
            raise ValueError("serving request identities must be non-empty")
        if isinstance(self.ordinal, bool) or self.ordinal < 0:
            raise ValueError("serving request ordinal must be non-negative")
        if self.cohort_sha256 != content_sha256(self.cohort_id):
            raise ValueError("cohort digest does not match its immutable identity")
        self.sampling.validate()
        if request is not None:
            request.validate()
            actual = (
                self.request_id,
                self.namespace,
                self.split,
                self.ordinal,
                self.input_token_ids,
                self.requested_output_tokens,
                self.arrival_us,
                self.cancellation_offset_us,
                self.cohort_id,
                self.sampling,
            )
            expected = (
                request.request_id,
                request.namespace,
                request.split,
                request.ordinal,
                request.input_token_ids,
                request.requested_output_tokens,
                request.arrival_us,
                request.cancellation_offset_us,
                request.cohort_id,
                request.sampling,
            )
            if actual != expected:
                raise ValueError("serving request differs from the immutable corpus")

    @property
    def sha256(self) -> str:
        self.validate()
        return content_sha256(self)


@dataclass(frozen=True)
class BenchServingResult:
    """Lossless subset of one official bench request result.

    ``chunks`` contains only observations supported by the adapter.  A
    multi-token response is represented as one coalesced chunk without
    per-token timestamps, while ``ttft_us`` retains the independently observed
    first response time.  Downstream ITL coverage therefore remains explicitly
    incomplete instead of receiving invented zero or evenly divided ITLs.
    """

    request_id: str
    success: bool
    generated_text: str
    output_tokens: int
    latency_us: int
    stop_reason: str | None
    error_code: str | None
    chunks: tuple[TokenChunkTiming, ...]
    generated_token_ids: tuple[int, ...] | None
    ttft_us: int | None = None

    def validate(self, request: BoundServingRequest) -> None:
        if self.request_id != request.request_id:
            raise ValueError("bench result belongs to another request")
        if (
            self.output_tokens < 0
            or self.output_tokens > request.requested_output_tokens
        ):
            raise ValueError("bench output length is outside the requested bound")
        if self.latency_us < 0:
            raise ValueError("bench latency cannot be negative")
        if self.ttft_us is not None and (
            not isinstance(self.ttft_us, int)
            or isinstance(self.ttft_us, bool)
            or self.ttft_us < 0
            or self.ttft_us > self.latency_us
            or self.output_tokens < 1
        ):
            raise ValueError("bench TTFT is outside the request lifetime")
        if self.success:
            if self.output_tokens < 1 or self.error_code is not None:
                raise ValueError(
                    "successful bench results require output without error"
                )
        elif not self.error_code:
            raise ValueError("failed bench results require a stable error code")
        if self.generated_token_ids is not None and (
            len(self.generated_token_ids) != self.output_tokens
            or any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
                for token_id in self.generated_token_ids
            )
        ):
            raise ValueError("generated token IDs do not cover the output exactly")
        covered = 0
        for chunk in self.chunks:
            chunk.validate()
            if (
                chunk.request_id != self.request_id
                or chunk.first_token_index != covered
            ):
                raise ValueError("bench chunks do not form an ordered request prefix")
            covered += chunk.token_count
        if covered != self.output_tokens:
            raise ValueError("bench chunks do not cover the reported output tokens")


class BenchServingTransport(Protocol):
    """Network boundary injected into the industrial executor."""

    async def submit(
        self,
        request: BoundServingRequest,
        *,
        base_url: str,
        served_model: str,
    ) -> BenchServingResult: ...

    async def abort(
        self,
        request_id: str,
        *,
        base_url: str,
    ) -> None: ...


_OfficialRequest = Callable[..., Awaitable[Any]]


class PinnedBenchServingTransport:
    """Adapter around the exact pinned official async SGLang request function."""

    def __init__(
        self,
        *,
        request_type: type,
        request_callable: _OfficialRequest,
        set_global_args: Callable[[Any], None],
        session_factory: Callable[[], Any] | None = None,
        headers_factory: Callable[[], dict[str, str]] | None = None,
        module_identity: str,
    ) -> None:
        if module_identity != "sglang.benchmark.serving.async_request_sglang_generate":
            raise ValueError("official bench module identity is not pinned")
        self._request_type = request_type
        self._request_callable = request_callable
        self._set_global_args = set_global_args
        self._session_factory = session_factory
        self._headers_factory = headers_factory
        self.module_identity = module_identity
        self._set_global_args(
            SimpleNamespace(
                cache_report=False,
                disable_ignore_eos=False,
                disable_stream=False,
                logprob_start_len=-1,
                return_logprob=False,
                return_routed_experts=False,
                temperature=0.0,
                token_ids_logprob=None,
                top_logprobs_num=0,
                top_p=1.0,
            )
        )

    @classmethod
    def from_checkout(cls, checkout: str | Path) -> PinnedBenchServingTransport:
        """Load the official request primitive only from a verified checkout."""

        verified = verify_patched_checkout(checkout)
        python_root = (verified / "python").resolve()
        loaded = sys.modules.get("sglang")
        if loaded is not None:
            loaded_path = Path(str(getattr(loaded, "__file__", ""))).resolve()
            if not loaded_path.is_relative_to(python_root):
                raise RuntimeError(
                    "another SGLang installation is already imported; refusing "
                    "an ambiguous pinned bench binding"
                )
        sys.path.insert(0, str(python_root))
        try:
            module = importlib.import_module("sglang.benchmark.serving")
        finally:
            if sys.path[0] == str(python_root):
                sys.path.pop(0)
        module_path = Path(str(getattr(module, "__file__", ""))).resolve()
        if not module_path.is_relative_to(python_root):
            raise RuntimeError(
                "bench_serving was not imported from the verified checkout"
            )
        return cls(
            request_type=module.RequestFuncInput,
            request_callable=module.async_request_sglang_generate,
            set_global_args=module.set_global_args,
            session_factory=module._create_bench_client_session,
            headers_factory=module.get_request_headers,
            module_identity=("sglang.benchmark.serving.async_request_sglang_generate"),
        )

    async def submit(
        self,
        request: BoundServingRequest,
        *,
        base_url: str,
        served_model: str,
    ) -> BenchServingResult:
        request.validate()
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("bench base_url must be an HTTP(S) URL")
        if not served_model:
            raise ValueError("served_model must not be empty")
        sampling = dict(request.sampling.items)
        requested = sampling.get("max_new_tokens")
        if requested is not None and requested != request.requested_output_tokens:
            raise ValueError("sampling max_new_tokens differs from immutable request")
        sampling["max_new_tokens"] = request.requested_output_tokens
        value = self._request_type(
            prompt=list(request.input_token_ids),
            api_url=base_url.rstrip("/") + "/generate",
            prompt_len=len(request.input_token_ids),
            output_len=request.requested_output_tokens,
            model=served_model,
            lora_name=None,
            image_data=None,
            extra_request_body={
                "rid": request.request_id,
                "sampling_params": sampling,
            },
            timestamp=request.arrival_us / 1000.0,
            routing_key=request.route_id,
        )
        raw = await self._request_callable(request_func_input=value, pbar=None)
        latency = float(raw.latency)
        if not math.isfinite(latency) or latency < 0:
            raise RuntimeError("official bench result has invalid latency")
        output_tokens = int(raw.output_len)
        latency_us = round(latency * 1_000_000)
        ttft = float(raw.ttft)
        if not math.isfinite(ttft) or ttft < 0 or ttft > latency:
            raise RuntimeError("official bench result has invalid TTFT")
        ttft_us = round(ttft * 1_000_000)
        chunks = (
            (
                TokenChunkTiming(
                    request_id=request.request_id,
                    first_token_index=0,
                    token_count=output_tokens,
                    chunk_observed_at_us=(
                        ttft_us if output_tokens == 1 else latency_us
                    ),
                    per_token_observed_at_us=None,
                ),
            )
            if output_tokens > 0
            else ()
        )
        success = bool(raw.success)
        result = BenchServingResult(
            request_id=request.request_id,
            success=success,
            generated_text=str(raw.generated_text),
            output_tokens=output_tokens,
            latency_us=latency_us,
            stop_reason=(
                "length"
                if success and output_tokens == request.requested_output_tokens
                else ("server_stop" if success else None)
            ),
            error_code=None if success else "bench_request_failed",
            chunks=chunks,
            generated_token_ids=(
                tuple(int(token_id) for token_id in raw.generated_token_ids)
                if getattr(raw, "generated_token_ids", None) is not None
                else None
            ),
            ttft_us=ttft_us if output_tokens > 0 else None,
        )
        result.validate(request)
        return result

    async def abort(self, request_id: str, *, base_url: str) -> None:
        """Use SGLang's native abort endpoint while retaining the result stream."""

        if not request_id or not base_url.startswith(("http://", "https://")):
            raise ValueError("abort requires a request ID and HTTP(S) base URL")
        if self._session_factory is None:
            raise RuntimeError("official bench session factory is unavailable")
        async with (
            self._session_factory() as session,
            session.post(
                url=base_url.rstrip("/") + "/abort_request",
                json={"rid": request_id, "abort_all": False},
                headers=(self._headers_factory() if self._headers_factory else {}),
            ) as response,
        ):
            if int(response.status) != 200:
                raise RuntimeError("SGLang did not acknowledge request abort")


def official_bench_argv(
    *,
    base_url: str,
    served_model: str,
    request_count: int,
    concurrency: int,
    arrival_kind: str,
) -> tuple[str, ...]:
    """Return the audit-equivalent official CLI surface for the adapter.

    Execution uses the official async request function directly because the
    CLI cannot accept the repository's already-tokenized immutable corpus.
    The argv is provenance, not a second command executed by the runner.
    """

    if request_count < 1 or concurrency < 1:
        raise ValueError("bench request count and concurrency must be positive")
    if arrival_kind not in {
        "closed_loop",
        "poisson",
        "immediate_burst",
        "external_shape",
    }:
        raise ValueError("arrival_kind is outside the registered load generators")
    return (
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--base-url",
        base_url,
        "--model",
        served_model,
        "--num-prompts",
        str(request_count),
        "--max-concurrency",
        str(concurrency),
        "--disable-tqdm",
    )
