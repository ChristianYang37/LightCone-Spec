from __future__ import annotations

import asyncio

import pytest

from lightcone_spec.experiments.load import (
    FrozenSamplingParameters,
    ImmutableRequest,
    RequestTemplate,
    TokenChunkTiming,
)
from lightcone_spec.experiments.serving import BenchServingResult, BoundServingRequest
from lightcone_spec.orchestration.executor import (
    RegisteredServingExecutionPolicy,
    execute_registered_serving_phase,
)
from lightcone_spec.orchestration.live_sglang import _execute_source_owned_phase


def _request(
    ordinal: int,
    *,
    split: str = "confirmation",
    arrival_us: int = 0,
    cancellation_offset_us: int | None = None,
) -> BoundServingRequest:
    immutable = ImmutableRequest.create(
        namespace="registered-serving-test",
        split=split,  # type: ignore[arg-type]
        ordinal=ordinal,
        template=RequestTemplate(
            input_token_ids=(ordinal + 1,),
            requested_output_tokens=1,
            sampling=FrozenSamplingParameters.from_mapping(
                {
                    "ignore_eos": True,
                    "max_new_tokens": 1,
                    "sampling_seed": ordinal,
                    "temperature": 0.0,
                    "top_p": 1.0,
                }
            ),
            cancellation_offset_us=cancellation_offset_us,
        ),
        arrival_us=arrival_us,
        cohort_id=f"cohort-{ordinal % 4}",
    )
    return BoundServingRequest.create(immutable, route_id="registered-route")


class _Transport:
    def __init__(
        self,
        *,
        delay_by_id: dict[str, float] | None = None,
        fail_ids: set[str] | None = None,
    ) -> None:
        self.delay_by_id = delay_by_id or {}
        self.fail_ids = fail_ids or set()
        self.submitted: list[str] = []
        self.aborted: list[str] = []

    async def submit(
        self,
        request: BoundServingRequest,
        *,
        base_url: str,
        served_model: str,
    ) -> BenchServingResult:
        assert base_url == "http://127.0.0.1:30000"
        assert served_model == "model"
        self.submitted.append(request.request_id)
        delay = self.delay_by_id.get(request.request_id, 0.0)
        if delay:
            await asyncio.sleep(delay)
        if request.request_id in self.fail_ids:
            raise OSError("registered transport failure")
        return BenchServingResult(
            request_id=request.request_id,
            success=True,
            generated_text="x",
            output_tokens=1,
            latency_us=max(1, round(delay * 1_000_000)),
            stop_reason="length",
            error_code=None,
            chunks=(
                TokenChunkTiming(
                    request_id=request.request_id,
                    first_token_index=0,
                    token_count=1,
                    chunk_observed_at_us=1,
                    per_token_observed_at_us=(1,),
                ),
            ),
            generated_token_ids=(10_000 + request.ordinal,),
            ttft_us=1,
        )

    async def abort(self, request_id: str, *, base_url: str) -> None:
        assert base_url == "http://127.0.0.1:30000"
        self.aborted.append(request_id)


def _run(
    requests: tuple[BoundServingRequest, ...],
    transport: _Transport,
    *,
    source_kind: str = "scheduled",
    arrival_duration_us: int = 1_000_000,
    request_deadline_us: int = 1_000_000,
    drain_duration_us: int = 1_000_000,
    concurrency: int = 2,
    complete_closed_loop_pool: bool = False,
):
    return asyncio.run(
        execute_registered_serving_phase(
            "scored",
            requests,
            source_kind=source_kind,
            arrival_duration_us=arrival_duration_us,
            request_deadline_us=request_deadline_us,
            drain_duration_us=drain_duration_us,
            concurrency=concurrency,
            transport=transport,
            base_url="http://127.0.0.1:30000",
            served_model="model",
            abort_grace_s=0.2,
            complete_closed_loop_pool=complete_closed_loop_pool,
        )
    )


def test_all_completed_partition_preserves_offer_and_admission() -> None:
    requests = tuple(_request(index) for index in range(4))
    phase = _run(requests, _Transport(), concurrency=2)

    assert tuple(row.request_id for row in phase.lifecycles) == tuple(
        row.request_id for row in requests
    )
    assert all(
        row.offered
        and row.submitted_to_server
        and row.outcome_status == "completed"
        and row.native_terminal_status == "completed"
        and row.offered_at_us is not None
        and row.admitted_at_us is not None
        for row in phase.lifecycles
    )


def test_semaphore_wait_counts_toward_deadline_and_overload_is_data() -> None:
    first, second = (_request(0), _request(1))
    transport = _Transport(delay_by_id={first.request_id: 0.05})
    phase = _run(
        (first, second),
        transport,
        request_deadline_us=20_000,
        drain_duration_us=50_000,
        concurrency=1,
    )

    by_id = {row.request_id: row for row in phase.lifecycles}
    assert by_id[first.request_id].outcome_status == "timed_out"
    assert by_id[first.request_id].submitted_to_server is True
    assert by_id[first.request_id].native_terminal_status == "aborted"
    assert by_id[second.request_id].outcome_status == "rejected"
    assert by_id[second.request_id].submitted_to_server is False
    assert set(transport.submitted) == {first.request_id}
    assert transport.aborted == [first.request_id]


def test_open_loop_drain_caps_an_admitted_request_deadline() -> None:
    request = _request(0, arrival_us=5_000)
    transport = _Transport(delay_by_id={request.request_id: 0.05})
    phase = _run(
        (request,),
        transport,
        arrival_duration_us=10_000,
        request_deadline_us=1_000_000,
        drain_duration_us=10_000,
        concurrency=1,
    )

    lifecycle = phase.lifecycles[0]
    assert lifecycle.outcome_status == "timed_out"
    assert lifecycle.effective_deadline_us == 20_000
    assert lifecycle.terminal_at_us == 20_000
    assert lifecycle.submitted_to_server is True
    assert transport.aborted == [request.request_id]


def test_cancellation_and_transport_unfinished_do_not_cancel_siblings() -> None:
    cancelled = _request(0, cancellation_offset_us=0)
    unfinished = _request(1)
    completed = _request(2)
    transport = _Transport(fail_ids={unfinished.request_id})
    phase = _run((cancelled, unfinished, completed), transport, concurrency=3)

    by_id = {row.request_id: row for row in phase.lifecycles}
    assert by_id[cancelled.request_id].outcome_status == "cancelled"
    assert by_id[cancelled.request_id].submitted_to_server is False
    assert by_id[unfinished.request_id].outcome_status == "unfinished"
    assert by_id[unfinished.request_id].terminal_at_us is None
    assert by_id[unfinished.request_id].submitted_to_server is False
    assert by_id[completed.request_id].outcome_status == "completed"
    assert completed.request_id in transport.submitted


def test_closed_loop_window_leaves_only_scored_pool_rows_unoffered() -> None:
    requests = tuple(_request(index) for index in range(8))
    transport = _Transport(
        delay_by_id={request.request_id: 0.005 for request in requests}
    )
    phase = _run(
        requests,
        transport,
        source_kind="closed_loop",
        arrival_duration_us=1_000,
        request_deadline_us=20_000,
        drain_duration_us=20_000,
        concurrency=2,
    )

    offered = tuple(row for row in phase.lifecycles if row.offered)
    unoffered = tuple(row for row in phase.lifecycles if not row.offered)
    assert len(offered) == 2
    assert len(unoffered) == 6
    assert all(row.outcome_status is None for row in unoffered)


def test_selected_closed_loop_exact_pool_offers_all_11000_rows() -> None:
    requests = tuple(_request(index) for index in range(11_000))
    phase = _run(
        requests,
        _Transport(),
        source_kind="closed_loop",
        arrival_duration_us=1,
        request_deadline_us=60_000_000,
        drain_duration_us=1,
        concurrency=64,
        complete_closed_loop_pool=True,
    )

    assert len(phase.executions) == 11_000
    assert len(phase.lifecycles) == 11_000
    assert all(row.offered for row in phase.lifecycles)
    assert all(row.outcome_status == "completed" for row in phase.lifecycles)


def test_exact_pool_holds_an_unfinished_lane_until_deadline_then_continues() -> None:
    requests = tuple(_request(index) for index in range(4))
    transport = _Transport(fail_ids={requests[0].request_id})
    phase = _run(
        requests,
        transport,
        source_kind="closed_loop",
        arrival_duration_us=1,
        request_deadline_us=1_000,
        drain_duration_us=1,
        concurrency=1,
        complete_closed_loop_pool=True,
    )

    assert len(phase.executions) == 4
    assert all(row.offered for row in phase.lifecycles)
    assert phase.lifecycles[0].outcome_status == "unfinished"
    assert all(row.outcome_status == "completed" for row in phase.lifecycles[1:])
    assert phase.lifecycles[1].offered_at_us == 1_000


def test_registered_warmup_cannot_use_noncompletion_exemption() -> None:
    request = _request(0, split="warmup", cancellation_offset_us=0)
    policy = RegisteredServingExecutionPolicy(
        schema_version=1,
        kind="registered_serving_execution_policy",
        source_kind="scheduled",
        warmup_duration_us=1_000_000,
        arrival_duration_us=1_000_000,
        request_deadline_us=1_000_000,
        drain_duration_us=1_000_000,
        max_concurrency=1,
        complete_closed_loop_pool=False,
    )
    with pytest.raises(RuntimeError, match="warmup requires strict native completion"):
        asyncio.run(
            _execute_source_owned_phase(
                "warmup",
                (request,),
                concurrency=1,
                transport=_Transport(),  # type: ignore[arg-type]
                base_url="http://127.0.0.1:30000",
                served_model="model",
                execution_policy=policy,
            )
        )
