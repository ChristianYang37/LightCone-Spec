from __future__ import annotations

from dataclasses import replace

import pytest

import lightcone_spec.experiments.load as load_module
from lightcone_spec.experiments.load import (
    ExternalShapeRow,
    ExternalWorkloadShape,
    FrozenSamplingParameters,
    ImmutableRequest,
    ProductionLoadPlan,
    ProductionWindow,
    ReplayBinding,
    RequestOutcome,
    RequestTemplate,
    TokenChunkTiming,
    account_scored_requests,
    assert_content_disjoint,
    assert_identical_replay,
    closed_loop_corpus,
    cohort_assignments,
    controlled_poisson_corpus,
    evaluate_token_timing,
    external_shape_corpus,
    immediate_burst_corpus,
)


def sampling(seed: int = 7) -> FrozenSamplingParameters:
    return FrozenSamplingParameters.from_mapping(
        {
            "ignore_eos": True,
            "seed": seed,
            "temperature": 0.0,
            "top_p": 1.0,
        }
    )


def templates(
    count: int, *, token_offset: int = 0, cancelled_index: int | None = None
) -> tuple[RequestTemplate, ...]:
    return tuple(
        RequestTemplate(
            input_token_ids=(token_offset + index, token_offset + index + 100),
            requested_output_tokens=index + 1,
            sampling=sampling(index),
            cancellation_offset_us=(30 if index == cancelled_index else None),
        )
        for index in range(count)
    )


def burst(
    split: str,
    values: tuple[RequestTemplate, ...],
    *,
    namespace: str,
):
    return immediate_burst_corpus(
        values,
        namespace=namespace,
        split=split,
        cohort_count=4,
        cohort_popularity="uniform",
        cohort_seed=19,
    )


def test_poisson_corpus_is_deterministic_and_hashes_every_trace_field() -> None:
    values = templates(8)
    first = controlled_poisson_corpus(
        values,
        namespace="load-v1",
        split="pilot",
        rate_per_second=12.5,
        arrival_seed=23,
        cohort_count=4,
        cohort_popularity="zipf",
        cohort_seed=29,
    )
    second = controlled_poisson_corpus(
        values,
        namespace="load-v1",
        split="pilot",
        rate_per_second=12.5,
        arrival_seed=23,
        cohort_count=4,
        cohort_popularity="zipf",
        cohort_seed=29,
    )

    assert first == second
    assert first.synthetic
    assert first.label == "synthetic controlled Poisson"
    assert first.requests[0].arrival_us > 0
    assert all(
        left.arrival_us < right.arrival_us
        for left, right in zip(first.requests, first.requests[1:])
    )
    assert len(set(first.hashes.__dict__.values())) == 8
    assert all(len(value) == 64 for value in first.hashes.__dict__.values())

    changed = controlled_poisson_corpus(
        (replace(values[0], requested_output_tokens=99), *values[1:]),
        namespace="load-v1",
        split="pilot",
        rate_per_second=12.5,
        arrival_seed=23,
        cohort_count=4,
        cohort_popularity="zipf",
        cohort_seed=29,
    )
    assert first.requests[0].request_id != changed.requests[0].request_id
    assert (
        first.hashes.requested_lengths_sha256 != changed.hashes.requested_lengths_sha256
    )
    assert first.hashes.arrivals_sha256 == changed.hashes.arrivals_sha256


def test_request_id_validation_rejects_mutation_and_synthetic_burstgpt_label() -> None:
    corpus = burst("pilot", templates(2), namespace="immutable")
    request = corpus.requests[0]
    with pytest.raises(ValueError, match="immutable request content"):
        replace(request, cohort_id="cohort-rewritten").validate()
    with pytest.raises(ValueError, match="never be labelled BurstGPT"):
        replace(corpus, label="BurstGPT replay").validate()


@pytest.mark.parametrize("cohort_count", [1, 4, 16, 64])
@pytest.mark.parametrize("popularity", ["uniform", "zipf"])
def test_registered_cohort_assignments_are_stable_and_bounded(
    cohort_count: int, popularity: str
) -> None:
    first = cohort_assignments(
        256,
        cohort_count=cohort_count,
        popularity=popularity,
        seed=31,
    )
    assert first == cohort_assignments(
        256,
        cohort_count=cohort_count,
        popularity=popularity,
        seed=31,
    )
    assert len(first) == 256
    assert all(
        0 <= int(value.removeprefix("cohort-")) < cohort_count for value in first
    )


def test_unregistered_cohort_cardinality_is_rejected_before_corpus_allocation() -> None:
    with pytest.raises(ValueError, match="one of 1, 4, 16, or 64"):
        cohort_assignments(10, cohort_count=2, popularity="uniform", seed=1)


def test_external_burstgpt_shape_requires_source_lock_and_exact_token_lengths(
    monkeypatch,
) -> None:
    rows = (
        ExternalShapeRow(arrival_us=0, input_tokens=2, requested_output_tokens=8),
        ExternalShapeRow(arrival_us=1_500, input_tokens=3, requested_output_tokens=4),
    )
    digest = ExternalWorkloadShape.digest_rows(rows)
    with pytest.raises(ValueError, match="not registered"):
        ExternalWorkloadShape.from_rows(
            source_name="BurstGPT",
            source_revision="public-shape-revision-1",
            rows=rows,
            declared_rows_sha256=digest,
        )
    monkeypatch.setattr(
        load_module,
        "REGISTERED_EXTERNAL_SHAPES",
        {("BurstGPT", "public-shape-revision-1"): digest},
    )
    shape = ExternalWorkloadShape.from_rows(
        source_name="BurstGPT",
        source_revision="public-shape-revision-1",
        rows=rows,
        declared_rows_sha256=digest,
    )
    corpus = external_shape_corpus(
        shape,
        namespace="burstgpt-public-shape-v1",
        split="broad_replication",
        tokenized_inputs=((1, 2), (3, 4, 5)),
        sampling=(sampling(1), sampling(2)),
        cohort_count=4,
        cohort_popularity="zipf",
        cohort_seed=37,
    )
    assert corpus.label == "BurstGPT workload-shape replay"
    assert corpus.source_kind == "external_shape"
    assert not corpus.synthetic
    assert [request.arrival_us for request in corpus.requests] == [0, 1_500]
    assert [request.requested_output_tokens for request in corpus.requests] == [8, 4]

    with pytest.raises(ValueError, match="source lock"):
        ExternalWorkloadShape.from_rows(
            source_name="BurstGPT",
            source_revision="public-shape-revision-1",
            rows=rows,
            declared_rows_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="does not match external input length"):
        external_shape_corpus(
            shape,
            namespace="bad-lengths",
            split="broad_replication",
            tokenized_inputs=((1,), (3, 4, 5)),
            sampling=(sampling(1), sampling(2)),
            cohort_count=1,
            cohort_popularity="uniform",
            cohort_seed=0,
        )


def test_content_split_identity_rejects_tokenized_input_leakage() -> None:
    tuning = burst("tuning", templates(3), namespace="tuning")
    confirmation = burst(
        "confirmation", templates(3, token_offset=1_000), namespace="confirmation"
    )
    identities = assert_content_disjoint((tuning, confirmation))
    assert [identity.split for identity in identities] == ["tuning", "confirmation"]
    assert identities[0].input_set_sha256 != identities[1].input_set_sha256

    leaked = burst(
        "confirmation",
        (templates(3)[0], *templates(2, token_offset=2_000)),
        namespace="leaked-confirmation",
    )
    with pytest.raises(ValueError, match="overlaps tuning/confirmation"):
        assert_content_disjoint((tuning, leaked))


def production_plan() -> ProductionLoadPlan:
    warmup = immediate_burst_corpus(
        templates(2, token_offset=10_000),
        namespace="warmup",
        split="warmup",
        cohort_count=1,
        cohort_popularity="uniform",
        cohort_seed=1,
    )
    scored = immediate_burst_corpus(
        templates(5, token_offset=20_000, cancelled_index=2),
        namespace="score",
        split="confirmation",
        cohort_count=4,
        cohort_popularity="uniform",
        cohort_seed=2,
    )
    return ProductionLoadPlan(
        warmup=warmup,
        scored=scored,
        window=ProductionWindow(
            warmup_duration_us=10,
            arrival_duration_us=10,
            request_deadline_us=100,
            drain_duration_us=90,
        ),
    )


def test_load_plan_binds_identical_trace_replay_across_methods() -> None:
    plan = production_plan()
    static = ReplayBinding.create(plan, method="static")
    tts = ReplayBinding.create(plan, method="tts")
    target = ReplayBinding.create(plan, method="target_only")
    assert assert_identical_replay((static, tts, target)) == plan.paired_replay_sha256
    assert len({static.binding_sha256, tts.binding_sha256, target.binding_sha256}) == 3
    assert static.request_ids_sha256 == tts.request_ids_sha256
    with pytest.raises(ValueError, match="does not match its content"):
        assert_identical_replay((replace(static, binding_sha256="0" * 64), tts))

    other = ReplayBinding.create(
        replace(
            plan,
            scored=immediate_burst_corpus(
                templates(5, token_offset=30_000, cancelled_index=2),
                namespace="other-score",
                split="confirmation",
                cohort_count=4,
                cohort_popularity="uniform",
                cohort_seed=2,
            ),
        ),
        method="l0",
    )
    with pytest.raises(ValueError, match="identical trace"):
        assert_identical_replay((static, other))


def test_fixed_trace_accounting_partitions_every_offered_request() -> None:
    plan = production_plan()
    request_ids = [request.request_id for request in plan.scored.requests]
    outcomes = (
        RequestOutcome(request_ids[0], "completed", 0, 20, "ok"),
        RequestOutcome(request_ids[1], "timed_out", 0, 100, "deadline"),
        RequestOutcome(request_ids[2], "cancelled", 0, 30, "client_cancel"),
        RequestOutcome(request_ids[3], "rejected", None, 0, "queue_full"),
        RequestOutcome(request_ids[4], "unfinished", 0, None, "drain_expired"),
    )
    accounting = account_scored_requests(plan, outcomes)
    assert accounting.offered == 5
    assert accounting.admitted == 4
    assert accounting.rejected == 1
    assert accounting.completed == 1
    assert accounting.timed_out == 1
    assert accounting.cancelled == 1
    assert accounting.unfinished == 1
    assert accounting.score_started_us == 0
    assert accounting.score_ended_us == 100
    assert accounting.elapsed_us == 100


def test_accounting_rejects_missing_duplicate_and_uses_last_terminal() -> None:
    plan = production_plan()
    request_ids = [request.request_id for request in plan.scored.requests]
    completed = RequestOutcome(request_ids[0], "completed", 0, 20, "ok")
    with pytest.raises(ValueError, match="coverage"):
        account_scored_requests(plan, (completed,))
    with pytest.raises(ValueError, match="duplicate"):
        account_scored_requests(plan, (completed, completed))

    outcomes = tuple(
        RequestOutcome(
            request.request_id,
            "cancelled" if request.cancellation_offset_us is not None else "completed",
            0,
            request.cancellation_offset_us or 40,
            "scheduled_cancel" if request.cancellation_offset_us is not None else "ok",
        )
        for request in plan.scored.requests
    )
    accounting = account_scored_requests(plan, outcomes)
    assert accounting.score_started_us == 0
    assert accounting.score_ended_us == 40
    assert accounting.elapsed_us == 40


def test_closed_loop_accounting_requires_terminal_replenishment_prefixes() -> None:
    with pytest.raises(ValueError, match="every client lane"):
        closed_loop_corpus(
            templates(1),
            namespace="undersized-closed-loop",
            split="tuning",
            concurrency=2,
            cohort_count=1,
            cohort_popularity="uniform",
            cohort_seed=3,
        )

    corpus = closed_loop_corpus(
        templates(4),
        namespace="closed-loop",
        split="tuning",
        concurrency=2,
        cohort_count=1,
        cohort_popularity="uniform",
        cohort_seed=3,
    )
    plan = ProductionLoadPlan(
        warmup=None,
        scored=corpus,
        window=ProductionWindow(0, 100, 100, 100),
    )
    rows = corpus.requests
    outcomes = (
        RequestOutcome(rows[0].request_id, "completed", 0, 10, "ok", 0),
        RequestOutcome(rows[1].request_id, "completed", 0, 15, "ok", 0),
        RequestOutcome(rows[2].request_id, "completed", 10, 20, "ok", 10),
        RequestOutcome(rows[3].request_id, "completed", 15, 25, "ok", 15),
    )
    accounting = account_scored_requests(plan, outcomes)
    assert accounting.offered == 4
    assert accounting.score_started_us == 0
    assert accounting.score_ended_us == 25

    with pytest.raises(ValueError, match="client lane"):
        account_scored_requests(plan, (outcomes[0], outcomes[2]))

    early_replenishment = replace(outcomes[2], admitted_at_us=9, offered_at_us=9)
    with pytest.raises(ValueError, match="prior terminal"):
        account_scored_requests(
            plan,
            (outcomes[0], outcomes[1], early_replenishment, outcomes[3]),
        )
    delayed_replenishment = replace(outcomes[2], admitted_at_us=11, offered_at_us=11)
    with pytest.raises(ValueError, match="prior terminal"):
        account_scored_requests(
            plan,
            (outcomes[0], outcomes[1], delayed_replenishment, outcomes[3]),
        )


def test_coalesced_chunks_are_missing_timing_not_zero_itls() -> None:
    chunks = (
        TokenChunkTiming("request-a", 0, 1, 10),
        TokenChunkTiming("request-a", 1, 2, 20),
        TokenChunkTiming("request-a", 3, 1, 30),
        TokenChunkTiming("request-a", 4, 2, 40, (35, 40)),
    )
    coverage = evaluate_token_timing(
        request_id="request-a",
        request_started_us=0,
        expected_output_tokens=6,
        chunks=chunks,
    )
    assert coverage.ttft_us == 10
    assert coverage.expected_itl_intervals == 5
    assert coverage.supported_itl_intervals == 2
    assert coverage.coalesced_tokens == 2
    assert coverage.itl_coverage == pytest.approx(0.4)
    assert coverage.supported_itls_us == (5, 5)
    assert 0 not in coverage.supported_itls_us
    assert coverage.itl_percentile_us(99) is None
    assert coverage.diagnostic_supported_percentile_us(99) == 5.0


def test_server_per_token_timestamps_enable_claimable_itl_percentile() -> None:
    coverage = evaluate_token_timing(
        request_id="request-b",
        request_started_us=5,
        expected_output_tokens=3,
        chunks=(TokenChunkTiming("request-b", 0, 3, 20, (10, 12, 15)),),
    )
    assert coverage.full_itl_coverage
    assert coverage.itl_coverage == 1.0
    assert coverage.ttft_us == 5
    assert coverage.supported_itls_us == (2, 3)
    assert coverage.itl_percentile_us(99) == pytest.approx(2.99)


def test_ambiguous_first_chunk_excludes_ttft_and_index_gaps_fail_closed() -> None:
    coverage = evaluate_token_timing(
        request_id="request-c",
        request_started_us=0,
        expected_output_tokens=2,
        chunks=(TokenChunkTiming("request-c", 0, 2, 10),),
    )
    assert coverage.ttft_us is None
    assert coverage.itl_coverage == 0.0
    assert coverage.itl_percentile_us(99) is None

    with pytest.raises(ValueError, match="cover token indices exactly once"):
        evaluate_token_timing(
            request_id="request-c",
            request_started_us=0,
            expected_output_tokens=2,
            chunks=(TokenChunkTiming("request-c", 1, 1, 10),),
        )
    with pytest.raises(ValueError, match="globally nondecreasing"):
        evaluate_token_timing(
            request_id="request-c",
            request_started_us=0,
            expected_output_tokens=2,
            chunks=(
                TokenChunkTiming("request-c", 0, 1, 10),
                TokenChunkTiming("request-c", 1, 1, 20, (5,)),
            ),
        )


def test_sampling_and_request_boundaries_reject_nonfinite_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        FrozenSamplingParameters.from_mapping({"temperature": float("nan")})
    with pytest.raises(ValueError, match="tokenized input"):
        RequestTemplate((), 1, sampling()).validate()
    valid = ImmutableRequest.create(
        namespace="valid",
        split="pilot",
        ordinal=0,
        template=templates(1)[0],
        arrival_us=0,
        cohort_id="cohort-00",
    )
    assert len(valid.field_hashes.sampling_sha256) == 64
