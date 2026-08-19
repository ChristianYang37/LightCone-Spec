from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from lightcone_spec.experiments.formal_slo_metrics import (
    FORMAL_PRIMARY_GOODPUT_ROLES,
    FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
    FormalSloGoodputObservation,
    FormalSloRequestEvidence,
    formal_prompt_bucket,
    linear_p99_ns,
    reduce_formal_slo_goodput,
    require_paired_completed_output_exactness,
    require_paired_primary_goodputs,
)


def _successful_request(
    request_id: str,
    *,
    input_token_count: int = 1,
    output_token_count: int = 2,
    request_started_ns: int = 0,
    request_terminal_ns: int = 1_000_000_000,
    first_token_ns: int = 100_000_000,
    token_step_ns: int = 50_000_000,
) -> FormalSloRequestEvidence:
    return FormalSloRequestEvidence(
        request_id=request_id,
        input_token_ids=tuple(range(input_token_count)),
        output_token_ids=tuple(range(10_000, 10_000 + output_token_count)),
        request_started_ns=request_started_ns,
        request_terminal_ns=request_terminal_ns,
        token_observed_ns=tuple(
            first_token_ns + index * token_step_ns
            for index in range(output_token_count)
        ),
        eligible=True,
        completed=True,
        error=False,
    )


def _failed_request(request_id: str) -> FormalSloRequestEvidence:
    return FormalSloRequestEvidence(
        request_id=request_id,
        input_token_ids=(1,),
        output_token_ids=(),
        request_started_ns=0,
        request_terminal_ns=1_000_000_000,
        token_observed_ns=(),
        eligible=True,
        completed=False,
        error=True,
    )


def test_prompt_buckets_and_linear_p99_are_exact() -> None:
    assert FORMAL_SLO_GOODPUT_PROTOCOL_SHA256 == (
        "0813afda9f08eac3eac1e043bbbe72b5a105495a7dbd77479ee9f6b80619c978"
    )
    assert formal_prompt_bucket(1) == "short"
    assert formal_prompt_bucket(2_048) == "short"
    assert formal_prompt_bucket(2_049) == "medium"
    assert formal_prompt_bucket(8_192) == "medium"
    assert formal_prompt_bucket(8_193) == "long"
    with pytest.raises(ValueError, match="positive"):
        formal_prompt_bucket(0)
    with pytest.raises(ValueError, match="positive"):
        formal_prompt_bucket(True)

    assert linear_p99_ns(()) is None
    assert linear_p99_ns((0, 100)) == Fraction(99)
    assert linear_p99_ns((10, 20, 30)) == Fraction(149, 5)
    with pytest.raises(ValueError, match="non-negative"):
        linear_p99_ns((1, -1))


def test_concurrent_goodput_uses_scored_wall_window_not_latency_sum() -> None:
    timestamps = tuple(1_000_000_000 + index * 50_000_000 for index in range(100))
    rows = tuple(
        FormalSloRequestEvidence(
            request_id=request_id,
            input_token_ids=(1,),
            output_token_ids=tuple(range(100)),
            request_started_ns=0,
            request_terminal_ns=10_000_000_000,
            token_observed_ns=timestamps,
            eligible=True,
            completed=True,
            error=False,
        )
        for request_id in ("request-a", "request-b")
    )

    observation = reduce_formal_slo_goodput(rows)

    assert observation.status == "PASS"
    assert observation.qualified_output_tokens == 200
    assert observation.scored_window_ns == 10_000_000_000
    assert observation.goodput_tokens_per_second == Fraction(20)
    summed_latency_goodput = Fraction(200 * 1_000_000_000, 20_000_000_000)
    assert observation.goodput_tokens_per_second != summed_latency_goodput


def test_slo_qualifies_requests_individually_and_keeps_full_window() -> None:
    rows = [_successful_request(f"request-{index:03d}") for index in range(99)]
    rows.append(
        _successful_request(
            "request-099",
            request_terminal_ns=4_000_000_000,
            first_token_ns=3_000_000_000,
        )
    )

    observation = reduce_formal_slo_goodput(rows)

    assert observation.status == "PASS"
    assert observation.eligible_requests == 100
    assert observation.qualified_requests == 99
    assert observation.qualified_output_tokens == 198
    assert observation.scored_window_ns == 4_000_000_000
    assert observation.goodput_tokens_per_second == Fraction(99, 2)
    assert "request-099" not in observation.qualified_request_ids

    unobservable_itl = reduce_formal_slo_goodput(
        (_successful_request("one-token", output_token_count=1),)
    )
    assert unobservable_itl.status == "FAIL"
    assert unobservable_itl.qualified_requests == 0

    exact_boundary = reduce_formal_slo_goodput(
        (
            _successful_request(
                "boundary",
                request_terminal_ns=2_200_000_000,
                first_token_ns=2_000_000_000,
                token_step_ns=100_000_000,
            ),
        )
    )
    assert exact_boundary.status == "PASS"
    over_boundary = reduce_formal_slo_goodput(
        (
            _successful_request(
                "boundary",
                request_terminal_ns=2_200_000_001,
                first_token_ns=2_000_000_001,
                token_step_ns=100_000_000,
            ),
        )
    )
    assert over_boundary.status == "FAIL"


def test_error_and_completion_thresholds_use_exact_integer_cross_products() -> None:
    one_failure = [_successful_request(f"request-{index:04d}") for index in range(999)]
    one_failure.append(_failed_request("request-0999"))
    passing = reduce_formal_slo_goodput(one_failure)
    assert passing.status == "PASS"
    assert passing.error_requests == 1
    assert passing.completed_requests == 999

    two_failures = [_successful_request(f"request-{index:04d}") for index in range(998)]
    two_failures.extend(
        (_failed_request("request-0998"), _failed_request("request-0999"))
    )
    failing = reduce_formal_slo_goodput(two_failures)
    assert failing.status == "FAIL"
    assert failing.error_requests == 2
    assert failing.completed_requests == 998

    with pytest.raises(ValueError, match="both completed and errored"):
        replace(_failed_request("contradictory-terminal"), completed=True)


def test_request_and_observation_codecs_reject_tamper() -> None:
    request = _successful_request("request-a")
    assert FormalSloRequestEvidence.from_dict(request.to_dict()) == request
    request_payload = request.to_dict()
    request_payload["token_observed_ns"] = [200_000_000, 100_000_000]
    with pytest.raises(ValueError, match="strictly increasing"):
        FormalSloRequestEvidence.from_dict(request_payload)

    observation = reduce_formal_slo_goodput((request,))
    assert observation.protocol_sha256 == FORMAL_SLO_GOODPUT_PROTOCOL_SHA256
    assert FormalSloGoodputObservation.from_dict(observation.to_dict()) == observation
    payload = observation.to_dict()
    payload["qualified_output_tokens"] = 3
    with pytest.raises(ValueError, match="digest differs"):
        FormalSloGoodputObservation.from_dict(payload)

    with pytest.raises(ValueError, match="every scored request eligible"):
        reduce_formal_slo_goodput((replace(request, eligible=False),))

    partial_error = FormalSloRequestEvidence(
        request_id="partial-error",
        input_token_ids=(1,),
        output_token_ids=(10, 11),
        request_started_ns=0,
        request_terminal_ns=1_000_000_000,
        token_observed_ns=(100_000_000, 200_000_000),
        eligible=True,
        completed=False,
        error=True,
    )
    assert not partial_error.qualifies
    assert FormalSloRequestEvidence.from_dict(partial_error.to_dict()) == partial_error
    with pytest.raises(ValueError, match="exact partial output timestamps"):
        replace(partial_error, token_observed_ns=(100_000_000,))


def test_primary_goodput_pairing_uses_source_requests_not_observed_outputs() -> None:
    base_rows = (
        _successful_request("request-a"),
        _successful_request("request-b"),
    )
    base = reduce_formal_slo_goodput(base_rows)
    observations = {role: base for role in FORMAL_PRIMARY_GOODPUT_ROLES}
    assert require_paired_primary_goodputs(observations) == tuple(
        (role, Fraction(4)) for role in FORMAL_PRIMARY_GOODPUT_ROLES
    )

    foreign = reduce_formal_slo_goodput(
        (_successful_request("request-a"), _successful_request("request-c"))
    )
    with pytest.raises(ValueError, match="not exactly paired"):
        require_paired_primary_goodputs({**observations, "LightCone": foreign})

    foreign_tokens = reduce_formal_slo_goodput(
        (
            _successful_request("request-a"),
            replace(
                _successful_request("request-b"),
                output_token_ids=(90_001, 90_002),
            ),
        )
    )
    assert require_paired_primary_goodputs(
        {**observations, "LightCone": foreign_tokens}
    )[-1] == ("LightCone", Fraction(4))

    failed = replace(
        base, qualified_request_ids=(), qualified_requests=0, status="FAIL"
    )
    assert require_paired_primary_goodputs({**observations, "TTS": failed})[1] == (
        "TTS",
        Fraction(4),
    )


def test_registered_closed_loop_pool_pairs_different_realized_prefixes() -> None:
    source_pool_sha256 = "a" * 64
    short_rows = (_successful_request("request-a"),)
    long_rows = (
        _successful_request("request-a"),
        _successful_request("request-b"),
    )
    observations = {
        "Static": reduce_formal_slo_goodput(
            short_rows,
            source_request_pool_sha256=source_pool_sha256,
        ),
        "TTS": reduce_formal_slo_goodput(
            long_rows,
            source_request_pool_sha256=source_pool_sha256,
        ),
        "LightCone": reduce_formal_slo_goodput(
            long_rows,
            source_request_pool_sha256=source_pool_sha256,
        ),
    }

    paired = require_paired_primary_goodputs(observations)
    assert tuple(role for role, _goodput in paired) == FORMAL_PRIMARY_GOODPUT_ROLES
    require_paired_completed_output_exactness(
        {
            "Static": short_rows,
            "TTS": long_rows,
            "LightCone": long_rows,
        },
        source_request_pool_sha256s={
            role: observation.source_request_pool_sha256
            for role, observation in observations.items()
        },
    )

    foreign = reduce_formal_slo_goodput(
        long_rows,
        source_request_pool_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="source request pools are not exactly paired"):
        require_paired_primary_goodputs({**observations, "LightCone": foreign})


def test_completed_output_exactness_is_separate_from_performance_pairing() -> None:
    completed = _successful_request("request-a")
    timed_out = FormalSloRequestEvidence(
        request_id="request-a",
        input_token_ids=completed.input_token_ids,
        output_token_ids=(),
        request_started_ns=completed.request_started_ns,
        request_terminal_ns=completed.request_terminal_ns,
        token_observed_ns=(),
        eligible=True,
        completed=False,
        error=False,
    )
    require_paired_completed_output_exactness(
        {"Static": (completed,), "LightCone": (timed_out,)}
    )

    different_completion = replace(completed, output_token_ids=(90_001, 90_002))
    with pytest.raises(ValueError, match="different output token trajectories"):
        require_paired_completed_output_exactness(
            {"Static": (completed,), "LightCone": (different_completion,)}
        )
