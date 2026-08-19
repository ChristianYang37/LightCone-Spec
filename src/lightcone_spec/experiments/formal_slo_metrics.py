"""Exact SLO-goodput accounting for proof-derived formal serving rows.

The serving reducers receive native terminal results and first-party client
timestamps.  This module is the single arithmetic boundary that turns those
integer observations into production-SLO-qualified goodput.  In particular,
goodput uses the full scored wall-clock window; summing per-request latencies
would undercount concurrent service and is deliberately not an accepted input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from types import MappingProxyType
from typing import Literal, Self

from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.statistics import (
    TTFT_LIMIT_MS,
    WITHIN_REQUEST_P99_ITL_LIMIT_MS,
)

FORMAL_PROMPT_BUCKET_LIMITS_TOKENS = (
    ("short", 2_048),
    ("medium", 8_192),
    ("long", None),
)
FORMAL_TTFT_LIMIT_NS: Mapping[str, int] = MappingProxyType(
    {
        "short": 2_000_000_000,
        "medium": 5_000_000_000,
        "long": 10_000_000_000,
    }
)
FORMAL_WITHIN_REQUEST_P99_ITL_LIMIT_NS = 100_000_000
FORMAL_PRIMARY_GOODPUT_ROLES = ("Static", "TTS", "LightCone")

if tuple(TTFT_LIMIT_MS) != tuple(FORMAL_TTFT_LIMIT_NS) or any(
    Fraction(str(TTFT_LIMIT_MS[name])) * 1_000_000 != limit
    for name, limit in FORMAL_TTFT_LIMIT_NS.items()
):  # pragma: no cover - an import-time protocol drift guard
    raise RuntimeError("formal integer TTFT limits differ from registered SLO")
if Fraction(str(WITHIN_REQUEST_P99_ITL_LIMIT_MS)) * 1_000_000 != (
    FORMAL_WITHIN_REQUEST_P99_ITL_LIMIT_NS
):  # pragma: no cover - an import-time protocol drift guard
    raise RuntimeError("formal integer ITL limit differs from registered SLO")


FORMAL_SLO_GOODPUT_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_formal_slo_goodput_protocol",
        "input": "verifier_reopened_terminal_and_first_party_client_timestamps",
        "eligibility": (
            "exact_source_registered_eligible_scored_request_universe_no_caller_filter"
        ),
        "prompt_bucket_by_input_tokens": {
            "short": "1..2048",
            "medium": "2049..8192",
            "long": "8193_plus",
        },
        "ttft_limit_ns": dict(FORMAL_TTFT_LIMIT_NS),
        "within_request_p99_itl_limit_ns": (FORMAL_WITHIN_REQUEST_P99_ITL_LIMIT_NS),
        "outcome_flags": ("verifier_derived_completed_and_error_mutually_exclusive"),
        "nonqualifying_output": (
            "preserve_exact_partial_output_token_and_timestamp_trajectory_"
            "but_never_count_it"
        ),
        "qualification": "completed_nonerror_request_meets_ttft_and_itl",
        "timestamp_order": "strictly_increasing_within_request",
        "global_gates": {
            "qualified": "at_least_99_percent",
            "errors": "at_most_0.1_percent",
            "completed": "at_least_99.9_percent",
        },
        "numerator": "output_tokens_from_individually_qualified_requests",
        "denominator": (
            "max_scored_request_terminal_ns_minus_min_scored_request_started_ns"
        ),
        "pairing": (
            "exact_registered_source_request_pool_sha256_plus_per_offered_"
            "request_identity_where_content_addressed_request_id_binds_requested_"
            "length_sampling_arrival_cohort_and_cancellation"
        ),
        "observed_output": (
            "retained_in_request_evidence_and_exactly_timestamped_for_completed_"
            "requests_but_not_part_of_pairing_identity"
        ),
        "completed_output_exactness": (
            "separate_gate_requires_identical_observed_output_for_every_pair_"
            "of_completed_methods_on_the_same_source_request"
        ),
        "forbidden_denominator": "sum_of_per_request_latencies",
        "arithmetic": "integer_counts_and_fraction_until_reporting_boundary",
    }
)


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def formal_prompt_bucket(input_token_count: int) -> Literal["short", "medium", "long"]:
    """Return the preregistered prompt bucket from an exact token count."""

    if type(input_token_count) is not int or input_token_count < 1:
        raise ValueError("formal SLO input token count must be positive")
    if input_token_count <= 2_048:
        return "short"
    if input_token_count <= 8_192:
        return "medium"
    return "long"


def linear_p99_ns(values: Sequence[int]) -> Fraction | None:
    """Return the exact linear p99, or ``None`` when ITL is unobservable."""

    rows = tuple(values)
    if not rows:
        return None
    if any(type(value) is not int or value < 0 for value in rows):
        raise ValueError("formal SLO ITL samples must be non-negative integers")
    ordered = tuple(sorted(rows))
    position = Fraction((len(ordered) - 1) * 99, 100)
    lower = position.numerator // position.denominator
    upper = min(lower + 1, len(ordered) - 1)
    remainder = position - lower
    return (
        Fraction(ordered[lower]) * (1 - remainder)
        + Fraction(ordered[upper]) * remainder
    )


@dataclass(frozen=True)
class FormalSloRequestEvidence:
    """One scored request reconstructed from terminal and client timing proofs."""

    request_id: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    request_started_ns: int
    request_terminal_ns: int
    token_observed_ns: tuple[int, ...]
    eligible: bool
    completed: bool
    error: bool

    def __post_init__(self) -> None:
        if (
            type(self.request_id) is not str
            or not self.request_id
            or self.request_id.strip() != self.request_id
        ):
            raise ValueError("formal SLO request ID must be canonical text")
        for label, token_ids, allow_empty in (
            ("input", self.input_token_ids, False),
            ("output", self.output_token_ids, True),
        ):
            if (
                type(token_ids) is not tuple
                or (not allow_empty and not token_ids)
                or any(
                    type(token_id) is not int or token_id < 0 for token_id in token_ids
                )
            ):
                raise ValueError(f"formal SLO {label} token IDs are invalid")
        if (
            type(self.request_started_ns) is not int
            or type(self.request_terminal_ns) is not int
            or self.request_started_ns < 0
            or self.request_terminal_ns <= self.request_started_ns
        ):
            raise ValueError("formal SLO request interval must be positive")
        if type(self.token_observed_ns) is not tuple or any(
            type(value) is not int for value in self.token_observed_ns
        ):
            raise TypeError("formal SLO token timestamps must be an integer tuple")
        if any(
            right <= left
            for left, right in zip(
                self.token_observed_ns,
                self.token_observed_ns[1:],
                strict=False,
            )
        ):
            raise ValueError("formal SLO token timestamps must be strictly increasing")
        if self.token_observed_ns and (
            self.token_observed_ns[0] < self.request_started_ns
            or self.token_observed_ns[-1] > self.request_terminal_ns
        ):
            raise ValueError("formal SLO token timestamps leave the request interval")
        if any(
            type(value) is not bool
            for value in (self.eligible, self.completed, self.error)
        ):
            raise TypeError(
                "formal SLO eligibility/completion/error flags must be bool"
            )
        if self.completed and self.error:
            raise ValueError("formal SLO request cannot be both completed and errored")
        successful = self.completed and not self.error
        if successful:
            if not self.output_token_ids or len(self.token_observed_ns) != len(
                self.output_token_ids
            ):
                raise ValueError(
                    "successful formal SLO request lacks exact output timestamps"
                )
        elif len(self.token_observed_ns) != len(self.output_token_ids):
            raise ValueError(
                "unsuccessful formal SLO request lacks its exact partial output timestamps"
            )

    @property
    def prompt_bucket(self) -> Literal["short", "medium", "long"]:
        return formal_prompt_bucket(len(self.input_token_ids))

    @cached_property
    def source_request_sha256(self) -> str:
        return content_sha256(
            {
                "request_id": self.request_id,
                "input_token_ids": list(self.input_token_ids),
            }
        )

    @property
    def trajectory_sha256(self) -> str:
        """Backward-compatible field name for the source request identity.

        Observed output is response data and can legitimately differ when one
        paired method completes while another times out or is rejected.  The
        content-addressed request ID already binds every registered source
        axis, including requested length, sampling, arrival, cohort, and
        cancellation.
        """

        return self.source_request_sha256

    @property
    def ttft_ns(self) -> int | None:
        if not self.token_observed_ns:
            return None
        return self.token_observed_ns[0] - self.request_started_ns

    @property
    def within_request_p99_itl_ns(self) -> Fraction | None:
        return linear_p99_ns(
            tuple(
                right - left
                for left, right in zip(
                    self.token_observed_ns,
                    self.token_observed_ns[1:],
                    strict=False,
                )
            )
        )

    @property
    def qualifies(self) -> bool:
        ttft = self.ttft_ns
        p99_itl = self.within_request_p99_itl_ns
        return (
            self.eligible
            and self.completed
            and not self.error
            and ttft is not None
            and p99_itl is not None
            and ttft <= FORMAL_TTFT_LIMIT_NS[self.prompt_bucket]
            and p99_itl <= FORMAL_WITHIN_REQUEST_P99_ITL_LIMIT_NS
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "input_token_ids": list(self.input_token_ids),
            "output_token_ids": list(self.output_token_ids),
            "request_started_ns": self.request_started_ns,
            "request_terminal_ns": self.request_terminal_ns,
            "token_observed_ns": list(self.token_observed_ns),
            "eligible": self.eligible,
            "completed": self.completed,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = set(cls.__dataclass_fields__)
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal SLO request evidence fields differ")
        row = dict(value)
        timestamps = row.pop("token_observed_ns")
        input_token_ids = row.pop("input_token_ids")
        output_token_ids = row.pop("output_token_ids")
        if any(
            type(values) is not list
            for values in (timestamps, input_token_ids, output_token_ids)
        ):
            raise TypeError("formal SLO token IDs and timestamps must be arrays")
        return cls(  # type: ignore[arg-type]
            **row,
            input_token_ids=tuple(input_token_ids),
            output_token_ids=tuple(output_token_ids),
            token_observed_ns=tuple(timestamps),
        )


@dataclass(frozen=True)
class FormalSloGoodputObservation:
    """Exact SLO accounting and goodput inputs for one formal serving cell."""

    schema_version: Literal[2]
    protocol_sha256: str
    source_request_pool_sha256: str
    request_ids: tuple[str, ...]
    request_trajectory_sha256s: tuple[tuple[str, str], ...]
    eligible_request_ids: tuple[str, ...]
    qualified_request_ids: tuple[str, ...]
    eligible_requests: int
    qualified_requests: int
    error_requests: int
    completed_requests: int
    qualified_output_tokens: int
    scored_window_ns: int
    status: Literal["PASS", "FAIL"]
    request_evidence_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 2
            or self.protocol_sha256 != FORMAL_SLO_GOODPUT_PROTOCOL_SHA256
        ):
            raise ValueError("formal SLO-goodput protocol identity differs")
        _sha256("formal SLO source request pool", self.source_request_pool_sha256)
        for label, values in (
            ("request", self.request_ids),
            ("eligible request", self.eligible_request_ids),
            ("qualified request", self.qualified_request_ids),
        ):
            if (
                type(values) is not tuple
                or values != tuple(sorted(set(values)))
                or any(type(value) is not str or not value for value in values)
            ):
                raise ValueError(f"formal SLO {label} IDs are not canonical")
        if (
            type(self.request_trajectory_sha256s) is not tuple
            or self.request_trajectory_sha256s
            != tuple(sorted(set(self.request_trajectory_sha256s)))
            or tuple(row[0] for row in self.request_trajectory_sha256s)
            != self.request_ids
        ):
            raise ValueError("formal SLO request trajectories are not canonical")
        for request_id, digest in self.request_trajectory_sha256s:
            if type(request_id) is not str or not request_id:
                raise ValueError("formal SLO trajectory request ID is invalid")
            _sha256("formal SLO request trajectory", digest)
        if (
            not self.request_ids
            or self.eligible_request_ids != self.request_ids
            or not set(self.qualified_request_ids) <= set(self.request_ids)
        ):
            raise ValueError("formal SLO request subsets differ")
        for label, value in (
            ("eligible requests", self.eligible_requests),
            ("qualified requests", self.qualified_requests),
            ("error requests", self.error_requests),
            ("completed requests", self.completed_requests),
            ("qualified output tokens", self.qualified_output_tokens),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"formal SLO {label} must be non-negative")
        if (
            self.eligible_requests != len(self.eligible_request_ids)
            or self.qualified_requests != len(self.qualified_request_ids)
            or self.eligible_requests > len(self.request_ids)
            or self.qualified_requests > self.eligible_requests
            or self.error_requests > self.eligible_requests
            or self.completed_requests > self.eligible_requests
            or type(self.scored_window_ns) is not int
            or self.scored_window_ns < 1
        ):
            raise ValueError("formal SLO accounting counts are inconsistent")
        passed = (
            self.eligible_requests > 0
            and self.qualified_requests * 100 >= self.eligible_requests * 99
            and self.error_requests * 1_000 <= self.eligible_requests
            and self.completed_requests * 1_000 >= self.eligible_requests * 999
            and self.qualified_output_tokens > 0
        )
        if self.status != ("PASS" if passed else "FAIL"):
            raise ValueError("formal SLO status differs from exact counts")
        _sha256("formal SLO request evidence", self.request_evidence_sha256)

    @property
    def goodput_tokens_per_second(self) -> Fraction:
        return Fraction(
            self.qualified_output_tokens * 1_000_000_000,
            self.scored_window_ns,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "protocol_sha256": self.protocol_sha256,
            "source_request_pool_sha256": self.source_request_pool_sha256,
            "request_ids": list(self.request_ids),
            "request_trajectory_sha256s": [
                list(row) for row in self.request_trajectory_sha256s
            ],
            "eligible_request_ids": list(self.eligible_request_ids),
            "qualified_request_ids": list(self.qualified_request_ids),
            "eligible_requests": self.eligible_requests,
            "qualified_requests": self.qualified_requests,
            "error_requests": self.error_requests,
            "completed_requests": self.completed_requests,
            "qualified_output_tokens": self.qualified_output_tokens,
            "scored_window_ns": self.scored_window_ns,
            "status": self.status,
            "request_evidence_sha256": self.request_evidence_sha256,
        }
        if include_sha256:
            value["observation_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "observation_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal SLO-goodput observation fields differ")
        row = dict(value)
        declared = _sha256(
            "formal SLO-goodput observation", row.pop("observation_sha256")
        )
        for name in (
            "request_ids",
            "eligible_request_ids",
            "qualified_request_ids",
        ):
            raw = row[name]
            if type(raw) is not list:
                raise TypeError(f"formal SLO {name} must be an array")
            row[name] = tuple(raw)
        raw_trajectories = row["request_trajectory_sha256s"]
        if type(raw_trajectories) is not list or any(
            type(item) is not list or len(item) != 2 for item in raw_trajectories
        ):
            raise TypeError("formal SLO request trajectories must be pair arrays")
        row["request_trajectory_sha256s"] = tuple(
            tuple(item) for item in raw_trajectories
        )
        observation = cls(**row)  # type: ignore[arg-type]
        if observation.sha256 != declared:
            raise ValueError("formal SLO-goodput observation digest differs")
        return observation


def reduce_formal_slo_goodput(
    requests: Sequence[FormalSloRequestEvidence],
    *,
    source_request_pool_sha256: str | None = None,
) -> FormalSloGoodputObservation:
    """Derive exact request qualification and scored-window goodput."""

    rows = tuple(requests)
    if not rows or any(type(row) is not FormalSloRequestEvidence for row in rows):
        raise TypeError("formal SLO-goodput requires exact request evidence")
    request_ids = tuple(row.request_id for row in rows)
    if request_ids != tuple(sorted(set(request_ids))):
        raise ValueError("formal SLO requests must be sorted and unique")
    if any(not row.eligible for row in rows):
        raise ValueError("formal serving SLO requires every scored request eligible")
    eligible = rows
    qualified = tuple(row for row in eligible if row.qualifies)
    evidence_sha256 = content_sha256([row.to_dict() for row in rows])
    eligible_count = len(eligible)
    qualified_count = len(qualified)
    error_count = sum(int(row.error) for row in eligible)
    completed_count = sum(int(row.completed) for row in eligible)
    qualified_tokens = sum(len(row.output_token_ids) for row in qualified)
    scored_window_ns = max(row.request_terminal_ns for row in rows) - min(
        row.request_started_ns for row in rows
    )
    if scored_window_ns < 1:
        raise ValueError("formal SLO scored wall window must be positive")
    passed = (
        qualified_count * 100 >= eligible_count * 99
        and error_count * 1_000 <= eligible_count
        and completed_count * 1_000 >= eligible_count * 999
        and qualified_tokens > 0
    )
    trajectories = tuple((row.request_id, row.trajectory_sha256) for row in rows)
    pool_sha256 = (
        content_sha256(
            {
                "kind": "offered_request_pool_fallback",
                "request_trajectory_sha256s": [list(row) for row in trajectories],
            }
        )
        if source_request_pool_sha256 is None
        else _sha256("formal SLO source request pool", source_request_pool_sha256)
    )
    return FormalSloGoodputObservation(
        schema_version=2,
        protocol_sha256=FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
        source_request_pool_sha256=pool_sha256,
        request_ids=request_ids,
        request_trajectory_sha256s=trajectories,
        eligible_request_ids=tuple(row.request_id for row in eligible),
        qualified_request_ids=tuple(row.request_id for row in qualified),
        eligible_requests=eligible_count,
        qualified_requests=qualified_count,
        error_requests=error_count,
        completed_requests=completed_count,
        qualified_output_tokens=qualified_tokens,
        scored_window_ns=scored_window_ns,
        status="PASS" if passed else "FAIL",
        request_evidence_sha256=evidence_sha256,
    )


def require_paired_primary_goodputs(
    observations: Mapping[str, FormalSloGoodputObservation],
) -> tuple[tuple[str, Fraction], ...]:
    """Require one source-request-paired Static/TTS/LightCone family.

    SLO failure is scientific data, especially at overload.  Selection and
    deployment gates inspect ``status`` separately; pairing must not erase the
    quantitative negative result.
    """

    if type(observations) is not dict or set(observations) != set(
        FORMAL_PRIMARY_GOODPUT_ROLES
    ):
        raise ValueError("formal primary goodput roles are not exact")
    rows = tuple(observations[role] for role in FORMAL_PRIMARY_GOODPUT_ROLES)
    if any(type(row) is not FormalSloGoodputObservation for row in rows):
        raise TypeError("formal primary goodput observations are not exact")
    if len({row.source_request_pool_sha256 for row in rows}) != 1:
        raise ValueError(
            "formal primary goodput source request pools are not exactly paired"
        )
    return tuple(
        (role, observations[role].goodput_tokens_per_second)
        for role in FORMAL_PRIMARY_GOODPUT_ROLES
    )


def require_paired_completed_output_exactness(
    requests_by_role: Mapping[str, Sequence[FormalSloRequestEvidence]],
    *,
    source_request_pool_sha256s: Mapping[str, str] | None = None,
) -> None:
    """Validate target-equivalent outputs without corrupting pairing identity.

    A completed response and a timeout are a valid paired performance
    observation, so response tokens cannot define the request-pool identity.
    Whenever two methods *both* complete the same source request, however,
    speculative exactness still requires byte-identical token trajectories.
    This independent gate preserves both rules.
    """

    if type(requests_by_role) is not dict or len(requests_by_role) < 2:
        raise ValueError("completed-output exactness requires at least two roles")
    indexed: dict[str, dict[str, FormalSloRequestEvidence]] = {}
    allow_prefix_difference = source_request_pool_sha256s is not None
    if source_request_pool_sha256s is not None:
        if set(source_request_pool_sha256s) != set(requests_by_role):
            raise ValueError("completed-output source pool roles differ")
        pools = {
            _sha256("completed-output source request pool", digest)
            for digest in source_request_pool_sha256s.values()
        }
        if len(pools) != 1:
            raise ValueError("completed-output source request pools are unpaired")
    source_identity: tuple[tuple[str, str], ...] | None = None
    for role, requests in requests_by_role.items():
        if type(role) is not str or not role:
            raise ValueError("completed-output exactness role is invalid")
        rows = tuple(requests)
        if not rows or any(type(row) is not FormalSloRequestEvidence for row in rows):
            raise TypeError("completed-output exactness evidence is not exact")
        by_id = {row.request_id: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError("completed-output exactness repeats a request")
        identity = tuple(
            sorted((row.request_id, row.source_request_sha256) for row in rows)
        )
        if source_identity is None:
            source_identity = identity
        elif not allow_prefix_difference and identity != source_identity:
            raise ValueError("completed-output exactness request pools are unpaired")
        indexed[role] = by_id
    assert source_identity is not None
    request_ids = (
        sorted({request_id for by_id in indexed.values() for request_id in by_id})
        if allow_prefix_difference
        else [request_id for request_id, _digest in source_identity]
    )
    for request_id in request_ids:
        identities = {
            row.source_request_sha256
            for by_id in indexed.values()
            if (row := by_id.get(request_id)) is not None
        }
        if len(identities) != 1:
            raise ValueError("completed-output source request identities differ")
        completed_outputs = {
            row.output_token_ids
            for by_id in indexed.values()
            if (row := by_id.get(request_id)) is not None
            and row.completed
            and not row.error
        }
        if len(completed_outputs) > 1:
            raise ValueError(
                "paired completed methods produced different output token trajectories"
            )


__all__ = (
    "FORMAL_PRIMARY_GOODPUT_ROLES",
    "FORMAL_PROMPT_BUCKET_LIMITS_TOKENS",
    "FORMAL_SLO_GOODPUT_PROTOCOL_SHA256",
    "FORMAL_TTFT_LIMIT_NS",
    "FORMAL_WITHIN_REQUEST_P99_ITL_LIMIT_NS",
    "FormalSloGoodputObservation",
    "FormalSloRequestEvidence",
    "formal_prompt_bucket",
    "linear_p99_ns",
    "reduce_formal_slo_goodput",
    "require_paired_completed_output_exactness",
    "require_paired_primary_goodputs",
)
