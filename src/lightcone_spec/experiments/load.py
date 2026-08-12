"""Deterministic request corpora and fail-closed production-load accounting.

This module is deliberately CPU-only.  It prepares immutable request identities
and replay receipts before a serving process is launched; it does not implement
a second scheduler or issue HTTP requests.  Arrival offsets are relative to the
start of their registered warm-up or scored window.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from itertools import pairwise
from types import MappingProxyType
from typing import Literal

JsonScalar = str | int | float | bool | None
SplitName = Literal["warmup", "tuning", "pilot", "confirmation", "broad_replication"]
SourceKind = Literal["poisson", "closed_loop", "immediate_burst", "external_shape"]
OutcomeStatus = Literal["rejected", "completed", "timed_out", "cancelled", "unfinished"]

REGISTERED_COHORT_COUNTS = frozenset({1, 4, 16, 64})
# External replay names are claims, not free-form labels. A source enters this
# immutable allowlist only with a reviewed public revision and canonical row
# digest. No BurstGPT asset is pinned in this source release, so its cells stay
# BLOCKED instead of accepting caller-authored rows under that name.
REGISTERED_EXTERNAL_SHAPES: Mapping[tuple[str, str], str] = MappingProxyType({})


def _sha256(value: object) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(body).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _require_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class FrozenSamplingParameters:
    """Canonical scalar sampling parameters with a stable content identity."""

    items: tuple[tuple[str, JsonScalar], ...]

    @classmethod
    def from_mapping(
        cls, parameters: Mapping[str, JsonScalar]
    ) -> FrozenSamplingParameters:
        items = tuple(sorted(parameters.items()))
        value = cls(items=items)
        value.validate()
        return value

    def validate(self) -> None:
        if not self.items:
            raise ValueError("sampling parameters must not be empty")
        keys = tuple(key for key, _ in self.items)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("sampling parameter keys must be unique and sorted")
        for key, value in self.items:
            if not isinstance(key, str) or not key:
                raise ValueError("sampling parameter keys must be non-empty strings")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise TypeError("sampling parameter values must be JSON scalars")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("sampling parameter floats must be finite")

    @property
    def sha256(self) -> str:
        self.validate()
        return _sha256(dict(self.items))


@dataclass(frozen=True)
class RequestTemplate:
    """Method-independent request content used by a trace generator."""

    input_token_ids: tuple[int, ...]
    requested_output_tokens: int
    sampling: FrozenSamplingParameters
    cancellation_offset_us: int | None = None

    def validate(self) -> None:
        if not self.input_token_ids:
            raise ValueError("tokenized input must not be empty")
        for token_id in self.input_token_ids:
            _require_nonnegative_integer(token_id, "input token ID")
        if (
            isinstance(self.requested_output_tokens, bool)
            or not isinstance(self.requested_output_tokens, int)
            or self.requested_output_tokens < 1
        ):
            raise ValueError("requested output length must be a positive integer")
        self.sampling.validate()
        if self.cancellation_offset_us is not None:
            _require_nonnegative_integer(
                self.cancellation_offset_us, "cancellation offset"
            )


@dataclass(frozen=True)
class RequestFieldHashes:
    """Independent identities for every registered request trace component."""

    tokenized_input_sha256: str
    requested_length_sha256: str
    arrival_sha256: str
    cohort_sha256: str
    cancellation_sha256: str
    sampling_sha256: str


@dataclass(frozen=True)
class ImmutableRequest:
    """A request whose ID changes if any replay-relevant field changes."""

    namespace: str
    split: SplitName
    ordinal: int
    request_id: str
    input_token_ids: tuple[int, ...]
    requested_output_tokens: int
    arrival_us: int
    cohort_id: str
    cancellation_offset_us: int | None
    sampling: FrozenSamplingParameters

    @classmethod
    def create(
        cls,
        *,
        namespace: str,
        split: SplitName,
        ordinal: int,
        template: RequestTemplate,
        arrival_us: int,
        cohort_id: str,
    ) -> ImmutableRequest:
        template.validate()
        payload = cls._identity_payload(
            namespace=namespace,
            split=split,
            ordinal=ordinal,
            input_token_ids=template.input_token_ids,
            requested_output_tokens=template.requested_output_tokens,
            arrival_us=arrival_us,
            cohort_id=cohort_id,
            cancellation_offset_us=template.cancellation_offset_us,
            sampling=template.sampling,
        )
        value = cls(
            namespace=namespace,
            split=split,
            ordinal=ordinal,
            request_id=f"req-{_sha256(payload)}",
            input_token_ids=template.input_token_ids,
            requested_output_tokens=template.requested_output_tokens,
            arrival_us=arrival_us,
            cohort_id=cohort_id,
            cancellation_offset_us=template.cancellation_offset_us,
            sampling=template.sampling,
        )
        value.validate()
        return value

    @staticmethod
    def _identity_payload(
        *,
        namespace: str,
        split: SplitName,
        ordinal: int,
        input_token_ids: tuple[int, ...],
        requested_output_tokens: int,
        arrival_us: int,
        cohort_id: str,
        cancellation_offset_us: int | None,
        sampling: FrozenSamplingParameters,
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "namespace": namespace,
            "split": split,
            "ordinal": ordinal,
            "input_token_ids": input_token_ids,
            "requested_output_tokens": requested_output_tokens,
            "arrival_us": arrival_us,
            "cohort_id": cohort_id,
            "cancellation": {
                "requested": cancellation_offset_us is not None,
                "offset_us": cancellation_offset_us,
            },
            "sampling": dict(sampling.items),
        }

    def validate(self) -> None:
        if not self.namespace:
            raise ValueError("request namespace must not be empty")
        if self.split not in {
            "warmup",
            "tuning",
            "pilot",
            "confirmation",
            "broad_replication",
        }:
            raise ValueError("request split is not registered")
        _require_nonnegative_integer(self.ordinal, "request ordinal")
        _require_nonnegative_integer(self.arrival_us, "arrival offset")
        if not self.cohort_id:
            raise ValueError("cohort identity must not be empty")
        template = RequestTemplate(
            input_token_ids=self.input_token_ids,
            requested_output_tokens=self.requested_output_tokens,
            sampling=self.sampling,
            cancellation_offset_us=self.cancellation_offset_us,
        )
        template.validate()
        expected = "req-" + _sha256(
            self._identity_payload(
                namespace=self.namespace,
                split=self.split,
                ordinal=self.ordinal,
                input_token_ids=self.input_token_ids,
                requested_output_tokens=self.requested_output_tokens,
                arrival_us=self.arrival_us,
                cohort_id=self.cohort_id,
                cancellation_offset_us=self.cancellation_offset_us,
                sampling=self.sampling,
            )
        )
        if self.request_id != expected:
            raise ValueError("request ID does not match immutable request content")

    @cached_property
    def field_hashes(self) -> RequestFieldHashes:
        self.validate()
        return RequestFieldHashes(
            tokenized_input_sha256=_sha256(self.input_token_ids),
            requested_length_sha256=_sha256(self.requested_output_tokens),
            arrival_sha256=_sha256(self.arrival_us),
            cohort_sha256=_sha256(self.cohort_id),
            cancellation_sha256=_sha256(
                {
                    "requested": self.cancellation_offset_us is not None,
                    "offset_us": self.cancellation_offset_us,
                }
            ),
            sampling_sha256=self.sampling.sha256,
        )


@dataclass(frozen=True)
class CorpusHashes:
    request_ids_sha256: str
    tokenized_inputs_sha256: str
    requested_lengths_sha256: str
    arrivals_sha256: str
    cohorts_sha256: str
    cancellations_sha256: str
    sampling_parameters_sha256: str
    corpus_sha256: str


@dataclass(frozen=True)
class RequestCorpus:
    """One ordered, immutable scored or warm-up request corpus."""

    label: str
    source_kind: SourceKind
    source_identity_sha256: str
    source_parameters: tuple[tuple[str, JsonScalar], ...]
    synthetic: bool
    split: SplitName
    requests: tuple[ImmutableRequest, ...]

    def validate(self) -> None:
        if not self.label:
            raise ValueError("corpus label must not be empty")
        if not _is_sha256(self.source_identity_sha256):
            raise ValueError("source identity must be a lowercase SHA-256")
        parameter_keys = tuple(name for name, _ in self.source_parameters)
        if (
            not self.source_parameters
            or parameter_keys != tuple(sorted(parameter_keys))
            or len(parameter_keys) != len(set(parameter_keys))
            or any(not name for name in parameter_keys)
            or any(
                not isinstance(value, (str, int, float, bool, type(None)))
                or (isinstance(value, float) and not math.isfinite(value))
                for _, value in self.source_parameters
            )
        ):
            raise ValueError("source parameters must be canonical JSON scalars")
        if self.source_identity_sha256 != _sha256(dict(self.source_parameters)):
            raise ValueError("source identity does not match its exact parameters")
        if not self.requests:
            raise ValueError("request corpus must not be empty")
        if self.synthetic and self.source_kind == "external_shape":
            raise ValueError("an external workload shape cannot be marked synthetic")
        if not self.synthetic and self.source_kind != "external_shape":
            raise ValueError("locally generated arrivals must be marked synthetic")
        if self.synthetic and "burstgpt" in self.label.casefold():
            raise ValueError("a synthetic trace must never be labelled BurstGPT")
        if self.source_kind == "closed_loop":
            parameters = dict(self.source_parameters)
            concurrency = parameters.get("concurrency")
            if (
                not isinstance(concurrency, int)
                or isinstance(concurrency, bool)
                or concurrency < 1
                or len(self.requests) < concurrency
                or parameters.get("request_count") != len(self.requests)
                or parameters.get("generator") != "closed_loop_zero_think_v1"
                or any(request.arrival_us != 0 for request in self.requests)
            ):
                raise ValueError(
                    "closed-loop corpus must bind one nonempty maximum pool per client"
                )
        identifiers: set[str] = set()
        last_arrival = -1
        for request in self.requests:
            request.validate()
            if request.split != self.split:
                raise ValueError("all requests must use the corpus split")
            if request.request_id in identifiers:
                raise ValueError("request IDs must be unique within a corpus")
            if request.arrival_us < last_arrival:
                raise ValueError("requests must be ordered by arrival offset")
            identifiers.add(request.request_id)
            last_arrival = request.arrival_us

    @cached_property
    def hashes(self) -> CorpusHashes:
        self.validate()
        fields = tuple(request.field_hashes for request in self.requests)
        component = {
            "request_ids_sha256": _sha256(
                tuple(request.request_id for request in self.requests)
            ),
            "tokenized_inputs_sha256": _sha256(
                tuple(field.tokenized_input_sha256 for field in fields)
            ),
            "requested_lengths_sha256": _sha256(
                tuple(field.requested_length_sha256 for field in fields)
            ),
            "arrivals_sha256": _sha256(tuple(field.arrival_sha256 for field in fields)),
            "cohorts_sha256": _sha256(tuple(field.cohort_sha256 for field in fields)),
            "cancellations_sha256": _sha256(
                tuple(field.cancellation_sha256 for field in fields)
            ),
            "sampling_parameters_sha256": _sha256(
                tuple(field.sampling_sha256 for field in fields)
            ),
        }
        corpus_sha256 = _sha256(
            {
                "schema_version": 1,
                "label": self.label,
                "source_kind": self.source_kind,
                "source_identity_sha256": self.source_identity_sha256,
                "source_parameters": self.source_parameters,
                "synthetic": self.synthetic,
                "split": self.split,
                **component,
            }
        )
        return CorpusHashes(corpus_sha256=corpus_sha256, **component)


def cohort_assignments(
    request_count: int,
    *,
    cohort_count: int,
    popularity: Literal["uniform", "zipf"],
    seed: int,
    zipf_exponent: float = 1.0,
) -> tuple[str, ...]:
    """Return deterministic IID uniform or Zipf cohort labels."""
    if (
        isinstance(request_count, bool)
        or not isinstance(request_count, int)
        or request_count < 1
    ):
        raise ValueError("request count must be positive")
    if cohort_count not in REGISTERED_COHORT_COUNTS:
        raise ValueError("cohort count must be one of 1, 4, 16, or 64")
    _require_nonnegative_integer(seed, "cohort seed")
    if popularity not in {"uniform", "zipf"}:
        raise ValueError("cohort popularity must be uniform or zipf")
    if not math.isfinite(zipf_exponent) or zipf_exponent <= 0:
        raise ValueError("Zipf exponent must be finite and positive")
    if popularity == "uniform":
        weights = (1.0,) * cohort_count
    else:
        weights = tuple(
            1.0 / ((rank + 1) ** zipf_exponent) for rank in range(cohort_count)
        )
    total = sum(weights)
    cumulative: list[float] = []
    running = 0.0
    for weight in weights:
        running += weight / total
        cumulative.append(running)
    assignments: list[str] = []
    for ordinal in range(request_count):
        digest = hashlib.sha256(f"cohort:{seed}:{ordinal}".encode()).digest()
        draw = (int.from_bytes(digest, "big") + 0.5) / (1 << 256)
        selected = next(
            index for index, upper_bound in enumerate(cumulative) if draw <= upper_bound
        )
        assignments.append(f"cohort-{selected:02d}")
    return tuple(assignments)


def _poisson_arrivals(
    request_count: int, *, rate_per_second: float, seed: int
) -> tuple[int, ...]:
    if not math.isfinite(rate_per_second) or rate_per_second <= 0:
        raise ValueError("Poisson rate must be finite and positive")
    _require_nonnegative_integer(seed, "arrival seed")
    arrivals: list[int] = []
    elapsed_us = 0
    for ordinal in range(request_count):
        digest = hashlib.sha256(f"arrival:{seed}:{ordinal}".encode()).digest()
        uniform = (int.from_bytes(digest, "big") + 0.5) / (1 << 256)
        gap_us = max(1, round(-math.log1p(-uniform) * 1_000_000 / rate_per_second))
        elapsed_us += gap_us
        arrivals.append(elapsed_us)
    return tuple(arrivals)


def _build_corpus(
    *,
    namespace: str,
    split: SplitName,
    templates: Sequence[RequestTemplate],
    arrivals_us: Sequence[int],
    cohorts: Sequence[str],
    label: str,
    source_kind: SourceKind,
    source_parameters: Mapping[str, JsonScalar],
    synthetic: bool,
) -> RequestCorpus:
    if not namespace:
        raise ValueError("corpus namespace must not be empty")
    if (
        not templates
        or len(templates) != len(arrivals_us)
        or len(templates) != len(cohorts)
    ):
        raise ValueError(
            "templates, arrivals, and cohorts must have equal nonzero length"
        )
    requests = tuple(
        ImmutableRequest.create(
            namespace=namespace,
            split=split,
            ordinal=ordinal,
            template=template,
            arrival_us=arrival_us,
            cohort_id=cohort,
        )
        for ordinal, (template, arrival_us, cohort) in enumerate(
            zip(templates, arrivals_us, cohorts, strict=True)
        )
    )
    canonical_source = tuple(sorted(source_parameters.items()))
    corpus = RequestCorpus(
        label=label,
        source_kind=source_kind,
        source_identity_sha256=_sha256(dict(canonical_source)),
        source_parameters=canonical_source,
        synthetic=synthetic,
        split=split,
        requests=requests,
    )
    corpus.validate()
    return corpus


def controlled_poisson_corpus(
    templates: Sequence[RequestTemplate],
    *,
    namespace: str,
    split: SplitName,
    rate_per_second: float,
    arrival_seed: int,
    cohort_count: int,
    cohort_popularity: Literal["uniform", "zipf"],
    cohort_seed: int,
    zipf_exponent: float = 1.0,
    registered_load_factor: float | None = None,
) -> RequestCorpus:
    """Generate a labelled-synthetic deterministic Poisson corpus."""
    arrivals = _poisson_arrivals(
        len(templates), rate_per_second=rate_per_second, seed=arrival_seed
    )
    cohorts = cohort_assignments(
        len(templates),
        cohort_count=cohort_count,
        popularity=cohort_popularity,
        seed=cohort_seed,
        zipf_exponent=zipf_exponent,
    )
    if registered_load_factor is not None and (
        not math.isfinite(registered_load_factor) or registered_load_factor <= 0
    ):
        raise ValueError("registered load factor must be finite and positive")
    source_parameters: dict[str, JsonScalar] = {
        "arrival_seed": arrival_seed,
        "cohort_count": cohort_count,
        "cohort_popularity": cohort_popularity,
        "cohort_seed": cohort_seed,
        "generator": "controlled_poisson_v1",
        "rate_per_second": rate_per_second,
        "registered_load_factor": registered_load_factor,
        "zipf_exponent": zipf_exponent,
    }
    return _build_corpus(
        namespace=namespace,
        split=split,
        templates=templates,
        arrivals_us=arrivals,
        cohorts=cohorts,
        label="synthetic controlled Poisson",
        source_kind="poisson",
        source_parameters=source_parameters,
        synthetic=True,
    )


def immediate_burst_corpus(
    templates: Sequence[RequestTemplate],
    *,
    namespace: str,
    split: SplitName,
    cohort_count: int,
    cohort_popularity: Literal["uniform", "zipf"],
    cohort_seed: int,
    zipf_exponent: float = 1.0,
) -> RequestCorpus:
    """Generate a labelled-synthetic trace with every request at offset zero."""
    cohorts = cohort_assignments(
        len(templates),
        cohort_count=cohort_count,
        popularity=cohort_popularity,
        seed=cohort_seed,
        zipf_exponent=zipf_exponent,
    )
    source_parameters: dict[str, JsonScalar] = {
        "cohort_count": cohort_count,
        "cohort_popularity": cohort_popularity,
        "cohort_seed": cohort_seed,
        "generator": "immediate_burst_v1",
        "request_count": len(templates),
        "zipf_exponent": zipf_exponent,
    }
    return _build_corpus(
        namespace=namespace,
        split=split,
        templates=templates,
        arrivals_us=(0,) * len(templates),
        cohorts=cohorts,
        label="synthetic immediate burst",
        source_kind="immediate_burst",
        source_parameters=source_parameters,
        synthetic=True,
    )


def closed_loop_corpus(
    templates: Sequence[RequestTemplate],
    *,
    namespace: str,
    split: SplitName,
    concurrency: int,
    cohort_count: int,
    cohort_popularity: Literal["uniform", "zipf"],
    cohort_seed: int,
    zipf_exponent: float = 1.0,
) -> RequestCorpus:
    """Prepare a zero-think closed population consumed under a fixed cap.

    Arrival offset zero means that request content is eligible at score start;
    the executor issues at most ``concurrency`` HTTP requests and replenishes a
    slot only after the preceding request reaches a terminal outcome.
    """

    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency < 1
    ):
        raise ValueError("closed-loop concurrency must be positive")
    if len(templates) < concurrency:
        raise ValueError("closed-loop request pool must cover every client lane")
    cohorts = cohort_assignments(
        len(templates),
        cohort_count=cohort_count,
        popularity=cohort_popularity,
        seed=cohort_seed,
        zipf_exponent=zipf_exponent,
    )
    return _build_corpus(
        namespace=namespace,
        split=split,
        templates=templates,
        arrivals_us=(0,) * len(templates),
        cohorts=cohorts,
        label="synthetic zero-think closed loop",
        source_kind="closed_loop",
        source_parameters={
            "cohort_count": cohort_count,
            "cohort_popularity": cohort_popularity,
            "cohort_seed": cohort_seed,
            "concurrency": concurrency,
            "generator": "closed_loop_zero_think_v1",
            "request_count": len(templates),
            "zipf_exponent": zipf_exponent,
        },
        synthetic=True,
    )


@dataclass(frozen=True)
class ExternalShapeRow:
    arrival_us: int
    input_tokens: int
    requested_output_tokens: int

    def validate(self) -> None:
        _require_nonnegative_integer(self.arrival_us, "external arrival offset")
        if (
            isinstance(self.input_tokens, bool)
            or not isinstance(self.input_tokens, int)
            or isinstance(self.requested_output_tokens, bool)
            or not isinstance(self.requested_output_tokens, int)
            or self.input_tokens < 1
            or self.requested_output_tokens < 1
        ):
            raise ValueError(
                "external input and requested output lengths must be positive"
            )


@dataclass(frozen=True)
class ExternalWorkloadShape:
    """Checksum-verified rows supplied from a public workload-shape asset."""

    source_name: str
    source_revision: str
    rows: tuple[ExternalShapeRow, ...]
    rows_sha256: str

    @classmethod
    def from_rows(
        cls,
        *,
        source_name: str,
        source_revision: str,
        rows: Sequence[ExternalShapeRow],
        declared_rows_sha256: str,
    ) -> ExternalWorkloadShape:
        registered = REGISTERED_EXTERNAL_SHAPES.get((source_name, source_revision))
        if registered is None:
            raise ValueError(
                "external workload source/revision is not registered in the source lock"
            )
        if declared_rows_sha256 != registered:
            raise ValueError(
                "external workload rows differ from the registered source lock"
            )
        value = cls(
            source_name=source_name,
            source_revision=source_revision,
            rows=tuple(rows),
            rows_sha256=declared_rows_sha256,
        )
        value.validate()
        return value

    @staticmethod
    def digest_rows(rows: Sequence[ExternalShapeRow]) -> str:
        return _sha256(tuple(asdict(row) for row in rows))

    def validate(self) -> None:
        if not self.source_name or not self.source_revision:
            raise ValueError("external workload shape requires source and revision")
        if "synthetic" in self.source_name.casefold():
            raise ValueError("external workload shape cannot claim a synthetic source")
        if not self.rows:
            raise ValueError("external workload shape must not be empty")
        registered = REGISTERED_EXTERNAL_SHAPES.get(
            (self.source_name, self.source_revision)
        )
        if registered is None or registered != self.rows_sha256:
            raise ValueError(
                "external workload source/revision is not registered in the source lock"
            )
        last_arrival = -1
        for row in self.rows:
            row.validate()
            if row.arrival_us < last_arrival:
                raise ValueError("external workload rows must be arrival ordered")
            last_arrival = row.arrival_us
        if self.rows_sha256 != self.digest_rows(self.rows):
            raise ValueError("external workload-shape row checksum does not match")

    @property
    def source_identity_sha256(self) -> str:
        self.validate()
        return _sha256(
            {
                "source_name": self.source_name,
                "source_revision": self.source_revision,
                "rows_sha256": self.rows_sha256,
            }
        )


def external_shape_corpus(
    shape: ExternalWorkloadShape,
    *,
    namespace: str,
    split: SplitName,
    tokenized_inputs: Sequence[tuple[int, ...]],
    sampling: Sequence[FrozenSamplingParameters],
    cancellation_offsets_us: Sequence[int | None] | None = None,
    cohort_count: int,
    cohort_popularity: Literal["uniform", "zipf"],
    cohort_seed: int,
    zipf_exponent: float = 1.0,
) -> RequestCorpus:
    """Map external arrival/length rows to checksum-bound tokenized requests."""
    shape.validate()
    if len(tokenized_inputs) != len(shape.rows) or len(sampling) != len(shape.rows):
        raise ValueError("external rows, tokenized inputs, and sampling must align")
    cancellations = (
        tuple(cancellation_offsets_us)
        if cancellation_offsets_us is not None
        else (None,) * len(shape.rows)
    )
    if len(cancellations) != len(shape.rows):
        raise ValueError("external cancellation flags must align with shape rows")
    templates: list[RequestTemplate] = []
    for row, token_ids, parameters, cancellation in zip(
        shape.rows, tokenized_inputs, sampling, cancellations, strict=True
    ):
        if len(token_ids) != row.input_tokens:
            raise ValueError("tokenized input does not match external input length")
        templates.append(
            RequestTemplate(
                input_token_ids=token_ids,
                requested_output_tokens=row.requested_output_tokens,
                sampling=parameters,
                cancellation_offset_us=cancellation,
            )
        )
    cohorts = cohort_assignments(
        len(templates),
        cohort_count=cohort_count,
        popularity=cohort_popularity,
        seed=cohort_seed,
        zipf_exponent=zipf_exponent,
    )
    return _build_corpus(
        namespace=namespace,
        split=split,
        templates=templates,
        arrivals_us=tuple(row.arrival_us for row in shape.rows),
        cohorts=cohorts,
        label=f"{shape.source_name} workload-shape replay",
        source_kind="external_shape",
        source_parameters={
            "rows_sha256": shape.rows_sha256,
            "source_name": shape.source_name,
            "source_revision": shape.source_revision,
        },
        synthetic=False,
    )


@dataclass(frozen=True)
class ContentSplitIdentity:
    split: SplitName
    corpus_sha256: str
    unique_input_count: int
    input_set_sha256: str


def assert_content_disjoint(
    corpora: Sequence[RequestCorpus],
) -> tuple[ContentSplitIdentity, ...]:
    """Return split identities after rejecting token-content leakage."""
    if not corpora:
        raise ValueError("at least one corpus is required")
    seen_by_split: dict[SplitName, set[str]] = {}
    identities: list[ContentSplitIdentity] = []
    for corpus in corpora:
        corpus.validate()
        if corpus.split in seen_by_split:
            raise ValueError("content-disjoint audit requires one corpus per split")
        input_hashes = {
            request.field_hashes.tokenized_input_sha256 for request in corpus.requests
        }
        for other_split, other_hashes in seen_by_split.items():
            if input_hashes & other_hashes:
                raise ValueError(
                    f"tokenized input content overlaps {other_split}/{corpus.split}"
                )
        seen_by_split[corpus.split] = input_hashes
        identities.append(
            ContentSplitIdentity(
                split=corpus.split,
                corpus_sha256=corpus.hashes.corpus_sha256,
                unique_input_count=len(input_hashes),
                input_set_sha256=_sha256(tuple(sorted(input_hashes))),
            )
        )
    return tuple(identities)


@dataclass(frozen=True)
class ProductionWindow:
    """Registered half-open arrival windows plus deadline-bounded drain."""

    warmup_duration_us: int
    arrival_duration_us: int
    request_deadline_us: int
    drain_duration_us: int

    def validate(self) -> None:
        _require_nonnegative_integer(self.warmup_duration_us, "warm-up duration")
        for value, name in (
            (self.arrival_duration_us, "arrival duration"),
            (self.request_deadline_us, "request deadline"),
            (self.drain_duration_us, "drain duration"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def scored_global_end_us(self) -> int:
        self.validate()
        return self.arrival_duration_us + self.drain_duration_us

    @cached_property
    def sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))

    def request_timeout_us(self, request: ImmutableRequest) -> int:
        return min(
            request.arrival_us + self.request_deadline_us,
            self.scored_global_end_us,
        )


@dataclass(frozen=True)
class ProductionLoadPlan:
    """Warm-up is bound to replay identity but excluded from score accounting."""

    warmup: RequestCorpus | None
    scored: RequestCorpus
    window: ProductionWindow
    cold_start_separate: bool = True

    def validate(self) -> None:
        self.window.validate()
        self.scored.validate()
        if self.scored.split == "warmup":
            raise ValueError("scored corpus cannot use the warmup split")
        if any(
            request.arrival_us >= self.window.arrival_duration_us
            for request in self.scored.requests
        ):
            raise ValueError("scored request lies outside fixed arrival window")
        if self.window.warmup_duration_us == 0:
            if self.warmup is not None:
                raise ValueError("zero-duration warm-up cannot contain requests")
        else:
            if self.warmup is None:
                raise ValueError("positive warm-up duration requires a warm-up corpus")
            self.warmup.validate()
            if self.warmup.split != "warmup":
                raise ValueError("warm-up corpus must use the warmup split")
            if any(
                request.arrival_us >= self.window.warmup_duration_us
                for request in self.warmup.requests
            ):
                raise ValueError("warm-up request lies outside excluded warm-up window")
        if not self.cold_start_separate:
            raise ValueError("cold start must be measured separately")

    @cached_property
    def paired_replay_sha256(self) -> str:
        self.validate()
        return _sha256(
            {
                "schema_version": 1,
                "warmup_corpus_sha256": (
                    self.warmup.hashes.corpus_sha256 if self.warmup else None
                ),
                "scored_hashes": asdict(self.scored.hashes),
                "window_sha256": self.window.sha256,
                "cold_start_separate": self.cold_start_separate,
            }
        )


@dataclass(frozen=True)
class ReplayBinding:
    method: str
    paired_replay_sha256: str
    request_ids_sha256: str
    binding_sha256: str

    def validate(self) -> None:
        if not self.method:
            raise ValueError("replay binding method must not be empty")
        if not _is_sha256(self.paired_replay_sha256) or not _is_sha256(
            self.request_ids_sha256
        ):
            raise ValueError("replay binding identities must be lowercase SHA-256")
        expected = _sha256(
            {
                "method": self.method,
                "paired_replay_sha256": self.paired_replay_sha256,
                "request_ids_sha256": self.request_ids_sha256,
            }
        )
        if self.binding_sha256 != expected:
            raise ValueError("replay binding hash does not match its content")

    @classmethod
    def create(cls, plan: ProductionLoadPlan, *, method: str) -> ReplayBinding:
        plan.validate()
        if not method:
            raise ValueError("replay binding method must not be empty")
        paired = plan.paired_replay_sha256
        request_ids = plan.scored.hashes.request_ids_sha256
        value = cls(
            method=method,
            paired_replay_sha256=paired,
            request_ids_sha256=request_ids,
            binding_sha256=_sha256(
                {
                    "method": method,
                    "paired_replay_sha256": paired,
                    "request_ids_sha256": request_ids,
                }
            ),
        )
        value.validate()
        return value


def assert_identical_replay(bindings: Sequence[ReplayBinding]) -> str:
    if len(bindings) < 2:
        raise ValueError("paired replay requires at least two method bindings")
    for binding in bindings:
        binding.validate()
    methods = {binding.method for binding in bindings}
    if len(methods) != len(bindings):
        raise ValueError("paired replay methods must be unique")
    identities = {
        (binding.paired_replay_sha256, binding.request_ids_sha256)
        for binding in bindings
    }
    if len(identities) != 1:
        raise ValueError("paired methods are not bound to an identical trace")
    return bindings[0].paired_replay_sha256


@dataclass(frozen=True)
class RequestOutcome:
    request_id: str
    status: OutcomeStatus
    admitted_at_us: int | None
    terminal_at_us: int | None
    code: str
    offered_at_us: int | None = None


@dataclass(frozen=True)
class LoadAccounting:
    offered: int
    admitted: int
    rejected: int
    completed: int
    timed_out: int
    cancelled: int
    unfinished: int
    score_started_us: int
    score_ended_us: int
    elapsed_us: int

    def validate(self) -> None:
        if self.offered != (
            self.rejected
            + self.completed
            + self.timed_out
            + self.cancelled
            + self.unfinished
        ):
            raise ValueError("terminal accounting does not partition offered requests")
        if self.admitted > self.offered or self.elapsed_us != (
            self.score_ended_us - self.score_started_us
        ):
            raise ValueError("load accounting interval is inconsistent")


def account_scored_requests(
    plan: ProductionLoadPlan, outcomes: Sequence[RequestOutcome]
) -> LoadAccounting:
    """Account every offered request exactly once, including unfinished work."""
    plan.validate()
    by_id: dict[str, RequestOutcome] = {}
    for outcome in outcomes:
        if outcome.request_id in by_id:
            raise ValueError("duplicate request outcome")
        if outcome.status not in {
            "rejected",
            "completed",
            "timed_out",
            "cancelled",
            "unfinished",
        }:
            raise ValueError("request outcome status is invalid")
        if not outcome.code:
            raise ValueError("request outcome code must not be empty")
        by_id[outcome.request_id] = outcome
    expected = {request.request_id for request in plan.scored.requests}
    closed_loop = plan.scored.source_kind == "closed_loop"
    if not by_id or set(by_id) - expected:
        raise ValueError("request outcome coverage does not match offered requests")
    if closed_loop:
        source = dict(plan.scored.source_parameters)
        concurrency = source.get("concurrency")
        if (
            not isinstance(concurrency, int)
            or isinstance(concurrency, bool)
            or concurrency < 1
        ):
            raise ValueError("closed-loop accounting lacks its population size")
        actual_ids = set(by_id)
        for lane in range(concurrency):
            planned = tuple(plan.scored.requests[lane::concurrency])
            if not planned or planned[0].request_id not in actual_ids:
                raise ValueError("closed-loop outcomes omit a registered client lane")
            seen_gap = False
            for request in planned:
                present = request.request_id in actual_ids
                if seen_gap and present:
                    raise ValueError("closed-loop outcomes are not a per-client prefix")
                seen_gap = seen_gap or not present
    elif set(by_id) != expected:
        raise ValueError("request outcome coverage does not match offered requests")

    counts = {
        status: 0
        for status in ("rejected", "completed", "timed_out", "cancelled", "unfinished")
    }
    admitted = 0
    offered_requests = tuple(
        request for request in plan.scored.requests if request.request_id in by_id
    )
    effective_arrivals: dict[str, int] = {}
    for request in offered_requests:
        outcome = by_id[request.request_id]
        offered_at_us = (
            outcome.offered_at_us
            if outcome.offered_at_us is not None
            else request.arrival_us
        )
        _require_nonnegative_integer(offered_at_us, "request offer time")
        if closed_loop:
            if not 0 <= offered_at_us < plan.window.arrival_duration_us:
                raise ValueError("closed-loop offer lies outside the arrival window")
        elif offered_at_us != request.arrival_us:
            raise ValueError("open-loop offer differs from the registered trace")
        effective_arrivals[request.request_id] = offered_at_us
        timeout_us = min(
            offered_at_us + plan.window.request_deadline_us,
            plan.window.scored_global_end_us,
        )
        cancel_us = (
            offered_at_us + request.cancellation_offset_us
            if request.cancellation_offset_us is not None
            else None
        )
        if outcome.admitted_at_us is not None:
            _require_nonnegative_integer(outcome.admitted_at_us, "admission time")
            if not offered_at_us <= outcome.admitted_at_us <= timeout_us:
                raise ValueError("admission time lies outside request lifetime")
            admitted += 1
        if (
            outcome.status in {"completed", "timed_out", "unfinished"}
            and outcome.admitted_at_us is None
        ):
            raise ValueError(
                "completed, timed-out, and unfinished requests must be admitted"
            )
        if outcome.status == "rejected" and outcome.admitted_at_us is not None:
            raise ValueError("rejected request cannot be admitted")
        if outcome.status == "unfinished":
            if outcome.terminal_at_us is not None:
                raise ValueError("unfinished request cannot have a terminal time")
            if cancel_us is not None and cancel_us <= timeout_us:
                raise ValueError("scheduled cancellation cannot remain unfinished")
        else:
            if outcome.terminal_at_us is None:
                raise ValueError("terminal request outcome requires a terminal time")
            _require_nonnegative_integer(outcome.terminal_at_us, "terminal time")
            if not offered_at_us <= outcome.terminal_at_us <= timeout_us:
                raise ValueError("terminal time lies outside request lifetime")
            if (
                outcome.admitted_at_us is not None
                and outcome.terminal_at_us < outcome.admitted_at_us
            ):
                raise ValueError("request terminated before admission")
            if outcome.status == "timed_out" and outcome.terminal_at_us != timeout_us:
                raise ValueError(
                    "timeout must occur at the registered effective deadline"
                )
            if outcome.status == "cancelled" and (
                cancel_us is None or outcome.terminal_at_us != cancel_us
            ):
                raise ValueError(
                    "cancellation must match the hashed cancellation schedule"
                )
            if (
                outcome.status == "completed"
                and cancel_us is not None
                and outcome.terminal_at_us > cancel_us
            ):
                raise ValueError(
                    "completion cannot occur after a scheduled cancellation"
                )
            if (
                outcome.status == "timed_out"
                and cancel_us is not None
                and cancel_us < timeout_us
            ):
                raise ValueError(
                    "timeout cannot bypass an earlier scheduled cancellation"
                )
        counts[outcome.status] += 1

    if closed_loop:
        concurrency = int(dict(plan.scored.source_parameters)["concurrency"])
        for lane in range(concurrency):
            previous_terminal: int | None = None
            for request in plan.scored.requests[lane::concurrency]:
                outcome = by_id.get(request.request_id)
                if outcome is None:
                    break
                offered_at_us = effective_arrivals[request.request_id]
                if previous_terminal is None and offered_at_us != 0:
                    raise ValueError(
                        "closed-loop client did not begin at the score boundary"
                    )
                if previous_terminal is not None and offered_at_us != previous_terminal:
                    raise ValueError(
                        "closed-loop client was not replenished at its prior terminal"
                    )
                if outcome.terminal_at_us is None:
                    previous_terminal = None
                    break
                previous_terminal = outcome.terminal_at_us

    # The denominator starts at the first registered open-loop arrival or the
    # first realized closed-loop offer. It ends at the latest observed terminal
    # event or, for unfinished work, that request's effective timeout. This
    # keeps every actual offer in the denominator without charging idle time
    # before the workload begins or after all offered work has ended.
    score_started = min(effective_arrivals.values())
    effective_terminals = []
    for request in offered_requests:
        outcome = by_id[request.request_id]
        offered_at_us = effective_arrivals[request.request_id]
        effective_terminals.append(
            min(
                offered_at_us + plan.window.request_deadline_us,
                plan.window.scored_global_end_us,
            )
            if outcome.status == "unfinished"
            else outcome.terminal_at_us
        )
    if any(value is None for value in effective_terminals):
        raise AssertionError("validated terminal accounting produced no endpoint")
    score_ended = max(int(value) for value in effective_terminals if value is not None)
    accounting = LoadAccounting(
        offered=len(offered_requests),
        admitted=admitted,
        rejected=counts["rejected"],
        completed=counts["completed"],
        timed_out=counts["timed_out"],
        cancelled=counts["cancelled"],
        unfinished=counts["unfinished"],
        score_started_us=score_started,
        score_ended_us=score_ended,
        elapsed_us=score_ended - score_started,
    )
    accounting.validate()
    return accounting


@dataclass(frozen=True)
class TokenChunkTiming:
    """One streamed chunk; multi-token chunks require per-token timestamps."""

    request_id: str
    first_token_index: int
    token_count: int
    chunk_observed_at_us: int
    per_token_observed_at_us: tuple[int, ...] | None = None

    def validate(self) -> None:
        if not self.request_id:
            raise ValueError("timing request ID must not be empty")
        _require_nonnegative_integer(self.first_token_index, "first token index")
        if (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, int)
            or self.token_count < 1
        ):
            raise ValueError("timing chunk token count must be positive")
        _require_nonnegative_integer(self.chunk_observed_at_us, "chunk timestamp")
        if self.per_token_observed_at_us is not None:
            if len(self.per_token_observed_at_us) != self.token_count:
                raise ValueError("per-token timestamp count does not match chunk")
            previous = -1
            for timestamp in self.per_token_observed_at_us:
                _require_nonnegative_integer(timestamp, "per-token timestamp")
                if timestamp < previous or timestamp > self.chunk_observed_at_us:
                    raise ValueError(
                        "per-token timestamps are not ordered within chunk"
                    )
                previous = timestamp


@dataclass(frozen=True)
class TimingCoverage:
    request_id: str
    output_tokens: int
    expected_itl_intervals: int
    supported_itl_intervals: int
    coalesced_tokens: int
    itl_coverage: float
    ttft_us: int | None
    supported_itls_us: tuple[int, ...]

    @property
    def full_itl_coverage(self) -> bool:
        return self.supported_itl_intervals == self.expected_itl_intervals

    def itl_percentile_us(self, percentile: float) -> float | None:
        """Return a claimable percentile only when every ITL is supported."""
        if not math.isfinite(percentile) or not 0 <= percentile <= 100:
            raise ValueError("percentile must be finite and lie in [0, 100]")
        if not self.full_itl_coverage or not self.supported_itls_us:
            return None
        return _linear_percentile(self.supported_itls_us, percentile)

    def diagnostic_supported_percentile_us(self, percentile: float) -> float | None:
        """Summarize the explicitly named supported subset, never as headline p99."""
        if not math.isfinite(percentile) or not 0 <= percentile <= 100:
            raise ValueError("percentile must be finite and lie in [0, 100]")
        if not self.supported_itls_us:
            return None
        return _linear_percentile(self.supported_itls_us, percentile)


def _linear_percentile(values: Sequence[int], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = percentile / 100 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def evaluate_token_timing(
    *,
    request_id: str,
    request_started_us: int,
    expected_output_tokens: int,
    chunks: Sequence[TokenChunkTiming],
) -> TimingCoverage:
    """Compute ITLs without inventing timestamps inside coalesced chunks."""
    if not request_id:
        raise ValueError("timing request ID must not be empty")
    _require_nonnegative_integer(request_started_us, "request start time")
    _require_nonnegative_integer(expected_output_tokens, "expected output tokens")
    exact_timestamps: list[int | None] = []
    coalesced_tokens = 0
    next_index = 0
    previous_chunk_time = request_started_us
    for chunk in chunks:
        chunk.validate()
        if chunk.request_id != request_id:
            raise ValueError("timing chunk belongs to a different request")
        if chunk.first_token_index != next_index:
            raise ValueError("timing chunks must cover token indices exactly once")
        if chunk.chunk_observed_at_us < previous_chunk_time:
            raise ValueError("chunk timestamps must be nondecreasing")
        if chunk.per_token_observed_at_us is not None:
            if any(
                timestamp < request_started_us
                for timestamp in chunk.per_token_observed_at_us
            ):
                raise ValueError("token timestamp precedes request start")
            exact_timestamps.extend(chunk.per_token_observed_at_us)
        elif chunk.token_count == 1:
            exact_timestamps.append(chunk.chunk_observed_at_us)
        else:
            exact_timestamps.extend((None,) * chunk.token_count)
            coalesced_tokens += chunk.token_count
        next_index += chunk.token_count
        previous_chunk_time = chunk.chunk_observed_at_us
    if next_index != expected_output_tokens:
        raise ValueError("timing chunks do not cover expected output tokens")
    known_timestamps = tuple(
        timestamp for timestamp in exact_timestamps if timestamp is not None
    )
    if any(current < previous for previous, current in pairwise(known_timestamps)):
        raise ValueError("exact token timestamps must be globally nondecreasing")

    supported_itls = tuple(
        current - previous
        for previous, current in pairwise(exact_timestamps)
        if previous is not None and current is not None
    )
    expected_itls = max(0, expected_output_tokens - 1)
    coverage = len(supported_itls) / expected_itls if expected_itls else 1.0
    first_timestamp = exact_timestamps[0] if exact_timestamps else None
    return TimingCoverage(
        request_id=request_id,
        output_tokens=expected_output_tokens,
        expected_itl_intervals=expected_itls,
        supported_itl_intervals=len(supported_itls),
        coalesced_tokens=coalesced_tokens,
        itl_coverage=coverage,
        ttft_us=(
            first_timestamp - request_started_us
            if first_timestamp is not None
            else None
        ),
        supported_itls_us=supported_itls,
    )
