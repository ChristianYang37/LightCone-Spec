"""Code-owned production arrival authority for the trusted operator.

This module keeps request *content* and request *arrival* identities separate.
E5 reuses the locked task prompts, while this file derives deterministic
closed-loop pools, Poisson arrivals, an immediate burst, and one real
BurstGPT-v2.0 arrival-shape replay.  No result-dependent scalar enters the
derivation: the only measured input is the request-rate anchor sealed by E3a.

The six release assets are pinned because they are the complete upstream v2.0
release.  The replay itself deliberately uses the latest independent,
failure-free trace (``BurstGPT_without_fails_3.csv``); combining the full and
failure-filtered files would double-count the same requests.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Literal, Self


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_positive_int(label: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class OfficialBurstGptAsset:
    name: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or Path(self.name).name != self.name
            or not self.name.endswith(".csv")
        ):
            raise ValueError("BurstGPT asset name is invalid")
        _require_positive_int("BurstGPT asset size", self.size)
        _require_sha256("BurstGPT asset", self.sha256)


BURSTGPT_V2_RELEASE_TAG = "v2.0"
BURSTGPT_V2_RELEASE_TAG_COMMIT = "7eb2c4f8350f8a6985272386f5c14af1f678b299"
BURSTGPT_V2_RELEASE_ID = 277_119_823
BURSTGPT_V2_RELEASE_PUBLISHED_AT = "2026-01-15T16:39:16Z"
BURSTGPT_V2_RELEASE_URL = "https://github.com/HPMLL/BurstGPT/releases/tag/v2.0"
BURSTGPT_V2_ACTIVE_ASSET = "BurstGPT_without_fails_3.csv"

BURSTGPT_V2_ASSETS = (
    OfficialBurstGptAsset(
        "BurstGPT_1.csv",
        52_283_111,
        "4bb3783693d0a435686fbfc885615d2349bd067239079fa4b749f2e679e12122",
    ),
    OfficialBurstGptAsset(
        "BurstGPT_2.csv",
        144_819_209,
        "44bf5942b03fca42c01a226a545ad3a750ec62688faecd67b6831c56a8928bf7",
    ),
    OfficialBurstGptAsset(
        "BurstGPT_3.csv",
        231_682_327,
        "2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f",
    ),
    OfficialBurstGptAsset(
        "BurstGPT_without_fails_1.csv",
        51_429_517,
        "a4d068a7113ec0290e74063a1b3447dc6001a30e4298eb313581b71006dda1f4",
    ),
    OfficialBurstGptAsset(
        "BurstGPT_without_fails_2.csv",
        142_376_815,
        "56193aa9b2bb26128ded43d2d29a960df6bf5af062bcfc9b005f3fcaa4e6e501",
    ),
    OfficialBurstGptAsset(
        "BurstGPT_without_fails_3.csv",
        217_312_026,
        "3326259f9efb11845bc5ef85fa97e6f691050b0974621c91ef22acd566c43a40",
    ),
)

if tuple(row.name for row in BURSTGPT_V2_ASSETS) != tuple(
    sorted(row.name for row in BURSTGPT_V2_ASSETS)
):
    raise RuntimeError("BurstGPT v2.0 asset pins are not canonical")


E5_MAX_BLOCKS = 24  # four excluded pilots plus at most twenty final blocks
E5_HEADLINE_ARRIVAL_DURATION_US = 60_000_000
E5_SOAK_ARRIVAL_DURATION_US = 300_000_000
E5_WARMUP_DURATION_US = 10_000_000
E5_REQUEST_DEADLINE_US = 120_000_000
E5_DRAIN_DURATION_US = 180_000_000
E5_P99_MINIMUM_COMPLETED_REQUESTS = 10_000
E5_P99_EXTENSION_OFFERED_REQUESTS = 11_000
E5_MAX_REQUEST_ROWS = 100_000
E5_SOAK_LOAD_FACTORS = {
    "moderate_soak": Fraction(3, 4),
    "saturation_soak": Fraction(1, 1),
    "overload_soak": Fraction(5, 4),
}


FORMAL_SINGLE_OPERATOR_E5_LOAD_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_e5_load_protocol",
        "lambda_star": (
            "exact_E3a_Static_request_rate_at_context_40928_"
            "short_input_long_generation_matched_width_common_load"
        ),
        "standard_arrival_duration_us": E5_HEADLINE_ARRIVAL_DURATION_US,
        "soak_arrival_duration_us": E5_SOAK_ARRIVAL_DURATION_US,
        "warmup_duration_us": E5_WARMUP_DURATION_US,
        "request_deadline_us": E5_REQUEST_DEADLINE_US,
        "drain_duration_us": E5_DRAIN_DURATION_US,
        "closed_loop_pool": (
            "max_4x_concurrency_or_2x_lambda_star_window_unless_selected_"
            "p99_family_uses_exact_extension_offered_pool"
        ),
        "poisson": "sha256_uniform_inverse_exponential_round_nearest_min_1us",
        "immediate_burst": "lambda_star_x_standard_window_all_at_zero",
        "soak_load_factors": {
            name: [value.numerator, value.denominator]
            for name, value in E5_SOAK_LOAD_FACTORS.items()
        },
        "burstgpt": {
            "release_tag": BURSTGPT_V2_RELEASE_TAG,
            "release_tag_commit": BURSTGPT_V2_RELEASE_TAG_COMMIT,
            "release_id": BURSTGPT_V2_RELEASE_ID,
            "assets": [asdict(row) for row in BURSTGPT_V2_ASSETS],
            "active_asset": BURSTGPT_V2_ACTIVE_ASSET,
            "selection": "one_centered_contiguous_window_in_each_of_24_row_strata",
            "projection": "arrival_offsets_only_content_remains_task_locked",
            "scaling": "preserve_interarrival_shape_scale_mean_to_lambda_star",
        },
        "pairing": "same_arrival_bytes_request_pool_and_block_for_every_method",
        "p99": {
            "minimum_completed": E5_P99_MINIMUM_COMPLETED_REQUESTS,
            "extension_offered": E5_P99_EXTENSION_OFFERED_REQUESTS,
            "scope": "selected_existing_family_cell_all_five_paired_methods",
            "closed_loop": "exact_extension_pool_at_registered_zero_think_time",
            "open_and_soak": (
                "exact_deterministic_poisson_prefix_at_registered_rate_and_"
                "duration_extended_to_last_arrival_plus_one_microsecond"
            ),
            "burstgpt": "exact_shape_preserving_extension_prefix",
            "immediate_burst": "exact_registered_all_at_zero_burst",
        },
    }
)


@dataclass(frozen=True)
class BurstGptVerifiedAsset:
    name: str
    size: int
    sha256: str
    row_count: int

    def __post_init__(self) -> None:
        OfficialBurstGptAsset(self.name, self.size, self.sha256)
        _require_positive_int("BurstGPT row count", self.row_count)


@dataclass(frozen=True)
class BurstGptV2ReleaseVerification:
    schema_version: Literal[1]
    kind: Literal["burstgpt_v2_release_verification"]
    release_tag: str
    release_tag_commit: str
    release_id: int
    release_published_at: str
    release_url: str
    active_asset: str
    assets: tuple[BurstGptVerifiedAsset, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "burstgpt_v2_release_verification"
            or self.release_tag != BURSTGPT_V2_RELEASE_TAG
            or self.release_tag_commit != BURSTGPT_V2_RELEASE_TAG_COMMIT
            or self.release_id != BURSTGPT_V2_RELEASE_ID
            or self.release_published_at != BURSTGPT_V2_RELEASE_PUBLISHED_AT
            or self.release_url != BURSTGPT_V2_RELEASE_URL
            or self.active_asset != BURSTGPT_V2_ACTIVE_ASSET
        ):
            raise ValueError("BurstGPT release identity differs from the source lock")
        if (
            type(self.assets) is not tuple
            or tuple(row.name for row in self.assets)
            != tuple(row.name for row in BURSTGPT_V2_ASSETS)
            or any(type(row) is not BurstGptVerifiedAsset for row in self.assets)
        ):
            raise ValueError("BurstGPT release verification coverage differs")
        for actual, expected in zip(self.assets, BURSTGPT_V2_ASSETS, strict=True):
            if (actual.name, actual.size, actual.sha256) != (
                expected.name,
                expected.size,
                expected.sha256,
            ):
                raise ValueError("BurstGPT verified asset differs from its pin")

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "release_tag": self.release_tag,
            "release_tag_commit": self.release_tag_commit,
            "release_id": self.release_id,
            "release_published_at": self.release_published_at,
            "release_url": self.release_url,
            "active_asset": self.active_asset,
            "assets": [asdict(row) for row in self.assets],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            "schema_version",
            "kind",
            "release_tag",
            "release_tag_commit",
            "release_id",
            "release_published_at",
            "release_url",
            "active_asset",
            "assets",
        }:
            raise ValueError("BurstGPT release verification fields differ")
        row = dict(value)
        raw_assets = row.pop("assets")
        if type(raw_assets) is not list:
            raise TypeError("BurstGPT verified assets must be an array")
        return cls(
            **row,
            assets=tuple(BurstGptVerifiedAsset(**item) for item in raw_assets),
        )  # type: ignore[arg-type]


_LEGACY_HEADER = (
    "Timestamp",
    "Model",
    "Request tokens",
    "Response tokens",
    "Total tokens",
    "Log Type",
)
_V2_HEADER = (
    "Timestamp",
    "Session ID",
    "Elapsed time",
    "Model",
    "Request tokens",
    "Response tokens",
    "Total tokens",
    "Log Type",
)


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _timestamp_us(value: str) -> int:
    try:
        timestamp = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("BurstGPT timestamp is not decimal") from error
    scaled = timestamp * 1_000_000
    if not scaled.is_finite() or scaled < 0 or scaled != scaled.to_integral_value():
        raise ValueError("BurstGPT timestamp is not an exact non-negative microsecond")
    return int(scaled)


def _positive_csv_int(label: str, value: str, *, allow_zero: bool) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"BurstGPT {label} is not an integer") from error
    if parsed < (0 if allow_zero else 1) or str(parsed) != value:
        raise ValueError(f"BurstGPT {label} is outside the registered domain")
    return parsed


def _validate_csv_asset(path: Path, *, require_failure_free: bool) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        header = tuple(reader.fieldnames or ())
        if header not in {_LEGACY_HEADER, _V2_HEADER}:
            raise ValueError("BurstGPT CSV header differs from release v2.0")
        last_timestamp = -1
        count = 0
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("BurstGPT CSV row width differs")
            timestamp = _timestamp_us(row["Timestamp"])
            request_tokens = _positive_csv_int(
                "request tokens",
                row["Request tokens"],
                allow_zero=not require_failure_free,
            )
            response_tokens = _positive_csv_int(
                "response tokens",
                row["Response tokens"],
                allow_zero=not require_failure_free,
            )
            total_tokens = _positive_csv_int(
                "total tokens",
                row["Total tokens"],
                allow_zero=not require_failure_free,
            )
            if timestamp < last_timestamp or total_tokens != (
                request_tokens + response_tokens
            ):
                raise ValueError("BurstGPT timestamp/token invariant differs")
            last_timestamp = timestamp
            count += 1
    return _require_positive_int("BurstGPT CSV row count", count)


def verify_burstgpt_v2_release(
    asset_paths: Mapping[str, str | Path],
) -> BurstGptV2ReleaseVerification:
    """Hash and semantically scan every official v2.0 release asset once."""

    if set(asset_paths) != {row.name for row in BURSTGPT_V2_ASSETS}:
        raise ValueError("BurstGPT release paths do not cover the six assets exactly")
    verified: list[BurstGptVerifiedAsset] = []
    for expected in BURSTGPT_V2_ASSETS:
        path = Path(asset_paths[expected.name]).resolve(strict=True)
        if not path.is_file() or path.is_symlink():
            raise ValueError("BurstGPT release asset is not a regular file")
        digest, size = _hash_file(path)
        if (digest, size) != (expected.sha256, expected.size):
            raise ValueError("BurstGPT release asset bytes differ from the pin")
        row_count = _validate_csv_asset(
            path,
            require_failure_free="without_fails" in expected.name,
        )
        verified.append(BurstGptVerifiedAsset(expected.name, size, digest, row_count))
    return BurstGptV2ReleaseVerification(
        schema_version=1,
        kind="burstgpt_v2_release_verification",
        release_tag=BURSTGPT_V2_RELEASE_TAG,
        release_tag_commit=BURSTGPT_V2_RELEASE_TAG_COMMIT,
        release_id=BURSTGPT_V2_RELEASE_ID,
        release_published_at=BURSTGPT_V2_RELEASE_PUBLISHED_AT,
        release_url=BURSTGPT_V2_RELEASE_URL,
        active_asset=BURSTGPT_V2_ACTIVE_ASSET,
        assets=tuple(verified),
    )


@dataclass(frozen=True)
class E3aLambdaStar:
    numerator_requests_x_1e9: int
    denominator_window_ns: int
    source_cell_id: str
    source_observation_sha256: str
    common_load: int
    matched_width: int
    rule: str

    def __post_init__(self) -> None:
        _require_positive_int("lambda-star numerator", self.numerator_requests_x_1e9)
        _require_positive_int("lambda-star denominator", self.denominator_window_ns)
        _require_sha256("lambda-star source cell", self.source_cell_id)
        _require_sha256(
            "lambda-star source observation", self.source_observation_sha256
        )
        _require_positive_int("lambda-star common load", self.common_load)
        if self.matched_width not in {4, 8, 16}:
            raise ValueError("lambda-star matched width differs")
        if self.rule != (
            "E3a_Static_context_40928_short_input_long_generation_"
            "matched_width_common_load_completed_requests_per_second"
        ):
            raise ValueError("lambda-star derivation rule differs")

    @property
    def requests_per_second(self) -> Fraction:
        return Fraction(
            self.numerator_requests_x_1e9,
            self.denominator_window_ns,
        )

    @classmethod
    def from_e3a_selection(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("E3a lambda-star selection must be an object")
        return cls(**value)


@dataclass(frozen=True)
class BurstGptArrivalWindow:
    release_verification_sha256: str
    active_asset: str
    active_asset_sha256: str
    block: int
    source_start_row: int
    source_row_count: int
    source_rows_sha256: str
    raw_span_us: int
    scaled_arrivals_us: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_sha256(
            "BurstGPT release verification", self.release_verification_sha256
        )
        _require_sha256("BurstGPT active asset", self.active_asset_sha256)
        _require_sha256("BurstGPT selected rows", self.source_rows_sha256)
        if (
            self.active_asset != BURSTGPT_V2_ACTIVE_ASSET
            or type(self.block) is not int
            or not 0 <= self.block < E5_MAX_BLOCKS
            or type(self.source_start_row) is not int
            or self.source_start_row < 0
            or self.source_row_count != len(self.scaled_arrivals_us)
            or self.source_row_count < 2
            or type(self.raw_span_us) is not int
            or self.raw_span_us < 1
            or self.scaled_arrivals_us[0] != 0
            or any(
                right < left
                for left, right in zip(
                    self.scaled_arrivals_us,
                    self.scaled_arrivals_us[1:],
                    strict=False,
                )
            )
        ):
            raise ValueError("BurstGPT selected arrival window differs")

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "scaled_arrivals_us": list(self.scaled_arrivals_us),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = set(cls.__dataclass_fields__)
        if type(value) is not dict or set(value) != expected:
            raise ValueError("BurstGPT arrival window fields differ")
        row = dict(value)
        raw_arrivals = row.pop("scaled_arrivals_us")
        if type(raw_arrivals) is not list:
            raise TypeError("BurstGPT scaled arrivals must be an array")
        return cls(**row, scaled_arrivals_us=tuple(raw_arrivals))  # type: ignore[arg-type]


def _round_fraction(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(2 * remainder >= value.denominator)


def select_burstgpt_arrival_window(
    *,
    active_asset_path: str | Path,
    verification: BurstGptV2ReleaseVerification,
    block: int,
    request_count: int,
    target_rate: Fraction,
) -> BurstGptArrivalWindow:
    """Select one disjoint, stratified real-trace window and rate-scale it."""

    verification.__post_init__()
    if type(block) is not int or not 0 <= block < E5_MAX_BLOCKS:
        raise ValueError("BurstGPT block is outside the registered prefix")
    _require_positive_int("BurstGPT window request count", request_count)
    if request_count < 2 or request_count > E5_MAX_REQUEST_ROWS or target_rate <= 0:
        raise ValueError("BurstGPT window size/rate differs")
    active = next(
        row for row in verification.assets if row.name == BURSTGPT_V2_ACTIVE_ASSET
    )
    path = Path(active_asset_path).resolve(strict=True)
    digest, size = _hash_file(path)
    if (digest, size) != (active.sha256, active.size):
        raise ValueError("BurstGPT active trace changed after verification")
    stratum_start = block * active.row_count // E5_MAX_BLOCKS
    stratum_end = (block + 1) * active.row_count // E5_MAX_BLOCKS
    if request_count > stratum_end - stratum_start:
        raise ValueError("BurstGPT request window exceeds its disjoint stratum")
    start = stratum_start + (stratum_end - stratum_start - request_count) // 2
    selected: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if tuple(reader.fieldnames or ()) != _V2_HEADER:
            raise ValueError("active BurstGPT trace requires the v2 extended header")
        for ordinal, row in enumerate(reader):
            if ordinal < start:
                continue
            if ordinal >= start + request_count:
                break
            timestamp = _timestamp_us(row["Timestamp"])
            request_tokens = _positive_csv_int(
                "request tokens", row["Request tokens"], allow_zero=False
            )
            response_tokens = _positive_csv_int(
                "response tokens", row["Response tokens"], allow_zero=False
            )
            total_tokens = _positive_csv_int(
                "total tokens", row["Total tokens"], allow_zero=False
            )
            if total_tokens != request_tokens + response_tokens:
                raise ValueError("selected BurstGPT token accounting differs")
            selected.append(
                {
                    "source_row": ordinal,
                    "timestamp_us": timestamp,
                    "request_tokens": request_tokens,
                    "response_tokens": response_tokens,
                    "model": row["Model"],
                    "log_type": row["Log Type"],
                }
            )
    if len(selected) != request_count:
        raise ValueError("BurstGPT selected window is truncated")
    timestamps = tuple(int(row["timestamp_us"]) for row in selected)
    if any(right < left for left, right in pairwise(timestamps)):
        raise ValueError("BurstGPT selected timestamps are not ordered")
    raw_span = timestamps[-1] - timestamps[0]
    if raw_span < 1:
        raise ValueError("BurstGPT selected window has no arrival span")
    target_span_us = Fraction((request_count - 1) * 1_000_000, 1) / target_rate
    scaled = tuple(
        _round_fraction(Fraction(timestamp - timestamps[0], raw_span) * target_span_us)
        for timestamp in timestamps
    )
    return BurstGptArrivalWindow(
        release_verification_sha256=verification.sha256,
        active_asset=active.name,
        active_asset_sha256=active.sha256,
        block=block,
        source_start_row=start,
        source_row_count=request_count,
        source_rows_sha256=_sha256(selected),
        raw_span_us=raw_span,
        scaled_arrivals_us=scaled,
    )


E5Family = Literal["closed_loop", "open_loop", "trace_or_soak", "topology_cohort"]


@dataclass(frozen=True)
class E5ArrivalPlan:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_e5_arrival_plan"]
    protocol_sha256: str
    cell_id: str
    paired_trace_sha256: str
    block: int
    family: E5Family
    arrival_policy: str
    lambda_star: E3aLambdaStar
    effective_rate_numerator: int | None
    effective_rate_denominator: int | None
    concurrency: int
    arrival_duration_us: int
    warmup_duration_us: int
    request_deadline_us: int
    drain_duration_us: int
    arrivals_us: tuple[int, ...]
    burstgpt_window: BurstGptArrivalWindow | None
    p99_extension_minimum_completed: int | None
    p99_extension_offered_requests: int | None

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_e5_arrival_plan"
            or self.protocol_sha256 != FORMAL_SINGLE_OPERATOR_E5_LOAD_PROTOCOL_SHA256
        ):
            raise ValueError("E5 arrival plan schema differs")
        _require_sha256("E5 arrival plan cell", self.cell_id)
        _require_sha256("E5 arrival plan paired trace", self.paired_trace_sha256)
        if (
            type(self.block) is not int
            or not 0 <= self.block < E5_MAX_BLOCKS
            or self.family
            not in {"closed_loop", "open_loop", "trace_or_soak", "topology_cohort"}
            or type(self.arrival_policy) is not str
            or not self.arrival_policy
            or type(self.lambda_star) is not E3aLambdaStar
        ):
            raise ValueError("E5 arrival plan scientific axes differ")
        _require_positive_int("E5 arrival plan concurrency", self.concurrency)
        _require_positive_int("E5 arrival duration", self.arrival_duration_us)
        _require_positive_int("E5 request deadline", self.request_deadline_us)
        _require_positive_int("E5 drain duration", self.drain_duration_us)
        if self.warmup_duration_us != E5_WARMUP_DURATION_US:
            raise ValueError("E5 warm-up duration differs")
        if (
            type(self.arrivals_us) is not tuple
            or not self.arrivals_us
            or len(self.arrivals_us) > E5_MAX_REQUEST_ROWS
            or any(type(value) is not int or value < 0 for value in self.arrivals_us)
            or any(
                right < left
                for left, right in zip(self.arrivals_us, self.arrivals_us[1:])
            )
            or self.arrivals_us[-1] >= self.arrival_duration_us
        ):
            raise ValueError("E5 arrival offsets differ")
        if (self.effective_rate_numerator is None) != (
            self.effective_rate_denominator is None
        ):
            raise ValueError("E5 effective rate is incomplete")
        if self.effective_rate_numerator is not None:
            _require_positive_int(
                "E5 effective rate numerator", self.effective_rate_numerator
            )
            _require_positive_int(
                "E5 effective rate denominator", self.effective_rate_denominator
            )
        if (self.arrival_policy == "burstgpt_shape") != (
            self.burstgpt_window is not None
        ):
            raise ValueError("E5 BurstGPT window presence differs")
        if self.burstgpt_window is not None and (
            self.burstgpt_window.block != self.block
            or self.burstgpt_window.scaled_arrivals_us != self.arrivals_us
        ):
            raise ValueError("E5 BurstGPT arrivals differ from their source window")
        if (self.p99_extension_minimum_completed is None) != (
            self.p99_extension_offered_requests is None
        ):
            raise ValueError("E5 p99 extension is incomplete")
        if self.p99_extension_minimum_completed is not None and (
            self.p99_extension_minimum_completed != E5_P99_MINIMUM_COMPLETED_REQUESTS
            or self.p99_extension_offered_requests != E5_P99_EXTENSION_OFFERED_REQUESTS
            or len(self.arrivals_us) != self.p99_extension_offered_requests
        ):
            raise ValueError("E5 p99 extension requirement differs")

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "cell_id": self.cell_id,
            "paired_trace_sha256": self.paired_trace_sha256,
            "block": self.block,
            "family": self.family,
            "arrival_policy": self.arrival_policy,
            "lambda_star": asdict(self.lambda_star),
            "effective_rate_numerator": self.effective_rate_numerator,
            "effective_rate_denominator": self.effective_rate_denominator,
            "concurrency": self.concurrency,
            "arrival_duration_us": self.arrival_duration_us,
            "warmup_duration_us": self.warmup_duration_us,
            "request_deadline_us": self.request_deadline_us,
            "drain_duration_us": self.drain_duration_us,
            "arrivals_us": list(self.arrivals_us),
            "burstgpt_window": (
                None if self.burstgpt_window is None else self.burstgpt_window.to_dict()
            ),
            "p99_extension_minimum_completed": (self.p99_extension_minimum_completed),
            "p99_extension_offered_requests": self.p99_extension_offered_requests,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = set(cls.__dataclass_fields__)
        if type(value) is not dict or set(value) != expected:
            raise ValueError("E5 arrival plan fields differ")
        row = dict(value)
        raw_lambda = row.pop("lambda_star")
        raw_arrivals = row.pop("arrivals_us")
        raw_burst = row.pop("burstgpt_window")
        if type(raw_lambda) is not dict or type(raw_arrivals) is not list:
            raise TypeError("E5 arrival plan nested fields differ")
        return cls(
            **row,
            lambda_star=E3aLambdaStar.from_e3a_selection(raw_lambda),
            arrivals_us=tuple(raw_arrivals),
            burstgpt_window=(
                None
                if raw_burst is None
                else BurstGptArrivalWindow.from_dict(raw_burst)
            ),
        )  # type: ignore[arg-type]


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _poisson_arrivals(
    *, request_count: int, rate_per_second: Fraction, seed: int
) -> tuple[int, ...]:
    if request_count < 1 or rate_per_second <= 0 or seed < 0:
        raise ValueError("E5 Poisson inputs differ")
    rate = float(rate_per_second)
    arrivals: list[int] = []
    elapsed = 0
    for ordinal in range(request_count):
        digest = hashlib.sha256(f"arrival:{seed}:{ordinal}".encode()).digest()
        uniform = (int.from_bytes(digest, "big") + 0.5) / (1 << 256)
        gap = max(1, round(-math.log1p(-uniform) * 1_000_000 / rate))
        elapsed += gap
        arrivals.append(elapsed)
    return tuple(arrivals)


def _seed(*, paired_trace_sha256: str, block: int, label: str) -> int:
    _require_sha256("E5 paired trace", paired_trace_sha256)
    digest = hashlib.sha256(
        (
            f"{FORMAL_SINGLE_OPERATOR_E5_LOAD_PROTOCOL_SHA256}:"
            f"{paired_trace_sha256}:{block}:{label}"
        ).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _paired_trace_sha256(
    *, family: E5Family, block: int, dimensions: Mapping[str, object]
) -> str:
    """Drop method/recipe lineage while retaining every load-family axis."""

    family_axes = {
        "backend_authority",
        "family_id",
        "topology",
        "concurrency",
        "load_factor",
        "arrival",
        "cohort_count",
        "cohort_distribution",
    }
    scientific = {
        name: dimensions[name] for name in sorted(family_axes & set(dimensions))
    }
    return _sha256(
        {
            "schema_version": 1,
            "kind": "formal_single_operator_e5_paired_trace_identity",
            "protocol_sha256": FORMAL_SINGLE_OPERATOR_E5_LOAD_PROTOCOL_SHA256,
            "family": family,
            "block": block,
            "scientific_axes": scientific,
        }
    )


def derive_e5_arrival_plan(
    *,
    cell_id: str,
    block: int,
    family: E5Family,
    dimensions: Mapping[str, object],
    lambda_star: E3aLambdaStar,
    burstgpt_verification: BurstGptV2ReleaseVerification | None = None,
    burstgpt_active_asset_path: str | Path | None = None,
    selected_p99_anchor: bool = False,
) -> E5ArrivalPlan:
    """Derive exact paired arrivals for one materialized E5 headline cell."""

    _require_sha256("E5 arrival cell", cell_id)
    if type(block) is not int or not 0 <= block < E5_MAX_BLOCKS:
        raise ValueError("E5 arrival block is outside the registered prefix")
    if type(lambda_star) is not E3aLambdaStar:
        raise TypeError("E5 arrival derivation requires the sealed E3a lambda-star")
    base_rate = lambda_star.requests_per_second
    paired_trace_sha256 = _paired_trace_sha256(
        family=family,
        block=block,
        dimensions=dimensions,
    )
    duration = E5_HEADLINE_ARRIVAL_DURATION_US
    burst_window = None
    effective_rate: Fraction | None = None
    extension_count = E5_P99_EXTENSION_OFFERED_REQUESTS if selected_p99_anchor else None

    if family == "closed_loop":
        concurrency = _require_positive_int(
            "E5 closed-loop concurrency", dimensions.get("concurrency")
        )
        count = (
            extension_count
            if extension_count is not None
            else max(
                4 * concurrency,
                _ceil_fraction(base_rate * 2 * Fraction(duration, 1_000_000)),
            )
        )
        arrivals = (0,) * count
        arrival_policy = "closed_loop"
    elif family == "topology_cohort":
        concurrency = lambda_star.common_load
        count = (
            extension_count
            if extension_count is not None
            else max(
                4 * concurrency,
                _ceil_fraction(base_rate * 2 * Fraction(duration, 1_000_000)),
            )
        )
        arrivals = (0,) * count
        arrival_policy = "closed_loop"
    elif family == "open_loop":
        raw_factor = dimensions.get("load_factor")
        if type(raw_factor) not in {int, float} or not math.isfinite(float(raw_factor)):
            raise ValueError("E5 open-loop load factor differs")
        factor = Fraction(str(raw_factor))
        if factor not in {
            Fraction(1, 4),
            Fraction(1, 2),
            Fraction(3, 4),
            Fraction(9, 10),
            Fraction(1, 1),
            Fraction(11, 10),
            Fraction(5, 4),
        }:
            raise ValueError("E5 open-loop load factor is not registered")
        effective_rate = base_rate * factor
        upper = (
            extension_count
            if extension_count is not None
            else min(
                E5_MAX_REQUEST_ROWS,
                _ceil_fraction(
                    effective_rate * Fraction(duration, 1_000_000) * Fraction(5, 4)
                )
                + 256,
            )
        )
        generated = _poisson_arrivals(
            request_count=upper,
            rate_per_second=effective_rate,
            seed=_seed(
                paired_trace_sha256=paired_trace_sha256,
                block=block,
                label="open_loop",
            ),
        )
        if extension_count is not None:
            arrivals = generated
            duration = max(duration, arrivals[-1] + 1)
        else:
            arrivals = tuple(value for value in generated if value < duration)
            if not arrivals or len(arrivals) == upper:
                raise ValueError("E5 open-loop request pool does not seal the window")
        concurrency = lambda_star.common_load
        arrival_policy = "poisson"
    elif family == "trace_or_soak":
        raw_arrival = dimensions.get("arrival")
        if type(raw_arrival) is not str:
            raise ValueError("E5 trace/soak arrival label differs")
        arrival_policy = raw_arrival
        concurrency = lambda_star.common_load
        if raw_arrival == "immediate_burst":
            count = (
                extension_count
                if extension_count is not None
                else max(
                    2,
                    _ceil_fraction(base_rate * Fraction(duration, 1_000_000)),
                )
            )
            arrivals = (0,) * count
        elif raw_arrival == "burstgpt_shape":
            if (
                type(burstgpt_verification) is not BurstGptV2ReleaseVerification
                or burstgpt_active_asset_path is None
            ):
                raise ValueError(
                    "E5 BurstGPT arrival requires its release verification"
                )
            count = (
                extension_count
                if extension_count is not None
                else max(
                    2,
                    _ceil_fraction(base_rate * Fraction(duration, 1_000_000)),
                )
            )
            burst_window = select_burstgpt_arrival_window(
                active_asset_path=burstgpt_active_asset_path,
                verification=burstgpt_verification,
                block=block,
                request_count=count,
                target_rate=base_rate,
            )
            arrivals = burst_window.scaled_arrivals_us
            # Rounding can place the last event exactly on the half-open edge.
            duration = max(E5_HEADLINE_ARRIVAL_DURATION_US, arrivals[-1] + 1)
            effective_rate = base_rate
        elif raw_arrival in E5_SOAK_LOAD_FACTORS:
            duration = E5_SOAK_ARRIVAL_DURATION_US
            effective_rate = base_rate * E5_SOAK_LOAD_FACTORS[raw_arrival]
            upper = (
                extension_count
                if extension_count is not None
                else min(
                    E5_MAX_REQUEST_ROWS,
                    _ceil_fraction(
                        effective_rate * Fraction(duration, 1_000_000) * Fraction(5, 4)
                    )
                    + 256,
                )
            )
            generated = _poisson_arrivals(
                request_count=upper,
                rate_per_second=effective_rate,
                seed=_seed(
                    paired_trace_sha256=paired_trace_sha256,
                    block=block,
                    label=raw_arrival,
                ),
            )
            if extension_count is not None:
                arrivals = generated
                duration = max(duration, arrivals[-1] + 1)
            else:
                arrivals = tuple(value for value in generated if value < duration)
                if not arrivals or len(arrivals) == upper:
                    raise ValueError("E5 soak request pool does not seal the window")
        else:
            raise ValueError("E5 trace/soak arrival is not registered")
    else:
        raise ValueError("E5 arrival family is not registered")

    if len(arrivals) > E5_MAX_REQUEST_ROWS:
        raise ValueError("E5 arrival plan exceeds the registered request-row ceiling")
    if extension_count is not None and len(arrivals) != extension_count:
        raise RuntimeError("E5 selected p99 family did not receive its exact extension")
    return E5ArrivalPlan(
        schema_version=1,
        kind="formal_single_operator_e5_arrival_plan",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_E5_LOAD_PROTOCOL_SHA256,
        cell_id=cell_id,
        paired_trace_sha256=paired_trace_sha256,
        block=block,
        family=family,
        arrival_policy=arrival_policy,
        lambda_star=lambda_star,
        effective_rate_numerator=(
            None if effective_rate is None else effective_rate.numerator
        ),
        effective_rate_denominator=(
            None if effective_rate is None else effective_rate.denominator
        ),
        concurrency=concurrency,
        arrival_duration_us=duration,
        warmup_duration_us=E5_WARMUP_DURATION_US,
        request_deadline_us=E5_REQUEST_DEADLINE_US,
        drain_duration_us=E5_DRAIN_DURATION_US,
        arrivals_us=arrivals,
        burstgpt_window=burst_window,
        p99_extension_minimum_completed=(
            E5_P99_MINIMUM_COMPLETED_REQUESTS if selected_p99_anchor else None
        ),
        p99_extension_offered_requests=(
            E5_P99_EXTENSION_OFFERED_REQUESTS if selected_p99_anchor else None
        ),
    )


__all__ = [
    "BURSTGPT_V2_ACTIVE_ASSET",
    "BURSTGPT_V2_ASSETS",
    "BURSTGPT_V2_RELEASE_ID",
    "BURSTGPT_V2_RELEASE_TAG",
    "BURSTGPT_V2_RELEASE_TAG_COMMIT",
    "BURSTGPT_V2_RELEASE_URL",
    "E5_P99_EXTENSION_OFFERED_REQUESTS",
    "E5_P99_MINIMUM_COMPLETED_REQUESTS",
    "FORMAL_SINGLE_OPERATOR_E5_LOAD_PROTOCOL_SHA256",
    "BurstGptArrivalWindow",
    "BurstGptV2ReleaseVerification",
    "E3aLambdaStar",
    "E5ArrivalPlan",
    "derive_e5_arrival_plan",
    "select_burstgpt_arrival_window",
    "verify_burstgpt_v2_release",
]
