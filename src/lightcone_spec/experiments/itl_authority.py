"""Formal raw authority for within-request inter-token latency timestamps.

The official serving adapter may receive several generated token IDs in one
SSE chunk.  A chunk arrival time is not a timestamp for every token in that
chunk, and dividing its gap by the token count would fabricate a latency
distribution.  Formal ITL evidence therefore has exactly two future source
modes: a native per-token timestamp hook, or raw SSE frame observations that
prove every frame contributes exactly one new token.

The pinned runtime exposes a CPU-only committed-token observation hook for
contract testing.  Its timestamps are host observations made while enumerating
tokens that are already committed; they are neither decode-production nor CUDA
event times and cannot support a formal p99 claim.  This release therefore
keeps the formal producer allowlist empty.  E2 prelaunch is named ``BLOCKED``
before a raw receipt path is opened.  The frozen reducer contract below exists
so a future GPU-validated first-party producer can be added without weakening
that gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Literal

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments.load import TokenChunkTiming
from lightcone_spec.experiments.registry import (
    ExperimentCell,
    ExperimentRegistry,
    WorkloadClass,
    content_sha256,
)

ItlTimestampSourceMode = Literal[
    "native_per_token_timestamp_hook",
    "sse_one_token_per_frame",
]

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
_MAX_RAW_RECEIPT_BYTES = 64 * 1024 * 1024

ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON = (
    "release_per_token_timestamp_producer_unavailable"
)
ITL_CPU_CONTRACT_ONLY_REASON = "cpu_contract_only_not_formal_itl_authority"
SGLANG_CPU_ITL_CONTRACT_HOOK = "sglang.schema_v3.native_per_token_timestamp.v1"
SGLANG_CPU_ITL_CONTRACT_SEMANTICS = "cpu_committed_token_observed_at_streamer_v1"
SGLANG_CPU_ITL_CONTRACT_RELEASE_STATUS = "CPU_CONTRACT_ONLY"
ITL_COALESCED_CHUNK_UNPROVEN_REASON = "coalesced_sse_chunk_has_no_token_timestamps"
ITL_RAW_RECEIPT_MISSING_REASON = "per_token_timestamp_raw_receipt_missing"
ITL_RAW_REQUEST_COVERAGE_INCOMPLETE_REASON = (
    "per_token_timestamp_request_coverage_incomplete"
)

ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_e2_per_token_itl_timestamp_authority",
        "sources": [
            "native_per_token_timestamp_hook",
            "sse_one_token_per_frame",
        ],
        "native_contract": (
            "ordered_token_id_and_monotonic_ns_for_every_generated_token"
        ),
        "sse_contract": "raw_frame_exactly_one_new_token_and_monotonic_ns",
        "request_binding": (
            "request_id_ordered_output_token_ids_start_terminal_and_full_coverage"
        ),
        "coalesced_without_native_timestamps": "BLOCKED",
        "chunk_gap_interpolation": "forbidden",
        "missing": "BLOCKED_and_None_never_zero",
    }
)


class ItlTimestampAuthorityBlocked(RuntimeError):
    """Raised before launch or promotion when formal ITL timing is unavailable."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"formal ITL timestamp authority is BLOCKED: {reason}")
        self.reason = reason


def reject_cpu_contract_only_itl_metadata(metadata: Mapping[str, object]) -> None:
    """Reject CPU observation metadata before any formal raw receipt is opened."""

    value = _strict_mapping("CPU ITL metadata", metadata)
    if any(
        value.get(field) == expected
        for field, expected in (
            ("native_token_timestamp_hook", SGLANG_CPU_ITL_CONTRACT_HOOK),
            (
                "native_token_timestamp_semantics",
                SGLANG_CPU_ITL_CONTRACT_SEMANTICS,
            ),
            (
                "native_token_timestamp_release_status",
                SGLANG_CPU_ITL_CONTRACT_RELEASE_STATUS,
            ),
        )
    ):
        raise ItlTimestampAuthorityBlocked(ITL_CPU_CONTRACT_ONLY_REASON)


def _require_sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_safe_id(label: str, value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe identifier")
    return value


def _strict_mapping(label: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    return value


def _strict_sequence(label: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    return value


def _strict_int(label: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_keys(
    label: str,
    value: Mapping[str, object],
    expected: set[str],
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


@dataclass(frozen=True)
class ReleaseItlTimestampProducer:
    producer_id: str
    source_mode: ItlTimestampSourceMode
    hook_id: str
    producer_version_sha256: str
    patched_sglang_tree: str
    clock: str
    protocol_sha256: str

    def __post_init__(self) -> None:
        _require_safe_id("ITL producer", self.producer_id)
        _require_safe_id("ITL producer hook", self.hook_id)
        _require_sha256("ITL producer version", self.producer_version_sha256)
        if self.source_mode not in {
            "native_per_token_timestamp_hook",
            "sse_one_token_per_frame",
        }:
            raise ValueError("ITL timestamp source mode is unsupported")
        expected_hook = {
            "native_per_token_timestamp_hook": (
                "sglang.schema_v3.native_per_token_timestamp.v1"
            ),
            "sse_one_token_per_frame": (
                "sglang.benchmark.serving.raw_sse_frame_observation.v1"
            ),
        }[self.source_mode]
        if self.hook_id != expected_hook:
            raise ValueError("ITL timestamp hook differs from its source mode")
        if self.patched_sglang_tree != PINNED_SGLANG_TREE:
            raise ValueError("ITL producer uses another patched SGLang tree")
        if self.clock != "monotonic_ns":
            raise ValueError("ITL producer must use the monotonic nanosecond clock")
        if self.protocol_sha256 != ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256:
            raise ValueError("ITL producer uses another authority protocol")

    def to_dict(self) -> dict[str, str]:
        return {
            "producer_id": self.producer_id,
            "source_mode": self.source_mode,
            "hook_id": self.hook_id,
            "producer_version_sha256": self.producer_version_sha256,
            "patched_sglang_tree": self.patched_sglang_tree,
            "clock": self.clock,
            "protocol_sha256": self.protocol_sha256,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


# Source-owned allowlist.  A future row requires the actual first-party hook and
# GPU-marked tests.  Caller probes and self-described capability JSON cannot add
# a producer.
RELEASE_ITL_TIMESTAMP_PRODUCERS: tuple[ReleaseItlTimestampProducer, ...] = ()


@dataclass(frozen=True)
class E2ItlTimestampPlan:
    schema_version: int
    kind: str
    registry_sha256: str
    cell_id: str
    cell_declaration_sha256: str
    patched_sglang_tree: str
    producer: ReleaseItlTimestampProducer | None
    interpolation_forbidden: bool
    full_request_coverage_required: bool
    protocol_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("E2 ITL timestamp plan schema is unsupported")
        if self.kind != "e2_itl_timestamp_plan":
            raise ValueError("E2 ITL timestamp plan schema is unsupported")
        for label, value in (
            ("ITL registry", self.registry_sha256),
            ("ITL cell", self.cell_id),
            ("ITL cell declaration", self.cell_declaration_sha256),
            ("ITL protocol", self.protocol_sha256),
        ):
            _require_sha256(label, value)
        if self.patched_sglang_tree != PINNED_SGLANG_TREE:
            raise ValueError("ITL plan uses another patched SGLang tree")
        if self.producer is not None:
            if type(self.producer) is not ReleaseItlTimestampProducer:
                raise TypeError("ITL plan requires an exact release producer")
            self.producer.__post_init__()
        if self.interpolation_forbidden is not True:
            raise ValueError("ITL plan cannot authorize chunk-gap interpolation")
        if self.full_request_coverage_required is not True:
            raise ValueError("ITL plan must require full request coverage")
        if self.protocol_sha256 != ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256:
            raise ValueError("ITL plan uses another authority protocol")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "registry_sha256": self.registry_sha256,
            "cell_id": self.cell_id,
            "cell_declaration_sha256": self.cell_declaration_sha256,
            "patched_sglang_tree": self.patched_sglang_tree,
            "producer": None if self.producer is None else self.producer.to_dict(),
            "interpolation_forbidden": self.interpolation_forbidden,
            "full_request_coverage_required": self.full_request_coverage_required,
            "protocol_sha256": self.protocol_sha256,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class ItlTimestampActivation:
    status: Literal["READY", "BLOCKED"]
    reason_code: str | None
    plan_sha256: str
    producer_sha256: str | None

    def __post_init__(self) -> None:
        if self.status not in {"READY", "BLOCKED"}:
            raise ValueError("ITL activation status is unsupported")
        _require_sha256("ITL activation plan", self.plan_sha256)
        if self.status == "BLOCKED":
            if not self.reason_code or self.producer_sha256 is not None:
                raise ValueError("blocked ITL activation must retain only its reason")
        else:
            if self.reason_code is not None or self.producer_sha256 is None:
                raise ValueError("ready ITL activation requires one producer")
            _require_sha256("ITL activation producer", self.producer_sha256)


def release_e2_itl_timestamp_plan(
    registry: ExperimentRegistry,
    cell: ExperimentCell,
) -> E2ItlTimestampPlan:
    """Derive an exact E2 timing plan without launching a serving process."""

    if type(registry) is not ExperimentRegistry or type(cell) is not ExperimentCell:
        raise TypeError("ITL planning requires exact registry and cell objects")
    matches = tuple(
        row for row in registry.cells_for("E2") if row.cell_id == cell.cell_id
    )
    if len(matches) != 1 or matches[0] != cell:
        raise ValueError("ITL cell is foreign to the E2 registry")
    if (
        cell.identity.experiment != "E2"
        or cell.identity.task != "LiveCodeBench_tuning"
        or cell.resources.workload_class is not WorkloadClass.TUNING
    ):
        raise ValueError("ITL plan requires an exact E2 tuning cell")
    if len(RELEASE_ITL_TIMESTAMP_PRODUCERS) > 1:
        raise RuntimeError("release ITL timestamp producer allowlist is ambiguous")
    producer = (
        RELEASE_ITL_TIMESTAMP_PRODUCERS[0] if RELEASE_ITL_TIMESTAMP_PRODUCERS else None
    )
    if producer is not None:
        producer.__post_init__()
    return E2ItlTimestampPlan(
        schema_version=1,
        kind="e2_itl_timestamp_plan",
        registry_sha256=registry.sha256,
        cell_id=cell.cell_id,
        cell_declaration_sha256=cell.sha256,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        producer=producer,
        interpolation_forbidden=True,
        full_request_coverage_required=True,
        protocol_sha256=ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256,
    )


def evaluate_e2_itl_timestamp_activation(
    plan: E2ItlTimestampPlan,
) -> ItlTimestampActivation:
    if type(plan) is not E2ItlTimestampPlan:
        raise TypeError("ITL activation requires an exact E2 plan")
    plan.__post_init__()
    if plan.producer is None:
        return ItlTimestampActivation(
            status="BLOCKED",
            reason_code=ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON,
            plan_sha256=plan.sha256,
            producer_sha256=None,
        )
    return ItlTimestampActivation(
        status="READY",
        reason_code=None,
        plan_sha256=plan.sha256,
        producer_sha256=plan.producer.sha256,
    )


def require_e2_itl_timestamp_prelaunch(
    plan: E2ItlTimestampPlan,
) -> ReleaseItlTimestampProducer:
    activation = evaluate_e2_itl_timestamp_activation(plan)
    if activation.status != "READY" or plan.producer is None:
        raise ItlTimestampAuthorityBlocked(
            activation.reason_code or ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON
        )
    return plan.producer


@dataclass(frozen=True)
class ItlRequestTimestamps:
    request_id: str
    request_started_ns: int
    request_terminal_ns: int
    output_token_ids: tuple[int, ...]
    token_observed_ns: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_safe_id("ITL request", self.request_id)
        _strict_int("ITL request start", self.request_started_ns)
        _strict_int("ITL request terminal", self.request_terminal_ns)
        if self.request_terminal_ns < self.request_started_ns:
            raise ValueError("ITL request terminal precedes its start")
        if len(self.output_token_ids) < 2:
            raise ValueError("formal ITL requires at least two generated tokens")
        if len(self.token_observed_ns) != len(self.output_token_ids):
            raise ValueError("ITL token timestamp coverage is incomplete")
        for token_id in self.output_token_ids:
            _strict_int("ITL output token ID", token_id)
        for observed_ns in self.token_observed_ns:
            _strict_int("ITL token timestamp", observed_ns)
        if any(
            current <= previous
            for previous, current in pairwise(self.token_observed_ns)
        ):
            raise ValueError("ITL token timestamps must be strictly increasing")
        if (
            self.token_observed_ns[0] < self.request_started_ns
            or self.token_observed_ns[-1] > self.request_terminal_ns
        ):
            raise ValueError("ITL token timestamps are outside the request lifetime")

    @property
    def inter_token_ns(self) -> tuple[int, ...]:
        return tuple(
            current - previous for previous, current in pairwise(self.token_observed_ns)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "request_started_ns": self.request_started_ns,
            "request_terminal_ns": self.request_terminal_ns,
            "output_token_ids": list(self.output_token_ids),
            "token_observed_ns": list(self.token_observed_ns),
        }


@dataclass(frozen=True)
class ItlRequestExpectation:
    """Terminal request identity derived outside the timestamp producer."""

    request_id: str
    request_started_ns: int
    request_terminal_ns: int
    output_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_safe_id("ITL expected request", self.request_id)
        _strict_int("ITL expected request start", self.request_started_ns)
        _strict_int("ITL expected request terminal", self.request_terminal_ns)
        if self.request_terminal_ns < self.request_started_ns:
            raise ValueError("ITL expected request terminal precedes its start")
        if len(self.output_token_ids) < 2:
            raise ValueError("formal ITL expectation requires at least two tokens")
        for token_id in self.output_token_ids:
            _strict_int("ITL expected output token ID", token_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "request_started_ns": self.request_started_ns,
            "request_terminal_ns": self.request_terminal_ns,
            "output_token_ids": list(self.output_token_ids),
        }


def itl_request_expectations_sha256(
    expectations: Sequence[ItlRequestExpectation],
) -> str:
    values = tuple(expectations)
    if (
        not values
        or any(type(value) is not ItlRequestExpectation for value in values)
        or tuple(value.request_id for value in values)
        != tuple(sorted({value.request_id for value in values}))
    ):
        raise ValueError("ITL expectations must be sorted, non-empty, and unique")
    for value in values:
        value.__post_init__()
    return content_sha256([value.to_dict() for value in values])


@dataclass(frozen=True)
class BoundItlTimestampAuthority:
    schema_version: int
    kind: str
    plan: E2ItlTimestampPlan
    raw_receipt_path: str
    raw_receipt_sha256: str
    producer_sha256: str
    expectations_sha256: str
    expectations: tuple[ItlRequestExpectation, ...]
    requests: tuple[ItlRequestTimestamps, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("bound ITL timestamp authority schema is unsupported")
        if self.kind != "bound_itl_timestamp_authority":
            raise ValueError("bound ITL timestamp authority schema is unsupported")
        if type(self.plan) is not E2ItlTimestampPlan:
            raise TypeError("bound ITL authority requires an exact E2 plan")
        producer = require_e2_itl_timestamp_prelaunch(self.plan)
        path = Path(self.raw_receipt_path)
        if not path.is_absolute() or path != path.resolve():
            raise ValueError("ITL raw receipt path must be absolute and resolved")
        _require_sha256("ITL raw receipt", self.raw_receipt_sha256)
        _require_sha256("ITL bound producer", self.producer_sha256)
        _require_sha256("ITL expectations", self.expectations_sha256)
        if self.producer_sha256 != producer.sha256:
            raise ValueError("ITL bound producer differs from the release plan")
        if self.expectations_sha256 != itl_request_expectations_sha256(
            self.expectations
        ):
            raise ValueError("ITL expected request binding changed")
        if not self.requests or tuple(row.request_id for row in self.requests) != tuple(
            sorted({row.request_id for row in self.requests})
        ):
            raise ValueError(
                "ITL request receipts must be sorted, non-empty, and unique"
            )
        for row in self.requests:
            row.__post_init__()
        if tuple(row.request_id for row in self.requests) != tuple(
            row.request_id for row in self.expectations
        ):
            raise ValueError("ITL raw receipt omits an expected request")
        for expected, observed in zip(self.expectations, self.requests, strict=True):
            if (
                expected.request_started_ns != observed.request_started_ns
                or expected.request_terminal_ns != observed.request_terminal_ns
                or expected.output_token_ids != observed.output_token_ids
            ):
                raise ValueError("ITL raw request differs from terminal expectations")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "plan": self.plan.to_dict(),
            "raw_receipt_path": self.raw_receipt_path,
            "raw_receipt_sha256": self.raw_receipt_sha256,
            "producer_sha256": self.producer_sha256,
            "expectations_sha256": self.expectations_sha256,
            "expectations": [value.to_dict() for value in self.expectations],
            "requests": [row.to_dict() for row in self.requests],
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def token_timestamps_for(self, request_id: str) -> tuple[int, ...]:
        matches = tuple(
            row.token_observed_ns
            for row in self.requests
            if row.request_id == request_id
        )
        if len(matches) != 1:
            raise KeyError(request_id)
        return matches[0]


def _read_stable_receipt(path_value: str | Path) -> tuple[Path, bytes]:
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError("ITL raw receipt path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ItlTimestampAuthorityBlocked(ITL_RAW_RECEIPT_MISSING_REASON) from error
    if resolved != path:
        raise ValueError("ITL raw receipt path must be resolved and non-symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("ITL raw receipt cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_RAW_RECEIPT_BYTES
        ):
            raise ValueError("ITL raw receipt must be a bounded regular file")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("ITL raw receipt changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("ITL raw receipt grew while being read")
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        identity = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or identity(before) != identity(after)
            or identity(after) != identity(current)
        ):
            raise ValueError("ITL raw receipt changed during coordinated read")
    finally:
        os.close(descriptor)
    return path, b"".join(chunks)


def _load_strict_json(raw: bytes) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"ITL raw receipt contains duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"ITL raw receipt contains non-finite {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ITL raw receipt is not strict UTF-8 JSON") from error
    return _strict_mapping("ITL raw receipt", value)


def _native_request_timing(row: Mapping[str, object]) -> ItlRequestTimestamps:
    _strict_keys(
        "native ITL request",
        row,
        {
            "request_id",
            "request_started_ns",
            "request_terminal_ns",
            "output_token_ids",
            "token_events",
        },
    )
    output_values = _strict_sequence("ITL output token IDs", row["output_token_ids"])
    output_ids = tuple(
        _strict_int("ITL output token ID", value) for value in output_values
    )
    events = _strict_sequence("native ITL token events", row["token_events"])
    observed: list[int] = []
    for index, event_value in enumerate(events):
        event = _strict_mapping("native ITL token event", event_value)
        _strict_keys(
            "native ITL token event",
            event,
            {"token_index", "token_id", "observed_ns"},
        )
        if (
            _strict_int("native ITL token index", event["token_index"]) != index
            or index >= len(output_ids)
            or _strict_int("native ITL token ID", event["token_id"])
            != output_ids[index]
        ):
            raise ValueError("native ITL events differ from ordered output tokens")
        observed.append(_strict_int("native ITL observation", event["observed_ns"]))
    return ItlRequestTimestamps(
        request_id=_require_safe_id("ITL request", row["request_id"]),
        request_started_ns=_strict_int("ITL request start", row["request_started_ns"]),
        request_terminal_ns=_strict_int(
            "ITL request terminal", row["request_terminal_ns"]
        ),
        output_token_ids=output_ids,
        token_observed_ns=tuple(observed),
    )


def _sse_request_timing(row: Mapping[str, object]) -> ItlRequestTimestamps:
    _strict_keys(
        "SSE ITL request",
        row,
        {
            "request_id",
            "request_started_ns",
            "request_terminal_ns",
            "output_token_ids",
            "raw_sse_frames",
        },
    )
    output_values = _strict_sequence("ITL output token IDs", row["output_token_ids"])
    output_ids = tuple(
        _strict_int("ITL output token ID", value) for value in output_values
    )
    frames = _strict_sequence("raw SSE frames", row["raw_sse_frames"])
    observed: list[int] = []
    for index, frame_value in enumerate(frames):
        frame = _strict_mapping("raw SSE frame", frame_value)
        _strict_keys(
            "raw SSE frame",
            frame,
            {"frame_index", "new_token_ids", "observed_ns"},
        )
        new_tokens = _strict_sequence("raw SSE new token IDs", frame["new_token_ids"])
        if len(new_tokens) != 1:
            raise ItlTimestampAuthorityBlocked(ITL_COALESCED_CHUNK_UNPROVEN_REASON)
        token_id = _strict_int("raw SSE token ID", new_tokens[0])
        if (
            _strict_int("raw SSE frame index", frame["frame_index"]) != index
            or index >= len(output_ids)
            or token_id != output_ids[index]
        ):
            raise ValueError("raw SSE frames differ from ordered output tokens")
        observed.append(_strict_int("raw SSE observation", frame["observed_ns"]))
    return ItlRequestTimestamps(
        request_id=_require_safe_id("ITL request", row["request_id"]),
        request_started_ns=_strict_int("ITL request start", row["request_started_ns"]),
        request_terminal_ns=_strict_int(
            "ITL request terminal", row["request_terminal_ns"]
        ),
        output_token_ids=output_ids,
        token_observed_ns=tuple(observed),
    )


def bind_itl_timestamp_authority(
    plan: E2ItlTimestampPlan,
    raw_receipt_path: str | Path,
    *,
    expected_requests: tuple[ItlRequestExpectation, ...],
) -> BoundItlTimestampAuthority:
    """Replay a first-party raw receipt without interpolating chunk gaps."""

    producer = require_e2_itl_timestamp_prelaunch(plan)
    expectations_sha256 = itl_request_expectations_sha256(expected_requests)
    path, raw = _read_stable_receipt(raw_receipt_path)
    receipt = _load_strict_json(raw)
    _strict_keys(
        "ITL raw receipt",
        receipt,
        {
            "schema_version",
            "kind",
            "plan_sha256",
            "producer_id",
            "producer_version_sha256",
            "source_mode",
            "hook_id",
            "clock",
            "complete",
            "requests",
        },
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["kind"] != "formal_itl_timestamp_raw_receipt"
        or receipt["plan_sha256"] != plan.sha256
        or receipt["producer_id"] != producer.producer_id
        or receipt["producer_version_sha256"] != producer.producer_version_sha256
        or receipt["source_mode"] != producer.source_mode
        or receipt["hook_id"] != producer.hook_id
        or receipt["clock"] != producer.clock
        or receipt["complete"] is not True
    ):
        raise ValueError("ITL raw receipt differs from its release plan")
    request_values = _strict_sequence("ITL raw requests", receipt["requests"])
    if not request_values:
        raise ItlTimestampAuthorityBlocked(ITL_RAW_REQUEST_COVERAGE_INCOMPLETE_REASON)
    reducer = (
        _native_request_timing
        if producer.source_mode == "native_per_token_timestamp_hook"
        else _sse_request_timing
    )
    requests = tuple(
        sorted(
            (
                reducer(_strict_mapping("ITL raw request", value))
                for value in request_values
            ),
            key=lambda value: value.request_id,
        )
    )
    return BoundItlTimestampAuthority(
        schema_version=1,
        kind="bound_itl_timestamp_authority",
        plan=plan,
        raw_receipt_path=str(path),
        raw_receipt_sha256=hashlib.sha256(raw).hexdigest(),
        producer_sha256=producer.sha256,
        expectations_sha256=expectations_sha256,
        expectations=expected_requests,
        requests=requests,
    )


def revalidate_itl_timestamp_authority(
    authority: BoundItlTimestampAuthority,
) -> BoundItlTimestampAuthority:
    if type(authority) is not BoundItlTimestampAuthority:
        raise TypeError("ITL revalidation requires an exact bound authority")
    authority.__post_init__()
    rebound = bind_itl_timestamp_authority(
        authority.plan,
        authority.raw_receipt_path,
        expected_requests=authority.expectations,
    )
    if rebound != authority or rebound.sha256 != authority.sha256:
        raise ValueError("ITL timestamp authority changed during revalidation")
    return rebound


def assess_serving_chunks_for_formal_itl(
    *,
    request_id: str,
    output_tokens: int,
    chunks: Sequence[TokenChunkTiming],
) -> str | None:
    """Return the named blocker for aggregate serving chunks, never an ITL."""

    _require_safe_id("ITL request", request_id)
    _strict_int("ITL output tokens", output_tokens, minimum=1)
    covered = 0
    for chunk in chunks:
        chunk.validate()
        if chunk.request_id != request_id or chunk.first_token_index != covered:
            raise ValueError("serving chunks do not exactly cover the ITL request")
        if chunk.per_token_observed_at_us is None and chunk.token_count > 1:
            return ITL_COALESCED_CHUNK_UNPROVEN_REASON
        covered += chunk.token_count
    if covered != output_tokens:
        raise ValueError("serving chunks do not cover every generated token")
    # Exact-looking client chunks still need one allowlisted raw producer receipt.
    return ITL_RAW_RECEIPT_MISSING_REASON


__all__ = [
    "ITL_COALESCED_CHUNK_UNPROVEN_REASON",
    "ITL_CPU_CONTRACT_ONLY_REASON",
    "ITL_RAW_RECEIPT_MISSING_REASON",
    "ITL_RAW_REQUEST_COVERAGE_INCOMPLETE_REASON",
    "ITL_TIMESTAMP_AUTHORITY_PROTOCOL_SHA256",
    "ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON",
    "RELEASE_ITL_TIMESTAMP_PRODUCERS",
    "SGLANG_CPU_ITL_CONTRACT_HOOK",
    "SGLANG_CPU_ITL_CONTRACT_RELEASE_STATUS",
    "SGLANG_CPU_ITL_CONTRACT_SEMANTICS",
    "BoundItlTimestampAuthority",
    "E2ItlTimestampPlan",
    "ItlRequestExpectation",
    "ItlRequestTimestamps",
    "ItlTimestampActivation",
    "ItlTimestampAuthorityBlocked",
    "ReleaseItlTimestampProducer",
    "assess_serving_chunks_for_formal_itl",
    "bind_itl_timestamp_authority",
    "evaluate_e2_itl_timestamp_activation",
    "itl_request_expectations_sha256",
    "reject_cpu_contract_only_itl_metadata",
    "release_e2_itl_timestamp_plan",
    "require_e2_itl_timestamp_prelaunch",
    "revalidate_itl_timestamp_authority",
]
