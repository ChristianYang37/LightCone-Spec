"""Deterministic token-level context compilation for E3b and E6.

The registered context axis is a prompt-plus-generation budget.  A label is
not evidence: every compiled request must consume that budget exactly after
tokenization.  Core task tokens are kept byte-for-token intact and contiguous;
only separately bound filler rows may be prefix-sliced.  Filler rows are never
cycled within one request, so insufficient registered content blocks the cell.

This module deliberately performs no tokenization and no network access.  It
consumes rows emitted by the first-party tokenizer worker, binds the tokenizer
member/revision, and returns typed values for the sharded schedule publisher.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Literal, Self

ContextCompilerRegime = Literal[
    "long_input_short_output",
    "short_input_long_generation",
    "multi_turn_shared_prefix",
    "native_mtp_transfer",
]

_SHA256 = frozenset("0123456789abcdef")
_REGIMES = {
    "long_input_short_output",
    "short_input_long_generation",
    "multi_turn_shared_prefix",
    "native_mtp_transfer",
}
_MAX_CONTEXT_TOKENS = 40_928


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
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{label} must be canonical nonempty text")
    return value


def _strict(label: str, value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return dict(value)


FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_context_compiler_protocol",
        "scope": ["E3b", "E6"],
        "axis": "exact_prompt_plus_requested_generation_tokens",
        "core": "complete_contiguous_token_sequence_never_truncated",
        "filler": (
            "ordered_tokenized_path_bound_workload_rows_without_within_request_"
            "cycling_final_filler_span_may_be_prefix_sliced"
        ),
        "long_input_short_output": (
            "input_equals_context_minus_min_256_or_quarter_context"
        ),
        "short_input_long_generation": (
            "unaltered_core_at_most_quarter_context_generation_gets_remainder"
        ),
        "multi_turn_shared_prefix": (
            "common_filler_prefix_at_least_half_input_then_complete_core"
        ),
        "native_mtp_transfer": (
            "complete_task_core_with_filler_and_at_most_2048_generation_tokens"
        ),
        "identity": "tokenizer_member_model_revision_plus_ordered_source_rows",
        "lineage": (
            "runtime_BOUND_content_source_binding_and_registered_workload_members"
        ),
        "forbidden": [
            "label_only_context",
            "space_padding",
            "filler_repetition_within_request",
            "core_truncation",
            "caller_token_scalars",
        ],
    }
)


class ContextCompilationBlocked(RuntimeError):
    """The exact context cannot be formed from the registered token sources."""

    def __init__(self, reason_code: str) -> None:
        _require_text("context compilation reason", reason_code)
        self.reason_code = reason_code
        super().__init__(f"context compilation is BLOCKED: {reason_code}")


@dataclass(frozen=True)
class TokenizedContextSourceRow:
    tokenizer_content_member_id: str
    tokenizer_model_id: str
    tokenizer_revision: str
    source_member_sha256: str
    source_sample_id: str
    prompt_sha256: str
    input_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_sha256(
            "context source tokenizer member", self.tokenizer_content_member_id
        )
        _require_text("context source tokenizer model", self.tokenizer_model_id)
        _require_text("context source tokenizer revision", self.tokenizer_revision)
        _require_sha256("context source member", self.source_member_sha256)
        _require_text("context source sample", self.source_sample_id)
        _require_sha256("context source prompt", self.prompt_sha256)
        if (
            type(self.input_token_ids) is not tuple
            or not self.input_token_ids
            or any(
                type(token) is not int or token < 0 for token in self.input_token_ids
            )
        ):
            raise ValueError("context source token IDs are invalid")

    @cached_property
    def token_ids_sha256(self) -> str:
        return _sha256(list(self.input_token_ids))

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    @cached_property
    def source_key(self) -> tuple[str, str]:
        return self.source_member_sha256, self.source_sample_id

    def to_dict(self) -> dict[str, object]:
        return {
            "tokenizer_content_member_id": self.tokenizer_content_member_id,
            "tokenizer_model_id": self.tokenizer_model_id,
            "tokenizer_revision": self.tokenizer_revision,
            "source_member_sha256": self.source_member_sha256,
            "source_sample_id": self.source_sample_id,
            "prompt_sha256": self.prompt_sha256,
            "input_token_ids": list(self.input_token_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "tokenized context source",
            value,
            {field.name for field in fields(cls)},
        )
        raw_tokens = row.pop("input_token_ids")
        if type(raw_tokens) is not list:
            raise TypeError("tokenized context source tokens must be an array")
        return cls(**row, input_token_ids=tuple(raw_tokens))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ContextFillerAuthority:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_context_filler_authority"]
    protocol_sha256: str
    content_source_binding_sha256: str
    tokenizer_content_member_id: str
    tokenizer_model_id: str
    tokenizer_revision: str
    registered_source_member_sha256s: tuple[str, ...]
    rows: tuple[TokenizedContextSourceRow, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_context_filler_authority"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256
        ):
            raise ValueError("context filler authority schema differs")
        _require_sha256(
            "context filler tokenizer member", self.tokenizer_content_member_id
        )
        _require_sha256(
            "context filler content source", self.content_source_binding_sha256
        )
        _require_text("context filler tokenizer model", self.tokenizer_model_id)
        _require_text("context filler tokenizer revision", self.tokenizer_revision)
        if (
            type(self.registered_source_member_sha256s) is not tuple
            or not self.registered_source_member_sha256s
            or self.registered_source_member_sha256s
            != tuple(sorted(set(self.registered_source_member_sha256s)))
        ):
            raise ValueError("context filler registered source members differ")
        for member_id in self.registered_source_member_sha256s:
            _require_sha256("context filler registered source member", member_id)
        if (
            type(self.rows) is not tuple
            or not self.rows
            or any(type(row) is not TokenizedContextSourceRow for row in self.rows)
            or len({row.source_key for row in self.rows}) != len(self.rows)
            or any(
                row.tokenizer_content_member_id != self.tokenizer_content_member_id
                or row.tokenizer_model_id != self.tokenizer_model_id
                or row.tokenizer_revision != self.tokenizer_revision
                for row in self.rows
            )
            or {row.source_member_sha256 for row in self.rows}
            - set(self.registered_source_member_sha256s)
        ):
            raise ValueError("context filler authority rows differ")

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "content_source_binding_sha256": self.content_source_binding_sha256,
            "tokenizer_content_member_id": self.tokenizer_content_member_id,
            "tokenizer_model_id": self.tokenizer_model_id,
            "tokenizer_revision": self.tokenizer_revision,
            "registered_source_member_sha256s": list(
                self.registered_source_member_sha256s
            ),
            "rows": [row.to_dict() for row in self.rows],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "context filler authority",
            value,
            {field.name for field in fields(cls)},
        )
        raw_rows = row.pop("rows")
        raw_members = row.pop("registered_source_member_sha256s")
        if type(raw_rows) is not list or type(raw_members) is not list:
            raise TypeError("context filler authority arrays differ")
        return cls(
            **row,
            registered_source_member_sha256s=tuple(raw_members),
            rows=tuple(TokenizedContextSourceRow.from_dict(item) for item in raw_rows),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class ContextFillerSpan:
    source_row_sha256: str
    source_sample_id: str
    source_start: int
    source_stop: int
    compiled_start: int
    compiled_stop: int

    def __post_init__(self) -> None:
        _require_sha256("context filler span source", self.source_row_sha256)
        _require_text("context filler span sample", self.source_sample_id)
        if (
            any(
                type(value) is not int or value < 0
                for value in (
                    self.source_start,
                    self.source_stop,
                    self.compiled_start,
                    self.compiled_stop,
                )
            )
            or self.source_start != 0
            or self.source_start >= self.source_stop
            or self.compiled_start >= self.compiled_stop
            or self.source_stop - self.source_start
            != self.compiled_stop - self.compiled_start
        ):
            raise ValueError("context filler span offsets differ")

    def to_dict(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict(
                "context filler span",
                value,
                {field.name for field in fields(cls)},
            )
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class CompiledContextRequest:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_compiled_context_request"]
    protocol_sha256: str
    regime: ContextCompilerRegime
    context_tokens: int
    requested_output_tokens: int
    tokenizer_content_member_id: str
    tokenizer_model_id: str
    tokenizer_revision: str
    filler_authority_sha256: str
    core_source_row_sha256: str
    core_source_member_sha256: str
    core_source_sample_id: str
    core_prompt_sha256: str
    core_token_ids_sha256: str
    core_compiled_start: int
    shared_prefix_tokens: int
    filler_spans: tuple[ContextFillerSpan, ...]
    input_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_compiled_context_request"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256
            or self.regime not in _REGIMES
        ):
            raise ValueError("compiled context request schema differs")
        for label, digest in (
            ("filler authority", self.filler_authority_sha256),
            ("core source row", self.core_source_row_sha256),
            ("core source member", self.core_source_member_sha256),
            ("core prompt", self.core_prompt_sha256),
            ("core tokens", self.core_token_ids_sha256),
            ("tokenizer member", self.tokenizer_content_member_id),
        ):
            _require_sha256(f"compiled context {label}", digest)
        _require_text("compiled context core sample", self.core_source_sample_id)
        _require_text("compiled context tokenizer model", self.tokenizer_model_id)
        _require_text("compiled context tokenizer revision", self.tokenizer_revision)
        if (
            type(self.context_tokens) is not int
            or not 1 <= self.context_tokens <= _MAX_CONTEXT_TOKENS
            or type(self.requested_output_tokens) is not int
            or self.requested_output_tokens < 1
            or type(self.input_token_ids) is not tuple
            or not self.input_token_ids
            or any(
                type(token) is not int or token < 0 for token in self.input_token_ids
            )
            or len(self.input_token_ids) + self.requested_output_tokens
            != self.context_tokens
            or type(self.core_compiled_start) is not int
            or not 0 <= self.core_compiled_start < len(self.input_token_ids)
            or type(self.shared_prefix_tokens) is not int
            or not 0 <= self.shared_prefix_tokens <= len(self.input_token_ids)
            or any(type(span) is not ContextFillerSpan for span in self.filler_spans)
        ):
            raise ValueError("compiled context token budget differs")
        if (
            _sha256(list(self.input_token_ids[self.core_compiled_start :]))
            != self.core_token_ids_sha256
        ):
            raise ValueError("compiled context changed the complete core tokens")
        rebuilt_core = TokenizedContextSourceRow(
            tokenizer_content_member_id=self.tokenizer_content_member_id,
            tokenizer_model_id=self.tokenizer_model_id,
            tokenizer_revision=self.tokenizer_revision,
            source_member_sha256=self.core_source_member_sha256,
            source_sample_id=self.core_source_sample_id,
            prompt_sha256=self.core_prompt_sha256,
            input_token_ids=self.input_token_ids[self.core_compiled_start :],
        )
        if rebuilt_core.sha256 != self.core_source_row_sha256:
            raise ValueError("compiled context core source identity differs")
        if self.filler_spans:
            expected_start = 0
            for span in self.filler_spans:
                if span.compiled_start != expected_start:
                    raise ValueError("compiled context filler spans are not contiguous")
                expected_start = span.compiled_stop
            if expected_start != self.core_compiled_start:
                raise ValueError("compiled context filler spans do not end at the core")
        elif self.core_compiled_start:
            raise ValueError("compiled context is missing filler spans")
        if self.regime == "long_input_short_output":
            expected_output = max(1, min(256, self.context_tokens // 4))
            if (
                len(self.input_token_ids) < math.ceil(0.75 * self.context_tokens)
                or self.requested_output_tokens != expected_output
            ):
                raise ValueError("compiled long-input regime differs")
        if self.regime == "short_input_long_generation" and (
            self.filler_spans
            or self.core_compiled_start != 0
            or len(self.input_token_ids) > self.context_tokens // 4
            or self.requested_output_tokens < math.ceil(0.75 * self.context_tokens)
        ):
            raise ValueError("compiled short-input regime differs")
        if self.regime in {"multi_turn_shared_prefix", "native_mtp_transfer"}:
            expected_output = max(1, min(2_048, self.context_tokens // 4))
            if self.requested_output_tokens != expected_output:
                raise ValueError("compiled fixed transfer output budget differs")
        if self.regime == "multi_turn_shared_prefix" and (
            self.shared_prefix_tokens != math.ceil(len(self.input_token_ids) / 2)
        ):
            raise ValueError("compiled shared-prefix regime differs")
        if self.regime != "multi_turn_shared_prefix" and self.shared_prefix_tokens:
            raise ValueError("non-multi-turn context carries a shared prefix")

    @cached_property
    def input_token_ids_sha256(self) -> str:
        return _sha256(list(self.input_token_ids))

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "regime": self.regime,
            "context_tokens": self.context_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "tokenizer_content_member_id": self.tokenizer_content_member_id,
            "tokenizer_model_id": self.tokenizer_model_id,
            "tokenizer_revision": self.tokenizer_revision,
            "filler_authority_sha256": self.filler_authority_sha256,
            "core_source_row_sha256": self.core_source_row_sha256,
            "core_source_member_sha256": self.core_source_member_sha256,
            "core_source_sample_id": self.core_source_sample_id,
            "core_prompt_sha256": self.core_prompt_sha256,
            "core_token_ids_sha256": self.core_token_ids_sha256,
            "core_compiled_start": self.core_compiled_start,
            "shared_prefix_tokens": self.shared_prefix_tokens,
            "filler_spans": [span.to_dict() for span in self.filler_spans],
            "input_token_ids": list(self.input_token_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "compiled context request",
            value,
            {field.name for field in fields(cls)},
        )
        raw_spans = row.pop("filler_spans")
        raw_tokens = row.pop("input_token_ids")
        if type(raw_spans) is not list or type(raw_tokens) is not list:
            raise TypeError("compiled context arrays differ")
        return cls(
            **row,
            filler_spans=tuple(ContextFillerSpan.from_dict(item) for item in raw_spans),
            input_token_ids=tuple(raw_tokens),
        )  # type: ignore[arg-type]


def _consume_filler(
    *,
    rows: tuple[TokenizedContextSourceRow, ...],
    token_count: int,
    compiled_start: int,
) -> tuple[tuple[int, ...], tuple[ContextFillerSpan, ...], int]:
    if token_count == 0:
        return (), (), 0
    tokens: list[int] = []
    spans: list[ContextFillerSpan] = []
    consumed_rows = 0
    for row in rows:
        remaining = token_count - len(tokens)
        if remaining <= 0:
            break
        count = min(remaining, len(row.input_token_ids))
        start = compiled_start + len(tokens)
        tokens.extend(row.input_token_ids[:count])
        spans.append(
            ContextFillerSpan(
                source_row_sha256=row.sha256,
                source_sample_id=row.source_sample_id,
                source_start=0,
                source_stop=count,
                compiled_start=start,
                compiled_stop=start + count,
            )
        )
        consumed_rows += 1
    if len(tokens) != token_count:
        raise ContextCompilationBlocked("registered_filler_token_pool_insufficient")
    return tuple(tokens), tuple(spans), consumed_rows


def _rotated_rows(
    rows: tuple[TokenizedContextSourceRow, ...], *, identity: str
) -> tuple[TokenizedContextSourceRow, ...]:
    if not rows:
        return rows
    offset = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16)
    offset %= len(rows)
    return rows[offset:] + rows[:offset]


def compile_context_requests(
    *,
    regime: ContextCompilerRegime,
    context_tokens: int,
    core_rows: tuple[TokenizedContextSourceRow, ...],
    filler_authority: ContextFillerAuthority,
) -> tuple[CompiledContextRequest, ...]:
    """Compile one paired request pool without any caller-supplied token counts."""

    if regime not in _REGIMES:
        raise ValueError("context compiler regime is unsupported")
    if (
        type(context_tokens) is not int
        or not 1 <= context_tokens <= _MAX_CONTEXT_TOKENS
    ):
        raise ValueError("context compiler budget is unsupported")
    if (
        type(core_rows) is not tuple
        or not core_rows
        or any(type(row) is not TokenizedContextSourceRow for row in core_rows)
        or len({row.source_key for row in core_rows}) != len(core_rows)
        or type(filler_authority) is not ContextFillerAuthority
    ):
        raise TypeError("context compiler source rows differ")
    if any(
        row.tokenizer_content_member_id != filler_authority.tokenizer_content_member_id
        or row.tokenizer_model_id != filler_authority.tokenizer_model_id
        or row.tokenizer_revision != filler_authority.tokenizer_revision
        for row in core_rows
    ):
        raise ContextCompilationBlocked("core_and_filler_tokenizer_identity_differs")
    if {row.source_member_sha256 for row in core_rows} - set(
        filler_authority.registered_source_member_sha256s
    ):
        raise ContextCompilationBlocked("core_source_member_not_registered")
    core_keys = {row.source_key for row in core_rows}
    filler_rows = filler_authority.rows
    if not filler_rows and regime != "short_input_long_generation":
        raise ContextCompilationBlocked("registered_filler_rows_pool_empty")

    if regime == "long_input_short_output":
        output_tokens = max(1, min(256, context_tokens // 4))
    elif regime in {"multi_turn_shared_prefix", "native_mtp_transfer"}:
        output_tokens = max(1, min(2_048, context_tokens // 4))
    else:
        output_tokens = -1

    common_prefix: tuple[int, ...] = ()
    common_spans: tuple[ContextFillerSpan, ...] = ()
    suffix_pool = filler_rows
    fixed_input_tokens = context_tokens - output_tokens
    if regime == "multi_turn_shared_prefix":
        shared_filler_rows = tuple(
            row for row in filler_rows if row.source_key not in core_keys
        )
        if not shared_filler_rows:
            raise ContextCompilationBlocked(
                "registered_shared_filler_rows_disjoint_pool_empty"
            )
        prefix_count = math.ceil(fixed_input_tokens / 2)
        common_prefix, common_spans, consumed = _consume_filler(
            rows=shared_filler_rows,
            token_count=prefix_count,
            compiled_start=0,
        )
        suffix_pool = shared_filler_rows[consumed:]

    compiled: list[CompiledContextRequest] = []
    for core in core_rows:
        if regime == "short_input_long_generation":
            if len(core.input_token_ids) > context_tokens // 4:
                raise ContextCompilationBlocked(
                    "complete_core_exceeds_short_input_quarter_budget"
                )
            input_ids = core.input_token_ids
            requested_output = context_tokens - len(input_ids)
            spans: tuple[ContextFillerSpan, ...] = ()
            core_start = 0
            shared = 0
        else:
            requested_output = output_tokens
            filler_needed = (
                fixed_input_tokens - len(common_prefix) - len(core.input_token_ids)
            )
            if filler_needed < 0:
                raise ContextCompilationBlocked(
                    "complete_core_exceeds_registered_input_budget"
                )
            candidates = _rotated_rows(
                tuple(row for row in suffix_pool if row.source_key != core.source_key),
                identity=core.sha256,
            )
            unique_filler, unique_spans, _consumed = _consume_filler(
                rows=candidates,
                token_count=filler_needed,
                compiled_start=len(common_prefix),
            )
            core_start = len(common_prefix) + len(unique_filler)
            input_ids = common_prefix + unique_filler + core.input_token_ids
            spans = common_spans + unique_spans
            shared = len(common_prefix)
        if input_ids[core_start:] != core.input_token_ids:
            raise AssertionError("context compiler changed the complete core tokens")
        compiled.append(
            CompiledContextRequest(
                schema_version=1,
                kind="formal_single_operator_compiled_context_request",
                protocol_sha256=(
                    FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256
                ),
                regime=regime,
                context_tokens=context_tokens,
                requested_output_tokens=requested_output,
                tokenizer_content_member_id=(
                    filler_authority.tokenizer_content_member_id
                ),
                tokenizer_model_id=filler_authority.tokenizer_model_id,
                tokenizer_revision=filler_authority.tokenizer_revision,
                filler_authority_sha256=filler_authority.sha256,
                core_source_row_sha256=core.sha256,
                core_source_member_sha256=core.source_member_sha256,
                core_source_sample_id=core.source_sample_id,
                core_prompt_sha256=core.prompt_sha256,
                core_token_ids_sha256=core.token_ids_sha256,
                core_compiled_start=core_start,
                shared_prefix_tokens=shared,
                filler_spans=spans,
                input_token_ids=input_ids,
            )
        )
    result = tuple(compiled)
    if regime == "multi_turn_shared_prefix":
        prefix = result[0].input_token_ids[: result[0].shared_prefix_tokens]
        if any(
            row.input_token_ids[: row.shared_prefix_tokens] != prefix for row in result
        ):
            raise AssertionError("context compiler shared prefix changed")
    return result


__all__ = [
    "FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256",
    "CompiledContextRequest",
    "ContextCompilationBlocked",
    "ContextCompilerRegime",
    "ContextFillerAuthority",
    "ContextFillerSpan",
    "TokenizedContextSourceRow",
    "compile_context_requests",
]
