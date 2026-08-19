from __future__ import annotations

import hashlib

import pytest

from lightcone_spec.experiments.formal_single_operator_context_compiler import (
    FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256,
    CompiledContextRequest,
    ContextCompilationBlocked,
    ContextFillerAuthority,
    TokenizedContextSourceRow,
    compile_context_requests,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _row(
    label: str,
    count: int,
    *,
    tokenizer_member: str | None = None,
    source_member: str | None = None,
    token_offset: int = 0,
) -> TokenizedContextSourceRow:
    return TokenizedContextSourceRow(
        tokenizer_content_member_id=tokenizer_member or _sha("tokenizer-a"),
        tokenizer_model_id="Qwen/Qwen3-8B",
        tokenizer_revision="1" * 40,
        source_member_sha256=source_member or _sha("workload-source"),
        source_sample_id=label,
        prompt_sha256=_sha(f"prompt:{label}"),
        input_token_ids=tuple(token_offset + index for index in range(count)),
    )


def _authority(
    *,
    tokenizer_member: str | None = None,
    mutate_first_token: bool = False,
    row_count: int = 8,
) -> ContextFillerAuthority:
    member = tokenizer_member or _sha("tokenizer-a")
    rows = tuple(
        _row(
            f"filler-{index}",
            600,
            tokenizer_member=member,
            token_offset=10_000 + index * 1_000,
        )
        for index in range(row_count)
    )
    if mutate_first_token:
        rows = tuple(
            TokenizedContextSourceRow(
                **{
                    **row.to_dict(),
                    "input_token_ids": (row.input_token_ids[0] + 1,)
                    + row.input_token_ids[1:],
                }
            )
            for row in rows
        )
    return ContextFillerAuthority(
        schema_version=1,
        kind="formal_single_operator_context_filler_authority",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256,
        content_source_binding_sha256=_sha("content-source-binding"),
        tokenizer_content_member_id=member,
        tokenizer_model_id="Qwen/Qwen3-8B",
        tokenizer_revision="1" * 40,
        registered_source_member_sha256s=(_sha("workload-source"),),
        rows=rows,
    )


def test_long_input_exact_budget_preserves_complete_core_and_roundtrips() -> None:
    core = _row("core", 20, token_offset=50_000)
    result = compile_context_requests(
        regime="long_input_short_output",
        context_tokens=1_024,
        core_rows=(core,),
        filler_authority=_authority(),
    )[0]

    assert len(result.input_token_ids) == 768
    assert result.requested_output_tokens == 256
    assert len(result.input_token_ids) + result.requested_output_tokens == 1_024
    assert result.input_token_ids[result.core_compiled_start :] == core.input_token_ids
    assert result.core_token_ids_sha256 == core.token_ids_sha256
    assert result.filler_spans[0].compiled_start == 0
    assert result.filler_spans[-1].compiled_stop == result.core_compiled_start
    assert len({span.source_row_sha256 for span in result.filler_spans}) == len(
        result.filler_spans
    )
    assert CompiledContextRequest.from_dict(result.to_dict()) == result


def test_short_input_allocates_remainder_to_generation_without_filler() -> None:
    core = _row("core", 100, token_offset=50_000)
    result = compile_context_requests(
        regime="short_input_long_generation",
        context_tokens=1_024,
        core_rows=(core,),
        filler_authority=_authority(),
    )[0]

    assert result.input_token_ids == core.input_token_ids
    assert result.requested_output_tokens == 924
    assert result.filler_spans == ()
    assert result.core_compiled_start == 0


def test_shared_prefix_is_identical_and_at_least_half_each_input() -> None:
    cores = (
        _row("core-a", 30, token_offset=50_000),
        _row("core-b", 45, token_offset=60_000),
    )
    results = compile_context_requests(
        regime="multi_turn_shared_prefix",
        context_tokens=1_024,
        core_rows=cores,
        filler_authority=_authority(),
    )

    assert {len(row.input_token_ids) for row in results} == {768}
    assert {row.requested_output_tokens for row in results} == {256}
    assert results[0].shared_prefix_tokens == 384
    assert results[1].shared_prefix_tokens == 384
    assert results[0].input_token_ids[:384] == results[1].input_token_ids[:384]
    assert results[0].input_token_ids[results[0].core_compiled_start :] == (
        cores[0].input_token_ids
    )
    assert results[1].input_token_ids[results[1].core_compiled_start :] == (
        cores[1].input_token_ids
    )


def test_native_mtp_transfer_realizes_registered_context() -> None:
    core = _row("math-core", 300, token_offset=50_000)
    result = compile_context_requests(
        regime="native_mtp_transfer",
        context_tokens=4_096,
        core_rows=(core,),
        filler_authority=_authority(),
    )[0]

    assert result.requested_output_tokens == 1_024
    assert len(result.input_token_ids) == 3_072
    assert result.input_token_ids[result.core_compiled_start :] == core.input_token_ids


def test_maximum_registered_total_context_budget_is_exact() -> None:
    result = compile_context_requests(
        regime="long_input_short_output",
        context_tokens=40_928,
        core_rows=(_row("max-context-core", 300, token_offset=90_000),),
        filler_authority=_authority(row_count=70),
    )[0]

    assert len(result.input_token_ids) == 40_672
    assert result.requested_output_tokens == 256
    assert len(result.input_token_ids) + result.requested_output_tokens == 40_928


def test_insufficient_filler_and_oversize_core_fail_closed() -> None:
    short_authority = ContextFillerAuthority(
        schema_version=1,
        kind="formal_single_operator_context_filler_authority",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256,
        content_source_binding_sha256=_sha("content-source-binding"),
        tokenizer_content_member_id=_sha("tokenizer-a"),
        tokenizer_model_id="Qwen/Qwen3-8B",
        tokenizer_revision="1" * 40,
        registered_source_member_sha256s=(_sha("workload-source"),),
        rows=(_row("only-filler", 10),),
    )
    with pytest.raises(
        ContextCompilationBlocked,
        match="registered_filler_token_pool_insufficient",
    ):
        compile_context_requests(
            regime="long_input_short_output",
            context_tokens=1_024,
            core_rows=(_row("core", 20),),
            filler_authority=short_authority,
        )
    with pytest.raises(
        ContextCompilationBlocked,
        match="complete_core_exceeds_short_input_quarter_budget",
    ):
        compile_context_requests(
            regime="short_input_long_generation",
            context_tokens=1_024,
            core_rows=(_row("large-core", 257),),
            filler_authority=_authority(),
        )


def test_token_mutation_and_tokenizer_identity_change_compiled_identity() -> None:
    original_authority = _authority()
    mutated_authority = _authority(mutate_first_token=True)
    original = compile_context_requests(
        regime="long_input_short_output",
        context_tokens=1_024,
        core_rows=(_row("core", 20),),
        filler_authority=original_authority,
    )[0]
    mutated = compile_context_requests(
        regime="long_input_short_output",
        context_tokens=1_024,
        core_rows=(_row("core", 20),),
        filler_authority=mutated_authority,
    )[0]

    assert mutated_authority.sha256 != original_authority.sha256
    assert mutated.input_token_ids_sha256 != original.input_token_ids_sha256
    assert mutated.sha256 != original.sha256

    other_member = _sha("tokenizer-b")
    other_authority = _authority(tokenizer_member=other_member)
    with pytest.raises(
        ContextCompilationBlocked,
        match="core_and_filler_tokenizer_identity_differs",
    ):
        compile_context_requests(
            regime="long_input_short_output",
            context_tokens=1_024,
            core_rows=(_row("core", 20),),
            filler_authority=other_authority,
        )
    other = compile_context_requests(
        regime="long_input_short_output",
        context_tokens=1_024,
        core_rows=(_row("core", 20, tokenizer_member=other_member),),
        filler_authority=other_authority,
    )[0]
    assert other.tokenizer_content_member_id == other_member
    assert other.sha256 != original.sha256


def test_roundtrip_detects_token_and_span_mutation() -> None:
    result = compile_context_requests(
        regime="long_input_short_output",
        context_tokens=1_024,
        core_rows=(_row("core", 20),),
        filler_authority=_authority(),
    )[0]
    token_mutation = result.to_dict()
    tokens = list(token_mutation["input_token_ids"])
    tokens[-1] += 1
    token_mutation["input_token_ids"] = tokens
    with pytest.raises(ValueError, match="changed the complete core tokens"):
        CompiledContextRequest.from_dict(token_mutation)

    span_mutation = result.to_dict()
    spans = list(span_mutation["filler_spans"])
    spans[0] = {
        **spans[0],
        "compiled_start": 1,
        "compiled_stop": spans[0]["compiled_stop"] + 1,
    }
    span_mutation["filler_spans"] = spans
    with pytest.raises(ValueError, match="not contiguous"):
        CompiledContextRequest.from_dict(span_mutation)


def test_authority_rejects_rows_from_a_different_tokenizer() -> None:
    with pytest.raises(ValueError, match="authority rows differ"):
        ContextFillerAuthority(
            schema_version=1,
            kind="formal_single_operator_context_filler_authority",
            protocol_sha256=FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256,
            content_source_binding_sha256=_sha("content-source-binding"),
            tokenizer_content_member_id=_sha("tokenizer-a"),
            tokenizer_model_id="Qwen/Qwen3-8B",
            tokenizer_revision="1" * 40,
            registered_source_member_sha256s=(_sha("workload-source"),),
            rows=(_row("filler", 10, tokenizer_member=_sha("tokenizer-b")),),
        )

    authority = _authority()
    assert ContextFillerAuthority.from_dict(authority.to_dict()) == authority
    with pytest.raises(
        ContextCompilationBlocked,
        match="core_source_member_not_registered",
    ):
        compile_context_requests(
            regime="long_input_short_output",
            context_tokens=1_024,
            core_rows=(
                _row(
                    "foreign-core",
                    20,
                    source_member=_sha("foreign-workload-source"),
                ),
            ),
            filler_authority=authority,
        )


def test_context_compiler_protocol_digest_is_stable() -> None:
    assert FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256 == (
        "8fca0627f062554f6c63d841c0de19d7f5b912eb6aa694491d42c433d899951c"
    )
