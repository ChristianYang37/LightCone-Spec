from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedModelSnapshotSpec,
    bind_trusted_locked_workload,
    bind_trusted_model_snapshot_member,
)
from lightcone_spec.experiments.formal_single_operator_lcb_tokenizer import (
    LIVECODEBENCH_PROMPT_TOKEN_QUANTILE_RULE,
    LIVECODEBENCH_TASK_NATIVE_CONTEXT_BUDGET,
    LiveCodeBenchPromptTokenStatistics,
    LiveCodeBenchTokenizedPrompt,
    LiveCodeBenchV6HardTokenizerAuthority,
    build_livecodebench_v6_hard_tokenizer_authority,
    load_livecodebench_v6_hard_tokenizer_authority,
    publish_livecodebench_v6_hard_tokenizer_authority,
    require_livecodebench_e1_e2_task_native_budget,
)
from lightcone_spec.experiments.registry import content_sha256
from lightcone_spec.experiments.workload_authority import (
    FORMAL_WORKLOAD_PROTOCOLS,
    LIVECODEBENCH_V6_HARD_RAW_FILE_SHA256,
    LIVECODEBENCH_V6_HARD_REPOSITORY_REVISION,
    LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS,
    LIVECODEBENCH_V6_HARD_SELECTED_RAW_ROWS_SHA256,
    LIVECODEBENCH_V6_HARD_SOURCE_LOCK,
    RELEASE_LIVECODEBENCH_V6_HARD_VERIFICATION_METADATA,
    LiveCodeBenchV6HardVerificationMetadata,
    bind_formal_workload_authority,
)


def _sha(label: object) -> str:
    return content_sha256(label)


def _tokenized_rows(
    counts: tuple[int, ...] = tuple(range(1, 81)),
) -> tuple[LiveCodeBenchTokenizedPrompt, ...]:
    assert len(counts) == 80
    return tuple(
        LiveCodeBenchTokenizedPrompt(
            source_sample_id=f"sample-{ordinal:03d}",
            source_row_id=source_row_id,
            prompt_sha256=_sha(f"prompt-{ordinal}"),
            input_token_count=count,
            input_token_ids_sha256=_sha(
                {"sample": ordinal, "tokens": list(range(count))}
            ),
        )
        for ordinal, (source_row_id, count) in enumerate(
            zip(LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS, counts, strict=True)
        )
    )


def _authority(
    counts: tuple[int, ...] = tuple(range(1, 81)),
) -> LiveCodeBenchV6HardTokenizerAuthority:
    release = RELEASE_LIVECODEBENCH_V6_HARD_VERIFICATION_METADATA
    rows = _tokenized_rows(counts)
    return LiveCodeBenchV6HardTokenizerAuthority(
        schema_version=1,
        kind="livecodebench_v6_hard_tokenizer_authority",
        trust_mode="trusted_single_operator_no_signature",
        signature=None,
        formal_measured_authorization=False,
        claim_scope="trusted_single_operator_empirical_tokenizer_metadata",
        statistics_status="BOUND",
        scope="E1_E2_task_native_prompts_only",
        controlled_long_context_axis_included=False,
        workload_id="livecodebench_v6_hard",
        repository=FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"].repository,
        repository_revision=LIVECODEBENCH_V6_HARD_REPOSITORY_REVISION,
        dataset_config="v6",
        split="test",
        raw_file_sha256=LIVECODEBENCH_V6_HARD_RAW_FILE_SHA256,
        raw_row_count=175,
        selected_row_count=80,
        selected_question_ids=LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS,
        selected_raw_rows_sha256=LIVECODEBENCH_V6_HARD_SELECTED_RAW_ROWS_SHA256,
        formal_samples_sha256=release.formal_samples_sha256,
        source_lock_sha256=LIVECODEBENCH_V6_HARD_SOURCE_LOCK.sha256,
        protocol_sha256=FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"].sha256,
        formal_workload_authority_sha256=_sha("workload-authority"),
        trusted_workload_member_sha256=_sha("workload-member"),
        tokenizer_model_id="Qwen/Qwen3-8B",
        tokenizer_revision="tokenizer-revision",
        tokenizer_content_member_id=_sha("tokenizer-member"),
        tokenizer_tree_sha256=_sha("tokenizer-tree"),
        tokenizer_content_sha256=_sha("tokenizer-content"),
        tokenizer_class="Qwen2TokenizerFast",
        tokenizer_vocab_size=151_643,
        transformers_version="4.57.0",
        tokenization_policy=(
            "AutoTokenizer_add_special_tokens_true_truncation_false_v1"
        ),
        prompt_token_statistics=LiveCodeBenchPromptTokenStatistics.from_counts(counts),
        tokenized_prompts=rows,
        task_native_context_budget=LIVECODEBENCH_TASK_NATIVE_CONTEXT_BUDGET,
    )


def test_nearest_rank_prompt_statistics_are_explicit() -> None:
    statistics = LiveCodeBenchPromptTokenStatistics.from_counts(tuple(range(1, 81)))
    assert statistics.to_dict() == {
        "count": 80,
        "minimum": 1,
        "median": 40,
        "p90": 72,
        "p95": 76,
        "maximum": 80,
        "quantile_rule": LIVECODEBENCH_PROMPT_TOKEN_QUANTILE_RULE,
    }


def test_authority_codec_no_replace_and_tamper(tmp_path: Path) -> None:
    authority = _authority()
    assert (
        LiveCodeBenchV6HardTokenizerAuthority.from_dict(authority.to_dict())
        == authority
    )
    output = (tmp_path / "lcb-tokenizer-authority.json").resolve()
    binding = publish_livecodebench_v6_hard_tokenizer_authority(authority, output)
    assert load_livecodebench_v6_hard_tokenizer_authority(binding) == authority
    with pytest.raises(RuntimeError, match="target already exists"):
        publish_livecodebench_v6_hard_tokenizer_authority(authority, output)

    tampered = authority.to_dict()
    tokenized = list(tampered["tokenized_prompts"])
    tokenized[0] = {**tokenized[0], "input_token_count": 99_999}
    tampered["tokenized_prompts"] = tokenized
    with pytest.raises(ValueError, match="statistics changed|digest differs"):
        LiveCodeBenchV6HardTokenizerAuthority.from_dict(tampered)


def test_e1_e2_budget_accepts_exact_limit_and_rejects_overflow() -> None:
    counts = (LIVECODEBENCH_TASK_NATIVE_CONTEXT_BUDGET - 1,) + (1,) * 79
    authority = _authority(counts)
    outputs = (1,) * 80
    require_livecodebench_e1_e2_task_native_budget(
        authority,
        stage="E1",
        tokenized_prompts=authority.tokenized_prompts,
        requested_output_tokens=outputs,
    )
    overflow_outputs = (2,) + (1,) * 79
    with pytest.raises(ValueError, match="overflow before GPU allocation"):
        require_livecodebench_e1_e2_task_native_budget(
            authority,
            stage="E2",
            tokenized_prompts=authority.tokenized_prompts,
            requested_output_tokens=overflow_outputs,
        )
    with pytest.raises(ValueError, match="E1/E2-only"):
        require_livecodebench_e1_e2_task_native_budget(
            authority,
            stage="E3b",
            tokenized_prompts=authority.tokenized_prompts,
            requested_output_tokens=outputs,
        )


def test_authority_rejects_controlled_axis_or_incomplete_hard_rows() -> None:
    authority = _authority()
    with pytest.raises(ValueError, match="release identity differs"):
        replace(authority, controlled_long_context_axis_included=True)
    with pytest.raises(ValueError, match="row coverage differs"):
        replace(
            authority,
            tokenized_prompts=authority.tokenized_prompts[:-1],
            prompt_token_statistics=LiveCodeBenchPromptTokenStatistics.from_counts(
                tuple(row.input_token_count for row in authority.tokenized_prompts[:-1])
            ),
        )


def test_legacy_pending_verification_metadata_still_decodes() -> None:
    legacy = RELEASE_LIVECODEBENCH_V6_HARD_VERIFICATION_METADATA
    decoded = LiveCodeBenchV6HardVerificationMetadata.from_dict(legacy.to_dict())
    assert decoded == legacy
    assert decoded.tokenizer_statistics_status == "PENDING_TOKENIZER_REVISION_LOCK"
    assert decoded.prompt_token_statistics is None


def _release_cache_path() -> Path | None:
    cache_value = os.environ.get("LIGHTCONE_CONTENT_SOURCE_CACHE")
    if cache_value is None:
        return None
    path = (
        Path(cache_value).resolve()
        / "livecodebench-code_generation_lite-0fe84c3"
        / "test6.jsonl"
    )
    return path if path.is_file() else None


def test_real_locked_cache_replay_builds_all_80_rows(tmp_path: Path) -> None:
    raw_path = _release_cache_path()
    if raw_path is None:
        pytest.skip("set LIGHTCONE_CONTENT_SOURCE_CACHE for real LCB replay")
    workload = bind_formal_workload_authority(
        "livecodebench_v6_hard",
        raw_path,
    )
    locked = bind_trusted_locked_workload("livecodebench_v6_hard", raw_path)
    snapshot = (tmp_path / "tokenizer-revision").resolve()
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text(
        '{"fixture":"content-bound-only"}\n',
        encoding="utf-8",
    )
    tokenizer = bind_trusted_model_snapshot_member(
        TrustedModelSnapshotSpec(
            model_id="Qwen/Qwen3-8B",
            revision=snapshot.name,
            role="tokenizer",
            stages=("E1", "E2"),
            local_snapshot_path=str(snapshot),
        )
    )
    rows = tuple(
        LiveCodeBenchTokenizedPrompt(
            source_sample_id=sample.sample_id,
            source_row_id=sample.source_row_id,
            prompt_sha256=content_sha256(sample.prompt),
            input_token_count=max(1, len(sample.prompt.encode("utf-8")) // 4),
            input_token_ids_sha256=content_sha256(list(sample.prompt.encode("utf-8"))),
        )
        for sample in workload.samples
    )
    authority = build_livecodebench_v6_hard_tokenizer_authority(
        workload_authority=workload,
        locked_workload=locked,
        tokenizer_member=tokenizer,
        tokenized_prompts=rows,
        tokenizer_class="ContentBoundReplayTokenizer",
        tokenizer_vocab_size=256,
        transformers_version="test-no-runtime-load",
    )
    assert authority.raw_row_count == 175
    assert authority.selected_row_count == 80
    assert len(authority.selected_question_ids) == 80
    assert authority.selected_raw_rows_sha256 == (
        LIVECODEBENCH_V6_HARD_SELECTED_RAW_ROWS_SHA256
    )
    assert authority.prompt_token_statistics.count == 80
    assert authority.controlled_long_context_axis_included is False

    (snapshot / "tokenizer.json").write_text(
        '{"fixture":"tampered-after-binding"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="changed"):
        build_livecodebench_v6_hard_tokenizer_authority(
            workload_authority=workload,
            locked_workload=locked,
            tokenizer_member=tokenizer,
            tokenized_prompts=rows,
            tokenizer_class="ContentBoundReplayTokenizer",
            tokenizer_vocab_size=256,
            transformers_version="test-no-runtime-load",
        )
