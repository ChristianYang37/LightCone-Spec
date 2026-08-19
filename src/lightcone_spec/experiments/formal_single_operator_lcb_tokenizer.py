"""Locked LiveCodeBench-v6-hard tokenizer statistics for E1/E2.

The release workload record deliberately predates a concrete model snapshot and
therefore retains its schema-1 ``PENDING_TOKENIZER_REVISION_LOCK`` fields for
legacy decoding.  This module closes that runtime gap without changing the raw
dataset authority: it binds the complete exact-hard selection to the tokenizer
member already present in the trusted content bundle and to the token IDs
emitted by the first-party tokenizer worker.

The authority is intentionally scoped to task-native E1/E2 prompts.  It is not
an authority for the controlled 1K--40,928 E3a/E3b context axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.registry import content_sha256
from lightcone_spec.experiments.workload_authority import (
    FORMAL_WORKLOAD_PROTOCOLS,
    LIVECODEBENCH_V6_HARD_RAW_FILE_SHA256,
    LIVECODEBENCH_V6_HARD_REPOSITORY_REVISION,
    LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS,
    LIVECODEBENCH_V6_HARD_SELECTED_RAW_ROWS_SHA256,
    LIVECODEBENCH_V6_HARD_SOURCE_LOCK,
    RELEASE_LIVECODEBENCH_V6_HARD_VERIFICATION_METADATA,
    FormalWorkloadAuthority,
    revalidate_formal_workload_authority,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

LIVECODEBENCH_TASK_NATIVE_CONTEXT_BUDGET = 40_928
LIVECODEBENCH_PROMPT_TOKEN_QUANTILE_RULE = (
    "nearest_rank_ceiling_p_times_n_one_indexed_sorted_counts_v1"
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(character in value for character in ("\n", "\r", "\x00"))
    ):
        raise ValueError(f"{label} must be canonical single-line text")
    return value


def _strict_object(label: str, value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


@dataclass(frozen=True)
class LiveCodeBenchTokenizedPrompt:
    """One complete selected prompt after the first-party tokenizer worker."""

    source_sample_id: str
    source_row_id: str
    prompt_sha256: str
    input_token_count: int
    input_token_ids_sha256: str

    def __post_init__(self) -> None:
        _require_text("LiveCodeBench source sample", self.source_sample_id)
        _require_text("LiveCodeBench source row", self.source_row_id)
        _require_sha256("LiveCodeBench prompt", self.prompt_sha256)
        _require_sha256("LiveCodeBench input token IDs", self.input_token_ids_sha256)
        if type(self.input_token_count) is not int or self.input_token_count < 1:
            raise ValueError("LiveCodeBench input token count must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_sample_id": self.source_sample_id,
            "source_row_id": self.source_row_id,
            "prompt_sha256": self.prompt_sha256,
            "input_token_count": self.input_token_count,
            "input_token_ids_sha256": self.input_token_ids_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict_object(
                "LiveCodeBench tokenized prompt",
                value,
                {field.name for field in fields(cls)},
            )
        )  # type: ignore[arg-type]


def _nearest_rank(sorted_counts: tuple[int, ...], probability: float) -> int:
    if not sorted_counts or not 0.0 < probability <= 1.0:
        raise ValueError("nearest-rank statistic input is invalid")
    rank = math.ceil(probability * len(sorted_counts))
    return sorted_counts[rank - 1]


@dataclass(frozen=True)
class LiveCodeBenchPromptTokenStatistics:
    count: int
    minimum: int
    median: int
    p90: int
    p95: int
    maximum: int
    quantile_rule: Literal[
        "nearest_rank_ceiling_p_times_n_one_indexed_sorted_counts_v1"
    ]

    def __post_init__(self) -> None:
        if (
            type(self.count) is not int
            or self.count < 1
            or any(
                type(value) is not int or value < 1
                for value in (
                    self.minimum,
                    self.median,
                    self.p90,
                    self.p95,
                    self.maximum,
                )
            )
            or not (self.minimum <= self.median <= self.p90 <= self.p95 <= self.maximum)
            or self.quantile_rule != LIVECODEBENCH_PROMPT_TOKEN_QUANTILE_RULE
        ):
            raise ValueError("LiveCodeBench prompt token statistics are invalid")

    @classmethod
    def from_counts(cls, counts: tuple[int, ...]) -> Self:
        if (
            type(counts) is not tuple
            or not counts
            or any(type(value) is not int or value < 1 for value in counts)
        ):
            raise ValueError("LiveCodeBench prompt token counts are invalid")
        ordered = tuple(sorted(counts))
        return cls(
            count=len(ordered),
            minimum=ordered[0],
            median=_nearest_rank(ordered, 0.50),
            p90=_nearest_rank(ordered, 0.90),
            p95=_nearest_rank(ordered, 0.95),
            maximum=ordered[-1],
            quantile_rule=LIVECODEBENCH_PROMPT_TOKEN_QUANTILE_RULE,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "minimum": self.minimum,
            "median": self.median,
            "p90": self.p90,
            "p95": self.p95,
            "maximum": self.maximum,
            "quantile_rule": self.quantile_rule,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        return cls(
            **_strict_object(
                "LiveCodeBench prompt token statistics",
                value,
                {field.name for field in fields(cls)},
            )
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class LiveCodeBenchV6HardTokenizerAuthority:
    """Path-free runtime authority for the exact 80 hard task-native prompts."""

    schema_version: Literal[1]
    kind: Literal["livecodebench_v6_hard_tokenizer_authority"]
    trust_mode: Literal["trusted_single_operator_no_signature"]
    signature: None
    formal_measured_authorization: Literal[False]
    claim_scope: Literal["trusted_single_operator_empirical_tokenizer_metadata"]
    statistics_status: Literal["BOUND"]
    scope: Literal["E1_E2_task_native_prompts_only"]
    controlled_long_context_axis_included: Literal[False]
    workload_id: Literal["livecodebench_v6_hard"]
    repository: str
    repository_revision: str
    dataset_config: str
    split: str
    raw_file_sha256: str
    raw_row_count: int
    selected_row_count: int
    selected_question_ids: tuple[str, ...]
    selected_raw_rows_sha256: str
    formal_samples_sha256: str
    source_lock_sha256: str
    protocol_sha256: str
    formal_workload_authority_sha256: str
    trusted_workload_member_sha256: str
    tokenizer_model_id: str
    tokenizer_revision: str
    tokenizer_content_member_id: str
    tokenizer_tree_sha256: str
    tokenizer_content_sha256: str
    tokenizer_class: str
    tokenizer_vocab_size: int
    transformers_version: str
    tokenization_policy: Literal[
        "AutoTokenizer_add_special_tokens_true_truncation_false_v1"
    ]
    prompt_token_statistics: LiveCodeBenchPromptTokenStatistics
    tokenized_prompts: tuple[LiveCodeBenchTokenizedPrompt, ...]
    task_native_context_budget: Literal[40928]

    def __post_init__(self) -> None:
        release = RELEASE_LIVECODEBENCH_V6_HARD_VERIFICATION_METADATA
        protocol = FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"]
        if (
            self.schema_version != 1
            or self.kind != "livecodebench_v6_hard_tokenizer_authority"
            or self.trust_mode != "trusted_single_operator_no_signature"
            or self.signature is not None
            or self.formal_measured_authorization is not False
            or self.claim_scope
            != "trusted_single_operator_empirical_tokenizer_metadata"
            or self.statistics_status != "BOUND"
            or self.scope != "E1_E2_task_native_prompts_only"
            or self.controlled_long_context_axis_included is not False
            or self.workload_id != "livecodebench_v6_hard"
            or self.repository != protocol.repository
            or self.repository_revision != LIVECODEBENCH_V6_HARD_REPOSITORY_REVISION
            or self.dataset_config != "v6"
            or self.split != "test"
            or self.raw_file_sha256 != LIVECODEBENCH_V6_HARD_RAW_FILE_SHA256
            or self.raw_row_count != 175
            or self.selected_row_count != 80
            or self.selected_question_ids != LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS
            or self.selected_raw_rows_sha256
            != LIVECODEBENCH_V6_HARD_SELECTED_RAW_ROWS_SHA256
            or self.formal_samples_sha256 != release.formal_samples_sha256
            or self.source_lock_sha256 != LIVECODEBENCH_V6_HARD_SOURCE_LOCK.sha256
            or self.protocol_sha256 != protocol.sha256
            or self.tokenization_policy
            != "AutoTokenizer_add_special_tokens_true_truncation_false_v1"
            or self.task_native_context_budget
            != LIVECODEBENCH_TASK_NATIVE_CONTEXT_BUDGET
        ):
            raise ValueError(
                "LiveCodeBench tokenizer authority release identity differs"
            )
        for label, value in (
            ("formal workload authority", self.formal_workload_authority_sha256),
            ("trusted workload member", self.trusted_workload_member_sha256),
            ("tokenizer member", self.tokenizer_content_member_id),
            ("tokenizer tree", self.tokenizer_tree_sha256),
            ("tokenizer content", self.tokenizer_content_sha256),
        ):
            _require_sha256(f"LiveCodeBench {label}", value)
        for label, value in (
            ("tokenizer model", self.tokenizer_model_id),
            ("tokenizer revision", self.tokenizer_revision),
            ("tokenizer class", self.tokenizer_class),
            ("transformers version", self.transformers_version),
        ):
            _require_text(f"LiveCodeBench {label}", value)
        if type(self.tokenizer_vocab_size) is not int or self.tokenizer_vocab_size < 1:
            raise ValueError("LiveCodeBench tokenizer vocab size must be positive")
        if (
            type(self.tokenized_prompts) is not tuple
            or len(self.tokenized_prompts) != 80
            or any(
                type(row) is not LiveCodeBenchTokenizedPrompt
                for row in self.tokenized_prompts
            )
            or tuple(row.source_row_id for row in self.tokenized_prompts)
            != self.selected_question_ids
            or len({row.source_sample_id for row in self.tokenized_prompts}) != 80
            or len({row.source_row_id for row in self.tokenized_prompts}) != 80
        ):
            raise ValueError("LiveCodeBench tokenizer row coverage differs")
        expected_statistics = LiveCodeBenchPromptTokenStatistics.from_counts(
            tuple(row.input_token_count for row in self.tokenized_prompts)
        )
        if self.prompt_token_statistics != expected_statistics:
            raise ValueError("LiveCodeBench prompt token statistics changed")

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "trust_mode": self.trust_mode,
            "signature": self.signature,
            "formal_measured_authorization": self.formal_measured_authorization,
            "claim_scope": self.claim_scope,
            "statistics_status": self.statistics_status,
            "scope": self.scope,
            "controlled_long_context_axis_included": (
                self.controlled_long_context_axis_included
            ),
            "workload_id": self.workload_id,
            "repository": self.repository,
            "repository_revision": self.repository_revision,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "raw_file_sha256": self.raw_file_sha256,
            "raw_row_count": self.raw_row_count,
            "selected_row_count": self.selected_row_count,
            "selected_question_ids": list(self.selected_question_ids),
            "selected_raw_rows_sha256": self.selected_raw_rows_sha256,
            "formal_samples_sha256": self.formal_samples_sha256,
            "source_lock_sha256": self.source_lock_sha256,
            "protocol_sha256": self.protocol_sha256,
            "formal_workload_authority_sha256": (self.formal_workload_authority_sha256),
            "trusted_workload_member_sha256": self.trusted_workload_member_sha256,
            "tokenizer_model_id": self.tokenizer_model_id,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_content_member_id": self.tokenizer_content_member_id,
            "tokenizer_tree_sha256": self.tokenizer_tree_sha256,
            "tokenizer_content_sha256": self.tokenizer_content_sha256,
            "tokenizer_class": self.tokenizer_class,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "transformers_version": self.transformers_version,
            "tokenization_policy": self.tokenization_policy,
            "prompt_token_statistics": self.prompt_token_statistics.to_dict(),
            "tokenized_prompts": [row.to_dict() for row in self.tokenized_prompts],
            "task_native_context_budget": self.task_native_context_budget,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {"authority_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "LiveCodeBench tokenizer authority",
            value,
            {field.name for field in fields(cls)} | {"authority_sha256"},
        )
        declared = _require_sha256(
            "LiveCodeBench tokenizer authority",
            row.pop("authority_sha256"),
        )
        selected_ids = row.pop("selected_question_ids")
        tokenized = row.pop("tokenized_prompts")
        statistics = row.pop("prompt_token_statistics")
        if type(selected_ids) is not list or type(tokenized) is not list:
            raise TypeError("LiveCodeBench tokenizer authority rows must be arrays")
        authority = cls(
            **row,
            selected_question_ids=tuple(selected_ids),
            prompt_token_statistics=LiveCodeBenchPromptTokenStatistics.from_dict(
                statistics
            ),
            tokenized_prompts=tuple(
                LiveCodeBenchTokenizedPrompt.from_dict(item) for item in tokenized
            ),
        )  # type: ignore[arg-type]
        if authority.sha256 != declared:
            raise ValueError("LiveCodeBench tokenizer authority digest differs")
        return authority


def _assemble_livecodebench_v6_hard_tokenizer_authority(
    *,
    workload_authority: FormalWorkloadAuthority,
    locked_workload: object,
    tokenizer_member: object,
    tokenized_prompts: tuple[LiveCodeBenchTokenizedPrompt, ...],
    tokenizer_class: str,
    tokenizer_vocab_size: int,
    transformers_version: str,
) -> LiveCodeBenchV6HardTokenizerAuthority:
    """Assemble from members already validated by the owning source boundary."""

    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedLockedWorkload,
        TrustedModelSnapshotMember,
    )

    if (
        type(workload_authority) is not FormalWorkloadAuthority
        or workload_authority.workload_id != "livecodebench_v6_hard"
        or type(locked_workload) is not TrustedLockedWorkload
        or locked_workload.workload_id != "livecodebench_v6_hard"
        or type(tokenizer_member) is not TrustedModelSnapshotMember
        or tokenizer_member.role != "tokenizer"
        or not {"E1", "E2"}.issubset(tokenizer_member.stages)
    ):
        raise TypeError("LiveCodeBench tokenizer authority inputs are not exact")
    workload = workload_authority
    tokenizer = tokenizer_member
    if (
        locked_workload.authority_sha256 != workload.sha256
        or locked_workload.raw_source_path != workload.raw_source_path
        or locked_workload.raw_file_sha256 != workload.raw_file_sha256
        or locked_workload.repository_revision != workload.repository_revision
        or locked_workload.raw_row_count != workload.raw_row_count
        or locked_workload.selected_row_count != workload.selected_row_count
        or locked_workload.formal_samples_sha256 != workload.selected_rows_sha256
        or locked_workload.source_lock_sha256 != workload.source_lock_sha256
        or locked_workload.protocol_sha256 != workload.protocol_sha256
    ):
        raise ValueError("LiveCodeBench locked workload differs from raw authority")
    expected_samples = tuple(
        (
            sample.sample_id,
            sample.source_row_id,
            content_sha256(sample.prompt),
        )
        for sample in workload.samples
    )
    observed_samples = tuple(
        (row.source_sample_id, row.source_row_id, row.prompt_sha256)
        for row in tokenized_prompts
    )
    if observed_samples != expected_samples:
        raise ValueError("LiveCodeBench tokenized prompts differ from exact hard rows")
    metadata = locked_workload.verification_metadata
    if metadata != RELEASE_LIVECODEBENCH_V6_HARD_VERIFICATION_METADATA:
        raise ValueError("LiveCodeBench locked verification metadata differs")
    statistics = LiveCodeBenchPromptTokenStatistics.from_counts(
        tuple(row.input_token_count for row in tokenized_prompts)
    )
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
        repository=metadata.repository,
        repository_revision=metadata.repository_revision,
        dataset_config=metadata.dataset_config,
        split=metadata.split,
        raw_file_sha256=metadata.raw_file_sha256,
        raw_row_count=metadata.raw_row_count,
        selected_row_count=metadata.selected_row_count,
        selected_question_ids=metadata.selected_question_ids,
        selected_raw_rows_sha256=metadata.selected_raw_rows_sha256,
        formal_samples_sha256=metadata.formal_samples_sha256,
        source_lock_sha256=metadata.source_lock_sha256,
        protocol_sha256=metadata.protocol_sha256,
        formal_workload_authority_sha256=workload.sha256,
        trusted_workload_member_sha256=locked_workload.sha256,
        tokenizer_model_id=tokenizer.model_id,
        tokenizer_revision=tokenizer.revision,
        tokenizer_content_member_id=tokenizer.sha256,
        tokenizer_tree_sha256=tokenizer.tree_sha256,
        tokenizer_content_sha256=tokenizer.content_sha256,
        tokenizer_class=tokenizer_class,
        tokenizer_vocab_size=tokenizer_vocab_size,
        transformers_version=transformers_version,
        tokenization_policy=(
            "AutoTokenizer_add_special_tokens_true_truncation_false_v1"
        ),
        prompt_token_statistics=statistics,
        tokenized_prompts=tokenized_prompts,
        task_native_context_budget=LIVECODEBENCH_TASK_NATIVE_CONTEXT_BUDGET,
    )


def build_livecodebench_v6_hard_tokenizer_authority(
    *,
    workload_authority: FormalWorkloadAuthority,
    locked_workload: object,
    tokenizer_member: object,
    tokenized_prompts: tuple[LiveCodeBenchTokenizedPrompt, ...],
    tokenizer_class: str,
    tokenizer_vocab_size: int,
    transformers_version: str,
) -> LiveCodeBenchV6HardTokenizerAuthority:
    """Deep-reopen standalone inputs, then bind exact first-party token rows."""

    from lightcone_spec.experiments.formal_single_operator_content import (
        revalidate_trusted_model_snapshot_member,
    )

    workload = revalidate_formal_workload_authority(workload_authority)
    tokenizer = revalidate_trusted_model_snapshot_member(tokenizer_member)  # type: ignore[arg-type]
    return _assemble_livecodebench_v6_hard_tokenizer_authority(
        workload_authority=workload,
        locked_workload=locked_workload,
        tokenizer_member=tokenizer,
        tokenized_prompts=tokenized_prompts,
        tokenizer_class=tokenizer_class,
        tokenizer_vocab_size=tokenizer_vocab_size,
        transformers_version=transformers_version,
    )


def build_livecodebench_v6_hard_tokenizer_authority_from_revalidated_bundle(
    *,
    content_bundle: object,
    workload_authority: FormalWorkloadAuthority,
    locked_workload: object,
    tokenizer_member: object,
    tokenized_prompts: tuple[LiveCodeBenchTokenizedPrompt, ...],
    tokenizer_class: str,
    tokenizer_vocab_size: int,
    transformers_version: str,
) -> LiveCodeBenchV6HardTokenizerAuthority:
    """Avoid a second multi-GB scan after ``FormalContentSourceBinding.reopen``.

    The caller must pass the exact bundle returned by that deep-reopen boundary;
    this reducer accepts only members of that bundle and performs no path scan of
    its own.  The public standalone builder above remains the deep-reopen API.
    """

    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedLockedWorkload,
        TrustedModelSnapshotMember,
        TrustedSingleOperatorContentBundle,
    )

    if (
        type(content_bundle) is not TrustedSingleOperatorContentBundle
        or type(workload_authority) is not FormalWorkloadAuthority
        or type(locked_workload) is not TrustedLockedWorkload
        or type(tokenizer_member) is not TrustedModelSnapshotMember
        or locked_workload not in content_bundle.locked_workloads
        or tokenizer_member not in content_bundle.model_members
    ):
        raise TypeError("LiveCodeBench authority inputs leave the revalidated bundle")
    content_bundle.__post_init__()
    workload_authority.__post_init__()
    locked_workload.__post_init__()
    tokenizer_member.__post_init__()
    return _assemble_livecodebench_v6_hard_tokenizer_authority(
        workload_authority=workload_authority,
        locked_workload=locked_workload,
        tokenizer_member=tokenizer_member,
        tokenized_prompts=tokenized_prompts,
        tokenizer_class=tokenizer_class,
        tokenizer_vocab_size=tokenizer_vocab_size,
        transformers_version=transformers_version,
    )


def require_livecodebench_e1_e2_task_native_budget(
    authority: LiveCodeBenchV6HardTokenizerAuthority,
    *,
    stage: str,
    tokenized_prompts: tuple[LiveCodeBenchTokenizedPrompt, ...],
    requested_output_tokens: tuple[int, ...],
) -> None:
    """Fail closed on exact prompt+generation length before GPU allocation."""

    if type(authority) is not LiveCodeBenchV6HardTokenizerAuthority:
        raise TypeError("LiveCodeBench task-native budget requires its authority")
    authority.__post_init__()
    if stage not in {"E1", "E2"}:
        raise ValueError("LiveCodeBench task-native budget gate is E1/E2-only")
    if (
        tokenized_prompts != authority.tokenized_prompts
        or type(requested_output_tokens) is not tuple
        or len(requested_output_tokens) != len(tokenized_prompts)
        or any(type(value) is not int or value < 1 for value in requested_output_tokens)
    ):
        raise ValueError("LiveCodeBench task-native budget row coverage differs")
    overflow = tuple(
        (
            row.source_sample_id,
            row.input_token_count,
            output_tokens,
            row.input_token_count + output_tokens,
        )
        for row, output_tokens in zip(
            tokenized_prompts,
            requested_output_tokens,
            strict=True,
        )
        if row.input_token_count + output_tokens
        > LIVECODEBENCH_TASK_NATIVE_CONTEXT_BUDGET
    )
    if overflow:
        sample_id, input_count, output_count, total = overflow[0]
        raise ValueError(
            "LiveCodeBench E1/E2 task-native context overflow before GPU allocation: "
            f"sample={sample_id}, input={input_count}, output={output_count}, "
            f"total={total}, limit={LIVECODEBENCH_TASK_NATIVE_CONTEXT_BUDGET}"
        )


def publish_livecodebench_v6_hard_tokenizer_authority(
    authority: LiveCodeBenchV6HardTokenizerAuthority,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(authority) is not LiveCodeBenchV6HardTokenizerAuthority:
        raise TypeError("LiveCodeBench tokenizer publisher requires its authority")
    authority.__post_init__()
    publish_canonical_json_no_replace(output_path, authority.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    if load_livecodebench_v6_hard_tokenizer_authority(binding) != authority:
        raise RuntimeError("published LiveCodeBench tokenizer authority changed")
    return binding


def load_livecodebench_v6_hard_tokenizer_authority(
    source: CanonicalJsonProofBinding | str | Path,
) -> LiveCodeBenchV6HardTokenizerAuthority:
    binding = (
        source
        if type(source) is CanonicalJsonProofBinding
        else CanonicalJsonProofBinding.bind(source)
    )
    if type(binding) is not CanonicalJsonProofBinding:
        raise TypeError("LiveCodeBench tokenizer authority binding differs")
    authority = LiveCodeBenchV6HardTokenizerAuthority.from_dict(binding.reopen())
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise ValueError("LiveCodeBench tokenizer authority path binding changed")
    return authority


__all__ = [
    "LIVECODEBENCH_PROMPT_TOKEN_QUANTILE_RULE",
    "LIVECODEBENCH_TASK_NATIVE_CONTEXT_BUDGET",
    "LiveCodeBenchPromptTokenStatistics",
    "LiveCodeBenchTokenizedPrompt",
    "LiveCodeBenchV6HardTokenizerAuthority",
    "build_livecodebench_v6_hard_tokenizer_authority",
    "build_livecodebench_v6_hard_tokenizer_authority_from_revalidated_bundle",
    "load_livecodebench_v6_hard_tokenizer_authority",
    "publish_livecodebench_v6_hard_tokenizer_authority",
    "require_livecodebench_e1_e2_task_native_budget",
]
