"""Release-owned authority for formal external benchmark workloads.

Formal workload rows are local, path-bound raw inputs.  This module never
downloads a dataset and never accepts a caller-provided revision or digest as
authority.  A release-owned source lock must bind the repository revision,
exact raw bytes, full row count, filtering protocol, and complete selected-row
digest before a local file is opened.

The static release registry contains the reviewed LiveCodeBench v6 ``test``
JSONL and MATH-500 ``test`` JSONL sources.  Formal staged execution may instead
consume a fresh offline-root-signed, challenge-bound workload authorization and
then deep-reopen the same exact local bytes.  Neither path may take the first N
rows, coerce a typed filter literal, or accept a caller-supplied bare digest as
authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

from lightcone_spec.experiments.data import PromptSample
from lightcone_spec.experiments.registry import content_sha256
from lightcone_spec.runtime.content_authorization import (
    VerifiedReleaseWorkloadSources,
)

FormalWorkloadId = Literal["livecodebench_v6_hard", "math500_level5"]

_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RAW_BYTES = 512 * 1024 * 1024

FORMAL_WORKLOAD_SOURCE_ALLOWLIST_EMPTY_REASON = "formal_workload_source_allowlist_empty"
FORMAL_WORKLOAD_SOURCE_NOT_REGISTERED_REASON = "formal_workload_source_not_registered"
FORMAL_WORKLOAD_LOCAL_SOURCE_MISSING_REASON = "formal_workload_local_source_missing"
FORMAL_WORKLOAD_FILTER_EMPTY_REASON = "formal_workload_filter_empty"
FORMAL_WORKLOAD_AUTHORITY_ARTIFACT_PREFIX = "formal_workload_authority:"


class FormalWorkloadAuthorityBlocked(RuntimeError):
    """Raised before execution when no release-owned workload is available."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"formal workload authority is BLOCKED: {reason}")
        self.reason = reason


def _require_sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_git_commit(label: str, value: object) -> str:
    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase Git commit")
    return value


def _require_positive_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_filter_literal(label: str, value: object) -> str | int:
    if type(value) is str:
        if not value:
            raise ValueError(f"{label} must be non-empty")
        return value
    if type(value) is int:
        return value
    raise TypeError(f"{label} must be exact JSON string or integer")


def _strict_mapping(label: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    return value


def _strict_sequence(label: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    return value


@dataclass(frozen=True)
class FormalWorkloadProtocol:
    workload_id: FormalWorkloadId
    repository: str
    dataset_config: str
    split: str
    identity_field: str
    prompt_field: str
    filter_field: str
    filter_value: str | int
    prompt_compiler: str
    selection_policy: str

    def __post_init__(self) -> None:
        if self.workload_id not in {
            "livecodebench_v6_hard",
            "math500_level5",
        }:
            raise ValueError("formal workload protocol is unsupported")
        for label, value in (
            ("repository", self.repository),
            ("dataset config", self.dataset_config),
            ("split", self.split),
            ("identity field", self.identity_field),
            ("prompt field", self.prompt_field),
            ("filter field", self.filter_field),
            ("prompt compiler", self.prompt_compiler),
            ("selection policy", self.selection_policy),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"formal workload {label} must be non-empty")
        _require_filter_literal("formal workload filter value", self.filter_value)
        if self.prompt_compiler != "verbatim_nfc_no_trim_v1":
            raise ValueError("formal workload prompt compiler is unsupported")
        if self.selection_policy != "all_exact_matches_in_raw_order":
            raise ValueError("formal workload selection policy is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "workload_id": self.workload_id,
            "repository": self.repository,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "identity_field": self.identity_field,
            "prompt_field": self.prompt_field,
            "filter_field": self.filter_field,
            "filter_value": self.filter_value,
            "prompt_compiler": self.prompt_compiler,
            "selection_policy": self.selection_policy,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


FORMAL_WORKLOAD_PROTOCOLS: Mapping[FormalWorkloadId, FormalWorkloadProtocol] = (
    MappingProxyType(
        {
            "livecodebench_v6_hard": FormalWorkloadProtocol(
                workload_id="livecodebench_v6_hard",
                repository="livecodebench/code_generation_lite",
                dataset_config="v6",
                split="test",
                identity_field="question_id",
                prompt_field="question_content",
                filter_field="difficulty",
                filter_value="hard",
                prompt_compiler="verbatim_nfc_no_trim_v1",
                selection_policy="all_exact_matches_in_raw_order",
            ),
            "math500_level5": FormalWorkloadProtocol(
                workload_id="math500_level5",
                repository="HuggingFaceH4/MATH-500",
                dataset_config="default",
                split="test",
                identity_field="unique_id",
                prompt_field="problem",
                filter_field="level",
                filter_value=5,
                prompt_compiler="verbatim_nfc_no_trim_v1",
                selection_policy="all_exact_matches_in_raw_order",
            ),
        }
    )
)

FORMAL_WORKLOAD_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_external_workload_protocol",
        "sources": {
            key: value.to_dict()
            for key, value in sorted(FORMAL_WORKLOAD_PROTOCOLS.items())
        },
        "source_authority": "release_allowlist_exact_revision_and_raw_bytes",
        "selection": "all_exact_filter_matches_in_raw_order_never_first_n",
        "missing": "BLOCKED_never_empty",
        "network": "forbidden",
    }
)


@dataclass(frozen=True)
class ReleaseWorkloadSourceLock:
    """One reviewed source identity compiled into a future release."""

    workload_id: FormalWorkloadId
    repository_revision: str
    raw_file_sha256: str
    raw_row_count: int
    selected_row_count: int
    selected_rows_sha256: str
    protocol_sha256: str

    def __post_init__(self) -> None:
        if self.workload_id not in FORMAL_WORKLOAD_PROTOCOLS:
            raise ValueError("source lock names an unsupported formal workload")
        _require_git_commit("workload repository revision", self.repository_revision)
        _require_sha256("workload raw file", self.raw_file_sha256)
        _require_sha256("workload selected rows", self.selected_rows_sha256)
        _require_sha256("workload protocol", self.protocol_sha256)
        _require_positive_int("workload raw row count", self.raw_row_count)
        _require_positive_int("workload selected row count", self.selected_row_count)
        if self.selected_row_count > self.raw_row_count:
            raise ValueError("selected workload rows exceed the raw source")
        expected_protocol = FORMAL_WORKLOAD_PROTOCOLS[self.workload_id].sha256
        if self.protocol_sha256 != expected_protocol:
            raise ValueError("source lock uses another workload filter protocol")

    def to_dict(self) -> dict[str, object]:
        return {
            "workload_id": self.workload_id,
            "repository_revision": self.repository_revision,
            "raw_file_sha256": self.raw_file_sha256,
            "raw_row_count": self.raw_row_count,
            "selected_row_count": self.selected_row_count,
            "selected_rows_sha256": self.selected_rows_sha256,
            "protocol_sha256": self.protocol_sha256,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


LIVECODEBENCH_V6_HARD_REPOSITORY_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LIVECODEBENCH_V6_HARD_RAW_FILE_SHA256 = (
    "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"
)
LIVECODEBENCH_V6_HARD_FORMAL_SAMPLES_SHA256 = (
    "83a25d4b656b74263b30969302697178d2e72a8d6b1797dedeff43299ccdc42e"
)
LIVECODEBENCH_V6_HARD_SELECTED_RAW_ROWS_SHA256 = (
    "57dc96f34aa8e68d62198fdb3c500cf4f9cadc56b42c85e745e4f314c56fccc6"
)
LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS_SHA256 = (
    "45fb8da37695fa085d89fe65c7cc8e0e391f64def87d0500778f07dc54f5d162"
)
LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS = (
    "abc387_f",
    "abc388_g",
    "abc388_f",
    "abc388_e",
    "abc389_g",
    "abc389_f",
    "abc389_e",
    "abc390_e",
    "abc390_g",
    "abc390_f",
    "abc391_f",
    "abc391_e",
    "abc391_g",
    "abc392_g",
    "abc392_f",
    "abc393_f",
    "abc393_e",
    "abc393_d",
    "abc394_f",
    "abc394_g",
    "abc394_e",
    "abc395_f",
    "abc395_e",
    "abc396_g",
    "abc396_f",
    "abc396_e",
    "abc397_d",
    "abc397_f",
    "abc397_g",
    "abc397_e",
    "abc398_f",
    "abc398_d",
    "abc398_g",
    "abc399_e",
    "abc399_f",
    "abc400_e",
    "abc400_g",
    "arc190_c",
    "arc190_a",
    "arc190_d",
    "arc191_d",
    "arc191_c",
    "arc192_e",
    "arc192_b",
    "arc192_d",
    "arc193_b",
    "arc193_a",
    "arc193_d",
    "arc194_c",
    "arc194_b",
    "arc194_e",
    "arc194_d",
    "arc195_e",
    "arc195_b",
    "arc195_c",
    "arc195_d",
    "arc196_b",
    "arc196_c",
    "arc196_a",
    "arc196_d",
    "3674",
    "3725",
    "3697",
    "3696",
    "3762",
    "3733",
    "3781",
    "3770",
    "3789",
    "3801",
    "3744",
    "3717",
    "3777",
    "3687",
    "3739",
    "3701",
    "3692",
    "3783",
    "3784",
    "3765",
)

MATH500_LEVEL5_REPOSITORY_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH500_LEVEL5_RAW_FILE_SHA256 = (
    "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
)
MATH500_LEVEL5_FORMAL_SAMPLES_SHA256 = (
    "115420d874458d4b6e57c77124bce5c105c68e4d0788cb336a9b0ea6c60f5294"
)
MATH500_LEVEL5_SELECTED_RAW_ROWS_SHA256 = (
    "14f1927ac889d4b734b5a882b92c2ea9c9ed8b8d9d486664f8e3f6280f2eb089"
)
MATH500_LEVEL5_SELECTED_SOURCE_ROW_IDS_SHA256 = (
    "ed4592216afa43006146becd492de7202bd786c4c2ce66fa3dcabaf26d4b1a6e"
)
MATH500_LEVEL5_SELECTED_SOURCE_ROW_IDS = (
    "test/intermediate_algebra/1994.json",
    "test/prealgebra/1139.json",
    "test/intermediate_algebra/1197.json",
    "test/number_theory/737.json",
    "test/precalculus/990.json",
    "test/algebra/1837.json",
    "test/algebra/2193.json",
    "test/algebra/2427.json",
    "test/intermediate_algebra/1388.json",
    "test/counting_and_probability/525.json",
    "test/intermediate_algebra/1454.json",
    "test/prealgebra/1558.json",
    "test/algebra/305.json",
    "test/number_theory/1055.json",
    "test/prealgebra/951.json",
    "test/intermediate_algebra/956.json",
    "test/precalculus/675.json",
    "test/intermediate_algebra/279.json",
    "test/intermediate_algebra/1849.json",
    "test/algebra/2700.json",
    "test/algebra/892.json",
    "test/prealgebra/1646.json",
    "test/intermediate_algebra/662.json",
    "test/geometry/183.json",
    "test/intermediate_algebra/582.json",
    "test/intermediate_algebra/558.json",
    "test/precalculus/44.json",
    "test/intermediate_algebra/232.json",
    "test/geometry/1140.json",
    "test/prealgebra/1297.json",
    "test/intermediate_algebra/2022.json",
    "test/counting_and_probability/159.json",
    "test/intermediate_algebra/1408.json",
    "test/intermediate_algebra/966.json",
    "test/intermediate_algebra/964.json",
    "test/precalculus/986.json",
    "test/geometry/172.json",
    "test/geometry/880.json",
    "test/prealgebra/1655.json",
    "test/algebra/2517.json",
    "test/prealgebra/1003.json",
    "test/counting_and_probability/181.json",
    "test/algebra/297.json",
    "test/algebra/686.json",
    "test/intermediate_algebra/183.json",
    "test/algebra/733.json",
    "test/geometry/702.json",
    "test/counting_and_probability/51.json",
    "test/intermediate_algebra/2196.json",
    "test/number_theory/838.json",
    "test/algebra/291.json",
    "test/intermediate_algebra/1510.json",
    "test/counting_and_probability/765.json",
    "test/prealgebra/1961.json",
    "test/intermediate_algebra/1354.json",
    "test/algebra/975.json",
    "test/geometry/229.json",
    "test/algebra/1143.json",
    "test/algebra/2626.json",
    "test/prealgebra/1251.json",
    "test/counting_and_probability/894.json",
    "test/algebra/2064.json",
    "test/counting_and_probability/1009.json",
    "test/geometry/826.json",
    "test/geometry/686.json",
    "test/algebra/1282.json",
    "test/prealgebra/1512.json",
    "test/precalculus/768.json",
    "test/intermediate_algebra/960.json",
    "test/number_theory/631.json",
    "test/precalculus/1172.json",
    "test/algebra/2486.json",
    "test/prealgebra/1044.json",
    "test/geometry/965.json",
    "test/intermediate_algebra/1544.json",
    "test/geometry/711.json",
    "test/prealgebra/1423.json",
    "test/intermediate_algebra/2152.json",
    "test/geometry/947.json",
    "test/prealgebra/954.json",
    "test/counting_and_probability/870.json",
    "test/intermediate_algebra/117.json",
    "test/algebra/2176.json",
    "test/counting_and_probability/216.json",
    "test/algebra/509.json",
    "test/intermediate_algebra/1166.json",
    "test/algebra/1078.json",
    "test/prealgebra/1353.json",
    "test/number_theory/820.json",
    "test/intermediate_algebra/955.json",
    "test/algebra/2264.json",
    "test/number_theory/427.json",
    "test/counting_and_probability/188.json",
    "test/intermediate_algebra/1111.json",
    "test/prealgebra/2066.json",
    "test/algebra/1031.json",
    "test/algebra/853.json",
    "test/algebra/2277.json",
    "test/precalculus/902.json",
    "test/intermediate_algebra/1791.json",
    "test/geometry/817.json",
    "test/intermediate_algebra/1806.json",
    "test/precalculus/398.json",
    "test/number_theory/769.json",
    "test/algebra/776.json",
    "test/algebra/1796.json",
    "test/precalculus/1202.json",
    "test/intermediate_algebra/2015.json",
    "test/number_theory/1002.json",
    "test/algebra/1339.json",
    "test/precalculus/1133.json",
    "test/intermediate_algebra/1462.json",
    "test/prealgebra/1640.json",
    "test/algebra/2043.json",
    "test/counting_and_probability/731.json",
    "test/number_theory/1000.json",
    "test/number_theory/13.json",
    "test/precalculus/323.json",
    "test/algebra/2780.json",
    "test/precalculus/703.json",
    "test/intermediate_algebra/158.json",
    "test/prealgebra/1930.json",
    "test/counting_and_probability/1003.json",
    "test/intermediate_algebra/1279.json",
    "test/number_theory/1128.json",
    "test/intermediate_algebra/1467.json",
    "test/number_theory/1090.json",
    "test/prealgebra/1203.json",
    "test/intermediate_algebra/1350.json",
    "test/prealgebra/1128.json",
    "test/algebra/2779.json",
    "test/intermediate_algebra/1930.json",
    "test/geometry/561.json",
    "test/intermediate_algebra/1508.json",
)

LIVECODEBENCH_V6_HARD_SOURCE_LOCK = ReleaseWorkloadSourceLock(
    workload_id="livecodebench_v6_hard",
    repository_revision=LIVECODEBENCH_V6_HARD_REPOSITORY_REVISION,
    raw_file_sha256=LIVECODEBENCH_V6_HARD_RAW_FILE_SHA256,
    raw_row_count=175,
    selected_row_count=80,
    selected_rows_sha256=LIVECODEBENCH_V6_HARD_FORMAL_SAMPLES_SHA256,
    protocol_sha256=FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"].sha256,
)

MATH500_LEVEL5_SOURCE_LOCK = ReleaseWorkloadSourceLock(
    workload_id="math500_level5",
    repository_revision=MATH500_LEVEL5_REPOSITORY_REVISION,
    raw_file_sha256=MATH500_LEVEL5_RAW_FILE_SHA256,
    raw_row_count=500,
    selected_row_count=134,
    selected_rows_sha256=MATH500_LEVEL5_FORMAL_SAMPLES_SHA256,
    protocol_sha256=FORMAL_WORKLOAD_PROTOCOLS["math500_level5"].sha256,
)

# Caller parameters cannot populate this tuple.  A source is added only by a
# reviewed release change that registers exact raw bytes and selected rows.
RELEASE_FORMAL_WORKLOAD_SOURCES: tuple[ReleaseWorkloadSourceLock, ...] = (
    LIVECODEBENCH_V6_HARD_SOURCE_LOCK,
    MATH500_LEVEL5_SOURCE_LOCK,
)


@dataclass(frozen=True)
class FormalWorkloadSample:
    source_row_id: str
    sample_id: str
    prompt: str
    seed: int

    def __post_init__(self) -> None:
        if not self.source_row_id or not self.sample_id:
            raise ValueError("formal workload sample identity must be non-empty")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("formal workload prompt must be non-empty")
        if self.prompt != self.prompt.strip():
            raise ValueError("formal workload prompts cannot be silently trimmed")
        if unicodedata.normalize("NFC", self.prompt) != self.prompt:
            raise ValueError("formal workload prompts must already be NFC normalized")
        if "\x00" in self.prompt:
            raise ValueError("formal workload prompts cannot contain NUL")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("formal workload seed must be an integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_row_id": self.source_row_id,
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_mapping("formal workload sample", value)
        if set(row) != {"source_row_id", "sample_id", "prompt", "seed"}:
            raise ValueError("formal workload sample fields differ from schema")
        return cls(
            source_row_id=row["source_row_id"],  # type: ignore[arg-type]
            sample_id=row["sample_id"],  # type: ignore[arg-type]
            prompt=row["prompt"],  # type: ignore[arg-type]
            seed=row["seed"],  # type: ignore[arg-type]
        )

    def as_prompt_sample(self) -> PromptSample:
        return PromptSample(
            sample_id=self.sample_id,
            prompt=self.prompt,
            seed=self.seed,
        )


def formal_workload_samples_sha256(
    samples: Sequence[FormalWorkloadSample],
) -> str:
    if not samples:
        raise ValueError("formal workload selection cannot be empty")
    return content_sha256([sample.to_dict() for sample in samples])


@dataclass(frozen=True)
class FormalWorkloadAuthority:
    schema_version: int
    kind: str
    workload_id: FormalWorkloadId
    raw_source_path: str
    raw_file_sha256: str
    repository_revision: str
    raw_row_count: int
    selected_row_count: int
    selected_rows_sha256: str
    source_lock_sha256: str
    protocol_sha256: str
    samples: tuple[FormalWorkloadSample, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "formal_workload_authority":
            raise ValueError("formal workload authority schema is unsupported")
        if self.workload_id not in FORMAL_WORKLOAD_PROTOCOLS:
            raise ValueError("formal workload authority names an unsupported workload")
        path = Path(self.raw_source_path)
        if not path.is_absolute() or path != path.resolve():
            raise ValueError(
                "formal workload source path must be absolute and resolved"
            )
        _require_sha256("formal raw workload", self.raw_file_sha256)
        _require_git_commit("formal workload revision", self.repository_revision)
        _require_sha256("formal selected workload", self.selected_rows_sha256)
        _require_sha256("formal source lock", self.source_lock_sha256)
        _require_sha256("formal workload protocol", self.protocol_sha256)
        _require_positive_int("formal raw row count", self.raw_row_count)
        _require_positive_int("formal selected row count", self.selected_row_count)
        if self.selected_row_count != len(self.samples):
            raise ValueError("formal workload selection count changed")
        if self.selected_rows_sha256 != formal_workload_samples_sha256(self.samples):
            raise ValueError("formal workload selection digest changed")
        if self.protocol_sha256 != FORMAL_WORKLOAD_PROTOCOLS[self.workload_id].sha256:
            raise ValueError("formal workload authority uses another protocol")

    @property
    def prompts(self) -> tuple[PromptSample, ...]:
        return tuple(sample.as_prompt_sample() for sample in self.samples)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "workload_id": self.workload_id,
            "raw_source_path": self.raw_source_path,
            "raw_file_sha256": self.raw_file_sha256,
            "repository_revision": self.repository_revision,
            "raw_row_count": self.raw_row_count,
            "selected_row_count": self.selected_row_count,
            "selected_rows_sha256": self.selected_rows_sha256,
            "source_lock_sha256": self.source_lock_sha256,
            "protocol_sha256": self.protocol_sha256,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_mapping("formal workload authority", value)
        expected = {
            "schema_version",
            "kind",
            "workload_id",
            "raw_source_path",
            "raw_file_sha256",
            "repository_revision",
            "raw_row_count",
            "selected_row_count",
            "selected_rows_sha256",
            "source_lock_sha256",
            "protocol_sha256",
            "samples",
        }
        if set(row) != expected:
            raise ValueError("formal workload authority fields differ from schema")
        samples = _strict_sequence("formal workload authority samples", row["samples"])
        return cls(
            schema_version=row["schema_version"],  # type: ignore[arg-type]
            kind=row["kind"],  # type: ignore[arg-type]
            workload_id=row["workload_id"],  # type: ignore[arg-type]
            raw_source_path=row["raw_source_path"],  # type: ignore[arg-type]
            raw_file_sha256=row["raw_file_sha256"],  # type: ignore[arg-type]
            repository_revision=row["repository_revision"],  # type: ignore[arg-type]
            raw_row_count=row["raw_row_count"],  # type: ignore[arg-type]
            selected_row_count=row["selected_row_count"],  # type: ignore[arg-type]
            selected_rows_sha256=row["selected_rows_sha256"],  # type: ignore[arg-type]
            source_lock_sha256=row["source_lock_sha256"],  # type: ignore[arg-type]
            protocol_sha256=row["protocol_sha256"],  # type: ignore[arg-type]
            samples=tuple(FormalWorkloadSample.from_dict(item) for item in samples),
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class LiveCodeBenchV6HardVerificationMetadata:
    """Path-free, publishable verification record for the locked hard split."""

    schema_version: int
    kind: Literal["livecodebench_v6_hard_verification_metadata"]
    workload_id: Literal["livecodebench_v6_hard"]
    repository: str
    repository_revision: str
    dataset_config: str
    split: str
    raw_file_name: str
    raw_file_sha256: str
    raw_row_count: int
    selected_row_count: int
    selected_question_ids: tuple[str, ...]
    selected_raw_rows_encoding: Literal[
        "registry_content_sha256_full_selected_source_rows_v1"
    ]
    selected_raw_rows_sha256: str
    formal_samples_sha256: str
    source_lock_sha256: str
    protocol_sha256: str
    tokenizer_statistics_status: Literal["PENDING_TOKENIZER_REVISION_LOCK"]
    tokenizer_revision: None
    prompt_token_statistics: None

    def __post_init__(self) -> None:
        protocol = FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"]
        if (
            self.schema_version != 1
            or self.kind != "livecodebench_v6_hard_verification_metadata"
            or self.workload_id != "livecodebench_v6_hard"
            or self.repository != protocol.repository
            or self.dataset_config != protocol.dataset_config
            or self.split != protocol.split
            or self.raw_file_name != "test6.jsonl"
            or self.selected_raw_rows_encoding
            != "registry_content_sha256_full_selected_source_rows_v1"
            or self.tokenizer_statistics_status != "PENDING_TOKENIZER_REVISION_LOCK"
            or self.tokenizer_revision is not None
            or self.prompt_token_statistics is not None
        ):
            raise ValueError("LiveCodeBench verification metadata identity differs")
        _require_git_commit(
            "LiveCodeBench verification revision",
            self.repository_revision,
        )
        for label, digest in (
            ("raw file", self.raw_file_sha256),
            ("selected raw rows", self.selected_raw_rows_sha256),
            ("formal samples", self.formal_samples_sha256),
            ("source lock", self.source_lock_sha256),
            ("protocol", self.protocol_sha256),
        ):
            _require_sha256(f"LiveCodeBench verification {label}", digest)
        _require_positive_int("LiveCodeBench raw row count", self.raw_row_count)
        _require_positive_int(
            "LiveCodeBench selected row count",
            self.selected_row_count,
        )
        if (
            type(self.selected_question_ids) is not tuple
            or len(self.selected_question_ids) != self.selected_row_count
            or len(set(self.selected_question_ids)) != self.selected_row_count
        ):
            raise ValueError("LiveCodeBench selected question ID coverage differs")
        for question_id in self.selected_question_ids:
            if (
                not isinstance(question_id, str)
                or not question_id
                or question_id != question_id.strip()
                or unicodedata.normalize("NFC", question_id) != question_id
            ):
                raise ValueError(
                    "LiveCodeBench selected question IDs must be unique NFC text"
                )
        if self.selected_row_count > self.raw_row_count:
            raise ValueError("LiveCodeBench selection exceeds its raw source")
        if self.protocol_sha256 != protocol.sha256:
            raise ValueError("LiveCodeBench verification protocol differs")
        if (
            self.repository_revision != LIVECODEBENCH_V6_HARD_REPOSITORY_REVISION
            or self.raw_file_sha256 != LIVECODEBENCH_V6_HARD_RAW_FILE_SHA256
            or self.raw_row_count != 175
            or self.selected_row_count != 80
            or self.selected_question_ids != LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS
            or self.selected_raw_rows_sha256
            != LIVECODEBENCH_V6_HARD_SELECTED_RAW_ROWS_SHA256
            or self.formal_samples_sha256 != LIVECODEBENCH_V6_HARD_FORMAL_SAMPLES_SHA256
            or self.source_lock_sha256 != LIVECODEBENCH_V6_HARD_SOURCE_LOCK.sha256
        ):
            raise ValueError("LiveCodeBench verification release identity differs")

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "workload_id": self.workload_id,
            "repository": self.repository,
            "repository_revision": self.repository_revision,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "raw_file_name": self.raw_file_name,
            "raw_file_sha256": self.raw_file_sha256,
            "raw_row_count": self.raw_row_count,
            "selected_row_count": self.selected_row_count,
            "selected_question_ids": list(self.selected_question_ids),
            "selected_raw_rows_encoding": self.selected_raw_rows_encoding,
            "selected_raw_rows_sha256": self.selected_raw_rows_sha256,
            "formal_samples_sha256": self.formal_samples_sha256,
            "source_lock_sha256": self.source_lock_sha256,
            "protocol_sha256": self.protocol_sha256,
            "tokenizer_statistics_status": self.tokenizer_statistics_status,
            "tokenizer_revision": self.tokenizer_revision,
            "prompt_token_statistics": self.prompt_token_statistics,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {"verification_metadata_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = dict(_strict_mapping("LiveCodeBench verification metadata", value))
        expected_fields = set(cls.__dataclass_fields__) | {
            "verification_metadata_sha256"
        }
        if set(row) != expected_fields:
            raise ValueError("LiveCodeBench verification metadata fields differ")
        declared = _require_sha256(
            "LiveCodeBench verification metadata",
            row.pop("verification_metadata_sha256"),
        )
        question_ids = _strict_sequence(
            "LiveCodeBench selected question IDs",
            row.pop("selected_question_ids"),
        )
        metadata = cls(
            **row,
            selected_question_ids=tuple(question_ids),
        )  # type: ignore[arg-type]
        if metadata.sha256 != declared:
            raise ValueError("LiveCodeBench verification metadata digest differs")
        return metadata


RELEASE_LIVECODEBENCH_V6_HARD_VERIFICATION_METADATA = (
    LiveCodeBenchV6HardVerificationMetadata(
        schema_version=1,
        kind="livecodebench_v6_hard_verification_metadata",
        workload_id="livecodebench_v6_hard",
        repository=FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"].repository,
        repository_revision=LIVECODEBENCH_V6_HARD_REPOSITORY_REVISION,
        dataset_config=FORMAL_WORKLOAD_PROTOCOLS[
            "livecodebench_v6_hard"
        ].dataset_config,
        split=FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"].split,
        raw_file_name="test6.jsonl",
        raw_file_sha256=LIVECODEBENCH_V6_HARD_RAW_FILE_SHA256,
        raw_row_count=175,
        selected_row_count=80,
        selected_question_ids=LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS,
        selected_raw_rows_encoding=(
            "registry_content_sha256_full_selected_source_rows_v1"
        ),
        selected_raw_rows_sha256=(LIVECODEBENCH_V6_HARD_SELECTED_RAW_ROWS_SHA256),
        formal_samples_sha256=LIVECODEBENCH_V6_HARD_FORMAL_SAMPLES_SHA256,
        source_lock_sha256=LIVECODEBENCH_V6_HARD_SOURCE_LOCK.sha256,
        protocol_sha256=FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"].sha256,
        tokenizer_statistics_status="PENDING_TOKENIZER_REVISION_LOCK",
        tokenizer_revision=None,
        prompt_token_statistics=None,
    )
)


@dataclass(frozen=True)
class Math500Level5VerificationMetadata:
    """Path-free verification record for the exact integer level-5 rows."""

    schema_version: int
    kind: Literal["math500_level5_verification_metadata"]
    workload_id: Literal["math500_level5"]
    repository: str
    repository_revision: str
    dataset_config: str
    split: str
    raw_file_name: str
    raw_file_sha256: str
    raw_row_count: int
    filter_field: Literal["level"]
    filter_value: Literal[5]
    selected_row_count: int
    selected_source_row_ids: tuple[str, ...]
    selected_raw_rows_encoding: Literal[
        "registry_content_sha256_full_selected_source_rows_v1"
    ]
    selected_raw_rows_sha256: str
    formal_samples_sha256: str
    source_lock_sha256: str
    protocol_sha256: str
    tokenizer_statistics_status: Literal["PENDING_TOKENIZER_REVISION_LOCK"]
    tokenizer_revision: None
    prompt_token_statistics: None

    def __post_init__(self) -> None:
        protocol = FORMAL_WORKLOAD_PROTOCOLS["math500_level5"]
        if (
            self.schema_version != 1
            or self.kind != "math500_level5_verification_metadata"
            or self.workload_id != "math500_level5"
            or self.repository != protocol.repository
            or self.dataset_config != protocol.dataset_config
            or self.split != protocol.split
            or self.raw_file_name != "test.jsonl"
            or self.filter_field != protocol.filter_field
            or type(self.filter_value) is not int
            or self.filter_value != protocol.filter_value
            or self.selected_raw_rows_encoding
            != "registry_content_sha256_full_selected_source_rows_v1"
            or self.tokenizer_statistics_status != "PENDING_TOKENIZER_REVISION_LOCK"
            or self.tokenizer_revision is not None
            or self.prompt_token_statistics is not None
        ):
            raise ValueError("MATH-500 verification metadata identity differs")
        _require_git_commit("MATH-500 verification revision", self.repository_revision)
        for label, digest in (
            ("raw file", self.raw_file_sha256),
            ("selected raw rows", self.selected_raw_rows_sha256),
            ("formal samples", self.formal_samples_sha256),
            ("source lock", self.source_lock_sha256),
            ("protocol", self.protocol_sha256),
        ):
            _require_sha256(f"MATH-500 verification {label}", digest)
        _require_positive_int("MATH-500 raw row count", self.raw_row_count)
        _require_positive_int("MATH-500 selected row count", self.selected_row_count)
        if (
            type(self.selected_source_row_ids) is not tuple
            or len(self.selected_source_row_ids) != self.selected_row_count
            or len(set(self.selected_source_row_ids)) != self.selected_row_count
        ):
            raise ValueError("MATH-500 selected source ID coverage differs")
        for source_id in self.selected_source_row_ids:
            if (
                type(source_id) is not str
                or not source_id
                or source_id != source_id.strip()
                or unicodedata.normalize("NFC", source_id) != source_id
            ):
                raise ValueError("MATH-500 selected source IDs must be unique NFC text")
        if self.selected_row_count > self.raw_row_count:
            raise ValueError("MATH-500 selection exceeds its raw source")
        if self.protocol_sha256 != protocol.sha256:
            raise ValueError("MATH-500 verification protocol differs")
        if (
            self.repository_revision != MATH500_LEVEL5_REPOSITORY_REVISION
            or self.raw_file_sha256 != MATH500_LEVEL5_RAW_FILE_SHA256
            or self.raw_row_count != 500
            or self.selected_row_count != 134
            or self.selected_source_row_ids != MATH500_LEVEL5_SELECTED_SOURCE_ROW_IDS
            or content_sha256(list(self.selected_source_row_ids))
            != MATH500_LEVEL5_SELECTED_SOURCE_ROW_IDS_SHA256
            or self.selected_raw_rows_sha256 != MATH500_LEVEL5_SELECTED_RAW_ROWS_SHA256
            or self.formal_samples_sha256 != MATH500_LEVEL5_FORMAL_SAMPLES_SHA256
            or self.source_lock_sha256 != MATH500_LEVEL5_SOURCE_LOCK.sha256
        ):
            raise ValueError("MATH-500 verification release identity differs")

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "workload_id": self.workload_id,
            "repository": self.repository,
            "repository_revision": self.repository_revision,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "raw_file_name": self.raw_file_name,
            "raw_file_sha256": self.raw_file_sha256,
            "raw_row_count": self.raw_row_count,
            "filter_field": self.filter_field,
            "filter_value": self.filter_value,
            "selected_row_count": self.selected_row_count,
            "selected_source_row_ids": list(self.selected_source_row_ids),
            "selected_raw_rows_encoding": self.selected_raw_rows_encoding,
            "selected_raw_rows_sha256": self.selected_raw_rows_sha256,
            "formal_samples_sha256": self.formal_samples_sha256,
            "source_lock_sha256": self.source_lock_sha256,
            "protocol_sha256": self.protocol_sha256,
            "tokenizer_statistics_status": self.tokenizer_statistics_status,
            "tokenizer_revision": self.tokenizer_revision,
            "prompt_token_statistics": self.prompt_token_statistics,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {"verification_metadata_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = dict(_strict_mapping("MATH-500 verification metadata", value))
        expected_fields = set(cls.__dataclass_fields__) | {
            "verification_metadata_sha256"
        }
        if set(row) != expected_fields:
            raise ValueError("MATH-500 verification metadata fields differ")
        declared = _require_sha256(
            "MATH-500 verification metadata",
            row.pop("verification_metadata_sha256"),
        )
        source_ids = _strict_sequence(
            "MATH-500 selected source IDs",
            row.pop("selected_source_row_ids"),
        )
        metadata = cls(
            **row,
            selected_source_row_ids=tuple(source_ids),
        )  # type: ignore[arg-type]
        if metadata.sha256 != declared:
            raise ValueError("MATH-500 verification metadata digest differs")
        return metadata


RELEASE_MATH500_LEVEL5_VERIFICATION_METADATA = Math500Level5VerificationMetadata(
    schema_version=1,
    kind="math500_level5_verification_metadata",
    workload_id="math500_level5",
    repository=FORMAL_WORKLOAD_PROTOCOLS["math500_level5"].repository,
    repository_revision=MATH500_LEVEL5_REPOSITORY_REVISION,
    dataset_config=FORMAL_WORKLOAD_PROTOCOLS["math500_level5"].dataset_config,
    split=FORMAL_WORKLOAD_PROTOCOLS["math500_level5"].split,
    raw_file_name="test.jsonl",
    raw_file_sha256=MATH500_LEVEL5_RAW_FILE_SHA256,
    raw_row_count=500,
    filter_field="level",
    filter_value=5,
    selected_row_count=134,
    selected_source_row_ids=MATH500_LEVEL5_SELECTED_SOURCE_ROW_IDS,
    selected_raw_rows_encoding=("registry_content_sha256_full_selected_source_rows_v1"),
    selected_raw_rows_sha256=MATH500_LEVEL5_SELECTED_RAW_ROWS_SHA256,
    formal_samples_sha256=MATH500_LEVEL5_FORMAL_SAMPLES_SHA256,
    source_lock_sha256=MATH500_LEVEL5_SOURCE_LOCK.sha256,
    protocol_sha256=FORMAL_WORKLOAD_PROTOCOLS["math500_level5"].sha256,
    tokenizer_statistics_status="PENDING_TOKENIZER_REVISION_LOCK",
    tokenizer_revision=None,
    prompt_token_statistics=None,
)


def formal_workload_authority_artifact_id(workload_id: FormalWorkloadId) -> str:
    if workload_id not in FORMAL_WORKLOAD_PROTOCOLS:
        raise ValueError("formal workload artifact names an unsupported workload")
    return f"{FORMAL_WORKLOAD_AUTHORITY_ARTIFACT_PREFIX}{workload_id}"


def formal_workload_authority_cli_artifact(
    authority: FormalWorkloadAuthority,
) -> dict[str, object]:
    """Serialize the diagnostic binding carried by the verified content set.

    The wrapper is deliberately non-authorizing.  A serving reducer must pair
    it with the root-verified workload authorization and replay the raw source.
    """

    if type(authority) is not FormalWorkloadAuthority:
        raise TypeError("formal workload CLI binding requires an exact authority")
    return {
        "schema_version": 1,
        "kind": "formal_workload_cli_binding",
        "formal_execution_authorized": False,
        "authority_sha256": authority.sha256,
        "authority": authority.to_dict(),
    }


def formal_workload_authority_from_cli_artifact(
    value: object,
) -> FormalWorkloadAuthority:
    """Strictly decode, but do not authorize, one path-bound workload wrapper."""

    row = _strict_mapping("formal workload CLI binding", value)
    if set(row) != {
        "schema_version",
        "kind",
        "formal_execution_authorized",
        "authority_sha256",
        "authority",
    }:
        raise ValueError("formal workload CLI binding fields differ from schema")
    if (
        row["schema_version"] != 1
        or row["kind"] != "formal_workload_cli_binding"
        or row["formal_execution_authorized"] is not False
    ):
        raise ValueError("formal workload CLI binding is diagnostic-only schema-v1")
    declared = _require_sha256(
        "formal workload CLI binding authority", row["authority_sha256"]
    )
    authority = FormalWorkloadAuthority.from_dict(row["authority"])
    if authority.sha256 != declared:
        raise ValueError("formal workload CLI binding changed authority identity")
    return authority


def _release_lock(workload_id: FormalWorkloadId) -> ReleaseWorkloadSourceLock:
    if workload_id not in FORMAL_WORKLOAD_PROTOCOLS:
        raise ValueError("formal workload is not registered")
    matches = tuple(
        lock
        for lock in RELEASE_FORMAL_WORKLOAD_SOURCES
        if lock.workload_id == workload_id
    )
    if not matches:
        raise FormalWorkloadAuthorityBlocked(
            FORMAL_WORKLOAD_SOURCE_ALLOWLIST_EMPTY_REASON
            if not RELEASE_FORMAL_WORKLOAD_SOURCES
            else FORMAL_WORKLOAD_SOURCE_NOT_REGISTERED_REASON
        )
    if len(matches) != 1:
        raise RuntimeError("formal workload release source lock is ambiguous")
    lock = matches[0]
    lock.__post_init__()
    return lock


def _read_stable_raw(path_value: str | Path) -> tuple[Path, bytes]:
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError("formal workload source path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise FormalWorkloadAuthorityBlocked(
            FORMAL_WORKLOAD_LOCAL_SOURCE_MISSING_REASON
        ) from error
    if resolved != path:
        raise ValueError("formal workload source path must be resolved and non-symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("formal workload source cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_RAW_BYTES
        ):
            raise ValueError("formal workload source must be a bounded regular file")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("formal workload source changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("formal workload source grew while being read")
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
            raise ValueError("formal workload source changed during coordinated read")
    finally:
        os.close(descriptor)
    return path, b"".join(chunks)


class _FormalWorkloadWholeDocumentDecodeError(ValueError):
    """Internal signal used to distinguish JSONL from a JSON document."""


def _load_strict_json(raw: bytes, *, label: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} has duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite {value}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _FormalWorkloadWholeDocumentDecodeError(
            f"{label} is not strict UTF-8 JSON"
        ) from error


def _load_raw_jsonl_rows(raw: bytes) -> tuple[Mapping[str, object], ...]:
    records = raw.splitlines()
    if not records or any(not record.strip() for record in records):
        raise ValueError("formal workload JSONL contains an empty record")
    rows: list[Mapping[str, object]] = []
    for index, record in enumerate(records):
        try:
            value = _load_strict_json(
                record,
                label=f"formal workload JSONL row {index}",
            )
        except _FormalWorkloadWholeDocumentDecodeError as error:
            raise ValueError(
                f"formal workload JSONL row {index} is not strict UTF-8 JSON"
            ) from error
        rows.append(_strict_mapping(f"formal workload JSONL row {index}", value))
    return tuple(rows)


def _load_locked_source_rows(
    raw: bytes,
    *,
    protocol: FormalWorkloadProtocol,
    lock: ReleaseWorkloadSourceLock,
) -> tuple[Mapping[str, object], ...]:
    """Decode either the legacy canonical envelope or locked upstream JSONL."""

    try:
        whole_value = _load_strict_json(raw, label="formal workload source")
    except _FormalWorkloadWholeDocumentDecodeError:
        rows = _load_raw_jsonl_rows(raw)
    else:
        whole = _strict_mapping("formal workload source", whole_value)
        envelope_fields = {
            "schema_version",
            "repository",
            "repository_revision",
            "dataset_config",
            "split",
            "rows",
        }
        raw_row_fields = {
            protocol.identity_field,
            protocol.prompt_field,
            protocol.filter_field,
        }
        if set(whole) == envelope_fields:
            if (
                whole["schema_version"] != 1
                or whole["repository"] != protocol.repository
                or whole["repository_revision"] != lock.repository_revision
                or whole["dataset_config"] != protocol.dataset_config
                or whole["split"] != protocol.split
            ):
                raise ValueError(
                    "formal workload source metadata differs from the release lock"
                )
            rows_value = _strict_sequence("formal workload rows", whole["rows"])
            rows = tuple(
                _strict_mapping(f"formal workload row {index}", value)
                for index, value in enumerate(rows_value)
            )
        elif raw_row_fields <= set(whole):
            # A one-record JSONL source is also a complete JSON document.
            rows = (whole,)
        else:
            raise ValueError("formal workload source envelope fields differ")
    if len(rows) != lock.raw_row_count:
        raise ValueError("formal workload raw row count differs from the release lock")
    return rows


def _sample_from_row(
    protocol: FormalWorkloadProtocol,
    row: Mapping[str, object],
) -> FormalWorkloadSample | None:
    identity = row.get(protocol.identity_field)
    prompt = row.get(protocol.prompt_field)
    filter_value = row.get(protocol.filter_field)
    if not isinstance(identity, str) or not identity:
        raise ValueError("formal workload row lacks its stable source identity")
    if unicodedata.normalize("NFC", identity) != identity:
        raise ValueError("formal workload source identities must be NFC normalized")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("formal workload row lacks its exact prompt")
    if type(filter_value) is not type(protocol.filter_value):
        raise TypeError(
            "formal workload row filter field differs from its exact JSON type"
        )
    if filter_value != protocol.filter_value:
        return None
    digest = content_sha256(
        {
            "workload_id": protocol.workload_id,
            "source_row_id": identity,
            "prompt": prompt,
            "prompt_compiler": protocol.prompt_compiler,
        }
    )
    return FormalWorkloadSample(
        source_row_id=identity,
        sample_id=f"{protocol.workload_id}-{digest[:16]}",
        prompt=prompt,
        seed=int(digest[:8], 16),
    )


def _select_all_rows(
    protocol: FormalWorkloadProtocol,
    rows_value: object,
) -> tuple[FormalWorkloadSample, ...]:
    samples, _selected_rows = _select_all_rows_with_raw(protocol, rows_value)
    return samples


def _select_all_rows_with_raw(
    protocol: FormalWorkloadProtocol,
    rows_value: object,
) -> tuple[
    tuple[FormalWorkloadSample, ...],
    tuple[Mapping[str, object], ...],
]:
    rows = _strict_sequence("formal workload rows", rows_value)
    if not rows:
        raise ValueError("formal workload raw source cannot be empty")
    samples: list[FormalWorkloadSample] = []
    selected_rows: list[Mapping[str, object]] = []
    raw_ids: set[str] = set()
    for index, value in enumerate(rows):
        row = _strict_mapping(f"formal workload row {index}", value)
        identity = row.get(protocol.identity_field)
        if not isinstance(identity, str) or identity in raw_ids:
            raise ValueError("formal workload source identities must be unique")
        raw_ids.add(identity)
        sample = _sample_from_row(protocol, row)
        if sample is not None:
            samples.append(sample)
            selected_rows.append(row)
    if not samples:
        raise FormalWorkloadAuthorityBlocked(FORMAL_WORKLOAD_FILTER_EMPTY_REASON)
    if len({sample.prompt for sample in samples}) != len(samples):
        raise ValueError("formal workload selected prompts must be content-unique")
    return tuple(samples), tuple(selected_rows)


def _bind_with_source_lock(
    workload_id: FormalWorkloadId,
    raw_source_path: str | Path,
    *,
    lock: ReleaseWorkloadSourceLock,
    expected_raw_file_size: int | None,
) -> FormalWorkloadAuthority:
    """Bind all exact protocol matches under one already-authorized lock."""

    lock.__post_init__()
    if lock.workload_id != workload_id:
        raise ValueError("formal workload source lock names another workload")
    path, raw = _read_stable_raw(raw_source_path)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if raw_sha256 != lock.raw_file_sha256 or (
        expected_raw_file_size is not None and len(raw) != expected_raw_file_size
    ):
        raise ValueError(
            "formal workload raw bytes differ from the release source lock"
        )
    protocol = FORMAL_WORKLOAD_PROTOCOLS[workload_id]
    rows = _load_locked_source_rows(raw, protocol=protocol, lock=lock)
    samples = _select_all_rows(protocol, rows)
    selected_sha256 = formal_workload_samples_sha256(samples)
    if (
        len(samples) != lock.selected_row_count
        or selected_sha256 != lock.selected_rows_sha256
    ):
        raise ValueError("formal workload filtered rows differ from the release lock")
    return FormalWorkloadAuthority(
        schema_version=1,
        kind="formal_workload_authority",
        workload_id=workload_id,
        raw_source_path=str(path),
        raw_file_sha256=raw_sha256,
        repository_revision=lock.repository_revision,
        raw_row_count=len(rows),
        selected_row_count=len(samples),
        selected_rows_sha256=selected_sha256,
        source_lock_sha256=lock.sha256,
        protocol_sha256=protocol.sha256,
        samples=samples,
    )


def bind_formal_workload_authority(
    workload_id: FormalWorkloadId,
    raw_source_path: str | Path,
) -> FormalWorkloadAuthority:
    """Bind under the legacy source allowlist, which remains fail-closed."""

    return _bind_with_source_lock(
        workload_id,
        raw_source_path,
        lock=_release_lock(workload_id),
        expected_raw_file_size=None,
    )


def bind_authorized_formal_workload_authority(
    workload_id: FormalWorkloadId,
    raw_source_path: str | Path,
    *,
    authorization: VerifiedReleaseWorkloadSources,
) -> FormalWorkloadAuthority:
    """Bind exact raw bytes under a verified offline-root authorization."""

    if type(authorization) is not VerifiedReleaseWorkloadSources:
        raise TypeError("authorized workload binding requires a verified token")
    source = authorization.source(workload_id)
    protocol = FORMAL_WORKLOAD_PROTOCOLS[workload_id]
    if (
        source.repository != protocol.repository
        or source.dataset_config != protocol.dataset_config
        or source.split != protocol.split
        or source.filter_field != protocol.filter_field
        or type(source.filter_value) is not type(protocol.filter_value)
        or source.filter_value != protocol.filter_value
        or source.prompt_compiler != protocol.prompt_compiler
        or source.selection_policy != protocol.selection_policy
        or source.protocol_sha256 != protocol.sha256
    ):
        raise ValueError("authorized workload source differs from formal protocol")
    lock = ReleaseWorkloadSourceLock(
        workload_id=workload_id,
        repository_revision=source.repository_revision,
        raw_file_sha256=source.raw_file_sha256,
        raw_row_count=source.raw_row_count,
        selected_row_count=source.selected_row_count,
        selected_rows_sha256=source.selected_rows_sha256,
        protocol_sha256=source.protocol_sha256,
    )
    return _bind_with_source_lock(
        workload_id,
        raw_source_path,
        lock=lock,
        expected_raw_file_size=source.raw_file_size,
    )


def revalidate_formal_workload_authority(
    authority: FormalWorkloadAuthority,
) -> FormalWorkloadAuthority:
    """Replay a frozen binding and reject path, bytes, lock, or filter drift."""

    if type(authority) is not FormalWorkloadAuthority:
        raise TypeError("formal workload revalidation requires an exact authority")
    authority.__post_init__()
    rebound = bind_formal_workload_authority(
        authority.workload_id,
        authority.raw_source_path,
    )
    if rebound != authority or rebound.sha256 != authority.sha256:
        raise ValueError("formal workload authority changed during revalidation")
    return rebound


def revalidate_authorized_formal_workload_authority(
    authority: FormalWorkloadAuthority,
    *,
    authorization: VerifiedReleaseWorkloadSources,
) -> FormalWorkloadAuthority:
    """Replay an authorized local binding and reject TOCTOU or wrapper drift."""

    if type(authority) is not FormalWorkloadAuthority:
        raise TypeError("authorized workload replay requires an exact authority")
    rebound = bind_authorized_formal_workload_authority(
        authority.workload_id,
        authority.raw_source_path,
        authorization=authorization,
    )
    if rebound != authority or rebound.sha256 != authority.sha256:
        raise ValueError("authorized formal workload changed during revalidation")
    return rebound


def build_livecodebench_v6_hard_verification_metadata(
    authority: FormalWorkloadAuthority,
) -> LiveCodeBenchV6HardVerificationMetadata:
    """Deep-reopen the release source and emit its path-free verification row."""

    if (
        type(authority) is not FormalWorkloadAuthority
        or authority.workload_id != "livecodebench_v6_hard"
    ):
        raise TypeError(
            "LiveCodeBench verification requires its exact formal authority"
        )
    lock = LIVECODEBENCH_V6_HARD_SOURCE_LOCK
    path, raw = _read_stable_raw(authority.raw_source_path)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    protocol = FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"]
    rows = _load_locked_source_rows(raw, protocol=protocol, lock=lock)
    samples, selected_rows = _select_all_rows_with_raw(protocol, rows)
    selected_raw_rows_sha256 = content_sha256(list(selected_rows))
    rebound = FormalWorkloadAuthority(
        schema_version=1,
        kind="formal_workload_authority",
        workload_id="livecodebench_v6_hard",
        raw_source_path=str(path),
        raw_file_sha256=raw_sha256,
        repository_revision=lock.repository_revision,
        raw_row_count=len(rows),
        selected_row_count=len(samples),
        selected_rows_sha256=formal_workload_samples_sha256(samples),
        source_lock_sha256=lock.sha256,
        protocol_sha256=protocol.sha256,
        samples=samples,
    )
    if rebound != authority or rebound.sha256 != authority.sha256:
        raise ValueError("LiveCodeBench authority changed during verification")
    metadata = LiveCodeBenchV6HardVerificationMetadata(
        schema_version=1,
        kind="livecodebench_v6_hard_verification_metadata",
        workload_id="livecodebench_v6_hard",
        repository=protocol.repository,
        repository_revision=lock.repository_revision,
        dataset_config=protocol.dataset_config,
        split=protocol.split,
        raw_file_name="test6.jsonl",
        raw_file_sha256=raw_sha256,
        raw_row_count=len(rows),
        selected_row_count=len(samples),
        selected_question_ids=tuple(sample.source_row_id for sample in samples),
        selected_raw_rows_encoding=(
            "registry_content_sha256_full_selected_source_rows_v1"
        ),
        selected_raw_rows_sha256=selected_raw_rows_sha256,
        formal_samples_sha256=formal_workload_samples_sha256(samples),
        source_lock_sha256=lock.sha256,
        protocol_sha256=protocol.sha256,
        tokenizer_statistics_status="PENDING_TOKENIZER_REVISION_LOCK",
        tokenizer_revision=None,
        prompt_token_statistics=None,
    )
    if metadata != RELEASE_LIVECODEBENCH_V6_HARD_VERIFICATION_METADATA:
        raise ValueError("LiveCodeBench verification differs from the release record")
    return metadata


def build_math500_level5_verification_metadata(
    authority: FormalWorkloadAuthority,
) -> Math500Level5VerificationMetadata:
    """Deep-reopen MATH-500 and verify all exact integer level-5 rows."""

    if (
        type(authority) is not FormalWorkloadAuthority
        or authority.workload_id != "math500_level5"
    ):
        raise TypeError("MATH-500 verification requires its exact formal authority")
    lock = MATH500_LEVEL5_SOURCE_LOCK
    path, raw = _read_stable_raw(authority.raw_source_path)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    protocol = FORMAL_WORKLOAD_PROTOCOLS["math500_level5"]
    rows = _load_locked_source_rows(raw, protocol=protocol, lock=lock)
    samples, selected_rows = _select_all_rows_with_raw(protocol, rows)
    rebound = FormalWorkloadAuthority(
        schema_version=1,
        kind="formal_workload_authority",
        workload_id="math500_level5",
        raw_source_path=str(path),
        raw_file_sha256=raw_sha256,
        repository_revision=lock.repository_revision,
        raw_row_count=len(rows),
        selected_row_count=len(samples),
        selected_rows_sha256=formal_workload_samples_sha256(samples),
        source_lock_sha256=lock.sha256,
        protocol_sha256=protocol.sha256,
        samples=samples,
    )
    if rebound != authority or rebound.sha256 != authority.sha256:
        raise ValueError("MATH-500 authority changed during verification")
    metadata = Math500Level5VerificationMetadata(
        schema_version=1,
        kind="math500_level5_verification_metadata",
        workload_id="math500_level5",
        repository=protocol.repository,
        repository_revision=lock.repository_revision,
        dataset_config=protocol.dataset_config,
        split=protocol.split,
        raw_file_name="test.jsonl",
        raw_file_sha256=raw_sha256,
        raw_row_count=len(rows),
        filter_field="level",
        filter_value=5,
        selected_row_count=len(samples),
        selected_source_row_ids=tuple(sample.source_row_id for sample in samples),
        selected_raw_rows_encoding=(
            "registry_content_sha256_full_selected_source_rows_v1"
        ),
        selected_raw_rows_sha256=content_sha256(list(selected_rows)),
        formal_samples_sha256=formal_workload_samples_sha256(samples),
        source_lock_sha256=lock.sha256,
        protocol_sha256=protocol.sha256,
        tokenizer_statistics_status="PENDING_TOKENIZER_REVISION_LOCK",
        tokenizer_revision=None,
        prompt_token_statistics=None,
    )
    if metadata != RELEASE_MATH500_LEVEL5_VERIFICATION_METADATA:
        raise ValueError("MATH-500 verification differs from the release record")
    return metadata


def require_authorized_formal_workload_authority(
    authority: FormalWorkloadAuthority,
    *,
    authorization: VerifiedReleaseWorkloadSources,
) -> tuple[PromptSample, ...]:
    return revalidate_authorized_formal_workload_authority(
        authority, authorization=authorization
    ).prompts


def require_formal_workload_authority(
    authority: FormalWorkloadAuthority,
) -> tuple[PromptSample, ...]:
    """Return complete selected prompts only after raw authority replay."""

    return revalidate_formal_workload_authority(authority).prompts


__all__ = [
    "FORMAL_WORKLOAD_AUTHORITY_ARTIFACT_PREFIX",
    "FORMAL_WORKLOAD_FILTER_EMPTY_REASON",
    "FORMAL_WORKLOAD_LOCAL_SOURCE_MISSING_REASON",
    "FORMAL_WORKLOAD_PROTOCOLS",
    "FORMAL_WORKLOAD_PROTOCOL_SHA256",
    "FORMAL_WORKLOAD_SOURCE_ALLOWLIST_EMPTY_REASON",
    "FORMAL_WORKLOAD_SOURCE_NOT_REGISTERED_REASON",
    "LIVECODEBENCH_V6_HARD_FORMAL_SAMPLES_SHA256",
    "LIVECODEBENCH_V6_HARD_RAW_FILE_SHA256",
    "LIVECODEBENCH_V6_HARD_REPOSITORY_REVISION",
    "LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS",
    "LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS_SHA256",
    "LIVECODEBENCH_V6_HARD_SELECTED_RAW_ROWS_SHA256",
    "LIVECODEBENCH_V6_HARD_SOURCE_LOCK",
    "MATH500_LEVEL5_FORMAL_SAMPLES_SHA256",
    "MATH500_LEVEL5_RAW_FILE_SHA256",
    "MATH500_LEVEL5_REPOSITORY_REVISION",
    "MATH500_LEVEL5_SELECTED_RAW_ROWS_SHA256",
    "MATH500_LEVEL5_SELECTED_SOURCE_ROW_IDS",
    "MATH500_LEVEL5_SELECTED_SOURCE_ROW_IDS_SHA256",
    "MATH500_LEVEL5_SOURCE_LOCK",
    "RELEASE_FORMAL_WORKLOAD_SOURCES",
    "RELEASE_LIVECODEBENCH_V6_HARD_VERIFICATION_METADATA",
    "RELEASE_MATH500_LEVEL5_VERIFICATION_METADATA",
    "FormalWorkloadAuthority",
    "FormalWorkloadAuthorityBlocked",
    "FormalWorkloadProtocol",
    "FormalWorkloadSample",
    "LiveCodeBenchV6HardVerificationMetadata",
    "Math500Level5VerificationMetadata",
    "ReleaseWorkloadSourceLock",
    "bind_authorized_formal_workload_authority",
    "bind_formal_workload_authority",
    "build_livecodebench_v6_hard_verification_metadata",
    "build_math500_level5_verification_metadata",
    "formal_workload_authority_artifact_id",
    "formal_workload_authority_cli_artifact",
    "formal_workload_authority_from_cli_artifact",
    "formal_workload_samples_sha256",
    "require_authorized_formal_workload_authority",
    "require_formal_workload_authority",
    "revalidate_authorized_formal_workload_authority",
    "revalidate_formal_workload_authority",
]
