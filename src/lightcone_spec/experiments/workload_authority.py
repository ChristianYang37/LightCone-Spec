"""Release-owned authority for formal external benchmark workloads.

Formal workload rows are local, path-bound raw inputs.  This module never
downloads a dataset and never accepts a caller-provided revision or digest as
authority.  A release-owned source lock must bind the repository revision,
exact raw bytes, full row count, filtering protocol, and complete selected-row
digest before a local file is opened.

The current release source allowlist is intentionally empty.  Consequently
LiveCodeBench v6 Hard and MATH-500 Level 5 both return a named ``BLOCKED``
outcome before a path is inspected or any output is created.  A future release
must add reviewed source locks; taking the first N rows is never part of this
formal path.
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
from typing import Literal

from lightcone_spec.experiments.data import PromptSample
from lightcone_spec.experiments.registry import content_sha256

FormalWorkloadId = Literal["livecodebench_v6_hard", "math500_level5"]

_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RAW_BYTES = 512 * 1024 * 1024

FORMAL_WORKLOAD_SOURCE_ALLOWLIST_EMPTY_REASON = "formal_workload_source_allowlist_empty"
FORMAL_WORKLOAD_SOURCE_NOT_REGISTERED_REASON = "formal_workload_source_not_registered"
FORMAL_WORKLOAD_LOCAL_SOURCE_MISSING_REASON = "formal_workload_local_source_missing"
FORMAL_WORKLOAD_FILTER_EMPTY_REASON = "formal_workload_filter_empty"


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
    filter_value: str
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
            ("filter value", self.filter_value),
            ("prompt compiler", self.prompt_compiler),
            ("selection policy", self.selection_policy),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"formal workload {label} must be non-empty")
        if self.prompt_compiler != "verbatim_nfc_no_trim_v1":
            raise ValueError("formal workload prompt compiler is unsupported")
        if self.selection_policy != "all_exact_matches_in_raw_order":
            raise ValueError("formal workload selection policy is unsupported")

    def to_dict(self) -> dict[str, str]:
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
                filter_value="Level 5",
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


# Caller parameters cannot populate this tuple.  A source is added only by a
# reviewed release change that registers exact raw bytes and selected rows.
RELEASE_FORMAL_WORKLOAD_SOURCES: tuple[ReleaseWorkloadSourceLock, ...] = ()


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

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


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


def _load_raw_envelope(raw: bytes) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"formal workload source has duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"formal workload source contains non-finite {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("formal workload source is not strict UTF-8 JSON") from error
    return _strict_mapping("formal workload source", value)


def _sample_from_row(
    protocol: FormalWorkloadProtocol,
    row: Mapping[str, object],
) -> FormalWorkloadSample | None:
    identity = row.get(protocol.identity_field)
    prompt = row.get(protocol.prompt_field)
    filter_value = row.get(protocol.filter_field)
    if not isinstance(identity, str) or not identity:
        raise ValueError("formal workload row lacks its stable source identity")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("formal workload row lacks its exact prompt")
    if not isinstance(filter_value, str):
        raise TypeError("formal workload row lacks its exact string filter field")
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
    rows = _strict_sequence("formal workload rows", rows_value)
    if not rows:
        raise ValueError("formal workload raw source cannot be empty")
    samples: list[FormalWorkloadSample] = []
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
    if not samples:
        raise FormalWorkloadAuthorityBlocked(FORMAL_WORKLOAD_FILTER_EMPTY_REASON)
    if len({sample.prompt for sample in samples}) != len(samples):
        raise ValueError("formal workload selected prompts must be content-unique")
    return tuple(samples)


def bind_formal_workload_authority(
    workload_id: FormalWorkloadId,
    raw_source_path: str | Path,
) -> FormalWorkloadAuthority:
    """Bind all exact protocol matches from one release-registered local file."""

    lock = _release_lock(workload_id)
    path, raw = _read_stable_raw(raw_source_path)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if raw_sha256 != lock.raw_file_sha256:
        raise ValueError(
            "formal workload raw bytes differ from the release source lock"
        )
    protocol = FORMAL_WORKLOAD_PROTOCOLS[workload_id]
    envelope = _load_raw_envelope(raw)
    expected_keys = {
        "schema_version",
        "repository",
        "repository_revision",
        "dataset_config",
        "split",
        "rows",
    }
    if set(envelope) != expected_keys:
        raise ValueError("formal workload source envelope fields differ")
    if (
        envelope["schema_version"] != 1
        or envelope["repository"] != protocol.repository
        or envelope["repository_revision"] != lock.repository_revision
        or envelope["dataset_config"] != protocol.dataset_config
        or envelope["split"] != protocol.split
    ):
        raise ValueError(
            "formal workload source metadata differs from the release lock"
        )
    rows = _strict_sequence("formal workload rows", envelope["rows"])
    if len(rows) != lock.raw_row_count:
        raise ValueError("formal workload raw row count differs from the release lock")
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


def require_formal_workload_authority(
    authority: FormalWorkloadAuthority,
) -> tuple[PromptSample, ...]:
    """Return complete selected prompts only after raw authority replay."""

    return revalidate_formal_workload_authority(authority).prompts


__all__ = [
    "FORMAL_WORKLOAD_FILTER_EMPTY_REASON",
    "FORMAL_WORKLOAD_LOCAL_SOURCE_MISSING_REASON",
    "FORMAL_WORKLOAD_PROTOCOLS",
    "FORMAL_WORKLOAD_PROTOCOL_SHA256",
    "FORMAL_WORKLOAD_SOURCE_ALLOWLIST_EMPTY_REASON",
    "FORMAL_WORKLOAD_SOURCE_NOT_REGISTERED_REASON",
    "RELEASE_FORMAL_WORKLOAD_SOURCES",
    "FormalWorkloadAuthority",
    "FormalWorkloadAuthorityBlocked",
    "FormalWorkloadProtocol",
    "FormalWorkloadSample",
    "ReleaseWorkloadSourceLock",
    "bind_formal_workload_authority",
    "formal_workload_samples_sha256",
    "require_formal_workload_authority",
    "revalidate_formal_workload_authority",
]
