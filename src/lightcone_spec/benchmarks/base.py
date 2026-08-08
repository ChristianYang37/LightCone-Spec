"""Benchmark adapter contract (spec 12.1, 12.2).

Each adapter locks: immutable source/revision, sample IDs, split, prompt
rendering, chat template use, stop condition, max new tokens, answer
extraction, scorer, judge config where applicable, timeout, invalid
output handling, license/provenance, and the deterministic sample
subset (seed-0 permutation). Overlapping datasets have exactly one
adapter, one locked version and one scorer.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import heapq
import os
from typing import Callable, Optional

import numpy as np

from lightcone_spec.exit_codes import LockError
from lightcone_spec.locking.hashing import sha256_json, stable_hash_int
from lightcone_spec.locking.lockfile import Lockfile


_SAMPLE_CACHE: dict[tuple[object, ...], tuple["BenchmarkSample", ...]] = {}


def _is_network_failure(exc: BaseException) -> bool:
    """Recognize transport failures without importing optional HTTP clients."""
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        module = type(current).__module__.split(".", 1)[0]
        name = type(current).__name__.lower()
        if isinstance(current, (ConnectionError, TimeoutError)) or module in {
            "httpx",
            "httpcore",
            "requests",
            "urllib3",
            "aiohttp",
        } or any(word in name for word in ("connectionerror", "connecterror", "timeout")):
            return True
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException):
                pending.append(linked)
    return False


@contextmanager
def _offline_hf_mode(datasets_module):
    """Temporarily force cache-only reads, restoring all process state afterwards."""
    old_env = {
        name: os.environ.get(name)
        for name in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")
    }
    patched: list[tuple[object, str, object]] = []
    try:
        for name in old_env:
            os.environ[name] = "1"
        targets = [getattr(datasets_module, "config", None)]
        try:
            import huggingface_hub.constants as hub_constants  # type: ignore

            targets.append(hub_constants)
        except ImportError:
            pass
        for target in targets:
            if target is None:
                continue
            for name in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
                if hasattr(target, name):
                    patched.append((target, name, getattr(target, name)))
                    setattr(target, name, True)
        yield
    finally:
        for target, name, value in reversed(patched):
            setattr(target, name, value)
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass
class BenchmarkSample:
    sample_id: str
    prompt: str
    gold_answer: Optional[str] = None
    test_code: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkAdapter:
    key: str
    source_group: str  # deepspec | tts
    hf_path: str
    hf_config: Optional[str]
    split: str
    quality_metric: str  # exact_match | accuracy | pass@1 | judge_score
    output_cap: int
    task_type: str  # math | code | chat | science
    prompt_template: str
    stop_strings: tuple[str, ...]
    scorer_kind: str  # exact_match | pass_at_1 | judge
    timeout_s: float
    license_note: str
    judge_model: Optional[str] = None
    judge_revision: Optional[str] = None
    data_file: Optional[str | tuple[str, ...]] = None
    loader: Optional[Callable[..., list[BenchmarkSample]]] = None

    # ---- deterministic subsets (spec 13.3/13.5) ------------------------

    def deterministic_subset(
        self,
        sample_ids: list[str],
        limit: int,
        seed: int = 0,
        offset: int = 0,
    ) -> list[str]:
        """One deterministic, non-wrapping window of the stable permutation."""
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        order = sorted(
            sample_ids, key=lambda sid: stable_hash_int(f"perm/{seed}/{self.key}/{sid}")
        )
        return order[offset : offset + limit]

    def sample_ids_hash(self, sample_ids: list[str]) -> str:
        return sha256_json(sorted(sample_ids))

    def validate_sample_ids(self, sample_ids: list[str]) -> None:
        """Require a non-empty, unique sequence-group identity domain.

        Prompt windows and replay splits both use ``sample_id`` as their
        stable group key.  Allowing one ID to occur twice could place the same
        group in two nominally disjoint permutation windows, so reject such a
        dataset both while locking and while consuming an older lockfile.
        """
        if not sample_ids:
            raise LockError(f"{self.key}: dataset contains no sample IDs")
        seen: set[str] = set()
        for sample_id in sample_ids:
            if sample_id in seen:
                raise LockError(
                    f"{self.key}: duplicate sample ID {sample_id!r} "
                    "would violate prompt-window isolation"
                )
            seen.add(sample_id)

    # ---- locked loading ----------------------------------------------

    def load_samples(
        self, lock: Lockfile, limit: int = 128, offset: int = 0
    ) -> list[BenchmarkSample]:
        """Load through the lockfile only: the dataset revision comes from
        the frozen lock entry; revision drift fails closed."""
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        locked = lock.find_dataset(self.key)
        observed_binding = (self.hf_path, self.hf_config, self.split)
        locked_binding = (locked.source, locked.config, locked.split)
        if observed_binding != locked_binding:
            raise LockError(
                f"{self.key}: adapter source/config/split drift against lockfile "
                "(fail closed)"
            )
        cache_key = (
            # Cache entries contain rendered BenchmarkSample objects rather
            # than raw rows.  Isolate adapter instances and invalidate an
            # entry when any source or rendering field is changed in-process.
            id(self),
            sha256_json(
                {
                    "class": f"{type(self).__module__}.{type(self).__qualname__}",
                    "source": observed_binding,
                    "data_files": self.locked_data_files(),
                    "prompt_template": self.prompt_template,
                    "field_maps": self._field_maps,
                    "loader": (
                        None
                        if self.loader is None
                        else (
                            getattr(self.loader, "__module__", ""),
                            getattr(self.loader, "__qualname__", repr(self.loader)),
                        )
                    ),
                }
            ),
            self.key,
            locked.source,
            locked.config,
            locked.split,
            locked.revision,
            locked.sample_ids_sha256,
            int(locked.num_samples),
            limit,
            offset,
        )
        cached = _SAMPLE_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)
        if self.loader is not None:
            samples = self.load_revision(locked.revision)
            ids = [s.sample_id for s in samples]
            self.validate_sample_ids(ids)
            selected_ids = set(
                self.deterministic_subset(ids, limit, offset=offset)
            )
            selected = [s for s in samples if s.sample_id in selected_ids]
        else:
            ids, selected_rows = self._select_hf_rows(
                locked.revision, limit, offset=offset
            )
            self.validate_sample_ids(ids)
            selected = [self.row_to_sample(index, row) for index, row in selected_rows]
        if len(ids) != locked.num_samples:
            raise LockError(
                f"{self.key}: sample count drift against lockfile "
                f"(locked {locked.num_samples}, found {len(ids)})"
            )
        if self.sample_ids_hash(ids) != locked.sample_ids_sha256:
            raise LockError(
                f"{self.key}: sample-id hash drift against lockfile (fail closed)"
            )
        expected_selected = min(limit, max(locked.num_samples - offset, 0))
        if len(selected) != expected_selected:
            raise LockError(
                f"{self.key}: deterministic subset is incomplete "
                f"(expected {expected_selected}, found {len(selected)})"
            )
        _SAMPLE_CACHE[cache_key] = tuple(selected)
        return list(selected)

    def load_sample_ids_revision(self, revision: str) -> list[str]:
        """Load only stable IDs for locking; never materialize large prompts/tests."""
        if self.loader is not None:
            sample_ids = [
                sample.sample_id for sample in self.load_revision(revision)
            ]
        else:
            sample_ids = [
                self.sample_id_from_row(index, dict(row))
                for index, row in enumerate(self._load_hf_dataset(revision))
            ]
        self.validate_sample_ids(sample_ids)
        return sample_ids

    def load_revision(self, revision: str) -> list[BenchmarkSample]:
        if self.loader is not None:
            return self.loader(revision=revision)
        return self._default_hf_loader(revision)

    def locked_data_files(self) -> tuple[str, ...]:
        if self.data_file is None:
            return ()
        if isinstance(self.data_file, str):
            return (self.data_file,)
        return self.data_file

    def _default_hf_loader(self, revision: str) -> list[BenchmarkSample]:
        return [
            self.row_to_sample(index, dict(row))
            for index, row in enumerate(self._load_hf_dataset(revision))
        ]

    def _load_hf_dataset(self, revision: str):
        try:
            import datasets  # type: ignore
        except ImportError as exc:
            raise LockError(
                "the `datasets` package (gpu extra) is required to load "
                f"benchmark {self.key}"
            ) from exc
        data_files = self.locked_data_files()
        if not data_files:
            args = (self.hf_path, self.hf_config)
            kwargs = {"split": self.split, "revision": revision}
        else:
            suffixes = {file.rsplit(".", 1)[-1].lower() for file in data_files}
            if suffixes <= {"json", "jsonl"}:
                builder = "json"
            elif suffixes == {"parquet"}:
                builder = "parquet"
            else:
                raise LockError(
                    f"{self.key}: unsupported locked data file types {sorted(suffixes)}"
                )
            data_uris = [
                f"hf://datasets/{self.hf_path}@{revision}/{file}"
                for file in data_files
            ]
            args = (builder,)
            kwargs = {
                "data_files": {
                    self.split: data_uris[0] if len(data_uris) == 1 else data_uris
                },
                "split": self.split,
            }
        try:
            return datasets.load_dataset(*args, **kwargs)
        except Exception as online_exc:
            if not _is_network_failure(online_exc):
                raise
        try:
            with _offline_hf_mode(datasets):
                return datasets.load_dataset(*args, **kwargs)
        except Exception as offline_exc:
            raise LockError(
                f"{self.key}: cannot load locked dataset {self.hf_path}@{revision}; "
                "the network failed and no complete local cache was available. "
                "Set HF_ENDPOINT to a reachable mirror and prepare the dataset "
                "cache before starting the GPU run."
            ) from offline_exc

    def _select_hf_rows(
        self, revision: str, limit: int, offset: int = 0
    ) -> tuple[list[str], list[tuple[int, dict]]]:
        """Scan once, retaining only the requested deterministic window."""
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        retain = limit + offset
        ids: list[str] = []
        selected: list[tuple[int, int, int, str, dict]] = []
        for index, source_row in enumerate(self._load_hf_dataset(revision)):
            row = dict(source_row)
            sample_id = self.sample_id_from_row(index, row)
            ids.append(sample_id)
            if not retain:
                continue
            rank = stable_hash_int(f"perm/0/{self.key}/{sample_id}")
            entry = (-rank, -index, index, sample_id, row)
            if len(selected) < retain:
                heapq.heappush(selected, entry)
            elif (rank, index) < (-selected[0][0], -selected[0][1]):
                heapq.heapreplace(selected, entry)
        ranked = sorted(
            (-entry[0], -entry[1], entry[2], entry[4]) for entry in selected
        )
        rows = sorted(
            (index, row)
            for _rank, _source_index, index, row in ranked[offset : offset + limit]
        )
        return ids, rows

    def sample_id_from_row(self, index: int, row: dict) -> str:
        return str(
            row.get(self.metadata_field("id_field", "id"), f"{self.key}-{index}")
        )

    def prompt_from_row(self, row: dict) -> str:
        prompt_field = self.metadata_field("prompt_field", "question")
        value = row.get(prompt_field, "")
        prompt_index = self.metadata_field("prompt_index", None)
        if prompt_index is not None:
            if not isinstance(value, (list, tuple)) or not (
                -len(value) <= int(prompt_index) < len(value)
            ):
                raise LockError(
                    f"{self.key}: prompt field {prompt_field!r} has no "
                    f"turn {prompt_index}"
                )
            value = value[int(prompt_index)]
        return str(value)

    def row_to_sample(self, index: int, row: dict) -> BenchmarkSample:
        """Field mapping per dataset; overridden via `field_map` metadata in
        the registry definitions."""
        answer_field = self.metadata_field("answer_field", "answer")
        test_field = self.metadata_field("test_field", None)
        sample_id = self.sample_id_from_row(index, row)
        return BenchmarkSample(
            sample_id=sample_id,
            prompt=self.prompt_template.format(question=self.prompt_from_row(row)),
            gold_answer=(
                str(row.get(answer_field)) if row.get(answer_field) is not None else None
            ),
            test_code=(str(row.get(test_field)) if test_field and row.get(test_field) else None),
            metadata={"index": index},
        )

    _field_maps: dict = field(default_factory=dict)

    def metadata_field(self, name: str, default):
        return self._field_maps.get(name, default)

    # ---- scoring ---------------------------------------------------------

    def score(self, sample: BenchmarkSample, output: str, judge_fn=None) -> float:
        from lightcone_spec.benchmarks.scorers import (
            LockedJudgeScorer,
            exact_match_score,
            pass_at_1_score,
        )

        if output is None or not output.strip():
            return 0.0  # invalid output handling: score 0, keep the row
        if self.scorer_kind == "exact_match":
            if sample.gold_answer is None:
                raise ValueError(f"{self.key}: sample without gold answer")
            return exact_match_score(output, sample.gold_answer)
        if self.scorer_kind == "pass_at_1":
            if sample.test_code is None:
                raise ValueError(f"{self.key}: sample without test code")
            return pass_at_1_score(output, sample.test_code, self.timeout_s)
        if self.scorer_kind == "judge":
            scorer = LockedJudgeScorer(
                judge_model=self.judge_model or "",
                judge_revision=self.judge_revision or "",
                judge_prompt_sha256=sha256_json(self.prompt_template),
                judge_fn=judge_fn,
            )
            return scorer.score(sample.prompt, output)
        raise ValueError(f"unknown scorer kind {self.scorer_kind}")
