"""Reusable, sharded tokenizer authority for controlled-context schedules.

E3b and E6 share the same locked LiveCodeBench-hard plus MATH-level5 filler
pool for one exact tokenizer snapshot.  This artifact tokenizes that pool once
per content/tokenizer identity, retains every first-party worker input/output,
and stores token rows in bounded shards.  Cell-local context compilation then
binds this reusable authority instead of duplicating the full corpus thousands
of times.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.formal_content_source import (
    FormalContentSourceBinding,
)
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedSingleOperatorContentBundle,
)
from lightcone_spec.experiments.formal_single_operator_context_compiler import (
    FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256,
    ContextFillerAuthority,
    TokenizedContextSourceRow,
)
from lightcone_spec.experiments.workload_authority import (
    bind_formal_workload_authority,
)
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.formal_sharded_artifact import (
    FormalCanonicalSequenceShardIndex,
    publish_formal_canonical_sequence_shards,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_SHA256_CHARS = frozenset("0123456789abcdef")
_BATCH_ROWS = 32

FORMAL_SINGLE_OPERATOR_CONTEXT_FILLER_ARTIFACT_PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "schema_version": 1,
            "kind": "formal_single_operator_context_filler_artifact_protocol",
            "compiler_protocol_sha256": (
                FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256
            ),
            "sources": ["livecodebench_v6_hard", "math500_level5"],
            "order": "locked_workload_order_then_selected_sample_order",
            "tokenizer": "exact_RUNTIME_BOUND_tokenizer_member",
            "worker": "first_party_offline_tokenizer_input_output_per_batch",
            "storage": "bounded_contiguous_token_row_shards",
            "reuse": "one_content_source_and_tokenizer_identity",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError(f"{label} must be canonical non-empty text")
    return value


def _strict(label: str, value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _worker_source() -> tuple[Path, str, int]:
    path = (
        Path(__file__).resolve().parent.parent
        / "sglang_bridge"
        / "formal_tokenize_worker.py"
    )
    if not path.is_file() or path.is_symlink():
        raise ValueError("context filler tokenizer worker is unavailable")
    body = path.read_bytes()
    return path, hashlib.sha256(body).hexdigest(), len(body)


def _expected_sources(
    *,
    content_source_binding: FormalContentSourceBinding,
) -> tuple[
    TrustedSingleOperatorContentBundle,
    tuple[tuple[str, str, str, str], ...],
]:
    if (
        type(content_source_binding) is not FormalContentSourceBinding
        or content_source_binding.mode != "trusted_single_operator"
    ):
        raise ValueError("context filler requires tagged trusted content")
    bundle = content_source_binding.reopen()
    if (
        type(bundle) is not TrustedSingleOperatorContentBundle
        or bundle.runtime_binding_status != "BOUND"
    ):
        raise ValueError("context filler requires one runtime-BOUND bundle")
    expected_ids = ("livecodebench_v6_hard", "math500_level5")
    if tuple(row.workload_id for row in bundle.locked_workloads) != expected_ids:
        raise ValueError("context filler locked workload coverage differs")
    values: list[tuple[str, str, str, str]] = []
    for member in bundle.locked_workloads:
        authority = bind_formal_workload_authority(
            member.workload_id,
            member.raw_source_path,
        )
        if (
            authority.sha256 != member.authority_sha256
            or authority.raw_file_sha256 != member.raw_file_sha256
            or authority.selected_rows_sha256 != member.formal_samples_sha256
            or authority.source_lock_sha256 != member.source_lock_sha256
            or authority.protocol_sha256 != member.protocol_sha256
        ):
            raise ValueError("context filler locked workload changed")
        values.extend(
            (member.sha256, sample.sample_id, sample.prompt, _sha256(sample.prompt))
            for sample in authority.samples
        )
    if len({(row[0], row[1]) for row in values}) != len(values):
        raise ValueError("context filler workload source keys repeat")
    return bundle, tuple(values)


def _tokenizer_member(
    *,
    bundle: TrustedSingleOperatorContentBundle,
    member_id: str,
    model_id: str,
    revision: str,
) -> object:
    rows = tuple(
        row
        for row in bundle.model_members
        if row.role == "tokenizer"
        and row.sha256 == member_id
        and row.model_id == model_id
        and row.revision == revision
    )
    if len(rows) != 1:
        raise ValueError("context filler tokenizer member is not exact")
    return rows[0]


def _source_identity(
    *,
    content_source_binding_sha256: str,
    tokenizer_content_member_id: str,
    sources: tuple[tuple[str, str, str, str], ...],
) -> str:
    return _sha256(
        {
            "content_source_binding_sha256": content_source_binding_sha256,
            "tokenizer_content_member_id": tokenizer_content_member_id,
            "ordered_sources": [
                {
                    "source_member_sha256": member,
                    "source_sample_id": sample,
                    "prompt_sha256": prompt_sha,
                }
                for member, sample, _prompt, prompt_sha in sources
            ],
        }
    )


@dataclass(frozen=True)
class TrustedContextFillerArtifact:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_context_filler_artifact"]
    protocol_sha256: str
    source_identity_sha256: str
    content_source_binding_sha256: str
    tokenizer_content_member_id: str
    tokenizer_model_id: str
    tokenizer_revision: str
    tokenizer_snapshot_path: str
    tokenizer_tree_sha256: str
    tokenizer_content_sha256: str
    registered_source_member_sha256s: tuple[str, ...]
    filler_authority_sha256: str
    row_count: int
    rows_shard_index: CanonicalJsonProofBinding
    tokenization_evidence: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_context_filler_artifact"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_CONTEXT_FILLER_ARTIFACT_PROTOCOL_SHA256
        ):
            raise ValueError("context filler artifact schema differs")
        for label, value in (
            ("source identity", self.source_identity_sha256),
            ("content source", self.content_source_binding_sha256),
            ("tokenizer member", self.tokenizer_content_member_id),
            ("tokenizer tree", self.tokenizer_tree_sha256),
            ("tokenizer content", self.tokenizer_content_sha256),
            ("filler authority", self.filler_authority_sha256),
        ):
            _require_sha256(f"context filler artifact {label}", value)
        for label, value in (
            ("tokenizer model", self.tokenizer_model_id),
            ("tokenizer revision", self.tokenizer_revision),
        ):
            _require_text(f"context filler artifact {label}", value)
        path = Path(self.tokenizer_snapshot_path)
        if not path.is_absolute() or path != path.resolve(strict=False):
            raise ValueError("context filler tokenizer path differs")
        if (
            type(self.registered_source_member_sha256s) is not tuple
            or self.registered_source_member_sha256s
            != tuple(sorted(set(self.registered_source_member_sha256s)))
            or type(self.row_count) is not int
            or self.row_count < 1
        ):
            raise ValueError("context filler artifact coverage differs")
        for value in self.registered_source_member_sha256s:
            _require_sha256("context filler registered source", value)
        for binding in (self.rows_shard_index, self.tokenization_evidence):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("context filler artifact is not path-bound")
            if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
                raise ValueError("context filler artifact binding changed")

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "source_identity_sha256": self.source_identity_sha256,
            "content_source_binding_sha256": self.content_source_binding_sha256,
            "tokenizer_content_member_id": self.tokenizer_content_member_id,
            "tokenizer_model_id": self.tokenizer_model_id,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_snapshot_path": self.tokenizer_snapshot_path,
            "tokenizer_tree_sha256": self.tokenizer_tree_sha256,
            "tokenizer_content_sha256": self.tokenizer_content_sha256,
            "registered_source_member_sha256s": list(
                self.registered_source_member_sha256s
            ),
            "filler_authority_sha256": self.filler_authority_sha256,
            "row_count": self.row_count,
            "rows_shard_index": self.rows_shard_index.to_dict(),
            "tokenization_evidence": self.tokenization_evidence.to_dict(),
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "context filler artifact",
            value,
            {
                "schema_version",
                "kind",
                "protocol_sha256",
                "source_identity_sha256",
                "content_source_binding_sha256",
                "tokenizer_content_member_id",
                "tokenizer_model_id",
                "tokenizer_revision",
                "tokenizer_snapshot_path",
                "tokenizer_tree_sha256",
                "tokenizer_content_sha256",
                "registered_source_member_sha256s",
                "filler_authority_sha256",
                "row_count",
                "rows_shard_index",
                "tokenization_evidence",
                "artifact_sha256",
            },
        )
        declared = _require_sha256(
            "context filler artifact", row.pop("artifact_sha256")
        )
        raw_members = row.pop("registered_source_member_sha256s")
        if type(raw_members) is not list:
            raise TypeError("context filler registered sources must be an array")
        row["rows_shard_index"] = CanonicalJsonProofBinding.from_dict(
            row["rows_shard_index"]
        )
        row["tokenization_evidence"] = CanonicalJsonProofBinding.from_dict(
            row["tokenization_evidence"]
        )
        result = cls(
            **row,
            registered_source_member_sha256s=tuple(raw_members),
        )  # type: ignore[arg-type]
        if result.sha256 != declared:
            raise ValueError("context filler artifact digest differs")
        return result


def _publish_tokenizer_input(
    *,
    path: Path,
    source_identity_sha256: str,
    launch: CompileLaunchManifest,
    rows: tuple[tuple[str, str, str, str], ...],
    start_ordinal: int,
) -> CanonicalJsonProofBinding:
    value = {
        "schema_version": 1,
        "kind": "formal_serving_tokenization_input",
        "protocol_sha256": (
            FORMAL_SINGLE_OPERATOR_CONTEXT_FILLER_ARTIFACT_PROTOCOL_SHA256
        ),
        "schedule_source_sha256": source_identity_sha256,
        "tokenizer_model_id": launch.tokenizer_model_id,
        "tokenizer_revision": launch.tokenizer_revision,
        "tokenizer_snapshot_path": launch.tokenizer_snapshot_path,
        "tokenizer_content_authority_sha256": None,
        "requests": [
            {
                "request_id": "context-"
                + _sha256(
                    {
                        "source_member_sha256": member,
                        "source_sample_id": sample,
                        "prompt_sha256": prompt_sha,
                    }
                ),
                "ordinal": start_ordinal + offset,
                "prompt": prompt,
                "prompt_sha256": prompt_sha,
            }
            for offset, (member, sample, prompt, prompt_sha) in enumerate(rows)
        ],
    }
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _invoke_worker(
    *, input_path: Path, output_path: Path
) -> tuple[CanonicalJsonProofBinding, str, int, str]:
    worker, worker_sha256, worker_size = _worker_source()
    executable = Path(sys.executable).resolve()
    argv = (
        str(executable),
        str(worker),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    )
    completed = subprocess.run(
        argv,
        cwd=Path(__file__).resolve().parents[3],
        env={
            "PATH": str(executable.parent),
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "LANG": "C",
            "LC_ALL": "C",
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=600.0,
    )
    if (
        completed.returncode != 0
        or completed.stdout
        or completed.stderr
        or not output_path.is_file()
        or output_path.is_symlink()
    ):
        raise RuntimeError("context filler first-party tokenizer failed")
    return (
        CanonicalJsonProofBinding.bind(output_path),
        worker_sha256,
        worker_size,
        _sha256({"argv": list(argv)}),
    )


def materialize_trusted_context_filler_artifact(
    *,
    content_source_binding: FormalContentSourceBinding,
    compile_launch_manifest_path: str | Path,
    output_directory: str | Path,
) -> CanonicalJsonProofBinding:
    """Tokenize and publish one reusable BOUND LCB+MATH filler authority."""

    root = Path(output_directory)
    if (
        not root.is_absolute()
        or root != root.resolve(strict=False)
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise ValueError("context filler output root must be an existing directory")
    artifact_path = root / "context-filler-authority.json"
    row_root = root / "context-filler-rows"
    token_root = root / "context-filler-tokenization"
    evidence_path = root / "context-filler-tokenization-evidence.json"
    if any(
        os.path.lexists(path)
        for path in (artifact_path, row_root, token_root, evidence_path)
    ):
        raise FileExistsError("context filler artifact output already exists")
    launch = CompileLaunchManifest.load(compile_launch_manifest_path)
    if (
        launch.schema_version != 2
        or launch.content_source_binding != content_source_binding
        or launch.tokenizer_content_authority_sha256 is not None
    ):
        raise ValueError("context filler compile launch content differs")
    bundle, sources = _expected_sources(content_source_binding=content_source_binding)
    member = _tokenizer_member(
        bundle=bundle,
        member_id=launch.tokenizer_content_member_id,
        model_id=launch.tokenizer_model_id,
        revision=launch.tokenizer_revision,
    )
    source_identity = _source_identity(
        content_source_binding_sha256=content_source_binding.sha256,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        sources=sources,
    )
    token_root.mkdir(mode=0o700)
    rows: list[TokenizedContextSourceRow] = []
    batches: list[dict[str, object]] = []
    worker_identity: tuple[str, int] | None = None
    tokenizer_identity: tuple[str, str, int] | None = None
    for batch_ordinal, start in enumerate(range(0, len(sources), _BATCH_ROWS)):
        selected = sources[start : start + _BATCH_ROWS]
        input_path = token_root / f"input-{batch_ordinal:06d}.json"
        output_path = token_root / f"output-{batch_ordinal:06d}.json"
        input_binding = _publish_tokenizer_input(
            path=input_path,
            source_identity_sha256=source_identity,
            launch=launch,
            rows=selected,
            start_ordinal=start,
        )
        output_binding, worker_sha, worker_size, argv_sha = _invoke_worker(
            input_path=input_path,
            output_path=output_path,
        )
        output = output_binding.reopen()
        raw_rows = output.get("requests") if type(output) is dict else None
        if type(raw_rows) is not list or len(raw_rows) != len(selected):
            raise ValueError("context filler tokenizer output coverage differs")
        for offset, (source, token_value) in enumerate(
            zip(selected, raw_rows, strict=True)
        ):
            member_sha, sample_id, _prompt, prompt_sha = source
            token_row = _strict(
                "context filler tokenizer output row",
                token_value,
                {
                    "request_id",
                    "ordinal",
                    "prompt_sha256",
                    "input_token_ids",
                    "input_token_ids_sha256",
                },
            )
            token_ids = token_row["input_token_ids"]
            if (
                type(token_ids) is not list
                or token_row["ordinal"] != start + offset
                or token_row["prompt_sha256"] != prompt_sha
                or token_row["input_token_ids_sha256"] != _sha256(token_ids)
            ):
                raise ValueError("context filler tokenizer row identity differs")
            rows.append(
                TokenizedContextSourceRow(
                    tokenizer_content_member_id=launch.tokenizer_content_member_id,
                    tokenizer_model_id=launch.tokenizer_model_id,
                    tokenizer_revision=launch.tokenizer_revision,
                    source_member_sha256=member_sha,
                    source_sample_id=sample_id,
                    prompt_sha256=prompt_sha,
                    input_token_ids=tuple(token_ids),
                )
            )
        current_worker = (worker_sha, worker_size)
        current_tokenizer = (
            str(output["transformers_version"]),
            str(output["tokenizer_class"]),
            int(output["tokenizer_vocab_size"]),
        )
        if worker_identity is None:
            worker_identity = current_worker
            tokenizer_identity = current_tokenizer
        elif (
            worker_identity != current_worker or tokenizer_identity != current_tokenizer
        ):
            raise ValueError("context filler tokenizer identity changed")
        batches.append(
            {
                "batch_ordinal": batch_ordinal,
                "start_ordinal": start,
                "end_ordinal_exclusive": start + len(selected),
                "tokenization_input": input_binding.to_dict(),
                "tokenization_output": output_binding.to_dict(),
                "tokenizer_worker_argv_sha256": argv_sha,
            }
        )
    assert worker_identity is not None and tokenizer_identity is not None
    evidence = {
        "schema_version": 1,
        "kind": "formal_single_operator_context_filler_tokenization_evidence",
        "protocol_sha256": (
            FORMAL_SINGLE_OPERATOR_CONTEXT_FILLER_ARTIFACT_PROTOCOL_SHA256
        ),
        "source_identity_sha256": source_identity,
        "tokenizer_worker_source_raw_sha256": worker_identity[0],
        "tokenizer_worker_source_size": worker_identity[1],
        "transformers_version": tokenizer_identity[0],
        "tokenizer_class": tokenizer_identity[1],
        "tokenizer_vocab_size": tokenizer_identity[2],
        "batch_count": len(batches),
        "batches": batches,
    }
    publish_canonical_json_no_replace(evidence_path, evidence)
    evidence_binding = CanonicalJsonProofBinding.bind(evidence_path)
    authority = ContextFillerAuthority(
        schema_version=1,
        kind="formal_single_operator_context_filler_authority",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256,
        content_source_binding_sha256=content_source_binding.sha256,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        registered_source_member_sha256s=tuple(
            sorted({row.source_member_sha256 for row in rows})
        ),
        rows=tuple(rows),
    )
    row_root.mkdir(mode=0o700)
    rows_binding, _index = publish_formal_canonical_sequence_shards(
        artifact_kind="formal_single_operator_context_filler_rows",
        artifact_id=source_identity,
        rows=tuple(row.to_dict() for row in authority.rows),
        output_directory=row_root,
        maximum_shard_bytes=1_000_000,
        maximum_shard_rows=64,
    )
    artifact = TrustedContextFillerArtifact(
        schema_version=1,
        kind="formal_single_operator_context_filler_artifact",
        protocol_sha256=(
            FORMAL_SINGLE_OPERATOR_CONTEXT_FILLER_ARTIFACT_PROTOCOL_SHA256
        ),
        source_identity_sha256=source_identity,
        content_source_binding_sha256=content_source_binding.sha256,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        tokenizer_snapshot_path=launch.tokenizer_snapshot_path,
        tokenizer_tree_sha256=member.tree_sha256,
        tokenizer_content_sha256=member.content_sha256,
        registered_source_member_sha256s=(authority.registered_source_member_sha256s),
        filler_authority_sha256=authority.sha256,
        row_count=len(authority.rows),
        rows_shard_index=rows_binding,
        tokenization_evidence=evidence_binding,
    )
    publish_canonical_json_no_replace(artifact_path, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(artifact_path)
    reopened = load_trusted_context_filler_artifact(
        binding.absolute_path,
        content_source_binding=content_source_binding,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
    )
    if reopened != authority:
        raise RuntimeError("context filler artifact changed after publication")
    return binding


def load_trusted_context_filler_artifact(
    path: str | Path,
    *,
    content_source_binding: FormalContentSourceBinding,
    tokenizer_content_member_id: str,
    tokenizer_model_id: str,
    tokenizer_revision: str,
) -> ContextFillerAuthority:
    """Deep-reopen rows, worker evidence, BOUND sources, and tokenizer member."""

    binding = CanonicalJsonProofBinding.bind(path)
    artifact = TrustedContextFillerArtifact.from_dict(binding.reopen())
    if binding.semantic_sha256 != _sha256(artifact.to_dict()):
        raise ValueError("context filler artifact path identity differs")
    bundle, sources = _expected_sources(content_source_binding=content_source_binding)
    member = _tokenizer_member(
        bundle=bundle,
        member_id=tokenizer_content_member_id,
        model_id=tokenizer_model_id,
        revision=tokenizer_revision,
    )
    expected_identity = _source_identity(
        content_source_binding_sha256=content_source_binding.sha256,
        tokenizer_content_member_id=tokenizer_content_member_id,
        sources=sources,
    )
    if (
        artifact.source_identity_sha256 != expected_identity
        or artifact.content_source_binding_sha256 != content_source_binding.sha256
        or artifact.tokenizer_content_member_id != tokenizer_content_member_id
        or artifact.tokenizer_model_id != tokenizer_model_id
        or artifact.tokenizer_revision != tokenizer_revision
        or artifact.tokenizer_tree_sha256 != member.tree_sha256
        or artifact.tokenizer_content_sha256 != member.content_sha256
        or artifact.registered_source_member_sha256s
        != tuple(sorted({row[0] for row in sources}))
    ):
        raise ValueError("context filler artifact source/tokenizer lineage differs")
    index = FormalCanonicalSequenceShardIndex.from_dict(
        artifact.rows_shard_index.reopen()
    )
    if (
        index.artifact_kind != "formal_single_operator_context_filler_rows"
        or index.artifact_id != expected_identity
        or index.total_rows != artifact.row_count
        or artifact.rows_shard_index.semantic_sha256 != _sha256(index.to_dict())
    ):
        raise ValueError("context filler row shard index differs")
    rows = tuple(
        TokenizedContextSourceRow.from_dict(row)
        for row in index.revalidate().iter_rows()
    )
    authority = ContextFillerAuthority(
        schema_version=1,
        kind="formal_single_operator_context_filler_authority",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256,
        content_source_binding_sha256=content_source_binding.sha256,
        tokenizer_content_member_id=tokenizer_content_member_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_revision=tokenizer_revision,
        registered_source_member_sha256s=artifact.registered_source_member_sha256s,
        rows=rows,
    )
    if authority.sha256 != artifact.filler_authority_sha256 or tuple(
        (row.source_member_sha256, row.source_sample_id, row.prompt_sha256)
        for row in authority.rows
    ) != tuple(
        (member_sha, sample, prompt_sha)
        for member_sha, sample, _p, prompt_sha in sources
    ):
        raise ValueError("context filler authority rows differ from locked sources")

    evidence = _strict(
        "context filler tokenization evidence",
        artifact.tokenization_evidence.reopen(),
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "source_identity_sha256",
            "tokenizer_worker_source_raw_sha256",
            "tokenizer_worker_source_size",
            "transformers_version",
            "tokenizer_class",
            "tokenizer_vocab_size",
            "batch_count",
            "batches",
        },
    )
    worker, worker_sha, worker_size = _worker_source()
    raw_batches = evidence["batches"]
    if (
        evidence["schema_version"] != 1
        or evidence["kind"]
        != "formal_single_operator_context_filler_tokenization_evidence"
        or evidence["protocol_sha256"]
        != FORMAL_SINGLE_OPERATOR_CONTEXT_FILLER_ARTIFACT_PROTOCOL_SHA256
        or evidence["source_identity_sha256"] != expected_identity
        or evidence["tokenizer_worker_source_raw_sha256"] != worker_sha
        or evidence["tokenizer_worker_source_size"] != worker_size
        or type(raw_batches) is not list
        or evidence["batch_count"] != len(raw_batches)
    ):
        raise ValueError("context filler tokenization evidence identity differs")
    cursor = 0
    for batch_ordinal, raw_batch in enumerate(raw_batches):
        batch = _strict(
            "context filler tokenization batch",
            raw_batch,
            {
                "batch_ordinal",
                "start_ordinal",
                "end_ordinal_exclusive",
                "tokenization_input",
                "tokenization_output",
                "tokenizer_worker_argv_sha256",
            },
        )
        input_binding = CanonicalJsonProofBinding.from_dict(batch["tokenization_input"])
        output_binding = CanonicalJsonProofBinding.from_dict(
            batch["tokenization_output"]
        )
        input_value = input_binding.reopen()
        output_value = output_binding.reopen()
        end = batch["end_ordinal_exclusive"]
        if (
            batch["batch_ordinal"] != batch_ordinal
            or batch["start_ordinal"] != cursor
            or type(end) is not int
            or not cursor < end <= len(sources)
        ):
            raise ValueError("context filler tokenization batch interval differs")
        expected_sources = sources[cursor:end]
        input_rows = input_value.get("requests") if type(input_value) is dict else None
        output_rows = (
            output_value.get("requests") if type(output_value) is dict else None
        )
        if (
            type(input_rows) is not list
            or type(output_rows) is not list
            or len(input_rows) != len(expected_sources)
            or len(output_rows) != len(expected_sources)
            or output_value.get("schedule_source_sha256") != expected_identity
            or input_value.get("schedule_source_sha256") != expected_identity
            or output_value.get("tokenizer_model_id") != tokenizer_model_id
            or output_value.get("tokenizer_revision") != tokenizer_revision
            or output_value.get("tokenizer_snapshot_path")
            != artifact.tokenizer_snapshot_path
            or output_value.get("tokenizer_content_authority_sha256") is not None
            or output_value.get("transformers_version")
            != evidence["transformers_version"]
            or output_value.get("tokenizer_class") != evidence["tokenizer_class"]
            or output_value.get("tokenizer_vocab_size")
            != evidence["tokenizer_vocab_size"]
        ):
            raise ValueError("context filler tokenizer batch output differs")
        executable = Path(sys.executable).resolve()
        expected_argv = (
            str(executable),
            str(worker),
            "--input",
            input_binding.absolute_path,
            "--output",
            output_binding.absolute_path,
        )
        if batch["tokenizer_worker_argv_sha256"] != _sha256(
            {"argv": list(expected_argv)}
        ):
            raise ValueError("context filler tokenizer worker argv differs")
        for offset, (expected, input_row, output_row) in enumerate(
            zip(expected_sources, input_rows, output_rows, strict=True)
        ):
            member_sha, sample_id, prompt, prompt_sha = expected
            expected_request_id = "context-" + _sha256(
                {
                    "source_member_sha256": member_sha,
                    "source_sample_id": sample_id,
                    "prompt_sha256": prompt_sha,
                }
            )
            expected_ordinal = cursor + offset
            if (
                input_row
                != {
                    "request_id": expected_request_id,
                    "ordinal": expected_ordinal,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha,
                }
                or type(output_row) is not dict
                or output_row.get("request_id") != expected_request_id
                or output_row.get("ordinal") != expected_ordinal
                or output_row.get("prompt_sha256") != prompt_sha
                or output_row.get("input_token_ids")
                != list(authority.rows[expected_ordinal].input_token_ids)
                or output_row.get("input_token_ids_sha256")
                != authority.rows[expected_ordinal].token_ids_sha256
            ):
                raise ValueError("context filler tokenizer row replay differs")
        cursor = end
    if cursor != len(sources):
        raise ValueError("context filler tokenization coverage differs")
    return authority


__all__ = [
    "FORMAL_SINGLE_OPERATOR_CONTEXT_FILLER_ARTIFACT_PROTOCOL_SHA256",
    "TrustedContextFillerArtifact",
    "load_trusted_context_filler_artifact",
    "materialize_trusted_context_filler_artifact",
]
