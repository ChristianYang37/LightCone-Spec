"""First-party, non-serving GPU preflight runner and immutable terminal.

The runner is deliberately narrow: it executes one source-owned qualification
suite, on one exact two-GPU inventory, after a dynamic root-authorized dispatch
decision.  It publishes the terminal and its no-replace result pointer even
when the suite fails, so a failure is evidence rather than an excuse to lose
the run.  Only an exact 8/8 JUnit result with zero skips and clean process
snapshots is eligible for formal activation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.runtime.content_authorization import (
    ContentVerificationReceipt,
    DatasetContentPathBinding,
    VerifiedDatasetContentRelease,
    revalidate_authorized_dataset_content_release,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    VerifiedControlArtifact,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import relocated_evidence_path

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")

PREFLIGHT_EXACTNESS_TEST_NAMES: tuple[str, ...] = (
    "test_dp2_sticky_replica_isolation",
    "test_dspark_real_serving_native_heads",
    "test_graph_fixed_address_no_host_sync",
    "test_native_itl_full_token_coverage",
    "test_nextn_tp1_real_serving",
    "test_nextn_tp2_real_serving",
    "test_tp2_one_rank_abort_zero_partial_publication",
    "test_tp2_shard_reference_parity",
)
PREFLIGHT_EXACTNESS_TEST_FILE = (
    "test/registered/unit/spec/test_formal_preflight_gpu_qualification.py"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(label: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or value != value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{label} must be canonical single-line text")
    return value


def _absolute_path(label: str, value: object) -> Path:
    path = Path(_require_text(label, value))
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError(f"{label} must be absolute and normalized")
    if path == Path(path.anchor):
        raise ValueError(f"{label} cannot be a filesystem root")
    return path


def _strict_object(
    label: str,
    value: object,
    fields: frozenset[str],
) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a JSON object with string keys")
    if set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return value


def _stable_file_bytes(path: Path, *, label: str) -> tuple[bytes, str, int]:
    path = relocated_evidence_path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be one regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        body = b"".join(chunks)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
            or len(body) != after.st_size
        ):
            raise RuntimeError(f"{label} changed while read")
        return body, hashlib.sha256(body).hexdigest(), len(body)
    finally:
        os.close(descriptor)


def _raw_file(path: Path, *, label: str) -> tuple[str, int]:
    _body, digest, size = _stable_file_bytes(path, label=label)
    return digest, size


def _publish_bytes(path: Path, body: bytes) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("preflight evidence parent must be a regular directory")
    temporary = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
    except FileExistsError:
        digest, size = _raw_file(path, label="existing preflight evidence")
        if size != len(body) or digest != hashlib.sha256(body).hexdigest():
            raise ValueError("immutable preflight evidence already differs")
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _publish_json(path: Path, value: object) -> None:
    _publish_bytes(path, _canonical_bytes(value) + b"\n")
    _publish_bytes(
        Path(f"{path}.sha256"),
        f"{_sha(value)}\n".encode("ascii"),
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    body, digest, size = _stable_file_bytes(path, label=label)
    if size > 16 * 1024 * 1024:
        raise ValueError(f"{label} is too large")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    if type(value) is not dict or body != _canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} is not canonical JSON")
    sidecar_body, sidecar_digest, sidecar_size = _stable_file_bytes(
        Path(f"{path}.sha256"), label=f"{label} sidecar"
    )
    del sidecar_digest
    if sidecar_size != 65 or sidecar_body != (f"{_sha(value)}\n".encode("ascii")):
        raise ValueError(f"{label} sidecar differs")
    if digest != hashlib.sha256(body).hexdigest():
        raise RuntimeError(f"{label} raw digest changed")
    return value


@dataclass(frozen=True)
class BurstGptSourcePin:
    source_id: str
    official_sha256: str

    def __post_init__(self) -> None:
        _require_text("BurstGPT source ID", self.source_id)
        _require_sha256("BurstGPT official source", self.official_sha256)


@dataclass(frozen=True)
class BurstGptShapeAuthority:
    schema_version: int
    kind: Literal["burstgpt_six_source_shape_authority"]
    sources: tuple[BurstGptSourcePin, ...]
    rows_sha256: str
    row_count: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "burstgpt_six_source_shape_authority"
        ):
            raise ValueError("BurstGPT shape authority schema is unsupported")
        if (
            type(self.sources) is not tuple
            or len(self.sources) != 6
            or tuple(row.source_id for row in self.sources)
            != tuple(sorted({row.source_id for row in self.sources}))
            or any(type(row) is not BurstGptSourcePin for row in self.sources)
        ):
            raise ValueError("BurstGPT authority requires six sorted official sources")
        _require_sha256("BurstGPT shape rows", self.rows_sha256)
        if type(self.row_count) is not int or self.row_count < 1:
            raise ValueError("BurstGPT shape authority row count must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "sources": [asdict(row) for row in self.sources],
            "rows_sha256": self.rows_sha256,
            "row_count": self.row_count,
        }

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "BurstGPT shape authority",
            value,
            frozenset(
                {"schema_version", "kind", "sources", "rows_sha256", "row_count"}
            ),
        )
        sources = row.pop("sources")
        if type(sources) is not list:
            raise TypeError("BurstGPT sources must be an array")
        return cls(
            sources=tuple(
                BurstGptSourcePin(
                    **_strict_object(
                        "BurstGPT source",
                        item,
                        frozenset({"source_id", "official_sha256"}),
                    )
                )
                for item in sources
            ),
            **row,
        )


def derive_burstgpt_shape_authority_from_content_receipt(
    receipt: ContentVerificationReceipt,
    *,
    current_ns: int,
) -> BurstGptShapeAuthority:
    """Derive the six-source shape authority from signed, path-bound content."""

    if type(receipt) is not ContentVerificationReceipt:
        raise TypeError("BurstGPT derivation requires an exact content receipt")
    verified_rows = receipt.revalidate(current_ns=current_ns)
    matches = tuple(
        row
        for row in verified_rows
        if type(row) is VerifiedDatasetContentRelease
        and row.authority_domain == "burstgpt_six_source"
    )
    if len(matches) != 1:
        raise ValueError("content receipt lacks one BurstGPT dataset authorization")
    path_artifacts = tuple(
        row
        for row in receipt.content_artifacts
        if row.artifact_id == "dataset:burstgpt_six_source:path_binding"
    )
    if len(path_artifacts) != 1:
        raise ValueError("content receipt lacks one BurstGPT path binding")
    binding = DatasetContentPathBinding.from_dict(path_artifacts[0].load())
    members = revalidate_authorized_dataset_content_release(
        binding,
        authorization=matches[0],
    )
    by_id = {row.member_id: row for row in members}
    shape_rows: list[dict[str, object]] = []
    for path_binding in binding.members:
        body, _digest, _size = _stable_file_bytes(
            Path(path_binding.request_shape_path),
            label=f"BurstGPT {path_binding.member_id} request shape",
        )
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("BurstGPT request shape is not UTF-8 JSON") from error
        if (
            type(value) is not dict
            or body != _canonical_bytes(value) + b"\n"
            or set(value) != {"schema_version", "kind", "requests"}
            or value["schema_version"] != 1
            or value["kind"] != "lightcone_request_shape_manifest"
            or type(value["requests"]) is not list
            or not value["requests"]
            or any(type(row) is not dict for row in value["requests"])
        ):
            raise ValueError("BurstGPT request shape manifest is not canonical")
        shape_rows.extend(
            {"source_id": path_binding.member_id, "request": row}
            for row in value["requests"]
        )
    authority = BurstGptShapeAuthority(
        schema_version=1,
        kind="burstgpt_six_source_shape_authority",
        sources=tuple(
            BurstGptSourcePin(
                source_id=member_id,
                official_sha256=by_id[member_id].raw_file_sha256,
            )
            for member_id in sorted(by_id)
        ),
        rows_sha256=_sha(shape_rows),
        row_count=len(shape_rows),
    )
    if tuple(row.source_id for row in authority.sources) != tuple(
        row.member_id for row in members
    ):
        raise RuntimeError("BurstGPT shape authority changed signed source order")
    return authority


@dataclass(frozen=True)
class PreflightInputLocks:
    prepared_model_set_sha256: str | None
    prepared_model_content_authority_sha256: str | None
    formal_workload_lock_sha256: str | None
    burstgpt_shape_authority: BurstGptShapeAuthority | None
    content_source_binding: object | None = None
    trusted_model_member_sha256s: tuple[str, ...] = ()
    trusted_workload_member_sha256: str | None = None
    trusted_burstgpt_release_sha256: str | None = None

    def __post_init__(self) -> None:
        from lightcone_spec.experiments.formal_content_source import (
            FormalContentSourceBinding,
        )

        if self.content_source_binding is None:
            for label, digest in (
                ("prepared model set", self.prepared_model_set_sha256),
                (
                    "prepared model content authority",
                    self.prepared_model_content_authority_sha256,
                ),
                ("formal workload lock", self.formal_workload_lock_sha256),
            ):
                _require_sha256(label, digest)
            if type(self.burstgpt_shape_authority) is not BurstGptShapeAuthority:
                raise TypeError("preflight inputs require exact BurstGPT authority")
            if (
                self.trusted_model_member_sha256s
                or self.trusted_workload_member_sha256 is not None
                or self.trusted_burstgpt_release_sha256 is not None
            ):
                raise ValueError("signed preflight locks carry trusted lineage")
            return
        if (
            type(self.content_source_binding) is not FormalContentSourceBinding
            or self.content_source_binding.mode != "trusted_single_operator"
            or self.prepared_model_set_sha256 is not None
            or self.prepared_model_content_authority_sha256 is not None
            or self.formal_workload_lock_sha256 is not None
            or self.burstgpt_shape_authority is not None
        ):
            raise ValueError("trusted preflight locks carry signed authorization")
        self.content_source_binding.reopen()
        if (
            type(self.trusted_model_member_sha256s) is not tuple
            or len(self.trusted_model_member_sha256s) != 3
            or self.trusted_model_member_sha256s
            != tuple(sorted(set(self.trusted_model_member_sha256s)))
        ):
            raise ValueError("trusted preflight model member set is not exact")
        for digest in self.trusted_model_member_sha256s:
            _require_sha256("trusted preflight model member", digest)
        _require_sha256(
            "trusted preflight workload member",
            self.trusted_workload_member_sha256,
        )
        _require_sha256(
            "trusted preflight BurstGPT release",
            self.trusted_burstgpt_release_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "prepared_model_set_sha256": self.prepared_model_set_sha256,
            "prepared_model_content_authority_sha256": (
                self.prepared_model_content_authority_sha256
            ),
            "formal_workload_lock_sha256": self.formal_workload_lock_sha256,
            "burstgpt_shape_authority": (
                None
                if self.burstgpt_shape_authority is None
                else self.burstgpt_shape_authority.to_dict()
            ),
        }
        if self.content_source_binding is not None:
            value.update(
                {
                    "content_source_binding": self.content_source_binding.to_dict(),
                    "trusted_model_member_sha256s": list(
                        self.trusted_model_member_sha256s
                    ),
                    "trusted_workload_member_sha256": (
                        self.trusted_workload_member_sha256
                    ),
                    "trusted_burstgpt_release_sha256": (
                        self.trusted_burstgpt_release_sha256
                    ),
                }
            )
        return value

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        base = {
            "prepared_model_set_sha256",
            "prepared_model_content_authority_sha256",
            "formal_workload_lock_sha256",
            "burstgpt_shape_authority",
        }
        trusted = {
            "content_source_binding",
            "trusted_model_member_sha256s",
            "trusted_workload_member_sha256",
            "trusted_burstgpt_release_sha256",
        }
        if type(value) is not dict:
            raise TypeError("preflight input locks must be an object")
        keys = frozenset(value)
        if keys not in {frozenset(base), frozenset(base | trusted)}:
            raise ValueError("preflight input locks fields differ from schema")
        row = dict(value)
        raw_burst = row.pop("burstgpt_shape_authority")
        burst = (
            None if raw_burst is None else BurstGptShapeAuthority.from_dict(raw_burst)
        )
        if "content_source_binding" not in row:
            return cls(burstgpt_shape_authority=burst, **row)
        from lightcone_spec.experiments.formal_content_source import (
            FormalContentSourceBinding,
        )

        raw_models = row.pop("trusted_model_member_sha256s")
        if type(raw_models) is not list:
            raise TypeError("trusted preflight model members must be an array")
        content_source = FormalContentSourceBinding.from_dict(
            row.pop("content_source_binding")
        )
        return cls(
            burstgpt_shape_authority=burst,
            content_source_binding=content_source,
            trusted_model_member_sha256s=tuple(raw_models),
            **row,
        )


@dataclass(frozen=True)
class ExactnessLoaderEnvironment:
    """Exact non-secret CUDA/NCCL loader inputs for the isolated child."""

    path_entries: tuple[str, ...]
    library_path_entries: tuple[str, ...]
    cuda_home: str

    def __post_init__(self) -> None:
        for label, values in (
            ("PATH", self.path_entries),
            ("LD_LIBRARY_PATH", self.library_path_entries),
        ):
            if (
                type(values) is not tuple
                or not values
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"exactness {label} entries must be unique")
            for value in values:
                directory = _absolute_path(f"exactness {label} entry", value)
                if not directory.is_dir() or directory.is_symlink():
                    raise ValueError(f"exactness {label} entry is unavailable")
        cuda_home = _absolute_path("exactness CUDA home", self.cuda_home)
        if not cuda_home.is_dir() or cuda_home.is_symlink():
            raise ValueError("exactness CUDA home is unavailable")

    def to_dict(self) -> dict[str, object]:
        return {
            "path_entries": list(self.path_entries),
            "library_path_entries": list(self.library_path_entries),
            "cuda_home": self.cuda_home,
        }

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    def child_environment(self) -> dict[str, str]:
        self.__post_init__()
        return {
            "PATH": os.pathsep.join(self.path_entries),
            "LD_LIBRARY_PATH": os.pathsep.join(self.library_path_entries),
            "CUDA_HOME": self.cuda_home,
            "CUDA_PATH": self.cuda_home,
            "NCCL_DEBUG": "INFO",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "exactness loader environment",
            value,
            frozenset({"path_entries", "library_path_entries", "cuda_home"}),
        )
        path_entries = row.pop("path_entries")
        library_entries = row.pop("library_path_entries")
        if type(path_entries) is not list or type(library_entries) is not list:
            raise TypeError("exactness loader entries must be arrays")
        return cls(
            path_entries=tuple(path_entries),
            library_path_entries=tuple(library_entries),
            **row,
        )


PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256 = _sha(
    {
        "schema_version": 3,
        "kind": "first_party_non_serving_exactness_runner",
        "source": (PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE),
        "test_file": PREFLIGHT_EXACTNESS_TEST_FILE,
        "test_names": PREFLIGHT_EXACTNESS_TEST_NAMES,
        "dispatch": "dynamic_root_signed_control_before_process_or_gpu_mutation",
        "raw_rank_authority": (
            "patched_suite_publishes_two_atomic_rank_terminals;runner_only_aggregates"
        ),
        "terminal": (
            "exact_junit_eight_pass_zero_skip_two_validated_raw_rank_terminals_"
            "and_clean_two_gpu_processes"
        ),
        "process_lifecycle": (
            "new_process_group_with_bounded_term_kill_and_final_gpu_snapshot"
        ),
        "loader": "path_bound_PATH_LD_LIBRARY_PATH_CUDA_HOME_no_inherited_env",
        "dispatch_binding": (
            "exact_physical_assignment_and_experiment_budget_from_sealed_preflight"
        ),
        "publication": "terminal_then_atomic_no_replace_result_pointer",
    }
)

PREFLIGHT_EXACTNESS_QUALIFICATION_PROTOCOL_SHA256 = _sha(
    {
        "schema_version": 1,
        "kind": "first_party_exactness_rank_aggregate_qualification",
        "runner_protocol_sha256": PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256,
        "test_names": PREFLIGHT_EXACTNESS_TEST_NAMES,
        "raw_authority": (
            "schema3_pointer_exact_junit_two_rank_terminals_and_native_terminal_"
            "identities"
        ),
        "control": (
            "post_run_dynamic_root_authorized_rank_aggregate_with_atomic_replay"
        ),
        "publication": (
            "proof_artifact_then_schema4_qualified_pointer_atomic_no_replace"
        ),
    }
)


@dataclass(frozen=True)
class ExactnessPreflightAssignment:
    schema_version: int
    kind: Literal["formal_exactness_preflight_assignment"]
    protocol_sha256: str
    registry_sha256: str
    cell_id: str
    runtime_sha256: str
    split_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    physical_assignment_sha256: str
    experiment_budget_sha256: str
    gpu_uuids: tuple[str, str]
    gpu_model: str
    patched_sglang_checkout: str
    patched_sglang_commit: str
    patched_sglang_tree: str
    python_executable: str
    python_raw_sha256: str
    nvidia_smi_executable: str
    nvidia_smi_raw_sha256: str
    python_version: str
    torch_version: str
    cuda_version: str
    driver_version: str
    input_locks: PreflightInputLocks
    loader_environment: ExactnessLoaderEnvironment
    evidence_directory: str

    def __post_init__(self) -> None:
        if self.schema_version != 3 or self.kind != (
            "formal_exactness_preflight_assignment"
        ):
            raise ValueError("exactness assignment schema is unsupported")
        if self.protocol_sha256 != PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256:
            raise ValueError("exactness assignment uses another runner")
        for label, digest in (
            ("registry", self.registry_sha256),
            ("cell", self.cell_id),
            ("runtime", self.runtime_sha256),
            ("split", self.split_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
            ("physical assignment", self.physical_assignment_sha256),
            ("experiment budget", self.experiment_budget_sha256),
            ("Python executable", self.python_raw_sha256),
            ("nvidia-smi executable", self.nvidia_smi_raw_sha256),
        ):
            _require_sha256(f"exactness {label}", digest)
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != 2
            or len(set(self.gpu_uuids)) != 2
        ):
            raise ValueError("exactness assignment requires exactly two unique GPUs")
        for value in self.gpu_uuids:
            _require_text("exactness GPU UUID", value)
        for label, value in (
            ("GPU model", self.gpu_model),
            ("Python version", self.python_version),
            ("Torch version", self.torch_version),
            ("CUDA version", self.cuda_version),
            ("driver version", self.driver_version),
        ):
            _require_text(f"exactness {label}", value)
        checkout = _absolute_path(
            "exactness patched SGLang checkout", self.patched_sglang_checkout
        )
        python = _absolute_path("exactness Python", self.python_executable)
        nvidia_smi = _absolute_path("exactness nvidia-smi", self.nvidia_smi_executable)
        evidence = _absolute_path(
            "exactness evidence directory", self.evidence_directory
        )
        if self.patched_sglang_commit != PINNED_SGLANG_COMMIT or (
            self.patched_sglang_tree != PINNED_SGLANG_TREE
        ):
            raise ValueError("exactness assignment uses another patched SGLang")
        if _GIT_SHA.fullmatch(self.patched_sglang_commit) is None or (
            _GIT_SHA.fullmatch(self.patched_sglang_tree) is None
        ):
            raise ValueError("exactness patched SGLang identity is invalid")
        if type(self.input_locks) is not PreflightInputLocks:
            raise TypeError("exactness assignment requires exact input locks")
        if type(self.loader_environment) is not ExactnessLoaderEnvironment:
            raise TypeError("exactness assignment requires exact loader environment")
        self.loader_environment.__post_init__()
        if not checkout.is_dir() or checkout.is_symlink():
            raise ValueError("exactness patched SGLang checkout is unavailable")
        if not evidence.is_dir() or evidence.is_symlink():
            raise ValueError("exactness evidence directory is unavailable")
        for label, path, digest in (
            ("Python", python, self.python_raw_sha256),
            ("nvidia-smi", nvidia_smi, self.nvidia_smi_raw_sha256),
        ):
            actual, size = _raw_file(path, label=f"exactness {label}")
            if size < 1 or actual != digest:
                raise ValueError(f"exactness {label} executable changed")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "gpu_uuids": list(self.gpu_uuids),
            "input_locks": self.input_locks.to_dict(),
            "loader_environment": self.loader_environment.to_dict(),
        }

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @cached_property
    def dispatch_lineage_sha256(self) -> str:
        return _sha(
            {
                "schema_version": 1,
                "kind": "formal_exactness_preflight_dispatch_lineage",
                "assignment_sha256": self.sha256,
                "inventory_sha256": self.inventory_sha256,
                "hardware_envelope_sha256": self.hardware_envelope_sha256,
                "input_locks_sha256": self.input_locks.sha256,
                "loader_environment_sha256": self.loader_environment.sha256,
                "patched_sglang_tree": self.patched_sglang_tree,
                "toolchain": {
                    "python": self.python_version,
                    "torch": self.torch_version,
                    "cuda": self.cuda_version,
                    "driver": self.driver_version,
                },
            }
        )

    def write(self, path: str | Path) -> Path:
        destination = _absolute_path("exactness assignment", str(path))
        _publish_json(destination, self.to_dict())
        return destination

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = frozenset(
            {
                "schema_version",
                "kind",
                "protocol_sha256",
                "registry_sha256",
                "cell_id",
                "runtime_sha256",
                "split_sha256",
                "inventory_sha256",
                "hardware_envelope_sha256",
                "gpu_uuids",
                "gpu_model",
                "physical_assignment_sha256",
                "experiment_budget_sha256",
                "patched_sglang_checkout",
                "patched_sglang_commit",
                "patched_sglang_tree",
                "python_executable",
                "python_raw_sha256",
                "nvidia_smi_executable",
                "nvidia_smi_raw_sha256",
                "python_version",
                "torch_version",
                "cuda_version",
                "driver_version",
                "input_locks",
                "loader_environment",
                "evidence_directory",
            }
        )
        row = _strict_object("exactness assignment", value, fields)
        gpu_uuids = row.pop("gpu_uuids")
        input_locks = PreflightInputLocks.from_dict(row.pop("input_locks"))
        loader_environment = ExactnessLoaderEnvironment.from_dict(
            row.pop("loader_environment")
        )
        if type(gpu_uuids) is not list:
            raise TypeError("exactness GPU UUIDs must be an array")
        return cls(
            gpu_uuids=tuple(gpu_uuids),
            input_locks=input_locks,
            loader_environment=loader_environment,
            **row,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = _absolute_path("exactness assignment", str(path))
        return cls.from_dict(_load_json(source, label="exactness assignment"))


@dataclass(frozen=True)
class EvidenceFileBinding:
    absolute_path: str
    raw_sha256: str
    size: int

    @classmethod
    def bind(cls, path: Path, *, label: str) -> Self:
        digest, size = _raw_file(path, label=label)
        if size < 1:
            raise ValueError(f"{label} is empty")
        return cls(str(path), digest, size)

    def reopen(self, *, label: str) -> None:
        identity = _absolute_path(label, self.absolute_path)
        digest, size = _raw_file(relocated_evidence_path(identity), label=label)
        if digest != self.raw_sha256 or size != self.size:
            raise ValueError(f"{label} changed")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object, *, label: str) -> Self:
        row = _strict_object(
            label,
            value,
            frozenset({"absolute_path", "raw_sha256", "size"}),
        )
        binding = cls(**row)
        _absolute_path(label, binding.absolute_path)
        _require_sha256(label, binding.raw_sha256)
        if type(binding.size) is not int or binding.size < 1:
            raise ValueError(f"{label} size is invalid")
        return binding


@dataclass(frozen=True)
class ExactnessControlVerificationReceipt:
    """Reopenable dynamic control decision for the non-serving GPU suite."""

    schema_version: int
    kind: Literal["formal_exactness_control_verification_receipt"]
    verified_ns: int
    assignment_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    control_envelope: ControlArtifactAttestation
    verified_control: VerifiedControlArtifact
    reservation_sha256: str
    reservation_record: EvidenceFileBinding

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_exactness_control_verification_receipt"
        ):
            raise ValueError("exactness control receipt schema is unsupported")
        if type(self.verified_ns) is not int or self.verified_ns < 0:
            raise ValueError("exactness control verification time is invalid")
        for label, digest in (
            ("assignment", self.assignment_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
            ("reservation", self.reservation_sha256),
        ):
            _require_sha256(f"exactness control {label}", digest)
        if type(self.control_envelope) is not ControlArtifactAttestation:
            raise TypeError("exactness control requires an exact envelope")
        if type(self.verified_control) is not VerifiedControlArtifact:
            raise TypeError("exactness control requires an exact verified result")
        if type(self.reservation_record) is not EvidenceFileBinding:
            raise TypeError("exactness control requires a path-bound reservation")
        replayed = verify_release_control_artifact_attestation(
            self.control_envelope,
            expected_inventory_sha256=self.inventory_sha256,
            now_ns=self.verified_ns,
            consumed_challenge_sha256s=(),
        )
        if replayed != self.verified_control:
            raise ValueError("exactness control verification result changed")
        subject = self.control_envelope.subject
        if (
            replayed.artifact_type != "non_serving_terminal"
            or replayed.artifact_sha256 != self.assignment_sha256
            or subject.protocol_sha256 != PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256
            or self.control_envelope.hardware_envelope_sha256
            != self.hardware_envelope_sha256
        ):
            raise ValueError("exactness control subject differs from assignment")
        expected_reservation = control_challenge_reservation_sha256(
            (replayed,), reserved_ns=self.verified_ns
        )
        if expected_reservation != self.reservation_sha256:
            raise ValueError("exactness control reservation identity changed")
        reservation_path = _absolute_path(
            "exactness control reservation", self.reservation_record.absolute_path
        )
        if reservation_path.name != f"reservation-{self.reservation_sha256}.json":
            raise ValueError("exactness control reservation path is not exact")
        self.reservation_record.reopen(label="exactness control reservation")
        reservation_body, _reservation_raw, _reservation_size = _stable_file_bytes(
            reservation_path, label="exactness control reservation"
        )
        try:
            reservation = json.loads(reservation_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "exactness control reservation is not UTF-8 JSON"
            ) from error
        if (
            type(reservation) is not dict
            or reservation_body != _canonical_bytes(reservation) + b"\n"
        ):
            raise ValueError("exactness control reservation is not canonical JSON")
        expected_challenges = tuple(
            sorted(
                {
                    replayed.challenge_sha256,
                    replayed.deployment_policy_challenge_sha256,
                }
            )
        )
        if reservation != {
            "schema_version": 2,
            "kind": "lightcone_control_challenge_reservation",
            "reserved_ns": self.verified_ns,
            "challenge_sha256s": list(expected_challenges),
        }:
            raise ValueError("exactness control reservation content differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "verified_ns": self.verified_ns,
            "assignment_sha256": self.assignment_sha256,
            "inventory_sha256": self.inventory_sha256,
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "control_envelope": self.control_envelope.to_dict(),
            "verified_control": asdict(self.verified_control),
            "reservation_sha256": self.reservation_sha256,
            "reservation_record": self.reservation_record.to_dict(),
        }

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "exactness control verification receipt",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "verified_ns",
                    "assignment_sha256",
                    "inventory_sha256",
                    "hardware_envelope_sha256",
                    "control_envelope",
                    "verified_control",
                    "reservation_sha256",
                    "reservation_record",
                }
            ),
        )
        envelope = ControlArtifactAttestation.from_dict(row.pop("control_envelope"))
        verified_row = row.pop("verified_control")
        if type(verified_row) is not dict:
            raise TypeError("exactness verified control must be an object")
        verified = VerifiedControlArtifact(**verified_row)
        reservation = EvidenceFileBinding.from_dict(
            row.pop("reservation_record"), label="exactness control reservation"
        )
        return cls(
            control_envelope=envelope,
            verified_control=verified,
            reservation_record=reservation,
            **row,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = _absolute_path("exactness control receipt", str(path))
        return cls.from_dict(_load_json(source, label="exactness control receipt"))


@dataclass(frozen=True)
class ExactnessRankTerminal:
    """One raw first-party rank terminal emitted by the patched GPU suite.

    This is deliberately not a second qualification receipt.  The patched
    runtime owns the native observation named by ``native_terminal_sha256``;
    this runner only validates and aggregates the two immutable rank files.
    """

    schema_version: int
    kind: Literal["formal_exactness_raw_rank_terminal"]
    runner_protocol_sha256: str
    assignment_sha256: str
    global_rank: int
    gpu_uuid: str
    status: Literal["PASSED", "FAILED", "ERROR"]
    started_ns: int
    finished_ns: int
    process_id: int
    process_started_ns: int
    completed_test_names: tuple[str, ...]
    native_terminal_sha256: str | None
    observation_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_exactness_raw_rank_terminal"
        ):
            raise ValueError("exactness rank terminal schema is unsupported")
        if self.runner_protocol_sha256 != PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256:
            raise ValueError("exactness rank terminal uses another runner")
        _require_sha256("exactness rank assignment", self.assignment_sha256)
        _require_sha256("exactness rank observation", self.observation_sha256)
        if self.native_terminal_sha256 is not None:
            _require_sha256(
                "exactness rank native terminal", self.native_terminal_sha256
            )
        if type(self.global_rank) is not int or self.global_rank not in {0, 1}:
            raise ValueError("exactness rank terminal rank must be zero or one")
        _require_text("exactness rank GPU UUID", self.gpu_uuid)
        if self.status not in {"PASSED", "FAILED", "ERROR"}:
            raise ValueError("exactness rank terminal status is unsupported")
        if (
            type(self.started_ns) is not int
            or type(self.finished_ns) is not int
            or self.started_ns < 1
            or self.finished_ns < self.started_ns
            or type(self.process_id) is not int
            or self.process_id < 1
            or type(self.process_started_ns) is not int
            or self.process_started_ns < 1
        ):
            raise ValueError("exactness rank terminal process/time fields are invalid")
        if (
            type(self.completed_test_names) is not tuple
            or self.completed_test_names
            != tuple(sorted(set(self.completed_test_names)))
            or any(
                name not in PREFLIGHT_EXACTNESS_TEST_NAMES
                for name in self.completed_test_names
            )
        ):
            raise ValueError("exactness rank completed tests are not an exact subset")
        complete = (
            self.completed_test_names == PREFLIGHT_EXACTNESS_TEST_NAMES
            and self.native_terminal_sha256 is not None
        )
        if (self.status == "PASSED") != complete:
            raise ValueError("exactness rank status differs from raw terminal coverage")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "completed_test_names": list(self.completed_test_names),
        }

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "exactness rank terminal",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "runner_protocol_sha256",
                    "assignment_sha256",
                    "global_rank",
                    "gpu_uuid",
                    "status",
                    "started_ns",
                    "finished_ns",
                    "process_id",
                    "process_started_ns",
                    "completed_test_names",
                    "native_terminal_sha256",
                    "observation_sha256",
                }
            ),
        )
        names = row.pop("completed_test_names")
        if type(names) is not list:
            raise TypeError("exactness rank completed tests must be an array")
        return cls(completed_test_names=tuple(names), **row)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = _absolute_path("exactness rank terminal", str(path))
        return cls.from_dict(_load_json(source, label="exactness rank terminal"))


@dataclass(frozen=True)
class ExactnessPreflightTerminal:
    schema_version: int
    kind: Literal["formal_exactness_preflight_terminal"]
    assignment_sha256: str
    dispatch_attestation_sha256: str
    replay_reservation_sha256: str
    status: Literal["PASSED", "FAILED", "ERROR"]
    started_ns: int
    finished_ns: int
    process_exit_code: int
    test_names: tuple[str, ...]
    tests_collected: int
    tests_passed: int
    tests_failed: int
    tests_errored: int
    tests_skipped: int
    terminal_rank_count: int
    rank_terminal_sha256s: tuple[str, ...]
    before_gpu_snapshot_sha256: str
    after_gpu_snapshot_sha256: str
    before_gpu_snapshot_status: Literal["AVAILABLE", "ERROR"]
    after_gpu_snapshot_status: Literal["AVAILABLE", "ERROR"]
    before_compute_process_count: int
    after_compute_process_count: int
    process_group_cleanup: Literal["CLEAN", "TERM_CLEAN", "KILL_CLEAN", "ERROR"]
    process_group_id: int | None
    runner_error_code: str | None
    junit_xml_sha256: str | None
    log_sha256: str
    authority_mode: Literal["dynamic_control", "formal_single_operator_v1"] = (
        "dynamic_control"
    )

    def __post_init__(self) -> None:
        if (
            self.kind != "formal_exactness_preflight_terminal"
            or (self.schema_version == 2 and self.authority_mode != "dynamic_control")
            or (
                self.schema_version == 3
                and self.authority_mode != "formal_single_operator_v1"
            )
            or self.schema_version not in {2, 3}
        ):
            raise ValueError("exactness terminal schema is unsupported")
        for label, digest in (
            ("assignment", self.assignment_sha256),
            ("dispatch attestation", self.dispatch_attestation_sha256),
            ("replay reservation", self.replay_reservation_sha256),
            ("before GPU snapshot", self.before_gpu_snapshot_sha256),
            ("after GPU snapshot", self.after_gpu_snapshot_sha256),
            ("log", self.log_sha256),
        ):
            _require_sha256(f"exactness terminal {label}", digest)
        if self.junit_xml_sha256 is not None:
            _require_sha256("exactness terminal JUnit", self.junit_xml_sha256)
        if (
            type(self.started_ns) is not int
            or type(self.finished_ns) is not int
            or self.started_ns < 1
            or self.finished_ns < self.started_ns
        ):
            raise ValueError("exactness terminal time order is invalid")
        for label, value in (
            ("exit code", self.process_exit_code),
            ("collected", self.tests_collected),
            ("passed", self.tests_passed),
            ("failed", self.tests_failed),
            ("errored", self.tests_errored),
            ("skipped", self.tests_skipped),
            ("terminal rank count", self.terminal_rank_count),
            ("before compute process count", self.before_compute_process_count),
            ("after compute process count", self.after_compute_process_count),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"exactness terminal {label} is invalid")
        if (
            type(self.rank_terminal_sha256s) is not tuple
            or self.terminal_rank_count != len(self.rank_terminal_sha256s)
            or len(set(self.rank_terminal_sha256s)) != len(self.rank_terminal_sha256s)
        ):
            raise ValueError("exactness terminal rank aggregate is invalid")
        for digest in self.rank_terminal_sha256s:
            _require_sha256("exactness terminal rank", digest)
        if self.before_gpu_snapshot_status not in {"AVAILABLE", "ERROR"} or (
            self.after_gpu_snapshot_status not in {"AVAILABLE", "ERROR"}
        ):
            raise ValueError("exactness terminal GPU snapshot status is unsupported")
        if self.process_group_cleanup not in {
            "CLEAN",
            "TERM_CLEAN",
            "KILL_CLEAN",
            "ERROR",
        }:
            raise ValueError("exactness process-group cleanup status is unsupported")
        if self.process_group_id is not None and (
            type(self.process_group_id) is not int or self.process_group_id < 1
        ):
            raise ValueError("exactness terminal process-group ID is invalid")
        if self.runner_error_code is not None:
            _require_text("exactness terminal runner error", self.runner_error_code)
        clean = (
            self.process_exit_code == 0
            and self.test_names == PREFLIGHT_EXACTNESS_TEST_NAMES
            and (
                self.tests_collected,
                self.tests_passed,
                self.tests_failed,
                self.tests_errored,
                self.tests_skipped,
                self.terminal_rank_count,
            )
            == (8, 8, 0, 0, 0, 2)
            and len(self.rank_terminal_sha256s) == 2
            and self.before_gpu_snapshot_status == "AVAILABLE"
            and self.after_gpu_snapshot_status == "AVAILABLE"
            and self.before_compute_process_count == 0
            and self.after_compute_process_count == 0
            and self.process_group_cleanup == "CLEAN"
            and self.process_group_id is not None
            and self.runner_error_code is None
            and self.junit_xml_sha256 is not None
        )
        if (self.status == "PASSED") != clean:
            raise ValueError("exactness terminal status differs from exact coverage")
        if self.status not in {"PASSED", "FAILED", "ERROR"}:
            raise ValueError("exactness terminal status is unsupported")

    def to_dict(self) -> dict[str, object]:
        value = {
            **asdict(self),
            "test_names": list(self.test_names),
            "rank_terminal_sha256s": list(self.rank_terminal_sha256s),
        }
        if self.schema_version == 2:
            value.pop("authority_mode")
        return value

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("exactness terminal must be an object")
        schema_version = value.get("schema_version")
        fields = {
            "schema_version",
            "kind",
            "assignment_sha256",
            "dispatch_attestation_sha256",
            "replay_reservation_sha256",
            "status",
            "started_ns",
            "finished_ns",
            "process_exit_code",
            "test_names",
            "tests_collected",
            "tests_passed",
            "tests_failed",
            "tests_errored",
            "tests_skipped",
            "terminal_rank_count",
            "rank_terminal_sha256s",
            "before_gpu_snapshot_sha256",
            "after_gpu_snapshot_sha256",
            "before_gpu_snapshot_status",
            "after_gpu_snapshot_status",
            "before_compute_process_count",
            "after_compute_process_count",
            "process_group_cleanup",
            "runner_error_code",
            "process_group_id",
            "junit_xml_sha256",
            "log_sha256",
        }
        if schema_version == 3:
            fields.add("authority_mode")
        row = _strict_object(
            "exactness terminal",
            value,
            frozenset(fields),
        )
        names = row.pop("test_names")
        rank_sha256s = row.pop("rank_terminal_sha256s")
        if type(names) is not list or type(rank_sha256s) is not list:
            raise TypeError("exactness terminal names/digests must be arrays")
        return cls(
            test_names=tuple(names),
            rank_terminal_sha256s=tuple(rank_sha256s),
            **row,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = _absolute_path("exactness terminal", str(path))
        return cls.from_dict(_load_json(source, label="exactness terminal"))


@dataclass(frozen=True)
class ExactnessQualificationPayload:
    """The complete post-run statement authorized by rank-aggregate control."""

    schema_version: int
    kind: Literal["formal_exactness_qualification_payload"]
    raw_result_pointer: EvidenceFileBinding
    raw_result_pointer_semantic_sha256: str
    assignment_sha256: str
    terminal_sha256: str
    junit_xml_sha256: str
    rank_terminal_sha256s: tuple[str, str]
    native_terminal_sha256s: tuple[str, str]
    registry_sha256: str
    runtime_sha256: str
    split_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    gpu_uuids: tuple[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_exactness_qualification_payload"
        ):
            raise ValueError("exactness qualification payload schema is unsupported")
        if type(self.raw_result_pointer) is not EvidenceFileBinding:
            raise TypeError("exactness qualification requires a raw result pointer")
        self.raw_result_pointer.reopen(
            label="exactness qualification raw result pointer"
        )
        for label, digest in (
            ("raw result semantic", self.raw_result_pointer_semantic_sha256),
            ("assignment", self.assignment_sha256),
            ("terminal", self.terminal_sha256),
            ("JUnit XML", self.junit_xml_sha256),
            ("registry", self.registry_sha256),
            ("runtime", self.runtime_sha256),
            ("split", self.split_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
        ):
            _require_sha256(f"exactness qualification {label}", digest)
        if (
            type(self.rank_terminal_sha256s) is not tuple
            or len(self.rank_terminal_sha256s) != 2
            or len(set(self.rank_terminal_sha256s)) != 2
            or type(self.native_terminal_sha256s) is not tuple
            or len(self.native_terminal_sha256s) != 2
            or len(set(self.native_terminal_sha256s)) != 2
        ):
            raise ValueError("exactness qualification requires two unique rank proofs")
        for label, values in (
            ("rank terminal", self.rank_terminal_sha256s),
            ("native terminal", self.native_terminal_sha256s),
        ):
            for digest in values:
                _require_sha256(f"exactness qualification {label}", digest)
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != 2
            or len(set(self.gpu_uuids)) != 2
        ):
            raise ValueError("exactness qualification requires two unique GPUs")
        for gpu_uuid in self.gpu_uuids:
            _require_text("exactness qualification GPU UUID", gpu_uuid)

    @cached_property
    def lineage_sha256(self) -> str:
        return _sha(
            {
                "schema_version": 1,
                "kind": "formal_exactness_qualification_lineage",
                "qualification_protocol_sha256": (
                    PREFLIGHT_EXACTNESS_QUALIFICATION_PROTOCOL_SHA256
                ),
                "runner_protocol_sha256": (PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256),
                "raw_result_pointer_semantic_sha256": (
                    self.raw_result_pointer_semantic_sha256
                ),
                "assignment_sha256": self.assignment_sha256,
                "terminal_sha256": self.terminal_sha256,
                "junit_xml_sha256": self.junit_xml_sha256,
                "rank_terminal_sha256s": self.rank_terminal_sha256s,
                "native_terminal_sha256s": self.native_terminal_sha256s,
                "inventory_sha256": self.inventory_sha256,
                "hardware_envelope_sha256": self.hardware_envelope_sha256,
                "gpu_uuids": self.gpu_uuids,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "raw_result_pointer": self.raw_result_pointer.to_dict(),
            "raw_result_pointer_semantic_sha256": (
                self.raw_result_pointer_semantic_sha256
            ),
            "assignment_sha256": self.assignment_sha256,
            "terminal_sha256": self.terminal_sha256,
            "junit_xml_sha256": self.junit_xml_sha256,
            "rank_terminal_sha256s": list(self.rank_terminal_sha256s),
            "native_terminal_sha256s": list(self.native_terminal_sha256s),
            "registry_sha256": self.registry_sha256,
            "runtime_sha256": self.runtime_sha256,
            "split_sha256": self.split_sha256,
            "inventory_sha256": self.inventory_sha256,
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "gpu_uuids": list(self.gpu_uuids),
        }

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "exactness qualification payload",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "raw_result_pointer",
                    "raw_result_pointer_semantic_sha256",
                    "assignment_sha256",
                    "terminal_sha256",
                    "junit_xml_sha256",
                    "rank_terminal_sha256s",
                    "native_terminal_sha256s",
                    "registry_sha256",
                    "runtime_sha256",
                    "split_sha256",
                    "inventory_sha256",
                    "hardware_envelope_sha256",
                    "gpu_uuids",
                }
            ),
        )
        raw_pointer = EvidenceFileBinding.from_dict(
            row.pop("raw_result_pointer"),
            label="exactness qualification raw result pointer",
        )
        rank_sha256s = row.pop("rank_terminal_sha256s")
        native_sha256s = row.pop("native_terminal_sha256s")
        gpu_uuids = row.pop("gpu_uuids")
        if any(
            type(values) is not list
            for values in (
                rank_sha256s,
                native_sha256s,
                gpu_uuids,
            )
        ):
            raise TypeError("exactness qualification arrays are malformed")
        return cls(
            raw_result_pointer=raw_pointer,
            rank_terminal_sha256s=tuple(rank_sha256s),
            native_terminal_sha256s=tuple(native_sha256s),
            gpu_uuids=tuple(gpu_uuids),
            **row,
        )


@dataclass(frozen=True)
class ExactnessPreflightResultPointer:
    schema_version: int
    kind: Literal["formal_exactness_preflight_result_pointer"]
    assignment: EvidenceFileBinding
    before_gpu_snapshot: EvidenceFileBinding
    after_gpu_snapshot: EvidenceFileBinding
    log: EvidenceFileBinding
    terminal: EvidenceFileBinding
    rank_terminals: tuple[EvidenceFileBinding, ...]
    junit_xml: EvidenceFileBinding | None
    control_verification_receipt: EvidenceFileBinding | None = None
    qualification_proof_artifact: EvidenceFileBinding | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {2, 3, 4} or self.kind != (
            "formal_exactness_preflight_result_pointer"
        ):
            raise ValueError("exactness result pointer schema is unsupported")
        if self.schema_version == 2 and (
            self.control_verification_receipt is not None
            or self.qualification_proof_artifact is not None
        ):
            raise ValueError("diagnostic exactness pointer cannot claim formal control")
        if (
            self.schema_version in {3, 4}
            and type(self.control_verification_receipt) is not EvidenceFileBinding
        ):
            raise TypeError("formal exactness pointer lacks path-bound control")
        if self.schema_version == 3 and self.qualification_proof_artifact is not None:
            raise ValueError(
                "raw exactness pointer cannot claim post-run qualification"
            )
        if (
            self.schema_version == 4
            and type(self.qualification_proof_artifact) is not EvidenceFileBinding
        ):
            raise TypeError("qualified exactness pointer lacks rank-aggregate proof")
        for label, binding in self.bindings().items():
            if type(binding) is not EvidenceFileBinding:
                raise TypeError(f"exactness result {label} binding is invalid")
            binding.reopen(label=f"exactness result {label}")
        if (
            type(self.rank_terminals) is not tuple
            or len(self.rank_terminals) > 2
            or len({binding.absolute_path for binding in self.rank_terminals})
            != len(self.rank_terminals)
        ):
            raise ValueError("exactness result rank terminal bindings are invalid")
        _validate_result_pointer_semantics(self)

    def bindings(self) -> dict[str, EvidenceFileBinding]:
        rows = {
            "assignment": self.assignment,
            "before_gpu_snapshot": self.before_gpu_snapshot,
            "after_gpu_snapshot": self.after_gpu_snapshot,
            "log": self.log,
            "terminal": self.terminal,
        }
        if self.junit_xml is not None:
            rows["junit_xml"] = self.junit_xml
        if self.control_verification_receipt is not None:
            rows["control_verification_receipt"] = self.control_verification_receipt
        if self.qualification_proof_artifact is not None:
            rows["qualification_proof_artifact"] = self.qualification_proof_artifact
        for index, binding in enumerate(self.rank_terminals):
            rows[f"rank_terminal_{index}"] = binding
        return rows

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "assignment": asdict(self.assignment),
            "before_gpu_snapshot": asdict(self.before_gpu_snapshot),
            "after_gpu_snapshot": asdict(self.after_gpu_snapshot),
            "log": asdict(self.log),
            "terminal": asdict(self.terminal),
            "rank_terminals": [asdict(binding) for binding in self.rank_terminals],
            "junit_xml": None if self.junit_xml is None else asdict(self.junit_xml),
        }
        if self.schema_version in {3, 4}:
            if self.control_verification_receipt is None:
                raise AssertionError("formal exactness pointer lost control binding")
            value["control_verification_receipt"] = asdict(
                self.control_verification_receipt
            )
        if self.schema_version == 4:
            if self.qualification_proof_artifact is None:
                raise AssertionError("qualified exactness pointer lost aggregate proof")
            value["qualification_proof_artifact"] = asdict(
                self.qualification_proof_artifact
            )
        return value

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("exactness result pointer must be a JSON object")
        schema_version = value.get("schema_version")
        fields = {
            "schema_version",
            "kind",
            "assignment",
            "before_gpu_snapshot",
            "after_gpu_snapshot",
            "log",
            "terminal",
            "rank_terminals",
            "junit_xml",
        }
        if schema_version in {3, 4}:
            fields.add("control_verification_receipt")
        if schema_version == 4:
            fields.add("qualification_proof_artifact")
        row = _strict_object(
            "exactness result pointer",
            value,
            frozenset(fields),
        )
        fields = {
            label: EvidenceFileBinding(
                **_strict_object(
                    f"exactness result {label}",
                    row[label],
                    frozenset({"absolute_path", "raw_sha256", "size"}),
                )
            )
            for label in (
                "assignment",
                "before_gpu_snapshot",
                "after_gpu_snapshot",
                "log",
                "terminal",
            )
        }
        rank_terminals = row["rank_terminals"]
        if type(rank_terminals) is not list:
            raise TypeError("exactness result rank terminals must be an array")
        junit = row["junit_xml"]
        control = row.get("control_verification_receipt")
        qualification = row.get("qualification_proof_artifact")
        return cls(
            schema_version=row["schema_version"],
            kind=row["kind"],
            rank_terminals=tuple(
                EvidenceFileBinding(
                    **_strict_object(
                        "exactness result rank terminal",
                        binding,
                        frozenset({"absolute_path", "raw_sha256", "size"}),
                    )
                )
                for binding in rank_terminals
            ),
            junit_xml=(
                None
                if junit is None
                else EvidenceFileBinding(
                    **_strict_object(
                        "exactness result JUnit",
                        junit,
                        frozenset({"absolute_path", "raw_sha256", "size"}),
                    )
                )
            ),
            control_verification_receipt=(
                None
                if control is None
                else EvidenceFileBinding.from_dict(
                    control, label="exactness control verification receipt"
                )
            ),
            qualification_proof_artifact=(
                None
                if qualification is None
                else EvidenceFileBinding.from_dict(
                    qualification,
                    label="exactness qualification proof artifact",
                )
            ),
            **fields,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = _absolute_path("exactness result pointer", str(path))
        return cls.from_dict(_load_json(source, label="exactness result pointer"))


@dataclass(frozen=True)
class ExactnessQualificationProofArtifact:
    """Durable local verification of the exact two-rank preflight aggregate."""

    schema_version: int
    kind: Literal["formal_exactness_qualification_proof_artifact"]
    payload: ExactnessQualificationPayload
    control_attestation: ControlArtifactAttestation
    replay_reservation: ChallengeReplayReservationBinding
    verified_control_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_exactness_qualification_proof_artifact"
        ):
            raise ValueError("exactness qualification proof schema is unsupported")
        if type(self.payload) is not ExactnessQualificationPayload:
            raise TypeError("exactness qualification proof requires its payload")
        if type(self.control_attestation) is not ControlArtifactAttestation:
            raise TypeError("exactness qualification proof requires exact control")
        if type(self.replay_reservation) is not ChallengeReplayReservationBinding:
            raise TypeError("exactness qualification proof requires replay binding")
        _require_sha256(
            "exactness qualification verified control",
            self.verified_control_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "payload": self.payload.to_dict(),
            "control_attestation": self.control_attestation.to_dict(),
            "replay_reservation": self.replay_reservation.to_dict(),
            "verified_control_sha256": self.verified_control_sha256,
        }

    @cached_property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "exactness qualification proof artifact",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "payload",
                    "control_attestation",
                    "replay_reservation",
                    "verified_control_sha256",
                }
            ),
        )
        return cls(
            payload=ExactnessQualificationPayload.from_dict(row.pop("payload")),
            control_attestation=ControlArtifactAttestation.from_dict(
                row.pop("control_attestation")
            ),
            replay_reservation=ChallengeReplayReservationBinding.from_dict(
                row.pop("replay_reservation")
            ),
            **row,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = _absolute_path("exactness qualification proof", str(path))
        return cls.from_dict(_load_json(source, label="exactness qualification proof"))

    def revalidate(self, *, now_ns: int) -> VerifiedControlArtifact:
        """Reopen raw evidence, signatures, and the committed replay group."""

        self.__post_init__()
        if type(now_ns) is not int or now_ns < self.replay_reservation.reserved_ns:
            raise ValueError("exactness qualification time precedes reservation")
        _revalidate_exactness_qualification_payload(self.payload)
        subject = self.control_attestation.subject
        if (
            subject.artifact_type != "rank_aggregate"
            or subject.artifact_sha256 != self.payload.sha256
            or subject.protocol_sha256
            != PREFLIGHT_EXACTNESS_QUALIFICATION_PROTOCOL_SHA256
            or subject.registry_sha256 != self.payload.registry_sha256
            or subject.lineage_sha256 != self.payload.lineage_sha256
            or self.control_attestation.hardware_envelope_sha256
            != self.payload.hardware_envelope_sha256
        ):
            raise ValueError("exactness qualification control subject differs")
        verified = verify_release_control_artifact_attestation(
            self.control_attestation,
            expected_inventory_sha256=self.payload.inventory_sha256,
            now_ns=self.replay_reservation.reserved_ns,
            consumed_challenge_sha256s=(),
        )
        expected_reservation = control_challenge_reservation_sha256(
            (verified,), reserved_ns=self.replay_reservation.reserved_ns
        )
        expected_challenges = tuple(
            sorted(
                {
                    verified.challenge_sha256,
                    verified.deployment_policy_challenge_sha256,
                }
            )
        )
        if (
            self.replay_reservation.revalidate() != expected_challenges
            or self.replay_reservation.reservation_sha256 != expected_reservation
        ):
            raise ValueError("exactness qualification replay reservation differs")
        if _sha(asdict(verified)) != self.verified_control_sha256:
            raise ValueError("exactness qualification verified control changed")
        return verified


class _RunnerOperationalError(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True)
class _ProcessOutcome:
    exit_code: int
    cleanup: Literal["CLEAN", "TERM_CLEAN", "KILL_CLEAN", "ERROR"]
    error_code: str | None
    process_group_id: int | None


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_process_group_gone(process_group_id: int, *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group_id):
            return True
        time.sleep(0.02)
    return not _process_group_exists(process_group_id)


def _bounded_process_group_cleanup(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = 2.0,
) -> Literal["CLEAN", "TERM_CLEAN", "KILL_CLEAN", "ERROR"]:
    """Reap a dedicated process group without ever targeting another group."""

    process_group_id = process.pid
    if process.poll() is None:
        try:
            process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass
    if not _process_group_exists(process_group_id):
        if process.poll() is None:
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                return "ERROR"
        return "CLEAN"
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return "CLEAN"
    except OSError:
        return "ERROR"
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    if _wait_process_group_gone(process_group_id, timeout_seconds=grace_seconds):
        return "TERM_CLEAN"
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        return "ERROR"
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    return (
        "KILL_CLEAN"
        if _wait_process_group_gone(process_group_id, timeout_seconds=grace_seconds)
        else "ERROR"
    )


def _capture_process_group(
    command: tuple[str, ...],
    *,
    timeout_seconds: float,
) -> bytes:
    try:
        process = subprocess.Popen(
            command,
            env={"LANG": "C", "LC_ALL": "C"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        raise _RunnerOperationalError("snapshot_process_start_failed") from error
    timed_out = False
    try:
        try:
            stdout, _stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout = b""
        cleanup = _bounded_process_group_cleanup(process)
        if timed_out:
            raise _RunnerOperationalError("snapshot_process_timeout")
        if cleanup != "CLEAN":
            raise _RunnerOperationalError("snapshot_process_group_not_clean")
        if process.returncode != 0:
            raise _RunnerOperationalError("snapshot_process_failed")
        return stdout
    finally:
        if _process_group_exists(process.pid):
            _bounded_process_group_cleanup(process)


def _snapshot_gpu(assignment: ExactnessPreflightAssignment) -> dict[str, object]:
    """Capture a terminal GPU snapshot; operational failures are evidence rows."""

    captured_ns = time.monotonic_ns()
    rows: list[dict[str, object]] = []
    process_rows: tuple[str, ...] = ()
    status: Literal["AVAILABLE", "ERROR"] = "ERROR"
    error_code: str | None = None
    try:
        gpu_stdout = _capture_process_group(
            (
                assignment.nvidia_smi_executable,
                "--query-gpu=uuid,name,memory.used",
                "--format=csv,noheader,nounits",
            ),
            timeout_seconds=30,
        )
        for line in gpu_stdout.decode("utf-8").splitlines():
            columns = tuple(item.strip() for item in line.split(","))
            if len(columns) != 3:
                raise _RunnerOperationalError("snapshot_gpu_row_malformed")
            uuid_value, name, memory_used = columns
            try:
                memory_used_mib = int(memory_used)
            except ValueError as error:
                raise _RunnerOperationalError(
                    "snapshot_gpu_memory_malformed"
                ) from error
            rows.append(
                {
                    "uuid": uuid_value,
                    "name": name,
                    "memory_used_mib": memory_used_mib,
                }
            )
        if tuple(row["uuid"] for row in rows) != assignment.gpu_uuids or any(
            row["name"] != assignment.gpu_model for row in rows
        ):
            raise _RunnerOperationalError("snapshot_inventory_mismatch")
        process_stdout = _capture_process_group(
            (
                assignment.nvidia_smi_executable,
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ),
            timeout_seconds=30,
        )
        process_rows = tuple(
            line.strip()
            for line in process_stdout.decode("utf-8").splitlines()
            if line.strip()
        )
        status = "AVAILABLE"
    except (UnicodeDecodeError, _RunnerOperationalError) as error:
        error_code = (
            error.reason_code
            if isinstance(error, _RunnerOperationalError)
            else "snapshot_output_not_utf8"
        )
    return {
        "schema_version": 2,
        "kind": "formal_exactness_gpu_snapshot",
        "assignment_sha256": assignment.sha256,
        "inventory_sha256": assignment.inventory_sha256,
        "status": status,
        "gpu_rows": rows,
        "compute_process_rows": list(process_rows),
        "error_code": error_code,
        "captured_ns": captured_ns,
    }


def _validate_gpu_snapshot(
    value: object,
    *,
    assignment: ExactnessPreflightAssignment,
) -> tuple[Literal["AVAILABLE", "ERROR"], int]:
    row = _strict_object(
        "exactness GPU snapshot",
        value,
        frozenset(
            {
                "schema_version",
                "kind",
                "assignment_sha256",
                "inventory_sha256",
                "status",
                "gpu_rows",
                "compute_process_rows",
                "error_code",
                "captured_ns",
            }
        ),
    )
    if row["schema_version"] != 2 or row["kind"] != ("formal_exactness_gpu_snapshot"):
        raise ValueError("exactness GPU snapshot schema is unsupported")
    if row["assignment_sha256"] != assignment.sha256 or (
        row["inventory_sha256"] != assignment.inventory_sha256
    ):
        raise ValueError("exactness GPU snapshot identity differs")
    if row["status"] not in {"AVAILABLE", "ERROR"}:
        raise ValueError("exactness GPU snapshot status is unsupported")
    if type(row["captured_ns"]) is not int or row["captured_ns"] < 1:
        raise ValueError("exactness GPU snapshot timestamp is invalid")
    gpu_rows = row["gpu_rows"]
    process_rows = row["compute_process_rows"]
    if (
        type(gpu_rows) is not list
        or type(process_rows) is not list
        or any(type(item) is not str or not item for item in process_rows)
    ):
        raise TypeError("exactness GPU snapshot rows are invalid")
    if row["status"] == "AVAILABLE":
        expected_rows = [
            {"uuid": gpu_uuid, "name": assignment.gpu_model}
            for gpu_uuid in assignment.gpu_uuids
        ]
        if len(gpu_rows) != 2:
            raise ValueError("exactness GPU snapshot requires two GPU rows")
        for observed, expected in zip(gpu_rows, expected_rows, strict=True):
            item = _strict_object(
                "exactness GPU row",
                observed,
                frozenset({"uuid", "name", "memory_used_mib"}),
            )
            if item["uuid"] != expected["uuid"] or item["name"] != expected["name"]:
                raise ValueError("exactness GPU snapshot inventory differs")
            if type(item["memory_used_mib"]) is not int or item["memory_used_mib"] < 0:
                raise ValueError("exactness GPU snapshot memory is invalid")
        if row["error_code"] is not None:
            raise ValueError("available exactness GPU snapshot has an error")
    else:
        _require_text("exactness GPU snapshot error", row["error_code"])
    return row["status"], len(process_rows)


def _run_logged_process_group(
    command: tuple[str, ...],
    *,
    cwd: str,
    environment: dict[str, str],
    log_fd: int,
    timeout_seconds: float,
) -> _ProcessOutcome:
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError:
        return _ProcessOutcome(
            255,
            "ERROR",
            "qualification_process_start_failed",
            None,
        )
    timed_out = False
    try:
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        cleanup = _bounded_process_group_cleanup(process)
        exit_code = process.returncode
        if type(exit_code) is not int:
            exit_code = 255
        elif exit_code < 0:
            exit_code = 128 + abs(exit_code)
        if timed_out:
            return _ProcessOutcome(
                124,
                cleanup,
                "qualification_process_timeout",
                process.pid,
            )
        if cleanup != "CLEAN":
            return _ProcessOutcome(
                exit_code,
                cleanup,
                "qualification_process_group_required_cleanup",
                process.pid,
            )
        return _ProcessOutcome(exit_code, cleanup, None, process.pid)
    finally:
        if _process_group_exists(process.pid):
            _bounded_process_group_cleanup(process)


def _valid_rank_terminals(
    bindings: tuple[EvidenceFileBinding, ...],
    *,
    assignment: ExactnessPreflightAssignment,
) -> tuple[ExactnessRankTerminal, ...]:
    rows: list[ExactnessRankTerminal] = []
    for binding in bindings:
        try:
            row = ExactnessRankTerminal.load(binding.absolute_path)
        except (OSError, TypeError, ValueError, RuntimeError):
            continue
        if (
            row.assignment_sha256 != assignment.sha256
            or row.gpu_uuid != assignment.gpu_uuids[row.global_rank]
            or any(existing.global_rank == row.global_rank for existing in rows)
        ):
            continue
        rows.append(row)
    return tuple(sorted(rows, key=lambda row: row.global_rank))


def derive_exactness_qualification_payload(
    raw_result_pointer_path: str | Path,
) -> ExactnessQualificationPayload:
    """Deep-reopen one raw schema-3 result and derive the only signable payload."""

    pointer_path = _absolute_path(
        "raw exactness result pointer", str(raw_result_pointer_path)
    )
    pointer = ExactnessPreflightResultPointer.load(pointer_path)
    if (
        pointer.schema_version != 3
        or pointer.control_verification_receipt is None
        or pointer.qualification_proof_artifact is not None
        or pointer.junit_xml is None
    ):
        raise ValueError("exactness qualification requires one raw schema-3 pointer")
    assignment = ExactnessPreflightAssignment.load(pointer.assignment.absolute_path)
    terminal = ExactnessPreflightTerminal.load(pointer.terminal.absolute_path)
    valid_ranks = _valid_rank_terminals(
        pointer.rank_terminals,
        assignment=assignment,
    )
    if (
        terminal.status != "PASSED"
        or terminal.test_names != PREFLIGHT_EXACTNESS_TEST_NAMES
        or (terminal.tests_collected, terminal.tests_passed) != (8, 8)
        or (terminal.tests_failed, terminal.tests_errored, terminal.tests_skipped)
        != (0, 0, 0)
        or tuple(row.global_rank for row in valid_ranks) != (0, 1)
        or any(row.status != "PASSED" for row in valid_ranks)
    ):
        raise ValueError("exactness qualification requires exact passed raw evidence")
    native_terminal_sha256s = tuple(row.native_terminal_sha256 for row in valid_ranks)
    if (
        any(digest is None for digest in native_terminal_sha256s)
        or len(set(native_terminal_sha256s)) != 2
    ):
        raise ValueError("exactness qualification lacks two native terminals")
    payload = ExactnessQualificationPayload(
        schema_version=1,
        kind="formal_exactness_qualification_payload",
        raw_result_pointer=EvidenceFileBinding.bind(
            pointer_path, label="raw exactness result pointer"
        ),
        raw_result_pointer_semantic_sha256=pointer.sha256,
        assignment_sha256=assignment.sha256,
        terminal_sha256=terminal.sha256,
        junit_xml_sha256=pointer.junit_xml.raw_sha256,
        rank_terminal_sha256s=tuple(row.sha256 for row in valid_ranks),
        native_terminal_sha256s=(
            str(native_terminal_sha256s[0]),
            str(native_terminal_sha256s[1]),
        ),
        registry_sha256=assignment.registry_sha256,
        runtime_sha256=assignment.runtime_sha256,
        split_sha256=assignment.split_sha256,
        inventory_sha256=assignment.inventory_sha256,
        hardware_envelope_sha256=assignment.hardware_envelope_sha256,
        gpu_uuids=assignment.gpu_uuids,
    )
    _revalidate_exactness_qualification_payload(payload)
    return payload


def _revalidate_exactness_qualification_payload(
    payload: ExactnessQualificationPayload,
) -> ExactnessPreflightResultPointer:
    if type(payload) is not ExactnessQualificationPayload:
        raise TypeError("exactness qualification requires an exact payload")
    payload.__post_init__()
    pointer = ExactnessPreflightResultPointer.load(
        payload.raw_result_pointer.absolute_path
    )
    if (
        pointer.schema_version != 3
        or pointer.sha256 != payload.raw_result_pointer_semantic_sha256
        or pointer.junit_xml is None
        or pointer.junit_xml.raw_sha256 != payload.junit_xml_sha256
    ):
        raise ValueError("exactness qualification raw pointer identity changed")
    assignment = ExactnessPreflightAssignment.load(pointer.assignment.absolute_path)
    terminal = ExactnessPreflightTerminal.load(pointer.terminal.absolute_path)
    valid_ranks = _valid_rank_terminals(
        pointer.rank_terminals,
        assignment=assignment,
    )
    native_sha256s = tuple(row.native_terminal_sha256 for row in valid_ranks)
    if (
        assignment.sha256 != payload.assignment_sha256
        or terminal.sha256 != payload.terminal_sha256
        or terminal.status != "PASSED"
        or tuple(row.sha256 for row in valid_ranks) != payload.rank_terminal_sha256s
        or native_sha256s != payload.native_terminal_sha256s
        or assignment.registry_sha256 != payload.registry_sha256
        or assignment.runtime_sha256 != payload.runtime_sha256
        or assignment.split_sha256 != payload.split_sha256
        or assignment.inventory_sha256 != payload.inventory_sha256
        or assignment.hardware_envelope_sha256 != payload.hardware_envelope_sha256
        or assignment.gpu_uuids != payload.gpu_uuids
    ):
        raise ValueError("exactness qualification payload differs from raw evidence")
    return pointer


def _validate_result_pointer_semantics(
    pointer: ExactnessPreflightResultPointer,
) -> None:
    assignment = ExactnessPreflightAssignment.load(pointer.assignment.absolute_path)
    terminal = ExactnessPreflightTerminal.from_dict(
        _load_json(Path(pointer.terminal.absolute_path), label="exactness terminal")
    )
    if terminal.assignment_sha256 != assignment.sha256:
        raise ValueError("exactness pointer terminal differs from assignment")
    if pointer.schema_version in {3, 4}:
        if pointer.control_verification_receipt is None:
            raise AssertionError("formal exactness pointer lost its control receipt")
        control = ExactnessControlVerificationReceipt.load(
            pointer.control_verification_receipt.absolute_path
        )
        subject = control.control_envelope.subject
        if (
            control.assignment_sha256 != assignment.sha256
            or control.inventory_sha256 != assignment.inventory_sha256
            or control.hardware_envelope_sha256 != assignment.hardware_envelope_sha256
            or subject.registry_sha256 != assignment.registry_sha256
            or subject.lineage_sha256 != assignment.dispatch_lineage_sha256
            or terminal.dispatch_attestation_sha256 != control.control_envelope.sha256
            or terminal.replay_reservation_sha256 != control.reservation_sha256
        ):
            raise ValueError("exactness pointer control differs from assignment")
    before = _load_json(
        Path(pointer.before_gpu_snapshot.absolute_path),
        label="exactness before GPU snapshot",
    )
    after = _load_json(
        Path(pointer.after_gpu_snapshot.absolute_path),
        label="exactness after GPU snapshot",
    )
    before_status, before_processes = _validate_gpu_snapshot(
        before, assignment=assignment
    )
    after_status, after_processes = _validate_gpu_snapshot(after, assignment=assignment)
    if (
        terminal.before_gpu_snapshot_sha256 != pointer.before_gpu_snapshot.raw_sha256
        or terminal.after_gpu_snapshot_sha256 != pointer.after_gpu_snapshot.raw_sha256
        or terminal.log_sha256 != pointer.log.raw_sha256
        or terminal.before_gpu_snapshot_status != before_status
        or terminal.after_gpu_snapshot_status != after_status
        or terminal.before_compute_process_count != before_processes
        or terminal.after_compute_process_count != after_processes
    ):
        raise ValueError("exactness terminal differs from bound raw evidence")
    if pointer.junit_xml is None:
        if terminal.junit_xml_sha256 is not None:
            raise ValueError("exactness terminal claims an absent JUnit file")
    else:
        if terminal.junit_xml_sha256 != pointer.junit_xml.raw_sha256:
            raise ValueError("exactness terminal JUnit digest differs")
        summary = _junit_summary(Path(pointer.junit_xml.absolute_path))
        if summary != (
            terminal.test_names,
            terminal.tests_collected,
            terminal.tests_passed,
            terminal.tests_failed,
            terminal.tests_errored,
            terminal.tests_skipped,
        ):
            raise ValueError("exactness terminal differs from bound JUnit")
    valid_ranks = _valid_rank_terminals(
        pointer.rank_terminals,
        assignment=assignment,
    )
    if terminal.rank_terminal_sha256s != tuple(
        row.sha256 for row in valid_ranks
    ) or terminal.terminal_rank_count != len(valid_ranks):
        raise ValueError("exactness terminal differs from bound rank terminals")
    if terminal.status == "PASSED":
        if tuple(row.global_rank for row in valid_ranks) != (0, 1) or any(
            row.status != "PASSED" for row in valid_ranks
        ):
            raise ValueError("passed exactness terminal lacks both passed ranks")
        native_sha256s = tuple(row.native_terminal_sha256 for row in valid_ranks)
        if None in native_sha256s or len(set(native_sha256s)) != 2:
            raise ValueError("passed exactness terminal reused native rank evidence")
    if pointer.schema_version == 4:
        if pointer.qualification_proof_artifact is None:
            raise AssertionError("qualified exactness pointer lost aggregate proof")
        proof = ExactnessQualificationProofArtifact.load(
            pointer.qualification_proof_artifact.absolute_path
        )
        proof.revalidate(now_ns=time.time_ns())
        raw_pointer = _revalidate_exactness_qualification_payload(proof.payload)
        if any(
            getattr(pointer, field) != getattr(raw_pointer, field)
            for field in (
                "assignment",
                "before_gpu_snapshot",
                "after_gpu_snapshot",
                "log",
                "terminal",
                "rank_terminals",
                "junit_xml",
                "control_verification_receipt",
            )
        ):
            raise ValueError("qualified exactness pointer differs from raw result")


def _junit_summary(path: Path) -> tuple[tuple[str, ...], int, int, int, int, int]:
    tree = ET.parse(path)
    cases = tuple(tree.iter("testcase"))
    names = tuple(sorted(str(case.attrib.get("name")) for case in cases))
    failed = sum(case.find("failure") is not None for case in cases)
    errored = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    passed = len(cases) - failed - errored - skipped
    return names, len(cases), passed, failed, errored, skipped


def publish_release_exactness_qualification_proof(
    raw_result_pointer_path: str | Path,
    *,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
    proof_artifact_path: str | Path,
) -> ExactnessQualificationProofArtifact:
    """Verify local post-run control and publish a durable aggregate proof."""

    payload = derive_exactness_qualification_payload(raw_result_pointer_path)
    if type(control_attestation) is not ControlArtifactAttestation:
        raise TypeError("exactness qualification requires exact rank control")
    subject = control_attestation.subject
    if (
        subject.artifact_type != "rank_aggregate"
        or subject.artifact_sha256 != payload.sha256
        or subject.protocol_sha256 != PREFLIGHT_EXACTNESS_QUALIFICATION_PROTOCOL_SHA256
        or subject.registry_sha256 != payload.registry_sha256
        or subject.lineage_sha256 != payload.lineage_sha256
        or control_attestation.hardware_envelope_sha256
        != payload.hardware_envelope_sha256
    ):
        raise ValueError("exactness qualification control differs from payload")
    output = _absolute_path(
        "exactness qualification proof output", str(proof_artifact_path)
    )
    if output.exists() or Path(f"{output}.sha256").exists():
        raise ValueError("exactness qualification proof output already exists")
    verified = verify_and_reserve_release_control_artifact_attestations(
        (control_attestation,),
        expected_inventory_sha256=payload.inventory_sha256,
        now_ns=now_ns,
        replay_store=replay_store,
    )[0]
    reservation_sha256 = control_challenge_reservation_sha256(
        (verified,), reserved_ns=now_ns
    )
    artifact = ExactnessQualificationProofArtifact(
        schema_version=1,
        kind="formal_exactness_qualification_proof_artifact",
        payload=payload,
        control_attestation=control_attestation,
        replay_reservation=replay_store.bind_reservation(reservation_sha256),
        verified_control_sha256=_sha(asdict(verified)),
    )
    artifact.revalidate(now_ns=now_ns)
    _publish_json(output, artifact.to_dict())
    reopened = ExactnessQualificationProofArtifact.load(output)
    reopened.revalidate(now_ns=now_ns)
    return reopened


def finalize_release_exactness_preflight(
    raw_result_pointer_path: str | Path,
    *,
    proof_artifact_path: str | Path,
    qualified_result_pointer_path: str | Path,
    now_ns: int,
) -> ExactnessPreflightResultPointer:
    """Publish schema 4 only after reopening the post-run rank aggregate proof."""

    raw_path = _absolute_path(
        "raw exactness result pointer", str(raw_result_pointer_path)
    )
    raw_pointer = ExactnessPreflightResultPointer.load(raw_path)
    if raw_pointer.schema_version != 3:
        raise ValueError("exactness finalization requires one raw schema-3 pointer")
    proof_path = _absolute_path(
        "exactness qualification proof", str(proof_artifact_path)
    )
    proof = ExactnessQualificationProofArtifact.load(proof_path)
    proof.revalidate(now_ns=now_ns)
    if (
        proof.payload.raw_result_pointer.absolute_path != str(raw_path)
        or proof.payload.raw_result_pointer_semantic_sha256 != raw_pointer.sha256
    ):
        raise ValueError("exactness qualification proof belongs to another result")
    output = _absolute_path(
        "qualified exactness result pointer",
        str(qualified_result_pointer_path),
    )
    if output.exists() or Path(f"{output}.sha256").exists():
        raise ValueError("qualified exactness result output already exists")
    pointer = ExactnessPreflightResultPointer(
        schema_version=4,
        kind="formal_exactness_preflight_result_pointer",
        assignment=raw_pointer.assignment,
        before_gpu_snapshot=raw_pointer.before_gpu_snapshot,
        after_gpu_snapshot=raw_pointer.after_gpu_snapshot,
        log=raw_pointer.log,
        terminal=raw_pointer.terminal,
        rank_terminals=raw_pointer.rank_terminals,
        junit_xml=raw_pointer.junit_xml,
        control_verification_receipt=raw_pointer.control_verification_receipt,
        qualification_proof_artifact=EvidenceFileBinding.bind(
            proof_path, label="exactness qualification proof"
        ),
    )
    _publish_json(output, pointer.to_dict())
    return ExactnessPreflightResultPointer.load(output)


def _execute_exactness_preflight(
    assignment_path: str | Path,
    *,
    dispatch_attestation: ControlArtifactAttestation | None,
    replay_store: ChallengeReplayStore | None,
    now_ns: int,
    timeout_seconds: float,
    single_operator_authority_sha256: str | None,
) -> ExactnessPreflightResultPointer:
    """Run the exact suite after one of the two closed launch authorities.

    The suite, not this function, writes the two raw rank terminals.  This
    function executes it in a dedicated process group and publishes an exact
    aggregate plus immutable pointer.  The legacy release branch persists its
    signed control and replay receipt.  The trusted-single-operator branch is
    reachable only through the current-source exact-ten adapter and persists a
    diagnostic pointer whose terminal is joined back to that adapter there.
    """

    assignment_source = _absolute_path("exactness assignment", str(assignment_path))
    assignment = ExactnessPreflightAssignment.load(assignment_source)
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("exactness execution time is invalid")
    if type(timeout_seconds) not in {int, float} or timeout_seconds <= 0:
        raise ValueError("exactness execution timeout is invalid")
    trusted_single_operator = single_operator_authority_sha256 is not None
    if trusted_single_operator:
        if dispatch_attestation is not None or replay_store is not None:
            raise ValueError("trusted exactness execution cannot claim sealed control")
        _require_sha256(
            "single-operator exactness authority",
            single_operator_authority_sha256,
        )
        dispatch_identity_sha256 = single_operator_authority_sha256
        reservation_sha256 = _sha(
            {
                "schema_version": 1,
                "kind": "formal_single_operator_exactness_execution_marker",
                "execution_authority_sha256": single_operator_authority_sha256,
                "assignment_sha256": assignment.sha256,
            }
        )
    else:
        if (
            type(dispatch_attestation) is not ControlArtifactAttestation
            or type(replay_store) is not ChallengeReplayStore
        ):
            raise TypeError("exactness runner requires an exact control attestation")
        subject = dispatch_attestation.subject
        if (
            subject.artifact_type != "non_serving_terminal"
            or subject.artifact_sha256 != assignment.sha256
            or subject.protocol_sha256 != PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256
            or subject.registry_sha256 != assignment.registry_sha256
            or subject.lineage_sha256 != assignment.dispatch_lineage_sha256
            or dispatch_attestation.hardware_envelope_sha256
            != assignment.hardware_envelope_sha256
        ):
            raise ValueError("exactness non-serving control differs from assignment")
        dispatch_identity_sha256 = dispatch_attestation.sha256
    evidence = Path(assignment.evidence_directory)
    prefix = f"exactness-{assignment.sha256}"
    paths = {
        name: evidence / f"{prefix}.{suffix}"
        for name, suffix in (
            ("before", "before.json"),
            ("after", "after.json"),
            ("log", "log.txt"),
            ("junit", "junit.xml"),
            ("terminal", "terminal.json"),
            ("control", "control.json"),
            ("pointer", "result.json"),
            ("rank0", "rank-0.json"),
            ("rank1", "rank-1.json"),
            ("fatal", "fatal-terminal.json"),
        )
    }
    if any(path.exists() or Path(f"{path}.sha256").exists() for path in paths.values()):
        raise ValueError("exactness evidence target already exists")
    if not trusted_single_operator:
        assert type(dispatch_attestation) is ControlArtifactAttestation
        assert type(replay_store) is ChallengeReplayStore
        verified = verify_and_reserve_release_control_artifact_attestations(
            (dispatch_attestation,),
            expected_inventory_sha256=assignment.inventory_sha256,
            now_ns=now_ns,
            replay_store=replay_store,
        )
        reservation_sha256 = control_challenge_reservation_sha256(
            verified, reserved_ns=now_ns
        )
        reservation_path = (
            Path(replay_store.root) / f"reservation-{reservation_sha256}.json"
        )
        control_receipt = ExactnessControlVerificationReceipt(
            schema_version=1,
            kind="formal_exactness_control_verification_receipt",
            verified_ns=now_ns,
            assignment_sha256=assignment.sha256,
            inventory_sha256=assignment.inventory_sha256,
            hardware_envelope_sha256=assignment.hardware_envelope_sha256,
            control_envelope=dispatch_attestation,
            verified_control=verified[0],
            reservation_sha256=reservation_sha256,
            reservation_record=EvidenceFileBinding.bind(
                reservation_path, label="exactness control reservation"
            ),
        )
        _publish_json(paths["control"], control_receipt.to_dict())
    started_ns = time.monotonic_ns()
    try:
        before = _snapshot_gpu(assignment)
        _publish_json(paths["before"], before)
        before_status, before_process_count = _validate_gpu_snapshot(
            before, assignment=assignment
        )
        command = (
            assignment.python_executable,
            "-m",
            "pytest",
            "-q",
            *(
                f"{PREFLIGHT_EXACTNESS_TEST_FILE}::{name}"
                for name in PREFLIGHT_EXACTNESS_TEST_NAMES
            ),
            f"--junitxml={paths['junit']}",
        )
        environment = {
            **assignment.loader_environment.child_environment(),
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": ",".join(assignment.gpu_uuids),
            "LIGHTCONE_PREFLIGHT_ASSIGNMENT_SHA256": assignment.sha256,
            "LIGHTCONE_PREFLIGHT_RUNNER_PROTOCOL_SHA256": (
                PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256
            ),
            "LIGHTCONE_PREFLIGHT_RANK0_TERMINAL_PATH": str(paths["rank0"]),
            "LIGHTCONE_PREFLIGHT_RANK1_TERMINAL_PATH": str(paths["rank1"]),
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        log_fd = os.open(paths["log"], flags, 0o600)
        outcome = _ProcessOutcome(255, "ERROR", None, None)
        launched = False
        try:
            if before_status == "AVAILABLE" and before_process_count == 0:
                launched = True
                outcome = _run_logged_process_group(
                    command,
                    cwd=assignment.patched_sglang_checkout,
                    environment=environment,
                    log_fd=log_fd,
                    timeout_seconds=timeout_seconds,
                )
            else:
                os.write(
                    log_fd,
                    b"qualification launch suppressed: GPU precondition failed\n",
                )
                outcome = _ProcessOutcome(
                    255,
                    "CLEAN",
                    (
                        "before_snapshot_unavailable"
                        if before_status == "ERROR"
                        else "before_snapshot_compute_process_present"
                    ),
                    None,
                )
        finally:
            if os.fstat(log_fd).st_size == 0:
                os.write(log_fd, b"qualification produced no log output\n")
            os.fsync(log_fd)
            os.close(log_fd)
        summary: tuple[tuple[str, ...], int, int, int, int, int] = (
            (),
            0,
            0,
            0,
            1,
            0,
        )
        junit_parse_error = False
        if paths["junit"].is_file() and not paths["junit"].is_symlink():
            try:
                summary = _junit_summary(paths["junit"])
            except (ET.ParseError, OSError, ValueError):
                junit_parse_error = True
        elif launched:
            junit_parse_error = True
        rank_bindings = tuple(
            EvidenceFileBinding.bind(path, label=f"exactness raw rank {rank}")
            for rank, path in enumerate((paths["rank0"], paths["rank1"]))
            if path.is_file() and not path.is_symlink()
        )
        valid_ranks = _valid_rank_terminals(
            rank_bindings,
            assignment=assignment,
        )
        runner_error_code = outcome.error_code
        if runner_error_code is None and junit_parse_error:
            runner_error_code = "qualification_junit_unavailable"
        exact_junit = summary == (
            PREFLIGHT_EXACTNESS_TEST_NAMES,
            8,
            8,
            0,
            0,
            0,
        )
        exact_ranks = (
            tuple(row.global_rank for row in valid_ranks) == (0, 1)
            and all(row.status == "PASSED" for row in valid_ranks)
            and len(
                {
                    row.native_terminal_sha256
                    for row in valid_ranks
                    if row.native_terminal_sha256 is not None
                }
            )
            == 2
        )
        if runner_error_code is None and outcome.exit_code == 0 and not exact_ranks:
            runner_error_code = "qualification_rank_terminals_incomplete"
        status: Literal["PASSED", "FAILED", "ERROR"]
        if runner_error_code is not None or outcome.cleanup != "CLEAN":
            status = "ERROR"
        elif outcome.exit_code != 0 or not exact_junit:
            status = "FAILED"
        elif exact_ranks:
            status = "PASSED"
        else:
            status = "ERROR"
        after = _snapshot_gpu(assignment)
        _publish_json(paths["after"], after)
        after_status, after_process_count = _validate_gpu_snapshot(
            after, assignment=assignment
        )
        if after_status != "AVAILABLE" or after_process_count != 0:
            status = "ERROR"
            if runner_error_code is None:
                runner_error_code = (
                    "after_snapshot_unavailable"
                    if after_status == "ERROR"
                    else "after_snapshot_compute_process_present"
                )
        log_binding = EvidenceFileBinding.bind(paths["log"], label="exactness log")
        before_binding = EvidenceFileBinding.bind(
            paths["before"], label="exactness before snapshot"
        )
        after_binding = EvidenceFileBinding.bind(
            paths["after"], label="exactness after snapshot"
        )
        junit_binding = (
            EvidenceFileBinding.bind(paths["junit"], label="exactness JUnit")
            if paths["junit"].is_file() and not paths["junit"].is_symlink()
            else None
        )
        names, collected, passed, failed, errored, skipped = summary
        terminal = ExactnessPreflightTerminal(
            schema_version=3 if trusted_single_operator else 2,
            kind="formal_exactness_preflight_terminal",
            assignment_sha256=assignment.sha256,
            dispatch_attestation_sha256=dispatch_identity_sha256,
            replay_reservation_sha256=reservation_sha256,
            status=status,
            started_ns=started_ns,
            finished_ns=time.monotonic_ns(),
            process_exit_code=outcome.exit_code,
            test_names=names,
            tests_collected=collected,
            tests_passed=passed,
            tests_failed=failed,
            tests_errored=errored,
            tests_skipped=skipped,
            terminal_rank_count=len(valid_ranks),
            rank_terminal_sha256s=tuple(row.sha256 for row in valid_ranks),
            before_gpu_snapshot_sha256=before_binding.raw_sha256,
            after_gpu_snapshot_sha256=after_binding.raw_sha256,
            before_gpu_snapshot_status=before_status,
            after_gpu_snapshot_status=after_status,
            before_compute_process_count=before_process_count,
            after_compute_process_count=after_process_count,
            process_group_cleanup=outcome.cleanup,
            process_group_id=outcome.process_group_id,
            runner_error_code=runner_error_code,
            junit_xml_sha256=(
                None if junit_binding is None else junit_binding.raw_sha256
            ),
            log_sha256=log_binding.raw_sha256,
            authority_mode=(
                "formal_single_operator_v1"
                if trusted_single_operator
                else "dynamic_control"
            ),
        )
        _publish_json(paths["terminal"], terminal.to_dict())
        pointer = ExactnessPreflightResultPointer(
            schema_version=2 if trusted_single_operator else 3,
            kind="formal_exactness_preflight_result_pointer",
            assignment=EvidenceFileBinding.bind(
                assignment_source, label="exactness assignment"
            ),
            before_gpu_snapshot=before_binding,
            after_gpu_snapshot=after_binding,
            log=log_binding,
            terminal=EvidenceFileBinding.bind(
                paths["terminal"], label="exactness terminal"
            ),
            rank_terminals=rank_bindings,
            junit_xml=junit_binding,
            control_verification_receipt=(
                None
                if trusted_single_operator
                else EvidenceFileBinding.bind(
                    paths["control"],
                    label="exactness control verification receipt",
                )
            ),
        )
        _publish_json(paths["pointer"], pointer.to_dict())
        return ExactnessPreflightResultPointer.load(paths["pointer"])
    except Exception:
        fatal = {
            "schema_version": 1,
            "kind": "formal_exactness_preflight_fatal_terminal",
            "assignment_sha256": assignment.sha256,
            "dispatch_attestation_sha256": dispatch_identity_sha256,
            "replay_reservation_sha256": reservation_sha256,
            "status": "ERROR",
            "error_code": "runner_evidence_publication_failed",
            "started_ns": started_ns,
            "finished_ns": time.monotonic_ns(),
        }
        try:
            _publish_json(paths["fatal"], fatal)
        except (OSError, RuntimeError, ValueError) as fatal_error:
            raise RuntimeError(
                "fatal exactness terminal publication failed"
            ) from fatal_error
        raise


def execute_release_exactness_preflight(
    assignment_path: str | Path,
    *,
    dispatch_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
    timeout_seconds: float = 3_600,
) -> ExactnessPreflightResultPointer:
    """Run and aggregate the exact suite after dynamic dispatch authorization."""

    return _execute_exactness_preflight(
        assignment_path,
        dispatch_attestation=dispatch_attestation,
        replay_store=replay_store,
        now_ns=now_ns,
        timeout_seconds=timeout_seconds,
        single_operator_authority_sha256=None,
    )


__all__ = [
    "PREFLIGHT_EXACTNESS_QUALIFICATION_PROTOCOL_SHA256",
    "PREFLIGHT_EXACTNESS_RUNNER_PROTOCOL_SHA256",
    "PREFLIGHT_EXACTNESS_TEST_FILE",
    "PREFLIGHT_EXACTNESS_TEST_NAMES",
    "BurstGptShapeAuthority",
    "BurstGptSourcePin",
    "EvidenceFileBinding",
    "ExactnessControlVerificationReceipt",
    "ExactnessLoaderEnvironment",
    "ExactnessPreflightAssignment",
    "ExactnessPreflightResultPointer",
    "ExactnessPreflightTerminal",
    "ExactnessQualificationPayload",
    "ExactnessQualificationProofArtifact",
    "ExactnessRankTerminal",
    "PreflightInputLocks",
    "derive_burstgpt_shape_authority_from_content_receipt",
    "derive_exactness_qualification_payload",
    "execute_release_exactness_preflight",
    "finalize_release_exactness_preflight",
    "publish_release_exactness_qualification_proof",
]
