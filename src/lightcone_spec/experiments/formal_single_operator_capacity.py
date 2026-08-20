"""Exact disk-capacity authority for the trusted single-operator workflow.

This is separate from the root-signed release capacity protocol.  It is an
unsigned, explicitly empirical authority for the trusted operator and must
never be reported as formal ``MEASURED`` capacity.

Bootstrap is acyclic: the public publisher accepts only an exact v03 content
path spec, an existing run root, and an output path.  It deep-scans the content
closure before doctor exists, binds all content/cache/run filesystem
identities, and derives every byte scalar from source.  Doctor consumes this
authority; the eventual runtime-BOUND content bundle must then reproduce the
same pre-doctor closure and bind the exact PASS doctor report.

Zero retry reserve is exact because this authority disables every automatic
retry.  A failed physical or auxiliary attempt durably stops/blocks before a
new attempt directory can be created.  A future retry protocol must ship a
reachable archive producer and reconciliation path before changing that rule.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedSingleOperatorContentBundle,
    TrustedSingleOperatorContentBundleBinding,
    TrustedSingleOperatorContentPathSpec,
    build_trusted_single_operator_content_bundle,
    load_trusted_single_operator_content_path_spec,
)
from lightcone_spec.experiments.registry import content_sha256
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

if TYPE_CHECKING:
    from lightcone_spec.orchestration.experiment_operator import (
        ExperimentOperatorStore,
        QueuedCommandSpec,
    )


TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES = 16 * 1024**3
TRUSTED_SINGLE_OPERATOR_CAPACITY_SAFETY_MARGIN_BYTES = 15 * 1024**3
TRUSTED_SINGLE_OPERATOR_MAXIMUM_AUTOMATIC_RETRIES = 0

TRUSTED_SINGLE_OPERATOR_RETRY_POLICY_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "trusted_single_operator_retry_policy",
        "automatic_infrastructure_retries": 0,
        "retry_reserve_bytes": 0,
        "failed_physical_attempt": "durable_dispatch_STOP_before_builder",
        "failed_auxiliary_attempt": "durable_DAG_BLOCKED_before_new_directory",
        "future_enablement": (
            "requires_reachable_TRANSFER_LOCAL_SHA_FULL_REHYDRATE_"
            "eviction_receipt_producer_and_restart_reconciliation"
        ),
    }
)

TRUSTED_SINGLE_OPERATOR_CAPACITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "trusted_single_operator_stage_capacity",
        "trust_mode": "trusted_single_operator_no_signature",
        "claim": "trusted_empirical_capacity_not_formal_MEASURED",
        "public_inputs": "v03_content_path_spec_run_root_output_paths_only",
        "bootstrap": "pre_doctor_deep_content_closure_then_doctor_then_BOUND_content",
        "initial_stage": "preflight",
        "current_physical_wave_high_water_bytes": (
            TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES
        ),
        "new_dispatch_running_waves": "sum_each_physical_command_high_water",
        "restart_existing_running": (
            "adopt_by_durable_identity_without_readding_already_written_high_water"
        ),
        "retained_evidence": "already_consumed_in_fresh_statvfs_f_bavail",
        "retry_reserve_bytes": 0,
        "retry_reserve_mode": "AUTOMATIC_RETRY_DISABLED_ZERO_RESERVE",
        "retry_policy_sha256": TRUSTED_SINGLE_OPERATOR_RETRY_POLICY_SHA256,
        "safety_margin_bytes": (TRUSTED_SINGLE_OPERATOR_CAPACITY_SAFETY_MARGIN_BYTES),
        "resident_group": (
            "sum_2_to_32_source_member_high_waters_plus_registered_shared_evidence"
        ),
        "publication": "canonical_atomic_no_replace_then_deep_reopen",
    }
)

_AUTHORITY_KIND = "trusted_single_operator_stage_capacity_authority"
_CLAIM_SCOPE = "trusted_single_operator_empirical_capacity_only"
_RETRY_RESERVE_MODE = "AUTOMATIC_RETRY_DISABLED_ZERO_RESERVE"
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


class TrustedSingleOperatorCapacityBlocked(RuntimeError):
    """A live, source-bound capacity check cannot safely admit one wave."""

    def __init__(self, decision: TrustedSingleOperatorCapacityDecision):
        self.decision = decision
        super().__init__(
            "trusted single-operator capacity is BLOCKED: "
            f"{decision.observed_free_bytes}<{decision.required_free_bytes} "
            f"for {decision.stage}"
        )


class TrustedSingleOperatorAutomaticRetryDisabled(RuntimeError):
    """The zero-reserve authority forbids a new automatic attempt."""


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _nonnegative_int(label: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_int(label: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _absolute_directory(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} is missing") from error
    if resolved != path or path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a resolved non-symlink directory")
    return path


def _normalized_future_path(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute() or path.resolve(strict=False) != path or not path.name:
        raise ValueError(f"{label} must be a normalized absolute file path")
    parent = _absolute_directory(path.parent, label=f"{label} parent")
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ValueError(f"{label} existing target is not a regular file")
    if parent != path.parent:
        raise ValueError(f"{label} parent identity differs")
    return path


def _filesystem_payload(path: Path) -> dict[str, int]:
    metadata = path.stat(follow_symlinks=False)
    filesystem = os.statvfs(path)
    filesystem_id = getattr(filesystem, "f_fsid", None)
    if type(filesystem_id) is not int:
        raise ValueError("filesystem does not expose an integral identity")
    values = {
        "device": int(metadata.st_dev),
        "filesystem_id": filesystem_id,
        "block_size": int(filesystem.f_bsize),
        "fragment_size": int(filesystem.f_frsize),
        "total_blocks": int(filesystem.f_blocks),
        "mount_flags": int(filesystem.f_flag),
        "maximum_name_length": int(filesystem.f_namemax),
    }
    for label, value in values.items():
        _nonnegative_int(f"filesystem {label}", value)
    if values["fragment_size"] == 0 or values["total_blocks"] == 0:
        raise ValueError("filesystem capacity identity is incomplete")
    return values


def _filesystem_sha256(path: Path) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_single_operator_filesystem_identity",
            **_filesystem_payload(path),
        }
    )


def _free_bytes(path: Path) -> int:
    filesystem = os.statvfs(path)
    available = int(filesystem.f_bavail) * int(filesystem.f_frsize)
    return _nonnegative_int("filesystem available bytes", available)


@dataclass(frozen=True, order=True)
class TrustedSingleOperatorCapacityRoot:
    """One stable directory and the mounted filesystem below it."""

    role: Literal["content", "cache", "run"]
    absolute_path: str
    device: int
    inode: int
    filesystem_sha256: str

    def __post_init__(self) -> None:
        if self.role not in {"content", "cache", "run"}:
            raise ValueError("trusted capacity root role is unsupported")
        path = _absolute_directory(self.absolute_path, label="trusted capacity root")
        metadata = path.stat(follow_symlinks=False)
        _nonnegative_int("trusted capacity root device", self.device)
        _positive_int("trusted capacity root inode", self.inode)
        _require_sha256("trusted capacity root filesystem", self.filesystem_sha256)
        if (metadata.st_dev, metadata.st_ino) != (self.device, self.inode):
            raise ValueError("trusted capacity root directory identity changed")
        if _filesystem_sha256(path) != self.filesystem_sha256:
            raise ValueError("trusted capacity root filesystem identity changed")

    @classmethod
    def bind(
        cls,
        path_value: str | Path,
        *,
        role: Literal["content", "cache", "run"],
    ) -> Self:
        path = _absolute_directory(path_value, label=f"trusted {role} root")
        metadata = path.stat(follow_symlinks=False)
        return cls(
            role=role,
            absolute_path=str(path),
            device=int(metadata.st_dev),
            inode=int(metadata.st_ino),
            filesystem_sha256=_filesystem_sha256(path),
        )

    def revalidate(self) -> Self:
        rebound = type(self).bind(self.absolute_path, role=self.role)
        if rebound != self:
            raise RuntimeError("trusted capacity root changed")
        return rebound

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("trusted capacity root fields differ")
        return cls(**value)  # type: ignore[arg-type]


def _deep_content_from_path_spec(
    path: str | Path,
) -> tuple[
    CanonicalJsonProofBinding,
    TrustedSingleOperatorContentPathSpec,
    TrustedSingleOperatorContentBundle,
]:
    from lightcone_spec.experiments.formal_single_operator_model_registry import (
        require_formal_v03_content_path_spec,
    )

    binding = CanonicalJsonProofBinding.bind(path)
    spec = load_trusted_single_operator_content_path_spec(path)
    require_formal_v03_content_path_spec(spec)
    if binding.reopen() != spec.to_dict():
        raise ValueError("trusted capacity content path-spec binding differs")
    pending = build_trusted_single_operator_content_bundle(
        repository_root=spec.repository_root,
        model_specs=spec.model_specs,
        livecodebench_raw_path=spec.livecodebench_raw_path,
        math500_raw_path=spec.math500_raw_path,
        burstgpt_asset_paths={
            row.name: row.absolute_path for row in spec.burstgpt_asset_paths
        },
        e0_task_native_specs=spec.e0_task_native_specs,
    )
    if (
        pending.runtime_binding_status != "PENDING_REMOTE_BINDING"
        or pending.runtime_observations is not None
    ):
        raise RuntimeError("trusted capacity pre-doctor content closure is not pending")
    return binding, spec, pending


def _content_root_paths(
    spec_binding: CanonicalJsonProofBinding,
    pending: TrustedSingleOperatorContentBundle,
) -> tuple[Path, ...]:
    paths = {
        Path(spec_binding.absolute_path).parent,
        Path(pending.source_snapshot.repository_root),
        *(Path(member.local_snapshot_path) for member in pending.model_members),
        *(Path(row.raw_source_path).parent for row in pending.locked_workloads),
        *(Path(row.absolute_path).parent for row in pending.burstgpt_release.assets),
        *(
            Path(row.source.absolute_path).parent
            for row in pending.e0_task_native_descriptors
        ),
    }
    return tuple(sorted(paths, key=str))


def _cache_root_paths(
    pending: TrustedSingleOperatorContentBundle,
) -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                Path(member.content_cache_root)
                for member in pending.model_members
                if member.content_cache_root is not None
            },
            key=str,
        )
    )


def _bind_roots(
    spec_binding: CanonicalJsonProofBinding,
    pending: TrustedSingleOperatorContentBundle,
    *,
    run_root: str | Path,
) -> tuple[
    tuple[TrustedSingleOperatorCapacityRoot, ...],
    tuple[TrustedSingleOperatorCapacityRoot, ...],
    TrustedSingleOperatorCapacityRoot,
]:
    content = tuple(
        TrustedSingleOperatorCapacityRoot.bind(path, role="content")
        for path in _content_root_paths(spec_binding, pending)
    )
    cache = tuple(
        TrustedSingleOperatorCapacityRoot.bind(path, role="cache")
        for path in _cache_root_paths(pending)
    )
    run = TrustedSingleOperatorCapacityRoot.bind(run_root, role="run")
    if not content:
        raise ValueError("trusted capacity authority has no content roots")
    return content, cache, run


@dataclass(frozen=True)
class TrustedSingleOperatorStageCapacityAuthority:
    """Unsigned path-bound authority for one exact initial physical wave."""

    schema_version: Literal[3]
    kind: Literal["trusted_single_operator_stage_capacity_authority"]
    protocol_sha256: str
    trust_mode: Literal["trusted_single_operator_no_signature"]
    signature: None
    formal_measured_authorization: Literal[False]
    claim_scope: Literal["trusted_single_operator_empirical_capacity_only"]
    initial_stage: Literal["preflight"]
    content_path_spec: CanonicalJsonProofBinding
    pre_doctor_content_closure_sha256: str
    content_roots: tuple[TrustedSingleOperatorCapacityRoot, ...]
    cache_roots: tuple[TrustedSingleOperatorCapacityRoot, ...]
    run_root: TrustedSingleOperatorCapacityRoot
    captured_free_bytes: int
    current_wave_high_water_bytes: int
    retry_reserve_mode: Literal["AUTOMATIC_RETRY_DISABLED_ZERO_RESERVE"]
    retry_policy_sha256: str
    maximum_automatic_infrastructure_retries: int
    retry_reserve_bytes: int
    safety_margin_bytes: int
    required_free_bytes: int
    status: Literal["AVAILABLE", "BLOCKED"]
    reason_code: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 3
            or self.kind != _AUTHORITY_KIND
            or self.protocol_sha256 != TRUSTED_SINGLE_OPERATOR_CAPACITY_PROTOCOL_SHA256
            or self.trust_mode != "trusted_single_operator_no_signature"
            or self.signature is not None
            or self.formal_measured_authorization is not False
            or self.claim_scope != _CLAIM_SCOPE
            or self.initial_stage != "preflight"
            or self.retry_reserve_mode != _RETRY_RESERVE_MODE
            or self.retry_policy_sha256 != TRUSTED_SINGLE_OPERATOR_RETRY_POLICY_SHA256
            or self.maximum_automatic_infrastructure_retries
            != TRUSTED_SINGLE_OPERATOR_MAXIMUM_AUTOMATIC_RETRIES
        ):
            raise ValueError("trusted single-operator capacity schema differs")
        if type(self.content_path_spec) is not CanonicalJsonProofBinding:
            raise TypeError("trusted capacity content path spec is not path-bound")
        _require_sha256(
            "trusted capacity pre-doctor content closure",
            self.pre_doctor_content_closure_sha256,
        )
        if (
            type(self.content_roots) is not tuple
            or not self.content_roots
            or any(row.role != "content" for row in self.content_roots)
            or self.content_roots
            != tuple(sorted(set(self.content_roots), key=lambda row: row.absolute_path))
            or type(self.cache_roots) is not tuple
            or any(row.role != "cache" for row in self.cache_roots)
            or self.cache_roots
            != tuple(sorted(set(self.cache_roots), key=lambda row: row.absolute_path))
            or type(self.run_root) is not TrustedSingleOperatorCapacityRoot
            or self.run_root.role != "run"
        ):
            raise ValueError("trusted capacity root coverage is not canonical")
        for label, amount in (
            ("captured free", self.captured_free_bytes),
            ("current wave high-water", self.current_wave_high_water_bytes),
            ("retry reserve", self.retry_reserve_bytes),
            ("safety margin", self.safety_margin_bytes),
            ("required free", self.required_free_bytes),
        ):
            _nonnegative_int(f"trusted capacity {label} bytes", amount)
        if (
            self.current_wave_high_water_bytes
            != TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES
            or self.retry_reserve_bytes != 0
            or self.safety_margin_bytes
            != TRUSTED_SINGLE_OPERATOR_CAPACITY_SAFETY_MARGIN_BYTES
            or self.required_free_bytes
            != self.current_wave_high_water_bytes
            + self.retry_reserve_bytes
            + self.safety_margin_bytes
        ):
            raise ValueError("trusted preflight capacity arithmetic differs")
        expected_status = (
            "AVAILABLE"
            if self.captured_free_bytes >= self.required_free_bytes
            else "BLOCKED"
        )
        expected_reason = (
            "trusted_single_operator_preflight_capacity_available"
            if expected_status == "AVAILABLE"
            else "trusted_single_operator_preflight_capacity_insufficient"
        )
        if self.status != expected_status or self.reason_code != expected_reason:
            raise ValueError("trusted preflight capacity decision differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "trust_mode": self.trust_mode,
            "signature": self.signature,
            "formal_measured_authorization": self.formal_measured_authorization,
            "claim_scope": self.claim_scope,
            "initial_stage": self.initial_stage,
            "content_path_spec": self.content_path_spec.to_dict(),
            "pre_doctor_content_closure_sha256": (
                self.pre_doctor_content_closure_sha256
            ),
            "content_roots": [row.to_dict() for row in self.content_roots],
            "cache_roots": [row.to_dict() for row in self.cache_roots],
            "run_root": self.run_root.to_dict(),
            "captured_free_bytes": self.captured_free_bytes,
            "current_wave_high_water_bytes": self.current_wave_high_water_bytes,
            "retry_reserve_mode": self.retry_reserve_mode,
            "retry_policy_sha256": self.retry_policy_sha256,
            "maximum_automatic_infrastructure_retries": (
                self.maximum_automatic_infrastructure_retries
            ),
            "retry_reserve_bytes": self.retry_reserve_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "required_free_bytes": self.required_free_bytes,
            "status": self.status,
            "reason_code": self.reason_code,
        }

    def to_dict(self) -> dict[str, object]:
        return {"authority_sha256": self.sha256, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = set(cls.__dataclass_fields__) | {"authority_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("trusted capacity authority fields differ")
        row = dict(value)
        declared = _require_sha256(
            "trusted capacity authority", row.pop("authority_sha256")
        )
        content_path_spec = CanonicalJsonProofBinding.from_dict(
            row.pop("content_path_spec")
        )
        raw_content = row.pop("content_roots")
        raw_cache = row.pop("cache_roots")
        raw_run = row.pop("run_root")
        if type(raw_content) is not list or type(raw_cache) is not list:
            raise TypeError("trusted capacity roots must be arrays")
        authority = cls(
            **row,
            content_path_spec=content_path_spec,
            content_roots=tuple(
                TrustedSingleOperatorCapacityRoot.from_dict(item)
                for item in raw_content
            ),
            cache_roots=tuple(
                TrustedSingleOperatorCapacityRoot.from_dict(item) for item in raw_cache
            ),
            run_root=TrustedSingleOperatorCapacityRoot.from_dict(raw_run),
        )  # type: ignore[arg-type]
        if authority.sha256 != declared:
            raise ValueError("trusted capacity authority digest differs")
        return authority


@dataclass(frozen=True)
class TrustedSingleOperatorCapacityDecision:
    """One fresh decision derived from an authority and physical commands."""

    stage: str
    wave_kind: Literal["ordinary", "resident_group", "operator_wave", "restart"]
    physical_member_count: int
    observed_free_bytes: int
    current_wave_high_water_bytes: int
    running_wave_high_water_bytes: int
    retry_reserve_bytes: int
    safety_margin_bytes: int
    required_free_bytes: int
    status: Literal["AVAILABLE", "BLOCKED"]
    reason_code: str
    authority_sha256: str

    def __post_init__(self) -> None:
        _registered_stage(self.stage)
        if self.wave_kind not in {
            "ordinary",
            "resident_group",
            "operator_wave",
            "restart",
        }:
            raise ValueError("trusted capacity wave kind is unsupported")
        _nonnegative_int(
            "trusted capacity physical member count", self.physical_member_count
        )
        if self.wave_kind != "restart" and self.physical_member_count < 1:
            raise ValueError("trusted capacity live wave has no physical members")
        for label, amount in (
            ("observed free", self.observed_free_bytes),
            ("wave high-water", self.current_wave_high_water_bytes),
            ("running high-water", self.running_wave_high_water_bytes),
            ("retry reserve", self.retry_reserve_bytes),
            ("safety margin", self.safety_margin_bytes),
            ("required free", self.required_free_bytes),
        ):
            _nonnegative_int(f"trusted capacity decision {label} bytes", amount)
        _require_sha256("trusted capacity decision authority", self.authority_sha256)
        if (
            self.safety_margin_bytes
            != TRUSTED_SINGLE_OPERATOR_CAPACITY_SAFETY_MARGIN_BYTES
            or self.required_free_bytes
            != self.current_wave_high_water_bytes
            + self.running_wave_high_water_bytes
            + self.retry_reserve_bytes
            + self.safety_margin_bytes
        ):
            raise ValueError("trusted capacity decision arithmetic differs")
        expected = (
            "AVAILABLE"
            if self.observed_free_bytes >= self.required_free_bytes
            else "BLOCKED"
        )
        expected_reason = (
            "trusted_single_operator_wave_capacity_available"
            if expected == "AVAILABLE"
            else "trusted_single_operator_wave_capacity_insufficient"
        )
        if self.status != expected or self.reason_code != expected_reason:
            raise ValueError("trusted capacity live status differs")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _registered_stage(stage: object) -> str:
    from lightcone_spec.experiments.formal_single_operator_stages import (
        FORMAL_SINGLE_OPERATOR_NODE_ORDER,
    )

    if type(stage) is not str or stage not in FORMAL_SINGLE_OPERATOR_NODE_ORDER:
        raise ValueError("trusted capacity stage is not a registered DAG node")
    return stage


def _authority_from_paths(
    *,
    content_path_spec_path: str | Path,
    run_root_path: str | Path,
) -> TrustedSingleOperatorStageCapacityAuthority:
    binding, _spec, pending = _deep_content_from_path_spec(content_path_spec_path)
    content_roots, cache_roots, run_root = _bind_roots(
        binding,
        pending,
        run_root=run_root_path,
    )
    observed = _free_bytes(Path(run_root.absolute_path))
    high_water = TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES
    retry = 0
    safety = TRUSTED_SINGLE_OPERATOR_CAPACITY_SAFETY_MARGIN_BYTES
    required = high_water + retry + safety
    status: Literal["AVAILABLE", "BLOCKED"] = (
        "AVAILABLE" if observed >= required else "BLOCKED"
    )
    return TrustedSingleOperatorStageCapacityAuthority(
        schema_version=3,
        kind=_AUTHORITY_KIND,
        protocol_sha256=TRUSTED_SINGLE_OPERATOR_CAPACITY_PROTOCOL_SHA256,
        trust_mode="trusted_single_operator_no_signature",
        signature=None,
        formal_measured_authorization=False,
        claim_scope=_CLAIM_SCOPE,
        initial_stage="preflight",
        content_path_spec=binding,
        pre_doctor_content_closure_sha256=pending.semantic_sha256,
        content_roots=content_roots,
        cache_roots=cache_roots,
        run_root=run_root,
        captured_free_bytes=observed,
        current_wave_high_water_bytes=high_water,
        retry_reserve_mode=_RETRY_RESERVE_MODE,
        retry_policy_sha256=TRUSTED_SINGLE_OPERATOR_RETRY_POLICY_SHA256,
        maximum_automatic_infrastructure_retries=(
            TRUSTED_SINGLE_OPERATOR_MAXIMUM_AUTOMATIC_RETRIES
        ),
        retry_reserve_bytes=retry,
        safety_margin_bytes=safety,
        required_free_bytes=required,
        status=status,
        reason_code=(
            "trusted_single_operator_preflight_capacity_available"
            if status == "AVAILABLE"
            else "trusted_single_operator_preflight_capacity_insufficient"
        ),
    )


def publish_trusted_single_operator_stage_capacity_authority(
    *,
    content_path_spec_path: str | Path,
    run_root_path: str | Path,
    output_path: str | Path,
) -> TrustedSingleOperatorStageCapacityAuthority:
    """Publish the path-only, fixed-policy preflight capacity authority."""

    authority = _authority_from_paths(
        content_path_spec_path=content_path_spec_path,
        run_root_path=run_root_path,
    )
    output = _normalized_future_path(output_path, label="trusted capacity output")
    _binding, spec, _pending = _deep_content_from_path_spec(
        authority.content_path_spec.absolute_path
    )
    repository = Path(spec.repository_root)
    if output == repository or output.is_relative_to(repository):
        raise ValueError("trusted capacity authority must stay outside source Git")
    publish_canonical_json_no_replace(output, authority.to_dict())
    proof, loaded = load_trusted_single_operator_stage_capacity_authority(output)
    if loaded != authority or proof.semantic_sha256 != content_sha256(
        authority.to_dict()
    ):
        raise RuntimeError("published trusted capacity authority changed")
    return authority


def load_trusted_single_operator_stage_capacity_authority(
    path: str | Path,
) -> tuple[CanonicalJsonProofBinding, TrustedSingleOperatorStageCapacityAuthority]:
    binding = CanonicalJsonProofBinding.bind(path)
    authority = TrustedSingleOperatorStageCapacityAuthority.from_dict(binding.reopen())
    if binding.semantic_sha256 != content_sha256(authority.to_dict()):
        raise ValueError("trusted capacity authority file identity differs")
    return binding, authority


def _live_decision(
    authority: TrustedSingleOperatorStageCapacityAuthority,
    *,
    stage: str,
    wave_kind: Literal["ordinary", "resident_group", "operator_wave", "restart"],
    physical_member_count: int,
    current_wave_high_water_bytes: int,
    running_wave_high_water_bytes: int,
    retry_reserve_bytes: int = 0,
) -> TrustedSingleOperatorCapacityDecision:
    _registered_stage(stage)
    observed = _free_bytes(Path(authority.run_root.absolute_path))
    required = (
        current_wave_high_water_bytes
        + running_wave_high_water_bytes
        + retry_reserve_bytes
        + TRUSTED_SINGLE_OPERATOR_CAPACITY_SAFETY_MARGIN_BYTES
    )
    status: Literal["AVAILABLE", "BLOCKED"] = (
        "AVAILABLE" if observed >= required else "BLOCKED"
    )
    return TrustedSingleOperatorCapacityDecision(
        stage=stage,
        wave_kind=wave_kind,
        physical_member_count=physical_member_count,
        observed_free_bytes=observed,
        current_wave_high_water_bytes=current_wave_high_water_bytes,
        running_wave_high_water_bytes=running_wave_high_water_bytes,
        retry_reserve_bytes=retry_reserve_bytes,
        safety_margin_bytes=TRUSTED_SINGLE_OPERATOR_CAPACITY_SAFETY_MARGIN_BYTES,
        required_free_bytes=required,
        status=status,
        reason_code=(
            "trusted_single_operator_wave_capacity_available"
            if status == "AVAILABLE"
            else "trusted_single_operator_wave_capacity_insufficient"
        ),
        authority_sha256=authority.sha256,
    )


def revalidate_trusted_single_operator_stage_capacity_authority(
    path: str | Path,
    *,
    expected_content_path_spec_path: str | Path | None = None,
    expected_run_root_path: str | Path | None = None,
    expected_output_path: str | Path | None = None,
    require_available: bool = True,
) -> tuple[
    CanonicalJsonProofBinding,
    TrustedSingleOperatorStageCapacityAuthority,
    TrustedSingleOperatorCapacityDecision,
]:
    """Deep-reopen content/root identity and re-probe live preflight capacity."""

    binding, authority = load_trusted_single_operator_stage_capacity_authority(path)
    if expected_content_path_spec_path is not None and (
        Path(authority.content_path_spec.absolute_path)
        != Path(expected_content_path_spec_path)
    ):
        raise ValueError("trusted capacity content path spec differs")
    if expected_run_root_path is not None and (
        Path(authority.run_root.absolute_path) != Path(expected_run_root_path)
    ):
        raise ValueError("trusted capacity run root differs")
    if expected_output_path is not None:
        output = Path(expected_output_path)
        if not output.is_absolute() or output.resolve(strict=False) != output:
            raise ValueError("trusted capacity output path must be normalized absolute")
        run_root = Path(authority.run_root.absolute_path)
        if output == run_root or not output.is_relative_to(run_root):
            raise ValueError("trusted capacity output leaves its run root")
    rebound = _authority_from_paths(
        content_path_spec_path=authority.content_path_spec.absolute_path,
        run_root_path=authority.run_root.absolute_path,
    )
    live_fields = {"captured_free_bytes", "status", "reason_code"}
    stable_fields = set(authority.__dataclass_fields__) - live_fields
    if any(
        getattr(rebound, name) != getattr(authority, name) for name in stable_fields
    ):
        raise RuntimeError("trusted capacity source or filesystem identity changed")
    decision = _live_decision(
        authority,
        stage="preflight",
        wave_kind="ordinary",
        physical_member_count=1,
        current_wave_high_water_bytes=authority.current_wave_high_water_bytes,
        running_wave_high_water_bytes=0,
    )
    if require_available and (
        authority.status != "AVAILABLE" or decision.status != "AVAILABLE"
    ):
        raise TrustedSingleOperatorCapacityBlocked(decision)
    return binding, authority, decision


def _running_high_water_bytes(
    authority: TrustedSingleOperatorStageCapacityAuthority,
    commands: tuple[object, ...],
) -> int:
    from lightcone_spec.orchestration.experiment_operator import QueuedCommandSpec

    if type(commands) is not tuple or any(
        type(row) is not QueuedCommandSpec for row in commands
    ):
        raise TypeError("trusted capacity running commands are not exact")
    identities = tuple((row.cell_id, row.attempt) for row in commands)
    if len(set(identities)) != len(identities):
        raise ValueError("trusted capacity running physical commands are duplicated")
    run_root = Path(authority.run_root.absolute_path)
    total = 0
    for command in commands:
        if (
            command.predicted_high_water_bytes
            < TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES
            or Path(command.monitored_path) != run_root
        ):
            raise ValueError(
                "trusted capacity running command bound is not source-owned"
            )
        total += command.predicted_high_water_bytes
    return total


def _running_auxiliary_high_water_bytes(
    authority: TrustedSingleOperatorStageCapacityAuthority,
    store: object | None,
) -> tuple[int, int]:
    if store is None:
        return 0, 0
    from lightcone_spec.orchestration.experiment_operator import (
        ExperimentOperatorStore,
    )

    if type(store) is not ExperimentOperatorStore:
        raise TypeError("trusted capacity auxiliary state requires the exact store")
    rows = tuple(
        row
        for row in store.controller_auxiliary_groups()
        if row.get("status") == "RUNNING"
    )
    identities = tuple((row.get("group_id"), row.get("attempt")) for row in rows)
    if len(identities) != len(set(identities)):
        raise ValueError("trusted capacity running auxiliary waves are duplicated")
    run_root = Path(authority.run_root.absolute_path)
    for row in rows:
        output = Path(str(row.get("output_directory") or ""))
        _registered_stage(row.get("node"))
        if (
            not output.is_absolute()
            or output.resolve(strict=False) != output
            or output == run_root
            or not output.is_relative_to(run_root)
        ):
            raise ValueError("trusted capacity running auxiliary output is foreign")
    return len(rows) * TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES, len(rows)


def require_trusted_single_operator_ordinary_capacity(
    authority_path: str | Path,
    *,
    stage: str,
    running_commands: tuple[object, ...] = (),
    store: object | None = None,
) -> TrustedSingleOperatorCapacityDecision:
    """Require one source-owned 16 GiB command before materializing it."""

    _binding, authority, _initial = (
        revalidate_trusted_single_operator_stage_capacity_authority(authority_path)
    )
    auxiliary_running, _auxiliary_count = _running_auxiliary_high_water_bytes(
        authority, store
    )
    running = _running_high_water_bytes(authority, running_commands) + auxiliary_running
    decision = _live_decision(
        authority,
        stage=stage,
        wave_kind="ordinary",
        physical_member_count=1,
        current_wave_high_water_bytes=TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES,
        running_wave_high_water_bytes=running,
    )
    if decision.status != "AVAILABLE":
        raise TrustedSingleOperatorCapacityBlocked(decision)
    return decision


def require_trusted_single_operator_restart_capacity(
    authority_path: str | Path,
    *,
    stage: str,
    running_commands: tuple[object, ...],
    store: object | None = None,
) -> TrustedSingleOperatorCapacityDecision:
    """Re-open capacity without orphaning or double-counting a running wave.

    Fresh ``f_bavail`` already reflects bytes written by retained RUNNING
    processes.  Restart therefore binds their exact durable count but does not
    add their full lifetime high-water again.  When nothing is running, this
    decision evaluates the next ordinary wave; callers may durably STOP new
    dispatch while still completing construction and reconciliation.
    """

    _binding, authority, _initial = (
        revalidate_trusted_single_operator_stage_capacity_authority(
            authority_path,
            require_available=False,
        )
    )
    auxiliary_running, auxiliary_count = _running_auxiliary_high_water_bytes(
        authority, store
    )
    owned_running = (
        _running_high_water_bytes(authority, running_commands) + auxiliary_running
    )
    current = 0 if owned_running else TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES
    decision = _live_decision(
        authority,
        stage=stage,
        wave_kind="restart",
        physical_member_count=len(running_commands) + auxiliary_count,
        current_wave_high_water_bytes=current,
        running_wave_high_water_bytes=0,
    )
    return decision


def trusted_single_operator_resident_group_high_water_bytes(
    commands: tuple[object, ...],
) -> int:
    """Derive one 2--32 member bound without accepting a byte scalar."""

    from lightcone_spec.orchestration.experiment_operator import QueuedCommandSpec
    from lightcone_spec.orchestration.formal_serving_session_group_production import (
        formal_serving_session_group_shared_evidence_bound_bytes,
    )

    if (
        type(commands) is not tuple
        or not 2 <= len(commands) <= 32
        or any(type(row) is not QueuedCommandSpec for row in commands)
        or any(
            row.predicted_high_water_bytes
            != TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES
            for row in commands
        )
        or len({(row.cell_id, row.attempt) for row in commands}) != len(commands)
    ):
        raise ValueError("trusted resident capacity commands are not exact members")
    return sum(row.predicted_high_water_bytes for row in commands) + (
        formal_serving_session_group_shared_evidence_bound_bytes(len(commands))
    )


def require_trusted_single_operator_resident_group_capacity(
    authority_path: str | Path,
    *,
    stage: str,
    commands: tuple[object, ...],
    running_commands: tuple[object, ...] = (),
    store: object | None = None,
) -> TrustedSingleOperatorCapacityDecision:
    """Require the complete resident sum before creating its group directory."""

    _binding, authority, _initial = (
        revalidate_trusted_single_operator_stage_capacity_authority(authority_path)
    )
    high_water = trusted_single_operator_resident_group_high_water_bytes(commands)
    auxiliary_running, _auxiliary_count = _running_auxiliary_high_water_bytes(
        authority, store
    )
    running = _running_high_water_bytes(authority, running_commands) + auxiliary_running
    decision = _live_decision(
        authority,
        stage=stage,
        wave_kind="resident_group",
        physical_member_count=len(commands),
        current_wave_high_water_bytes=high_water,
        running_wave_high_water_bytes=running,
    )
    if decision.status != "AVAILABLE":
        raise TrustedSingleOperatorCapacityBlocked(decision)
    return decision


def require_trusted_single_operator_operator_wave_capacity(
    authority_path: str | Path,
    *,
    stage: str,
    store: object,
) -> TrustedSingleOperatorCapacityDecision:
    """Gate one source-owned auxiliary wave without caller byte scalars."""

    from lightcone_spec.orchestration.experiment_operator import (
        ExperimentOperatorStore,
    )

    if type(store) is not ExperimentOperatorStore:
        raise TypeError("trusted operator-wave capacity requires the exact store")
    _binding, authority, _initial = (
        revalidate_trusted_single_operator_stage_capacity_authority(authority_path)
    )
    physical = _running_high_water_bytes(
        authority, store.physical_commands(status="RUNNING")
    )
    auxiliary, _auxiliary_count = _running_auxiliary_high_water_bytes(authority, store)
    decision = _live_decision(
        authority,
        stage=stage,
        wave_kind="operator_wave",
        physical_member_count=1,
        current_wave_high_water_bytes=TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES,
        running_wave_high_water_bytes=physical + auxiliary,
    )
    if decision.status != "AVAILABLE":
        raise TrustedSingleOperatorCapacityBlocked(decision)
    return decision


def trusted_single_operator_capacity_free_bytes(
    authority_path: str | Path,
    monitored_path: str | Path,
) -> int:
    """Scheduler probe that rejects root or mount replacement before free bytes."""

    _binding, authority, _initial = (
        revalidate_trusted_single_operator_stage_capacity_authority(authority_path)
    )
    monitored = Path(monitored_path)
    if not monitored.is_absolute() or monitored.resolve(strict=False) != monitored:
        raise ValueError("trusted capacity monitored path is not normalized absolute")
    run_root = Path(authority.run_root.absolute_path)
    if monitored != run_root and not monitored.is_relative_to(run_root):
        raise ValueError("trusted capacity monitored path leaves run root")
    probe = monitored
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if _filesystem_sha256(probe) != authority.run_root.filesystem_sha256:
        raise RuntimeError("trusted capacity monitored filesystem changed")
    return _free_bytes(probe)


def require_trusted_single_operator_bound_content_closure(
    authority: TrustedSingleOperatorStageCapacityAuthority,
    *,
    bundle: TrustedSingleOperatorContentBundle,
    doctor_report_path: str | Path,
) -> None:
    """Match one in-memory BOUND bundle to the pre-doctor capacity closure."""

    if type(authority) is not TrustedSingleOperatorStageCapacityAuthority:
        raise TypeError("trusted content closure requires an exact capacity authority")
    if type(bundle) is not TrustedSingleOperatorContentBundle:
        raise TypeError("trusted content closure requires an exact content bundle")
    runtime = bundle.runtime_observations
    if (
        bundle.runtime_binding_status != "BOUND"
        or runtime is None
        or Path(runtime.doctor.absolute_path) != Path(doctor_report_path)
    ):
        raise ValueError("trusted capacity content lacks the exact BOUND doctor")
    _path_binding, spec, pending = _deep_content_from_path_spec(
        authority.content_path_spec.absolute_path
    )
    if Path(spec.doctor_path) != Path(doctor_report_path) or Path(
        spec.inventory_path
    ) != Path(runtime.inventory.absolute_path):
        raise ValueError("trusted capacity runtime observation paths differ")
    projected = replace(
        bundle,
        runtime_binding_status="PENDING_REMOTE_BINDING",
        runtime_observations=None,
    )
    if (
        pending.semantic_sha256 != authority.pre_doctor_content_closure_sha256
        or projected != pending
        or projected.semantic_sha256 != authority.pre_doctor_content_closure_sha256
    ):
        raise ValueError("BOUND content differs from pre-doctor capacity closure")


def _require_bound_content_source_closure(
    authority: TrustedSingleOperatorStageCapacityAuthority,
    *,
    content_source_path: str | Path,
    doctor_report_path: str | Path,
) -> None:
    content_binding = TrustedSingleOperatorContentBundleBinding.bind(
        content_source_path
    )
    require_trusted_single_operator_bound_content_closure(
        authority,
        bundle=content_binding.reopen(),
        doctor_report_path=doctor_report_path,
    )


def trusted_single_operator_capacity_authority_from_doctor(
    doctor_report_path: str | Path,
    *,
    expected_bound_content_source_path: str | Path | None = None,
    expected_run_root_path: str | Path | None = None,
    expected_output_path: str | Path | None = None,
    require_available: bool = True,
) -> tuple[
    CanonicalJsonProofBinding,
    TrustedSingleOperatorStageCapacityAuthority,
    TrustedSingleOperatorCapacityDecision,
]:
    """Deep-reopen the tagged authority carried by one complete PASS doctor."""

    if type(require_available) is not bool:
        raise TypeError("trusted doctor capacity availability policy must be boolean")

    doctor_binding = CanonicalJsonProofBinding.bind(doctor_report_path)
    doctor = doctor_binding.reopen()
    if type(doctor) is not dict:
        raise TypeError("trusted capacity doctor report must be an object")
    readiness = doctor.get("readiness")
    checks = doctor.get("checks")
    report = doctor.get("stage_capacity")
    if (
        doctor.get("schema_version") != 2
        or doctor.get("status") != "PASS"
        or type(readiness) is not dict
        or readiness.get("status") != "PASS"
        or readiness.get("fail_count") != 0
        or readiness.get("unknown_count") != 0
        or type(checks) is not dict
        or not checks
        or any(
            type(row) is not dict or row.get("status") != "PASS"
            for row in checks.values()
        )
        or type(report) is not dict
        or report.get("mode") != "TRUSTED_SINGLE_OPERATOR_NO_SIGNATURE"
        or report.get("status") != "AVAILABLE"
        or report.get("formal_measured_authorization") is not False
        or report.get("retry_reserve_mode") != _RETRY_RESERVE_MODE
        or report.get("retry_policy_sha256")
        != TRUSTED_SINGLE_OPERATOR_RETRY_POLICY_SHA256
        or type(report.get("authority")) is not dict
    ):
        raise ValueError("doctor lacks trusted single-operator capacity authority")
    authority_binding = CanonicalJsonProofBinding.from_dict(report["authority"])
    rebound, authority, decision = (
        revalidate_trusted_single_operator_stage_capacity_authority(
            authority_binding.absolute_path,
            expected_run_root_path=expected_run_root_path,
            expected_output_path=expected_output_path,
            require_available=require_available,
        )
    )
    _path_binding, spec, _pending = _deep_content_from_path_spec(
        authority.content_path_spec.absolute_path
    )
    if Path(spec.doctor_path) != Path(doctor_binding.absolute_path):
        raise ValueError("doctor path differs from the capacity content recipe")
    if expected_bound_content_source_path is not None:
        _require_bound_content_source_closure(
            authority,
            content_source_path=expected_bound_content_source_path,
            doctor_report_path=doctor_binding.absolute_path,
        )
    if (
        rebound != authority_binding
        or report.get("authority_sha256") != authority.sha256
        or report.get("required_free_bytes") != decision.required_free_bytes
        or report.get("current_wave_high_water_bytes")
        != decision.current_wave_high_water_bytes
        or report.get("retry_reserve_bytes") != decision.retry_reserve_bytes
        or report.get("safety_margin_bytes") != decision.safety_margin_bytes
    ):
        raise ValueError("doctor trusted capacity authority identity differs")
    return rebound, authority, decision


def require_trusted_single_operator_retry_capacity(
    authority_path: str | Path,
    *,
    store: ExperimentOperatorStore,
    previous_command: QueuedCommandSpec,
    stage: str,
) -> TrustedSingleOperatorCapacityDecision:
    """Reject every automatic retry before new paths or free-space assumptions."""

    from lightcone_spec.orchestration.experiment_operator import (
        ExperimentOperatorStore,
        QueuedCommandSpec,
    )

    if type(store) is not ExperimentOperatorStore:
        raise TypeError("trusted retry capacity requires the exact operator store")
    if type(previous_command) is not QueuedCommandSpec:
        raise TypeError("trusted retry capacity requires one exact prior command")
    _registered_stage(stage)
    _binding, authority = load_trusted_single_operator_stage_capacity_authority(
        authority_path
    )
    if authority.maximum_automatic_infrastructure_retries != 0:
        raise RuntimeError("trusted zero-reserve retry policy changed")
    raise TrustedSingleOperatorAutomaticRetryDisabled(
        "trusted zero-reserve capacity disables automatic retries"
    )


__all__ = [
    "TRUSTED_SINGLE_OPERATOR_CAPACITY_PROTOCOL_SHA256",
    "TRUSTED_SINGLE_OPERATOR_CAPACITY_SAFETY_MARGIN_BYTES",
    "TRUSTED_SINGLE_OPERATOR_CELL_HIGH_WATER_BYTES",
    "TRUSTED_SINGLE_OPERATOR_MAXIMUM_AUTOMATIC_RETRIES",
    "TRUSTED_SINGLE_OPERATOR_RETRY_POLICY_SHA256",
    "TrustedSingleOperatorAutomaticRetryDisabled",
    "TrustedSingleOperatorCapacityBlocked",
    "TrustedSingleOperatorCapacityDecision",
    "TrustedSingleOperatorCapacityRoot",
    "TrustedSingleOperatorStageCapacityAuthority",
    "load_trusted_single_operator_stage_capacity_authority",
    "publish_trusted_single_operator_stage_capacity_authority",
    "require_trusted_single_operator_bound_content_closure",
    "require_trusted_single_operator_operator_wave_capacity",
    "require_trusted_single_operator_ordinary_capacity",
    "require_trusted_single_operator_resident_group_capacity",
    "require_trusted_single_operator_restart_capacity",
    "require_trusted_single_operator_retry_capacity",
    "revalidate_trusted_single_operator_stage_capacity_authority",
    "trusted_single_operator_capacity_authority_from_doctor",
    "trusted_single_operator_capacity_free_bytes",
    "trusted_single_operator_resident_group_high_water_bytes",
]
