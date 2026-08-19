"""Trusted single-operator physical NEXTN TP2 interface/fit evidence.

This module deliberately does not mint ``VerifiedNextNTp2Authority``.  That
token belongs to the offline-root-signed path.  Instead it runs the same
source-owned, non-skippable ``nextn_tp2`` live qualification once for each of
the two immutable E6 targets and publishes a replayable empirical receipt.
The two receipts are then shared by E6 pilot and final materializations.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.config import load_run_config
from lightcone_spec.experiments.e6_stage_authority import (
    E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256,
)
from lightcone_spec.experiments.formal_content_source import (
    FormalContentSourceBinding,
)
from lightcone_spec.experiments.formal_protocol import (
    E6_MODELS,
    ProtocolLock,
    content_sha256,
    reject_banned_model_identity,
)
from lightcone_spec.experiments.formal_registry import protocol_lock_from_dict
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedModelSnapshotMember,
    TrustedSingleOperatorContentBundle,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FormalSingleOperatorJsonBinding,
    publish_formal_single_operator_json_artifact,
    rebuild_formal_single_operator_stage_completion,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
from lightcone_spec.runtime.native_qualification_runner import (
    NATIVE_RUNTIME_GPU_TEST_FILES,
    NATIVE_RUNTIME_GPU_TEST_NAMES,
    NativeRuntimeQualificationAssignment,
    NativeRuntimeQualificationObservation,
    _junit_summary,
    _process_group_exists,
    _publish_snapshot,
    _source_capability_sha256,
    _terminate_process_group,
    _validate_gpu_snapshot,
)
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.readiness import (
    NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S,
)

FORMAL_SINGLE_OPERATOR_E6_INTERFACE_FIT_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_e6_interface_fit_protocol",
        "trust_mode": "trusted_single_operator_empirical_no_signature",
        "models": list(E6_MODELS),
        "topology": "tp2_dp1",
        "physical_runs": 2,
        "suite": "nextn_tp2_exact_8_of_8_zero_fail_error_skip",
        "reuse": "same_two_terminals_for_pilot_and_final",
    }
)
FORMAL_SINGLE_OPERATOR_E6_BUILT_IN_MTP_INTERFACE_FIT_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "formal_single_operator_e6_interface_fit_protocol",
        "trust_mode": "trusted_single_operator_empirical_no_signature",
        "models": list(E6_MODELS),
        "topology": "tp2_dp1",
        "physical_runs": 2,
        "nextn_mode": "built_in_mtp_same_target_checkpoint",
        "component_authority": (
            "frozen_config_weight_index_safetensors_headers_and_target_snapshot"
        ),
        "external_drafter": "forbidden",
        "suite": "nextn_tp2_exact_8_of_8_zero_fail_error_skip",
        "reuse": "same_two_terminals_for_pilot_and_final",
    }
)
FORMAL_SINGLE_OPERATOR_E6_INTERFACE_REPLAY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_e6_interface_replay_protocol",
        "source": "current_e6_execution_source_and_shared_auxiliary_bundle",
        "execution": "no_additional_gpu_work",
    }
)
FORMAL_SINGLE_OPERATOR_E6_TRUSTED_SERVING_AUTHORITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_e6_trusted_serving_authority_protocol",
        "trust_mode": "trusted_single_operator_empirical_no_signature",
        "formal_measured_authorization": False,
        "source": "deep_replayed_two_model_interface_fit_auxiliary",
        "qualification": "nextn_tp2_exact_8_of_8_zero_fail_error_skip",
    }
)
NEXTN_MTP_INTERFACE_SHA256 = hashlib.sha256(
    b'{"drafter":"frozen_kv_mtp","hidden":"actual_mtp_rows",'
    b'"mask":"target_verified_rows","schema_version":1,'
    b'"source":"adapter_version","teacher":"target_verify_logits",'
    b'"tp2_models":["target_two_shards","drafter_two_shards"]}'
).hexdigest()
NEXTN_BUILT_IN_MTP_INTERFACE_SHA256 = hashlib.sha256(
    b'{"checkpoint":"target_and_mtp_same_frozen_snapshot",'
    b'"component":"config_index_safetensors_header_bound_mtp_namespace",'
    b'"external_drafter":false,"hidden":"actual_mtp_rows",'
    b'"mask":"target_verified_rows","schema_version":2,'
    b'"source":"adapter_version","teacher":"target_verify_logits",'
    b'"topology":"target_tp2_plus_builtin_mtp_tp2"}'
).hexdigest()
_E6_INTERFACE_FIT_SUBPROCESS_TIMEOUT_SECONDS = 1_800
_E6_INTERFACE_FIT_SUBPROCESS_CLEANUP_SECONDS = 60
_E6_INTERFACE_FIT_PUBLICATION_GRACE_SECONDS = 15 * 60


class FormalSingleOperatorE6InterfaceFitBlocked(RuntimeError):
    """A required model, TP2 launch, doctor, or GPU proof is unavailable."""


def _sha(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or "\n" in value
        or "\r" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be canonical text")
    return value


def _strict(label: str, value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _binding(path: str | Path, *, label: str) -> CanonicalJsonProofBinding:
    try:
        return CanonicalJsonProofBinding.bind(path)
    except (OSError, TypeError, ValueError) as error:
        raise FormalSingleOperatorE6InterfaceFitBlocked(
            f"{label}_unavailable"
        ) from error


def _root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute() or root != root.resolve(strict=False):
        raise ValueError("E6 interface/fit output root must be absolute")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("E6 interface/fit output root is unavailable")
    return root


def _member(
    bundle: TrustedSingleOperatorContentBundle,
    *,
    member_id: str,
    role: str,
) -> TrustedModelSnapshotMember:
    matches = tuple(
        row
        for row in bundle.model_members
        if row.sha256 == member_id and row.role == role and "E6" in row.stages
    )
    if len(matches) != 1:
        raise FormalSingleOperatorE6InterfaceFitBlocked(
            f"exact_e6_{role}_model_content_unavailable"
        )
    return matches[0]


def _trusted_shard_manifest(
    member: TrustedModelSnapshotMember,
    *,
    gpu_uuids: tuple[str, str],
) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_single_operator_nextn_tp2_snapshot_shards",
            "member_sha256": member.sha256,
            "model_id": member.model_id,
            "revision": member.revision,
            "tree_sha256": member.tree_sha256,
            "content_sha256": member.content_sha256,
            "tensor_parallel_size": 2,
            "gpu_uuids": list(gpu_uuids),
        }
    )


def _topology_sha256(
    inventory: GpuInventory,
    gpu_uuids: tuple[str, str],
) -> str:
    groups = tuple(
        row for row in inventory.topology_groups if set(gpu_uuids) <= set(row.gpu_uuids)
    )
    if len(groups) != 1:
        raise FormalSingleOperatorE6InterfaceFitBlocked(
            "exact_tp2_topology_group_unavailable"
        )
    group = groups[0]
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_single_operator_nextn_tp2_topology",
            "inventory_sha256": inventory.sha256,
            "group_id": group.group_id,
            "host_id": group.host_id,
            "fabric": group.fabric,
            "bandwidth_class": group.bandwidth_class,
            "ordered_gpu_uuids": list(gpu_uuids),
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
        }
    )


def _doctor(
    binding: CanonicalJsonProofBinding,
    *,
    inventory: GpuInventory,
    launch: CompileLaunchManifest,
) -> tuple[str, str]:
    value = binding.reopen()
    if type(value) is not dict:
        raise FormalSingleOperatorE6InterfaceFitBlocked(
            "runtime_doctor_report_malformed"
        )
    readiness = value.get("readiness")
    checks = value.get("checks")
    roots = value.get("roots")
    python = value.get("python")
    gpu = value.get("gpu")
    commands = value.get("commands")
    parsed = None if type(gpu) is not dict else gpu.get("parsed_inventory")
    devices = None if type(parsed) is not dict else parsed.get("devices")
    if (
        value.get("schema_version") != 2
        or value.get("status") != "PASS"
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
        or type(roots) is not dict
        or roots.get("patched_sglang") != launch.patched_sglang_checkout
        or type(python) is not dict
        or type(commands) is not dict
        or type(devices) is not list
    ):
        raise FormalSingleOperatorE6InterfaceFitBlocked(
            "complete_pass_runtime_doctor_unavailable"
        )
    by_uuid = {row.get("uuid"): row for row in devices if type(row) is dict}
    for device in inventory.devices:
        row = by_uuid.get(device.uuid)
        if (
            type(row) is not dict
            or row.get("name") != device.model
            or row.get("compute_capability")
            != f"{device.compute_capability[0]}.{device.compute_capability[1]}"
        ):
            raise FormalSingleOperatorE6InterfaceFitBlocked(
                "doctor_gpu_identity_differs"
            )
    python_path = Path(str(python.get("executable"))).resolve(strict=False)
    nvidia = shutil.which("nvidia-smi", path=os.pathsep.join(launch.path_entries))
    if (
        not python_path.is_file()
        or python_path.is_symlink()
        or nvidia is None
        or not Path(nvidia).resolve(strict=True).is_file()
        or not str(commands.get("nvidia_smi", "")).strip()
    ):
        raise FormalSingleOperatorE6InterfaceFitBlocked(
            "doctor_runtime_tools_unavailable"
        )
    return str(python_path), str(Path(nvidia).resolve(strict=True))


@dataclass(frozen=True)
class FormalSingleOperatorE6InterfaceFitPlan:
    schema_version: Literal[1, 2]
    kind: Literal["formal_single_operator_e6_interface_fit_plan"]
    protocol_sha256: str
    protocol_lock: FormalSingleOperatorJsonBinding
    predecessor_completion: FormalSingleOperatorJsonBinding
    content_source: FormalContentSourceBinding
    inventory: CanonicalJsonProofBinding
    doctor: CanonicalJsonProofBinding
    launch_manifest: CanonicalJsonProofBinding
    native_assignment: CanonicalJsonProofBinding
    model: str
    target_member_sha256: str
    drafter_member_sha256: str
    target_shard_manifest_sha256: str
    drafter_shard_manifest_sha256: str
    interface_sha256: str
    topology_sha256: str
    source_adapter_version: Literal[0]
    gpu_uuids: tuple[str, str]
    physical_run_index: int
    evidence_directory: str
    nextn_mtp_mode: Literal["external_drafter", "built_in_mtp"] = "external_drafter"
    target_snapshot_sha256: str | None = None
    mtp_component_sha256: str | None = None
    mtp_component: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {1, 2}
            or self.kind != "formal_single_operator_e6_interface_fit_plan"
            or self.model not in E6_MODELS
            or self.source_adapter_version != 0
            or self.physical_run_index != E6_MODELS.index(self.model)
            or len(self.gpu_uuids) != 2
            or len(set(self.gpu_uuids)) != 2
        ):
            raise ValueError("E6 interface/fit plan identity differs")
        built_in = self.schema_version == 2
        if built_in:
            if (
                self.protocol_sha256
                != FORMAL_SINGLE_OPERATOR_E6_BUILT_IN_MTP_INTERFACE_FIT_PROTOCOL_SHA256
                or self.nextn_mtp_mode != "built_in_mtp"
                or self.interface_sha256 != NEXTN_BUILT_IN_MTP_INTERFACE_SHA256
                or self.target_snapshot_sha256 is None
                or self.mtp_component_sha256 is None
                or self.target_snapshot_sha256 == self.mtp_component_sha256
                or type(self.mtp_component) is not CanonicalJsonProofBinding
                or self.mtp_component.semantic_sha256 != self.mtp_component_sha256
                or self.target_member_sha256 != self.drafter_member_sha256
                or self.target_shard_manifest_sha256
                != self.drafter_shard_manifest_sha256
            ):
                raise ValueError("E6 built-in MTP plan identity differs")
        elif (
            self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_E6_INTERFACE_FIT_PROTOCOL_SHA256
            or self.nextn_mtp_mode != "external_drafter"
            or self.interface_sha256 != NEXTN_MTP_INTERFACE_SHA256
            or self.target_snapshot_sha256 is not None
            or self.mtp_component_sha256 is not None
            or self.mtp_component is not None
            or self.target_member_sha256 == self.drafter_member_sha256
        ):
            raise ValueError("E6 external-drafter plan identity differs")
        for label, digest in (
            ("target member", self.target_member_sha256),
            ("drafter member", self.drafter_member_sha256),
            ("target shards", self.target_shard_manifest_sha256),
            ("drafter shards", self.drafter_shard_manifest_sha256),
            ("interface", self.interface_sha256),
            ("topology", self.topology_sha256),
        ):
            _sha(f"E6 interface/fit {label}", digest)
        if built_in:
            _sha("E6 target snapshot", self.target_snapshot_sha256)
            _sha("E6 MTP component", self.mtp_component_sha256)
        if type(self.protocol_lock) is not FormalSingleOperatorJsonBinding:
            raise TypeError("E6 interface/fit ProtocolLock is not path-bound")
        if type(self.predecessor_completion) is not FormalSingleOperatorJsonBinding:
            raise TypeError("E6 interface/fit predecessor is not path-bound")
        if (
            type(self.content_source) is not FormalContentSourceBinding
            or self.content_source.mode != "trusted_single_operator"
        ):
            raise TypeError("E6 interface/fit content source is not trusted")
        for value in (
            self.inventory,
            self.doctor,
            self.launch_manifest,
            self.native_assignment,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("E6 interface/fit plan input is not path-bound")
        root = Path(self.evidence_directory)
        if not root.is_absolute() or root != root.resolve(strict=False):
            raise ValueError("E6 interface/fit evidence directory differs")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = {
            **asdict(self),
            "protocol_lock": self.protocol_lock.to_dict(),
            "predecessor_completion": self.predecessor_completion.to_dict(),
            "content_source": self.content_source.to_dict(),
            "inventory": self.inventory.to_dict(),
            "doctor": self.doctor.to_dict(),
            "launch_manifest": self.launch_manifest.to_dict(),
            "native_assignment": self.native_assignment.to_dict(),
            "gpu_uuids": list(self.gpu_uuids),
        }
        if self.schema_version == 1:
            value.pop("nextn_mtp_mode")
            value.pop("target_snapshot_sha256")
            value.pop("mtp_component_sha256")
            value.pop("mtp_component")
        else:
            assert self.mtp_component is not None
            value["mtp_component"] = self.mtp_component.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("E6 interface/fit plan must be an object")
        schema_version = value.get("schema_version")
        expected = set(cls.__dataclass_fields__)
        if schema_version == 1:
            expected -= {
                "nextn_mtp_mode",
                "target_snapshot_sha256",
                "mtp_component_sha256",
                "mtp_component",
            }
        row = _strict("E6 interface/fit plan", value, expected)
        row["protocol_lock"] = FormalSingleOperatorJsonBinding.from_dict(
            row["protocol_lock"]
        )
        row["predecessor_completion"] = FormalSingleOperatorJsonBinding.from_dict(
            row["predecessor_completion"]
        )
        row["content_source"] = FormalContentSourceBinding.from_dict(
            row["content_source"]
        )
        for name in (
            "inventory",
            "doctor",
            "launch_manifest",
            "native_assignment",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        if schema_version == 1:
            row["nextn_mtp_mode"] = "external_drafter"
            row["target_snapshot_sha256"] = None
            row["mtp_component_sha256"] = None
            row["mtp_component"] = None
        else:
            row["mtp_component"] = CanonicalJsonProofBinding.from_dict(
                row["mtp_component"]
            )
        raw_gpus = row.pop("gpu_uuids")
        if type(raw_gpus) is not list:
            raise TypeError("E6 interface/fit GPU UUIDs must be an array")
        return cls(**row, gpu_uuids=tuple(raw_gpus))  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorE6InterfaceFitCampaign:
    schema_version: Literal[1, 2]
    kind: Literal["formal_single_operator_e6_interface_fit_campaign"]
    protocol_sha256: str
    protocol_lock_sha256: str
    predecessor_completion_sha256: str
    trusted_content_sha256: str
    inventory_sha256: str
    models: tuple[str, str]
    gpu_uuids: tuple[str, str]
    plans: tuple[CanonicalJsonProofBinding, CanonicalJsonProofBinding]
    physical_run_count: Literal[2]
    nextn_mtp_mode: Literal["external_drafter", "built_in_mtp"] = "external_drafter"
    target_snapshot_sha256s: tuple[str, str] | None = None
    mtp_component_sha256s: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {1, 2}
            or self.kind != "formal_single_operator_e6_interface_fit_campaign"
            or self.models != E6_MODELS
            or self.physical_run_count != 2
            or len(self.gpu_uuids) != 2
            or len(set(self.gpu_uuids)) != 2
            or len(self.plans) != 2
            or len(set(self.plans)) != 2
        ):
            raise ValueError("E6 interface/fit campaign identity differs")
        if self.schema_version == 2:
            if (
                self.protocol_sha256
                != FORMAL_SINGLE_OPERATOR_E6_BUILT_IN_MTP_INTERFACE_FIT_PROTOCOL_SHA256
                or self.nextn_mtp_mode != "built_in_mtp"
                or self.target_snapshot_sha256s is None
                or self.mtp_component_sha256s is None
                or len(set(self.target_snapshot_sha256s)) != 2
                or len(set(self.mtp_component_sha256s)) != 2
                or any(
                    target == component
                    for target, component in zip(
                        self.target_snapshot_sha256s,
                        self.mtp_component_sha256s,
                        strict=True,
                    )
                )
            ):
                raise ValueError("E6 built-in MTP campaign identity differs")
            for digest in (
                *self.target_snapshot_sha256s,
                *self.mtp_component_sha256s,
            ):
                _sha("E6 built-in MTP campaign digest", digest)
        elif (
            self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_E6_INTERFACE_FIT_PROTOCOL_SHA256
            or self.nextn_mtp_mode != "external_drafter"
            or self.target_snapshot_sha256s is not None
            or self.mtp_component_sha256s is not None
        ):
            raise ValueError("E6 external campaign identity differs")
        for label, digest in (
            ("ProtocolLock", self.protocol_lock_sha256),
            ("predecessor", self.predecessor_completion_sha256),
            ("trusted content", self.trusted_content_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _sha(f"E6 campaign {label}", digest)
        if any(type(row) is not CanonicalJsonProofBinding for row in self.plans):
            raise TypeError("E6 campaign plan is not path-bound")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = {
            **asdict(self),
            "models": list(self.models),
            "gpu_uuids": list(self.gpu_uuids),
            "plans": [row.to_dict() for row in self.plans],
        }
        if self.schema_version == 1:
            value.pop("nextn_mtp_mode")
            value.pop("target_snapshot_sha256s")
            value.pop("mtp_component_sha256s")
        else:
            assert self.target_snapshot_sha256s is not None
            assert self.mtp_component_sha256s is not None
            value["target_snapshot_sha256s"] = list(self.target_snapshot_sha256s)
            value["mtp_component_sha256s"] = list(self.mtp_component_sha256s)
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("E6 interface/fit campaign must be an object")
        schema_version = value.get("schema_version")
        expected = set(cls.__dataclass_fields__)
        if schema_version == 1:
            expected -= {
                "nextn_mtp_mode",
                "target_snapshot_sha256s",
                "mtp_component_sha256s",
            }
        row = _strict("E6 interface/fit campaign", value, expected)
        for name in ("models", "gpu_uuids", "plans"):
            if type(row[name]) is not list:
                raise TypeError(f"E6 campaign {name} must be an array")
        if schema_version == 1:
            row["nextn_mtp_mode"] = "external_drafter"
            row["target_snapshot_sha256s"] = None
            row["mtp_component_sha256s"] = None
        else:
            for name in ("target_snapshot_sha256s", "mtp_component_sha256s"):
                if type(row[name]) is not list:
                    raise TypeError(f"E6 campaign {name} must be an array")
                row[name] = tuple(row[name])
        return cls(
            **{
                **row,
                "models": tuple(row["models"]),
                "gpu_uuids": tuple(row["gpu_uuids"]),
                "plans": tuple(
                    CanonicalJsonProofBinding.from_dict(item) for item in row["plans"]
                ),
            }
        )  # type: ignore[arg-type]


def _load_campaign(path: str | Path) -> FormalSingleOperatorE6InterfaceFitCampaign:
    binding = CanonicalJsonProofBinding.bind(path)
    campaign = FormalSingleOperatorE6InterfaceFitCampaign.from_dict(binding.reopen())
    if campaign.sha256 != binding.semantic_sha256:
        raise ValueError("E6 interface/fit campaign binding differs")
    plans = tuple(
        revalidate_formal_single_operator_e6_interface_fit_plan(row.absolute_path)
        for row in campaign.plans
    )
    if (
        tuple(row.model for row in plans) != E6_MODELS
        or any(row.gpu_uuids != campaign.gpu_uuids for row in plans)
        or {row.protocol_lock.semantic_sha256 for row in plans}
        != {campaign.protocol_lock_sha256}
        or {row.predecessor_completion.semantic_sha256 for row in plans}
        != {campaign.predecessor_completion_sha256}
        or {row.content_source.content_sha256 for row in plans}
        != {campaign.trusted_content_sha256}
        or {row.inventory.semantic_sha256 for row in plans}
        != {campaign.inventory_sha256}
        or {row.nextn_mtp_mode for row in plans} != {campaign.nextn_mtp_mode}
        or (
            None
            if campaign.schema_version == 1
            else tuple(row.target_snapshot_sha256 for row in plans)
        )
        != campaign.target_snapshot_sha256s
        or (
            None
            if campaign.schema_version == 1
            else tuple(row.mtp_component_sha256 for row in plans)
        )
        != campaign.mtp_component_sha256s
    ):
        raise ValueError("E6 interface/fit campaign plan set differs")
    return campaign


def formal_single_operator_e6_interface_fit_process_hard_timeout_ns(
    campaign_path: str | Path,
) -> int:
    """Deep-replay the exact-two campaign and return its whole-worker cap."""

    campaign = _load_campaign(campaign_path)
    if campaign.physical_run_count != 2 or len(campaign.plans) != 2:
        raise ValueError("E6 interface/fit timeout requires exact-two coverage")
    seconds = (
        campaign.physical_run_count
        * (
            _E6_INTERFACE_FIT_SUBPROCESS_TIMEOUT_SECONDS
            + _E6_INTERFACE_FIT_SUBPROCESS_CLEANUP_SECONDS
        )
        + _E6_INTERFACE_FIT_PUBLICATION_GRACE_SECONDS
    )
    return seconds * 1_000_000_000


def materialize_formal_single_operator_e6_interface_fit_campaign(
    *,
    protocol_lock_path: str | Path,
    predecessor_completion_path: str | Path,
    trusted_content_bundle_path: str | Path,
    launch_manifest_paths: dict[str, str | Path],
    output_root: str | Path,
) -> FormalSingleOperatorE6InterfaceFitCampaign:
    """Prepare exactly two immutable source-owned NEXTN TP2 physical plans."""

    if (
        type(launch_manifest_paths) is not dict
        or tuple(model for model in E6_MODELS if model in launch_manifest_paths)
        != E6_MODELS
        or set(launch_manifest_paths) != set(E6_MODELS)
    ):
        raise ValueError("E6 interface/fit launches must cover both exact models")
    lock_binding = FormalSingleOperatorJsonBinding.bind(
        protocol_lock_path,
        label="E6 interface/fit ProtocolLock",
    )
    lock = protocol_lock_from_dict(
        lock_binding.reopen(label="E6 interface/fit ProtocolLock")
    )
    if (
        lock.sha256 != lock_binding.semantic_sha256
        or lock.schema_version != 5
        or lock.content_source_mode != "trusted_single_operator"
    ):
        raise FormalSingleOperatorE6InterfaceFitBlocked(
            "schema5_trusted_protocol_lock_required"
        )
    predecessor_binding = FormalSingleOperatorJsonBinding.bind(
        predecessor_completion_path,
        label="E6 interface/fit E5 predecessor",
    )
    predecessor = rebuild_formal_single_operator_stage_completion(
        predecessor_binding.absolute_path
    )
    if (
        predecessor.artifact.node != "e5_final"
        or predecessor.artifact.protocol_lock_sha256 != lock.sha256
        or predecessor.decision.payload.get("status") != "CONFIRMED"
    ):
        raise FormalSingleOperatorE6InterfaceFitBlocked(
            "confirmed_current_e5_predecessor_required"
        )
    content = FormalContentSourceBinding.bind_trusted_single_operator(
        str(trusted_content_bundle_path)
    )
    bundle = content.reopen()
    assert type(bundle) is TrustedSingleOperatorContentBundle
    if (
        bundle.runtime_binding_status != "BOUND"
        or content.content_sha256 != lock.trusted_single_operator_content_bundle_sha256
        or bundle.runtime_observations is None
    ):
        raise FormalSingleOperatorE6InterfaceFitBlocked(
            "runtime_bound_trusted_content_required"
        )
    inventory_artifact = bundle.runtime_observations.inventory
    doctor_artifact = bundle.runtime_observations.doctor
    inventory_binding = _binding(
        inventory_artifact.absolute_path,
        label="trusted_runtime_inventory",
    )
    doctor_binding = _binding(
        doctor_artifact.absolute_path,
        label="trusted_runtime_doctor",
    )
    if (
        inventory_binding.raw_sha256 != inventory_artifact.raw_sha256
        or inventory_binding.semantic_sha256 != inventory_artifact.semantic_sha256
        or doctor_binding.raw_sha256 != doctor_artifact.raw_sha256
        or doctor_binding.semantic_sha256 != doctor_artifact.semantic_sha256
    ):
        raise RuntimeError("E6 trusted runtime observations changed")
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    if inventory.sha256 != inventory_binding.semantic_sha256:
        raise ValueError("E6 trusted inventory canonical identity differs")
    root = _root(output_root)
    plan_bindings = []
    gpu_order: tuple[str, str] | None = None
    for index, model in enumerate(E6_MODELS):
        launch_binding = _binding(
            launch_manifest_paths[model],
            label=f"e6_{index}_compile_launch",
        )
        launch = CompileLaunchManifest.load(launch_binding.absolute_path)
        config = load_run_config(launch.run_config_path)
        if (
            launch.schema_version not in {2, 3}
            or launch.formal_stage != "E6"
            or launch.content_source_binding != content
            or launch.inventory_sha256 != inventory.sha256
            or config.model.algorithm != "NEXTN"
            or config.model.target != model
            or config.model.target_revision != launch.target_revision
            or config.model.drafter != launch.drafter_model_id
            or config.model.drafter_revision != launch.drafter_revision
            or config.runtime.tensor_parallel_size != 2
            or config.runtime.data_parallel_size != 1
            or not config.runtime.speculation_enabled
            or len(launch.gpu_uuids) != 2
            or len(set(launch.gpu_uuids)) != 2
            or launch.drafter_content_member_id is None
        ):
            raise FormalSingleOperatorE6InterfaceFitBlocked(
                f"exact_nextn_tp2_launch_unavailable_for_model_{index}"
            )
        built_in = launch.schema_version == 3
        if built_in != (config.model.nextn_mtp_mode == "built_in_mtp"):
            raise FormalSingleOperatorE6InterfaceFitBlocked(
                f"exact_nextn_mtp_mode_unavailable_for_model_{index}"
            )
        ordered_gpus = tuple(launch.gpu_uuids)
        if gpu_order is None:
            gpu_order = ordered_gpus  # type: ignore[assignment]
        elif gpu_order != ordered_gpus:
            raise ValueError("E6 interface/fit launches change GPU placement")
        target = _member(
            bundle,
            member_id=launch.target_content_member_id,
            role="target",
        )
        if built_in:
            from lightcone_spec.experiments.formal_single_operator_e6_builtin_mtp import (
                revalidate_formal_single_operator_e6_builtin_mtp_component,
            )

            if (
                launch.drafter_content_member_id != launch.target_content_member_id
                or launch.drafter_model_id != launch.target_model_id
                or launch.drafter_revision != launch.target_revision
                or launch.drafter_snapshot_path != launch.target_snapshot_path
                or launch.mtp_component_binding is None
            ):
                raise FormalSingleOperatorE6InterfaceFitBlocked(
                    f"builtin_mtp_same_snapshot_unavailable_for_model_{index}"
                )
            component = revalidate_formal_single_operator_e6_builtin_mtp_component(
                launch.mtp_component_binding.absolute_path,
                member=target,
            )
            drafter = target
        else:
            component = None
            drafter = _member(
                bundle,
                member_id=launch.drafter_content_member_id,
                role="drafter",
            )
        if (
            target.model_id != model
            or target.revision != launch.target_revision
            or drafter.model_id != launch.drafter_model_id
            or drafter.revision != launch.drafter_revision
        ):
            raise FormalSingleOperatorE6InterfaceFitBlocked(
                f"exact_nextn_model_pair_unavailable_for_model_{index}"
            )
        topology = _topology_sha256(inventory, ordered_gpus)  # type: ignore[arg-type]
        python_path, nvidia_path = _doctor(
            doctor_binding,
            inventory=inventory,
            launch=launch,
        )
        devices = {row.uuid: row for row in inventory.devices}
        if any(uuid not in devices for uuid in ordered_gpus):
            raise ValueError("E6 interface/fit GPU placement leaves inventory")
        model_root = root / f"model-{index}"
        model_root.mkdir(mode=0o700, exist_ok=False)
        python_evidence = EvidenceFileBinding.bind(
            python_path,
            label="E6 interface/fit Python",
        )
        nvidia_evidence = EvidenceFileBinding.bind(
            nvidia_path,
            label="E6 interface/fit nvidia-smi",
        )
        run_nonce = content_sha256(
            {
                "schema_version": 1,
                "kind": "formal_single_operator_e6_interface_fit_run_nonce",
                "protocol_lock_sha256": lock.sha256,
                "predecessor_completion_sha256": (predecessor_binding.semantic_sha256),
                "model": model,
                "launch_manifest_sha256": launch.sha256,
                "physical_run_index": index,
            }
        )
        assignment = NativeRuntimeQualificationAssignment(
            schema_version=1,
            kind="formal_native_runtime_gpu_qualification_assignment",
            suite_id="nextn_tp2",
            runner_protocol_sha256=(
                NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S["nextn_tp2"]
            ),
            registry_sha256=lock.registry_sha256,
            runtime_sha256=lock.formal_runtime_authority_manifest_sha256,
            topology_sha256=topology,
            inventory_sha256=inventory.sha256,
            hardware_envelope_sha256=content_sha256(
                {
                    "inventory": inventory.sha256,
                    "doctor": doctor_binding.semantic_sha256,
                    "gpus": list(ordered_gpus),
                }
            ),
            run_nonce_sha256=run_nonce,
            gpu_uuids=ordered_gpus,
            gpu_models=tuple(devices[uuid].model for uuid in ordered_gpus),
            launch_manifest=launch_binding,
            base_exactness_result_pointer=None,
            eagle3_selector_status=None,
            eagle3_compatibility_authority_sha256=None,
            eagle3_model_selector_sha256=None,
            python_executable=python_evidence.absolute_path,
            python_executable_raw_sha256=python_evidence.raw_sha256,
            python_executable_size=python_evidence.size,
            nvidia_smi_executable=nvidia_evidence.absolute_path,
            nvidia_smi_raw_sha256=nvidia_evidence.raw_sha256,
            nvidia_smi_size=nvidia_evidence.size,
            evidence_directory=str(model_root),
        )
        assignment_path = model_root / "native-assignment.json"
        assignment_binding = assignment.write(assignment_path)
        plan = FormalSingleOperatorE6InterfaceFitPlan(
            schema_version=2 if built_in else 1,
            kind="formal_single_operator_e6_interface_fit_plan",
            protocol_sha256=(
                FORMAL_SINGLE_OPERATOR_E6_BUILT_IN_MTP_INTERFACE_FIT_PROTOCOL_SHA256
                if built_in
                else FORMAL_SINGLE_OPERATOR_E6_INTERFACE_FIT_PROTOCOL_SHA256
            ),
            protocol_lock=lock_binding,
            predecessor_completion=predecessor_binding,
            content_source=content,
            inventory=inventory_binding,
            doctor=doctor_binding,
            launch_manifest=launch_binding,
            native_assignment=assignment_binding,
            model=model,
            target_member_sha256=target.sha256,
            drafter_member_sha256=drafter.sha256,
            target_shard_manifest_sha256=_trusted_shard_manifest(
                target,
                gpu_uuids=ordered_gpus,  # type: ignore[arg-type]
            ),
            drafter_shard_manifest_sha256=_trusted_shard_manifest(
                drafter,
                gpu_uuids=ordered_gpus,  # type: ignore[arg-type]
            ),
            interface_sha256=(
                NEXTN_BUILT_IN_MTP_INTERFACE_SHA256
                if built_in
                else NEXTN_MTP_INTERFACE_SHA256
            ),
            topology_sha256=topology,
            source_adapter_version=0,
            gpu_uuids=ordered_gpus,
            physical_run_index=index,
            evidence_directory=str(model_root),
            nextn_mtp_mode=("built_in_mtp" if built_in else "external_drafter"),
            target_snapshot_sha256=(
                None if component is None else component.target_snapshot_sha256
            ),
            mtp_component_sha256=(None if component is None else component.sha256),
            mtp_component=(None if component is None else launch.mtp_component_binding),
        )
        plan_path = model_root / "e6-interface-fit-plan.json"
        publish_canonical_json_no_replace(plan_path, plan.to_dict())
        plan_bindings.append(
            CanonicalJsonProofBinding.bind(plan_path, semantic_sha256=plan.sha256)
        )
    assert gpu_order is not None
    plan_values = tuple(
        revalidate_formal_single_operator_e6_interface_fit_plan(row.absolute_path)
        for row in plan_bindings
    )
    modes = {row.nextn_mtp_mode for row in plan_values}
    if len(modes) != 1:
        raise ValueError("E6 interface/fit launch modes are mixed")
    campaign_built_in = modes == {"built_in_mtp"}
    campaign = FormalSingleOperatorE6InterfaceFitCampaign(
        schema_version=2 if campaign_built_in else 1,
        kind="formal_single_operator_e6_interface_fit_campaign",
        protocol_sha256=(
            FORMAL_SINGLE_OPERATOR_E6_BUILT_IN_MTP_INTERFACE_FIT_PROTOCOL_SHA256
            if campaign_built_in
            else FORMAL_SINGLE_OPERATOR_E6_INTERFACE_FIT_PROTOCOL_SHA256
        ),
        protocol_lock_sha256=lock.sha256,
        predecessor_completion_sha256=predecessor_binding.semantic_sha256,
        trusted_content_sha256=content.content_sha256,
        inventory_sha256=inventory.sha256,
        models=E6_MODELS,
        gpu_uuids=gpu_order,
        plans=tuple(plan_bindings),  # type: ignore[arg-type]
        physical_run_count=2,
        nextn_mtp_mode=("built_in_mtp" if campaign_built_in else "external_drafter"),
        target_snapshot_sha256s=(
            tuple(row.target_snapshot_sha256 for row in plan_values)
            if campaign_built_in
            else None
        ),
        mtp_component_sha256s=(
            tuple(row.mtp_component_sha256 for row in plan_values)
            if campaign_built_in
            else None
        ),
    )
    campaign_path = root / "e6-interface-fit-campaign.json"
    publish_canonical_json_no_replace(campaign_path, campaign.to_dict())
    return _load_campaign(campaign_path)


def revalidate_formal_single_operator_e6_interface_fit_plan(
    path: str | Path,
) -> FormalSingleOperatorE6InterfaceFitPlan:
    binding = CanonicalJsonProofBinding.bind(path)
    plan = FormalSingleOperatorE6InterfaceFitPlan.from_dict(binding.reopen())
    if plan.sha256 != binding.semantic_sha256:
        raise ValueError("E6 interface/fit plan binding differs")
    lock = protocol_lock_from_dict(
        plan.protocol_lock.reopen(label="E6 interface/fit ProtocolLock")
    )
    predecessor = rebuild_formal_single_operator_stage_completion(
        plan.predecessor_completion.absolute_path
    )
    bundle = plan.content_source.reopen()
    if (
        lock.schema_version != 5
        or lock.sha256 != plan.protocol_lock.semantic_sha256
        or lock.trusted_single_operator_content_bundle_sha256
        != plan.content_source.content_sha256
        or predecessor.artifact.node != "e5_final"
        or predecessor.artifact.protocol_lock_sha256 != lock.sha256
        or predecessor.decision.payload.get("status") != "CONFIRMED"
        or type(bundle) is not TrustedSingleOperatorContentBundle
        or bundle.runtime_binding_status != "BOUND"
        or bundle.runtime_observations is None
    ):
        raise ValueError("E6 interface/fit trusted lineage differs")
    inventory = GpuInventory.from_dict(plan.inventory.reopen())
    launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    assignment = NativeRuntimeQualificationAssignment.load(
        plan.native_assignment.absolute_path
    )
    target = _member(bundle, member_id=plan.target_member_sha256, role="target")
    if plan.schema_version == 2:
        from lightcone_spec.experiments.formal_single_operator_e6_builtin_mtp import (
            revalidate_formal_single_operator_e6_builtin_mtp_component,
        )

        if plan.mtp_component is None:
            raise ValueError("E6 built-in MTP plan lacks its component")
        component = revalidate_formal_single_operator_e6_builtin_mtp_component(
            plan.mtp_component.absolute_path,
            member=target,
        )
        drafter = target
    else:
        component = None
        drafter = _member(
            bundle,
            member_id=plan.drafter_member_sha256,
            role="drafter",
        )
    _doctor(plan.doctor, inventory=inventory, launch=launch)
    observations = bundle.runtime_observations
    assert observations is not None
    if (
        inventory.sha256 != plan.inventory.semantic_sha256
        or (
            plan.inventory.absolute_path,
            plan.inventory.raw_sha256,
            plan.inventory.semantic_sha256,
            plan.inventory.size,
        )
        != (
            observations.inventory.absolute_path,
            observations.inventory.raw_sha256,
            observations.inventory.semantic_sha256,
            observations.inventory.size,
        )
        or (
            plan.doctor.absolute_path,
            plan.doctor.raw_sha256,
            plan.doctor.semantic_sha256,
            plan.doctor.size,
        )
        != (
            observations.doctor.absolute_path,
            observations.doctor.raw_sha256,
            observations.doctor.semantic_sha256,
            observations.doctor.size,
        )
        or assignment.launch_manifest != plan.launch_manifest
        or assignment.suite_id != "nextn_tp2"
        or assignment.registry_sha256 != lock.registry_sha256
        or assignment.runtime_sha256 != lock.formal_runtime_authority_manifest_sha256
        or assignment.inventory_sha256 != inventory.sha256
        or assignment.gpu_uuids != plan.gpu_uuids
        or assignment.topology_sha256 != plan.topology_sha256
        or launch.schema_version != (3 if plan.schema_version == 2 else 2)
        or launch.formal_stage != "E6"
        or launch.content_source_binding != plan.content_source
        or launch.target_model_id != plan.model
        or launch.target_content_member_id != target.sha256
        or launch.drafter_content_member_id != drafter.sha256
        or launch.nextn_mtp_mode
        != ("built_in_mtp" if plan.schema_version == 2 else None)
        or launch.target_snapshot_sha256 != plan.target_snapshot_sha256
        or launch.mtp_component_sha256 != plan.mtp_component_sha256
        or launch.mtp_component_binding != plan.mtp_component
        or (
            component is not None
            and (
                component.target_snapshot_sha256 != plan.target_snapshot_sha256
                or component.sha256 != plan.mtp_component_sha256
                or component.target_member_sha256 != target.sha256
            )
        )
        or plan.target_shard_manifest_sha256
        != _trusted_shard_manifest(target, gpu_uuids=plan.gpu_uuids)
        or plan.drafter_shard_manifest_sha256
        != _trusted_shard_manifest(drafter, gpu_uuids=plan.gpu_uuids)
        or plan.topology_sha256 != _topology_sha256(inventory, plan.gpu_uuids)
        or Path(plan.evidence_directory) != Path(binding.absolute_path).parent
        or Path(plan.native_assignment.absolute_path).parent
        != Path(plan.evidence_directory)
    ):
        raise ValueError("E6 interface/fit plan replay differs")
    return plan


@dataclass(frozen=True)
class FormalSingleOperatorE6InterfaceFitTerminal:
    schema_version: Literal[1, 2]
    kind: Literal["formal_single_operator_e6_interface_fit_terminal"]
    protocol_sha256: str
    plan: CanonicalJsonProofBinding
    model: str
    before_gpu_snapshot: CanonicalJsonProofBinding
    after_gpu_snapshot: CanonicalJsonProofBinding
    junit_xml: EvidenceFileBinding
    runner_log: EvidenceFileBinding
    live_observation: CanonicalJsonProofBinding
    live_native_terminal: CanonicalJsonProofBinding
    live_native_itl: CanonicalJsonProofBinding
    live_graph: CanonicalJsonProofBinding
    live_worker_hook: CanonicalJsonProofBinding
    live_rank_terminals: tuple[CanonicalJsonProofBinding, CanonicalJsonProofBinding]
    live_server_receipt: CanonicalJsonProofBinding
    live_server_log: EvidenceFileBinding
    native_gpu_proof_sha256: str
    distributed_gpu_proof_sha256: str
    trusted_authority_sha256: str
    started_ns: int
    finished_ns: int
    status: Literal["COMPLETE"]
    physical_execution_count: Literal[1]
    nextn_mtp_mode: Literal["external_drafter", "built_in_mtp"] = "external_drafter"
    target_snapshot_sha256: str | None = None
    mtp_component_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {1, 2}
            or self.kind != "formal_single_operator_e6_interface_fit_terminal"
            or self.model not in E6_MODELS
            or self.status != "COMPLETE"
            or self.physical_execution_count != 1
            or len(self.live_rank_terminals) != 2
            or len(set(self.live_rank_terminals)) != 2
            or self.started_ns < 1
            or self.finished_ns <= self.started_ns
        ):
            raise ValueError("E6 interface/fit terminal identity differs")
        if self.schema_version == 2:
            if (
                self.protocol_sha256
                != FORMAL_SINGLE_OPERATOR_E6_BUILT_IN_MTP_INTERFACE_FIT_PROTOCOL_SHA256
                or self.nextn_mtp_mode != "built_in_mtp"
                or self.target_snapshot_sha256 is None
                or self.mtp_component_sha256 is None
            ):
                raise ValueError("E6 built-in MTP terminal identity differs")
            _sha("E6 terminal target snapshot", self.target_snapshot_sha256)
            _sha("E6 terminal MTP component", self.mtp_component_sha256)
        elif (
            self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_E6_INTERFACE_FIT_PROTOCOL_SHA256
            or self.nextn_mtp_mode != "external_drafter"
            or self.target_snapshot_sha256 is not None
            or self.mtp_component_sha256 is not None
        ):
            raise ValueError("E6 external terminal identity differs")
        for label, digest in (
            ("native GPU proof", self.native_gpu_proof_sha256),
            ("distributed GPU proof", self.distributed_gpu_proof_sha256),
            ("trusted authority", self.trusted_authority_sha256),
        ):
            _sha(f"E6 terminal {label}", digest)
        for value in (
            self.plan,
            self.before_gpu_snapshot,
            self.after_gpu_snapshot,
            self.live_observation,
            self.live_native_terminal,
            self.live_native_itl,
            self.live_graph,
            self.live_worker_hook,
            *self.live_rank_terminals,
            self.live_server_receipt,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("E6 terminal JSON evidence is not path-bound")
        for value in (self.junit_xml, self.runner_log, self.live_server_log):
            if type(value) is not EvidenceFileBinding:
                raise TypeError("E6 terminal raw evidence is not path-bound")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = {
            **asdict(self),
            "plan": self.plan.to_dict(),
            "before_gpu_snapshot": self.before_gpu_snapshot.to_dict(),
            "after_gpu_snapshot": self.after_gpu_snapshot.to_dict(),
            "junit_xml": self.junit_xml.to_dict(),
            "runner_log": self.runner_log.to_dict(),
            "live_observation": self.live_observation.to_dict(),
            "live_native_terminal": self.live_native_terminal.to_dict(),
            "live_native_itl": self.live_native_itl.to_dict(),
            "live_graph": self.live_graph.to_dict(),
            "live_worker_hook": self.live_worker_hook.to_dict(),
            "live_rank_terminals": [row.to_dict() for row in self.live_rank_terminals],
            "live_server_receipt": self.live_server_receipt.to_dict(),
            "live_server_log": self.live_server_log.to_dict(),
        }
        if self.schema_version == 1:
            value.pop("nextn_mtp_mode")
            value.pop("target_snapshot_sha256")
            value.pop("mtp_component_sha256")
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("E6 interface/fit terminal must be an object")
        schema_version = value.get("schema_version")
        expected = set(cls.__dataclass_fields__)
        if schema_version == 1:
            expected -= {
                "nextn_mtp_mode",
                "target_snapshot_sha256",
                "mtp_component_sha256",
            }
        row = _strict("E6 interface/fit terminal", value, expected)
        for name in (
            "plan",
            "before_gpu_snapshot",
            "after_gpu_snapshot",
            "live_observation",
            "live_native_terminal",
            "live_native_itl",
            "live_graph",
            "live_worker_hook",
            "live_server_receipt",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        for name in ("junit_xml", "runner_log", "live_server_log"):
            row[name] = EvidenceFileBinding.from_dict(
                row[name],
                label=f"E6 terminal {name}",
            )
        raw_ranks = row.pop("live_rank_terminals")
        if type(raw_ranks) is not list:
            raise TypeError("E6 rank terminals must be an array")
        if schema_version == 1:
            row["nextn_mtp_mode"] = "external_drafter"
            row["target_snapshot_sha256"] = None
            row["mtp_component_sha256"] = None
        return cls(
            **row,
            live_rank_terminals=tuple(
                CanonicalJsonProofBinding.from_dict(item) for item in raw_ranks
            ),
        )  # type: ignore[arg-type]


def _harness_bindings(
    plan: FormalSingleOperatorE6InterfaceFitPlan,
    assignment: NativeRuntimeQualificationAssignment,
) -> tuple[
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
    CanonicalJsonProofBinding,
    tuple[CanonicalJsonProofBinding, CanonicalJsonProofBinding],
    CanonicalJsonProofBinding,
    EvidenceFileBinding,
]:
    root = Path(plan.evidence_directory)
    return (
        CanonicalJsonProofBinding.bind(root / "nextn_tp2.live-native-terminal.json"),
        CanonicalJsonProofBinding.bind(root / "nextn_tp2.live-native-itl.json"),
        CanonicalJsonProofBinding.bind(root / "nextn_tp2.live-graph.json"),
        CanonicalJsonProofBinding.bind(root / "nextn_tp2.live-worker-hook.json"),
        (
            CanonicalJsonProofBinding.bind(root / "nextn_tp2.rank-0.json"),
            CanonicalJsonProofBinding.bind(root / "nextn_tp2.rank-1.json"),
        ),
        CanonicalJsonProofBinding.bind(root / "nextn_tp2.live-server-receipt.json"),
        EvidenceFileBinding.bind(
            assignment.evidence_path("live-observation.json").with_suffix(
                ".live-server.log"
            ),
            label="E6 live server log",
        ),
    )


def _evidence_proofs(
    *,
    plan: FormalSingleOperatorE6InterfaceFitPlan,
    observation: NativeRuntimeQualificationObservation,
    junit: EvidenceFileBinding,
    live_native: CanonicalJsonProofBinding,
    live_itl: CanonicalJsonProofBinding,
    live_graph: CanonicalJsonProofBinding,
    live_worker: CanonicalJsonProofBinding,
    ranks: tuple[CanonicalJsonProofBinding, CanonicalJsonProofBinding],
    server: CanonicalJsonProofBinding,
) -> tuple[str, str, str]:
    native = content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_single_operator_nextn_native_gpu_proof",
            "plan_sha256": plan.sha256,
            "observation_sha256": observation.sha256,
            "junit_raw_sha256": junit.raw_sha256,
            "native_terminal_sha256": live_native.semantic_sha256,
            "native_itl_sha256": live_itl.semantic_sha256,
            "graph_sha256": live_graph.semantic_sha256,
            "worker_sha256": live_worker.semantic_sha256,
            "interface_sha256": plan.interface_sha256,
        }
    )
    distributed = content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_single_operator_nextn_tp2_distributed_gpu_proof",
            "plan_sha256": plan.sha256,
            "server_receipt_sha256": server.semantic_sha256,
            "rank_terminal_sha256s": [row.semantic_sha256 for row in ranks],
            "topology_sha256": plan.topology_sha256,
            "gpu_uuids": list(plan.gpu_uuids),
        }
    )
    authority_value: dict[str, object] = {
        "schema_version": 1,
        "kind": "trusted_single_operator_nextn_tp2_empirical_authority",
        "trust_mode": "no_signature_not_formal_measured",
        "model": plan.model,
        "target_member_sha256": plan.target_member_sha256,
        "drafter_member_sha256": plan.drafter_member_sha256,
        "target_shard_manifest_sha256": plan.target_shard_manifest_sha256,
        "drafter_shard_manifest_sha256": plan.drafter_shard_manifest_sha256,
        "native_gpu_proof_sha256": native,
        "distributed_gpu_proof_sha256": distributed,
    }
    if getattr(plan, "nextn_mtp_mode", "external_drafter") == "built_in_mtp":
        authority_value.update(
            {
                "nextn_mtp_mode": "built_in_mtp",
                "target_snapshot_sha256": plan.target_snapshot_sha256,
                "mtp_component_sha256": plan.mtp_component_sha256,
            }
        )
    authority = content_sha256(authority_value)
    return native, distributed, authority


def _validate_observation_evidence(
    observation: NativeRuntimeQualificationObservation,
    *,
    live_native: CanonicalJsonProofBinding,
    live_itl: CanonicalJsonProofBinding,
    live_graph: CanonicalJsonProofBinding,
    live_worker: CanonicalJsonProofBinding,
    ranks: tuple[CanonicalJsonProofBinding, CanonicalJsonProofBinding],
    server: CanonicalJsonProofBinding,
) -> None:
    if (
        observation.native_terminal_sha256 != live_native.semantic_sha256
        or observation.native_itl_pointer_sha256 != live_itl.semantic_sha256
        or observation.graph_observation_sha256 != live_graph.semantic_sha256
        or observation.worker_hook_observation_sha256 != live_worker.semantic_sha256
        or observation.rank_terminal_sha256s
        != tuple(row.semantic_sha256 for row in ranks)
        or observation.live_server_receipt_sha256 != server.semantic_sha256
    ):
        raise ValueError("E6 live NEXTN observation evidence differs")


def _execute_formal_single_operator_e6_interface_fit_plan_unlocked(
    plan_path: str | Path,
) -> FormalSingleOperatorE6InterfaceFitTerminal:
    """Execute one exact source-owned NEXTN TP2 8/8 live qualification."""

    plan = revalidate_formal_single_operator_e6_interface_fit_plan(plan_path)
    assignment = NativeRuntimeQualificationAssignment.load(
        plan.native_assignment.absolute_path
    )
    launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    before = _publish_snapshot(assignment, phase="before")
    before_value = _validate_gpu_snapshot(
        before.reopen(), assignment=assignment, phase="before"
    )
    if before_value["status"] != "AVAILABLE" or before_value["compute_process_rows"]:
        raise FormalSingleOperatorE6InterfaceFitBlocked(
            "nextn_tp2_gpu_precondition_not_clean"
        )
    log_path = assignment.evidence_path("runner.log")
    junit_path = assignment.evidence_path("junit.xml")
    observation_path = assignment.evidence_path("live-observation.json")
    test_file = (
        Path(launch.patched_sglang_checkout)
        / NATIVE_RUNTIME_GPU_TEST_FILES["nextn_tp2"]
    )
    command = (
        assignment.python_executable,
        "-m",
        "pytest",
        "-q",
        *tuple(
            f"{test_file}::{name}"
            for name in NATIVE_RUNTIME_GPU_TEST_NAMES["nextn_tp2"]
        ),
        f"--junitxml={junit_path}",
    )
    environment = launch.child_environment()
    environment.update(
        {
            "PYTHONPATH": launch.patched_sglang_checkout,
            "LIGHTCONE_NATIVE_QUALIFICATION_ASSIGNMENT_PATH": (
                plan.native_assignment.absolute_path
            ),
            "LIGHTCONE_NATIVE_QUALIFICATION_ASSIGNMENT_SHA256": assignment.sha256,
            "LIGHTCONE_NATIVE_QUALIFICATION_RUNNER_PROTOCOL_SHA256": (
                assignment.runner_protocol_sha256
            ),
            "LIGHTCONE_NATIVE_QUALIFICATION_SOURCE_CAPABILITY_SHA256": (
                _source_capability_sha256("nextn_tp2")
            ),
            "LIGHTCONE_NATIVE_QUALIFICATION_OBSERVATION_PATH": str(observation_path),
            "LIGHTCONE_COMPILE_LAUNCH_MANIFEST_PATH": (
                plan.launch_manifest.absolute_path
            ),
            "LIGHTCONE_COMPILE_LAUNCH_MANIFEST_SHA256": launch.sha256,
        }
    )
    if getattr(plan, "nextn_mtp_mode", "external_drafter") == "built_in_mtp":
        assert plan.target_snapshot_sha256 is not None
        assert plan.mtp_component_sha256 is not None
        assert plan.mtp_component is not None
        environment.update(
            {
                "LIGHTCONE_NEXTN_MTP_MODE": plan.nextn_mtp_mode,
                "LIGHTCONE_NEXTN_TARGET_SNAPSHOT_SHA256": (plan.target_snapshot_sha256),
                "LIGHTCONE_NEXTN_MTP_COMPONENT_SHA256": (plan.mtp_component_sha256),
                "LIGHTCONE_NEXTN_MTP_COMPONENT_PATH": (
                    plan.mtp_component.absolute_path
                ),
            }
        )
    started_ns = time.monotonic_ns()
    process: subprocess.Popen[bytes] | None = None
    with log_path.open("xb") as log:
        log.write(b"trusted single-operator E6 NEXTN TP2 interface/fit\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=launch.patched_sglang_checkout,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        try:
            process.wait(timeout=float(_E6_INTERFACE_FIT_SUBPROCESS_TIMEOUT_SECONDS))
        except subprocess.TimeoutExpired as error:
            _terminate_process_group(process)
            raise FormalSingleOperatorE6InterfaceFitBlocked(
                "nextn_tp2_interface_fit_timeout"
            ) from error
        finally:
            log.flush()
            os.fsync(log.fileno())
    if process.returncode != 0:
        if _process_group_exists(process.pid):
            _terminate_process_group(process)
        raise FormalSingleOperatorE6InterfaceFitBlocked(
            "nextn_tp2_interface_fit_pytest_failed"
        )
    if _process_group_exists(process.pid):
        _terminate_process_group(process)
        raise FormalSingleOperatorE6InterfaceFitBlocked(
            "nextn_tp2_interface_fit_left_process_group"
        )
    after = _publish_snapshot(assignment, phase="after")
    after_value = _validate_gpu_snapshot(
        after.reopen(), assignment=assignment, phase="after"
    )
    if after_value["status"] != "AVAILABLE" or after_value["compute_process_rows"]:
        raise FormalSingleOperatorE6InterfaceFitBlocked(
            "nextn_tp2_gpu_postcondition_not_clean"
        )
    expected_names = tuple(sorted(NATIVE_RUNTIME_GPU_TEST_NAMES["nextn_tp2"]))
    if _junit_summary(junit_path) != (expected_names, 8, 8, 0, 0, 0):
        raise FormalSingleOperatorE6InterfaceFitBlocked(
            "nextn_tp2_junit_not_exact_8_of_8"
        )
    junit = EvidenceFileBinding.bind(junit_path, label="E6 interface/fit JUnit")
    runner_log = EvidenceFileBinding.bind(log_path, label="E6 interface/fit log")
    observation_binding = CanonicalJsonProofBinding.bind(observation_path)
    observation = NativeRuntimeQualificationObservation.from_dict(
        observation_binding.reopen()
    )
    if observation.sha256 != observation_binding.semantic_sha256:
        raise ValueError("E6 interface/fit observation digest differs")
    observation.validate_assignment(assignment)
    (
        live_native,
        live_itl,
        live_graph,
        live_worker,
        ranks,
        server,
        live_server_log,
    ) = _harness_bindings(plan, assignment)
    _validate_observation_evidence(
        observation,
        live_native=live_native,
        live_itl=live_itl,
        live_graph=live_graph,
        live_worker=live_worker,
        ranks=ranks,
        server=server,
    )
    native, distributed, authority = _evidence_proofs(
        plan=plan,
        observation=observation,
        junit=junit,
        live_native=live_native,
        live_itl=live_itl,
        live_graph=live_graph,
        live_worker=live_worker,
        ranks=ranks,
        server=server,
    )
    nextn_mtp_mode = getattr(plan, "nextn_mtp_mode", "external_drafter")
    terminal = FormalSingleOperatorE6InterfaceFitTerminal(
        schema_version=2 if nextn_mtp_mode == "built_in_mtp" else 1,
        kind="formal_single_operator_e6_interface_fit_terminal",
        protocol_sha256=(
            FORMAL_SINGLE_OPERATOR_E6_BUILT_IN_MTP_INTERFACE_FIT_PROTOCOL_SHA256
            if nextn_mtp_mode == "built_in_mtp"
            else FORMAL_SINGLE_OPERATOR_E6_INTERFACE_FIT_PROTOCOL_SHA256
        ),
        plan=CanonicalJsonProofBinding.bind(plan_path, semantic_sha256=plan.sha256),
        model=plan.model,
        before_gpu_snapshot=before,
        after_gpu_snapshot=after,
        junit_xml=junit,
        runner_log=runner_log,
        live_observation=observation_binding,
        live_native_terminal=live_native,
        live_native_itl=live_itl,
        live_graph=live_graph,
        live_worker_hook=live_worker,
        live_rank_terminals=ranks,
        live_server_receipt=server,
        live_server_log=live_server_log,
        native_gpu_proof_sha256=native,
        distributed_gpu_proof_sha256=distributed,
        trusted_authority_sha256=authority,
        started_ns=started_ns,
        finished_ns=max(time.monotonic_ns(), started_ns + 1),
        status="COMPLETE",
        physical_execution_count=1,
        nextn_mtp_mode=nextn_mtp_mode,
        target_snapshot_sha256=getattr(plan, "target_snapshot_sha256", None),
        mtp_component_sha256=getattr(plan, "mtp_component_sha256", None),
    )
    output = Path(plan.evidence_directory) / "e6-interface-fit-terminal.json"
    publish_canonical_json_no_replace(output, terminal.to_dict())
    return revalidate_formal_single_operator_e6_interface_fit_terminal(output)


def execute_formal_single_operator_e6_interface_fit_plan(
    plan_path: str | Path,
) -> FormalSingleOperatorE6InterfaceFitTerminal:
    """Run one plan under the campaign-wide exclusive two-GPU gang lock."""

    plan = revalidate_formal_single_operator_e6_interface_fit_plan(plan_path)
    lock_path = Path(plan.evidence_directory).parent / ".e6-nextn-tp2-gang.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FormalSingleOperatorE6InterfaceFitBlocked(
                "nextn_tp2_campaign_gang_busy"
            ) from error
        try:
            return _execute_formal_single_operator_e6_interface_fit_plan_unlocked(
                plan_path
            )
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def revalidate_formal_single_operator_e6_interface_fit_terminal(
    path: str | Path,
) -> FormalSingleOperatorE6InterfaceFitTerminal:
    binding = CanonicalJsonProofBinding.bind(path)
    terminal = FormalSingleOperatorE6InterfaceFitTerminal.from_dict(binding.reopen())
    if terminal.sha256 != binding.semantic_sha256:
        raise ValueError("E6 interface/fit terminal binding differs")
    plan = revalidate_formal_single_operator_e6_interface_fit_plan(
        terminal.plan.absolute_path
    )
    assignment = NativeRuntimeQualificationAssignment.load(
        plan.native_assignment.absolute_path
    )
    before = _validate_gpu_snapshot(
        terminal.before_gpu_snapshot.reopen(), assignment=assignment, phase="before"
    )
    after = _validate_gpu_snapshot(
        terminal.after_gpu_snapshot.reopen(), assignment=assignment, phase="after"
    )
    terminal.junit_xml.reopen(label="E6 interface/fit JUnit")
    terminal.runner_log.reopen(label="E6 interface/fit runner log")
    terminal.live_server_log.reopen(label="E6 interface/fit server log")
    expected_names = tuple(sorted(NATIVE_RUNTIME_GPU_TEST_NAMES["nextn_tp2"]))
    observation = NativeRuntimeQualificationObservation.from_dict(
        terminal.live_observation.reopen()
    )
    observation.validate_assignment(assignment)
    expected_harness = _harness_bindings(plan, assignment)
    _validate_observation_evidence(
        observation,
        live_native=terminal.live_native_terminal,
        live_itl=terminal.live_native_itl,
        live_graph=terminal.live_graph,
        live_worker=terminal.live_worker_hook,
        ranks=terminal.live_rank_terminals,
        server=terminal.live_server_receipt,
    )
    native, distributed, authority = _evidence_proofs(
        plan=plan,
        observation=observation,
        junit=terminal.junit_xml,
        live_native=terminal.live_native_terminal,
        live_itl=terminal.live_native_itl,
        live_graph=terminal.live_graph,
        live_worker=terminal.live_worker_hook,
        ranks=terminal.live_rank_terminals,
        server=terminal.live_server_receipt,
    )
    plan_mode = getattr(plan, "nextn_mtp_mode", "external_drafter")
    if (
        plan.model != terminal.model
        or (2 if plan_mode == "built_in_mtp" else 1) != terminal.schema_version
        or (
            FORMAL_SINGLE_OPERATOR_E6_BUILT_IN_MTP_INTERFACE_FIT_PROTOCOL_SHA256
            if plan_mode == "built_in_mtp"
            else FORMAL_SINGLE_OPERATOR_E6_INTERFACE_FIT_PROTOCOL_SHA256
        )
        != terminal.protocol_sha256
        or plan_mode != terminal.nextn_mtp_mode
        or getattr(plan, "target_snapshot_sha256", None)
        != terminal.target_snapshot_sha256
        or getattr(plan, "mtp_component_sha256", None) != terminal.mtp_component_sha256
        or Path(binding.absolute_path).parent != Path(plan.evidence_directory)
        or before["status"] != "AVAILABLE"
        or before["compute_process_rows"]
        or after["status"] != "AVAILABLE"
        or after["compute_process_rows"]
        or _junit_summary(Path(terminal.junit_xml.absolute_path))
        != (expected_names, 8, 8, 0, 0, 0)
        or observation.sha256 != terminal.live_observation.semantic_sha256
        or (
            terminal.live_native_terminal,
            terminal.live_native_itl,
            terminal.live_graph,
            terminal.live_worker_hook,
            terminal.live_rank_terminals,
            terminal.live_server_receipt,
            terminal.live_server_log,
        )
        != expected_harness
        or terminal.native_gpu_proof_sha256 != native
        or terminal.distributed_gpu_proof_sha256 != distributed
        or terminal.trusted_authority_sha256 != authority
    ):
        raise ValueError("E6 interface/fit terminal replay differs")
    return terminal


@dataclass(frozen=True)
class FormalSingleOperatorE6CompatibilityRow:
    model: str
    source_input_sha256: str
    dynamic_artifact_sha256: str
    verified_authority_sha256: str
    interface_sha256: str
    target_member_id: str
    drafter_member_id: str
    target_model_id: str
    drafter_model_id: str
    target_revision: str
    drafter_revision: str
    target_shard_manifest_sha256: str
    drafter_shard_manifest_sha256: str
    topology_sha256: str
    source_adapter_version: int
    native_gpu_proof_sha256: str
    distributed_gpu_proof_sha256: str
    content_verification_receipt_sha256: str
    inventory_sha256: str
    gpu_uuids: tuple[str, str]
    terminal_sha256: str
    nextn_mtp_mode: Literal["external_drafter", "built_in_mtp"] = "external_drafter"
    target_snapshot_sha256: str | None = None
    mtp_component_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            self.model not in E6_MODELS
            or self.target_model_id != self.model
            or self.source_adapter_version != 0
            or len(self.gpu_uuids) != 2
            or len(set(self.gpu_uuids)) != 2
        ):
            raise ValueError("trusted E6 compatibility row identity differs")
        if self.nextn_mtp_mode == "built_in_mtp":
            if (
                self.target_model_id != self.drafter_model_id
                or self.target_revision != self.drafter_revision
                or self.target_member_id != self.drafter_member_id
                or self.target_shard_manifest_sha256
                != self.drafter_shard_manifest_sha256
                or self.interface_sha256 != NEXTN_BUILT_IN_MTP_INTERFACE_SHA256
                or self.target_snapshot_sha256 is None
                or self.mtp_component_sha256 is None
                or self.target_snapshot_sha256 == self.mtp_component_sha256
            ):
                raise ValueError("trusted E6 built-in MTP row identity differs")
        elif (
            self.nextn_mtp_mode != "external_drafter"
            or self.target_model_id == self.drafter_model_id
            or self.interface_sha256 != NEXTN_MTP_INTERFACE_SHA256
            or self.target_snapshot_sha256 is not None
            or self.mtp_component_sha256 is not None
        ):
            raise ValueError("trusted E6 external row identity differs")
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith("sha256") and value is not None:
                _sha(f"trusted E6 row {field.name}", value)
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = {**asdict(self), "gpu_uuids": list(self.gpu_uuids)}
        if self.nextn_mtp_mode == "external_drafter":
            value.pop("nextn_mtp_mode")
            value.pop("target_snapshot_sha256")
            value.pop("mtp_component_sha256")
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("trusted E6 compatibility row must be an object")
        expected = set(cls.__dataclass_fields__)
        external = "nextn_mtp_mode" not in value
        if external:
            expected -= {
                "nextn_mtp_mode",
                "target_snapshot_sha256",
                "mtp_component_sha256",
            }
        row = _strict("trusted E6 compatibility row", value, expected)
        raw_gpus = row.pop("gpu_uuids")
        if type(raw_gpus) is not list:
            raise TypeError("trusted E6 compatibility GPUs must be an array")
        if external:
            row["nextn_mtp_mode"] = "external_drafter"
            row["target_snapshot_sha256"] = None
            row["mtp_component_sha256"] = None
        return cls(**row, gpu_uuids=tuple(raw_gpus))  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorE6CompatibilityReceipt:
    schema_version: Literal[1]
    protocol_lock_sha256: str
    registry_sha256: str
    trusted_content_bundle_sha256: str
    protocol_sha256: str
    inventory_sha256: str
    models: tuple[
        FormalSingleOperatorE6CompatibilityRow,
        FormalSingleOperatorE6CompatibilityRow,
    ]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.protocol_sha256 != E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256
            or tuple(row.model for row in self.models) != E6_MODELS
            or len({row.terminal_sha256 for row in self.models}) != 2
            or len({row.inventory_sha256 for row in self.models}) != 1
            or {row.inventory_sha256 for row in self.models} != {self.inventory_sha256}
            or len({row.content_verification_receipt_sha256 for row in self.models})
            != 1
            or {row.content_verification_receipt_sha256 for row in self.models}
            != {self.trusted_content_bundle_sha256}
            or len({row.nextn_mtp_mode for row in self.models}) != 1
        ):
            raise ValueError("trusted E6 compatibility coverage differs")
        for label, digest in (
            ("ProtocolLock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("trusted content", self.trusted_content_bundle_sha256),
            ("protocol", self.protocol_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _sha(f"trusted E6 receipt {label}", digest)
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "registry_sha256": self.registry_sha256,
            "trusted_content_bundle_sha256": self.trusted_content_bundle_sha256,
            "protocol_sha256": self.protocol_sha256,
            "inventory_sha256": self.inventory_sha256,
            "models": [row.to_dict() for row in self.models],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "trusted E6 compatibility receipt",
            value,
            set(cls.__dataclass_fields__),
        )
        raw_models = row.pop("models")
        if type(raw_models) is not list:
            raise TypeError("trusted E6 compatibility models must be an array")
        return cls(
            **row,
            models=tuple(
                FormalSingleOperatorE6CompatibilityRow.from_dict(item)
                for item in raw_models
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalSingleOperatorE6InterfaceFitBundle:
    schema_version: Literal[2, 3]
    kind: Literal["formal_single_operator_e6_interface_fit_bundle"]
    trust_mode: Literal["trusted_single_operator_empirical_no_signature"]
    protocol_sha256: str
    campaign: CanonicalJsonProofBinding
    protocol_lock_sha256: str
    expected_inventory_sha256: str
    verified_ns: int
    models: tuple[str, str]
    terminals: tuple[CanonicalJsonProofBinding, CanonicalJsonProofBinding]
    compatibility: FormalSingleOperatorE6CompatibilityReceipt
    compatibility_sha256: str
    physical_execution_count: Literal[2]
    reuse_scope: Literal["e6_pilot_and_e6_final"]
    nextn_mtp_mode: Literal["external_drafter", "built_in_mtp"] = "external_drafter"
    target_snapshot_sha256s: tuple[str, str] | None = None
    mtp_component_sha256s: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {2, 3}
            or self.kind != "formal_single_operator_e6_interface_fit_bundle"
            or self.trust_mode != "trusted_single_operator_empirical_no_signature"
            or self.models != E6_MODELS
            or len(self.terminals) != 2
            or len(set(self.terminals)) != 2
            or self.compatibility.sha256 != self.compatibility_sha256
            or self.compatibility.protocol_lock_sha256 != self.protocol_lock_sha256
            or self.compatibility.inventory_sha256 != self.expected_inventory_sha256
            or self.physical_execution_count != 2
            or self.reuse_scope != "e6_pilot_and_e6_final"
            or self.verified_ns < 1
        ):
            raise ValueError("trusted E6 interface/fit bundle identity differs")
        if self.schema_version == 3:
            if (
                self.protocol_sha256
                != FORMAL_SINGLE_OPERATOR_E6_BUILT_IN_MTP_INTERFACE_FIT_PROTOCOL_SHA256
                or self.nextn_mtp_mode != "built_in_mtp"
                or self.target_snapshot_sha256s is None
                or self.mtp_component_sha256s is None
                or tuple(
                    row.target_snapshot_sha256 for row in self.compatibility.models
                )
                != self.target_snapshot_sha256s
                or tuple(row.mtp_component_sha256 for row in self.compatibility.models)
                != self.mtp_component_sha256s
            ):
                raise ValueError("trusted E6 built-in MTP bundle identity differs")
        elif (
            self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_E6_INTERFACE_FIT_PROTOCOL_SHA256
            or self.nextn_mtp_mode != "external_drafter"
            or self.target_snapshot_sha256s is not None
            or self.mtp_component_sha256s is not None
        ):
            raise ValueError("trusted E6 external bundle identity differs")
        for digest in (
            self.protocol_lock_sha256,
            self.expected_inventory_sha256,
            self.compatibility_sha256,
        ):
            _sha("trusted E6 bundle digest", digest)
        if type(self.campaign) is not CanonicalJsonProofBinding or any(
            type(row) is not CanonicalJsonProofBinding for row in self.terminals
        ):
            raise TypeError("trusted E6 bundle input is not path-bound")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "trust_mode": self.trust_mode,
            "protocol_sha256": self.protocol_sha256,
            "campaign": self.campaign.to_dict(),
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "expected_inventory_sha256": self.expected_inventory_sha256,
            "verified_ns": self.verified_ns,
            "models": list(self.models),
            "terminals": [row.to_dict() for row in self.terminals],
            "compatibility": self.compatibility.to_dict(),
            "compatibility_sha256": self.compatibility_sha256,
            "physical_execution_count": self.physical_execution_count,
            "reuse_scope": self.reuse_scope,
            "nextn_mtp_mode": self.nextn_mtp_mode,
            "target_snapshot_sha256s": self.target_snapshot_sha256s,
            "mtp_component_sha256s": self.mtp_component_sha256s,
        }
        if self.schema_version == 2:
            value.pop("nextn_mtp_mode")
            value.pop("target_snapshot_sha256s")
            value.pop("mtp_component_sha256s")
        else:
            assert self.target_snapshot_sha256s is not None
            assert self.mtp_component_sha256s is not None
            value["target_snapshot_sha256s"] = list(self.target_snapshot_sha256s)
            value["mtp_component_sha256s"] = list(self.mtp_component_sha256s)
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("trusted E6 interface/fit bundle must be an object")
        schema_version = value.get("schema_version")
        expected = set(cls.__dataclass_fields__)
        if schema_version == 2:
            expected -= {
                "nextn_mtp_mode",
                "target_snapshot_sha256s",
                "mtp_component_sha256s",
            }
        row = _strict("trusted E6 interface/fit bundle", value, expected)
        raw_models = row.pop("models")
        raw_terminals = row.pop("terminals")
        raw_campaign = row.pop("campaign")
        raw_compatibility = row.pop("compatibility")
        if type(raw_models) is not list or type(raw_terminals) is not list:
            raise TypeError("trusted E6 interface/fit arrays differ")
        if schema_version == 2:
            row["nextn_mtp_mode"] = "external_drafter"
            row["target_snapshot_sha256s"] = None
            row["mtp_component_sha256s"] = None
        else:
            for name in ("target_snapshot_sha256s", "mtp_component_sha256s"):
                if type(row[name]) is not list:
                    raise TypeError(f"trusted E6 bundle {name} must be an array")
                row[name] = tuple(row[name])
        return cls(
            **row,
            campaign=CanonicalJsonProofBinding.from_dict(raw_campaign),
            models=tuple(raw_models),
            terminals=tuple(
                CanonicalJsonProofBinding.from_dict(item) for item in raw_terminals
            ),
            compatibility=FormalSingleOperatorE6CompatibilityReceipt.from_dict(
                raw_compatibility
            ),
        )  # type: ignore[arg-type]


def compatibility_row_for_terminal(
    plan: FormalSingleOperatorE6InterfaceFitPlan,
    terminal: FormalSingleOperatorE6InterfaceFitTerminal,
) -> FormalSingleOperatorE6CompatibilityRow:
    """Project one deeply replayed physical terminal into its E6 source row."""

    if terminal.plan.semantic_sha256 != plan.sha256 or terminal.model != plan.model:
        raise ValueError("E6 interface/fit plan and terminal differ")
    bundle = plan.content_source.reopen()
    assert type(bundle) is TrustedSingleOperatorContentBundle
    launch = CompileLaunchManifest.load(plan.launch_manifest.absolute_path)
    assert launch.drafter_model_id is not None
    assert launch.drafter_revision is not None
    return FormalSingleOperatorE6CompatibilityRow(
        model=plan.model,
        source_input_sha256=plan.sha256,
        dynamic_artifact_sha256=terminal.sha256,
        verified_authority_sha256=terminal.trusted_authority_sha256,
        interface_sha256=plan.interface_sha256,
        target_member_id=plan.target_member_sha256,
        drafter_member_id=plan.drafter_member_sha256,
        target_model_id=launch.target_model_id,
        drafter_model_id=launch.drafter_model_id,
        target_revision=launch.target_revision,
        drafter_revision=launch.drafter_revision,
        target_shard_manifest_sha256=plan.target_shard_manifest_sha256,
        drafter_shard_manifest_sha256=plan.drafter_shard_manifest_sha256,
        topology_sha256=plan.topology_sha256,
        source_adapter_version=plan.source_adapter_version,
        native_gpu_proof_sha256=terminal.native_gpu_proof_sha256,
        distributed_gpu_proof_sha256=terminal.distributed_gpu_proof_sha256,
        content_verification_receipt_sha256=plan.content_source.content_sha256,
        inventory_sha256=GpuInventory.from_dict(plan.inventory.reopen()).sha256,
        gpu_uuids=plan.gpu_uuids,
        terminal_sha256=terminal.sha256,
        nextn_mtp_mode=plan.nextn_mtp_mode,
        target_snapshot_sha256=plan.target_snapshot_sha256,
        mtp_component_sha256=plan.mtp_component_sha256,
    )


def finalize_formal_single_operator_e6_interface_fit_bundle(
    *,
    campaign_path: str | Path,
    output_path: str | Path,
) -> FormalSingleOperatorE6InterfaceFitBundle:
    """Publish the exact two-model auxiliary after exactly two real runs."""

    campaign_binding = CanonicalJsonProofBinding.bind(campaign_path)
    campaign = _load_campaign(campaign_path)
    plans = tuple(
        revalidate_formal_single_operator_e6_interface_fit_plan(row.absolute_path)
        for row in campaign.plans
    )
    terminal_bindings = tuple(
        CanonicalJsonProofBinding.bind(
            Path(plan.evidence_directory) / "e6-interface-fit-terminal.json"
        )
        for plan in plans
    )
    terminals = tuple(
        revalidate_formal_single_operator_e6_interface_fit_terminal(row.absolute_path)
        for row in terminal_bindings
    )
    if (
        tuple(row.model for row in terminals) != E6_MODELS
        or any(
            terminal.plan != plan_binding
            for terminal, plan_binding in zip(terminals, campaign.plans, strict=True)
        )
        or sum(row.physical_execution_count for row in terminals) != 2
    ):
        raise ValueError("E6 interface/fit physical execution coverage differs")
    lock = protocol_lock_from_dict(
        plans[0].protocol_lock.reopen(label="E6 interface/fit ProtocolLock")
    )
    compatibility = FormalSingleOperatorE6CompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=lock.sha256,
        registry_sha256=lock.registry_sha256,
        trusted_content_bundle_sha256=plans[0].content_source.content_sha256,
        protocol_sha256=E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256,
        inventory_sha256=campaign.inventory_sha256,
        models=tuple(
            compatibility_row_for_terminal(plan, terminal)
            for plan, terminal in zip(plans, terminals, strict=True)
        ),  # type: ignore[arg-type]
    )
    built_in = campaign.schema_version == 2
    bundle = FormalSingleOperatorE6InterfaceFitBundle(
        schema_version=3 if built_in else 2,
        kind="formal_single_operator_e6_interface_fit_bundle",
        trust_mode="trusted_single_operator_empirical_no_signature",
        protocol_sha256=campaign.protocol_sha256,
        campaign=campaign_binding,
        protocol_lock_sha256=lock.sha256,
        expected_inventory_sha256=campaign.inventory_sha256,
        verified_ns=max(row.finished_ns for row in terminals),
        models=E6_MODELS,
        terminals=terminal_bindings,  # type: ignore[arg-type]
        compatibility=compatibility,
        compatibility_sha256=compatibility.sha256,
        physical_execution_count=2,
        reuse_scope="e6_pilot_and_e6_final",
        nextn_mtp_mode=campaign.nextn_mtp_mode,
        target_snapshot_sha256s=campaign.target_snapshot_sha256s,
        mtp_component_sha256s=campaign.mtp_component_sha256s,
    )
    publish_formal_single_operator_json_artifact(output_path, bundle.to_dict())
    return revalidate_formal_single_operator_e6_interface_fit_bundle(output_path)


def revalidate_formal_single_operator_e6_interface_fit_bundle_value(
    value: object,
    *,
    protocol_lock: ProtocolLock,
) -> FormalSingleOperatorE6InterfaceFitBundle:
    bundle = FormalSingleOperatorE6InterfaceFitBundle.from_dict(value)
    campaign = _load_campaign(bundle.campaign.absolute_path)
    terminals = tuple(
        revalidate_formal_single_operator_e6_interface_fit_terminal(row.absolute_path)
        for row in bundle.terminals
    )
    plans = tuple(
        revalidate_formal_single_operator_e6_interface_fit_plan(
            terminal.plan.absolute_path
        )
        for terminal in terminals
    )
    expected = FormalSingleOperatorE6CompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        trusted_content_bundle_sha256=plans[0].content_source.content_sha256,
        protocol_sha256=E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256,
        inventory_sha256=campaign.inventory_sha256,
        models=tuple(
            compatibility_row_for_terminal(plan, terminal)
            for plan, terminal in zip(plans, terminals, strict=True)
        ),  # type: ignore[arg-type]
    )
    if (
        protocol_lock.schema_version != 5
        or bundle.protocol_lock_sha256 != protocol_lock.sha256
        or campaign.protocol_lock_sha256 != protocol_lock.sha256
        or tuple(row.model for row in terminals) != E6_MODELS
        or tuple(row.plan for row in terminals) != campaign.plans
        or bundle.compatibility != expected
        or bundle.compatibility_sha256 != expected.sha256
        or bundle.expected_inventory_sha256 != campaign.inventory_sha256
        or bundle.physical_execution_count != 2
        or bundle.nextn_mtp_mode != campaign.nextn_mtp_mode
        or bundle.target_snapshot_sha256s != campaign.target_snapshot_sha256s
        or bundle.mtp_component_sha256s != campaign.mtp_component_sha256s
    ):
        raise ValueError("trusted E6 interface/fit bundle replay differs")
    return bundle


def revalidate_formal_single_operator_e6_interface_fit_bundle(
    path: str | Path,
) -> FormalSingleOperatorE6InterfaceFitBundle:
    binding = FormalSingleOperatorJsonBinding.bind(
        path,
        label="trusted E6 interface/fit bundle",
    )
    value = binding.reopen(label="trusted E6 interface/fit bundle")
    preliminary = FormalSingleOperatorE6InterfaceFitBundle.from_dict(value)
    campaign = _load_campaign(preliminary.campaign.absolute_path)
    plan = revalidate_formal_single_operator_e6_interface_fit_plan(
        campaign.plans[0].absolute_path
    )
    lock = protocol_lock_from_dict(
        plan.protocol_lock.reopen(label="E6 interface/fit ProtocolLock")
    )
    bundle = revalidate_formal_single_operator_e6_interface_fit_bundle_value(
        value,
        protocol_lock=lock,
    )
    if binding.semantic_sha256 != content_sha256(bundle.to_dict()):
        raise ValueError("trusted E6 interface/fit bundle binding differs")
    return bundle


def terminal_for_model(
    bundle: FormalSingleOperatorE6InterfaceFitBundle,
    model: str,
) -> CanonicalJsonProofBinding:
    if model not in E6_MODELS:
        raise ValueError("E6 terminal model is outside the exact panel")
    matches = tuple(
        binding
        for binding in bundle.terminals
        if revalidate_formal_single_operator_e6_interface_fit_terminal(
            binding.absolute_path
        ).model
        == model
    )
    if len(matches) != 1:
        raise ValueError("E6 interface/fit terminal model coverage differs")
    return matches[0]


@dataclass(frozen=True)
class FormalSingleOperatorTrustedNextnTp2ServingAuthority:
    """Unsigned empirical NEXTN/TP2 authority for one current E6 serving cell.

    This is deliberately not a ``VerifiedNextNTp2Authority``.  It binds the
    current cell to the source-owned two-model auxiliary and its exact 8/8 GPU
    terminal while explicitly withholding the repository's formal-MEASURED
    authorization claim.
    """

    schema_version: Literal[1]
    kind: Literal["formal_single_operator_e6_trusted_serving_authority"]
    protocol_sha256: str
    trust_mode: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured_authorization: Literal[False]
    execution_source: CanonicalJsonProofBinding
    execution_source_sha256: str
    materialized_cell_id: str
    node: Literal["e6_pilot", "e6_final"]
    auxiliary_bundle: FormalSingleOperatorJsonBinding
    auxiliary_bundle_sha256: str
    protocol_lock_sha256: str
    compatibility_receipt_sha256: str
    compatibility_row_sha256: str
    interface_fit_plan: CanonicalJsonProofBinding
    interface_fit_plan_sha256: str
    interface_fit_terminal: CanonicalJsonProofBinding
    interface_fit_terminal_sha256: str
    content_source: FormalContentSourceBinding
    trusted_content_bundle_sha256: str
    inventory: CanonicalJsonProofBinding
    inventory_sha256: str
    doctor: CanonicalJsonProofBinding
    doctor_sha256: str
    model: str
    target_model_id: str
    drafter_model_id: str
    target_revision: str
    drafter_revision: str
    target_member_sha256: str
    drafter_member_sha256: str
    target_shard_manifest_sha256: str
    drafter_shard_manifest_sha256: str
    interface_sha256: str
    topology_sha256: str
    source_adapter_version: Literal[0]
    gpu_uuids: tuple[str, str]
    native_gpu_proof_sha256: str
    distributed_gpu_proof_sha256: str
    empirical_authority_sha256: str
    junit_raw_sha256: str
    qualified_test_count: Literal[8]
    passed_test_count: Literal[8]
    failed_test_count: Literal[0]
    error_test_count: Literal[0]
    skipped_test_count: Literal[0]
    physical_execution_count: Literal[1]
    nextn_mtp_mode: Literal["external_drafter", "built_in_mtp"] = "external_drafter"
    target_snapshot_sha256: str | None = None
    mtp_component_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_e6_trusted_serving_authority"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_E6_TRUSTED_SERVING_AUTHORITY_PROTOCOL_SHA256
            or self.trust_mode != "trusted_single_operator_empirical_no_signature"
            or self.formal_measured_authorization is not False
            or self.node not in {"e6_pilot", "e6_final"}
            or self.model not in E6_MODELS
            or self.target_model_id != self.model
            or self.source_adapter_version != 0
            or len(self.gpu_uuids) != 2
            or len(set(self.gpu_uuids)) != 2
            or (
                self.qualified_test_count,
                self.passed_test_count,
                self.failed_test_count,
                self.error_test_count,
                self.skipped_test_count,
                self.physical_execution_count,
            )
            != (8, 8, 0, 0, 0, 1)
        ):
            raise ValueError("trusted E6 serving authority identity differs")
        if self.nextn_mtp_mode == "built_in_mtp":
            if (
                self.target_model_id != self.drafter_model_id
                or self.target_revision != self.drafter_revision
                or self.target_member_sha256 != self.drafter_member_sha256
                or self.target_shard_manifest_sha256
                != self.drafter_shard_manifest_sha256
                or self.interface_sha256 != NEXTN_BUILT_IN_MTP_INTERFACE_SHA256
                or self.target_snapshot_sha256 is None
                or self.mtp_component_sha256 is None
            ):
                raise ValueError("trusted E6 built-in MTP serving authority differs")
        elif (
            self.nextn_mtp_mode != "external_drafter"
            or self.target_model_id == self.drafter_model_id
            or self.interface_sha256 != NEXTN_MTP_INTERFACE_SHA256
            or self.target_snapshot_sha256 is not None
            or self.mtp_component_sha256 is not None
        ):
            raise ValueError("trusted E6 external serving authority differs")
        for field in fields(self):
            if field.name.endswith("sha256") and getattr(self, field.name) is not None:
                _sha(
                    f"trusted E6 serving authority {field.name}",
                    getattr(self, field.name),
                )
        if type(self.execution_source) is not CanonicalJsonProofBinding:
            raise TypeError("trusted E6 authority execution source is not path-bound")
        if type(self.auxiliary_bundle) is not FormalSingleOperatorJsonBinding:
            raise TypeError("trusted E6 authority auxiliary is not path-bound")
        for binding in (
            self.interface_fit_plan,
            self.interface_fit_terminal,
            self.inventory,
            self.doctor,
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError("trusted E6 authority evidence is not path-bound")
            if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
                raise ValueError("trusted E6 authority evidence changed")
        if (
            type(self.content_source) is not FormalContentSourceBinding
            or self.content_source.mode != "trusted_single_operator"
        ):
            raise TypeError("trusted E6 authority content source differs")
        if (
            self.content_source.content_sha256 != self.trusted_content_bundle_sha256
            or self.auxiliary_bundle.semantic_sha256 != self.auxiliary_bundle_sha256
            or self.interface_fit_plan.semantic_sha256 != self.interface_fit_plan_sha256
            or self.interface_fit_terminal.semantic_sha256
            != self.interface_fit_terminal_sha256
            or self.inventory.semantic_sha256 != self.inventory_sha256
            or self.doctor.semantic_sha256 != self.doctor_sha256
        ):
            raise ValueError("trusted E6 authority evidence identity differs")
        self.content_source.reopen()
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["execution_source"] = self.execution_source.to_dict()
        value["auxiliary_bundle"] = self.auxiliary_bundle.to_dict()
        value["interface_fit_plan"] = self.interface_fit_plan.to_dict()
        value["interface_fit_terminal"] = self.interface_fit_terminal.to_dict()
        value["content_source"] = self.content_source.to_dict()
        value["inventory"] = self.inventory.to_dict()
        value["doctor"] = self.doctor.to_dict()
        value["gpu_uuids"] = list(self.gpu_uuids)
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "trusted E6 serving authority",
            value,
            set(cls.__dataclass_fields__),
        )
        row["execution_source"] = CanonicalJsonProofBinding.from_dict(
            row["execution_source"]
        )
        row["auxiliary_bundle"] = FormalSingleOperatorJsonBinding.from_dict(
            row["auxiliary_bundle"]
        )
        for name in (
            "interface_fit_plan",
            "interface_fit_terminal",
            "inventory",
            "doctor",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        row["content_source"] = FormalContentSourceBinding.from_dict(
            row["content_source"]
        )
        raw_gpus = row.pop("gpu_uuids")
        if type(raw_gpus) is not list:
            raise TypeError("trusted E6 serving authority GPUs must be an array")
        return cls(**row, gpu_uuids=tuple(raw_gpus))  # type: ignore[arg-type]


def derive_formal_single_operator_trusted_nextn_tp2_serving_authority(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    compile_launch_manifest: CanonicalJsonProofBinding,
    inventory: CanonicalJsonProofBinding,
    content_source: FormalContentSourceBinding,
) -> FormalSingleOperatorTrustedNextnTp2ServingAuthority:
    """Deep-project one current E6 serving cell's empirical TP2 authority."""

    from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
        route_formal_single_operator_cell,
    )

    source_binding = CanonicalJsonProofBinding.bind(execution_source_path)
    source, cell, route = route_formal_single_operator_cell(
        execution_source_path=source_binding.absolute_path,
        materialized_cell_id=materialized_cell_id,
    )
    if (
        source.node not in {"e6_pilot", "e6_final"}
        or source.schema_version != 3
        or source.stage != "E6"
        or cell.stage != "E6"
        or cell.task not in {"LiveCodeBench", "MATH-500"}
        or route.physical_kind != "serving"
        or type(compile_launch_manifest) is not CanonicalJsonProofBinding
        or type(inventory) is not CanonicalJsonProofBinding
        or type(content_source) is not FormalContentSourceBinding
        or content_source.mode != "trusted_single_operator"
        or source.content_source_binding != content_source
    ):
        raise ValueError("trusted E6 serving authority source differs")
    lock = protocol_lock_from_dict(
        source.protocol_lock_source.reopen(
            label="trusted E6 serving authority ProtocolLock"
        )
    )
    auxiliary = source.auxiliary_source_binding("e6_interface_fit")
    bundle = revalidate_formal_single_operator_e6_interface_fit_bundle_value(
        auxiliary.reopen(label="trusted E6 serving interface/fit auxiliary"),
        protocol_lock=lock,
    )
    rows = tuple(row for row in bundle.compatibility.models if row.model == cell.model)
    if len(rows) != 1:
        raise ValueError("trusted E6 serving model is absent from auxiliary")
    row = rows[0]
    terminal_binding = terminal_for_model(bundle, cell.model)
    terminal = revalidate_formal_single_operator_e6_interface_fit_terminal(
        terminal_binding.absolute_path
    )
    plan = revalidate_formal_single_operator_e6_interface_fit_plan(
        terminal.plan.absolute_path
    )
    projected_row = compatibility_row_for_terminal(plan, terminal)
    launch = CompileLaunchManifest.load(compile_launch_manifest.absolute_path)
    config = load_run_config(launch.run_config_path)
    runtime_inventory = GpuInventory.from_dict(inventory.reopen())
    trusted_content = content_source.reopen()
    if (
        type(trusted_content) is not TrustedSingleOperatorContentBundle
        or trusted_content.runtime_observations is None
    ):
        raise TypeError("trusted E6 serving content source has the wrong type")
    dimensions = dict(cell.dimensions)
    expected_dimensions: dict[str, object] = {
        "content_verification_receipt_sha256": row.content_verification_receipt_sha256,
        "distributed_gpu_proof_sha256": row.distributed_gpu_proof_sha256,
        "drafter_member_id": row.drafter_member_id,
        "drafter_model_id": row.drafter_model_id,
        "drafter_revision": row.drafter_revision,
        "drafter_shard_manifest_sha256": row.drafter_shard_manifest_sha256,
        "e6_model_compatibility_row_sha256": row.sha256,
        "e6_verified_authority_sha256": row.verified_authority_sha256,
        "gpu_uuid_order_sha256": content_sha256(row.gpu_uuids),
        "interface_sha256": row.interface_sha256,
        "inventory_sha256": row.inventory_sha256,
        "native_gpu_proof_sha256": row.native_gpu_proof_sha256,
        "signed_e6_model_compatibility_sha256": auxiliary.semantic_sha256,
        "source_adapter_version": row.source_adapter_version,
        "target_member_id": row.target_member_id,
        "target_model_id": row.target_model_id,
        "target_revision": row.target_revision,
        "target_shard_manifest_sha256": row.target_shard_manifest_sha256,
        "topology": "tp2_dp1",
        "topology_authority_sha256": row.topology_sha256,
    }
    if row.nextn_mtp_mode == "built_in_mtp":
        expected_dimensions.update(
            {
                "nextn_mtp_mode": row.nextn_mtp_mode,
                "target_snapshot_sha256": row.target_snapshot_sha256,
                "mtp_component_sha256": row.mtp_component_sha256,
            }
        )
    if any(
        dimensions.get(name) != expected
        for name, expected in expected_dimensions.items()
    ):
        raise ValueError("trusted E6 serving cell differs from interface authority")
    if (
        lock.schema_version != 5
        or lock.content_source_mode != "trusted_single_operator"
        or auxiliary.semantic_sha256 != bundle.sha256
        or projected_row != row
        or terminal_binding.semantic_sha256 != terminal.sha256
        or terminal.sha256 != row.terminal_sha256
        or terminal.plan.semantic_sha256 != plan.sha256
        or terminal.trusted_authority_sha256 != row.verified_authority_sha256
        or terminal.native_gpu_proof_sha256 != row.native_gpu_proof_sha256
        or terminal.distributed_gpu_proof_sha256 != row.distributed_gpu_proof_sha256
        or terminal.status != "COMPLETE"
        or terminal.physical_execution_count != 1
        or plan.model != cell.model
        or plan.content_source != content_source
        or content_source.content_sha256 != row.content_verification_receipt_sha256
        or plan.inventory != inventory
        or plan.inventory.semantic_sha256 != runtime_inventory.sha256
        or (
            trusted_content.runtime_observations.inventory.absolute_path,
            trusted_content.runtime_observations.inventory.raw_sha256,
            trusted_content.runtime_observations.inventory.semantic_sha256,
            trusted_content.runtime_observations.inventory.size,
        )
        != (
            inventory.absolute_path,
            inventory.raw_sha256,
            inventory.semantic_sha256,
            inventory.size,
        )
        or plan.doctor.semantic_sha256
        != trusted_content.runtime_observations.doctor.semantic_sha256
        or (
            trusted_content.runtime_observations.doctor.absolute_path,
            trusted_content.runtime_observations.doctor.raw_sha256,
            trusted_content.runtime_observations.doctor.size,
        )
        != (
            plan.doctor.absolute_path,
            plan.doctor.raw_sha256,
            plan.doctor.size,
        )
        or inventory.semantic_sha256 != runtime_inventory.sha256
        or runtime_inventory.sha256 != row.inventory_sha256
        or bundle.expected_inventory_sha256 != runtime_inventory.sha256
        or compile_launch_manifest.semantic_sha256 != launch.sha256
        or launch.schema_version != (3 if row.nextn_mtp_mode == "built_in_mtp" else 2)
        or launch.formal_stage != "E6"
        or launch.content_source_binding != content_source
        or launch.inventory_sha256 != runtime_inventory.sha256
        or launch.gpu_uuids != row.gpu_uuids
        or config.model.algorithm != "NEXTN"
        or getattr(config.model, "nextn_mtp_mode", "external_drafter")
        != row.nextn_mtp_mode
        or getattr(config.model, "target_snapshot_sha256", None)
        != row.target_snapshot_sha256
        or getattr(config.model, "mtp_component_sha256", None)
        != row.mtp_component_sha256
        or config.model.target != row.target_model_id
        or config.model.drafter != row.drafter_model_id
        or config.model.target_revision != row.target_revision
        or config.model.drafter_revision != row.drafter_revision
        or launch.target_content_member_id != row.target_member_id
        or launch.drafter_content_member_id != row.drafter_member_id
        or getattr(launch, "nextn_mtp_mode", None)
        != ("built_in_mtp" if row.nextn_mtp_mode == "built_in_mtp" else None)
        or getattr(launch, "target_snapshot_sha256", None) != row.target_snapshot_sha256
        or getattr(launch, "mtp_component_sha256", None) != row.mtp_component_sha256
        or config.runtime.topology_mode != "tp2_dp1"
        or config.runtime.tensor_parallel_size != 2
        or config.runtime.data_parallel_size != 1
    ):
        raise ValueError("trusted E6 serving empirical authority differs")
    return FormalSingleOperatorTrustedNextnTp2ServingAuthority(
        schema_version=1,
        kind="formal_single_operator_e6_trusted_serving_authority",
        protocol_sha256=(
            FORMAL_SINGLE_OPERATOR_E6_TRUSTED_SERVING_AUTHORITY_PROTOCOL_SHA256
        ),
        trust_mode="trusted_single_operator_empirical_no_signature",
        formal_measured_authorization=False,
        execution_source=source_binding,
        execution_source_sha256=source.sha256,
        materialized_cell_id=cell.cell_id,
        node=source.node,
        auxiliary_bundle=auxiliary,
        auxiliary_bundle_sha256=bundle.sha256,
        protocol_lock_sha256=lock.sha256,
        compatibility_receipt_sha256=bundle.compatibility.sha256,
        compatibility_row_sha256=row.sha256,
        interface_fit_plan=terminal.plan,
        interface_fit_plan_sha256=plan.sha256,
        interface_fit_terminal=terminal_binding,
        interface_fit_terminal_sha256=terminal.sha256,
        content_source=content_source,
        trusted_content_bundle_sha256=content_source.content_sha256,
        inventory=inventory,
        inventory_sha256=runtime_inventory.sha256,
        doctor=plan.doctor,
        doctor_sha256=plan.doctor.semantic_sha256,
        model=cell.model,
        target_model_id=row.target_model_id,
        drafter_model_id=row.drafter_model_id,
        target_revision=row.target_revision,
        drafter_revision=row.drafter_revision,
        target_member_sha256=row.target_member_id,
        drafter_member_sha256=row.drafter_member_id,
        target_shard_manifest_sha256=row.target_shard_manifest_sha256,
        drafter_shard_manifest_sha256=row.drafter_shard_manifest_sha256,
        interface_sha256=row.interface_sha256,
        topology_sha256=row.topology_sha256,
        source_adapter_version=0,
        gpu_uuids=row.gpu_uuids,
        native_gpu_proof_sha256=row.native_gpu_proof_sha256,
        distributed_gpu_proof_sha256=row.distributed_gpu_proof_sha256,
        empirical_authority_sha256=row.verified_authority_sha256,
        junit_raw_sha256=terminal.junit_xml.raw_sha256,
        qualified_test_count=8,
        passed_test_count=8,
        failed_test_count=0,
        error_test_count=0,
        skipped_test_count=0,
        physical_execution_count=1,
        nextn_mtp_mode=row.nextn_mtp_mode,
        target_snapshot_sha256=row.target_snapshot_sha256,
        mtp_component_sha256=row.mtp_component_sha256,
    )


@dataclass(frozen=True)
class FormalSingleOperatorE6InterfaceReplayPlan:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_e6_interface_replay_plan"]
    protocol_sha256: str
    execution_source: CanonicalJsonProofBinding
    materialized_cell_id: str
    node: Literal["e6_pilot", "e6_final"]
    model: str
    auxiliary_bundle: FormalSingleOperatorJsonBinding
    shared_terminal: CanonicalJsonProofBinding
    physical_execution_reused: Literal[True]
    additional_gpu_runs: Literal[0]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_e6_interface_replay_plan"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_E6_INTERFACE_REPLAY_PROTOCOL_SHA256
            or self.node not in {"e6_pilot", "e6_final"}
            or self.model not in E6_MODELS
            or self.physical_execution_reused is not True
            or self.additional_gpu_runs != 0
        ):
            raise ValueError("E6 interface replay plan identity differs")
        _sha("E6 replay cell", self.materialized_cell_id)
        if type(self.execution_source) is not CanonicalJsonProofBinding:
            raise TypeError("E6 replay execution source is not path-bound")
        if type(self.auxiliary_bundle) is not FormalSingleOperatorJsonBinding:
            raise TypeError("E6 replay auxiliary is not path-bound")
        if type(self.shared_terminal) is not CanonicalJsonProofBinding:
            raise TypeError("E6 replay terminal is not path-bound")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "execution_source": self.execution_source.to_dict(),
            "auxiliary_bundle": self.auxiliary_bundle.to_dict(),
            "shared_terminal": self.shared_terminal.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "E6 interface replay plan",
            value,
            set(cls.__dataclass_fields__),
        )
        row["execution_source"] = CanonicalJsonProofBinding.from_dict(
            row["execution_source"]
        )
        row["auxiliary_bundle"] = FormalSingleOperatorJsonBinding.from_dict(
            row["auxiliary_bundle"]
        )
        row["shared_terminal"] = CanonicalJsonProofBinding.from_dict(
            row["shared_terminal"]
        )
        return cls(**row)  # type: ignore[arg-type]


def materialize_formal_single_operator_e6_interface_replay_plan(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    output_path: str | Path,
) -> FormalSingleOperatorE6InterfaceReplayPlan:
    from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
        route_formal_single_operator_cell,
    )

    source_binding = CanonicalJsonProofBinding.bind(execution_source_path)
    source, cell, route = route_formal_single_operator_cell(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
    )
    if (
        source.node not in {"e6_pilot", "e6_final"}
        or route.physical_kind != "e6_interface_preflight"
        or cell.task != "immutable_metadata_interface_and_fit_preflight"
    ):
        raise ValueError("E6 replay plan requires one interface/fit cell")
    auxiliary = source.auxiliary_source_binding("e6_interface_fit")
    lock = protocol_lock_from_dict(
        source.protocol_lock_source.reopen(label="E6 replay ProtocolLock")
    )
    bundle = revalidate_formal_single_operator_e6_interface_fit_bundle_value(
        auxiliary.reopen(label="E6 replay interface/fit auxiliary"),
        protocol_lock=lock,
    )
    terminal = terminal_for_model(bundle, cell.model)
    plan = FormalSingleOperatorE6InterfaceReplayPlan(
        schema_version=1,
        kind="formal_single_operator_e6_interface_replay_plan",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_E6_INTERFACE_REPLAY_PROTOCOL_SHA256,
        execution_source=source_binding,
        materialized_cell_id=cell.cell_id,
        node=source.node,
        model=cell.model,
        auxiliary_bundle=auxiliary,
        shared_terminal=terminal,
        physical_execution_reused=True,
        additional_gpu_runs=0,
    )
    publish_canonical_json_no_replace(output_path, plan.to_dict())
    return revalidate_formal_single_operator_e6_interface_replay_plan(output_path)


def revalidate_formal_single_operator_e6_interface_replay_plan(
    path: str | Path,
) -> FormalSingleOperatorE6InterfaceReplayPlan:
    from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
        route_formal_single_operator_cell,
    )

    binding = CanonicalJsonProofBinding.bind(path)
    plan = FormalSingleOperatorE6InterfaceReplayPlan.from_dict(binding.reopen())
    source, cell, route = route_formal_single_operator_cell(
        execution_source_path=plan.execution_source.absolute_path,
        materialized_cell_id=plan.materialized_cell_id,
    )
    lock = protocol_lock_from_dict(
        source.protocol_lock_source.reopen(label="E6 replay ProtocolLock")
    )
    auxiliary = source.auxiliary_source_binding("e6_interface_fit")
    bundle = revalidate_formal_single_operator_e6_interface_fit_bundle_value(
        auxiliary.reopen(label="E6 replay interface/fit auxiliary"),
        protocol_lock=lock,
    )
    terminal = terminal_for_model(bundle, cell.model)
    if (
        plan.sha256 != binding.semantic_sha256
        or source.node != plan.node
        or route.physical_kind != "e6_interface_preflight"
        or cell.task != "immutable_metadata_interface_and_fit_preflight"
        or cell.model != plan.model
        or auxiliary != plan.auxiliary_bundle
        or terminal != plan.shared_terminal
    ):
        raise ValueError("E6 interface replay plan lineage differs")
    return plan


__all__ = [
    "FORMAL_SINGLE_OPERATOR_E6_BUILT_IN_MTP_INTERFACE_FIT_PROTOCOL_SHA256",
    "FORMAL_SINGLE_OPERATOR_E6_INTERFACE_FIT_PROTOCOL_SHA256",
    "FORMAL_SINGLE_OPERATOR_E6_INTERFACE_REPLAY_PROTOCOL_SHA256",
    "FORMAL_SINGLE_OPERATOR_E6_TRUSTED_SERVING_AUTHORITY_PROTOCOL_SHA256",
    "NEXTN_BUILT_IN_MTP_INTERFACE_SHA256",
    "NEXTN_MTP_INTERFACE_SHA256",
    "FormalSingleOperatorE6CompatibilityReceipt",
    "FormalSingleOperatorE6CompatibilityRow",
    "FormalSingleOperatorE6InterfaceFitBlocked",
    "FormalSingleOperatorE6InterfaceFitBundle",
    "FormalSingleOperatorE6InterfaceFitCampaign",
    "FormalSingleOperatorE6InterfaceFitPlan",
    "FormalSingleOperatorE6InterfaceFitTerminal",
    "FormalSingleOperatorE6InterfaceReplayPlan",
    "FormalSingleOperatorTrustedNextnTp2ServingAuthority",
    "compatibility_row_for_terminal",
    "derive_formal_single_operator_trusted_nextn_tp2_serving_authority",
    "execute_formal_single_operator_e6_interface_fit_plan",
    "finalize_formal_single_operator_e6_interface_fit_bundle",
    "formal_single_operator_e6_interface_fit_process_hard_timeout_ns",
    "materialize_formal_single_operator_e6_interface_fit_campaign",
    "materialize_formal_single_operator_e6_interface_replay_plan",
    "revalidate_formal_single_operator_e6_interface_fit_bundle",
    "revalidate_formal_single_operator_e6_interface_fit_bundle_value",
    "revalidate_formal_single_operator_e6_interface_fit_plan",
    "revalidate_formal_single_operator_e6_interface_fit_terminal",
    "revalidate_formal_single_operator_e6_interface_replay_plan",
    "terminal_for_model",
]
