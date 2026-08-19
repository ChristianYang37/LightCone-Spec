"""Trusted single-operator publisher for the 108 E0 compatibility probes.

The publisher never executes a model and never turns missing or failed probe
evidence into ``N/A``.  It accepts only complete, timed terminal receipts and
derives each disposition from code-owned interface, workload, and one-request
smoke rules.  The resulting bundle is the exact auxiliary object consumed by
``formal_single_operator_downstream``.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.e0_stage_authority import (
    E0OnlineSpecSourceAuthority,
)
from lightcone_spec.experiments.formal_protocol import ProtocolLock, content_sha256
from lightcone_spec.experiments.formal_registry import (
    e0_compatibility_receipt_to_dict,
    e0_onlinespec_source_authority_to_dict,
)
from lightcone_spec.experiments.stage_materialization import (
    E0_BACKENDS,
    E0_MODELS,
    E0_TASKS,
    E0CompatibilityDecision,
    E0CompatibilityReceipt,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

E0_COMPATIBILITY_VALID_REASON = "PROBE_COMPATIBLE"
E0_COMPATIBILITY_NA_REASONS = frozenset(
    {
        "MODEL_BACKEND_INTERFACE_UNSUPPORTED",
        "TOKENIZER_TASK_WORKLOAD_UNSUPPORTED",
        "GPU_SMOKE_REGISTERED_UNSUPPORTED",
    }
)
E0_BACKEND_REQUIRES_GPU_SMOKE = {
    "EAGLE3": True,
    "DFLASH": True,
    "DSPARK": True,
}

E0_TRUSTED_MODEL_BACKEND_INTERFACE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "formal_single_operator_e0_model_backend_interface",
        "identity": ("exact_target_drafter_tokenizer_model_revision_content_members"),
        "launch": "path_bound_schema2_compile_launch",
        "eagle3": (
            "nine_task_keyed_exact_execution_compatibility_selector_native_"
            "gpu_path_bound_proof_rows_or_explicit_empty_for_other_backends"
        ),
    }
)
E0_TRUSTED_PREPROBE_MODEL_BACKEND_INTERFACE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "formal_single_operator_e0_model_backend_interface",
        "identity": ("exact_target_drafter_tokenizer_model_revision_content_members"),
        "launch": "path_bound_schema2_static_preprobe_compile_launch",
        "eagle3": (
            "preprobe_rows_empty_postprobe_task_row_bound_by_terminal_and_final_bundle"
        ),
    }
)


@dataclass(frozen=True)
class E0Eagle3RuntimeProofRow:
    """Four reopenable EAGLE3 authorities for one exact E0 task."""

    schema_version: Literal[1, 2]
    task: str
    execution_authority_sha256: str
    compatibility_authority_sha256: str
    model_selector_sha256: str
    native_gpu_proof_sha256: str
    execution_authority: CanonicalJsonProofBinding
    compatibility_authority: CanonicalJsonProofBinding
    model_selector_authority: CanonicalJsonProofBinding
    native_gpu_proof: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2} or self.task not in E0_TASKS:
            raise ValueError("E0 EAGLE3 proof-row scope differs")
        for label, digest in (
            ("execution authority", self.execution_authority_sha256),
            ("compatibility authority", self.compatibility_authority_sha256),
            ("model selector", self.model_selector_sha256),
            ("native GPU proof", self.native_gpu_proof_sha256),
        ):
            _sha256(f"E0 EAGLE3 {label}", digest)
        for label, binding in (
            ("execution authority", self.execution_authority),
            ("compatibility authority", self.compatibility_authority),
            ("model selector authority", self.model_selector_authority),
            ("native GPU proof", self.native_gpu_proof),
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError(f"E0 EAGLE3 {label} must be path-bound")

        execution = self.execution_authority.reopen()
        if type(execution) is not dict:
            raise ValueError("E0 EAGLE3 execution authority is not an object")
        if (
            self.execution_authority.semantic_sha256 != self.execution_authority_sha256
            or content_sha256(execution) != self.execution_authority_sha256
            or execution.get("stage") != "E0"
            or execution.get("task") != self.task
            or execution.get("compatibility_authority_sha256")
            != self.compatibility_authority_sha256
            or execution.get("model_selector_sha256") != self.model_selector_sha256
            or execution.get("native_gpu_receipt_sha256")
            != self.native_gpu_proof_sha256
        ):
            raise ValueError("E0 EAGLE3 execution authority replay differs")

        compatibility_value = self.compatibility_authority.reopen()
        if type(compatibility_value) is not dict:
            raise ValueError("E0 EAGLE3 compatibility authority is not an object")
        selector = self.model_selector_authority.reopen()
        if type(selector) is not dict:
            raise ValueError("E0 EAGLE3 selector authority is not an object")
        native_value = self.native_gpu_proof.reopen()
        if type(native_value) is not dict:
            raise ValueError("E0 EAGLE3 native proof is not an object")
        if self.schema_version == 2:
            expected_common = {
                "task": self.task,
                "interface_sha256": execution.get("interface_sha256"),
                "target_revision": execution.get("target_revision"),
                "drafter_revision": execution.get("drafter_revision"),
                "model": execution.get("model"),
            }
            core = CanonicalJsonProofBinding.from_dict(
                native_value.get("core_evidence")
            )
            result = CanonicalJsonProofBinding.from_dict(native_value.get("result"))
            lifecycle = CanonicalJsonProofBinding.from_dict(
                native_value.get("lifecycle")
            )
            core_value = core.reopen()
            result_value = result.reopen()
            lifecycle_value = lifecycle.reopen()
            if type(core_value) is not dict:
                raise ValueError("E0 EAGLE3 post-probe core is not an object")
            from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding

            for name in ("plan", "interface_receipt", "workload_authority"):
                CanonicalJsonProofBinding.from_dict(core_value.get(name)).reopen()
            for name in ("stdout", "stderr", "junit", "server_stdout", "server_stderr"):
                EvidenceFileBinding.from_dict(
                    core_value.get(name),
                    label=f"E0 EAGLE3 post-probe {name}",
                ).reopen(label=f"E0 EAGLE3 post-probe {name}")
            if (
                self.compatibility_authority.semantic_sha256
                != self.compatibility_authority_sha256
                or content_sha256(compatibility_value)
                != self.compatibility_authority_sha256
                or self.model_selector_authority.semantic_sha256
                != self.model_selector_sha256
                or content_sha256(selector) != self.model_selector_sha256
                or self.native_gpu_proof.semantic_sha256 != self.native_gpu_proof_sha256
                or content_sha256(native_value) != self.native_gpu_proof_sha256
                or execution.get("schema_version") != 1
                or execution.get("kind")
                != "trusted_single_operator_e0_eagle3_postprobe_execution_authority"
                or compatibility_value.get("schema_version") != 1
                or compatibility_value.get("kind")
                != "trusted_single_operator_e0_eagle3_postprobe_compatibility_authority"
                or selector.get("schema_version") != 1
                or selector.get("kind")
                != "trusted_single_operator_e0_eagle3_postprobe_model_selector"
                or native_value.get("schema_version") != 1
                or native_value.get("kind")
                != "trusted_single_operator_e0_eagle3_postprobe_native_gpu_proof"
                or any(
                    value.get(name) != expected
                    for value in (compatibility_value, selector, native_value)
                    for name, expected in expected_common.items()
                )
                or compatibility_value.get("status") != "COMPATIBLE"
                or compatibility_value.get("reason_code")
                != E0_COMPATIBILITY_VALID_REASON
                or compatibility_value.get("model_selector_sha256")
                != self.model_selector_sha256
                or compatibility_value.get("native_gpu_proof_sha256")
                != self.native_gpu_proof_sha256
                or CanonicalJsonProofBinding.from_dict(
                    compatibility_value.get("core_evidence")
                )
                != core
                or CanonicalJsonProofBinding.from_dict(
                    compatibility_value.get("model_selector")
                )
                != self.model_selector_authority
                or CanonicalJsonProofBinding.from_dict(
                    compatibility_value.get("native_gpu_proof")
                )
                != self.native_gpu_proof
                or CanonicalJsonProofBinding.from_dict(selector.get("core_evidence"))
                != core
                or CanonicalJsonProofBinding.from_dict(selector.get("result")) != result
                or execution.get("core_evidence_sha256") != core.semantic_sha256
                or execution.get("compatibility_authority_sha256")
                != self.compatibility_authority_sha256
                or execution.get("model_selector_sha256") != self.model_selector_sha256
                or execution.get("native_gpu_receipt_sha256")
                != self.native_gpu_proof_sha256
                or selector.get("core_evidence_sha256") != core.semantic_sha256
                or native_value.get("core_evidence_sha256") != core.semantic_sha256
                or native_value.get("result_sha256") != result.semantic_sha256
                or native_value.get("lifecycle_sha256") != lifecycle.semantic_sha256
                or core_value.get("kind")
                != "formal_single_operator_e0_compatibility_probe_evidence"
                or CanonicalJsonProofBinding.from_dict(core_value.get("result"))
                != result
                or CanonicalJsonProofBinding.from_dict(core_value.get("lifecycle"))
                != lifecycle
                or type(result_value) is not dict
                or result_value.get("output_token_count") != 1
                or type(lifecycle_value) is not dict
                or lifecycle_value.get("status") != "COMPLETE"
                or native_value.get("inventory_sha256")
                != execution.get("inventory_sha256")
                or native_value.get("gpu_uuids") != execution.get("gpu_uuids")
            ):
                raise ValueError("E0 EAGLE3 post-probe proof replay differs")
            return

        from lightcone_spec.runtime.backend import Eagle3CompatibilityAuthority
        from lightcone_spec.runtime.readiness import NativeRuntimeGpuProofReceipt

        compatibility = Eagle3CompatibilityAuthority(**compatibility_value)
        native = NativeRuntimeGpuProofReceipt.from_dict(native_value)
        if (
            content_sha256(compatibility_value)
            != self.compatibility_authority.semantic_sha256
            or compatibility.sha256 != self.compatibility_authority_sha256
            or compatibility.model_selector_sha256 != self.model_selector_sha256
            or content_sha256(selector) != self.model_selector_authority.semantic_sha256
            or selector.get("task") != self.task
            or selector.get("model_selector_sha256") != self.model_selector_sha256
            or selector.get("interface_sha256") != compatibility.interface_sha256
            or selector.get("target_revision") != compatibility.target_revision
            or selector.get("drafter_revision") != compatibility.drafter_revision
            or native.sha256 != self.native_gpu_proof_sha256
            or native.sha256 != self.native_gpu_proof.semantic_sha256
            or native.suite_id != "eagle3_tp1"
            or native.topology_mode != "tp1_dp1"
            or native.inventory_sha256 != execution.get("inventory_sha256")
            or list(native.gpu_uuids) != execution.get("gpu_uuids")
            or execution.get("target_revision") != compatibility.target_revision
            or execution.get("drafter_revision") != compatibility.drafter_revision
            or execution.get("interface_sha256") != compatibility.interface_sha256
        ):
            raise ValueError("E0 EAGLE3 four-proof replay differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("E0 EAGLE3 runtime proof-row fields differ")
        row = dict(value)
        for field in (
            "execution_authority",
            "compatibility_authority",
            "model_selector_authority",
            "native_gpu_proof",
        ):
            row[field] = CanonicalJsonProofBinding.from_dict(row[field])
        return cls(**row)  # type: ignore[arg-type]


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_time(label: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer timestamp")
    return value


def e0_preprobe_interface_sha256(
    *,
    protocol_lock_sha256: str,
    upstream_e6_confirmation_sha256: str,
    model: str,
    backend: str,
    target_model_id: str,
    target_revision: str,
    drafter_model_id: str,
    drafter_revision: str,
    tokenizer_model_id: str,
    tokenizer_revision: str,
    target_member_sha256: str,
    drafter_member_sha256: str,
    tokenizer_member_sha256: str,
    compile_launch_manifest_sha256: str | None,
    preprobe_evidence_sha256: str,
) -> str:
    """Derive the fresh interface identity from source-owned immutable fields."""

    return content_sha256(
        {
            "protocol_sha256": (
                E0_TRUSTED_PREPROBE_MODEL_BACKEND_INTERFACE_PROTOCOL_SHA256
            ),
            "protocol_lock_sha256": protocol_lock_sha256,
            "upstream_e6_confirmation_sha256": upstream_e6_confirmation_sha256,
            "model": model,
            "backend": backend,
            "target": [target_model_id, target_revision],
            "drafter": [drafter_model_id, drafter_revision],
            "tokenizer": [tokenizer_model_id, tokenizer_revision],
            "members": [
                target_member_sha256,
                drafter_member_sha256,
                tokenizer_member_sha256,
            ],
            "compile_launch_manifest_sha256": compile_launch_manifest_sha256,
            "eagle3_proof_rows": [],
            "preprobe_evidence_sha256": preprobe_evidence_sha256,
        }
    )


@dataclass(frozen=True)
class E0PreparedModelBackendInterfaceReceipt:
    """Prepared model/tokenizer identity and backend interface result."""

    schema_version: Literal[1, 2, 3]
    protocol_lock_sha256: str
    upstream_e6_confirmation_sha256: str
    model: str
    backend: str
    tokenizer_sha256: str
    interface_sha256: str
    prepared_model_manifest_sha256: str
    support_status: Literal["READY", "UNSUPPORTED"]
    reason_code: str
    requires_gpu_smoke: bool
    evidence_sha256: str
    target_model_id: str | None = None
    target_revision: str | None = None
    drafter_model_id: str | None = None
    drafter_revision: str | None = None
    tokenizer_model_id: str | None = None
    tokenizer_revision: str | None = None
    target_member_sha256: str | None = None
    drafter_member_sha256: str | None = None
    tokenizer_member_sha256: str | None = None
    compile_launch_manifest: CanonicalJsonProofBinding | None = None
    eagle3_runtime_proof_rows: tuple[E0Eagle3RuntimeProofRow, ...] = ()
    preprobe_evidence: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2, 3}:
            raise ValueError("E0 prepared interface schema differs")
        if self.model not in E0_MODELS or self.backend not in E0_BACKENDS:
            raise ValueError("E0 prepared interface lies outside the 12 pairs")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("E6 confirmation", self.upstream_e6_confirmation_sha256),
            ("tokenizer", self.tokenizer_sha256),
            ("interface", self.interface_sha256),
            ("prepared-model manifest", self.prepared_model_manifest_sha256),
            ("interface evidence", self.evidence_sha256),
        ):
            _sha256(f"E0 {label}", digest)
        if self.support_status == "READY":
            if self.reason_code != "INTERFACE_READY":
                raise ValueError("READY E0 interface requires INTERFACE_READY")
        elif self.support_status == "UNSUPPORTED":
            if self.reason_code != "MODEL_BACKEND_INTERFACE_UNSUPPORTED":
                raise ValueError("unsupported E0 interface reason differs")
        else:
            raise ValueError("E0 interface support status differs")
        if type(self.requires_gpu_smoke) is not bool:
            raise TypeError("E0 GPU-smoke requirement must be boolean")
        if self.requires_gpu_smoke != E0_BACKEND_REQUIRES_GPU_SMOKE[self.backend]:
            raise ValueError("E0 GPU-smoke requirement differs from backend policy")
        trusted_identity_fields = (
            self.target_model_id,
            self.target_revision,
            self.drafter_model_id,
            self.drafter_revision,
            self.tokenizer_model_id,
            self.tokenizer_revision,
            self.target_member_sha256,
            self.drafter_member_sha256,
            self.tokenizer_member_sha256,
        )
        if self.schema_version == 1:
            if (
                any(
                    value is not None
                    for value in (
                        *trusted_identity_fields,
                        self.compile_launch_manifest,
                        self.preprobe_evidence,
                    )
                )
                or self.eagle3_runtime_proof_rows
            ):
                raise ValueError("legacy E0 interface carries trusted launch fields")
            return
        if any(value is None for value in trusted_identity_fields):
            raise ValueError("trusted E0 interface identity is incomplete")
        assert self.target_model_id is not None
        assert self.target_revision is not None
        assert self.drafter_model_id is not None
        assert self.drafter_revision is not None
        assert self.tokenizer_model_id is not None
        assert self.tokenizer_revision is not None
        assert self.target_member_sha256 is not None
        assert self.drafter_member_sha256 is not None
        assert self.tokenizer_member_sha256 is not None
        for label, value in (
            ("target model", self.target_model_id),
            ("drafter model", self.drafter_model_id),
            ("tokenizer model", self.tokenizer_model_id),
        ):
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"trusted E0 {label} differs")
        for label, value in (
            ("target revision", self.target_revision),
            ("drafter revision", self.drafter_revision),
            ("tokenizer revision", self.tokenizer_revision),
        ):
            if len(value) not in {40, 64} or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"trusted E0 {label} is not immutable")
        for label, value in (
            ("target member", self.target_member_sha256),
            ("drafter member", self.drafter_member_sha256),
            ("tokenizer member", self.tokenizer_member_sha256),
        ):
            _sha256(f"trusted E0 {label}", value)
        proof_by_task = {row.task: row for row in self.eagle3_runtime_proof_rows}
        if (
            self.backend == "EAGLE3"
            and self.support_status == "READY"
            and self.schema_version == 2
            and (
                len(self.eagle3_runtime_proof_rows) != len(E0_TASKS)
                or set(proof_by_task) != set(E0_TASKS)
            )
        ):
            raise ValueError("READY EAGLE3 interface lacks nine exact task proofs")
        if self.schema_version == 3 and self.eagle3_runtime_proof_rows:
            raise ValueError("fresh pre-probe E0 interface carries post-probe proofs")
        if (
            self.backend != "EAGLE3" or self.support_status != "READY"
        ) and self.eagle3_runtime_proof_rows:
            raise ValueError("non-ready EAGLE3 interface carries EAGLE3 proofs")
        if self.schema_version == 2 and self.preprobe_evidence is not None:
            raise ValueError(
                "schema-2 E0 interface carries schema-3 pre-probe evidence"
            )
        if self.schema_version == 3:
            if type(self.preprobe_evidence) is not CanonicalJsonProofBinding:
                raise TypeError(
                    "fresh E0 interface lacks path-bound pre-probe evidence"
                )
            evidence = self.preprobe_evidence.reopen()
            if (
                type(evidence) is not dict
                or evidence.get("schema_version") != 1
                or evidence.get("kind")
                != "formal_single_operator_e0_preprobe_interface_evidence"
                or evidence.get("protocol_lock_sha256") != self.protocol_lock_sha256
                or evidence.get("upstream_e6_confirmation_sha256")
                != self.upstream_e6_confirmation_sha256
                or evidence.get("model") != self.model
                or evidence.get("backend") != self.backend
                or evidence.get("target_member_sha256") != self.target_member_sha256
                or evidence.get("drafter_member_sha256") != self.drafter_member_sha256
                or evidence.get("tokenizer_member_sha256")
                != self.tokenizer_member_sha256
                or evidence.get("compile_launch_manifest_sha256")
                != (
                    None
                    if self.compile_launch_manifest is None
                    else self.compile_launch_manifest.semantic_sha256
                )
                or self.preprobe_evidence.semantic_sha256 != self.evidence_sha256
            ):
                raise ValueError("fresh E0 pre-probe evidence replay differs")

        if self.support_status == "UNSUPPORTED":
            if self.compile_launch_manifest is not None:
                raise ValueError(
                    "unsupported E0 interface carries an executable launch"
                )
            expected_interface = content_sha256(
                {
                    "protocol_sha256": (
                        E0_TRUSTED_PREPROBE_MODEL_BACKEND_INTERFACE_PROTOCOL_SHA256
                        if self.schema_version == 3
                        else E0_TRUSTED_MODEL_BACKEND_INTERFACE_PROTOCOL_SHA256
                    ),
                    "protocol_lock_sha256": self.protocol_lock_sha256,
                    "upstream_e6_confirmation_sha256": (
                        self.upstream_e6_confirmation_sha256
                    ),
                    "model": self.model,
                    "backend": self.backend,
                    "target": [self.target_model_id, self.target_revision],
                    "drafter": [self.drafter_model_id, self.drafter_revision],
                    "tokenizer": [self.tokenizer_model_id, self.tokenizer_revision],
                    "members": [
                        self.target_member_sha256,
                        self.drafter_member_sha256,
                        self.tokenizer_member_sha256,
                    ],
                    "compile_launch_manifest_sha256": None,
                    "eagle3_proof_rows": [],
                    **(
                        {"preprobe_evidence_sha256": self.evidence_sha256}
                        if self.schema_version == 3
                        else {}
                    ),
                }
            )
            if self.interface_sha256 != expected_interface:
                raise ValueError(
                    "trusted unsupported E0 interface digest differs from identity"
                )
            return
        if self.compile_launch_manifest is None:
            raise ValueError("READY E0 interface lacks its compile launch")

        from lightcone_spec.config import load_run_config
        from lightcone_spec.runtime.compile_cache import (
            CompileCacheLaunchPlan,
            validate_compile_key_for_run_config,
        )
        from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

        launch = CompileLaunchManifest.load(self.compile_launch_manifest.absolute_path)
        config = load_run_config(launch.run_config_path)
        cache = CompileCacheLaunchPlan.load(launch.compile_cache_plan_path)
        validate_compile_key_for_run_config(cache, config=config)
        if (
            self.compile_launch_manifest.semantic_sha256 != launch.sha256
            or launch.schema_version != 2
            or launch.formal_stage != "E0"
            or config.model.algorithm != self.backend
            or config.model.target != self.target_model_id
            or config.model.target_revision != self.target_revision
            or config.model.drafter != self.drafter_model_id
            or config.model.drafter_revision != self.drafter_revision
            or launch.target_model_id != self.target_model_id
            or launch.target_revision != self.target_revision
            or launch.drafter_model_id != self.drafter_model_id
            or launch.drafter_revision != self.drafter_revision
            or launch.tokenizer_model_id != self.tokenizer_model_id
            or launch.tokenizer_revision != self.tokenizer_revision
            or launch.target_content_member_id != self.target_member_sha256
            or launch.drafter_content_member_id != self.drafter_member_sha256
            or launch.tokenizer_content_member_id != self.tokenizer_member_sha256
            or launch.prepared_model_content_manifest_sha256
            != self.prepared_model_manifest_sha256
            or self.tokenizer_sha256 != self.tokenizer_member_sha256
            or config.method != "static"
            or config.adaptation is not None
            or config.online_spec is not None
            or config.runtime.topology_mode != "tp1_dp1"
            or len(launch.gpu_uuids) != 1
        ):
            raise ValueError("trusted E0 interface launch identity differs")
        expected_interface = content_sha256(
            {
                "protocol_sha256": (
                    E0_TRUSTED_PREPROBE_MODEL_BACKEND_INTERFACE_PROTOCOL_SHA256
                    if self.schema_version == 3
                    else E0_TRUSTED_MODEL_BACKEND_INTERFACE_PROTOCOL_SHA256
                ),
                "protocol_lock_sha256": self.protocol_lock_sha256,
                "upstream_e6_confirmation_sha256": (
                    self.upstream_e6_confirmation_sha256
                ),
                "model": self.model,
                "backend": self.backend,
                "target": [self.target_model_id, self.target_revision],
                "drafter": [self.drafter_model_id, self.drafter_revision],
                "tokenizer": [self.tokenizer_model_id, self.tokenizer_revision],
                "members": [
                    self.target_member_sha256,
                    self.drafter_member_sha256,
                    self.tokenizer_member_sha256,
                ],
                "compile_launch_manifest_sha256": launch.sha256,
                "eagle3_proof_rows": [
                    [row.task, row.sha256]
                    for row in sorted(
                        self.eagle3_runtime_proof_rows,
                        key=lambda item: item.task,
                    )
                ],
                **(
                    {"preprobe_evidence_sha256": self.evidence_sha256}
                    if self.schema_version == 3
                    else {}
                ),
            }
        )
        if self.interface_sha256 != expected_interface:
            raise ValueError("trusted E0 interface digest differs from exact launch")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        if self.schema_version == 1:
            for name in (
                "target_model_id",
                "target_revision",
                "drafter_model_id",
                "drafter_revision",
                "tokenizer_model_id",
                "tokenizer_revision",
                "target_member_sha256",
                "drafter_member_sha256",
                "tokenizer_member_sha256",
                "compile_launch_manifest",
                "eagle3_runtime_proof_rows",
                "preprobe_evidence",
            ):
                value.pop(name)
        elif self.schema_version == 2:
            value.pop("preprobe_evidence")
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("E0 prepared interface receipt must be an object")
        row = dict(value)
        schema_version = row.get("schema_version")
        legacy = {
            "schema_version",
            "protocol_lock_sha256",
            "upstream_e6_confirmation_sha256",
            "model",
            "backend",
            "tokenizer_sha256",
            "interface_sha256",
            "prepared_model_manifest_sha256",
            "support_status",
            "reason_code",
            "requires_gpu_smoke",
            "evidence_sha256",
        }
        expected = (
            legacy
            if schema_version == 1
            else set(cls.__dataclass_fields__)
            - ({"preprobe_evidence"} if schema_version == 2 else set())
        )
        if set(row) != expected:
            raise ValueError("E0 prepared interface receipt fields differ")
        if schema_version == 1:
            return cls(**row)  # type: ignore[arg-type]
        if row["compile_launch_manifest"] is not None:
            row["compile_launch_manifest"] = CanonicalJsonProofBinding.from_dict(
                row["compile_launch_manifest"]
            )
        if schema_version == 3:
            row["preprobe_evidence"] = CanonicalJsonProofBinding.from_dict(
                row["preprobe_evidence"]
            )
        proof_rows = row["eagle3_runtime_proof_rows"]
        if type(proof_rows) is not list:
            raise TypeError("E0 EAGLE3 proof rows must be an array")
        row["eagle3_runtime_proof_rows"] = tuple(
            E0Eagle3RuntimeProofRow.from_dict(item) for item in proof_rows
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class E0TaskNativeWorkloadAuthority:
    """One model-tokenizer/task request-shape authority."""

    schema_version: Literal[1]
    protocol_lock_sha256: str
    upstream_e6_confirmation_sha256: str
    model: str
    task: str
    tokenizer_sha256: str
    task_native_workload_sha256: str
    source_revision_sha256: str
    support_status: Literal["READY", "UNSUPPORTED"]
    reason_code: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("E0 task-native workload schema differs")
        if self.model not in E0_MODELS or self.task not in E0_TASKS:
            raise ValueError("E0 workload authority lies outside the 36 pairs")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("E6 confirmation", self.upstream_e6_confirmation_sha256),
            ("tokenizer", self.tokenizer_sha256),
            ("task-native workload", self.task_native_workload_sha256),
            ("workload source revision", self.source_revision_sha256),
            ("workload evidence", self.evidence_sha256),
        ):
            _sha256(f"E0 {label}", digest)
        if self.support_status == "READY":
            if self.reason_code != "TASK_WORKLOAD_READY":
                raise ValueError("READY E0 workload requires TASK_WORKLOAD_READY")
        elif self.support_status == "UNSUPPORTED":
            if self.reason_code != "TOKENIZER_TASK_WORKLOAD_UNSUPPORTED":
                raise ValueError("unsupported E0 workload reason differs")
        else:
            raise ValueError("E0 workload support status differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("E0 task-native workload authority fields differ")
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class E0CompatibilityProbeTerminal:
    """Actual terminal for one model/backend/task compatibility decision."""

    schema_version: Literal[1, 2, 3]
    protocol_lock_sha256: str
    upstream_e6_confirmation_sha256: str
    model: str
    backend: str
    task: str
    interface_sha256: str
    task_native_workload_sha256: str
    tokenizer_sha256: str
    command_sha256: str
    started_ns: int
    finished_ns: int
    terminal_status: Literal["COMPLETE", "FAILED"]
    exit_code: int
    stdout_sha256: str
    stderr_sha256: str
    junit_sha256: str
    junit_status: Literal["PASS", "FAIL"]
    evidence_sha256: str
    smoke_status: Literal["PASS", "NOT_REQUIRED", "REGISTERED_UNSUPPORTED"]
    completed_request_count: int
    disposition: Literal["VALID", "N/A"]
    reason_code: str
    interface_receipt_sha256: str | None = None
    compile_launch_manifest_sha256: str | None = None
    eagle3_runtime_proof_row_sha256: str | None = None
    eagle3_runtime_proof_row: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2, 3}:
            raise ValueError("E0 compatibility terminal schema differs")
        if (
            self.model not in E0_MODELS
            or self.backend not in E0_BACKENDS
            or self.task not in E0_TASKS
        ):
            raise ValueError("E0 compatibility terminal lies outside 108 probes")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("E6 confirmation", self.upstream_e6_confirmation_sha256),
            ("interface", self.interface_sha256),
            ("task-native workload", self.task_native_workload_sha256),
            ("tokenizer", self.tokenizer_sha256),
            ("command", self.command_sha256),
            ("stdout", self.stdout_sha256),
            ("stderr", self.stderr_sha256),
            ("JUnit", self.junit_sha256),
            ("probe evidence", self.evidence_sha256),
        ):
            _sha256(f"E0 terminal {label}", digest)
        started = _positive_time("E0 probe started_ns", self.started_ns)
        finished = _positive_time("E0 probe finished_ns", self.finished_ns)
        if finished <= started:
            raise ValueError("E0 probe finish must follow its start")
        if type(self.exit_code) is not int:
            raise TypeError("E0 probe exit code must be an integer")
        if (
            type(self.completed_request_count) is not int
            or self.completed_request_count < 0
        ):
            raise ValueError("E0 completed request count must be non-negative")
        if self.disposition not in {"VALID", "N/A"}:
            raise ValueError("E0 probe disposition differs")
        if self.reason_code not in {
            E0_COMPATIBILITY_VALID_REASON,
            *E0_COMPATIBILITY_NA_REASONS,
        }:
            raise ValueError("E0 probe reason is not code-owned")
        if self.junit_status not in {"PASS", "FAIL"}:
            raise ValueError("E0 probe JUnit status differs")
        trusted = (
            self.interface_receipt_sha256,
            self.compile_launch_manifest_sha256,
            self.eagle3_runtime_proof_row_sha256,
            self.eagle3_runtime_proof_row,
        )
        if self.schema_version == 1:
            if any(value is not None for value in trusted):
                raise ValueError("legacy E0 terminal carries trusted launch lineage")
        else:
            _sha256(
                "trusted E0 terminal interface receipt",
                self.interface_receipt_sha256,
            )
            if self.reason_code == "MODEL_BACKEND_INTERFACE_UNSUPPORTED":
                if self.compile_launch_manifest_sha256 is not None:
                    raise ValueError("unsupported E0 terminal carries a compile launch")
            else:
                _sha256(
                    "trusted E0 terminal compile launch",
                    self.compile_launch_manifest_sha256,
                )
            if (
                self.backend == "EAGLE3"
                and self.reason_code == E0_COMPATIBILITY_VALID_REASON
            ):
                _sha256(
                    "trusted E0 EAGLE3 task proof row",
                    self.eagle3_runtime_proof_row_sha256,
                )
            elif self.eagle3_runtime_proof_row_sha256 is not None:
                raise ValueError("non-EAGLE3 terminal carries EAGLE3 task authority")
            if self.schema_version == 2:
                if self.eagle3_runtime_proof_row is not None:
                    raise ValueError("schema-2 terminal carries schema-3 proof binding")
            elif self.eagle3_runtime_proof_row_sha256 is None:
                if self.eagle3_runtime_proof_row is not None:
                    raise ValueError("terminal carries an unclaimed EAGLE3 proof row")
            else:
                if type(self.eagle3_runtime_proof_row) is not CanonicalJsonProofBinding:
                    raise TypeError(
                        "fresh EAGLE3 terminal lacks a path-bound proof row"
                    )
                proof = E0Eagle3RuntimeProofRow.from_dict(
                    self.eagle3_runtime_proof_row.reopen()
                )
                execution = proof.execution_authority.reopen()
                native = proof.native_gpu_proof.reopen()
                core = CanonicalJsonProofBinding.from_dict(native.get("core_evidence"))
                result = CanonicalJsonProofBinding.from_dict(native.get("result"))
                lifecycle = CanonicalJsonProofBinding.from_dict(native.get("lifecycle"))
                core_value = core.reopen()
                result_value = result.reopen()
                lifecycle_value = lifecycle.reopen()
                if type(core_value) is not dict:
                    raise ValueError("fresh EAGLE3 terminal core evidence differs")
                plan = CanonicalJsonProofBinding.from_dict(core_value.get("plan"))
                interface = CanonicalJsonProofBinding.from_dict(
                    core_value.get("interface_receipt")
                )
                workload = CanonicalJsonProofBinding.from_dict(
                    core_value.get("workload_authority")
                )
                plan_value = plan.reopen()
                workload_value = workload.reopen()
                execution_gpus = execution.get("gpu_uuids")
                from lightcone_spec.runtime.preflight_runner import (
                    EvidenceFileBinding,
                )

                stdout = EvidenceFileBinding.from_dict(
                    core_value.get("stdout"), label="fresh E0 EAGLE3 stdout"
                )
                stderr = EvidenceFileBinding.from_dict(
                    core_value.get("stderr"), label="fresh E0 EAGLE3 stderr"
                )
                junit = EvidenceFileBinding.from_dict(
                    core_value.get("junit"), label="fresh E0 EAGLE3 JUnit"
                )
                stdout.reopen(label="fresh E0 EAGLE3 stdout")
                stderr.reopen(label="fresh E0 EAGLE3 stderr")
                junit.reopen(label="fresh E0 EAGLE3 JUnit")
                if (
                    proof.schema_version != 2
                    or proof.task != self.task
                    or proof.sha256 != self.eagle3_runtime_proof_row_sha256
                    or self.eagle3_runtime_proof_row.semantic_sha256 != proof.sha256
                    or type(execution) is not dict
                    or execution.get("model") != self.model
                    or execution.get("interface_sha256") != self.interface_sha256
                    or execution.get("compile_launch_manifest_sha256")
                    != self.compile_launch_manifest_sha256
                    or type(native) is not dict
                    or native.get("core_evidence_sha256") != self.evidence_sha256
                    or native.get("compile_launch_manifest_sha256")
                    != self.compile_launch_manifest_sha256
                    or core.semantic_sha256 != self.evidence_sha256
                    or type(plan_value) is not dict
                    or plan_value.get("model") != self.model
                    or plan_value.get("backend") != self.backend
                    or plan_value.get("task") != self.task
                    or type(execution_gpus) is not list
                    or len(execution_gpus) != 1
                    or plan_value.get("gpu_uuid") != execution_gpus[0]
                    or interface.semantic_sha256 != self.interface_receipt_sha256
                    or type(workload_value) is not dict
                    or workload_value.get("model") != self.model
                    or workload_value.get("task") != self.task
                    or workload_value.get("task_native_workload_sha256")
                    != self.task_native_workload_sha256
                    or core_value.get("command_sha256") != self.command_sha256
                    or core_value.get("started_ns") != self.started_ns
                    or core_value.get("finished_ns") != self.finished_ns
                    or core_value.get("completed_request_count")
                    != self.completed_request_count
                    or stdout.raw_sha256 != self.stdout_sha256
                    or stderr.raw_sha256 != self.stderr_sha256
                    or junit.raw_sha256 != self.junit_sha256
                    or type(result_value) is not dict
                    or result_value.get("output_token_count") != 1
                    or type(lifecycle_value) is not dict
                    or lifecycle_value.get("status") != "COMPLETE"
                    or lifecycle_value.get("request_started_ns") != self.started_ns
                    or lifecycle_value.get("finished_ns") != self.finished_ns
                    or lifecycle_value.get("completed_request_count")
                    != self.completed_request_count
                    or lifecycle_value.get("gpu_uuid") != plan_value.get("gpu_uuid")
                ):
                    raise ValueError("fresh EAGLE3 terminal proof row differs")

    @cached_property
    def key(self) -> tuple[str, str, str]:
        return self.model, self.backend, self.task

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        if self.schema_version == 1:
            value.pop("interface_receipt_sha256")
            value.pop("compile_launch_manifest_sha256")
            value.pop("eagle3_runtime_proof_row_sha256")
            value.pop("eagle3_runtime_proof_row")
        elif self.schema_version == 2:
            value.pop("eagle3_runtime_proof_row")
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("E0 compatibility terminal must be an object")
        row = dict(value)
        legacy = set(cls.__dataclass_fields__) - {
            "interface_receipt_sha256",
            "compile_launch_manifest_sha256",
            "eagle3_runtime_proof_row_sha256",
            "eagle3_runtime_proof_row",
        }
        schema_version = row.get("schema_version")
        expected = (
            legacy
            if schema_version == 1
            else set(cls.__dataclass_fields__)
            - ({"eagle3_runtime_proof_row"} if schema_version == 2 else set())
        )
        if set(row) != expected:
            raise ValueError("E0 compatibility terminal fields differ")
        if schema_version == 3 and row["eagle3_runtime_proof_row"] is not None:
            row["eagle3_runtime_proof_row"] = CanonicalJsonProofBinding.from_dict(
                row["eagle3_runtime_proof_row"]
            )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class E0CompatibilityEvidenceManifest:
    schema_version: Literal[1, 2, 3]
    protocol_lock_sha256: str
    upstream_e6_materialization_sha256: str
    upstream_e6_confirmation_sha256: str
    started_ns: int
    finished_ns: int
    interface_receipt_sha256s: tuple[str, ...]
    workload_authority_sha256s: tuple[str, ...]
    probe_terminal_sha256s: tuple[str, ...]
    interface_receipts: tuple[CanonicalJsonProofBinding, ...] = ()
    workload_authorities: tuple[CanonicalJsonProofBinding, ...] = ()
    probe_terminals: tuple[CanonicalJsonProofBinding, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2, 3}:
            raise ValueError("E0 compatibility evidence manifest schema differs")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("E6 materialization", self.upstream_e6_materialization_sha256),
            ("E6 confirmation", self.upstream_e6_confirmation_sha256),
        ):
            _sha256(f"E0 evidence {label}", digest)
        started = _positive_time("E0 evidence started_ns", self.started_ns)
        finished = _positive_time("E0 evidence finished_ns", self.finished_ns)
        if finished <= started:
            raise ValueError("E0 evidence finish must follow its start")
        for label, values, count in (
            ("interface receipts", self.interface_receipt_sha256s, 12),
            ("workload authorities", self.workload_authority_sha256s, 36),
            ("probe terminals", self.probe_terminal_sha256s, 108),
        ):
            if (
                type(values) is not tuple
                or len(values) != count
                or values != tuple(sorted(set(values)))
            ):
                raise ValueError(f"E0 evidence {label} coverage differs")
            for digest in values:
                _sha256(f"E0 evidence {label}", digest)
        if self.schema_version == 1:
            if (
                self.interface_receipts
                or self.workload_authorities
                or self.probe_terminals
            ):
                raise ValueError("legacy E0 evidence carries trusted path bindings")
            return
        if (
            len(self.interface_receipts) != 12
            or len(set(self.interface_receipts)) != 12
            or len(self.workload_authorities) != 36
            or len(set(self.workload_authorities)) != 36
            or len(self.probe_terminals) != 108
            or len(set(self.probe_terminals)) != 108
        ):
            raise ValueError("trusted E0 evidence path coverage differs")
        interfaces = tuple(
            E0PreparedModelBackendInterfaceReceipt.from_dict(row.reopen())
            for row in self.interface_receipts
        )
        terminals = tuple(
            E0CompatibilityProbeTerminal.from_dict(row.reopen())
            for row in self.probe_terminals
        )
        workloads = tuple(
            E0TaskNativeWorkloadAuthority.from_dict(row.reopen())
            for row in self.workload_authorities
        )
        if (
            any(
                row.schema_version != self.schema_version
                for row in (*interfaces, *terminals)
            )
            or tuple(sorted(row.sha256 for row in interfaces))
            != self.interface_receipt_sha256s
            or tuple(sorted(row.sha256 for row in workloads))
            != self.workload_authority_sha256s
            or tuple(sorted(row.sha256 for row in terminals))
            != self.probe_terminal_sha256s
            or tuple(row.semantic_sha256 for row in self.interface_receipts)
            != tuple(row.sha256 for row in interfaces)
            or tuple(row.semantic_sha256 for row in self.workload_authorities)
            != tuple(row.sha256 for row in workloads)
            or tuple(row.semantic_sha256 for row in self.probe_terminals)
            != tuple(row.sha256 for row in terminals)
        ):
            raise ValueError("trusted E0 evidence bound receipt identity differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "upstream_e6_materialization_sha256": (
                self.upstream_e6_materialization_sha256
            ),
            "upstream_e6_confirmation_sha256": (self.upstream_e6_confirmation_sha256),
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
            "interface_receipt_sha256s": list(self.interface_receipt_sha256s),
            "workload_authority_sha256s": list(self.workload_authority_sha256s),
            "probe_terminal_sha256s": list(self.probe_terminal_sha256s),
        }
        if self.schema_version in {2, 3}:
            value["interface_receipts"] = [
                row.to_dict() for row in self.interface_receipts
            ]
            value["workload_authorities"] = [
                row.to_dict() for row in self.workload_authorities
            ]
            value["probe_terminals"] = [row.to_dict() for row in self.probe_terminals]
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise TypeError("E0 compatibility evidence must be an object")
        row = dict(value)
        expected = {
            "schema_version",
            "protocol_lock_sha256",
            "upstream_e6_materialization_sha256",
            "upstream_e6_confirmation_sha256",
            "started_ns",
            "finished_ns",
            "interface_receipt_sha256s",
            "workload_authority_sha256s",
            "probe_terminal_sha256s",
        }
        if row.get("schema_version") in {2, 3}:
            expected |= {
                "interface_receipts",
                "workload_authorities",
                "probe_terminals",
            }
        if set(row) != expected:
            raise ValueError("E0 compatibility evidence fields differ")
        for name in (
            "interface_receipt_sha256s",
            "workload_authority_sha256s",
            "probe_terminal_sha256s",
        ):
            raw = row[name]
            if type(raw) is not list:
                raise TypeError(f"E0 compatibility {name} must be an array")
            row[name] = tuple(raw)
        if row["schema_version"] in {2, 3}:
            for name in (
                "interface_receipts",
                "workload_authorities",
                "probe_terminals",
            ):
                raw = row[name]
                if type(raw) is not list:
                    raise TypeError(f"trusted E0 {name} must be an array")
                row[name] = tuple(
                    CanonicalJsonProofBinding.from_dict(item) for item in raw
                )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class E0CompatibilityPublication:
    compatibility: E0CompatibilityReceipt
    evidence_manifest: E0CompatibilityEvidenceManifest
    bundle: dict[str, object]

    def __post_init__(self) -> None:
        if type(self.compatibility) is not E0CompatibilityReceipt:
            raise TypeError("E0 publication compatibility type differs")
        if type(self.evidence_manifest) is not E0CompatibilityEvidenceManifest:
            raise TypeError("E0 publication evidence manifest type differs")
        expected_fields = {
            "schema_version",
            "kind",
            "protocol_lock_sha256",
            "upstream_e6_materialization_sha256",
            "upstream_e6_confirmation_sha256",
            "compatibility",
            "compatibility_sha256",
            "compatibility_evidence_manifest_sha256",
            "onlinespec_source_authority",
            "onlinespec_source_authority_sha256",
            "started_ns",
            "finished_ns",
            "bundle_sha256",
        }
        if self.bundle.get("schema_version") in {2, 3}:
            expected_fields.add("compatibility_evidence_manifest")
        if type(self.bundle) is not dict or set(self.bundle) != expected_fields:
            raise ValueError("E0 compatibility bundle fields differ")
        payload = dict(self.bundle)
        expected_sha256 = payload.pop("bundle_sha256")
        if expected_sha256 != content_sha256(payload):
            raise ValueError("E0 compatibility bundle digest differs")
        if self.bundle["schema_version"] == 1:
            if self.evidence_manifest.schema_version != 1:
                raise ValueError("legacy E0 bundle/evidence schema differs")
        elif self.bundle["schema_version"] in {2, 3}:
            binding = CanonicalJsonProofBinding.from_dict(
                self.bundle["compatibility_evidence_manifest"]
            )
            observed = E0CompatibilityEvidenceManifest.from_dict(binding.reopen())
            if (
                self.evidence_manifest.schema_version != self.bundle["schema_version"]
                or observed != self.evidence_manifest
                or binding.semantic_sha256 != self.evidence_manifest.sha256
                or self.bundle["compatibility_evidence_manifest_sha256"]
                != self.evidence_manifest.sha256
            ):
                raise ValueError("trusted E0 evidence manifest binding differs")
        else:
            raise ValueError("E0 compatibility publication schema differs")


def _e6_identity(
    protocol_lock: ProtocolLock,
    completion: object,
) -> tuple[str, str]:
    from lightcone_spec.experiments.formal_single_operator_stages import (
        RebuiltFormalSingleOperatorStageCompletion,
    )

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E0 compatibility requires an exact ProtocolLock")
    if type(completion) is not RebuiltFormalSingleOperatorStageCompletion:
        raise TypeError("E0 compatibility requires an exact E6 completion")
    if completion.artifact.node != "e6_final":
        raise ValueError("E0 compatibility requires the E6 final completion")
    materialization_sha256 = _sha256(
        "E0 upstream E6 materialization",
        completion.materialization.sha256,
    )
    confirmation_sha256 = _sha256(
        "E0 upstream E6 confirmation",
        completion.decision.payload.get("confirmation_sha256"),
    )
    if (
        completion.materialization.protocol_lock_sha256 != protocol_lock.sha256
        or completion.decision.next_materialization_source_decision_sha256
        != confirmation_sha256
        or completion.decision.next_materialization_upstream_receipt_sha256s
        != (materialization_sha256,)
    ):
        raise ValueError("E0 compatibility E6 lineage differs")
    return materialization_sha256, confirmation_sha256


def reduce_e0_compatibility_probes(
    *,
    protocol_lock: ProtocolLock,
    e6_completion: object,
    interface_receipts: tuple[E0PreparedModelBackendInterfaceReceipt, ...],
    workload_authorities: tuple[E0TaskNativeWorkloadAuthority, ...],
    probe_terminals: tuple[E0CompatibilityProbeTerminal, ...],
    onlinespec_source_authority: E0OnlineSpecSourceAuthority | None,
) -> E0CompatibilityPublication:
    """Reduce complete actual probes into the canonical 13-field bundle."""

    e6_materialization_sha256, e6_confirmation_sha256 = _e6_identity(
        protocol_lock,
        e6_completion,
    )
    if type(interface_receipts) is not tuple or any(
        type(row) is not E0PreparedModelBackendInterfaceReceipt
        for row in interface_receipts
    ):
        raise TypeError("E0 interface inputs require exact receipts")
    if type(workload_authorities) is not tuple or any(
        type(row) is not E0TaskNativeWorkloadAuthority for row in workload_authorities
    ):
        raise TypeError("E0 workload inputs require exact authorities")
    if type(probe_terminals) is not tuple or any(
        type(row) is not E0CompatibilityProbeTerminal for row in probe_terminals
    ):
        raise TypeError("E0 probe inputs require exact terminals")
    interface_by_key = {(row.model, row.backend): row for row in interface_receipts}
    workload_by_key = {(row.model, row.task): row for row in workload_authorities}
    terminal_by_key = {row.key: row for row in probe_terminals}
    expected_interfaces = {
        (model, backend) for model in E0_MODELS for backend in E0_BACKENDS
    }
    expected_workloads = {(model, task) for model in E0_MODELS for task in E0_TASKS}
    expected_terminals = {
        (model, backend, task)
        for model in E0_MODELS
        for backend in E0_BACKENDS
        for task in E0_TASKS
    }
    if (
        len(interface_receipts) != 12
        or set(interface_by_key) != expected_interfaces
        or len(workload_authorities) != 36
        or set(workload_by_key) != expected_workloads
        or len(probe_terminals) != 108
        or set(terminal_by_key) != expected_terminals
    ):
        raise ValueError("E0 compatibility source coverage is not 12/36/108")
    decisions = []
    for key in sorted(expected_terminals):
        model, backend, task = key
        interface = interface_by_key[(model, backend)]
        workload = workload_by_key[(model, task)]
        terminal = terminal_by_key[key]
        if any(
            row.protocol_lock_sha256 != protocol_lock.sha256
            or row.upstream_e6_confirmation_sha256 != e6_confirmation_sha256
            for row in (interface, workload, terminal)
        ):
            raise ValueError("E0 compatibility source lineage differs")
        if (
            interface.tokenizer_sha256 != workload.tokenizer_sha256
            or terminal.tokenizer_sha256 != interface.tokenizer_sha256
            or terminal.interface_sha256 != interface.interface_sha256
            or terminal.task_native_workload_sha256
            != workload.task_native_workload_sha256
        ):
            raise ValueError(
                "E0 compatibility model/tokenizer/workload identity differs"
            )
        if interface.schema_version != terminal.schema_version:
            raise ValueError("E0 compatibility interface/terminal schema differs")
        if interface.schema_version in {2, 3}:
            launch_sha256 = (
                None
                if interface.compile_launch_manifest is None
                else interface.compile_launch_manifest.semantic_sha256
            )
            proof_row_sha256 = None
            if interface.backend == "EAGLE3" and terminal.disposition == "VALID":
                proof_row_sha256 = e0_eagle3_runtime_proof_row_for_task(
                    interface,
                    task=task,
                    terminal=terminal,
                ).sha256
            if (
                terminal.interface_receipt_sha256 != interface.sha256
                or terminal.compile_launch_manifest_sha256 != launch_sha256
                or terminal.eagle3_runtime_proof_row_sha256 != proof_row_sha256
                or (
                    interface.schema_version == 3
                    and (terminal.eagle3_runtime_proof_row is None)
                    != (proof_row_sha256 is None)
                )
            ):
                raise ValueError(
                    "trusted E0 terminal differs from its exact interface launch"
                )
        if (
            terminal.terminal_status != "COMPLETE"
            or terminal.exit_code != 0
            or terminal.junit_status != "PASS"
        ):
            raise RuntimeError("E0 compatibility probe did not complete successfully")
        if interface.support_status == "UNSUPPORTED":
            expected_disposition = "N/A"
            expected_reason = "MODEL_BACKEND_INTERFACE_UNSUPPORTED"
            expected_smoke = "NOT_REQUIRED"
            expected_requests = 0
        elif workload.support_status == "UNSUPPORTED":
            expected_disposition = "N/A"
            expected_reason = "TOKENIZER_TASK_WORKLOAD_UNSUPPORTED"
            expected_smoke = "NOT_REQUIRED"
            expected_requests = 0
        elif terminal.smoke_status == "REGISTERED_UNSUPPORTED":
            if not interface.requires_gpu_smoke:
                raise ValueError("non-GPU E0 probe cannot register GPU incompatibility")
            expected_disposition = "N/A"
            expected_reason = "GPU_SMOKE_REGISTERED_UNSUPPORTED"
            expected_smoke = "REGISTERED_UNSUPPORTED"
            expected_requests = 0
        else:
            expected_disposition = "VALID"
            expected_reason = E0_COMPATIBILITY_VALID_REASON
            expected_smoke = "PASS" if interface.requires_gpu_smoke else "NOT_REQUIRED"
            expected_requests = 1 if interface.requires_gpu_smoke else 0
        if (
            terminal.disposition != expected_disposition
            or terminal.reason_code != expected_reason
            or terminal.smoke_status != expected_smoke
            or terminal.completed_request_count != expected_requests
        ):
            raise ValueError("E0 probe result differs from code-owned decision rule")
        decisions.append(
            E0CompatibilityDecision(
                model=model,
                backend=backend,
                task=task,
                disposition=expected_disposition,  # type: ignore[arg-type]
                reason_code=expected_reason,
                interface_sha256=interface.interface_sha256,
                task_native_workload_sha256=(workload.task_native_workload_sha256),
            )
        )
    compatibility = E0CompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_e6_receipt_sha256=e6_materialization_sha256,
        decisions=tuple(sorted(decisions, key=lambda row: row.decision_id)),
    )
    manifest = E0CompatibilityEvidenceManifest(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_e6_materialization_sha256=e6_materialization_sha256,
        upstream_e6_confirmation_sha256=e6_confirmation_sha256,
        started_ns=min(row.started_ns for row in probe_terminals),
        finished_ns=max(row.finished_ns for row in probe_terminals),
        interface_receipt_sha256s=tuple(
            sorted(row.sha256 for row in interface_receipts)
        ),
        workload_authority_sha256s=tuple(
            sorted(row.sha256 for row in workload_authorities)
        ),
        probe_terminal_sha256s=tuple(sorted(row.sha256 for row in probe_terminals)),
    )
    if compatibility.valid_count:
        if type(onlinespec_source_authority) is not E0OnlineSpecSourceAuthority:
            raise ValueError("VALID E0 decisions require OnlineSPEC source authority")
        onlinespec_source_authority.revalidate()
        authority_value: object = e0_onlinespec_source_authority_to_dict(
            onlinespec_source_authority
        )
        authority_sha256: object = onlinespec_source_authority.sha256
    else:
        if onlinespec_source_authority is not None:
            raise ValueError("all-N/A E0 publication cannot claim OnlineSPEC source")
        authority_value = None
        authority_sha256 = None
    bundle: dict[str, object] = {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_compatibility_bundle",
        "protocol_lock_sha256": protocol_lock.sha256,
        "upstream_e6_materialization_sha256": e6_materialization_sha256,
        "upstream_e6_confirmation_sha256": e6_confirmation_sha256,
        "compatibility": e0_compatibility_receipt_to_dict(compatibility),
        "compatibility_sha256": compatibility.sha256,
        "compatibility_evidence_manifest_sha256": manifest.sha256,
        "onlinespec_source_authority": authority_value,
        "onlinespec_source_authority_sha256": authority_sha256,
        "started_ns": manifest.started_ns,
        "finished_ns": manifest.finished_ns,
    }
    bundle["bundle_sha256"] = content_sha256(bundle)
    return E0CompatibilityPublication(compatibility, manifest, bundle)


def load_e0_prepared_model_backend_interface_receipt(
    path: str | Path,
) -> E0PreparedModelBackendInterfaceReceipt:
    """Deep-open one canonical interface receipt and all schema-2 launch inputs."""

    binding = CanonicalJsonProofBinding.bind(path)
    receipt = E0PreparedModelBackendInterfaceReceipt.from_dict(binding.reopen())
    if binding.semantic_sha256 != receipt.sha256:
        raise ValueError("E0 interface receipt binding differs")
    return receipt


def load_e0_compatibility_probe_terminal(
    path: str | Path,
) -> E0CompatibilityProbeTerminal:
    """Deep-open one canonical compatibility terminal."""

    binding = CanonicalJsonProofBinding.bind(path)
    terminal = E0CompatibilityProbeTerminal.from_dict(binding.reopen())
    if binding.semantic_sha256 != terminal.sha256:
        raise ValueError("E0 compatibility terminal binding differs")
    return terminal


def load_e0_task_native_workload_authority(
    path: str | Path,
) -> E0TaskNativeWorkloadAuthority:
    """Deep-open one canonical model/tokenizer/task workload authority."""

    binding = CanonicalJsonProofBinding.bind(path)
    authority = E0TaskNativeWorkloadAuthority.from_dict(binding.reopen())
    if binding.semantic_sha256 != authority.sha256:
        raise ValueError("E0 task-native workload authority binding differs")
    return authority


def e0_eagle3_runtime_proof_row_for_task(
    receipt: E0PreparedModelBackendInterfaceReceipt,
    *,
    task: str,
    terminal: E0CompatibilityProbeTerminal | None = None,
) -> E0Eagle3RuntimeProofRow:
    """Resolve one legacy pre-bound or fresh terminal-bound EAGLE3 row."""

    if (
        type(receipt) is not E0PreparedModelBackendInterfaceReceipt
        or receipt.schema_version not in {2, 3}
        or receipt.backend != "EAGLE3"
        or receipt.support_status != "READY"
        or task not in E0_TASKS
    ):
        raise ValueError("E0 EAGLE3 task authority is unavailable")
    if receipt.schema_version == 2:
        if terminal is not None and terminal.schema_version != 2:
            raise ValueError("legacy EAGLE3 terminal schema differs")
        matches = tuple(
            row for row in receipt.eagle3_runtime_proof_rows if row.task == task
        )
    else:
        if (
            type(terminal) is not E0CompatibilityProbeTerminal
            or terminal.schema_version != 3
            or terminal.task != task
            or terminal.model != receipt.model
            or terminal.backend != "EAGLE3"
            or terminal.disposition != "VALID"
            or terminal.interface_receipt_sha256 != receipt.sha256
            or type(terminal.eagle3_runtime_proof_row) is not CanonicalJsonProofBinding
        ):
            raise ValueError("fresh EAGLE3 terminal authority is unavailable")
        row = E0Eagle3RuntimeProofRow.from_dict(
            terminal.eagle3_runtime_proof_row.reopen()
        )
        if (
            row.schema_version != 2
            or row.sha256 != terminal.eagle3_runtime_proof_row_sha256
            or terminal.eagle3_runtime_proof_row.semantic_sha256 != row.sha256
        ):
            raise ValueError("fresh EAGLE3 terminal authority differs")
        matches = (row,)
    if len(matches) != 1:
        raise ValueError("E0 EAGLE3 task authority is not unique")
    selected = matches[0]
    if receipt.schema_version == 3:
        execution = selected.execution_authority.reopen()
        if (
            type(execution) is not dict
            or execution.get("model") != receipt.model
            or execution.get("interface_sha256") != receipt.interface_sha256
            or execution.get("target_revision") != receipt.target_revision
            or execution.get("drafter_revision") != receipt.drafter_revision
        ):
            raise ValueError("fresh EAGLE3 row/interface scope differs")
    return selected


def e0_eagle3_runtime_authority_for_task(
    receipt: E0PreparedModelBackendInterfaceReceipt,
    *,
    task: str,
    terminal: E0CompatibilityProbeTerminal | None = None,
) -> dict[str, str]:
    """Project the four exact adaptive-runtime claims for one E0 task."""

    row = e0_eagle3_runtime_proof_row_for_task(
        receipt,
        task=task,
        terminal=terminal,
    )
    return {
        "eagle3_e0_execution_authority_sha256": row.execution_authority_sha256,
        "eagle3_compatibility_authority_sha256": (row.compatibility_authority_sha256),
        "eagle3_model_selector_sha256": row.model_selector_sha256,
        "eagle3_native_gpu_proof_sha256": row.native_gpu_proof_sha256,
    }


def _e0_eagle3_native_scope(
    row: E0Eagle3RuntimeProofRow,
) -> tuple[str, str, tuple[str, ...]]:
    if row.schema_version == 1:
        from lightcone_spec.runtime.readiness import NativeRuntimeGpuProofReceipt

        native = NativeRuntimeGpuProofReceipt.from_dict(row.native_gpu_proof.reopen())
        return native.source_identity_sha256, native.inventory_sha256, native.gpu_uuids
    native = row.native_gpu_proof.reopen()
    if type(native) is not dict:
        raise ValueError("fresh EAGLE3 native proof is not an object")
    source = _sha256(
        "fresh EAGLE3 native source identity",
        native.get("source_identity_sha256"),
    )
    inventory = _sha256(
        "fresh EAGLE3 native inventory",
        native.get("inventory_sha256"),
    )
    raw_gpus = native.get("gpu_uuids")
    if (
        type(raw_gpus) is not list
        or len(raw_gpus) != 1
        or type(raw_gpus[0]) is not str
        or not raw_gpus[0].startswith("GPU-")
    ):
        raise ValueError("fresh EAGLE3 native GPU scope differs")
    return source, inventory, tuple(raw_gpus)


@dataclass(frozen=True)
class TrustedSingleOperatorEagle3ExecutionAuthority:
    """Path-replayed EAGLE3 authority for empirical single-operator runs.

    This explicitly is not the repository's signed ``MEASURED`` token.  Its
    authority comes only from reopening the current schema-2 compatibility
    auxiliary and every task-keyed proof path immediately before launch.
    """

    schema_version: Literal[1]
    kind: Literal["trusted_single_operator_eagle3_execution_authority"]
    trust_mode: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measured_authorization: Literal[False]
    execution_source: CanonicalJsonProofBinding
    execution_source_sha256: str
    materialized_cell_id: str
    compile_launch_manifest: CanonicalJsonProofBinding
    interface_receipt: CanonicalJsonProofBinding
    compatibility_terminal: CanonicalJsonProofBinding
    execution_authority: CanonicalJsonProofBinding
    compatibility_authority: CanonicalJsonProofBinding
    model_selector_authority: CanonicalJsonProofBinding
    native_gpu_receipt: CanonicalJsonProofBinding
    proof_row_sha256: str
    model: str
    backend: Literal["EAGLE3"]
    task: str
    method: str
    target_revision: str
    drafter_revision: str
    inventory_sha256: str
    gpu_uuids: tuple[str, ...]
    eagle3_e0_execution_authority_sha256: str
    eagle3_compatibility_authority_sha256: str
    eagle3_model_selector_sha256: str
    eagle3_native_gpu_proof_sha256: str
    native_source_identity_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "trusted_single_operator_eagle3_execution_authority"
            or self.trust_mode != "trusted_single_operator_empirical_no_signature"
            or self.formal_measured_authorization is not False
            or self.backend != "EAGLE3"
        ):
            raise ValueError("trusted EAGLE3 empirical authority schema differs")
        for label, value in (
            ("execution source", self.execution_source_sha256),
            ("materialized cell", self.materialized_cell_id),
            ("proof row", self.proof_row_sha256),
            ("inventory", self.inventory_sha256),
            ("execution authority", self.eagle3_e0_execution_authority_sha256),
            ("compatibility authority", self.eagle3_compatibility_authority_sha256),
            ("model selector", self.eagle3_model_selector_sha256),
            ("native GPU receipt", self.eagle3_native_gpu_proof_sha256),
            ("native source identity", self.native_source_identity_sha256),
        ):
            _sha256(f"trusted EAGLE3 {label}", value)
        for label, binding in (
            ("execution source", self.execution_source),
            ("compile launch", self.compile_launch_manifest),
            ("interface receipt", self.interface_receipt),
            ("compatibility terminal", self.compatibility_terminal),
            ("execution authority", self.execution_authority),
            ("compatibility authority", self.compatibility_authority),
            ("model selector", self.model_selector_authority),
            ("native GPU receipt", self.native_gpu_receipt),
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError(f"trusted EAGLE3 {label} is not path-bound")
            if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
                raise ValueError(f"trusted EAGLE3 {label} changed")
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != 1
            or len(set(self.gpu_uuids)) != 1
            or self.model not in E0_MODELS
            or self.task not in E0_TASKS
            or not self.method
        ):
            raise ValueError("trusted EAGLE3 empirical scope differs")

        from lightcone_spec.config import load_run_config
        from lightcone_spec.experiments.formal_registry import (
            stage_materialization_receipt_from_dict,
        )
        from lightcone_spec.experiments.formal_single_operator_stages import (
            load_formal_single_operator_execution_source,
        )
        from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

        source = load_formal_single_operator_execution_source(
            self.execution_source.absolute_path
        )
        materialization = stage_materialization_receipt_from_dict(
            source.materialization_source.reopen()
        )
        cells = {cell.cell_id: cell for cell in materialization.cells}
        cell = cells.get(self.materialized_cell_id)
        if cell is None:
            raise ValueError("trusted EAGLE3 cell is outside current materialization")
        auxiliary = source.auxiliary_source_binding("e0_compatibility")
        publication = revalidate_trusted_e0_compatibility_bundle_value(
            auxiliary.reopen(label="trusted EAGLE3 compatibility auxiliary")
        )
        if (
            self.execution_source.semantic_sha256
            != content_sha256(self.execution_source.reopen())
            or source.sha256 != self.execution_source_sha256
            or source.stage != "E0"
            or (cell.stage, cell.model, cell.backend, cell.task)
            != ("E0", self.model, "EAGLE3", self.task)
            or self.interface_receipt
            not in publication.evidence_manifest.interface_receipts
            or self.compatibility_terminal
            not in publication.evidence_manifest.probe_terminals
        ):
            raise ValueError("trusted EAGLE3 current-source lineage differs")
        decisions = tuple(
            row
            for row in publication.compatibility.decisions
            if (row.model, row.backend, row.task) == (self.model, "EAGLE3", self.task)
        )
        interface = load_e0_prepared_model_backend_interface_receipt(
            self.interface_receipt.absolute_path
        )
        terminal = load_e0_compatibility_probe_terminal(
            self.compatibility_terminal.absolute_path
        )
        if interface.schema_version == 2:
            proof_rows = tuple(
                row
                for row in interface.eagle3_runtime_proof_rows
                if row.task == self.task
            )
            if len(proof_rows) != 1:
                raise ValueError("trusted EAGLE3 task proof row is not unique")
            proof_row = proof_rows[0]
        else:
            proof_row = e0_eagle3_runtime_proof_row_for_task(
                interface,
                task=self.task,
                terminal=terminal,
            )
        launch = CompileLaunchManifest.load(self.compile_launch_manifest.absolute_path)
        config = load_run_config(launch.run_config_path)
        adaptation = config.adaptation
        if interface.schema_version == 2:
            from lightcone_spec.runtime.readiness import NativeRuntimeGpuProofReceipt

            legacy_native = NativeRuntimeGpuProofReceipt.from_dict(
                proof_row.native_gpu_proof.reopen()
            )
            native_source = legacy_native.source_identity_sha256
            native_inventory = legacy_native.inventory_sha256
            native_gpus = legacy_native.gpu_uuids
            native_claim_sha256 = legacy_native.sha256
            claims = e0_eagle3_runtime_authority_for_task(
                interface,
                task=self.task,
            )
        else:
            native_source, native_inventory, native_gpus = _e0_eagle3_native_scope(
                proof_row
            )
            native_claim_sha256 = proof_row.native_gpu_proof.semantic_sha256
            claims = e0_eagle3_runtime_authority_for_task(
                interface,
                task=self.task,
                terminal=terminal,
            )
        if (
            len(decisions) != 1
            or decisions[0].disposition != "VALID"
            or interface.schema_version not in {2, 3}
            or interface.support_status != "READY"
            or terminal.schema_version != interface.schema_version
            or terminal.disposition != "VALID"
            or terminal.interface_receipt_sha256 != interface.sha256
            or terminal.eagle3_runtime_proof_row_sha256 != proof_row.sha256
            or proof_row.sha256 != self.proof_row_sha256
            or proof_row.execution_authority != self.execution_authority
            or proof_row.compatibility_authority != self.compatibility_authority
            or proof_row.model_selector_authority != self.model_selector_authority
            or proof_row.native_gpu_proof != self.native_gpu_receipt
            or self.compile_launch_manifest.semantic_sha256 != launch.sha256
            or launch.formal_stage != "E0"
            or config.model.algorithm != "EAGLE3"
            or config.model.target != self.model
            or config.model.target_revision != self.target_revision
            or config.model.drafter_revision != self.drafter_revision
            or config.method != self.method
            or adaptation is None
            or launch.inventory_sha256 != self.inventory_sha256
            or launch.gpu_uuids != self.gpu_uuids
            or native_claim_sha256 != self.eagle3_native_gpu_proof_sha256
            or native_source != self.native_source_identity_sha256
            or native_inventory != self.inventory_sha256
            or native_gpus != self.gpu_uuids
            or source.content_source_binding is None
            or launch.content_source_binding != source.content_source_binding
            or any(getattr(adaptation, name) != value for name, value in claims.items())
            or claims
            != {
                "eagle3_e0_execution_authority_sha256": (
                    self.eagle3_e0_execution_authority_sha256
                ),
                "eagle3_compatibility_authority_sha256": (
                    self.eagle3_compatibility_authority_sha256
                ),
                "eagle3_model_selector_sha256": self.eagle3_model_selector_sha256,
                "eagle3_native_gpu_proof_sha256": (self.eagle3_native_gpu_proof_sha256),
            }
        ):
            raise ValueError("trusted EAGLE3 empirical replay differs")
        execution = self.execution_authority.reopen()
        if (
            type(execution) is not dict
            or execution.get("task") != self.task
            or execution.get("target_revision") != self.target_revision
            or execution.get("drafter_revision") != self.drafter_revision
            or execution.get("interface_sha256") != decisions[0].interface_sha256
            or execution.get("inventory_sha256") != self.inventory_sha256
            or tuple(execution.get("gpu_uuids", ())) != self.gpu_uuids
        ):
            raise ValueError("trusted EAGLE3 execution projection differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["gpu_uuids"] = list(self.gpu_uuids)
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("trusted EAGLE3 empirical authority fields differ")
        row = dict(value)
        for name in (
            "execution_source",
            "compile_launch_manifest",
            "interface_receipt",
            "compatibility_terminal",
            "execution_authority",
            "compatibility_authority",
            "model_selector_authority",
            "native_gpu_receipt",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        raw_gpus = row.pop("gpu_uuids")
        if type(raw_gpus) is not list:
            raise TypeError("trusted EAGLE3 GPU UUIDs must be an array")
        return cls(**row, gpu_uuids=tuple(raw_gpus))  # type: ignore[arg-type]


def derive_trusted_single_operator_eagle3_execution_authority(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    compile_launch_manifest_path: str | Path,
) -> TrustedSingleOperatorEagle3ExecutionAuthority:
    """Derive the tagged authority without accepting scientific values."""

    from lightcone_spec.config import load_run_config
    from lightcone_spec.experiments.formal_registry import (
        stage_materialization_receipt_from_dict,
    )
    from lightcone_spec.experiments.formal_single_operator_stages import (
        load_formal_single_operator_execution_source,
    )
    from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

    source_binding = CanonicalJsonProofBinding.bind(execution_source_path)
    source = load_formal_single_operator_execution_source(source_binding.absolute_path)
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen()
    )
    cells = {cell.cell_id: cell for cell in materialization.cells}
    cell = cells.get(materialized_cell_id)
    if cell is None:
        raise ValueError("trusted EAGLE3 cell is outside current materialization")
    auxiliary = source.auxiliary_source_binding("e0_compatibility")
    publication = revalidate_trusted_e0_compatibility_bundle_value(
        auxiliary.reopen(label="trusted EAGLE3 derivation auxiliary")
    )
    interface_bindings = tuple(
        binding
        for binding in publication.evidence_manifest.interface_receipts
        if (raw := binding.reopen()).get("model") == cell.model
        and raw.get("backend") == "EAGLE3"
    )
    terminal_bindings = tuple(
        binding
        for binding in publication.evidence_manifest.probe_terminals
        if (raw := binding.reopen()).get("model") == cell.model
        and raw.get("backend") == "EAGLE3"
        and raw.get("task") == cell.task
    )
    if len(interface_bindings) != 1 or len(terminal_bindings) != 1:
        raise ValueError("trusted EAGLE3 derivation path coverage differs")
    interface = load_e0_prepared_model_backend_interface_receipt(
        interface_bindings[0].absolute_path
    )
    terminal = load_e0_compatibility_probe_terminal(terminal_bindings[0].absolute_path)
    proof_row = e0_eagle3_runtime_proof_row_for_task(
        interface,
        task=cell.task,
        terminal=terminal,
    )
    launch_binding = CanonicalJsonProofBinding.bind(compile_launch_manifest_path)
    launch = CompileLaunchManifest.load(launch_binding.absolute_path)
    config = load_run_config(launch.run_config_path)
    native_source, native_inventory, native_gpus = _e0_eagle3_native_scope(proof_row)
    claims = e0_eagle3_runtime_authority_for_task(
        interface,
        task=cell.task,
        terminal=terminal,
    )
    if native_inventory != launch.inventory_sha256 or native_gpus != launch.gpu_uuids:
        raise ValueError("trusted EAGLE3 derivation launch/native scope differs")
    return TrustedSingleOperatorEagle3ExecutionAuthority(
        schema_version=1,
        kind="trusted_single_operator_eagle3_execution_authority",
        trust_mode="trusted_single_operator_empirical_no_signature",
        formal_measured_authorization=False,
        execution_source=source_binding,
        execution_source_sha256=source.sha256,
        materialized_cell_id=cell.cell_id,
        compile_launch_manifest=launch_binding,
        interface_receipt=interface_bindings[0],
        compatibility_terminal=terminal_bindings[0],
        execution_authority=proof_row.execution_authority,
        compatibility_authority=proof_row.compatibility_authority,
        model_selector_authority=proof_row.model_selector_authority,
        native_gpu_receipt=proof_row.native_gpu_proof,
        proof_row_sha256=proof_row.sha256,
        model=cell.model,
        backend="EAGLE3",
        task=cell.task,
        method=config.method,
        target_revision=config.model.target_revision,
        drafter_revision=config.model.drafter_revision,
        inventory_sha256=launch.inventory_sha256,
        gpu_uuids=launch.gpu_uuids,
        native_source_identity_sha256=native_source,
        **claims,
    )


def publish_trusted_single_operator_eagle3_execution_authority(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    compile_launch_manifest_path: str | Path,
    output_path: str | Path,
) -> TrustedSingleOperatorEagle3ExecutionAuthority:
    authority = derive_trusted_single_operator_eagle3_execution_authority(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
        compile_launch_manifest_path=compile_launch_manifest_path,
    )
    publish_canonical_json_no_replace(output_path, authority.to_dict())
    rebound = load_trusted_single_operator_eagle3_execution_authority(output_path)
    if rebound != authority:
        raise RuntimeError("trusted EAGLE3 authority changed during publication")
    return rebound


def load_trusted_single_operator_eagle3_execution_authority(
    path: str | Path,
) -> TrustedSingleOperatorEagle3ExecutionAuthority:
    binding = CanonicalJsonProofBinding.bind(path)
    authority = TrustedSingleOperatorEagle3ExecutionAuthority.from_dict(
        binding.reopen()
    )
    if binding.semantic_sha256 != authority.sha256:
        raise ValueError("trusted EAGLE3 authority binding differs")
    return authority


def publish_e0_prepared_model_backend_interface_receipt(
    receipt: E0PreparedModelBackendInterfaceReceipt,
    *,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish one source-produced interface receipt without replacement."""

    if type(
        receipt
    ) is not E0PreparedModelBackendInterfaceReceipt or receipt.schema_version not in {
        2,
        3,
    }:
        raise TypeError("trusted E0 interface publisher requires schema 2 or 3")
    publish_canonical_json_no_replace(output_path, receipt.to_dict())
    rebound = load_e0_prepared_model_backend_interface_receipt(output_path)
    if rebound != receipt:
        raise RuntimeError("published E0 interface receipt changed")
    return CanonicalJsonProofBinding.bind(output_path, semantic_sha256=receipt.sha256)


def publish_e0_compatibility_probe_terminal(
    terminal: E0CompatibilityProbeTerminal,
    *,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish one source-produced compatibility terminal without replacement."""

    if type(
        terminal
    ) is not E0CompatibilityProbeTerminal or terminal.schema_version not in {2, 3}:
        raise TypeError("trusted E0 terminal publisher requires schema 2 or 3")
    publish_canonical_json_no_replace(output_path, terminal.to_dict())
    rebound = load_e0_compatibility_probe_terminal(output_path)
    if rebound != terminal:
        raise RuntimeError("published E0 compatibility terminal changed")
    return CanonicalJsonProofBinding.bind(output_path, semantic_sha256=terminal.sha256)


def publish_e0_task_native_workload_authority(
    authority: E0TaskNativeWorkloadAuthority,
    *,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish one exact model/tokenizer/task authority without replacement."""

    if type(authority) is not E0TaskNativeWorkloadAuthority:
        raise TypeError("E0 workload publisher requires an exact authority")
    publish_canonical_json_no_replace(output_path, authority.to_dict())
    rebound = load_e0_task_native_workload_authority(output_path)
    if rebound != authority:
        raise RuntimeError("published E0 task-native workload authority changed")
    return CanonicalJsonProofBinding.bind(output_path, semantic_sha256=authority.sha256)


def publish_trusted_e0_compatibility_probe_sources(
    *,
    protocol_lock: ProtocolLock,
    e6_completion: object,
    interface_receipt_paths: tuple[str | Path, ...],
    workload_authority_paths: tuple[str | Path, ...],
    probe_terminal_paths: tuple[str | Path, ...],
    onlinespec_source_authority: E0OnlineSpecSourceAuthority | None,
    bundle_output_path: str | Path,
    evidence_manifest_output_path: str | Path,
) -> E0CompatibilityPublication:
    """Publish one trusted schema-2/3 bundle from paths, never summary SHA alone."""

    interface_bindings = tuple(
        CanonicalJsonProofBinding.bind(path) for path in interface_receipt_paths
    )
    workload_bindings = tuple(
        CanonicalJsonProofBinding.bind(path) for path in workload_authority_paths
    )
    terminal_bindings = tuple(
        CanonicalJsonProofBinding.bind(path) for path in probe_terminal_paths
    )
    interfaces = tuple(
        load_e0_prepared_model_backend_interface_receipt(row.absolute_path)
        for row in interface_bindings
    )
    terminals = tuple(
        load_e0_compatibility_probe_terminal(row.absolute_path)
        for row in terminal_bindings
    )
    workload_authorities = tuple(
        load_e0_task_native_workload_authority(row.absolute_path)
        for row in workload_bindings
    )
    trusted_schema = {row.schema_version for row in (*interfaces, *terminals)}
    if (
        len(trusted_schema) != 1
        or next(iter(trusted_schema), None) not in {2, 3}
        or tuple(row.key for row in terminals)
        != tuple(sorted(row.key for row in terminals))
        or tuple((row.model, row.backend) for row in interfaces)
        != tuple(sorted((row.model, row.backend) for row in interfaces))
        or tuple((row.model, row.task) for row in workload_authorities)
        != tuple(sorted((row.model, row.task) for row in workload_authorities))
    ):
        raise ValueError("trusted E0 source paths are not one canonical trusted schema")
    publication_schema = next(iter(trusted_schema))
    legacy_projection = reduce_e0_compatibility_probes(
        protocol_lock=protocol_lock,
        e6_completion=e6_completion,
        interface_receipts=interfaces,
        workload_authorities=workload_authorities,
        probe_terminals=terminals,
        onlinespec_source_authority=onlinespec_source_authority,
    )
    legacy_manifest = legacy_projection.evidence_manifest
    manifest = E0CompatibilityEvidenceManifest(
        schema_version=publication_schema,  # type: ignore[arg-type]
        protocol_lock_sha256=legacy_manifest.protocol_lock_sha256,
        upstream_e6_materialization_sha256=(
            legacy_manifest.upstream_e6_materialization_sha256
        ),
        upstream_e6_confirmation_sha256=(
            legacy_manifest.upstream_e6_confirmation_sha256
        ),
        started_ns=legacy_manifest.started_ns,
        finished_ns=legacy_manifest.finished_ns,
        interface_receipt_sha256s=legacy_manifest.interface_receipt_sha256s,
        workload_authority_sha256s=legacy_manifest.workload_authority_sha256s,
        probe_terminal_sha256s=legacy_manifest.probe_terminal_sha256s,
        interface_receipts=interface_bindings,
        workload_authorities=workload_bindings,
        probe_terminals=terminal_bindings,
    )
    publish_canonical_json_no_replace(
        evidence_manifest_output_path,
        manifest.to_dict(),
    )
    manifest_binding = CanonicalJsonProofBinding.bind(
        evidence_manifest_output_path,
        semantic_sha256=manifest.sha256,
    )
    bundle = dict(legacy_projection.bundle)
    bundle.pop("bundle_sha256")
    bundle.update(
        {
            "schema_version": publication_schema,
            "compatibility_evidence_manifest": manifest_binding.to_dict(),
            "compatibility_evidence_manifest_sha256": manifest.sha256,
        }
    )
    bundle["bundle_sha256"] = content_sha256(bundle)
    publication = E0CompatibilityPublication(
        compatibility=legacy_projection.compatibility,
        evidence_manifest=manifest,
        bundle=bundle,
    )
    publish_canonical_json_no_replace(bundle_output_path, publication.bundle)
    rebound = revalidate_trusted_e0_compatibility_bundle(bundle_output_path)
    if rebound.bundle != publication.bundle:
        raise RuntimeError("published trusted E0 compatibility bundle changed")
    return rebound


def revalidate_trusted_e0_compatibility_bundle(
    path: str | Path,
) -> E0CompatibilityPublication:
    """Deep-replay one trusted bundle, manifest, and all 120 bound sources."""

    binding = CanonicalJsonProofBinding.bind(path)
    value = binding.reopen()
    publication = revalidate_trusted_e0_compatibility_bundle_value(value)
    if binding.semantic_sha256 != content_sha256(value):
        raise ValueError("trusted E0 compatibility bundle binding differs")
    return publication


def revalidate_trusted_e0_compatibility_bundle_value(
    value: object,
) -> E0CompatibilityPublication:
    """Deep-replay a trusted auxiliary value through its bound source paths."""

    if type(value) is not dict or value.get("schema_version") not in {2, 3}:
        raise ValueError("trusted E0 compatibility bundle requires schema 2 or 3")
    manifest_binding = CanonicalJsonProofBinding.from_dict(
        value.get("compatibility_evidence_manifest")
    )
    manifest = E0CompatibilityEvidenceManifest.from_dict(manifest_binding.reopen())
    from lightcone_spec.experiments.formal_registry import (
        e0_compatibility_receipt_from_dict,
    )

    compatibility = e0_compatibility_receipt_from_dict(value.get("compatibility"))
    publication = E0CompatibilityPublication(
        compatibility=compatibility,
        evidence_manifest=manifest,
        bundle=value,
    )
    if (
        compatibility.sha256 != value["compatibility_sha256"]
        or manifest.sha256 != value["compatibility_evidence_manifest_sha256"]
    ):
        raise ValueError("trusted E0 compatibility bundle binding differs")
    interfaces = tuple(
        load_e0_prepared_model_backend_interface_receipt(row.absolute_path)
        for row in manifest.interface_receipts
    )
    terminals = tuple(
        load_e0_compatibility_probe_terminal(row.absolute_path)
        for row in manifest.probe_terminals
    )
    interface_by_key = {(row.model, row.backend): row for row in interfaces}
    if len(interface_by_key) != 12:
        raise ValueError("trusted E0 interface source coverage differs")
    for terminal in terminals:
        interface = interface_by_key[(terminal.model, terminal.backend)]
        proof_row_sha256 = None
        if interface.backend == "EAGLE3" and terminal.disposition == "VALID":
            proof_row_sha256 = e0_eagle3_runtime_proof_row_for_task(
                interface,
                task=terminal.task,
                terminal=terminal,
            ).sha256
        if (
            terminal.interface_receipt_sha256 != interface.sha256
            or terminal.interface_sha256 != interface.interface_sha256
            or terminal.tokenizer_sha256 != interface.tokenizer_sha256
            or terminal.compile_launch_manifest_sha256
            != (
                None
                if interface.compile_launch_manifest is None
                else interface.compile_launch_manifest.semantic_sha256
            )
            or terminal.eagle3_runtime_proof_row_sha256 != proof_row_sha256
            or (
                interface.schema_version == 3
                and (terminal.eagle3_runtime_proof_row is None)
                != (proof_row_sha256 is None)
            )
        ):
            raise ValueError("trusted E0 terminal/interface replay differs")
    return publication


def publish_e0_compatibility_probes(
    publication: E0CompatibilityPublication,
    *,
    bundle_output_path: str | Path,
    evidence_manifest_output_path: str | Path,
) -> None:
    """Publish evidence first and the canonical bundle last, without overwrite."""

    if type(publication) is not E0CompatibilityPublication:
        raise TypeError("E0 compatibility publisher requires exact publication")
    evidence = {
        **asdict(publication.evidence_manifest),
        "evidence_manifest_sha256": publication.evidence_manifest.sha256,
    }
    _atomic_no_replace_json(evidence_manifest_output_path, evidence)
    _atomic_no_replace_json(bundle_output_path, publication.bundle)


def _atomic_no_replace_json(path: str | Path, value: object) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace E0 publication: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".partial",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "E0_BACKEND_REQUIRES_GPU_SMOKE",
    "E0_COMPATIBILITY_NA_REASONS",
    "E0_COMPATIBILITY_VALID_REASON",
    "E0_TRUSTED_MODEL_BACKEND_INTERFACE_PROTOCOL_SHA256",
    "E0_TRUSTED_PREPROBE_MODEL_BACKEND_INTERFACE_PROTOCOL_SHA256",
    "E0CompatibilityEvidenceManifest",
    "E0CompatibilityProbeTerminal",
    "E0CompatibilityPublication",
    "E0Eagle3RuntimeProofRow",
    "E0PreparedModelBackendInterfaceReceipt",
    "E0TaskNativeWorkloadAuthority",
    "TrustedSingleOperatorEagle3ExecutionAuthority",
    "derive_trusted_single_operator_eagle3_execution_authority",
    "e0_eagle3_runtime_authority_for_task",
    "e0_eagle3_runtime_proof_row_for_task",
    "e0_preprobe_interface_sha256",
    "load_e0_compatibility_probe_terminal",
    "load_e0_prepared_model_backend_interface_receipt",
    "load_e0_task_native_workload_authority",
    "load_trusted_single_operator_eagle3_execution_authority",
    "publish_e0_compatibility_probe_terminal",
    "publish_e0_compatibility_probes",
    "publish_e0_prepared_model_backend_interface_receipt",
    "publish_e0_task_native_workload_authority",
    "publish_trusted_e0_compatibility_probe_sources",
    "publish_trusted_single_operator_eagle3_execution_authority",
    "reduce_e0_compatibility_probes",
    "revalidate_trusted_e0_compatibility_bundle",
    "revalidate_trusted_e0_compatibility_bundle_value",
]
