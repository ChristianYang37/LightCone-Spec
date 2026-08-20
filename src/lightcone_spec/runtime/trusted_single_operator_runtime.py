"""Verifier-sealed runtime authority for trusted-v03 empirical execution.

This module deliberately does not turn unsigned qualification output into a
release proof.  A token produced here is usable only by the trusted
``formal_single_operator_v1`` lane, carries ``formal_measurement=False``, and
must be rebuilt from its complete path-bound evidence before every server
launch.  The patched SGLang runtime accepts the exact token type but cannot
construct one itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

TrustedRuntimeAuthorityKind = Literal[
    "preflight_qualification",
    "e6_nextn",
    "e0_eagle3",
]
TrustedRuntimeAuthorityRole = Literal["distributed", "native"]

TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "trusted_single_operator_runtime_authority",
        "trust_mode": "trusted_single_operator_empirical_no_signature",
        "formal_measurement": False,
        "qualification_only": False,
        "consumer": (
            "path_bound_current_execution_source_cell_run_config_launch_inventory"
        ),
        "evidence": (
            "deep_replayed_complete_empirical_result_assignment_terminal_junit"
        ),
        "roles": ["distributed", "native"],
        "claim": "UNMEASURED",
    }
)

_VERIFIED_TRUSTED_RUNTIME_AUTHORITY_SENTINEL = object()
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_ENVIRONMENT = (
    "LIGHTCONE_FORMAL_RUNTIME_AUTHORITY_SOURCE_PATH",
    "LIGHTCONE_FORMAL_RUNTIME_AUTHORITY_SOURCE_RAW_SHA256",
    "LIGHTCONE_FORMAL_RUNTIME_AUTHORITY_SOURCE_SEMANTIC_SHA256",
    "LIGHTCONE_FORMAL_RUNTIME_AUTHORITY_SOURCE_SIZE",
    "LIGHTCONE_FORMAL_RUNTIME_AUTHORITY_SOURCE_KIND",
)


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or any(character in value for character in ("\n", "\r", "\x00"))
    ):
        raise ValueError(f"{label} must be canonical single-line text")
    return value


@dataclass(frozen=True, init=False)
class VerifiedTrustedSingleOperatorRuntimeGpuAuthority:
    """Opaque, non-release GPU runtime token issued by the deep revalidator."""

    schema_version: Literal[1]
    kind: Literal["verified_trusted_single_operator_runtime_gpu_authority"]
    protocol_sha256: str
    trust_mode: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measurement: Literal[False]
    qualification_only: Literal[False]
    role: TrustedRuntimeAuthorityRole
    authority_kind: TrustedRuntimeAuthorityKind
    source_suite_id: str
    authority_source_sha256: str
    consumer_identity_sha256: str
    evidence_sha256s: tuple[str, ...]
    receipt_sha256: str
    source_capability_sha256: str
    role_source_identity_sha256: str
    source_identity_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    topology_sha256: str
    gpu_uuids: tuple[str, ...]
    backend_capabilities: tuple[str, ...]

    def __init__(
        self,
        *,
        role: TrustedRuntimeAuthorityRole,
        authority_kind: TrustedRuntimeAuthorityKind,
        source_suite_id: str,
        authority_source_sha256: str,
        consumer_identity_sha256: str,
        evidence_sha256s: tuple[str, ...],
        receipt_sha256: str,
        source_capability_sha256: str,
        role_source_identity_sha256: str,
        source_identity_sha256: str,
        inventory_sha256: str,
        hardware_envelope_sha256: str,
        topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"],
        topology_sha256: str,
        gpu_uuids: tuple[str, ...],
        backend_capabilities: tuple[str, ...],
        _verification_tag: object,
    ) -> None:
        if _verification_tag is not _VERIFIED_TRUSTED_RUNTIME_AUTHORITY_SENTINEL:
            raise TypeError(
                "trusted runtime authority can only come from source revalidation"
            )
        values = {
            "schema_version": 1,
            "kind": "verified_trusted_single_operator_runtime_gpu_authority",
            "protocol_sha256": (
                TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_PROTOCOL_SHA256
            ),
            "trust_mode": "trusted_single_operator_empirical_no_signature",
            "formal_measurement": False,
            "qualification_only": False,
            "role": role,
            "authority_kind": authority_kind,
            "source_suite_id": source_suite_id,
            "authority_source_sha256": authority_source_sha256,
            "consumer_identity_sha256": consumer_identity_sha256,
            "evidence_sha256s": evidence_sha256s,
            "receipt_sha256": receipt_sha256,
            "source_capability_sha256": source_capability_sha256,
            "role_source_identity_sha256": role_source_identity_sha256,
            "source_identity_sha256": source_identity_sha256,
            "inventory_sha256": inventory_sha256,
            "hardware_envelope_sha256": hardware_envelope_sha256,
            "topology_mode": topology_mode,
            "topology_sha256": topology_sha256,
            "gpu_uuids": gpu_uuids,
            "backend_capabilities": backend_capabilities,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self._validate()

    def _validate(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "verified_trusted_single_operator_runtime_gpu_authority"
            or self.protocol_sha256
            != TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_PROTOCOL_SHA256
            or self.trust_mode != "trusted_single_operator_empirical_no_signature"
            or self.formal_measurement is not False
            or self.qualification_only is not False
            or self.role not in {"distributed", "native"}
            or self.authority_kind
            not in {"preflight_qualification", "e6_nextn", "e0_eagle3"}
            or self.topology_mode not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}
        ):
            raise ValueError("trusted runtime authority identity differs")
        _require_text("trusted runtime suite", self.source_suite_id)
        for label, value in (
            ("protocol", self.protocol_sha256),
            ("authority source", self.authority_source_sha256),
            ("consumer identity", self.consumer_identity_sha256),
            ("receipt", self.receipt_sha256),
            ("source capability", self.source_capability_sha256),
            ("role source identity", self.role_source_identity_sha256),
            ("common source identity", self.source_identity_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
            ("topology", self.topology_sha256),
        ):
            _require_sha256(f"trusted runtime {label}", value)
        if (
            type(self.evidence_sha256s) is not tuple
            or not self.evidence_sha256s
            or self.evidence_sha256s != tuple(sorted(set(self.evidence_sha256s)))
        ):
            raise ValueError("trusted runtime evidence set is not canonical")
        for digest in self.evidence_sha256s:
            _require_sha256("trusted runtime evidence", digest)
        expected_gpu_count = 1 if self.topology_mode == "tp1_dp1" else 2
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != expected_gpu_count
            or len(set(self.gpu_uuids)) != expected_gpu_count
            or any(
                type(value) is not str or not value.startswith("GPU-")
                for value in self.gpu_uuids
            )
        ):
            raise ValueError("trusted runtime GPU coverage differs")
        if (
            type(self.backend_capabilities) is not tuple
            or self.backend_capabilities
            != tuple(sorted(set(self.backend_capabilities)))
            or (
                self.role == "distributed"
                and (self.topology_mode == "tp1_dp1" or self.backend_capabilities)
            )
            or (self.role == "native" and not self.backend_capabilities)
        ):
            raise ValueError("trusted runtime role capabilities differ")

    @property
    def sha256(self) -> str:
        return content_sha256(asdict(self))


@dataclass(frozen=True)
class TrustedSingleOperatorRuntimeRoleSource:
    role: TrustedRuntimeAuthorityRole
    source_suite_id: str
    source_capability_sha256: str
    role_source_identity_sha256: str
    evidence_sha256s: tuple[str, ...]
    backend_capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in {"distributed", "native"}:
            raise ValueError("trusted runtime source role differs")
        _require_text("trusted runtime source suite", self.source_suite_id)
        for label, value in (
            ("source capability", self.source_capability_sha256),
            ("role source identity", self.role_source_identity_sha256),
        ):
            _require_sha256(f"trusted runtime source {label}", value)
        if (
            type(self.evidence_sha256s) is not tuple
            or not self.evidence_sha256s
            or self.evidence_sha256s != tuple(sorted(set(self.evidence_sha256s)))
        ):
            raise ValueError("trusted runtime source evidence is not canonical")
        for digest in self.evidence_sha256s:
            _require_sha256("trusted runtime source evidence", digest)
        if (
            type(self.backend_capabilities) is not tuple
            or self.backend_capabilities
            != tuple(sorted(set(self.backend_capabilities)))
            or (self.role == "distributed" and self.backend_capabilities)
            or (self.role == "native" and not self.backend_capabilities)
        ):
            raise ValueError("trusted runtime source capabilities differ")

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "source_suite_id": self.source_suite_id,
            "source_capability_sha256": self.source_capability_sha256,
            "role_source_identity_sha256": self.role_source_identity_sha256,
            "evidence_sha256s": list(self.evidence_sha256s),
            "backend_capabilities": list(self.backend_capabilities),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("trusted runtime role source fields differ")
        row = dict(value)
        for name in ("evidence_sha256s", "backend_capabilities"):
            raw = row[name]
            if type(raw) is not list:
                raise TypeError(f"trusted runtime role {name} must be an array")
            row[name] = tuple(raw)
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class TrustedSingleOperatorRuntimeAuthoritySource:
    """Path-only source that must be rebuilt immediately before SGLang import."""

    schema_version: Literal[1]
    kind: Literal["trusted_single_operator_runtime_authority_source"]
    protocol_sha256: str
    trust_mode: Literal["trusted_single_operator_empirical_no_signature"]
    formal_measurement: Literal[False]
    authority_kind: TrustedRuntimeAuthorityKind
    algorithm: Literal["DFLASH", "DSPARK", "NEXTN", "EAGLE3"]
    consumer_source: CanonicalJsonProofBinding
    execution_source: CanonicalJsonProofBinding
    materialized_cell_id: str
    launch_manifest: CanonicalJsonProofBinding
    preflight_inputs: CanonicalJsonProofBinding
    authority_evidence: CanonicalJsonProofBinding
    authority_sha256: str
    consumer_identity_sha256: str
    source_identity_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
    topology_sha256: str
    gpu_uuids: tuple[str, ...]
    roles: tuple[TrustedSingleOperatorRuntimeRoleSource, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "trusted_single_operator_runtime_authority_source"
            or self.protocol_sha256
            != TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_PROTOCOL_SHA256
            or self.trust_mode != "trusted_single_operator_empirical_no_signature"
            or self.formal_measurement is not False
            or self.authority_kind
            not in {"preflight_qualification", "e6_nextn", "e0_eagle3"}
            or self.algorithm not in {"DFLASH", "DSPARK", "NEXTN", "EAGLE3"}
            or self.topology_mode not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}
        ):
            raise ValueError("trusted runtime authority source schema differs")
        for label, binding in (
            ("consumer", self.consumer_source),
            ("execution", self.execution_source),
            ("launch", self.launch_manifest),
            ("preflight", self.preflight_inputs),
            ("authority evidence", self.authority_evidence),
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError(f"trusted runtime {label} source is not path-bound")
            if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
                raise ValueError(f"trusted runtime {label} source changed")
        for label, value in (
            ("materialized cell", self.materialized_cell_id),
            ("authority", self.authority_sha256),
            ("consumer identity", self.consumer_identity_sha256),
            ("source identity", self.source_identity_sha256),
            ("inventory", self.inventory_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
            ("topology", self.topology_sha256),
        ):
            _require_sha256(f"trusted runtime source {label}", value)
        expected_gpu_count = 1 if self.topology_mode == "tp1_dp1" else 2
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != expected_gpu_count
            or len(set(self.gpu_uuids)) != expected_gpu_count
            or any(not value.startswith("GPU-") for value in self.gpu_uuids)
            or type(self.roles) is not tuple
            or not self.roles
            or tuple(row.role for row in self.roles)
            != tuple(sorted({row.role for row in self.roles}))
        ):
            raise ValueError("trusted runtime source GPU/role coverage differs")
        expected_roles = (
            ("native",)
            if self.algorithm in {"DSPARK", "EAGLE3"}
            and self.topology_mode == "tp1_dp1"
            else (
                ("distributed",)
                if self.algorithm == "DFLASH"
                else ("distributed", "native")
            )
        )
        if tuple(row.role for row in self.roles) != expected_roles:
            raise ValueError("trusted runtime source algorithm roles differ")
        if (
            (self.authority_kind == "e6_nextn") != (self.algorithm == "NEXTN")
            or (self.authority_kind == "e0_eagle3") != (self.algorithm == "EAGLE3")
            or (
                self.authority_kind == "preflight_qualification"
                and self.algorithm not in {"DFLASH", "DSPARK"}
            )
        ):
            raise ValueError("trusted runtime source authority/backend differs")

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for name in (
            "consumer_source",
            "execution_source",
            "launch_manifest",
            "preflight_inputs",
            "authority_evidence",
        ):
            value[name] = getattr(self, name).to_dict()
        value["gpu_uuids"] = list(self.gpu_uuids)
        value["roles"] = [row.to_dict() for row in self.roles]
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("trusted runtime authority source fields differ")
        row = dict(value)
        for name in (
            "consumer_source",
            "execution_source",
            "launch_manifest",
            "preflight_inputs",
            "authority_evidence",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        raw_gpus = row.pop("gpu_uuids")
        raw_roles = row.pop("roles")
        if type(raw_gpus) is not list or type(raw_roles) is not list:
            raise TypeError("trusted runtime source GPUs/roles must be arrays")
        return cls(
            **row,
            gpu_uuids=tuple(raw_gpus),
            roles=tuple(
                TrustedSingleOperatorRuntimeRoleSource.from_dict(item)
                for item in raw_roles
            ),
        )  # type: ignore[arg-type]


def trusted_single_operator_runtime_authority_environment(
    binding: CanonicalJsonProofBinding,
) -> dict[str, str]:
    """Return the exact child environment for one already-published source."""

    if type(binding) is not CanonicalJsonProofBinding or (
        CanonicalJsonProofBinding.bind(binding.absolute_path) != binding
    ):
        raise ValueError("trusted runtime authority environment binding changed")
    value = TrustedSingleOperatorRuntimeAuthoritySource.from_dict(binding.reopen())
    if value.sha256 != binding.semantic_sha256:
        raise ValueError("trusted runtime authority environment source differs")
    return {
        TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_ENVIRONMENT[0]: (
            binding.absolute_path
        ),
        TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_ENVIRONMENT[1]: binding.raw_sha256,
        TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_ENVIRONMENT[2]: (
            binding.semantic_sha256
        ),
        TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_ENVIRONMENT[3]: str(binding.size),
        TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_ENVIRONMENT[4]: value.kind,
    }


def bind_trusted_single_operator_runtime_authority_environment(
    environment: Mapping[str, str],
) -> CanonicalJsonProofBinding | None:
    """Bind an all-or-nothing child authority environment without issuing tokens."""

    if not isinstance(environment, Mapping):
        raise TypeError("trusted runtime authority environment is not a mapping")
    values = tuple(
        environment.get(name)
        for name in (TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_ENVIRONMENT)
    )
    if all(value is None for value in values):
        return None
    if any(type(value) is not str or not value for value in values):
        raise ValueError("trusted runtime authority environment is incomplete")
    path, raw_sha256, semantic_sha256, raw_size, kind = values
    assert isinstance(path, str)
    assert isinstance(raw_sha256, str)
    assert isinstance(semantic_sha256, str)
    assert isinstance(raw_size, str)
    assert isinstance(kind, str)
    if not raw_size.isdecimal() or str(int(raw_size)) != raw_size:
        raise ValueError("trusted runtime authority environment size is not canonical")
    binding = CanonicalJsonProofBinding.bind(path)
    if (
        binding.raw_sha256 != raw_sha256
        or binding.semantic_sha256 != semantic_sha256
        or binding.size != int(raw_size)
    ):
        raise ValueError("trusted runtime authority environment binding differs")
    value = TrustedSingleOperatorRuntimeAuthoritySource.from_dict(binding.reopen())
    if kind != value.kind or value.sha256 != binding.semantic_sha256:
        raise ValueError("trusted runtime authority environment identity differs")
    return binding


def _revalidate_consumer_source(binding: CanonicalJsonProofBinding) -> object:
    """Deep-reopen one of the four trusted serving consumer descriptors."""

    import time

    raw = binding.reopen()
    kind = raw.get("kind")
    if kind == "formal_single_operator_early_run_plan_inputs":
        from lightcone_spec.experiments.formal_single_operator_early_execution import (
            revalidate_formal_single_operator_early_run_plan_inputs,
        )

        value = revalidate_formal_single_operator_early_run_plan_inputs(
            binding.absolute_path
        )
    elif kind == "formal_single_operator_downstream_run_plan_inputs":
        from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
            FormalSingleOperatorDownstreamRunPlanInputs,
        )

        value = FormalSingleOperatorDownstreamRunPlanInputs.from_dict(raw)
    elif kind == "formal_single_operator_prepared_downstream_run_plan_inputs":
        from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
            revalidate_formal_single_operator_prepared_downstream_run_plan_inputs,
        )

        value = revalidate_formal_single_operator_prepared_downstream_run_plan_inputs(
            binding.absolute_path,
            current_ns=time.time_ns(),
        )
    elif kind == "formal_single_operator_e5_failure_execution_descriptor":
        from lightcone_spec.experiments.formal_failure_execution import (
            revalidate_formal_single_operator_e5_failure_execution_descriptor,
        )

        value = revalidate_formal_single_operator_e5_failure_execution_descriptor(
            binding.absolute_path,
            current_ns=time.time_ns(),
        )
    else:
        raise ValueError("trusted runtime consumer source kind is unsupported")
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict) or content_sha256(to_dict()) != binding.semantic_sha256:
        raise ValueError("trusted runtime consumer source digest differs")
    return value


def _runtime_topology_sha256(
    *,
    inventory_sha256: str,
    topology_mode: str,
    gpu_uuids: tuple[str, ...],
) -> str:
    return content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_single_operator_runtime_topology",
            "inventory_sha256": inventory_sha256,
            "topology_mode": topology_mode,
            "gpu_uuids": list(gpu_uuids),
        }
    )


def _preflight_role_sources(
    *,
    algorithm: Literal["DFLASH", "DSPARK"],
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"],
    preflight_inputs: object,
    inventory_sha256: str,
    gpu_uuids: tuple[str, ...],
) -> tuple[
    CanonicalJsonProofBinding,
    str,
    str,
    str,
    tuple[TrustedSingleOperatorRuntimeRoleSource, ...],
]:
    from lightcone_spec.experiments.formal_content_source import (
        FormalContentSourceBinding,
    )
    from lightcone_spec.experiments.formal_preflight_inputs import (
        FormalSingleOperatorPreflightAuthority,
    )
    from lightcone_spec.experiments.formal_single_operator_preflight_qualification import (
        load_formal_single_operator_preflight_qualification_plan,
        load_formal_single_operator_preflight_qualification_plan_index,
        revalidate_formal_single_operator_preflight_qualification_result,
    )
    from lightcone_spec.runtime.distributed import (
        DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES,
    )
    from lightcone_spec.runtime.native_qualification_runner import (
        NativeRuntimeQualificationAssignment,
    )
    from lightcone_spec.runtime.readiness import (
        NATIVE_RUNTIME_RELEASE_CAPABILITY,
        NATIVE_RUNTIME_SUITE_CAPABILITIES,
    )

    role_suites: tuple[tuple[TrustedRuntimeAuthorityRole, str], ...] | None = {
        ("DFLASH", "tp2_dp1"): (("distributed", "tp2_dp1"),),
        ("DFLASH", "tp1_dp2"): (("distributed", "tp1_dp2"),),
        ("DSPARK", "tp1_dp1"): (("native", "dspark_tp1"),),
        ("DSPARK", "tp2_dp1"): (
            ("distributed", "tp2_dp1"),
            ("native", "dspark_tp2"),
        ),
        ("DSPARK", "tp1_dp2"): (
            ("distributed", "tp1_dp2"),
            ("native", "dspark_dp2"),
        ),
    }.get((algorithm, topology_mode))
    if role_suites is None:
        raise ValueError("trusted runtime preflight suite mapping is unavailable")
    index_binding = getattr(preflight_inputs, "qualification_plan_index", None)
    if type(index_binding) is not CanonicalJsonProofBinding:
        raise ValueError("trusted runtime preflight qualification index is absent")
    index = load_formal_single_operator_preflight_qualification_plan_index(
        index_binding.absolute_path
    )
    preflight_authority_binding = getattr(preflight_inputs, "execution_authority", None)
    if type(preflight_authority_binding) is not CanonicalJsonProofBinding:
        raise ValueError("trusted runtime preflight authority is absent")
    preflight_authority = FormalSingleOperatorPreflightAuthority.from_dict(
        preflight_authority_binding.reopen()
    )
    if preflight_authority.sha256 != preflight_authority_binding.semantic_sha256:
        raise ValueError("trusted runtime preflight authority digest differs")
    expected_plan_lineage = (
        preflight_authority.protocol_lock,
        getattr(preflight_inputs, "content_source_binding", None),
        getattr(preflight_inputs, "inventory", None),
        getattr(preflight_inputs, "doctor_report", None),
        getattr(preflight_inputs, "exactness_assignment", None),
    )
    if (
        type(expected_plan_lineage[0]) is not CanonicalJsonProofBinding
        or type(expected_plan_lineage[1]) is not FormalContentSourceBinding
        or type(expected_plan_lineage[2]) is not CanonicalJsonProofBinding
        or type(expected_plan_lineage[3]) is not CanonicalJsonProofBinding
        or type(expected_plan_lineage[4]) is not CanonicalJsonProofBinding
        or preflight_authority.inventory != expected_plan_lineage[2]
    ):
        raise ValueError("trusted runtime preflight lineage is incomplete")
    plans_by_suite: dict[str, tuple[CanonicalJsonProofBinding, object]] = {}
    for binding in index.plans:
        plan = load_formal_single_operator_preflight_qualification_plan(
            binding.absolute_path
        )
        if plan.suite_id in plans_by_suite:
            raise ValueError("trusted runtime qualification suite is duplicated")
        if (
            plan.protocol_lock,
            plan.content_source,
            plan.inventory,
            plan.doctor,
            plan.exactness_assignment,
        ) != expected_plan_lineage:
            raise ValueError("trusted runtime qualification plan lineage differs")
        plans_by_suite[plan.suite_id] = (binding, plan)
    if len(plans_by_suite) != 6:
        raise ValueError("trusted runtime qualification gate is not exact six")
    completed_by_suite: dict[str, tuple[object, object, object]] = {}
    all_result_sha256s: list[str] = []
    for suite_id, (plan_binding, plan) in sorted(plans_by_suite.items()):
        result_binding = CanonicalJsonProofBinding.bind(plan.result_path)
        result = revalidate_formal_single_operator_preflight_qualification_result(
            result_binding.absolute_path
        )
        assignment = NativeRuntimeQualificationAssignment.load(
            result.assignment.absolute_path
        )
        if (
            result.plan != plan_binding
            or result.plan.semantic_sha256 != plan.sha256
            or result.status != "COMPLETE"
            or assignment.schema_version != 2
            or assignment.suite_id != suite_id
            or assignment.inventory_sha256 != inventory_sha256
            or assignment.topology_sha256 != plan.topology_sha256
            or assignment.gpu_uuids != plan.gpu_uuids
        ):
            raise ValueError("trusted runtime exact-six qualification gate differs")
        completed_by_suite[suite_id] = (result_binding, result, assignment)
        all_result_sha256s.append(result.sha256)
    roles: list[TrustedSingleOperatorRuntimeRoleSource] = []
    hardware_sha256s: set[str] = set()
    topology_sha256s: set[str] = set()
    for role, suite_id in role_suites:
        plan_row = plans_by_suite.get(suite_id)
        if plan_row is None:
            raise ValueError("trusted runtime qualification suite is absent")
        plan_binding, plan = plan_row
        result_binding, result, assignment = completed_by_suite[suite_id]
        if (
            result.plan != plan_binding
            or result.plan.semantic_sha256 != plan.sha256
            or assignment.schema_version != 2
            or assignment.suite_id != suite_id
            or assignment.inventory_sha256 != inventory_sha256
            or assignment.gpu_uuids != gpu_uuids
            or plan.topology_mode != topology_mode
            or plan.gpu_uuids != gpu_uuids
            or assignment.topology_sha256 != plan.topology_sha256
            or result.status != "COMPLETE"
        ):
            raise ValueError(
                "trusted runtime qualification result differs from consumer"
            )
        evidence = tuple(
            sorted(
                {
                    result_binding.semantic_sha256,
                    result.assignment.semantic_sha256,
                    result.empirical_proof.semantic_sha256,
                    result.runner_terminal.semantic_sha256,
                    result.live_observation.semantic_sha256,
                    result.live_native_terminal.semantic_sha256,
                    result.junit_xml.raw_sha256,
                }
            )
        )
        hardware_sha256s.add(assignment.hardware_envelope_sha256)
        topology_sha256s.add(assignment.topology_sha256)
        if role == "distributed":
            capability_sha256 = DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES[
                topology_mode
            ].sha256
            backend_capabilities: tuple[str, ...] = ()
        else:
            capability_sha256 = NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256
            backend_capabilities = tuple(
                sorted(NATIVE_RUNTIME_SUITE_CAPABILITIES[suite_id])
            )
        roles.append(
            TrustedSingleOperatorRuntimeRoleSource(
                role=role,
                source_suite_id=suite_id,
                source_capability_sha256=capability_sha256,
                role_source_identity_sha256=assignment.source_identity_sha256,
                evidence_sha256s=evidence,
                backend_capabilities=backend_capabilities,
            )
        )
    if len(hardware_sha256s) != 1 or len(topology_sha256s) != 1:
        raise ValueError("trusted runtime qualification role scopes differ")
    authority_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_single_operator_preflight_runtime_authority",
            "qualification_plan_index_sha256": index.sha256,
            "algorithm": algorithm,
            "topology_mode": topology_mode,
            "inventory_sha256": inventory_sha256,
            "gpu_uuids": list(gpu_uuids),
            "exact_six_result_sha256s": sorted(all_result_sha256s),
            "roles": [row.to_dict() for row in roles],
        }
    )
    return (
        index_binding,
        authority_sha256,
        next(iter(hardware_sha256s)),
        next(iter(topology_sha256s)),
        tuple(sorted(roles, key=lambda row: row.role)),
    )


def _e6_role_sources(
    *,
    execution_source_path: str,
    materialized_cell_id: str,
    launch_manifest: CanonicalJsonProofBinding,
    inventory: CanonicalJsonProofBinding,
    content_source: object,
) -> tuple[
    CanonicalJsonProofBinding,
    str,
    str,
    str,
    tuple[str, str],
    tuple[TrustedSingleOperatorRuntimeRoleSource, ...],
]:
    from lightcone_spec.experiments.formal_single_operator_e6_interface import (
        derive_formal_single_operator_trusted_nextn_tp2_serving_authority,
        revalidate_formal_single_operator_e6_interface_fit_plan,
    )
    from lightcone_spec.runtime.distributed import (
        DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES,
    )
    from lightcone_spec.runtime.native_qualification_runner import (
        NativeRuntimeQualificationAssignment,
    )
    from lightcone_spec.runtime.readiness import (
        NATIVE_RUNTIME_RELEASE_CAPABILITY,
        NATIVE_RUNTIME_SUITE_CAPABILITIES,
    )

    authority = derive_formal_single_operator_trusted_nextn_tp2_serving_authority(
        execution_source_path=execution_source_path,
        materialized_cell_id=materialized_cell_id,
        compile_launch_manifest=launch_manifest,
        inventory=inventory,
        content_source=content_source,  # type: ignore[arg-type]
    )
    plan = revalidate_formal_single_operator_e6_interface_fit_plan(
        authority.interface_fit_plan.absolute_path
    )
    assignment = NativeRuntimeQualificationAssignment.load(
        plan.native_assignment.absolute_path
    )
    if (
        assignment.suite_id != "nextn_tp2"
        or assignment.inventory_sha256 != authority.inventory_sha256
        or assignment.topology_sha256 != authority.topology_sha256
        or assignment.gpu_uuids != authority.gpu_uuids
    ):
        raise ValueError("trusted runtime E6 assignment differs")
    evidence = tuple(
        sorted(
            {
                authority.sha256,
                authority.interface_fit_plan_sha256,
                authority.interface_fit_terminal_sha256,
                plan.native_assignment.semantic_sha256,
                authority.native_gpu_proof_sha256,
                authority.distributed_gpu_proof_sha256,
                authority.junit_raw_sha256,
            }
        )
    )
    distributed_capability = DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES["tp2_dp1"]
    roles = (
        TrustedSingleOperatorRuntimeRoleSource(
            role="distributed",
            source_suite_id="nextn_tp2",
            source_capability_sha256=distributed_capability.sha256,
            role_source_identity_sha256=assignment.source_identity_sha256,
            evidence_sha256s=evidence,
            backend_capabilities=(),
        ),
        TrustedSingleOperatorRuntimeRoleSource(
            role="native",
            source_suite_id="nextn_tp2",
            source_capability_sha256=NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256,
            role_source_identity_sha256=assignment.source_identity_sha256,
            evidence_sha256s=evidence,
            backend_capabilities=tuple(
                sorted(NATIVE_RUNTIME_SUITE_CAPABILITIES["nextn_tp2"])
            ),
        ),
    )
    return (
        authority.interface_fit_terminal,
        authority.sha256,
        assignment.hardware_envelope_sha256,
        authority.topology_sha256,
        authority.gpu_uuids,
        roles,
    )


def _e0_role_sources(
    *,
    authority_binding: CanonicalJsonProofBinding,
    inventory: object,
    doctor: CanonicalJsonProofBinding,
) -> tuple[
    str,
    str,
    tuple[str, ...],
    tuple[TrustedSingleOperatorRuntimeRoleSource, ...],
]:
    from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
        load_trusted_single_operator_eagle3_execution_authority,
    )
    from lightcone_spec.runtime.readiness import (
        NATIVE_RUNTIME_RELEASE_CAPABILITY,
        NATIVE_RUNTIME_SUITE_CAPABILITIES,
    )

    authority = load_trusted_single_operator_eagle3_execution_authority(
        authority_binding.absolute_path
    )
    device = inventory.device(authority.gpu_uuids[0])
    hardware = content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_single_operator_eagle3_runtime_hardware",
            "inventory_sha256": authority.inventory_sha256,
            "doctor_sha256": doctor.semantic_sha256,
            "gpu_uuid": authority.gpu_uuids[0],
            "device_hardware_envelope_sha256": device.hardware_envelope_sha256,
        }
    )
    evidence = tuple(
        sorted(
            {
                authority_binding.semantic_sha256,
                authority.execution_source.semantic_sha256,
                authority.compile_launch_manifest.semantic_sha256,
                authority.interface_receipt.semantic_sha256,
                authority.compatibility_terminal.semantic_sha256,
                authority.execution_authority.semantic_sha256,
                authority.compatibility_authority.semantic_sha256,
                authority.model_selector_authority.semantic_sha256,
                authority.native_gpu_receipt.semantic_sha256,
                authority.proof_row_sha256,
            }
        )
    )
    role = TrustedSingleOperatorRuntimeRoleSource(
        role="native",
        source_suite_id="eagle3_tp1",
        source_capability_sha256=NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256,
        role_source_identity_sha256=authority.native_source_identity_sha256,
        evidence_sha256s=evidence,
        backend_capabilities=tuple(
            sorted(NATIVE_RUNTIME_SUITE_CAPABILITIES["eagle3_tp1"])
        ),
    )
    return authority.sha256, hardware, authority.gpu_uuids, (role,)


def _derive_trusted_runtime_authority_source(
    *,
    consumer_source: CanonicalJsonProofBinding,
    execution_source: CanonicalJsonProofBinding,
    materialized_cell_id: str,
    launch_manifest: CanonicalJsonProofBinding,
    preflight_inputs: CanonicalJsonProofBinding,
) -> TrustedSingleOperatorRuntimeAuthoritySource | None:
    from lightcone_spec.config import load_run_config, run_config_sha256
    from lightcone_spec.experiments.formal_preflight_inputs import (
        FormalPreflightExecutionInputs,
        FormalSingleOperatorPreflightAuthority,
        FormalSingleOperatorPreflightCompletion,
        revalidate_formal_single_operator_preflight_completion,
    )
    from lightcone_spec.experiments.formal_single_operator_run_dispatch import (
        route_formal_single_operator_cell,
    )
    from lightcone_spec.experiments.formal_single_operator_stages import (
        rebuild_formal_single_operator_stage_completion,
    )
    from lightcone_spec.experiments.gpu_pool import GpuInventory
    from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

    consumer = _revalidate_consumer_source(consumer_source)
    source, cell, route = route_formal_single_operator_cell(
        execution_source_path=execution_source.absolute_path,
        materialized_cell_id=materialized_cell_id,
    )
    if source.predecessor_completion_source is None:
        raise ValueError("trusted runtime execution lacks preflight ancestry")
    predecessor = rebuild_formal_single_operator_stage_completion(
        source.predecessor_completion_source.absolute_path
    )
    while predecessor.artifact.node != "preflight":
        if predecessor.predecessor is None:
            raise ValueError("trusted runtime execution lost preflight ancestry")
        predecessor = predecessor.predecessor
    actual_sources = tuple(
        sorted(
            {
                row.source.absolute_path: row.source
                for row in predecessor.artifact.actual_results
            }.values(),
            key=lambda row: row.absolute_path,
        )
    )
    if len(actual_sources) != 1:
        raise ValueError("trusted runtime preflight completion source is ambiguous")
    raw_completion = actual_sources[0].reopen()
    serialized_completion = FormalSingleOperatorPreflightCompletion.from_dict(
        raw_completion
    )
    completed_preflight = revalidate_formal_single_operator_preflight_completion(
        actual_sources[0].absolute_path,
        current_ns=serialized_completion.finished_ns,
    )
    expected_preflight_inputs = completed_preflight.execution_inputs
    if preflight_inputs != expected_preflight_inputs:
        raise ValueError("trusted runtime preflight inputs were spliced")
    preflight = FormalPreflightExecutionInputs.from_dict(preflight_inputs.reopen())
    launch = CompileLaunchManifest.load(launch_manifest.absolute_path)
    config = load_run_config(launch.run_config_path)
    inventory = GpuInventory.from_dict(preflight.inventory.reopen())
    if consumer.kind == "formal_single_operator_downstream_run_plan_inputs":
        from lightcone_spec.experiments.formal_single_operator_e4_execution import (
            revalidate_formal_single_operator_e4_compile_launch,
        )

        preflight_authority = FormalSingleOperatorPreflightAuthority.from_dict(
            preflight.execution_authority.reopen()
        )
        e4_context = revalidate_formal_single_operator_e4_compile_launch(
            execution_source_path=execution_source.absolute_path,
            materialized_cell_id=materialized_cell_id,
            repository_root=preflight_authority.repository_root,
            inventory_path=preflight.inventory.absolute_path,
            compile_launch_manifest_path=launch_manifest.absolute_path,
        )
        if e4_context.launch != launch:
            raise ValueError("trusted runtime E4 launch differs from its mapper")
    algorithm = config.model.algorithm
    topology_mode = config.runtime.topology_mode
    if config.adaptation is None or (
        algorithm == "DFLASH" and topology_mode == "tp1_dp1"
    ):
        return None
    if algorithm not in {"DFLASH", "DSPARK", "NEXTN", "EAGLE3"}:
        return None
    consumer_cell_id = getattr(consumer, "materialized_cell_id", None)
    if consumer_cell_id is None:
        failure_subject = getattr(consumer, "failure_subject", None)
        consumer_cell_id = getattr(failure_subject, "materialized_cell_id", None)
    consumer_preflight = getattr(consumer, "preflight_inputs", None)
    if (
        getattr(consumer, "execution_source", None) != execution_source
        or getattr(consumer, "compile_launch_manifest", None) != launch_manifest
        or consumer_cell_id != materialized_cell_id
        or (consumer_preflight is not None and consumer_preflight != preflight_inputs)
        or source.sha256 != execution_source.semantic_sha256
        or cell.cell_id != materialized_cell_id
        or route.physical_kind != "serving"
        or preflight.schema_version != 4
        or preflight.sha256 != preflight_inputs.semantic_sha256
        or launch.sha256 != launch_manifest.semantic_sha256
        or launch.inventory_sha256 != inventory.sha256
        or config.runtime.device_identity
        not in {launch.gpu_uuids[0], ",".join(launch.gpu_uuids)}
        or set(launch.gpu_uuids) - {row.uuid for row in inventory.devices}
    ):
        raise ValueError("trusted runtime consumer/launch lineage differs")
    authority_kind: TrustedRuntimeAuthorityKind
    authority_evidence: CanonicalJsonProofBinding
    authority_sha256: str
    hardware_envelope_sha256: str
    topology_sha256: str
    gpu_uuids = launch.gpu_uuids
    roles: tuple[TrustedSingleOperatorRuntimeRoleSource, ...]
    if algorithm in {"DFLASH", "DSPARK"}:
        (
            authority_evidence,
            authority_sha256,
            hardware_envelope_sha256,
            topology_sha256,
            roles,
        ) = _preflight_role_sources(
            algorithm=algorithm,
            topology_mode=topology_mode,
            preflight_inputs=preflight,
            inventory_sha256=inventory.sha256,
            gpu_uuids=gpu_uuids,
        )
        authority_kind = "preflight_qualification"
    elif algorithm == "NEXTN":
        if (
            topology_mode != "tp2_dp1"
            or source.content_source_binding is None
            or source.content_source_binding.mode != "trusted_single_operator"
        ):
            raise ValueError("trusted runtime NEXTN scope differs")
        (
            authority_evidence,
            authority_sha256,
            hardware_envelope_sha256,
            topology_sha256,
            e6_gpus,
            roles,
        ) = _e6_role_sources(
            execution_source_path=execution_source.absolute_path,
            materialized_cell_id=materialized_cell_id,
            launch_manifest=launch_manifest,
            inventory=preflight.inventory,
            content_source=source.content_source_binding,
        )
        if e6_gpus != gpu_uuids:
            raise ValueError("trusted runtime E6 GPU placement differs")
        authority_kind = "e6_nextn"
    else:
        if topology_mode != "tp1_dp1":
            raise ValueError("trusted runtime EAGLE3 topology differs")
        raw_authority = getattr(consumer, "trusted_eagle3_execution_authority", None)
        if type(raw_authority) is not CanonicalJsonProofBinding:
            raise ValueError("trusted runtime EAGLE3 authority is absent")
        authority_evidence = raw_authority
        (
            authority_sha256,
            hardware_envelope_sha256,
            e0_gpus,
            roles,
        ) = _e0_role_sources(
            authority_binding=raw_authority,
            inventory=inventory,
            doctor=preflight.doctor_report,
        )
        if e0_gpus != gpu_uuids:
            raise ValueError("trusted runtime E0 GPU placement differs")
        topology_sha256 = _runtime_topology_sha256(
            inventory_sha256=inventory.sha256,
            topology_mode=topology_mode,
            gpu_uuids=gpu_uuids,
        )
        authority_kind = "e0_eagle3"
    source_identity_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_single_operator_runtime_common_source_identity",
            "protocol_sha256": (
                TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_PROTOCOL_SHA256
            ),
            "trust_mode": "trusted_single_operator_empirical_no_signature",
            "formal_measurement": False,
            "authority_kind": authority_kind,
            "algorithm": algorithm,
            "authority_sha256": authority_sha256,
            "inventory_sha256": inventory.sha256,
            "hardware_envelope_sha256": hardware_envelope_sha256,
            "topology_mode": topology_mode,
            "topology_sha256": topology_sha256,
            "gpu_uuids": list(gpu_uuids),
            "roles": [row.to_dict() for row in roles],
        }
    )
    consumer_identity_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_single_operator_runtime_consumer_identity",
            "consumer_source_sha256": consumer_source.semantic_sha256,
            "execution_source_sha256": source.sha256,
            "materialized_cell_id": materialized_cell_id,
            "launch_manifest_sha256": launch.sha256,
            "run_config_sha256": run_config_sha256(config),
            "preflight_inputs_sha256": preflight.sha256,
            "source_identity_sha256": source_identity_sha256,
        }
    )
    return TrustedSingleOperatorRuntimeAuthoritySource(
        schema_version=1,
        kind="trusted_single_operator_runtime_authority_source",
        protocol_sha256=TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_PROTOCOL_SHA256,
        trust_mode="trusted_single_operator_empirical_no_signature",
        formal_measurement=False,
        authority_kind=authority_kind,
        algorithm=algorithm,
        consumer_source=consumer_source,
        execution_source=execution_source,
        materialized_cell_id=materialized_cell_id,
        launch_manifest=launch_manifest,
        preflight_inputs=preflight_inputs,
        authority_evidence=authority_evidence,
        authority_sha256=authority_sha256,
        consumer_identity_sha256=consumer_identity_sha256,
        source_identity_sha256=source_identity_sha256,
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=hardware_envelope_sha256,
        topology_mode=topology_mode,
        topology_sha256=topology_sha256,
        gpu_uuids=gpu_uuids,
        roles=roles,
    )


def publish_trusted_single_operator_runtime_authority_source(
    *,
    consumer_source_path: str | Path,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    launch_manifest_path: str | Path,
    preflight_inputs_path: str | Path,
    output_path: str | Path,
) -> CanonicalJsonProofBinding | None:
    """Publish the exact child-process authority source, or ``None`` if unused."""

    consumer = CanonicalJsonProofBinding.bind(consumer_source_path)
    execution = CanonicalJsonProofBinding.bind(execution_source_path)
    launch = CanonicalJsonProofBinding.bind(launch_manifest_path)
    preflight = CanonicalJsonProofBinding.bind(preflight_inputs_path)
    value = _derive_trusted_runtime_authority_source(
        consumer_source=consumer,
        execution_source=execution,
        materialized_cell_id=materialized_cell_id,
        launch_manifest=launch,
        preflight_inputs=preflight,
    )
    if value is None:
        return None
    publish_canonical_json_no_replace(output_path, value.to_dict())
    binding = CanonicalJsonProofBinding.bind(
        output_path,
        semantic_sha256=value.sha256,
    )
    rebuilt, _tokens = verify_trusted_single_operator_runtime_authority_source(
        binding.absolute_path,
        expected_consumer_source=consumer,
        expected_launch_manifest=launch,
    )
    if rebuilt != value:
        raise RuntimeError("published trusted runtime authority source changed")
    return binding


def verify_trusted_single_operator_runtime_authority_source(
    path: str | Path,
    *,
    expected_source_binding: CanonicalJsonProofBinding | None = None,
    expected_consumer_source: CanonicalJsonProofBinding | None = None,
    expected_launch_manifest: CanonicalJsonProofBinding | None = None,
) -> tuple[
    TrustedSingleOperatorRuntimeAuthoritySource,
    tuple[VerifiedTrustedSingleOperatorRuntimeGpuAuthority, ...],
]:
    """Deep-rebuild one source and issue its process-local opaque role tokens."""

    binding = CanonicalJsonProofBinding.bind(path)
    if expected_source_binding is not None and binding != expected_source_binding:
        raise ValueError("trusted runtime authority environment binding changed")
    value = TrustedSingleOperatorRuntimeAuthoritySource.from_dict(binding.reopen())
    if value.sha256 != binding.semantic_sha256:
        raise ValueError("trusted runtime authority source digest differs")
    if (
        expected_consumer_source is not None
        and value.consumer_source != expected_consumer_source
    ):
        raise ValueError("trusted runtime source consumer differs from launch")
    if (
        expected_launch_manifest is not None
        and value.launch_manifest != expected_launch_manifest
    ):
        raise ValueError("trusted runtime source launch differs from child")
    rebuilt = _derive_trusted_runtime_authority_source(
        consumer_source=value.consumer_source,
        execution_source=value.execution_source,
        materialized_cell_id=value.materialized_cell_id,
        launch_manifest=value.launch_manifest,
        preflight_inputs=value.preflight_inputs,
    )
    if rebuilt != value:
        raise ValueError("trusted runtime authority source replay differs")
    if CanonicalJsonProofBinding.bind(path) != binding:
        raise ValueError("trusted runtime authority source changed during replay")
    tokens = tuple(
        _issue_verified_trusted_single_operator_runtime_gpu_authority(
            role=row.role,
            authority_kind=value.authority_kind,
            source_suite_id=row.source_suite_id,
            authority_source_sha256=binding.semantic_sha256,
            consumer_identity_sha256=value.consumer_identity_sha256,
            evidence_sha256s=row.evidence_sha256s,
            source_capability_sha256=row.source_capability_sha256,
            role_source_identity_sha256=row.role_source_identity_sha256,
            source_identity_sha256=value.source_identity_sha256,
            inventory_sha256=value.inventory_sha256,
            hardware_envelope_sha256=value.hardware_envelope_sha256,
            topology_mode=value.topology_mode,
            topology_sha256=value.topology_sha256,
            gpu_uuids=value.gpu_uuids,
            backend_capabilities=row.backend_capabilities,
        )
        for row in value.roles
    )
    return value, tokens


def _issue_verified_trusted_single_operator_runtime_gpu_authority(
    *,
    role: TrustedRuntimeAuthorityRole,
    authority_kind: TrustedRuntimeAuthorityKind,
    source_suite_id: str,
    authority_source_sha256: str,
    consumer_identity_sha256: str,
    evidence_sha256s: tuple[str, ...],
    source_capability_sha256: str,
    role_source_identity_sha256: str,
    source_identity_sha256: str,
    inventory_sha256: str,
    hardware_envelope_sha256: str,
    topology_mode: Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"],
    topology_sha256: str,
    gpu_uuids: tuple[str, ...],
    backend_capabilities: tuple[str, ...],
) -> VerifiedTrustedSingleOperatorRuntimeGpuAuthority:
    """Issue one role only after the caller completed the deep source replay."""

    canonical_evidence = tuple(sorted(set(evidence_sha256s)))
    receipt_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "trusted_single_operator_runtime_gpu_authority_receipt",
            "protocol_sha256": (
                TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_PROTOCOL_SHA256
            ),
            "trust_mode": "trusted_single_operator_empirical_no_signature",
            "formal_measurement": False,
            "qualification_only": False,
            "role": role,
            "authority_kind": authority_kind,
            "source_suite_id": source_suite_id,
            "authority_source_sha256": authority_source_sha256,
            "consumer_identity_sha256": consumer_identity_sha256,
            "evidence_sha256s": list(canonical_evidence),
            "source_capability_sha256": source_capability_sha256,
            "role_source_identity_sha256": role_source_identity_sha256,
            "source_identity_sha256": source_identity_sha256,
            "inventory_sha256": inventory_sha256,
            "hardware_envelope_sha256": hardware_envelope_sha256,
            "topology_mode": topology_mode,
            "topology_sha256": topology_sha256,
            "gpu_uuids": list(gpu_uuids),
            "backend_capabilities": list(backend_capabilities),
        }
    )
    return VerifiedTrustedSingleOperatorRuntimeGpuAuthority(
        role=role,
        authority_kind=authority_kind,
        source_suite_id=source_suite_id,
        authority_source_sha256=authority_source_sha256,
        consumer_identity_sha256=consumer_identity_sha256,
        evidence_sha256s=canonical_evidence,
        receipt_sha256=receipt_sha256,
        source_capability_sha256=source_capability_sha256,
        role_source_identity_sha256=role_source_identity_sha256,
        source_identity_sha256=source_identity_sha256,
        inventory_sha256=inventory_sha256,
        hardware_envelope_sha256=hardware_envelope_sha256,
        topology_mode=topology_mode,
        topology_sha256=topology_sha256,
        gpu_uuids=gpu_uuids,
        backend_capabilities=tuple(sorted(set(backend_capabilities))),
        _verification_tag=_VERIFIED_TRUSTED_RUNTIME_AUTHORITY_SENTINEL,
    )


__all__ = (
    "TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_ENVIRONMENT",
    "TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_PROTOCOL_SHA256",
    "TrustedSingleOperatorRuntimeAuthoritySource",
    "TrustedSingleOperatorRuntimeRoleSource",
    "VerifiedTrustedSingleOperatorRuntimeGpuAuthority",
    "bind_trusted_single_operator_runtime_authority_environment",
    "publish_trusted_single_operator_runtime_authority_source",
    "trusted_single_operator_runtime_authority_environment",
    "verify_trusted_single_operator_runtime_authority_source",
)
