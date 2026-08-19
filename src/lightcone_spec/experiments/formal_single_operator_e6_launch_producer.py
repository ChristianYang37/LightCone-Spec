"""Source-owned exact-two E6 launches for checkpoints with built-in MTP.

The two registered Qwen checkpoints carry their NEXTN/MTP tensors inside the
target checkpoint.  This producer accepts paths only, discovers the frozen
target and tokenizer members from the trusted content bundle, scans the MTP
component metadata, and emits exactly two TP2 ``CompileLaunchManifest`` rows.
No external draft-model path or caller-authored model/scientific scalar crosses
this boundary.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.config import load_run_config, run_config_sha256
from lightcone_spec.config.schema import ModelPair, RunConfig, RuntimeConfig
from lightcone_spec.experiments.formal_content_source import FormalContentSourceBinding
from lightcone_spec.experiments.formal_preflight_inputs import _compile_key
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
from lightcone_spec.experiments.formal_single_operator_e6_builtin_mtp import (
    FormalSingleOperatorE6BuiltInMtpComponent,
    publish_formal_single_operator_e6_builtin_mtp_component,
    revalidate_formal_single_operator_e6_builtin_mtp_component,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    rebuild_formal_single_operator_stage_completion,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.orchestration.runtime import _render_server
from lightcone_spec.runtime.compile_cache import (
    CompileCacheLaunchPlan,
    CompileOnlyPrewarmManifest,
    CompileOnlyPrewarmPayload,
)
from lightcone_spec.runtime.compile_runner import (
    TRUSTED_SINGLE_OPERATOR_BUILT_IN_MTP_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
    CompileLaunchManifest,
    write_compile_prewarm_manifest,
)
from lightcone_spec.runtime.distributed import (
    DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_SINGLE_OPERATOR_E6_BUILT_IN_MTP_LAUNCH_INDEX_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_e6_builtin_mtp_launch_index_protocol",
        "models": list(E6_MODELS),
        "mode": "built_in_mtp_same_frozen_target_snapshot",
        "component": ("strict_config_weight_index_and_safetensors_header_metadata"),
        "topology": "tp2_dp1",
        "sampling": {
            "schema_version": 2,
            "purpose": "natural",
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": False,
        },
        "mem_fraction_static": 0.75,
        "runtime": (
            "BOUND_content_inventory_doctor_protocol_lock_and_base_environment"
        ),
        "launch_count": 2,
        "external_drafter": "forbidden",
        "caller_scientific_scalars": "forbidden",
    }
)

_PORTS = (35_620, 35_621)
_FIXED_MEM_FRACTION_STATIC = 0.75


class FormalSingleOperatorE6BuiltInMtpLaunchBlocked(RuntimeError):
    """The frozen trusted sources cannot yield the exact two E6 launches."""


def _strict(label: str, value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _sha(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _absolute_new_directory(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute() or root != root.resolve(strict=False):
        raise ValueError("E6 built-in MTP launch root must be normalized and absolute")
    if os.path.lexists(root):
        raise FileExistsError("E6 built-in MTP launch root already exists")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise ValueError("E6 built-in MTP launch parent is unavailable")
    root.mkdir(mode=0o700)
    return root


def _member(
    bundle: TrustedSingleOperatorContentBundle,
    *,
    role: Literal["target", "tokenizer"],
    model: str,
) -> TrustedModelSnapshotMember:
    matches = tuple(
        row
        for row in bundle.model_members
        if row.role == role and row.model_id == model and "E6" in row.stages
    )
    if len(matches) != 1:
        raise FormalSingleOperatorE6BuiltInMtpLaunchBlocked(
            f"exact_e6_{role}_member_missing"
        )
    return matches[0]


def _draft_depth(member: TrustedModelSnapshotMember) -> int:
    matches = tuple(
        row
        for row in member.runtime_bindings
        if row.stage == "E6"
        and row.target_model_id == member.model_id
        and row.backend == "NEXTN"
    )
    if len(matches) != 1:
        raise FormalSingleOperatorE6BuiltInMtpLaunchBlocked(
            "exact_e6_builtin_mtp_runtime_binding_missing"
        )
    return matches[0].draft_depth


def _raw_binding(path: str | Path) -> CanonicalJsonProofBinding:
    return CanonicalJsonProofBinding.bind(Path(path).resolve())


def _validate_predecessor(
    binding: CanonicalJsonProofBinding,
    *,
    lock: ProtocolLock,
) -> object:
    predecessor = rebuild_formal_single_operator_stage_completion(binding.absolute_path)
    if (
        predecessor.artifact.node != "e5_final"
        or predecessor.artifact.protocol_lock_sha256 != lock.sha256
        or predecessor.decision.payload.get("status") != "CONFIRMED"
    ):
        raise FormalSingleOperatorE6BuiltInMtpLaunchBlocked(
            "confirmed_current_e5_predecessor_required"
        )
    return predecessor


def _fresh_tp2_qualification(predecessor: object) -> object:
    """Recover the completed trusted TP2 qualification from the E5 chain."""

    from lightcone_spec.experiments.formal_single_operator_prerequisite_launch_producer import (
        _preflight_qualification_authorities,
    )

    rows = []
    current = predecessor
    while current is not None:
        rows.append(current)
        current = current.predecessor
    authorities = _preflight_qualification_authorities(tuple(rows))
    authority = authorities.get("tp2_dp1")
    if (
        authority is None
        or authority.authority_kind != "preflight_native_qualification"
        or authority.source_stage != "preflight"
        or len(authority.authority_sources) < 5
    ):
        raise FormalSingleOperatorE6BuiltInMtpLaunchBlocked(
            "fresh_trusted_tp2_qualification_missing"
        )
    return authority


def _validate_runtime_sources(
    *,
    lock: ProtocolLock,
    content: FormalContentSourceBinding,
    bundle: TrustedSingleOperatorContentBundle,
    base: CompileLaunchManifest,
) -> tuple[
    GpuInventory,
    CanonicalJsonProofBinding,
    dict[str, object],
    CanonicalJsonProofBinding,
    CompileCacheLaunchPlan,
    RunConfig,
]:
    if (
        lock.schema_version != 5
        or lock.content_source_mode != "trusted_single_operator"
        or lock.trusted_single_operator_content_bundle_sha256 != content.content_sha256
        or bundle.runtime_binding_status != "BOUND"
        or bundle.runtime_observations is None
        or base.schema_version != 2
        or base.content_source_binding != content
    ):
        raise FormalSingleOperatorE6BuiltInMtpLaunchBlocked(
            "one_bound_trusted_runtime_required"
        )
    inventory_binding = _raw_binding(
        bundle.runtime_observations.inventory.absolute_path
    )
    doctor_binding = _raw_binding(bundle.runtime_observations.doctor.absolute_path)
    if (
        inventory_binding.raw_sha256 != bundle.runtime_observations.inventory.raw_sha256
        or inventory_binding.semantic_sha256
        != bundle.runtime_observations.inventory.semantic_sha256
        or doctor_binding.raw_sha256 != bundle.runtime_observations.doctor.raw_sha256
        or doctor_binding.semantic_sha256
        != bundle.runtime_observations.doctor.semantic_sha256
    ):
        raise RuntimeError("trusted E6 runtime observation changed")
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    doctor = doctor_binding.reopen()
    if type(doctor) is not dict:
        raise TypeError("trusted E6 doctor report is not an object")
    base_config = RunConfig.model_validate(
        CanonicalJsonProofBinding.bind(base.run_config_path).reopen()
    )
    base_plan = CompileCacheLaunchPlan.load(base.compile_cache_plan_path)
    ready = {row.uuid for row in inventory.devices if row.ready}
    groups = tuple(
        row
        for row in inventory.topology_groups
        if set(row.gpu_uuids) == set(base.gpu_uuids)
    )
    if (
        inventory.sha256 != inventory_binding.semantic_sha256
        or base.inventory_sha256 != inventory.sha256
        or base_config.runtime.topology_mode != "tp2_dp1"
        or len(base.gpu_uuids) != 2
        or len(set(base.gpu_uuids)) != 2
        or set(base.gpu_uuids) - ready
        or len(groups) != 1
    ):
        raise FormalSingleOperatorE6BuiltInMtpLaunchBlocked(
            "exact_tp2_base_environment_unavailable"
        )
    base_lock = ModelLock(
        schema_version=2,
        models=tuple(
            sorted(
                {
                    LockedModel(
                        base_config.model.target,
                        base_config.model.target_revision,
                    ),
                    LockedModel(
                        base_config.model.drafter,
                        base_config.model.drafter_revision,
                    ),
                },
                key=lambda row: row.model_id,
            )
        ),
    )
    expected = _compile_key(
        doctor=doctor,
        config=base_config,
        gpu_uuid=base.gpu_uuids[0],
    )
    # ``_compile_key`` is the source-owned doctor projection used by the
    # preflight producers; model-lock validation independently prevents a
    # duplicate/ambiguous external pair.
    base_lock.validate()
    if expected != base_plan.key:
        raise ValueError("E6 base environment differs from its bound doctor")
    if base_config.runtime.distributed_capability_receipt_sha256 is None:
        raise FormalSingleOperatorE6BuiltInMtpLaunchBlocked(
            "fresh_trusted_tp2_dispatch_authority_missing"
        )
    return (
        inventory,
        inventory_binding,
        doctor,
        doctor_binding,
        base_plan,
        base_config,
    )


def _run_config(
    *,
    model: str,
    member: TrustedModelSnapshotMember,
    component: FormalSingleOperatorE6BuiltInMtpComponent,
    sampling: SamplingProfile,
    gpu_uuids: tuple[str, str],
    distributed_capability_receipt_sha256: str,
) -> RunConfig:
    depth = _draft_depth(member)
    release = DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES["tp2_dp1"]
    return RunConfig(
        method="static",
        model=ModelPair(
            key=f"e6_builtin_mtp_{E6_MODELS.index(model)}",
            target=model,
            drafter=model,
            target_revision=member.revision,
            drafter_revision=member.revision,
            algorithm="NEXTN",
            max_context_length=40_960,
            draft_depth=depth,
            nextn_mtp_mode="built_in_mtp",
            target_snapshot_sha256=member.content_sha256,
            mtp_component_sha256=component.sha256,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256=sampling.sha256,
            speculation_enabled=True,
            tensor_parallel_size=2,
            data_parallel_size=1,
            device_identity=",".join(gpu_uuids),
            rendezvous_identity=f"e6-builtin-mtp-{E6_MODELS.index(model)}",
            router_identity="single-replica",
            distributed_runtime_capability="patched_two_gpu_v1",
            distributed_release_capability_sha256=release.sha256,
            distributed_capability_receipt_sha256=(
                distributed_capability_receipt_sha256
            ),
            process_group_backend=release.process_group_backend,
            speculative_num_draft_tokens=depth + 1,
            max_running_requests=1,
        ),
    )


def _publish_launch(
    *,
    root: Path,
    index: int,
    lock: ProtocolLock,
    predecessor: CanonicalJsonProofBinding,
    content: FormalContentSourceBinding,
    bundle: TrustedSingleOperatorContentBundle,
    inventory: GpuInventory,
    doctor: dict[str, object],
    base: CompileLaunchManifest,
    base_plan: CompileCacheLaunchPlan,
    base_config: RunConfig,
) -> tuple[CanonicalJsonProofBinding, CanonicalJsonProofBinding]:
    model = E6_MODELS[index]
    member = _member(bundle, role="target", model=model)
    tokenizer = _member(bundle, role="tokenizer", model=model)
    row_root = root / f"model-{index}"
    row_root.mkdir(mode=0o700)
    component_path = row_root / "built-in-mtp-component.json"
    component_binding = publish_formal_single_operator_e6_builtin_mtp_component(
        member,
        component_path,
    )
    component = revalidate_formal_single_operator_e6_builtin_mtp_component(
        component_binding.absolute_path,
        member=member,
    )
    # The qualified base launch contributes only current host/toolchain/cache
    # identity. E6 scientific sampling and memory settings are source-owned.
    sampling = SamplingProfile(purpose="natural", ignore_eos=False)
    sampling_path = row_root / "sampling-profile.json"
    sampling.write(sampling_path)
    distributed_capability_receipt_sha256 = (
        base_config.runtime.distributed_capability_receipt_sha256
    )
    assert distributed_capability_receipt_sha256 is not None
    config = _run_config(
        model=model,
        member=member,
        component=component,
        sampling=sampling,
        gpu_uuids=base.gpu_uuids,  # type: ignore[arg-type]
        distributed_capability_receipt_sha256=(distributed_capability_receipt_sha256),
    )
    model_lock = ModelLock(
        schema_version=2,
        models=(LockedModel(model, member.revision),),
    )
    model_lock.validate()
    key = _compile_key(
        doctor=doctor,
        config=config,
        gpu_uuid=base.gpu_uuids[0],
    )
    cache = CompileCacheLaunchPlan.issue(
        key=key,
        cache_root=base_plan.cache_root,
        cache_mode="build",
    )
    cache_path = row_root / "compile-cache-plan.json"
    cache.write(cache_path)
    prewarm = CompileOnlyPrewarmManifest(
        schema_version=1,
        kind="compile_only_prewarm_manifest",
        model_lock_sha256=model_lock.sha256,
        sampling_profile_sha256=sampling.sha256,
        payloads=(
            CompileOnlyPrewarmPayload(
                request_id=f"e6-builtin-mtp-{index}",
                graph_bucket=1,
                input_token_ids=(1,),
                requested_output_tokens=1,
                sampling_seed=1,
            ),
        ),
    )
    prewarm_path = write_compile_prewarm_manifest(
        prewarm,
        row_root / "compile-prewarm.json",
    )
    rendered = _render_server(
        output=row_root,
        method="static",
        config=config,
        verified_checkout=Path(base.patched_sglang_checkout),
        roots={model: member.local_snapshot_path},
        target_id=model,
        drafter_id=model,
        adaptation_reserve_mb=0,
        mem_fraction_static=_FIXED_MEM_FRACTION_STATIC,
        host="127.0.0.1",
        port=_PORTS[index],
        compile_cache_plan_path=cache_path,
    )
    if "--speculative-draft-model-path" in rendered.argv:
        raise ValueError("built-in MTP launch rendered an external drafter path")
    config_path = Path(rendered.run_config).resolve()
    config_binding = _raw_binding(config_path)
    cache_binding = _raw_binding(cache_path)
    prewarm_binding = _raw_binding(prewarm_path)
    sampling_binding = _raw_binding(sampling_path)
    trusted = content.trusted_single_operator
    assert trusted is not None
    launch = CompileLaunchManifest(
        schema_version=3,
        kind="first_party_compile_launch_manifest",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_BUILT_IN_MTP_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256
        ),
        patched_sglang_checkout=base.patched_sglang_checkout,
        patched_sglang_commit=base.patched_sglang_commit,
        patched_sglang_tree=base.patched_sglang_tree,
        run_config_path=str(config_path),
        run_config_raw_sha256=config_binding.raw_sha256,
        run_config_semantic_sha256=run_config_sha256(config),
        compile_cache_plan_path=str(cache_path),
        compile_cache_plan_raw_sha256=cache_binding.raw_sha256,
        compile_cache_plan_sha256=cache.sha256,
        prewarm_manifest_path=str(prewarm_path),
        prewarm_manifest_raw_sha256=prewarm_binding.raw_sha256,
        prewarm_manifest_sha256=prewarm.sha256,
        sampling_profile_path=sampling_binding.absolute_path,
        sampling_profile_raw_sha256=sampling_binding.raw_sha256,
        prepared_model_content_manifest_path=trusted.absolute_path,
        prepared_model_content_manifest_raw_sha256=trusted.raw_sha256,
        prepared_model_content_manifest_sha256=trusted.semantic_sha256,
        prepared_model_content_manifest_size=trusted.size,
        target_content_member_id=member.sha256,
        target_model_id=model,
        target_snapshot_path=member.local_snapshot_path,
        target_revision=member.revision,
        target_content_authority_sha256=None,
        drafter_content_member_id=member.sha256,
        drafter_model_id=model,
        drafter_snapshot_path=member.local_snapshot_path,
        drafter_revision=member.revision,
        drafter_content_authority_sha256=None,
        tokenizer_content_member_id=tokenizer.sha256,
        tokenizer_model_id=tokenizer.model_id,
        tokenizer_snapshot_path=tokenizer.local_snapshot_path,
        tokenizer_revision=tokenizer.revision,
        tokenizer_content_authority_sha256=None,
        server_argv=rendered.argv,
        server_argv_sha256=content_sha256({"argv": list(rendered.argv)}),
        localhost_port=_PORTS[index],
        model_lock_sha256=model_lock.sha256,
        sampling_profile_sha256=sampling.sha256,
        physical_assignment_sha256=content_sha256(
            {
                "schema_version": 1,
                "kind": "e6_builtin_mtp_interface_physical_assignment",
                "protocol_lock_sha256": lock.sha256,
                "model": model,
                "inventory_sha256": inventory.sha256,
                "gpu_uuids": list(base.gpu_uuids),
            }
        ),
        experiment_budget_sha256=content_sha256(
            {
                "schema_version": 1,
                "kind": "e6_builtin_mtp_interface_budget",
                "predecessor_completion_sha256": predecessor.semantic_sha256,
                "model": model,
                "physical_execution_count": 1,
            }
        ),
        budget_materialization_authority_sha256=predecessor.semantic_sha256,
        inventory_sha256=inventory.sha256,
        gpu_uuids=base.gpu_uuids,
        path_entries=base.path_entries,
        library_path_entries=base.library_path_entries,
        cuda_home=base.cuda_home,
        formal_stage="E6",
        content_source_binding=content,
        nextn_mtp_mode="built_in_mtp",
        target_snapshot_sha256=member.content_sha256,
        mtp_component_sha256=component.sha256,
        mtp_component_binding=component_binding,
    )
    launch.validate(reopen_inputs=True)
    launch_path = row_root / "compile-launch.json"
    launch.write(launch_path)
    return component_binding, CanonicalJsonProofBinding.bind(
        launch_path,
        semantic_sha256=launch.sha256,
    )


@dataclass(frozen=True)
class FormalSingleOperatorE6BuiltInMtpLaunchIndex:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_e6_builtin_mtp_launch_index"]
    protocol_sha256: str
    protocol_lock: CanonicalJsonProofBinding
    protocol_lock_sha256: str
    predecessor_completion: CanonicalJsonProofBinding
    predecessor_completion_sha256: str
    content_source: FormalContentSourceBinding
    inventory: CanonicalJsonProofBinding
    inventory_sha256: str
    doctor: CanonicalJsonProofBinding
    doctor_sha256: str
    base_environment_launch: CanonicalJsonProofBinding
    tp2_qualification_sources: tuple[CanonicalJsonProofBinding, ...]
    tp2_qualification_authority_sha256: str
    models: tuple[str, str]
    gpu_uuids: tuple[str, str]
    components: tuple[CanonicalJsonProofBinding, CanonicalJsonProofBinding]
    launches: tuple[CanonicalJsonProofBinding, CanonicalJsonProofBinding]
    physical_launch_count: Literal[2]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_e6_builtin_mtp_launch_index"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_E6_BUILT_IN_MTP_LAUNCH_INDEX_PROTOCOL_SHA256
            or self.models != E6_MODELS
            or len(set(self.gpu_uuids)) != 2
            or len(set(self.components)) != 2
            or len(set(self.launches)) != 2
            or self.physical_launch_count != 2
        ):
            raise ValueError("E6 built-in MTP launch index identity differs")
        for label, value in (
            ("ProtocolLock", self.protocol_lock_sha256),
            ("predecessor", self.predecessor_completion_sha256),
            ("inventory", self.inventory_sha256),
            ("doctor", self.doctor_sha256),
            ("TP2 qualification authority", self.tp2_qualification_authority_sha256),
        ):
            _sha(f"E6 launch index {label}", value)
        for value in (
            self.protocol_lock,
            self.predecessor_completion,
            self.inventory,
            self.doctor,
            self.base_environment_launch,
            *self.tp2_qualification_sources,
            *self.components,
            *self.launches,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("E6 launch index artifact is not path-bound")
            if CanonicalJsonProofBinding.bind(value.absolute_path) != value:
                raise ValueError("E6 launch index artifact changed")
        if not self.tp2_qualification_sources or tuple(
            row.absolute_path for row in self.tp2_qualification_sources
        ) != tuple(
            sorted({row.absolute_path for row in self.tp2_qualification_sources})
        ):
            raise ValueError("E6 TP2 qualification source set differs")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @property
    def launch_manifest_paths(self) -> dict[str, str]:
        return {
            model: binding.absolute_path
            for model, binding in zip(self.models, self.launches, strict=True)
        }

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["protocol_lock"] = self.protocol_lock.to_dict()
        value["predecessor_completion"] = self.predecessor_completion.to_dict()
        value["content_source"] = self.content_source.to_dict()
        value["inventory"] = self.inventory.to_dict()
        value["doctor"] = self.doctor.to_dict()
        value["base_environment_launch"] = self.base_environment_launch.to_dict()
        value["tp2_qualification_sources"] = [
            row.to_dict() for row in self.tp2_qualification_sources
        ]
        value["models"] = list(self.models)
        value["gpu_uuids"] = list(self.gpu_uuids)
        value["components"] = [row.to_dict() for row in self.components]
        value["launches"] = [row.to_dict() for row in self.launches]
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "E6 built-in MTP launch index", value, {f.name for f in fields(cls)}
        )
        for name in (
            "protocol_lock",
            "predecessor_completion",
            "inventory",
            "doctor",
            "base_environment_launch",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        raw_qualification = row.pop("tp2_qualification_sources")
        if type(raw_qualification) is not list:
            raise TypeError("E6 TP2 qualification sources must be an array")
        row["tp2_qualification_sources"] = tuple(
            CanonicalJsonProofBinding.from_dict(item) for item in raw_qualification
        )
        row["content_source"] = FormalContentSourceBinding.from_dict(
            row["content_source"]
        )
        for name in ("models", "gpu_uuids"):
            raw = row.pop(name)
            if type(raw) is not list:
                raise TypeError(f"E6 launch index {name} must be an array")
            row[name] = tuple(raw)
        for name in ("components", "launches"):
            raw = row.pop(name)
            if type(raw) is not list:
                raise TypeError(f"E6 launch index {name} must be an array")
            row[name] = tuple(CanonicalJsonProofBinding.from_dict(item) for item in raw)
        return cls(**row)  # type: ignore[arg-type]


def revalidate_formal_single_operator_e6_builtin_mtp_launch_index(
    path: str | Path,
) -> FormalSingleOperatorE6BuiltInMtpLaunchIndex:
    binding = CanonicalJsonProofBinding.bind(path)
    index = FormalSingleOperatorE6BuiltInMtpLaunchIndex.from_dict(binding.reopen())
    if index.sha256 != binding.semantic_sha256:
        raise ValueError("E6 built-in MTP launch index binding differs")
    lock = protocol_lock_from_dict(index.protocol_lock.reopen())
    predecessor = _validate_predecessor(index.predecessor_completion, lock=lock)
    tp2 = _fresh_tp2_qualification(predecessor)
    content = index.content_source
    bundle = content.reopen()
    if type(bundle) is not TrustedSingleOperatorContentBundle:
        raise TypeError("E6 launch index content bundle is not exact")
    base = CompileLaunchManifest.load(index.base_environment_launch.absolute_path)
    (
        inventory,
        inventory_binding,
        _doctor,
        doctor_binding,
        _base_plan,
        base_config,
    ) = _validate_runtime_sources(
        lock=lock,
        content=content,
        bundle=bundle,
        base=base,
    )
    if (
        index.protocol_lock.semantic_sha256 != lock.sha256
        or index.protocol_lock_sha256 != lock.sha256
        or index.predecessor_completion.semantic_sha256
        != index.predecessor_completion_sha256
        or index.inventory != inventory_binding
        or index.inventory_sha256 != inventory.sha256
        or index.doctor != doctor_binding
        or index.doctor_sha256 != doctor_binding.semantic_sha256
        or index.base_environment_launch.semantic_sha256 != base.sha256
        or index.base_environment_launch != tp2.binding
        or index.tp2_qualification_sources != tp2.authority_sources
        or index.tp2_qualification_authority_sha256
        != base_config.runtime.distributed_capability_receipt_sha256
        or index.gpu_uuids != base.gpu_uuids
    ):
        raise ValueError("E6 built-in MTP launch index sources differ")
    snapshot_shas: list[str] = []
    component_shas: list[str] = []
    for model, component_binding, launch_binding in zip(
        index.models,
        index.components,
        index.launches,
        strict=True,
    ):
        member = _member(bundle, role="target", model=model)
        tokenizer = _member(bundle, role="tokenizer", model=model)
        component = revalidate_formal_single_operator_e6_builtin_mtp_component(
            component_binding.absolute_path,
            member=member,
        )
        launch = CompileLaunchManifest.load(launch_binding.absolute_path)
        config = load_run_config(launch.run_config_path)
        if (
            component_binding.semantic_sha256 != component.sha256
            or launch_binding.semantic_sha256 != launch.sha256
            or launch.schema_version != 3
            or launch.formal_stage != "E6"
            or launch.content_source_binding != content
            or launch.inventory_sha256 != inventory.sha256
            or launch.gpu_uuids != index.gpu_uuids
            or launch.target_content_member_id != member.sha256
            or launch.drafter_content_member_id != member.sha256
            or launch.target_model_id != model
            or launch.drafter_model_id != model
            or launch.target_revision != member.revision
            or launch.drafter_revision != member.revision
            or launch.target_snapshot_path != member.local_snapshot_path
            or launch.drafter_snapshot_path != member.local_snapshot_path
            or launch.tokenizer_content_member_id != tokenizer.sha256
            or launch.nextn_mtp_mode != "built_in_mtp"
            or launch.target_snapshot_sha256 != member.content_sha256
            or launch.mtp_component_sha256 != component.sha256
            or launch.mtp_component_binding != component_binding
            or config.method != "static"
            or config.model.algorithm != "NEXTN"
            or config.model.target != model
            or config.model.drafter != model
            or config.model.target_revision != member.revision
            or config.model.drafter_revision != member.revision
            or config.model.nextn_mtp_mode != "built_in_mtp"
            or config.model.target_snapshot_sha256 != member.content_sha256
            or config.model.mtp_component_sha256 != component.sha256
            or config.runtime.topology_mode != "tp2_dp1"
            or config.runtime.tensor_parallel_size != 2
            or config.runtime.data_parallel_size != 1
            or config.runtime.device_identity != ",".join(index.gpu_uuids)
            or config.runtime.distributed_capability_receipt_sha256
            != index.tp2_qualification_authority_sha256
            or config.runtime.sampling_profile_sha256 != launch.sampling_profile_sha256
            or "--speculative-draft-model-path" in launch.server_argv
        ):
            raise ValueError("E6 built-in MTP launch row differs")
        snapshot_shas.append(member.content_sha256)
        component_shas.append(component.sha256)
    if len(set(snapshot_shas)) != 2 or len(set(component_shas)) != 2:
        raise ValueError("E6 built-in MTP launch components are not exact-two")
    return index


def publish_formal_single_operator_e6_builtin_mtp_launch_index(
    *,
    protocol_lock_path: str | Path,
    predecessor_completion_path: str | Path,
    trusted_content_bundle_path: str | Path,
    base_environment_launch_manifest_path: str | Path,
    output_root: str | Path,
) -> FormalSingleOperatorE6BuiltInMtpLaunchIndex:
    """Publish exactly two source-owned TP2 launches from path-only inputs."""

    root = _absolute_new_directory(output_root)
    lock_binding = _raw_binding(protocol_lock_path)
    lock = protocol_lock_from_dict(lock_binding.reopen())
    predecessor_binding = _raw_binding(predecessor_completion_path)
    predecessor = _validate_predecessor(predecessor_binding, lock=lock)
    tp2 = _fresh_tp2_qualification(predecessor)
    content = FormalContentSourceBinding.bind_trusted_single_operator(
        str(trusted_content_bundle_path)
    )
    bundle = content.reopen()
    if type(bundle) is not TrustedSingleOperatorContentBundle:
        raise TypeError("E6 launch producer content bundle is not exact")
    base_binding = _raw_binding(base_environment_launch_manifest_path)
    base = CompileLaunchManifest.load(base_binding.absolute_path)
    if base_binding != tp2.binding:
        raise FormalSingleOperatorE6BuiltInMtpLaunchBlocked(
            "base_environment_is_not_fresh_tp2_qualification"
        )
    (
        inventory,
        inventory_binding,
        doctor,
        doctor_binding,
        base_plan,
        base_config,
    ) = _validate_runtime_sources(
        lock=lock,
        content=content,
        bundle=bundle,
        base=base,
    )
    rows = tuple(
        _publish_launch(
            root=root,
            index=index,
            lock=lock,
            predecessor=predecessor_binding,
            content=content,
            bundle=bundle,
            inventory=inventory,
            doctor=doctor,
            base=base,
            base_plan=base_plan,
            base_config=base_config,
        )
        for index in range(2)
    )
    tp2_qualification_authority_sha256 = (
        base_config.runtime.distributed_capability_receipt_sha256
    )
    assert tp2_qualification_authority_sha256 is not None
    index = FormalSingleOperatorE6BuiltInMtpLaunchIndex(
        schema_version=1,
        kind="formal_single_operator_e6_builtin_mtp_launch_index",
        protocol_sha256=(
            FORMAL_SINGLE_OPERATOR_E6_BUILT_IN_MTP_LAUNCH_INDEX_PROTOCOL_SHA256
        ),
        protocol_lock=lock_binding,
        protocol_lock_sha256=lock.sha256,
        predecessor_completion=predecessor_binding,
        predecessor_completion_sha256=predecessor_binding.semantic_sha256,
        content_source=content,
        inventory=inventory_binding,
        inventory_sha256=inventory.sha256,
        doctor=doctor_binding,
        doctor_sha256=doctor_binding.semantic_sha256,
        base_environment_launch=base_binding,
        tp2_qualification_sources=tp2.authority_sources,
        tp2_qualification_authority_sha256=(tp2_qualification_authority_sha256),
        models=E6_MODELS,
        gpu_uuids=base.gpu_uuids,  # type: ignore[arg-type]
        components=tuple(row[0] for row in rows),  # type: ignore[arg-type]
        launches=tuple(row[1] for row in rows),  # type: ignore[arg-type]
        physical_launch_count=2,
    )
    index_path = root / "e6-built-in-mtp-launch-index.json"
    publish_canonical_json_no_replace(index_path, index.to_dict())
    return revalidate_formal_single_operator_e6_builtin_mtp_launch_index(index_path)


__all__ = [
    "FORMAL_SINGLE_OPERATOR_E6_BUILT_IN_MTP_LAUNCH_INDEX_PROTOCOL_SHA256",
    "FormalSingleOperatorE6BuiltInMtpLaunchBlocked",
    "FormalSingleOperatorE6BuiltInMtpLaunchIndex",
    "publish_formal_single_operator_e6_builtin_mtp_launch_index",
    "revalidate_formal_single_operator_e6_builtin_mtp_launch_index",
]
