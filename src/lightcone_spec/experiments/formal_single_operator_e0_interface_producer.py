"""Source-owned exact-12 pre-probe E0 interface launch producer.

Fresh E0 compatibility cannot require task-keyed GPU evidence before those
tasks have been probed.  This module therefore publishes only the immutable
model/backend interface and one generic Static launch for each of the twelve
registered pairs.  EAGLE3 task proof rows are deliberately empty here; the
physical compatibility worker publishes them after each successful real
one-request probe.
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
from lightcone_spec.experiments.formal_protocol import ProtocolLock, content_sha256
from lightcone_spec.experiments.formal_registry import protocol_lock_from_dict
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedModelSnapshotMember,
    TrustedSingleOperatorContentBundle,
)
from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
    E0PreparedModelBackendInterfaceReceipt,
    e0_preprobe_interface_sha256,
    load_e0_prepared_model_backend_interface_receipt,
    publish_e0_prepared_model_backend_interface_receipt,
)
from lightcone_spec.experiments.formal_single_operator_prerequisite_launch_producer import (
    _preflight_qualification_authorities,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    RebuiltFormalSingleOperatorStageCompletion,
    rebuild_formal_single_operator_stage_completion,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.stage_materialization import (
    E0_BACKENDS,
    E0_MODELS,
)
from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.orchestration.runtime import _render_server
from lightcone_spec.runtime.compile_cache import (
    CompileCacheLaunchPlan,
    CompileOnlyPrewarmManifest,
    CompileOnlyPrewarmPayload,
)
from lightcone_spec.runtime.compile_runner import (
    TRUSTED_SINGLE_OPERATOR_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
    CompileLaunchManifest,
    write_compile_prewarm_manifest,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_SINGLE_OPERATOR_E0_PREPROBE_INTERFACE_INDEX_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_preprobe_interface_index_protocol",
        "coverage": "4_models_x_3_backends_exact_12",
        "inputs": (
            "schema5_protocol_lock_confirmed_e6_bound_content_runtime_and_fresh_"
            "preflight_tp1_environment"
        ),
        "model_pairs": "content_runtime_bindings_only_no_caller_scientific_scalar",
        "launch": "static_tp1_dp1_one_request_compile_descriptor",
        "eagle3_preprobe_task_rows": "empty",
        "publication": (
            "no_replace_retain_partial_retry_completed_pair_resume_index_last"
        ),
    }
)

_PORTS = tuple(range(35_640, 35_652))
_MEM_FRACTION_STATIC = 0.75


class FormalSingleOperatorE0PreprobeInterfaceBlocked(RuntimeError):
    """The current trusted sources cannot yield all twelve interfaces."""


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
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _binding(path: str | Path) -> CanonicalJsonProofBinding:
    return CanonicalJsonProofBinding.bind(Path(path).resolve())


def _private_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute() or root != root.resolve(strict=False):
        raise ValueError("E0 pre-probe output root must be normalized and absolute")
    if root == Path(root.anchor):
        raise ValueError("E0 pre-probe output root cannot be a filesystem root")
    if os.path.lexists(root):
        if not root.is_dir() or root.is_symlink() or root.stat().st_mode & 0o077:
            raise ValueError("existing E0 pre-probe output root is not private")
    else:
        if not root.parent.is_dir() or root.parent.is_symlink():
            raise ValueError("E0 pre-probe output parent is unavailable")
        root.mkdir(mode=0o700)
    return root


def _predecessor(
    binding: CanonicalJsonProofBinding,
    *,
    lock: ProtocolLock,
) -> tuple[RebuiltFormalSingleOperatorStageCompletion, str]:
    completion = rebuild_formal_single_operator_stage_completion(binding.absolute_path)
    confirmation = _sha(
        "E0 upstream E6 confirmation",
        completion.decision.payload.get("confirmation_sha256"),
    )
    if (
        completion.artifact.node != "e6_final"
        or completion.artifact.protocol_lock_sha256 != lock.sha256
        or completion.materialization.protocol_lock_sha256 != lock.sha256
        or completion.decision.payload.get("status") != "CONFIRMED"
        or completion.decision.next_materialization_source_decision_sha256
        != confirmation
        or completion.decision.next_materialization_upstream_receipt_sha256s
        != (completion.materialization.sha256,)
    ):
        raise FormalSingleOperatorE0PreprobeInterfaceBlocked(
            "confirmed_current_e6_predecessor_required"
        )
    return completion, confirmation


def _fresh_tp1_environment(
    completion: RebuiltFormalSingleOperatorStageCompletion,
) -> object:
    chain = []
    current: RebuiltFormalSingleOperatorStageCompletion | None = completion
    while current is not None:
        chain.append(current)
        current = current.predecessor
    authorities = _preflight_qualification_authorities(tuple(chain))
    authority = authorities.get("dspark_tp1")
    if (
        authority is None
        or authority.authority_kind != "preflight_native_qualification"
        or authority.source_stage != "preflight"
        or not authority.authority_sources
    ):
        raise FormalSingleOperatorE0PreprobeInterfaceBlocked(
            "fresh_trusted_tp1_environment_missing"
        )
    return authority


def _member(
    bundle: TrustedSingleOperatorContentBundle,
    *,
    role: Literal["target", "tokenizer"],
    model: str,
) -> TrustedModelSnapshotMember:
    matches = tuple(
        row
        for row in bundle.model_members
        if row.role == role and row.model_id == model and "E0" in row.stages
    )
    if len(matches) != 1:
        raise FormalSingleOperatorE0PreprobeInterfaceBlocked(
            f"exact_e0_{role}_{model}_member_missing"
        )
    return matches[0]


def _drafter(
    bundle: TrustedSingleOperatorContentBundle,
    *,
    model: str,
    backend: str,
) -> tuple[TrustedModelSnapshotMember, int]:
    matches = []
    for row in bundle.model_members:
        if row.role != "drafter" or "E0" not in row.stages:
            continue
        bindings = tuple(
            item
            for item in row.runtime_bindings
            if item.stage == "E0"
            and item.target_model_id == model
            and item.backend == backend
        )
        if len(bindings) == 1:
            matches.append((row, bindings[0].draft_depth))
        elif bindings:
            raise ValueError("E0 drafter runtime binding is ambiguous")
    if len(matches) != 1:
        raise FormalSingleOperatorE0PreprobeInterfaceBlocked(
            f"exact_e0_drafter_{model}_{backend}_member_missing"
        )
    return matches[0]


def _runtime_sources(
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
]:
    if (
        lock.schema_version != 5
        or lock.content_source_mode != "trusted_single_operator"
        or lock.trusted_single_operator_content_bundle_sha256 != content.content_sha256
        or bundle.runtime_binding_status != "BOUND"
        or bundle.runtime_observations is None
        or base.schema_version != 2
        or base.content_source_binding != content
        or len(base.gpu_uuids) != 1
    ):
        raise FormalSingleOperatorE0PreprobeInterfaceBlocked(
            "one_bound_trusted_tp1_runtime_required"
        )
    inventory_binding = _binding(bundle.runtime_observations.inventory.absolute_path)
    doctor_binding = _binding(bundle.runtime_observations.doctor.absolute_path)
    if (
        inventory_binding.raw_sha256 != bundle.runtime_observations.inventory.raw_sha256
        or inventory_binding.semantic_sha256
        != bundle.runtime_observations.inventory.semantic_sha256
        or doctor_binding.raw_sha256 != bundle.runtime_observations.doctor.raw_sha256
        or doctor_binding.semantic_sha256
        != bundle.runtime_observations.doctor.semantic_sha256
    ):
        raise RuntimeError("trusted E0 runtime observation changed")
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    doctor = doctor_binding.reopen()
    base_config = load_run_config(base.run_config_path)
    base_plan = CompileCacheLaunchPlan.load(base.compile_cache_plan_path)
    if (
        type(doctor) is not dict
        or inventory.sha256 != inventory_binding.semantic_sha256
        or base.inventory_sha256 != inventory.sha256
        or base_config.runtime.topology_mode != "tp1_dp1"
        or base.gpu_uuids[0] not in {row.uuid for row in inventory.devices if row.ready}
        or _compile_key(
            doctor=doctor,
            config=base_config,
            gpu_uuid=base.gpu_uuids[0],
        )
        != base_plan.key
    ):
        raise FormalSingleOperatorE0PreprobeInterfaceBlocked(
            "fresh_tp1_environment_differs_from_bound_runtime"
        )
    return inventory, inventory_binding, doctor, doctor_binding, base_plan


def _run_config(
    *,
    model: str,
    backend: str,
    target: TrustedModelSnapshotMember,
    drafter: TrustedModelSnapshotMember,
    draft_depth: int,
    sampling: SamplingProfile,
    gpu_uuid: str,
) -> RunConfig:
    return RunConfig(
        method="static",
        model=ModelPair(
            key=f"e0_preprobe_{E0_MODELS.index(model)}_{E0_BACKENDS.index(backend)}",
            target=model,
            drafter=drafter.model_id,
            target_revision=target.revision,
            drafter_revision=drafter.revision,
            algorithm=backend,  # type: ignore[arg-type]
            max_context_length=40_960,
            draft_depth=draft_depth,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256=sampling.sha256,
            speculation_enabled=True,
            tensor_parallel_size=1,
            data_parallel_size=1,
            device_identity=gpu_uuid,
            speculative_num_draft_tokens=draft_depth + 1,
            speculative_eagle_topk=1 if backend == "EAGLE3" else None,
            max_running_requests=1,
        ),
    )


def _resume_pair_or_allocate_attempt(
    *,
    root: Path,
    pair_index: int,
    model: str,
    backend: str,
) -> tuple[CanonicalJsonProofBinding | None, Path | None]:
    """Reuse one complete pair or allocate a new no-replace retry directory."""

    prefix = f"pair-{pair_index:02d}"
    attempts: list[tuple[int, Path]] = []
    for candidate in root.iterdir():
        if candidate.name == prefix:
            attempt = 0
        elif candidate.name.startswith(f"{prefix}-retry-"):
            suffix = candidate.name.removeprefix(f"{prefix}-retry-")
            if len(suffix) != 3 or not suffix.isdigit() or suffix == "000":
                continue
            attempt = int(suffix)
        else:
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError("E0 pre-probe pair attempt is not a real directory")
        attempts.append((attempt, candidate))
    completed: list[CanonicalJsonProofBinding] = []
    for _attempt, candidate in sorted(attempts):
        receipt_path = candidate / "preprobe-interface.json"
        if not os.path.lexists(receipt_path):
            continue
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError("E0 pre-probe pair receipt is not a real file")
        receipt = load_e0_prepared_model_backend_interface_receipt(receipt_path)
        if (receipt.model, receipt.backend, receipt.schema_version) != (
            model,
            backend,
            3,
        ):
            raise ValueError("resumed E0 pre-probe pair identity differs")
        completed.append(_binding(receipt_path))
    if len(completed) > 1:
        raise ValueError("E0 pre-probe pair has ambiguous completed attempts")
    if completed:
        return completed[0], None
    if not attempts:
        return None, root / prefix
    next_attempt = max(attempt for attempt, _path in attempts) + 1
    if next_attempt > 999:
        raise FormalSingleOperatorE0PreprobeInterfaceBlocked(
            f"preprobe_pair_retry_limit_reached:{pair_index}"
        )
    return None, root / f"{prefix}-retry-{next_attempt:03d}"


def _publish_pair(
    *,
    root: Path,
    pair_index: int,
    lock: ProtocolLock,
    predecessor: CanonicalJsonProofBinding,
    e6_confirmation_sha256: str,
    content: FormalContentSourceBinding,
    bundle: TrustedSingleOperatorContentBundle,
    inventory: GpuInventory,
    inventory_binding: CanonicalJsonProofBinding,
    doctor: dict[str, object],
    doctor_binding: CanonicalJsonProofBinding,
    base: CompileLaunchManifest,
    base_binding: CanonicalJsonProofBinding,
    base_plan: CompileCacheLaunchPlan,
    qualification_sources: tuple[CanonicalJsonProofBinding, ...],
) -> CanonicalJsonProofBinding:
    model = E0_MODELS[pair_index // len(E0_BACKENDS)]
    backend = E0_BACKENDS[pair_index % len(E0_BACKENDS)]
    resumed, row_root = _resume_pair_or_allocate_attempt(
        root=root,
        pair_index=pair_index,
        model=model,
        backend=backend,
    )
    if resumed is not None:
        return resumed
    assert row_root is not None
    receipt_path = row_root / "preprobe-interface.json"
    if os.path.lexists(row_root):
        raise FileExistsError(f"refusing to replace E0 pair attempt: {row_root}")
    row_root.mkdir(mode=0o700)
    target = _member(bundle, role="target", model=model)
    tokenizer = _member(bundle, role="tokenizer", model=model)
    drafter, depth = _drafter(bundle, model=model, backend=backend)
    sampling = SamplingProfile(purpose="natural", ignore_eos=False)
    sampling_path = row_root / "sampling-profile.json"
    sampling.write(sampling_path)
    config = _run_config(
        model=model,
        backend=backend,
        target=target,
        drafter=drafter,
        draft_depth=depth,
        sampling=sampling,
        gpu_uuid=base.gpu_uuids[0],
    )
    model_lock = ModelLock(
        schema_version=2,
        models=tuple(
            sorted(
                {
                    LockedModel(model, target.revision),
                    LockedModel(drafter.model_id, drafter.revision),
                },
                key=lambda row: row.model_id,
            )
        ),
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
                request_id=f"e0-preprobe-{pair_index:02d}",
                graph_bucket=1,
                input_token_ids=(1,),
                requested_output_tokens=1,
                sampling_seed=pair_index + 1,
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
        roots={
            model: target.local_snapshot_path,
            drafter.model_id: drafter.local_snapshot_path,
        },
        target_id=model,
        drafter_id=drafter.model_id,
        adaptation_reserve_mb=0,
        mem_fraction_static=_MEM_FRACTION_STATIC,
        host="127.0.0.1",
        port=_PORTS[pair_index],
        compile_cache_plan_path=cache_path,
    )
    config_path = Path(rendered.run_config).resolve()
    config_binding = _binding(config_path)
    cache_binding = _binding(cache_path)
    prewarm_binding = _binding(prewarm_path)
    sampling_binding = _binding(sampling_path)
    trusted = content.trusted_single_operator
    assert trusted is not None
    launch = CompileLaunchManifest(
        schema_version=2,
        kind="first_party_compile_launch_manifest",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256
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
        target_content_member_id=target.sha256,
        target_model_id=model,
        target_snapshot_path=target.local_snapshot_path,
        target_revision=target.revision,
        target_content_authority_sha256=None,
        drafter_content_member_id=drafter.sha256,
        drafter_model_id=drafter.model_id,
        drafter_snapshot_path=drafter.local_snapshot_path,
        drafter_revision=drafter.revision,
        drafter_content_authority_sha256=None,
        tokenizer_content_member_id=tokenizer.sha256,
        tokenizer_model_id=tokenizer.model_id,
        tokenizer_snapshot_path=tokenizer.local_snapshot_path,
        tokenizer_revision=tokenizer.revision,
        tokenizer_content_authority_sha256=None,
        server_argv=rendered.argv,
        server_argv_sha256=content_sha256({"argv": list(rendered.argv)}),
        localhost_port=_PORTS[pair_index],
        model_lock_sha256=model_lock.sha256,
        sampling_profile_sha256=sampling.sha256,
        physical_assignment_sha256=content_sha256(
            {
                "schema_version": 1,
                "kind": "e0_preprobe_interface_physical_assignment",
                "protocol_lock_sha256": lock.sha256,
                "model": model,
                "backend": backend,
                "inventory_sha256": inventory.sha256,
                "gpu_uuids": list(base.gpu_uuids),
            }
        ),
        experiment_budget_sha256=content_sha256(
            {
                "schema_version": 1,
                "kind": "e0_preprobe_interface_budget",
                "predecessor_completion_sha256": predecessor.semantic_sha256,
                "model": model,
                "backend": backend,
                "physical_probe_group_count": 1,
            }
        ),
        budget_materialization_authority_sha256=predecessor.semantic_sha256,
        inventory_sha256=inventory.sha256,
        gpu_uuids=base.gpu_uuids,
        path_entries=base.path_entries,
        library_path_entries=base.library_path_entries,
        cuda_home=base.cuda_home,
        formal_stage="E0",
        content_source_binding=content,
    )
    launch.validate(reopen_inputs=True)
    launch_path = row_root / "compile-launch.json"
    launch.write(launch_path)
    launch_binding = _binding(launch_path)
    evidence_value = {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_preprobe_interface_evidence",
        "protocol_lock_sha256": lock.sha256,
        "upstream_e6_confirmation_sha256": e6_confirmation_sha256,
        "model": model,
        "backend": backend,
        "content_source": content.to_dict(),
        "inventory": inventory_binding.to_dict(),
        "doctor": doctor_binding.to_dict(),
        "base_environment_launch": base_binding.to_dict(),
        "qualification_sources": [row.to_dict() for row in qualification_sources],
        "target_member_sha256": target.sha256,
        "drafter_member_sha256": drafter.sha256,
        "tokenizer_member_sha256": tokenizer.sha256,
        "compile_launch_manifest_sha256": launch.sha256,
        "compile_launch_manifest": launch_binding.to_dict(),
    }
    evidence_path = row_root / "preprobe-evidence.json"
    publish_canonical_json_no_replace(evidence_path, evidence_value)
    evidence = _binding(evidence_path)
    interface_sha = e0_preprobe_interface_sha256(
        protocol_lock_sha256=lock.sha256,
        upstream_e6_confirmation_sha256=e6_confirmation_sha256,
        model=model,
        backend=backend,
        target_model_id=model,
        target_revision=target.revision,
        drafter_model_id=drafter.model_id,
        drafter_revision=drafter.revision,
        tokenizer_model_id=tokenizer.model_id,
        tokenizer_revision=tokenizer.revision,
        target_member_sha256=target.sha256,
        drafter_member_sha256=drafter.sha256,
        tokenizer_member_sha256=tokenizer.sha256,
        compile_launch_manifest_sha256=launch.sha256,
        preprobe_evidence_sha256=evidence.semantic_sha256,
    )
    receipt = E0PreparedModelBackendInterfaceReceipt(
        schema_version=3,
        protocol_lock_sha256=lock.sha256,
        upstream_e6_confirmation_sha256=e6_confirmation_sha256,
        model=model,
        backend=backend,
        tokenizer_sha256=tokenizer.sha256,
        interface_sha256=interface_sha,
        prepared_model_manifest_sha256=trusted.semantic_sha256,
        support_status="READY",
        reason_code="INTERFACE_READY",
        requires_gpu_smoke=True,
        evidence_sha256=evidence.semantic_sha256,
        target_model_id=model,
        target_revision=target.revision,
        drafter_model_id=drafter.model_id,
        drafter_revision=drafter.revision,
        tokenizer_model_id=tokenizer.model_id,
        tokenizer_revision=tokenizer.revision,
        target_member_sha256=target.sha256,
        drafter_member_sha256=drafter.sha256,
        tokenizer_member_sha256=tokenizer.sha256,
        compile_launch_manifest=launch_binding,
        eagle3_runtime_proof_rows=(),
        preprobe_evidence=evidence,
    )
    return publish_e0_prepared_model_backend_interface_receipt(
        receipt,
        output_path=receipt_path,
    )


@dataclass(frozen=True)
class FormalSingleOperatorE0PreprobeInterfaceIndex:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_e0_preprobe_interface_index"]
    protocol_sha256: str
    protocol_lock: CanonicalJsonProofBinding
    protocol_lock_sha256: str
    predecessor_completion: CanonicalJsonProofBinding
    predecessor_completion_sha256: str
    upstream_e6_confirmation_sha256: str
    content_source: FormalContentSourceBinding
    inventory: CanonicalJsonProofBinding
    inventory_sha256: str
    doctor: CanonicalJsonProofBinding
    doctor_sha256: str
    base_environment_launch: CanonicalJsonProofBinding
    tp1_qualification_sources: tuple[CanonicalJsonProofBinding, ...]
    pair_keys: tuple[str, ...]
    interface_descriptors: tuple[CanonicalJsonProofBinding, ...]
    launch_descriptor_count: Literal[12]

    def __post_init__(self) -> None:
        expected_keys = tuple(
            f"{model}|{backend}" for model in E0_MODELS for backend in E0_BACKENDS
        )
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_e0_preprobe_interface_index"
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_E0_PREPROBE_INTERFACE_INDEX_PROTOCOL_SHA256
            or self.pair_keys != expected_keys
            or len(set(self.interface_descriptors)) != 12
            or self.launch_descriptor_count != 12
            or not self.tp1_qualification_sources
        ):
            raise ValueError("E0 pre-probe interface index identity differs")
        for label, digest in (
            ("ProtocolLock", self.protocol_lock_sha256),
            ("predecessor", self.predecessor_completion_sha256),
            ("E6 confirmation", self.upstream_e6_confirmation_sha256),
            ("inventory", self.inventory_sha256),
            ("doctor", self.doctor_sha256),
        ):
            _sha(f"E0 pre-probe index {label}", digest)
        for item in (
            self.protocol_lock,
            self.predecessor_completion,
            self.inventory,
            self.doctor,
            self.base_environment_launch,
            *self.tp1_qualification_sources,
            *self.interface_descriptors,
        ):
            if type(item) is not CanonicalJsonProofBinding:
                raise TypeError("E0 pre-probe index input is not path-bound")
            if _binding(item.absolute_path) != item:
                raise ValueError("E0 pre-probe index input changed")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @property
    def interface_descriptor_paths(self) -> dict[str, str]:
        return {
            key: binding.absolute_path
            for key, binding in zip(
                self.pair_keys,
                self.interface_descriptors,
                strict=True,
            )
        }

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["protocol_lock"] = self.protocol_lock.to_dict()
        value["predecessor_completion"] = self.predecessor_completion.to_dict()
        value["content_source"] = self.content_source.to_dict()
        value["inventory"] = self.inventory.to_dict()
        value["doctor"] = self.doctor.to_dict()
        value["base_environment_launch"] = self.base_environment_launch.to_dict()
        value["tp1_qualification_sources"] = [
            row.to_dict() for row in self.tp1_qualification_sources
        ]
        value["pair_keys"] = list(self.pair_keys)
        value["interface_descriptors"] = [
            row.to_dict() for row in self.interface_descriptors
        ]
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "E0 pre-probe interface index",
            value,
            {field.name for field in fields(cls)},
        )
        for name in (
            "protocol_lock",
            "predecessor_completion",
            "inventory",
            "doctor",
            "base_environment_launch",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        row["content_source"] = FormalContentSourceBinding.from_dict(
            row["content_source"]
        )
        for name in ("tp1_qualification_sources", "interface_descriptors"):
            raw = row[name]
            if type(raw) is not list:
                raise TypeError(f"E0 pre-probe {name} must be an array")
            row[name] = tuple(CanonicalJsonProofBinding.from_dict(item) for item in raw)
        raw_keys = row.pop("pair_keys")
        if type(raw_keys) is not list:
            raise TypeError("E0 pre-probe pair keys must be an array")
        return cls(**row, pair_keys=tuple(raw_keys))  # type: ignore[arg-type]


def revalidate_formal_single_operator_e0_preprobe_interface_index(
    index_path: str | Path,
) -> FormalSingleOperatorE0PreprobeInterfaceIndex:
    binding = _binding(index_path)
    index = FormalSingleOperatorE0PreprobeInterfaceIndex.from_dict(binding.reopen())
    if index.sha256 != binding.semantic_sha256:
        raise ValueError("E0 pre-probe interface index binding differs")
    lock = protocol_lock_from_dict(index.protocol_lock.reopen())
    predecessor, confirmation = _predecessor(
        index.predecessor_completion,
        lock=lock,
    )
    tp1 = _fresh_tp1_environment(predecessor)
    content = index.content_source
    bundle = content.reopen()
    if type(bundle) is not TrustedSingleOperatorContentBundle:
        raise TypeError("E0 pre-probe index content bundle is not exact")
    base = CompileLaunchManifest.load(index.base_environment_launch.absolute_path)
    inventory, inventory_binding, _doctor, doctor_binding, _base_plan = (
        _runtime_sources(
            lock=lock,
            content=content,
            bundle=bundle,
            base=base,
        )
    )
    if (
        index.protocol_lock.semantic_sha256 != lock.sha256
        or index.protocol_lock_sha256 != lock.sha256
        or index.predecessor_completion.semantic_sha256
        != index.predecessor_completion_sha256
        or index.upstream_e6_confirmation_sha256 != confirmation
        or index.inventory != inventory_binding
        or index.inventory_sha256 != inventory.sha256
        or index.doctor != doctor_binding
        or index.doctor_sha256 != doctor_binding.semantic_sha256
        or index.base_environment_launch != tp1.binding
        or index.base_environment_launch.semantic_sha256 != base.sha256
        or index.tp1_qualification_sources != tp1.authority_sources
    ):
        raise ValueError("E0 pre-probe interface index sources differ")
    for pair_index, descriptor in enumerate(index.interface_descriptors):
        model = E0_MODELS[pair_index // len(E0_BACKENDS)]
        backend = E0_BACKENDS[pair_index % len(E0_BACKENDS)]
        receipt = load_e0_prepared_model_backend_interface_receipt(
            descriptor.absolute_path
        )
        target = _member(bundle, role="target", model=model)
        tokenizer = _member(bundle, role="tokenizer", model=model)
        drafter, depth = _drafter(bundle, model=model, backend=backend)
        if receipt.compile_launch_manifest is None or receipt.preprobe_evidence is None:
            raise ValueError("READY E0 pre-probe descriptor lacks evidence")
        launch = CompileLaunchManifest.load(
            receipt.compile_launch_manifest.absolute_path
        )
        config = load_run_config(launch.run_config_path)
        evidence = receipt.preprobe_evidence.reopen()
        if (
            descriptor.semantic_sha256 != receipt.sha256
            or receipt.schema_version != 3
            or (receipt.model, receipt.backend) != (model, backend)
            or receipt.protocol_lock_sha256 != lock.sha256
            or receipt.upstream_e6_confirmation_sha256 != confirmation
            or receipt.support_status != "READY"
            or receipt.reason_code != "INTERFACE_READY"
            or receipt.eagle3_runtime_proof_rows
            or receipt.target_member_sha256 != target.sha256
            or receipt.drafter_member_sha256 != drafter.sha256
            or receipt.tokenizer_member_sha256 != tokenizer.sha256
            or launch.schema_version != 2
            or launch.formal_stage != "E0"
            or launch.content_source_binding != content
            or launch.inventory_sha256 != inventory.sha256
            or launch.gpu_uuids != base.gpu_uuids
            or launch.localhost_port != _PORTS[pair_index]
            or config.method != "static"
            or config.adaptation is not None
            or config.online_spec is not None
            or config.model.target != model
            or config.model.target_revision != target.revision
            or config.model.drafter != drafter.model_id
            or config.model.drafter_revision != drafter.revision
            or config.model.algorithm != backend
            or config.model.draft_depth != depth
            or config.runtime.topology_mode != "tp1_dp1"
            or config.runtime.max_running_requests != 1
            or config.runtime.speculative_num_draft_tokens != depth + 1
            or config.runtime.speculative_eagle_topk
            != (1 if backend == "EAGLE3" else None)
            or type(evidence) is not dict
            or CanonicalJsonProofBinding.from_dict(
                evidence.get("compile_launch_manifest")
            )
            != receipt.compile_launch_manifest
            or CanonicalJsonProofBinding.from_dict(evidence.get("inventory"))
            != inventory_binding
            or CanonicalJsonProofBinding.from_dict(evidence.get("doctor"))
            != doctor_binding
            or CanonicalJsonProofBinding.from_dict(
                evidence.get("base_environment_launch")
            )
            != index.base_environment_launch
        ):
            raise ValueError("E0 pre-probe interface descriptor replay differs")
        raw_sources = evidence.get("qualification_sources")
        if (
            type(raw_sources) is not list
            or tuple(CanonicalJsonProofBinding.from_dict(item) for item in raw_sources)
            != index.tp1_qualification_sources
        ):
            raise ValueError("E0 pre-probe qualification source replay differs")
    return index


def publish_formal_single_operator_e0_preprobe_interface_index(
    *,
    protocol_lock_path: str | Path,
    predecessor_completion_path: str | Path,
    trusted_content_bundle_path: str | Path,
    output_root: str | Path,
) -> FormalSingleOperatorE0PreprobeInterfaceIndex:
    """Publish/resume the exact twelve source-owned pre-probe descriptors."""

    root = _private_root(output_root)
    index_path = root / "e0-preprobe-interface-index.json"
    if os.path.lexists(index_path):
        return revalidate_formal_single_operator_e0_preprobe_interface_index(index_path)
    lock_binding = _binding(protocol_lock_path)
    lock = protocol_lock_from_dict(lock_binding.reopen())
    predecessor_binding = _binding(predecessor_completion_path)
    predecessor, confirmation = _predecessor(predecessor_binding, lock=lock)
    tp1 = _fresh_tp1_environment(predecessor)
    content = FormalContentSourceBinding.bind_trusted_single_operator(
        str(trusted_content_bundle_path)
    )
    bundle = content.reopen()
    if type(bundle) is not TrustedSingleOperatorContentBundle:
        raise TypeError("E0 pre-probe producer content bundle is not exact")
    base_binding = tp1.binding
    base = CompileLaunchManifest.load(base_binding.absolute_path)
    (
        inventory,
        inventory_binding,
        doctor,
        doctor_binding,
        base_plan,
    ) = _runtime_sources(
        lock=lock,
        content=content,
        bundle=bundle,
        base=base,
    )
    descriptors = tuple(
        _publish_pair(
            root=root,
            pair_index=pair_index,
            lock=lock,
            predecessor=predecessor_binding,
            e6_confirmation_sha256=confirmation,
            content=content,
            bundle=bundle,
            inventory=inventory,
            inventory_binding=inventory_binding,
            doctor=doctor,
            doctor_binding=doctor_binding,
            base=base,
            base_binding=base_binding,
            base_plan=base_plan,
            qualification_sources=tp1.authority_sources,
        )
        for pair_index in range(12)
    )
    pair_keys = tuple(
        f"{model}|{backend}" for model in E0_MODELS for backend in E0_BACKENDS
    )
    index = FormalSingleOperatorE0PreprobeInterfaceIndex(
        schema_version=1,
        kind="formal_single_operator_e0_preprobe_interface_index",
        protocol_sha256=(
            FORMAL_SINGLE_OPERATOR_E0_PREPROBE_INTERFACE_INDEX_PROTOCOL_SHA256
        ),
        protocol_lock=lock_binding,
        protocol_lock_sha256=lock.sha256,
        predecessor_completion=predecessor_binding,
        predecessor_completion_sha256=predecessor_binding.semantic_sha256,
        upstream_e6_confirmation_sha256=confirmation,
        content_source=content,
        inventory=inventory_binding,
        inventory_sha256=inventory.sha256,
        doctor=doctor_binding,
        doctor_sha256=doctor_binding.semantic_sha256,
        base_environment_launch=base_binding,
        tp1_qualification_sources=tp1.authority_sources,
        pair_keys=pair_keys,
        interface_descriptors=descriptors,
        launch_descriptor_count=12,
    )
    publish_canonical_json_no_replace(index_path, index.to_dict())
    return revalidate_formal_single_operator_e0_preprobe_interface_index(index_path)


__all__ = [
    "FORMAL_SINGLE_OPERATOR_E0_PREPROBE_INTERFACE_INDEX_PROTOCOL_SHA256",
    "FormalSingleOperatorE0PreprobeInterfaceBlocked",
    "FormalSingleOperatorE0PreprobeInterfaceIndex",
    "publish_formal_single_operator_e0_preprobe_interface_index",
    "revalidate_formal_single_operator_e0_preprobe_interface_index",
]
