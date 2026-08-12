from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

from lightcone_spec.adaptation import (
    TRAINABLE_PLAN_REDUCER_PROTOCOL_SHA256,
    PreparedDrafterParameterInventory,
    TrainablePlanAuthorityBinding,
    audit_trainable_plan_authority_for_method,
    bind_trainable_plan_authority,
    materialize_trainable_plan_authority_manifest,
    replay_trainable_plan_authority,
    require_trainable_plan_authority_for_method,
    trainable_plan_authority_binding_from_dict,
    trainable_plan_authority_binding_to_dict,
)
from lightcone_spec.adaptation.parameters import DFlashParameterPlan, ParameterEntry
from lightcone_spec.config import run_config_sha256
from lightcone_spec.config.schema import (
    AdaptationConfig,
    ModelPair,
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
)
from lightcone_spec.experiments.registry import (
    CellIdentity,
    CellStatus,
    ExperimentCell,
    ResourceClaim,
    WorkloadClass,
    content_sha256,
)
from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.locking.prepared_models import (
    bind_prepared_model_content_authority,
    bind_prepared_models,
    materialize_prepared_model_content_manifest,
    revalidate_prepared_model_content_authority,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _write_bound_json(path: Path, value: object) -> None:
    body = _canonical(value)
    digest = hashlib.sha256(body).hexdigest()
    path.write_bytes(body)
    Path(f"{path}.sha256").write_text(f"{digest}\n", encoding="ascii")


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...]]],
) -> None:
    sizes = {"BF16": 2, "F32": 4}
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name in sorted(tensors):
        dtype, shape = tensors[name]
        count = 1
        for dimension in shape:
            count *= dimension
        end = offset + count * sizes[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, end],
        }
        offset = end
    encoded = _canonical(header)
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def _cell_to_dict(cell: ExperimentCell) -> dict[str, object]:
    identity = cell.identity
    return {
        "identity": {
            "experiment": identity.experiment,
            "model": identity.model,
            "backend": identity.backend,
            "task": identity.task,
            "method": identity.method,
            "scope": identity.scope,
            "rank": identity.rank,
            "alpha_over_rank": identity.alpha_over_rank,
            "optimizer": identity.optimizer,
            "learning_rate": identity.learning_rate,
            "schedule": identity.schedule,
            "context": identity.context,
            "regime": identity.regime,
            "width": identity.width,
            "arrival": identity.arrival,
            "slo": identity.slo,
            "cohort": identity.cohort,
            "topology": identity.topology,
            "seed": identity.seed,
            "block": identity.block,
            "gpu_uuids": list(identity.gpu_uuids),
            "parameterization": identity.parameterization,
            "variant": identity.variant,
            "concurrency": identity.concurrency,
            "load_factor": identity.load_factor,
            "cohort_count": identity.cohort_count,
        },
        "resources": {
            "gpu_uuids": list(cell.resources.gpu_uuids),
            "ports": list(cell.resources.ports),
            "cache_root": cell.resources.cache_root,
            "evidence_root": cell.resources.evidence_root,
            "workload_class": cell.resources.workload_class.value,
        },
        "status": cell.status.value,
        "reason_code": cell.reason_code,
        "reason": cell.reason,
    }


def _inputs(
    tmp_path: Path,
    *,
    method: str = "tts",
    mode: str = "lora",
    target_id: str = "Qwen/Qwen3-8B",
    drafter_id: str = "z-lab/Qwen3-8B-DFlash-b16",
    target_revision: str = "1" * 40,
    drafter_revision: str = "2" * 40,
) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_lock = ModelLock(
        schema_version=2,
        models=(
            LockedModel(target_id, target_revision),
            LockedModel(drafter_id, drafter_revision),
        ),
    )
    model_lock_path = (tmp_path / "model-lock.json").resolve()
    model_lock.write(model_lock_path)

    target_root = (tmp_path / "target" / "snapshots" / target_revision).resolve()
    drafter_root = (tmp_path / "drafter" / "snapshots" / drafter_revision).resolve()
    target_root.mkdir(parents=True)
    drafter_root.mkdir(parents=True)
    for name, body in {
        "config.json": b'{"model_type":"qwen3"}',
        "generation_config.json": b'{"do_sample":false}',
        "merges.txt": b"#version: 0.2\na b\n",
        "tokenizer.json": b'{"version":"1.0"}',
        "tokenizer_config.json": b'{"model_max_length":40960}',
        "vocab.json": b'{"a":0,"b":1}',
    }.items():
        (target_root / name).write_bytes(body)
    target_tensor = {"model.embed_tokens.weight": ("BF16", (4, 2))}
    target_shard = "model-00001-of-00001.safetensors"
    _write_safetensors(target_root / target_shard, target_tensor)
    (target_root / "model.safetensors.index.json").write_bytes(
        _canonical(
            {
                "metadata": {"total_size": 16},
                "weight_map": {
                    "model.embed_tokens.weight": target_shard,
                },
            }
        )
    )
    for name, body in {
        "config.json": b'{"model_type":"dflash"}',
        "dflash.py": b"class DFlash: pass\n",
        "modeling_dflash.py": b"class DFlashModel: pass\n",
        "utils.py": b"BLOCK_SIZE = 16\n",
    }.items():
        (drafter_root / name).write_bytes(body)
    _write_safetensors(
        drafter_root / "model.safetensors",
        {
            "layers.0.input_layernorm.weight": ("F32", (4,)),
            "layers.0.self_attn.q_proj.weight": ("BF16", (8, 4)),
            "lm_head.weight": ("BF16", (32, 4)),
            "target_model.layers.0.weight": ("BF16", (4, 4)),
        },
    )
    prepared_models = bind_prepared_models(
        model_lock,
        {target_id: target_root, drafter_id: drafter_root},
    )
    content_manifest = materialize_prepared_model_content_manifest(
        model_lock, prepared_models
    )
    content_manifest_path = (tmp_path / "prepared-content.json").resolve()
    _write_bound_json(content_manifest_path, content_manifest)
    content_authority = bind_prepared_model_content_authority(
        model_lock,
        prepared_models,
        content_manifest_path,
        expected_release_manifest_sha256=hashlib.sha256(
            _canonical(content_manifest)
        ).hexdigest(),
    )

    config = RunConfig(
        method=method,
        model=ModelPair(
            target=target_id,
            drafter=drafter_id,
            target_revision=target_revision,
            drafter_revision=drafter_revision,
            algorithm="DFLASH",
            max_context_length=40960,
            draft_depth=15,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256="3" * 64,
            speculation_enabled=True,
            speculative_num_draft_tokens=16,
            max_running_requests=1,
        ),
        adaptation=AdaptationConfig(
            weight_update_mode=mode,
            parameter_scope="last1",
            adaptation_group_id="authority-cohort",
            optimizer=OptimizerConfig(
                name="adamw",
                learning_rate=1e-4,
                weight_decay=0.01,
            ),
            rank=2 if mode == "lora" else None,
            lora_alpha=2 if mode == "lora" else None,
            stride=10,
            canvas_tokens=16,
        ),
    )
    run_config_path = (tmp_path / "run-config.json").resolve()
    _write_bound_json(run_config_path, config.model_dump(mode="json"))

    identity = CellIdentity(
        experiment="E1",
        model=target_id,
        backend="DFLASH",
        task="GSM8K",
        method=method,
        scope="last1",
        rank=2 if mode == "lora" else None,
        alpha_over_rank=1.0 if mode == "lora" else None,
        optimizer="adamw",
        learning_rate=1e-4,
        schedule="constant",
        context=40960,
        regime="latency_sensitive",
        width=16,
        arrival="closed_loop",
        slo="registered",
        cohort="K=1:uniform",
        topology="tp1_dp1",
        seed=1,
        block=0,
        gpu_uuids=("GPU-authority",),
        parameterization=mode,
        variant="authority-test",
        concurrency=1,
    )
    cell = ExperimentCell(
        identity=identity,
        resources=ResourceClaim(
            gpu_uuids=identity.gpu_uuids,
            ports=(31000,),
            cache_root="authority-cache",
            evidence_root="authority-evidence",
            workload_class=WorkloadClass.HEADLINE,
        ),
        status=CellStatus.UNMEASURED,
        reason_code="awaiting_registered_measurement",
        reason="No complete content-bound measurement exists.",
    )
    cell_path = (tmp_path / "cell.json").resolve()
    _write_bound_json(cell_path, _cell_to_dict(cell))

    split = {
        "schema_version": 1,
        "kind": "authority_test_execution_split",
        "cell_id": cell.cell_id,
        "run_config_sha256": run_config_sha256(config),
        "split_population_sha256": "4" * 64,
    }
    split_path = (tmp_path / "split.json").resolve()
    _write_bound_json(split_path, split)

    content_result = revalidate_prepared_model_content_authority(
        model_lock,
        content_authority,
        expected_release_manifest_sha256=(content_authority.release_manifest_sha256),
    )
    inventory = {
        "schema_version": 1,
        "kind": "prepared_drafter_parameter_inventory",
        "model_lock_sha256": model_lock.sha256,
        "drafter_model_id": drafter_id,
        "prepared_drafter_revision": drafter_revision,
        "dspark_native_heads": None,
        "parameters": [
            {
                "name": tensor.name,
                "shape": list(tensor.shape),
                "dtype": tensor.dtype,
                "ownership": "sharded",
            }
            for tensor in content_result.snapshot(drafter_id).tensors
        ],
    }
    prepared_path = (tmp_path / "prepared-drafter.json").resolve()
    _write_bound_json(prepared_path, inventory)

    manifest = materialize_trainable_plan_authority_manifest(
        model_lock_artifact=model_lock_path,
        prepared_drafter_artifact=prepared_path,
        run_config_artifact=run_config_path,
        split_artifact=split_path,
        cell_artifact=cell_path,
        prepared_model_content_authority=content_authority,
    )
    manifest_path = (tmp_path / "trainable-plan-authority.json").resolve()
    _write_bound_json(manifest_path, manifest)
    binding = bind_trainable_plan_authority(
        manifest_path,
        prepared_model_content_authority=content_authority,
    )
    return {
        "binding": binding,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "model_lock": model_lock,
        "model_lock_path": model_lock_path,
        "prepared": inventory,
        "prepared_path": prepared_path,
        "prepared_model_content_authority": content_authority,
        "prepared_model_content_manifest_path": content_manifest_path,
        "run_config": config,
        "run_config_path": run_config_path,
        "split": split,
        "split_path": split_path,
        "cell": cell,
        "cell_path": cell_path,
    }


def _expected(binding: TrainablePlanAuthorityBinding) -> dict[str, object]:
    return {
        "expected_model_lock_sha256": binding.model_lock_sha256,
        "expected_prepared_model_content_manifest_sha256": (
            binding.prepared_model_content_manifest_sha256
        ),
        "expected_run_config_sha256": binding.run_config_sha256,
        "expected_split_sha256": binding.split_sha256,
        "expected_cell_id": binding.cell_id,
        "expected_cell_declaration_sha256": binding.cell_declaration_sha256,
        "expected_target_model_id": binding.target_model_id,
        "expected_target_revision": binding.target_revision,
        "expected_drafter_model_id": binding.drafter_model_id,
        "expected_prepared_drafter_revision": binding.prepared_drafter_revision,
        "expected_backend": binding.backend,
        "expected_mode": binding.mode,
        "expected_scope": binding.scope,
        "expected_optimizer": binding.optimizer,
        "expected_rank": binding.rank,
        "expected_lora_alpha": binding.lora_alpha,
    }


def test_raw_authority_replays_selector_state_memory_and_strict_codec(
    tmp_path: Path,
) -> None:
    values = _inputs(tmp_path)
    binding = values["binding"]
    assert isinstance(binding, TrainablePlanAuthorityBinding)
    result = replay_trainable_plan_authority(binding)
    assert binding.reducer_protocol_sha256 == TRAINABLE_PLAN_REDUCER_PROTOCOL_SHA256
    assert binding.prepared_model_content_manifest_sha256 == (
        values["prepared_model_content_authority"].release_manifest_sha256
    )
    assert result.plan.sha256 == binding.trainable_plan_sha256
    assert result.plan.state_layout_sha256 == binding.state_layout_sha256
    assert result.plan.allocation_memory_sha256 == binding.allocation_memory_sha256
    assert result.plan.predict_memory("adamw").peak_bytes > 0
    assert tuple(entry.name for entry in result.plan.entries) == (
        "layers.0.self_attn.q_proj.weight",
    )
    assert result.plan.entries[0].ownership == "sharded"
    assert "layers.0.input_layernorm.weight" in result.plan.frozen_names
    assert PreparedDrafterParameterInventory.from_dict(values["prepared"]) == (
        result.prepared_drafter
    )
    encoded = trainable_plan_authority_binding_to_dict(binding)
    assert trainable_plan_authority_binding_from_dict(encoded) == binding
    with pytest.raises(ValueError, match="fields differ"):
        trainable_plan_authority_binding_from_dict({**encoded, "summary": {}})
    with pytest.raises(ValueError, match="binding identity"):
        trainable_plan_authority_binding_from_dict({**encoded, "schema_version": True})


def test_core_method_gate_requires_exact_raw_identity_and_no_baseline_state(
    tmp_path: Path,
) -> None:
    values = _inputs(tmp_path)
    binding = values["binding"]
    assert isinstance(binding, TrainablePlanAuthorityBinding)
    plan = audit_trainable_plan_authority_for_method(
        "tts", binding, **_expected(binding)
    )
    assert plan is not None and plan.sha256 == binding.trainable_plan_sha256
    with pytest.raises(
        RuntimeError,
        match="prepared_model_content_release_manifest_pin_unavailable",
    ):
        require_trainable_plan_authority_for_method(
            "tts", binding, **_expected(binding)
        )
    for method in ("target_only", "static"):
        assert require_trainable_plan_authority_for_method(method, None) is None
        with pytest.raises(ValueError, match="must not carry"):
            require_trainable_plan_authority_for_method(method, binding)
    with pytest.raises(ValueError, match="requires exact path-bound"):
        require_trainable_plan_authority_for_method("l0", None)
    with pytest.raises(ValueError, match="differs"):
        audit_trainable_plan_authority_for_method(
            "tts",
            binding,
            **{
                **_expected(binding),
                "expected_split_sha256": "f" * 64,
            },
        )

    l0 = _inputs(tmp_path / "l0", method="l0")["binding"]
    assert isinstance(l0, TrainablePlanAuthorityBinding)
    assert (
        audit_trainable_plan_authority_for_method("l0", l0, **_expected(l0)) is not None
    )

    full = _inputs(tmp_path / "full", mode="full")["binding"]
    assert isinstance(full, TrainablePlanAuthorityBinding)
    full_plan = audit_trainable_plan_authority_for_method(
        "tts", full, **_expected(full)
    )
    assert full_plan is not None and full_plan.mode == "full"

    with pytest.raises(
        RuntimeError,
        match="prepared_parameter_inventory_first_party_extractor_unavailable",
    ):
        materialize_trainable_plan_authority_manifest(
            model_lock_artifact=values["model_lock_path"],
            prepared_drafter_artifact=values["prepared_path"],
            run_config_artifact=values["run_config_path"],
            split_artifact=values["split_path"],
            cell_artifact=values["cell_path"],
        )

    unavailable = _inputs(tmp_path / "unavailable-extractor")
    unavailable_binding = unavailable["binding"]
    assert isinstance(unavailable_binding, TrainablePlanAuthorityBinding)
    snapshot = unavailable_binding.prepared_model_content_authority.prepared_model_set
    drafter_root = next(
        item.root
        for item in snapshot.snapshots
        if item.model_id == unavailable_binding.drafter_model_id
    )
    (Path(drafter_root) / "model.safetensors").unlink()
    with pytest.raises(
        RuntimeError,
        match="prepared_parameter_inventory_first_party_extractor_unavailable",
    ):
        audit_trainable_plan_authority_for_method(
            "tts", unavailable_binding, **_expected(unavailable_binding)
        )


def test_jointly_rehashed_serialized_plan_cannot_replace_raw_reducer(
    tmp_path: Path,
) -> None:
    values = _inputs(tmp_path)
    manifest = dict(values["manifest"])
    forged = DFlashParameterPlan(
        backend="DFLASH",
        mode="lora",
        scope="last1",
        rank=2,
        lora_alpha=2,
        entries=(
            ParameterEntry(
                name="layers.0.self_attn.q_proj.weight",
                shape=(800, 400),
                dtype="torch.float16",
                parameterization="lora",
                ownership="replicated",
            ),
        ),
        frozen_names=tuple(manifest["frozen_names"]),
    )
    entries = [
        {
            "name": entry.name,
            "shape": list(entry.shape),
            "dtype": entry.dtype,
            "parameterization": entry.parameterization,
            "ownership": entry.ownership,
        }
        for entry in forged.entries
    ]
    state_layout = [
        {
            "name": row["name"],
            "parameterization": row["parameterization"],
            "ownership": row["ownership"],
            "state_shapes": [list(shape) for shape in row["state_shapes"]],
        }
        for row in forged.state_layout
    ]
    memory = {
        "optimizer": "adamw",
        **forged.predict_memory("adamw").to_dict(),
    }
    manifest["entries"] = entries
    manifest["entries_sha256"] = content_sha256(entries)
    manifest["state_layout"] = state_layout
    manifest["state_layout_sha256"] = content_sha256(state_layout)
    manifest["trainable_parameter_count"] = forged.trainable_parameter_count
    manifest["optimizer_memory_prediction"] = memory
    manifest["optimizer_memory_sha256"] = content_sha256(memory)
    manifest["allocation_memory_sha256"] = forged.allocation_memory_sha256
    manifest["trainable_plan_sha256"] = forged.sha256
    _write_bound_json(values["manifest_path"], manifest)
    with pytest.raises(ValueError, match="serialized trainable plan differs"):
        bind_trainable_plan_authority(
            values["manifest_path"],
            prepared_model_content_authority=values["prepared_model_content_authority"],
        )

    forged_source = _inputs(tmp_path / "caller-inventory")
    forged_inventory = dict(forged_source["prepared"])
    forged_inventory["parameters"] = [
        {
            **row,
            "ownership": (
                "replicated"
                if row["name"] == "layers.0.self_attn.q_proj.weight"
                else row["ownership"]
            ),
        }
        for row in forged_inventory["parameters"]
    ]
    _write_bound_json(forged_source["prepared_path"], forged_inventory)
    with pytest.raises(ValueError, match="first-party snapshot extraction"):
        materialize_trainable_plan_authority_manifest(
            model_lock_artifact=forged_source["model_lock_path"],
            prepared_drafter_artifact=forged_source["prepared_path"],
            run_config_artifact=forged_source["run_config_path"],
            split_artifact=forged_source["split_path"],
            cell_artifact=forged_source["cell_path"],
            prepared_model_content_authority=forged_source[
                "prepared_model_content_authority"
            ],
        )


def test_model_revision_swap_and_raw_source_replacement_fail_closed(
    tmp_path: Path,
) -> None:
    values = _inputs(tmp_path)
    binding = values["binding"]
    assert isinstance(binding, TrainablePlanAuthorityBinding)
    lock = values["model_lock"].to_dict()
    lock["models"][0]["revision"] = "9" * 40
    _write_bound_json(values["model_lock_path"], lock)
    with pytest.raises(RuntimeError, match="model_lock or sidecar changed"):
        replay_trainable_plan_authority(binding)

    other = _inputs(tmp_path / "other")
    run_config = other["run_config"].model_dump(mode="json")
    run_config["model"]["target"] = "foreign/target"
    _write_bound_json(other["run_config_path"], run_config)
    with pytest.raises(ValueError, match="run config is invalid|registry cell differs"):
        materialize_trainable_plan_authority_manifest(
            model_lock_artifact=other["model_lock_path"],
            prepared_drafter_artifact=other["prepared_path"],
            run_config_artifact=other["run_config_path"],
            split_artifact=other["split_path"],
            cell_artifact=other["cell_path"],
            prepared_model_content_authority=other["prepared_model_content_authority"],
        )

    swapped = _inputs(
        tmp_path / "jointly-swapped",
        target_revision="a" * 40,
        drafter_revision="b" * 40,
    )["binding"]
    assert isinstance(swapped, TrainablePlanAuthorityBinding)
    with pytest.raises(ValueError, match="differs"):
        audit_trainable_plan_authority_for_method("tts", swapped, **_expected(binding))


def test_path_symlink_duplicate_key_nonfinite_and_tamper_are_rejected(
    tmp_path: Path,
) -> None:
    values = _inputs(tmp_path)
    symlink = (tmp_path / "manifest-link.json").resolve()
    os.symlink(values["manifest_path"], symlink)
    with pytest.raises(ValueError, match="absolute, resolved, and symlink-free"):
        bind_trainable_plan_authority(symlink)

    duplicate = (tmp_path / "duplicate.json").resolve()
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n')
    Path(f"{duplicate}.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        bind_trainable_plan_authority(duplicate)

    nonfinite = (tmp_path / "nonfinite.json").resolve()
    nonfinite.write_text('{"value":NaN}\n')
    Path(f"{nonfinite}.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        bind_trainable_plan_authority(nonfinite)

    overflowing = (tmp_path / "overflowing.json").resolve()
    overflowing.write_text('{"value":1e999}\n')
    Path(f"{overflowing}.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(ValueError, match="non-finite JSON number"):
        bind_trainable_plan_authority(overflowing)

    surrogate = (tmp_path / "surrogate.json").resolve()
    surrogate.write_bytes(b'{"value":"\\ud800"}\n')
    Path(f"{surrogate}.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(ValueError, match="unpaired JSON surrogate"):
        bind_trainable_plan_authority(surrogate)

    binding = values["binding"]
    assert isinstance(binding, TrainablePlanAuthorityBinding)
    _write_bound_json(Path(binding.split.path), {"tampered": True})
    with pytest.raises(RuntimeError, match="split or sidecar changed"):
        replay_trainable_plan_authority(binding)
