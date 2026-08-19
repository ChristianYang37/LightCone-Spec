from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments import (
    formal_single_operator_e6_launch_producer as launch_producer,
)
from lightcone_spec.experiments.formal_protocol import E6_MODELS
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedModelRuntimeBinding,
    TrustedModelSnapshotSpec,
    bind_trusted_model_snapshot_member,
)
from lightcone_spec.experiments.formal_single_operator_e6_builtin_mtp import (
    FormalSingleOperatorE6BuiltInMtpBlocked,
    publish_formal_single_operator_e6_builtin_mtp_component,
    revalidate_formal_single_operator_e6_builtin_mtp_component,
    scan_formal_single_operator_e6_builtin_mtp_component,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.orchestration import runtime as runtime_module


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _write_safetensors(path: Path, tensors: dict[str, dict[str, object]]) -> None:
    header = json.dumps(
        tensors,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    capacity = max(
        int(row["data_offsets"][1])  # type: ignore[index]
        for row in tensors.values()
    )
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(capacity))


def _snapshot(
    tmp_path: Path,
    *,
    model: str = E6_MODELS[0],
    include_layer: bool = True,
):
    revision = "1" * 40
    root = tmp_path / revision
    root.mkdir(parents=True)
    _write_json(
        root / "config.json",
        {
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": False,
            },
        },
    )
    names = [
        "model.layers.0.input_layernorm.weight",
        "mtp.fc.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ]
    if include_layer:
        names.append("mtp.layers.0.input_layernorm.weight")
    tensors = {
        name: {
            "dtype": "BF16",
            "shape": [1],
            "data_offsets": [index * 2, index * 2 + 2],
        }
        for index, name in enumerate(names)
    }
    shard = "model-00001-of-00001.safetensors"
    _write_safetensors(root / shard, tensors)
    _write_json(
        root / "model.safetensors.index.json",
        {
            "metadata": {"total_size": len(names) * 2},
            "weight_map": {name: shard for name in names},
        },
    )
    return bind_trusted_model_snapshot_member(
        TrustedModelSnapshotSpec(
            model_id=model,
            revision=revision,
            role="target",
            stages=("E6",),
            local_snapshot_path=str(root.resolve()),
            runtime_bindings=(
                TrustedModelRuntimeBinding(
                    stage="E6",
                    target_model_id=model,
                    backend="NEXTN",
                    draft_depth=1,
                ),
            ),
        )
    )


def test_same_snapshot_builtin_mtp_is_distinct_component_not_external_drafter(
    tmp_path: Path,
) -> None:
    member = _snapshot(tmp_path)
    component = scan_formal_single_operator_e6_builtin_mtp_component(member)

    assert component.mode == "built_in_mtp"
    assert component.model_id == member.model_id
    assert component.revision == member.revision
    assert component.snapshot_root == member.local_snapshot_path
    assert component.target_member_sha256 == member.sha256
    assert component.target_snapshot_sha256 == member.content_sha256
    assert component.sha256 not in {member.sha256, member.content_sha256}
    assert {row.name for row in component.tensors} == {
        "mtp.fc.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.input_layernorm.weight",
    }
    assert not hasattr(component, "drafter_model_id")
    assert not hasattr(component, "drafter_snapshot_path")


def test_builtin_mtp_publication_deep_replays_headers_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    member = _snapshot(tmp_path)
    output = tmp_path / "component.json"
    binding = publish_formal_single_operator_e6_builtin_mtp_component(member, output)
    assert (
        revalidate_formal_single_operator_e6_builtin_mtp_component(
            binding.absolute_path,
            member=member,
        ).sha256
        == binding.semantic_sha256
    )

    config = Path(member.local_snapshot_path) / "config.json"
    raw = json.loads(config.read_text(encoding="utf-8"))
    raw["text_config"]["mtp_use_dedicated_embeddings"] = True
    _write_json(config, raw)
    with pytest.raises((ValueError, FormalSingleOperatorE6BuiltInMtpBlocked)):
        revalidate_formal_single_operator_e6_builtin_mtp_component(
            binding.absolute_path,
            member=member,
        )


def test_builtin_mtp_missing_structural_layer_fails_closed(tmp_path: Path) -> None:
    member = _snapshot(tmp_path, include_layer=False)
    with pytest.raises(ValueError, match="structural tensors"):
        scan_formal_single_operator_e6_builtin_mtp_component(member)


def test_builtin_mtp_foreign_model_and_foreign_runtime_binding_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="trusted E6 runtime binding"):
        _snapshot(tmp_path, model="foreign/model")

    member = _snapshot(tmp_path / "valid")
    foreign = member.__class__(
        **{
            **member.__dict__,
            "runtime_bindings": (),
        }
    )
    with pytest.raises(
        FormalSingleOperatorE6BuiltInMtpBlocked,
        match="runtime_binding",
    ):
        scan_formal_single_operator_e6_builtin_mtp_component(foreign)


@pytest.mark.parametrize("model", E6_MODELS)
def test_exact_two_registered_models_accept_same_snapshot_semantics(
    tmp_path: Path,
    model: str,
) -> None:
    member = _snapshot(tmp_path, model=model)
    component = scan_formal_single_operator_e6_builtin_mtp_component(member)
    assert component.model_id == model
    assert component.mtp_num_hidden_layers == 1
    assert component.mtp_use_dedicated_embeddings is False


def test_exact_two_launch_producer_public_boundary_has_no_scientific_scalars() -> None:
    parameters = inspect.signature(
        launch_producer.publish_formal_single_operator_e6_builtin_mtp_launch_index
    ).parameters

    assert tuple(parameters) == (
        "protocol_lock_path",
        "predecessor_completion_path",
        "trusted_content_bundle_path",
        "base_environment_launch_manifest_path",
        "output_root",
    )
    assert not {
        "model",
        "drafter",
        "draft_model",
        "draft_depth",
        "topology",
        "sampling",
        "mem_fraction_static",
    }.intersection(parameters)


@pytest.mark.parametrize("model", E6_MODELS)
def test_source_owned_e6_run_config_is_same_snapshot_tp2_builtin_mtp(
    tmp_path: Path,
    model: str,
) -> None:
    member = _snapshot(tmp_path, model=model)
    component = scan_formal_single_operator_e6_builtin_mtp_component(member)
    sampling = SamplingProfile(purpose="natural", ignore_eos=False)
    receipt_sha256 = "a" * 64

    config = launch_producer._run_config(
        model=model,
        member=member,
        component=component,
        sampling=sampling,
        gpu_uuids=("GPU-0", "GPU-1"),
        distributed_capability_receipt_sha256=receipt_sha256,
    )

    assert config.model.target == config.model.drafter == model
    assert config.model.target_revision == config.model.drafter_revision
    assert config.model.nextn_mtp_mode == "built_in_mtp"
    assert config.model.target_snapshot_sha256 == member.content_sha256
    assert config.model.mtp_component_sha256 == component.sha256
    assert config.model.target_snapshot_sha256 != config.model.mtp_component_sha256
    assert config.runtime.topology_mode == "tp2_dp1"
    assert config.runtime.tensor_parallel_size == 2
    assert config.runtime.data_parallel_size == 1
    assert config.runtime.distributed_capability_receipt_sha256 == receipt_sha256
    assert config.runtime.sampling_profile_sha256 == sampling.sha256


def test_builtin_mtp_server_render_has_no_external_drafter_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member = _snapshot(tmp_path / "snapshot")
    component = scan_formal_single_operator_e6_builtin_mtp_component(member)
    config = launch_producer._run_config(
        model=member.model_id,
        member=member,
        component=component,
        sampling=SamplingProfile(purpose="natural", ignore_eos=False),
        gpu_uuids=("GPU-0", "GPU-1"),
        distributed_capability_receipt_sha256="b" * 64,
    )
    cache_path = (tmp_path / "compile-cache-plan.json").resolve()
    cache_path.write_text("{}\n", encoding="utf-8")
    fake_plan = SimpleNamespace(
        sha256="c" * 64,
        key=SimpleNamespace(sha256="d" * 64, dtype="bfloat16"),
    )
    monkeypatch.setattr(
        runtime_module.CompileCacheLaunchPlan,
        "load",
        classmethod(lambda _cls, _path: fake_plan),
    )
    monkeypatch.setattr(
        runtime_module,
        "validate_compile_key_for_run_config",
        lambda _plan, *, config: None,
    )

    rendered = runtime_module._render_server(
        output=(tmp_path / "rendered").resolve(),
        method="static",
        config=config,
        verified_checkout=(tmp_path / "checkout").resolve(),
        roots={member.model_id: member.local_snapshot_path},
        target_id=member.model_id,
        drafter_id=member.model_id,
        adaptation_reserve_mb=0,
        mem_fraction_static=0.75,
        host="127.0.0.1",
        port=35_620,
        compile_cache_plan_path=cache_path,
    )

    assert "--speculative-draft-model-path" not in rendered.argv
    assert rendered.argv.count("--model-path") == 1
    assert rendered.argv[rendered.argv.index("--model-path") + 1] == (
        member.local_snapshot_path
    )


def test_e6_exact_two_recovers_fresh_tp2_qualification_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.experiments import (
        formal_single_operator_prerequisite_launch_producer as prerequisite,
    )

    tail = SimpleNamespace(predecessor=None)
    predecessor = SimpleNamespace(predecessor=tail)
    sources = tuple(SimpleNamespace() for _ in range(5))
    authority = SimpleNamespace(
        authority_kind="preflight_native_qualification",
        source_stage="preflight",
        authority_sources=sources,
    )
    observed: dict[str, object] = {}

    def authorities(chain: tuple[object, ...]) -> dict[str, object]:
        observed["chain"] = chain
        return {"tp2_dp1": authority}

    monkeypatch.setattr(
        prerequisite,
        "_preflight_qualification_authorities",
        authorities,
    )
    assert launch_producer._fresh_tp2_qualification(predecessor) is authority
    assert observed["chain"] == (predecessor, tail)

    authority.authority_kind = "legacy_protocol_scalar"
    with pytest.raises(
        launch_producer.FormalSingleOperatorE6BuiltInMtpLaunchBlocked,
        match="fresh_trusted_tp2_qualification_missing",
    ):
        launch_producer._fresh_tp2_qualification(predecessor)
