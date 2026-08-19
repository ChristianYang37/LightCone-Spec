from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_formal_single_operator_content import (
    _model_spec,
    _source_repository,
    _tiny_burstgpt_assets,
)

from lightcone_spec.experiments import formal_registry, formal_single_operator_stages
from lightcone_spec.experiments import (
    formal_single_operator_context_artifact as context_artifact,
)
from lightcone_spec.experiments.formal_content_source import (
    FormalContentSourceBinding,
)
from lightcone_spec.experiments.formal_registry import (
    stage_materialization_receipt_to_dict,
)
from lightcone_spec.experiments.formal_single_operator_content import (
    bind_trusted_locked_workload,
    bind_trusted_single_operator_runtime_observations,
    build_trusted_single_operator_content_bundle,
    publish_trusted_single_operator_content_bundle,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    MaterializedCell,
    StageMaterializationReceipt,
)
from lightcone_spec.experiments.workload_authority import (
    bind_formal_workload_authority,
    formal_workload_authority_cli_artifact,
)
from lightcone_spec.orchestration import formal_physical_dispatch as dispatch
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _fake_tokenizer_worker(
    *, input_path: Path, output_path: Path
) -> tuple[CanonicalJsonProofBinding, str, int, str]:
    value = CanonicalJsonProofBinding.bind(input_path).reopen()
    rows = []
    for item in value["requests"]:
        token_ids = [int(item["ordinal"]) + 1, len(str(item["prompt"])) + 7]
        rows.append(
            {
                "request_id": item["request_id"],
                "ordinal": item["ordinal"],
                "prompt_sha256": item["prompt_sha256"],
                "input_token_ids": token_ids,
                "input_token_ids_sha256": context_artifact._sha256(token_ids),
            }
        )
    output = {
        "schema_version": 1,
        "kind": "formal_serving_tokenization_output",
        "protocol_sha256": value["protocol_sha256"],
        "schedule_source_sha256": value["schedule_source_sha256"],
        "tokenizer_model_id": value["tokenizer_model_id"],
        "tokenizer_revision": value["tokenizer_revision"],
        "tokenizer_snapshot_path": value["tokenizer_snapshot_path"],
        "tokenizer_content_authority_sha256": None,
        "tokenizer_class": "FixtureTokenizer",
        "tokenizer_vocab_size": 256,
        "transformers_version": "fixture",
        "requests": rows,
    }
    publish_canonical_json_no_replace(output_path, output)
    worker, worker_sha, worker_size = context_artifact._worker_source()
    executable = Path(sys.executable).resolve()
    argv = (
        str(executable),
        str(worker),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    )
    return (
        CanonicalJsonProofBinding.bind(output_path),
        worker_sha,
        worker_size,
        context_artifact._sha256({"argv": list(argv)}),
    )


def test_context_filler_artifact_roundtrip_and_worker_evidence_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = _source_repository(tmp_path)
    raw_specs = tuple(
        _model_spec(tmp_path, role) for role in ("target", "drafter", "tokenizer")
    )
    specs = tuple(
        replace(
            spec,
            revision=Path(spec.local_snapshot_path).name,
            stages=("E3b",),
        )
        if spec.role == "tokenizer"
        else spec
        for spec in raw_specs
    )
    cache_text = os.environ.get("LIGHTCONE_TEST_SOURCE_CACHE")
    if cache_text is None:
        pytest.skip("LIGHTCONE_TEST_SOURCE_CACHE is not configured")
    cache = Path(cache_text).resolve()
    livecodebench_path = (
        cache / "livecodebench-code_generation_lite-0fe84c3/test6.jsonl"
    )
    math_path = cache / "math-500-6e4ed1a/test.jsonl"
    if not livecodebench_path.is_file() or not math_path.is_file():
        pytest.skip("locked formal workload cache is unavailable")
    locked = {
        "livecodebench_v6_hard": bind_trusted_locked_workload(
            "livecodebench_v6_hard",
            livecodebench_path,
        ),
        "math500_level5": bind_trusted_locked_workload(
            "math500_level5",
            math_path,
        ),
    }
    burst_paths = _tiny_burstgpt_assets(tmp_path, monkeypatch)
    pending = build_trusted_single_operator_content_bundle(
        repository_root=source_root,
        model_specs=specs,
        livecodebench_raw_path=locked["livecodebench_v6_hard"].raw_source_path,
        math500_raw_path=locked["math500_level5"].raw_source_path,
        burstgpt_asset_paths=burst_paths,
    )
    inventory = (tmp_path / "inventory.json").resolve()
    doctor = (tmp_path / "doctor.json").resolve()
    inventory.write_text('{"gpu":"GPU-test"}\n', encoding="utf-8")
    doctor.write_text('{"driver":"test"}\n', encoding="utf-8")
    bound = bind_trusted_single_operator_runtime_observations(
        pending,
        inventory_path=inventory,
        doctor_path=doctor,
    )
    bundle_path = (tmp_path.parent / f"{tmp_path.name}-bundle.json").resolve()
    publish_trusted_single_operator_content_bundle(bound, bundle_path)
    content_source = FormalContentSourceBinding.bind_trusted_single_operator(
        str(bundle_path)
    )

    monkeypatch.setattr(context_artifact, "_invoke_worker", _fake_tokenizer_worker)
    tokenizer = next(row for row in bound.model_members if row.role == "tokenizer")
    launch = SimpleNamespace(
        schema_version=2,
        content_source_binding=content_source,
        tokenizer_content_authority_sha256=None,
        tokenizer_content_member_id=tokenizer.sha256,
        tokenizer_model_id=tokenizer.model_id,
        tokenizer_revision=tokenizer.revision,
        tokenizer_snapshot_path=tokenizer.local_snapshot_path,
    )
    monkeypatch.setattr(
        context_artifact.CompileLaunchManifest,
        "load",
        lambda _path: launch,
    )
    launch_path = (tmp_path / "launch.json").resolve()
    publish_canonical_json_no_replace(launch_path, {"launch": True})
    output = (tmp_path / "context-artifact").resolve()
    output.mkdir()

    binding = context_artifact.materialize_trusted_context_filler_artifact(
        content_source_binding=content_source,
        compile_launch_manifest_path=launch_path,
        output_directory=output,
    )
    authority = context_artifact.load_trusted_context_filler_artifact(
        binding.absolute_path,
        content_source_binding=content_source,
        tokenizer_content_member_id=tokenizer.sha256,
        tokenizer_model_id=tokenizer.model_id,
        tokenizer_revision=tokenizer.revision,
    )

    assert len(authority.rows) == 214
    assert authority.registered_source_member_sha256s == tuple(
        sorted(row.sha256 for row in bound.locked_workloads)
    )
    artifact = context_artifact.TrustedContextFillerArtifact.from_dict(binding.reopen())
    assert artifact.row_count == len(authority.rows)
    assert max(path.stat().st_size for path in output.rglob("*.json")) < 2 * 1024 * 1024

    workload = bind_formal_workload_authority(
        "livecodebench_v6_hard",
        livecodebench_path,
    )
    workload_path = (tmp_path / "workload-authority.json").resolve()
    publish_canonical_json_no_replace(
        workload_path,
        formal_workload_authority_cli_artifact(workload),
    )
    cell = MaterializedCell(
        stage="E3b",
        method_role="Static",
        model="fixture/target",
        backend="DFLASH",
        task="heldout_long_context_confirmation",
        publication_policy="none",
        recipe_sha256=None,
        dimensions=tuple(
            sorted(
                {
                    "block": 0,
                    "block_phase": "excluded_pilot",
                    "context": 16,
                    "load": "concurrency_one",
                    "regime": "short_input_long_generation",
                    "width_panel": "matched",
                }.items()
            )
        ),
    )
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E3b",
        protocol_lock_sha256="1" * 64,
        upstream_receipt_sha256s=("2" * 64,),
        source_decision_sha256="3" * 64,
        materialization_rule="test_exact_controlled_context_cell",
        expected_cell_count=1,
        cells=(cell,),
        gpu_hours=GpuHourEstimate(
            status="UNMEASURED",
            source_pilot_receipt_sha256=None,
            compute_gpu_hours=None,
            reserved_gpu_hours=None,
            estimated_wall_hours=None,
            retry_reserve_gpu_hours=None,
            profile_reserve_gpu_hours=None,
            evidence_reserve_gpu_hours=None,
        ),
    )
    materialization_path = (tmp_path / "materialization.json").resolve()
    publish_canonical_json_no_replace(
        materialization_path,
        stage_materialization_receipt_to_dict(materialization),
    )
    execution_source = SimpleNamespace(
        schema_version=3,
        content_source_binding=content_source,
        protocol_lock_source=SimpleNamespace(reopen=lambda **_kwargs: {}),
        materialization_source=SimpleNamespace(absolute_path=str(materialization_path)),
        materialization_sha256=materialization.sha256,
        stage="E3b",
    )
    monkeypatch.setattr(
        formal_single_operator_stages,
        "load_formal_single_operator_execution_source",
        lambda _path: execution_source,
    )
    monkeypatch.setattr(
        formal_registry,
        "protocol_lock_from_dict",
        lambda _value: SimpleNamespace(
            schema_version=5,
            trusted_single_operator_content_bundle_sha256=(
                content_source.content_sha256
            ),
        ),
    )
    monkeypatch.setattr(
        formal_registry,
        "stage_materialization_receipt_from_dict",
        lambda _value: materialization,
    )
    sampling = SamplingProfile()
    sampling_path = (tmp_path / "sampling.json").resolve()
    sampling.write(sampling_path)
    run_config_path = (tmp_path / "run-config.json").resolve()
    publish_canonical_json_no_replace(run_config_path, {"runtime": True})
    launch_semantic = CanonicalJsonProofBinding.bind(launch_path).semantic_sha256
    launch = SimpleNamespace(
        schema_version=2,
        sha256=launch_semantic,
        content_source_binding=content_source,
        formal_stage="E3b",
        run_config_path=str(run_config_path),
        run_config_semantic_sha256="4" * 64,
        sampling_profile_path=str(sampling_path),
        sampling_profile_sha256=sampling.sha256,
        tokenizer_content_authority_sha256=None,
        tokenizer_content_member_id=tokenizer.sha256,
        tokenizer_model_id=tokenizer.model_id,
        tokenizer_revision=tokenizer.revision,
        tokenizer_snapshot_path=tokenizer.local_snapshot_path,
    )
    monkeypatch.setattr(
        context_artifact.CompileLaunchManifest,
        "load",
        lambda _path: launch,
    )
    run_config = SimpleNamespace(
        runtime=SimpleNamespace(
            topology_mode="tp1_dp1",
            max_running_requests=1,
            context_length=40_928,
        )
    )
    monkeypatch.setattr(dispatch, "load_run_config", lambda _path: run_config)
    monkeypatch.setattr(dispatch, "run_config_sha256", lambda _config: "4" * 64)
    cell_output = (tmp_path / "cell-output").resolve()
    cell_output.mkdir(mode=0o700)
    receipt = dispatch.materialize_trusted_single_operator_request_schedule(
        execution_source_path=(tmp_path / "execution-source.json").resolve(),
        materialized_cell_id=cell.cell_id,
        compile_launch_manifest_path=launch_path,
        workload_source_path=workload_path,
        execution_binding_sha256="5" * 64,
        subject_sha256="6" * 64,
        private_output_root=cell_output,
        context_filler_artifact_path=binding.absolute_path,
    )
    assert receipt.schema_version == 7
    request_rows = tuple(dispatch.formal_serving_request_schedule_rows(receipt))
    assert request_rows
    assert all(
        len(row.request.input_token_ids) + row.request.requested_output_tokens == 16
        for row in request_rows
    )
    receipt.reopen()

    evidence = artifact.tokenization_evidence.reopen()
    first_output = Path(evidence["batches"][0]["tokenization_output"]["absolute_path"])
    value = json.loads(first_output.read_text())
    value["requests"][0]["input_token_ids"][0] += 1
    first_output.chmod(0o600)
    first_output.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    first_output.chmod(0o400)
    with pytest.raises(ValueError, match="changed"):
        context_artifact.load_trusted_context_filler_artifact(
            binding.absolute_path,
            content_source_binding=content_source,
            tokenizer_content_member_id=tokenizer.sha256,
            tokenizer_model_id=tokenizer.model_id,
            tokenizer_revision=tokenizer.revision,
        )
