from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_preflight_interference import (
    _control as _release_control,
)
from test_preflight_interference import (
    _release_control_context,
)

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.config import (
    ModelPair,
    RunConfig,
    RuntimeConfig,
    run_config_sha256,
)
from lightcone_spec.execution import ControlledExecutionPolicy
from lightcone_spec.experiments.formal_registry import (
    stage_materialization_receipt_to_dict,
)
from lightcone_spec.experiments.formal_single_operator_loads import (
    E5_DRAIN_DURATION_US,
    E5_REQUEST_DEADLINE_US,
    E5_WARMUP_DURATION_US,
    FORMAL_SINGLE_OPERATOR_E5_LOAD_PROTOCOL_SHA256,
    E3aLambdaStar,
    E5ArrivalPlan,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.load import (
    FrozenSamplingParameters,
    ImmutableRequest,
    RequestTemplate,
)
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    MaterializedCell,
    StageMaterializationReceipt,
    _materialize_tts_calibration_diagnostic,
)
from lightcone_spec.experiments.workload_authority import (
    FORMAL_WORKLOAD_PROTOCOLS,
    FormalWorkloadAuthority,
    FormalWorkloadSample,
    formal_workload_authority_cli_artifact,
    formal_workload_samples_sha256,
)
from lightcone_spec.orchestration import formal_physical_dispatch as dispatch
from lightcone_spec.orchestration import formal_serving_lift as serving_lift
from lightcone_spec.orchestration import formal_terminal_result as terminal_result
from lightcone_spec.orchestration.formal_single_operator_admission import (
    publish_formal_single_operator_admission,
)
from lightcone_spec.runtime.compile_runner import (
    COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
    CompileLaunchManifest,
)
from lightcone_spec.runtime.control_attestation import ChallengeReplayStore
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return dispatch._sha256({"test": label})


def _immutable(*, prompt_tokens: tuple[int, ...], phase: str, ordinal: int):
    split = "warmup" if phase == "warmup" else "confirmation"
    sampling = FrozenSamplingParameters.from_mapping({"temperature": 0.0, "top_p": 1.0})
    return ImmutableRequest.create(
        namespace="formal-test",
        split=split,
        ordinal=ordinal,
        template=RequestTemplate(
            input_token_ids=prompt_tokens,
            requested_output_tokens=8,
            sampling=sampling,
        ),
        arrival_us=ordinal * 100,
        cohort_id=f"cohort-{ordinal % 2}",
    )


def _source_row(
    *, prompt: str, tokens: tuple[int, ...], phase: str, ordinal: int, route=None
):
    request = _immutable(prompt_tokens=tokens, phase=phase, ordinal=ordinal)
    return dispatch.FormalServingRequestScheduleSourceRow(
        source_member_sha256=_sha("workload-member"),
        source_raw_file_sha256=_sha("workload-raw"),
        source_selected_rows_sha256=_sha("workload-selected"),
        source_sample_id=f"sample-{ordinal}",
        prompt=prompt,
        prompt_sha256=dispatch._sha256(prompt),
        phase=phase,
        namespace=request.namespace,
        split=request.split,
        ordinal=request.ordinal,
        requested_output_tokens=request.requested_output_tokens,
        arrival_us=request.arrival_us,
        cancellation_offset_us=None,
        cohort_id=request.cohort_id,
        routed_dp_rank=route,
        sampling=request.sampling.items,
    )


def _source(*, subject_sha256: str, topology: str):
    routes = (None, None) if topology != "tp1_dp2" else (0, 1)
    rows = (
        _source_row(
            prompt="alpha beta",
            tokens=(11, 12),
            phase="warmup",
            ordinal=0,
            route=routes[0],
        ),
        _source_row(
            prompt="gamma delta",
            tokens=(21, 22),
            phase="scored",
            ordinal=1,
            route=routes[1],
        ),
    )
    return dispatch.FormalServingRequestScheduleSource(
        schema_version=3,
        kind="formal_serving_request_schedule_source",
        protocol_sha256=dispatch.FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        derivation_protocol_sha256=(
            dispatch.FORMAL_SERVING_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
        ),
        subject_sha256=subject_sha256,
        materialization_receipt_sha256=_sha("materialization"),
        materialized_cell_id=_sha("cell"),
        workload_authority_sha256=_sha("workload-authority"),
        workload_id="livecodebench_v6_hard",
        workload_source_descriptor_sha256=_sha("workload-member"),
        workload_source_authority_sha256=_sha("workload-source-authority"),
        tts_tuning_window_sha256=None,
        tts_tuning_entry_ids=(),
        sampling_profile_sha256=_sha("sampling-profile"),
        load_protocol_sha256=_sha("load-protocol"),
        context_tokens=4096,
        regime="short_input_long_generation",
        arrival_policy="closed_loop_zero_think",
        max_running_requests=1,
        cohort_count=4 if topology == "tp1_dp2" else 1,
        topology_mode=topology,
        tokenizer_content_member_id="tokenizer-member",
        tokenizer_model_id="example/tokenizer",
        tokenizer_revision="1" * 40,
        tokenizer_content_authority_sha256=_sha("tokenizer-authority"),
        requests=rows,
    )


def test_schedule_source_codec_and_dp2_sticky_routing_fail_closed() -> None:
    source = _source(subject_sha256=_sha("subject"), topology="tp1_dp2")
    assert (
        dispatch.FormalServingRequestScheduleSource.from_dict(source.to_dict())
        == source
    )
    crossed = source.requests[1].to_dict()
    crossed["cohort_id"] = source.requests[0].cohort_id
    with pytest.raises(ValueError, match="sticky"):
        dispatch.FormalServingRequestScheduleSource(
            **{
                **source.to_dict(),
                "tts_tuning_entry_ids": source.tts_tuning_entry_ids,
                "requests": (
                    source.requests[0],
                    dispatch.FormalServingRequestScheduleSourceRow.from_dict(crossed),
                ),
            }
        )


def test_trusted_schedule_source_shards_cover_ten_thousand_rows(
    tmp_path: Path,
) -> None:
    base = _source(subject_sha256=_sha("subject"), topology="tp1_dp1")
    template = base.requests[1]
    rows = tuple(
        replace(
            template,
            source_sample_id=f"sample-{ordinal}",
            phase="warmup" if ordinal == 0 else "scored",
            split="warmup" if ordinal == 0 else "confirmation",
            ordinal=ordinal,
            arrival_us=ordinal,
        )
        for ordinal in range(10_000)
    )
    trusted = replace(
        base,
        schema_version=5,
        protocol_sha256=(
            dispatch.TRUSTED_SINGLE_OPERATOR_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        ),
        derivation_protocol_sha256=(
            dispatch.TRUSTED_SINGLE_OPERATOR_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
        ),
        workload_authority_sha256=None,
        workload_source_descriptor_sha256=_sha("workload-member"),
        workload_source_authority_sha256=None,
        tokenizer_content_authority_sha256=None,
        requests=rows,
        content_source_binding_sha256=_sha("content-source"),
        trusted_workload_member_sha256=_sha("workload-member"),
    )
    root = tmp_path / "source-shards"
    root.mkdir()

    sharded = dispatch.publish_trusted_schedule_source_shards(
        source=trusted,
        output_directory=root,
    )

    assert sharded.schema_version == 6
    assert sharded.requests == ()
    assert sharded.request_count == 10_000
    assert (
        dispatch.FormalServingRequestScheduleSource.from_dict(sharded.to_dict())
        == sharded
    )
    reopened = tuple(dispatch.formal_serving_request_schedule_source_rows(sharded))
    assert reopened == rows
    assert max(path.stat().st_size for path in root.iterdir()) < 2 * 1024 * 1024

    first = json.loads(Path(sharded.requests_shard_index.absolute_path).read_text())
    first_shard = Path(first["shards"][0]["binding"]["absolute_path"])
    mutated = json.loads(first_shard.read_text())
    mutated["rows"][0]["ordinal"] = 1
    first_shard.chmod(0o600)
    first_shard.write_text(
        json.dumps(mutated, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    first_shard.chmod(0o400)
    with pytest.raises(ValueError, match="changed"):
        tuple(dispatch.formal_serving_request_schedule_source_rows(sharded))


def test_controlled_context_schema7_roundtrip_and_shard_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from lightcone_spec.experiments.formal_content_source import (
        FormalContentSourceBinding,
    )
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundleBinding,
    )
    from lightcone_spec.experiments.formal_single_operator_context_compiler import (
        FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256,
        ContextFillerAuthority,
        TokenizedContextSourceRow,
        compile_context_requests,
    )

    base = _source(subject_sha256=_sha("context-subject"), topology="tp1_dp1")
    uncompiled = replace(
        base,
        schema_version=5,
        protocol_sha256=(
            dispatch.TRUSTED_SINGLE_OPERATOR_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        ),
        derivation_protocol_sha256=(
            dispatch.TRUSTED_SINGLE_OPERATOR_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
        ),
        workload_authority_sha256=None,
        workload_source_descriptor_sha256=_sha("workload-member"),
        workload_source_authority_sha256=None,
        tokenizer_content_member_id=_sha("context-tokenizer-member"),
        tokenizer_content_authority_sha256=None,
        context_tokens=16,
        regime="short_input_long_generation",
        content_source_binding_sha256=_sha("content-source"),
        trusted_workload_member_sha256=_sha("workload-member"),
    )
    uncompiled_path = (tmp_path / "uncompiled.json").resolve()
    publish_canonical_json_no_replace(uncompiled_path, uncompiled.to_dict())
    uncompiled_binding = CanonicalJsonProofBinding.bind(
        uncompiled_path,
        semantic_sha256=uncompiled.sha256,
    )
    filler_path = (tmp_path / "filler.json").resolve()
    publish_canonical_json_no_replace(filler_path, {"fixture": "filler"})
    filler_binding = CanonicalJsonProofBinding.bind(filler_path)
    core_rows = tuple(
        TokenizedContextSourceRow(
            tokenizer_content_member_id=uncompiled.tokenizer_content_member_id,
            tokenizer_model_id=uncompiled.tokenizer_model_id,
            tokenizer_revision=uncompiled.tokenizer_revision,
            source_member_sha256=row.source_member_sha256,
            source_sample_id=row.source_sample_id,
            prompt_sha256=row.prompt_sha256,
            input_token_ids=(ordinal + 1, ordinal + 2),
        )
        for ordinal, row in enumerate(uncompiled.requests)
    )
    authority = ContextFillerAuthority(
        schema_version=1,
        kind="formal_single_operator_context_filler_authority",
        protocol_sha256=(FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256),
        content_source_binding_sha256=_sha("content-source"),
        tokenizer_content_member_id=uncompiled.tokenizer_content_member_id,
        tokenizer_model_id=uncompiled.tokenizer_model_id,
        tokenizer_revision=uncompiled.tokenizer_revision,
        registered_source_member_sha256s=(_sha("workload-member"),),
        rows=core_rows,
    )
    compiled = compile_context_requests(
        regime="short_input_long_generation",
        context_tokens=16,
        core_rows=core_rows,
        filler_authority=authority,
    )
    adjusted = replace(
        uncompiled,
        requests=tuple(
            replace(
                row,
                requested_output_tokens=compiled_row.requested_output_tokens,
                sampling=FrozenSamplingParameters.from_mapping(
                    {
                        **dict(row.sampling),
                        "max_new_tokens": compiled_row.requested_output_tokens,
                    }
                ).items,
            )
            for row, compiled_row in zip(
                uncompiled.requests,
                compiled,
                strict=True,
            )
        ),
    )
    compiled_root = tmp_path / "compiled"
    source_root = tmp_path / "source"
    compiled_root.mkdir()
    source_root.mkdir()
    schema7 = dispatch.publish_trusted_controlled_context_schedule_source_shards(
        source=adjusted,
        uncompiled_source=uncompiled_binding,
        context_filler_artifact=filler_binding,
        compiled=compiled,
        compiled_output_directory=compiled_root,
        source_output_directory=source_root,
    )

    assert (
        dispatch.FormalServingRequestScheduleSource.from_dict(schema7.to_dict())
        == schema7
    )
    assert tuple(dispatch.formal_serving_request_schedule_source_rows(schema7)) == (
        adjusted.requests
    )
    assert tuple(dispatch.formal_serving_controlled_context_requests(schema7)) == (
        compiled
    )
    assert all(
        path.stat().st_size < 2 * 1024 * 1024
        for path in (*compiled_root.iterdir(), *source_root.iterdir())
    )

    schedule_path = (tmp_path / "schedule.json").resolve()
    publish_canonical_json_no_replace(schedule_path, schema7.to_dict())
    from lightcone_spec.runtime.content_authorization import (
        ContentJsonArtifactBinding,
    )

    schedule_binding = ContentJsonArtifactBinding.from_path(
        "derived_formal_serving_request_schedule:test",
        schedule_path,
    )
    fixture_path = (tmp_path / "fixture.json").resolve()
    publish_canonical_json_no_replace(fixture_path, {"fixture": True})
    fixture_binding = CanonicalJsonProofBinding.bind(fixture_path)
    workload_binding = ContentJsonArtifactBinding.from_path(
        "formal_workload:livecodebench_v6_hard",
        fixture_path,
    )
    bundle_path = (tmp_path / "bundle.json").resolve()
    publish_canonical_json_no_replace(bundle_path, {"bundle": True})
    bundle_semantic = _sha("bundle-semantic")
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: SimpleNamespace(
            runtime_binding_status="BOUND",
            semantic_sha256=bundle_semantic,
        ),
    )
    trusted_bundle_binding = TrustedSingleOperatorContentBundleBinding(
        absolute_path=str(bundle_path),
        size=bundle_path.stat().st_size,
        raw_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        semantic_sha256=bundle_semantic,
        runtime_binding_status="BOUND",
    )
    content_source = FormalContentSourceBinding(
        schema_version=1,
        kind="formal_content_source_binding",
        mode="trusted_single_operator",
        offline_root_signed=None,
        trusted_single_operator=trusted_bundle_binding,
    )
    receipt_rows = dispatch._materialized_controlled_context_rows(
        source=schema7,
        compiled=compiled,
    )
    base_receipt = dispatch.FormalServingRequestScheduleReceipt(
        schema_version=5,
        kind="formal_serving_request_schedule_receipt",
        protocol_sha256=(
            dispatch.TRUSTED_SINGLE_OPERATOR_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        ),
        formal_execution_authorized=False,
        execution_binding_sha256=_sha("execution"),
        subject_sha256=schema7.subject_sha256,
        materialized_cell_id=schema7.materialized_cell_id,
        workload_authority_sha256=None,
        content_verification_receipt_sha256=None,
        topology_mode=schema7.topology_mode,
        materialization=fixture_binding,
        content_verification_receipt=None,
        workload_source=workload_binding,
        compile_launch_manifest=fixture_binding,
        sampling_profile=fixture_binding,
        schedule_source=schedule_binding,
        tokenization_input=filler_binding,
        tokenization_output=schema7.compiled_context_requests_shard_index,
        tokenizer_worker_source_raw_sha256=_sha("worker"),
        tokenizer_worker_source_size=1,
        tokenizer_worker_argv_sha256=_sha("argv"),
        tokenizer_model_id=schema7.tokenizer_model_id,
        tokenizer_revision=schema7.tokenizer_revision,
        tokenizer_snapshot_path=str((tmp_path / schema7.tokenizer_revision).resolve()),
        tokenizer_content_member_id=schema7.tokenizer_content_member_id,
        tokenizer_content_authority_sha256=None,
        transformers_version="test",
        tokenizer_class="TestTokenizer",
        tokenizer_vocab_size=128,
        requests=receipt_rows,
        content_source_binding=content_source,
        trusted_workload_member_sha256=schema7.trusted_workload_member_sha256,
    )
    receipt_root = tmp_path / "receipt"
    receipt_root.mkdir()
    schema7_receipt = (
        dispatch.publish_trusted_controlled_context_schedule_receipt_shards(
            receipt=base_receipt,
            uncompiled_source=uncompiled_binding,
            context_filler_artifact=filler_binding,
            compiled_context_requests_shard_index=(
                schema7.compiled_context_requests_shard_index
            ),
            output_directory=receipt_root,
        )
    )
    assert (
        dispatch.FormalServingRequestScheduleReceipt.from_dict(
            schema7_receipt.to_dict()
        )
        == schema7_receipt
    )
    assert tuple(dispatch.formal_serving_request_schedule_rows(schema7_receipt)) == (
        receipt_rows
    )

    compiled_index = json.loads(
        Path(schema7.compiled_context_requests_shard_index.absolute_path).read_text()
    )
    compiled_shard = Path(compiled_index["shards"][0]["binding"]["absolute_path"])
    value = json.loads(compiled_shard.read_text())
    value["rows"][0]["requested_output_tokens"] -= 1
    compiled_shard.chmod(0o600)
    compiled_shard.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    compiled_shard.chmod(0o400)
    with pytest.raises(ValueError, match="changed"):
        tuple(dispatch.formal_serving_controlled_context_requests(schema7))


@pytest.mark.parametrize(
    "prompt",
    (
        "Solve the task.\n```python\ndef answer(x):\n    return x + 1\n```\n",
        "def has_close_elements(numbers, threshold):\n    # HumanEval\n    pass\n",
        "def windows_line_endings():\r\n    return 'preserved'\r\n",
    ),
)
def test_schedule_source_preserves_multiline_task_native_prompts(prompt: str) -> None:
    row = _source_row(
        prompt=prompt,
        tokens=(1, 2, 3),
        phase="scored",
        ordinal=0,
    )
    reopened = dispatch.FormalServingRequestScheduleSourceRow.from_dict(row.to_dict())
    assert reopened.prompt == prompt
    assert reopened.prompt_sha256 == dispatch._sha256(prompt)

    with pytest.raises(ValueError, match="NFC"):
        _source_row(
            prompt="Cafe\u0301\npass\n",
            tokens=(1,),
            phase="scored",
            ordinal=1,
        )
    with pytest.raises(ValueError, match="NFC"):
        _source_row(
            prompt="def f():\n\x00pass",
            tokens=(1,),
            phase="scored",
            ordinal=2,
        )


def test_schedule_source_schema4_path_binds_e5_arrival_plan(tmp_path) -> None:
    source = _source(subject_sha256=_sha("subject"), topology="tp1_dp1")
    plan = E5ArrivalPlan(
        schema_version=1,
        kind="formal_single_operator_e5_arrival_plan",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_E5_LOAD_PROTOCOL_SHA256,
        cell_id=source.materialized_cell_id,
        paired_trace_sha256=_sha("paired-trace"),
        block=0,
        family="closed_loop",
        arrival_policy="closed_loop_zero_think",
        lambda_star=E3aLambdaStar(
            numerator_requests_x_1e9=1_000_000_000,
            denominator_window_ns=1_000_000_000,
            source_cell_id=_sha("lambda-star-cell"),
            source_observation_sha256=_sha("lambda-star-observation"),
            common_load=1,
            matched_width=8,
            rule=(
                "E3a_Static_context_40928_short_input_long_generation_"
                "matched_width_common_load_completed_requests_per_second"
            ),
        ),
        effective_rate_numerator=None,
        effective_rate_denominator=None,
        concurrency=1,
        arrival_duration_us=1_000_000,
        warmup_duration_us=E5_WARMUP_DURATION_US,
        request_deadline_us=E5_REQUEST_DEADLINE_US,
        drain_duration_us=E5_DRAIN_DURATION_US,
        arrivals_us=(0,),
        burstgpt_window=None,
        p99_extension_minimum_completed=None,
        p99_extension_offered_requests=None,
    )
    plan_path = tmp_path / "e5-arrival-plan.json"
    publish_canonical_json_no_replace(plan_path, plan.to_dict())
    binding = CanonicalJsonProofBinding.bind(plan_path)
    current = replace(
        source,
        schema_version=4,
        load_protocol_sha256=plan.sha256,
        e5_arrival_plan=binding,
    )
    assert (
        dispatch.FormalServingRequestScheduleSource.from_dict(current.to_dict())
        == current
    )
    with pytest.raises(ValueError, match="legacy formal schedule"):
        replace(current, schema_version=3)


def test_e5_policy_requires_its_path_bound_arrival_plan() -> None:
    schedule = SimpleNamespace(schema_version=4, e5_arrival_plan=None)
    with pytest.raises(ValueError, match="path-bound arrival plan"):
        dispatch._registered_serving_execution_policy(stage="E5", schedule=schedule)


@pytest.mark.parametrize(
    ("topology", "allowed"),
    (
        ("tp1_dp1", frozenset({"tp1_dp1"})),
        ("tp2_dp1", frozenset({"tp2_dp1", "tp1_dp2"})),
        ("tp1_dp2", frozenset({"tp2_dp1", "tp1_dp2"})),
    ),
)
def test_schema4_exact_pool_cap_exceeds_one_hour_for_every_runner_topology(
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
    allowed: frozenset[str],
) -> None:
    from lightcone_spec.orchestration.executor import (
        RegisteredServingExecutionPolicy,
    )

    policy = RegisteredServingExecutionPolicy(
        schema_version=1,
        kind="registered_serving_execution_policy",
        source_kind="closed_loop",
        warmup_duration_us=60_000_000,
        arrival_duration_us=300_000_000,
        request_deadline_us=120_000_000,
        drain_duration_us=60_000_000,
        max_concurrency=64,
        complete_closed_loop_pool=True,
    )
    rows = (SimpleNamespace(phase="warmup"),) + tuple(
        SimpleNamespace(phase="scored") for _ in range(11_000)
    )
    schedule = SimpleNamespace()
    monkeypatch.setattr(
        dispatch,
        "formal_serving_request_schedule_rows",
        lambda value: rows if value is schedule else (),
    )
    timeout_ns = dispatch._registered_process_hard_timeout_ns(
        policy=policy,
        schedule=schedule,
    )
    expected_waves = 1 + (11_000 + policy.max_concurrency - 1) // (
        policy.max_concurrency
    )
    assert (
        timeout_ns
        == (
            expected_waves * (policy.request_deadline_us + 60_000_000)
            + dispatch._CURRENT_PROCESS_STARTUP_RESERVE_US
            + dispatch._CURRENT_PROCESS_CLEANUP_RESERVE_US
        )
        * 1_000
    )
    assert timeout_ns > 3_600 * 1_000_000_000
    plan = SimpleNamespace(
        schema_version=4,
        topology_mode=topology,
        serving_execution_policy=policy,
        process_hard_timeout_ns=timeout_ns,
    )
    assert dispatch._registered_plan_process_hard_timeout_seconds(
        plan,
        allowed_topologies=allowed,
    ) == pytest.approx(timeout_ns / 1_000_000_000)

    plan.process_hard_timeout_ns = policy.minimum_process_timeout_us * 1_000 - 1
    with pytest.raises(ValueError, match="outside source bounds"):
        dispatch._registered_plan_process_hard_timeout_seconds(
            plan,
            allowed_topologies=allowed,
        )


def test_runtime_contract_excludes_attempt_paths_and_rejects_timeout_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lightcone_spec.orchestration.executor import (
        RegisteredServingExecutionPolicy,
    )

    policy = RegisteredServingExecutionPolicy(
        schema_version=1,
        kind="registered_serving_execution_policy",
        source_kind="scheduled",
        warmup_duration_us=600_000_000,
        arrival_duration_us=3_600_000_000,
        request_deadline_us=600_000_000,
        drain_duration_us=120_000_000,
        max_concurrency=1,
        complete_closed_loop_pool=False,
    )
    timeout_ns = 5_640_000_000_000
    root = tmp_path.resolve()

    def plan(*, attempt: str, cap: int = timeout_ns):
        return SimpleNamespace(
            schema_version=4,
            protocol_sha256=_sha("physical-protocol"),
            execution_binding_sha256=_sha("execution-binding"),
            subject_sha256=_sha("subject"),
            materialized_cell_id=_sha("cell"),
            stage="E5",
            method="lightcone",
            topology_mode="tp1_dp1",
            inventory_sha256=_sha("inventory"),
            gpu_uuids=("GPU-test-0",),
            runtime_gpu_proof_sha256s=(_sha("gpu-proof"),),
            nextn_tp2_authority_sha256=None,
            native_terminal_binding=SimpleNamespace(
                begin_payload=lambda: {"run": "source-owned"}
            ),
            serving_execution_policy=policy,
            process_hard_timeout_ns=cap,
            sha256=_sha(f"plan-{attempt}"),
            server_log_output_path=str(root / f"{attempt}.server.log"),
            server_stdout_output_path=str(root / f"{attempt}.stdout.log"),
            server_stderr_output_path=str(root / f"{attempt}.stderr.log"),
        )

    launch = SimpleNamespace(sha256=_sha("launch"))
    schedule = SimpleNamespace(sha256=_sha("schedule"))
    current = {"plan": plan(attempt="one")}
    monkeypatch.setattr(
        dispatch,
        "_load_formal_single_operator_trusted_run_plan",
        lambda _path: (current["plan"], launch, schedule),
    )
    first = dispatch.formal_serving_process_runtime_contract(root / "plan-one.json")
    assert first.process_hard_timeout_ns > 3_600 * 1_000_000_000
    assert (
        first.outer_max_runtime_seconds
        == (timeout_ns + dispatch.FORMAL_SERVING_TERMINAL_PUBLICATION_GRACE_NS)
        // 1_000_000_000
    )

    current["plan"] = plan(attempt="two")
    second = dispatch.formal_serving_process_runtime_contract(root / "plan-two.json")
    assert second.plan_sha256 != first.plan_sha256
    assert second.progress_log_paths != first.progress_log_paths
    assert second.scientific_command_sha256 == first.scientific_command_sha256

    current["plan"] = plan(
        attempt="tampered",
        cap=policy.minimum_process_timeout_us * 1_000 - 1,
    )
    with pytest.raises(ValueError, match="outside source bounds"):
        dispatch.formal_serving_process_runtime_contract(root / "tampered.json")


def test_five_actual_roles_share_trusted_and_controlled_context_request_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lightcone_spec.experiments.formal_content_source import (
        FormalContentSourceBinding,
    )
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundle,
        TrustedSingleOperatorContentBundleBinding,
    )
    from lightcone_spec.experiments.formal_single_operator_context_compiler import (
        FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256,
        ContextFillerAuthority,
        TokenizedContextSourceRow,
        compile_context_requests,
    )
    from lightcone_spec.experiments.serving import BoundServingRequest
    from lightcone_spec.experiments.workload_authority import (
        formal_workload_authority_artifact_id,
    )
    from lightcone_spec.runtime.content_authorization import (
        ContentJsonArtifactBinding,
    )

    dimensions = tuple(
        sorted(
            {
                "block": 0,
                "block_phase": "excluded_pilot",
                "context": 4096,
                "load": "concurrency_one",
                "regime": "short_input_long_generation",
                "width_panel": "matched",
            }.items()
        )
    )
    roles = ("Target-only", "Static", "TTS", "L0-naive", "LightCone")
    cells = tuple(
        sorted(
            (
                MaterializedCell(
                    stage="E3b",
                    method_role=role,
                    model="Qwen/Qwen3-8B",
                    backend="NONE" if role == "Target-only" else "DFLASH",
                    task="heldout_long_context_confirmation",
                    publication_policy=(
                        "fixed_barrier"
                        if role == "TTS"
                        else "first_ready"
                        if role in {"L0-naive", "LightCone"}
                        else "none"
                    ),
                    recipe_sha256=(
                        _sha("tts-recipe")
                        if role in {"TTS", "L0-naive"}
                        else _sha("lightcone-recipe")
                        if role == "LightCone"
                        else None
                    ),
                    dimensions=dimensions,
                )
                for role in roles
            ),
            key=lambda row: row.cell_id,
        )
    )
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E3b",
        protocol_lock_sha256=_sha("protocol-lock"),
        upstream_receipt_sha256s=(_sha("upstream"),),
        source_decision_sha256=_sha("decision"),
        materialization_rule="five_actual_roles_pairing_fixture",
        expected_cell_count=len(cells),
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    paired_pool_ids = {
        dispatch._load_protocol_for_cell(
            cell=cell,
            max_running_requests=1,
            server_context_limit=40_960,
        )["paired_request_pool_sha256"]
        for cell in cells
    }
    assert len(paired_pool_ids) == 1
    next_block_dimensions = dict(cells[0].dimensions)
    next_block_dimensions["block"] = 1
    next_block = replace(
        cells[0],
        dimensions=tuple(sorted(next_block_dimensions.items())),
    )
    assert (
        dispatch._load_protocol_for_cell(
            cell=next_block,
            max_running_requests=1,
            server_context_limit=40_960,
        )["paired_request_pool_sha256"]
        not in paired_pool_ids
    )
    raw_path = (tmp_path / "lcb.json").resolve()
    raw_path.write_text('{"fixture":true}\n', encoding="utf-8")
    samples = tuple(
        FormalWorkloadSample(
            source_row_id=f"row-{index}",
            sample_id=f"sample-{index}",
            prompt=f"prompt {index}",
            seed=index + 1,
        )
        for index in range(6)
    )
    workload = FormalWorkloadAuthority(
        schema_version=1,
        kind="formal_workload_authority",
        workload_id="livecodebench_v6_hard",
        raw_source_path=str(raw_path),
        raw_file_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        repository_revision="a" * 40,
        raw_row_count=len(samples),
        selected_row_count=len(samples),
        selected_rows_sha256=formal_workload_samples_sha256(samples),
        source_lock_sha256=_sha("source-lock"),
        protocol_sha256=FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"].sha256,
        samples=samples,
    )
    workload_path = (tmp_path / "workload.json").resolve()
    publish_canonical_json_no_replace(
        workload_path,
        formal_workload_authority_cli_artifact(workload),
    )
    workload_binding = ContentJsonArtifactBinding.from_path(
        formal_workload_authority_artifact_id(workload.workload_id),
        workload_path,
    )
    workload_member_sha256 = _sha("trusted-workload-member")
    tokenizer_member_sha256 = _sha("trusted-tokenizer-member")
    fake_bundle = object.__new__(TrustedSingleOperatorContentBundle)
    object.__setattr__(fake_bundle, "runtime_binding_status", "BOUND")
    object.__setattr__(fake_bundle, "semantic_sha256", _sha("trusted-bundle"))
    object.__setattr__(
        fake_bundle,
        "model_members",
        (
            SimpleNamespace(
                role="tokenizer",
                stages=("E3b",),
                model_id="example/tokenizer",
                revision="1" * 40,
                sha256=tokenizer_member_sha256,
            ),
        ),
    )
    object.__setattr__(
        fake_bundle,
        "locked_workloads",
        (
            SimpleNamespace(
                workload_id=workload.workload_id,
                authority_sha256=workload.sha256,
                raw_source_path=workload.raw_source_path,
                raw_file_sha256=workload.raw_file_sha256,
                repository_revision=workload.repository_revision,
                raw_row_count=workload.raw_row_count,
                selected_row_count=workload.selected_row_count,
                formal_samples_sha256=workload.selected_rows_sha256,
                source_lock_sha256=workload.source_lock_sha256,
                protocol_sha256=workload.protocol_sha256,
                sha256=workload_member_sha256,
            ),
        ),
    )
    object.__setattr__(fake_bundle, "e0_task_native_descriptors", ())
    bundle_path = (tmp_path / "content-bundle.json").resolve()
    bundle_path.write_text('{"fixture":true}\n', encoding="utf-8")
    bundle_binding = TrustedSingleOperatorContentBundleBinding(
        absolute_path=str(bundle_path),
        size=bundle_path.stat().st_size,
        raw_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        semantic_sha256=fake_bundle.semantic_sha256,
        runtime_binding_status="BOUND",
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: fake_bundle,
    )
    content_source = FormalContentSourceBinding(
        schema_version=1,
        kind="formal_content_source_binding",
        mode="trusted_single_operator",
        offline_root_signed=None,
        trusted_single_operator=bundle_binding,
    )

    sources = tuple(
        dispatch.rebuild_trusted_single_operator_request_schedule_source(
            subject_sha256=_sha(f"subject-{cell.method_role}"),
            content_source_binding=content_source,
            topology_mode="tp1_dp1",
            materialization=materialization,
            materialized_cell_id=cell.cell_id,
            workload_source=workload,
            workload_source_binding=workload_binding,
            sampling_profile=SamplingProfile(),
            max_running_requests=1,
            server_context_limit=40_960,
            tokenizer_content_member_id=tokenizer_member_sha256,
            tokenizer_model_id="example/tokenizer",
            tokenizer_revision="1" * 40,
        )
        for cell in cells
    )
    scientific_rows = tuple(
        tuple(
            (
                row.namespace,
                row.split,
                row.ordinal,
                row.prompt,
                row.requested_output_tokens,
                row.arrival_us,
                row.cohort_id,
                row.sampling,
            )
            for row in source.requests
        )
        for source in sources
    )
    assert len(set(scientific_rows)) == 1

    controlled_request_ids = []
    controlled_rows_by_role: dict[str, tuple[object, ...]] = {}
    source_rows_by_object: dict[int, tuple[object, ...]] = {}
    for cell, source in zip(cells, sources, strict=True):
        core_rows = tuple(
            TokenizedContextSourceRow(
                tokenizer_content_member_id=tokenizer_member_sha256,
                tokenizer_model_id=source.tokenizer_model_id,
                tokenizer_revision=source.tokenizer_revision,
                source_member_sha256=row.source_member_sha256,
                source_sample_id=row.source_sample_id,
                prompt_sha256=row.prompt_sha256,
                input_token_ids=(row.ordinal + 1, row.ordinal + 101),
            )
            for row in source.requests
        )
        filler = ContextFillerAuthority(
            schema_version=1,
            kind="formal_single_operator_context_filler_authority",
            protocol_sha256=FORMAL_SINGLE_OPERATOR_CONTEXT_COMPILER_PROTOCOL_SHA256,
            content_source_binding_sha256=content_source.sha256,
            tokenizer_content_member_id=tokenizer_member_sha256,
            tokenizer_model_id=source.tokenizer_model_id,
            tokenizer_revision=source.tokenizer_revision,
            registered_source_member_sha256s=(workload_member_sha256,),
            rows=core_rows,
        )
        compiled = compile_context_requests(
            regime="short_input_long_generation",
            context_tokens=source.context_tokens,
            core_rows=core_rows,
            filler_authority=filler,
        )
        adjusted = tuple(
            replace(
                row,
                requested_output_tokens=compiled_row.requested_output_tokens,
                sampling=FrozenSamplingParameters.from_mapping(
                    {
                        **dict(row.sampling),
                        "max_new_tokens": compiled_row.requested_output_tokens,
                    }
                ).items,
            )
            for row, compiled_row in zip(source.requests, compiled, strict=True)
        )
        schema7 = SimpleNamespace(schema_version=7, topology_mode="tp1_dp1")
        source_rows_by_object[id(schema7)] = adjusted
        monkeypatch.setattr(
            dispatch,
            "formal_serving_request_schedule_source_rows",
            lambda value, rows_by_id=source_rows_by_object: rows_by_id[id(value)],
        )
        materialized = dispatch._materialized_controlled_context_rows(
            source=schema7,
            compiled=compiled,
        )
        assert all(type(row.request) is BoundServingRequest for row in materialized)
        controlled_request_ids.append(
            tuple(row.request.request_id for row in materialized)
        )
        controlled_rows_by_role[cell.method_role] = materialized
    assert len(set(controlled_request_ids)) == 1

    # Carry the real materialized five-role request pool through the terminal
    # projection and the production pairing reducer.  Response evidence is
    # method-specific, while the registered source-pool identity is not.
    from lightcone_spec.experiments import formal_single_operator_downstream

    pool_sha256s = {
        role: dispatch._sha256(
            [
                {
                    "request_id": row.request.request_id,
                    "input_token_ids": list(row.request.input_token_ids),
                }
                for row in rows
            ]
        )
        for role, rows in controlled_rows_by_role.items()
    }
    assert len(set(pool_sha256s.values())) == 1
    cells_by_role = {cell.method_role: cell for cell in cells}
    reducer_rows = {}
    for role, materialized_rows in controlled_rows_by_role.items():
        terminal_rows = []
        for index, row in enumerate(
            sorted(materialized_rows, key=lambda value: value.request.request_id)
        ):
            started_ns = 1 + index * 1_000
            terminal_rows.append(
                {
                    "request_id": row.request.request_id,
                    "input_token_ids": list(row.request.input_token_ids),
                    "output_token_ids": [90_001, 90_002],
                    "request_started_ns": started_ns,
                    "request_terminal_ns": started_ns + 500,
                    "token_observed_ns": [started_ns + 100, started_ns + 200],
                    "terminal_status": "completed",
                    "terminal_reason": "FINISH_LENGTH",
                    "submitted_to_server": True,
                }
            )
        # A fixed-window closed-loop method may realize a shorter contiguous
        # prefix without changing its complete registered pool identity.
        if role == "TTS":
            terminal_rows.pop()
        reducer_rows[role] = (
            cells_by_role[role],
            {
                "source_request_pool_sha256": pool_sha256s[role],
                "requests": terminal_rows,
            },
        )
    paired = formal_single_operator_downstream._paired_role_goodputs(reducer_rows)
    assert set(paired) == set(roles)
    assert len({row.source_request_pool_sha256 for row in paired.values()}) == 1


def test_tts_registry_window_label_maps_to_source_owned_closed_loop_arrival() -> None:
    source = build_industrial_registry().cells_for("TTS-Cal")[0]
    cell = MaterializedCell(
        stage="TTS-Cal",
        method_role="TTS",
        model=source.identity.model,
        backend=source.identity.backend,
        task=source.identity.task,
        publication_policy="fixed_barrier",
        recipe_sha256=_sha("tts-recipe"),
        dimensions=(("registry_cell_id", source.cell_id),),
    )
    protocol = dispatch._load_protocol_for_cell(
        cell=cell,
        max_running_requests=1,
        server_context_limit=40_960,
    )
    assert protocol["arrival_policy"] == "closed_loop_zero_think"
    assert protocol["context_tokens"] == source.identity.context
    assert protocol["regime"] == source.identity.regime
    assert dispatch._workload_id_for_cell(cell) == "livecodebench_v6_hard"


def test_tts_four_blocks_replay_same_76_rows_and_never_schedule_four_holdouts(
    tmp_path: Path,
) -> None:
    from lightcone_spec.runtime.content_authorization import (
        TTS_CALIBRATION_TUNING_SELECTOR_NAMESPACE,
        TtsCalibrationTuningWindow,
        TtsCalibrationTuningWindowEntry,
    )

    descriptor_sha256 = _sha("trusted-locked-lcb-h80")
    samples = tuple(
        FormalWorkloadSample(
            source_row_id=f"problem-{index:03d}",
            sample_id=f"sample-{index:03d}",
            prompt=f"Solve exact problem {index:03d}.",
            seed=index + 1,
        )
        for index in range(80)
    )
    workload = FormalWorkloadAuthority(
        schema_version=1,
        kind="formal_workload_authority",
        workload_id="livecodebench_v6_hard",
        raw_source_path=str((tmp_path / "lcb-h80.json").resolve()),
        raw_file_sha256=_sha("lcb-h80-raw"),
        repository_revision="3" * 40,
        raw_row_count=175,
        selected_row_count=len(samples),
        selected_rows_sha256=formal_workload_samples_sha256(samples),
        source_lock_sha256=_sha("lcb-h80-lock"),
        protocol_sha256=FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"].sha256,
        samples=samples,
    )
    ranked = tuple(
        sorted(
            samples,
            key=lambda row: (
                dispatch._sha256(
                    {
                        "selector_namespace": (
                            TTS_CALIBRATION_TUNING_SELECTOR_NAMESPACE
                        ),
                        "source_problem_id": row.source_row_id,
                    }
                ),
                row.source_row_id,
                row.sample_id,
            ),
        )
    )
    holdout_problem_ids = {row.source_row_id for row in ranked[:4]}

    def entry(sample: FormalWorkloadSample) -> TtsCalibrationTuningWindowEntry:
        return TtsCalibrationTuningWindowEntry(
            workload_id="livecodebench_v6_hard",
            source_problem_id=sample.source_row_id,
            source_sample_id=sample.sample_id,
            source_descriptor_sha256=descriptor_sha256,
            prompt_sha256=dispatch._sha256(sample.prompt),
        )

    tuning_entries = tuple(
        sorted(
            (
                entry(row)
                for row in samples
                if row.source_row_id not in holdout_problem_ids
            ),
            key=lambda row: row.entry_id,
        )
    )
    holdout_entries = tuple(
        sorted(
            (entry(row) for row in samples if row.source_row_id in holdout_problem_ids),
            key=lambda row: row.entry_id,
        )
    )
    window = TtsCalibrationTuningWindow(
        schema_version=5,
        kind="lightcone_tts_disjoint_tuning_window_source",
        tuning_entries=tuning_entries,
        excluded_pilot_entries=holdout_entries,
        selector_namespace=TTS_CALIBRATION_TUNING_SELECTOR_NAMESPACE,
        workload_authority_sha256=workload.sha256,
        ordered_domain_sha256=_sha("lcb-h80-ordered-domain"),
        tuning_problem_ids=tuple(
            sorted(str(row.source_problem_id) for row in tuning_entries)
        ),
        excluded_problem_ids=tuple(
            sorted(str(row.source_problem_id) for row in holdout_entries)
        ),
        trusted_content_bundle_sha256=_sha("trusted-content-bundle"),
        trusted_locked_workload_sha256=descriptor_sha256,
    )
    materialization = _materialize_tts_calibration_diagnostic(
        protocol_lock_sha256=_sha("protocol-lock"),
        upstream_e3a_receipt_sha256=_sha("e3a-selection"),
        calibration_authority_sha256=_sha("tts-authority"),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    candidate = materialization.cells[0].recipe_sha256
    cells = tuple(
        cell for cell in materialization.cells if cell.recipe_sha256 == candidate
    )
    assert {dict(cell.dimensions)["block"] for cell in cells} == {0, 1, 2, 3}
    assert all(cell.method_role == "TTS" for cell in cells)

    schedules = tuple(
        dispatch.rebuild_formal_serving_request_schedule_source(
            subject_sha256=_sha(f"subject:{cell.cell_id}"),
            workload_authority_sha256=workload.sha256,
            topology_mode="tp1_dp1",
            materialization=materialization,
            materialized_cell_id=cell.cell_id,
            workload_source=workload,
            workload_source_descriptor_sha256=descriptor_sha256,
            tts_tuning_window=window,
            sampling_profile=SamplingProfile(),
            max_running_requests=1,
            server_context_limit=40_928,
            tokenizer_content_member_id="tokenizer-member",
            tokenizer_model_id="Qwen/Qwen3-8B",
            tokenizer_revision="4" * 40,
            tokenizer_content_authority_sha256=_sha("tokenizer-authority"),
        )
        for cell in cells
    )
    expected_tuning_samples = {row.source_sample_id for row in tuning_entries}
    holdout_samples = {row.source_sample_id for row in holdout_entries}
    expected_entry_ids = tuple(sorted(row.entry_id for row in tuning_entries))
    assert len(expected_tuning_samples) == 76
    assert len(holdout_samples) == 4
    assert all(len(source.requests) == 76 for source in schedules)
    assert all(
        {row.source_sample_id for row in source.requests} == expected_tuning_samples
        and not ({row.source_sample_id for row in source.requests} & holdout_samples)
        and source.tts_tuning_entry_ids == expected_entry_ids
        for source in schedules
    )
    assert len({source.requests[0].namespace for source in schedules}) == 4


def test_tokenizer_worker_path_and_real_subprocess_no_replace(tmp_path) -> None:
    worker, raw_sha256, size = dispatch._tokenizer_worker_source()
    assert worker == (
        Path(dispatch.__file__).resolve().parent.parent
        / "sglang_bridge"
        / "formal_tokenize_worker.py"
    )
    assert worker.is_file() and len(raw_sha256) == 64 and size == worker.stat().st_size
    fake_root = tmp_path / "fake-packages"
    package = fake_root / "transformers"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        """
__version__ = "0.test"
class _Tokenizer:
    vocab_size = 101
    def __call__(self, prompt, **kwargs):
        return {"input_ids": [len(prompt), len(prompt) + 1]}
class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return _Tokenizer()
""".lstrip(),
        encoding="utf-8",
    )
    snapshot = tmp_path / ("1" * 40)
    snapshot.mkdir()
    source_sha = _sha("subprocess-source")
    prompt = "worker subprocess prompt"
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    publish_canonical_json_no_replace(
        input_path,
        {
            "schema_version": 1,
            "kind": "formal_serving_tokenization_input",
            "protocol_sha256": dispatch.FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
            "schedule_source_sha256": source_sha,
            "tokenizer_model_id": "test/tokenizer",
            "tokenizer_revision": snapshot.name,
            "tokenizer_snapshot_path": str(snapshot),
            "tokenizer_content_authority_sha256": _sha("tokenizer"),
            "requests": [
                {
                    "request_id": "source-request",
                    "ordinal": 0,
                    "prompt": prompt,
                    "prompt_sha256": dispatch._sha256(prompt),
                }
            ],
        },
    )
    source_root = Path(dispatch.__file__).resolve().parents[2]
    environment = {
        "PATH": str(Path(sys.executable).resolve().parent),
        "PYTHONPATH": os.pathsep.join((str(fake_root), str(source_root))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C",
        "LC_ALL": "C",
    }
    command = (
        sys.executable,
        str(worker),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    )
    first = subprocess.run(command, env=environment, capture_output=True, check=False)
    assert first.returncode == 0 and not first.stdout and not first.stderr
    output = CanonicalJsonProofBinding.bind(output_path).reopen()
    assert output["requests"][0]["input_token_ids"] == [len(prompt), len(prompt) + 1]
    assert output["transformers_version"] == "0.test"
    second = subprocess.run(command, env=environment, capture_output=True, check=False)
    assert second.returncode != 0


@pytest.mark.parametrize(
    ("field", "foreign_value"),
    (
        ("tokenizer_model_id", "foreign/model"),
        ("tokenizer_revision", "f" * 40),
        ("tokenizer_snapshot_path", "/private/foreign/snapshot"),
        ("tokenizer_content_authority_sha256", "e" * 64),
    ),
)
def test_tokenizer_worker_metadata_tamper_is_rejected(
    tmp_path, field, foreign_value
) -> None:
    source = _source(subject_sha256=_sha("subject"), topology="tp1_dp1")
    snapshot = (tmp_path / source.tokenizer_revision).resolve()
    snapshot.mkdir()
    launch = SimpleNamespace(
        tokenizer_model_id=source.tokenizer_model_id,
        tokenizer_revision=source.tokenizer_revision,
        tokenizer_snapshot_path=str(snapshot),
        tokenizer_content_authority_sha256=(source.tokenizer_content_authority_sha256),
    )
    input_path = tmp_path / "token-input.json"
    input_binding = dispatch._publish_tokenization_input(
        path=input_path,
        source=source,
        launch=launch,
    )
    output_value = {
        "schema_version": 1,
        "kind": "formal_serving_tokenization_output",
        "protocol_sha256": dispatch.FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
        "schedule_source_sha256": source.sha256,
        "tokenizer_model_id": source.tokenizer_model_id,
        "tokenizer_revision": source.tokenizer_revision,
        "tokenizer_snapshot_path": str(snapshot),
        "tokenizer_content_authority_sha256": source.tokenizer_content_authority_sha256,
        "tokenizer_class": "TestTokenizer",
        "tokenizer_vocab_size": 100,
        "transformers_version": "test",
        "requests": [
            {
                "request_id": row.source_request_key,
                "ordinal": row.ordinal,
                "prompt_sha256": row.prompt_sha256,
                "input_token_ids": [row.ordinal + 1],
                "input_token_ids_sha256": dispatch._sha256([row.ordinal + 1]),
            }
            for row in source.requests
        ],
    }
    output_value[field] = foreign_value
    output_path = tmp_path / "token-output.json"
    publish_canonical_json_no_replace(output_path, output_value)
    with pytest.raises(ValueError, match="schema/coverage"):
        dispatch._materialized_schedule_rows(
            source=source,
            launch=launch,
            tokenization_input=input_binding,
            tokenization_output=CanonicalJsonProofBinding.bind(output_path),
        )


def _install_materialization_fakes(
    monkeypatch,
    tmp_path: Path,
    *,
    topology: str = "tp1_dp1",
    algorithm: str = "DFLASH",
    stage: str = "E3a",
    method: str = "static",
    method_role: str = "Static",
    task: str = "controlled_baseline",
    max_running_requests: int = 2,
    sample_count: int = 4,
    server_argv_override: tuple[str, ...] | None = None,
    localhost_port: int = 32109,
    cell_context_tokens: int = 4096,
    real_run_config: bool = False,
):
    output_root = tmp_path / "private"
    output_root.mkdir(mode=0o700)
    subject_sha = _sha("subject")
    inventory = GpuInventory(
        schema_version=1,
        devices=tuple(
            GpuDevice(
                uuid=f"GPU-test-{index}",
                host_id="test-host",
                model="Test GPU",
                memory_bytes=96_000_000_000,
                compute_capability=(12, 0),
                pci_bus_id=f"0000:{index + 1:02x}:00.0",
                pci_root="root-0",
                numa_node=0,
                interconnects=("PCIe",),
                peer_access_class="peer-enabled",
                clock_policy="locked",
                power_limit_watts=600.0,
                thermal_limit_celsius=85.0,
                availability=GpuAvailability.READY,
                reserved_processes=(),
                allowed_topology_groups=("dual-card",),
            )
            for index in range(2)
        ),
        topology_groups=(
            GpuTopologyGroup(
                group_id="dual-card",
                host_id="test-host",
                gpu_uuids=("GPU-test-0", "GPU-test-1"),
                fabric="PCIe",
                bandwidth_class="test",
            ),
        ),
        source_receipt_sha256=_sha("inventory-source"),
    )
    inventory_path = (tmp_path / "inventory.json").resolve()
    publish_canonical_json_no_replace(inventory_path, inventory.to_dict())
    cell = MaterializedCell(
        stage=stage,
        method_role=method_role,
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task=task,
        publication_policy="none",
        recipe_sha256=None,
        dimensions=tuple(
            sorted(
                (
                    ("concurrency", max_running_requests),
                    ("context", cell_context_tokens),
                    ("regime", "short_input_long_generation"),
                    *((("cohort_count", 4),) if topology == "tp1_dp2" else ()),
                    *((("arrival", "closed_loop"),) if stage == "E5" else ()),
                )
            )
        ),
    )
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage=stage,
        protocol_lock_sha256=_sha("protocol-lock"),
        upstream_receipt_sha256s=(_sha("preflight"),),
        source_decision_sha256=_sha("e3a-source"),
        materialization_rule="test_exact_single_cell",
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
    raw_path = (tmp_path / "workload.json").resolve()
    raw_path.write_text('{"test":"workload"}\n', encoding="utf-8")
    samples = tuple(
        FormalWorkloadSample(
            source_row_id=f"row-{index}",
            sample_id=f"sample-{index}",
            prompt=f"prompt number {index}",
            seed=index + 1,
        )
        for index in range(sample_count)
    )
    protocol = FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"]
    workload = FormalWorkloadAuthority(
        schema_version=1,
        kind="formal_workload_authority",
        workload_id="livecodebench_v6_hard",
        raw_source_path=str(raw_path),
        raw_file_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        repository_revision="a" * 40,
        raw_row_count=sample_count,
        selected_row_count=sample_count,
        selected_rows_sha256=formal_workload_samples_sha256(samples),
        source_lock_sha256=_sha("workload-source-lock"),
        protocol_sha256=protocol.sha256,
        samples=samples,
    )
    workload_path = (tmp_path / "workload-authority.json").resolve()
    publish_canonical_json_no_replace(
        workload_path,
        formal_workload_authority_cli_artifact(workload),
    )
    content_path = (tmp_path / "content-receipt.json").resolve()
    publish_canonical_json_no_replace(content_path, {"kind": "test-content"})
    content_sha = CanonicalJsonProofBinding.bind(content_path).semantic_sha256

    class FakeContentReceipt:
        def __init__(self):
            self.content_artifacts = ()
            self.sha256 = content_sha
            self.verified_ns = 20

        @classmethod
        def from_dict(cls, _value):
            return cls()

        def revalidate_formal_scope(self, *, current_ns: int):
            assert current_ns == 20
            return ()

    monkeypatch.setattr(dispatch, "ContentVerificationReceipt", FakeContentReceipt)

    class FakeWorkloadAuthorization:
        @staticmethod
        def source(_workload_id):
            return SimpleNamespace(sha256=_sha("workload-member"))

    monkeypatch.setattr(
        dispatch,
        "_verified_workload_authorization",
        lambda *_args, **_kwargs: FakeWorkloadAuthorization(),
    )
    monkeypatch.setattr(
        dispatch,
        "revalidate_authorized_formal_workload_authority",
        lambda value, **_kwargs: value,
    )
    sampling_path = (tmp_path / "sampling.json").resolve()
    SamplingProfile().write(sampling_path)
    diagnostic_config = SimpleNamespace(
        method=method,
        adaptation=(
            None
            if method in {"static", "target_only"}
            else SimpleNamespace(
                reset_scope=("request" if method in {"tts", "l0"} else "cohort"),
                request_admission_policy=(
                    "serialized_native_scheduler_v1"
                    if method in {"tts", "l0"}
                    else "cohort_batching_v1"
                ),
            )
        ),
        runtime=SimpleNamespace(
            topology_mode=topology,
            max_running_requests=max_running_requests,
            context_length=40960,
            sampling_profile_sha256=SamplingProfile().sha256,
            tensor_parallel_size=(2 if topology == "tp2_dp1" else 1),
            data_parallel_size=(2 if topology == "tp1_dp2" else 1),
            node_count=1,
            cuda_graph_mode="disabled",
            execution_policy_sha256=ControlledExecutionPolicy().sha256,
            random_seed=1,
            disable_radix_cache=True,
            disable_cuda_graph=True,
            target_reference_disable_overlap_schedule=True,
            speculative_disable_overlap_schedule=False,
            enable_deterministic_inference=False,
            incremental_streaming_output=False,
            native_graph_release_capability_sha256=None,
            cuda_graph_batch_sizes=(1,),
            telemetry_detail="headline",
            adaptation_microbatch_size=1,
            adaptation_publication_coalescing=1,
            adaptation_stream_priority="default",
        ),
        model=SimpleNamespace(
            algorithm=algorithm,
            target="example/target",
            drafter="example/drafter",
            target_revision="2" * 40,
            drafter_revision="3" * 40,
        ),
    )
    if real_run_config:
        from lightcone_spec.runtime.distributed import (
            DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES,
        )

        release = DISTRIBUTED_RUNTIME_RELEASE_CAPABILITIES.get(topology)
        config = RunConfig(
            method="static",
            model=ModelPair(
                target="example/target",
                drafter="example/drafter",
                target_revision="2" * 40,
                drafter_revision="3" * 40,
            ),
            runtime=RuntimeConfig(
                sampling_profile_sha256=SamplingProfile().sha256,
                max_running_requests=max_running_requests,
                tensor_parallel_size=2 if topology == "tp2_dp1" else 1,
                data_parallel_size=2 if topology == "tp1_dp2" else 1,
                router_identity=(
                    "sticky-router-v1" if topology == "tp1_dp2" else "single-replica"
                ),
                process_group_backend=(
                    "nccl" if release is None else release.process_group_backend
                ),
                distributed_runtime_capability=(
                    "single_rank" if release is None else "patched_two_gpu_v1"
                ),
                distributed_release_capability_sha256=(
                    None if release is None else release.sha256
                ),
                distributed_capability_receipt_sha256=(
                    None if release is None else _sha(f"{topology}-runtime-envelope")
                ),
            ),
        )
    else:
        config = diagnostic_config
    config_sha256 = run_config_sha256(config) if real_run_config else _sha("run-config")
    identity = SimpleNamespace(
        run_id="formal-run",
        run_nonce_sha256=_sha("run-nonce"),
        execution_plan_sha256=_sha("execution-plan"),
        rank_config_sha256=_sha("rank-config"),
        attempt_id="attempt-0",
    )
    gpu_proof_path = tmp_path / "runtime-gpu-proof.json"
    publish_canonical_json_no_replace(
        gpu_proof_path,
        {"schema_version": 1, "kind": "test-runtime-gpu-proof"},
    )
    gpu_proof_binding = CanonicalJsonProofBinding.bind(gpu_proof_path)
    subject = SimpleNamespace(
        sha256=subject_sha,
        materialization_receipt_sha256=materialization.sha256,
        materialized_cell_id=cell.cell_id,
        workload_authority_sha256=_sha("workload-authorization"),
        content_verification_receipt_sha256=content_sha,
        topology_mode=topology,
        run_config_sha256=config_sha256,
        inventory_sha256=inventory.sha256,
        gpu_uuids=(
            ("GPU-test-0",) if topology == "tp1_dp1" else ("GPU-test-0", "GPU-test-1")
        ),
        workload_member_sha256s=(_sha("workload-member"),),
        prepared_model_member_sha256s=(_sha("tokenizer-authority"),),
        execution_identity=identity,
        stage=stage,
        method=method,
        runtime_gpu_proof_artifacts=(gpu_proof_binding,),
    )
    verified = SimpleNamespace(
        subject=subject,
        run_config=config,
        sha256=_sha("binding"),
        runtime_gpu_proof_sha256s=(gpu_proof_binding.semantic_sha256,),
        verified_nextn_tp2_authority=None,
    )
    monkeypatch.setattr(
        dispatch,
        "require_verified_formal_serving_execution_binding",
        lambda value: value,
    )
    monkeypatch.setattr(
        dispatch,
        "_revalidate_backend_runtime_proofs",
        lambda **_kwargs: None,
    )
    launch_path = (tmp_path / "launch.json").resolve()
    checkout = (tmp_path / "patched-sglang").resolve()
    target = (tmp_path / "models" / "target" / ("2" * 40)).resolve()
    drafter = (tmp_path / "models" / "drafter" / ("3" * 40)).resolve()
    tokenizer = (tmp_path / ("1" * 40)).resolve()
    cuda_home = (tmp_path / "cuda").resolve()
    library = (tmp_path / "lib").resolve()
    for directory in (checkout, target, drafter, tokenizer, cuda_home, library):
        directory.mkdir(parents=True, exist_ok=True)
    run_config_path = (tmp_path / "run.json").resolve()
    if real_run_config:
        run_config_path.write_text(
            json.dumps(
                config.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        Path(f"{run_config_path}.sha256").write_text(
            f"{config_sha256}\n", encoding="ascii"
        )
    else:
        run_config_path.write_text("{}\n", encoding="utf-8")
    cache_path = (tmp_path / "cache-plan.json").resolve()
    prewarm_path = (tmp_path / "prewarm.json").resolve()
    prepared_path = (tmp_path / "prepared.json").resolve()
    for path in (cache_path, prewarm_path, prepared_path):
        path.write_text("{}\n", encoding="utf-8")
    for path, semantic in (
        (cache_path, _sha("cache-plan")),
        (prewarm_path, _sha("prewarm")),
        (prepared_path, _sha("prepared")),
    ):
        Path(f"{path}.sha256").write_text(f"{semantic}\n", encoding="ascii")
    server_argv = server_argv_override or (
        str(Path(sys.executable).resolve()),
        "-c",
        "import time; time.sleep(60)",
        "--host",
        "127.0.0.1",
        "--port",
        str(localhost_port),
        "--model-path",
        str(target),
        "--speculative-draft-model-path",
        str(drafter),
    )
    launch_artifact = CompileLaunchManifest(
        schema_version=1,
        kind="first_party_compile_launch_manifest",
        protocol_sha256=COMPILE_LAUNCH_MANIFEST_PROTOCOL_SHA256,
        patched_sglang_checkout=str(checkout),
        patched_sglang_commit=PINNED_SGLANG_COMMIT,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        run_config_path=str(run_config_path),
        run_config_raw_sha256=hashlib.sha256(run_config_path.read_bytes()).hexdigest(),
        run_config_semantic_sha256=subject.run_config_sha256,
        compile_cache_plan_path=str(cache_path),
        compile_cache_plan_raw_sha256=hashlib.sha256(
            cache_path.read_bytes()
        ).hexdigest(),
        compile_cache_plan_sha256=_sha("cache-plan"),
        prewarm_manifest_path=str(prewarm_path),
        prewarm_manifest_raw_sha256=hashlib.sha256(
            prewarm_path.read_bytes()
        ).hexdigest(),
        prewarm_manifest_sha256=_sha("prewarm"),
        sampling_profile_path=str(sampling_path),
        sampling_profile_raw_sha256=hashlib.sha256(
            sampling_path.read_bytes()
        ).hexdigest(),
        prepared_model_content_manifest_path=str(prepared_path),
        prepared_model_content_manifest_raw_sha256=hashlib.sha256(
            prepared_path.read_bytes()
        ).hexdigest(),
        prepared_model_content_manifest_sha256=_sha("prepared"),
        prepared_model_content_manifest_size=prepared_path.stat().st_size,
        target_content_member_id="target-member",
        target_model_id="example/target",
        target_snapshot_path=str(target),
        target_revision="2" * 40,
        target_content_authority_sha256=_sha("target-authority"),
        drafter_content_member_id="drafter-member",
        drafter_model_id="example/drafter",
        drafter_snapshot_path=str(drafter),
        drafter_revision="3" * 40,
        drafter_content_authority_sha256=_sha("drafter-authority"),
        tokenizer_content_member_id="tokenizer-member",
        tokenizer_model_id="example/tokenizer",
        tokenizer_snapshot_path=str(tokenizer),
        tokenizer_revision="1" * 40,
        tokenizer_content_authority_sha256=_sha("tokenizer-authority"),
        server_argv=server_argv,
        server_argv_sha256=dispatch._sha256({"argv": list(server_argv)}),
        localhost_port=localhost_port,
        model_lock_sha256=_sha("model-lock"),
        sampling_profile_sha256=SamplingProfile().sha256,
        physical_assignment_sha256=_sha("physical-assignment"),
        experiment_budget_sha256=_sha("experiment-budget"),
        budget_materialization_authority_sha256=_sha("budget-authority"),
        inventory_sha256=subject.inventory_sha256,
        gpu_uuids=subject.gpu_uuids,
        path_entries=(str(Path(sys.executable).resolve().parent),),
        library_path_entries=(str(library),),
        cuda_home=str(cuda_home),
    )
    launch_artifact.write(launch_path)
    launch_binding = CanonicalJsonProofBinding.bind(launch_path)
    launch = SimpleNamespace(
        schema_version=1,
        tokenizer_content_member_id="tokenizer-member",
        tokenizer_model_id="example/tokenizer",
        tokenizer_revision="1" * 40,
        tokenizer_content_authority_sha256=_sha("tokenizer-authority"),
        tokenizer_snapshot_path=str(tokenizer),
        run_config_path=str(tmp_path / "run.json"),
        sampling_profile_path=str(sampling_path),
        sampling_profile_sha256=SamplingProfile().sha256,
        inventory_sha256=subject.inventory_sha256,
        gpu_uuids=subject.gpu_uuids,
        sha256=launch_binding.semantic_sha256,
        target_model_id="example/target",
        physical_assignment_sha256=_sha("physical-assignment"),
        experiment_budget_sha256=_sha("experiment-budget"),
        localhost_port=localhost_port,
        server_argv=server_argv,
        server_argv_sha256=dispatch._sha256({"argv": list(server_argv)}),
        patched_sglang_checkout=str(checkout),
        patched_sglang_commit=PINNED_SGLANG_COMMIT,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        child_environment=lambda: {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": ",".join(subject.gpu_uuids),
        },
    )

    class FakeCompileLaunchManifest:
        @staticmethod
        def load(path):
            assert Path(path) == launch_path
            return launch

    monkeypatch.setattr(dispatch, "CompileLaunchManifest", FakeCompileLaunchManifest)
    monkeypatch.setattr(dispatch, "load_run_config", lambda _path: config)
    monkeypatch.setattr(
        dispatch, "run_config_sha256", lambda _config: subject.run_config_sha256
    )

    def fake_tokenizer(*, input_path: Path, output_path: Path):
        source_input = CanonicalJsonProofBinding.bind(input_path).reopen()
        token_map = {
            row["request_id"]: (row["ordinal"] * 10 + 1, row["ordinal"] * 10 + 2)
            for row in source_input["requests"]
        }
        output = {
            "schema_version": 1,
            "kind": "formal_serving_tokenization_output",
            "protocol_sha256": dispatch.FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
            "schedule_source_sha256": source_input["schedule_source_sha256"],
            "tokenizer_model_id": source.tokenizer_model_id,
            "tokenizer_revision": source.tokenizer_revision,
            "tokenizer_snapshot_path": launch.tokenizer_snapshot_path,
            "tokenizer_content_authority_sha256": source.tokenizer_content_authority_sha256,
            "tokenizer_class": "TestTokenizer",
            "tokenizer_vocab_size": 32,
            "transformers_version": "test-only",
            "requests": [
                {
                    "request_id": row["request_id"],
                    "ordinal": row["ordinal"],
                    "prompt_sha256": row["prompt_sha256"],
                    "input_token_ids": list(token_map[row["request_id"]]),
                    "input_token_ids_sha256": dispatch._sha256(
                        list(token_map[row["request_id"]])
                    ),
                }
                for row in source_input["requests"]
            ],
        }
        publish_canonical_json_no_replace(output_path, output)
        return (
            CanonicalJsonProofBinding.bind(output_path),
            dispatch._tokenizer_worker_source()[1],
            dispatch._tokenizer_worker_source()[2],
            _sha("worker-argv"),
        )

    monkeypatch.setattr(dispatch, "_invoke_tokenizer_worker", fake_tokenizer)
    source = dispatch.rebuild_formal_serving_request_schedule_source(
        subject_sha256=subject.sha256,
        workload_authority_sha256=subject.workload_authority_sha256,
        topology_mode=topology,
        materialization=materialization,
        materialized_cell_id=cell.cell_id,
        workload_source=workload,
        workload_source_descriptor_sha256=_sha("workload-member"),
        tts_tuning_window=None,
        sampling_profile=SamplingProfile(),
        max_running_requests=max_running_requests,
        server_context_limit=40960,
        tokenizer_content_member_id=launch.tokenizer_content_member_id,
        tokenizer_model_id=launch.tokenizer_model_id,
        tokenizer_revision=launch.tokenizer_revision,
        tokenizer_content_authority_sha256=(launch.tokenizer_content_authority_sha256),
    )
    return (
        output_root,
        content_path,
        workload_path,
        materialization_path,
        verified,
        launch_path,
        source,
        inventory_path,
    )


@pytest.mark.parametrize("selected_schema", (5, 6))
def test_profiler_subject_schedule_preserves_trusted_lineage_and_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    selected_schema: int,
) -> None:
    from lightcone_spec.experiments.formal_content_source import (
        FormalContentSourceBinding,
    )
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundleBinding,
    )
    from lightcone_spec.experiments.serving import BoundServingRequest
    from lightcone_spec.runtime.content_authorization import (
        ContentJsonArtifactBinding,
    )

    (
        _output_root,
        _content_path,
        workload_path,
        materialization_path,
        _verified,
        launch_path,
        offline_source,
        _inventory_path,
    ) = _install_materialization_fakes(
        monkeypatch,
        tmp_path,
        stage="E4",
        max_running_requests=1,
        sample_count=2,
    )
    launch = dispatch.CompileLaunchManifest.load(launch_path)
    trusted_launch_values = vars(launch).copy()
    trusted_launch_values["tokenizer_content_authority_sha256"] = None
    trusted_launch = SimpleNamespace(**trusted_launch_values)

    bundle_path = (tmp_path / "trusted-bundle.json").resolve()
    publish_canonical_json_no_replace(bundle_path, {"fixture": "trusted-bundle"})
    bundle_semantic = _sha("trusted-bundle")
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: SimpleNamespace(
            runtime_binding_status="BOUND",
            semantic_sha256=bundle_semantic,
        ),
    )
    bundle_binding = TrustedSingleOperatorContentBundleBinding(
        absolute_path=str(bundle_path),
        size=bundle_path.stat().st_size,
        raw_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        semantic_sha256=bundle_semantic,
        runtime_binding_status="BOUND",
    )
    content_source = FormalContentSourceBinding(
        schema_version=1,
        kind="formal_content_source_binding",
        mode="trusted_single_operator",
        offline_root_signed=None,
        trusted_single_operator=bundle_binding,
    )
    source = replace(
        offline_source,
        schema_version=5,
        protocol_sha256=(
            dispatch.TRUSTED_SINGLE_OPERATOR_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        ),
        derivation_protocol_sha256=(
            dispatch.TRUSTED_SINGLE_OPERATOR_REQUEST_SCHEDULE_DERIVATION_PROTOCOL_SHA256
        ),
        workload_authority_sha256=None,
        workload_source_authority_sha256=None,
        tokenizer_content_authority_sha256=None,
        content_source_binding_sha256=content_source.sha256,
        trusted_workload_member_sha256=_sha("workload-member"),
    )
    selected_root = (tmp_path / "selected-schedule").resolve()
    selected_root.mkdir(mode=0o700)
    if selected_schema == 6:
        source_row_root = selected_root / "source-rows"
        source_row_root.mkdir(mode=0o700)
        source = dispatch.publish_trusted_schedule_source_shards(
            source=source,
            output_directory=source_row_root,
        )
    source_path = selected_root / "source.json"
    publish_canonical_json_no_replace(source_path, source.to_dict())
    schedule_binding = ContentJsonArtifactBinding.from_path(
        "formal_request_schedule:selected-headline",
        source_path,
    )

    rows = []
    for source_row in dispatch.formal_serving_request_schedule_source_rows(source):
        input_token_ids = (source_row.ordinal * 10 + 1, source_row.ordinal * 10 + 2)
        immutable = ImmutableRequest.create(
            namespace=source_row.namespace,
            split=source_row.split,
            ordinal=source_row.ordinal,
            template=RequestTemplate(
                input_token_ids=input_token_ids,
                requested_output_tokens=source_row.requested_output_tokens,
                sampling=FrozenSamplingParameters(items=source_row.sampling),
                cancellation_offset_us=source_row.cancellation_offset_us,
            ),
            arrival_us=source_row.arrival_us,
            cohort_id=source_row.cohort_id,
        )
        rows.append(
            dispatch.FormalServingRequestScheduleRow(
                source_member_sha256=source_row.source_member_sha256,
                source_sample_id=source_row.source_sample_id,
                prompt_sha256=source_row.prompt_sha256,
                phase=source_row.phase,
                routed_dp_rank=source_row.routed_dp_rank,
                request=BoundServingRequest.create(
                    immutable,
                    route_id=source.topology_mode,
                ),
                tokenized_input_sha256=dispatch._sha256(list(input_token_ids)),
            )
        )
    fixture_path = (selected_root / "fixture.json").resolve()
    publish_canonical_json_no_replace(fixture_path, {"fixture": True})
    fixture_binding = CanonicalJsonProofBinding.bind(fixture_path)
    launch_binding = CanonicalJsonProofBinding.bind(launch_path)
    sampling_binding = CanonicalJsonProofBinding.bind(
        trusted_launch.sampling_profile_path
    )
    materialization_binding = CanonicalJsonProofBinding.bind(materialization_path)
    selected = dispatch.FormalServingRequestScheduleReceipt(
        schema_version=5,
        kind="formal_serving_request_schedule_receipt",
        protocol_sha256=(
            dispatch.TRUSTED_SINGLE_OPERATOR_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256
        ),
        formal_execution_authorized=False,
        execution_binding_sha256=_sha("selected-execution"),
        subject_sha256=source.subject_sha256,
        materialized_cell_id=source.materialized_cell_id,
        workload_authority_sha256=None,
        content_verification_receipt_sha256=None,
        topology_mode=source.topology_mode,
        materialization=materialization_binding,
        content_verification_receipt=None,
        workload_source=ContentJsonArtifactBinding.from_path(
            "formal_workload:livecodebench_v6_hard",
            workload_path,
        ),
        compile_launch_manifest=launch_binding,
        sampling_profile=sampling_binding,
        schedule_source=schedule_binding,
        tokenization_input=fixture_binding,
        tokenization_output=fixture_binding,
        tokenizer_worker_source_raw_sha256=_sha("selected-worker"),
        tokenizer_worker_source_size=1,
        tokenizer_worker_argv_sha256=_sha("selected-worker-argv"),
        tokenizer_model_id=trusted_launch.tokenizer_model_id,
        tokenizer_revision=trusted_launch.tokenizer_revision,
        tokenizer_snapshot_path=trusted_launch.tokenizer_snapshot_path,
        tokenizer_content_member_id=trusted_launch.tokenizer_content_member_id,
        tokenizer_content_authority_sha256=None,
        transformers_version="test-only",
        tokenizer_class="TestTokenizer",
        tokenizer_vocab_size=32,
        requests=tuple(rows),
        content_source_binding=content_source,
        trusted_workload_member_sha256=source.trusted_workload_member_sha256,
    )
    if selected_schema == 6:
        receipt_row_root = selected_root / "receipt-rows"
        receipt_row_root.mkdir(mode=0o700)
        selected = dispatch.publish_trusted_schedule_receipt_shards(
            receipt=selected,
            output_directory=receipt_row_root,
        )
    selected_path = selected_root / "receipt.json"
    publish_canonical_json_no_replace(selected_path, selected.to_dict())

    def fake_tokenizer(*, input_path: Path, output_path: Path):
        token_input = CanonicalJsonProofBinding.bind(input_path).reopen()
        token_output = {
            "schema_version": 1,
            "kind": "formal_serving_tokenization_output",
            "protocol_sha256": dispatch.FORMAL_SERVING_PHYSICAL_DISPATCH_PROTOCOL_SHA256,
            "schedule_source_sha256": token_input["schedule_source_sha256"],
            "tokenizer_model_id": token_input["tokenizer_model_id"],
            "tokenizer_revision": token_input["tokenizer_revision"],
            "tokenizer_snapshot_path": token_input["tokenizer_snapshot_path"],
            "tokenizer_content_authority_sha256": token_input[
                "tokenizer_content_authority_sha256"
            ],
            "tokenizer_class": "TestTokenizer",
            "tokenizer_vocab_size": 32,
            "transformers_version": "test-only",
            "requests": [
                {
                    "request_id": row["request_id"],
                    "ordinal": row["ordinal"],
                    "prompt_sha256": row["prompt_sha256"],
                    "input_token_ids": [
                        row["ordinal"] * 10 + 1,
                        row["ordinal"] * 10 + 2,
                    ],
                    "input_token_ids_sha256": dispatch._sha256(
                        [row["ordinal"] * 10 + 1, row["ordinal"] * 10 + 2]
                    ),
                }
                for row in token_input["requests"]
            ],
        }
        publish_canonical_json_no_replace(output_path, token_output)
        worker_path, worker_sha, worker_size = dispatch._tokenizer_worker_source()
        return (
            CanonicalJsonProofBinding.bind(output_path),
            worker_sha,
            worker_size,
            dispatch._sha256({"worker": str(worker_path)}),
        )

    monkeypatch.setattr(dispatch, "_invoke_tokenizer_worker", fake_tokenizer)
    monkeypatch.setattr(
        dispatch.FormalServingRequestScheduleReceipt,
        "reopen",
        lambda _self: None,
    )
    profile_root = (tmp_path / "profile-schedule").resolve()
    profile_root.mkdir(mode=0o700)
    input_path = (tmp_path / "profiler-input.json").resolve()
    publish_canonical_json_no_replace(input_path, {"input": "profiler"})
    profile_subject_sha256 = _sha("profile-subject")
    result = dispatch._materialize_formal_single_operator_profiler_subject_schedule(
        inputs=SimpleNamespace(
            selected_request_schedule=CanonicalJsonProofBinding.bind(selected_path),
            subject_sha256=profile_subject_sha256,
            private_output_root=str(profile_root),
            source_headline_cell_id=source.materialized_cell_id,
            profile_compile_launch_manifest=launch_binding,
        ),
        input_binding=CanonicalJsonProofBinding.bind(input_path),
        launch=trusted_launch,
    )

    assert result.schema_version == selected_schema
    assert result.content_source_binding == content_source
    assert result.trusted_workload_member_sha256 == _sha("workload-member")
    assert result.workload_authority_sha256 is None
    assert result.content_verification_receipt_sha256 is None
    assert result.tokenizer_content_authority_sha256 is None
    assert tuple(dispatch.formal_serving_request_schedule_rows(result)) == tuple(rows)
    result_source = dispatch.FormalServingRequestScheduleSource.from_dict(
        result.schedule_source.load()
    )
    assert result_source.schema_version == selected_schema
    assert result_source.subject_sha256 == profile_subject_sha256
    assert result_source.content_source_binding_sha256 == content_source.sha256


def test_materialized_plan_replays_and_reaches_tp1_live_runner_without_requests(
    monkeypatch, tmp_path
) -> None:
    (
        output_root,
        content_path,
        workload_path,
        materialization_path,
        verified,
        launch_path,
        source,
        inventory_path,
    ) = _install_materialization_fakes(monkeypatch, tmp_path)
    plan = dispatch.materialize_formal_serving_run_plan(
        execution_binding=verified,
        content_verification_receipt_path=content_path,
        workload_authority_path=workload_path,
        materialization_path=materialization_path,
        compile_launch_manifest_path=launch_path,
        private_output_root=output_root,
        now_ns=20,
    )
    plan_path = output_root / "formal-serving-run-plan.json"
    admission = publish_formal_single_operator_admission(
        plan_path=plan_path,
        inventory_path=inventory_path,
    )
    assert plan.schema_version == 1
    assert plan.single_operator_execution_rebuild_source is None
    assert "single_operator_execution_rebuild_source" not in plan.to_dict()
    assert dispatch.FormalServingRunPlan.from_dict(plan.to_dict()) == plan
    current_source_path = (tmp_path / "current-execution-source.json").resolve()
    runtime_authority_path = (tmp_path / "runtime-authority.json").resolve()
    publish_canonical_json_no_replace(
        current_source_path,
        {"schema_version": 1, "kind": "test-current-execution-source"},
    )
    publish_canonical_json_no_replace(
        runtime_authority_path,
        {"schema_version": 1, "kind": "test-runtime-authority"},
    )
    rebuild_source = dispatch.FormalSingleOperatorExecutionRebuildSource(
        schema_version=1,
        kind="formal_single_operator_execution_rebuild_source",
        protocol_sha256=(
            dispatch.FORMAL_SINGLE_OPERATOR_EXECUTION_REBUILD_SOURCE_PROTOCOL_SHA256
        ),
        execution_binding_sha256=plan.execution_binding_sha256,
        subject_sha256=plan.subject_sha256,
        materialized_cell_id=plan.materialized_cell_id,
        execution_source=CanonicalJsonProofBinding.bind(current_source_path),
        execution_source_sha256=_sha("current-execution-source"),
        formal_runtime_authority_manifest=CanonicalJsonProofBinding.bind(
            runtime_authority_path
        ),
        compile_launch_manifest=plan.launch_manifest,
        inventory=CanonicalJsonProofBinding.bind(inventory_path),
        content_verification_receipt=CanonicalJsonProofBinding.bind(content_path),
        runtime_gpu_proof_artifacts=plan.runtime_gpu_proof_artifacts,
        tts_calibration_authority=None,
        e1_recipe_anchor_authority=None,
        formal_registry_verification_receipt=None,
        repository_root=None,
    )
    rebuild_source_path = (
        output_root / "formal-single-operator-execution-rebuild-source.json"
    )
    publish_canonical_json_no_replace(
        rebuild_source_path,
        rebuild_source.to_dict(),
    )
    rebuild_source_binding = CanonicalJsonProofBinding.bind(rebuild_source_path)
    assert (
        dispatch.revalidate_formal_single_operator_execution_rebuild_source(
            rebuild_source_path
        )
        == rebuild_source
    )
    schema2 = replace(
        plan,
        schema_version=2,
        single_operator_execution_rebuild_source=rebuild_source_binding,
    )
    assert dispatch.FormalServingRunPlan.from_dict(schema2.to_dict()) == schema2
    missing_source = schema2.to_dict()
    del missing_source["single_operator_execution_rebuild_source"]
    with pytest.raises(ValueError, match="fields differ"):
        dispatch.FormalServingRunPlan.from_dict(missing_source)
    legacy_with_source = plan.to_dict()
    legacy_with_source["single_operator_execution_rebuild_source"] = (
        rebuild_source_binding.to_dict()
    )
    with pytest.raises(ValueError, match="fields differ"):
        dispatch.FormalServingRunPlan.from_dict(legacy_with_source)
    schedule = dispatch._reopen_schedule_receipt(plan.request_schedule_receipt)
    assert tuple(row.request.input_token_ids for row in schedule.requests) == tuple(
        (row.ordinal * 10 + 1, row.ordinal * 10 + 2) for row in source.requests
    )
    assert (
        "warmup_requests"
        not in inspect.signature(
            dispatch.execute_formal_tp1_serving_run_plan
        ).parameters
    )
    observed = {}

    async def fake_live_runner(**kwargs):
        observed.update(kwargs)
        return "live-result"

    monkeypatch.setattr(
        dispatch, "execute_unsigned_native_serving_run", fake_live_runner
    )
    result = asyncio.run(
        dispatch.execute_formal_serving_run_plan(
            plan_path=plan_path,
            launch_admission_path=admission.absolute_path,
            execution_binding=verified,
            nvidia_smi_tool=object(),
        )
    )
    assert result == "live-result"
    assert (output_root / "formal-single-operator-admission-consumed.json").is_file()
    assert tuple(row.request_id for row in observed["warmup_requests"]) == tuple(
        row.request.request_id for row in schedule.requests if row.phase == "warmup"
    )
    assert tuple(row.request_id for row in observed["scored_requests"]) == tuple(
        row.request.request_id for row in schedule.requests if row.phase == "scored"
    )
    assert observed["launch_manifest_path"] == str(launch_path)

    trusted_root = (tmp_path / "trusted-direct-run").resolve()
    trusted_root.mkdir(mode=0o700)
    trusted_rebuild_path = (
        trusted_root / "formal-single-operator-execution-rebuild-source.json"
    )
    publish_canonical_json_no_replace(
        trusted_rebuild_path,
        rebuild_source.to_dict(),
    )
    trusted_plan = replace(
        plan,
        schema_version=2,
        single_operator_execution_rebuild_source=(
            CanonicalJsonProofBinding.bind(trusted_rebuild_path)
        ),
        private_output_root=str(trusted_root),
        terminal_output_path=str(trusted_root / "unsigned-native-terminal.json"),
        native_itl_pointer_output_path=str(trusted_root / "unsigned-native-itl.json"),
        live_run_receipt_output_path=str(trusted_root / "unsigned-live-run.json"),
        lifecycle_timing_output_path=str(trusted_root / "unsigned-lifecycle.json"),
        server_log_output_path=str(trusted_root / "server.log"),
        server_stdout_output_path=str(trusted_root / "stdout.log"),
        server_stderr_output_path=str(trusted_root / "stderr.log"),
        junit_output_path=str(trusted_root / "junit.xml"),
        before_gpu_snapshot_output_path=str(trusted_root / "before-gpu.json"),
        ready_gpu_snapshot_output_path=str(trusted_root / "ready-gpu.json"),
        after_gpu_snapshot_output_path=str(trusted_root / "after-gpu.json"),
        fatal_output_path=str(trusted_root / "fatal.json"),
    )
    trusted_plan_path = trusted_root / "formal-serving-run-plan.json"
    publish_canonical_json_no_replace(trusted_plan_path, trusted_plan.to_dict())
    monkeypatch.setattr(
        dispatch,
        "rebuild_formal_single_operator_execution_binding_from_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("trusted execution must not rebuild the legacy token")
        ),
    )
    observed.clear()
    result = asyncio.run(
        dispatch.execute_formal_single_operator_serving_run_plan(
            plan_path=trusted_plan_path,
            nvidia_smi_tool=object(),
        )
    )
    assert result == "live-result"
    assert observed["formal_launch_admission"] is None
    assert observed["formal_launch_consumption"] is None
    assert observed["budget_consumption"] is None
    assert observed["timeout_seconds"] == 3_600.0


def test_plan_materialization_rejects_caller_owned_schedule_values() -> None:
    parameters = inspect.signature(
        dispatch.materialize_formal_serving_run_plan
    ).parameters
    assert set(parameters) == {
        "execution_binding",
        "content_verification_receipt_path",
        "workload_authority_path",
        "materialization_path",
        "compile_launch_manifest_path",
        "private_output_root",
        "now_ns",
        "verified_nextn_tp2_authority",
    }
    forbidden = {"prompt", "input_token_ids", "requests", "port", "argv", "transport"}
    assert forbidden.isdisjoint(parameters)


def test_single_operator_early_inputs_materialize_and_execute_without_old_token(
    monkeypatch, tmp_path
) -> None:
    from lightcone_spec.experiments import formal_single_operator_early_execution
    from lightcone_spec.experiments.formal_preflight_inputs import (
        FORMAL_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256,
        FormalPreflightExecutionInputs,
    )
    from lightcone_spec.experiments.formal_registry import (
        stage_materialization_receipt_from_dict,
    )
    from lightcone_spec.experiments.formal_single_operator_early_execution import (
        FormalSingleOperatorEarlyRunPlanInputs,
    )
    from lightcone_spec.experiments.workload_authority import (
        formal_workload_authority_artifact_id,
    )
    from lightcone_spec.runtime.content_authorization import (
        ContentJsonArtifactBinding,
    )

    (
        output_root,
        content_path,
        workload_path,
        materialization_path,
        verified,
        launch_path,
        _source,
        inventory_path,
    ) = _install_materialization_fakes(monkeypatch, tmp_path)
    execution_source_path = (tmp_path / "current-execution-source.json").resolve()
    publish_canonical_json_no_replace(
        execution_source_path,
        {"schema_version": 1, "kind": "test-current-execution-source"},
    )
    execution_source = CanonicalJsonProofBinding.bind(execution_source_path)
    execution_source_value = execution_source.reopen()
    materialization_binding = CanonicalJsonProofBinding.bind(materialization_path)
    materialization_sha256 = stage_materialization_receipt_from_dict(
        materialization_binding.reopen()
    ).sha256
    monkeypatch.setattr(
        formal_single_operator_early_execution.FormalSingleOperatorExecutionSource,
        "from_dict",
        classmethod(
            lambda _cls, _value: SimpleNamespace(
                sha256=execution_source.semantic_sha256,
                materialization_sha256=materialization_sha256,
                to_dict=lambda: execution_source_value,
            )
        ),
    )

    def auxiliary_binding(name: str, ordinal: int = 0):
        path = (tmp_path / f"{name}-{ordinal}.json").resolve()
        publish_canonical_json_no_replace(
            path,
            {"kind": f"test-{name}", "ordinal": ordinal},
        )
        return CanonicalJsonProofBinding.bind(path)

    preflight_inputs = FormalPreflightExecutionInputs(
        schema_version=2,
        kind="formal_single_operator_exact_ten_preflight_inputs",
        protocol_sha256=FORMAL_SINGLE_OPERATOR_PREFLIGHT_INPUTS_PROTOCOL_SHA256,
        authority_mode="formal_single_operator_v1",
        execution_authority=auxiliary_binding("execution-authority"),
        inventory=CanonicalJsonProofBinding.bind(inventory_path),
        content_receipt=CanonicalJsonProofBinding.bind(content_path),
        workload_authority=ContentJsonArtifactBinding.from_path(
            formal_workload_authority_artifact_id("livecodebench_v6_hard"),
            workload_path,
        ),
        doctor_report=auxiliary_binding("doctor"),
        compile_assignment_plan=auxiliary_binding("compile-assignment"),
        exactness_assignment=auxiliary_binding("exactness-assignment"),
        interference_manifest=auxiliary_binding("interference-manifest"),
        request_schedule_sources=tuple(
            auxiliary_binding("request-schedule", index) for index in range(8)
        ),
        tokenization_inputs=tuple(
            auxiliary_binding("tokenization-input", index) for index in range(8)
        ),
        tokenization_outputs=tuple(
            auxiliary_binding("tokenization-output", index) for index in range(8)
        ),
    )
    preflight_inputs_path = (tmp_path / "preflight-inputs.json").resolve()
    publish_canonical_json_no_replace(
        preflight_inputs_path,
        preflight_inputs.to_dict(),
    )
    inputs = FormalSingleOperatorEarlyRunPlanInputs(
        schema_version=1,
        kind="formal_single_operator_early_run_plan_inputs",
        execution_source=execution_source,
        execution_source_sha256=execution_source.semantic_sha256,
        materialized_cell_id=verified.subject.materialized_cell_id,
        stage="E3a",
        materialization=materialization_binding,
        materialization_sha256=materialization_sha256,
        preflight_inputs=CanonicalJsonProofBinding.bind(preflight_inputs_path),
        compile_launch_manifest=CanonicalJsonProofBinding.bind(launch_path),
        private_output_root=str(output_root),
    )
    inputs_path = output_root / "formal-single-operator-early-run-plan-inputs.json"
    publish_canonical_json_no_replace(inputs_path, inputs.to_dict())
    plan = dispatch.materialize_formal_single_operator_serving_run_plan(
        early_run_plan_inputs_path=inputs_path,
    )
    assert plan.schema_version == 4
    assert plan.serving_execution_policy is not None
    assert plan.process_hard_timeout_ns is not None
    assert plan.execution_binding_sha256 == inputs.sha256
    assert plan.runtime_gpu_proof_sha256s == ()
    assert plan.runtime_gpu_proof_artifacts == ()
    observed: dict[str, object] = {}

    async def fake_live_runner(**kwargs):
        observed.update(kwargs)
        return "direct-live-result"

    monkeypatch.setattr(
        dispatch,
        "execute_unsigned_native_serving_run",
        fake_live_runner,
    )
    monkeypatch.setattr(
        dispatch,
        "rebuild_formal_single_operator_execution_binding_from_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("trusted direct execution must not rebuild old authority")
        ),
    )
    result = asyncio.run(
        dispatch.execute_formal_single_operator_serving_run_plan(
            plan_path=output_root / "formal-serving-run-plan.json",
            nvidia_smi_tool=object(),
        )
    )
    assert result == "direct-live-result"
    assert observed["formal_launch_admission"] is None
    assert observed["timeout_seconds"] == (plan.process_hard_timeout_ns / 1_000_000_000)
    assert observed["execution_policy"] == plan.serving_execution_policy


@pytest.mark.parametrize(
    ("stage", "regime", "workload_id"),
    (
        ("E3a", "short_input_long_generation", "livecodebench_v6_hard"),
        ("E3a", "long_input_short_output", "math500_level5"),
        ("TTS-Cal", "short_input_long_generation", "livecodebench_v6_hard"),
    ),
)
def test_trusted_direct_workload_source_is_rebuilt_from_bound_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
    regime: str,
    workload_id: str,
) -> None:
    from lightcone_spec.experiments.formal_content_source import (
        FormalContentSourceBinding,
    )
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundle,
        TrustedSingleOperatorContentBundleBinding,
    )
    from lightcone_spec.experiments.workload_authority import (
        formal_workload_authority_artifact_id,
        formal_workload_authority_from_cli_artifact,
    )
    from lightcone_spec.runtime.content_authorization import (
        ContentJsonArtifactBinding,
    )

    raw_path = (tmp_path / f"{workload_id}-raw.json").resolve()
    raw_path.write_text('{"fixture":true}\n', encoding="utf-8")
    samples = (
        FormalWorkloadSample(
            source_row_id="source-0",
            sample_id="sample-0",
            prompt="A source-owned prompt.",
            seed=17,
        ),
    )
    authority = FormalWorkloadAuthority(
        schema_version=1,
        kind="formal_workload_authority",
        workload_id=workload_id,
        raw_source_path=str(raw_path),
        raw_file_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        repository_revision="a" * 40,
        raw_row_count=1,
        selected_row_count=1,
        selected_rows_sha256=formal_workload_samples_sha256(samples),
        source_lock_sha256=_sha(f"{workload_id}-source-lock"),
        protocol_sha256=FORMAL_WORKLOAD_PROTOCOLS[workload_id].sha256,
        samples=samples,
    )
    locked = SimpleNamespace(
        workload_id=workload_id,
        authority_sha256=authority.sha256,
        raw_source_path=authority.raw_source_path,
        raw_file_sha256=authority.raw_file_sha256,
        repository_revision=authority.repository_revision,
        raw_row_count=authority.raw_row_count,
        selected_row_count=authority.selected_row_count,
        formal_samples_sha256=authority.selected_rows_sha256,
        source_lock_sha256=authority.source_lock_sha256,
        protocol_sha256=authority.protocol_sha256,
    )
    bundle = object.__new__(TrustedSingleOperatorContentBundle)
    object.__setattr__(bundle, "runtime_binding_status", "BOUND")
    object.__setattr__(bundle, "semantic_sha256", _sha("trusted-workload-bundle"))
    object.__setattr__(bundle, "locked_workloads", (locked,))
    object.__setattr__(bundle, "e0_task_native_descriptors", ())
    bundle_path = (tmp_path / "trusted-workload-bundle.json").resolve()
    bundle_path.write_text('{"fixture":true}\n', encoding="utf-8")
    bundle_binding = TrustedSingleOperatorContentBundleBinding(
        absolute_path=str(bundle_path),
        size=bundle_path.stat().st_size,
        raw_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        semantic_sha256=bundle.semantic_sha256,
        runtime_binding_status="BOUND",
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: bundle,
    )
    content_source = FormalContentSourceBinding(
        schema_version=1,
        kind="formal_content_source_binding",
        mode="trusted_single_operator",
        offline_root_signed=None,
        trusted_single_operator=bundle_binding,
    )
    rebound: list[tuple[str, str]] = []

    def bind_locked(selected_workload_id, selected_raw_path):
        rebound.append((selected_workload_id, str(selected_raw_path)))
        return authority

    monkeypatch.setattr(dispatch, "bind_formal_workload_authority", bind_locked)
    cell = MaterializedCell(
        stage=stage,
        method_role="TTS" if stage == "TTS-Cal" else "Static",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="content_owned_workload_fixture",
        publication_policy="fixed_barrier" if stage == "TTS-Cal" else "tuning_only",
        recipe_sha256=_sha("tts-recipe") if stage == "TTS-Cal" else None,
        dimensions=(("regime", regime),),
    )

    output = dispatch._materialize_trusted_single_operator_workload_source(
        content_source_binding=content_source,
        cell=cell,
        private_output_root=tmp_path.resolve(),
    )
    assert output == (tmp_path / "trusted-workload-source.json").resolve()
    binding = ContentJsonArtifactBinding.from_path(
        formal_workload_authority_artifact_id(workload_id),
        output,
    )
    assert formal_workload_authority_from_cli_artifact(binding.load()) == authority
    assert rebound == [(workload_id, authority.raw_source_path)]

    with pytest.raises(RuntimeError, match="target already exists"):
        dispatch._materialize_trusted_single_operator_workload_source(
            content_source_binding=content_source,
            cell=cell,
            private_output_root=tmp_path.resolve(),
        )

    bad_root = (tmp_path / "tampered").resolve()
    bad_root.mkdir(mode=0o700)
    object.__setattr__(
        bundle,
        "locked_workloads",
        (SimpleNamespace(**{**vars(locked), "authority_sha256": _sha("foreign")}),),
    )
    with pytest.raises(ValueError, match="differs from locked workload"):
        dispatch._materialize_trusted_single_operator_workload_source(
            content_source_binding=content_source,
            cell=cell,
            private_output_root=bad_root,
        )
    assert not (bad_root / "trusted-workload-source.json").exists()


def test_trusted_direct_e0_workload_uses_exact_content_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lightcone_spec.experiments.formal_content_source import (
        FormalContentSourceBinding,
    )
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundle,
        TrustedSingleOperatorContentBundleBinding,
    )
    from lightcone_spec.experiments.formal_single_operator_e0_workloads import (
        E0TaskNativeSourceAuthority,
    )

    descriptor_path = (tmp_path / "gsm8k-source.json").resolve()
    descriptor_path.write_text('{"fixture":true}\n', encoding="utf-8")
    descriptor = SimpleNamespace(
        task="GSM8K",
        source=SimpleNamespace(absolute_path=str(descriptor_path)),
    )
    bundle = object.__new__(TrustedSingleOperatorContentBundle)
    object.__setattr__(bundle, "runtime_binding_status", "BOUND")
    object.__setattr__(bundle, "semantic_sha256", _sha("trusted-e0-bundle"))
    object.__setattr__(bundle, "locked_workloads", ())
    object.__setattr__(bundle, "e0_task_native_descriptors", (descriptor,))
    bundle_path = (tmp_path / "trusted-e0-bundle.json").resolve()
    bundle_path.write_text('{"fixture":true}\n', encoding="utf-8")
    bundle_binding = TrustedSingleOperatorContentBundleBinding(
        absolute_path=str(bundle_path),
        size=bundle_path.stat().st_size,
        raw_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        semantic_sha256=bundle.semantic_sha256,
        runtime_binding_status="BOUND",
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: bundle,
    )
    content_source = FormalContentSourceBinding(
        schema_version=1,
        kind="formal_content_source_binding",
        mode="trusted_single_operator",
        offline_root_signed=None,
        trusted_single_operator=bundle_binding,
    )
    authority = object.__new__(E0TaskNativeSourceAuthority)
    object.__setattr__(authority, "task", "GSM8K")
    object.__setattr__(authority, "support_status", "READY")
    loaded: list[str] = []

    def load_descriptor(path):
        loaded.append(str(path))
        return authority

    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_single_operator_e0_workloads."
        "load_e0_task_native_source_authority",
        load_descriptor,
    )
    cell = MaterializedCell(
        stage="E0",
        method_role="Static",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="GSM8K",
        publication_policy="confirmation_only",
        recipe_sha256=None,
        dimensions=(),
    )

    assert (
        dispatch._materialize_trusted_single_operator_workload_source(
            content_source_binding=content_source,
            cell=cell,
            private_output_root=tmp_path.resolve(),
        )
        == descriptor_path
    )
    assert loaded == [str(descriptor_path)]
    assert not (tmp_path / "trusted-workload-source.json").exists()

    object.__setattr__(bundle, "e0_task_native_descriptors", ())
    with pytest.raises(ValueError, match="lacks one exact E0 descriptor"):
        dispatch._materialize_trusted_single_operator_workload_source(
            content_source_binding=content_source,
            cell=cell,
            private_output_root=tmp_path.resolve(),
        )

    object.__setattr__(bundle, "e0_task_native_descriptors", (descriptor,))
    object.__setattr__(authority, "support_status", "UNSUPPORTED")
    with pytest.raises(ValueError, match="not serving-ready"):
        dispatch._materialize_trusted_single_operator_workload_source(
            content_source_binding=content_source,
            cell=cell,
            private_output_root=tmp_path.resolve(),
        )


@pytest.mark.parametrize("stage", ("E3a", "TTS-Cal", "E1", "E2"))
def test_schema4_early_schedule_uses_trusted_content_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
) -> None:
    from lightcone_spec.experiments import (
        formal_single_operator_preflight_qualification as qualification_module,
    )
    from lightcone_spec.experiments import (
        formal_single_operator_stages as stages_module,
    )
    from lightcone_spec.experiments.formal_content_source import (
        FormalContentSourceBinding,
    )
    from lightcone_spec.experiments.formal_preflight_inputs import (
        TRUSTED_SINGLE_OPERATOR_QUALIFIED_PREFLIGHT_INPUTS_PROTOCOL_SHA256,
        FormalPreflightExecutionInputs,
    )
    from lightcone_spec.experiments.formal_single_operator_content import (
        TrustedSingleOperatorContentBundleBinding,
    )
    from lightcone_spec.runtime.content_authorization import (
        ContentJsonArtifactBinding,
    )

    def proof(name: str) -> CanonicalJsonProofBinding:
        path = (tmp_path / f"{name}.json").resolve()
        publish_canonical_json_no_replace(path, {"kind": name})
        return CanonicalJsonProofBinding.bind(path)

    bundle_path = (tmp_path / "trusted-content.json").resolve()
    publish_canonical_json_no_replace(bundle_path, {"kind": "trusted-content"})
    bundle_proof = CanonicalJsonProofBinding.bind(bundle_path)
    bundle_binding = TrustedSingleOperatorContentBundleBinding(
        absolute_path=str(bundle_path),
        size=bundle_proof.size,
        raw_sha256=bundle_proof.raw_sha256,
        semantic_sha256=bundle_proof.semantic_sha256,
        runtime_binding_status="BOUND",
    )
    reopen_calls: list[str] = []

    def reopen_bundle(binding):
        rebound = CanonicalJsonProofBinding.bind(binding.absolute_path)
        if (
            rebound.size != binding.size
            or rebound.raw_sha256 != binding.raw_sha256
            or rebound.semantic_sha256 != binding.semantic_sha256
        ):
            raise RuntimeError("trusted content bundle binding changed")
        reopen_calls.append(binding.absolute_path)
        return SimpleNamespace(
            runtime_binding_status="BOUND",
            semantic_sha256=binding.semantic_sha256,
        )

    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        reopen_bundle,
    )
    content_source = FormalContentSourceBinding(
        schema_version=1,
        kind="formal_content_source_binding",
        mode="trusted_single_operator",
        offline_root_signed=None,
        trusted_single_operator=bundle_binding,
    )
    monkeypatch.setattr(
        qualification_module,
        "load_formal_single_operator_preflight_qualification_plan_index",
        lambda _path: object(),
    )
    workload_path = (tmp_path / "workload.json").resolve()
    publish_canonical_json_no_replace(workload_path, {"kind": "workload"})
    workload = ContentJsonArtifactBinding.from_path(
        "formal_workload_authority:livecodebench_v6_hard",
        workload_path,
    )
    common = proof("common")
    request_bindings = tuple(proof(f"request-{index}") for index in range(8))
    preflight = FormalPreflightExecutionInputs(
        schema_version=4,
        kind="formal_single_operator_exact_ten_preflight_inputs",
        protocol_sha256=(
            TRUSTED_SINGLE_OPERATOR_QUALIFIED_PREFLIGHT_INPUTS_PROTOCOL_SHA256
        ),
        authority_mode="formal_single_operator_v1",
        execution_authority=common,
        inventory=proof("inventory"),
        content_receipt=None,
        workload_authority=workload,
        doctor_report=proof("doctor"),
        compile_assignment_plan=proof("compile-assignment"),
        exactness_assignment=proof("exactness-assignment"),
        interference_manifest=proof("interference"),
        request_schedule_sources=request_bindings,
        tokenization_inputs=request_bindings,
        tokenization_outputs=request_bindings,
        content_source_binding=content_source,
        qualification_plan_index=proof("qualification-index"),
    )
    execution_source = proof("execution-source")
    launch_binding = proof("launch")
    protocol_binding = SimpleNamespace(reopen=lambda **_kwargs: {"lock": "current"})
    current_source = SimpleNamespace(
        schema_version=3,
        content_source_binding=content_source,
        protocol_lock_source=protocol_binding,
    )
    monkeypatch.setattr(
        stages_module,
        "load_formal_single_operator_execution_source",
        lambda _path: current_source,
    )
    tts_authority = proof("tts-authority")
    trusted_sources = SimpleNamespace(
        tts_calibration_authority_source=SimpleNamespace(
            absolute_path=tts_authority.absolute_path
        )
    )
    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_registry.protocol_lock_from_dict",
        lambda _value: SimpleNamespace(
            schema_version=5,
            trusted_single_operator_source_bindings=trusted_sources,
        ),
    )
    captured: dict[str, object] = {}
    expected = object()

    def trusted_materializer(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        dispatch,
        "materialize_trusted_single_operator_request_schedule",
        trusted_materializer,
    )
    resolved_workload_path = (tmp_path / "content-owned-workload.json").resolve()
    publish_canonical_json_no_replace(
        resolved_workload_path,
        {"kind": "content-owned-workload"},
    )
    resolved: dict[str, object] = {}

    def trusted_workload_resolver(**kwargs):
        resolved.update(kwargs)
        return resolved_workload_path

    monkeypatch.setattr(
        dispatch,
        "_materialize_trusted_single_operator_workload_source",
        trusted_workload_resolver,
    )
    inputs = SimpleNamespace(
        execution_source=execution_source,
        compile_launch_manifest=launch_binding,
        private_output_root=str(tmp_path.resolve()),
    )
    cell = MaterializedCell(
        stage=stage,
        method_role="TTS" if stage == "TTS-Cal" else "Static",
        model="Qwen/Qwen3-8B",
        backend="DFLASH",
        task="trusted_early_schedule_fixture",
        publication_policy="fixed_barrier" if stage == "TTS-Cal" else "none",
        recipe_sha256=_sha("tts-recipe") if stage == "TTS-Cal" else None,
        dimensions=(
            ("concurrency", 1),
            ("context", 4096),
            ("regime", "short_input_long_generation"),
        ),
    )
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage=stage,
        protocol_lock_sha256=_sha("trusted-protocol-lock"),
        upstream_receipt_sha256s=(_sha("trusted-upstream"),),
        source_decision_sha256=_sha("trusted-source-decision"),
        materialization_rule="trusted_schema4_early_schedule_fixture",
        expected_cell_count=1,
        cells=(cell,),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    result = dispatch._materialize_single_operator_direct_schedule(
        inputs=inputs,
        preflight_inputs=preflight,
        input_binding=proof("early-plan-inputs"),
        launch=SimpleNamespace(schema_version=2),
        materialization=materialization,
        cell=cell,
        subject_sha256=_sha(f"{stage}-subject"),
    )
    assert result is expected
    assert captured["execution_source_path"] == execution_source.absolute_path
    assert captured["workload_source_path"] == resolved_workload_path
    assert captured["workload_source_path"] != workload.path
    assert captured["materialized_cell_id"] == cell.cell_id
    assert captured["tts_calibration_authority_path"] == (
        tts_authority.absolute_path if stage == "TTS-Cal" else None
    )
    assert reopen_calls
    assert resolved == {
        "content_source_binding": content_source,
        "cell": cell,
        "private_output_root": inputs.private_output_root,
    }

    if stage != "E3a":
        return
    foreign_path = (tmp_path / "foreign-content.json").resolve()
    publish_canonical_json_no_replace(foreign_path, {"kind": "foreign-content"})
    foreign_proof = CanonicalJsonProofBinding.bind(foreign_path)
    foreign_source = FormalContentSourceBinding(
        schema_version=1,
        kind="formal_content_source_binding",
        mode="trusted_single_operator",
        offline_root_signed=None,
        trusted_single_operator=TrustedSingleOperatorContentBundleBinding(
            absolute_path=foreign_proof.absolute_path,
            size=foreign_proof.size,
            raw_sha256=foreign_proof.raw_sha256,
            semantic_sha256=foreign_proof.semantic_sha256,
            runtime_binding_status="BOUND",
        ),
    )
    foreign_preflight = replace(preflight, content_source_binding=foreign_source)
    with pytest.raises(ValueError, match="content lineage differs"):
        dispatch._materialize_single_operator_direct_schedule(
            inputs=inputs,
            preflight_inputs=foreign_preflight,
            input_binding=proof("foreign-plan-inputs"),
            launch=SimpleNamespace(schema_version=2),
            materialization=materialization,
            cell=cell,
            subject_sha256=_sha("foreign-subject"),
        )
    bundle_path.write_text('{"kind":"tampered-content"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="bundle binding changed"):
        dispatch._materialize_single_operator_direct_schedule(
            inputs=inputs,
            preflight_inputs=preflight,
            input_binding=proof("tampered-plan-inputs"),
            launch=SimpleNamespace(schema_version=2),
            materialization=materialization,
            cell=cell,
            subject_sha256=_sha("tampered-subject"),
        )


def _run_tp1_operator_fixture(monkeypatch, tmp_path: Path):
    """Run one source-owned TP1 plan against a real local HTTP child."""

    import socket

    from test_live_sglang_runner import (
        _FAKE_SERVER_SOURCE,
        _dynamic_nvidia_smi_tool,
        _fake_live_server_configuration,
        _real_http_transport,
    )

    from lightcone_spec.orchestration import live_sglang

    server_source = (tmp_path / "fake-formal-tp1-server.py").resolve()
    server_source.write_text(_FAKE_SERVER_SOURCE, encoding="utf-8")
    server_config = (tmp_path / "fake-formal-tp1-config.json").resolve()
    pid_path = (tmp_path / "fake-formal-tp1.pid").resolve()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server_argv = (
        str(Path(sys.executable).resolve()),
        str(server_source),
        str(port),
        str(server_config),
        str(pid_path),
        "GPU-test-0",
        "--disable-cuda-graph",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-path",
        str((tmp_path / "models" / "target" / ("2" * 40)).resolve()),
        "--speculative-draft-model-path",
        str((tmp_path / "models" / "drafter" / ("3" * 40)).resolve()),
    )
    (
        output_root,
        content_path,
        workload_path,
        materialization_path,
        verified,
        launch_path,
        _source_value,
        inventory_path,
    ) = _install_materialization_fakes(
        monkeypatch,
        tmp_path,
        method="static",
        method_role="Static",
        max_running_requests=1,
        sample_count=2,
        cell_context_tokens=4,
        real_run_config=True,
        server_argv_override=server_argv,
        localhost_port=port,
    )
    plan = dispatch.materialize_formal_serving_run_plan(
        execution_binding=verified,
        content_verification_receipt_path=content_path,
        workload_authority_path=workload_path,
        materialization_path=materialization_path,
        compile_launch_manifest_path=launch_path,
        private_output_root=output_root,
        now_ns=20,
    )
    plan_path = output_root / "formal-serving-run-plan.json"
    admission_binding = publish_formal_single_operator_admission(
        plan_path=plan_path,
        inventory_path=inventory_path,
    )
    config, _warmup, _scored = _fake_live_server_configuration(
        plan.native_terminal_binding,
        warmup_inputs=(1, 2),
        warmup_outputs=(3, 4),
        scored_inputs=(11, 12),
        scored_outputs=(6, 7),
    )
    server_config.write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    launch = dispatch.CompileLaunchManifest.load(launch_path)
    run_config = verified.run_config
    monkeypatch.setattr(
        live_sglang.CompileLaunchManifest,
        "load",
        classmethod(lambda _cls, _path: launch),
    )
    monkeypatch.setattr(live_sglang, "load_run_config", lambda _path: run_config)
    transport = _real_http_transport()
    _bound_request_type, transport_type = live_sglang._serving_runtime_types()
    monkeypatch.setattr(
        transport_type,
        "from_checkout",
        classmethod(lambda _cls, _path: transport),
    )
    tool = _dynamic_nvidia_smi_tool(
        tmp_path,
        pid_path=pid_path,
        gpu_uuid="GPU-test-0",
    )
    result = asyncio.run(
        dispatch.execute_formal_serving_run_plan(
            plan_path=plan_path,
            launch_admission_path=admission_binding.absolute_path,
            execution_binding=verified,
            nvidia_smi_tool=tool,
        )
    )
    return plan, plan_path, verified, inventory_path, admission_binding, result


def test_live_sglang_lazy_transport_resolves_in_fresh_interpreter() -> None:
    repository = Path(__file__).parents[1].resolve()
    code = """
from lightcone_spec.orchestration import live_sglang
assert live_sglang.PinnedBenchServingTransport is None
request_type, transport_type = live_sglang._serving_runtime_types()
from lightcone_spec.experiments.serving import (
    BoundServingRequest,
    PinnedBenchServingTransport,
)
assert request_type is BoundServingRequest
assert transport_type is PinnedBenchServingTransport
assert live_sglang.PinnedBenchServingTransport is PinnedBenchServingTransport
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _run_distributed_operator_fixture(
    monkeypatch, tmp_path, *, topology: str = "tp2_dp1"
):
    (
        output_root,
        content_path,
        workload_path,
        materialization_path,
        verified,
        launch_path,
        _source_value,
        inventory_path,
    ) = _install_materialization_fakes(
        monkeypatch,
        tmp_path,
        topology=topology,
        real_run_config=True,
        sample_count=16 if topology == "tp1_dp2" else 4,
    )
    plan = dispatch.materialize_formal_serving_run_plan(
        execution_binding=verified,
        content_verification_receipt_path=content_path,
        workload_authority_path=workload_path,
        materialization_path=materialization_path,
        compile_launch_manifest_path=launch_path,
        private_output_root=output_root,
        now_ns=20,
    )
    plan_path = output_root / "formal-serving-run-plan.json"
    admission = publish_formal_single_operator_admission(
        plan_path=plan_path,
        inventory_path=inventory_path,
    )
    observed: dict[str, object] = {}
    schedule = dispatch._reopen_schedule_receipt(plan.request_schedule_receipt)

    def fake_snapshot(
        *,
        tool,
        gpu_uuids,
        inventory_sha256,
        phase,
        output_path,
        expected_server_process_group_ids=None,
        shared_server_process_group_id=None,
    ):
        if phase == "ready":
            assert shared_server_process_group_id is not None
            observed["shared_process_group_id"] = shared_server_process_group_id
        process_group_ids = (
            [shared_server_process_group_id] * len(gpu_uuids)
            if shared_server_process_group_id is not None
            else None
        )
        processes = (
            []
            if phase != "ready"
            else [
                {
                    "gpu_uuid": gpu_uuid,
                    "pid": 10_000 + rank,
                    "process_group_id": shared_server_process_group_id,
                    "used_gpu_memory_mib": 1,
                }
                for rank, gpu_uuid in enumerate(gpu_uuids)
            ]
        )
        publish_canonical_json_no_replace(
            output_path,
            {
                "schema_version": 1,
                "kind": "unsigned_pinned_sglang_gpu_process_snapshot",
                "protocol_sha256": _sha("snapshot-protocol"),
                "phase": phase,
                "captured_ns": {"before": 1, "ready": 2, "after": 3}[phase],
                "gpu_uuids": list(gpu_uuids),
                "inventory_sha256": inventory_sha256,
                "server_process_group_ids": process_group_ids,
                "nvidia_smi": {"test": True},
                "gpu_rows": [
                    {"uuid": gpu_uuid, "name": "test", "memory_used_mib": 1}
                    for gpu_uuid in gpu_uuids
                ],
                "compute_process_rows": processes,
            },
        )
        return CanonicalJsonProofBinding.bind(output_path)

    native_identity = dispatch._formal_gang_native_identity(plan)
    full_schedule = {
        phase: [
            {
                "request_id": row.request.request_id,
                "cohort_sha256": row.request.cohort_sha256,
                "routed_dp_rank": row.routed_dp_rank,
            }
            for row in schedule.requests
            if row.phase == phase
        ]
        for phase in ("warmup", "scored")
    }
    sticky_routes = sorted(
        {
            row.request.cohort_sha256: row.routed_dp_rank for row in schedule.requests
        }.items()
    )

    def phase_request_rows(phase):
        phase_rows = tuple(row for row in schedule.requests if row.phase == phase)
        return [
            {
                "request_id": row.request.request_id,
                "input_token_ids": list(row.request.input_token_ids),
                "output_token_ids": [71, 72],
                "native_itl_semantics": (
                    "scheduler_committed_token_at_result_processor_v1"
                ),
                "native_itl_event_count": 2,
                "native_itl_events_sha256": dispatch._sha256(
                    [
                        {
                            "token_index": 0,
                            "token_id": 71,
                            "observed_ns": 120 + ordinal * 100,
                        },
                        {
                            "token_index": 1,
                            "token_id": 72,
                            "observed_ns": 140 + ordinal * 100,
                        },
                    ]
                ),
                "terminal_status": "completed",
                "terminal_reason": "FINISH_LENGTH",
            }
            for ordinal, row in enumerate(phase_rows)
        ]

    def allocation_free_native_state():
        scheduler_state = {
            "schema_version": 2,
            "scheduler_idle": True,
            "active_requests": 0,
            "queued_requests": 0,
            "request_pool_active_slots": 0,
            "allocator_current_hbm_bytes": 32,
            "allocator_reserved_hbm_bytes": 64,
            "allocator_peak_hbm_bytes": 3072,
            "kv_token_capacity": 1024,
            "kv_available_tokens": 1024,
            "kv_state_sha256": _sha("gang-kv-state"),
            "rng_state_sha256": _sha("gang-rng-state"),
            "adapter_state_sha256": _sha("gang-adapter-state"),
            "adapter_reset_verified": True,
            "adapter_reset_scope": native_identity[0],
            "adapter_request_admission_policy": native_identity[1],
            "adapter_request_source_point_reset_protocol_sha256": (native_identity[2]),
            "adapter_runtime_trust_mode": native_identity[3],
            "adapter_formal_measurement": native_identity[4],
            "adapter_active_request_id": None,
            "adapter_request_epoch": 0,
            "adapter_source_round": 0,
            "adapter_active_version": 0,
            "adapter_epoch": 0,
            "optimizer_generation": 0,
            "telemetry_generation": 1,
            "completion_event_generation": 1,
            "completion_event_complete": True,
        }
        return {
            "scheduler": scheduler_state,
            "round_rows": [],
            "update_rows": [],
            "performance_counters": {
                "target_calls": 0,
                "peak_hbm_bytes": scheduler_state["allocator_peak_hbm_bytes"],
                "updates_launched": 0,
                "updates_published": 0,
                "exposed_update_ms": None,
                "exactness_violations": 0,
                "version_mismatches": 0,
                "fallbacks": 0,
                "nonfinite_updates": 0,
                "oom_events": 0,
                "retractions": 0,
                "communicator_failures": 0,
            },
            "historical_kv_source_versions": {},
            "request_source_point_resets": {
                "schema_version": 1,
                "reset_scope": native_identity[0],
                "request_admission_policy": native_identity[1],
                "protocol_sha256": native_identity[2],
                "final_archive_sha256": "0" * 64,
                "receipts": [],
            },
            "adaptation": None,
        }

    def rank_terminals_for(phase, *, client_lifecycle_sha256):
        request_rows = phase_request_rows(phase)
        values = []
        for rank, gpu_uuid in enumerate(plan.gpu_uuids):
            local_routes = (
                full_schedule[phase]
                if plan.topology_mode == "tp2_dp1"
                else [
                    row for row in full_schedule[phase] if row["routed_dp_rank"] == rank
                ]
            )
            local_ids = [row["request_id"] for row in local_routes]
            local_request_rows = [
                row for row in request_rows if row["request_id"] in set(local_ids)
            ]
            native_state = allocation_free_native_state()
            rank_value = {
                "schema_version": 2,
                "kind": "sglang_formal_gang_rank_terminal",
                "hook": "sglang.lightcone_formal_gang_serving.v1",
                "protocol_sha256": dispatch.FORMAL_GANG_SERVING_PROTOCOL_SHA256,
                "topology": plan.topology_mode,
                "rank": rank,
                "world_size": 2,
                "gpu_uuid": gpu_uuid,
                "execution_plan_sha256": (
                    plan.native_terminal_binding.execution_plan_sha256
                ),
                "rank_config_sha256": (plan.native_terminal_binding.rank_config_sha256),
                "run_nonce_sha256": plan.native_terminal_binding.run_nonce_sha256,
                "method": plan.method,
                "reset_scope": native_identity[0],
                "request_admission_policy": native_identity[1],
                "request_source_point_reset_protocol_sha256": native_identity[2],
                "runtime_trust_mode": native_identity[3],
                "formal_measurement": native_identity[4],
                "phase": phase,
                "full_schedule_sha256": dispatch._sha256(full_schedule),
                "local_request_routes_sha256": dispatch._sha256(local_routes),
                "sticky_cohort_routes_sha256": dispatch._sha256(sticky_routes),
                "expected_request_ids_sha256": dispatch._sha256(local_ids),
                "request_terminals": local_request_rows,
                "request_terminal_sha256s": [
                    dispatch._sha256(row) for row in local_request_rows
                ],
                "native_state": native_state,
                "native_state_sha256": dispatch._sha256(native_state),
                "client_lifecycle_sha256": client_lifecycle_sha256,
                "non_submitted_request_ids_sha256": (
                    None
                    if phase == "warmup" or client_lifecycle_sha256 is None
                    else dispatch._sha256([])
                ),
                "status": "COMPLETE",
                "reason_code": None,
            }
            values.append(
                {
                    **rank_value,
                    "terminal_sha256": dispatch._sha256(rank_value),
                }
            )
        return values

    class FakeTransport:
        @classmethod
        def from_checkout(cls, checkout):
            assert Path(checkout) == tmp_path / "patched-sglang"
            return cls()

        async def open(self, **_kwargs):
            observed["opened"] = True

        async def close(self):
            observed["closed"] = True

        def bind_native_admin_base_url(self, value):
            observed["base_url"] = value

        async def get_json(self, path):
            assert path.endswith("/capability")
            rank_rows = [
                {
                    "schema_version": 2,
                    "kind": "sglang_formal_gang_rank_capability",
                    "hook": "sglang.lightcone_formal_gang_serving.v1",
                    "protocol_sha256": dispatch.FORMAL_GANG_SERVING_PROTOCOL_SHA256,
                    "execution_plan_sha256": (
                        plan.native_terminal_binding.execution_plan_sha256
                    ),
                    "rank_config_sha256": (
                        plan.native_terminal_binding.rank_config_sha256
                    ),
                    "run_nonce_sha256": plan.native_terminal_binding.run_nonce_sha256,
                    "topology": plan.topology_mode,
                    "rank": rank,
                    "world_size": 2,
                    "method": plan.method,
                    "reset_scope": native_identity[0],
                    "request_admission_policy": native_identity[1],
                    "request_source_point_reset_protocol_sha256": native_identity[2],
                    "runtime_trust_mode": native_identity[3],
                    "formal_measurement": native_identity[4],
                    "assignment_sha256": _sha("physical-assignment"),
                    "inventory_sha256": plan.inventory_sha256,
                    "gpu_uuid": gpu_uuid,
                    "process_id": 10_000 + rank,
                }
                for rank, gpu_uuid in enumerate(plan.gpu_uuids)
            ]
            return {
                "schema_version": 2,
                "kind": "sglang_formal_gang_capability",
                "hook": "sglang.lightcone_formal_gang_serving.v1",
                "protocol_sha256": dispatch.FORMAL_GANG_SERVING_PROTOCOL_SHA256,
                "topology": plan.topology_mode,
                "world_size": 2,
                "method": plan.method,
                "reset_scope": native_identity[0],
                "request_admission_policy": native_identity[1],
                "request_source_point_reset_protocol_sha256": native_identity[2],
                "runtime_trust_mode": native_identity[3],
                "formal_measurement": native_identity[4],
                "execution_plan_sha256": plan.native_terminal_binding.execution_plan_sha256,
                "rank_config_sha256": plan.native_terminal_binding.rank_config_sha256,
                "run_nonce_sha256": plan.native_terminal_binding.run_nonce_sha256,
                "rank_capabilities": rank_rows,
                "rank_capability_sha256s": [dispatch._sha256(row) for row in rank_rows],
            }

        async def post_json(self, path, payload):
            action = path.rsplit("/", 1)[-1]
            observed.setdefault("actions", []).append(action)
            if action == "begin":
                value = {
                    "schema_version": 2,
                    "kind": "sglang_formal_gang_begin",
                    "hook": "sglang.lightcone_formal_gang_serving.v1",
                    "protocol_sha256": dispatch.FORMAL_GANG_SERVING_PROTOCOL_SHA256,
                    "topology": plan.topology_mode,
                    "world_size": 2,
                    "execution_plan_sha256": plan.native_terminal_binding.execution_plan_sha256,
                    "schedule_sha256": dispatch._sha256(payload),
                    "reset_scope": native_identity[0],
                    "request_admission_policy": native_identity[1],
                    "request_source_point_reset_protocol_sha256": native_identity[2],
                    "runtime_trust_mode": native_identity[3],
                    "formal_measurement": native_identity[4],
                    "rank_begin_sha256s": [_sha("begin-0"), _sha("begin-1")],
                }
                return {**value, "begin_sha256": dispatch._sha256(value)}
            phase = "warmup" if action == "reset" else "scored"
            lifecycle_sha256 = (
                None
                if action == "reset" or "client_lifecycle_rows" not in payload
                else dispatch._sha256(payload["client_lifecycle_rows"])
            )
            rank_terminals = rank_terminals_for(
                phase,
                client_lifecycle_sha256=lifecycle_sha256,
            )
            rank_terminal_sha256s = [row["terminal_sha256"] for row in rank_terminals]
            value = {
                "schema_version": 2,
                "kind": "sglang_formal_gang_all_rank_terminal",
                "hook": "sglang.lightcone_formal_gang_serving.v1",
                "protocol_sha256": dispatch.FORMAL_GANG_SERVING_PROTOCOL_SHA256,
                "topology": plan.topology_mode,
                "world_size": 2,
                "action": f"formal_gang_{action}",
                "reset_scope": native_identity[0],
                "request_admission_policy": native_identity[1],
                "request_source_point_reset_protocol_sha256": native_identity[2],
                "runtime_trust_mode": native_identity[3],
                "formal_measurement": native_identity[4],
                "decision": "COMMITTED",
                "published_ranks": [0, 1],
                "reason_code": None,
                "cross_replica_gradient_collective": False,
                "rank_terminals": rank_terminals,
                "rank_terminal_sha256s": rank_terminal_sha256s,
            }
            if action == "reset":
                value["rank_reset_sha256s"] = [_sha("reset-0"), _sha("reset-1")]
            return {**value, "aggregate_sha256": dispatch._sha256(value)}

    class FakeRequestResult:
        def __init__(self, request):
            self.request_id = request.request_id
            self.input_token_ids = request.input_token_ids
            self.output_token_ids = (71, 72)
            self.terminal_status = "completed"
            self.terminal_reason = "FINISH_LENGTH"
            self.submitted_to_server = True

        def validate(self):
            return None

    async def fake_phase(_phase, requests, **_kwargs):
        pointers = []
        lifecycle_rows = []
        for ordinal, request in enumerate(requests):
            pointer = {
                "schema_version": 1,
                "kind": "sglang_native_itl_result_pointer",
                "hook": "sglang.schema_v3.native_per_token_timestamp.v2",
                "semantics": "scheduler_committed_token_at_result_processor_v1",
                "release_status": "IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF",
                "request_id": request.request_id,
                "request_started_ns": 100 + ordinal * 100,
                "request_terminal_ns": 180 + ordinal * 100,
                "terminal_status": "completed",
                "terminal_reason": "FINISH_LENGTH",
                "events": [
                    {
                        "token_index": index,
                        "token_id": token,
                        "observed_ns": 120 + ordinal * 100 + index * 20,
                    }
                    for index, token in enumerate((71, 72))
                ],
            }
            result_pointer_sha256 = dispatch._sha256(pointer)
            pointers.append(
                json.dumps(
                    {
                        **pointer,
                        "result_pointer_sha256": result_pointer_sha256,
                    }
                )
            )
            lifecycle_rows.append(
                {
                    "request_id": request.request_id,
                    "phase": _phase,
                    "scheduled_arrival_us": ordinal,
                    "offered": True,
                    "offered_at_us": ordinal + 1,
                    "admitted_at_us": ordinal + 1,
                    "effective_deadline_us": 1_000_000,
                    "cancellation_at_us": None,
                    "terminal_at_us": ordinal + 2,
                    "outcome_status": "completed",
                    "outcome_code": "completed",
                    "submitted_to_server": True,
                    "native_terminal_status": "completed",
                    "native_result_pointer_sha256": result_pointer_sha256,
                }
            )
        return SimpleNamespace(
            requests=tuple(FakeRequestResult(request) for request in requests),
            native_result_pointer_json=tuple(pointers),
            client_lifecycle_rows=tuple(lifecycle_rows),
        )

    monkeypatch.setattr(dispatch, "_capture_gpu_process_snapshot", fake_snapshot)
    monkeypatch.setattr(dispatch, "_wait_server_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dispatch, "PinnedBenchServingTransport", FakeTransport)
    monkeypatch.setattr(
        dispatch,
        "_observe_live_server_execution_policy",
        lambda **_kwargs: asyncio.sleep(0),
    )
    monkeypatch.setattr(dispatch, "_execute_source_owned_phase", fake_phase)
    result = asyncio.run(
        dispatch.execute_formal_distributed_serving_run_plan(
            plan_path=plan_path,
            launch_admission_path=admission.absolute_path,
            execution_binding=verified,
            nvidia_smi_tool=object(),
        )
    )
    assert result.receipt.reopen()["process_group_empty"] is True
    receipt = result.receipt.reopen()
    assert receipt["process_exit_code"] in {0, -15}
    assert (output_root / "formal-single-operator-admission-consumed.json").is_file()
    assert observed["actions"] == ["begin", "reset", "finalize"]
    assert observed["opened"] is True and observed["closed"] is True
    assert type(observed["shared_process_group_id"]) is int
    assert result.formal_gang_terminal.reopen()["published_ranks"] == [0, 1]
    binding = (
        terminal_result.build_formal_distributed_terminal_external_control_binding(
            result.receipt.absolute_path,
            plan_path=str(plan_path),
            expected_inventory_sha256=plan.inventory_sha256,
            expected_registry_sha256=admission.reopen()["registry_sha256"],
        )
    )
    assert binding.topology_mode == topology
    assert binding.execution_plan_sha256 == (
        plan.native_terminal_binding.execution_plan_sha256
    )
    outcome = terminal_result.validate_formal_distributed_physical_outcome(
        plan_path=str(plan_path),
        run_receipt_path=result.receipt.absolute_path,
        expected_inventory_sha256=plan.inventory_sha256,
        expected_registry_sha256=admission.reopen()["registry_sha256"],
    )
    assert outcome.status == "COMPLETE"
    assert outcome.finished_ns >= outcome.execution_started_ns
    assert outcome.process_exit_code in {0, -15}
    assert outcome.server_stdout.absolute_path == plan.server_stdout_output_path
    assert outcome.server_stderr.absolute_path == plan.server_stderr_output_path
    return plan, result, admission, inventory_path, verified


def test_distributed_operator_runs_real_child_and_source_owned_gang_protocol(
    monkeypatch, tmp_path
) -> None:
    _run_distributed_operator_fixture(monkeypatch, tmp_path)


@pytest.mark.parametrize(
    ("topology", "expected"),
    (("tp1_dp1", "tp1"), ("tp2_dp1", "gang"), ("tp1_dp2", "gang")),
)
def test_public_formal_operator_routes_only_from_sealed_topology(
    monkeypatch, topology, expected
) -> None:
    plan = SimpleNamespace(topology_mode=topology)
    binding = SimpleNamespace(verified_nextn_tp2_authority=None)
    observed: list[str] = []

    monkeypatch.setattr(
        dispatch,
        "load_formal_serving_run_plan",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        dispatch,
        "require_verified_formal_serving_execution_binding",
        lambda value: value,
    )

    async def fake_tp1(**_kwargs):
        observed.append("tp1")
        return "tp1-result"

    async def fake_gang(**_kwargs):
        observed.append("gang")
        return "gang-result"

    monkeypatch.setattr(dispatch, "execute_formal_tp1_serving_run_plan", fake_tp1)
    monkeypatch.setattr(
        dispatch,
        "execute_formal_distributed_serving_run_plan",
        fake_gang,
    )
    result = asyncio.run(
        dispatch.execute_formal_serving_run_plan(
            plan_path="/sealed/formal-plan.json",
            launch_admission_path="/sealed/formal-stage-launch-admission.json",
            execution_binding=binding,
            nvidia_smi_tool=object(),
        )
    )
    assert observed == [expected]
    assert result == f"{expected}-result"
    parameters = inspect.signature(dispatch.execute_formal_serving_run_plan).parameters
    assert parameters["execution_binding"].default is None
    assert {
        "prompt",
        "input_token_ids",
        "requests",
        "port",
        "argv",
        "transport",
    }.isdisjoint(parameters)
    assert tuple(
        inspect.signature(
            dispatch.rebuild_formal_single_operator_execution_binding_from_plan
        ).parameters
    ) == ("plan_path", "current_ns")
    assert tuple(
        inspect.signature(
            dispatch.execute_formal_single_operator_serving_run_plan
        ).parameters
    ) == ("plan_path", "nvidia_smi_tool")


def _distributed_lifecycle_fixture(monkeypatch, tmp_path: Path):
    plan, result, admission, inventory_path, _verified = (
        _run_distributed_operator_fixture(monkeypatch, tmp_path)
    )
    plan_path = Path(plan.private_output_root) / "formal-serving-run-plan.json"
    inventory = GpuInventory.from_dict(
        CanonicalJsonProofBinding.bind(inventory_path).reopen()
    )
    hardware = inventory.devices[0].hardware_envelope_sha256
    assert {row.hardware_envelope_sha256 for row in inventory.devices} == {hardware}
    context = _release_control_context(
        monkeypatch,
        hardware_envelope_sha256=hardware,
    )
    registry_sha256 = admission.reopen()["registry_sha256"]
    replay_root = (tmp_path / "lifecycle-replay").resolve()
    replay_root.mkdir(mode=0o700)
    replay_store = ChallengeReplayStore(str(replay_root))
    subject = terminal_result.build_formal_terminal_control_subject(
        plan_path=str(plan_path),
        expected_inventory_sha256=plan.inventory_sha256,
        expected_registry_sha256=registry_sha256,
    )
    control = _release_control(
        context,
        inventory_sha256=plan.inventory_sha256,
        subject=subject,
        label="distributed-terminal",
    )
    terminal_proof_path = Path(plan.private_output_root) / "terminal-proof.json"
    terminal_result.publish_formal_terminal_result_proof_artifact(
        plan_path=str(plan_path),
        control_attestation=control,
        replay_store=replay_store,
        expected_inventory_sha256=plan.inventory_sha256,
        expected_registry_sha256=registry_sha256,
        expected_root_manifest_sha256=context.root_binding.semantic_sha256,
        now_ns=2_000_000_000,
        proof_artifact_path=str(terminal_proof_path),
    )
    proof_path = Path(plan.private_output_root) / "lifecycle-proof.json"
    proof = serving_lift.publish_formal_serving_lifecycle_proof(
        plan_path,
        native_result_proof_path=terminal_proof_path,
        expected_registry_sha256=registry_sha256,
        expected_root_manifest_sha256=context.root_binding.semantic_sha256,
        now_ns=2_000_000_000,
        proof_artifact_path=proof_path,
    )
    assert result.lifecycle_timing.absolute_path == plan.lifecycle_timing_output_path
    return (
        plan,
        proof,
        proof_path,
        registry_sha256,
        context.root_binding.semantic_sha256,
    )


def test_distributed_lifecycle_proof_reopens_terminal_dag(
    monkeypatch, tmp_path
) -> None:
    plan, proof, proof_path, registry_sha256, root_sha256 = (
        _distributed_lifecycle_fixture(monkeypatch, tmp_path)
    )
    artifact = serving_lift.validate_formal_distributed_lifecycle_timing_proof_artifact(
        proof_path,
        expected_inventory_sha256=plan.inventory_sha256,
        expected_registry_sha256=registry_sha256,
        expected_root_manifest_sha256=root_sha256,
        now_ns=2_000_000_000,
    )
    assert artifact.sha256 == proof.semantic_sha256
    assert artifact.topology_mode == "tp2_dp1"


def test_distributed_lifecycle_proof_rejects_raw_timing_tamper(
    monkeypatch, tmp_path
) -> None:
    plan, _proof, proof_path, registry_sha256, root_sha256 = (
        _distributed_lifecycle_fixture(monkeypatch, tmp_path)
    )
    Path(plan.lifecycle_timing_output_path).write_text(
        '{"kind":"tampered"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed"):
        serving_lift.validate_formal_distributed_lifecycle_timing_proof_artifact(
            proof_path,
            expected_inventory_sha256=plan.inventory_sha256,
            expected_registry_sha256=registry_sha256,
            expected_root_manifest_sha256=root_sha256,
            now_ns=2_000_000_000,
        )


def test_distributed_lifecycle_proof_rejects_foreign_registry(
    monkeypatch, tmp_path
) -> None:
    plan, _proof, proof_path, _registry_sha256, root_sha256 = (
        _distributed_lifecycle_fixture(monkeypatch, tmp_path)
    )
    with pytest.raises(ValueError, match="authority differs"):
        serving_lift.validate_formal_distributed_lifecycle_timing_proof_artifact(
            proof_path,
            expected_inventory_sha256=plan.inventory_sha256,
            expected_registry_sha256=_sha("foreign-registry"),
            expected_root_manifest_sha256=root_sha256,
            now_ns=2_000_000_000,
        )
