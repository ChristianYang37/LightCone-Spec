from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.config import (
    AdaptationConfig,
    ModelPair,
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
)
from lightcone_spec.experiments import e0_stage_authority
from lightcone_spec.experiments.e0_stage_authority import (
    E0OnlineSpecSourceAuthority,
)
from lightcone_spec.experiments.formal_protocol import ProtocolLock, content_sha256
from lightcone_spec.experiments.formal_single_operator_downstream import (
    _e0_compatibility_from_auxiliary,
)
from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
    E0CompatibilityProbeTerminal,
    E0PreparedModelBackendInterfaceReceipt,
    E0TaskNativeWorkloadAuthority,
    TrustedSingleOperatorEagle3ExecutionAuthority,
    e0_preprobe_interface_sha256,
    publish_e0_compatibility_probes,
    reduce_e0_compatibility_probes,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    RebuiltFormalSingleOperatorStageCompletion,
)
from lightcone_spec.experiments.onlinespec import (
    ONLINE_SPEC_COMMIT,
    ONLINE_SPEC_SOURCE_AUDIT_SHA256,
    ONLINE_SPEC_TREE,
)
from lightcone_spec.experiments.stage_materialization import (
    E0_BACKENDS,
    E0_MODELS,
    E0_TASKS,
)
from lightcone_spec.orchestration.live_sglang import _eagle3_execution_authority
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _lock() -> ProtocolLock:
    return ProtocolLock(
        schema_version=4,
        protocol_id="e0-compatibility-test",
        code_git_head="1" * 40,
        code_git_tree="2" * 40,
        patch_manifest_sha256=_sha("patch"),
        registry_sha256=_sha("registry"),
        english_protocol_sha256=_sha("english"),
        chinese_protocol_sha256=_sha("chinese"),
        tts_calibration_authority_sha256=_sha("tts"),
        chronobelief_authority_sha256=_sha("chronobelief"),
        e1_recipe_anchor_authority_sha256=_sha("e1"),
        e2_recipe_grid_authority_sha256=_sha("e2"),
        formal_runtime_authority_manifest_sha256=_sha("runtime"),
        offline_release_trust_root_sha256=_sha("root"),
        prepared_model_content_authorization_sha256=_sha("models"),
        formal_workload_e3a_authorization_sha256=_sha("e3a-workload"),
        formal_workload_e0_authorization_sha256=_sha("e0-workload"),
        burstgpt_shape_authorization_sha256=_sha("burstgpt"),
        native_runtime_qualification_protocol_sha256=_sha("native-protocol"),
        native_runtime_qualification_runner_sha256=_sha("native-runner"),
        native_runtime_qualification_test_set_sha256=_sha("native-tests"),
        compile_qualification_protocol_sha256=_sha("compile-protocol"),
        compile_qualification_runner_sha256=_sha("compile-runner"),
        compile_qualification_test_set_sha256=_sha("compile-tests"),
        exactness_qualification_protocol_sha256=_sha("exactness-protocol"),
        exactness_qualification_runner_sha256=_sha("exactness-runner"),
        exactness_qualification_test_set_sha256=_sha("exactness-tests"),
    )


def _e6_completion(lock: ProtocolLock) -> RebuiltFormalSingleOperatorStageCompletion:
    materialization_sha256 = _sha("e6-materialization")
    confirmation_sha256 = _sha("e6-confirmation")
    return RebuiltFormalSingleOperatorStageCompletion(
        artifact=SimpleNamespace(node="e6_final"),
        predecessor=None,
        node_materialization=SimpleNamespace(),
        materialization=SimpleNamespace(
            sha256=materialization_sha256,
            protocol_lock_sha256=lock.sha256,
        ),
        decision=SimpleNamespace(
            payload={"confirmation_sha256": confirmation_sha256},
            next_materialization_source_decision_sha256=confirmation_sha256,
            next_materialization_upstream_receipt_sha256s=(materialization_sha256,),
        ),
    )


def _source_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> E0OnlineSpecSourceAuthority:
    checkout = (tmp_path / "onlinespec").resolve()
    checkout.mkdir()
    audit = (tmp_path / "onlinespec-audit.json").resolve()
    audit.write_text("{}\n", encoding="utf-8")
    verified = {
        "commit": ONLINE_SPEC_COMMIT,
        "tree": ONLINE_SPEC_TREE,
        "key_files": {"optimizer.py": _sha("optimizer")},
    }
    monkeypatch.setattr(
        e0_stage_authority,
        "verify_onlinespec_source_checkout",
        lambda *_args, **_kwargs: verified,
    )
    return E0OnlineSpecSourceAuthority(
        schema_version=1,
        checkout_path=str(checkout),
        audit_path=str(audit),
        source_audit_sha256=ONLINE_SPEC_SOURCE_AUDIT_SHA256,
        commit=ONLINE_SPEC_COMMIT,
        tree=ONLINE_SPEC_TREE,
        verification_sha256=content_sha256(verified),
    )


def _probe_inputs(
    lock: ProtocolLock,
    *,
    unsupported_interfaces: set[tuple[str, str]] | None = None,
    unsupported_workloads: set[tuple[str, str]] | None = None,
    unsupported_smokes: set[tuple[str, str, str]] | None = None,
):
    unsupported_interfaces = unsupported_interfaces or set()
    unsupported_workloads = unsupported_workloads or set()
    unsupported_smokes = unsupported_smokes or set()
    e6_confirmation = _sha("e6-confirmation")
    interfaces = []
    workloads = []
    terminals = []
    interface_by_key = {}
    workload_by_key = {}
    for model in E0_MODELS:
        tokenizer = _sha(f"tokenizer:{model}")
        for backend in E0_BACKENDS:
            unsupported = (model, backend) in unsupported_interfaces
            receipt = E0PreparedModelBackendInterfaceReceipt(
                schema_version=1,
                protocol_lock_sha256=lock.sha256,
                upstream_e6_confirmation_sha256=e6_confirmation,
                model=model,
                backend=backend,
                tokenizer_sha256=tokenizer,
                interface_sha256=_sha(f"interface:{model}:{backend}"),
                prepared_model_manifest_sha256=_sha(f"manifest:{model}:{backend}"),
                support_status="UNSUPPORTED" if unsupported else "READY",
                reason_code=(
                    "MODEL_BACKEND_INTERFACE_UNSUPPORTED"
                    if unsupported
                    else "INTERFACE_READY"
                ),
                requires_gpu_smoke=True,
                evidence_sha256=_sha(f"interface-evidence:{model}:{backend}"),
            )
            interfaces.append(receipt)
            interface_by_key[(model, backend)] = receipt
        for task in E0_TASKS:
            unsupported = (model, task) in unsupported_workloads
            authority = E0TaskNativeWorkloadAuthority(
                schema_version=1,
                protocol_lock_sha256=lock.sha256,
                upstream_e6_confirmation_sha256=e6_confirmation,
                model=model,
                task=task,
                tokenizer_sha256=tokenizer,
                task_native_workload_sha256=_sha(f"workload:{model}:{task}"),
                source_revision_sha256=_sha(f"revision:{task}"),
                support_status="UNSUPPORTED" if unsupported else "READY",
                reason_code=(
                    "TOKENIZER_TASK_WORKLOAD_UNSUPPORTED"
                    if unsupported
                    else "TASK_WORKLOAD_READY"
                ),
                evidence_sha256=_sha(f"workload-evidence:{model}:{task}"),
            )
            workloads.append(authority)
            workload_by_key[(model, task)] = authority
    index = 0
    for model in E0_MODELS:
        for backend in E0_BACKENDS:
            for task in E0_TASKS:
                interface = interface_by_key[(model, backend)]
                workload = workload_by_key[(model, task)]
                key = (model, backend, task)
                if interface.support_status == "UNSUPPORTED":
                    disposition = "N/A"
                    reason = "MODEL_BACKEND_INTERFACE_UNSUPPORTED"
                    smoke = "NOT_REQUIRED"
                    completed = 0
                elif workload.support_status == "UNSUPPORTED":
                    disposition = "N/A"
                    reason = "TOKENIZER_TASK_WORKLOAD_UNSUPPORTED"
                    smoke = "NOT_REQUIRED"
                    completed = 0
                elif key in unsupported_smokes:
                    disposition = "N/A"
                    reason = "GPU_SMOKE_REGISTERED_UNSUPPORTED"
                    smoke = "REGISTERED_UNSUPPORTED"
                    completed = 0
                else:
                    disposition = "VALID"
                    reason = "PROBE_COMPATIBLE"
                    smoke = "PASS"
                    completed = 1
                started = 1_000 + index * 10
                terminals.append(
                    E0CompatibilityProbeTerminal(
                        schema_version=1,
                        protocol_lock_sha256=lock.sha256,
                        upstream_e6_confirmation_sha256=e6_confirmation,
                        model=model,
                        backend=backend,
                        task=task,
                        interface_sha256=interface.interface_sha256,
                        task_native_workload_sha256=(
                            workload.task_native_workload_sha256
                        ),
                        tokenizer_sha256=interface.tokenizer_sha256,
                        command_sha256=_sha(f"command:{model}:{backend}:{task}"),
                        started_ns=started,
                        finished_ns=started + 5,
                        terminal_status="COMPLETE",
                        exit_code=0,
                        stdout_sha256=_sha(f"stdout:{model}:{backend}:{task}"),
                        stderr_sha256=_sha(f"stderr:{model}:{backend}:{task}"),
                        junit_sha256=_sha(f"junit:{model}:{backend}:{task}"),
                        junit_status="PASS",
                        evidence_sha256=_sha(f"evidence:{model}:{backend}:{task}"),
                        smoke_status=smoke,
                        completed_request_count=completed,
                        disposition=disposition,
                        reason_code=reason,
                    )
                )
                index += 1
    return tuple(interfaces), tuple(workloads), tuple(terminals)


def test_publisher_builds_exact_108_valid_bundle_and_deep_reopens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock = _lock()
    completion = _e6_completion(lock)
    interfaces, workloads, terminals = _probe_inputs(lock)
    authority = _source_authority(monkeypatch, tmp_path)
    publication = reduce_e0_compatibility_probes(
        protocol_lock=lock,
        e6_completion=completion,
        interface_receipts=interfaces,
        workload_authorities=workloads,
        probe_terminals=terminals,
        onlinespec_source_authority=authority,
    )
    assert len(publication.compatibility.decisions) == 108
    assert publication.compatibility.valid_count == 108
    assert len(publication.evidence_manifest.probe_terminal_sha256s) == 108
    assert publication.bundle["started_ns"] == 1_000
    assert publication.bundle["finished_ns"] == 2_075
    assert publication.bundle["onlinespec_source_authority_sha256"] == authority.sha256
    payload = dict(publication.bundle)
    bundle_sha256 = payload.pop("bundle_sha256")
    assert bundle_sha256 == content_sha256(payload)
    reopened, reopened_authority, reopened_sha, evidence_sha = (
        _e0_compatibility_from_auxiliary(completion, lock, publication.bundle)
    )
    assert reopened == publication.compatibility
    assert reopened_authority == authority
    assert reopened_sha == bundle_sha256
    assert evidence_sha == publication.evidence_manifest.sha256


def test_publisher_derives_registered_valid_and_na_reasons(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock = _lock()
    interfaces, workloads, terminals = _probe_inputs(
        lock,
        unsupported_interfaces={(E0_MODELS[0], E0_BACKENDS[0])},
        unsupported_workloads={(E0_MODELS[1], E0_TASKS[0])},
        unsupported_smokes={(E0_MODELS[2], E0_BACKENDS[1], E0_TASKS[1])},
    )
    publication = reduce_e0_compatibility_probes(
        protocol_lock=lock,
        e6_completion=_e6_completion(lock),
        interface_receipts=interfaces,
        workload_authorities=workloads,
        probe_terminals=terminals,
        onlinespec_source_authority=_source_authority(monkeypatch, tmp_path),
    )
    assert publication.compatibility.valid_count == 95
    reasons = [
        row.reason_code
        for row in publication.compatibility.decisions
        if row.disposition == "N/A"
    ]
    assert reasons.count("MODEL_BACKEND_INTERFACE_UNSUPPORTED") == 9
    assert reasons.count("TOKENIZER_TASK_WORKLOAD_UNSUPPORTED") == 3
    assert reasons.count("GPU_SMOKE_REGISTERED_UNSUPPORTED") == 1


def test_missing_failed_or_mismatched_probe_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock = _lock()
    completion = _e6_completion(lock)
    interfaces, workloads, terminals = _probe_inputs(lock)
    authority = _source_authority(monkeypatch, tmp_path)
    arguments = {
        "protocol_lock": lock,
        "e6_completion": completion,
        "interface_receipts": interfaces,
        "workload_authorities": workloads,
        "onlinespec_source_authority": authority,
    }
    with pytest.raises(ValueError, match="12/36/108"):
        reduce_e0_compatibility_probes(
            **arguments,
            probe_terminals=terminals[:-1],
        )
    failed = E0CompatibilityProbeTerminal(
        **{
            **asdict(terminals[0]),
            "terminal_status": "FAILED",
            "exit_code": 1,
        }
    )
    with pytest.raises(RuntimeError, match="did not complete"):
        reduce_e0_compatibility_probes(
            **arguments,
            probe_terminals=(failed, *terminals[1:]),
        )
    mismatched = E0CompatibilityProbeTerminal(
        **{
            **asdict(terminals[0]),
            "disposition": "N/A",
            "reason_code": "MODEL_BACKEND_INTERFACE_UNSUPPORTED",
        }
    )
    with pytest.raises(ValueError, match="code-owned decision rule"):
        reduce_e0_compatibility_probes(
            **arguments,
            probe_terminals=(mismatched, *terminals[1:]),
        )


def test_all_na_requires_null_onlinespec_and_publishes_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock = _lock()
    unsupported = {(model, backend) for model in E0_MODELS for backend in E0_BACKENDS}
    interfaces, workloads, terminals = _probe_inputs(
        lock,
        unsupported_interfaces=unsupported,
    )
    kwargs = {
        "protocol_lock": lock,
        "e6_completion": _e6_completion(lock),
        "interface_receipts": interfaces,
        "workload_authorities": workloads,
        "probe_terminals": terminals,
    }
    publication = reduce_e0_compatibility_probes(
        **kwargs,
        onlinespec_source_authority=None,
    )
    assert publication.compatibility.valid_count == 0
    assert publication.bundle["onlinespec_source_authority"] is None
    assert publication.bundle["onlinespec_source_authority_sha256"] is None
    with pytest.raises(ValueError, match="all-N/A"):
        reduce_e0_compatibility_probes(
            **kwargs,
            onlinespec_source_authority=_source_authority(monkeypatch, tmp_path),
        )
    bundle_path = tmp_path / "bundle.json"
    evidence_path = tmp_path / "evidence.json"
    publish_e0_compatibility_probes(
        publication,
        bundle_output_path=bundle_path,
        evidence_manifest_output_path=evidence_path,
    )
    assert json.loads(bundle_path.read_text()) == publication.bundle
    assert json.loads(evidence_path.read_text())["evidence_manifest_sha256"] == (
        publication.evidence_manifest.sha256
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        publish_e0_compatibility_probes(
            publication,
            bundle_output_path=bundle_path,
            evidence_manifest_output_path=evidence_path,
        )


def test_trusted_eagle3_task_row_is_shared_across_methods_but_tokens_are_not(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One task proof row may serve roles, but every launch token is method-bound."""

    from lightcone_spec import config as config_module
    from lightcone_spec.experiments import formal_registry
    from lightcone_spec.experiments import (
        formal_single_operator_e0_compatibility as compatibility_module,
    )
    from lightcone_spec.experiments import formal_single_operator_stages as stages
    from lightcone_spec.runtime import compile_runner, readiness

    def binding(name: str, value: dict[str, object]) -> CanonicalJsonProofBinding:
        path = (tmp_path / f"{name}.json").resolve()
        publish_canonical_json_no_replace(path, value)
        return CanonicalJsonProofBinding.bind(path)

    task = E0_TASKS[0]
    other_task = E0_TASKS[1]
    model = E0_MODELS[0]
    target_revision = "1" * 40
    drafter_revision = "2" * 40
    inventory_sha256 = _sha("eagle3-inventory")
    gpu_uuids = ("GPU-0",)
    claims = {
        "eagle3_e0_execution_authority_sha256": _sha("execution"),
        "eagle3_compatibility_authority_sha256": _sha("compatibility"),
        "eagle3_model_selector_sha256": _sha("selector"),
        "eagle3_native_gpu_proof_sha256": _sha("native"),
    }

    source_binding = binding("execution-source", {"source": "current-E0"})
    auxiliary_binding = binding("compatibility-bundle", {"bundle": "schema2"})
    interface_binding = binding("interface", {"interface": "EAGLE3"})
    terminal_binding = binding("terminal", {"task": task})
    execution_binding = binding(
        "execution-authority",
        {
            "stage": "E0",
            "task": task,
            "target_revision": target_revision,
            "drafter_revision": drafter_revision,
            "interface_sha256": _sha("interface-claim"),
            "inventory_sha256": inventory_sha256,
            "gpu_uuids": list(gpu_uuids),
        },
    )
    compatibility_binding = binding("compatibility-authority", {"proof": "c"})
    selector_binding = binding("selector-authority", {"proof": "s"})
    native_binding = binding("native-receipt", {"proof": "n"})
    launch_tts_binding = binding("launch-tts", {"launch": "tts"})
    launch_l0_binding = binding("launch-l0", {"launch": "l0"})

    def config(method: str) -> RunConfig:
        return RunConfig(
            method=method,  # type: ignore[arg-type]
            model=ModelPair(
                target=model,
                drafter="example/eagle3-drafter",
                target_revision=target_revision,
                drafter_revision=drafter_revision,
                algorithm="EAGLE3",
                draft_depth=3,
            ),
            runtime=RuntimeConfig(
                sampling_profile_sha256=_sha("sampling"),
                device_identity="GPU-0",
                speculative_num_draft_tokens=4,
                speculative_eagle_topk=1,
            ),
            adaptation=AdaptationConfig(
                weight_update_mode="full",
                parameter_scope="all",
                adaptation_group_id=f"{method}-group",
                optimizer=OptimizerConfig(name="adam", learning_rate=1e-5),
                canvas_tokens=4,
                **claims,
            ),
        )

    configs = {"tts-config": config("tts"), "l0-config": config("l0")}
    content_binding = object()
    cells = (
        SimpleNamespace(
            cell_id=_sha("tts-cell"),
            stage="E0",
            model=model,
            backend="EAGLE3",
            task=task,
        ),
        SimpleNamespace(
            cell_id=_sha("l0-cell"),
            stage="E0",
            model=model,
            backend="EAGLE3",
            task=task,
        ),
    )
    source = SimpleNamespace(
        sha256=_sha("current-source"),
        stage="E0",
        materialization_source=SimpleNamespace(reopen=lambda: {"cells": "current"}),
        content_source_binding=content_binding,
        auxiliary_source_binding=lambda kind: (
            SimpleNamespace(reopen=lambda **_kwargs: auxiliary_binding.reopen())
            if kind == "e0_compatibility"
            else (_ for _ in ()).throw(KeyError(kind))
        ),
    )
    proof_row = SimpleNamespace(
        task=task,
        sha256=_sha("task-proof-row"),
        execution_authority=execution_binding,
        compatibility_authority=compatibility_binding,
        model_selector_authority=selector_binding,
        native_gpu_proof=native_binding,
    )
    interface = SimpleNamespace(
        schema_version=2,
        support_status="READY",
        sha256=_sha("interface-receipt"),
        eagle3_runtime_proof_rows=(proof_row,),
    )
    terminal = SimpleNamespace(
        schema_version=2,
        disposition="VALID",
        interface_receipt_sha256=interface.sha256,
        eagle3_runtime_proof_row_sha256=proof_row.sha256,
    )
    publication = SimpleNamespace(
        evidence_manifest=SimpleNamespace(
            interface_receipts=(interface_binding,),
            probe_terminals=(terminal_binding,),
        ),
        compatibility=SimpleNamespace(
            decisions=(
                SimpleNamespace(
                    model=model,
                    backend="EAGLE3",
                    task=task,
                    disposition="VALID",
                    interface_sha256=_sha("interface-claim"),
                ),
            )
        ),
    )
    launches = {
        launch_tts_binding.absolute_path: SimpleNamespace(
            sha256=launch_tts_binding.semantic_sha256,
            run_config_path="tts-config",
            formal_stage="E0",
            inventory_sha256=inventory_sha256,
            gpu_uuids=gpu_uuids,
            content_source_binding=content_binding,
        ),
        launch_l0_binding.absolute_path: SimpleNamespace(
            sha256=launch_l0_binding.semantic_sha256,
            run_config_path="l0-config",
            formal_stage="E0",
            inventory_sha256=inventory_sha256,
            gpu_uuids=gpu_uuids,
            content_source_binding=content_binding,
        ),
    }
    native = SimpleNamespace(
        sha256=claims["eagle3_native_gpu_proof_sha256"],
        source_identity_sha256=_sha("native-source"),
        inventory_sha256=inventory_sha256,
        gpu_uuids=gpu_uuids,
    )

    monkeypatch.setattr(
        stages, "load_formal_single_operator_execution_source", lambda _path: source
    )
    monkeypatch.setattr(
        formal_registry,
        "stage_materialization_receipt_from_dict",
        lambda _raw: SimpleNamespace(cells=cells),
    )
    monkeypatch.setattr(
        compatibility_module,
        "revalidate_trusted_e0_compatibility_bundle_value",
        lambda _raw: publication,
    )
    monkeypatch.setattr(
        compatibility_module,
        "load_e0_prepared_model_backend_interface_receipt",
        lambda _path: interface,
    )
    monkeypatch.setattr(
        compatibility_module,
        "load_e0_compatibility_probe_terminal",
        lambda _path: terminal,
    )
    monkeypatch.setattr(
        compatibility_module,
        "e0_eagle3_runtime_authority_for_task",
        lambda receipt, *, task: (
            claims
            if receipt is interface and task == proof_row.task
            else (_ for _ in ()).throw(ValueError("wrong task"))
        ),
    )
    monkeypatch.setattr(
        compile_runner.CompileLaunchManifest,
        "load",
        classmethod(lambda _cls, path: launches[str(path)]),
    )
    monkeypatch.setattr(config_module, "load_run_config", lambda path: configs[path])
    monkeypatch.setattr(
        readiness.NativeRuntimeGpuProofReceipt,
        "from_dict",
        classmethod(lambda _cls, _raw: native),
    )

    def token(
        *, cell_id: str, launch: CanonicalJsonProofBinding, method: str
    ) -> TrustedSingleOperatorEagle3ExecutionAuthority:
        return TrustedSingleOperatorEagle3ExecutionAuthority(
            schema_version=1,
            kind="trusted_single_operator_eagle3_execution_authority",
            trust_mode="trusted_single_operator_empirical_no_signature",
            formal_measured_authorization=False,
            execution_source=source_binding,
            execution_source_sha256=source.sha256,
            materialized_cell_id=cell_id,
            compile_launch_manifest=launch,
            interface_receipt=interface_binding,
            compatibility_terminal=terminal_binding,
            execution_authority=execution_binding,
            compatibility_authority=compatibility_binding,
            model_selector_authority=selector_binding,
            native_gpu_receipt=native_binding,
            proof_row_sha256=proof_row.sha256,
            model=model,
            backend="EAGLE3",
            task=task,
            method=method,
            target_revision=target_revision,
            drafter_revision=drafter_revision,
            inventory_sha256=inventory_sha256,
            gpu_uuids=gpu_uuids,
            native_source_identity_sha256=native.source_identity_sha256,
            **claims,
        )

    tts_token = token(cell_id=cells[0].cell_id, launch=launch_tts_binding, method="tts")
    l0_token = token(cell_id=cells[1].cell_id, launch=launch_l0_binding, method="l0")
    assert tts_token.proof_row_sha256 == l0_token.proof_row_sha256
    assert _eagle3_execution_authority(
        config=configs["tts-config"],
        verified_authority=None,
        expected_source_identity_sha256=None,
        inventory_sha256=inventory_sha256,
        gpu_uuids=gpu_uuids,
        trusted_single_operator_authority=tts_token,
    ) == (*claims.values(), native.source_identity_sha256)

    with pytest.raises(ValueError, match="current-source lineage"):
        replace(tts_token, task=other_task)
    with pytest.raises(ValueError, match="empirical replay"):
        replace(tts_token, method="l0")
    with pytest.raises(ValueError, match="live launch"):
        _eagle3_execution_authority(
            config=configs["tts-config"],
            verified_authority=None,
            expected_source_identity_sha256=None,
            inventory_sha256=inventory_sha256,
            gpu_uuids=gpu_uuids,
            trusted_single_operator_authority=l0_token,
        )

    original = configs["tts-config"]
    assert original.adaptation is not None
    configs["tts-config"] = original.model_copy(
        update={
            "adaptation": original.adaptation.model_copy(
                update={"eagle3_model_selector_sha256": _sha("mutated-selector")}
            )
        }
    )
    with pytest.raises(ValueError, match="empirical replay"):
        tts_token.__post_init__()


def test_schema3_preprobe_interface_is_executable_but_has_no_task_proofs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lightcone_spec import config as config_module
    from lightcone_spec.runtime import compile_cache, compile_runner

    def binding(name: str, value: object) -> CanonicalJsonProofBinding:
        path = (tmp_path / f"{name}.json").resolve()
        publish_canonical_json_no_replace(path, value)
        return CanonicalJsonProofBinding.bind(path)

    model = E0_MODELS[0]
    backend = "EAGLE3"
    target_revision = "1" * 40
    drafter_revision = "2" * 40
    tokenizer_revision = "3" * 40
    launch_binding = binding("launch", {"launch": "static"})
    members = {
        "target": _sha("target-member"),
        "drafter": _sha("drafter-member"),
        "tokenizer": _sha("tokenizer-member"),
    }
    evidence = binding(
        "preprobe-evidence",
        {
            "schema_version": 1,
            "kind": "formal_single_operator_e0_preprobe_interface_evidence",
            "protocol_lock_sha256": _sha("lock"),
            "upstream_e6_confirmation_sha256": _sha("e6-confirmation"),
            "model": model,
            "backend": backend,
            "target_member_sha256": members["target"],
            "drafter_member_sha256": members["drafter"],
            "tokenizer_member_sha256": members["tokenizer"],
            "compile_launch_manifest_sha256": launch_binding.semantic_sha256,
        },
    )
    interface_sha = e0_preprobe_interface_sha256(
        protocol_lock_sha256=_sha("lock"),
        upstream_e6_confirmation_sha256=_sha("e6-confirmation"),
        model=model,
        backend=backend,
        target_model_id=model,
        target_revision=target_revision,
        drafter_model_id="example/eagle3-drafter",
        drafter_revision=drafter_revision,
        tokenizer_model_id=model,
        tokenizer_revision=tokenizer_revision,
        target_member_sha256=members["target"],
        drafter_member_sha256=members["drafter"],
        tokenizer_member_sha256=members["tokenizer"],
        compile_launch_manifest_sha256=launch_binding.semantic_sha256,
        preprobe_evidence_sha256=evidence.semantic_sha256,
    )
    launch = SimpleNamespace(
        sha256=launch_binding.semantic_sha256,
        schema_version=2,
        formal_stage="E0",
        run_config_path="config.json",
        compile_cache_plan_path="cache.json",
        target_model_id=model,
        target_revision=target_revision,
        drafter_model_id="example/eagle3-drafter",
        drafter_revision=drafter_revision,
        tokenizer_model_id=model,
        tokenizer_revision=tokenizer_revision,
        target_content_member_id=members["target"],
        drafter_content_member_id=members["drafter"],
        tokenizer_content_member_id=members["tokenizer"],
        prepared_model_content_manifest_sha256=_sha("content"),
        gpu_uuids=("GPU-0",),
    )
    config = SimpleNamespace(
        model=SimpleNamespace(
            algorithm=backend,
            target=model,
            target_revision=target_revision,
            drafter="example/eagle3-drafter",
            drafter_revision=drafter_revision,
        ),
        method="static",
        adaptation=None,
        online_spec=None,
        runtime=SimpleNamespace(topology_mode="tp1_dp1"),
    )
    monkeypatch.setattr(
        compile_runner.CompileLaunchManifest,
        "load",
        classmethod(lambda _cls, _path: launch),
    )
    monkeypatch.setattr(config_module, "load_run_config", lambda _path: config)
    monkeypatch.setattr(
        compile_cache.CompileCacheLaunchPlan,
        "load",
        classmethod(lambda _cls, _path: object()),
    )
    monkeypatch.setattr(
        compile_cache,
        "validate_compile_key_for_run_config",
        lambda *_args, **_kwargs: None,
    )

    receipt = E0PreparedModelBackendInterfaceReceipt(
        schema_version=3,
        protocol_lock_sha256=_sha("lock"),
        upstream_e6_confirmation_sha256=_sha("e6-confirmation"),
        model=model,
        backend=backend,
        tokenizer_sha256=members["tokenizer"],
        interface_sha256=interface_sha,
        prepared_model_manifest_sha256=_sha("content"),
        support_status="READY",
        reason_code="INTERFACE_READY",
        requires_gpu_smoke=True,
        evidence_sha256=evidence.semantic_sha256,
        target_model_id=model,
        target_revision=target_revision,
        drafter_model_id="example/eagle3-drafter",
        drafter_revision=drafter_revision,
        tokenizer_model_id=model,
        tokenizer_revision=tokenizer_revision,
        target_member_sha256=members["target"],
        drafter_member_sha256=members["drafter"],
        tokenizer_member_sha256=members["tokenizer"],
        compile_launch_manifest=launch_binding,
        eagle3_runtime_proof_rows=(),
        preprobe_evidence=evidence,
    )

    assert receipt.schema_version == 3
    assert receipt.eagle3_runtime_proof_rows == ()
