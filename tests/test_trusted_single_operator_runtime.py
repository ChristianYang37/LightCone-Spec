from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.sglang_spawn_runtime_authority import (
    VerifiedSglangSpawnRuntimeAuthorityBridge,
    revalidate_sglang_spawn_runtime_authority_environment,
)
from lightcone_spec.runtime.trusted_single_operator_runtime import (
    TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_ENVIRONMENT,
    TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_PROTOCOL_SHA256,
    TrustedSingleOperatorRuntimeAuthoritySource,
    TrustedSingleOperatorRuntimeRoleSource,
    VerifiedTrustedSingleOperatorRuntimeGpuAuthority,
    _issue_verified_trusted_single_operator_runtime_gpu_authority,
    _preflight_role_sources,
    bind_trusted_single_operator_runtime_authority_environment,
    trusted_single_operator_runtime_authority_environment,
)


def _binding(tmp_path, name: str) -> CanonicalJsonProofBinding:
    path = tmp_path / name
    publish_canonical_json_no_replace(path, {"name": name})
    return CanonicalJsonProofBinding.bind(path)


def _issue(
    *,
    role: str,
    topology_mode: str,
    backend_capabilities: tuple[str, ...],
) -> VerifiedTrustedSingleOperatorRuntimeGpuAuthority:
    return _issue_verified_trusted_single_operator_runtime_gpu_authority(
        role=role,  # type: ignore[arg-type]
        authority_kind="preflight_qualification",
        source_suite_id="dspark_tp2",
        authority_source_sha256="1" * 64,
        consumer_identity_sha256="2" * 64,
        evidence_sha256s=("4" * 64, "3" * 64),
        source_capability_sha256="5" * 64,
        role_source_identity_sha256="6" * 64,
        source_identity_sha256="7" * 64,
        inventory_sha256="8" * 64,
        hardware_envelope_sha256="9" * 64,
        topology_mode=topology_mode,  # type: ignore[arg-type]
        topology_sha256="a" * 64,
        gpu_uuids=("GPU-a", "GPU-b"),
        backend_capabilities=backend_capabilities,
    )


def test_verified_trusted_runtime_authority_is_sealed_and_explicitly_unmeasured() -> (
    None
):
    token = _issue(
        role="native",
        topology_mode="tp2_dp1",
        backend_capabilities=("native_itl", "dspark", "native_itl"),
    )

    assert token.protocol_sha256 == (
        TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_PROTOCOL_SHA256
    )
    assert token.trust_mode == "trusted_single_operator_empirical_no_signature"
    assert token.formal_measurement is False
    assert token.qualification_only is False
    assert token.evidence_sha256s == ("3" * 64, "4" * 64)
    assert token.backend_capabilities == ("dspark", "native_itl")
    assert len(token.sha256) == 64

    with pytest.raises(TypeError, match="source revalidation"):
        VerifiedTrustedSingleOperatorRuntimeGpuAuthority(
            role="native",
            authority_kind="preflight_qualification",
            source_suite_id="dspark_tp2",
            authority_source_sha256="1" * 64,
            consumer_identity_sha256="2" * 64,
            evidence_sha256s=("3" * 64,),
            receipt_sha256="4" * 64,
            source_capability_sha256="5" * 64,
            role_source_identity_sha256="6" * 64,
            source_identity_sha256="7" * 64,
            inventory_sha256="8" * 64,
            hardware_envelope_sha256="9" * 64,
            topology_mode="tp2_dp1",
            topology_sha256="a" * 64,
            gpu_uuids=("GPU-a", "GPU-b"),
            backend_capabilities=("dspark",),
            _verification_tag=object(),
        )


def test_trusted_runtime_role_tokens_share_common_source_but_not_receipt() -> None:
    native = _issue(
        role="native",
        topology_mode="tp2_dp1",
        backend_capabilities=("dspark",),
    )
    distributed = _issue(
        role="distributed",
        topology_mode="tp2_dp1",
        backend_capabilities=(),
    )

    assert native.source_identity_sha256 == distributed.source_identity_sha256
    assert native.consumer_identity_sha256 == distributed.consumer_identity_sha256
    assert native.receipt_sha256 != distributed.receipt_sha256


def test_trusted_runtime_distributed_role_rejects_single_rank_or_backend_caps() -> None:
    with pytest.raises(ValueError, match="role capabilities"):
        _issue(
            role="distributed",
            topology_mode="tp2_dp1",
            backend_capabilities=("dspark",),
        )


def test_spawn_runtime_bridge_is_sealed_and_replays_the_trusted_source(
    tmp_path,
    monkeypatch,
) -> None:
    import lightcone_spec.config as config_module
    import lightcone_spec.runtime.sglang_spawn_runtime_authority as spawn_bridge
    import lightcone_spec.runtime.trusted_single_operator_runtime as runtime_authority
    from lightcone_spec.runtime import compile_runner

    binding = _binding(tmp_path, "spawn-source.json")
    distributed = _issue(
        role="distributed",
        topology_mode="tp2_dp1",
        backend_capabilities=(),
    )
    native = _issue(
        role="native",
        topology_mode="tp2_dp1",
        backend_capabilities=("dspark",),
    )
    source = SimpleNamespace(
        roles=(
            SimpleNamespace(role="distributed"),
            SimpleNamespace(role="native"),
        ),
        topology_mode="tp2_dp1",
        sha256="a" * 64,
        launch_manifest=binding,
    )
    monkeypatch.delenv("LIGHTCONE_NATIVE_QUALIFICATION_MODE", raising=False)
    monkeypatch.setattr(
        runtime_authority,
        "bind_trusted_single_operator_runtime_authority_environment",
        lambda _environment: binding,
    )
    monkeypatch.setattr(
        runtime_authority,
        "verify_trusted_single_operator_runtime_authority_source",
        lambda _path, *, expected_source_binding: (source, (distributed, native)),
    )
    monkeypatch.setattr(
        compile_runner.CompileLaunchManifest,
        "load",
        classmethod(
            lambda _cls, _path: SimpleNamespace(run_config_path="/run-config.json")
        ),
    )
    monkeypatch.setattr(config_module, "load_run_config", lambda _path: object())
    monkeypatch.setattr(
        spawn_bridge,
        "_revalidate_spawn_adaptation_payload",
        lambda **_kwargs: None,
    )

    bridge = revalidate_sglang_spawn_runtime_authority_environment()
    assert type(bridge) is VerifiedSglangSpawnRuntimeAuthorityBridge
    assert bridge.proofs_by_role == (
        ("distributed", distributed),
        ("native", native),
    )
    assert bridge.rank_publication_mode == "formal_nccl_v1"

    with pytest.raises(TypeError, match="source revalidation"):
        VerifiedSglangSpawnRuntimeAuthorityBridge(
            source_environment_sha256="b" * 64,
            proofs_by_role=(("native", native),),
            rank_publication_mode="none",
            _verification_tag=object(),
        )
    with pytest.raises(ValueError, match="role capabilities"):
        _issue_verified_trusted_single_operator_runtime_gpu_authority(
            role="distributed",
            authority_kind="preflight_qualification",
            source_suite_id="dspark_tp1",
            authority_source_sha256="1" * 64,
            consumer_identity_sha256="2" * 64,
            evidence_sha256s=("3" * 64,),
            source_capability_sha256="4" * 64,
            role_source_identity_sha256="5" * 64,
            source_identity_sha256="6" * 64,
            inventory_sha256="7" * 64,
            hardware_envelope_sha256="8" * 64,
            topology_mode="tp1_dp1",
            topology_sha256="9" * 64,
            gpu_uuids=("GPU-a",),
            backend_capabilities=(),
        )


def test_trusted_runtime_authority_source_codec_is_exact_and_path_bound(
    tmp_path,
) -> None:
    bindings = {
        name: _binding(tmp_path, f"{name}.json")
        for name in (
            "consumer",
            "execution",
            "launch",
            "preflight",
            "evidence",
        )
    }
    value = TrustedSingleOperatorRuntimeAuthoritySource(
        schema_version=1,
        kind="trusted_single_operator_runtime_authority_source",
        protocol_sha256=TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_PROTOCOL_SHA256,
        trust_mode="trusted_single_operator_empirical_no_signature",
        formal_measurement=False,
        authority_kind="preflight_qualification",
        algorithm="DSPARK",
        consumer_source=bindings["consumer"],
        execution_source=bindings["execution"],
        materialized_cell_id="1" * 64,
        launch_manifest=bindings["launch"],
        preflight_inputs=bindings["preflight"],
        authority_evidence=bindings["evidence"],
        authority_sha256="2" * 64,
        consumer_identity_sha256="3" * 64,
        source_identity_sha256="4" * 64,
        inventory_sha256="5" * 64,
        hardware_envelope_sha256="6" * 64,
        topology_mode="tp1_dp1",
        topology_sha256="7" * 64,
        gpu_uuids=("GPU-a",),
        roles=(
            TrustedSingleOperatorRuntimeRoleSource(
                role="native",
                source_suite_id="dspark_tp1",
                source_capability_sha256="8" * 64,
                role_source_identity_sha256="9" * 64,
                evidence_sha256s=("a" * 64,),
                backend_capabilities=("dspark",),
            ),
        ),
    )

    assert (
        TrustedSingleOperatorRuntimeAuthoritySource.from_dict(value.to_dict()) == value
    )
    assert len(value.sha256) == 64

    source_path = tmp_path / "runtime-authority.json"
    publish_canonical_json_no_replace(source_path, value.to_dict())
    source_binding = CanonicalJsonProofBinding.bind(source_path)
    environment = trusted_single_operator_runtime_authority_environment(source_binding)
    assert (
        bind_trusted_single_operator_runtime_authority_environment(environment)
        == source_binding
    )
    partial = dict(environment)
    partial.pop(TRUSTED_SINGLE_OPERATOR_RUNTIME_AUTHORITY_ENVIRONMENT[-1])
    with pytest.raises(ValueError, match="incomplete"):
        bind_trusted_single_operator_runtime_authority_environment(partial)

    Path(bindings["consumer"].absolute_path).write_text(
        '{"name":"changed"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="consumer source changed"):
        TrustedSingleOperatorRuntimeAuthoritySource.from_dict(value.to_dict())


def test_trusted_runtime_role_source_codec_rejects_missing_identity() -> None:
    with pytest.raises(ValueError, match="fields differ"):
        TrustedSingleOperatorRuntimeRoleSource.from_dict(
            {
                "role": "native",
                "source_suite_id": "dspark_tp2",
                "source_capability_sha256": "1" * 64,
                "evidence_sha256s": ["3" * 64],
                "backend_capabilities": ["dspark"],
            }
        )


def test_dspark_tp2_runtime_source_joins_distributed_and_native_suites(
    tmp_path,
    monkeypatch,
) -> None:
    import lightcone_spec.experiments.formal_single_operator_preflight_qualification as qualification
    import lightcone_spec.runtime.native_qualification_runner as native_runner
    from lightcone_spec.experiments.formal_content_source import (
        FormalContentSourceBinding,
    )
    from lightcone_spec.experiments.formal_preflight_inputs import (
        FormalSingleOperatorPreflightAuthority,
    )

    index_binding = _binding(tmp_path, "qualification-index.json")
    execution_authority = _binding(tmp_path, "preflight-authority.json")
    protocol_lock = _binding(tmp_path, "protocol-lock.json")
    inventory = _binding(tmp_path, "inventory.json")
    doctor = _binding(tmp_path, "doctor.json")
    exactness = _binding(tmp_path, "exactness.json")
    content_source = object.__new__(FormalContentSourceBinding)
    object.__setattr__(content_source, "schema_version", 1)
    object.__setattr__(content_source, "kind", "formal_content_source_binding")
    object.__setattr__(content_source, "mode", "trusted_single_operator")
    object.__setattr__(content_source, "offline_root_signed", None)
    object.__setattr__(content_source, "trusted_single_operator", object())
    suites = (
        "chronobelief_gpu_parity",
        "dspark_dp2",
        "dspark_tp1",
        "dspark_tp2",
        "tp1_dp2",
        "tp2_dp1",
    )
    plan_bindings = {
        suite: _binding(tmp_path, f"{suite}-plan.json") for suite in suites
    }
    result_bindings = {
        suite: _binding(tmp_path, f"{suite}-result.json") for suite in plan_bindings
    }
    assignment_bindings = {
        suite: _binding(tmp_path, f"{suite}-assignment.json") for suite in plan_bindings
    }
    evidence = {
        name: _binding(tmp_path, f"{name}.json")
        for name in ("proof", "terminal", "observation", "native")
    }
    suite_topology = {
        "chronobelief_gpu_parity": "tp1_dp1",
        "dspark_tp1": "tp1_dp1",
        "dspark_tp2": "tp2_dp1",
        "tp2_dp1": "tp2_dp1",
        "dspark_dp2": "tp1_dp2",
        "tp1_dp2": "tp1_dp2",
    }
    plans = {
        suite: SimpleNamespace(
            suite_id=suite,
            protocol_lock=protocol_lock,
            content_source=content_source,
            inventory=inventory,
            doctor=doctor,
            exactness_assignment=exactness,
            result_path=result_bindings[suite].absolute_path,
            topology_mode=suite_topology[suite],
            topology_sha256=(
                "a" * 64
                if suite_topology[suite] == "tp2_dp1"
                else plan_bindings[suite].semantic_sha256
            ),
            gpu_uuids=(
                ("GPU-a",) if suite_topology[suite] == "tp1_dp1" else ("GPU-a", "GPU-b")
            ),
            sha256=plan_bindings[suite].semantic_sha256,
        )
        for suite in plan_bindings
    }
    results = {
        suite: SimpleNamespace(
            plan=plan_bindings[suite],
            assignment=assignment_bindings[suite],
            empirical_proof=evidence["proof"],
            runner_terminal=evidence["terminal"],
            live_observation=evidence["observation"],
            live_native_terminal=evidence["native"],
            junit_xml=SimpleNamespace(raw_sha256="b" * 64),
            status="COMPLETE",
            sha256=result_bindings[suite].semantic_sha256,
        )
        for suite in plan_bindings
    }
    assignments = {
        suite: SimpleNamespace(
            schema_version=2,
            suite_id=suite,
            inventory_sha256="e" * 64,
            gpu_uuids=plans[suite].gpu_uuids,
            topology_sha256=plans[suite].topology_sha256,
            hardware_envelope_sha256=(
                "f" * 64
                if suite_topology[suite] == "tp2_dp1"
                else assignment_bindings[suite].semantic_sha256
            ),
            source_identity_sha256=assignment_bindings[suite].semantic_sha256,
        )
        for suite in plan_bindings
    }
    monkeypatch.setattr(
        qualification,
        "load_formal_single_operator_preflight_qualification_plan_index",
        lambda _path: SimpleNamespace(
            plans=tuple(plan_bindings.values()),
            sha256=index_binding.semantic_sha256,
        ),
    )
    monkeypatch.setattr(
        qualification,
        "load_formal_single_operator_preflight_qualification_plan",
        lambda path: plans[Path(path).name.removesuffix("-plan.json")],
    )
    monkeypatch.setattr(
        qualification,
        "revalidate_formal_single_operator_preflight_qualification_result",
        lambda path: results[Path(path).name.removesuffix("-result.json")],
    )
    monkeypatch.setattr(
        native_runner.NativeRuntimeQualificationAssignment,
        "load",
        lambda path: assignments[Path(path).name.removesuffix("-assignment.json")],
    )
    monkeypatch.setattr(
        FormalSingleOperatorPreflightAuthority,
        "from_dict",
        classmethod(
            lambda _cls, _value: SimpleNamespace(
                sha256=execution_authority.semantic_sha256,
                protocol_lock=protocol_lock,
                inventory=inventory,
            )
        ),
    )

    authority, _sha256, hardware, topology, roles = _preflight_role_sources(
        algorithm="DSPARK",
        topology_mode="tp2_dp1",
        preflight_inputs=SimpleNamespace(
            qualification_plan_index=index_binding,
            execution_authority=execution_authority,
            content_source_binding=content_source,
            inventory=inventory,
            doctor_report=doctor,
            exactness_assignment=exactness,
        ),
        inventory_sha256="e" * 64,
        gpu_uuids=("GPU-a", "GPU-b"),
    )

    assert authority == index_binding
    assert hardware == "f" * 64
    assert topology == "a" * 64
    assert tuple((row.role, row.source_suite_id) for row in roles) == (
        ("distributed", "tp2_dp1"),
        ("native", "dspark_tp2"),
    )
    assert roles[0].role_source_identity_sha256 != roles[1].role_source_identity_sha256

    plans["tp2_dp1"].doctor = _binding(tmp_path, "foreign-doctor.json")
    with pytest.raises(ValueError, match="plan lineage differs"):
        _preflight_role_sources(
            algorithm="DSPARK",
            topology_mode="tp2_dp1",
            preflight_inputs=SimpleNamespace(
                qualification_plan_index=index_binding,
                execution_authority=execution_authority,
                content_source_binding=content_source,
                inventory=inventory,
                doctor_report=doctor,
                exactness_assignment=exactness,
            ),
            inventory_sha256="e" * 64,
            gpu_uuids=("GPU-a", "GPU-b"),
        )
