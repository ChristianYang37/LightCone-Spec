from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import lightcone_spec.runtime.release_trust_root as root_module
from lightcone_spec.experiments.failure_actuator import (
    FAILURE_ACTUATION_EXTERNAL_CONTROL_PROTOCOL_SHA256,
    FAILURE_ACTUATOR_PROTOCOL_SHA256,
    FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON,
    FAILURE_SCENARIO_SEMANTICS,
    FAILURE_SCENARIO_SEMANTICS_SHA256,
    RELEASE_FAILURE_ACTUATOR_CAPABILITIES,
    FailureActuatorBlocked,
    FailureActuatorContext,
    FailureActuatorLaunchBinding,
    FailurePhaseObservation,
    FailureRankLaunchBinding,
    FailureScenarioSemantics,
    FailureTerminalObservation,
    ReleaseFailureActuatorCapability,
    SglangHttpFailureNativeControlTransport,
    build_failure_actuation_external_control_binding,
    execute_failure_actuator_for_cpu_test,
    execute_release_failure_actuator_unsigned,
    failure_phase_observation_sha256,
    failure_semantics,
    linux_process_start_monotonic_ns,
    publish_failure_actuation_proof_artifact,
    release_failure_actuator_capability,
    require_release_failure_actuator,
    validate_failure_actuation_proof_artifact,
    validate_failure_recovery_receipt,
)
from lightcone_spec.experiments.failure_authority import (
    FailureInjectionAuthorityResult,
    ReleaseFailurePlan,
    bind_failure_injection_authority,
    release_failure_plan_for_cell,
    require_failure_injection_authority,
    revalidate_failure_injection_authority,
)
from lightcone_spec.experiments.registry import (
    E5_FAILURES,
    ExperimentRegistry,
    content_sha256,
)
from lightcone_spec.experiments.registry import (
    build_legacy_industrial_registry as build_industrial_registry,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
    attestation_message,
)
from lightcone_spec.runtime.attester_bundle import (
    AttestationNoncePolicy,
    TrustedAttesterPolicyBundle,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayStore,
    ControlArtifactAttestation,
    ControlArtifactSubject,
)
from lightcone_spec.runtime.release_trust_root import (
    DeploymentPolicyAuthorization,
    SourceReleaseEd25519Root,
    SourceReleaseRootBinding,
    deployment_policy_subject_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(f"failure-actuator:{label}".encode()).hexdigest()


_NOW_NS = 2_000_000_000
_HARDWARE_SHA256 = _sha("hardware")


def _public_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _control_authority(
    monkeypatch: pytest.MonkeyPatch,
    *,
    inventory_sha256: str,
):
    root_private = Ed25519PrivateKey.generate()
    artifact_private = Ed25519PrivateKey.generate()
    root_public = _public_bytes(root_private)
    root_spki = root_private.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    root = SourceReleaseEd25519Root(
        schema_version=1,
        kind="lightcone_source_release_ed25519_root",
        root_id="lightcone-release-root-2026q3",
        key_id="lightcone-release-root-key-2026q3",
        algorithm="Ed25519",
        public_key_base64=base64.b64encode(root_public).decode("ascii"),
        public_key_sha256=hashlib.sha256(root_public).hexdigest(),
        spki_sha256=hashlib.sha256(root_spki).hexdigest(),
    )
    root_binding = SourceReleaseRootBinding(
        root=root,
        path="/validation/failure-release-root.json",
        sidecar_path="/validation/failure-release-root.json.sha256",
        semantic_sha256=root.sha256,
        file_sha256=_sha("root-file"),
        sidecar_file_sha256=_sha("root-sidecar"),
    )
    artifact_public = _public_bytes(artifact_private)
    fingerprint = hashlib.sha256(artifact_public).hexdigest()
    bundle = TrustedAttesterPolicyBundle(
        schema_version=1,
        kind="lightcone_trusted_attester_policy_bundle",
        bundle_id="failure-control-bundle-v1",
        valid_from_ns=1,
        expires_ns=10_000_000_000,
        nonce_policy=AttestationNoncePolicy(
            schema_version=1,
            kind="lightcone_attestation_nonce_policy",
            nonce_bytes=32,
            minimum_lifetime_ns=100_000_000,
            maximum_lifetime_ns=2_000_000_000,
            maximum_clock_skew_ns=100_000_000,
            replay_policy="external_single_use_store",
            subject_binding_required=True,
        ),
        hardware_envelope_sha256_allowlist=(_HARDWARE_SHA256,),
        trusted_attester_policy=TrustedAttesterPolicy(
            policy_id="failure-control-policy-v1",
            trusted_attesters=(
                ("failure-coordinator", "failure-coordinator-key", fingerprint),
            ),
            public_keys=(
                (fingerprint, base64.b64encode(artifact_public).decode("ascii")),
            ),
        ),
    )
    deployment_subject = deployment_policy_subject_sha256(
        root_manifest_sha256=root_binding.semantic_sha256,
        inventory_sha256=inventory_sha256,
        bundle_sha256=bundle.sha256,
    )
    deployment_challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id="failure-deployment-policy-1",
        nonce_base64=base64.b64encode(b"d" * 32).decode("ascii"),
        subject_sha256=deployment_subject,
        issued_ns=1_500_000_000,
        expires_ns=3_000_000_000,
    )
    authorization = DeploymentPolicyAuthorization(
        schema_version=1,
        kind="lightcone_deployment_policy_authorization",
        root_manifest_sha256=root_binding.semantic_sha256,
        inventory_sha256=inventory_sha256,
        bundle=bundle,
        challenge=deployment_challenge,
        signature_base64=base64.b64encode(
            root_private.sign(
                attestation_message(
                    deployment_challenge,
                    payload_sha256=bundle.sha256,
                )
            )
        ).decode("ascii"),
    )
    monkeypatch.setattr(
        root_module,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    return artifact_private, root_binding, bundle, authorization


def _control_envelope(
    *,
    private_key: Ed25519PrivateKey,
    root_binding: SourceReleaseRootBinding,
    bundle: TrustedAttesterPolicyBundle,
    authorization: DeploymentPolicyAuthorization,
    binding_sha256: str,
    registry_sha256: str,
    lineage_sha256: str,
) -> ControlArtifactAttestation:
    subject = ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="non_serving_terminal",
        artifact_sha256=binding_sha256,
        protocol_sha256=FAILURE_ACTUATION_EXTERNAL_CONTROL_PROTOCOL_SHA256,
        registry_sha256=registry_sha256,
        lineage_sha256=lineage_sha256,
    )
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id="failure-terminal-control-1",
        nonce_base64=base64.b64encode(b"f" * 32).decode("ascii"),
        subject_sha256=subject.sha256,
        issued_ns=1_600_000_000,
        expires_ns=2_600_000_000,
    )
    public = _public_bytes(private_key)
    signature = private_key.sign(
        attestation_message(challenge, payload_sha256=binding_sha256)
    )
    return ControlArtifactAttestation(
        schema_version=1,
        kind="lightcone_control_artifact_attestation",
        subject=subject,
        hardware_envelope_sha256=_HARDWARE_SHA256,
        trust_anchor_sha256=root_binding.sha256,
        trust_bundle_sha256=bundle.sha256,
        trusted_attester_policy_sha256=bundle.trusted_attester_policy.sha256,
        deployment_policy_authorization=authorization,
        challenge=challenge,
        attestation=SignedAttestation(
            schema_version=1,
            kind="lightcone_signed_attestation",
            algorithm="Ed25519",
            attester_id="failure-coordinator",
            key_id="failure-coordinator-key",
            environment="release",
            public_key_base64=base64.b64encode(public).decode("ascii"),
            challenge_sha256=challenge.sha256,
            payload_sha256=binding_sha256,
            signature_base64=base64.b64encode(signature).decode("ascii"),
        ),
    )


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return build_industrial_registry()


def _authority(
    tmp_path: Path,
    registry: ExperimentRegistry,
    scenario: str,
) -> FailureInjectionAuthorityResult:
    cell = next(
        value
        for value in registry.cells_for("E5")
        if value.identity.task == "failure_injection"
        and value.identity.arrival == f"failure:{scenario}"
    )
    plan = release_failure_plan_for_cell(registry, cell)
    path = tmp_path / f"plan-{scenario}.json"
    path.write_text(
        json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    binding = bind_failure_injection_authority(path.resolve(), registry=registry)
    return revalidate_failure_injection_authority(binding, registry=registry)


def _token_cell_authority(
    tmp_path: Path,
    registry: ExperimentRegistry,
    scenario: str,
):
    authority = _authority(tmp_path, registry, scenario)
    cell = next(
        row for row in registry.cells_for("E5") if row.cell_id == authority.plan.cell_id
    )
    token = require_failure_injection_authority(
        authority.binding,
        registry=registry,
    )
    return token, cell, authority


def _launch(
    tmp_path: Path,
    *,
    authority: FailureInjectionAuthorityResult,
    registry: ExperimentRegistry,
) -> FailureActuatorLaunchBinding:
    rows = []
    for index, (topology, rank) in enumerate(
        (
            ("tp1_dp1", 0),
            ("tp2_dp1", 0),
            ("tp2_dp1", 1),
            ("tp1_dp2", 0),
            ("tp1_dp2", 1),
        )
    ):
        quota = (tmp_path / f"quota-{topology}-{rank}").resolve()
        quota.mkdir(mode=0o700)
        rows.append(
            FailureRankLaunchBinding(
                topology=topology,
                rank=rank,
                process_id=4_000 + index,
                process_group_id=5_000 + index,
                process_start_monotonic_ns=1,
                gpu_uuid=f"GPU-failure-{index}",
                control_url=(
                    f"http://127.0.0.1:{18000 + index}"
                    "/v1/lightcone-spec/failure-actuator"
                ),
                temp_quota_root=str(quota),
            )
        )
    return FailureActuatorLaunchBinding(
        schema_version=1,
        kind="e5_failure_actuator_launch_binding",
        assignment_sha256=_sha("assignment"),
        inventory_sha256=_sha("inventory"),
        registry_sha256=registry.sha256,
        plan_sha256=authority.plan.sha256,
        run_nonce_sha256=_sha(f"release-nonce:{authority.plan.scenario}"),
        ranks=tuple(rows),
    )


class _FakeNativeFailureTransport:
    def __init__(self, plan: ReleaseFailurePlan) -> None:
        self.plan = plan
        self.calls: list[tuple[str, int, str]] = []

    def invoke(self, rank, payload):  # type: ignore[no-untyped-def]
        assert payload["process_id"] == rank.process_id
        assert payload["process_group_id"] == rank.process_group_id
        assert payload["gpu_uuid"] == rank.gpu_uuid
        assert payload["temp_quota_root"] == rank.temp_quota_root
        phase = payload["phase"]
        operation = payload["operation"]
        self.calls.append((rank.topology, rank.rank, phase))
        context = FailureActuatorContext(
            plan_sha256=payload["plan_sha256"],
            scenario_semantics_sha256=payload["scenario_semantics_sha256"],
            topology=rank.topology,
            rank=rank.rank,
            world_size=1 if rank.topology == "tp1_dp1" else 2,
            process_id=rank.process_id,
            process_start_monotonic_ns=rank.process_start_monotonic_ns,
            session_epoch=0,
            run_nonce_sha256=payload["run_nonce_sha256"],
        )
        semantics = failure_semantics(payload["scenario"])
        order = {"arm": 10, "trigger": 20, "proof": 30, "recover": 40, "terminal": 50}
        response = {
            "phase": phase,
            "operation": operation,
            "monotonic_ns": order[phase],
            "event_count": 1,
            "observation_sha256": failure_phase_observation_sha256(
                context,
                semantics,
                phase=phase,
                operation=operation,
                event_count=1,
            ),
        }
        if phase == "terminal":
            counters = {row.name: 0 for row in self.plan.expected_counters}
            if rank.rank == 0:
                counters[semantics.terminal_counter] = 1
            response.update(counters=counters, recovery_valid=True)
        return response


class _FakeActuator:
    actuator_id = "cpu.fake.first_party_fault_actuator.v1"
    actuator_version_sha256 = _sha("actuator-version")

    def __init__(
        self,
        *,
        wrong_proof: bool = False,
        recovery_valid: bool = True,
        trigger_count: int = 1,
        fail_rank: tuple[str, int] | None = None,
    ) -> None:
        self.wrong_proof = wrong_proof
        self.recovery_valid = recovery_valid
        self.trigger_count = trigger_count
        self.fail_rank = fail_rank

    def fresh_context(
        self,
        *,
        plan: ReleaseFailurePlan,
        semantics: FailureScenarioSemantics,
        topology: str,
        rank: int,
        run_nonce_sha256: str,
    ) -> FailureActuatorContext:
        topology_offset = {
            "tp1_dp1": 0,
            "tp2_dp1": 100,
            "tp1_dp2": 200,
        }[topology]
        return FailureActuatorContext(
            plan_sha256=plan.sha256,
            scenario_semantics_sha256=semantics.sha256,
            topology=topology,
            rank=rank,
            world_size=1 if topology == "tp1_dp1" else 2,
            process_id=2000 + topology_offset + rank,
            process_start_monotonic_ns=1,
            session_epoch=0,
            run_nonce_sha256=run_nonce_sha256,
        )

    def _observation(
        self,
        context: FailureActuatorContext,
        phase: str,
        operation: str,
        time: int,
    ) -> FailurePhaseObservation:
        return FailurePhaseObservation(
            phase=phase,
            operation=operation,
            monotonic_ns=time,
            event_count=1,
            observation_sha256=failure_phase_observation_sha256(
                context,
                failure_semantics_from_context(context),
                phase=phase,
                operation=operation,
                event_count=1,
            ),
        )

    def arm(
        self,
        context: FailureActuatorContext,
        semantics: FailureScenarioSemantics,
    ) -> FailurePhaseObservation:
        return self._observation(context, "arm", semantics.arm_operation, 10)

    def trigger(
        self,
        context: FailureActuatorContext,
        semantics: FailureScenarioSemantics,
    ) -> FailurePhaseObservation:
        if self.fail_rank == (context.topology, context.rank):
            raise RuntimeError("deterministic rank actuation failure")
        return self._observation(context, "trigger", semantics.trigger_operation, 20)

    def prove(
        self,
        context: FailureActuatorContext,
        semantics: FailureScenarioSemantics,
    ) -> FailurePhaseObservation:
        operation = "generic_self_report" if self.wrong_proof else semantics.proof_event
        return self._observation(context, "proof", operation, 30)

    def recover(
        self,
        context: FailureActuatorContext,
        semantics: FailureScenarioSemantics,
    ) -> FailurePhaseObservation:
        return self._observation(
            context,
            "recover",
            semantics.recovery_invariant,
            40,
        )

    def terminal(
        self,
        context: FailureActuatorContext,
        semantics: FailureScenarioSemantics,
    ) -> FailureTerminalObservation:
        counters = {
            "exactness_violations": 0,
            "fallbacks": 0,
            "partial_target_continuations": 0,
            "retractions": 0,
            "version_mismatches": 0,
            semantics.terminal_counter: self.trigger_count if context.rank == 0 else 0,
        }
        return FailureTerminalObservation(
            phase="terminal",
            operation=semantics.terminal_counter,
            monotonic_ns=50,
            event_count=1,
            observation_sha256=failure_phase_observation_sha256(
                context,
                semantics,
                phase="terminal",
                operation=semantics.terminal_counter,
                event_count=1,
            ),
            counters=tuple(sorted(counters.items())),
            recovery_valid=self.recovery_valid,
        )


def failure_semantics_from_context(
    context: FailureActuatorContext,
) -> FailureScenarioSemantics:
    return next(
        value
        for value in FAILURE_SCENARIO_SEMANTICS
        if value.sha256 == context.scenario_semantics_sha256
    )


def test_source_semantics_exactly_cover_all_eleven_registered_scenarios() -> None:
    assert tuple(value.scenario for value in FAILURE_SCENARIO_SEMANTICS) == E5_FAILURES
    assert len(FAILURE_SCENARIO_SEMANTICS) == 11
    assert len({value.sha256 for value in FAILURE_SCENARIO_SEMANTICS}) == 11
    assert FAILURE_SCENARIO_SEMANTICS_SHA256 == content_sha256(
        [value.to_dict() for value in FAILURE_SCENARIO_SEMANTICS]
    )
    assert all(
        len(
            {
                value.arm_operation,
                value.trigger_operation,
                value.proof_event,
                value.recovery_invariant,
                value.terminal_counter,
            }
        )
        == 5
        for value in FAILURE_SCENARIO_SEMANTICS
    )


@pytest.mark.parametrize("scenario", E5_FAILURES)
def test_cpu_fake_lifecycle_is_atomic_all_rank_and_recovered_for_each_scenario(
    tmp_path: Path,
    registry: ExperimentRegistry,
    scenario: str,
) -> None:
    authority = _authority(tmp_path, registry, scenario)
    output = (tmp_path / f"receipt-{scenario}.json").resolve()
    receipt = execute_failure_actuator_for_cpu_test(
        authority,
        _FakeActuator(),
        run_nonce_sha256=_sha(f"nonce:{scenario}"),
        receipt_path=output,
    )

    assert output.is_file()
    assert receipt.scenario == scenario
    assert receipt.scenario_semantics_sha256 == failure_semantics(scenario).sha256
    assert receipt.formal_execution_authorized is False
    assert receipt.correctness_only is True
    assert receipt.recovered is receipt.committed is True
    assert tuple((value.topology, value.rank) for value in receipt.rank_receipts) == (
        ("tp1_dp1", 0),
        ("tp2_dp1", 0),
        ("tp2_dp1", 1),
        ("tp1_dp2", 0),
        ("tp1_dp2", 1),
    )
    assert all(value.session_epoch == 0 for value in receipt.rank_receipts)
    assert all(value.recovery_valid for value in receipt.rank_receipts)
    validate_failure_recovery_receipt(receipt, authority)


@pytest.mark.parametrize(
    ("actuator", "message"),
    [
        (_FakeActuator(wrong_proof=True), "source-owned scenario semantics"),
        (_FakeActuator(recovery_valid=False), "did not prove recovery"),
        (_FakeActuator(trigger_count=0), "missing from a registered topology"),
    ],
)
def test_generic_proof_missing_recovery_or_missing_counter_cannot_commit(
    tmp_path: Path,
    registry: ExperimentRegistry,
    actuator: _FakeActuator,
    message: str,
) -> None:
    authority = _authority(tmp_path, registry, "communicator_failure")
    output = (tmp_path / "must-not-exist.json").resolve()
    with pytest.raises(ValueError, match=message):
        execute_failure_actuator_for_cpu_test(
            authority,
            actuator,
            run_nonce_sha256=_sha("negative-nonce"),
            receipt_path=output,
        )
    assert not output.exists()


def test_one_rank_failure_prevents_atomic_receipt(
    tmp_path: Path,
    registry: ExperimentRegistry,
) -> None:
    authority = _authority(tmp_path, registry, "slow_rank")
    output = (tmp_path / "partial.json").resolve()
    with pytest.raises(RuntimeError, match="rank actuation failure"):
        execute_failure_actuator_for_cpu_test(
            authority,
            _FakeActuator(fail_rank=("tp1_dp2", 1)),
            run_nonce_sha256=_sha("partial-nonce"),
            receipt_path=output,
        )
    assert not output.exists()


@pytest.mark.parametrize("scenario", E5_FAILURES)
def test_release_unsigned_terminal_uses_source_actuator_and_exact_child_scopes(
    tmp_path: Path,
    registry: ExperimentRegistry,
    scenario: str,
) -> None:
    token, cell, authority = _token_cell_authority(tmp_path, registry, scenario)
    launch = _launch(tmp_path, authority=authority, registry=registry)
    transport = _FakeNativeFailureTransport(authority.plan)
    output = (tmp_path / f"unsigned-{scenario}.json").resolve()

    binding = execute_release_failure_actuator_unsigned(
        token,
        cell=cell,
        expected_registry_sha256=registry.sha256,
        launch=launch,
        transport=transport,
        raw_terminal_path=output,
    )

    assert binding.absolute_path == str(output)
    assert binding.reopen()["formal_execution_authorized"] is False
    assert transport.calls == [
        (topology, rank, phase)
        for topology, rank in (
            ("tp1_dp1", 0),
            ("tp2_dp1", 0),
            ("tp2_dp1", 1),
            ("tp1_dp2", 0),
            ("tp1_dp2", 1),
        )
        for phase in ("arm", "trigger", "proof", "recover", "terminal")
    ]
    signing_binding = build_failure_actuation_external_control_binding(
        str(output),
        token=token,
        cell=cell,
        expected_registry_sha256=registry.sha256,
        expected_assignment_sha256=launch.assignment_sha256,
        expected_inventory_sha256=launch.inventory_sha256,
    )
    assert signing_binding.launch_binding_sha256 == launch.sha256
    assert signing_binding.scenario == scenario
    assert len(FAILURE_ACTUATION_EXTERNAL_CONTROL_PROTOCOL_SHA256) == 64


def test_release_unsigned_terminal_has_zero_partial_publication_on_rank_failure(
    tmp_path: Path,
    registry: ExperimentRegistry,
) -> None:
    token, cell, authority = _token_cell_authority(
        tmp_path, registry, "communicator_failure"
    )
    launch = _launch(tmp_path, authority=authority, registry=registry)
    transport = _FakeNativeFailureTransport(authority.plan)
    original = transport.invoke

    def fail_one_rank(rank, payload):  # type: ignore[no-untyped-def]
        if (
            rank.topology == "tp1_dp2"
            and rank.rank == 1
            and payload["phase"] == "proof"
        ):
            raise RuntimeError("native rank terminated")
        return original(rank, payload)

    transport.invoke = fail_one_rank  # type: ignore[method-assign]
    output = (tmp_path / "must-not-publish-unsigned.json").resolve()
    with pytest.raises(RuntimeError, match="native rank terminated"):
        execute_release_failure_actuator_unsigned(
            token,
            cell=cell,
            expected_registry_sha256=registry.sha256,
            launch=launch,
            transport=transport,
            raw_terminal_path=output,
        )
    assert not output.exists()


def test_concrete_sglang_http_transport_is_exact_localhost_and_revalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightcone_spec.experiments.failure_actuator as actuator_module

    received: list[object] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            size = int(self.headers["Content-Length"])
            received.append(json.loads(self.rfile.read(size)))
            body = b'{"transport":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    quota = (tmp_path / "transport-quota").resolve()
    quota.mkdir(mode=0o700)
    rank = FailureRankLaunchBinding(
        topology="tp1_dp1",
        rank=0,
        process_id=os.getpid(),
        process_group_id=os.getpgid(0),
        process_start_monotonic_ns=1,
        gpu_uuid="GPU-transport",
        control_url=(
            f"http://127.0.0.1:{server.server_port}/v1/lightcone-spec/failure-actuator"
        ),
        temp_quota_root=str(quota),
    )
    checks: list[FailureRankLaunchBinding] = []
    monkeypatch.setattr(
        actuator_module,
        "_revalidate_failure_rank_process",
        lambda value: checks.append(value),
    )
    semantics = failure_semantics("queue_saturation")
    payload = {
        "schema_version": 1,
        "kind": "lightcone_e5_failure_actuator_command",
        "protocol_sha256": FAILURE_ACTUATOR_PROTOCOL_SHA256,
        "launch_binding_sha256": _sha("transport-launch"),
        "assignment_sha256": _sha("transport-assignment"),
        "inventory_sha256": _sha("transport-inventory"),
        "plan_sha256": _sha("transport-plan"),
        "scenario": semantics.scenario,
        "scenario_semantics_sha256": semantics.sha256,
        "phase": "arm",
        "operation": semantics.arm_operation,
        "parameters": [list(row) for row in semantics.parameters],
        "topology": rank.topology,
        "rank": rank.rank,
        "process_id": rank.process_id,
        "process_group_id": rank.process_group_id,
        "process_start_monotonic_ns": rank.process_start_monotonic_ns,
        "gpu_uuid": rank.gpu_uuid,
        "temp_quota_root": rank.temp_quota_root,
        "run_nonce_sha256": _sha("transport-nonce"),
    }
    try:
        response = SglangHttpFailureNativeControlTransport().invoke(rank, payload)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response == {"transport": "ok"}
    assert received == [payload]
    assert checks == [rank, rank]
    with pytest.raises(ValueError, match="process ID"):
        linux_process_start_monotonic_ns(0)


def test_failure_external_control_proof_is_durable_and_replay_bound(
    tmp_path: Path,
    registry: ExperimentRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, cell, authority = _token_cell_authority(
        tmp_path, registry, "replica_restart"
    )
    launch = _launch(tmp_path, authority=authority, registry=registry)
    raw_path = (tmp_path / "unsigned-replica-restart.json").resolve()
    execute_release_failure_actuator_unsigned(
        token,
        cell=cell,
        expected_registry_sha256=registry.sha256,
        launch=launch,
        transport=_FakeNativeFailureTransport(authority.plan),
        raw_terminal_path=raw_path,
    )
    signing_binding = build_failure_actuation_external_control_binding(
        str(raw_path),
        token=token,
        cell=cell,
        expected_registry_sha256=registry.sha256,
        expected_assignment_sha256=launch.assignment_sha256,
        expected_inventory_sha256=launch.inventory_sha256,
    )
    private, root_binding, bundle, authorization = _control_authority(
        monkeypatch,
        inventory_sha256=launch.inventory_sha256,
    )
    control = _control_envelope(
        private_key=private,
        root_binding=root_binding,
        bundle=bundle,
        authorization=authorization,
        binding_sha256=signing_binding.sha256,
        registry_sha256=registry.sha256,
        lineage_sha256=signing_binding.lineage_sha256,
    )
    replay_root = tmp_path / "failure-replay"
    replay_root.mkdir(mode=0o700)
    replay_store = ChallengeReplayStore(str(replay_root.resolve()))
    proof_path = (tmp_path / "failure-proof.json").resolve()
    publish_failure_actuation_proof_artifact(
        str(raw_path),
        token=token,
        cell=cell,
        expected_registry_sha256=registry.sha256,
        expected_assignment_sha256=launch.assignment_sha256,
        expected_inventory_sha256=launch.inventory_sha256,
        expected_root_manifest_sha256=root_binding.semantic_sha256,
        control_attestation=control,
        replay_store=replay_store,
        now_ns=_NOW_NS,
        proof_artifact_path=str(proof_path),
    )
    reservations = tuple(replay_root.glob("reservation-*.json"))
    assert len(reservations) == 1

    result = validate_failure_actuation_proof_artifact(
        str(proof_path),
        token=token,
        cell=cell,
        expected_registry_sha256=registry.sha256,
        expected_assignment_sha256=launch.assignment_sha256,
        expected_inventory_sha256=launch.inventory_sha256,
        expected_root_manifest_sha256=root_binding.semantic_sha256,
        now_ns=_NOW_NS + 20_000_000_000,
    )
    assert result.scenario == "replica_restart"
    assert result.correctness_only
    assert tuple(
        (row.topology, row.rank) for row in result.recovery_receipt.rank_receipts
    ) == (
        ("tp1_dp1", 0),
        ("tp2_dp1", 0),
        ("tp2_dp1", 1),
        ("tp1_dp2", 0),
        ("tp1_dp2", 1),
    )
    assert tuple(replay_root.glob("reservation-*.json")) == reservations

    with pytest.raises(ValueError, match="formal execution identity differs"):
        validate_failure_actuation_proof_artifact(
            str(proof_path),
            token=token,
            cell=cell,
            expected_registry_sha256=registry.sha256,
            expected_assignment_sha256=_sha("foreign-assignment"),
            expected_inventory_sha256=launch.inventory_sha256,
            expected_root_manifest_sha256=root_binding.semantic_sha256,
            now_ns=_NOW_NS,
        )
    with pytest.raises((ValueError, RuntimeError), match="replay|consumed"):
        publish_failure_actuation_proof_artifact(
            str(raw_path),
            token=token,
            cell=cell,
            expected_registry_sha256=registry.sha256,
            expected_assignment_sha256=launch.assignment_sha256,
            expected_inventory_sha256=launch.inventory_sha256,
            expected_root_manifest_sha256=root_binding.semantic_sha256,
            control_attestation=control,
            replay_store=replay_store,
            now_ns=_NOW_NS,
            proof_artifact_path=str((tmp_path / "replayed-proof.json").resolve()),
        )


def test_release_capability_is_named_block_before_actuator_or_output(
    tmp_path: Path,
) -> None:
    assert len(RELEASE_FAILURE_ACTUATOR_CAPABILITIES) == 1
    assert RELEASE_FAILURE_ACTUATOR_CAPABILITIES[0].actuator_id == (
        "release.sglang_child_fault_actuator.v1"
    )
    output = tmp_path / "never-created.json"
    with pytest.raises(FailureActuatorBlocked) as blocked:
        require_release_failure_actuator()
    assert blocked.value.reason == "failure_actuator_trusted_signer_unavailable"
    assert not output.exists()


def test_release_registry_binds_callable_and_identity_as_one_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightcone_spec.experiments.failure_actuator as module

    def release_failure_actuator_factory():
        return _FakeActuator()

    release_failure_actuator_factory.__module__ = (
        "lightcone_spec.experiments.failure_actuator"
    )
    release_failure_actuator_factory.__qualname__ = "release_failure_actuator_factory"
    capability = ReleaseFailureActuatorCapability(
        actuator_id="release.first_party_fault_actuator.v1",
        actuator_version_sha256=_sha("release-actuator-version"),
        factory_module=release_failure_actuator_factory.__module__,
        factory_qualname=release_failure_actuator_factory.__qualname__,
        factory=release_failure_actuator_factory,
    )
    monkeypatch.setattr(
        module,
        "RELEASE_FAILURE_ACTUATOR_CAPABILITIES",
        (capability,),
    )
    assert release_failure_actuator_capability() == capability
    assert capability.sha256 == content_sha256(
        {
            "schema_version": 1,
            "kind": "release_failure_actuator_capability",
            "actuator_id": capability.actuator_id,
            "actuator_version_sha256": capability.actuator_version_sha256,
            "factory_module": capability.factory_module,
            "factory_qualname": capability.factory_qualname,
        }
    )
    release_failure_actuator_factory.__qualname__ = "replaced_factory"
    with pytest.raises(FailureActuatorBlocked) as replaced:
        release_failure_actuator_capability()
    assert replaced.value.reason == FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON

    monkeypatch.setattr(
        module,
        "RELEASE_FAILURE_ACTUATOR_CAPABILITIES",
        ((capability.actuator_id, capability.actuator_version_sha256),),
    )
    with pytest.raises(FailureActuatorBlocked) as blocked:
        release_failure_actuator_capability()
    assert blocked.value.reason == FAILURE_ACTUATOR_RELEASE_UNAVAILABLE_REASON


def test_cpu_diagnostic_actuator_cannot_be_registered_for_release() -> None:
    def source_factory():
        return _FakeActuator()

    source_factory.__module__ = "lightcone_spec.experiments.failure_actuator"
    source_factory.__qualname__ = "source_factory"
    with pytest.raises(ValueError, match="diagnostic actuator identities"):
        ReleaseFailureActuatorCapability(
            actuator_id=_FakeActuator.actuator_id,
            actuator_version_sha256=_FakeActuator.actuator_version_sha256,
            factory_module=source_factory.__module__,
            factory_qualname=source_factory.__qualname__,
            factory=source_factory,
        )


def test_scenario_semantics_cannot_be_posthoc_substituted(
    tmp_path: Path,
    registry: ExperimentRegistry,
) -> None:
    authority = _authority(tmp_path, registry, "disk_quota")
    wrong = replace(
        failure_semantics("disk_quota"),
        proof_event=failure_semantics("oom_candidate").proof_event,
    )
    context = _FakeActuator().fresh_context(
        plan=authority.plan,
        semantics=wrong,
        topology="tp2_dp1",
        rank=0,
        run_nonce_sha256=_sha("substitution"),
    )
    assert context.scenario_semantics_sha256 != failure_semantics("disk_quota").sha256

    class ForeignContextActuator(_FakeActuator):
        def fresh_context(self, **kwargs):  # type: ignore[no-untyped-def]
            valid = super().fresh_context(**kwargs)
            return replace(valid, scenario_semantics_sha256=wrong.sha256)

    output = (tmp_path / "foreign-context.json").resolve()
    with pytest.raises(ValueError, match="foreign execution context"):
        execute_failure_actuator_for_cpu_test(
            authority,
            ForeignContextActuator(),
            run_nonce_sha256=_sha("substitution"),
            receipt_path=output,
        )
    assert not output.exists()
