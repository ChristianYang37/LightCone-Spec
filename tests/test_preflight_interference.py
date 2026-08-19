from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import socket
import stat
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_live_sglang_runner import (
    _FAKE_SERVER_SOURCE,
    _fake_live_server_configuration,
)

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.config.schema import RuntimeConfig
from lightcone_spec.experiments import formal_preflight_execution
from lightcone_spec.experiments.formal_dispatch import (
    FormalPreflightExecutionBinding,
)
from lightcone_spec.experiments.formal_preflight_execution import (
    FormalPreflightInterferenceExecutionManifest,
    FormalPreflightInterferenceRunInput,
    execute_formal_preflight_interference_raw,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAssignment,
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
    registry_pool_work_item,
)
from lightcone_spec.experiments.interference_authority import (
    InterferenceRawObservation,
)
from lightcone_spec.experiments.itl_authority import (
    STAGE_ITL_TIMESTAMP_PROOF_PROTOCOL_SHA256,
    StageItlExecutionIdentity,
    StageItlTimestampProofRequest,
    build_stage_itl_external_control_binding,
    publish_stage_itl_timestamp_proof_artifacts,
    publish_stage_itl_timestamp_raw_receipt,
)
from lightcone_spec.experiments.preflight_interference import (
    _SLO_POLICY_SHA256,
    FORMAL_PREFLIGHT_INTERFERENCE_PROOF_PROTOCOL_SHA256,
    FORMAL_PREFLIGHT_INTERFERENCE_RAW_PROTOCOL_SHA256,
    FormalPreflightInterferenceFatalTerminal,
    FormalPreflightInterferenceProofRow,
    FormalPreflightInterferenceQualificationLock,
    FormalPreflightInterferenceQualificationRow,
    FormalPreflightInterferenceRawBatch,
    FormalPreflightInterferenceRawRow,
    _diagnose,
    build_formal_preflight_interference_aggregate_binding,
    publish_formal_preflight_interference_proof_artifact,
    validate_formal_preflight_interference_proof_artifact,
)
from lightcone_spec.experiments.registry import (
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.serving import PinnedBenchServingTransport
from lightcone_spec.orchestration import live_sglang
from lightcone_spec.orchestration.formal_terminal_result import (
    publish_formal_current_preflight_tp1_terminal_result_proof_artifact,
    publish_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact,
    validate_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact,
)
from lightcone_spec.orchestration.live_sglang import (
    PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
    PinnedNvidiaSmiTool,
    PinnedSglangServingRunSpec,
    _derive_actual_group_overlap,
    execute_unsigned_native_serving_group,
)
from lightcone_spec.orchestration.native_terminal import (
    NATIVE_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256,
    NativeTerminalRunBinding,
    build_native_terminal_external_control_binding,
    publish_native_terminal_result_proof_artifacts,
)
from lightcone_spec.runtime import release_trust_root as root_module
from lightcone_spec.runtime.attestation import (
    NO_TRUSTED_ATTESTERS,
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
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.readiness import (
    NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256,
    NATIVE_RUNTIME_QUALIFICATION_TESTS,
    NATIVE_RUNTIME_RELEASE_CAPABILITY,
    NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S,
    NativeRuntimeGpuProofReceipt,
    build_native_runtime_gpu_proof_artifact,
    verify_native_runtime_gpu_proof,
)
from lightcone_spec.runtime.release_trust_root import (
    DeploymentPolicyAuthorization,
    SourceReleaseEd25519Root,
    SourceReleaseRootBinding,
    deployment_policy_subject_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _public_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _release_control_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hardware_envelope_sha256: str,
) -> SimpleNamespace:
    root_private = Ed25519PrivateKey.generate()
    controller_private = Ed25519PrivateKey.generate()
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
    root_path = "/validation/preflight-interference-root.json"
    root_binding = SourceReleaseRootBinding(
        root=root,
        path=root_path,
        sidecar_path=f"{root_path}.sha256",
        semantic_sha256=root.sha256,
        file_sha256=_sha("root-file"),
        sidecar_file_sha256=_sha("root-sidecar"),
    )
    controller_public = _public_bytes(controller_private)
    fingerprint = hashlib.sha256(controller_public).hexdigest()
    bundle = TrustedAttesterPolicyBundle(
        schema_version=1,
        kind="lightcone_trusted_attester_policy_bundle",
        bundle_id="preflight-interference-test-bundle-v1",
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
        hardware_envelope_sha256_allowlist=(hardware_envelope_sha256,),
        trusted_attester_policy=TrustedAttesterPolicy(
            policy_id="preflight-interference-test-policy-v1",
            trusted_attesters=(
                (
                    "validation-signer",
                    "validation-signer-key",
                    fingerprint,
                ),
            ),
            public_keys=(
                (
                    fingerprint,
                    base64.b64encode(controller_public).decode("ascii"),
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        root_module,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    return SimpleNamespace(
        root_private=root_private,
        controller_private=controller_private,
        root_binding=root_binding,
        bundle=bundle,
        hardware_envelope_sha256=hardware_envelope_sha256,
    )


def _deployment_authorization(
    context: SimpleNamespace,
    *,
    inventory_sha256: str,
    label: str,
) -> DeploymentPolicyAuthorization:
    subject_sha256 = deployment_policy_subject_sha256(
        root_manifest_sha256=context.root_binding.semantic_sha256,
        inventory_sha256=inventory_sha256,
        bundle_sha256=context.bundle.sha256,
    )
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id=f"preflight-deployment-{label}",
        nonce_base64=base64.b64encode(
            hashlib.sha256(f"deployment-{label}".encode()).digest()
        ).decode("ascii"),
        subject_sha256=subject_sha256,
        issued_ns=1_500_000_000,
        expires_ns=3_000_000_000,
    )
    return DeploymentPolicyAuthorization(
        schema_version=1,
        kind="lightcone_deployment_policy_authorization",
        root_manifest_sha256=context.root_binding.semantic_sha256,
        inventory_sha256=inventory_sha256,
        bundle=context.bundle,
        challenge=challenge,
        signature_base64=base64.b64encode(
            context.root_private.sign(
                attestation_message(
                    challenge,
                    payload_sha256=context.bundle.sha256,
                )
            )
        ).decode("ascii"),
    )


def _control(
    context: SimpleNamespace,
    *,
    inventory_sha256: str,
    subject: ControlArtifactSubject,
    label: str,
) -> ControlArtifactAttestation:
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id=f"preflight-control-{label}",
        nonce_base64=base64.b64encode(
            hashlib.sha256(f"control-{label}".encode()).digest()
        ).decode("ascii"),
        subject_sha256=subject.sha256,
        issued_ns=1_600_000_000,
        expires_ns=2_600_000_000,
    )
    payload = subject.artifact_sha256
    return ControlArtifactAttestation(
        schema_version=1,
        kind="lightcone_control_artifact_attestation",
        subject=subject,
        hardware_envelope_sha256=context.hardware_envelope_sha256,
        trust_anchor_sha256=context.root_binding.sha256,
        trust_bundle_sha256=context.bundle.sha256,
        trusted_attester_policy_sha256=(context.bundle.trusted_attester_policy.sha256),
        deployment_policy_authorization=_deployment_authorization(
            context,
            inventory_sha256=inventory_sha256,
            label=label,
        ),
        challenge=challenge,
        attestation=SignedAttestation(
            schema_version=1,
            kind="lightcone_signed_attestation",
            algorithm="Ed25519",
            attester_id="validation-signer",
            key_id="validation-signer-key",
            environment="release",
            public_key_base64=base64.b64encode(
                _public_bytes(context.controller_private)
            ).decode("ascii"),
            challenge_sha256=challenge.sha256,
            payload_sha256=payload,
            signature_base64=base64.b64encode(
                context.controller_private.sign(
                    attestation_message(challenge, payload_sha256=payload)
                )
            ).decode("ascii"),
        ),
    )


def _subject(
    *,
    artifact_type: str,
    artifact_sha256: str,
    protocol_sha256: str,
    registry_sha256: str,
    lineage_sha256: str,
) -> ControlArtifactSubject:
    return ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type=artifact_type,
        artifact_sha256=artifact_sha256,
        protocol_sha256=protocol_sha256,
        registry_sha256=registry_sha256,
        lineage_sha256=lineage_sha256,
    )


def _binding(mode: str, repetition: int, slot: int) -> NativeTerminalRunBinding:
    suffix = f"{mode}-{repetition}-{slot}"
    return NativeTerminalRunBinding(
        run_id=f"preflight-{suffix}",
        run_nonce_sha256=_sha(f"run-nonce-{suffix}"),
        execution_plan_sha256=_sha("execution-plan"),
        rank_config_sha256=_sha(f"rank-config-{suffix}"),
        attempt_id=f"attempt-{suffix}",
        session_id=f"session-{suffix}",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=_sha(f"challenge-{suffix}"),
        method="static",
        warmup_request_ids=("warm-0",),
        scored_request_ids=("score-0",),
    )


def _json_binding(path: Path, value: object) -> CanonicalJsonProofBinding:
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _dynamic_two_gpu_tool(
    tmp_path: Path,
    *,
    pid_paths: tuple[Path, Path],
) -> PinnedNvidiaSmiTool:
    executable = (tmp_path / "dynamic-two-gpu-nvidia-smi").resolve()
    executable.write_text(
        "#!"
        + sys.executable
        + "\nimport os, pathlib, sys\n"
        + f"ROWS = {[(f'GPU-{index}', str(path)) for index, path in enumerate(pid_paths)]!r}\n"
        + "query = ' '.join(sys.argv[1:])\n"
        + "if '--query-compute-apps=' in query:\n"
        + "    for gpu, raw_path in ROWS:\n"
        + "        path = pathlib.Path(raw_path)\n"
        + "        if not path.is_file():\n"
        + "            continue\n"
        + "        pid = int(path.read_text())\n"
        + "        try:\n"
        + "            os.kill(pid, 0)\n"
        + "        except ProcessLookupError:\n"
        + "            continue\n"
        + "        print(f'{gpu}, {pid}, 1')\n"
        + "else:\n"
        + "    print('GPU-0, Test GPU, 1')\n"
        + "    print('GPU-1, Test GPU, 1')\n",
        encoding="utf-8",
    )
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return PinnedNvidiaSmiTool.bind(executable)


def _dynamic_many_run_two_gpu_tool(
    tmp_path: Path,
    *,
    pid_rows: tuple[tuple[str, Path], ...],
) -> PinnedNvidiaSmiTool:
    executable = (tmp_path / "dynamic-many-run-two-gpu-nvidia-smi").resolve()
    executable.write_text(
        "#!"
        + sys.executable
        + "\nimport os, pathlib, sys\n"
        + f"ROWS = {[(gpu, str(path)) for gpu, path in pid_rows]!r}\n"
        + "query = ' '.join(sys.argv[1:])\n"
        + "selector = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--id='))\n"
        + "selected = set(selector.split(','))\n"
        + "if '--query-compute-apps=' in query:\n"
        + "    for gpu, raw_path in ROWS:\n"
        + "        if gpu not in selected:\n"
        + "            continue\n"
        + "        path = pathlib.Path(raw_path)\n"
        + "        if not path.is_file():\n"
        + "            continue\n"
        + "        pid = int(path.read_text())\n"
        + "        try:\n"
        + "            os.kill(pid, 0)\n"
        + "        except ProcessLookupError:\n"
        + "            continue\n"
        + "        print(f'{gpu}, {pid}, 1')\n"
        + "else:\n"
        + "    for gpu in ('GPU-0', 'GPU-1'):\n"
        + "        if gpu in selected:\n"
        + "            print(f'{gpu}, Test GPU, 1')\n",
        encoding="utf-8",
    )
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return PinnedNvidiaSmiTool.bind(executable)


def _inventory() -> GpuInventory:
    devices = tuple(
        GpuDevice(
            uuid=f"GPU-{slot}",
            host_id="preflight-host",
            model="Test GPU",
            memory_bytes=96_000_000_000,
            compute_capability=(12, 0),
            pci_bus_id=f"0000:{slot + 1:02x}:00.0",
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
        for slot in range(2)
    )
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id="dual-card",
                host_id="preflight-host",
                gpu_uuids=("GPU-0", "GPU-1"),
                fabric="PCIe",
                bandwidth_class="test",
            ),
        ),
        source_receipt_sha256=_sha("inventory-source"),
    )


def _overlap_transport() -> PinnedBenchServingTransport:
    class RequestInput:
        def __init__(self, **kwargs: object) -> None:
            vars(self).update(kwargs)

    class TraceConfig:
        def __init__(self) -> None:
            self.on_connection_create_end: list[object] = []
            self.on_connection_reuseconn: list[object] = []

        def freeze(self) -> None:
            return

    class Response:
        def __init__(self, value: object) -> None:
            self.status = 200
            self._value = value

        async def json(self, *, content_type=None):
            assert content_type is None
            return self._value

    class RequestContext:
        def __init__(self, request: urllib.request.Request) -> None:
            self._request = request
            self._response: Response | None = None

        async def __aenter__(self) -> Response:
            def load() -> object:
                with urllib.request.urlopen(self._request, timeout=5.0) as stream:
                    return json.loads(stream.read())

            self._response = Response(await asyncio.to_thread(load))
            return self._response

        async def __aexit__(self, *_args: object) -> None:
            self._response = None

    class Session:
        def __init__(self) -> None:
            self.closed = False
            self.trace_configs: list[object] = []

        def get(self, *, url: str, headers: dict[str, str]) -> RequestContext:
            return RequestContext(urllib.request.Request(url, headers=headers))

        def post(
            self,
            *,
            url: str,
            json: dict[str, object],
            headers: dict[str, str],
        ) -> RequestContext:
            return RequestContext(
                urllib.request.Request(
                    url,
                    data=__import__("json").dumps(json).encode(),
                    headers={"Content-Type": "application/json", **headers},
                    method="POST",
                )
            )

    async def open_bench_client_session(*, total_timeout_s: float = 6 * 60 * 60):
        assert total_timeout_s > 0
        return Session()

    async def close_bench_client_session(client_session) -> None:
        client_session.closed = True

    async def async_request_sglang_generate(
        request_func_input,
        pbar=None,
        *,
        client_session=None,
        timeout_s=None,
    ):
        assert pbar is None
        assert client_session is not None
        assert timeout_s is not None
        request_id = request_func_input.extra_request_body["rid"]
        token_ids = (3,) if request_id == "warm-0" else (6, 7)
        started_ns = __import__("time").monotonic_ns()
        pointer = {
            "schema_version": 1,
            "kind": "sglang_native_itl_result_pointer",
            "hook": "sglang.schema_v3.native_per_token_timestamp.v2",
            "semantics": "scheduler_committed_token_at_result_processor_v1",
            "release_status": "IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF",
            "request_id": request_id,
            "request_started_ns": started_ns,
            "request_terminal_ns": started_ns + 30_000_000,
            "terminal_status": "completed",
            "terminal_reason": "FINISH_LENGTH",
            "events": [
                {
                    "token_index": index,
                    "token_id": token_id,
                    "observed_ns": started_ns + (index + 1) * 10_000_000,
                }
                for index, token_id in enumerate(token_ids)
            ],
        }
        unsigned = dict(pointer)
        pointer["result_pointer_sha256"] = content_sha256(unsigned)
        await asyncio.sleep(0.04)
        return SimpleNamespace(
            success=True,
            generated_text=request_id,
            generated_token_ids=list(token_ids),
            output_len=len(token_ids),
            latency=0.04,
            ttft=0.01,
            native_token_timestamp_result_pointer=pointer,
        )

    async def async_request_sglang_abort(
        request_id,
        base_url,
        *,
        client_session=None,
        timeout_s=None,
    ) -> None:
        raise AssertionError((request_id, base_url, client_session, timeout_s))

    return PinnedBenchServingTransport(
        request_type=RequestInput,
        request_callable=async_request_sglang_generate,
        abort_callable=async_request_sglang_abort,
        open_session_callable=open_bench_client_session,
        close_session_callable=close_bench_client_session,
        set_global_args=lambda _value: None,
        trace_config_factory=TraceConfig,
        headers_factory=dict,
        module_identity="sglang.benchmark.serving.async_request_sglang_generate",
    )


def _qualification(
    path: Path,
    *,
    materialized_cell_id: str,
    registry_cell_id: str,
    scored_request_sha256: str,
) -> CanonicalJsonProofBinding:
    value = FormalPreflightInterferenceQualificationLock(
        schema_version=1,
        kind="formal_preflight_interference_request_qualification_lock",
        registry_sha256=_sha("registry"),
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        materialized_cell_id=materialized_cell_id,
        registry_cell_id=registry_cell_id,
        workload_authorization_sha256=_sha("workload"),
        scored_request_inputs_sha256=content_sha256([scored_request_sha256]),
        slo_policy_sha256=_SLO_POLICY_SHA256,
        rows=(
            FormalPreflightInterferenceQualificationRow(
                request_id="score-0",
                prompt_bucket="short",
                eligible=True,
            ),
        ),
    )
    return _json_binding(path, value.to_dict())


def _error_row(
    tmp_path: Path,
    *,
    mode: str,
    repetition: int,
    slot: int,
    inventory_sha256: str,
    shared_group_source: CanonicalJsonProofBinding | None = None,
) -> FormalPreflightInterferenceRawRow:
    binding = _binding(mode, repetition, slot)
    run_sha256 = content_sha256(binding.begin_payload())
    reason = "preflight_test_failure"
    source = shared_group_source
    if source is None:
        source = _json_binding(
            (tmp_path / f"source-fatal-{mode}-{repetition}-{slot}.json").resolve(),
            {
                "schema_version": 1,
                "kind": "unsigned_pinned_sglang_serving_fatal_pointer",
                "protocol_sha256": PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
                "status": "ERROR",
                "formal_execution_authorized": False,
                "reason_code": reason,
                "run_binding_sha256": run_sha256,
            },
        )
    materialized = _sha(f"materialized-{mode}-{repetition}-{slot}")
    cell = _sha(f"cell-{mode}-{repetition}-{slot}")
    fatal = FormalPreflightInterferenceFatalTerminal(
        schema_version=2,
        kind="formal_preflight_interference_fatal_terminal",
        registry_cell_id=cell,
        assignment_sha256=_sha(f"assignment-{mode}-{repetition}-{slot}"),
        experiment_budget_sha256=_sha(f"budget-{mode}-{repetition}-{slot}"),
        inventory_sha256=inventory_sha256,
        run_binding_sha256=run_sha256,
        error_code=reason,
        source_fatal_terminal=source,
    )
    fatal_binding = _json_binding(
        (tmp_path / f"fatal-{mode}-{repetition}-{slot}.json").resolve(),
        fatal.to_dict(),
    )
    return FormalPreflightInterferenceRawRow(
        materialized_cell_id=materialized,
        registry_cell_id=cell,
        assignment_sha256=fatal.assignment_sha256,
        experiment_budget_sha256=fatal.experiment_budget_sha256,
        inventory_sha256=inventory_sha256,
        gpu_uuid=f"GPU-{slot}",
        mode=mode,  # type: ignore[arg-type]
        repetition=repetition,
        slot=slot,
        run_binding=binding,
        status="ERROR",
        launch_manifest=None,
        live_run_receipt=None,
        raw_terminal=None,
        native_itl_pointer_artifact=None,
        qualification_lock=None,
        concurrent_group_receipt=None,
        fatal_terminal=fatal_binding,
    )


def test_exact_eight_raw_batch_deep_reopens_shared_group_and_rejects_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _sha("inventory")
    checkout = (tmp_path / "patched-sglang").resolve()
    checkout.mkdir()
    pid_paths = (
        (tmp_path / "server-0.pid").resolve(),
        (tmp_path / "server-1.pid").resolve(),
    )
    tool = _dynamic_two_gpu_tool(tmp_path, pid_paths=pid_paths)
    launches: dict[str, object] = {}
    specs: list[PinnedSglangServingRunSpec] = []
    success_inputs: list[dict[str, object]] = []
    run_config = SimpleNamespace(
        method="static",
        adaptation=None,
        model=SimpleNamespace(target="target/test"),
        runtime=RuntimeConfig(
            sampling_profile_sha256=_sha("sampling-profile"),
            max_running_requests=2,
        ),
    )
    for slot in range(2):
        binding = _binding("concurrent", 0, slot)
        config, warmup, scored = _fake_live_server_configuration(binding)
        server_source = (tmp_path / f"server-{slot}.py").resolve()
        server_source.write_text(_FAKE_SERVER_SOURCE, encoding="utf-8")
        server_config = (tmp_path / f"server-{slot}.json").resolve()
        server_config.write_text(
            json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        port = _unused_port()
        launch_path = (tmp_path / f"launch-{slot}.json").resolve()
        launch_binding = _json_binding(launch_path, {"slot": slot})
        server_argv = (
            str(Path(sys.executable).resolve()),
            str(server_source),
            str(port),
            str(server_config),
            str(pid_paths[slot]),
            f"GPU-{slot}",
            "--disable-cuda-graph",
        )
        assignment = _sha(f"assignment-concurrent-0-{slot}")
        budget = _sha(f"budget-concurrent-0-{slot}")

        class FakeLaunch:
            @staticmethod
            def child_environment(
                gpu_uuid: str = f"GPU-{slot}",
            ) -> dict[str, str]:
                return {
                    "PATH": str(Path(sys.executable).parent),
                    "LD_LIBRARY_PATH": "",
                    "CUDA_HOME": str(tmp_path),
                    "CUDA_PATH": str(tmp_path),
                    "CUDA_VISIBLE_DEVICES": gpu_uuid,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                    "LANG": "C",
                    "LC_ALL": "C",
                }

        launch = FakeLaunch()
        launch.sha256 = launch_binding.semantic_sha256
        launch.run_config_path = str((tmp_path / "run-config.json").resolve())
        launch.target_model_id = "target/test"
        launch.patched_sglang_checkout = str(checkout)
        launch.patched_sglang_commit = PINNED_SGLANG_COMMIT
        launch.patched_sglang_tree = PINNED_SGLANG_TREE
        launch.server_argv = server_argv
        launch.server_argv_sha256 = _sha(f"argv-{slot}")
        launch.physical_assignment_sha256 = assignment
        launch.experiment_budget_sha256 = budget
        launch.inventory_sha256 = inventory
        launch.gpu_uuids = (f"GPU-{slot}",)
        launch.localhost_port = port
        launches[str(launch_path)] = launch
        row_dir = (tmp_path / f"row-{slot}").resolve()
        row_dir.mkdir(mode=0o700)
        qualification = _qualification(
            row_dir / "qualification.json",
            materialized_cell_id=_sha(f"materialized-concurrent-0-{slot}"),
            registry_cell_id=_sha(f"cell-concurrent-0-{slot}"),
            scored_request_sha256=scored[0].sha256,
        )
        specs.append(
            PinnedSglangServingRunSpec(
                launch_manifest=launch_binding,
                binding=binding,
                warmup_requests=warmup,
                scored_requests=scored,
                terminal_output_path=str(row_dir / "terminal.json"),
                native_itl_pointer_output_path=str(row_dir / "itl.json"),
                live_run_receipt_output_path=str(row_dir / "live.json"),
                server_log_output_path=str(row_dir / "server.log"),
            )
        )
        success_inputs.append(
            {
                "materialized_cell_id": _sha(f"materialized-concurrent-0-{slot}"),
                "registry_cell_id": _sha(f"cell-concurrent-0-{slot}"),
                "assignment_sha256": assignment,
                "experiment_budget_sha256": budget,
                "inventory_sha256": inventory,
                "gpu_uuid": f"GPU-{slot}",
                "mode": "concurrent",
                "repetition": 0,
                "slot": slot,
                "run_binding": binding,
                "status": "WAITING_FOR_LOCAL_CONTROL",
                "launch_manifest": launch_binding,
                "qualification_lock": qualification,
                "fatal_terminal": None,
            }
        )

    monkeypatch.setattr(
        live_sglang.CompileLaunchManifest,
        "load",
        classmethod(lambda _cls, path: launches[str(Path(path).resolve())]),
    )
    monkeypatch.setattr(live_sglang, "load_run_config", lambda _path: run_config)
    monkeypatch.setattr(
        PinnedBenchServingTransport,
        "from_checkout",
        classmethod(lambda _cls, _path: _overlap_transport()),
    )
    group_dir = (tmp_path / "group").resolve()
    group_dir.mkdir(mode=0o700)
    group = asyncio.run(
        execute_unsigned_native_serving_group(
            specs=(specs[0], specs[1]),
            nvidia_smi_tool=tool,
            inventory_sha256=inventory,
            before_gpu_snapshot_output_path=group_dir / "before.json",
            ready_gpu_snapshot_output_path=group_dir / "ready.json",
            after_gpu_snapshot_output_path=group_dir / "after.json",
            group_receipt_output_path=group_dir / "receipt.json",
            fatal_output_path=group_dir / "fatal.json",
            timeout_seconds=20.0,
        )
    )
    success_rows = [
        FormalPreflightInterferenceRawRow(
            **success_inputs[slot],
            live_run_receipt=group.runs[slot].receipt_binding,
            raw_terminal=CanonicalJsonProofBinding.bind(
                specs[slot].terminal_output_path
            ),
            native_itl_pointer_artifact=CanonicalJsonProofBinding.bind(
                specs[slot].native_itl_pointer_output_path
            ),
            concurrent_group_receipt=group.receipt_binding,
        )
        for slot in range(2)
    ]

    error_rows = [
        _error_row(
            tmp_path,
            mode="isolated",
            repetition=repetition,
            slot=slot,
            inventory_sha256=inventory,
        )
        for repetition in range(2)
        for slot in range(2)
    ]
    group_source = _json_binding(
        (tmp_path / "failed-group-source.json").resolve(),
        {
            "schema_version": 1,
            "kind": "unsigned_pinned_sglang_concurrent_group_fatal_pointer",
            "protocol_sha256": PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
            "status": "ERROR",
            "formal_execution_authorized": False,
            "reason_code": "preflight_test_failure",
            "inventory_sha256": inventory,
            "run_binding_sha256s": [
                content_sha256(_binding("concurrent", 1, slot).begin_payload())
                for slot in range(2)
            ],
        },
    )
    error_rows.extend(
        _error_row(
            tmp_path,
            mode="concurrent",
            repetition=1,
            slot=slot,
            inventory_sha256=inventory,
            shared_group_source=group_source,
        )
        for slot in range(2)
    )
    rows = tuple(
        sorted((*success_rows, *error_rows), key=lambda row: row.registry_cell_id)
    )
    batch = FormalPreflightInterferenceRawBatch(
        schema_version=3,
        kind="formal_preflight_interference_raw_batch",
        protocol_sha256=FORMAL_PREFLIGHT_INTERFERENCE_RAW_PROTOCOL_SHA256,
        dispatch_sha256=_sha("dispatch"),
        registry_sha256=_sha("registry"),
        activation_sha256=_sha("activation"),
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
        inventory_sha256=inventory,
        nvidia_smi_tool=tool,
        status="ERROR",
        rows=rows,
    )
    batch.revalidate()
    assert group.receipt.overlap_duration_ns > 0
    assert (
        group.receipt.server_process_group_ids[0]
        != (group.receipt.server_process_group_ids[1])
    )

    reused_second_repetition = tuple(
        FormalPreflightInterferenceRawRow(
            **{
                **success_rows[slot].__dict__,
                "materialized_cell_id": _sha(f"materialized-concurrent-1-{slot}"),
                "registry_cell_id": _sha(f"cell-concurrent-1-{slot}"),
                "assignment_sha256": _sha(f"assignment-concurrent-1-{slot}"),
                "experiment_budget_sha256": _sha(f"budget-concurrent-1-{slot}"),
                "repetition": 1,
                "run_binding": _binding("concurrent", 1, slot),
            }
        )
        for slot in range(2)
    )
    reused = tuple(
        sorted(
            (
                *success_rows,
                *(row for row in error_rows if row.mode == "isolated"),
                *reused_second_repetition,
            ),
            key=lambda row: row.registry_cell_id,
        )
    )
    with pytest.raises(ValueError, match="reused a group receipt"):
        FormalPreflightInterferenceRawBatch(**{**batch.__dict__, "rows": reused})

    group_path = Path(group.receipt_binding.absolute_path)
    group_path.write_bytes(group_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="canonical JSON|SHA-256"):
        batch.revalidate()


def test_raw_batch_rejects_duplicate_run_nonce_without_opening_gpu(
    tmp_path: Path,
) -> None:
    tool = _dynamic_two_gpu_tool(
        tmp_path,
        pid_paths=(tmp_path / "missing-0", tmp_path / "missing-1"),
    )
    inventory = _sha("inventory")
    rows = tuple(
        sorted(
            (
                _error_row(
                    tmp_path,
                    mode=mode,
                    repetition=repetition,
                    slot=slot,
                    inventory_sha256=inventory,
                )
                for mode in ("isolated", "concurrent")
                for repetition in range(2)
                for slot in range(2)
            ),
            key=lambda row: row.registry_cell_id,
        )
    )
    duplicate = FormalPreflightInterferenceRawRow(
        **{
            **rows[1].__dict__,
            "run_binding": NativeTerminalRunBinding(
                **{
                    **rows[1].run_binding.__dict__,
                    "run_nonce_sha256": rows[0].run_binding.run_nonce_sha256,
                }
            ),
        }
    )
    changed = (rows[0], duplicate, *rows[2:])
    with pytest.raises(ValueError, match="run/replay identities repeat"):
        FormalPreflightInterferenceRawBatch(
            schema_version=3,
            kind="formal_preflight_interference_raw_batch",
            protocol_sha256=FORMAL_PREFLIGHT_INTERFERENCE_RAW_PROTOCOL_SHA256,
            dispatch_sha256=_sha("dispatch"),
            registry_sha256=_sha("registry"),
            activation_sha256=_sha("activation"),
            runtime_sha256=_sha("runtime"),
            split_sha256=_sha("split"),
            inventory_sha256=inventory,
            nvidia_smi_tool=tool,
            status="ERROR",
            rows=changed,
        )


def test_sequential_concurrent_intervals_and_equal_but_bad_slo_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="did not overlap"):
        _derive_actual_group_overlap(
            shared_origin_ns=100,
            scored_intervals=((110, 120), (121, 130)),
        )

    registry = build_industrial_registry()
    safety = (
        ("exactness_violations", 0),
        ("version_mismatches", 0),
        ("fallbacks", 0),
        ("nonfinite_updates", 0),
        ("oom_events", 0),
        ("retractions", 0),
        ("communicator_failures", 0),
    )
    rows: list[FormalPreflightInterferenceProofRow] = []
    interference_cells = tuple(
        cell
        for cell in registry.cells_for("preflight")
        if cell.identity.task == "simultaneous_single_gpu_interference"
    )
    assert len(interference_cells) == 8
    for cell in interference_cells:
        proof = _json_binding(
            (tmp_path / f"proof-{cell.cell_id}.json").resolve(),
            {"proof_cell_id": cell.cell_id},
        )
        variant = str(cell.identity.variant)
        mode = "concurrent" if variant.startswith("concurrent") else "isolated"
        repetition = int(cell.identity.block)
        slot = int(variant.rsplit("_", 1)[1])
        if mode == "isolated":
            sequence = repetition * 2 + slot
            started_ns = 100 + sequence * 100
        else:
            started_ns = 1_000 + repetition * 100
        finished_ns = started_ns + 50
        request_id = f"score-{repetition}-{slot}"
        observation = InterferenceRawObservation(
            observation_id=f"preflight-{mode}-{repetition}-{slot}",
            terminal_authority_sha256=proof.semantic_sha256,
            mode=mode,
            repetition=repetition,
            slot=slot,
            started_ns=started_ns,
            finished_ns=finished_ns,
            request_ids=(request_id,),
            token_trajectory_sha256=_sha(f"trajectory-{repetition}-{slot}"),
            completed_requests=1,
            output_tokens=2,
            goodput_tps=10.0,
            p99_itl_ms=10.0,
            safety_counters=safety,
            hardware_valid=True,
        )
        rows.append(
            FormalPreflightInterferenceProofRow(
                materialized_cell_id=_sha(f"materialized-{cell.cell_id}"),
                registry_cell_id=cell.cell_id,
                assignment_sha256=_sha(f"assignment-{cell.cell_id}"),
                experiment_budget_sha256=_sha(f"budget-{cell.cell_id}"),
                gpu_uuid=f"GPU-{slot}",
                mode=mode,
                repetition=repetition,
                slot=slot,
                run_binding=_binding(mode, repetition, slot),
                load_plan_sha256=_sha("load-plan"),
                topology_sha256=_sha("topology"),
                hardware_envelope_sha256=_sha("hardware"),
                native_result_proof=proof,
                native_itl_proof=proof,
                slo_accounting_sha256=_sha(f"slo-{cell.cell_id}"),
                slo_status="PASS",
                qualified_request_ids=(request_id,),
                observation=observation,
            )
        )
    exact_rows = tuple(rows)
    passing = _diagnose(
        exact_rows,
        registry=registry,
        inventory_sha256=_sha("inventory"),
        hardware_envelope_sha256=_sha("hardware"),
    )
    assert passing.status == "PASS"

    bad = (replace(exact_rows[0], slo_status="FAIL"), *exact_rows[1:])
    failing = _diagnose(
        bad,
        registry=registry,
        inventory_sha256=_sha("inventory"),
        hardware_envelope_sha256=_sha("hardware"),
    )
    assert failing.status == "FAIL"
    assert "request_slo_qualification_failed" in failing.reason_codes


def test_exact_eight_first_party_remote_phase_is_explicitly_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = build_industrial_registry()
    inventory = _inventory()
    activation = SimpleNamespace(
        sha256=_sha("activation"),
        runtime_sha256=_sha("runtime"),
        split_sha256=_sha("split"),
    )
    source_bindings = tuple(
        sorted(
            {
                "burstgpt_shape": _sha("burstgpt"),
                "compile_qualification": _sha("compile"),
                "exactness_qualification": _sha("exactness"),
                "formal_workload_e0": _sha("workload-e0"),
                "formal_workload_e3a": _sha("workload-e3a"),
                "native_runtime_qualification": _sha("native"),
                "offline_release_trust_root": _sha("root"),
                "prepared_model_content": _sha("prepared"),
            }.items()
        )
    )
    bindings: list[FormalPreflightExecutionBinding] = []
    next_port = 31_000
    for cell in registry.cells_for("preflight"):
        item = registry_pool_work_item(cell, estimated_duration_seconds=1.0)
        if cell.identity.task == "simultaneous_single_gpu_interference":
            slot = int(str(cell.identity.variant).rsplit("_", 1)[1])
            gpu_uuids = (f"GPU-{slot}",)
            runner_kind = "first_party_interference"
        elif cell.identity.task == "exactness_memory_telemetry_preflight":
            gpu_uuids = ("GPU-0", "GPU-1")
            runner_kind = "first_party_exactness"
        else:
            gpu_uuids = ("GPU-0", "GPU-1")
            runner_kind = "first_party_compile"
        tp = item.claim.gang_shape.tensor_parallel_size
        rank_groups = tuple(
            gpu_uuids[index : index + tp] for index in range(0, len(gpu_uuids), tp)
        )
        ports = tuple(range(next_port, next_port + item.claim.port_count))
        next_port += item.claim.port_count
        assignment = GpuAssignment(item, gpu_uuids, rank_groups, ports)
        bindings.append(
            FormalPreflightExecutionBinding(
                materialized_cell_id=_sha(f"materialized-{cell.cell_id}"),
                registry_cell_id=cell.cell_id,
                runner_kind=runner_kind,  # type: ignore[arg-type]
                work_item_sha256=item.sha256,
                assignment_sha256=assignment.sha256,
                experiment_budget_sha256=_sha(f"budget-{cell.cell_id}"),
                source_authority_bindings=source_bindings,
                cell=cell,
                assignment=assignment,
                gpu_uuids=gpu_uuids,
                rank_groups=rank_groups,
            )
        )
    dispatch_plan = SimpleNamespace(sha256=_sha("execution-plan"))
    token = SimpleNamespace(
        sha256=_sha("dispatch-token"),
        protocol_lock=SimpleNamespace(sha256=_sha("protocol-lock")),
        manifest=SimpleNamespace(registry_sha256=registry.sha256),
        subject=SimpleNamespace(
            execution_bindings=tuple(bindings),
            inventory_sha256=inventory.sha256,
            budget_plan_sha256=_sha("budget-plan"),
        ),
        dispatch_context=SimpleNamespace(
            registry=registry,
            inventory=inventory,
            activation_artifact=activation,
        ),
        dispatch_plan=dispatch_plan,
    )

    checkout = (tmp_path / "exact8-patched-sglang").resolve()
    checkout.mkdir()
    run_config_path = (tmp_path / "exact8-run-config.json").resolve()
    run_config_path.write_text("{}\n", encoding="utf-8")
    run_config = SimpleNamespace(
        method="static",
        adaptation=None,
        model=SimpleNamespace(target="target/test"),
        runtime=RuntimeConfig(
            sampling_profile_sha256=_sha("sampling-profile"),
            max_running_requests=2,
        ),
    )
    pid_rows: list[tuple[str, Path]] = []
    launches: dict[str, object] = {}
    execution_inputs: dict[str, FormalPreflightInterferenceRunInput] = {}
    interference_bindings = tuple(
        row for row in bindings if row.runner_kind == "first_party_interference"
    )
    for binding in interference_bindings:
        variant = str(binding.cell.identity.variant)
        mode = "concurrent" if variant.startswith("concurrent") else "isolated"
        repetition = int(binding.cell.identity.block)
        slot = int(variant.rsplit("_", 1)[1])
        run_binding = NativeTerminalRunBinding(
            **{
                **_binding(mode, repetition, slot).__dict__,
                "execution_plan_sha256": dispatch_plan.sha256,
            }
        )
        config, warmup, scored = _fake_live_server_configuration(run_binding)
        server_source = (
            tmp_path / f"exact8-server-{binding.registry_cell_id}.py"
        ).resolve()
        server_source.write_text(_FAKE_SERVER_SOURCE, encoding="utf-8")
        server_config = (
            tmp_path / f"exact8-server-{binding.registry_cell_id}.json"
        ).resolve()
        server_config.write_text(
            json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        port = binding.assignment.ports[0]
        launch_path = (
            tmp_path / f"exact8-launch-{binding.registry_cell_id}.json"
        ).resolve()
        launch_binding = _json_binding(
            launch_path,
            {"registry_cell_id": binding.registry_cell_id},
        )
        server_argv = (
            str(Path(sys.executable).resolve()),
            str(server_source),
            str(port),
            str(server_config),
            str((tmp_path / f"exact8-server-{binding.registry_cell_id}.pid").resolve()),
            f"GPU-{slot}",
            "--disable-cuda-graph",
        )
        pid_rows.append((f"GPU-{slot}", Path(server_argv[4])))
        environment = {
            "PATH": str(Path(sys.executable).parent),
            "LD_LIBRARY_PATH": "",
            "CUDA_HOME": str(tmp_path),
            "CUDA_PATH": str(tmp_path),
            "CUDA_VISIBLE_DEVICES": f"GPU-{slot}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "LANG": "C",
            "LC_ALL": "C",
        }
        launch = SimpleNamespace(
            sha256=launch_binding.semantic_sha256,
            run_config_path=str(run_config_path),
            target_model_id="target/test",
            patched_sglang_checkout=str(checkout),
            patched_sglang_commit=PINNED_SGLANG_COMMIT,
            patched_sglang_tree=PINNED_SGLANG_TREE,
            server_argv=server_argv,
            server_argv_sha256=_sha(f"argv-{binding.registry_cell_id}"),
            physical_assignment_sha256=binding.assignment_sha256,
            experiment_budget_sha256=binding.experiment_budget_sha256,
            inventory_sha256=inventory.sha256,
            gpu_uuids=(f"GPU-{slot}",),
            localhost_port=port,
            child_environment=lambda value=environment: dict(value),
        )
        launches[str(launch_path)] = launch
        execution_inputs[binding.registry_cell_id] = (
            FormalPreflightInterferenceRunInput(
                registry_cell_id=binding.registry_cell_id,
                launch_manifest_path=str(launch_path),
                run_binding=run_binding,
                warmup_requests=warmup,  # type: ignore[arg-type]
                scored_requests=scored,  # type: ignore[arg-type]
                qualification_rows=(
                    FormalPreflightInterferenceQualificationRow(
                        request_id="score-0",
                        prompt_bucket="short",
                        eligible=True,
                    ),
                ),
            )
        )

    monkeypatch.setattr(formal_preflight_execution, "_verified", lambda _: token)
    preflight_module = __import__(
        "lightcone_spec.experiments.preflight_interference",
        fromlist=["require_verified_formal_preflight_dispatch"],
    )
    monkeypatch.setattr(
        preflight_module,
        "require_verified_formal_preflight_dispatch",
        lambda _: token,
    )
    monkeypatch.setattr(
        live_sglang.CompileLaunchManifest,
        "load",
        classmethod(lambda _cls, path: launches[str(Path(path).resolve())]),
    )
    monkeypatch.setattr(live_sglang, "load_run_config", lambda _path: run_config)
    monkeypatch.setattr(
        PinnedBenchServingTransport,
        "from_checkout",
        classmethod(lambda _cls, _path: _overlap_transport()),
    )
    binding_by_slot = {
        formal_preflight_execution._interference_mode_repetition_slot(binding): binding
        for binding in interference_bindings
    }
    cap_by_cell: dict[str, object] = {}
    for mode in ("isolated", "concurrent"):
        for repetition in range(2):
            members = tuple(
                binding_by_slot[(mode, repetition, slot)] for slot in range(2)
            )
            for binding in members:
                cap_by_cell[binding.registry_cell_id] = SimpleNamespace(
                    runner_kind="first_party_interference",
                    wave_index=(
                        repetition * 2
                        + int(str(binding.cell.identity.variant).rsplit("_", 1)[1])
                        if mode == "isolated"
                        else 4 + repetition
                    ),
                    wave_cell_ids=(
                        (binding.materialized_cell_id,)
                        if mode == "isolated"
                        else tuple(sorted(row.materialized_cell_id for row in members))
                    ),
                    process_hard_timeout_ns=20_000_000_000,
                )
    cap_schedule = SimpleNamespace(
        protocol_lock_sha256=token.protocol_lock.sha256,
        registry_sha256=token.manifest.registry_sha256,
        inventory_sha256=token.subject.inventory_sha256,
        budget_plan_sha256=token.subject.budget_plan_sha256,
        cap_for_registry_cell=lambda cell_id: cap_by_cell[cell_id],
    )
    monkeypatch.setattr(
        formal_preflight_execution,
        "revalidate_formal_preflight_launch_cap_schedule",
        lambda *_args, **_kwargs: cap_schedule,
    )
    consumption_index = 0

    def consume_wave(*_args, **_kwargs):
        nonlocal consumption_index
        binding = _json_binding(
            (tmp_path / f"launch-consumption-{consumption_index}.json").resolve(),
            {"launch_consumption": consumption_index},
        )
        consumption_index += 1
        return binding

    monkeypatch.setattr(
        formal_preflight_execution,
        "consume_formal_preflight_launch_wave",
        consume_wave,
    )
    tool = _dynamic_many_run_two_gpu_tool(
        tmp_path,
        pid_rows=tuple(pid_rows),
    )
    evidence_root = (tmp_path / "exact8-evidence").resolve()
    evidence_root.mkdir(mode=0o700)
    remote = asyncio.run(
        execute_formal_preflight_interference_raw(
            token,
            launch_cap_schedule_path=(tmp_path / "source-cap-schedule.json").resolve(),
            execution_inputs=execution_inputs,
            nvidia_smi_tool=tool,
            evidence_root=evidence_root,
            now_ns=2_000_000_000,
        )
    )
    assert remote.status == "WAITING_FOR_LOCAL_CONTROL", [
        None if row.fatal_terminal is None else row.fatal_terminal.reopen()
        for row in remote.rows
    ]
    assert len(remote.rows) == 8
    assert all(row.status == "WAITING_FOR_LOCAL_CONTROL" for row in remote.rows)
    concurrent = tuple(row for row in remote.rows if row.mode == "concurrent")
    assert len({row.concurrent_group_receipt for row in concurrent}) == 2

    execution_manifest = FormalPreflightInterferenceExecutionManifest(
        schema_version=1,
        kind="formal_preflight_interference_execution_manifest",
        dispatch_receipt_semantic_sha256=token.sha256,
        inputs=tuple(
            sorted(execution_inputs.values(), key=lambda row: row.registry_cell_id)
        ),
    )
    execution_manifest_path = (tmp_path / "exact8-execution-manifest.json").resolve()
    publish_canonical_json_no_replace(
        execution_manifest_path,
        execution_manifest.to_dict(),
    )

    trusted_row = remote.rows[0]
    assert trusted_row.raw_terminal is not None
    trusted_root_sha256 = _sha("trusted-single-operator-root")
    trusted_raw_proof = (tmp_path / "trusted-current-raw-terminal-proof.json").resolve()
    publish_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact(
        execution_manifest_path=str(execution_manifest_path),
        interference_raw_batch_path=remote.raw_batch.absolute_path,
        raw_terminal_path=trusted_row.raw_terminal.absolute_path,
        materialized_cell_id=trusted_row.materialized_cell_id,
        registry_cell_id=trusted_row.registry_cell_id,
        expected_inventory_sha256=inventory.sha256,
        expected_registry_sha256=registry.sha256,
        expected_root_manifest_sha256=trusted_root_sha256,
        proof_artifact_path=str(trusted_raw_proof),
    )
    trusted_projection = (
        validate_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact(
            str(trusted_raw_proof),
            expected_inventory_sha256=inventory.sha256,
            expected_registry_sha256=registry.sha256,
            expected_root_manifest_sha256=trusted_root_sha256,
            expected_execution_plan_sha256=(
                trusted_row.run_binding.execution_plan_sha256
            ),
            expected_rank_config_sha256=trusted_row.run_binding.rank_config_sha256,
            expected_run_id=trusted_row.run_binding.run_id,
            expected_run_nonce_sha256=trusted_row.run_binding.run_nonce_sha256,
            expected_attempt_id=trusted_row.run_binding.attempt_id,
            expected_method="static",
            now_ns=2_000_000_000,
        )
    )
    assert trusted_projection.scored_request_ids == (
        trusted_row.run_binding.scored_request_ids
    )
    with pytest.raises(ValueError, match="identity differs"):
        validate_formal_single_operator_preflight_tp1_raw_terminal_proof_artifact(
            str(trusted_raw_proof),
            expected_inventory_sha256=inventory.sha256,
            expected_registry_sha256=_sha("foreign-registry"),
            expected_root_manifest_sha256=trusted_root_sha256,
            expected_execution_plan_sha256=(
                trusted_row.run_binding.execution_plan_sha256
            ),
            expected_rank_config_sha256=trusted_row.run_binding.rank_config_sha256,
            expected_run_id=trusted_row.run_binding.run_id,
            expected_run_nonce_sha256=trusted_row.run_binding.run_nonce_sha256,
            expected_attempt_id=trusted_row.run_binding.attempt_id,
            expected_method="static",
            now_ns=2_000_000_000,
        )

    hardware_sha256 = inventory.devices[0].hardware_envelope_sha256
    assert {device.hardware_envelope_sha256 for device in inventory.devices} == {
        hardware_sha256
    }
    control_context = _release_control_context(
        monkeypatch,
        hardware_envelope_sha256=hardware_sha256,
    )
    monkeypatch.setattr(
        preflight_module,
        "load_source_release_ed25519_root",
        lambda: control_context.root_binding,
    )
    replay_root = (tmp_path / "exact8-replay").resolve()
    replay_root.mkdir(mode=0o700)
    replay_store = ChallengeReplayStore(str(replay_root))
    now_ns = 2_000_000_000

    result_controls: list[ControlArtifactAttestation] = []
    for row in remote.rows:
        assert row.raw_terminal is not None
        terminal_control_binding = build_native_terminal_external_control_binding(
            row.raw_terminal.reopen(),
            trusted_attester_policy=NO_TRUSTED_ATTESTERS,
            inventory_sha256=inventory.sha256,
            expected_binding=row.run_binding,
        )
        result_controls.append(
            _control(
                control_context,
                inventory_sha256=inventory.sha256,
                subject=_subject(
                    artifact_type="non_serving_terminal",
                    artifact_sha256=terminal_control_binding.sha256,
                    protocol_sha256=(NATIVE_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256),
                    registry_sha256=registry.sha256,
                    lineage_sha256=terminal_control_binding.lineage_sha256,
                ),
                label=f"result-{row.registry_cell_id}",
            )
        )

    bad_result_paths = tuple(
        str((tmp_path / f"bad-result-{index}.json").resolve()) for index in range(8)
    )
    wrong_control = _control(
        control_context,
        inventory_sha256=inventory.sha256,
        subject=_subject(
            artifact_type="non_serving_terminal",
            artifact_sha256=_sha("wrong-terminal-control-subject"),
            protocol_sha256=NATIVE_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256,
            registry_sha256=registry.sha256,
            lineage_sha256=_sha("wrong-terminal-lineage"),
        ),
        label="wrong-result",
    )
    reservations_before = tuple(replay_root.glob("reservation-*.json"))
    with pytest.raises(ValueError, match="subject is not exact"):
        publish_native_terminal_result_proof_artifacts(
            tuple(
                row.raw_terminal.absolute_path  # type: ignore[union-attr]
                for row in remote.rows
            ),
            control_attestations=(wrong_control, *result_controls[1:]),
            replay_store=replay_store,
            expected_inventory_sha256=inventory.sha256,
            expected_registry_sha256=registry.sha256,
            expected_root_manifest_sha256=(
                control_context.root_binding.semantic_sha256
            ),
            now_ns=now_ns,
            proof_artifact_paths=bad_result_paths,
            expected_bindings=tuple(row.run_binding for row in remote.rows),
        )
    assert not any(Path(path).exists() for path in bad_result_paths)
    assert tuple(replay_root.glob("reservation-*.json")) == reservations_before

    result_paths = tuple(
        str((tmp_path / f"result-{index}.json").resolve()) for index in range(8)
    )
    publish_native_terminal_result_proof_artifacts(
        tuple(
            row.raw_terminal.absolute_path  # type: ignore[union-attr]
            for row in remote.rows
        ),
        control_attestations=tuple(result_controls),
        replay_store=replay_store,
        expected_inventory_sha256=inventory.sha256,
        expected_registry_sha256=registry.sha256,
        expected_root_manifest_sha256=control_context.root_binding.semantic_sha256,
        now_ns=now_ns,
        proof_artifact_paths=result_paths,
        expected_bindings=tuple(row.run_binding for row in remote.rows),
    )
    current_result_paths = tuple(
        str((tmp_path / f"current-result-{index}.json").resolve()) for index in range(8)
    )
    for index, row in enumerate(remote.rows):
        publish_formal_current_preflight_tp1_terminal_result_proof_artifact(
            execution_manifest_path=str(execution_manifest_path),
            interference_raw_batch_path=remote.raw_batch.absolute_path,
            native_result_proof_path=result_paths[index],
            materialized_cell_id=row.materialized_cell_id,
            registry_cell_id=row.registry_cell_id,
            expected_inventory_sha256=inventory.sha256,
            expected_registry_sha256=registry.sha256,
            expected_root_manifest_sha256=(
                control_context.root_binding.semantic_sha256
            ),
            now_ns=now_ns,
            proof_artifact_path=current_result_paths[index],
        )
    result_paths = current_result_paths
    result_by_cell = {
        row.registry_cell_id: result_paths[index]
        for index, row in enumerate(remote.rows)
    }

    capability = NATIVE_RUNTIME_RELEASE_CAPABILITY
    gpu_receipt = NativeRuntimeGpuProofReceipt(
        schema_version=1,
        kind="lightcone_native_runtime_gpu_proof",
        suite_id="native_hot_path_tp1",
        topology_mode="tp1_dp1",
        topology_sha256=_sha("native-topology"),
        runner_protocol_sha256=(
            NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S["native_hot_path_tp1"]
        ),
        assignment_sha256=_sha("native-assignment"),
        qualification_observation_sha256=_sha("native-observation"),
        source_capability_sha256=capability.sha256,
        pinned_sglang_commit=capability.pinned_sglang_commit,
        patched_sglang_tree=capability.patched_sglang_tree,
        semantic_patch_sha256=capability.semantic_patch_sha256,
        run_nonce_sha256=_sha("native-run-nonce"),
        qualification_authority_sha256=(NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256),
        source_identity_sha256=_sha("native-source-identity"),
        inventory_sha256=inventory.sha256,
        gpu_uuids=("GPU-0",),
        hardware_envelope_sha256=hardware_sha256,
        junit_xml_sha256=_sha("native-junit"),
        test_names=NATIVE_RUNTIME_QUALIFICATION_TESTS["native_hot_path_tp1"],
        tests_collected=8,
        tests_passed=8,
        tests_failed=0,
        tests_errored=0,
        tests_skipped=0,
    )
    gpu_receipt_path = str((tmp_path / "native-gpu-receipt.json").resolve())
    gpu_receipt_binding = gpu_receipt.write_unsigned(gpu_receipt_path)
    gpu_control = _control(
        control_context,
        inventory_sha256=inventory.sha256,
        subject=_subject(
            artifact_type="non_serving_terminal",
            artifact_sha256=gpu_receipt_binding.raw_sha256,
            protocol_sha256=capability.suite_protocol_sha256,
            registry_sha256=gpu_receipt.source_identity_sha256,
            lineage_sha256=gpu_receipt.control_lineage_sha256,
        ),
        label="native-gpu",
    )
    verified_gpu = verify_native_runtime_gpu_proof(
        gpu_receipt_path,
        control_attestation=gpu_control,
        replay_store=replay_store,
        expected_suite_id="native_hot_path_tp1",
        expected_topology_sha256=gpu_receipt.topology_sha256,
        expected_source_identity_sha256=gpu_receipt.source_identity_sha256,
        expected_inventory_sha256=inventory.sha256,
        expected_gpu_uuids=("GPU-0",),
        expected_hardware_envelope_sha256=hardware_sha256,
        expected_run_nonce_sha256=gpu_receipt.run_nonce_sha256,
        now_ns=now_ns,
    )
    gpu_artifact = build_native_runtime_gpu_proof_artifact(
        receipt_path=gpu_receipt_path,
        control_attestation=gpu_control,
        replay_store=replay_store,
        verified_proof=verified_gpu,
    )
    gpu_proof_path = str((tmp_path / "native-gpu-proof.json").resolve())
    publish_canonical_json_no_replace(gpu_proof_path, gpu_artifact.to_dict())

    itl_by_cell: dict[str, str] = {}
    itl_requests: list[StageItlTimestampProofRequest] = []
    for index, row in enumerate(remote.rows):
        assert row.native_itl_pointer_artifact is not None
        pointer_bundle = row.native_itl_pointer_artifact.reopen()
        pointers = tuple(pointer_bundle["native_result_pointers"])
        execution_identity = StageItlExecutionIdentity(
            schema_version=1,
            kind="stage_itl_execution_identity",
            materialized_cell_id=row.materialized_cell_id,
            inventory_sha256=inventory.sha256,
            registry_sha256=registry.sha256,
            execution_plan_sha256=row.run_binding.execution_plan_sha256,
            rank_config_sha256=row.run_binding.rank_config_sha256,
            run_id=row.run_binding.run_id,
            run_nonce_sha256=row.run_binding.run_nonce_sha256,
            attempt_id=row.run_binding.attempt_id,
            method="static",
        )
        raw_itl_path = str((tmp_path / f"raw-itl-{index}.json").resolve())
        publish_stage_itl_timestamp_raw_receipt(
            raw_itl_path,
            native_result_proof_path=result_paths[index],
            native_gpu_proof_path=gpu_proof_path,
            execution_identity=execution_identity,
            expected_root_manifest_sha256=(
                control_context.root_binding.semantic_sha256
            ),
            native_result_pointers=pointers,
            now_ns=now_ns,
        )
        itl_control_binding = build_stage_itl_external_control_binding(
            raw_itl_path,
            native_result_proof_path=result_paths[index],
            native_gpu_proof_path=gpu_proof_path,
            execution_identity=execution_identity,
            expected_root_manifest_sha256=(
                control_context.root_binding.semantic_sha256
            ),
            now_ns=now_ns,
        )
        itl_control = _control(
            control_context,
            inventory_sha256=inventory.sha256,
            subject=_subject(
                artifact_type="non_serving_terminal",
                artifact_sha256=itl_control_binding.sha256,
                protocol_sha256=STAGE_ITL_TIMESTAMP_PROOF_PROTOCOL_SHA256,
                registry_sha256=registry.sha256,
                lineage_sha256=itl_control_binding.lineage_sha256,
            ),
            label=f"itl-{row.registry_cell_id}",
        )
        itl_path = str((tmp_path / f"itl-proof-{index}.json").resolve())
        itl_requests.append(
            StageItlTimestampProofRequest(
                raw_receipt_path=raw_itl_path,
                native_result_proof_path=result_paths[index],
                native_gpu_proof_path=gpu_proof_path,
                execution_identity=execution_identity,
                control_attestation=itl_control,
                proof_artifact_path=itl_path,
            )
        )
        itl_by_cell[row.registry_cell_id] = itl_path

    wrong_itl_control = _control(
        control_context,
        inventory_sha256=inventory.sha256,
        subject=_subject(
            artifact_type="non_serving_terminal",
            artifact_sha256=_sha("wrong-itl-control-subject"),
            protocol_sha256=STAGE_ITL_TIMESTAMP_PROOF_PROTOCOL_SHA256,
            registry_sha256=registry.sha256,
            lineage_sha256=_sha("wrong-itl-lineage"),
        ),
        label="wrong-itl",
    )
    bad_itl_paths = tuple(
        str((tmp_path / f"bad-itl-{index}.json").resolve()) for index in range(8)
    )
    bad_itl_requests = tuple(
        replace(
            request,
            control_attestation=(
                wrong_itl_control if index == 0 else request.control_attestation
            ),
            proof_artifact_path=bad_itl_paths[index],
        )
        for index, request in enumerate(itl_requests)
    )
    reservations_before = tuple(replay_root.glob("reservation-*.json"))
    with pytest.raises(ValueError, match="control subject is not exact"):
        publish_stage_itl_timestamp_proof_artifacts(
            bad_itl_requests,
            replay_store=replay_store,
            expected_root_manifest_sha256=(
                control_context.root_binding.semantic_sha256
            ),
            now_ns=now_ns,
        )
    assert tuple(replay_root.glob("reservation-*.json")) == reservations_before
    assert not any(Path(path).exists() for path in bad_itl_paths)

    published_itl = publish_stage_itl_timestamp_proof_artifacts(
        tuple(itl_requests),
        replay_store=replay_store,
        expected_root_manifest_sha256=(control_context.root_binding.semantic_sha256),
        now_ns=now_ns,
    )
    assert len(published_itl) == 8
    assert len({binding.semantic_sha256 for binding in published_itl}) == 8
    itl_reservations = {
        artifact.reopen()["replay_reservation"]["reservation_sha256"]
        for artifact in published_itl
    }
    assert len(itl_reservations) == 1

    aggregate_binding = build_formal_preflight_interference_aggregate_binding(
        token,
        raw_batch_path=remote.raw_batch.absolute_path,
        native_result_proof_paths=result_by_cell,
        native_itl_proof_paths=itl_by_cell,
        expected_root_manifest_sha256=(control_context.root_binding.semantic_sha256),
        now_ns=now_ns,
    )
    assert aggregate_binding.status == "PASSED"
    aggregate_control = _control(
        control_context,
        inventory_sha256=inventory.sha256,
        subject=_subject(
            artifact_type="interference",
            artifact_sha256=aggregate_binding.sha256,
            protocol_sha256=(FORMAL_PREFLIGHT_INTERFERENCE_PROOF_PROTOCOL_SHA256),
            registry_sha256=registry.sha256,
            lineage_sha256=aggregate_binding.lineage_sha256,
        ),
        label="aggregate",
    )
    aggregate_path = str((tmp_path / "interference-proof.json").resolve())
    publish_formal_preflight_interference_proof_artifact(
        token,
        raw_batch_path=remote.raw_batch.absolute_path,
        native_result_proof_paths=result_by_cell,
        native_itl_proof_paths=itl_by_cell,
        control_attestation=aggregate_control,
        replay_store=replay_store,
        now_ns=now_ns,
        proof_artifact_path=aggregate_path,
    )
    verified = validate_formal_preflight_interference_proof_artifact(
        aggregate_path,
        registry=registry,
        expected_activation_sha256=activation.sha256,
        expected_runtime_sha256=activation.runtime_sha256,
        expected_split_sha256=activation.split_sha256,
        expected_inventory_sha256=inventory.sha256,
        now_ns=now_ns + 1,
    )
    assert verified.status == "PASSED"

    replay_output = (tmp_path / "interference-proof-replay.json").resolve()
    with pytest.raises(ValueError, match="already consumed"):
        publish_formal_preflight_interference_proof_artifact(
            token,
            raw_batch_path=remote.raw_batch.absolute_path,
            native_result_proof_paths=result_by_cell,
            native_itl_proof_paths=itl_by_cell,
            control_attestation=aggregate_control,
            replay_store=replay_store,
            now_ns=now_ns,
            proof_artifact_path=replay_output,
        )
    assert not replay_output.exists()
