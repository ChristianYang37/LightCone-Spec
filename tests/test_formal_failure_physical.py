from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_control_attestation import NOW_NS, _bundle, _root_binding
from test_formal_physical_dispatch import _install_materialization_fakes
from test_stage_itl_proof import _authorization, _control

import lightcone_spec.experiments.failure_actuator as failure_source
import lightcone_spec.experiments.formal_failure_actuator as actuator
import lightcone_spec.orchestration.formal_failure_physical as physical
import lightcone_spec.orchestration.formal_physical_dispatch as dispatch
import lightcone_spec.runtime.release_trust_root as root_module
from lightcone_spec.experiments.registry import content_sha256
from lightcone_spec.orchestration.formal_single_operator_admission import (
    FormalSingleOperatorAdmission,
    publish_formal_single_operator_admission,
)
from lightcone_spec.runtime.control_attestation import ChallengeReplayStore
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return content_sha256({"formal-failure-physical-test": label})


_FAKE_SERVER = r"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lightcone_spec.experiments.failure_actuator import (
    FAILURE_ACTUATOR_PROTOCOL_SHA256,
    FailureActuatorContext,
    failure_phase_observation_sha256,
    failure_semantics,
)

PORT = int(sys.argv[1])
TOPOLOGY = "tp1_dp1"
HOOK = "sglang.lightcone_e5_failure_actuator.v1"
QUOTA = os.path.join(
    os.environ["LIGHTCONE_FAILURE_ACTUATOR_QUOTA_ROOT_BASE"],
    "tp1_dp1-rank-0",
)
os.mkdir(QUOTA, 0o700)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def _send(self, value):
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health_generate":
            self._send({"status": "ok"})
            return
        if self.path == "/v1/lightcone-spec/failure-actuator/capability":
            self._send({
                "schema_version": 1,
                "hook": HOOK,
                "topology": TOPOLOGY,
                "world_size": 1,
                "rank_bindings": [{
                    "schema_version": 1,
                    "hook": HOOK,
                    "protocol_sha256": FAILURE_ACTUATOR_PROTOCOL_SHA256,
                    "assignment_sha256": os.environ[
                        "LIGHTCONE_FAILURE_ACTUATOR_ASSIGNMENT_SHA256"
                    ],
                    "inventory_sha256": os.environ[
                        "LIGHTCONE_FAILURE_ACTUATOR_INVENTORY_SHA256"
                    ],
                    "plan_sha256": os.environ[
                        "LIGHTCONE_FAILURE_ACTUATOR_PLAN_SHA256"
                    ],
                    "run_nonce_sha256": os.environ[
                        "LIGHTCONE_FAILURE_ACTUATOR_RUN_NONCE_SHA256"
                    ],
                    "topology": TOPOLOGY,
                    "rank": 0,
                    "world_size": 1,
                    "process_id": os.getpid(),
                    "process_group_id": os.getpgrp(),
                    "process_start_monotonic_ns": 1,
                    "gpu_uuid": "GPU-test-0",
                    "temp_quota_root": QUOTA,
                }],
            })
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/lightcone-spec/failure-actuator":
            self.send_error(404)
            return
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        semantics = failure_semantics(payload["scenario"])
        context = FailureActuatorContext(
            plan_sha256=payload["plan_sha256"],
            scenario_semantics_sha256=payload["scenario_semantics_sha256"],
            topology=payload["topology"],
            rank=payload["rank"],
            world_size=1,
            process_id=payload["process_id"],
            process_start_monotonic_ns=payload["process_start_monotonic_ns"],
            session_epoch=0,
            run_nonce_sha256=payload["run_nonce_sha256"],
        )
        response = {
            "phase": payload["phase"],
            "operation": payload["operation"],
            "monotonic_ns": time.monotonic_ns(),
            "event_count": 1,
            "observation_sha256": failure_phase_observation_sha256(
                context,
                semantics,
                phase=payload["phase"],
                operation=payload["operation"],
                event_count=1,
            ),
        }
        if payload["phase"] == "terminal":
            response["counters"] = {semantics.terminal_counter: 1}
            response["recovery_valid"] = True
        self._send(response)


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
"""


class _RealHttpAdmin:
    @classmethod
    def from_checkout(cls, _checkout):
        return cls()

    async def open(self, **_kwargs):
        return None

    async def close(self):
        return None

    def bind_native_admin_base_url(self, value):
        self.base_url = value

    async def get_json(self, path):
        def fetch():
            with urllib.request.urlopen(f"{self.base_url}{path}") as response:
                return json.loads(response.read())

        return await asyncio.to_thread(fetch)


def _snapshot(**kwargs):
    phase = kwargs["phase"]
    group = kwargs.get("shared_server_process_group_id")
    value = {
        "schema_version": 1,
        "kind": "unsigned-test-gpu-snapshot",
        "phase": phase,
        "captured_ns": 1,
        "gpu_uuids": list(kwargs["gpu_uuids"]),
        "inventory_sha256": kwargs["inventory_sha256"],
        "server_process_group_ids": None if group is None else [group],
        "nvidia_smi": {"test": True},
        "gpu_rows": [],
        "compute_process_rows": [],
    }
    publish_canonical_json_no_replace(kwargs["output_path"], value)
    return CanonicalJsonProofBinding.bind(kwargs["output_path"])


def test_integrated_failure_runner_uses_one_real_child_and_durable_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (
        root,
        content_path,
        workload_path,
        materialization_path,
        serving,
        compile_launch_path,
        _schedule_source,
        inventory_path,
    ) = _install_materialization_fakes(
        monkeypatch,
        tmp_path,
        stage="E5",
        method="l0",
        method_role="L0-naive",
        task="deterministic_failure_injection",
    )
    plan = dispatch.materialize_formal_serving_run_plan(
        execution_binding=serving,
        content_verification_receipt_path=content_path,
        workload_authority_path=workload_path,
        materialization_path=materialization_path,
        compile_launch_manifest_path=compile_launch_path,
        private_output_root=root,
        now_ns=20,
    )
    plan_path = root / "formal-serving-run-plan.json"
    admission_binding = publish_formal_single_operator_admission(
        plan_path=plan_path,
        inventory_path=inventory_path,
    )
    admission = FormalSingleOperatorAdmission.from_dict(admission_binding.reopen())
    serving.subject.execution_identity.registry_sha256 = admission.registry_sha256
    server_script = (tmp_path / "fake_failure_server.py").resolve()
    server_script.write_text(_FAKE_SERVER, encoding="utf-8")
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    inventory = plan.inventory_sha256
    execution_plan = plan.native_terminal_binding.execution_plan_sha256
    run_nonce = plan.native_terminal_binding.run_nonce_sha256
    cell = plan.materialized_cell_id
    failure_subject = SimpleNamespace(
        assignment_sha256=_sha("assignment"),
        inventory_sha256=inventory,
        registry_sha256=admission.registry_sha256,
        serving_execution_plan_sha256=execution_plan,
        materialized_cell_id=cell,
        scenario="queue_saturation",
        topology="tp1_dp1",
        run_nonce_sha256=run_nonce,
    )
    failure = SimpleNamespace(
        sha256=_sha("failure-binding"),
        subject=failure_subject,
        serving_execution=serving,
    )
    monkeypatch.setattr(
        physical,
        "VerifiedFormalFailureExecutionBinding",
        type(failure),
    )
    launch = SimpleNamespace(
        server_argv=(
            str(Path(sys.executable).resolve()),
            str(server_script),
            str(port),
        ),
        patched_sglang_checkout=str(tmp_path.resolve()),
        localhost_port=port,
        gpu_uuids=("GPU-test-0",),
        inventory_sha256=inventory,
        child_environment=lambda: {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "GPU-test-0",
            "PYTHONPATH": os.pathsep.join(path for path in sys.path if path),
        },
    )
    monkeypatch.setattr(
        physical, "require_verified_formal_serving_execution_binding", lambda x: x
    )
    monkeypatch.setattr(
        physical, "require_verified_formal_failure_execution_binding", lambda x: x
    )
    monkeypatch.setattr(
        actuator, "require_verified_formal_failure_execution_binding", lambda x: x
    )
    monkeypatch.setattr(
        physical,
        "revalidate_formal_serving_run_plan",
        lambda *_a, **_k: (launch, object()),
    )
    monkeypatch.setattr(physical, "_capture_gpu_process_snapshot", _snapshot)
    monkeypatch.setattr(physical, "PinnedBenchServingTransport", _RealHttpAdmin)
    monkeypatch.setattr(
        physical,
        "_observe_live_server_execution_policy",
        lambda **_kwargs: asyncio.sleep(0),
    )

    def revalidate_process(rank):
        os.kill(rank.process_id, 0)
        assert os.getpgid(rank.process_id) == rank.process_group_id
        assert Path(rank.temp_quota_root).is_dir()

    monkeypatch.setattr(
        failure_source, "_revalidate_failure_rank_process", revalidate_process
    )

    result = asyncio.run(
        physical.execute_formal_e5_failure_run_plan(
            plan_path=plan_path,
            launch_admission_path=root / "formal-single-operator-admission.json",
            execution_binding=serving,
            failure_execution_binding=failure,
            nvidia_smi_tool=SimpleNamespace(to_dict=lambda: {"test": True}),
        )
    )
    receipt = physical.validate_formal_e5_failure_lifecycle_raw_receipt(
        result.lifecycle_receipt.absolute_path,
        plan_path=plan_path,
        execution_binding=serving,
        failure_execution_binding=failure,
    )
    raw = result.raw_failure_terminal.reopen()
    assert raw["recovery_receipt"]["recovered"] is True
    assert len(raw["recovery_receipt"]["rank_receipts"][0]["phases"]) == 5
    assert receipt.process_group_empty is True
    assert receipt.raw_failure_terminal == result.raw_failure_terminal
    outcome = physical.validate_formal_single_operator_e5_physical_outcome(
        plan_path=plan_path,
        run_receipt_path=plan.live_run_receipt_output_path,
        lifecycle_receipt_path=result.lifecycle_receipt.absolute_path,
        execution_binding=serving,
    )
    assert outcome.status == "COMPLETE"
    assert outcome.process_exit_code in {0, -15}
    assert outcome.finished_ns >= outcome.started_ns
    outcome.server_stdout.reopen(label="test E5 stdout")
    outcome.server_stderr.reopen(label="test E5 stderr")
    outcome.junit.reopen(label="test E5 JUnit")
    assert outcome.server_stdout.size > 0
    assert outcome.server_stderr.size > 0
    assert b"testsuite" in Path(outcome.junit.absolute_path).read_bytes()

    root_private = Ed25519PrivateKey.generate()
    artifact_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    bundle = _bundle(artifact_private)
    monkeypatch.setattr(
        root_module,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    subject = physical.build_formal_e5_failure_lifecycle_control_subject(
        result.lifecycle_receipt.absolute_path,
        plan_path=plan_path,
        execution_binding=serving,
        failure_execution_binding=failure,
    )
    control = _control(
        artifact_private=artifact_private,
        root_binding=root_binding,
        bundle=bundle,
        authorization=_authorization(
            root_private=root_private,
            root_binding=root_binding,
            bundle=bundle,
            suffix="failure-lifecycle",
            nonce_byte=b"8",
            inventory_sha256=inventory,
        ),
        subject=subject,
        suffix="failure-lifecycle",
        nonce_byte=b"9",
    )
    replay_root = (tmp_path / "replay").resolve()
    replay_root.mkdir(mode=0o700)
    replay_store = ChallengeReplayStore(str(replay_root))
    proof_path = root / "controlled-lifecycle.json"
    proof = physical.publish_formal_e5_failure_lifecycle_proof_artifact(
        result.lifecycle_receipt.absolute_path,
        plan_path=plan_path,
        execution_binding=serving,
        failure_execution_binding=failure,
        control_attestation=control,
        replay_store=replay_store,
        expected_root_manifest_sha256=root_binding.semantic_sha256,
        now_ns=NOW_NS,
        proof_artifact_path=proof_path,
    )
    controlled = physical.validate_formal_e5_failure_lifecycle_proof_artifact(
        proof.absolute_path,
        plan_path=plan_path,
        execution_binding=serving,
        failure_execution_binding=failure,
        expected_root_manifest_sha256=root_binding.semantic_sha256,
        now_ns=NOW_NS,
    )
    assert controlled.sha256 == receipt.sha256

    Path(plan.server_log_output_path).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        physical.validate_formal_e5_failure_lifecycle_raw_receipt(
            result.lifecycle_receipt.absolute_path,
            plan_path=plan_path,
            execution_binding=serving,
            failure_execution_binding=failure,
        )


def test_failure_runner_surface_has_no_caller_launch_or_fault_values() -> None:
    import inspect

    parameters = inspect.signature(
        physical.execute_formal_e5_failure_run_plan
    ).parameters
    assert {
        "scenario",
        "request",
        "token_ids",
        "port",
        "argv",
        "transport",
        "quota_root",
    }.isdisjoint(parameters)
