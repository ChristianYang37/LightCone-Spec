from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import os
import socket
import stat
import subprocess
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_control_attestation import (
    HARDWARE_SHA256 as CONTROL_HARDWARE_SHA256,
)
from test_control_attestation import (
    INVENTORY_SHA256 as CONTROL_INVENTORY_SHA256,
)
from test_control_attestation import NOW_NS as CONTROL_NOW_NS
from test_control_attestation import _bundle as _control_bundle
from test_control_attestation import (
    _deployment_authorization as _control_deployment_authorization,
)
from test_control_attestation import _public_bytes as _control_public_bytes
from test_control_attestation import _root_binding as _control_root_binding

import lightcone_spec.runtime.release_trust_root as release_root_module
from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.config.schema import RuntimeConfig
from lightcone_spec.execution import (
    ControlledExecutionPolicy,
    FixedAddressGraphExecutionPolicy,
)
from lightcone_spec.experiments.serving import PinnedBenchServingTransport
from lightcone_spec.orchestration import live_sglang
from lightcone_spec.orchestration.live_sglang import (
    PINNED_SGLANG_LIFECYCLE_TIMING_PROTOCOL_SHA256,
    PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
    PinnedNvidiaSmiTool,
    PinnedSglangServingRunError,
    PinnedSglangServingRunSpec,
    UnsignedPinnedSglangServingGroupReceipt,
    execute_unsigned_native_serving_group,
    execute_unsigned_native_serving_run,
    publish_pinned_sglang_lifecycle_timing_proof_artifact,
    validate_pinned_sglang_gpu_process_snapshot,
    validate_pinned_sglang_lifecycle_timing_proof_artifact,
    validate_unsigned_pinned_sglang_lifecycle_timing_receipt,
    validate_unsigned_pinned_sglang_serving_group_receipt,
    validate_unsigned_pinned_sglang_serving_run_receipt,
)
from lightcone_spec.orchestration.native_terminal import (
    NATIVE_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256,
    NativeTerminalRunBinding,
    build_native_terminal_external_control_binding,
    publish_native_terminal_result_proof_artifact,
)
from lightcone_spec.runtime.attestation import (
    NO_TRUSTED_ATTESTERS,
    AttestationChallenge,
    SignedAttestation,
    attestation_message,
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
from lightcone_spec.runtime.readiness import NATIVE_RUNTIME_RELEASE_CAPABILITY
from lightcone_spec.runtime.release_trust_root import (
    DeploymentPolicyAuthorization,
    deployment_policy_subject_sha256,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _deployment_authorization_with_nonce(
    *,
    root_private_key: Ed25519PrivateKey,
    root_binding,
    bundle,
    challenge_id: str,
    nonce_byte: bytes,
) -> DeploymentPolicyAuthorization:
    subject_sha256 = deployment_policy_subject_sha256(
        root_manifest_sha256=root_binding.semantic_sha256,
        inventory_sha256=CONTROL_INVENTORY_SHA256,
        bundle_sha256=bundle.sha256,
    )
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id=challenge_id,
        nonce_base64=base64.b64encode(nonce_byte * 32).decode("ascii"),
        subject_sha256=subject_sha256,
        issued_ns=1_500_000_000,
        expires_ns=3_000_000_000,
    )
    return DeploymentPolicyAuthorization(
        schema_version=1,
        kind="lightcone_deployment_policy_authorization",
        root_manifest_sha256=root_binding.semantic_sha256,
        inventory_sha256=CONTROL_INVENTORY_SHA256,
        bundle=bundle,
        challenge=challenge,
        signature_base64=base64.b64encode(
            root_private_key.sign(
                attestation_message(challenge, payload_sha256=bundle.sha256)
            )
        ).decode("ascii"),
    )


def _local_control(
    *,
    private_key: Ed25519PrivateKey,
    root_binding,
    bundle,
    authorization: DeploymentPolicyAuthorization,
    artifact_sha256: str,
    protocol_sha256: str,
    registry_sha256: str,
    lineage_sha256: str,
    challenge_id: str,
    nonce_byte: bytes,
) -> ControlArtifactAttestation:
    subject = ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="non_serving_terminal",
        artifact_sha256=artifact_sha256,
        protocol_sha256=protocol_sha256,
        registry_sha256=registry_sha256,
        lineage_sha256=lineage_sha256,
    )
    challenge = AttestationChallenge(
        schema_version=1,
        kind="lightcone_attestation_challenge",
        challenge_id=challenge_id,
        nonce_base64=base64.b64encode(nonce_byte * 32).decode("ascii"),
        subject_sha256=subject.sha256,
        issued_ns=1_600_000_000,
        expires_ns=2_600_000_000,
    )
    public_key = _control_public_bytes(private_key)
    signature = private_key.sign(
        attestation_message(challenge, payload_sha256=artifact_sha256)
    )
    return ControlArtifactAttestation(
        schema_version=1,
        kind="lightcone_control_artifact_attestation",
        subject=subject,
        hardware_envelope_sha256=CONTROL_HARDWARE_SHA256,
        trust_anchor_sha256=root_binding.sha256,
        trust_bundle_sha256=bundle.sha256,
        trusted_attester_policy_sha256=bundle.trusted_attester_policy.sha256,
        deployment_policy_authorization=authorization,
        challenge=challenge,
        attestation=SignedAttestation(
            schema_version=1,
            kind="lightcone_signed_attestation",
            algorithm="Ed25519",
            attester_id="validation-signer",
            key_id="validation-signer-key",
            environment="release",
            public_key_base64=base64.b64encode(public_key).decode("ascii"),
            challenge_sha256=challenge.sha256,
            payload_sha256=artifact_sha256,
            signature_base64=base64.b64encode(signature).decode("ascii"),
        ),
    )


_FAKE_SERVER_SOURCE = r"""
import copy
import hashlib
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(sys.argv[1])
config_path = sys.argv[2]
pid_path = sys.argv[3]
expected_gpu = sys.argv[4]
if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu:
    raise SystemExit(41)
with open(config_path, "r", encoding="utf-8") as stream:
    config = json.load(stream)
pid = os.getpid()
started_ns = time.time_ns()
with open(pid_path, "x", encoding="ascii") as stream:
    stream.write(str(pid))

def sha(value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(body).hexdigest()

begin = None
reset = None

def begin_response():
    global begin
    begin = copy.deepcopy(config["begin"])
    begin["server_process_id"] = pid
    begin["server_process_started_ns"] = started_ns
    begin.pop("begin_sha256", None)
    begin["begin_sha256"] = sha(begin)
    return begin

def reset_response():
    global reset
    if begin is None:
        raise RuntimeError("begin missing")
    reset = copy.deepcopy(config["reset"])
    reset["server_process_id"] = pid
    reset["server_process_started_ns"] = started_ns
    reset["begin_sha256"] = begin["begin_sha256"]
    reset.pop("reset_sha256", None)
    reset["reset_sha256"] = sha(reset)
    return reset

def terminal_response():
    if reset is None:
        raise RuntimeError("reset missing")
    value = copy.deepcopy(config["terminal"])
    value["server_process_id"] = pid
    value["server_process_started_ns"] = started_ns
    value["reset_receipt_sha256"] = reset["reset_sha256"]
    value.pop("terminal_sha256", None)
    value.pop("attestation", None)
    value["terminal_sha256"] = sha(value)
    message = {
        "schema_version": 1,
        "kind": "lightcone_terminal_attestation_challenge",
        "hook": value["hook"],
        "challenge_nonce_sha256": value["challenge_nonce_sha256"],
        "terminal_sha256": value["terminal_sha256"],
        "run_id": value["run_id"],
        "run_nonce_sha256": value["run_nonce_sha256"],
        "server_process_id": pid,
        "server_process_started_ns": started_ns,
        "session_id": value["session_id"],
        "session_epoch": value["session_epoch"],
        "attempt_id": value["attempt_id"],
    }
    value["attestation"] = {
        "schema_version": 1,
        "status": "UNAVAILABLE",
        "challenge_nonce_sha256": value["challenge_nonce_sha256"],
        "message_sha256": sha(message),
        "attester_id": None,
        "trust_domain": None,
        "signature_hex": None,
    }
    return value

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def send_json(self, value):
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health_generate":
            self.send_json({"ready": True})
            return
        if self.path == "/server_info":
            self.send_json(config["server_info"])
            return
        if self.path.endswith("/capability"):
            self.send_json(config["capability"])
            return
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/generate":
            response = copy.deepcopy(config["generate"][value["request_id"]])
            pointer = response["native_result_pointer"]
            started_ns = time.monotonic_ns()
            time.sleep(float(config.get("generate_delay_seconds", 0.0)))
            terminal_ns = time.monotonic_ns()
            for index, event in enumerate(pointer["events"]):
                event["observed_ns"] = started_ns + max(
                    1,
                    ((index + 1) * (terminal_ns - started_ns))
                    // (len(pointer["events"]) + 1),
                )
            pointer["request_started_ns"] = started_ns
            pointer["request_terminal_ns"] = max(
                terminal_ns,
                pointer["events"][-1]["observed_ns"] + 1,
            )
            pointer.pop("result_pointer_sha256", None)
            pointer["result_pointer_sha256"] = sha(pointer)
            self.send_json(response)
            return
        if self.path.endswith("/terminal-evidence"):
            action = value["action"]
            response = begin_response() if action == "begin" else reset_response() if action == "reset" else terminal_response()
            self.send_json(response)
            return
        self.send_error(404)

ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def _binding(
    suffix: str = "0",
    *,
    method: str = "static",
    warmup_request_ids: tuple[str, ...] = (),
    scored_request_ids: tuple[str, ...] | None = None,
) -> NativeTerminalRunBinding:
    return NativeTerminalRunBinding(
        run_id=f"live-{suffix}",
        run_nonce_sha256=SHA_A,
        execution_plan_sha256=SHA_B,
        rank_config_sha256=SHA_C,
        attempt_id=f"attempt-{suffix}",
        session_id=f"session-{suffix}",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=SHA_D,
        method=method,
        warmup_request_ids=warmup_request_ids,
        scored_request_ids=(
            (f"score-{suffix}",) if scored_request_ids is None else scored_request_ids
        ),
    )


def _json_binding(path: Path, value: object) -> CanonicalJsonProofBinding:
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _nvidia_smi_tool(
    tmp_path: Path,
    *,
    gpu_rows: tuple[tuple[str, str, int], ...],
    process_rows: tuple[tuple[str, int, int], ...],
    name: str = "nvidia-smi",
) -> PinnedNvidiaSmiTool:
    executable = (tmp_path / name).resolve()
    gpu_text = "".join(
        f"{uuid}, {model}, {memory}\n" for uuid, model, memory in gpu_rows
    )
    process_text = "".join(
        f"{uuid}, {pid}, {memory}\n" for uuid, pid, memory in process_rows
    )
    executable.write_text(
        "#!"
        + sys.executable
        + "\nimport sys\n"
        + f"GPU = {gpu_text!r}\n"
        + f"PROCESSES = {process_text!r}\n"
        + "query = ' '.join(sys.argv[1:])\n"
        + "sys.stdout.write(PROCESSES if '--query-compute-apps=' in query else GPU)\n",
        encoding="utf-8",
    )
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return PinnedNvidiaSmiTool.bind(executable)


def _dynamic_nvidia_smi_tool(
    tmp_path: Path, *, pid_path: Path, gpu_uuid: str
) -> PinnedNvidiaSmiTool:
    executable = (tmp_path / "dynamic-nvidia-smi").resolve()
    executable.write_text(
        "#!"
        + sys.executable
        + "\nimport os, pathlib, sys\n"
        + f"pid_path = pathlib.Path({str(pid_path)!r})\n"
        + f"gpu_uuid = {gpu_uuid!r}\n"
        + "query = ' '.join(sys.argv[1:])\n"
        + "if '--query-compute-apps=' in query:\n"
        + "    if pid_path.is_file():\n"
        + "        pid = int(pid_path.read_text())\n"
        + "        try:\n"
        + "            os.kill(pid, 0)\n"
        + "        except ProcessLookupError:\n"
        + "            pass\n"
        + "        else:\n"
        + "            print(f'{gpu_uuid}, {pid}, 1')\n"
        + "else:\n"
        + "    print(f'{gpu_uuid}, Test GPU, 1')\n",
        encoding="utf-8",
    )
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return PinnedNvidiaSmiTool.bind(executable)


def _dynamic_group_nvidia_smi_tool(
    tmp_path: Path, *, pid_paths: tuple[Path, Path]
) -> PinnedNvidiaSmiTool:
    executable = (tmp_path / "dynamic-group-nvidia-smi").resolve()
    rows = (("GPU-0", str(pid_paths[0])), ("GPU-1", str(pid_paths[1])))
    executable.write_text(
        "#!"
        + sys.executable
        + "\nimport os, pathlib, sys\n"
        + f"ROWS = {rows!r}\n"
        + "query = ' '.join(sys.argv[1:])\n"
        + "if '--query-compute-apps=' in query:\n"
        + "    for gpu_uuid, raw_path in ROWS:\n"
        + "        path = pathlib.Path(raw_path)\n"
        + "        if not path.is_file():\n"
        + "            continue\n"
        + "        pid = int(path.read_text())\n"
        + "        try:\n"
        + "            os.kill(pid, 0)\n"
        + "        except ProcessLookupError:\n"
        + "            continue\n"
        + "        print(f'{gpu_uuid}, {pid}, 1')\n"
        + "else:\n"
        + "    for gpu_uuid, _raw_path in ROWS:\n"
        + "        print(f'{gpu_uuid}, Test GPU, 1')\n",
        encoding="utf-8",
    )
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return PinnedNvidiaSmiTool.bind(executable)


def _fake_live_server_configuration(
    binding: NativeTerminalRunBinding,
    *,
    generate_delay_seconds: float = 0.0,
    warmup_inputs: tuple[int, ...] = (1, 2),
    warmup_outputs: tuple[int, ...] = (3,),
    scored_inputs: tuple[int, ...] = (4, 5),
    scored_outputs: tuple[int, ...] = (6, 7),
) -> tuple[dict[str, object], tuple[object, ...], tuple[object, ...]]:
    from test_native_terminal_provider import (
        FakeAdminTransport,
        _bound_request,
        _native_itl_pointer,
        _server_request,
    )

    assert len(binding.warmup_request_ids) == 1
    assert len(binding.scored_request_ids) == 1
    warmup_id = binding.warmup_request_ids[0]
    scored_id = binding.scored_request_ids[0]
    warmup_expected = (
        _server_request(warmup_id, inputs=warmup_inputs, outputs=warmup_outputs),
    )
    scored_expected = (
        _server_request(scored_id, inputs=scored_inputs, outputs=scored_outputs),
    )
    admin = FakeAdminTransport(
        binding=binding,
        warmup=warmup_expected,
        scored=scored_expected,
    )
    begin = admin._begin(binding.begin_payload())
    admin.begin_receipt = begin
    reset = admin._reset(
        {
            "hook": begin["hook"],
            "run_id": binding.run_id,
            "begin_sha256": begin["begin_sha256"],
        }
    )
    admin.reset_receipt = reset
    terminal = admin._terminal(
        {
            "hook": begin["hook"],
            "run_id": binding.run_id,
            "reset_sha256": reset["reset_sha256"],
            "client_terminal_rows": [],
        }
    )
    capability = asyncio.run(admin.get_json("/capability"))
    server_info = ControlledExecutionPolicy().server_info_fields(role="speculative")
    server_info.update(
        {
            "lightcone_adaptation_mechanism_enabled": binding.method
            not in {"target_only", "static"},
            "lightcone_adaptation_microbatch_size": 1,
            "lightcone_adaptation_publication_coalescing": 1,
            "lightcone_adaptation_stream_priority": "default",
        }
    )
    config = {
        "server_info": server_info,
        "capability": capability,
        "begin": begin,
        "reset": reset,
        "terminal": terminal,
        "generate": {
            request.request_id: {
                "generated_text": request.request_id,
                "generated_token_ids": list(request.output_token_ids or ()),
                "native_result_pointer": json.loads(_native_itl_pointer(request)),
            }
            for request in (*warmup_expected, *scored_expected)
        },
        "generate_delay_seconds": generate_delay_seconds,
    }
    warmup = (
        _bound_request(
            warmup_id,
            inputs=warmup_inputs,
            requested_output_tokens=len(warmup_outputs),
            ordinal=0,
        ),
    )
    scored = (
        _bound_request(
            scored_id,
            inputs=scored_inputs,
            requested_output_tokens=len(scored_outputs),
            ordinal=0,
        ),
    )
    return config, warmup, scored


def _real_http_transport() -> PinnedBenchServingTransport:
    class Response:
        def __init__(self, *, status: int, body: bytes) -> None:
            self.status = status
            self._body = body

        async def json(self, content_type=None):
            del content_type
            return json.loads(self._body)

        def raise_for_status(self) -> None:
            if self.status != 200:
                raise RuntimeError(f"fake HTTP status {self.status}")

    class RequestContext:
        def __init__(
            self,
            *,
            method: str,
            url: str,
            value: object | None,
            timeout: object,
        ) -> None:
            self._method = method
            self._url = url
            self._value = value
            self._timeout = timeout
            self._response: Response | None = None

        async def __aenter__(self) -> Response:
            def request() -> Response:
                body = (
                    None
                    if self._value is None
                    else json.dumps(self._value, separators=(",", ":")).encode()
                )
                request = urllib.request.Request(
                    self._url,
                    data=body,
                    method=self._method,
                    headers={"Content-Type": "application/json"},
                )
                timeout = (
                    float(self._timeout)
                    if isinstance(self._timeout, (int, float))
                    else 30.0
                )
                with urllib.request.urlopen(request, timeout=timeout) as opened:
                    return Response(status=opened.status, body=opened.read())

            self._response = await asyncio.to_thread(request)
            return self._response

        async def __aexit__(self, *_args: object) -> None:
            self._response = None

    class Session:
        def __init__(self) -> None:
            self.trace_configs: list[object] = []
            self.closed = False

        def get(self, *, url: str, headers: object) -> RequestContext:
            del headers
            return RequestContext(method="GET", url=url, value=None, timeout=30.0)

        def post(
            self,
            url: str | None = None,
            *,
            json: object,
            headers: object | None = None,
            timeout: object = 30.0,
        ) -> RequestContext:
            del headers
            assert url is not None
            return RequestContext(method="POST", url=url, value=json, timeout=timeout)

    class TraceConfig:
        def __init__(self) -> None:
            self.on_connection_create_end: list[object] = []
            self.on_connection_reuseconn: list[object] = []

        def freeze(self) -> None:
            return None

    class RequestInput:
        def __init__(self, **kwargs: object) -> None:
            vars(self).update(kwargs)

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
        started = asyncio.get_running_loop().time()
        async with client_session.post(
            request_func_input.api_url,
            json={"request_id": request_func_input.extra_request_body["rid"]},
            timeout=timeout_s,
        ) as response:
            response.raise_for_status()
            value = await response.json()
        latency = asyncio.get_running_loop().time() - started
        return SimpleNamespace(
            success=True,
            generated_text=value["generated_text"],
            generated_token_ids=value["generated_token_ids"],
            output_len=len(value["generated_token_ids"]),
            latency=latency,
            ttft=latency / 2,
            native_token_timestamp_result_pointer=value["native_result_pointer"],
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


def _run_source_owned_fake_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    second_scored_arrival_us: int,
) -> tuple[
    object,
    tuple[PinnedSglangServingRunSpec, PinnedSglangServingRunSpec],
    PinnedNvidiaSmiTool,
]:
    bindings = tuple(
        _binding(
            str(index),
            warmup_request_ids=(f"warm-{index}",),
            scored_request_ids=(f"score-{index}",),
        )
        for index in range(2)
    )
    configurations_and_requests = tuple(
        _fake_live_server_configuration(
            binding,
            generate_delay_seconds=0.08,
        )
        for binding in bindings
    )
    server_source = (tmp_path / "fake-pinned-group-server.py").resolve()
    server_source.write_text(_FAKE_SERVER_SOURCE, encoding="utf-8")
    checkout = (tmp_path / "patched-sglang").resolve()
    checkout.mkdir()
    ports: list[int] = []
    for _index in range(2):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            ports.append(listener.getsockname()[1])
    if ports[0] == ports[1]:  # pragma: no cover - OS allocation invariant
        raise RuntimeError("test group ports collided")
    pid_paths = tuple(
        (tmp_path / f"group-server-{index}.pid").resolve() for index in range(2)
    )
    launch_by_path: dict[str, object] = {}
    config_by_path: dict[str, object] = {}
    specs: list[PinnedSglangServingRunSpec] = []
    for index, (binding, config_and_requests) in enumerate(
        zip(bindings, configurations_and_requests, strict=True)
    ):
        config, warmup, scored = config_and_requests
        if index == 1 and second_scored_arrival_us:
            scored = (replace(scored[0], arrival_us=second_scored_arrival_us),)
        server_config = (tmp_path / f"group-server-{index}.json").resolve()
        server_config.write_text(
            json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        launch_path = (tmp_path / f"group-launch-{index}.json").resolve()
        launch_binding = _json_binding(launch_path, {"test_group_launch": index})
        run_config_path = (tmp_path / f"group-run-config-{index}.json").resolve()
        server_argv = (
            str(Path(sys.executable).resolve()),
            str(server_source),
            str(ports[index]),
            str(server_config),
            str(pid_paths[index]),
            f"GPU-{index}",
            "--disable-cuda-graph",
        )
        environment = {
            "PATH": str(Path(sys.executable).parent),
            "LD_LIBRARY_PATH": "",
            "CUDA_HOME": str(tmp_path),
            "CUDA_PATH": str(tmp_path),
            "CUDA_VISIBLE_DEVICES": f"GPU-{index}",
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
            server_argv_sha256=hashlib.sha256(
                json.dumps(list(server_argv), separators=(",", ":")).encode()
            ).hexdigest(),
            physical_assignment_sha256=(SHA_B if index == 0 else SHA_C),
            experiment_budget_sha256=SHA_C,
            inventory_sha256=SHA_D,
            gpu_uuids=(f"GPU-{index}",),
            localhost_port=ports[index],
            child_environment=lambda environment=environment: dict(environment),
        )
        launch_by_path[str(launch_path)] = launch
        config_by_path[str(run_config_path)] = SimpleNamespace(
            method="static",
            model=SimpleNamespace(target="target/test"),
            adaptation=None,
            runtime=RuntimeConfig(
                sampling_profile_sha256=SHA_A,
                max_running_requests=1,
            ),
        )
        specs.append(
            PinnedSglangServingRunSpec(
                launch_manifest=launch_binding,
                binding=binding,
                warmup_requests=warmup,
                scored_requests=scored,
                terminal_output_path=str(
                    (tmp_path / f"group-terminal-{index}.json").resolve()
                ),
                native_itl_pointer_output_path=str(
                    (tmp_path / f"group-itl-{index}.json").resolve()
                ),
                live_run_receipt_output_path=str(
                    (tmp_path / f"group-receipt-{index}.json").resolve()
                ),
                server_log_output_path=str(
                    (tmp_path / f"group-server-{index}.log").resolve()
                ),
                lifecycle_timing_output_path=str(
                    (tmp_path / f"group-lifecycle-{index}.json").resolve()
                ),
            )
        )
    monkeypatch.setattr(
        live_sglang.CompileLaunchManifest,
        "load",
        classmethod(lambda _cls, path: launch_by_path[str(Path(path).resolve())]),
    )
    monkeypatch.setattr(
        live_sglang,
        "load_run_config",
        lambda path: config_by_path[str(Path(path).resolve())],
    )
    monkeypatch.setattr(
        PinnedBenchServingTransport,
        "from_checkout",
        classmethod(lambda _cls, _path: _real_http_transport()),
    )
    tool = _dynamic_group_nvidia_smi_tool(tmp_path, pid_paths=pid_paths)
    exact_specs = (specs[0], specs[1])
    result = asyncio.run(
        execute_unsigned_native_serving_group(
            specs=exact_specs,
            nvidia_smi_tool=tool,
            inventory_sha256=SHA_D,
            before_gpu_snapshot_output_path=(tmp_path / "group-before.json").resolve(),
            ready_gpu_snapshot_output_path=(tmp_path / "group-ready.json").resolve(),
            after_gpu_snapshot_output_path=(tmp_path / "group-after.json").resolve(),
            group_receipt_output_path=(tmp_path / "group-result.json").resolve(),
            fatal_output_path=(tmp_path / "group-fatal.json").resolve(),
            timeout_seconds=20.0,
        )
    )
    return result, exact_specs, tool


def test_live_module_imports_cleanly_and_has_no_callback_boundary() -> None:
    import_statement = (
        "from lightcone_spec.orchestration.live_sglang import "
        + "execute_unsigned_native_serving_group,execute_unsigned_native_serving_run"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            import_statement,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    single = inspect.signature(execute_unsigned_native_serving_run).parameters
    group = inspect.signature(execute_unsigned_native_serving_group).parameters
    assert "execute_requests" not in single
    assert "transport" not in single
    assert "specs" in group and "nvidia_smi_tool" in group


def test_e4_mechanism_knobs_bind_config_argv_and_live_server_observation() -> None:
    from test_schema import config_value

    from lightcone_spec.config.schema import RunConfig
    from lightcone_spec.orchestration.runtime import _adaptation_mechanism_argv
    from lightcone_spec.sglang_bridge.config import sglang_adaptation_payload

    value = config_value("l0")
    value["runtime"].update(
        adaptation_microbatch_size=8,
        adaptation_publication_coalescing=4,
        adaptation_stream_priority="high",
    )
    config = RunConfig.model_validate(value)
    mechanism_argv = tuple(_adaptation_mechanism_argv(config.runtime))
    server_argv = ("/absolute/python", "--disable-cuda-graph", *mechanism_argv)
    expected = live_sglang._mechanism_execution_authority(
        config=config,
        server_argv=server_argv,
    )
    assert expected == {
        "lightcone_adaptation_mechanism_enabled": True,
        "lightcone_adaptation_microbatch_size": 8,
        "lightcone_adaptation_publication_coalescing": 4,
        "lightcone_adaptation_stream_priority": "high",
    }
    payload = sglang_adaptation_payload(config)
    assert payload is not None
    assert payload["adaptation_microbatch_size"] == 8
    assert payload["adaptation_publication_coalescing"] == 4
    assert payload["adaptation_stream_priority"] == "high"

    class Transport:
        async def get_json(self, path: str) -> dict[str, object]:
            assert path == "/server_info"
            return {
                **ControlledExecutionPolicy().server_info_fields(role="speculative"),
                **expected,
            }

    canonical, digest = asyncio.run(
        live_sglang._observe_live_server_execution_policy(
            transport=Transport(),  # type: ignore[arg-type]
            config=config,
        )
    )
    assert json.loads(canonical)["lightcone_adaptation_microbatch_size"] == 8
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == digest

    with pytest.raises(ValueError, match="mechanism argv differs"):
        live_sglang._mechanism_execution_authority(
            config=config,
            server_argv=tuple("1" if value == "8" else value for value in server_argv),
        )
    static = RunConfig.model_validate(config_value("static"))
    with pytest.raises(ValueError, match="allocation-free"):
        live_sglang._mechanism_execution_authority(
            config=static,
            server_argv=server_argv,
        )


def test_source_owned_single_runner_exercises_real_http_and_native_admin_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding(warmup_request_ids=("warm-0",), scored_request_ids=("score-0",))
    config, warmup, scored = _fake_live_server_configuration(binding)
    server_source = (tmp_path / "fake-pinned-server.py").resolve()
    server_source.write_text(_FAKE_SERVER_SOURCE, encoding="utf-8")
    server_config = (tmp_path / "fake-server-config.json").resolve()
    server_config.write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    pid_path = (tmp_path / "server.pid").resolve()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    checkout = (tmp_path / "patched-sglang").resolve()
    checkout.mkdir()
    manifest_path = (tmp_path / "launch.json").resolve()
    launch_binding = _json_binding(manifest_path, {"test_launch": True})
    server_argv = (
        str(Path(sys.executable).resolve()),
        str(server_source),
        str(port),
        str(server_config),
        str(pid_path),
        "GPU-0",
        "--disable-cuda-graph",
    )

    class FakeLaunch:
        sha256 = launch_binding.semantic_sha256
        run_config_path = str((tmp_path / "run-config.json").resolve())
        target_model_id = "target/test"
        patched_sglang_checkout = str(checkout)
        patched_sglang_commit = PINNED_SGLANG_COMMIT
        patched_sglang_tree = PINNED_SGLANG_TREE
        server_argv_sha256 = hashlib.sha256(
            json.dumps(list(server_argv), separators=(",", ":")).encode()
        ).hexdigest()
        physical_assignment_sha256 = SHA_B
        experiment_budget_sha256 = SHA_C
        inventory_sha256 = CONTROL_INVENTORY_SHA256
        gpu_uuids = ("GPU-0",)
        localhost_port = port

        @staticmethod
        def child_environment() -> dict[str, str]:
            return {
                "PATH": str(Path(sys.executable).parent),
                "LD_LIBRARY_PATH": "",
                "CUDA_HOME": str(tmp_path),
                "CUDA_PATH": str(tmp_path),
                "CUDA_VISIBLE_DEVICES": "GPU-0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "LANG": "C",
                "LC_ALL": "C",
            }

    FakeLaunch.server_argv = server_argv
    launch = FakeLaunch()
    run_config = SimpleNamespace(
        method="static",
        model=SimpleNamespace(target="target/test"),
        adaptation=None,
        runtime=RuntimeConfig(
            sampling_profile_sha256=SHA_A,
            max_running_requests=2,
        ),
    )
    transport = _real_http_transport()
    monkeypatch.setattr(
        live_sglang.CompileLaunchManifest,
        "load",
        classmethod(lambda _cls, _path: launch),
    )
    monkeypatch.setattr(live_sglang, "load_run_config", lambda _path: run_config)
    monkeypatch.setattr(
        PinnedBenchServingTransport,
        "from_checkout",
        classmethod(lambda _cls, _path: transport),
    )
    tool = _dynamic_nvidia_smi_tool(tmp_path, pid_path=pid_path, gpu_uuid="GPU-0")
    terminal_path = (tmp_path / "terminal.json").resolve()
    pointer_path = (tmp_path / "itl.json").resolve()
    receipt_path = (tmp_path / "receipt.json").resolve()
    log_path = (tmp_path / "server.log").resolve()
    lifecycle_path = (tmp_path / "lifecycle.json").resolve()
    result = asyncio.run(
        execute_unsigned_native_serving_run(
            launch_manifest_path=manifest_path,
            binding=binding,
            warmup_requests=warmup,
            scored_requests=scored,
            terminal_output_path=terminal_path,
            native_itl_pointer_output_path=pointer_path,
            live_run_receipt_output_path=receipt_path,
            server_log_output_path=log_path,
            nvidia_smi_tool=tool,
            before_gpu_snapshot_output_path=(tmp_path / "before.json").resolve(),
            ready_gpu_snapshot_output_path=(tmp_path / "ready.json").resolve(),
            after_gpu_snapshot_output_path=(tmp_path / "after.json").resolve(),
            fatal_output_path=(tmp_path / "fatal.json").resolve(),
            timeout_seconds=20.0,
            lifecycle_timing_output_path=lifecycle_path,
        )
    )
    assert result.receipt.server_process_id == int(pid_path.read_text())
    assert result.receipt.snapshot_gpu_uuids == ("GPU-0",)
    assert result.receipt.server_process_group_ids == (
        result.receipt.server_process_id,
    )
    assert result.receipt.process_group_empty is True
    with pytest.raises(ProcessLookupError):
        os.kill(result.receipt.server_process_id, 0)

    lifecycle_binding = CanonicalJsonProofBinding.bind(lifecycle_path)
    lifecycle = validate_unsigned_pinned_sglang_lifecycle_timing_receipt(
        lifecycle_binding,
        expected_live_run_receipt=result.receipt_binding,
        expected_binding=binding,
        expected_telemetry_detail="headline",
    )
    assert lifecycle.inventory_sha256 == CONTROL_INVENTORY_SHA256
    assert lifecycle.phase_durations_ns["reserved_wall_ns"] > 0
    assert lifecycle.phase_durations_ns["startup_ns"] > 0
    assert lifecycle.phase_durations_ns["warmup_ns"] > 0
    assert lifecycle.phase_durations_ns["scored_request_window_ns"] > 0
    assert lifecycle.phase_durations_ns["evidence_flush_ns"] >= 0
    assert lifecycle.phase_durations_ns["profile_reserved_ns"] == 0

    root_private_key = Ed25519PrivateKey.generate()
    control_private_key = Ed25519PrivateKey.generate()
    root_binding = _control_root_binding(root_private_key)
    control_bundle = _control_bundle(control_private_key)
    terminal_authorization = _control_deployment_authorization(
        root_private_key=root_private_key,
        root_binding=root_binding,
        bundle=control_bundle,
    )
    lifecycle_authorization = _deployment_authorization_with_nonce(
        root_private_key=root_private_key,
        root_binding=root_binding,
        bundle=control_bundle,
        challenge_id="validation-lifecycle-deployment-policy-1",
        nonce_byte=b"l",
    )
    monkeypatch.setattr(
        release_root_module,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    external_terminal_binding = build_native_terminal_external_control_binding(
        result.receipt.terminal_artifact.reopen(),
        trusted_attester_policy=NO_TRUSTED_ATTESTERS,
        inventory_sha256=CONTROL_INVENTORY_SHA256,
        expected_binding=binding,
    )
    terminal_control = _local_control(
        private_key=control_private_key,
        root_binding=root_binding,
        bundle=control_bundle,
        authorization=terminal_authorization,
        artifact_sha256=external_terminal_binding.sha256,
        protocol_sha256=NATIVE_TERMINAL_EXTERNAL_CONTROL_PROTOCOL_SHA256,
        registry_sha256=SHA_A,
        lineage_sha256=external_terminal_binding.lineage_sha256,
        challenge_id="validation-native-result-control-1",
        nonce_byte=b"t",
    )
    replay_root = (tmp_path / "proof-replay").resolve()
    replay_root.mkdir(mode=0o700)
    replay_store = ChallengeReplayStore(str(replay_root))
    native_result_path = (tmp_path / "native-result-proof.json").resolve()
    native_result_binding = publish_native_terminal_result_proof_artifact(
        str(terminal_path),
        control_attestation=terminal_control,
        replay_store=replay_store,
        expected_inventory_sha256=CONTROL_INVENTORY_SHA256,
        expected_registry_sha256=SHA_A,
        expected_root_manifest_sha256=root_binding.semantic_sha256,
        now_ns=CONTROL_NOW_NS,
        proof_artifact_path=str(native_result_path),
        expected_binding=binding,
    )
    lifecycle_control = _local_control(
        private_key=control_private_key,
        root_binding=root_binding,
        bundle=control_bundle,
        authorization=lifecycle_authorization,
        artifact_sha256=lifecycle_binding.raw_sha256,
        protocol_sha256=PINNED_SGLANG_LIFECYCLE_TIMING_PROTOCOL_SHA256,
        registry_sha256=SHA_A,
        lineage_sha256=live_sglang._lifecycle_control_lineage_sha256(
            timing_binding=lifecycle_binding,
            timing=lifecycle,
            live_run_receipt=result.receipt_binding,
            native_result_proof=native_result_binding,
            registry_sha256=SHA_A,
        ),
        challenge_id="validation-lifecycle-control-1",
        nonce_byte=b"p",
    )
    lifecycle_proof_path = (tmp_path / "lifecycle-proof.json").resolve()
    lifecycle_proof_binding = publish_pinned_sglang_lifecycle_timing_proof_artifact(
        str(lifecycle_path),
        live_run_receipt_path=str(receipt_path),
        native_result_proof_artifact_path=str(native_result_path),
        control_attestation=lifecycle_control,
        replay_store=replay_store,
        expected_binding=binding,
        expected_inventory_sha256=CONTROL_INVENTORY_SHA256,
        expected_registry_sha256=SHA_A,
        expected_root_manifest_sha256=root_binding.semantic_sha256,
        expected_telemetry_detail="headline",
        now_ns=CONTROL_NOW_NS,
        proof_artifact_path=str(lifecycle_proof_path),
    )
    verified_lifecycle = validate_pinned_sglang_lifecycle_timing_proof_artifact(
        str(lifecycle_proof_path),
        expected_binding=binding,
        expected_inventory_sha256=CONTROL_INVENTORY_SHA256,
        expected_registry_sha256=SHA_A,
        expected_root_manifest_sha256=root_binding.semantic_sha256,
        expected_gpu_uuids=("GPU-0",),
        expected_telemetry_detail="headline",
        now_ns=CONTROL_NOW_NS,
    )
    assert lifecycle_proof_binding.semantic_sha256 == (
        CanonicalJsonProofBinding.bind(lifecycle_proof_path).semantic_sha256
    )
    assert verified_lifecycle.raw_timing_sha256 == lifecycle.sha256
    assert verified_lifecycle.native_result_proof_sha256 == (
        native_result_binding.semantic_sha256
    )
    assert verified_lifecycle.phase_durations == lifecycle.phase_durations_ns
    assert len(tuple(replay_root.glob("reservation-*.json"))) == 2

    log_path.write_bytes(log_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="server log changed|raw digest"):
        validate_unsigned_pinned_sglang_serving_run_receipt(
            result.receipt_binding,
            expected_launch_manifest=launch_binding,
            expected_binding=binding,
            expected_terminal_artifact=result.receipt.terminal_artifact,
            expected_native_itl_pointer_artifact=(
                result.receipt.native_itl_pointer_artifact
            ),
            expected_scored_request_inputs_sha256=live_sglang.canonical_sha256(
                [request.sha256 for request in scored]
            ),
            expected_gpu_uuids=("GPU-0",),
            expected_inventory_sha256=CONTROL_INVENTORY_SHA256,
            expected_physical_assignment_sha256=SHA_B,
            expected_experiment_budget_sha256=SHA_C,
            expected_tool=tool,
            expected_snapshot_gpu_uuids=("GPU-0",),
            expected_server_process_group_ids=(result.receipt.server_process_id,),
        )


def test_ready_snapshot_rejects_foreign_process_group_and_tamper(
    tmp_path: Path,
) -> None:
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        for _index in range(2)
    ]
    try:
        tool = _nvidia_smi_tool(
            tmp_path,
            gpu_rows=(("GPU-0", "Test GPU", 1), ("GPU-1", "Test GPU", 1)),
            process_rows=(
                ("GPU-0", processes[0].pid, 1),
                ("GPU-1", processes[1].pid, 1),
            ),
        )
        snapshot = live_sglang._capture_gpu_process_snapshot(
            tool=tool,
            gpu_uuids=("GPU-0", "GPU-1"),
            inventory_sha256=SHA_A,
            phase="ready",
            output_path=(tmp_path / "ready.json").resolve(),
            expected_server_process_group_ids=(processes[0].pid, processes[1].pid),
        )
        value = validate_pinned_sglang_gpu_process_snapshot(
            snapshot,
            expected_tool=tool,
            expected_gpu_uuids=("GPU-0", "GPU-1"),
            expected_inventory_sha256=SHA_A,
            expected_phase="ready",
            expected_server_process_group_ids=(processes[0].pid, processes[1].pid),
        )
        assert [row["process_group_id"] for row in value["compute_process_rows"]] == [
            processes[0].pid,
            processes[1].pid,
        ]

        foreign_tool = _nvidia_smi_tool(
            tmp_path,
            gpu_rows=(("GPU-0", "Test GPU", 1), ("GPU-1", "Test GPU", 1)),
            process_rows=(("GPU-0", processes[1].pid, 1),),
            name="foreign-nvidia-smi",
        )
        with pytest.raises(RuntimeError, match="foreign process"):
            live_sglang._capture_gpu_process_snapshot(
                tool=foreign_tool,
                gpu_uuids=("GPU-0", "GPU-1"),
                inventory_sha256=SHA_A,
                phase="ready",
                output_path=(tmp_path / "foreign.json").resolve(),
                expected_server_process_group_ids=(
                    processes[0].pid,
                    processes[1].pid,
                ),
            )

        Path(snapshot.absolute_path).write_bytes(
            Path(snapshot.absolute_path).read_bytes() + b" "
        )
        with pytest.raises(ValueError, match="canonical JSON|raw digest"):
            snapshot.reopen()
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, 15)
            process.wait(timeout=5)


def test_clean_snapshot_rejects_any_compute_process(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        tool = _nvidia_smi_tool(
            tmp_path,
            gpu_rows=(("GPU-0", "Test GPU", 1),),
            process_rows=(("GPU-0", process.pid, 1),),
        )
        with pytest.raises(RuntimeError, match="process gate is not empty"):
            live_sglang._capture_gpu_process_snapshot(
                tool=tool,
                gpu_uuids=("GPU-0",),
                inventory_sha256=SHA_A,
                phase="before",
                output_path=(tmp_path / "before.json").resolve(),
            )
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 15)
        process.wait(timeout=5)


def test_group_receipt_round_trip_binds_ready_rows_and_process_groups(
    tmp_path: Path,
) -> None:
    bindings = tuple(
        _json_binding((tmp_path / f"artifact-{index}.json").resolve(), {"i": index})
        for index in range(8)
    )
    receipt = UnsignedPinnedSglangServingGroupReceipt(
        schema_version=1,
        kind="unsigned_pinned_sglang_concurrent_group_receipt",
        protocol_sha256=PINNED_SGLANG_LIVE_SERVING_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        inventory_sha256=SHA_A,
        gpu_uuids=("GPU-0", "GPU-1"),
        localhost_ports=(30001, 30002),
        server_process_group_ids=(101, 202),
        ready_compute_process_rows_sha256=SHA_B,
        launch_manifests=(bindings[0], bindings[1]),
        run_binding_sha256s=(SHA_C, SHA_D),
        live_run_receipts=(bindings[2], bindings[3]),
        before_gpu_snapshot=bindings[4],
        ready_gpu_snapshot=bindings[5],
        after_gpu_snapshot=bindings[6],
        shared_scored_origin_ns=10,
        overlap_started_ns=10,
        overlap_finished_ns=20,
        overlap_duration_ns=10,
    )
    assert (
        UnsignedPinnedSglangServingGroupReceipt.from_dict(receipt.to_dict()) == receipt
    )


def test_shared_barrier_cannot_fabricate_actual_request_overlap() -> None:
    with pytest.raises(RuntimeError, match="native request intervals did not overlap"):
        live_sglang._derive_actual_group_overlap(
            shared_origin_ns=100,
            scored_intervals=((110, 150), (200, 250)),
        )
    assert live_sglang._derive_actual_group_overlap(
        shared_origin_ns=100,
        scored_intervals=((110, 230), (200, 250)),
    ) == (200, 230)


def test_source_owned_group_runs_two_live_servers_and_deep_reopens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, specs, tool = _run_source_owned_fake_group(
        tmp_path,
        monkeypatch,
        second_scored_arrival_us=0,
    )
    receipt = result.receipt
    assert receipt.gpu_uuids == ("GPU-0", "GPU-1")
    assert receipt.localhost_ports[0] != receipt.localhost_ports[1]
    assert receipt.server_process_group_ids == tuple(
        run.receipt.server_process_id for run in result.runs
    )
    assert all(run.receipt.process_group_empty for run in result.runs)
    assert receipt.shared_scored_origin_ns <= min(
        run.receipt.scored_started_ns for run in result.runs
    )
    assert receipt.overlap_started_ns == max(
        run.receipt.scored_started_ns for run in result.runs
    )
    assert receipt.overlap_finished_ns == min(
        run.receipt.scored_finished_ns for run in result.runs
    )
    assert receipt.overlap_duration_ns > 0
    reopened = validate_unsigned_pinned_sglang_serving_group_receipt(
        result.receipt_binding,
        expected_specs=specs,
        expected_tool=tool,
        expected_inventory_sha256=SHA_D,
    )
    assert reopened.receipt.sha256 == receipt.sha256
    for process_id in receipt.server_process_group_ids:
        with pytest.raises(ProcessLookupError):
            os.kill(process_id, 0)


def test_source_owned_group_rejects_delayed_nonoverlapping_native_intervals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(PinnedSglangServingRunError) as caught:
        _run_source_owned_fake_group(
            tmp_path,
            monkeypatch,
            second_scored_arrival_us=300_000,
        )
    assert caught.value.reason_code == "concurrent_group_no_overlap"
    fatal = json.loads((tmp_path / "group-fatal.json").read_text(encoding="utf-8"))
    assert fatal["status"] == "ERROR"
    assert not (tmp_path / "group-result.json").exists()


def test_single_prelaunch_failure_publishes_immutable_fatal_pointer(
    tmp_path: Path,
) -> None:
    tool = _nvidia_smi_tool(
        tmp_path,
        gpu_rows=(("GPU-0", "Test GPU", 0),),
        process_rows=(),
    )
    fatal = (tmp_path / "fatal.json").resolve()
    with pytest.raises(PinnedSglangServingRunError) as caught:
        asyncio.run(
            execute_unsigned_native_serving_run(
                launch_manifest_path=(tmp_path / "missing-launch.json").resolve(),
                binding=_binding(),
                warmup_requests=(),
                scored_requests=(),
                terminal_output_path=(tmp_path / "terminal.json").resolve(),
                native_itl_pointer_output_path=(tmp_path / "itl.json").resolve(),
                live_run_receipt_output_path=(tmp_path / "receipt.json").resolve(),
                server_log_output_path=(tmp_path / "server.log").resolve(),
                nvidia_smi_tool=tool,
                before_gpu_snapshot_output_path=(tmp_path / "before.json").resolve(),
                ready_gpu_snapshot_output_path=(tmp_path / "ready.json").resolve(),
                after_gpu_snapshot_output_path=(tmp_path / "after.json").resolve(),
                fatal_output_path=fatal,
                timeout_seconds=10.0,
            )
        )
    assert caught.value.reason_code == "prelaunch_validation_failed"
    value = json.loads(fatal.read_text(encoding="utf-8"))
    assert value["status"] == "ERROR"
    assert value["formal_execution_authorized"] is False
    with pytest.raises(RuntimeError, match="already exists"):
        _json_binding(fatal, {"replacement": True})


def test_graph_launch_without_dynamic_proof_fails_before_process_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability_sha256 = NATIVE_RUNTIME_RELEASE_CAPABILITY.sha256
    policy = FixedAddressGraphExecutionPolicy(
        native_runtime_release_capability_sha256=capability_sha256
    )
    runtime = RuntimeConfig(
        sampling_profile_sha256=SHA_A,
        cuda_graph_mode="fixed_address_publication_v1",
        disable_cuda_graph=False,
        native_graph_release_capability_sha256=capability_sha256,
        execution_policy_sha256=policy.sha256,
    )
    launch_path = (tmp_path / "graph-launch.json").resolve()
    launch_binding = _json_binding(launch_path, {"graph_launch": True})
    launch = SimpleNamespace(
        sha256=launch_binding.semantic_sha256,
        run_config_path=str((tmp_path / "graph-run.json").resolve()),
        target_model_id="target/test",
        patched_sglang_checkout=str((tmp_path / "patched").resolve()),
        patched_sglang_commit=PINNED_SGLANG_COMMIT,
        patched_sglang_tree=PINNED_SGLANG_TREE,
        server_argv=(
            str(Path(sys.executable).resolve()),
            "--lightcone-fixed-address-publication-graph",
            "--lightcone-graph-batch-sizes",
            "1",
            "--lightcone-disable-graph-eager-fallback",
        ),
        server_argv_sha256=SHA_A,
        physical_assignment_sha256=SHA_B,
        experiment_budget_sha256=SHA_C,
        inventory_sha256=SHA_D,
        gpu_uuids=("GPU-0",),
        localhost_port=30001,
    )
    config = SimpleNamespace(
        method="static",
        model=SimpleNamespace(target="target/test"),
        adaptation=None,
        runtime=runtime,
    )
    monkeypatch.setattr(
        live_sglang.CompileLaunchManifest,
        "load",
        classmethod(lambda _cls, _path: launch),
    )
    monkeypatch.setattr(live_sglang, "load_run_config", lambda _path: config)
    monkeypatch.setattr(
        live_sglang,
        "_spawn_server",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("graph proof gate ran after process allocation")
        ),
    )
    tool = _nvidia_smi_tool(
        tmp_path,
        gpu_rows=(("GPU-0", "Test GPU", 1),),
        process_rows=(),
    )
    fatal = (tmp_path / "graph-fatal.json").resolve()
    with pytest.raises(PinnedSglangServingRunError) as caught:
        asyncio.run(
            execute_unsigned_native_serving_run(
                launch_manifest_path=launch_path,
                binding=_binding(),
                warmup_requests=(),
                scored_requests=(),
                terminal_output_path=(tmp_path / "graph-terminal.json").resolve(),
                native_itl_pointer_output_path=(tmp_path / "graph-itl.json").resolve(),
                live_run_receipt_output_path=(
                    tmp_path / "graph-receipt.json"
                ).resolve(),
                server_log_output_path=(tmp_path / "graph-server.log").resolve(),
                nvidia_smi_tool=tool,
                before_gpu_snapshot_output_path=(
                    tmp_path / "graph-before.json"
                ).resolve(),
                ready_gpu_snapshot_output_path=(
                    tmp_path / "graph-ready.json"
                ).resolve(),
                after_gpu_snapshot_output_path=(
                    tmp_path / "graph-after.json"
                ).resolve(),
                fatal_output_path=fatal,
                timeout_seconds=10.0,
                expected_graph_source_identity_sha256=SHA_A,
            )
        )
    assert caught.value.reason_code == "prelaunch_validation_failed"
    assert not (tmp_path / "graph-server.log").exists()
    assert json.loads(fatal.read_text(encoding="utf-8"))["error_type"] == (
        "NativeReadinessBlocked"
    )


def test_group_prelaunch_failure_publishes_fatal_pointer(tmp_path: Path) -> None:
    tool = _nvidia_smi_tool(
        tmp_path,
        gpu_rows=(("GPU-0", "Test GPU", 0), ("GPU-1", "Test GPU", 0)),
        process_rows=(),
    )
    specs = []
    for index in range(2):
        launch = _json_binding(
            (tmp_path / f"invalid-launch-{index}.json").resolve(),
            {"not": "a compile launch manifest"},
        )
        specs.append(
            PinnedSglangServingRunSpec(
                launch_manifest=launch,
                binding=_binding(str(index)),
                warmup_requests=(),
                scored_requests=(),
                terminal_output_path=str(
                    (tmp_path / f"terminal-{index}.json").resolve()
                ),
                native_itl_pointer_output_path=str(
                    (tmp_path / f"itl-{index}.json").resolve()
                ),
                live_run_receipt_output_path=str(
                    (tmp_path / f"receipt-{index}.json").resolve()
                ),
                server_log_output_path=str(
                    (tmp_path / f"server-{index}.log").resolve()
                ),
            )
        )
    fatal = (tmp_path / "group-fatal.json").resolve()
    with pytest.raises(PinnedSglangServingRunError) as caught:
        asyncio.run(
            execute_unsigned_native_serving_group(
                specs=(specs[0], specs[1]),
                nvidia_smi_tool=tool,
                inventory_sha256=SHA_A,
                before_gpu_snapshot_output_path=(tmp_path / "before.json").resolve(),
                ready_gpu_snapshot_output_path=(tmp_path / "ready.json").resolve(),
                after_gpu_snapshot_output_path=(tmp_path / "after.json").resolve(),
                group_receipt_output_path=(tmp_path / "group.json").resolve(),
                fatal_output_path=fatal,
                timeout_seconds=10.0,
            )
        )
    assert caught.value.reason_code == "group_prelaunch_validation_failed"
    assert json.loads(fatal.read_text(encoding="utf-8"))["status"] == "ERROR"
