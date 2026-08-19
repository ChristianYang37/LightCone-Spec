"""First-party child and consumer for the pinned COMPILE lifecycle.

The child delays every SGLang/Torch import and server fork until the parent has
sent its path-bound launch manifest and private cache environment.  Requests
reuse the official pinned bench transport; only native scheduler terminal rows
and counters can complete the lifecycle.  Dynamic release control, outside
this module, decides whether the measured receipt may authorize formal work.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.runtime.compile_cache import (
    COMPILE_CACHE_ENVIRONMENT_VARIABLES,
    CompileCacheLaunchPlan,
    CompileOnlyPrewarmManifest,
    CompileOnlyPrewarmPayload,
    _content_sha256,
)
from lightcone_spec.runtime.compile_runner import (
    COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
    CompileAssignmentPlan,
    CompileLaunchManifest,
)

if TYPE_CHECKING:
    from lightcone_spec.experiments.sampling import SamplingProfile
    from lightcone_spec.experiments.serving import PinnedBenchServingTransport

NATIVE_COMPILE_BEGIN_PATH = "/v1/lightcone-spec/compile-cache-evidence/begin"
NATIVE_COMPILE_TERMINAL_PATH = "/v1/lightcone-spec/compile-cache-evidence/terminal"
NATIVE_COMPILE_FINALIZE_PATH = "/v1/lightcone-spec/compile-cache-evidence/finalize"
SOURCE_OWNED_COMPILE_HOOK = "sglang.schema_v3.source_owned_compile_cache_lifecycle.v1"
SOURCE_OWNED_COMPILE_RELEASE_STATUS = "IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF"
GPU_COMPILE_SEMANTICS = "MEASURED_PENDING_PROOF"
GPU_COMPILE_REASON = None

_GPU_MEASUREMENTS = (
    "cache_hits",
    "cache_misses",
    "jit_time_ns",
    "graph_capture_count",
    "graph_replay_count",
    "cache_write_count",
)
SOURCE_OWNED_COMPILE_PROTOCOL_SHA256 = _content_sha256(
    {
        "schema_version": 2,
        "hook": SOURCE_OWNED_COMPILE_HOOK,
        "release_status": SOURCE_OWNED_COMPILE_RELEASE_STATUS,
        "lifecycle": ("begin", "ordered_request_terminals", "finalize"),
        "source_owned": (
            "patched_tree",
            "process_identity",
            "process_start",
            "request_terminals",
            "begin_and_final_scheduler_state",
        ),
        "counter_source": "sglang_compilation_counter_plus_scheduler_graph_replay",
        "gpu_compile_semantics": GPU_COMPILE_SEMANTICS,
        "gpu_measurements": {name: "nonnegative_integer" for name in _GPU_MEASUREMENTS},
    }
)

CompilePrewarmSubmitter = Callable[[CompileOnlyPrewarmPayload], Awaitable[object]]


def _serving_runtime() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    """Import the Torch-bearing experiment package only after child start."""

    from lightcone_spec.experiments.load import FrozenSamplingParameters
    from lightcone_spec.experiments.sampling import SamplingProfile
    from lightcone_spec.experiments.serving import (
        BoundServingRequest,
        PinnedBenchServingTransport,
    )

    return (
        FrozenSamplingParameters,
        SamplingProfile,
        BoundServingRequest,
        PinnedBenchServingTransport,
    )


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lower-case SHA-256")
    return value


def _integer(label: str, value: object, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{label} is invalid")
    return value


def _text(label: str, value: object) -> str:
    if type(value) is not str or not value or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be non-empty single-line text")
    return value


def _exact_object(
    label: str,
    value: object,
    fields: set[str],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ from source protocol")
    return value


def _canonical_raw(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


_IDENTITY_FIELDS = {
    "assignment_plan_sha256",
    "compile_key_sha256",
    "model_lock_sha256",
    "sampling_profile_sha256",
    "physical_assignment_sha256",
    "experiment_budget_sha256",
    "inventory_sha256",
    "prewarm_manifest_sha256",
    "prewarm_sha256",
}
_COMMON_FIELDS = {
    "schema_version",
    "hook",
    "protocol_sha256",
    "release_status",
    "gpu_compile_semantics",
    "gpu_compile_reason",
    "patched_sglang_tree",
    "process_id",
    "process_started_ns",
    *_IDENTITY_FIELDS,
}
_BEGIN_FIELDS = {
    *_COMMON_FIELDS,
    "ordered_prewarm",
    "begin_state",
    "begin_sha256",
}
_FINAL_FIELDS = {
    *_COMMON_FIELDS,
    "begin_sha256",
    "begin_state",
    "ordered_terminals",
    "final_state",
    "gpu_measurements",
    "completion_marker",
    "compile_receipt_sha256",
}
_TERMINAL_FIELDS = {
    "sequence",
    "graph_bucket",
    "request_id",
    "input_token_count",
    "input_token_ids_sha256",
    "requested_output_tokens",
    "sampling_seed",
    "output_token_count",
    "output_token_ids_sha256",
    "terminal_status",
    "terminal_reason",
    "terminal_sha256",
}


def _drained_state(label: str, value: object) -> dict[str, object]:
    state = _exact_object(label, value, {"active_requests", "queued_requests"})
    if _integer(f"{label} active requests", state["active_requests"]) or _integer(
        f"{label} queued requests", state["queued_requests"]
    ):
        raise ValueError(f"{label} is not drained")
    return state


def _validate_common(
    value: dict[str, object],
    *,
    expected: Mapping[str, object],
) -> None:
    if (
        value["schema_version"] != 2
        or value["hook"] != SOURCE_OWNED_COMPILE_HOOK
        or value["protocol_sha256"] != SOURCE_OWNED_COMPILE_PROTOCOL_SHA256
        or value["release_status"] != SOURCE_OWNED_COMPILE_RELEASE_STATUS
        or value["gpu_compile_semantics"] != GPU_COMPILE_SEMANTICS
        or value["gpu_compile_reason"] != GPU_COMPILE_REASON
        or value["patched_sglang_tree"] != PINNED_SGLANG_TREE
    ):
        raise ValueError("native compile source identity or disposition differs")
    _integer("native compile process ID", value["process_id"], positive=True)
    _integer("native compile process start", value["process_started_ns"], positive=True)
    for name in _IDENTITY_FIELDS:
        _sha256(f"native compile {name}", value[name])
        if value[name] != expected[name]:
            raise ValueError(f"native compile {name} differs from assignment")


def _begin_body(plan: CompileAssignmentPlan) -> dict[str, object]:
    _assignment, _cache_plan, manifest, _launch = plan.revalidate()
    ordered = [
        {
            "graph_bucket": payload.graph_bucket,
            "request_id": payload.request_id,
            "input_token_ids_sha256": _content_sha256(list(payload.input_token_ids)),
            "requested_output_tokens": payload.requested_output_tokens,
            "sampling_seed": payload.sampling_seed,
        }
        for payload in manifest.payloads
    ]
    return {
        "assignment_plan_sha256": plan.sha256,
        "compile_key_sha256": plan.compile_key_sha256,
        "model_lock_sha256": plan.model_lock_sha256,
        "sampling_profile_sha256": manifest.sampling_profile_sha256,
        "physical_assignment_sha256": plan.physical_assignment_sha256,
        "experiment_budget_sha256": plan.experiment_budget_sha256,
        "inventory_sha256": plan.inventory_sha256,
        "prewarm_manifest_sha256": plan.prewarm_manifest_sha256,
        "prewarm": ordered,
    }


@dataclass(frozen=True)
class SourceOwnedCompileNativeContract:
    """Measured native receipts that still require dynamic release control."""

    begin_raw: bytes
    final_raw: bytes
    begin_sha256: str
    compile_receipt_sha256: str
    process_id: int
    process_started_ns: int
    ordered_terminal_sha256s: tuple[str, ...]
    release_status: str
    gpu_compile_semantics: str
    gpu_compile_reason: str | None
    formal_execution_authorized: bool

    def validate(self, plan: CompileAssignmentPlan) -> None:
        if type(plan) is not CompileAssignmentPlan:
            raise TypeError("native compile contract requires an exact assignment plan")
        begin_body = _begin_body(plan)
        ordered_prewarm = begin_body.pop("prewarm")
        expected = {
            **begin_body,
            "prewarm_sha256": _content_sha256(ordered_prewarm),
        }
        begin = _exact_object(
            "native compile begin",
            json.loads(self.begin_raw),
            _BEGIN_FIELDS,
        )
        final = _exact_object(
            "native compile final",
            json.loads(self.final_raw),
            _FINAL_FIELDS,
        )
        _validate_common(begin, expected=expected)
        _validate_common(final, expected=expected)
        if begin["ordered_prewarm"] != ordered_prewarm:
            raise ValueError("native compile prewarm order differs from assignment")
        _drained_state("native compile begin state", begin["begin_state"])
        _drained_state("native compile final begin state", final["begin_state"])
        _drained_state("native compile final state", final["final_state"])
        if final["begin_state"] != begin["begin_state"]:
            raise ValueError("native compile begin state changed in final receipt")
        begin_digest = _content_sha256(
            {key: value for key, value in begin.items() if key != "begin_sha256"}
        )
        if begin["begin_sha256"] != begin_digest or self.begin_sha256 != begin_digest:
            raise ValueError("native compile begin receipt digest differs")
        if final["begin_sha256"] != begin_digest:
            raise ValueError("native compile final names another begin receipt")
        terminals = final["ordered_terminals"]
        if type(terminals) is not list or len(terminals) != len(ordered_prewarm):
            raise ValueError("native compile terminal coverage is incomplete")
        terminal_sha256s: list[str] = []
        for index, (raw, prewarm) in enumerate(
            zip(terminals, ordered_prewarm, strict=True)
        ):
            row = _exact_object("native compile terminal", raw, _TERMINAL_FIELDS)
            if (
                row["sequence"] != index
                or row["graph_bucket"] != prewarm["graph_bucket"]
                or row["request_id"] != prewarm["request_id"]
                or row["input_token_ids_sha256"] != prewarm["input_token_ids_sha256"]
                or row["requested_output_tokens"] != prewarm["requested_output_tokens"]
                or row["sampling_seed"] != prewarm["sampling_seed"]
                or row["terminal_status"] != "completed"
            ):
                raise ValueError("native compile ordered terminal differs from plan")
            _integer("native compile input token count", row["input_token_count"])
            _integer("native compile output token count", row["output_token_count"])
            _sha256("native compile output token IDs", row["output_token_ids_sha256"])
            _text("native compile terminal reason", row["terminal_reason"])
            terminal_digest = _content_sha256(
                {key: value for key, value in row.items() if key != "terminal_sha256"}
            )
            if row["terminal_sha256"] != terminal_digest:
                raise ValueError("native compile terminal digest differs")
            terminal_sha256s.append(terminal_digest)
        measurements = _exact_object(
            "native compile GPU measurements",
            final["gpu_measurements"],
            set(_GPU_MEASUREMENTS),
        )
        for name, value in measurements.items():
            _integer(f"native compile {name}", value)
        final_digest = _content_sha256(
            {
                key: value
                for key, value in final.items()
                if key != "compile_receipt_sha256"
            }
        )
        if (
            final["completion_marker"] != "GPU_COMPILE_COMPLETE"
            or final["compile_receipt_sha256"] != final_digest
            or self.compile_receipt_sha256 != final_digest
            or self.process_id != final["process_id"]
            or self.process_started_ns != final["process_started_ns"]
            or self.ordered_terminal_sha256s != tuple(terminal_sha256s)
            or self.release_status != SOURCE_OWNED_COMPILE_RELEASE_STATUS
            or self.gpu_compile_semantics != GPU_COMPILE_SEMANTICS
            or self.gpu_compile_reason != GPU_COMPILE_REASON
            or self.formal_execution_authorized is not False
        ):
            raise ValueError("native compile contract disposition differs")


class PinnedCompileLifecycleWorker:
    """Run one ordered native COMPILE contract on the official serving pool."""

    def __init__(self, transport: PinnedBenchServingTransport) -> None:
        *_, transport_type = _serving_runtime()
        if not isinstance(transport, transport_type):
            raise TypeError("compile worker requires the pinned bench transport")
        self._transport = transport
        self._used = False

    async def execute(
        self,
        plan: CompileAssignmentPlan,
        *,
        submit_prewarm: CompilePrewarmSubmitter,
    ) -> SourceOwnedCompileNativeContract:
        if self._used:
            raise RuntimeError("compile worker is one-shot")
        if type(plan) is not CompileAssignmentPlan:
            raise TypeError("compile worker requires an exact assignment plan")
        if not callable(submit_prewarm):
            raise TypeError("compile worker requires a prewarm submitter")
        self._used = True
        _assignment, _cache_plan, manifest, _launch = plan.revalidate()
        begin = await self._transport.post_json(
            NATIVE_COMPILE_BEGIN_PATH,
            _begin_body(plan),
        )
        begin_row = _exact_object("native compile begin", begin, _BEGIN_FIELDS)
        begin_sha256 = _sha256(
            "native compile begin receipt", begin_row["begin_sha256"]
        )
        observed_terminal_sha256s: list[str] = []
        for payload in manifest.payloads:
            # The response is intentionally ignored.  Native scheduler state,
            # not a submitter summary, supplies the terminal evidence.
            await submit_prewarm(payload)
            terminal = await self._transport.post_json(
                NATIVE_COMPILE_TERMINAL_PATH,
                {"request_id": payload.request_id},
            )
            terminal_row = _exact_object(
                "native compile terminal", terminal, _TERMINAL_FIELDS
            )
            observed_terminal_sha256s.append(
                _sha256(
                    "native compile terminal",
                    terminal_row["terminal_sha256"],
                )
            )
        final = await self._transport.post_json(
            NATIVE_COMPILE_FINALIZE_PATH,
            {"begin_sha256": begin_sha256},
        )
        final_row = _exact_object("native compile final", final, _FINAL_FIELDS)
        terminals = final_row["ordered_terminals"]
        if type(terminals) is not list:
            raise ValueError("native compile terminal rows must be an array")
        terminal_sha256s = tuple(
            _sha256(
                "native compile terminal",
                _exact_object("native compile terminal", row, _TERMINAL_FIELDS)[
                    "terminal_sha256"
                ],
            )
            for row in terminals
        )
        if terminal_sha256s != tuple(observed_terminal_sha256s):
            raise ValueError("native compile terminal query/finalize identities differ")
        contract = SourceOwnedCompileNativeContract(
            begin_raw=_canonical_raw(begin_row),
            final_raw=_canonical_raw(final_row),
            begin_sha256=begin_sha256,
            compile_receipt_sha256=_sha256(
                "native compile terminal receipt",
                final_row["compile_receipt_sha256"],
            ),
            process_id=_integer(
                "native compile process ID", final_row["process_id"], positive=True
            ),
            process_started_ns=_integer(
                "native compile process start",
                final_row["process_started_ns"],
                positive=True,
            ),
            ordered_terminal_sha256s=terminal_sha256s,
            release_status=SOURCE_OWNED_COMPILE_RELEASE_STATUS,
            gpu_compile_semantics=GPU_COMPILE_SEMANTICS,
            gpu_compile_reason=GPU_COMPILE_REASON,
            formal_execution_authorized=False,
        )
        contract.validate(plan)
        return contract


_MAX_MESSAGE_BYTES = 1024 * 1024
_ASSIGNMENT_ENVIRONMENT = "LIGHTCONE_COMPILE_ASSIGNMENT_PLAN_SHA256"


def _raw_sha256(path: Path, *, label: str) -> str:
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError(f"{label} must be absolute and normalized")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_protocol_message() -> dict[str, object]:
    encoded = sys.stdin.buffer.readline(_MAX_MESSAGE_BYTES + 1)
    if not encoded or len(encoded) > _MAX_MESSAGE_BYTES or not encoded.endswith(b"\n"):
        raise RuntimeError("compile parent command channel is closed or oversized")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("compile parent message is not canonical JSON") from error
    if type(value) is not dict:
        raise TypeError("compile parent message must be a JSON object")
    if _canonical_raw(value) != encoded:
        raise ValueError("compile parent message is not canonical JSON")
    return value


def _write_protocol_message(value: Mapping[str, object]) -> None:
    encoded = _canonical_raw(dict(value))
    if len(encoded) > _MAX_MESSAGE_BYTES:
        raise ValueError("compile child message exceeds the protocol bound")
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def _require_protocol_row(
    value: dict[str, object],
    *,
    assignment_plan_sha256: str,
    kind: str,
    fields: set[str],
) -> dict[str, object]:
    expected = {
        "kind",
        "protocol_sha256",
        "assignment_plan_sha256",
        *fields,
    }
    if set(value) != expected or value["kind"] != kind:
        raise ValueError("compile parent message fields or order differ")
    if (
        value["protocol_sha256"] != COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256
        or value["assignment_plan_sha256"] != assignment_plan_sha256
    ):
        raise ValueError("compile parent message uses another protocol identity")
    return value


def _load_launch_from_start(
    row: dict[str, object],
) -> tuple[CompileLaunchManifest, dict[str, str]]:
    launch_binding = _exact_object(
        "compile launch binding",
        row["launch_manifest"],
        {"path", "raw_sha256", "semantic_sha256"},
    )
    launch_path = Path(_text("compile launch path", launch_binding["path"]))
    expected_raw = _sha256("compile launch raw", launch_binding["raw_sha256"])
    expected_semantic = _sha256(
        "compile launch semantic", launch_binding["semantic_sha256"]
    )
    if _raw_sha256(launch_path, label="compile launch manifest") != expected_raw:
        raise ValueError("compile launch raw identity changed before child start")
    launch = CompileLaunchManifest.load(launch_path)
    if launch.sha256 != expected_semantic:
        raise ValueError("compile launch semantic identity changed before child start")
    cache_raw = _exact_object(
        "compile private cache environment",
        row["cache_environment"],
        set(COMPILE_CACHE_ENVIRONMENT_VARIABLES),
    )
    cache_environment: dict[str, str] = {}
    for name in COMPILE_CACHE_ENVIRONMENT_VARIABLES:
        path = Path(_text(f"compile cache {name}", cache_raw[name]))
        if (
            not path.is_absolute()
            or path != path.resolve(strict=False)
            or not path.is_dir()
            or path.is_symlink()
        ):
            raise ValueError("compile private cache path is unavailable")
        cache_environment[name] = str(path)
    return launch, cache_environment


def _begin_body_from_launch(
    launch: CompileLaunchManifest,
    *,
    assignment_plan_sha256: str,
) -> tuple[
    dict[str, object],
    CompileOnlyPrewarmManifest,
    SamplingProfile,
]:
    _, sampling_profile_type, _, _ = _serving_runtime()
    cache_plan = CompileCacheLaunchPlan.load(launch.compile_cache_plan_path)
    manifest = CompileOnlyPrewarmManifest.from_dict(
        json.loads(Path(launch.prewarm_manifest_path).read_text(encoding="utf-8"))
    )
    sampling = sampling_profile_type.load(launch.sampling_profile_path)
    if (
        cache_plan.sha256 != launch.compile_cache_plan_sha256
        or manifest.sha256 != launch.prewarm_manifest_sha256
        or sampling.sha256 != launch.sampling_profile_sha256
    ):
        raise ValueError("compile launch inputs changed inside the child")
    ordered = [
        {
            "graph_bucket": payload.graph_bucket,
            "request_id": payload.request_id,
            "input_token_ids_sha256": _content_sha256(list(payload.input_token_ids)),
            "requested_output_tokens": payload.requested_output_tokens,
            "sampling_seed": payload.sampling_seed,
        }
        for payload in manifest.payloads
    ]
    return (
        {
            "assignment_plan_sha256": assignment_plan_sha256,
            "compile_key_sha256": cache_plan.key.sha256,
            "model_lock_sha256": launch.model_lock_sha256,
            "sampling_profile_sha256": launch.sampling_profile_sha256,
            "physical_assignment_sha256": launch.physical_assignment_sha256,
            "experiment_budget_sha256": launch.experiment_budget_sha256,
            "inventory_sha256": launch.inventory_sha256,
            "prewarm_manifest_sha256": manifest.sha256,
            "prewarm": ordered,
        },
        manifest,
        sampling,
    )


def _wait_for_server(process: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.monotonic() + 590.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("compile SGLang server exited before readiness")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError("compile SGLang server readiness timed out")


def _spawn_server(launch: CompileLaunchManifest) -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        launch.server_argv,
        cwd=launch.patched_sglang_checkout,
        env=dict(os.environ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    _wait_for_server(process, launch.localhost_port)
    return process


def _terminate_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


class _CompileWorkerChild:
    """Single-use JSON-lines child with one exact native server lifecycle."""

    def __init__(self, assignment_plan_sha256: str) -> None:
        self.assignment_plan_sha256 = _sha256(
            "compile child assignment plan", assignment_plan_sha256
        )
        self.launch: CompileLaunchManifest | None = None
        self.transport: PinnedBenchServingTransport | None = None
        self.server: subprocess.Popen[bytes] | None = None
        self.begin_sha256: str | None = None
        self.manifest: CompileOnlyPrewarmManifest | None = None
        self.sampling: SamplingProfile | None = None
        self.next_prewarm = 0

    def response(self, kind: str, **fields: object) -> dict[str, object]:
        return {
            "kind": kind,
            "protocol_sha256": COMPILE_SUBPROCESS_LIFECYCLE_PROTOCOL_SHA256,
            "assignment_plan_sha256": self.assignment_plan_sha256,
            **fields,
        }

    async def start(self, row: dict[str, object]) -> None:
        _require_protocol_row(
            row,
            assignment_plan_sha256=self.assignment_plan_sha256,
            kind="compile_subprocess_start",
            fields={"cache_environment", "launch_manifest"},
        )
        launch, cache_environment = _load_launch_from_start(row)
        os.environ.update(cache_environment)
        *_, transport_type = _serving_runtime()
        begin_body, manifest, sampling = _begin_body_from_launch(
            launch,
            assignment_plan_sha256=self.assignment_plan_sha256,
        )
        server = await asyncio.to_thread(_spawn_server, launch)
        self.server = server
        transport = transport_type.from_checkout(launch.patched_sglang_checkout)
        await transport.open(request_timeout_s=580.0, abort_timeout_s=30.0)
        transport.bind_native_admin_base_url(
            f"http://127.0.0.1:{launch.localhost_port}"
        )
        begin = _exact_object(
            "native compile begin",
            await transport.post_json(NATIVE_COMPILE_BEGIN_PATH, begin_body),
            _BEGIN_FIELDS,
        )
        self.begin_sha256 = _sha256(
            "native compile begin receipt", begin["begin_sha256"]
        )
        self.launch = launch
        self.transport = transport
        self.manifest = manifest
        self.sampling = sampling

    async def prewarm(self, row: dict[str, object]) -> str:
        _require_protocol_row(
            row,
            assignment_plan_sha256=self.assignment_plan_sha256,
            kind="compile_subprocess_prewarm",
            fields={
                "request_id",
                "graph_bucket",
                "input_token_ids",
                "requested_output_tokens",
                "sampling_seed",
            },
        )
        if (
            self.launch is None
            or self.transport is None
            or self.manifest is None
            or self.sampling is None
            or self.next_prewarm >= len(self.manifest.payloads)
        ):
            raise RuntimeError("compile child prewarm is out of lifecycle order")
        payload = CompileOnlyPrewarmPayload.from_dict(
            {
                key: value
                for key, value in row.items()
                if key
                in {
                    "request_id",
                    "graph_bucket",
                    "input_token_ids",
                    "requested_output_tokens",
                    "sampling_seed",
                }
            }
        )
        if payload != self.manifest.payloads[self.next_prewarm]:
            raise ValueError("compile child prewarm differs from launch manifest")
        frozen_sampling_type, _, request_type, _ = _serving_runtime()
        sampling = frozen_sampling_type.from_mapping(
            self.sampling.parameters(
                seed=payload.sampling_seed,
                max_new_tokens=payload.requested_output_tokens,
            )
        )
        cohort_id = f"compile:{self.assignment_plan_sha256}"
        request = request_type(
            request_id=payload.request_id,
            namespace="compile",
            split="warmup",
            ordinal=self.next_prewarm,
            input_token_ids=payload.input_token_ids,
            requested_output_tokens=payload.requested_output_tokens,
            arrival_us=0,
            cancellation_offset_us=None,
            cohort_id=cohort_id,
            cohort_sha256=_content_sha256(cohort_id),
            route_id="compile-localhost",
            sampling=sampling,
        )
        request.validate()
        result = await self.transport.submit(
            request,
            base_url=f"http://127.0.0.1:{self.launch.localhost_port}",
            served_model=self.launch.target_model_id,
        )
        if (
            not result.success
            or result.output_tokens != payload.requested_output_tokens
        ):
            raise RuntimeError("compile prewarm request did not complete exactly")
        terminal = _exact_object(
            "native compile terminal",
            await self.transport.post_json(
                NATIVE_COMPILE_TERMINAL_PATH,
                {"request_id": payload.request_id},
            ),
            _TERMINAL_FIELDS,
        )
        if (
            terminal["request_id"] != payload.request_id
            or terminal["graph_bucket"] != payload.graph_bucket
        ):
            raise ValueError("native compile terminal differs from prewarm payload")
        self.next_prewarm += 1
        return _sha256("native compile terminal", terminal["terminal_sha256"])

    async def shutdown(self, row: dict[str, object]) -> str:
        _require_protocol_row(
            row,
            assignment_plan_sha256=self.assignment_plan_sha256,
            kind="compile_subprocess_shutdown",
            fields=set(),
        )
        if (
            self.transport is None
            or self.manifest is None
            or self.begin_sha256 is None
            or self.next_prewarm != len(self.manifest.payloads)
        ):
            raise RuntimeError("compile child shutdown lacks complete prewarm coverage")
        final = _exact_object(
            "native compile final",
            await self.transport.post_json(
                NATIVE_COMPILE_FINALIZE_PATH,
                {"begin_sha256": self.begin_sha256},
            ),
            _FINAL_FIELDS,
        )
        provider_ack = _sha256(
            "native compile receipt", final["compile_receipt_sha256"]
        )
        await self.transport.close()
        self.transport = None
        if self.server is None:
            raise AssertionError("compile child lost its server process")
        _terminate_server(self.server)
        return provider_ack

    async def abort(self) -> None:
        if self.transport is not None:
            with suppress(Exception):
                await self.transport.close()
            self.transport = None
        if self.server is not None:
            _terminate_server(self.server)


async def _child_main() -> int:
    assignment_plan_sha256 = os.environ.get(_ASSIGNMENT_ENVIRONMENT, "")
    child = _CompileWorkerChild(assignment_plan_sha256)
    _write_protocol_message(
        child.response("compile_subprocess_ready", process_id=os.getpid())
    )
    try:
        start = await asyncio.to_thread(_read_protocol_message)
        await child.start(start)
        _write_protocol_message(
            child.response("compile_subprocess_started", process_id=os.getpid())
        )
        while True:
            row = await asyncio.to_thread(_read_protocol_message)
            kind = row.get("kind")
            if kind == "compile_subprocess_prewarm":
                terminal_sha256 = await child.prewarm(row)
                _write_protocol_message(
                    child.response(
                        "compile_subprocess_prewarm_complete",
                        request_id=row["request_id"],
                        graph_bucket=row["graph_bucket"],
                        completed=True,
                        provider_receipt_sha256=terminal_sha256,
                    )
                )
                continue
            if kind != "compile_subprocess_shutdown":
                raise ValueError("compile child received an out-of-order command")
            provider_ack = await child.shutdown(row)
            _write_protocol_message(
                child.response(
                    "compile_subprocess_drained",
                    active_requests=0,
                    queued_requests=0,
                    provider_ack_sha256=provider_ack,
                )
            )
            return 0
    except Exception:  # noqa: BLE001 - protocol boundary must fail without stdout
        await child.abort()
        return 1


def main() -> int:
    """Run the private child protocol; never emit non-protocol stdout."""

    try:
        return asyncio.run(_child_main())
    except Exception:  # noqa: BLE001 - keep child stdout protocol-only
        return 1


__all__ = [
    "GPU_COMPILE_REASON",
    "GPU_COMPILE_SEMANTICS",
    "NATIVE_COMPILE_BEGIN_PATH",
    "NATIVE_COMPILE_FINALIZE_PATH",
    "NATIVE_COMPILE_TERMINAL_PATH",
    "SOURCE_OWNED_COMPILE_HOOK",
    "SOURCE_OWNED_COMPILE_PROTOCOL_SHA256",
    "SOURCE_OWNED_COMPILE_RELEASE_STATUS",
    "PinnedCompileLifecycleWorker",
    "SourceOwnedCompileNativeContract",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
