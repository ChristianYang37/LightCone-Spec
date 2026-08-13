"""Thin consumer for the pinned source-owned COMPILE CPU contract.

The worker reuses the official pinned bench transport and its caller-owned
pool.  A submitter may actuate each registered prewarm request, but its return
value is never evidence: only the native scheduler's ordered terminal receipt
can complete this lifecycle.  GPU compile semantics remain unavailable.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments.serving import PinnedBenchServingTransport
from lightcone_spec.runtime.compile_cache import (
    CompileOnlyPrewarmPayload,
    _content_sha256,
)
from lightcone_spec.runtime.compile_runner import CompileAssignmentPlan

NATIVE_COMPILE_BEGIN_PATH = "/v1/lightcone-spec/compile-cache-evidence/begin"
NATIVE_COMPILE_FINALIZE_PATH = "/v1/lightcone-spec/compile-cache-evidence/finalize"
SOURCE_OWNED_COMPILE_HOOK = "sglang.schema_v3.source_owned_compile_cache_lifecycle.v1"
SOURCE_OWNED_COMPILE_RELEASE_STATUS = "CPU_CONTRACT_ONLY"
GPU_COMPILE_SEMANTICS = "PENDING"
GPU_COMPILE_REASON = "gpu_compile_semantics_unavailable"

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
        "schema_version": 1,
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
        "caller_runtime_counters_forbidden": True,
        "gpu_compile_semantics": GPU_COMPILE_SEMANTICS,
        "gpu_measurements": {name: None for name in _GPU_MEASUREMENTS},
    }
)

CompilePrewarmSubmitter = Callable[[CompileOnlyPrewarmPayload], Awaitable[object]]


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
        value["schema_version"] != 1
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
    _assignment, _cache_plan, manifest = plan.revalidate()
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
class SourceOwnedCompileCPUContract:
    """Raw native receipts retained only as a non-authorizing CPU contract."""

    begin_raw: bytes
    final_raw: bytes
    begin_sha256: str
    compile_receipt_sha256: str
    process_id: int
    process_started_ns: int
    ordered_terminal_sha256s: tuple[str, ...]
    release_status: str
    gpu_compile_semantics: str
    gpu_compile_reason: str
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
        if any(
            value != {"value": None, "reason": GPU_COMPILE_REASON}
            for value in measurements.values()
        ):
            raise ValueError("native compile source invented a GPU measurement")
        final_digest = _content_sha256(
            {
                key: value
                for key, value in final.items()
                if key != "compile_receipt_sha256"
            }
        )
        if (
            final["completion_marker"] != "COMPILE_CPU_CONTRACT_COMPLETE"
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
            raise ValueError("native compile CPU contract disposition differs")


class PinnedCompileLifecycleWorker:
    """Run one ordered native COMPILE contract on the official serving pool."""

    def __init__(self, transport: PinnedBenchServingTransport) -> None:
        if not isinstance(transport, PinnedBenchServingTransport):
            raise TypeError("compile worker requires the pinned bench transport")
        self._transport = transport
        self._used = False

    async def execute(
        self,
        plan: CompileAssignmentPlan,
        *,
        submit_prewarm: CompilePrewarmSubmitter,
    ) -> SourceOwnedCompileCPUContract:
        if self._used:
            raise RuntimeError("compile worker is one-shot")
        if type(plan) is not CompileAssignmentPlan:
            raise TypeError("compile worker requires an exact assignment plan")
        if not callable(submit_prewarm):
            raise TypeError("compile worker requires a prewarm submitter")
        self._used = True
        _assignment, _cache_plan, manifest = plan.revalidate()
        begin = await self._transport.post_json(
            NATIVE_COMPILE_BEGIN_PATH,
            _begin_body(plan),
        )
        begin_row = _exact_object("native compile begin", begin, _BEGIN_FIELDS)
        begin_sha256 = _sha256(
            "native compile begin receipt", begin_row["begin_sha256"]
        )
        for payload in manifest.payloads:
            # The response is intentionally ignored.  Native scheduler state,
            # not a submitter summary, supplies the terminal evidence.
            await submit_prewarm(payload)
        final = await self._transport.post_json(
            NATIVE_COMPILE_FINALIZE_PATH,
            {"begin_sha256": begin_sha256},
        )
        final_row = _exact_object("native compile final", final, _FINAL_FIELDS)
        terminals = final_row["ordered_terminals"]
        if type(terminals) is not list:
            raise ValueError("native compile terminal rows must be an array")
        contract = SourceOwnedCompileCPUContract(
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
            ordered_terminal_sha256s=tuple(
                _sha256(
                    "native compile terminal",
                    _exact_object("native compile terminal", row, _TERMINAL_FIELDS)[
                        "terminal_sha256"
                    ],
                )
                for row in terminals
            ),
            release_status=SOURCE_OWNED_COMPILE_RELEASE_STATUS,
            gpu_compile_semantics=GPU_COMPILE_SEMANTICS,
            gpu_compile_reason=GPU_COMPILE_REASON,
            formal_execution_authorized=False,
        )
        contract.validate(plan)
        return contract


__all__ = [
    "GPU_COMPILE_REASON",
    "GPU_COMPILE_SEMANTICS",
    "NATIVE_COMPILE_BEGIN_PATH",
    "NATIVE_COMPILE_FINALIZE_PATH",
    "SOURCE_OWNED_COMPILE_HOOK",
    "SOURCE_OWNED_COMPILE_PROTOCOL_SHA256",
    "SOURCE_OWNED_COMPILE_RELEASE_STATUS",
    "PinnedCompileLifecycleWorker",
    "SourceOwnedCompileCPUContract",
]
