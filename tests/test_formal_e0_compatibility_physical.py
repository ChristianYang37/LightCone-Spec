from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
    E0CompatibilityProbeTerminal,
    E0Eagle3RuntimeProofRow,
)
from lightcone_spec.orchestration import formal_e0_compatibility_physical as physical
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _json_binding(
    tmp_path: Path, name: str, value: object
) -> CanonicalJsonProofBinding:
    path = (tmp_path / f"{name}.json").resolve()
    publish_canonical_json_no_replace(path, value)
    return CanonicalJsonProofBinding.bind(path)


def _raw_binding(tmp_path: Path, name: str, body: bytes) -> EvidenceFileBinding:
    path = (tmp_path / name).resolve()
    path.write_bytes(body)
    return EvidenceFileBinding.bind(path, label=name)


def test_e0_physical_universe_is_canonical_exact_108() -> None:
    keys = physical.expected_e0_compatibility_probe_keys()

    assert len(keys) == 108
    assert keys == tuple(sorted(set(keys)))
    assert {model for model, _, _ in keys} == set(physical.E0_MODELS)
    assert {backend for _, backend, _ in keys} == set(physical.E0_BACKENDS)
    assert {task for _, _, task in keys} == set(physical.E0_TASKS)


def test_e0_campaign_process_timeout_replays_exact_12x9_source_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = tuple(
        SimpleNamespace(absolute_path=f"/bound/group-{index}.json")
        for index in range(12)
    )
    groups = {
        binding.absolute_path: SimpleNamespace(
            probes=tuple(object() for _ in range(9)),
            compile_launch_manifest=object(),
        )
        for binding in bindings
    }
    monkeypatch.setattr(
        physical,
        "revalidate_formal_e0_compatibility_physical_campaign",
        lambda _path: SimpleNamespace(groups=bindings),
    )
    monkeypatch.setattr(
        physical,
        "revalidate_formal_e0_compatibility_probe_group",
        lambda path: groups[path],
    )

    assert (
        physical.formal_e0_compatibility_process_hard_timeout_ns("/bound/campaign.json")
        == 75_120 * 1_000_000_000
    )

    groups[bindings[0].absolute_path] = SimpleNamespace(
        probes=tuple(object() for _ in range(8)),
        compile_launch_manifest=object(),
    )
    with pytest.raises(ValueError, match="12x9"):
        physical.formal_e0_compatibility_process_hard_timeout_ns("/bound/campaign.json")


def test_probe_plans_do_not_accept_caller_timestamps() -> None:
    field_names = {field.name for field in fields(physical.E0CompatibilityProbePlan)}

    assert "started_ns" not in field_names
    assert "finished_ns" not in field_names
    assert "timestamp" not in field_names


def test_clock_after_uses_a_later_real_clock_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter((17, 17, 18))
    monkeypatch.setattr(physical.time, "monotonic_ns", lambda: next(readings))

    assert physical._clock_after(17) == 18


@pytest.mark.parametrize(
    ("interface_status", "workload_status", "expected"),
    (
        (
            "UNSUPPORTED",
            "READY",
            (
                "N/A",
                "MODEL_BACKEND_INTERFACE_UNSUPPORTED",
                "NOT_REQUIRED",
                0,
            ),
        ),
        (
            "READY",
            "UNSUPPORTED",
            (
                "N/A",
                "TOKENIZER_TASK_WORKLOAD_UNSUPPORTED",
                "NOT_REQUIRED",
                0,
            ),
        ),
        (
            "READY",
            "READY",
            ("VALID", "PROBE_COMPATIBLE", "PASS", 1),
        ),
    ),
)
def test_na_is_derived_only_from_registered_source_status(
    interface_status: str,
    workload_status: str,
    expected: tuple[str, str, str, int],
) -> None:
    assert (
        physical._disposition(
            interface=SimpleNamespace(support_status=interface_status),
            workload=SimpleNamespace(support_status=workload_status),
        )
        == expected
    )


class _FakeRequest:
    request_id = "e0-probe-request"
    sha256 = content_sha256("request")
    input_token_ids = (11, 12)


class _FakeResult:
    def __init__(self, *, success: bool) -> None:
        self.success = success
        self.output_tokens = 1 if success else 0
        self.generated_token_ids = (13,) if success else ()
        self.latency_us = 100
        self.ttft_us = 50 if success else None
        self.stop_reason = "length" if success else "error"

    def validate(self, request: object) -> None:
        assert request is _FAKE_REQUEST


class _FakeTransport:
    def __init__(self, *, success: bool) -> None:
        self.success = success
        self.calls: list[tuple[object, str, str]] = []

    async def submit(
        self, request: object, *, base_url: str, served_model: str
    ) -> _FakeResult:
        self.calls.append((request, base_url, served_model))
        return _FakeResult(success=self.success)

    def metrics(self) -> dict[str, int]:
        return {"submitted_requests": len(self.calls)}


_FAKE_REQUEST = _FakeRequest()


def _patch_one_request_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        physical,
        "_load_probe_source",
        lambda binding, *, task: ("sample", "prompt", 7, "a", "b", "READY"),
    )
    monkeypatch.setattr(
        physical,
        "_bound_probe_request",
        lambda *, plan, launch, prompt: _FAKE_REQUEST,
    )
    monkeypatch.setattr(
        physical,
        "load_run_config",
        lambda path: SimpleNamespace(model=SimpleNamespace(target="target")),
    )


def test_ready_probe_submits_exactly_one_real_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_one_request_inputs(monkeypatch)
    plan = SimpleNamespace(
        workload_source=object(), task="GSM8K", model="Qwen3-4B", backend="DFLASH"
    )
    launch = SimpleNamespace(localhost_port=31900, run_config_path="config.json")
    transport = _FakeTransport(success=True)

    rows = asyncio.run(
        physical._run_gpu_probes(plans=(plan,), launch=launch, transport=transport)
    )

    assert len(transport.calls) == 1
    assert len(rows) == 1
    assert rows[0].result is not None
    assert rows[0].result["output_token_count"] == 1
    assert rows[0].result["output_token_ids"] == [13]


def test_gpu_probe_failure_is_not_converted_to_na(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_one_request_inputs(monkeypatch)
    plan = SimpleNamespace(
        workload_source=object(), task="GSM8K", model="Qwen3-4B", backend="DFLASH"
    )
    launch = SimpleNamespace(localhost_port=31900, run_config_path="config.json")
    transport = _FakeTransport(success=False)

    with pytest.raises(
        physical.FormalE0CompatibilityPhysicalBlocked,
        match="one_request_gpu_smoke_did_not_complete",
    ):
        asyncio.run(
            physical._run_gpu_probes(plans=(plan,), launch=launch, transport=transport)
        )

    assert len(transport.calls) == 1


def test_missing_source_is_blocking_not_na(tmp_path: Path) -> None:
    with pytest.raises(
        physical.FormalE0CompatibilityPhysicalBlocked,
        match="source_owned_livecodebench_v6_hard_missing",
    ):
        physical._source_bindings(
            bundle=SimpleNamespace(locked_workloads=(), e0_task_native_descriptors=()),
            root=tmp_path,
        )


def test_raw_evidence_is_atomic_no_replace(tmp_path: Path) -> None:
    path = tmp_path / "raw.log"

    first = physical._publish_raw_no_replace(path, b"first\n")
    with pytest.raises(FileExistsError):
        physical._publish_raw_no_replace(path, b"replacement\n")

    first.reopen(label="test raw")
    assert path.read_bytes() == b"first\n"


def test_existing_terminal_is_deep_revalidated_for_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "probe-terminal.json"
    path.write_text("{}\n", encoding="utf-8")
    plan = SimpleNamespace(terminal_output_path=str(path))
    monkeypatch.setattr(
        physical,
        "revalidate_formal_e0_compatibility_physical_terminal",
        lambda candidate: (_ for _ in ()).throw(ValueError("tampered evidence")),
    )

    with pytest.raises(ValueError, match="tampered evidence"):
        physical._completed_terminal(plan)


def test_restart_reuses_nine_complete_terminals_and_repairs_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = tuple(
        SimpleNamespace(absolute_path=f"/probe/{index}.json") for index in range(9)
    )
    group = SimpleNamespace(probes=bindings)
    plans = tuple(SimpleNamespace(task=str(index)) for index in range(9))
    terminals = tuple(SimpleNamespace(task=str(index)) for index in range(9))
    by_path = {
        binding.absolute_path: plan
        for binding, plan in zip(bindings, plans, strict=True)
    }
    calls: list[int] = []
    monkeypatch.setattr(
        physical,
        "revalidate_formal_e0_compatibility_probe_group",
        lambda path: group,
    )
    monkeypatch.setattr(
        physical,
        "revalidate_formal_e0_compatibility_probe_plan",
        lambda path: by_path[path],
    )
    monkeypatch.setattr(
        physical,
        "_completed_terminal",
        lambda plan: terminals[int(plan.task)],
    )
    monkeypatch.setattr(
        physical,
        "_attempt_root",
        lambda group: (_ for _ in ()).throw(AssertionError("unexpected retry")),
    )
    monkeypatch.setattr(
        physical,
        "_publish_or_revalidate_group_completion",
        lambda **kwargs: calls.append(
            kwargs["physical_server_launch_count_this_attempt"]
        ),
    )

    resumed = asyncio.run(physical._execute_group_unlocked("/group.json"))

    assert resumed == terminals
    assert calls == [0]


def test_group_failure_receipt_explicitly_forbids_na(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readings = iter((101, 102))
    monkeypatch.setattr(physical.time, "monotonic_ns", lambda: next(readings))
    path = tmp_path / "failure.json"
    group = SimpleNamespace(
        sha256=content_sha256("group"),
        argv_sha256=content_sha256("argv"),
        model="Qwen3-4B",
        backend="DFLASH",
    )

    physical._publish_group_failure(
        path=path,
        group=group,
        attempt_started_ns=101,
        error=RuntimeError("secret caller text is not serialized"),
        process_id=None,
        process_exit_code=None,
        process_started_ns=None,
        server_ready_ns=None,
        process_exited_ns=None,
        completed_request_count=0,
        before_snapshot=None,
        ready_snapshot=None,
        after_snapshot=None,
        server_stdout=None,
        server_stderr=None,
    )
    value = physical.CanonicalJsonProofBinding.bind(path).reopen()

    assert value["started_ns"] == 101
    assert value["finished_ns"] == 102
    assert value["status"] == "FAILED"
    assert value["failure_is_na"] is False
    assert value["reason_code"] == "physical_probe_failed:RuntimeError"
    assert "secret caller text" not in path.read_text(encoding="utf-8")


def test_eagle3_postprobe_row_is_derived_from_core_without_terminal_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = physical.E0_TASKS[0]
    model = physical.E0_MODELS[0]
    workload_sha256 = _sha("workload")
    command_sha256 = _sha("command")
    started_ns = 1
    finished_ns = 2
    plan_binding = _json_binding(
        tmp_path,
        "plan",
        {
            "model": model,
            "backend": "EAGLE3",
            "task": task,
            "gpu_uuid": "GPU-0",
        },
    )
    interface_binding = _json_binding(tmp_path, "interface", {"preprobe": True})
    workload_binding = _json_binding(
        tmp_path,
        "workload",
        {
            "model": model,
            "task": task,
            "task_native_workload_sha256": workload_sha256,
        },
    )
    result = _json_binding(
        tmp_path,
        "result",
        {"output_token_count": 1, "output_token_ids": [17]},
    )
    lifecycle = _json_binding(
        tmp_path,
        "lifecycle",
        {
            "status": "COMPLETE",
            "request_started_ns": started_ns,
            "finished_ns": finished_ns,
            "completed_request_count": 1,
            "gpu_uuid": "GPU-0",
        },
    )
    raw = {
        name: _raw_binding(tmp_path, f"{name}.log", f"{name}\n".encode())
        for name in ("stdout", "stderr", "junit", "server_stdout", "server_stderr")
    }
    core = _json_binding(
        tmp_path,
        "core",
        {
            "kind": "formal_single_operator_e0_compatibility_probe_evidence",
            "plan": plan_binding.to_dict(),
            "interface_receipt": interface_binding.to_dict(),
            "workload_authority": workload_binding.to_dict(),
            "lifecycle": lifecycle.to_dict(),
            "result": result.to_dict(),
            "command_sha256": command_sha256,
            "started_ns": started_ns,
            "finished_ns": finished_ns,
            "completed_request_count": 1,
            **{name: binding.to_dict() for name, binding in raw.items()},
        },
    )
    inventory = _json_binding(tmp_path, "inventory", {"inventory": "bound"})
    launch = _json_binding(tmp_path, "launch", {"launch": "static-eagle3"})
    plan = SimpleNamespace(
        backend="EAGLE3",
        execution_kind="GPU_ONE_REQUEST",
        gpu_uuid="GPU-0",
        task=task,
        model=model,
        sha256=_sha("plan"),
    )
    interface = SimpleNamespace(
        schema_version=3,
        backend="EAGLE3",
        support_status="READY",
        compile_launch_manifest=launch,
        interface_sha256=_sha("interface"),
        target_revision="1" * 40,
        drafter_revision="2" * 40,
    )
    group = SimpleNamespace(
        compile_launch_manifest=launch,
        inventory=inventory,
    )
    monkeypatch.setattr(
        physical.CompileLaunchManifest,
        "load",
        classmethod(lambda _cls, _path: SimpleNamespace(gpu_uuids=("GPU-0",))),
    )

    proof_binding = physical._publish_eagle3_postprobe_proof_row(
        plan=plan,
        attempt_root=tmp_path,
        interface=interface,
        group=group,
        core=core,
        result=result,
        lifecycle=lifecycle,
    )
    proof = E0Eagle3RuntimeProofRow.from_dict(proof_binding.reopen())
    native = proof.native_gpu_proof.reopen()

    assert proof.schema_version == 2
    assert CanonicalJsonProofBinding.from_dict(native["core_evidence"]) == core
    assert CanonicalJsonProofBinding.from_dict(native["result"]) == result
    assert CanonicalJsonProofBinding.from_dict(native["lifecycle"]) == lifecycle
    assert "terminal" not in json.dumps(proof.to_dict(), sort_keys=True)

    terminal = E0CompatibilityProbeTerminal(
        schema_version=3,
        protocol_lock_sha256=_sha("lock"),
        upstream_e6_confirmation_sha256=_sha("e6"),
        model=model,
        backend="EAGLE3",
        task=task,
        interface_sha256=interface.interface_sha256,
        task_native_workload_sha256=workload_sha256,
        tokenizer_sha256=_sha("tokenizer"),
        command_sha256=command_sha256,
        started_ns=started_ns,
        finished_ns=finished_ns,
        terminal_status="COMPLETE",
        exit_code=0,
        stdout_sha256=raw["stdout"].raw_sha256,
        stderr_sha256=raw["stderr"].raw_sha256,
        junit_sha256=raw["junit"].raw_sha256,
        junit_status="PASS",
        evidence_sha256=core.semantic_sha256,
        smoke_status="PASS",
        completed_request_count=1,
        disposition="VALID",
        reason_code="PROBE_COMPATIBLE",
        interface_receipt_sha256=interface_binding.semantic_sha256,
        compile_launch_manifest_sha256=launch.semantic_sha256,
        eagle3_runtime_proof_row_sha256=proof.sha256,
        eagle3_runtime_proof_row=proof_binding,
    )
    assert terminal.eagle3_runtime_proof_row == proof_binding
    with pytest.raises(ValueError, match="proof row differs"):
        replace(terminal, stdout_sha256=_sha("mutated-stdout"))
    with pytest.raises(ValueError, match="proof row differs"):
        replace(
            terminal,
            compile_launch_manifest_sha256=_sha("mutated-launch"),
        )
    with pytest.raises(ValueError, match="proof row differs"):
        replace(terminal, task=physical.E0_TASKS[1])

    mutated = proof.to_dict()
    mutated["task"] = physical.E0_TASKS[1]
    with pytest.raises(ValueError, match="replay differs"):
        E0Eagle3RuntimeProofRow.from_dict(mutated)


def test_fresh_eagle3_preprobe_receipt_has_no_task_authority() -> None:
    receipt = SimpleNamespace(
        backend="EAGLE3",
        support_status="READY",
        schema_version=3,
    )

    assert (
        physical._eagle3_runtime_proof_row_sha256(
            receipt,
            task=physical.E0_TASKS[0],
        )
        is None
    )
