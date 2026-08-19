from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments import formal_single_operator_profiler as profiler
from lightcone_spec.experiments.formal_registry import (
    stage_materialization_receipt_to_dict,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    publish_formal_single_operator_json_artifact,
)
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    _materialize_e4_profiler_diagnostic,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _current_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    materialization = _materialize_e4_profiler_diagnostic(
        protocol_lock_sha256=_sha("lock"),
        upstream_local_receipt_sha256=_sha("local"),
        source_decision_sha256=_sha("decision"),
        selected_configuration_sha256=_sha("configuration"),
        model="Qwen/Qwen3-8B",
        lightcone_recipe_sha256=_sha("recipe"),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    materialization_path = tmp_path / "e4-profiler-materialization.json"
    materialization_binding = publish_formal_single_operator_json_artifact(
        materialization_path,
        stage_materialization_receipt_to_dict(materialization),
    )
    execution_source_path = tmp_path / "e4-profiler-execution-source.json"
    publish_formal_single_operator_json_artifact(
        execution_source_path,
        {"kind": "test-current-e4-profiler-execution-source"},
    )
    source = SimpleNamespace(
        node="e4_profiler",
        stage="E4",
        phase="profiler",
        sha256=_sha("current-execution-source"),
        materialization_source=materialization_binding,
        materialization_sha256=materialization.sha256,
        protocol_lock_sha256=materialization.protocol_lock_sha256,
    )
    monkeypatch.setattr(
        profiler,
        "load_formal_single_operator_execution_source",
        lambda path: source,
    )
    return execution_source_path, materialization


def _fake_tool(tmp_path: Path, tool: str) -> tuple[Path, Path]:
    path = tmp_path / tool
    log = tmp_path / f"{tool}-argv.json"
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

if sys.argv[1:] == ["--version"]:
    print("fake-nsight-tool 2026.3")
    raise SystemExit(0)

prefix = "--output=" if pathlib.Path(sys.argv[0]).name == "nsys" else "--export="
suffix = ".nsys-rep" if pathlib.Path(sys.argv[0]).name == "nsys" else ".ncu-rep"
output = next(value.removeprefix(prefix) for value in sys.argv[1:] if value.startswith(prefix))
pathlib.Path(output + suffix).write_bytes(b"cpu-fake-profiler-report")
pathlib.Path(os.environ["LIGHTCONE_TEST_PROFILER_ARGV"]).write_text(
    json.dumps(sys.argv[1:]), encoding="utf-8"
)
""",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path, log


def _canonical_binding(tmp_path: Path, name: str) -> CanonicalJsonProofBinding:
    path = tmp_path / f"{name}.json"
    publish_canonical_json_no_replace(path, {"kind": name})
    return CanonicalJsonProofBinding.bind(path)


@pytest.mark.parametrize(
    ("variant", "tool", "trace_flag", "suffix"),
    [
        ("nvtx", "nsys", "--trace=cuda,nvtx", ".nsys-rep"),
        (
            "nsight_systems",
            "nsys",
            "--trace=cuda,nvtx,osrt,cublas,cudnn",
            ".nsys-rep",
        ),
        ("nsight_compute", "ncu", "--target-processes=all", ".ncu-rep"),
    ],
)
def test_private_capture_uses_current_cell_registered_template_and_terminal_last(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    variant: str,
    tool: str,
    trace_flag: str,
    suffix: str,
) -> None:
    monkeypatch.setattr(
        profiler,
        "_revalidate_profiler_subject_argv",
        lambda *args, **kwargs: None,
    )
    execution_source_path, materialization = _current_source(
        monkeypatch,
        tmp_path,
    )
    cell = next(
        row
        for row in materialization.cells
        if dict(row.dimensions)["profiler"] == variant
    )
    tool_path, log_path = _fake_tool(tmp_path, tool)
    monkeypatch.setenv("LIGHTCONE_TEST_PROFILER_ARGV", str(log_path))
    output = tmp_path / f"capture-{variant}"

    receipt = profiler._run_formal_single_operator_profiler_capture(
        execution_source_path=execution_source_path,
        materialized_cell_id=cell.cell_id,
        tool_path=tool_path,
        subject_argv=("/usr/bin/true", "--source-owned-test-subject"),
        output_directory=output,
    )

    assert receipt.status == "COMPLETE"
    assert receipt.variant == variant
    assert receipt.tool_identity.absolute_path == str(tool_path)
    assert (
        receipt.tool_identity.raw_sha256
        == hashlib.sha256(tool_path.read_bytes()).hexdigest()
    )
    assert "fake-nsight-tool 2026.3" in receipt.tool_identity.version_stdout
    assert receipt.headline_eligible is False
    recorded = json.loads(log_path.read_text(encoding="utf-8"))
    assert trace_flag in recorded
    assert recorded[-3:] == ["--", "/usr/bin/true", "--source-owned-test-subject"]
    raw_path = output / f"profile{suffix}"
    terminal_path = output / "profiler-terminal.json"
    assert raw_path.read_bytes() == b"cpu-fake-profiler-report"
    assert terminal_path.stat().st_mtime_ns >= raw_path.stat().st_mtime_ns
    assert (
        profiler.load_formal_single_operator_profiler_terminal(terminal_path) == receipt
    )

    with pytest.raises(RuntimeError, match="already exists"):
        profiler._run_formal_single_operator_profiler_capture(
            execution_source_path=execution_source_path,
            materialized_cell_id=cell.cell_id,
            tool_path=tool_path,
            subject_argv=("/usr/bin/true",),
            output_directory=output,
        )


def test_public_runner_accepts_only_generated_plan_and_uses_its_exact_subject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parameters = inspect.signature(
        profiler.run_formal_single_operator_profiler
    ).parameters
    assert set(parameters) == {"profiler_plan_path", "current_ns"}
    assert all("argv" not in name and "digest" not in name for name in parameters)
    expected = object()
    plan = SimpleNamespace(
        execution_source=SimpleNamespace(absolute_path="/source.json"),
        materialized_cell_id=_sha("cell"),
        tool_identity=SimpleNamespace(absolute_path="/tools/nsys"),
        subject_argv=("/python", "-m", "lightcone_spec.cli.main"),
        output_directory=str(tmp_path / "capture"),
    )
    monkeypatch.setattr(
        profiler,
        "revalidate_formal_single_operator_profiler_plan",
        lambda path, current_ns: plan,
    )
    observed: dict[str, object] = {}

    def capture(**kwargs):
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(
        profiler,
        "_run_formal_single_operator_profiler_capture",
        capture,
    )
    assert (
        profiler.run_formal_single_operator_profiler(
            profiler_plan_path=tmp_path / "profiler-plan.json",
            current_ns=10,
        )
        is expected
    )
    assert observed == {
        "execution_source_path": "/source.json",
        "materialized_cell_id": _sha("cell"),
        "tool_path": "/tools/nsys",
        "subject_argv": plan.subject_argv,
        "output_directory": str(tmp_path / "capture"),
    }


def test_capture_plan_codec_rejects_any_subject_command_mutation(
    tmp_path: Path,
) -> None:
    tool_path, _log = _fake_tool(tmp_path, "nsys")
    tool = profiler.probe_formal_single_operator_profiler_tool(
        expected_tool="nsys",
        tool_path=tool_path,
    )
    subject_plan = _canonical_binding(tmp_path, "subject-plan")
    argv = profiler._profiler_subject_argv(
        repository_root=tmp_path,
        run_plan_path=subject_plan.absolute_path,
    )
    plan = profiler.FormalSingleOperatorProfilerCapturePlan(
        schema_version=1,
        kind="formal_single_operator_profiler_capture_plan",
        protocol_sha256=profiler.FORMAL_SINGLE_OPERATOR_PROFILER_PROTOCOL_SHA256,
        execution_source=_canonical_binding(tmp_path, "execution-source"),
        execution_source_sha256=_sha("execution-source-semantic"),
        prepared_launch_bundle=_canonical_binding(tmp_path, "prepared-bundle"),
        prepared_launch_bundle_sha256=_sha("prepared-bundle-semantic"),
        prepared_launch_entry_sha256=_sha("prepared-entry"),
        materialized_cell_id=_sha("profiler-cell"),
        variant="nvtx",
        subject_inputs=_canonical_binding(tmp_path, "subject-inputs"),
        subject_inputs_sha256=_sha("subject-inputs-semantic"),
        subject_run_plan=subject_plan,
        subject_run_plan_sha256=_sha("subject-plan-semantic"),
        subject_argv=argv,
        subject_argv_sha256=profiler._content_sha256({"argv": list(argv)}),
        tool_identity=tool,
        repository_root=str(tmp_path),
        output_directory=str(tmp_path / "capture"),
        created_ns=1,
        headline_eligible=False,
    )
    assert (
        profiler.FormalSingleOperatorProfilerCapturePlan.from_dict(plan.to_dict())
        == plan
    )
    tampered = plan.to_dict()
    tampered["subject_argv"][-1] = "/tmp/foreign-plan.json"
    with pytest.raises(ValueError, match="subject argv digest"):
        profiler.FormalSingleOperatorProfilerCapturePlan.from_dict(tampered)


def test_subject_argv_revalidator_joins_current_profiler_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _canonical_binding(tmp_path, "current-execution-source")
    subject_inputs = _canonical_binding(tmp_path, "subject-inputs-command")
    subject_plan = _canonical_binding(tmp_path, "subject-run-plan-command")
    profiler_cell_id = _sha("current-profiler-cell")
    headline_cell_id = _sha("selected-headline-cell")
    argv = profiler._profiler_subject_argv(
        repository_root=tmp_path,
        run_plan_path=subject_plan.absolute_path,
    )
    from lightcone_spec.orchestration import formal_physical_dispatch as dispatch

    monkeypatch.setattr(
        dispatch,
        "revalidate_formal_single_operator_profiler_subject_run_plan",
        lambda path, current_ns: SimpleNamespace(
            single_operator_execution_rebuild_source=subject_inputs,
            materialized_cell_id=headline_cell_id,
        ),
    )
    monkeypatch.setattr(
        profiler,
        "revalidate_formal_single_operator_profiler_subject_inputs",
        lambda path, current_ns: SimpleNamespace(
            execution_source=source,
            profiler_cell_id=profiler_cell_id,
            repository_root=str(tmp_path),
            source_headline_cell_id=headline_cell_id,
        ),
    )
    profiler._revalidate_profiler_subject_argv(
        argv,
        execution_source_path=source.absolute_path,
        materialized_cell_id=profiler_cell_id,
        current_ns=1,
    )
    mutated = (*argv[:2], "foreign.module", *argv[3:])
    with pytest.raises(ValueError, match="not code-owned"):
        profiler._revalidate_profiler_subject_argv(
            mutated,
            execution_source_path=source.absolute_path,
            materialized_cell_id=profiler_cell_id,
            current_ns=1,
        )


def test_probe_rejects_non_absolute_or_wrong_tool_and_rehash_detects_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        profiler,
        "_revalidate_profiler_subject_argv",
        lambda *args, **kwargs: None,
    )
    execution_source_path, materialization = _current_source(
        monkeypatch,
        tmp_path,
    )
    cell = next(
        row
        for row in materialization.cells
        if dict(row.dimensions)["profiler"] == "nvtx"
    )
    tool_path, log_path = _fake_tool(tmp_path, "nsys")
    monkeypatch.setenv("LIGHTCONE_TEST_PROFILER_ARGV", str(log_path))
    with pytest.raises(ValueError, match="absolute and normalized"):
        profiler.probe_formal_single_operator_profiler_tool(
            expected_tool="nsys",
            tool_path="nsys",
        )
    with pytest.raises(ValueError, match="basename differs"):
        profiler.probe_formal_single_operator_profiler_tool(
            expected_tool="ncu",
            tool_path=tool_path,
        )

    output = tmp_path / "capture-tamper"
    profiler._run_formal_single_operator_profiler_capture(
        execution_source_path=execution_source_path,
        materialized_cell_id=cell.cell_id,
        tool_path=tool_path,
        subject_argv=("/usr/bin/true",),
        output_directory=output,
    )
    raw = output / "profile.nsys-rep"
    raw.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="raw output identity changed"):
        profiler.load_formal_single_operator_profiler_terminal(
            output / "profiler-terminal.json"
        )
