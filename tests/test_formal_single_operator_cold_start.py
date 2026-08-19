from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_formal_method_authority import _plan_source, _tts_window
from test_formal_preflight_inputs import _inventory
from test_formal_single_operator_content import _tiny_burstgpt_assets

import lightcone_spec.experiments.formal_preflight_inputs as preflight_module
import lightcone_spec.experiments.formal_single_operator_content as content_module
import lightcone_spec.experiments.workload_authority as workload_module
import lightcone_spec.orchestration.formal_physical_dispatch as dispatch_module
from lightcone_spec.cli.main import main
from lightcone_spec.experiments import e0_stage_authority
from lightcone_spec.experiments.formal_method_authority import (
    TTS_DRAFTER_NATIVE_LOSS_SOURCE,
)
from lightcone_spec.experiments.formal_registry import (
    e0_onlinespec_source_authority_from_dict,
)
from lightcone_spec.experiments.formal_runtime_manifest import (
    FORMAL_RUNTIME_SOURCE_LAYOUT,
)
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedModelRuntimeBinding,
    TrustedModelSnapshotSpec,
    TrustedNamedInputPath,
    TrustedSingleOperatorContentPathSpec,
    bind_trusted_locked_workload,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FORMAL_SINGLE_OPERATOR_NODE_ORDER,
)
from lightcone_spec.experiments.onlinespec import (
    ONLINE_SPEC_COMMIT,
    ONLINE_SPEC_TREE,
)
from lightcone_spec.orchestration.experiment_operator import (
    ExperimentOperatorStore,
    default_formal_stage_plan,
)
from lightcone_spec.orchestration.formal_single_operator_bootstrap import (
    FormalSingleOperatorBootstrapSupervisor,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
    )


def _clean_runtime_fixture_checkout(tmp_path: Path) -> Path:
    """Create the smallest clean checkout accepted by source-owned binders."""

    source = Path(__file__).resolve().parents[1]
    checkout = (tmp_path / "clean-checkout").resolve()
    checkout.mkdir()
    relative_paths = {
        "pyproject.toml",
        "docs/en/experiment-protocol.md",
        "docs/zh-CN/experiment-protocol.md",
        "src/lightcone_spec/experiments/formal_protocol.py",
        "patches/sglang/manifest.json",
        *(
            relative
            for layout in FORMAL_RUNTIME_SOURCE_LAYOUT
            for relative in layout.runner_sources
        ),
        *(
            node.partition("::")[0]
            for layout in FORMAL_RUNTIME_SOURCE_LAYOUT
            for node in layout.test_nodes
        ),
    }
    manifest = json.loads(
        (source / "patches/sglang/manifest.json").read_text(encoding="utf-8")
    )
    relative_paths.update(
        f"patches/sglang/{row['file']}" for row in manifest["patches"]
    )
    for relative in sorted(relative_paths):
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    _git(checkout, "init")
    _git(checkout, "add", ".")
    _git(
        checkout,
        "-c",
        "user.name=Cold Start Fixture",
        "-c",
        "user.email=cold-start@example.invalid",
        "commit",
        "-m",
        "source-owned cold-start fixture",
    )
    return checkout


def _cached_workload_sources() -> tuple[Path, Path]:
    raw_root = os.environ.get("LIGHTCONE_CONTENT_SOURCE_CACHE")
    if raw_root is None:
        pytest.skip("set LIGHTCONE_CONTENT_SOURCE_CACHE for cold-start replay")
    root = Path(raw_root).resolve(strict=True)
    livecodebench = root / "livecodebench-code_generation_lite-0fe84c3/test6.jsonl"
    math500 = root / "math-500-6e4ed1a/test.jsonl"
    if not livecodebench.is_file() or not math500.is_file():
        pytest.skip("cold-start replay source cache is incomplete")
    return livecodebench.resolve(), math500.resolve()


def _publish_runtime_inputs(
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    inventory = _inventory()
    inventory_path = (tmp_path / "sources/inventory.json").resolve()
    inventory_path.parent.mkdir(parents=True)
    publish_canonical_json_no_replace(inventory_path, inventory.to_dict())

    cuda_home = (tmp_path / "runtime/cuda").resolve()
    bin_root = cuda_home / "bin"
    (cuda_home / "lib64").mkdir(parents=True)
    bin_root.mkdir()
    for name in ("nvidia-smi", "nvcc"):
        executable = bin_root / name
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    monkeypatch.setenv("PATH", f"{bin_root}{os.pathsep}{os.environ['PATH']}")

    patched_sglang = (tmp_path / "runtime/patched-sglang").resolve()
    patched_sglang.mkdir()
    devices = [
        {
            "uuid": device.uuid,
            "name": device.model,
            "compute_capability": (
                f"{device.compute_capability[0]}.{device.compute_capability[1]}"
            ),
            "driver_version": "580.95.05",
        }
        for device in inventory.devices
    ]
    doctor = {
        "schema_version": 2,
        "status": "PASS",
        "readiness": {
            "status": "PASS",
            "pass_count": 1,
            "fail_count": 0,
            "unknown_count": 0,
        },
        "checks": {"cold_start_fixture": {"status": "PASS"}},
        "roots": {"patched_sglang": str(patched_sglang)},
        "python": {"executable": sys.executable, "version": "3.12.13"},
        "gpu": {
            "torch": {"version": "2.7.1", "cuda_build": "12.8"},
            "parsed_inventory": {"devices": devices},
        },
        "commands": {"nvcc": "Cuda compilation tools, release 12.8, V12.8.93"},
        "packages": {"triton": "3.3.1"},
    }
    doctor_path = (tmp_path / "sources/doctor.json").resolve()
    publish_canonical_json_no_replace(doctor_path, doctor)
    return inventory_path, doctor_path, patched_sglang, cuda_home


def _model_specs(tmp_path: Path) -> tuple[TrustedModelSnapshotSpec, ...]:
    revision = {
        "target": "1" * 40,
        "dflash": "2" * 40,
        "dspark": "3" * 40,
    }
    roots: dict[str, Path] = {}
    for name in ("target", "dflash", "dspark"):
        root = (tmp_path / "models" / name / revision[name]).resolve()
        root.mkdir(parents=True)
        (root / "config.json").write_text(
            json.dumps({"model_type": name}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        roots[name] = root
    rows = (
        TrustedModelSnapshotSpec(
            model_id="z-lab/Qwen3-8B-DFlash-b16",
            revision=revision["dflash"],
            role="drafter",
            stages=("preflight",),
            local_snapshot_path=str(roots["dflash"]),
            runtime_bindings=(
                TrustedModelRuntimeBinding(
                    stage="preflight",
                    target_model_id="Qwen/Qwen3-8B",
                    backend="DFLASH",
                    draft_depth=15,
                ),
            ),
        ),
        TrustedModelSnapshotSpec(
            model_id="z-lab/Qwen3-8B-DSpark",
            revision=revision["dspark"],
            role="drafter",
            stages=("preflight",),
            local_snapshot_path=str(roots["dspark"]),
            runtime_bindings=(
                TrustedModelRuntimeBinding(
                    stage="preflight",
                    target_model_id="Qwen/Qwen3-8B",
                    backend="DSPARK",
                    draft_depth=15,
                ),
            ),
        ),
        TrustedModelSnapshotSpec(
            model_id="Qwen/Qwen3-8B",
            revision=revision["target"],
            role="target",
            stages=("preflight",),
            local_snapshot_path=str(roots["target"]),
        ),
        TrustedModelSnapshotSpec(
            model_id="Qwen/Qwen3-8B",
            revision=revision["target"],
            role="tokenizer",
            stages=("preflight",),
            local_snapshot_path=str(roots["target"]),
        ),
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.role,
                row.model_id,
                row.revision,
                row.local_snapshot_path,
            ),
        )
    )


def _install_code_owned_tokenizer_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invoke(*, input_path: Path, output_path: Path):
        source = json.loads(input_path.read_text(encoding="utf-8"))
        rows = []
        for request in source["requests"]:
            token_ids = list(hashlib.sha256(request["prompt"].encode("utf-8")).digest())
            rows.append(
                {
                    "request_id": request["request_id"],
                    "ordinal": request["ordinal"],
                    "prompt_sha256": request["prompt_sha256"],
                    "input_token_ids": token_ids,
                    "input_token_ids_sha256": content_module.content_sha256(token_ids),
                }
            )
        value = {
            **{
                name: source[name]
                for name in (
                    "protocol_sha256",
                    "schedule_source_sha256",
                    "tokenizer_model_id",
                    "tokenizer_revision",
                    "tokenizer_snapshot_path",
                    "tokenizer_content_authority_sha256",
                )
            },
            "schema_version": 1,
            "kind": "formal_serving_tokenization_output",
            "tokenizer_class": "CodeOwnedColdStartTokenizer",
            "tokenizer_vocab_size": 256,
            "transformers_version": "fixture-not-imported",
            "requests": rows,
        }
        publish_canonical_json_no_replace(output_path, value)
        worker = Path(dispatch_module.__file__).resolve()
        body = worker.read_bytes()
        argv_sha256 = content_module.content_sha256(
            {
                "fixture": "code_owned_cold_start_tokenizer",
                "input": str(input_path),
                "output": str(output_path),
            }
        )
        return (
            CanonicalJsonProofBinding.bind(output_path),
            hashlib.sha256(body).hexdigest(),
            len(body),
            argv_sha256,
        )

    monkeypatch.setattr(dispatch_module, "_invoke_tokenizer_worker", invoke)
    monkeypatch.setattr(preflight_module, "_invoke_tokenizer_worker", invoke)


def _publish_method_authorities(tmp_path: Path, capsys) -> tuple[Path, Path, Path]:
    tts_plan, _ = _plan_source(tmp_path / "tts-plan", scope="all")
    e1_plan, _ = _plan_source(tmp_path / "e1-plan", scope="last1")
    tts_pdf = (tmp_path / "sources/tts.pdf").resolve()
    tts_tex = (tmp_path / "sources/tts.tex").resolve()
    tts_pdf.write_bytes(b"%PDF-1.7\nsource-owned cold-start TTS\n")
    tts_tex.write_text("source-owned cold-start TTS\n", encoding="utf-8")
    window = (tmp_path / "sources/tts-window.json").resolve()
    loss = (tmp_path / "sources/tts-loss.json").resolve()
    publish_canonical_json_no_replace(window, _tts_window().to_dict())
    publish_canonical_json_no_replace(loss, TTS_DRAFTER_NATIVE_LOSS_SOURCE)
    tts = (tmp_path / "sources/tts-authority.json").resolve()
    assert (
        main(
            [
                "publish-tts-calibration-source-authority",
                "--paper-pdf",
                str(tts_pdf),
                "--paper-source",
                str(tts_tex),
                "--tuning-window",
                str(window),
                "--trainable-plan-authority",
                str(tts_plan),
                "--drafter-native-loss",
                str(loss),
                "--output",
                str(tts),
            ]
        )
        == 0
    )
    capsys.readouterr()

    chrono_pdf = (tmp_path / "sources/chronobelief.pdf").resolve()
    chrono_tex = (tmp_path / "sources/chronobelief.tex").resolve()
    chrono_pdf.write_bytes(b"%PDF-1.7\nsource-owned ChronoBelief\n")
    chrono_tex.write_text("equations 5.5--5.8\n", encoding="utf-8")
    chrono = (tmp_path / "sources/chronobelief-authority.json").resolve()
    assert (
        main(
            [
                "publish-chronobelief-source-authority",
                "--paper-pdf",
                str(chrono_pdf),
                "--tex-source",
                str(chrono_tex),
                "--output",
                str(chrono),
            ]
        )
        == 0
    )
    capsys.readouterr()

    e1 = (tmp_path / "sources/e1-anchor-authority.json").resolve()
    assert (
        main(
            [
                "publish-e1-recipe-anchor-authority",
                "--trainable-plan-authority",
                str(e1_plan),
                "--output",
                str(e1),
            ]
        )
        == 0
    )
    capsys.readouterr()
    return tts, chrono, e1


def _publish_onlinespec_source_authority(
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> tuple[Path, list[str]]:
    checkout = (tmp_path / "sources/onlinespec-checkout").resolve()
    checkout.mkdir(parents=True)
    audit = (tmp_path / "sources/onlinespec-audit.json").resolve()
    audit.write_text("{}\n", encoding="utf-8")
    verified = {
        "commit": ONLINE_SPEC_COMMIT,
        "tree": ONLINE_SPEC_TREE,
        "clean": True,
        "key_files": {"optimizer.py": "code-owned-fixture"},
    }
    monkeypatch.setattr(
        e0_stage_authority,
        "verify_onlinespec_source_checkout",
        lambda *_args, **_kwargs: verified,
    )
    output = (tmp_path / "sources/onlinespec-source-authority.json").resolve()
    argv = [
        "formal-single-operator",
        "publish-onlinespec-source-authority",
        "--checkout",
        str(checkout),
        "--audit",
        str(audit),
        "--output",
        str(output),
    ]
    assert main(argv) == 0
    assert len(json.loads(capsys.readouterr().out)["semantic_sha256"]) == 64
    authority = e0_onlinespec_source_authority_from_dict(
        CanonicalJsonProofBinding.bind(output).reopen()
    )
    assert authority.revalidate() == verified
    return output, argv


def test_public_onlinespec_source_authority_is_path_only_and_no_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    _output, argv = _publish_onlinespec_source_authority(
        tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
    )
    with pytest.raises(RuntimeError, match="target already exists"):
        main(argv)


@pytest.mark.integration
def test_v03_cold_start_reaches_exact_first_gpu_boundary_without_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Replay the public handoff from paths through an exact-ten PENDING queue."""

    livecodebench, math500 = _cached_workload_sources()
    checkout = _clean_runtime_fixture_checkout(tmp_path)
    inventory, doctor, _sglang, _cuda = _publish_runtime_inputs(
        tmp_path,
        monkeypatch=monkeypatch,
    )

    locked = {
        workload_id: bind_trusted_locked_workload(workload_id, source)
        for workload_id, source in (
            ("livecodebench_v6_hard", livecodebench),
            ("math500_level5", math500),
        )
    }
    authorities = {
        workload_id: workload_module.bind_formal_workload_authority(workload_id, source)
        for workload_id, source in (
            ("livecodebench_v6_hard", livecodebench),
            ("math500_level5", math500),
        )
    }
    monkeypatch.setattr(
        content_module,
        "bind_trusted_locked_workload",
        lambda workload_id, _path: locked[workload_id],
    )
    monkeypatch.setattr(
        workload_module,
        "bind_formal_workload_authority",
        lambda workload_id, _path: authorities[workload_id],
    )
    burst_root = tmp_path / "burstgpt"
    burst_root.mkdir()
    burst = _tiny_burstgpt_assets(burst_root, monkeypatch)
    _install_code_owned_tokenizer_fixture(monkeypatch)

    content_spec = TrustedSingleOperatorContentPathSpec(
        schema_version=1,
        kind="trusted_single_operator_content_path_spec",
        repository_root=str(checkout),
        model_specs=_model_specs(tmp_path),
        livecodebench_raw_path=str(livecodebench),
        math500_raw_path=str(math500),
        burstgpt_asset_paths=tuple(
            TrustedNamedInputPath(name=name, absolute_path=str(path))
            for name, path in sorted(burst.items())
        ),
        e0_task_native_specs=(),
        inventory_path=str(inventory),
        doctor_path=str(doctor),
    )
    spec_path = (tmp_path / "sources/content-path-spec.json").resolve()
    publish_canonical_json_no_replace(spec_path, content_spec.to_dict())
    content = (tmp_path / "sources/trusted-content.json").resolve()
    assert (
        main(
            [
                "formal-single-operator",
                "publish-trusted-content",
                "--spec",
                str(spec_path),
                "--output",
                str(content),
            ]
        )
        == 0
    )
    content_result = json.loads(capsys.readouterr().out)
    assert content_result["runtime_binding_status"] == "BOUND"

    workload = (tmp_path / "sources/preflight-workload.json").resolve()
    assert (
        main(
            [
                "formal-single-operator",
                "publish-preflight-workload",
                "--content-source",
                str(content),
                "--output",
                str(workload),
            ]
        )
        == 0
    )
    capsys.readouterr()

    runtime = (tmp_path / "sources/runtime-authority.json").resolve()
    assert (
        main(
            [
                "publish-formal-runtime-authority-manifest",
                "--repository-root",
                str(checkout),
                "--output",
                str(runtime),
            ]
        )
        == 0
    )
    capsys.readouterr()
    tts, chrono, e1 = _publish_method_authorities(tmp_path, capsys)
    lock = (tmp_path / "sources/protocol-lock.json").resolve()
    assert (
        main(
            [
                "formal-single-operator",
                "build-trusted-protocol-lock",
                "--protocol-id",
                "lightcone-v03-cold-start-fixture",
                "--trusted-content-bundle",
                str(content),
                "--runtime-authority-manifest",
                str(runtime),
                "--tts-calibration-authority",
                str(tts),
                "--chronobelief-authority",
                str(chrono),
                "--e1-recipe-anchor-authority",
                str(e1),
                "--output",
                str(lock),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["schema_version"] == 5
    onlinespec_authority, _ = _publish_onlinespec_source_authority(
        tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
    )

    run_root = (tmp_path / "formal-v03-cold-start").resolve()
    catalog = (tmp_path / "prerequisite-catalog").resolve()
    run_root.mkdir()
    catalog.mkdir()
    driver_config = (run_root / "driver-config.json").resolve()
    driver_argv = [
        "formal-single-operator",
        "write-dag-driver-config",
        "--repository-root",
        str(checkout),
        "--run-root",
        str(run_root),
        "--protocol-lock",
        str(lock),
        "--content-source",
        str(content),
        "--runtime-authority-manifest",
        str(runtime),
        "--inventory",
        str(inventory),
        "--doctor-report",
        str(doctor),
        "--preflight-workload-authority",
        str(workload),
        "--prerequisite-catalog",
        str(catalog),
        "--output",
        str(driver_config),
    ]
    assert main(driver_argv) == 0
    capsys.readouterr()
    with pytest.raises(FileExistsError):
        main(driver_argv)
    capsys.readouterr()

    bootstrap_config = (run_root / "bootstrap-config.json").resolve()
    bootstrap_argv = [
        "formal-single-operator",
        "write-bootstrap-config",
        "--driver-config",
        str(driver_config),
        "--onlinespec-source-authority",
        str(onlinespec_authority),
        "--output",
        str(bootstrap_config),
    ]
    assert main(bootstrap_argv) == 0
    capsys.readouterr()
    with pytest.raises(FileExistsError):
        main(bootstrap_argv)
    capsys.readouterr()

    status = main(["formal-single-operator", "status"])
    assert status == 0
    readiness = json.loads(capsys.readouterr().out)
    assert tuple(row["node"] for row in readiness["nodes"]) == (
        FORMAL_SINGLE_OPERATOR_NODE_ORDER
    )
    assert all(row["code_capability_ready"] for row in readiness["nodes"])

    once = [
        "formal-single-operator",
        "bootstrap-once",
        "--config",
        str(bootstrap_config),
    ]
    assert main(once) == 0
    first = json.loads(capsys.readouterr().out)
    assert (first["controller_node"], first["controller_action"]) == (
        "preflight",
        "MATERIALIZED",
    )
    assert main(once) == 0
    second = json.loads(capsys.readouterr().out)
    assert (second["controller_node"], second["controller_action"]) == (
        "preflight",
        "PLANNED",
    )

    database = run_root / "operator.sqlite3"
    with ExperimentOperatorStore(database) as store:
        expected_plan = default_formal_stage_plan()
        assert tuple(row["node"] for row in store.controller_nodes()) == tuple(
            row.node for row in expected_plan
        )
        assert store.controller_node("preflight")["state"] == "PLANNED"
        assert all(
            store.controller_node(node)["state"] == "UNMATERIALIZED"
            for node in FORMAL_SINGLE_OPERATOR_NODE_ORDER[1:]
        )
        attempts = store.latest_stage_attempts("preflight")
        assert len(attempts) == 10
        assert {row["status"] for row in attempts} == {"PENDING"}
        assert all(row["pid"] is None and row["pgid"] is None for row in attempts)
        assert len(store.physical_attempt_groups()) == 1
        group = store.physical_attempt_groups()[0]
        assert group["status"] == "PENDING"
        assert len(group["members"]) == 10
        assert len(store.physical_attempt_group_commands(group["group_id"])) == 10
        assert all(
            store.latest_stage_attempts(node) == ()
            for node in FORMAL_SINGLE_OPERATOR_NODE_ORDER[1:]
        )

    progress = run_root / "results/progress"
    expected_exports = {
        "stage_plan.csv",
        "cell_ledger.csv",
        "metrics_long.parquet",
        "stage_summary.csv",
        "selection_decisions.jsonl",
        "watchdog_events.jsonl",
        "dashboard.md",
        "instance_billing.csv",
        "controller_state.csv",
        "export_manifest.json",
    }
    assert {path.name for path in progress.iterdir()} == expected_exports
    with (progress / "stage_plan.csv").open(newline="", encoding="utf-8") as handle:
        stage_rows = list(csv.DictReader(handle))
    assert len(stage_rows) == 21
    assert [row["expected_formula"] for row in stage_rows] == [
        entry.expected_formula for entry in default_formal_stage_plan()
    ]
    with (progress / "cell_ledger.csv").open(newline="", encoding="utf-8") as handle:
        ledger_rows = list(csv.DictReader(handle))
    assert len(ledger_rows) == 10
    assert {row["status"] for row in ledger_rows} == {"PENDING"}

    # Restart opens and deep-validates every binding without replacing the
    # materialization, adding attempts, or crossing into scheduler launch.
    supervisor = FormalSingleOperatorBootstrapSupervisor(bootstrap_config)
    try:
        assert supervisor.config.onlinespec_source_authority is not None
        assert supervisor.config.onlinespec_source_authority.absolute_path == str(
            onlinespec_authority
        )
        assert supervisor.driver.store.controller_node("preflight")["state"] == (
            "PLANNED"
        )
        assert len(supervisor.driver.store.latest_stage_attempts("preflight")) == 10
    finally:
        supervisor.close()

    # This is the exact handoff boundary: another bootstrap cycle observes a
    # PLANNED node, commits the two-GPU group RUNNING, and invokes its setsid
    # child.  The cold-start test deliberately does not execute that cycle.
    assert all(not row["pid"] for row in ledger_rows)
