from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import pytest
from test_formal_preflight_inputs import _inventory
from test_formal_single_operator_content import _tiny_burstgpt_assets
from test_trainable_plan_authority import _write_safetensors

import lightcone_spec.experiments.formal_preflight_inputs as preflight_module
import lightcone_spec.experiments.formal_single_operator_content as content_module
import lightcone_spec.experiments.workload_authority as workload_module
import lightcone_spec.orchestration.formal_physical_dispatch as dispatch_module
from lightcone_spec.cli.main import main
from lightcone_spec.experiments import e0_stage_authority
from lightcone_spec.experiments.formal_registry import (
    e0_onlinespec_source_authority_from_dict,
)
from lightcone_spec.experiments.formal_runtime_manifest import (
    FORMAL_RUNTIME_SOURCE_LAYOUT,
)
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedSingleOperatorContentBundleBinding,
    bind_trusted_locked_workload,
)
from lightcone_spec.experiments.formal_single_operator_e0_workloads import (
    E0_TASK_NATIVE_SOURCE_PINS,
)
from lightcone_spec.experiments.formal_single_operator_model_registry import (
    FORMAL_V03_E0_SOURCE_AUTHORITY_INDEX_FILE_NAME,
    FORMAL_V03_MODEL_SNAPSHOT_REGISTRY,
    FormalV03NamedDirectoryPath,
    FormalV03NamedFilePath,
    load_formal_v03_e0_source_authority_index,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    FORMAL_SINGLE_OPERATOR_NODE_ORDER,
)
from lightcone_spec.experiments.formal_stage_execution import (
    load_e1_recipe_anchor_authority_artifact,
)
from lightcone_spec.experiments.onlinespec import (
    ONLINE_SPEC_COMMIT,
    ONLINE_SPEC_TREE,
)
from lightcone_spec.orchestration.experiment_operator import (
    ExperimentOperatorStore,
    default_formal_stage_plan,
)
from lightcone_spec.orchestration.experiment_operator_production import (
    ProductionSchedulerRuntime,
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
        "manifests/runtime/industrial_compatibility_v1.json",
        "manifests/runtime/industrial_compatibility_v1.json.sha256",
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


def _cold_start_inventory():
    inventory = _inventory()
    return replace(
        inventory,
        devices=tuple(
            replace(
                device,
                model="NVIDIA RTX PRO 6000 Blackwell Server Edition",
                memory_bytes=96_000 * 1024**2,
            )
            for device in inventory.devices
        ),
    )


def _publish_runtime_inputs(
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> tuple[Path, Path, Path, Path]:
    inventory = _cold_start_inventory()
    inventory_path = (tmp_path / "sources/inventory.json").resolve()
    inventory_path.parent.mkdir(parents=True)
    receipt_path = (tmp_path / "sources/inventory-receipt.json").resolve()
    inventory_receipt = {
        "schema_version": 1,
        "kind": "code_owned_cold_start_gpu_inventory_receipt",
        "receipt_sha256": inventory.source_receipt_sha256,
    }
    cli_module = import_module("lightcone_spec.cli.main")
    monkeypatch.setattr(
        cli_module,
        "collect_gpu_inventory",
        lambda **_kwargs: (inventory, inventory_receipt),
    )
    assert (
        main(
            [
                "collect-gpu-inventory",
                "--challenge-nonce-sha256",
                "a" * 64,
                "--receipt-output",
                str(receipt_path),
                "--output",
                str(inventory_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert CanonicalJsonProofBinding.bind(inventory_path).reopen() == (
        inventory.to_dict()
    )

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
    doctor_path = (tmp_path / "sources/doctor.json").resolve()
    return inventory_path, doctor_path, patched_sglang, cuda_home


def _publish_trusted_capacity_doctor(
    *,
    checkout: Path,
    patched_sglang: Path,
    content_spec_path: Path,
    run_root: Path,
    doctor_path: Path,
    inventory_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> Path:
    """Publish capacity, then a source-produced trusted PASS doctor report."""

    from test_doctor_industrial import _passing_facts

    capacity = (doctor_path.parent / "stage-capacity.json").resolve()
    assert (
        main(
            [
                "formal-single-operator",
                "publish-stage-capacity",
                "--content-path-spec",
                str(content_spec_path),
                "--run-root",
                str(run_root),
                "--output",
                str(capacity),
            ]
        )
        == 0
    )
    capacity_result = json.loads(capsys.readouterr().out)
    assert capacity_result["status"] == "AVAILABLE"
    assert capacity_result["formal_measured_authorization"] is False

    inventory = _cold_start_inventory()
    assert json.loads(inventory_path.read_text(encoding="utf-8")) == (
        inventory.to_dict()
    )
    facts = _passing_facts(checkout, patched_sglang)
    facts["python"]["executable"] = sys.executable
    facts["gpu"]["inventory"]["devices"] = [
        {
            "uuid": device.uuid,
            "name": device.model,
            "memory_total_mib": device.memory_bytes // (1024**2),
            "driver_version": "580.95.05",
            "compute_capability": (
                f"{device.compute_capability[0]}.{device.compute_capability[1]}"
            ),
            "pci_bus_id": device.pci_bus_id,
        }
        for device in inventory.devices
    ]
    monkeypatch.setattr("lightcone_spec.doctor._collect_facts", lambda *_args: facts)
    assert (
        main(
            [
                "doctor",
                "--project-root",
                str(checkout),
                "--sglang-root",
                str(patched_sglang),
                "--trusted-single-operator-capacity",
                str(capacity),
                "--output",
                str(doctor_path),
            ]
        )
        == 0
    )
    doctor = json.loads(capsys.readouterr().out)
    assert doctor["schema_version"] == 2
    assert doctor["status"] == "PASS"
    assert doctor["stage_capacity"]["status"] == "AVAILABLE"
    assert doctor["stage_capacity"]["formal_measured_authorization"] is False
    assert CanonicalJsonProofBinding.bind(doctor_path).reopen() == doctor
    return capacity


def _model_snapshot_paths(
    tmp_path: Path,
) -> tuple[FormalV03NamedDirectoryPath, ...]:
    rows = []
    for snapshot in FORMAL_V03_MODEL_SNAPSHOT_REGISTRY:
        repository = "models--" + snapshot.snapshot_model_id.replace("/", "--")
        cache_root = (tmp_path / "models" / repository).resolve()
        root = (cache_root / "snapshots" / snapshot.revision).resolve()
        root.mkdir(parents=True)
        (root / "config.json").write_text(
            json.dumps({"model_type": snapshot.snapshot_key}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if snapshot.snapshot_key == "qwen3_8b_target":
            for name, body in {
                "generation_config.json": b'{"do_sample":false}',
                "merges.txt": b"#version: 0.2\na b\n",
                "tokenizer.json": b'{"version":"1.0"}',
                "tokenizer_config.json": b'{"model_max_length":40960}',
                "vocab.json": b'{"a":0,"b":1}',
            }.items():
                (root / name).write_bytes(body)
            target_shard = "model-00001-of-00001.safetensors"
            _write_safetensors(
                root / target_shard,
                {"model.embed_tokens.weight": ("BF16", (4, 2))},
            )
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 16},
                        "weight_map": {
                            "model.embed_tokens.weight": target_shard,
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
        elif snapshot.snapshot_key == "qwen3_8b_dflash_core":
            for name, body in {
                "dflash.py": b"class DFlash: pass\n",
                "modeling_dflash.py": b"class DFlashModel: pass\n",
                "utils.py": b"BLOCK_SIZE = 16\n",
            }.items():
                (root / name).write_bytes(body)
            _write_safetensors(
                root / "model.safetensors",
                {
                    "layers.0.input_layernorm.weight": ("F32", (4,)),
                    "layers.0.self_attn.q_proj.weight": ("BF16", (8, 4)),
                    "lm_head.weight": ("BF16", (32, 4)),
                    "target_model.layers.0.weight": ("BF16", (4, 4)),
                },
            )
        rows.append(
            FormalV03NamedDirectoryPath(
                name=snapshot.snapshot_key,
                absolute_path=str(root),
            )
        )
    return tuple(sorted(rows))


def _publish_v03_e0_source_authorities(
    tmp_path: Path,
    *,
    capsys,
) -> tuple[FormalV03NamedFilePath, ...]:
    cache = Path(os.environ["LIGHTCONE_CONTENT_SOURCE_CACHE"]).resolve(strict=True)
    raw_root = (cache / "e0-task-native").resolve(strict=True)
    inputs_path = (tmp_path / "sources/e0-raw-source-paths.json").resolve()
    writer_argv = [
        "formal-single-operator",
        "write-v03-e0-raw-source-path-inputs",
        "--output",
        str(inputs_path),
    ]
    for task, pin in sorted(E0_TASK_NATIVE_SOURCE_PINS.items()):
        writer_argv.extend(
            (
                "--source",
                f"{task}={(raw_root / pin.source_file_name).resolve(strict=True)}",
            )
        )
    assert main(writer_argv) == 0
    writer_result = json.loads(capsys.readouterr().out)
    assert writer_result["absolute_path"] == str(inputs_path)
    CanonicalJsonProofBinding.bind(inputs_path).reopen()
    output = (tmp_path / "sources/e0-authorities").resolve()
    output.mkdir()
    assert (
        main(
            [
                "formal-single-operator",
                "publish-v03-e0-source-authorities",
                "--inputs",
                str(inputs_path),
                "--output-directory",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    index_path = (output / FORMAL_V03_E0_SOURCE_AUTHORITY_INDEX_FILE_NAME).resolve(
        strict=True
    )
    assert result["index_path"] == str(index_path)
    index = load_formal_v03_e0_source_authority_index(index_path)
    assert tuple(row.name for row in index.authority_paths) == tuple(
        sorted(E0_TASK_NATIVE_SOURCE_PINS)
    )
    return index.authority_paths


def _publish_v03_content_path_spec(
    tmp_path: Path,
    *,
    repository_root: Path,
    model_snapshot_paths: tuple[FormalV03NamedDirectoryPath, ...],
    livecodebench: Path,
    math500: Path,
    burst: dict[str, Path],
    e0_source_authority_paths: tuple[FormalV03NamedFilePath, ...],
    inventory: Path,
    doctor: Path,
    capsys,
) -> Path:
    inputs_path = (tmp_path / "sources/content-path-inputs.json").resolve()
    writer_argv = [
        "formal-single-operator",
        "write-v03-content-path-inputs",
        "--repository-root",
        str(repository_root),
        "--livecodebench-raw",
        str(livecodebench),
        "--math500-raw",
        str(math500),
        "--inventory",
        str(inventory),
        "--doctor-output",
        str(doctor),
        "--output",
        str(inputs_path),
    ]
    for row in model_snapshot_paths:
        writer_argv.extend(("--model-snapshot", f"{row.name}={row.absolute_path}"))
    for name, path in sorted(burst.items()):
        writer_argv.extend(("--burstgpt-asset", f"{name}={path}"))
    for row in e0_source_authority_paths:
        writer_argv.extend(("--e0-source-authority", f"{row.name}={row.absolute_path}"))
    assert main(writer_argv) == 0
    writer_result = json.loads(capsys.readouterr().out)
    assert writer_result["absolute_path"] == str(inputs_path)
    CanonicalJsonProofBinding.bind(inputs_path).reopen()
    output = (tmp_path / "sources/content-path-spec.json").resolve()
    argv = [
        "formal-single-operator",
        "publish-v03-content-path-spec",
        "--inputs",
        str(inputs_path),
        "--output",
        str(output),
    ]
    assert main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["path"] == str(output)
    with pytest.raises(FileExistsError):
        main(argv)
    capsys.readouterr()
    return output


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


def _publish_method_authorities(
    tmp_path: Path,
    *,
    trusted_content: Path,
    capsys,
) -> tuple[Path, Path, Path]:
    cache = Path(os.environ["LIGHTCONE_CONTENT_SOURCE_CACHE"]).resolve(strict=True)
    primary = cache / "method-primary-sources"
    tts_pdf = (primary / "tts-2605.09329v2.pdf").resolve(strict=True)
    tts_source = (primary / "tts-2605.09329v2-source.tar.gz").resolve(strict=True)
    chrono_pdf = (primary / "chronobelief-project-preregistration-v1.pdf").resolve(
        strict=True
    )
    chrono_tex = (primary / "chronobelief-project-preregistration-v1.tex").resolve(
        strict=True
    )

    tts_plan = (tmp_path / "sources/tts-trainable-plan.json").resolve()
    tts_plan_argv = [
        "formal-single-operator",
        "publish-tts-cal-trainable-plan",
        "--trusted-content-bundle",
        str(trusted_content),
        "--output",
        str(tts_plan),
    ]
    assert main(tts_plan_argv) == 0
    tts_plan_result = json.loads(capsys.readouterr().out)
    assert len(tts_plan_result["trainable_plan_sha256"]) == 64
    with pytest.raises(FileExistsError, match="already exists"):
        main(tts_plan_argv)
    capsys.readouterr()

    e1_plan = (tmp_path / "sources/e1-trainable-plan.json").resolve()
    assert (
        main(
            [
                "formal-single-operator",
                "publish-e1-anchor-trainable-plan",
                "--trusted-content-bundle",
                str(trusted_content),
                "--output",
                str(e1_plan),
            ]
        )
        == 0
    )
    e1_plan_result = json.loads(capsys.readouterr().out)
    assert len(e1_plan_result["trainable_plan_sha256"]) == 64

    window = (tmp_path / "sources/tts-window.json").resolve()
    assert (
        main(
            [
                "publish-tts-calibration-tuning-window",
                "--trusted-content-bundle",
                str(trusted_content),
                "--output",
                str(window),
            ]
        )
        == 0
    )
    capsys.readouterr()
    loss = (tmp_path / "sources/tts-loss.json").resolve()
    assert (
        main(
            [
                "publish-tts-drafter-native-loss-source",
                "--output",
                str(loss),
            ]
        )
        == 0
    )
    capsys.readouterr()
    tts = (tmp_path / "sources/tts-authority.json").resolve()
    assert (
        main(
            [
                "publish-tts-calibration-source-authority",
                "--paper-pdf",
                str(tts_pdf),
                "--paper-source",
                str(tts_source),
                "--trusted-content-bundle",
                str(trusted_content),
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
                "--trusted-content-bundle",
                str(trusted_content),
                "--output",
                str(e1),
            ]
        )
        == 0
    )
    capsys.readouterr()
    e1_artifact = load_e1_recipe_anchor_authority_artifact(e1)
    assert e1_artifact.schema_version == 3
    assert e1_artifact.trusted_content_bundle_source == (
        TrustedSingleOperatorContentBundleBinding.bind(trusted_content)
    )
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

    gpu_spawn_calls: list[tuple[str, tuple[str, ...]]] = []

    def reject_gpu_spawn(self, command, gpu_uuids):
        gpu_spawn_calls.append((command.cell_id, gpu_uuids))
        pytest.fail("cold-start boundary crossed into a GPU process spawn")

    monkeypatch.setattr(ProductionSchedulerRuntime, "launch", reject_gpu_spawn)
    livecodebench, math500 = _cached_workload_sources()
    checkout = _clean_runtime_fixture_checkout(tmp_path)
    inventory, doctor, patched_sglang, _cuda = _publish_runtime_inputs(
        tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
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
    e0_authorities = _publish_v03_e0_source_authorities(tmp_path, capsys=capsys)
    spec_path = _publish_v03_content_path_spec(
        tmp_path,
        repository_root=checkout,
        model_snapshot_paths=_model_snapshot_paths(tmp_path),
        livecodebench=livecodebench,
        math500=math500,
        burst=burst,
        e0_source_authority_paths=e0_authorities,
        inventory=inventory,
        doctor=doctor,
        capsys=capsys,
    )
    run_root = (tmp_path / "formal-v03-cold-start").resolve()
    run_root.mkdir()
    capacity = _publish_trusted_capacity_doctor(
        checkout=checkout,
        patched_sglang=patched_sglang,
        content_spec_path=spec_path,
        run_root=run_root,
        doctor_path=doctor,
        inventory_path=inventory,
        monkeypatch=monkeypatch,
        capsys=capsys,
    )
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
    tts, chrono, e1 = _publish_method_authorities(
        tmp_path,
        trusted_content=content,
        capsys=capsys,
    )
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

    catalog = (tmp_path / "prerequisite-catalog").resolve()
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
        assert supervisor.driver_config.doctor_report.absolute_path == str(doctor)
        doctor_value = CanonicalJsonProofBinding.bind(doctor).reopen()
        assert doctor_value["stage_capacity"]["authority"]["absolute_path"] == str(
            capacity
        )
        assert supervisor.driver.store.controller_node("preflight")["state"] == (
            "PLANNED"
        )
        assert len(supervisor.driver.store.latest_stage_attempts("preflight")) == 10
    finally:
        supervisor.close()

    # A third cycle would commit the physical group RUNNING and invoke its
    # setsid child.  This test proves the exact no-spawn handoff and stops.
    assert gpu_spawn_calls == []
    assert all(not row["pid"] for row in ledger_rows)

    content.write_bytes(content.read_bytes() + b" ")
    with pytest.raises((RuntimeError, ValueError), match="canonical|changed"):
        load_e1_recipe_anchor_authority_artifact(e1)
