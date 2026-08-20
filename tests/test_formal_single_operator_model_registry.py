from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_formal_preflight_inputs import _inventory

import lightcone_spec.experiments.formal_single_operator_model_registry as registry
from lightcone_spec.cli.main import main
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedRuntimeObservations,
    TrustedSingleOperatorContentBundle,
    bind_trusted_json_artifact,
)
from lightcone_spec.experiments.formal_single_operator_e0_workloads import (
    E0_TASK_NATIVE_SOURCE_PINS,
)
from lightcone_spec.experiments.formal_single_operator_loads import (
    BURSTGPT_V2_ASSETS,
)
from lightcone_spec.experiments.formal_single_operator_model_registry import (
    FORMAL_V03_E0_SOURCE_AUTHORITY_INDEX_FILE_NAME,
    FORMAL_V03_MODEL_SNAPSHOT_REGISTRY,
    FormalV03ContentPathInputs,
    FormalV03NamedDirectoryPath,
    FormalV03NamedFilePath,
    build_formal_v03_content_path_spec,
    build_formal_v03_model_lock,
    load_formal_v03_content_path_inputs,
    load_formal_v03_e0_raw_source_path_inputs,
    load_formal_v03_e0_source_authority_index,
    publish_formal_v03_content_path_spec_from_inputs,
    publish_formal_v03_e0_source_authorities_from_inputs,
    publish_formal_v03_model_lock,
    require_formal_v03_content_path_spec,
    require_formal_v03_pass_runtime_doctor,
)
from lightcone_spec.experiments.registry import E0_BACKENDS, E0_MODELS
from lightcone_spec.locking.models import ModelLock, prepare_models
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_EXPECTED_SNAPSHOTS = {
    "gemma4_12b_dflash_e0": (
        "deepseek-ai/dflash_gemma4_12b_block7",
        "7490ce60c7630107917fe558e2bbe3dcec6195cb",
    ),
    "gemma4_12b_dspark_e0": (
        "deepseek-ai/dspark_gemma4_12b_block7",
        "2fa72e765eec2965fc4d86a8663ce6769eba6218",
    ),
    "gemma4_12b_eagle3_e0": (
        "deepseek-ai/eagle3_gemma4_12b_ttt7",
        "0bc24c312350910419cf371e54082f040d65cc82",
    ),
    "gemma4_12b_target": (
        "google/gemma-4-12B-it",
        "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    ),
    "qwen35_122b_a10b_fp8_nextn": (
        "Qwen/Qwen3.5-122B-A10B-FP8",
        "a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9",
    ),
    "qwen36_35b_a3b_nextn": (
        "Qwen/Qwen3.6-35B-A3B",
        "995ad96eacd98c81ed38be0c5b274b04031597b0",
    ),
    "qwen3_14b_dflash_e0": (
        "deepseek-ai/dflash_qwen3_14b_block7",
        "ab0a8b28236654620bb41d64b336d00a14cb467f",
    ),
    "qwen3_14b_dspark_e0": (
        "deepseek-ai/dspark_qwen3_14b_block7",
        "83207b416acf99f41c2184648923632fccea6dd0",
    ),
    "qwen3_14b_eagle3_e0": (
        "deepseek-ai/eagle3_qwen3_14b_ttt7",
        "d7ea05d0b0009badfff0df2dcaedf82cce0f74f8",
    ),
    "qwen3_14b_target": (
        "Qwen/Qwen3-14B",
        "40c069824f4251a91eefaf281ebe4c544efd3e18",
    ),
    "qwen3_4b_dflash_e0": (
        "deepseek-ai/dflash_qwen3_4b_block7",
        "02d530b7962ea1412beaf41a05c0b8e36d5f9b1d",
    ),
    "qwen3_4b_dspark_e0": (
        "deepseek-ai/dspark_qwen3_4b_block7",
        "3457dff1417cb84927f6098a5fcb7cee85c934b7",
    ),
    "qwen3_4b_eagle3_e0": (
        "deepseek-ai/eagle3_qwen3_4b_ttt7",
        "b0b90fd15d052217c226be5e46d468d8d129e0cd",
    ),
    "qwen3_4b_target": (
        "Qwen/Qwen3-4B",
        "1cfa9a7208912126459214e8b04321603b3df60c",
    ),
    "qwen3_8b_dflash_core": (
        "z-lab/Qwen3-8B-DFlash-b16",
        "9b41424b7109f9c5413454f481b09a82b85333f4",
    ),
    "qwen3_8b_dflash_e0": (
        "deepseek-ai/dflash_qwen3_8b_block7",
        "9e44dbbb6cb68b0c943abf9c5fc3c17c00897cdf",
    ),
    "qwen3_8b_dspark_e0_core": (
        "deepseek-ai/dspark_qwen3_8b_block7",
        "03326e5043815da1f81b109078b2889737c26017",
    ),
    "qwen3_8b_eagle3_e0": (
        "deepseek-ai/eagle3_qwen3_8b_ttt7",
        "f6485ba8d21e11942958617dbe7e71b467f38f38",
    ),
    "qwen3_8b_target": (
        "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218",
    ),
}


def _model_paths(tmp_path: Path) -> tuple[FormalV03NamedDirectoryPath, ...]:
    rows = []
    for snapshot in FORMAL_V03_MODEL_SNAPSHOT_REGISTRY:
        repository = "models--" + snapshot.snapshot_model_id.replace("/", "--")
        path = (
            tmp_path / "hf" / repository / "snapshots" / snapshot.revision
        ).resolve()
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}\n", encoding="utf-8")
        rows.append(
            FormalV03NamedDirectoryPath(
                name=snapshot.snapshot_key,
                absolute_path=str(path),
            )
        )
    return tuple(sorted(rows))


def _content_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> FormalV03ContentPathInputs:
    repository = (tmp_path / "repository").resolve()
    repository.mkdir()
    livecodebench = (tmp_path / "livecodebench.jsonl").resolve()
    math500 = (tmp_path / "math500.jsonl").resolve()
    inventory = (tmp_path / "inventory.json").resolve()
    for path in (livecodebench, math500, inventory):
        path.write_text("{}\n", encoding="utf-8")
    burst_root = (tmp_path / "burst").resolve()
    burst_root.mkdir()
    burst_paths = []
    for asset in BURSTGPT_V2_ASSETS:
        path = (burst_root / asset.name).resolve()
        path.write_text("fixture\n", encoding="utf-8")
        burst_paths.append(
            FormalV03NamedFilePath(name=asset.name, absolute_path=str(path))
        )
    authority_root = (tmp_path / "e0-authorities").resolve()
    authority_root.mkdir()
    authority_tasks: dict[str, str] = {}
    authority_paths = []
    for task in sorted(E0_TASK_NATIVE_SOURCE_PINS):
        path = (authority_root / f"{task.lower().replace('-', '_')}.json").resolve()
        path.write_text("{}\n", encoding="utf-8")
        authority_tasks[str(path)] = task
        authority_paths.append(
            FormalV03NamedFilePath(name=task, absolute_path=str(path))
        )

    def load_authority(path: str):
        task = authority_tasks[path]
        pin = E0_TASK_NATIVE_SOURCE_PINS[task]
        return SimpleNamespace(
            task=task,
            repository=pin.repository,
            repository_revision=pin.repository_revision,
            support_status="UNSUPPORTED" if task == "MT-Bench" else "READY",
        )

    monkeypatch.setattr(
        registry, "load_e0_task_native_source_authority", load_authority
    )
    return FormalV03ContentPathInputs(
        schema_version=1,
        kind="formal_v03_content_path_inputs",
        repository_root=str(repository),
        model_snapshot_paths=_model_paths(tmp_path),
        livecodebench_raw_path=str(livecodebench),
        math500_raw_path=str(math500),
        burstgpt_asset_paths=tuple(burst_paths),
        e0_source_authority_paths=tuple(authority_paths),
        inventory_path=str(inventory),
        doctor_path=str((tmp_path / "future-doctor.json").resolve()),
    )


def test_registry_freezes_exact_snapshots_and_runtime_product() -> None:
    observed = {
        row.snapshot_key: (row.snapshot_model_id, row.revision)
        for row in FORMAL_V03_MODEL_SNAPSHOT_REGISTRY
    }
    assert observed == _EXPECTED_SNAPSHOTS
    assert len(FORMAL_V03_MODEL_SNAPSHOT_REGISTRY) == 19
    assert sum(len(row.members) for row in FORMAL_V03_MODEL_SNAPSHOT_REGISTRY) == 25

    bindings = tuple(
        binding
        for snapshot in FORMAL_V03_MODEL_SNAPSHOT_REGISTRY
        for member in snapshot.members
        for binding in member.runtime_bindings
    )
    assert {
        (row.backend, row.target_model_id, row.draft_depth)
        for row in bindings
        if row.stage == "preflight"
    } == {
        ("DFLASH", "Qwen/Qwen3-8B", 15),
        ("DSPARK", "Qwen/Qwen3-8B", 15),
    }
    core_dspark = next(
        row
        for row in FORMAL_V03_MODEL_SNAPSHOT_REGISTRY
        if row.snapshot_key == "qwen3_8b_dspark_e0_core"
    ).members[0]
    assert core_dspark.stages == ("preflight", "E1a", "E5", "E0")
    assert {
        (row.target_model_id, row.backend, row.draft_depth)
        for row in bindings
        if row.stage == "E6"
    } == {
        ("Qwen/Qwen3.5-122B-A10B-FP8", "NEXTN", 1),
        ("Qwen/Qwen3.6-35B-A3B", "NEXTN", 1),
    }
    assert {
        (row.target_model_id, row.backend, row.draft_depth)
        for row in bindings
        if row.stage == "E0"
    } == {(model, backend, 7) for model in E0_MODELS for backend in E0_BACKENDS}

    gemma = next(
        row
        for row in FORMAL_V03_MODEL_SNAPSHOT_REGISTRY
        if row.snapshot_key == "gemma4_12b_target"
    )
    assert gemma.snapshot_model_id == "google/gemma-4-12B-it"
    assert {(row.model_id, row.role) for row in gemma.members} == {
        ("Gemma4-12B", "target"),
        ("Gemma4-12B", "tokenizer"),
    }


def test_formal_v03_model_lock_is_exact_nineteen_and_canonically_sorted() -> None:
    lock = build_formal_v03_model_lock()
    expected = tuple(sorted(_EXPECTED_SNAPSHOTS.values()))

    assert type(lock) is ModelLock
    assert lock.schema_version == 2
    assert len(lock.models) == 19
    assert len({row.model_id for row in lock.models}) == 19
    assert tuple((row.model_id, row.revision) for row in lock.models) == expected
    assert tuple(row.model_id for row in lock.models) == tuple(
        sorted(row.model_id for row in lock.models)
    )
    lock.validate()


def test_formal_v03_model_lock_publishes_for_existing_offline_consumers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (tmp_path / "formal-v03-model-lock.json").resolve()
    published = publish_formal_v03_model_lock(output_path=output)

    assert ModelLock.load(output) == published == build_formal_v03_model_lock()
    assert Path(f"{output}.sha256").read_text(encoding="ascii") == (
        f"{published.sha256}\n"
    )

    calls: list[tuple[str, str, bool]] = []

    def snapshot_download(
        *,
        repo_id: str,
        revision: str,
        cache_dir: str | Path,
        token: str | None,
        local_files_only: bool,
    ) -> str:
        assert token is None
        calls.append((repo_id, revision, local_files_only))
        root = (Path(cache_dir) / repo_id.replace("/", "--") / revision).resolve()
        root.mkdir(parents=True)
        return str(root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    roots = prepare_models(
        ModelLock.load(output),
        (tmp_path / "prepared-cache").resolve(),
        local_files_only=True,
    )
    assert tuple(roots) == tuple(row.model_id for row in published.models)
    assert calls == [(row.model_id, row.revision, True) for row in published.models]


def test_formal_v03_model_lock_publication_never_replaces_or_repairs_partial_pair(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "formal-v03-model-lock.json").resolve()
    sidecar = Path(f"{output}.sha256")
    sidecar.write_text("partial-sidecar\n", encoding="ascii")
    with pytest.raises(FileExistsError, match="output or sidecar already exists"):
        publish_formal_v03_model_lock(output_path=output)
    assert not output.exists()
    assert sidecar.read_text(encoding="ascii") == "partial-sidecar\n"

    sidecar.unlink()
    output.write_text("partial-lock", encoding="ascii")
    with pytest.raises(FileExistsError, match="output or sidecar already exists"):
        publish_formal_v03_model_lock(output_path=output)
    assert output.read_text(encoding="ascii") == "partial-lock"
    assert not sidecar.exists()

    output.unlink()
    publish_formal_v03_model_lock(output_path=output)
    before = (output.read_bytes(), sidecar.read_bytes())
    with pytest.raises(FileExistsError, match="output or sidecar already exists"):
        publish_formal_v03_model_lock(output_path=output)
    assert (output.read_bytes(), sidecar.read_bytes()) == before


def test_formal_v03_model_lock_publication_rejects_revision_tamper_on_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = build_formal_v03_model_lock()
    tampered = replace(
        expected,
        models=(
            replace(expected.models[0], revision="f" * 40),
            *expected.models[1:],
        ),
    )
    tampered.validate()
    monkeypatch.setattr(
        registry.ModelLock,
        "load",
        staticmethod(lambda _path: tampered),
    )

    with pytest.raises(RuntimeError, match="changed during publication"):
        publish_formal_v03_model_lock(
            output_path=(tmp_path / "formal-v03-model-lock.json").resolve()
        )


def test_formal_v03_model_lock_sidecar_rejects_post_publication_revision_tamper(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "formal-v03-model-lock.json").resolve()
    publish_formal_v03_model_lock(output_path=output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["models"][0]["revision"] = "f" * 40
    output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sidecar"):
        ModelLock.load(output)


def test_formal_v03_model_lock_race_never_replaces_winner_or_leaves_own_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (tmp_path / "formal-v03-model-lock.json").resolve()
    sidecar = Path(f"{output}.sha256")
    real_link = registry.os.link

    def concurrent_link(source, destination, *, follow_symlinks=True):
        if Path(destination) == output:
            output.write_text("concurrent-winner", encoding="ascii")
            raise FileExistsError("concurrent winner")
        return real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(registry.os, "link", concurrent_link)
    with pytest.raises(FileExistsError, match="output or sidecar already exists"):
        publish_formal_v03_model_lock(output_path=output)

    assert output.read_text(encoding="ascii") == "concurrent-winner"
    assert not sidecar.exists()
    assert tuple(tmp_path.iterdir()) == (output,)


def test_path_only_producer_requires_exact_members_before_mt_bench_na(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _content_inputs(tmp_path, monkeypatch)
    assert not Path(inputs.doctor_path).exists()
    spec = build_formal_v03_content_path_spec(inputs)
    require_formal_v03_content_path_spec(spec)
    assert len(spec.model_specs) == 25
    assert len(spec.e0_task_native_specs) == 7
    assert "sha256" not in json.dumps(inputs.to_dict(), sort_keys=True)
    assert "sha256" not in json.dumps(spec.to_dict(), sort_keys=True)

    calls = 0

    def forbidden_loader(_path: str):
        nonlocal calls
        calls += 1
        raise AssertionError("E0 authority opened before model coverage")

    monkeypatch.setattr(
        registry, "load_e0_task_native_source_authority", forbidden_loader
    )
    incomplete = replace(spec, model_specs=spec.model_specs[:-1])
    with pytest.raises(ValueError, match="model member coverage differs"):
        require_formal_v03_content_path_spec(incomplete)
    assert calls == 0


def test_path_only_producer_rejects_missing_foreign_and_wrong_cache_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _content_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="snapshot path coverage differs"):
        replace(inputs, model_snapshot_paths=inputs.model_snapshot_paths[:-1])

    spec = build_formal_v03_content_path_spec(inputs)
    tokenizer = next(row for row in spec.model_specs if row.role == "tokenizer")
    foreign = replace(tokenizer, model_id="foreign/protocol-tokenizer")
    foreign_rows = tuple(
        sorted(
            (foreign, *(row for row in spec.model_specs if row is not tokenizer)),
            key=lambda row: (
                row.role,
                row.model_id,
                row.revision,
                row.local_snapshot_path,
            ),
        )
    )
    with pytest.raises(ValueError, match="foreign or duplicated"):
        require_formal_v03_content_path_spec(replace(spec, model_specs=foreign_rows))

    first = inputs.model_snapshot_paths[0]
    wrong = (tmp_path / "hf" / "models--foreign--repo" / "snapshots").resolve()
    wrong = wrong / Path(first.absolute_path).name
    wrong.mkdir(parents=True)
    (wrong / "config.json").write_text("{}\n", encoding="utf-8")
    wrong_paths = (
        FormalV03NamedDirectoryPath(name=first.name, absolute_path=str(wrong)),
        *inputs.model_snapshot_paths[1:],
    )
    with pytest.raises(ValueError, match="path/revision differs"):
        build_formal_v03_content_path_spec(
            replace(inputs, model_snapshot_paths=wrong_paths)
        )


@pytest.mark.parametrize(
    "foreign_stages",
    (
        ("preflight", "E5", "E0"),
        ("preflight", "E3a", "E1a", "E5", "E0"),
    ),
)
def test_core_dspark_stage_coverage_rejects_missing_or_foreign_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    foreign_stages: tuple[str, ...],
) -> None:
    spec = build_formal_v03_content_path_spec(_content_inputs(tmp_path, monkeypatch))
    dspark = next(
        row
        for row in spec.model_specs
        if row.model_id == "deepseek-ai/dspark_qwen3_8b_block7"
    )
    assert dspark.stages == ("preflight", "E1a", "E5", "E0")
    mutated = replace(dspark, stages=foreign_stages)
    rows = tuple(mutated if row is dspark else row for row in spec.model_specs)
    with pytest.raises(ValueError, match="binding differs from registry"):
        require_formal_v03_content_path_spec(replace(spec, model_specs=rows))


def test_content_path_spec_publication_round_trips_without_caller_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _content_inputs(tmp_path, monkeypatch)
    input_path = (tmp_path / "v03-content-inputs.json").resolve()
    output_path = (tmp_path / "v03-content-path-spec.json").resolve()
    writer_argv = [
        "formal-single-operator",
        "write-v03-content-path-inputs",
        "--repository-root",
        inputs.repository_root,
        "--livecodebench-raw",
        inputs.livecodebench_raw_path,
        "--math500-raw",
        inputs.math500_raw_path,
        "--inventory",
        inputs.inventory_path,
        "--doctor-output",
        inputs.doctor_path,
        "--output",
        str(input_path),
    ]
    for row in inputs.model_snapshot_paths:
        writer_argv.extend(("--model-snapshot", f"{row.name}={row.absolute_path}"))
    for row in inputs.burstgpt_asset_paths:
        writer_argv.extend(("--burstgpt-asset", f"{row.name}={row.absolute_path}"))
    for row in inputs.e0_source_authority_paths:
        writer_argv.extend(("--e0-source-authority", f"{row.name}={row.absolute_path}"))
    assert main(writer_argv) == 0
    assert load_formal_v03_content_path_inputs(input_path) == inputs
    with pytest.raises(RuntimeError, match="target already exists"):
        main(writer_argv)

    spec = publish_formal_v03_content_path_spec_from_inputs(
        inputs_path=input_path,
        output_path=output_path,
    )

    assert output_path.is_file()
    assert len(spec.model_specs) == 25
    with pytest.raises(FileExistsError):
        publish_formal_v03_content_path_spec_from_inputs(
            inputs_path=input_path,
            output_path=output_path,
        )


def test_runtime_doctor_is_future_path_at_spec_time_but_must_deep_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _content_inputs(tmp_path, monkeypatch)
    build_formal_v03_content_path_spec(inputs)
    doctor = Path(inputs.doctor_path)
    with pytest.raises(ValueError):
        require_formal_v03_pass_runtime_doctor(doctor)

    publish_canonical_json_no_replace(
        doctor,
        {
            "schema_version": 2,
            "status": "PASS",
            "readiness": {
                "status": "PASS",
                "fail_count": 0,
                "unknown_count": 0,
            },
            "checks": {"fixture": {"status": "PASS"}},
        },
    )
    # A caller-authored PASS-shaped object is not a source-replayed doctor.
    with pytest.raises(ValueError):
        require_formal_v03_pass_runtime_doctor(doctor)

    from lightcone_spec import doctor as doctor_module

    observed: dict[str, object] = {}

    def revalidate(
        path,
        *,
        expected_bound_content_bundle=None,
        require_capacity_available=True,
    ):
        observed.update(
            path=str(path),
            expected_bound_content_bundle=expected_bound_content_bundle,
            require_capacity_available=require_capacity_available,
        )
        return CanonicalJsonProofBinding.bind(path)

    monkeypatch.setattr(
        doctor_module,
        "revalidate_trusted_single_operator_doctor_report",
        revalidate,
    )
    assert require_formal_v03_pass_runtime_doctor(doctor) == (
        CanonicalJsonProofBinding.bind(doctor)
    )
    assert observed == {
        "path": str(doctor),
        "expected_bound_content_bundle": None,
        "require_capacity_available": True,
    }


def _runtime_observation_fixture(
    tmp_path: Path,
    *,
    doctor_devices: list[dict[str, object]],
    inventory_payload: object | None = None,
) -> TrustedRuntimeObservations:
    inventory_path = (tmp_path / "runtime-inventory.json").resolve()
    doctor_path = (tmp_path / "runtime-doctor.json").resolve()
    inventory = _inventory()
    publish_canonical_json_no_replace(
        inventory_path,
        inventory.to_dict() if inventory_payload is None else inventory_payload,
    )
    publish_canonical_json_no_replace(
        doctor_path,
        {"gpu": {"parsed_inventory": {"devices": doctor_devices}}},
    )
    return TrustedRuntimeObservations(
        inventory=bind_trusted_json_artifact("remote_gpu_inventory", inventory_path),
        doctor=bind_trusted_json_artifact("remote_runtime_doctor", doctor_path),
    )


def _doctor_devices_from_inventory() -> list[dict[str, object]]:
    return [
        {
            "uuid": device.uuid,
            "name": device.model,
            "compute_capability": (
                f"{device.compute_capability[0]}.{device.compute_capability[1]}"
            ),
        }
        for device in _inventory().devices
    ]


def _inventory_with_device_count(device_count: int):
    inventory = _inventory()
    if device_count == 1:
        return replace(
            inventory,
            devices=inventory.devices[:1],
            topology_groups=(
                replace(inventory.topology_groups[0], gpu_uuids=("GPU-0",)),
            ),
        )
    if device_count == 3:
        third = replace(
            inventory.devices[-1],
            uuid="GPU-2",
            pci_bus_id="0000:03:00.0",
        )
        return replace(
            inventory,
            devices=(*inventory.devices, third),
            topology_groups=(
                replace(
                    inventory.topology_groups[0],
                    gpu_uuids=("GPU-0", "GPU-1", "GPU-2"),
                ),
            ),
        )
    raise AssertionError("unsupported inventory fixture size")


@pytest.mark.parametrize("device_count", (1, 3))
def test_runtime_observations_reject_non_exact_two_before_doctor_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_count: int,
) -> None:
    inventory = _inventory_with_device_count(device_count)
    runtime = _runtime_observation_fixture(
        tmp_path,
        doctor_devices=[],
        inventory_payload=inventory.to_dict(),
    )
    bundle = SimpleNamespace(runtime_observations=runtime)
    from lightcone_spec import doctor as doctor_module

    monkeypatch.setattr(
        doctor_module,
        "revalidate_trusted_single_operator_doctor_report",
        lambda *_args, **_kwargs: pytest.fail("non-exact-two inventory reached doctor"),
    )
    with pytest.raises(ValueError, match="exactly two GPUs"):
        registry._require_formal_v03_runtime_observation_identity(bundle)  # type: ignore[arg-type]


def test_runtime_observations_decode_inventory_and_match_exact_doctor_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_observation_fixture(
        tmp_path,
        doctor_devices=_doctor_devices_from_inventory(),
    )
    bundle = SimpleNamespace(runtime_observations=runtime)
    observed: dict[str, object] = {}

    from lightcone_spec import doctor as doctor_module

    def revalidate(
        path,
        *,
        expected_bound_content_bundle=None,
        require_capacity_available=True,
    ):
        observed["bundle"] = expected_bound_content_bundle
        observed["require_capacity_available"] = require_capacity_available
        return CanonicalJsonProofBinding.bind(path)

    monkeypatch.setattr(
        doctor_module,
        "revalidate_trusted_single_operator_doctor_report",
        revalidate,
    )
    registry._require_formal_v03_runtime_observation_identity(bundle)  # type: ignore[arg-type]
    assert observed["bundle"] is bundle
    assert observed["require_capacity_available"] is True


def test_bound_content_capacity_policy_reaches_both_doctor_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = object.__new__(TrustedSingleOperatorContentBundle)
    object.__setattr__(bundle, "runtime_binding_status", "BOUND")
    object.__setattr__(
        bundle,
        "runtime_observations",
        SimpleNamespace(doctor=SimpleNamespace(absolute_path="/unused/doctor.json")),
    )
    object.__setattr__(bundle, "model_members", ())
    object.__setattr__(bundle, "e0_task_native_descriptors", ())
    observed: list[tuple[str, bool]] = []

    monkeypatch.setattr(
        registry,
        "_require_formal_v03_model_coverage",
        lambda _rows: None,
    )
    monkeypatch.setattr(
        registry,
        "_require_formal_v03_e0_member_coverage",
        lambda _rows: None,
    )
    monkeypatch.setattr(
        registry,
        "_require_formal_v03_runtime_observation_identity",
        lambda _bundle, *, require_capacity_available=True, revalidate_runtime_observations=True: (
            observed.append(
                (
                    "runtime",
                    require_capacity_available,
                    revalidate_runtime_observations,
                )
            )
        ),
    )
    from lightcone_spec import doctor as doctor_module

    monkeypatch.setattr(
        doctor_module,
        "revalidate_trusted_single_operator_doctor_report",
        lambda _path, *, require_capacity_available=True, **_kwargs: observed.append(
            ("closure", require_capacity_available)
        ),
    )

    registry.require_formal_v03_bound_content_bundle(bundle)
    registry.require_formal_v03_bound_content_bundle(
        bundle,
        require_capacity_available=False,
        revalidate_runtime_observations=False,
    )
    assert observed == [
        ("runtime", True, True),
        ("closure", True),
        ("runtime", False, False),
    ]


def test_runtime_identity_only_reopens_bytes_without_fresh_doctor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_observation_fixture(
        tmp_path,
        doctor_devices=_doctor_devices_from_inventory(),
    )
    bundle = SimpleNamespace(runtime_observations=runtime)
    from lightcone_spec import doctor as doctor_module

    monkeypatch.setattr(
        doctor_module,
        "revalidate_trusted_single_operator_doctor_report",
        lambda *_args, **_kwargs: pytest.fail("identity-only replay probed doctor"),
    )

    registry._require_formal_v03_runtime_observation_identity(  # type: ignore[arg-type]
        bundle,
        require_capacity_available=False,
        revalidate_runtime_observations=False,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("uuid", "GPU-foreign"),
        ("name", "Foreign GPU"),
        ("compute_capability", "9.9"),
    ),
)
def test_runtime_observations_reject_doctor_device_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    devices = _doctor_devices_from_inventory()
    devices[0][field] = value
    runtime = _runtime_observation_fixture(tmp_path, doctor_devices=devices)
    bundle = SimpleNamespace(runtime_observations=runtime)

    from lightcone_spec import doctor as doctor_module

    monkeypatch.setattr(
        doctor_module,
        "revalidate_trusted_single_operator_doctor_report",
        lambda path, *, expected_bound_content_bundle=None, **_kwargs: (
            CanonicalJsonProofBinding.bind(path)
        ),
    )
    with pytest.raises(ValueError, match="GPU device sets differ"):
        registry._require_formal_v03_runtime_observation_identity(bundle)  # type: ignore[arg-type]


def test_runtime_observations_reject_arbitrary_canonical_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_observation_fixture(
        tmp_path,
        doctor_devices=_doctor_devices_from_inventory(),
        inventory_payload={"gpu": ["GPU-0", "GPU-1"]},
    )
    bundle = SimpleNamespace(runtime_observations=runtime)

    from lightcone_spec import doctor as doctor_module

    monkeypatch.setattr(
        doctor_module,
        "revalidate_trusted_single_operator_doctor_report",
        lambda path, *, expected_bound_content_bundle=None, **_kwargs: (
            CanonicalJsonProofBinding.bind(path)
        ),
    )
    with pytest.raises(ValueError, match="GPU inventory fields differ"):
        registry._require_formal_v03_runtime_observation_identity(bundle)  # type: ignore[arg-type]


def test_e0_source_authority_producer_scans_exact_seven_and_seals_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = (tmp_path / "raw-e0").resolve()
    raw_root.mkdir()
    raw_paths = []
    for task, pin in sorted(E0_TASK_NATIVE_SOURCE_PINS.items()):
        task_root = raw_root / task.lower().replace("-", "_")
        task_root.mkdir()
        path = (task_root / pin.source_file_name).resolve()
        path.write_text("fixture\n", encoding="utf-8")
        raw_paths.append(FormalV03NamedFilePath(name=task, absolute_path=str(path)))
    inputs_path = (tmp_path / "e0-raw-inputs.json").resolve()
    writer_argv = [
        "formal-single-operator",
        "write-v03-e0-raw-source-path-inputs",
        "--output",
        str(inputs_path),
    ]
    for row in raw_paths:
        writer_argv.extend(("--source", f"{row.name}={row.absolute_path}"))
    assert main(writer_argv) == 0
    assert tuple(
        row.name
        for row in load_formal_v03_e0_raw_source_path_inputs(inputs_path).source_paths
    ) == tuple(sorted(E0_TASK_NATIVE_SOURCE_PINS))
    with pytest.raises(RuntimeError, match="target already exists"):
        main(writer_argv)
    output = (tmp_path / "published-e0").resolve()
    output.mkdir()

    @dataclass(frozen=True)
    class FakeAuthority:
        task: str
        support_status: str

        def revalidate(self):
            return self

    scanned: list[str] = []
    published: dict[str, FakeAuthority] = {}

    def scan(*, task: str, raw_source_path: str):
        assert (
            Path(raw_source_path).name
            == E0_TASK_NATIVE_SOURCE_PINS[task].source_file_name
        )
        scanned.append(task)
        return FakeAuthority(
            task=task,
            support_status="UNSUPPORTED" if task == "MT-Bench" else "READY",
        )

    def publish(authority: FakeAuthority, *, output_path: Path):
        publish_canonical_json_no_replace(output_path, {"task": authority.task})
        published[str(output_path)] = authority

    def load(path: str | Path):
        return published[str(path)]

    monkeypatch.setattr(registry, "E0TaskNativeSourceAuthority", FakeAuthority)
    monkeypatch.setattr(registry, "scan_e0_task_native_source", scan)
    monkeypatch.setattr(registry, "publish_e0_task_native_source_authority", publish)
    monkeypatch.setattr(registry, "load_e0_task_native_source_authority", load)

    binding = publish_formal_v03_e0_source_authorities_from_inputs(
        inputs_path=inputs_path,
        output_directory=output,
    )
    index = load_formal_v03_e0_source_authority_index(binding.absolute_path)

    assert scanned == sorted(E0_TASK_NATIVE_SOURCE_PINS)
    assert tuple(row.name for row in index.authority_paths) == tuple(
        sorted(E0_TASK_NATIVE_SOURCE_PINS)
    )
    assert Path(binding.absolute_path).name == (
        FORMAL_V03_E0_SOURCE_AUTHORITY_INDEX_FILE_NAME
    )
    assert len(tuple(output.iterdir())) == 8
    with pytest.raises(FileExistsError, match="not empty"):
        publish_formal_v03_e0_source_authorities_from_inputs(
            inputs_path=inputs_path,
            output_directory=output,
        )
