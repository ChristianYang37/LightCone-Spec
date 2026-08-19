from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments import (
    formal_single_operator_e0_interface_producer as producer,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.stage_materialization import E0_BACKENDS, E0_MODELS


def test_public_preprobe_producer_accepts_paths_only() -> None:
    parameters = inspect.signature(
        producer.publish_formal_single_operator_e0_preprobe_interface_index
    ).parameters

    assert tuple(parameters) == (
        "protocol_lock_path",
        "predecessor_completion_path",
        "trusted_content_bundle_path",
        "output_root",
    )
    assert len(producer._PORTS) == 12
    assert len(set(producer._PORTS)) == 12


def test_source_owned_static_configs_cover_exact_twelve_without_task_authority() -> (
    None
):
    sampling = SamplingProfile(purpose="natural", ignore_eos=False)
    configs = []
    for model in E0_MODELS:
        target = SimpleNamespace(revision="1" * 40)
        for backend in E0_BACKENDS:
            drafter = SimpleNamespace(
                model_id=f"example/{model.rsplit('/', 1)[-1]}-{backend}",
                revision="2" * 40,
            )
            config = producer._run_config(
                model=model,
                backend=backend,
                target=target,
                drafter=drafter,
                draft_depth=3,
                sampling=sampling,
                gpu_uuid="GPU-0",
            )
            configs.append(config)
            assert config.method == "static"
            assert config.adaptation is None
            assert config.online_spec is None
            assert config.runtime.topology_mode == "tp1_dp1"
            assert config.runtime.max_running_requests == 1
            assert config.runtime.speculative_num_draft_tokens == 4
            assert config.runtime.speculative_eagle_topk == (
                1 if backend == "EAGLE3" else None
            )

    assert len(configs) == 12
    assert len({config.model.key for config in configs}) == 12


def test_drafter_is_discovered_from_one_exact_runtime_binding() -> None:
    member = SimpleNamespace(
        role="drafter",
        stages=("E0",),
        runtime_bindings=(
            SimpleNamespace(
                stage="E0",
                target_model_id=E0_MODELS[0],
                backend=E0_BACKENDS[0],
                draft_depth=7,
            ),
        ),
    )
    bundle = SimpleNamespace(model_members=(member,))

    assert producer._drafter(
        bundle,
        model=E0_MODELS[0],
        backend=E0_BACKENDS[0],
    ) == (member, 7)

    with pytest.raises(
        producer.FormalSingleOperatorE0PreprobeInterfaceBlocked,
        match="member_missing",
    ):
        producer._drafter(
            bundle,
            model=E0_MODELS[1],
            backend=E0_BACKENDS[0],
        )


def test_private_output_root_is_created_once_and_is_resumable(tmp_path: Path) -> None:
    root = (tmp_path / "e0-preprobe").resolve()

    assert producer._private_root(root) == root
    assert root.is_dir()
    assert producer._private_root(root) == root

    visible = (tmp_path / "visible").resolve()
    visible.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="not private"):
        producer._private_root(visible)


def test_partial_pair_is_retained_and_resume_uses_a_new_attempt(
    tmp_path: Path,
) -> None:
    root = producer._private_root((tmp_path / "e0-preprobe").resolve())
    first = root / "pair-00"
    first.mkdir(mode=0o700)
    (first / "interrupted-evidence.json").write_text("{}\n", encoding="utf-8")

    resumed, attempt = producer._resume_pair_or_allocate_attempt(
        root=root,
        pair_index=0,
        model=E0_MODELS[0],
        backend=E0_BACKENDS[0],
    )

    assert resumed is None
    assert attempt == root / "pair-00-retry-001"
    assert (first / "interrupted-evidence.json").is_file()


def test_unique_completed_retry_is_resumed_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = producer._private_root((tmp_path / "e0-preprobe").resolve())
    (root / "pair-00").mkdir(mode=0o700)
    completed = root / "pair-00-retry-001"
    completed.mkdir(mode=0o700)
    receipt_path = completed / "preprobe-interface.json"
    receipt_path.write_text('{"receipt":"complete"}\n', encoding="utf-8")
    monkeypatch.setattr(
        producer,
        "load_e0_prepared_model_backend_interface_receipt",
        lambda _path: SimpleNamespace(
            model=E0_MODELS[0],
            backend=E0_BACKENDS[0],
            schema_version=3,
        ),
    )

    resumed, attempt = producer._resume_pair_or_allocate_attempt(
        root=root,
        pair_index=0,
        model=E0_MODELS[0],
        backend=E0_BACKENDS[0],
    )

    assert attempt is None
    assert resumed is not None
    assert resumed.absolute_path == str(receipt_path.resolve())
