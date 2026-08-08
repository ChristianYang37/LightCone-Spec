"""Config counterexamples for every spec-10.2 compatibility constraint,
plus locking fail-closed behaviour."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from lightcone_spec.config.loader import validate_adaptation_config_dict
from lightcone_spec.exit_codes import ConfigError, LockError
from lightcone_spec.locking import resolvers
from lightcone_spec.locking.resolvers import (
    _git_head_from_files,
    _runtime_source_sha256,
)


def test_every_catalog_manifest_passes_its_hash_sidecar():
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    root = Path(__file__).resolve().parents[1] / "manifests"
    paths = sorted(root.rglob("*.json"))
    assert paths
    for path in paths:
        assert Path(str(path) + ".sha256").is_file(), path
        ExperimentManifest.load(path)


def base_cfg(**over) -> dict:
    cfg = {
        "schema_version": 1,
        "method": "tts",
        "update_stride": 10,
        "optimizer": "adamw",
        "lr": 1e-4,
        "async": {"enabled": False, "logical_delay_rounds": 0, "max_in_flight": 1},
        "trace": {"artifact_root": "/tmp/lc_test"},
        "model": {"pair_id": "toy_markov4"},
        "dataset": {"adapter": "markov4_world"},
    }
    cfg.update(over)
    return cfg


def test_valid_config_passes():
    cfg = validate_adaptation_config_dict(base_cfg())
    assert cfg.method == "tts"


def _l3_evaluation_cfg(**trace_overrides):
    trace = {
        "artifact_root": "/tmp/lc_test",
        "trace_capture_max_bytes": 1 << 20,
        "l3_evaluation_only": True,
        **trace_overrides,
    }
    return base_cfg(
        method="lc_transport",
        async_={"enabled": True, "logical_delay_rounds": 0, "max_in_flight": 1},
        controller={"artifact_path": "/tmp/controller.json"},
        transport={"basis_path": "/tmp/transport.json"},
        trace=trace,
    )


def test_l3_evaluation_mode_is_explicit_bounded_and_identity_neutral():
    from lightcone_spec.methods.registry import controller_runtime_identity

    # Use the public key spelling rather than the Python field alias.
    raw = _l3_evaluation_cfg()
    raw["async"] = raw.pop("async_")
    evaluation = validate_adaptation_config_dict(raw)
    production_raw = {**raw, "trace": {"artifact_root": "/tmp/lc_test"}}
    production = validate_adaptation_config_dict(production_raw)

    assert evaluation.trace.l3_evaluation_only is True
    assert controller_runtime_identity(evaluation) == controller_runtime_identity(
        production
    )


@pytest.mark.parametrize(
    "method,trace",
    [
        (
            "lc_transport",
            {
                "artifact_root": "/tmp/lc_test",
                "trace_capture_max_bytes": 0,
                "l3_evaluation_only": True,
            },
        ),
        (
            "lc_transport",
            {
                "artifact_root": "/tmp/lc_test",
                "privacy_mode": "private",
                "trace_capture_max_bytes": 1 << 20,
                "l3_evaluation_only": True,
            },
        ),
        (
            "naive_async",
            {
                "artifact_root": "/tmp/lc_test",
                "trace_capture_max_bytes": 1 << 20,
                "l3_evaluation_only": True,
            },
        ),
    ],
)
def test_l3_evaluation_mode_fails_closed_outside_its_contract(method, trace):
    raw = _l3_evaluation_cfg()
    raw["async"] = raw.pop("async_")
    raw["method"] = method
    raw["trace"] = trace
    if method != "lc_transport":
        raw.pop("controller")
        raw.pop("transport")

    with pytest.raises(ConfigError, match="l3_evaluation_only"):
        validate_adaptation_config_dict(raw)


@pytest.mark.parametrize(
    "over",
    [
        {"method": "no_such_method"},
        {"method": "static"},  # static + adamw optimizer
        {"method": "static", "optimizer": "none",
         "controller": {"artifact_path": "/tmp/x.json"}},
        {"method": "static", "optimizer": "none",
         "async": {"enabled": True, "logical_delay_rounds": 1, "max_in_flight": 1}},
        {"method": "sync_fresh", "optimizer": "sgd"},
        {"method": "sync_fresh",
         "async": {"enabled": True, "logical_delay_rounds": 1, "max_in_flight": 1}},
        {"method": "tts", "optimizer": "sgd"},
        {"method": "tts",
         "async": {"enabled": False, "logical_delay_rounds": 0, "max_in_flight": 2}},
        {"method": "onlinespec_ogd", "optimizer": "adamw"},
        {"method": "lc_gate"},  # missing controller artifact
        {"method": "lc_transport",
         "controller": {"artifact_path": "/tmp/x.json"}},  # missing basis
        {"method": "naive_async",
         "async": {"enabled": True, "logical_delay_rounds": 1, "max_in_flight": 2}},
        {"trainable_scope": "full-drafter"},  # tail tiers only
        {"sampling": {"temperature": 0.5, "top_p": 1.0, "max_new_tokens": 8}},
        {"sampling": {"temperature": 1.0, "top_p": 0.9, "max_new_tokens": 8}},
        {"unknown_field": 1},  # extra=forbid
        {"schema_version": 2},
    ],
)
def test_config_counterexamples_fail_closed(over):
    with pytest.raises(ConfigError):
        validate_adaptation_config_dict(base_cfg(**over))


def test_lockfile_offline_verify_fail_closed(tmp_path):
    from lightcone_spec.locking.lockfile import load_lockfile

    bogus = tmp_path / "lock.json"
    bogus.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(Exception):
        load_lockfile(bogus)


def test_canonical_json_hash_stable():
    from lightcone_spec.locking.hashing import sha256_json

    a = sha256_json({"b": 1, "a": [2, 3]})
    b = sha256_json({"a": [2, 3], "b": 1})
    assert a == b and len(a) == 64


def test_alpaca_uses_locked_raw_json_without_dataset_script(monkeypatch):
    from lightcone_spec.benchmarks.registry import get_adapter

    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"instruction": "hello"}]

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset)
    )
    revision = "a" * 40
    samples = get_adapter("alpaca").load_revision(revision)
    assert samples[0].prompt == "hello"
    assert calls == [
        (
            ("json",),
            {
                "data_files": {
                    "eval": (
                        "hf://datasets/tatsu-lab/alpaca_eval@"
                        f"{revision}/alpaca_eval.json"
                    )
                },
                "split": "eval",
            },
        )
    ]


def test_livecodebench_release_v6_uses_six_locked_jsonl_files(monkeypatch):
    from lightcone_spec.benchmarks.registry import get_adapter

    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [
            {
                "question_id": "q1",
                "question_content": "solve",
                "public_test_cases": "[]",
            }
        ]

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset)
    )
    revision = "b" * 40
    samples = get_adapter("livecodebench").load_revision(revision)
    assert samples[0].sample_id == "q1"
    assert calls[0][0] == ("json",)
    uris = calls[0][1]["data_files"]["test"]
    assert len(uris) == 6
    assert uris[0].endswith(f"@{revision}/test.jsonl")
    assert uris[-1].endswith(f"@{revision}/test6.jsonl")


def test_locked_dataset_retries_from_cache_after_network_failure(monkeypatch):
    from lightcone_spec.benchmarks.registry import get_adapter

    calls = []
    fake_datasets = SimpleNamespace(
        config=SimpleNamespace(HF_DATASETS_OFFLINE=False),
    )

    def load_dataset(*args, **kwargs):
        calls.append(fake_datasets.config.HF_DATASETS_OFFLINE)
        if len(calls) == 1:
            raise ConnectionError("mirror unavailable")
        return [{"question_id": "q1", "question_content": "solve"}]

    fake_datasets.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    samples = get_adapter("livecodebench").load_revision("b" * 40)
    assert samples[0].sample_id == "q1"
    assert calls == [False, True]
    assert fake_datasets.config.HF_DATASETS_OFFLINE is False


def test_locked_dataset_reports_actionable_cache_miss(monkeypatch):
    from lightcone_spec.benchmarks.registry import get_adapter

    attempts = 0
    fake_datasets = SimpleNamespace(
        config=SimpleNamespace(HF_DATASETS_OFFLINE=False),
    )

    def load_dataset(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("mirror unavailable")
        raise FileNotFoundError("not cached")

    fake_datasets.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    with pytest.raises(LockError, match="no complete local cache"):
        get_adapter("livecodebench").load_revision("b" * 40)
    assert attempts == 2
    assert fake_datasets.config.HF_DATASETS_OFFLINE is False


def test_mt_bench_uses_first_turn_as_the_paired_task_suffix():
    from lightcone_spec.benchmarks.registry import get_adapter

    sample = get_adapter("mt_bench").row_to_sample(
        0,
        {
            "prompt_id": 7,
            "prompt": ["First turn", "Second turn depends on the first answer"],
        },
    )
    assert sample.sample_id == "7"
    assert sample.prompt == "First turn"


def test_dataset_lock_records_raw_file_metadata(monkeypatch):
    from lightcone_spec.benchmarks.base import BenchmarkSample
    from lightcone_spec.benchmarks.registry import get_adapter
    from lightcone_spec.cli.main import _lock_dataset

    adapter = get_adapter("math500")
    monkeypatch.setattr(
        adapter,
        "load_sample_ids_revision",
        lambda revision: ["m1"],
    )

    class Api:
        @staticmethod
        def dataset_info(repo_id, files_metadata=False):
            assert repo_id == "HuggingFaceH4/MATH-500"
            assert files_metadata is True
            return SimpleNamespace(
                sha="c" * 40,
                siblings=[
                    SimpleNamespace(
                        rfilename="test.jsonl",
                        size=123,
                        lfs=SimpleNamespace(sha256="d" * 64),
                        blob_id=None,
                    )
                ],
            )

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    locked = _lock_dataset("math500")
    assert locked.revision == "c" * 40
    assert locked.files[0].relpath == "test.jsonl"
    assert locked.files[0].size_bytes == 123
    assert locked.files[0].sha256 == "d" * 64


def test_locked_dataset_subset_materializes_only_selected_rows(monkeypatch):
    from lightcone_spec.benchmarks.base import BenchmarkAdapter, _SAMPLE_CACHE
    from lightcone_spec.locking.lockfile import LockedDataset, LockedEnvironment, Lockfile

    adapter = BenchmarkAdapter(
        key="large",
        source_group="test",
        hf_path="org/large",
        hf_config=None,
        split="test",
        quality_metric="exact_match",
        output_cap=1,
        task_type="math",
        prompt_template="{question}",
        stop_strings=(),
        scorer_kind="exact_match",
        timeout_s=0,
        license_note="test",
        data_file="test.jsonl",
    )
    rows = [{"id": f"id-{i}", "question": f"q-{i}"} for i in range(100)]
    ids = [row["id"] for row in rows]
    monkeypatch.setattr(adapter, "_load_hf_dataset", lambda revision: rows)
    materialized = []
    original = adapter.row_to_sample

    def tracked(index, row):
        materialized.append(index)
        return original(index, row)

    monkeypatch.setattr(adapter, "row_to_sample", tracked)
    lock = Lockfile(
        created_utc="2026-08-02T00:00:00Z",
        datasets=[
            LockedDataset(
                adapter_key="large",
                source="org/large",
                split="test",
                revision="a" * 40,
                sample_ids_sha256=adapter.sample_ids_hash(ids),
                num_samples=len(ids),
            )
        ],
        environment=LockedEnvironment(python_version="3.12", torch_version="2.11"),
    )
    _SAMPLE_CACHE.clear()
    first = adapter.load_samples(lock, limit=7)
    second = adapter.load_samples(lock, limit=7)
    assert len(first) == len(second) == 7
    assert len(materialized) == 7
    assert {sample.sample_id for sample in first} == set(
        adapter.deterministic_subset(ids, 7)
    )

    next_window = adapter.load_samples(lock, limit=7, offset=7)
    assert len(next_window) == 7
    assert {sample.sample_id for sample in first}.isdisjoint(
        sample.sample_id for sample in next_window
    )
    assert {sample.sample_id for sample in next_window} == set(
        adapter.deterministic_subset(ids, 7, offset=7)
    )


def test_sample_cache_isolates_source_and_rendering_identity(monkeypatch):
    from lightcone_spec.benchmarks.base import BenchmarkAdapter, _SAMPLE_CACHE
    from lightcone_spec.locking.lockfile import (
        LockedDataset,
        LockedEnvironment,
        Lockfile,
    )

    def make_adapter(source: str, question: str) -> BenchmarkAdapter:
        adapter = BenchmarkAdapter(
            key="shared-key",
            source_group="test",
            hf_path=source,
            hf_config=None,
            split="test",
            quality_metric="exact_match",
            output_cap=1,
            task_type="math",
            prompt_template="first:{question}",
            stop_strings=(),
            scorer_kind="exact_match",
            timeout_s=0,
            license_note="test",
        )
        monkeypatch.setattr(
            adapter,
            "_load_hf_dataset",
            lambda revision: [{"id": "same-id", "question": question}],
        )
        return adapter

    def make_lock(adapter: BenchmarkAdapter) -> Lockfile:
        return Lockfile(
            created_utc="2026-08-04T00:00:00Z",
            datasets=[
                LockedDataset(
                    adapter_key=adapter.key,
                    source=adapter.hf_path,
                    split=adapter.split,
                    revision="a" * 40,
                    sample_ids_sha256=adapter.sample_ids_hash(["same-id"]),
                    num_samples=1,
                )
            ],
            environment=LockedEnvironment(
                python_version="3.12", torch_version="2.11"
            ),
        )

    _SAMPLE_CACHE.clear()
    first = make_adapter("org/first", "one")
    second = make_adapter("org/second", "two")
    assert first.load_samples(make_lock(first), limit=1)[0].prompt == "first:one"
    assert second.load_samples(make_lock(second), limit=1)[0].prompt == "first:two"

    second.prompt_template = "changed:{question}"
    assert second.load_samples(make_lock(second), limit=1)[0].prompt == "changed:two"


def test_duplicate_sample_ids_fail_closed_when_locking_and_loading(monkeypatch):
    from lightcone_spec.benchmarks.base import BenchmarkAdapter, _SAMPLE_CACHE
    from lightcone_spec.cli.main import _lock_dataset
    from lightcone_spec.locking.lockfile import (
        LockedDataset,
        LockedEnvironment,
        Lockfile,
    )

    adapter = BenchmarkAdapter(
        key="duplicates",
        source_group="test",
        hf_path="org/duplicates",
        hf_config=None,
        split="test",
        quality_metric="exact_match",
        output_cap=1,
        task_type="math",
        prompt_template="{question}",
        stop_strings=(),
        scorer_kind="exact_match",
        timeout_s=0,
        license_note="test",
    )
    rows = [
        {"id": "duplicate", "question": "one"},
        {"id": "duplicate", "question": "two"},
    ]
    monkeypatch.setattr(adapter, "_load_hf_dataset", lambda revision: rows)
    ids = [row["id"] for row in rows]
    lock = Lockfile(
        created_utc="2026-08-04T00:00:00Z",
        datasets=[
            LockedDataset(
                adapter_key=adapter.key,
                source=adapter.hf_path,
                split=adapter.split,
                revision="a" * 40,
                sample_ids_sha256=adapter.sample_ids_hash(ids),
                num_samples=len(ids),
            )
        ],
        environment=LockedEnvironment(
            python_version="3.12", torch_version="2.11"
        ),
    )
    _SAMPLE_CACHE.clear()
    with pytest.raises(LockError, match="duplicate sample ID"):
        adapter.load_samples(lock, limit=1)

    from lightcone_spec.benchmarks.registry import BENCHMARK_ADAPTERS

    monkeypatch.setitem(BENCHMARK_ADAPTERS, adapter.key, adapter)
    monkeypatch.setattr(
        adapter, "load_sample_ids_revision", lambda revision: ids
    )

    class Api:
        @staticmethod
        def dataset_info(repo_id, files_metadata=False):
            return SimpleNamespace(sha="a" * 40, siblings=[])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    with pytest.raises(LockError, match="duplicate sample ID"):
        _lock_dataset(adapter.key)


def test_locked_dataset_rejects_adapter_binding_and_count_drift(monkeypatch):
    from lightcone_spec.benchmarks.base import BenchmarkAdapter, _SAMPLE_CACHE
    from lightcone_spec.exit_codes import LockError
    from lightcone_spec.locking.lockfile import (
        LockedDataset,
        LockedEnvironment,
        Lockfile,
    )

    adapter = BenchmarkAdapter(
        key="bound",
        source_group="test",
        hf_path="org/source",
        hf_config="cfg",
        split="test",
        quality_metric="exact_match",
        output_cap=1,
        task_type="math",
        prompt_template="{question}",
        stop_strings=(),
        scorer_kind="exact_match",
        timeout_s=0,
        license_note="test",
    )
    rows = [{"id": "one", "question": "q"}]
    monkeypatch.setattr(adapter, "_load_hf_dataset", lambda revision: rows)

    def make_lock(*, source="org/source", num_samples=1):
        return Lockfile(
            created_utc="2026-08-03T00:00:00Z",
            datasets=[
                LockedDataset(
                    adapter_key="bound",
                    source=source,
                    config="cfg",
                    split="test",
                    revision="a" * 40,
                    sample_ids_sha256=adapter.sample_ids_hash(["one"]),
                    num_samples=num_samples,
                )
            ],
            environment=LockedEnvironment(
                python_version="3.12", torch_version="2.11"
            ),
        )

    _SAMPLE_CACHE.clear()
    with pytest.raises(LockError, match="source/config/split drift"):
        adapter.load_samples(make_lock(source="org/other"), limit=1)
    with pytest.raises(LockError, match="sample count drift"):
        adapter.load_samples(make_lock(num_samples=2), limit=1)


def test_prepare_datasets_writes_hash_bound_receipt(monkeypatch, tmp_path):
    import json
    from argparse import Namespace

    from lightcone_spec.benchmarks.base import _SAMPLE_CACHE
    from lightcone_spec.benchmarks.registry import get_adapter
    from lightcone_spec.cli.main import cmd_prepare_datasets
    from lightcone_spec.locking.lockfile import (
        LockedDataset,
        LockedEnvironment,
        Lockfile,
    )

    adapter = get_adapter("math500")
    rows = [
        {"unique_id": "m1", "problem": "one", "answer": "1"},
        {"unique_id": "m2", "problem": "two", "answer": "2"},
    ]
    monkeypatch.setattr(adapter, "_load_hf_dataset", lambda revision: rows)
    ids = ["m1", "m2"]
    lock = Lockfile(
        created_utc="2026-08-03T00:00:00Z",
        datasets=[
            LockedDataset(
                adapter_key="math500",
                source=adapter.hf_path,
                config=adapter.hf_config,
                split=adapter.split,
                revision="a" * 40,
                sample_ids_sha256=adapter.sample_ids_hash(ids),
                num_samples=len(ids),
                license_note=adapter.license_note,
            )
        ],
        environment=LockedEnvironment(
            python_version="3.12", torch_version="2.11"
        ),
    )
    lock_path = tmp_path / "lock.json"
    lock.write(lock_path)
    output = tmp_path / "receipt.json"
    _SAMPLE_CACHE.clear()

    assert cmd_prepare_datasets(
        Namespace(
            lockfile=str(lock_path),
            datasets=["math500"],
            limit=1,
            offset=1,
            output=str(output),
        )
    ) == 0
    receipt = json.loads(output.read_text())
    assert receipt["lockfile_sha256"] == lock.content_sha256()
    assert receipt["offset"] == 1
    assert receipt["datasets"][0]["selected_count"] == 1
    assert Path(str(output) + ".sha256").read_text().strip()


def test_runtime_source_hash_and_git_head_fallback(tmp_path):
    source = tmp_path / "runtime"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n")
    (source / "ignored.txt").write_text("first\n")
    (source / "._module.py").write_text("apple metadata\n")
    digest = _runtime_source_sha256(source)
    assert digest is not None

    (source / "ignored.txt").write_text("second\n")
    assert _runtime_source_sha256(source) == digest
    (source / "module.py").write_text("value = 2\n")
    assert _runtime_source_sha256(source) != digest

    git_dir = source / ".git"
    branch = git_dir / "refs" / "heads"
    branch.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/runtime\n")
    (branch / "runtime").write_text("a" * 40 + "\n")
    assert _git_head_from_files(source) == "a" * 40


def test_lock_can_rebind_host_without_reresolving_inputs(tmp_path, monkeypatch):
    from lightcone_spec.cli.main import cmd_lock
    from lightcone_spec.locking.lockfile import (
        LockedDataset,
        LockedEnvironment,
        LockedGitRepo,
        LockedHFSnapshot,
        Lockfile,
        load_lockfile,
    )

    source = Lockfile(
        created_utc="2026-08-01T00:00:00Z",
        git_repos=[
            LockedGitRepo(
                name="sglang", url="https://example.invalid/sglang", commit_sha="a" * 40
            )
        ],
        hf_snapshots=[
            LockedHFSnapshot(
                repo_id="org/model", snapshot_sha="b" * 40, role="target"
            )
        ],
        datasets=[
            LockedDataset(
                adapter_key="bench",
                source="org/bench",
                split="test",
                revision="c" * 40,
                sample_ids_sha256="d" * 64,
                num_samples=1,
            )
        ],
        environment=LockedEnvironment(
            python_version="3.12", torch_version="2.11"
        ),
    )
    source_path = tmp_path / "inputs.lock.json"
    source.write(source_path)
    rebound_environment = LockedEnvironment(
        python_version="3.12", cuda_version="12.9", torch_version="2.11"
    )
    monkeypatch.setattr(resolvers, "now_utc_iso", lambda: "2026-08-02T00:00:00Z")
    monkeypatch.setattr(resolvers, "resolve_environment", lambda: rebound_environment)
    monkeypatch.setattr(resolvers, "resolve_gpus", lambda: [])

    output = tmp_path / "host.lock.json"
    args = SimpleNamespace(
        output=str(output),
        pairs=[],
        datasets=[],
        reuse_inputs_from=str(source_path),
        refresh=True,
        offline_verify=None,
        skip_git=False,
    )
    assert cmd_lock(args) == 0
    rebound = load_lockfile(output)
    assert rebound.created_utc == "2026-08-02T00:00:00Z"
    assert rebound.git_repos == source.git_repos
    assert rebound.hf_snapshots == source.hf_snapshots
    assert rebound.datasets == source.datasets
    assert rebound.environment == rebound_environment


def test_qwen3_4b_dflash_pair_matches_8b_backend_capabilities():
    from lightcone_spec.config.schema import MODEL_PAIRS

    pair = MODEL_PAIRS["qwen3_4b_dflash16"]
    reference = MODEL_PAIRS["qwen3_8b_dflash16"]
    assert pair == {
        **reference,
        "target": "Qwen/Qwen3-4B",
        "drafter": "z-lab/Qwen3-4B-DFlash-b16",
    }


def test_lock_rebind_incrementally_resolves_only_missing_inputs(
    tmp_path, monkeypatch
):
    from lightcone_spec.cli import main as cli_main
    from lightcone_spec.locking.lockfile import (
        LockedDataset,
        LockedEnvironment,
        LockedHFSnapshot,
        Lockfile,
        load_lockfile,
    )

    target = LockedHFSnapshot(
        repo_id="Qwen/Qwen3-4B", snapshot_sha="a" * 40, role="target"
    )
    math500 = LockedDataset(
        adapter_key="math500",
        source="HuggingFaceH4/MATH-500",
        split="test",
        revision="b" * 40,
        sample_ids_sha256="c" * 64,
        num_samples=1,
    )
    source = Lockfile(
        created_utc="2026-08-01T00:00:00Z",
        hf_snapshots=[target, target],
        datasets=[math500, math500],
        environment=LockedEnvironment(
            python_version="3.12", torch_version="2.11"
        ),
    )
    source_path = tmp_path / "inputs.lock.json"
    source.write(source_path)

    model_calls = []

    def resolve_model(repo_id, role, revision, include_chat_template=False):
        model_calls.append((repo_id, role, revision, include_chat_template))
        return LockedHFSnapshot(
            repo_id=repo_id, snapshot_sha="d" * 40, role=role
        )

    dataset_calls = []

    def resolve_dataset(key):
        dataset_calls.append(key)
        return LockedDataset(
            adapter_key=key,
            source="lmsys/mt_bench_human_judgments",
            split="test",
            revision="e" * 40,
            sample_ids_sha256="f" * 64,
            num_samples=1,
        )

    monkeypatch.setattr(resolvers, "resolve_hf_snapshot", resolve_model)
    monkeypatch.setattr(cli_main, "_lock_dataset", resolve_dataset)
    monkeypatch.setattr(
        resolvers,
        "resolve_environment",
        lambda: LockedEnvironment(python_version="3.12", torch_version="2.11"),
    )
    monkeypatch.setattr(resolvers, "resolve_gpus", lambda: [])

    output = tmp_path / "host.lock.json"
    args = SimpleNamespace(
        output=str(output),
        pairs=["qwen3_4b_dflash16", "qwen3_4b_dflash16"],
        datasets=["math500", "mt_bench", "mt_bench"],
        reuse_inputs_from=str(source_path),
        refresh=True,
        offline_verify=None,
        skip_git=True,
    )
    assert cli_main.cmd_lock(args) == 0

    rebound = load_lockfile(output)
    assert [(item.repo_id, item.role) for item in rebound.hf_snapshots] == [
        ("Qwen/Qwen3-4B", "target"),
        ("z-lab/Qwen3-4B-DFlash-b16", "drafter"),
    ]
    assert [item.adapter_key for item in rebound.datasets] == [
        "math500",
        "mt_bench",
    ]
    assert model_calls == [
        ("z-lab/Qwen3-4B-DFlash-b16", "drafter", "main", False)
    ]
    assert dataset_calls == ["mt_bench"]


def test_lock_rebind_increment_rejects_snapshot_role_conflict(
    tmp_path, monkeypatch
):
    from lightcone_spec.cli.main import cmd_lock
    from lightcone_spec.locking.lockfile import (
        LockedEnvironment,
        LockedHFSnapshot,
        Lockfile,
    )

    source = Lockfile(
        created_utc="2026-08-01T00:00:00Z",
        hf_snapshots=[
            LockedHFSnapshot(
                repo_id="Qwen/Qwen3-4B",
                snapshot_sha="a" * 40,
                role="drafter",
            )
        ],
        environment=LockedEnvironment(
            python_version="3.12", torch_version="2.11"
        ),
    )
    source_path = tmp_path / "inputs.lock.json"
    source.write(source_path)
    monkeypatch.setattr(
        resolvers,
        "resolve_hf_snapshot",
        lambda *args, **kwargs: pytest.fail("role conflict must fail before resolve"),
    )

    args = SimpleNamespace(
        output=str(tmp_path / "host.lock.json"),
        pairs=["qwen3_4b_dflash16"],
        datasets=[],
        reuse_inputs_from=str(source_path),
        refresh=True,
        offline_verify=None,
        skip_git=True,
    )
    with pytest.raises(LockError, match="requested role is 'target'"):
        cmd_lock(args)
