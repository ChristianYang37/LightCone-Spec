"""CLI entry point (spec 9.2/9.3).

Subcommands: lock | serve | exactness | replay | run-manifest | analyze
| validate-artifacts. Exit codes are stable: 0 success, 2 config, 3
lock, 4 resource skip, 5 exactness, 6 numerical, 7 runtime/GPU, 8
artifact validation, 9 incomplete coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from lightcone_spec import exit_codes
from lightcone_spec.exit_codes import LightconeError


def build_parser() -> argparse.ArgumentParser:
    from lightcone_spec.config.schema import WEIGHT_UPDATE_MODE_CHOICES

    p = argparse.ArgumentParser(
        prog="lightcone-spec",
        description="LightCone-Spec adaptation and experiment framework",
    )
    sub = p.add_subparsers(dest="command", required=True)

    lock = sub.add_parser("lock", help="resolve and freeze all mutable inputs")
    lock.add_argument("--output", default="lock/lightcone.lock.json")
    lock.add_argument("--pairs", nargs="*", default=[])
    lock.add_argument("--datasets", nargs="*", default=[])
    lock.add_argument(
        "--reuse-inputs-from",
        default=None,
        metavar="LOCKFILE",
        help=(
            "reuse the exact git/model/dataset inputs from an existing lock "
            "while rebinding the current Python/CUDA/GPU environment; with "
            "--pairs/--datasets, resolve and append only missing inputs"
        ),
    )
    lock.add_argument("--refresh", action="store_true",
                      help="explicitly regenerate a new lockfile")
    lock.add_argument("--offline-verify", metavar="ROOTS_JSON", default=None,
                      help="JSON mapping repo_id -> local dir; verify without network")
    lock.add_argument("--skip-git", action="store_true")

    doctor = sub.add_parser("doctor", help="read-only GPU host preflight")
    doctor.add_argument("--runtime-root", default="~/lightcone-tts-runtime")
    doctor.add_argument("--min-free-gib", type=int, default=80)
    doctor.add_argument("--no-network", action="store_true")
    doctor.add_argument("--strict", action="store_true",
                        help="return runtime/GPU failure unless install-ready")

    prep = sub.add_parser(
        "prepare-models", help="download and verify locked model snapshots"
    )
    prep.add_argument("--lockfile", required=True)
    prep.add_argument("--model-cache", required=True)
    prep.add_argument("--pairs", nargs="*", default=[])
    prep.add_argument("--output", required=True,
                      help="write repo_id -> verified local directory JSON")

    prep_data = sub.add_parser(
        "prepare-datasets",
        help="materialize and verify locked datasets before loading a model",
    )
    prep_data.add_argument("--lockfile", required=True)
    prep_data.add_argument("--datasets", nargs="*", default=[])
    prep_data.add_argument("--limit", type=int, default=128)
    prep_data.add_argument(
        "--offset",
        type=int,
        default=0,
        help="start index in the deterministic locked-dataset permutation",
    )
    prep_data.add_argument("--output", required=True,
                           help="write a content-addressed preflight receipt")

    serve = sub.add_parser("serve", help="start the adapted SGLang server")
    serve.add_argument("--config", required=True)
    serve.add_argument("--lockfile", required=True)
    serve.add_argument("--port", type=int, default=30000)
    serve.add_argument("--model-roots", required=True)
    serve.add_argument(
        "--weight-update-mode",
        choices=WEIGHT_UPDATE_MODE_CHOICES,
        default=None,
        help="override the non-static config with a cache-safe tail update tier",
    )

    ex = sub.add_parser("exactness", help="run the exactness toy suite (P0-B)")
    ex.add_argument("--output", default=None)
    ex.add_argument("--mc-samples", type=int, default=20000)
    ex.add_argument("--seed", type=int, default=0)
    ex.add_argument("--with-p0a", action="store_true",
                    help="also run the P0-A delayed-optimization toy")

    rp = sub.add_parser("replay", help="counterfactual replay + controller fit")
    rp.add_argument("--pair", default="toy_markov4")
    rp.add_argument("--dataset", default="phase_switch")
    rp.add_argument("--output-dir", required=True)
    rp.add_argument("--num-traces", type=int, default=24)
    rp.add_argument("--rounds", type=int, default=40)
    rp.add_argument("--delay", type=int, default=5)
    rp.add_argument("--cadence", type=int, default=6)
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--transport-rank", type=int, default=16, choices=(8, 16, 32))
    rp.add_argument("--select-distance-weights", action="store_true")
    rp.add_argument(
        "--trace-root", default=None,
        help="fit from bounded real-model replay shards instead of toy traces",
    )

    rm = sub.add_parser("run-manifest", help="execute an immutable manifest")
    rm.add_argument("--manifest", required=True)
    rm.add_argument("--artifact-root", required=True)
    rm.add_argument("--lockfile", default=None)
    rm.add_argument("--runtime-root", default="~/lightcone-tts-runtime")
    rm.add_argument("--model-roots", default=None)
    rm.add_argument("--controller-root", default=None)
    rm.add_argument(
        "--peak-tflops-per-gpu",
        type=float,
        default=None,
        help="dense BF16/FP16 peak used only to normalize estimated MFU",
    )
    rm.add_argument("--no-resume", action="store_true")
    rm.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "execute only these methods from the immutable source manifest; "
            "the effective manifest identity is recomputed"
        ),
    )
    rm.add_argument(
        "--lifecycles",
        nargs="+",
        default=None,
        help=(
            "execute only these lifecycles from the immutable source manifest; "
            "the effective manifest identity is recomputed"
        ),
    )
    rm.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="override the effective manifest engine learning rate",
    )
    rm.add_argument(
        "--weight-update-mode",
        choices=WEIGHT_UPDATE_MODE_CHOICES,
        default=None,
        help=(
            "override only non-static units in memory; source manifest remains "
            "unchanged and effective identities are recomputed"
        ),
    )

    an = sub.add_parser("analyze", help="tables, claim gates and figures")
    an.add_argument("--artifact-root", required=True)
    an.add_argument("--output-dir", required=True)
    an.add_argument(
        "--manifest",
        default=None,
        help="optional experiment manifest; require complete coverage before analysis",
    )
    an.add_argument(
        "--weight-update-mode",
        choices=WEIGHT_UPDATE_MODE_CHOICES,
        default=None,
        help=(
            "apply the same non-static mode overlay used by run-manifest "
            "before checking coverage"
        ),
    )
    an.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="apply the same method subset used by run-manifest",
    )
    an.add_argument(
        "--lifecycles",
        nargs="+",
        default=None,
        help="apply the same lifecycle subset used by run-manifest",
    )
    an.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="apply the same learning-rate overlay used by run-manifest",
    )
    an.add_argument("--baseline", default="static")
    an.add_argument("--itl-slo-ms", type=float, default=50.0)

    va = sub.add_parser("validate-artifacts", help="validate run artifacts")
    va.add_argument("--artifact-root", required=True)
    va.add_argument("--manifest", default=None)
    va.add_argument(
        "--weight-update-mode",
        choices=WEIGHT_UPDATE_MODE_CHOICES,
        default=None,
        help=(
            "apply the same non-static mode overlay used by run-manifest "
            "before checking coverage"
        ),
    )
    va.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="apply the same method subset used by run-manifest",
    )
    va.add_argument(
        "--lifecycles",
        nargs="+",
        default=None,
        help="apply the same lifecycle subset used by run-manifest",
    )
    va.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="apply the same learning-rate overlay used by run-manifest",
    )
    va.add_argument("--coverage-output", default=None)
    return p


# ---------------------------------------------------------------------------


def cmd_lock(args) -> int:
    from lightcone_spec.locking.lockfile import Lockfile, load_lockfile
    from lightcone_spec.locking.resolvers import (
        PINNED_GIT_REPOS,
        now_utc_iso,
        resolve_environment,
        resolve_git_repo,
        resolve_gpus,
        resolve_hf_snapshot,
    )
    from lightcone_spec.locking.verify import verify_lockfile_offline
    from lightcone_spec.config.schema import MODEL_PAIRS

    out_path = Path(args.output)
    if args.offline_verify:
        lock = load_lockfile(out_path)
        roots = json.loads(Path(args.offline_verify).read_text())
        verified = verify_lockfile_offline(lock, roots)
        print(f"offline verify OK for {len(verified)} snapshots")
        return exit_codes.SUCCESS
    if out_path.is_file() and not args.refresh:
        raise exit_codes.LockError(
            f"lockfile {out_path} already exists; use --refresh to regenerate"
        )
    if args.reuse_inputs_from:
        source_lock = load_lockfile(args.reuse_inputs_from)
        git_repos = source_lock.git_repos
        snapshots = list(source_lock.hf_snapshots)
        datasets = list(source_lock.datasets)

        # Preserve the original rebind-only contract byte-for-byte at the
        # input-list level.  Incremental mode is entered only when the caller
        # explicitly requests pairs or datasets; there we normalize exact
        # duplicates and fail closed on ambiguous identities.
        if args.pairs or args.datasets:
            snapshots_by_repo = {}
            for snapshot in snapshots:
                previous = snapshots_by_repo.get(snapshot.repo_id)
                if previous is not None and previous != snapshot:
                    raise exit_codes.LockError(
                        "reused lock contains conflicting snapshots for "
                        f"{snapshot.repo_id!r}"
                    )
                snapshots_by_repo[snapshot.repo_id] = snapshot
            snapshots = list(snapshots_by_repo.values())

            datasets_by_key = {}
            for dataset in datasets:
                previous = datasets_by_key.get(dataset.adapter_key)
                if previous is not None and previous != dataset:
                    raise exit_codes.LockError(
                        "reused lock contains conflicting datasets for "
                        f"{dataset.adapter_key!r}"
                    )
                datasets_by_key[dataset.adapter_key] = dataset
            datasets = list(datasets_by_key.values())

            for pair_id in dict.fromkeys(args.pairs):
                if pair_id not in MODEL_PAIRS:
                    raise exit_codes.ConfigError(f"unknown model pair {pair_id}")
                pair = MODEL_PAIRS[pair_id]
                for repo_id, role, include_chat_template in (
                    (pair["target"], "target", True),
                    (pair["drafter"], "drafter", False),
                ):
                    existing = snapshots_by_repo.get(repo_id)
                    if existing is not None:
                        if existing.role != role:
                            raise exit_codes.LockError(
                                f"reused snapshot {repo_id!r} has role "
                                f"{existing.role!r}; requested role is {role!r}"
                            )
                        continue
                    resolved = resolve_hf_snapshot(
                        repo_id,
                        role,
                        "main",
                        include_chat_template=include_chat_template,
                    )
                    snapshots.append(resolved)
                    snapshots_by_repo[repo_id] = resolved

            for key in dict.fromkeys(args.datasets):
                if key in datasets_by_key:
                    continue
                resolved = _lock_dataset(key)
                datasets.append(resolved)
                datasets_by_key[key] = resolved
    else:
        git_repos = []
        if not args.skip_git:
            for name, (url, sha) in PINNED_GIT_REPOS.items():
                git_repos.append(resolve_git_repo(name, url, sha))
        snapshots = []
        resolved_snapshot_repos: set[str] = set()
        for pair_id in args.pairs:
            if pair_id not in MODEL_PAIRS:
                raise exit_codes.ConfigError(f"unknown model pair {pair_id}")
            pair = MODEL_PAIRS[pair_id]
            if pair["target"] not in resolved_snapshot_repos:
                snapshots.append(
                    resolve_hf_snapshot(
                        pair["target"], "target", "main", include_chat_template=True
                    )
                )
                resolved_snapshot_repos.add(pair["target"])
            if pair["drafter"] not in resolved_snapshot_repos:
                snapshots.append(
                    resolve_hf_snapshot(pair["drafter"], "drafter", "main")
                )
                resolved_snapshot_repos.add(pair["drafter"])
        datasets = [_lock_dataset(key) for key in args.datasets]
    lock = Lockfile(
        created_utc=now_utc_iso(),
        git_repos=git_repos,
        hf_snapshots=snapshots,
        datasets=datasets,
        environment=resolve_environment(),
        gpus=resolve_gpus(),
    )
    digest = lock.write(out_path)
    print(f"lockfile written: {out_path} sha256={digest}")
    return exit_codes.SUCCESS


def cmd_doctor(args) -> int:
    from lightcone_spec.doctor import collect_doctor_report

    report = collect_doctor_report(
        args.runtime_root,
        min_free_gib=args.min_free_gib,
        check_network=not args.no_network,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not report["ready_for_install"]:
        return exit_codes.RUNTIME_GPU_FAILURE
    return exit_codes.SUCCESS


def cmd_prepare_models(args) -> int:
    from lightcone_spec.config.schema import MODEL_PAIRS
    from lightcone_spec.locking.download import (
        prepare_locked_models,
        write_model_roots,
    )
    from lightcone_spec.locking.lockfile import load_lockfile

    lock = load_lockfile(args.lockfile)
    repo_ids: list[str] = []
    for pair_id in args.pairs:
        if pair_id not in MODEL_PAIRS:
            raise exit_codes.ConfigError(f"unknown model pair {pair_id}")
        pair = MODEL_PAIRS[pair_id]
        repo_ids.extend((pair["target"], pair["drafter"]))
    repo_ids = list(dict.fromkeys(repo_ids))
    roots = prepare_locked_models(
        lock, args.model_cache, repo_ids=(repo_ids or None)
    )
    write_model_roots(roots, args.output)
    print(json.dumps({"verified_model_roots": roots, "output": args.output}, indent=2))
    return exit_codes.SUCCESS


def _prepare_locked_datasets(
    lock, dataset_keys: list[str], limit: int, offset: int = 0
) -> dict:
    """Materialize immutable benchmark inputs and return a stable receipt."""
    from lightcone_spec.benchmarks.registry import get_adapter

    if limit <= 0:
        raise exit_codes.ConfigError("dataset preflight --limit must be positive")
    if offset < 0:
        raise exit_codes.ConfigError("dataset preflight --offset must be nonnegative")
    available = {dataset.adapter_key for dataset in lock.datasets}
    requested = list(dict.fromkeys(dataset_keys or sorted(available)))
    unknown = sorted(set(requested) - available)
    if unknown:
        raise exit_codes.LockError(
            f"requested datasets are not locked: {unknown}"
        )
    prepared = []
    for key in requested:
        try:
            adapter = get_adapter(key)
        except KeyError as exc:
            raise exit_codes.ConfigError(
                f"no benchmark adapter is registered for locked dataset {key!r}"
            ) from exc
        locked = lock.find_dataset(key)
        samples = adapter.load_samples(lock, limit=limit, offset=offset)
        expected = min(limit, max(locked.num_samples - offset, 0))
        if len(samples) != expected:
            raise exit_codes.LockError(
                f"{key}: preflight selected {len(samples)} samples; expected {expected}"
            )
        prepared.append(
            {
                "adapter_key": key,
                "source": locked.source,
                "config": locked.config,
                "split": locked.split,
                "revision": locked.revision,
                "num_samples": locked.num_samples,
                "sample_ids_sha256": locked.sample_ids_sha256,
                "selected_count": len(samples),
                "selected_sample_ids_sha256": adapter.sample_ids_hash(
                    [sample.sample_id for sample in samples]
                ),
                "files": [file.model_dump(mode="json") for file in locked.files],
            }
        )
    return {
        "schema_version": 1,
        "lockfile_sha256": lock.content_sha256(),
        "limit": limit,
        "offset": offset,
        "datasets": prepared,
    }


def _write_dataset_receipt(receipt: dict, output_path: str | Path) -> str:
    from lightcone_spec.locking.hashing import canonical_json, sha256_bytes

    body = canonical_json(receipt)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(output)
    digest = sha256_bytes(body.encode("utf-8"))
    Path(str(output) + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


@contextmanager
def _artifact_root_lock(artifact_root: str | Path) -> Iterator[Path]:
    """Serialize manifest publication and resume decisions for one root."""
    import fcntl

    root = Path(artifact_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with root.joinpath(".run-manifest.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield root
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def cmd_prepare_datasets(args) -> int:
    from lightcone_spec.locking.lockfile import load_lockfile

    lock = load_lockfile(args.lockfile)
    receipt = _prepare_locked_datasets(
        lock,
        list(args.datasets),
        int(args.limit),
        int(getattr(args, "offset", 0)),
    )
    output = Path(args.output).expanduser().resolve()
    digest = _write_dataset_receipt(receipt, output)
    print(json.dumps({"receipt": str(output), "sha256": digest}, indent=2))
    return exit_codes.SUCCESS


def _lock_dataset(key: str):
    from lightcone_spec.benchmarks.registry import get_adapter
    from lightcone_spec.locking.lockfile import LockedDataset, LockedFile

    adapter = get_adapter(key)
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(adapter.hf_path, files_metadata=True)
    ids = adapter.load_sample_ids_revision(info.sha)
    # Keep the lock writer fail-closed even when a custom adapter overrides or
    # a test double replaces ``load_sample_ids_revision``.
    adapter.validate_sample_ids(ids)
    siblings = {sibling.rfilename: sibling for sibling in info.siblings or []}
    files = []
    for relpath in adapter.locked_data_files():
        sibling = siblings.get(relpath)
        if sibling is None:
            raise exit_codes.LockError(
                f"{key}: locked data file missing at revision {info.sha}: {relpath}"
            )
        digest = None
        if sibling.lfs is not None:
            digest = sibling.lfs.sha256
        elif sibling.blob_id is not None:
            digest = f"gitblob:{sibling.blob_id}"
        if not digest:
            raise exit_codes.LockError(
                f"{key}: no content hash for data file {relpath}"
            )
        files.append(
            LockedFile(
                relpath=relpath,
                size_bytes=int(sibling.size or 0),
                sha256=digest,
            )
        )
    return LockedDataset(
        adapter_key=key,
        source=adapter.hf_path,
        config=adapter.hf_config,
        split=adapter.split,
        revision=info.sha,
        files=files,
        sample_ids_sha256=adapter.sample_ids_hash(ids),
        num_samples=len(ids),
        license_note=adapter.license_note,
    )


def cmd_serve(args) -> int:
    from lightcone_spec.config.loader import load_adaptation_config
    from lightcone_spec.config.schema import (
        AdaptationConfig,
        canonical_weight_update_mode,
    )
    from lightcone_spec.locking.lockfile import load_lockfile
    from lightcone_spec.locking.verify import check_pair_lock
    from lightcone_spec.config.schema import MODEL_PAIRS

    config = load_adaptation_config(args.config)
    mode = getattr(args, "weight_update_mode", None)
    if mode is not None and config.method != "static":
        resolved = config.model_dump(by_alias=True, mode="json")
        resolved["trainable_scope"] = canonical_weight_update_mode(mode)
        config = AdaptationConfig.model_validate(resolved)
    if config.controller.artifact_path is not None:
        from lightcone_spec.controller.artifact import ControllerArtifact
        from lightcone_spec.methods.registry import validate_controller_artifact

        validate_controller_artifact(
            config, ControllerArtifact.load(config.controller.artifact_path)
        )
    lock = load_lockfile(args.lockfile)
    if config.model.pair_id in MODEL_PAIRS:
        pair = MODEL_PAIRS[config.model.pair_id]
        check_pair_lock(lock, pair["target"], pair["drafter"])
    import torch

    if not torch.cuda.is_available():
        raise exit_codes.RuntimeGpuFailure(
            "serve requires CUDA GPUs and the pinned SGLang fork; this host "
            "has no CUDA device"
        )
    try:
        from sglang.srt.entrypoints.http_server import launch_server
        from sglang.srt.server_args import ServerArgs
    except ImportError as exc:
        raise exit_codes.RuntimeGpuFailure(
            f"pinned SGLang fork not importable: {exc}"
        ) from exc
    pair = MODEL_PAIRS[config.model.pair_id]
    if args.model_roots:
        from lightcone_spec.locking.download import load_model_roots

        roots = load_model_roots(args.model_roots)
    else:
        roots = {}
    if roots:
        from lightcone_spec.locking.verify import verify_lockfile_offline

        verify_lockfile_offline(
            lock,
            {k: v for k, v in roots.items() if k in (pair["target"], pair["drafter"])},
            require_all=False,
        )
    target_path = roots.get(pair["target"], pair["target"])
    drafter_path = roots.get(pair["drafter"], pair["drafter"])
    from tempfile import TemporaryDirectory

    import yaml

    from lightcone_spec.sglang_bridge.client import _pair_server_args

    # Always materialize the validated representation.  This both preserves
    # the source file and guarantees that legacy `adapter/full` spellings do
    # not leak into runtime evidence.
    with TemporaryDirectory(prefix="lightcone-serve-") as tmp:
        runtime_config_path = Path(tmp) / "adaptation.runtime.yaml"
        runtime_config_path.write_text(
            yaml.safe_dump(
                config.model_dump(by_alias=True, mode="json"), sort_keys=True
            )
        )
        kwargs = _pair_server_args(
            pair,
            target_path=target_path,
            drafter_path=drafter_path,
            adaptation_config_path=str(runtime_config_path),
            num_draft_tokens=config.runtime.speculative_num_draft_tokens,
            tensor_parallel_size=config.runtime.tensor_parallel_size,
            random_seed=config.runtime.seed,
            adaptation_reserve_mb=config.runtime.calibrated_reserve_mb,
        )
        kwargs["port"] = args.port
        launch_server(ServerArgs(**kwargs))
    return exit_codes.SUCCESS


def cmd_exactness(args) -> int:
    from lightcone_spec.runtime.exactness_harness import run_exactness_suite

    report = run_exactness_suite(mc_samples=args.mc_samples, seed=args.seed)
    payload = {"exactness": report.to_dict()}
    if args.with_p0a:
        from lightcone_spec.toys import run_p0a

        p0a = run_p0a(seed=args.seed)
        payload["p0a"] = p0a.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)
    if not report.passed:
        return exit_codes.EXACTNESS_VERSION_FAILURE
    return exit_codes.SUCCESS


def cmd_replay(args) -> int:
    if args.trace_root:
        from lightcone_spec.replay.real import fit_real_replay

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result = fit_real_replay(
            args.trace_root,
            model_pair_id=args.pair,
            transport_rank=args.transport_rank,
            seed=args.seed,
        )
        from lightcone_spec.controller.artifact import controller_artifact_filename

        runtime_identity = result.artifact.extra.get(
            "controller_runtime_identity", {}
        )
        mode = runtime_identity.get("candidate", {}).get("weight_update_mode")
        layout_sha = result.artifact.extra.get("parameter_layout_sha256")
        artifact_path = out / controller_artifact_filename(
            args.pair, mode, layout_sha
        )
        digest = result.artifact.freeze(artifact_path)
        report = {
            "artifact_path": str(artifact_path),
            "artifact_sha256": digest,
            "utility_metric": result.artifact.extra.get(
                "controller_utility_metric"
            ),
            "utility_diagnostics": result.artifact.extra.get(
                "utility_diagnostics"
            ),
            "evaluations": {k: v.to_dict() for k, v in result.evaluations.items()},
            "h1": result.h1,
            "harmful_rate": result.harmful_rate,
            "mean_gradient_cosine": result.mean_cosine,
            "split_sizes": result.split_sizes,
            "trace_exactness": result.artifact.extra["trace_exactness"],
            "l3_gate": result.artifact.extra["l3_gate"],
            "oracle_replay_gate": result.artifact.extra[
                "oracle_replay_gate"
            ],
            "tts_paired_gate": result.artifact.extra["tts_paired_gate"],
            "learned_policy_gate": result.artifact.extra[
                "learned_policy_gate"
            ],
        }
        (out / "replay_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True)
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return exit_codes.SUCCESS
    if not args.pair.startswith("toy_"):
        raise exit_codes.ConfigError(
            "real model replay requires --trace-root; refusing to label toy "
            "replay as a real model pair"
        )
    import numpy as np

    from lightcone_spec.benchmarks.synthetic import synthetic_world
    from lightcone_spec.plots.figures import (
        delay_drift_utility_figure,
        predictor_comparison_figure,
    )
    from lightcone_spec.replay.counterfactual import ReplayConfig, replay_trace
    from lightcone_spec.replay.pipeline import (
        fit_replay_pipeline,
        select_distance_weights,
    )
    from lightcone_spec.replay.trace import generate_trace
    from lightcone_spec.runtime.toy_model import make_toy_pair
    from lightcone_spec.trajectory.distance import DistanceWeights
    from lightcone_spec.trajectory.zvector import default_zvectorizer

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    world = synthetic_world(args.dataset, seed=args.seed)
    pair = make_toy_pair(world, seed=args.seed)
    traces = [
        generate_trace(
            pair,
            base_prompt_id=f"prompt-{i // 2}",
            trajectory_id=f"traj-{i}",
            dataset=args.dataset,
            num_rounds=args.rounds,
            seed=args.seed,
        )
        for i in range(args.num_traces)
    ]
    zvec = default_zvectorizer()
    zvec.fit_normalization([s for t in traces for s in t.states])
    rcfg = ReplayConfig(cadence=args.cadence, delay=args.delay)

    def records_for(w: DistanceWeights):
        recs = []
        for t in traces:
            recs.extend(replay_trace(pair, t, rcfg, w, zvec))
        return recs

    if args.select_distance_weights:
        weights, _ = select_distance_weights(records_for, seed=args.seed)
    else:
        weights = DistanceWeights(a_p=1 / 3, a_h=1 / 3, a_e=1 / 3)
        weights.frozen = True
    records = records_for(weights)
    result = fit_replay_pipeline(
        records,
        model_pair_id=args.pair,
        zvec=zvec,
        transport_rank=args.transport_rank,
        seed=args.seed,
    )
    result.artifact.distance_weights = weights
    artifact_path = out / f"{args.pair}.controller.json"
    digest = result.artifact.freeze(artifact_path)
    report = {
        "artifact_path": str(artifact_path),
        "artifact_sha256": digest,
        "evaluations": {k: v.to_dict() for k, v in result.evaluations.items()},
        "h1": result.h1,
        "harmful_rate": result.harmful_rate,
        "mean_gradient_cosine": result.mean_cosine,
        "split_sizes": result.split_sizes,
        "distance_weights": weights.to_dict(),
    }
    (out / "replay_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    predictor_comparison_figure(
        {k: v.to_dict() for k, v in result.evaluations.items()},
        out / "predictor_comparison.png",
    )
    rows = [r.row for r in records]
    delay_drift_utility_figure(
        np.asarray([r.round_delay for r in rows]),
        np.asarray([r.rho_path for r in rows]),
        np.asarray([r.utility for r in rows]),
        out / "delay_drift_utility.png",
    )
    print(json.dumps({"h1": result.h1, "artifact": str(artifact_path)}, indent=2))
    return exit_codes.SUCCESS


def _apply_manifest_overlays(manifest, args):
    """Resolve every shared CLI execution overlay in one canonical order."""
    return (
        manifest.with_methods(getattr(args, "methods", None))
        .with_lifecycles(getattr(args, "lifecycles", None))
        .with_weight_update_mode(getattr(args, "weight_update_mode", None))
        .with_learning_rate(getattr(args, "learning_rate", None))
    )


def _require_manifest_for_derived_overlays(args) -> None:
    requested = [
        flag
        for flag, value in (
            ("--lifecycles", getattr(args, "lifecycles", None)),
            ("--learning-rate", getattr(args, "learning_rate", None)),
        )
        if value is not None
    ]
    if requested and not getattr(args, "manifest", None):
        raise exit_codes.ConfigError(
            f"{', '.join(requested)} requires --manifest to define the "
            "effective experiment"
        )


def cmd_run_manifest(args) -> int:
    from lightcone_spec.config.schema import CONTROLLER_METHODS, SYNTHETIC_DATASET_KEYS
    from lightcone_spec.doctor import configure_runtime_cuda_toolkit
    from lightcone_spec.orchestration.executor import execute_manifest
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    manifest = _apply_manifest_overlays(
        ExperimentManifest.load(args.manifest), args
    )
    from dataclasses import replace
    from lightcone_spec.locking.lockfile import load_lockfile

    engine_params = dict(manifest.engine_params)
    real_units = [
        unit for unit in manifest.units if not unit.model_pair.startswith("toy_")
    ]
    if real_units and not args.lockfile:
        raise exit_codes.ConfigError(
            "real-model manifests require --lockfile before model loading"
        )
    if real_units and not args.model_roots:
        raise exit_codes.ConfigError(
            "real-model manifests require --model-roots from prepare-models"
        )
    if any(unit.method in CONTROLLER_METHODS for unit in real_units) and not (
        args.controller_root or engine_params.get("controller_artifact_path")
    ):
        raise exit_codes.ConfigError(
            "controller methods require --controller-root or a manifest-bound "
            "controller_artifact_path before model loading"
        )
    if getattr(args, "weight_update_mode", None) is not None and any(
        unit.method != "static" for unit in manifest.units
    ):
        from lightcone_spec.config.schema import canonical_weight_update_mode

        engine_params["weight_update_mode_override"] = (
            canonical_weight_update_mode(args.weight_update_mode)
        )
    runtime_root = Path(args.runtime_root).expanduser().resolve()
    engine_params["runtime_root"] = str(runtime_root)
    if real_units:
        toolkit = configure_runtime_cuda_toolkit(runtime_root)
        if not toolkit["supported"]:
            raise exit_codes.ConfigError(
                "real-model GPU runs require an isolated CUDA >=12.9 toolkit "
                f"with nvcc under {runtime_root}/cuda-*; found "
                f"{toolkit.get('version') or 'none'}"
            )
        engine_params["cuda_toolkit_root"] = toolkit["root"]
        engine_params["cuda_toolkit_version"] = toolkit["version"]
    lock = None
    if args.lockfile:
        lock = load_lockfile(args.lockfile)
        engine_params["lockfile_path"] = str(Path(args.lockfile).resolve())
        digest = lock.content_sha256()
        if manifest.lockfile_sha256 not in (None, digest):
            raise exit_codes.LockError(
                "manifest lockfile_sha256 does not match --lockfile"
            )
        manifest = replace(manifest, lockfile_sha256=digest)
    if args.model_roots:
        engine_params["model_roots_path"] = str(Path(args.model_roots).resolve())
    if args.controller_root:
        engine_params["controller_root"] = str(Path(args.controller_root).resolve())
    if args.peak_tflops_per_gpu is not None:
        if args.peak_tflops_per_gpu <= 0:
            raise exit_codes.ConfigError("--peak-tflops-per-gpu must be positive")
        engine_params["peak_tflops_per_gpu"] = args.peak_tflops_per_gpu
    receipt = None
    if real_units:
        assert lock is not None
        from lightcone_spec.orchestration.runtime_config import (
            preflight_gpu_manifest_inputs,
            runtime_implementation_fingerprint,
        )

        compiler_versions = getattr(
            getattr(lock, "environment", None), "compiler_versions", {}
        )
        locked_reference = {
            key: compiler_versions[key]
            for key in (
                "lightcone_runtime_source_sha256",
                "sglang_runtime_source_sha256",
                "sglang_fork_commit",
                "sglang_fork_dirty",
            )
            if key in compiler_versions
        }
        engine_params["runtime_implementation_fingerprint"] = (
            runtime_implementation_fingerprint(
                locked_reference=locked_reference
            )
        )
        engine_params["runtime_input_preflight"] = preflight_gpu_manifest_inputs(
            real_units, engine_params
        )
        dataset_keys = sorted(
            {
                unit.dataset
                for unit in real_units
                if unit.dataset not in SYNTHETIC_DATASET_KEYS
            }
        )
        receipt = _prepare_locked_datasets(
            lock,
            dataset_keys,
            int(engine_params.get("prompt_limit", 128)),
            int(engine_params.get("prompt_offset", 0)),
        )
    with _artifact_root_lock(args.artifact_root) as artifact_root:
        if receipt is not None:
            engine_params["dataset_preflight_sha256"] = _write_dataset_receipt(
                receipt, artifact_root / "dataset-preflight.json"
            )
        manifest = replace(manifest, engine_params=engine_params)
        report = execute_manifest(
            manifest, artifact_root, resume=not args.no_resume
        )
    print(json.dumps({"counts": report.counts()}, indent=2))
    if not report.ok:
        for o in report.outcomes:
            if o.status.startswith("failed"):
                print(f"unit {o.unit_id[:12]} {o.status}: {o.detail}", file=sys.stderr)
        if any(o.status == "failed_exactness" for o in report.outcomes):
            return exit_codes.EXACTNESS_VERSION_FAILURE
        return exit_codes.RUNTIME_GPU_FAILURE
    return exit_codes.SUCCESS


def _expected_unit_ids(expected: list[dict] | None) -> set[str] | None:
    if expected is None:
        return None
    return {str(unit["unit_id"]) for unit in expected}


def _scope_unit_status(
    unit_status: dict[str, str], expected_unit_ids: set[str] | None
) -> dict[str, str]:
    if expected_unit_ids is None:
        return dict(unit_status)
    return {
        unit_id: status
        for unit_id, status in unit_status.items()
        if unit_id in expected_unit_ids
    }


def _write_analysis_provenance(
    artifact_root: Path,
    output_dir: Path,
    *,
    baseline: str,
    itl_slo_ms: float,
    run_status: dict[str, str],
    expected_manifest_sha256: str | None = None,
    weight_update_mode: str | None = None,
    methods: list[str] | tuple[str, ...] | None = None,
    lifecycles: list[str] | tuple[str, ...] | None = None,
    learning_rate: float | None = None,
    validated_run_ids: set[str] | None = None,
) -> tuple[Path, Path]:
    """Bind derived outputs transitively to immutable run hash ledgers."""
    from lightcone_spec.locking.hashing import (
        canonical_json,
        sha256_bytes,
        sha256_file,
    )

    provenance_names = {
        "analysis-manifest.json",
        "analysis-manifest.sha256",
        "analysis-hashes.json",
    }
    input_runs = []
    for run_dir in sorted(path for path in artifact_root.iterdir() if path.is_dir()):
        if validated_run_ids is not None and run_dir.name not in validated_run_ids:
            continue
        manifest_path = run_dir / "manifest.json"
        hashes_path = run_dir / "hashes.json"
        if not manifest_path.is_file() or not hashes_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        unit_id = str(manifest.get("unit_id", run_dir.name))
        exit_path = run_dir / "exit.json"
        if (
            run_status.get(unit_id) != "complete_valid"
            or not exit_path.is_file()
            or json.loads(exit_path.read_text()).get("status") != "complete_valid"
        ):
            continue
        input_runs.append(
            {
                "run_id": run_dir.name,
                "unit_id": unit_id,
                "manifest_sha256": sha256_file(manifest_path),
                "hashes_sha256": sha256_file(hashes_path),
            }
        )

    derived = {}
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        if relative in provenance_names:
            continue
        derived[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    if not derived:
        raise exit_codes.ArtifactValidationFailure(
            "analysis produced no derived outputs to attest"
        )

    manifest_path = output_dir / "analysis-manifest.json"
    manifest_sha_path = output_dir / "analysis-manifest.sha256"
    manifest_body = canonical_json(
        {
            "schema_version": 1,
            "analysis": {
                "baseline": baseline,
                "itl_slo_ms": float(itl_slo_ms),
                "expected_manifest_sha256": expected_manifest_sha256,
                "weight_update_mode_overlay": weight_update_mode,
                "methods_overlay": list(methods) if methods is not None else None,
                "lifecycles_overlay": (
                    list(lifecycles) if lifecycles is not None else None
                ),
                "learning_rate_overlay": learning_rate,
            },
            "input_runs": input_runs,
            "derived_outputs": derived,
        }
    )
    manifest_path.write_text(manifest_body)
    manifest_sha_path.write_text(
        sha256_bytes(manifest_body.encode("utf-8")) + "\n"
    )
    attested = {
        **derived,
        manifest_path.name: {
            "sha256": sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        manifest_sha_path.name: {
            "sha256": sha256_file(manifest_sha_path),
            "bytes": manifest_sha_path.stat().st_size,
        },
    }
    hashes_path = output_dir / "analysis-hashes.json"
    hashes_path.write_text(json.dumps(attested, indent=2, sort_keys=True) + "\n")
    return manifest_path, hashes_path


def _continuous_prefix_windows(engine_params: dict) -> tuple[tuple[int, int], ...]:
    """Validate optional P5 single-request true-prefix windows.

    Independent exact-prefix checkpoints and a continuously adapted request
    answer different questions.  The latter is explicitly declared in the
    resolved run manifest and is bucketed by the true prefix observed before
    each proposal. Keeping this contract separate prevents a
    128-token request started at 40K from being pooled with a request whose
    optimizer has actually followed the whole 0--40K trajectory.
    """

    raw = engine_params.get("p5_continuous_prefix_windows")
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise exit_codes.ArtifactValidationFailure(
            "p5_continuous_prefix_windows must be a non-empty list"
        )
    windows: list[tuple[int, int]] = []
    for index, value in enumerate(raw):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise exit_codes.ArtifactValidationFailure(
                "each continuous prefix window must be [start, end] integers"
            )
        start, end = value
        if start < 0 or end <= start:
            raise exit_codes.ArtifactValidationFailure(
                f"invalid continuous prefix window at index {index}: {value}"
            )
        if windows and start != windows[-1][1]:
            raise exit_codes.ArtifactValidationFailure(
                "continuous prefix windows must be ordered and contiguous"
            )
        windows.append((start, end))
    return tuple(windows)


def _p5_update_stride(run_manifest: dict) -> int:
    """Resolve the authoritative P5 stride with a legacy-table fallback."""

    from lightcone_spec.statistics.tables import DEFAULT_P5_UPDATE_STRIDE

    raw = run_manifest.get(
        "stride", run_manifest.get("update_stride", DEFAULT_P5_UPDATE_STRIDE)
    )
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise exit_codes.ArtifactValidationFailure(
            "P5 run manifest update stride must be a positive integer"
        )
    return raw


def _aggregate_p5_run_counter(
    frame,
    group_keys: list[str],
    *,
    counter: str,
    run_key: str = "analysis_run_id",
):
    """Deduplicate one replicated run counter, then sum independent runs."""

    import pandas as pd

    missing = [name for name in (*group_keys, run_key, counter) if name not in frame]
    if missing:
        raise ValueError(f"P5 summaries lack counter fields: {missing}")
    values = pd.to_numeric(frame[counter], errors="coerce")
    if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError(f"P5 {counter} must contain nonnegative integers")
    normalized = frame.copy()
    normalized[counter] = values.astype(int)
    per_run = (
        normalized.groupby([*group_keys, run_key], as_index=False)
        .agg(**{counter: (counter, "max")})
    )
    return per_run.groupby(group_keys, as_index=False).agg(
        **{counter: (counter, "sum")}
    )


def _p5_scientific_sample_pass(
    paired_prompt_clusters: int, benchmark_repetitions: int
) -> bool:
    """Apply the locked P5 minimum: two prompts and five repetitions."""

    return paired_prompt_clusters >= 2 and benchmark_repetitions >= 5


def _p5_window_dominance(group, *, continuous_trajectory: bool):
    """Require a positive paired gain CI in every measured context bucket.

    The aggregate LCAG/elasticity gate answers a different question and stays
    unchanged.  This stricter companion is used only when a claim says the
    method wins throughout all declared contexts.
    """

    order = (
        ["prefix_window_start", "prefix_window_end"]
        if continuous_trajectory
        else ["context_length"]
    )
    failures = []
    for row in group.sort_values(order).to_dict("records"):
        raw_ci = row.get("acceptance_gain_ci_low")
        try:
            ci_low = float(raw_ci)
        except (TypeError, ValueError):
            ci_low = float("nan")
        raw_clusters = row.get("gain_prompt_clusters", 0)
        try:
            clusters = int(raw_clusters)
        except (TypeError, ValueError, OverflowError):
            clusters = 0
        reasons = []
        if clusters < 2:
            reasons.append("insufficient_paired_prompt_clusters")
        if not math.isfinite(ci_low):
            reasons.append("nonfinite_acceptance_gain_ci_low")
        elif ci_low <= 0.0:
            reasons.append("acceptance_gain_ci_low_not_positive")
        if not reasons:
            continue
        if continuous_trajectory:
            location = {
                "prefix_window_start": int(row["prefix_window_start"]),
                "prefix_window_end": int(row["prefix_window_end"]),
            }
        else:
            location = {"context_length": int(row["context_length"])}
        failures.append(
            {
                **location,
                "acceptance_gain_ci_low": (
                    ci_low if math.isfinite(ci_low) else None
                ),
                "paired_prompt_clusters": clusters,
                "reasons": reasons,
            }
        )
    return bool(len(group) and not failures), failures


def _p5_checkpoint_performance_scope(engine_params: dict) -> str:
    """Expose per-context performance only under an explicit timing contract."""

    timing_contract = engine_params.get("p5_context_timing_contract")
    if timing_contract is None:
        return "mixed_workload_global"
    if timing_contract == "independent_exact_context_group_v1":
        return "checkpoint_request"
    raise exit_codes.ArtifactValidationFailure(
        f"unsupported P5 context timing contract: {timing_contract!r}"
    )


def cmd_analyze(args) -> int:
    import pandas as pd
    import pyarrow.parquet as pq

    from lightcone_spec.artifacts.rundir import RunDirectory
    from lightcone_spec.plots.figures import (
        acceptance_cost_pareto_figure,
        acceptance_shape_figure,
        long_context_acceptance_figure,
        speedup_figure,
    )
    from lightcone_spec.statistics.hypotheses import ClaimGateVerdict, harmful_rate_gate
    from lightcone_spec.statistics.tables import (
        P5_IDENTITY_COLUMNS,
        acceptance_elasticity_table,
        expand_static_p5_identities,
        long_context_acceptance_table,
        method_table,
        p5_prompt_acceptance_table,
        select_load_profiles,
    )
    from lightcone_spec.artifacts.coverage import build_coverage
    from lightcone_spec.artifacts.validator import validate_artifact_root
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    _require_manifest_for_derived_overlays(args)
    root = Path(args.artifact_root)
    if not root.is_dir():
        raise exit_codes.ArtifactValidationFailure(
            f"artifact root does not exist: {root}"
        )
    expected = None
    expected_manifest_sha256 = None
    weight_update_mode_overlay = getattr(args, "weight_update_mode", None)
    methods = getattr(args, "methods", None)
    lifecycles = getattr(args, "lifecycles", None)
    learning_rate = getattr(args, "learning_rate", None)
    manifest_path = getattr(args, "manifest", None)
    if manifest_path:
        effective_manifest = _apply_manifest_overlays(
            ExperimentManifest.load(manifest_path), args
        )
        expected = effective_manifest.expected_units()
        expected_manifest_sha256 = effective_manifest.content_sha256()
    expected_ids = _expected_unit_ids(expected)
    validation = validate_artifact_root(root, expected)
    scoped_status = _scope_unit_status(validation.unit_status, expected_ids)
    validated_run_ids = (
        set(validation.checked_runs) if expected_ids is not None else None
    )
    if expected is not None:
        incomplete = build_coverage(expected, scoped_status).missing_required()
        if incomplete:
            raise exit_codes.ArtifactValidationFailure(
                f"analysis input has incomplete required units: {incomplete}"
            )
    if not validation.ok:
        raise exit_codes.ArtifactValidationFailure(
            "analysis input failed artifact validation: " + validation.errors[0]
        )
    failed_runs = {
        unit_id: status
        for unit_id, status in scoped_status.items()
        if status in ("failed_exactness", "failed_runtime", "invalid_artifact")
    }
    if failed_runs:
        raise exit_codes.ArtifactValidationFailure(
            f"analysis input contains failed units: {failed_runs}"
        )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    p5_rounds = []
    p5_summary_frames = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if validated_run_ids is not None and run_dir.name not in validated_run_ids:
            continue
        rd = RunDirectory(root, run_dir.name)
        if not rd.is_complete:
            continue
        manifest = rd.read_manifest()
        unit_id = str(manifest.get("unit_id", run_dir.name))
        if expected_ids is not None and unit_id not in expected_ids:
            continue
        if rd.read_exit().get("status") != "complete_valid":
            continue
        t = pq.read_table(run_dir / "request_summary.parquet")
        if t.num_rows:
            frame = t.to_pandas()
            phase = str(manifest.get("phase", ""))
            continuous_windows = _continuous_prefix_windows(
                manifest.get("engine_params", {})
            )
            frame["analysis_phase"] = phase
            if phase.startswith("p5"):
                from lightcone_spec.config.schema import (
                    canonical_weight_update_mode,
                )

                frame["weight_update_mode"] = canonical_weight_update_mode(
                    manifest.get("trainable_scope", "output_residual")
                )
                run_update_stride = _p5_update_stride(manifest)
                frame["update_stride"] = run_update_stride
            frames.append(frame)
            if phase.startswith("p5"):
                model_pair = str(manifest["model_pair"])
                run_weight_update_mode = str(frame["weight_update_mode"].iloc[0])
                checkpoint_path = run_dir / "prefix-checkpoints.json"
                if not checkpoint_path.is_file():
                    raise exit_codes.ArtifactValidationFailure(
                        f"P5 run {run_dir.name} has no prefix-checkpoints.json"
                    )
                checkpoints = json.loads(checkpoint_path.read_text()).get(
                    "checkpoints", []
                )
                checkpoint_context = {}
                checkpoint_prompt_group = {}
                for row in checkpoints:
                    group = hashlib.sha256(
                        str(row["sample_id"]).encode("utf-8")
                    ).hexdigest()
                    checkpoint_context[group] = int(row["context_length"])
                    checkpoint_prompt_group[group] = hashlib.sha256(
                        str(row["source_sample_id"]).encode("utf-8")
                    ).hexdigest()
                if not checkpoint_context:
                    raise exit_codes.ArtifactValidationFailure(
                        f"P5 run {run_dir.name} has no exact context checkpoints"
                    )
                initial_contexts = sorted(set(checkpoint_context.values()))
                if continuous_windows and len(initial_contexts) != 1:
                    raise exit_codes.ArtifactValidationFailure(
                        "continuous P5 trajectories require exactly one locked "
                        f"initial prefix, got {initial_contexts}"
                    )
                if continuous_windows and continuous_windows[0][0] != initial_contexts[0]:
                    raise exit_codes.ArtifactValidationFailure(
                        "continuous P5 prefix windows must begin at the locked "
                        f"initial prefix {initial_contexts[0]}, got "
                        f"{continuous_windows[0][0]}"
                    )
                frame = frame.copy()
                frame["model_pair"] = model_pair
                frame["weight_update_mode"] = run_weight_update_mode
                frame["peak_tflops_basis"] = manifest.get(
                    "engine_params", {}
                ).get("peak_tflops_basis")
                # Request summaries replicate this run-level counter.  Keep
                # the run identity so later aggregation can take one max per
                # run and only then sum independent runs, never requests.
                frame["analysis_run_id"] = run_dir.name
                frame_context = frame["prompt_id_hash"].astype(str).str.extract(
                    r":ctx-(\d+)(?::repeat-\d+)?$", expand=False
                )
                if frame_context.isna().any():
                    raise exit_codes.ArtifactValidationFailure(
                        f"P5 run {run_dir.name} has request rows without context ids"
                    )
                frame["context_length"] = frame_context.astype(int)
                if continuous_windows:
                    initial_context = initial_contexts[0]
                    if set(frame["context_length"].astype(int)) != {initial_context}:
                        raise exit_codes.ArtifactValidationFailure(
                            "continuous P5 request summary does not match its "
                            f"locked initial prefix {initial_context}"
                        )
                    # Request-summary latency, throughput and HBM cover the one
                    # complete trajectory.  Keep exactly one performance row;
                    # duplicating it into every prefix window would manufacture
                    # context-resolved speedups that were never measured.
                    frame["context_length"] = int(continuous_windows[-1][1])
                    frame["trajectory_kind"] = "continuous_prefix"
                    frame["prefix_window_start"] = int(continuous_windows[0][0])
                    frame["prefix_window_end"] = int(continuous_windows[-1][1])
                    frame["performance_scope"] = "full_continuous_trajectory"
                    p5_summary_frames.append(frame)
                else:
                    frame["trajectory_kind"] = "independent_checkpoint"
                    frame["prefix_window_start"] = None
                    frame["prefix_window_end"] = None
                    frame["performance_scope"] = (
                        _p5_checkpoint_performance_scope(
                            manifest.get("engine_params", {})
                        )
                    )
                    p5_summary_frames.append(frame)
                request_aggregates = {}
                telemetry_paths = sorted((run_dir / "runtime").glob("*.jsonl"))
                rank_zero = [path for path in telemetry_paths if "-r0." in path.name]
                selected_telemetry_paths = rank_zero or telemetry_paths
                for telemetry_path in selected_telemetry_paths:
                    with telemetry_path.open(encoding="utf-8") as source:
                        for line in source:
                            record = json.loads(line)
                            if record.pop("kind", None) != "round":
                                continue
                            request_id = str(record.get("request_id", ""))
                            if not request_id.startswith("lightcone-g"):
                                continue
                            group_match = __import__("re").match(
                                r"lightcone-g([0-9a-f]{64})-", request_id
                            )
                            checkpoint_group = (
                                group_match.group(1) if group_match else None
                            )
                            checkpoint_length = (
                                checkpoint_context.get(checkpoint_group)
                                if checkpoint_group
                                else None
                            )
                            if checkpoint_length is None:
                                raise exit_codes.ArtifactValidationFailure(
                                    f"P5 telemetry request has no checkpoint context: "
                                    f"{request_id}"
                                )
                            if record.get("prefix_feature_exact") is not True:
                                raise exit_codes.ArtifactValidationFailure(
                                    "P5 telemetry requires an exact per-round "
                                    f"prefix feature: {request_id}"
                                )
                            if "prefix_len_before" not in record:
                                raise exit_codes.ArtifactValidationFailure(
                                    "P5 telemetry lacks prefix_len_before for "
                                    f"{request_id}"
                                )
                            prefix = int(record["prefix_len_before"])
                            if prefix < checkpoint_length:
                                raise exit_codes.ArtifactValidationFailure(
                                    "P5 proposal prefix precedes its locked "
                                    f"checkpoint: {prefix} < {checkpoint_length}"
                                )
                            window_start = None
                            window_end = None
                            if continuous_windows:
                                selected = next(
                                    (
                                        (start, end)
                                        for start, end in continuous_windows
                                        if start <= prefix < end
                                    ),
                                    None,
                                )
                                # Rounds outside declared analysis windows are
                                # valid execution evidence but deliberately do
                                # not contribute to a trajectory bucket.
                                if selected is None:
                                    continue
                                window_start, window_end = selected
                                context_length = int(window_end)
                                aggregate_key = (request_id, context_length)
                            else:
                                context_length = checkpoint_length
                                next_context = min(
                                    (
                                        value
                                        for value in checkpoint_context.values()
                                        if value > context_length
                                    ),
                                    default=None,
                                )
                                if next_context is not None and prefix >= next_context:
                                    raise exit_codes.ArtifactValidationFailure(
                                        "P5 request crossed into the next context "
                                        "bucket; split the generation window so "
                                        "acceptance and throughput stay paired "
                                        f"({prefix} >= {next_context})"
                                    )
                                aggregate_key = request_id
                            aggregate = request_aggregates.setdefault(
                                aggregate_key,
                                {
                                    "request_id": request_id,
                                    "method": manifest["method"],
                                    "model_pair": model_pair,
                                    "weight_update_mode": run_weight_update_mode,
                                    "update_stride": run_update_stride,
                                    "dataset": manifest["dataset"],
                                    "lifecycle": manifest["lifecycle"],
                                    "seed": int(manifest["seed"]),
                                    "offered_concurrency": int(
                                        manifest["concurrency"]
                                    ),
                                    "context_length": context_length,
                                    "trajectory_kind": (
                                        "continuous_prefix"
                                        if continuous_windows
                                        else "independent_checkpoint"
                                    ),
                                    "initial_prefix_len": checkpoint_length,
                                    "prefix_window_start": window_start,
                                    "prefix_window_end": window_end,
                                    "benchmark_repetitions": int(
                                        manifest.get("engine_params", {}).get(
                                            "benchmark_repetitions", 1
                                        )
                                    ),
                                    "prompt_cluster": checkpoint_prompt_group[
                                        checkpoint_group
                                    ],
                                    "round_count": 0,
                                    "semantic_round_count": 0,
                                    "accepted_sum": 0.0,
                                    "semantic_accepted_sum": 0.0,
                                    "committed_sum": 0.0,
                                    "semantic_committed_sum": 0.0,
                                    "verified_sum": 0.0,
                                    "semantic_verified_sum": 0.0,
                                    "waste_sum": 0.0,
                                    "semantic_waste_sum": 0.0,
                                    "target_calls_sum": 0.0,
                                    "semantic_target_calls_sum": 0.0,
                                    "algorithmic_censored_count": 0,
                                    "draft_cuda_us_sum": 0.0,
                                    "verify_cuda_us_sum": 0.0,
                                    "accept_cuda_us_sum": 0.0,
                                    "signal_prep_cuda_us_sum": 0.0,
                                    "signal_prep_timed_rounds": 0,
                                    "signal_prep_unknown_rounds": 0,
                                    "batch_size_sum": 0.0,
                                    "batch_reciprocal_sum": 0.0,
                                    "version_mismatch_count": 0,
                                    "observed_prefix_min": None,
                                    "observed_prefix_max": None,
                                    "draft_tokens": 0,
                                },
                            )
                            accepted = int(record.get("accepted_drafts", 0) or 0)
                            verify_len = int(
                                record.get("verify_len")
                                or int(record.get("draft_tokens", 0) or 0) + 1
                            )
                            verified = max(verify_len - 1, 0)
                            aggregate["round_count"] += 1
                            aggregate["accepted_sum"] += accepted
                            aggregate["committed_sum"] += int(
                                record.get("committed_per_verify", 0) or 0
                            )
                            aggregate["verified_sum"] += verified
                            aggregate["waste_sum"] += max(verified - accepted, 0)
                            aggregate["target_calls_sum"] += int(
                                record.get("target_calls", 0) or 0
                            )
                            censored = bool(
                                record.get("algorithmic_censored", False)
                            )
                            aggregate["algorithmic_censored_count"] += int(
                                censored
                            )
                            if not censored:
                                aggregate["semantic_round_count"] += 1
                                aggregate["semantic_accepted_sum"] += accepted
                                aggregate["semantic_committed_sum"] += int(
                                    record.get("committed_per_verify", 0) or 0
                                )
                                aggregate["semantic_verified_sum"] += verified
                                aggregate["semantic_waste_sum"] += max(
                                    verified - accepted, 0
                                )
                                aggregate["semantic_target_calls_sum"] += int(
                                    record.get("target_calls", 0) or 0
                                )
                            for name in (
                                "draft_cuda_us",
                                "verify_cuda_us",
                                "accept_cuda_us",
                                "batch_size",
                            ):
                                aggregate[f"{name}_sum"] += float(
                                    record.get(name, 0) or 0
                                )
                            signal_prep = record.get("signal_prep_cuda_us")
                            if signal_prep is None:
                                aggregate["signal_prep_unknown_rounds"] += 1
                            else:
                                try:
                                    signal_prep = float(signal_prep)
                                except (TypeError, ValueError, OverflowError) as exc:
                                    raise exit_codes.ArtifactValidationFailure(
                                        "P5 signal_prep_cuda_us must be numeric"
                                    ) from exc
                                if not math.isfinite(signal_prep) or signal_prep < 0:
                                    raise exit_codes.ArtifactValidationFailure(
                                        "P5 signal_prep_cuda_us must be finite "
                                        "and non-negative"
                                    )
                                aggregate["signal_prep_cuda_us_sum"] += signal_prep
                                aggregate["signal_prep_timed_rounds"] += 1
                            batch_size = float(record.get("batch_size", 0) or 0)
                            if batch_size <= 0:
                                raise exit_codes.ArtifactValidationFailure(
                                    "P5 telemetry batch_size must be positive"
                                )
                            # One scheduler step of size B produces B request
                            # rows.  Summing 1/B over those rows recovers one
                            # step, which keeps the reported batch mean from
                            # being biased toward large batches.
                            aggregate["batch_reciprocal_sum"] += 1.0 / batch_size
                            aggregate["version_mismatch_count"] += int(
                                not bool(record.get("version_canary_ok", False))
                            )
                            aggregate["observed_prefix_min"] = (
                                prefix
                                if aggregate["observed_prefix_min"] is None
                                else min(aggregate["observed_prefix_min"], prefix)
                            )
                            aggregate["observed_prefix_max"] = (
                                prefix
                                if aggregate["observed_prefix_max"] is None
                                else max(aggregate["observed_prefix_max"], prefix)
                            )
                            gamma = int(record.get("draft_tokens", 0) or 0)
                            aggregate["draft_tokens"] = max(
                                aggregate["draft_tokens"], gamma
                            )
                            if not censored:
                                for k in range(1, gamma + 1):
                                    name = f"semantic_survival_count_k{k}"
                                    aggregate[name] = aggregate.get(name, 0) + int(
                                        accepted >= k
                                    )
                # Update work is emitted on a different lifecycle from rounds.
                # Read it separately so inclusive candidate time and its
                # backward/optimizer breakdown are joined by the exact source
                # prefix without complicating the round hot path.  Missing
                # component fields remain explicit unknown evidence.
                update_fields = (
                    "side_queue_cuda_us",
                    "candidate_cuda_us",
                    "backward_cuda_us",
                    "optimizer_cuda_us",
                    "controller_cuda_us",
                    "publish_cuda_us",
                    "barrier_wait_cpu_us",
                )
                update_aggregates = {}
                unmapped_update_cost = False
                for telemetry_path in selected_telemetry_paths:
                    with telemetry_path.open(encoding="utf-8") as source:
                        for line in source:
                            update = json.loads(line)
                            if update.pop("kind", None) != "update":
                                continue
                            if update.get("launch_ts_us") is None:
                                continue
                            request_id = str(update.get("request_id", ""))
                            if not request_id.startswith("lightcone-g"):
                                continue
                            group_match = __import__("re").match(
                                r"lightcone-g([0-9a-f]{64})-", request_id
                            )
                            checkpoint_group = (
                                group_match.group(1) if group_match else None
                            )
                            checkpoint_length = (
                                checkpoint_context.get(checkpoint_group)
                                if checkpoint_group
                                else None
                            )
                            if checkpoint_length is None:
                                raise exit_codes.ArtifactValidationFailure(
                                    "P5 update request has no checkpoint context: "
                                    f"{request_id}"
                                )
                            source_prefix = update.get("source_prefix_len")
                            if source_prefix is None:
                                unmapped_update_cost = True
                                continue
                            try:
                                source_prefix = int(source_prefix)
                            except (TypeError, ValueError, OverflowError) as exc:
                                raise exit_codes.ArtifactValidationFailure(
                                    "P5 update source_prefix_len must be integral"
                                ) from exc
                            if source_prefix < checkpoint_length:
                                raise exit_codes.ArtifactValidationFailure(
                                    "P5 update prefix precedes its checkpoint: "
                                    f"{source_prefix} < {checkpoint_length}"
                                )
                            if continuous_windows:
                                selected = next(
                                    (
                                        (start, end)
                                        for start, end in continuous_windows
                                        if start <= source_prefix < end
                                    ),
                                    None,
                                )
                                if selected is None:
                                    continue
                                aggregate_key = (request_id, int(selected[1]))
                            else:
                                next_context = min(
                                    (
                                        value
                                        for value in checkpoint_context.values()
                                        if value > checkpoint_length
                                    ),
                                    default=None,
                                )
                                if (
                                    next_context is not None
                                    and source_prefix >= next_context
                                ):
                                    raise exit_codes.ArtifactValidationFailure(
                                        "P5 update crossed into the next context "
                                        f"bucket: {source_prefix} >= {next_context}"
                                    )
                                aggregate_key = request_id
                            evidence = update_aggregates.setdefault(
                                aggregate_key,
                                {
                                    "update_count": 0,
                                    "update_cost_evidence_complete": True,
                                    **{f"{name}_sum": 0.0 for name in update_fields},
                                },
                            )
                            evidence["update_count"] += 1
                            for name in update_fields:
                                value = update.get(name)
                                if value is None:
                                    evidence["update_cost_evidence_complete"] = False
                                    continue
                                try:
                                    value = float(value)
                                except (TypeError, ValueError, OverflowError) as exc:
                                    raise exit_codes.ArtifactValidationFailure(
                                        f"P5 update {name} must be numeric"
                                    ) from exc
                                if not math.isfinite(value) or value < 0:
                                    raise exit_codes.ArtifactValidationFailure(
                                        f"P5 update {name} must be finite and "
                                        "non-negative"
                                    )
                                evidence[f"{name}_sum"] += value

                if not request_aggregates:
                    raise exit_codes.ArtifactValidationFailure(
                        f"P5 run {run_dir.name} has no measured round telemetry"
                    )
                expected_aggregates = t.num_rows * (
                    len(continuous_windows) if continuous_windows else 1
                )
                if len(request_aggregates) != expected_aggregates:
                    raise exit_codes.ArtifactValidationFailure(
                        f"P5 run {run_dir.name} has incomplete raw telemetry: "
                        f"{len(request_aggregates)} measured request/buckets for "
                        f"{t.num_rows} request summaries and "
                        f"{len(continuous_windows) if continuous_windows else 1} "
                        "expected buckets"
                    )
                for aggregate_key, aggregate in request_aggregates.items():
                    count = max(int(aggregate["round_count"]), 1)
                    update_cost = update_aggregates.get(
                        aggregate_key,
                        {
                            "update_count": 0,
                            "update_cost_evidence_complete": True,
                            **{f"{name}_sum": 0.0 for name in update_fields},
                        },
                    )
                    aggregate.update(update_cost)
                    aggregate["update_cost_evidence_complete"] = bool(
                        update_cost["update_cost_evidence_complete"]
                        and not unmapped_update_cost
                    )
                    aggregate.update(
                        {
                            "accepted_drafts": aggregate["accepted_sum"] / count,
                            "committed_per_verify": aggregate["committed_sum"] / count,
                            "verify_len": aggregate["verified_sum"] / count + 1,
                            "target_calls": aggregate["target_calls_sum"] / count,
                            "draft_cuda_us": aggregate["draft_cuda_us_sum"] / count,
                            "verify_cuda_us": aggregate["verify_cuda_us_sum"] / count,
                            "accept_cuda_us": aggregate["accept_cuda_us_sum"] / count,
                            "batch_size": aggregate["batch_size_sum"] / count,
                            "prefix_len_before": aggregate["observed_prefix_min"],
                            "version_canary_ok": aggregate["version_mismatch_count"] == 0,
                        }
                    )
                    p5_rounds.append(aggregate)
    if not frames:
        raise exit_codes.ArtifactValidationFailure("no completed runs to analyze")
    summaries = pd.concat(frames, ignore_index=True)
    load_profiles = select_load_profiles(
        summaries, itl_slo_ms=float(args.itl_slo_ms)
    )
    (out / "load_profiles.json").write_text(
        json.dumps(load_profiles, indent=2, sort_keys=True) + "\n"
    )
    verdict = ClaimGateVerdict()
    outputs = {}
    # P5 spans several backends and update parameterizations.  Its paired,
    # identity-aware tables are emitted below; feeding those rows into the
    # legacy aggregate table would manufacture a cross-backend speedup.
    headline_summaries = summaries[
        ~summaries["analysis_phase"].astype(str).str.startswith("p5")
    ]
    for lifecycle in sorted(headline_summaries["lifecycle"].unique()):
        table = method_table(
            headline_summaries, lifecycle, baseline_method=args.baseline
        )
        if table.empty:
            continue
        csv_path = out / f"main_table_{lifecycle}.csv"
        table.to_csv(csv_path, index=False)
        outputs[lifecycle] = str(csv_path)
        if len(table) > 1:
            speedup_figure(
                table.to_dict("records"), out / f"speedup_{lifecycle}.png"
            )
    n_failed = int((summaries["status"] == "failed_exactness").sum())
    verdict.add(
        "exactness",
        n_failed == 0,
        {"failed_exactness_requests": n_failed},
    )
    if p5_rounds:
        raw_rounds = pd.DataFrame(p5_rounds)
        trajectory_kinds = set(raw_rounds["trajectory_kind"].dropna().astype(str))
        if len(trajectory_kinds) != 1:
            raise exit_codes.ArtifactValidationFailure(
                "independent-checkpoint and continuous P5 trajectories must "
                f"be analyzed in separate artifact roots, got {trajectory_kinds}"
            )
        continuous_trajectory = trajectory_kinds == {"continuous_prefix"}
        acceptance = long_context_acceptance_table(
            raw_rounds, baseline_method=args.baseline
        )
        prompt_acceptance = p5_prompt_acceptance_table(raw_rounds)
        shape = acceptance_elasticity_table(
            raw_rounds, baseline_method=args.baseline
        )
        p5_summaries = pd.concat(p5_summary_frames, ignore_index=True)
        p5_summaries = expand_static_p5_identities(p5_summaries)
        p5_summaries["exactness_violation"] = (
            p5_summaries["status"] == "failed_exactness"
        ).astype(int)
        group_keys = ["method", *P5_IDENTITY_COLUMNS, "context_length"]
        try:
            fallback_by_cell = _aggregate_p5_run_counter(
                p5_summaries,
                group_keys,
                counter="adaptation_fallback_count",
            )
        except ValueError as exc:
            raise exit_codes.ArtifactValidationFailure(
                str(exc)
            ) from exc
        performance = (
            p5_summaries.groupby(group_keys, as_index=False)
            .agg(
                decode_goodput_tps=("decode_tps", "mean"),
                peak_hbm_bytes=("peak_hbm_bytes", "max"),
                kv_retractions=("kv_retracted_requests", "max"),
                exactness_violations=("exactness_violation", "sum"),
                performance_scope=("performance_scope", "first"),
                trajectory_prefix_start=("prefix_window_start", "min"),
                trajectory_prefix_end=("prefix_window_end", "max"),
                p50_itl_ms=("p50_itl_ms", "mean"),
                p95_itl_ms=("p95_itl_ms", "mean"),
                p99_itl_ms=("p99_itl_ms", "mean"),
                estimated_tflops_per_gpu=("estimated_tflops_per_gpu", "mean"),
                estimated_mfu=("estimated_mfu", "mean"),
                peak_tflops_per_gpu=("peak_tflops_per_gpu", "first"),
                peak_tflops_basis=("peak_tflops_basis", "first"),
                decode_batch_fill_ratio=("decode_batch_fill_ratio", "mean"),
                decode_batch_size_step_mean=(
                    "decode_batch_size_step_mean",
                    "mean",
                ),
                decode_batch_size_time_mean=(
                    "decode_batch_size_time_mean",
                    "mean",
                ),
                peak_running_requests=("peak_running_requests", "max"),
                peak_queue_requests=("peak_queue_requests", "max"),
                nvml_gpu_utilization_mean=(
                    "nvml_gpu_utilization_mean",
                    "mean",
                ),
                nvml_gpu_busy_fraction_90=(
                    "nvml_gpu_busy_fraction_90",
                    "mean",
                ),
            )
            .merge(
                fallback_by_cell,
                on=group_keys,
                how="left",
                validate="one_to_one",
            )
        )
        baseline_performance = performance[performance["method"] == args.baseline][
            group_keys[1:] + ["decode_goodput_tps"]
        ].rename(columns={"decode_goodput_tps": "baseline_decode_goodput_tps"})
        performance = performance.merge(
            baseline_performance, on=group_keys[1:], how="left"
        )
        performance["throughput_speedup_vs_baseline"] = (
            performance["decode_goodput_tps"]
            / performance["baseline_decode_goodput_tps"]
        )
        if args.baseline == "static":
            performance["static_decode_goodput_tps"] = performance[
                "baseline_decode_goodput_tps"
            ]
            performance["throughput_speedup_vs_static"] = performance[
                "throughput_speedup_vs_baseline"
            ]
        if continuous_trajectory:
            performance.to_parquet(
                out / "p5_continuous_performance.parquet", index=False
            )
            performance.to_csv(
                out / "p5_continuous_performance.csv", index=False
            )
            safety_keys = group_keys[:-1]
            safety = (
                performance.groupby(safety_keys, as_index=False)
                .agg(
                    exactness_violations=("exactness_violations", "max"),
                    adaptation_fallback_count=(
                        "adaptation_fallback_count",
                        "max",
                    ),
                )
            )
            acceptance = acceptance.merge(safety, on=safety_keys, how="left")
            # Algorithmic prefix windows intentionally carry no throughput/HBM
            # value.  The separately emitted trajectory table is the only
            # valid performance evidence for this run.
            for column in (
                "decode_goodput_tps",
                "peak_hbm_bytes",
                "kv_retractions",
                "baseline_decode_goodput_tps",
                "throughput_speedup_vs_baseline",
            ):
                acceptance[column] = float("nan")
            acceptance["performance_scope"] = "prefix_window_algorithmic_only"
            if args.baseline == "static":
                acceptance["static_decode_goodput_tps"] = float("nan")
                acceptance["throughput_speedup_vs_static"] = float("nan")
        else:
            acceptance = acceptance.merge(performance, on=group_keys, how="left")
        acceptance["decode_round_cuda_us"] = (
            acceptance["draft_cuda_us"]
            + acceptance["verify_cuda_us"]
            + acceptance["accept_cuda_us"]
        )
        acceptance["round_cuda_us"] = acceptance[
            "decode_round_cuda_us"
        ] + acceptance["adaptation_cuda_us"]
        acceptance.to_parquet(out / "p5_long_context_acceptance.parquet", index=False)
        acceptance.to_csv(out / "p5_long_context_acceptance.csv", index=False)
        prompt_acceptance.to_parquet(
            out / "p5_prompt_acceptance.parquet", index=False
        )
        prompt_acceptance.to_csv(out / "p5_prompt_acceptance.csv", index=False)
        if continuous_trajectory:
            acceptance.to_parquet(
                out / "p5_continuous_trajectory.parquet", index=False
            )
            acceptance.to_csv(out / "p5_continuous_trajectory.csv", index=False)
        shape.to_parquet(out / "p5_acceptance_elasticity.parquet", index=False)
        shape.to_csv(out / "p5_acceptance_elasticity.csv", index=False)
        long_context_acceptance_figure(
            acceptance, out / "p5_acceptance_and_committed.png"
        )
        acceptance_shape_figure(shape, out / "p5_acceptance_shape.png")
        if not continuous_trajectory:
            acceptance_cost_pareto_figure(
                acceptance, out / "p5_acceptance_cost_pareto.png"
            )
        p5_gates = []
        identity = ["method", *P5_IDENTITY_COLUMNS]
        for key, group in acceptance.groupby(identity, dropna=False):
            method = key[0]
            if method == args.baseline:
                continue
            if continuous_trajectory:
                long_rows = group[
                    pd.to_numeric(
                        group["prefix_window_start"], errors="coerce"
                    ).ge(4096)
                ]
            else:
                long_rows = group[group["context_length"] >= 4096]
            shape_rows = shape[
                (shape["metric"] == "elasticity")
                & (shape["method"] == method)
                & (shape["context_left"] >= 4096)
            ]
            for name, value in zip(P5_IDENTITY_COLUMNS, key[1:]):
                shape_rows = shape_rows[shape_rows[name] == value]
            lcag_low = (
                float(long_rows["lcag_ci_low"].iloc[0])
                if len(long_rows)
                else float("nan")
            )
            delta_e = (
                float(shape_rows["delta_vs_baseline"].mean())
                if len(shape_rows)
                else float("nan")
            )
            paired_clusters = (
                int(long_rows["lcag_prompt_clusters"].iloc[0])
                if len(long_rows)
                else 0
            )
            repetitions = (
                int(group["benchmark_repetitions"].min())
                if "benchmark_repetitions" in group
                else 1
            )
            scientific_sample_pass = _p5_scientific_sample_pass(
                paired_clusters, repetitions
            )
            algorithmic = bool(
                scientific_sample_pass and lcag_low > 0 and delta_e < 0
            )
            speedup = (
                float(long_rows["throughput_speedup_vs_baseline"].mean())
                if len(long_rows)
                else float("nan")
            )
            target_gain = float(long_rows["target_calls_per_output_token"].mean())
            baseline_rows = acceptance[
                (acceptance["method"] == args.baseline)
                & (acceptance["context_length"] >= 4096)
            ]
            for name, value in zip(P5_IDENTITY_COLUMNS, key[1:]):
                baseline_rows = baseline_rows[baseline_rows[name] == value]
            baseline_target = baseline_rows[
                "target_calls_per_output_token"
            ].mean()
            adaptation_fallback_count = (
                int(long_rows["adaptation_fallback_count"].max())
                if len(long_rows)
                else 0
            )
            exact = (
                int(long_rows["version_mismatch_count"].sum()) == 0
                and int(long_rows["exactness_violations"].sum()) == 0
                and adaptation_fallback_count == 0
            )
            window_dominance, window_failures = _p5_window_dominance(
                group, continuous_trajectory=continuous_trajectory
            )
            gate = {
                **dict(zip(identity, key)),
                "baseline_method": args.baseline,
                "lcag_ci_low": lcag_low,
                "mean_delta_acceptance_elasticity": delta_e,
                "long_context_throughput_speedup": speedup,
                "paired_prompt_clusters": paired_clusters,
                "benchmark_repetitions": repetitions,
                "scientific_sample_pass": scientific_sample_pass,
                "algorithmic_pass": algorithmic and exact,
                "window_dominance_pass": window_dominance,
                "window_dominance_scope": (
                    "declared_prefix_windows"
                    if continuous_trajectory
                    else "declared_context_points"
                ),
                "declared_window_count": int(len(group)),
                "window_dominance_failures": window_failures,
                "engineering_pass": bool(
                    algorithmic
                    and exact
                    and not continuous_trajectory
                    and speedup >= 1.0
                    and target_gain < baseline_target
                ),
                "engineering_evidence": (
                    "trajectory_level_only"
                    if continuous_trajectory
                    else "context_resolved"
                ),
                "exactness_pass": exact,
                "adaptation_fallback_count": adaptation_fallback_count,
            }
            for field in (
                "lcag_ci_low",
                "mean_delta_acceptance_elasticity",
                "long_context_throughput_speedup",
            ):
                if not math.isfinite(gate[field]):
                    gate[field] = None
            p5_gates.append(gate)
        (out / "p5_claim_gates.json").write_text(
            json.dumps(p5_gates, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        outputs["p5"] = str(out / "p5_long_context_acceptance.parquet")
        outputs["p5_prompt"] = str(out / "p5_prompt_acceptance.parquet")
        if continuous_trajectory:
            outputs["p5_continuous"] = str(
                out / "p5_continuous_trajectory.parquet"
            )
            outputs["p5_continuous_performance"] = str(
                out / "p5_continuous_performance.parquet"
            )
    harm_path = root / "replay_report.json"
    if harm_path.is_file():
        harm = json.loads(harm_path.read_text())
        verdict.gates["h1"] = harm.get("h1", {})
        verdict.gates["harmful_rate"] = harmful_rate_gate(harm.get("harmful_rate", 0.0))
    (out / "claim_gates.json").write_text(
        json.dumps(verdict.to_dict(), indent=2, sort_keys=True)
    )
    analysis_manifest, analysis_hashes = _write_analysis_provenance(
        root,
        out,
        baseline=args.baseline,
        itl_slo_ms=float(args.itl_slo_ms),
        run_status=scoped_status,
        expected_manifest_sha256=expected_manifest_sha256,
        weight_update_mode=weight_update_mode_overlay,
        methods=methods,
        lifecycles=lifecycles,
        learning_rate=learning_rate,
        validated_run_ids=validated_run_ids,
    )
    print(verdict.human_readable())
    print(
        json.dumps(
            {
                "tables": outputs,
                "load_profiles": str(out / "load_profiles.json"),
                "analysis_manifest": str(analysis_manifest),
                "analysis_hashes": str(analysis_hashes),
            },
            indent=2,
        )
    )
    return exit_codes.SUCCESS


def cmd_validate_artifacts(args) -> int:
    from lightcone_spec.artifacts.coverage import build_coverage
    from lightcone_spec.artifacts.validator import validate_artifact_root
    from lightcone_spec.orchestration.manifest import ExperimentManifest

    _require_manifest_for_derived_overlays(args)
    expected = None
    if args.manifest:
        expected = _apply_manifest_overlays(
            ExperimentManifest.load(args.manifest), args
        ).expected_units()
    report = validate_artifact_root(args.artifact_root, expected)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if expected is not None:
        expected_ids = _expected_unit_ids(expected)
        cov = build_coverage(
            expected,
            _scope_unit_status(report.unit_status, expected_ids),
        )
        if args.coverage_output:
            cov.write(args.coverage_output)
        missing = cov.missing_required()
        if missing:
            print(f"incomplete required units: {missing}", file=sys.stderr)
            return exit_codes.INCOMPLETE_COVERAGE
    if not report.ok:
        return exit_codes.ARTIFACT_VALIDATION_FAILURE
    return exit_codes.SUCCESS


COMMANDS = {
    "lock": cmd_lock,
    "doctor": cmd_doctor,
    "prepare-models": cmd_prepare_models,
    "prepare-datasets": cmd_prepare_datasets,
    "serve": cmd_serve,
    "exactness": cmd_exactness,
    "replay": cmd_replay,
    "run-manifest": cmd_run_manifest,
    "analyze": cmd_analyze,
    "validate-artifacts": cmd_validate_artifacts,
}


def main(argv: list[str] | None = None) -> int:
    # Run before any command-specific module can import OpenMP-backed native
    # libraries. Some managed GPU images export the invalid value ``0``.
    from lightcone_spec.doctor import configure_runtime_threads

    configure_runtime_threads()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = COMMANDS[args.command](args)
    except LightconeError as exc:
        print(f"error ({exit_codes.meaning(exc.exit_code)}): {exc.message}",
              file=sys.stderr)
        return exc.exit_code
    return code


if __name__ == "__main__":
    sys.exit(main())
