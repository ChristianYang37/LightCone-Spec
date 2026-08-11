"""Render matched method configs and argv-only SGLang launch plans."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.config.schema import (
    AdaptationConfig,
    ModelPair,
    OnlineSpecConfig,
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
)
from lightcone_spec.execution import ControlledExecutionPolicy
from lightcone_spec.experiments.onlinespec import (
    OnlineSpecCandidate,
    OnlineSpecSelection,
)
from lightcone_spec.experiments.protocol import (
    DFLASH_BLOCK_SIZE,
    DFLASH_LOSS_POSITION_DECAY,
    TuningCandidate,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.selection import SelectionArtifact
from lightcone_spec.locking.models import ModelLock
from lightcone_spec.sglang_bridge.checkout import verify_patched_checkout
from lightcone_spec.sglang_bridge.config import sglang_adaptation_payload


@dataclass(frozen=True)
class ServerLaunch:
    method: str
    base_url: str
    exclusive_device: bool
    run_config: str
    adaptation_config: str | None
    telemetry_path: str | None
    argv: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeChoice:
    phase: str
    candidate: TuningCandidate | None
    selected_concurrency: int
    model_lock_sha256: str
    selection_sha256: str | None = None

    @property
    def sha256(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(body).hexdigest()


def _immutable_json(path: Path, value: object) -> None:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != body:
        raise ValueError(f"refusing to overwrite immutable runtime file {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    sidecar = Path(f"{path}.sha256")
    if sidecar.exists() and sidecar.read_text(encoding="utf-8").strip() != digest:
        raise ValueError(f"runtime sidecar does not match {path}")
    sidecar.write_text(digest + "\n", encoding="utf-8")


def _execution_argv(
    runtime: RuntimeConfig, *, role: Literal["target_reference", "speculative"]
) -> list[str]:
    """Render the server controls attested by every formal runtime."""
    policy = ControlledExecutionPolicy(
        context_length=runtime.context_length,
        random_seed=runtime.random_seed,
        disable_radix_cache=runtime.disable_radix_cache,
        disable_cuda_graph=runtime.disable_cuda_graph,
        target_reference_disable_overlap_schedule=(
            runtime.target_reference_disable_overlap_schedule
        ),
        speculative_disable_overlap_schedule=(
            runtime.speculative_disable_overlap_schedule
        ),
        enable_deterministic_inference=runtime.enable_deterministic_inference,
        incremental_streaming_output=runtime.incremental_streaming_output,
    )
    if runtime.execution_policy_sha256 != policy.sha256:
        raise ValueError("runtime execution-policy identity mismatch")
    argv = [
        "--context-length",
        str(policy.context_length),
        "--random-seed",
        str(policy.random_seed),
    ]
    if policy.disable_radix_cache:
        argv.append("--disable-radix-cache")
    if policy.disable_cuda_graph:
        argv.append("--disable-cuda-graph")
    if policy.overlap_disabled(role=role):
        argv.append("--disable-overlap-schedule")
    if policy.enable_deterministic_inference:
        argv.append("--enable-deterministic-inference")
    if policy.incremental_streaming_output:
        argv.append("--incremental-streaming-output")
    return argv


def _render_server(
    *,
    output: Path,
    method: str,
    config: RunConfig,
    verified_checkout: Path,
    roots: dict[str, str],
    target_id: str,
    drafter_id: str,
    adaptation_reserve_mb: int,
    mem_fraction_static: float,
    host: str,
    port: int,
) -> ServerLaunch:
    method_root = output / method
    config_path = method_root / "run-config.json"
    _immutable_json(config_path, config.model_dump(mode="json"))
    adaptation_path = None
    telemetry_path = None
    if method != "static":
        adaptation_path = method_root / "adaptation-config.json"
        payload = sglang_adaptation_payload(config)
        if payload is None:
            raise AssertionError("adapted config produced no payload")
        _immutable_json(adaptation_path, payload)
        telemetry_path = method_root / "adaptation-telemetry.json"
    argv = [
        sys.executable,
        "-m",
        "lightcone_spec.sglang_bridge.launch",
        "--checkout",
        str(verified_checkout),
        "--",
        "--model-path",
        str(Path(roots[target_id]).resolve()),
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        str(Path(roots[drafter_id]).resolve()),
        "--speculative-num-draft-tokens",
        "16",
        "--speculative-draft-window-size",
        "16",
        "--speculative-accept-threshold-single",
        "1.0",
        "--speculative-accept-threshold-acc",
        "1.0",
        "--speculative-use-rejection-sampling",
        "--speculative-speed-study-metrics",
        "--max-running-requests",
        str(config.runtime.max_running_requests),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--tp-size",
        "1",
        "--host",
        host,
        "--port",
        str(port),
    ]
    argv.extend(_execution_argv(config.runtime, role="speculative"))
    if adaptation_path is not None and telemetry_path is not None:
        argv.extend(
            (
                "--speculative-adaptation-config",
                str(adaptation_path),
                "--speculative-adaptation-reserve-mb",
                str(adaptation_reserve_mb),
                "--speculative-adaptation-telemetry-path",
                str(telemetry_path),
            )
        )
    return ServerLaunch(
        method=method,
        base_url=f"http://{host}:{port}",
        exclusive_device=True,
        run_config=str(config_path),
        adaptation_config=None if adaptation_path is None else str(adaptation_path),
        telemetry_path=None if telemetry_path is None else str(telemetry_path),
        argv=tuple(argv),
    )


def _render_choice_plan(
    *,
    output_root: str | Path,
    choice: RuntimeChoice,
    model_lock: ModelLock,
    model_roots: dict,
    sampling_profile: SamplingProfile,
    sglang_checkout: str | Path,
    adaptation_group_id: str | None,
    adaptation_reserve_mb: int,
    mem_fraction_static: float,
    include_static: bool = True,
    host: str = "127.0.0.1",
    first_port: int = 30000,
) -> tuple[ServerLaunch, ...]:
    model_lock.validate()
    sampling_profile.validate()
    verified_checkout = verify_patched_checkout(sglang_checkout)
    if choice.model_lock_sha256 != model_lock.sha256:
        raise ValueError("runtime choice and model lock identities differ")
    if choice.phase not in {
        "static_load_screen",
        "shared_config_tuning",
        "controlled_confirmation",
        "natural_task_replication",
        "independent_profiler",
    }:
        raise ValueError("runtime choice phase is invalid")
    if choice.selected_concurrency < 1:
        raise ValueError("runtime concurrency must be positive")
    static_only = choice.phase == "static_load_screen"
    if not include_static and choice.phase != "shared_config_tuning":
        raise ValueError("only tuning plans may omit their shared Static baseline")
    if static_only != (choice.candidate is None):
        raise ValueError(
            "Static load runtime must be the only phase without a tuning candidate"
        )
    if static_only:
        if adaptation_group_id is not None or adaptation_reserve_mb != 0:
            raise ValueError(
                "Static load runtime forbids adaptation identity and HBM reserve"
            )
    elif not adaptation_group_id:
        raise ValueError("adaptation_group_id must be non-empty")
    elif adaptation_reserve_mb <= 0:
        raise ValueError("adaptation HBM reserve must be explicit and positive")
    if not 0.0 < mem_fraction_static < 1.0:
        raise ValueError("mem_fraction_static must be in (0, 1)")
    if not 1 <= first_port <= 65535:
        raise ValueError("first_port must be a valid TCP port")
    if model_roots.get("schema_version") != 2:
        raise ValueError("model roots must use schema version 2")
    if model_roots.get("lock_sha256") != model_lock.sha256:
        raise ValueError("model roots belong to a different model lock")
    roots = model_roots.get("roots")
    if not isinstance(roots, dict):
        raise TypeError("model roots mapping is missing")
    target_id = "Qwen/Qwen3-8B"
    drafter_id = "z-lab/Qwen3-8B-DFlash-b16"
    revisions = {model.model_id: model.revision for model in model_lock.models}
    if target_id not in revisions or drafter_id not in revisions:
        raise ValueError("model lock lacks the formal Qwen3-8B/DFlash pair")
    for model_id in (target_id, drafter_id):
        root = roots.get(model_id)
        if not isinstance(root, str) or not Path(root).is_dir():
            raise ValueError(f"verified local model root is missing: {model_id}")
    model = ModelPair(
        target=target_id,
        drafter=drafter_id,
        target_revision=revisions[target_id],
        drafter_revision=revisions[drafter_id],
    )
    runtime = RuntimeConfig(
        sampling_profile_sha256=sampling_profile.sha256,
        tensor_parallel_size=1,
        speculative_num_draft_tokens=16,
        max_running_requests=choice.selected_concurrency,
        telemetry_detail=(
            "profile" if choice.phase == "independent_profiler" else "headline"
        ),
    )
    selected = choice.candidate
    adaptation = None
    if selected is not None:
        optimizer = OptimizerConfig(
            name=selected.optimizer,
            learning_rate=selected.learning_rate,
            weight_decay=selected.weight_decay,
            beta1=selected.beta1,
            beta2=selected.beta2,
            grad_clip=selected.grad_clip,
            momentum=selected.momentum,
            muon_ns_steps=selected.muon_ns_steps,
            muon_auxiliary_learning_rate=(selected.muon_auxiliary_learning_rate),
            muon_auxiliary_weight_decay=(selected.muon_auxiliary_weight_decay),
        )
        adaptation = AdaptationConfig(
            weight_update_mode=selected.weight_update_mode,
            parameter_scope=selected.parameter_scope,
            adaptation_group_id=str(adaptation_group_id),
            optimizer=optimizer,
            rank=selected.rank,
            stride=selected.stride,
            canvas_tokens=DFLASH_BLOCK_SIZE,
            loss_position_decay=DFLASH_LOSS_POSITION_DECAY,
        )
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    launches: list[ServerLaunch] = []
    methods = (
        ("static",)
        if static_only
        else (
            ("static", "tts", "naive_async")
            if include_static
            else ("tts", "naive_async")
        )
    )
    for method in methods:
        config = RunConfig(
            method=method,
            model=model,
            runtime=runtime,
            adaptation=None if method == "static" else adaptation,
        )
        launches.append(
            _render_server(
                output=output,
                method=method,
                config=config,
                verified_checkout=verified_checkout,
                roots=roots,
                target_id=target_id,
                drafter_id=drafter_id,
                adaptation_reserve_mb=adaptation_reserve_mb,
                mem_fraction_static=mem_fraction_static,
                host=host,
                port=first_port,
            )
        )
    _immutable_json(
        output / "launch-plan.json",
        {
            "schema_version": 2,
            "execution_mode": "sequential_exclusive_device",
            "phase": choice.phase,
            "runtime_choice_sha256": choice.sha256,
            "selection_sha256": choice.selection_sha256,
            "model_lock_sha256": model_lock.sha256,
            "sampling_profile_sha256": sampling_profile.sha256,
            "execution_policy_sha256": runtime.execution_policy_sha256,
            "patched_sglang_tree": PINNED_SGLANG_TREE,
            "sglang_checkout": str(verified_checkout),
            "servers": [asdict(launch) for launch in launches],
        },
    )
    return tuple(launches)


def render_static_load_runtime_plan(
    *,
    output_root: str | Path,
    concurrency: int,
    model_lock: ModelLock,
    model_roots: dict,
    sampling_profile: SamplingProfile,
    sglang_checkout: str | Path,
    mem_fraction_static: float,
    host: str = "127.0.0.1",
    first_port: int = 30000,
) -> tuple[ServerLaunch, ...]:
    """Render one native Static endpoint with no adaptation allocation."""
    return _render_choice_plan(
        output_root=output_root,
        choice=RuntimeChoice(
            phase="static_load_screen",
            candidate=None,
            selected_concurrency=concurrency,
            model_lock_sha256=model_lock.sha256,
        ),
        model_lock=model_lock,
        model_roots=model_roots,
        sampling_profile=sampling_profile,
        sglang_checkout=sglang_checkout,
        adaptation_group_id=None,
        adaptation_reserve_mb=0,
        mem_fraction_static=mem_fraction_static,
        host=host,
        first_port=first_port,
    )


def render_target_runtime_plan(
    *,
    output_root: str | Path,
    concurrency: int,
    model_lock: ModelLock,
    model_roots: dict,
    sampling_profile: SamplingProfile,
    sglang_checkout: str | Path,
    mem_fraction_static: float,
    host: str = "127.0.0.1",
    first_port: int = 30000,
) -> tuple[ServerLaunch, ...]:
    """Render the target-only endpoint used by token-exactness gates."""
    model_lock.validate()
    sampling_profile.validate()
    if sampling_profile.purpose != "controlled":
        raise ValueError("target reference requires controlled greedy sampling")
    if concurrency < 1:
        raise ValueError("target reference concurrency must be positive")
    if not 0.0 < mem_fraction_static < 1.0:
        raise ValueError("mem_fraction_static must be in (0, 1)")
    if not 1 <= first_port <= 65535:
        raise ValueError("first_port must be a valid TCP port")
    verified_checkout = verify_patched_checkout(sglang_checkout)
    model, roots = _locked_dflash_pair(model_lock, model_roots)
    runtime = RuntimeConfig(
        sampling_profile_sha256=sampling_profile.sha256,
        tensor_parallel_size=1,
        speculative_num_draft_tokens=16,
        max_running_requests=concurrency,
        telemetry_detail="headline",
    )
    output = Path(output_root).resolve()
    method = "target_only"
    method_root = output / method
    config_path = method_root / "run-config.json"
    _immutable_json(
        config_path,
        {
            "schema_version": 2,
            "purpose": "greedy_target_reference",
            "target_model": model.target,
            "target_revision": model.target_revision,
            "runtime": runtime.model_dump(mode="json"),
            "mem_fraction_static": mem_fraction_static,
        },
    )
    argv = [
        sys.executable,
        "-m",
        "lightcone_spec.sglang_bridge.launch",
        "--checkout",
        str(verified_checkout),
        "--",
        "--model-path",
        str(Path(roots[model.target]).resolve()),
        "--max-running-requests",
        str(concurrency),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--tp-size",
        "1",
        "--host",
        host,
        "--port",
        str(first_port),
    ]
    argv.extend(_execution_argv(runtime, role="target_reference"))
    launch = ServerLaunch(
        method=method,
        base_url=f"http://{host}:{first_port}",
        exclusive_device=True,
        run_config=str(config_path),
        adaptation_config=None,
        telemetry_path=None,
        argv=tuple(argv),
    )
    _immutable_json(
        output / "launch-plan.json",
        {
            "schema_version": 2,
            "execution_mode": "sequential_exclusive_device",
            "phase": "greedy_target_reference",
            "model_lock_sha256": model_lock.sha256,
            "sampling_profile_sha256": sampling_profile.sha256,
            "execution_policy_sha256": runtime.execution_policy_sha256,
            "patched_sglang_tree": PINNED_SGLANG_TREE,
            "sglang_checkout": str(verified_checkout),
            "servers": [asdict(launch)],
        },
    )
    return (launch,)


def render_runtime_plan(
    *,
    output_root: str | Path,
    selection: SelectionArtifact,
    model_lock: ModelLock,
    model_roots: dict,
    sampling_profile: SamplingProfile,
    sglang_checkout: str | Path,
    adaptation_group_id: str,
    adaptation_reserve_mb: int,
    mem_fraction_static: float,
    host: str = "127.0.0.1",
    first_port: int = 30000,
) -> tuple[ServerLaunch, ...]:
    selection.validate()
    return _render_choice_plan(
        output_root=output_root,
        choice=RuntimeChoice(
            phase="controlled_confirmation",
            candidate=selection.candidate,
            selected_concurrency=selection.selected_concurrency,
            model_lock_sha256=selection.model_lock_sha256,
            selection_sha256=selection.sha256,
        ),
        model_lock=model_lock,
        model_roots=model_roots,
        sampling_profile=sampling_profile,
        sglang_checkout=sglang_checkout,
        adaptation_group_id=adaptation_group_id,
        adaptation_reserve_mb=adaptation_reserve_mb,
        mem_fraction_static=mem_fraction_static,
        host=host,
        first_port=first_port,
    )


def render_tuning_runtime_plan(
    *,
    output_root: str | Path,
    candidate: TuningCandidate,
    concurrency: int,
    model_lock: ModelLock,
    model_roots: dict,
    sampling_profile: SamplingProfile,
    sglang_checkout: str | Path,
    adaptation_group_id: str,
    adaptation_reserve_mb: int,
    mem_fraction_static: float,
    host: str = "127.0.0.1",
    first_port: int = 30000,
) -> tuple[ServerLaunch, ...]:
    return _render_choice_plan(
        output_root=output_root,
        choice=RuntimeChoice(
            phase="shared_config_tuning",
            candidate=candidate,
            selected_concurrency=concurrency,
            model_lock_sha256=model_lock.sha256,
        ),
        model_lock=model_lock,
        model_roots=model_roots,
        sampling_profile=sampling_profile,
        sglang_checkout=sglang_checkout,
        adaptation_group_id=adaptation_group_id,
        adaptation_reserve_mb=adaptation_reserve_mb,
        mem_fraction_static=mem_fraction_static,
        include_static=False,
        host=host,
        first_port=first_port,
    )


def render_replication_runtime_plan(
    *,
    output_root: str | Path,
    selection: SelectionArtifact,
    model_lock: ModelLock,
    model_roots: dict,
    sampling_profile: SamplingProfile,
    sglang_checkout: str | Path,
    adaptation_group_id: str,
    adaptation_reserve_mb: int,
    mem_fraction_static: float,
    phase: str,
    host: str = "127.0.0.1",
    first_port: int = 30000,
) -> tuple[ServerLaunch, ...]:
    selection.validate()
    if phase not in {"natural_task_replication", "independent_profiler"}:
        raise ValueError("replication runtime phase is invalid")
    return _render_choice_plan(
        output_root=output_root,
        choice=RuntimeChoice(
            phase=phase,
            candidate=selection.candidate,
            selected_concurrency=selection.selected_concurrency,
            model_lock_sha256=selection.model_lock_sha256,
            selection_sha256=selection.sha256,
        ),
        model_lock=model_lock,
        model_roots=model_roots,
        sampling_profile=sampling_profile,
        sglang_checkout=sglang_checkout,
        adaptation_group_id=adaptation_group_id,
        adaptation_reserve_mb=adaptation_reserve_mb,
        mem_fraction_static=mem_fraction_static,
        host=host,
        first_port=first_port,
    )


def _locked_dflash_pair(
    model_lock: ModelLock,
    model_roots: dict,
) -> tuple[ModelPair, dict[str, str]]:
    if model_roots.get("schema_version") != 2:
        raise ValueError("model roots must use schema version 2")
    if model_roots.get("lock_sha256") != model_lock.sha256:
        raise ValueError("model roots belong to a different model lock")
    roots = model_roots.get("roots")
    if not isinstance(roots, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in roots.items()
    ):
        raise TypeError("model roots mapping is missing or malformed")
    target_id = "Qwen/Qwen3-8B"
    drafter_id = "z-lab/Qwen3-8B-DFlash-b16"
    revisions = {model.model_id: model.revision for model in model_lock.models}
    for model_id in (target_id, drafter_id):
        root = roots.get(model_id)
        if model_id not in revisions or root is None or not Path(root).is_dir():
            raise ValueError(f"verified local model root is missing: {model_id}")
    return (
        ModelPair(
            target=target_id,
            drafter=drafter_id,
            target_revision=revisions[target_id],
            drafter_revision=revisions[drafter_id],
        ),
        roots,
    )


def _onlinespec_run_config(
    candidate: OnlineSpecCandidate,
    *,
    model: ModelPair,
    sampling_profile: SamplingProfile,
    concurrency: int,
    adaptation_group_id: str,
) -> RunConfig:
    candidate.validate()
    return RunConfig(
        method=candidate.method,
        model=model,
        runtime=RuntimeConfig(
            sampling_profile_sha256=sampling_profile.sha256,
            tensor_parallel_size=1,
            speculative_num_draft_tokens=16,
            max_running_requests=concurrency,
            telemetry_detail="headline",
        ),
        adaptation=AdaptationConfig(
            weight_update_mode=candidate.weight_update_mode,
            parameter_scope=candidate.parameter_scope,
            adaptation_group_id=adaptation_group_id,
            optimizer=OptimizerConfig(
                name="sgd",
                learning_rate=candidate.learning_rate,
                weight_decay=0.0,
                grad_clip=candidate.grad_clip,
            ),
            rank=candidate.rank,
            stride=candidate.stride,
            canvas_tokens=DFLASH_BLOCK_SIZE,
            loss_position_decay=DFLASH_LOSS_POSITION_DECAY,
        ),
        online_spec=OnlineSpecConfig(
            projection_radius=candidate.projection_radius,
            additional_learning_rates=candidate.additional_learning_rates,
            hedge_learning_rate=candidate.hedge_learning_rate,
        ),
    )


def render_onlinespec_tuning_runtime_plan(
    *,
    output_root: str | Path,
    candidate: OnlineSpecCandidate,
    concurrency: int,
    model_lock: ModelLock,
    model_roots: dict,
    sampling_profile: SamplingProfile,
    sglang_checkout: str | Path,
    adaptation_group_id: str,
    adaptation_reserve_mb: int,
    mem_fraction_static: float,
    host: str = "127.0.0.1",
    first_port: int = 30000,
) -> tuple[ServerLaunch, ...]:
    """Render one paired Static/candidate OnlineSPEC tuning plan."""
    candidate.validate()
    model_lock.validate()
    sampling_profile.validate()
    if sampling_profile.purpose != "controlled":
        raise ValueError("OnlineSPEC tuning requires the controlled profile")
    if concurrency not in {1, 2, 4, 8, 16, 32, 48}:
        raise ValueError("OnlineSPEC tuning load is outside the registered grid")
    if not adaptation_group_id or adaptation_reserve_mb <= 0:
        raise ValueError("OnlineSPEC tuning requires a cohort and HBM reserve")
    if not 0.0 < mem_fraction_static < 1.0:
        raise ValueError("mem_fraction_static must be in (0, 1)")
    if not 1 <= first_port <= 65535:
        raise ValueError("first_port must be a valid TCP port")
    verified_checkout = verify_patched_checkout(sglang_checkout)
    model, roots = _locked_dflash_pair(model_lock, model_roots)
    runtime = RuntimeConfig(
        sampling_profile_sha256=sampling_profile.sha256,
        tensor_parallel_size=1,
        speculative_num_draft_tokens=16,
        max_running_requests=concurrency,
        telemetry_detail="headline",
    )
    configs = {
        "static": RunConfig(method="static", model=model, runtime=runtime),
        candidate.method: _onlinespec_run_config(
            candidate,
            model=model,
            sampling_profile=sampling_profile,
            concurrency=concurrency,
            adaptation_group_id=adaptation_group_id,
        ),
    }
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    launches = tuple(
        _render_server(
            output=output,
            method=method,
            config=configs[method],
            verified_checkout=verified_checkout,
            roots=roots,
            target_id=model.target,
            drafter_id=model.drafter,
            adaptation_reserve_mb=(
                0 if method == "static" else adaptation_reserve_mb
            ),
            mem_fraction_static=mem_fraction_static,
            host=host,
            port=first_port,
        )
        for method in ("static", candidate.method)
    )
    _immutable_json(
        output / "launch-plan.json",
        {
            "schema_version": 2,
            "execution_mode": "sequential_exclusive_device",
            "phase": "onlinespec_tuning",
            "candidate_id": candidate.candidate_id,
            "model_lock_sha256": model_lock.sha256,
            "sampling_profile_sha256": sampling_profile.sha256,
            "execution_policy_sha256": runtime.execution_policy_sha256,
            "patched_sglang_tree": PINNED_SGLANG_TREE,
            "sglang_checkout": str(verified_checkout),
            "servers": [asdict(launch) for launch in launches],
        },
    )
    return launches


def render_onlinespec_runtime_plan(
    *,
    output_root: str | Path,
    selection: OnlineSpecSelection,
    model_lock: ModelLock,
    model_roots: dict,
    sampling_profile: SamplingProfile,
    sglang_checkout: str | Path,
    adaptation_group_id: str,
    adaptation_reserve_mb: int,
    mem_fraction_static: float,
    host: str = "127.0.0.1",
    first_port: int = 30000,
) -> tuple[ServerLaunch, ...]:
    """Render one sequential Static/OnlineSPEC paired baseline plan."""
    selection.validate()
    model_lock.validate()
    sampling_profile.validate()
    if sampling_profile.purpose != "controlled":
        raise ValueError("OnlineSPEC confirmation requires the controlled profile")
    verified_checkout = verify_patched_checkout(sglang_checkout)
    if selection.model_lock_sha256 != model_lock.sha256:
        raise ValueError("OnlineSPEC selection and model lock identities differ")
    if selection.sampling_profile_sha256 != sampling_profile.sha256:
        raise ValueError("OnlineSPEC selection and sampling profile differ")
    if not adaptation_group_id or adaptation_reserve_mb <= 0:
        raise ValueError("OnlineSPEC requires a cohort id and positive HBM reserve")
    if not 0.0 < mem_fraction_static < 1.0:
        raise ValueError("mem_fraction_static must be in (0, 1)")
    if not 1 <= first_port <= 65535:
        raise ValueError("first_port must be a valid TCP port")
    model, roots = _locked_dflash_pair(model_lock, model_roots)
    runtime = RuntimeConfig(
        sampling_profile_sha256=sampling_profile.sha256,
        tensor_parallel_size=1,
        speculative_num_draft_tokens=16,
        max_running_requests=selection.selected_concurrency,
        telemetry_detail="headline",
    )
    configs = {
        "static": RunConfig(method="static", model=model, runtime=runtime),
    }
    for candidate in selection.selected:
        configs[candidate.method] = _onlinespec_run_config(
            candidate,
            model=model,
            sampling_profile=sampling_profile,
            concurrency=selection.selected_concurrency,
            adaptation_group_id=adaptation_group_id,
        )
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    launches = tuple(
        _render_server(
            output=output,
            method=method,
            config=configs[method],
            verified_checkout=verified_checkout,
            roots=roots,
            target_id=model.target,
            drafter_id=model.drafter,
            adaptation_reserve_mb=(0 if method == "static" else adaptation_reserve_mb),
            mem_fraction_static=mem_fraction_static,
            host=host,
            port=first_port,
        )
        for method in (
            "static",
            "onlinespec_ogd",
            "onlinespec_opt",
            "onlinespec_ens",
        )
    )
    _immutable_json(
        output / "launch-plan.json",
        {
            "schema_version": 2,
            "execution_mode": "sequential_exclusive_device",
            "phase": "onlinespec_paired_confirmation",
            "selection_sha256": selection.sha256,
            "model_lock_sha256": model_lock.sha256,
            "sampling_profile_sha256": sampling_profile.sha256,
            "execution_policy_sha256": runtime.execution_policy_sha256,
            "patched_sglang_tree": PINNED_SGLANG_TREE,
            "sglang_checkout": str(verified_checkout),
            "servers": [asdict(launch) for launch in launches],
        },
    )
    return launches
