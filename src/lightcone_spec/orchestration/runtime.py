"""Render matched method configs and argv-only SGLang launch plans."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.config.schema import (
    AdaptationConfig,
    ModelPair,
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
)
from lightcone_spec.experiments.protocol import TuningCandidate
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
        body = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode()
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
            grad_clip=selected.grad_clip,
        )
        adaptation = AdaptationConfig(
            weight_update_mode=selected.weight_update_mode,
            parameter_scope=selected.parameter_scope,
            adaptation_group_id=str(adaptation_group_id),
            optimizer=optimizer,
            rank=selected.rank,
            stride=selected.stride,
            canvas_tokens=16,
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
        # Formal methods are restarted sequentially on one exclusive device.
        # Sharing one port makes simultaneous launch fail immediately instead
        # of silently splitting HBM and changing the KV capacity.
        port = first_port
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
            str(choice.selected_concurrency),
            "--mem-fraction-static",
            str(mem_fraction_static),
            "--tp-size",
            "1",
            "--host",
            host,
            "--port",
            str(port),
        ]
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
        launches.append(
            ServerLaunch(
                method=method,
                base_url=f"http://{host}:{port}",
                exclusive_device=True,
                run_config=str(config_path),
                adaptation_config=(
                    None if adaptation_path is None else str(adaptation_path)
                ),
                telemetry_path=(
                    None if telemetry_path is None else str(telemetry_path)
                ),
                argv=tuple(argv),
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
