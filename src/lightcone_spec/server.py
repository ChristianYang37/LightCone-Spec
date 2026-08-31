"""Direct SGLang process lifecycle for one experiment cell."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .client import ScheduledRun, SGLangClient
from .config import ExperimentConfig
from .protocol import FORMAL_ADAPTATION_STRIDE, Job, uses_formal_adaptation_stride

ADAPTIVE_METHODS = {
    "tts",
    "tts_lora_batched",
    "l0_naive",
    "lightcone",
    "lightcone_candidate",
    "onlinespec_candidate",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
}

E2_MINIMUM_UPDATES = (2, 4, 8, 16)
COHORT_TELEMETRY_ROUND_ITEMS = 3_000_000


def _topology(job: Job) -> tuple[int, int]:
    topology = str(job.parameters.get("topology", "tp1_dp1"))
    tp = 2 if topology == "tp2_dp1" or job.node.startswith("E6") else 1
    return tp, 1


def _speculative_canvas(job: Job) -> int:
    return 8 if job.backend == "DSPARK" else int(job.width or 16)


def _request_scoped_adaptation(job: Job) -> bool:
    return job.method in {"tts", "l0_naive"}


def _telemetry_round_items(job: Job) -> int:
    if _request_scoped_adaptation(job):
        return int(job.context or 40960)
    return COHORT_TELEMETRY_ROUND_ITEMS


def adaptation_payload(job: Job, selection: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if job.method not in ADAPTIVE_METHODS:
        return None
    chosen = dict(selection or {})
    chosen.update({name: value for name, value in job.parameters.items() if value is not None})
    if job.method in {"tts", "l0_naive"}:
        chosen.update(
            optimizer="adam",
            weight_decay=0.0,
            grad_clip=0.0,
            parameterization="full",
            scope="all",
        )
    if job.method == "tts_lora_batched":
        chosen.update(
            optimizer="adam",
            weight_decay=0.0,
            grad_clip=0.0,
            parameterization="lora",
        )
    method = {
        "tts": "tts",
        "tts_lora_batched": "tts",
        "l0_naive": "l0",
        "lightcone": "l0",
        "lightcone_candidate": "l0",
        "onlinespec_candidate": "onlinespec_ogd",
    }.get(job.method, job.method)
    rank = chosen.get("rank", 8)
    parameterization = chosen.get("parameterization", "lora")
    if parameterization == "none":
        return None
    if uses_formal_adaptation_stride(job):
        chosen["stride"] = FORMAL_ADAPTATION_STRIDE
    stride = int(chosen.get("stride", FORMAL_ADAPTATION_STRIDE))
    if uses_formal_adaptation_stride(job) and stride != FORMAL_ADAPTATION_STRIDE:
        raise ValueError("formal adaptive jobs must resolve to stride S=10")
    coalescing = int(chosen.get("coalescing", 1))
    optimizer = {
        "name": chosen.get("optimizer", "adam"),
        "learning_rate": float(chosen.get("learning_rate", 1e-3)),
        "weight_decay": float(chosen.get("weight_decay", 0.0)),
        "beta1": float(chosen.get("beta1", 0.9)),
        "beta2": float(chosen.get("beta2", 0.999)),
        "epsilon": float(chosen.get("epsilon", 1e-8)),
        "grad_clip": float(chosen.get("grad_clip", 1.0)),
        "momentum": chosen.get(
            "momentum",
            0.9
            if chosen.get("optimizer") in {"sgdm", "nag"}
            else 0.95
            if chosen.get("optimizer") == "muon"
            else None,
        ),
        "schedule": chosen.get("schedule", "constant"),
    }
    if optimizer["name"] == "muon":
        optimizer.update(
            muon_ns_steps=int(chosen.get("muon_ns_steps", 5)),
            muon_auxiliary_learning_rate=float(
                chosen.get("muon_auxiliary_learning_rate", optimizer["learning_rate"])
            ),
            muon_auxiliary_weight_decay=float(
                chosen.get("muon_auxiliary_weight_decay", optimizer["weight_decay"])
            ),
        )
    if optimizer["schedule"] == "cosine_to_zero":
        round_index = int(chosen.get("round", 0))
        registered_requests = int(chosen.get("registered_request_count", 1))
        generation_tokens = int(chosen.get("generation_tokens", 0))
        if registered_requests < 1 or generation_tokens < 1:
            raise ValueError("cosine schedule requires a registered request/output budget")
        expected_rounds = registered_requests * generation_tokens
        optimizer["schedule_total_published_updates"] = max(
            E2_MINIMUM_UPDATES[round_index],
            math.ceil(expected_rounds / (stride * coalescing)),
        )
    payload = {
        "schema_version": 1,
        "method": method,
        "algorithm": job.backend,
        "parameter_scope": chosen.get("scope", "all"),
        "weight_update_mode": parameterization,
        "rank": rank if parameterization == "lora" else None,
        "optimizer": optimizer,
        "stride": stride,
        "canvas_tokens": _speculative_canvas(job),
        "loss_position_decay": float(chosen.get("loss_position_decay", math.exp(-1.0 / 7.0))),
        "teacher_row_policy": chosen.get("teacher_row_policy", "latest_update_round_only"),
        "extra_logical_delay": int(chosen.get("logical_delay", 0)),
        "adaptation_microbatch_size": int(chosen.get("microbatch", 1)),
        "update_coalescing": coalescing,
        "stream_priority": chosen.get("stream_priority", "default"),
        "max_in_flight": _concurrency(job),
        "kv_history_policy": "frozen",
        "reset_scope": (
            "request_batched"
            if job.method == "tts_lora_batched"
            else "request"
            if _request_scoped_adaptation(job)
            else "cohort"
        ),
        "telemetry_round_items": _telemetry_round_items(job),
        "adaptation_group_id": f"{job.node}-{job.ordinal}",
        "telemetry_detail": "profile" if job.node == "E4-profile" else "headline",
        "verification_mode": chosen.get("verification", "native_scheduler"),
        "controlled_candidate_replay": bool(chosen.get("controlled_replay", False)),
        "controlled_candidate_role": chosen.get("controlled_candidate_role"),
        "failure_injection": chosen.get("failure"),
    }
    scientific_node = str(job.parameters.get("source_node", job.node))
    if scientific_node == "E1a" and chosen.get("verification") == "fixed_budget":
        payload["fixed_total_token_budget"] = int(
            chosen.get("proposal_budget", 8)
        ) * _concurrency(job)
    if scientific_node == "E1a":
        payload["confidence_loss_weight"] = float(chosen.get("confidence_loss_weight", 0.1))
        payload["save_confidence_outcomes"] = bool(
            chosen.get("save_confidence_outcomes", False)
        )
        if chosen.get("confidence_threshold") is not None:
            payload["confidence_threshold"] = float(chosen["confidence_threshold"])
        if chosen.get("confidence_temperature") is not None:
            payload["confidence_temperature"] = float(chosen["confidence_temperature"])
    if method.startswith("onlinespec_"):
        payload["online_spec"] = {
            "projection_radius": chosen.get("projection_radius"),
            "additional_learning_rates": chosen.get("additional_learning_rates", ()),
            "hedge_learning_rate": chosen.get("hedge_learning_rate"),
        }
    return payload


def server_command(
    config: ExperimentConfig,
    job: Job,
    *,
    port: int,
    output_dir: Path,
    adaptation: dict[str, Any] | None,
) -> list[str]:
    target = config.model_path(job.model)
    tp, dp = _topology(job)
    memory_fraction = config.server.mem_fraction_static
    if job.backend == "DSPARK" or job.parameters.get("exactness_bootstrap"):
        memory_fraction = min(memory_fraction, 0.80)
    max_running = _server_capacity(job, adaptation)
    argv = [
        str(config.server.python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(target),
        "--host",
        config.server.host,
        "--port",
        str(port),
        "--tp-size",
        str(tp),
        "--dp-size",
        str(dp),
        "--context-length",
        str(max(40960, job.context or 0)),
        "--max-running-requests",
        str(max_running),
        "--mem-fraction-static",
        str(memory_fraction),
        "--random-seed",
        str(config.protocol.seed),
        "--skip-server-warmup",
        "--speculative-speed-study-metrics",
    ]
    if job.parameters.get("deterministic_exactness"):
        argv.append("--enable-deterministic-inference")
    if job.parameters.get("regime") != "multi_turn_shared_prefix" and not job.parameters.get(
        "prefix_reuse"
    ):
        argv.append("--disable-radix-cache")
    if job.parameters.get("graph_replay", True) is False:
        argv.append("--disable-cuda-graph")
    else:
        graph_sizes = tuple(
            size for size in (1, 2, 4, 8, 16, 32, 64, 128, 256) if size <= max_running
        )
        argv.extend(
            [
                "--cuda-graph-bs-decode",
                *map(str, graph_sizes),
            ]
        )
    argv.extend(
        [
            "--chunked-prefill-size",
            "8192" if job.parameters.get("chunked_prefill") else "-1",
        ]
    )
    if job.method == "target_only" or job.backend == "NONE":
        return argv
    argv.extend(
        [
            "--speculative-algorithm",
            job.backend,
            "--speculative-num-draft-tokens",
            str(_speculative_canvas(job)),
            "--speculative-num-steps",
            str(_speculative_canvas(job) - 1 if job.backend == "NEXTN" else 1),
            "--speculative-draft-window-size",
            str(_speculative_canvas(job)),
            "--speculative-use-rejection-sampling",
        ]
    )
    if job.backend == "NEXTN":
        argv.extend(["--speculative-eagle-topk", "1"])
    if job.backend == "DSPARK":
        argv.extend(["--attention-backend", "triton"])
    draft = None if job.backend == "NEXTN" else config.draft_path(job.model, job.backend)
    if draft is not None:
        argv.extend(["--speculative-draft-model-path", str(draft)])
    if job.backend == "DSPARK" and config.server.dspark_sps_table is not None:
        argv.extend(
            [
                "--speculative-dspark-sps-table-path",
                str(config.server.dspark_sps_table),
            ]
        )
    if adaptation is not None:
        reserve_mb = config.server.adaptation_reserve_mb
        if job.backend == "NEXTN" and adaptation["weight_update_mode"] == "lora":
            reserve_mb = min(reserve_mb, 8192)
        adaptation_path = output_dir / "adaptation.json"
        adaptation_path.write_text(
            json.dumps(adaptation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        telemetry_path = (
            Path("/dev/full")
            if job.parameters.get("failure") in {"telemetry_backpressure", "disk_quota"}
            else output_dir / "cycles.jsonl"
        )
        argv.extend(
            [
                "--speculative-adaptation-config",
                str(adaptation_path),
                "--speculative-adaptation-telemetry-path",
                str(telemetry_path),
                "--speculative-adaptation-reserve-mb",
                str(reserve_mb),
            ]
        )
    profile = job.parameters.get("profiler")
    if profile == "nsys":
        return [
            str(config.profiler_tools["nsys"]),
            "profile",
            "--force-overwrite=true",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "-o",
            str(output_dir / "nsys"),
            *argv,
        ]
    if profile == "ncu":
        return [
            str(config.profiler_tools["ncu"]),
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "--export",
            str(output_dir / "ncu"),
            *argv,
        ]
    return argv


def server_session_key(job: Job, selection: dict[str, Any] | None = None) -> tuple[object, ...]:
    adaptation = adaptation_payload(job, selection)
    if adaptation is None:
        adaptation_layout: tuple[object, ...] = (
            "target" if job.method == "target_only" else "static",
        )
    else:
        optimizer = adaptation["optimizer"]
        online = adaptation.get("online_spec") or {}
        optimizer_state = (
            "two_moment" if optimizer["name"] in {"adam", "adamw", "chronobelief"} else "one_moment"
        )
        adaptation_layout = (
            "online" if str(adaptation["method"]).startswith("onlinespec_") else "adaptive",
            adaptation["weight_update_mode"],
            adaptation["parameter_scope"],
            adaptation["rank"],
            optimizer_state,
            len(online.get("additional_learning_rates", ())),
            adaptation["telemetry_round_items"],
        )
    return (
        job.model,
        job.backend,
        job.parameters.get("topology", "tp1_dp1"),
        *_topology(job),
        _server_capacity(job, adaptation),
        _speculative_canvas(job),
        bool(job.parameters.get("graph_replay", True)),
        bool(job.parameters.get("chunked_prefill")),
        bool(job.parameters.get("prefix_reuse"))
        or job.parameters.get("regime") == "multi_turn_shared_prefix",
        bool(job.parameters.get("deterministic_exactness")),
        job.parameters.get("profiler"),
        (
            "full-device"
            if job.parameters.get("failure") in {"telemetry_backpressure", "disk_quota"}
            else "normal-device"
        ),
        *adaptation_layout,
    )


def _concurrency(job: Job) -> int:
    if job.load and job.load.startswith("c") and job.load[1:].isdigit():
        return int(job.load[1:])
    if job.load and job.load.startswith("closed_loop_c"):
        return int(job.load.removeprefix("closed_loop_c"))
    return 1


def _server_capacity(job: Job, adaptation: dict[str, Any] | None) -> int:
    if adaptation is not None and adaptation.get("reset_scope") == "request":
        return 1
    if job.parameters.get("exactness_bootstrap"):
        return 1
    bundled = job.parameters.get("server_capacity")
    if isinstance(bundled, int) and bundled > 0:
        return bundled
    return _concurrency(job)


class GpuSampler:
    def __init__(self, gpus: tuple[int, ...], output: Path):
        self.gpus = gpus
        self.output = output
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.started = False

    def start(self) -> None:
        if not self.output.exists():
            self.output.write_text(
                "timestamp,index,memory_used_mb,gpu_util_pct,memory_util_pct,"
                "power_w,energy_mj,temperature_c,sm_clock_mhz,pstate,throttle\n",
                encoding="utf-8",
            )
        self.thread.start()
        self.started = True

    def stop(self) -> None:
        if self.started:
            self.stop_event.set()
            self.thread.join(timeout=5)

    def _run(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,"
            "power.draw,total_energy_consumption,temperature.gpu,clocks.sm,pstate,"
            "clocks_event_reasons.active",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(map(str, self.gpus)),
        ]
        while not self.stop_event.is_set():
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                with self.output.open("a", encoding="utf-8") as stream:
                    stream.write(result.stdout)
            else:
                fallback = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,index,memory.used,utilization.gpu,"
                        "utilization.memory,power.draw,temperature.gpu,clocks.sm,pstate",
                        "--format=csv,noheader,nounits",
                        "-i",
                        ",".join(map(str, self.gpus)),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if fallback.returncode == 0:
                    with self.output.open("a", encoding="utf-8") as stream:
                        for line in fallback.stdout.splitlines():
                            fields = [part.strip() for part in line.split(",")]
                            stream.write(",".join((*fields[:6], "N/A", *fields[6:], "N/A")) + "\n")
            self.stop_event.wait(1.0)


class ServerProcess:
    def __init__(
        self,
        config: ExperimentConfig,
        job: Job,
        *,
        gpus: tuple[int, ...],
        port: int,
        output_dir: Path,
        selection: dict[str, Any] | None,
    ):
        self.config = config
        self.job = job
        self.gpus = gpus
        self.port = port
        self.output_dir = output_dir
        self.adaptation = adaptation_payload(job, selection)
        self.session_key = server_session_key(job, selection)
        self.process: subprocess.Popen[str] | None = None
        self.log = None
        self.sampler = GpuSampler(gpus, output_dir / "gpu.csv")

    def configure(self, job: Job, selection: dict[str, Any] | None) -> SGLangClient:
        if (
            self.process is None
            or self.process.poll() is not None
            or server_session_key(job, selection) != self.session_key
        ):
            return self.restart_for(job, selection)
        adaptation = adaptation_payload(job, selection)
        if adaptation is not None:
            (self.output_dir / "adaptation.json").write_text(
                json.dumps(adaptation, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self.job = job
        self.adaptation = adaptation
        client = self.client
        client.reset()
        return client

    @property
    def client(self) -> SGLangClient:
        return SGLangClient(
            f"http://{self.config.server.host}:{self.port}",
            self.config.server.request_timeout_seconds,
        )

    def start(self) -> SGLangClient:
        argv = server_command(
            self.config,
            self.job,
            port=self.port,
            output_dir=self.output_dir,
            adaptation=self.adaptation,
        )
        (self.output_dir / "server-command.json").write_text(
            json.dumps(argv, indent=2) + "\n", encoding="utf-8"
        )
        environment = dict(os.environ)
        if self.config.server.cuda_home is not None:
            cuda_home = self.config.server.cuda_home
            environment["CUDA_HOME"] = str(cuda_home)
            environment["CUDA_PATH"] = str(cuda_home)
            environment["PATH"] = os.pathsep.join(
                (str(cuda_home / "bin"), environment.get("PATH", ""))
            )
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                (str(cuda_home / "lib64"), environment.get("LD_LIBRARY_PATH", ""))
            )
        roots = [str(self.config.sglang_root / "python"), str(Path(__file__).parents[1])]
        if environment.get("PYTHONPATH"):
            roots.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(roots)
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpus))
        environment["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "false"
        if self.job.backend == "DSPARK":
            environment["SGLANG_RAGGED_VERIFY_MODE"] = "compact"
        (self.output_dir / "server.stopped").unlink(missing_ok=True)
        self.log = (self.output_dir / "server.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            argv,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        (self.output_dir / "server.pid").write_text(f"{self.process.pid}\n", encoding="utf-8")
        self.sampler.start()
        client = self.client
        deadline = time.monotonic() + self.config.server.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                if self.log is not None:
                    self.log.flush()
                log_path = self.output_dir / "server.log"
                tail = log_path.read_bytes()[-4096:].decode(errors="replace")
                raise RuntimeError(
                    f"SGLang exited during startup with {self.process.returncode}: {tail}"
                )
            if client.health():
                return client
            time.sleep(1)
        raise TimeoutError("SGLang did not become healthy before the startup timeout")

    def restart(self) -> SGLangClient:
        self.stop()
        time.sleep(1)
        self.process = None
        self.log = None
        self.sampler = GpuSampler(self.gpus, self.output_dir / "gpu.csv")
        return self.start()

    def restart_for(self, job: Job, selection: dict[str, Any] | None) -> SGLangClient:
        self.stop()
        self.job = job
        self.adaptation = adaptation_payload(job, selection)
        self.session_key = server_session_key(job, selection)
        self.process = None
        self.log = None
        self.sampler = GpuSampler(self.gpus, self.output_dir / "gpu.csv")
        return self.start()

    def inject_failure(self, kind: str) -> SGLangClient:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("cannot inject a fault into a stopped server")
        if kind in {"replica_restart", "replica_drain"}:
            return self.restart()
        children = subprocess.run(
            ["pgrep", "-P", str(self.process.pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        victim = int(children[-1]) if children else self.process.pid
        if kind == "communicator_failure":
            os.kill(victim, signal.SIGKILL)
            return self.restart()
        if kind == "slow_rank":
            os.kill(victim, signal.SIGSTOP)
            time.sleep(0.5)
            os.kill(victim, signal.SIGCONT)
        return self.client

    def stop(self) -> None:
        self.sampler.stop()
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self.log is not None:
            self.log.close()
        (self.output_dir / "server.stopped").write_text("stopped\n", encoding="utf-8")

    def __enter__(self) -> SGLangClient:
        try:
            return self.start()
        except Exception:
            self.stop()
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


class StickyReplicaClient:
    def __init__(self, replicas: tuple[SGLangClient, SGLangClient]):
        self.replicas = replicas

    def replica_index(self, routing_key: str | None) -> int:
        if routing_key is None:
            return 0
        return sum((offset + 1) * ord(value) for offset, value in enumerate(routing_key)) % 2

    def _replica(self, routing_key: str | None) -> SGLangClient:
        return self.replicas[self.replica_index(routing_key)]

    def health(self) -> bool:
        return all(replica.health() for replica in self.replicas)

    def reset(self) -> None:
        for replica in self.replicas:
            replica.reset()

    def warmup(self, prompt, *, max_new_tokens: int, seed: int) -> None:
        for index, replica in enumerate(self.replicas):
            replica.run_batch(
                (prompt,),
                max_new_tokens=max_new_tokens,
                seed=seed + index,
                routing_key=f"replica-warmup-{index}",
                request_id_prefix=f"replica-warmup-{index}",
            )
        self.reset()

    def server_info(self) -> dict[str, object]:
        states = []
        for replica in self.replicas:
            info = replica.server_info()
            nested = info.get("internal_states")
            states.extend(
                nested if isinstance(nested, list) else [info.get("internal_state", info)]
            )
        return {"internal_states": states}

    def tokenize(self, text: str) -> tuple[int, ...]:
        return self.replicas[0].tokenize(text)

    def run_batch(self, prompts, *, routing_key=None, **kwargs):
        return self._replica(routing_key).run_batch(prompts, routing_key=routing_key, **kwargs)

    def abort(self, request_id: str) -> None:
        for replica in self.replicas:
            replica.abort(request_id)

    def run_bounded(self, prompts, **kwargs):
        from .client import _run_bounded

        return _run_bounded(self, prompts, **kwargs)

    def run_closed_loop(self, prompts, **kwargs):
        from .client import _run_closed_loop

        return _run_closed_loop(self, prompts, **kwargs)

    def run_scheduled(
        self,
        prompts,
        arrival_offsets,
        *,
        max_new_tokens,
        seed: int,
        temperature: float = 0.0,
        routing_keys=None,
        max_in_flight: int = 256,
        deadline_seconds: float = 120.0,
        drain_seconds: float = 180.0,
    ):
        prompt_rows = tuple(prompts)
        offsets = tuple(arrival_offsets)
        keys = tuple(routing_keys) if routing_keys is not None else (None,) * len(prompt_rows)
        budgets = (
            (max_new_tokens,) * len(prompt_rows)
            if isinstance(max_new_tokens, int)
            else tuple(max_new_tokens)
        )
        groups: list[list[int]] = [[], []]
        for index, key in enumerate(keys):
            groups[self.replica_index(key)].append(index)
        started = time.perf_counter()

        def run_group(replica_index: int) -> ScheduledRun:
            indexes = groups[replica_index]
            return self.replicas[replica_index].run_scheduled(
                (prompt_rows[index] for index in indexes),
                (offsets[index] for index in indexes),
                max_new_tokens=tuple(budgets[index] for index in indexes),
                seed=seed,
                temperature=temperature,
                routing_keys=(keys[index] for index in indexes),
                request_ids=tuple(f"scheduled-{index:05d}" for index in indexes),
                max_in_flight=max(1, max_in_flight // 2),
                deadline_seconds=deadline_seconds,
                drain_seconds=drain_seconds,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            runs = tuple(
                future.result()
                for future in (pool.submit(run_group, index) for index in range(2) if groups[index])
            )
        results = sorted(
            (result for run in runs for result in run.results),
            key=lambda row: row.request_id,
        )
        outcomes = sorted(
            (outcome for run in runs for outcome in run.outcomes),
            key=lambda row: row.request_id,
        )
        return ScheduledRun(tuple(results), tuple(outcomes), time.perf_counter() - started)


class ReplicaServerProcess:
    def __init__(
        self,
        config: ExperimentConfig,
        job: Job,
        *,
        gpus: tuple[int, int],
        port: int,
        output_dir: Path,
        selection: dict[str, Any] | None,
    ):
        second_dir = output_dir / "replica-1"
        second_dir.mkdir(parents=True, exist_ok=True)
        self.replicas = (
            ServerProcess(
                config, job, gpus=(gpus[0],), port=port, output_dir=output_dir, selection=selection
            ),
            ServerProcess(
                config,
                job,
                gpus=(gpus[1],),
                port=port + 1,
                output_dir=second_dir,
                selection=selection,
            ),
        )
        self.output_dir = output_dir

    @property
    def process(self):
        return self.replicas[0].process

    @property
    def client(self) -> StickyReplicaClient:
        return StickyReplicaClient(tuple(replica.client for replica in self.replicas))

    def start(self) -> StickyReplicaClient:
        started = []
        try:
            for replica in self.replicas:
                replica.start()
                started.append(replica)
            return self.client
        except Exception:
            for replica in started:
                replica.stop()
            raise

    def configure(self, job: Job, selection: dict[str, Any] | None):
        for replica in self.replicas:
            replica.configure(job, selection)
        return self.client

    def restart(self):
        self.stop()
        return self.start()

    def restart_for(self, job: Job, selection: dict[str, Any] | None):
        for replica in self.replicas:
            replica.restart_for(job, selection)
        return self.client

    def inject_failure(self, kind: str):
        if kind in {"replica_restart", "replica_drain"}:
            self.replicas[1].restart()
        elif kind == "communicator_failure":
            self.replicas[0].restart()
        else:
            self.replicas[0].inject_failure(kind)
        return self.client

    def stop(self) -> None:
        for replica in self.replicas:
            replica.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
