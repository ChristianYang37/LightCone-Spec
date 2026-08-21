"""Direct SGLang process lifecycle for one experiment cell."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .client import SGLangClient
from .config import ExperimentConfig
from .protocol import Job

ADAPTIVE_METHODS = {
    "tts",
    "l0_naive",
    "lightcone",
    "lightcone_candidate",
    "onlinespec_candidate",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
}


def _topology(job: Job) -> tuple[int, int]:
    topology = str(job.parameters.get("topology", "tp1_dp1"))
    tp = 2 if topology == "tp2_dp1" or job.node.startswith("E6") else 1
    dp = 2 if topology == "two_replica_tp1_dp2" else 1
    return tp, dp


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
    method = {
        "tts": "tts",
        "l0_naive": "l0",
        "lightcone": "l0",
        "lightcone_candidate": "l0",
        "onlinespec_candidate": "onlinespec_ogd",
    }.get(job.method, job.method)
    rank = chosen.get("rank", 8)
    parameterization = chosen.get("parameterization", "lora")
    if parameterization == "none":
        return None
    payload = {
        "schema_version": 1,
        "method": method,
        "algorithm": job.backend,
        "parameter_scope": chosen.get("scope", "all"),
        "weight_update_mode": parameterization,
        "rank": rank if parameterization == "lora" else None,
        "optimizer": {
            "name": chosen.get("optimizer", "adam"),
            "learning_rate": float(chosen.get("learning_rate", 1e-3)),
            "weight_decay": float(chosen.get("weight_decay", 0.0)),
            "beta1": float(chosen.get("beta1", 0.9)),
            "beta2": float(chosen.get("beta2", 0.999)),
            "epsilon": float(chosen.get("epsilon", 1e-8)),
            "grad_clip": float(chosen.get("grad_clip", 1.0)),
            "momentum": chosen.get("momentum"),
            "schedule": chosen.get("schedule", "constant"),
        },
        "stride": int(chosen.get("stride", 10)),
        "canvas_tokens": int(job.width or 16),
        "loss_position_decay": float(
            chosen.get("loss_position_decay", math.exp(-1.0 / 7.0))
        ),
        "teacher_row_policy": chosen.get(
            "teacher_row_policy", "latest_update_round_only"
        ),
        "extra_logical_delay": int(chosen.get("logical_delay", 0)),
        "adaptation_microbatch_size": int(chosen.get("microbatch", 1)),
        "update_coalescing": int(chosen.get("coalescing", 1)),
        "stream_priority": chosen.get("stream_priority", "default"),
        "max_in_flight": 1,
        "kv_history_policy": "frozen",
        "reset_scope": "request" if job.method in {"tts", "l0_naive"} else "cohort",
        "adaptation_group_id": f"{job.node}-{job.ordinal}",
        "telemetry_detail": "profile" if job.node == "E4-profile" else "headline",
        "verification_mode": chosen.get("verification", "native_scheduler"),
        "controlled_candidate_replay": bool(chosen.get("controlled_replay", False)),
        "failure_injection": chosen.get("failure"),
    }
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
        str(_concurrency(job)),
        "--mem-fraction-static",
        str(config.server.mem_fraction_static),
        "--random-seed",
        str(config.protocol.seed),
        "--speculative-speed-study-metrics",
    ]
    if job.parameters.get("regime") != "multi_turn_shared_prefix" and not job.parameters.get(
        "prefix_reuse"
    ):
        argv.append("--disable-radix-cache")
    if not job.parameters.get("graph_replay", False):
        argv.append("--disable-cuda-graph")
    argv.extend(
        [
            "--chunked-prefill-size",
            "8192" if job.parameters.get("chunked_prefill") else "-1",
        ]
    )
    if job.method == "target_only" or job.backend == "NONE":
        argv.append("--disable-overlap-schedule")
        return argv
    argv.extend(
        [
            "--speculative-algorithm",
            job.backend,
            "--speculative-num-draft-tokens",
            str(job.width or 16),
            "--speculative-num-steps",
            "1",
            "--speculative-draft-window-size",
            str(job.width or 16),
            "--speculative-use-rejection-sampling",
        ]
    )
    draft = None if job.backend == "NEXTN" else config.draft_path(job.model, job.backend)
    if draft is not None:
        argv.extend(["--speculative-draft-model-path", str(draft)])
    if adaptation is not None:
        adaptation_path = output_dir / "adaptation.json"
        adaptation_path.write_text(json.dumps(adaptation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        telemetry_path = (
            Path("/dev/full")
            if job.parameters.get("failure")
            in {"telemetry_backpressure", "disk_quota"}
            else output_dir / "cycles.jsonl"
        )
        argv.extend(
            [
                "--speculative-adaptation-config",
                str(adaptation_path),
                "--speculative-adaptation-telemetry-path",
                str(telemetry_path),
                "--speculative-adaptation-reserve-mb",
                str(config.server.adaptation_reserve_mb),
            ]
        )
    profile = job.parameters.get("profiler")
    if profile == "nsys":
        return [
            str(config.profiler_tools["nsys"]),
            "profile",
            "--force-overwrite=true",
            "-o",
            str(output_dir / "nsys"),
            *argv,
        ]
    if profile == "ncu":
        return [
            str(config.profiler_tools["ncu"]),
            "--target-processes",
            "all",
            "--export",
            str(output_dir / "ncu"),
            *argv,
        ]
    return argv


def server_session_key(
    job: Job, selection: dict[str, Any] | None = None
) -> tuple[object, ...]:
    adaptation = adaptation_payload(job, selection)
    if adaptation is None:
        adaptation_layout: tuple[object, ...] = (
            "target" if job.method == "target_only" else "static",
        )
    else:
        optimizer = adaptation["optimizer"]
        online = adaptation.get("online_spec") or {}
        adaptation_layout = (
            "online" if str(adaptation["method"]).startswith("onlinespec_") else "adaptive",
            adaptation["weight_update_mode"],
            adaptation["parameter_scope"],
            adaptation["rank"],
            optimizer["name"],
            len(online.get("additional_learning_rates", ())),
            adaptation["stream_priority"],
        )
    return (
        job.model,
        job.backend,
        *_topology(job),
        job.context,
        _concurrency(job),
        job.width,
        bool(job.parameters.get("graph_replay")),
        bool(job.parameters.get("chunked_prefill")),
        bool(job.parameters.get("prefix_reuse"))
        or job.parameters.get("regime") == "multi_turn_shared_prefix",
        job.parameters.get("profiler"),
        (
            "full-device"
            if job.parameters.get("failure")
            in {"telemetry_backpressure", "disk_quota"}
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


class GpuSampler:
    def __init__(self, gpus: tuple[int, ...], output: Path):
        self.gpus = gpus
        self.output = output
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.started = False

    def start(self) -> None:
        if not self.output.exists():
            self.output.write_text("timestamp,index,memory_used_mb,power_w,temperature_c,sm_clock_mhz\n", encoding="utf-8")
        self.thread.start()
        self.started = True

    def stop(self) -> None:
        if self.started:
            self.stop_event.set()
            self.thread.join(timeout=5)

    def _run(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=timestamp,index,memory.used,power.draw,temperature.gpu,clocks.sm",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(map(str, self.gpus)),
        ]
        while not self.stop_event.is_set():
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                with self.output.open("a", encoding="utf-8") as stream:
                    stream.write(result.stdout)
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

    def configure(
        self, job: Job, selection: dict[str, Any] | None
    ) -> SGLangClient:
        if server_session_key(job, selection) != self.session_key:
            raise RuntimeError("server session layout changed")
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
        roots = [str(self.config.sglang_root / "python"), str(Path(__file__).parents[1])]
        if environment.get("PYTHONPATH"):
            roots.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(roots)
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpus))
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
                raise RuntimeError(f"SGLang exited during startup with {self.process.returncode}")
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
