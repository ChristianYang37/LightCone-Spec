"""Small YAML configuration for local two-GPU paper experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _absolute(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path


@dataclass(frozen=True)
class ServerConfig:
    python: Path
    cuda_home: Path | None = None
    host: str = "127.0.0.1"
    base_port: int = 30000
    mem_fraction_static: float = 0.90
    adaptation_reserve_mb: int = 32768
    startup_timeout_seconds: int = 900
    request_timeout_seconds: int = 900
    requests_per_cell: int = 16
    max_new_tokens: int = 256
    warmup_requests: int = 1


@dataclass(frozen=True)
class ProtocolConfig:
    preset: str = "paper-v1"
    start_stage: str | None = None
    end_stage: str | None = None
    max_process_retries: int = 1
    final_blocks: int | None = None
    seed: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    source: Path
    run_name: str
    sglang_root: Path
    results_root: Path
    models: dict[str, Path]
    drafts: dict[str, Path]
    datasets: dict[str, Path]
    gpu_ids: tuple[int, int]
    server: ServerConfig
    protocol: ProtocolConfig
    profiler_tools: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        source = Path(path).expanduser().resolve()
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        root = _mapping(data, "configuration")
        paths = _mapping(root.get("paths"), "paths")
        server_data = _mapping(root.get("server", {}), "server")
        protocol_data = _mapping(root.get("protocol", {}), "protocol")
        models = {
            str(name): _absolute(value, f"models.{name}")
            for name, value in _mapping(paths.get("models"), "paths.models").items()
        }
        drafts = {
            str(name): _absolute(value, f"drafts.{name}")
            for name, value in _mapping(paths.get("drafts"), "paths.drafts").items()
        }
        datasets = {
            str(name): _absolute(value, f"datasets.{name}")
            for name, value in _mapping(paths.get("datasets"), "paths.datasets").items()
        }
        raw_gpus = root.get("gpu_ids", [0, 1])
        if (
            not isinstance(raw_gpus, list)
            or len(raw_gpus) != 2
            or any(not isinstance(item, int) or item < 0 for item in raw_gpus)
            or raw_gpus[0] == raw_gpus[1]
        ):
            raise ValueError("gpu_ids must name two distinct non-negative devices")
        host = server_data.get("host", "127.0.0.1")
        if not isinstance(host, str) or not host:
            raise ValueError("server.host must be text")
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("server.host must bind only to loopback")
        server = ServerConfig(
            python=_absolute(server_data.get("python"), "server.python"),
            cuda_home=(
                _absolute(server_data["cuda_home"], "server.cuda_home")
                if server_data.get("cuda_home") is not None
                else None
            ),
            host=host,
            base_port=int(server_data.get("base_port", 30000)),
            mem_fraction_static=float(server_data.get("mem_fraction_static", 0.90)),
            adaptation_reserve_mb=int(server_data.get("adaptation_reserve_mb", 32768)),
            startup_timeout_seconds=int(server_data.get("startup_timeout_seconds", 900)),
            request_timeout_seconds=int(server_data.get("request_timeout_seconds", 900)),
            requests_per_cell=int(server_data.get("requests_per_cell", 16)),
            max_new_tokens=int(server_data.get("max_new_tokens", 256)),
            warmup_requests=int(server_data.get("warmup_requests", 1)),
        )
        if not 0 < server.mem_fraction_static < 1:
            raise ValueError("server.mem_fraction_static must be between zero and one")
        if (
            server.requests_per_cell < 1
            or server.max_new_tokens < 1
            or server.warmup_requests < 0
            or server.adaptation_reserve_mb < 0
        ):
            raise ValueError("server request counts and token budget are invalid")
        if not 1024 <= server.base_port <= 65531:
            raise ValueError("server.base_port must leave four valid local ports")
        if server.startup_timeout_seconds < 1 or server.request_timeout_seconds < 1:
            raise ValueError("server timeouts must be positive")
        protocol = ProtocolConfig(
            preset=str(protocol_data.get("preset", "paper-v1")),
            start_stage=protocol_data.get("start_stage"),
            end_stage=protocol_data.get("end_stage"),
            max_process_retries=int(protocol_data.get("max_process_retries", 1)),
            final_blocks=(
                int(protocol_data["final_blocks"])
                if protocol_data.get("final_blocks") is not None
                else None
            ),
            seed=int(protocol_data.get("seed", 0)),
        )
        if protocol.preset != "paper-v1":
            raise ValueError("only protocol preset paper-v1 is supported")
        if protocol.max_process_retries not in {0, 1}:
            raise ValueError("max_process_retries must be zero or one")
        if protocol.final_blocks is not None and not 12 <= protocol.final_blocks <= 20:
            raise ValueError("final_blocks must be between 12 and 20")
        from .protocol import PAPER_NODES

        for name, value in (
            ("start_stage", protocol.start_stage),
            ("end_stage", protocol.end_stage),
        ):
            if value is not None and value not in PAPER_NODES:
                raise ValueError(f"protocol.{name} must name a paper stage")
        if (
            protocol.start_stage is not None
            and protocol.end_stage is not None
            and PAPER_NODES.index(protocol.start_stage) > PAPER_NODES.index(protocol.end_stage)
        ):
            raise ValueError("protocol.start_stage must not follow end_stage")
        profilers = {
            str(name): _absolute(value, f"profiler_tools.{name}")
            for name, value in _mapping(root.get("profiler_tools", {}), "profiler_tools").items()
        }
        run_name = root.get("run_name")
        if not isinstance(run_name, str) or not run_name.strip():
            raise ValueError("run_name must be non-empty text")
        return cls(
            source=source,
            run_name=run_name,
            sglang_root=_absolute(paths.get("sglang_root"), "paths.sglang_root"),
            results_root=_absolute(paths.get("results_root"), "paths.results_root"),
            models=models,
            drafts=drafts,
            datasets=datasets,
            gpu_ids=(raw_gpus[0], raw_gpus[1]),
            server=server,
            protocol=protocol,
            profiler_tools=profilers,
        )

    @property
    def run_dir(self) -> Path:
        return self.results_root / self.run_name

    def normalized(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "gpu_ids": list(self.gpu_ids),
            "paths": {
                "sglang_root": str(self.sglang_root),
                "results_root": str(self.results_root),
                "models": {name: str(path) for name, path in self.models.items()},
                "drafts": {name: str(path) for name, path in self.drafts.items()},
                "datasets": {name: str(path) for name, path in self.datasets.items()},
            },
            "server": {
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(self.server).items()
            },
            "protocol": dict(vars(self.protocol)),
            "profiler_tools": {
                name: str(path) for name, path in self.profiler_tools.items()
            },
        }

    def validate_local_paths(self) -> None:
        required = {
            "SGLang checkout": self.sglang_root,
            "Python interpreter": self.server.python,
            **{f"model {name}": path for name, path in self.models.items()},
            **{f"draft {name}": path for name, path in self.drafts.items()},
            **{f"dataset {name}": path for name, path in self.datasets.items()},
            **{f"profiler {name}": path for name, path in self.profiler_tools.items()},
        }
        if self.server.cuda_home is not None:
            required["CUDA toolkit"] = self.server.cuda_home
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing configured paths:\n" + "\n".join(missing))

    def model_path(self, model: str) -> Path:
        try:
            return self.models[model]
        except KeyError as error:
            raise KeyError(f"no local model path configured for {model}") from error

    def draft_path(self, model: str, backend: str) -> Path | None:
        if backend in {"NONE", "TARGET_ONLY"}:
            return None
        key = f"{model}|{backend}"
        if key in self.drafts:
            return self.drafts[key]
        raise KeyError(f"no local draft path configured for {model} with {backend}")

    def has_exact_draft(self, model: str, backend: str) -> bool:
        if backend in {"NONE", "TARGET_ONLY", "NEXTN"}:
            return True
        return f"{model}|{backend}" in self.drafts

    def dataset_path(self, task: str) -> Path:
        try:
            return self.datasets[task]
        except KeyError as error:
            raise KeyError(f"no exact local dataset path configured for {task}") from error
