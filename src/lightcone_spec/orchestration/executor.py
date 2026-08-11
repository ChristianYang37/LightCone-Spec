"""Content-bound execution of one materialised industrial serving cell.

Planning is separate from execution.  The planner validates every local
artifact and emits immutable server/bench argv without starting a process.
The async runner performs network and process mutation only through injected
interfaces, accounts every offered request, and publishes evidence only via
the repository's durable :class:`EvidenceWriter` terminal-receipt protocol.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from itertools import pairwise
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pyarrow.parquet as pq

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.experiments.load import (
    LoadAccounting,
    ProductionLoadPlan,
    RequestCorpus,
    RequestOutcome,
    TimingCoverage,
    TokenChunkTiming,
    account_scored_requests,
    evaluate_token_timing,
)
from lightcone_spec.experiments.registry import (
    CellStatus,
    ExperimentCell,
    ExperimentReceipt,
    WorkloadClass,
    content_sha256,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.serving import (
    BenchServingResult,
    BenchServingTransport,
    BoundServingRequest,
    official_bench_argv,
)
from lightcone_spec.orchestration.industrial import IndustrialRuntimePlan
from lightcone_spec.orchestration.runtime import (
    ServerLaunch,
    _immutable_json,
    _render_server,
)
from lightcone_spec.sglang_bridge.checkout import verify_patched_checkout
from lightcone_spec.sglang_bridge.config import sglang_adaptation_payload
from lightcone_spec.telemetry.records import (
    OUTPUT_HASH_FORMAT,
    PerformanceRecord,
    RequestRecord,
    RoundRecord,
    RunRecord,
    UpdateRecord,
)
from lightcone_spec.telemetry.writer import EvidenceWriter, load_completed_evidence

NATIVE_TERMINAL_EVIDENCE_HOOK = (
    "sglang.schema_v3.content_bound_terminal_speculative_evidence.v1"
)
NATIVE_TERMINAL_EVIDENCE_FIELDS = (
    "run_id",
    "run_nonce_sha256",
    "execution_plan_sha256",
    "rank_config_sha256",
    "server_process_id",
    "attempt_id",
    "request_round_rows",
    "update_rows",
    "performance_counters",
    "terminal_sha256",
)
MISSING_NATIVE_EVIDENCE_REASON = "missing_content_bound_native_speculative_evidence"
MAX_IN_MEMORY_REQUEST_EXECUTIONS = 100_000

type EvidenceItem = (
    PerformanceRecord | RequestRecord | RoundRecord | RunRecord | UpdateRecord
)


class _AsyncEvidenceSink:
    """Bounded single-writer bridge from the event loop to durable WAL."""

    def __init__(self, writer: EvidenceWriter, *, max_queued_rows: int = 1024) -> None:
        self._writer = writer
        self._queue: asyncio.Queue[EvidenceItem | object] = asyncio.Queue(
            maxsize=max_queued_rows
        )
        self._stop = object()
        self._error: BaseException | None = None
        self._closed = False
        self._backpressure_events = 0
        self._overflow_lock = asyncio.Lock()
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is self._stop:
                    return
                flush_after = self._queue.empty()
                if not await asyncio.to_thread(
                    self._write_one, item, flush_after=flush_after
                ):
                    raise RuntimeError("bounded evidence writer dropped a row")
            except BaseException as error:  # noqa: BLE001 - background boundary
                self._error = error
                while not self._queue.empty():
                    self._queue.get_nowait()
                    self._queue.task_done()
                return
            finally:
                self._queue.task_done()

    def _write_one(self, record: EvidenceItem, *, flush_after: bool) -> bool:
        written = self._writer.write(record)
        if written and flush_after:
            self._writer.flush()
        return written

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("background evidence writer failed") from self._error

    @property
    def backpressure_events(self) -> int:
        return self._backpressure_events

    async def write(self, record: EvidenceItem) -> None:
        if self._closed:
            raise RuntimeError("background evidence sink is closed")
        self._raise_error()
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull as error:
            self._backpressure_events += 1
            # Saturation makes the run nonclaimable, so measured-path latency no
            # longer matters.  Drain the bounded queue and durably persist the
            # triggering terminal row before aborting; a negative row is never
            # silently dropped to preserve zero-think timing.
            async with self._overflow_lock:
                await self.flush()
                await self._queue.put(record)
                await self.flush()
            raise RuntimeError(
                "background evidence queue saturated; run is nonclaimable"
            ) from error
        self._raise_error()

    async def flush(self) -> None:
        await self._queue.join()
        self._raise_error()

    async def close(self) -> None:
        if self._closed:
            self._raise_error()
            return
        if not self._task.done():
            await self._queue.put(self._stop)
        await self._task
        self._closed = True
        self._raise_error()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class ArtifactBinding:
    """One file bound by raw bytes plus its optional semantic identity."""

    name: str
    path: str
    content_sha256: str
    file_sha256: str
    size: int
    experiment: str | None = None

    @classmethod
    def from_path(
        cls,
        *,
        name: str,
        path: str | Path,
        expected_sha256: str | None = None,
        semantic_sha256: str | None = None,
        experiment: str | None = None,
    ) -> ArtifactBinding:
        unresolved = Path(path)
        if unresolved.is_symlink():
            raise ValueError("artifact must be a regular non-symlinked file")
        resolved = unresolved.resolve()
        if (
            not name
            or "\n" in name
            or (experiment is not None and (not experiment or "\n" in experiment))
        ):
            raise ValueError("artifact identities must be non-empty single-line text")
        if resolved.is_symlink() or not resolved.is_file():
            raise ValueError("artifact must be a regular non-symlinked file")
        file_digest = _file_sha256(resolved)
        content_digest = semantic_sha256 or file_digest
        if not _is_sha256(content_digest):
            raise ValueError("artifact semantic identity must be a lowercase SHA-256")
        if expected_sha256 is not None and content_digest != expected_sha256:
            raise ValueError(f"artifact {name!r} differs from its locked digest")
        value = cls(
            name=name,
            path=str(resolved),
            content_sha256=content_digest,
            file_sha256=file_digest,
            size=resolved.stat().st_size,
            experiment=experiment,
        )
        value.assert_unchanged()
        return value

    def assert_unchanged(self) -> None:
        if (
            not self.name
            or not _is_sha256(self.content_sha256)
            or not _is_sha256(self.file_sha256)
            or self.size < 0
        ):
            raise ValueError("artifact binding is malformed")
        path = Path(self.path)
        if (
            not path.is_absolute()
            or path.resolve() != path
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != self.size
            or _file_sha256(path) != self.file_sha256
        ):
            raise RuntimeError(f"bound artifact changed: {self.name}")

    def identity_dict(self) -> dict[str, object]:
        return {
            "experiment": self.experiment,
            "name": self.name,
            "content_sha256": self.content_sha256,
            "file_sha256": self.file_sha256,
            "size": self.size,
        }


def _bound_sampling_profile(binding: ArtifactBinding) -> SamplingProfile:
    """Load the exact schema-v2 sampling semantics from a byte-bound file."""

    binding.assert_unchanged()
    try:
        value = json.loads(Path(binding.path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("sampling artifact is not valid JSON") from error
    expected_fields = {
        "schema_version",
        "purpose",
        "temperature",
        "top_p",
        "ignore_eos",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("sampling artifact fields do not match schema-v2")
    try:
        profile = SamplingProfile(**value)
        profile.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("sampling artifact has invalid schema-v2 semantics") from error
    if profile.sha256 != binding.content_sha256:
        raise ValueError("sampling artifact differs from its semantic digest")
    return profile


def _validate_request_sampling(
    request: BoundServingRequest,
    *,
    profile: SamplingProfile,
) -> None:
    parameters = dict(request.sampling.items)
    expected = {
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "ignore_eos": profile.ignore_eos,
        "max_new_tokens": request.requested_output_tokens,
    }
    for name, expected_value in expected.items():
        actual = parameters.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(
                f"request sampling {name} differs from the bound sampling profile"
            )
    sampling_seed = parameters.get("sampling_seed")
    if (
        isinstance(sampling_seed, bool)
        or not isinstance(sampling_seed, int)
        or sampling_seed < 0
    ):
        raise ValueError("request sampling requires a non-negative sampling_seed")


def _expected_scored_split(experiment: str, block: int) -> str:
    if experiment in {"E3b", "E5"}:
        return "pilot" if block in {0, 1, 2, 3} else "confirmation"
    if experiment == "E0":
        return "broad_replication"
    if experiment in {"E3a", "E1", "E2", "E4", "E1a"}:
        return "tuning"
    raise ValueError("serving cell has no registered scored split")


def _validate_cell_load_contract(
    cell: ExperimentCell,
    load_plan: ProductionLoadPlan,
) -> None:
    """Bind registry workload semantics to one immutable production trace."""

    identity = cell.identity
    corpus = load_plan.scored
    expected_split = _expected_scored_split(identity.experiment, identity.block)
    if corpus.split != expected_split:
        raise ValueError("load split differs from the registry stage/block")

    arrival = identity.arrival
    if arrival.startswith("closed_loop"):
        expected_source = "closed_loop"
    elif arrival in {"immediate_burst", "deterministic_stratified_requests"}:
        expected_source = "immediate_burst"
    elif arrival in {
        "poisson",
        "moderate_soak",
        "saturation_soak",
        "overload_soak",
    }:
        expected_source = "poisson"
    elif arrival == "burstgpt_shape":
        expected_source = "external_shape"
    else:
        raise ValueError("cell arrival policy is not materialized for execution")
    if corpus.source_kind != expected_source:
        raise ValueError("load source kind differs from the registry arrival policy")
    arrivals = tuple(request.arrival_us for request in corpus.requests)
    if expected_source in {"closed_loop", "immediate_burst"} and any(
        value != 0 for value in arrivals
    ):
        raise ValueError(
            "closed-loop/immediate-burst requests must start at offset zero"
        )
    if expected_source == "poisson" and (
        not arrivals
        or arrivals[0] <= 0
        or any(right <= left for left, right in pairwise(arrivals))
    ):
        raise ValueError("registered Poisson arrivals require strictly positive gaps")

    source = dict(corpus.source_parameters)
    cohort_text = identity.cohort
    try:
        count_text, popularity = cohort_text.removeprefix("K=").split(":", 1)
        cohort_count = int(count_text)
    except (AttributeError, ValueError) as error:
        raise ValueError("registry cohort policy is malformed") from error
    if (
        cohort_count != identity.cohort_count
        or source.get("cohort_count") != cohort_count
        or source.get("cohort_popularity") != popularity
    ):
        raise ValueError("load cohort generator differs from the registry cell")
    if (
        expected_source == "closed_loop"
        and source.get("concurrency") != identity.concurrency
    ):
        raise ValueError("closed-loop population cap differs from the registry cell")
    if identity.load_factor is not None and (
        expected_source != "poisson"
        or source.get("registered_load_factor") != identity.load_factor
    ):
        raise ValueError("Poisson load factor differs from the registry cell")

    lengths = tuple(
        (len(request.input_token_ids), request.requested_output_tokens)
        for request in corpus.requests
    )
    context = identity.context
    assert context is not None
    if identity.regime == "long_input_short_output" and any(
        input_tokens < math.ceil(0.75 * context)
        or output_tokens > math.floor(0.25 * context)
        for input_tokens, output_tokens in lengths
    ):
        raise ValueError("corpus violates the long-input/short-output regime")
    if identity.regime == "short_input_long_generation" and any(
        input_tokens > math.floor(0.25 * context)
        or output_tokens < math.ceil(0.75 * context)
        for input_tokens, output_tokens in lengths
    ):
        raise ValueError("corpus violates the short-input/long-generation regime")
    if identity.regime == "multi_turn_shared_prefix":
        inputs = tuple(request.input_token_ids for request in corpus.requests)
        shared = min((len(row) for row in inputs), default=0) // 2
        if (
            len(inputs) < 2
            or shared < 1
            or not all(row[:shared] == inputs[0][:shared] for row in inputs)
        ):
            raise ValueError("multi-turn corpus lacks a shared token prefix")
    if identity.regime == "production_mix" and not (
        any(left > right for left, right in lengths)
        and any(left < right for left, right in lengths)
    ):
        raise ValueError("production-mix corpus lacks both request regimes")


def industrial_execution_split_contract(
    *,
    registry_sha256: str,
    cell: ExperimentCell,
    load_plan: ProductionLoadPlan,
    sampling_profile_sha256: str,
    model_lock_sha256: str,
) -> dict[str, object]:
    """Return the exact local split/load identity required by the executor."""

    if not _is_sha256(registry_sha256):
        raise ValueError("split contract requires the registry SHA-256")
    if not _is_sha256(sampling_profile_sha256) or not _is_sha256(model_lock_sha256):
        raise ValueError("split contract requires sampling/model-lock SHA-256 values")
    load_plan.validate()
    _validate_cell_load_contract(cell, load_plan)
    hashes = load_plan.scored.hashes
    return {
        "schema_version": 1,
        "kind": "industrial_execution_split",
        "registry_sha256": registry_sha256,
        "cell_id": cell.cell_id,
        "experiment": cell.identity.experiment,
        "block": cell.identity.block,
        "task": cell.identity.task,
        "variant": cell.identity.variant,
        "scored_split": load_plan.scored.split,
        "paired_replay_sha256": load_plan.paired_replay_sha256,
        "warmup_corpus_sha256": (
            None if load_plan.warmup is None else load_plan.warmup.hashes.corpus_sha256
        ),
        "corpus_sha256": hashes.corpus_sha256,
        "request_ids_sha256": hashes.request_ids_sha256,
        "arrivals_sha256": hashes.arrivals_sha256,
        "cohorts_sha256": hashes.cohorts_sha256,
        "cancellations_sha256": hashes.cancellations_sha256,
        "source_kind": load_plan.scored.source_kind,
        "source_identity_sha256": load_plan.scored.source_identity_sha256,
        "source_parameters": dict(load_plan.scored.source_parameters),
        "window_sha256": load_plan.window.sha256,
        "sampling_profile_sha256": sampling_profile_sha256,
        "model_lock_sha256": model_lock_sha256,
    }


def _request_routes(
    corpus: RequestCorpus | None,
    *,
    route_id: str,
) -> tuple[BoundServingRequest, ...]:
    if corpus is None:
        return ()
    corpus.validate()
    return tuple(
        BoundServingRequest.create(request, route_id=route_id)
        for request in corpus.requests
    )


@dataclass(frozen=True)
class IndustrialExecutionPlan:
    """Immutable local plan for exactly one rank and one serving cell."""

    runtime_plan: IndustrialRuntimePlan
    load_plan: ProductionLoadPlan
    server_launch: ServerLaunch
    dependency_receipt_sha256s: tuple[str, ...]
    expected_dependency_outputs: tuple[tuple[str, str, str], ...]
    dependency_artifacts: tuple[ArtifactBinding, ...]
    split_artifact: ArtifactBinding
    sampling_artifact: ArtifactBinding
    model_lock_artifact: ArtifactBinding
    warmup_requests: tuple[BoundServingRequest, ...]
    scored_requests: tuple[BoundServingRequest, ...]
    bench_argv: tuple[str, ...]
    patched_sglang_tree: str = PINNED_SGLANG_TREE
    startup_timeout_s: float = 300.0
    shutdown_timeout_s: float = 30.0
    abort_grace_s: float = 30.0

    def validate(self) -> None:
        cell = self.runtime_plan.cell
        if not cell.runnable or cell.status is not CellStatus.UNMEASURED:
            raise ValueError("execution plan requires one runnable UNMEASURED cell")
        if cell.identity.experiment == "preflight" or cell.resources.workload_class in {
            WorkloadClass.DOWNLOAD,
            WorkloadClass.COMPILE,
        }:
            raise ValueError("non-serving/preflight cells cannot enter this executor")
        if len(self.runtime_plan.rank_configs) != 1:
            raise ValueError(
                "the current strict RunConfig exposes only one-rank serving execution"
            )
        self.load_plan.validate()
        config = self.runtime_plan.rank_configs[0]
        if config.runtime.max_running_requests != cell.identity.concurrency:
            raise ValueError("execution concurrency differs from the registry cell")
        if (
            config.runtime.sampling_profile_sha256
            != self.sampling_artifact.content_sha256
        ):
            raise ValueError("sampling artifact differs from the RunConfig")
        sampling_profile = _bound_sampling_profile(self.sampling_artifact)
        expected_split = industrial_execution_split_contract(
            registry_sha256=self.runtime_plan.registry_sha256,
            cell=cell,
            load_plan=self.load_plan,
            sampling_profile_sha256=self.sampling_artifact.content_sha256,
            model_lock_sha256=self.model_lock_artifact.content_sha256,
        )
        try:
            actual_split = json.loads(
                Path(self.split_artifact.path).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("split artifact is not valid JSON") from error
        if actual_split != expected_split:
            raise ValueError("split artifact differs from the exact cell/load contract")
        if self.patched_sglang_tree != PINNED_SGLANG_TREE:
            raise ValueError("execution plan has the wrong patched SGLang tree")
        for value, name in (
            (self.startup_timeout_s, "startup timeout"),
            (self.shutdown_timeout_s, "shutdown timeout"),
            (self.abort_grace_s, "abort grace"),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for artifact in (
            *self.dependency_artifacts,
            self.split_artifact,
            self.sampling_artifact,
            self.model_lock_artifact,
        ):
            artifact.assert_unchanged()
        expected = tuple(sorted(self.expected_dependency_outputs))
        actual = tuple(
            sorted(
                (
                    str(artifact.experiment),
                    artifact.name,
                    artifact.content_sha256,
                )
                for artifact in self.dependency_artifacts
            )
        )
        if actual != expected:
            raise ValueError(
                "dependency artifacts do not cover the locked outputs exactly"
            )
        if self.dependency_receipt_sha256s != (
            self.runtime_plan.dependency_receipt_sha256s
        ):
            raise ValueError("execution receipt chain differs from the runtime plan")
        route_id = config.runtime.router_identity
        expected_warmup = _request_routes(self.load_plan.warmup, route_id=route_id)
        expected_scored = _request_routes(self.load_plan.scored, route_id=route_id)
        if (
            self.warmup_requests != expected_warmup
            or self.scored_requests != expected_scored
        ):
            raise ValueError("execution requests differ from the immutable load plan")
        for request in (*self.warmup_requests, *self.scored_requests):
            _validate_request_sampling(request, profile=sampling_profile)
        cohorts = {request.cohort_id for request in self.scored_requests}
        if len(cohorts) != cell.identity.cohort_count:
            raise ValueError(
                "observed cohort cardinality differs from the registry cell"
            )
        context = cell.identity.context
        if context is None or any(
            len(request.input_token_ids) + request.requested_output_tokens > context
            for request in (*self.warmup_requests, *self.scored_requests)
        ):
            raise ValueError("request corpus exceeds the cell context bound")
        _validate_server_launch(self.runtime_plan, self.server_launch)
        expected_bench = official_bench_argv(
            base_url=self.server_launch.base_url,
            served_model=config.model.target,
            request_count=len(self.scored_requests),
            concurrency=config.runtime.max_running_requests,
            arrival_kind=self.load_plan.scored.source_kind,
        )
        if self.bench_argv != expected_bench:
            raise ValueError("bench argv differs from the official adapter surface")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        hashes = self.load_plan.scored.hashes
        return {
            "schema_version": 1,
            "runtime_plan_sha256": self.runtime_plan.sha256,
            "rank_config_sha256": self.rank_config_sha256,
            "topology_sha256": self.topology_sha256,
            "topology_receipt_sha256": self.runtime_plan.topology_receipt_sha256,
            "runtime_plan": self.runtime_plan.to_dict(),
            "load": {
                "paired_replay_sha256": self.load_plan.paired_replay_sha256,
                "warmup_corpus_sha256": (
                    self.load_plan.warmup.hashes.corpus_sha256
                    if self.load_plan.warmup is not None
                    else None
                ),
                "scored_corpus_sha256": hashes.corpus_sha256,
                "request_ids_sha256": hashes.request_ids_sha256,
                "arrivals_sha256": hashes.arrivals_sha256,
                "cohorts_sha256": hashes.cohorts_sha256,
                "cancellations_sha256": hashes.cancellations_sha256,
                "window_sha256": self.load_plan.window.sha256,
            },
            "server_launch": asdict(self.server_launch),
            "bench_adapter": ("sglang.benchmark.serving.async_request_sglang_generate"),
            "bench_argv": list(self.bench_argv),
            "dependency_receipt_sha256s": list(self.dependency_receipt_sha256s),
            "dependency_artifacts": [
                artifact.identity_dict()
                for artifact in sorted(
                    self.dependency_artifacts,
                    key=lambda value: (str(value.experiment), value.name),
                )
            ],
            "split_artifact": self.split_artifact.identity_dict(),
            "sampling_artifact": self.sampling_artifact.identity_dict(),
            "model_lock_artifact": self.model_lock_artifact.identity_dict(),
            "warmup_request_bindings": [
                request.sha256 for request in self.warmup_requests
            ],
            "scored_request_bindings": [
                request.sha256 for request in self.scored_requests
            ],
            "patched_sglang_tree": self.patched_sglang_tree,
            "startup_timeout_s": self.startup_timeout_s,
            "shutdown_timeout_s": self.shutdown_timeout_s,
            "abort_grace_s": self.abort_grace_s,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @property
    def rank_config_sha256(self) -> str:
        values = self.runtime_plan.to_dict()["rank_config_sha256s"]
        if (
            not isinstance(values, list)
            or len(values) != 1
            or not isinstance(values[0], str)
            or not _is_sha256(values[0])
        ):
            raise ValueError("runtime plan lacks exactly one rank-config digest")
        return values[0]

    @property
    def topology_sha256(self) -> str:
        """Canonical cell topology identity consumed by industrial reduction."""

        config = self.runtime_plan.rank_configs[0]
        cell = self.runtime_plan.cell
        return content_sha256(
            {
                "schema_version": 1,
                "cell_id": cell.cell_id,
                "topology": cell.identity.topology,
                "gpu_uuids": list(cell.resources.gpu_uuids),
                "tensor_parallel_size": config.runtime.tensor_parallel_size,
                "data_parallel_size": config.runtime.data_parallel_size,
                "world_size": len(self.runtime_plan.rank_configs),
            }
        )


def _validate_server_launch(
    runtime_plan: IndustrialRuntimePlan,
    launch: ServerLaunch,
) -> None:
    config = runtime_plan.rank_configs[0]
    if launch.method != config.method or not launch.exclusive_device:
        raise ValueError("server launch method/isolation differs from the runtime plan")
    parsed = urlsplit(launch.base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port != runtime_plan.cell.resources.ports[0]
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("industrial server launch must use its reserved loopback port")
    config_path = Path(launch.run_config)
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("server launch lacks a regular run-config artifact")
    try:
        rendered = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("server run-config artifact is invalid") from error
    if rendered != config.model_dump(mode="json"):
        raise ValueError("server run-config artifact differs from the runtime plan")
    adapted = config.method not in {"target_only", "static"}
    if adapted != (
        launch.adaptation_config is not None and launch.telemetry_path is not None
    ):
        raise ValueError("server adaptation artifacts differ from the method contract")
    argv = launch.argv
    if len(argv) < 18 or argv[:4] != (
        sys.executable,
        "-m",
        "lightcone_spec.sglang_bridge.launch",
        "--checkout",
    ):
        raise ValueError("server launch argv is not the registered SGLang launcher")
    checkout = Path(argv[4])
    if not checkout.is_dir() or argv[5] != "--":
        raise ValueError("server argv lacks its local disposable checkout")
    base = argv[6:18]
    expected_keys = (
        "--model-path",
        "--max-running-requests",
        "--mem-fraction-static",
        "--tp-size",
        "--host",
        "--port",
    )
    if base[::2] != expected_keys:
        raise ValueError("server base argv fields are incomplete or reordered")
    model_root = Path(base[1])
    if model_root.is_symlink() or not model_root.is_dir():
        raise ValueError("server target model root is not a regular directory")
    try:
        max_running = int(base[3])
        mem_fraction = float(base[5])
        tp_size = int(base[7])
        port = int(base[11])
    except ValueError as error:
        raise ValueError("server base argv values are malformed") from error
    if (
        max_running != config.runtime.max_running_requests
        or not math.isfinite(mem_fraction)
        or not 0 < mem_fraction < 1
        or tp_size != config.runtime.tensor_parallel_size
        or base[9] != parsed.hostname
        or port != parsed.port
    ):
        raise ValueError("server base argv differs from the RunConfig/base URL")
    remainder = argv[18:]
    if config.method == "target_only":
        if remainder:
            raise ValueError("target-only argv cannot enable speculative state")
        return
    if len(remainder) < 13:
        raise ValueError("speculative server argv is incomplete")
    expected_speculative_keys = (
        "--speculative-algorithm",
        "--speculative-draft-model-path",
        "--speculative-num-draft-tokens",
        "--speculative-draft-window-size",
        "--speculative-accept-threshold-single",
        "--speculative-accept-threshold-acc",
    )
    if remainder[:12:2] != expected_speculative_keys:
        raise ValueError("speculative server argv fields are incomplete or reordered")
    drafter_root = Path(remainder[3])
    try:
        draft_tokens = int(remainder[5])
        draft_window = int(remainder[7])
        accept_single = float(remainder[9])
        accept_acc = float(remainder[11])
    except ValueError as error:
        raise ValueError("speculative server argv values are malformed") from error
    if (
        remainder[1] != config.model.algorithm
        or drafter_root.is_symlink()
        or not drafter_root.is_dir()
        or draft_tokens != config.runtime.speculative_num_draft_tokens
        or draft_window != config.runtime.speculative_num_draft_tokens
        or accept_single != 1.0
        or accept_acc != 1.0
        or remainder[12] != "--speculative-use-rejection-sampling"
    ):
        raise ValueError("speculative server argv differs from the RunConfig")
    adaptation_argv = remainder[13:]
    if not adapted:
        if adaptation_argv:
            raise ValueError("Static argv cannot enable adaptation")
        return
    if not adaptation_argv or adaptation_argv[0] != "--speculative-speed-study-metrics":
        raise ValueError("adapted server argv lacks native speed-study evidence")
    adaptation_argv = adaptation_argv[1:]
    if len(adaptation_argv) != 6 or adaptation_argv[::2] != (
        "--speculative-adaptation-config",
        "--speculative-adaptation-reserve-mb",
        "--speculative-adaptation-telemetry-path",
    ):
        raise ValueError("adapted server argv fields are incomplete or reordered")
    if (
        adaptation_argv[1] != launch.adaptation_config
        or adaptation_argv[5] != launch.telemetry_path
    ):
        raise ValueError("adaptation argv paths differ from the launch contract")
    try:
        reserve_mb = int(adaptation_argv[3])
    except ValueError as error:
        raise ValueError("adaptation reserve is malformed") from error
    adaptation_path = Path(adaptation_argv[1])
    if reserve_mb < 1 or adaptation_path.is_symlink() or not adaptation_path.is_file():
        raise ValueError("adaptation reserve/config artifact is invalid")
    try:
        adaptation_payload = json.loads(adaptation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("adaptation config artifact is invalid") from error
    if adaptation_payload != sglang_adaptation_payload(config):
        raise ValueError("adaptation config artifact differs from the RunConfig")


def build_industrial_execution_plan(
    *,
    runtime_plan: IndustrialRuntimePlan,
    load_plan: ProductionLoadPlan,
    server_launch: ServerLaunch,
    dependency_receipts: tuple[ExperimentReceipt, ...],
    dependency_artifacts: tuple[ArtifactBinding, ...],
    split_artifact: ArtifactBinding,
    sampling_artifact: ArtifactBinding,
    model_lock_artifact: ArtifactBinding,
    startup_timeout_s: float = 300.0,
    shutdown_timeout_s: float = 30.0,
    abort_grace_s: float = 30.0,
) -> IndustrialExecutionPlan:
    """Bind a rendered runtime, exact trace, and content-locked artifacts."""

    receipt_sha256s = tuple(receipt.sha256 for receipt in dependency_receipts)
    if receipt_sha256s != runtime_plan.dependency_receipt_sha256s:
        raise ValueError("dependency receipt order differs from the runtime plan")
    expected = tuple(
        sorted(
            (
                (receipt.experiment, output.name, output.content_sha256)
                for receipt in dependency_receipts
                for output in receipt.outputs
            ),
            key=lambda row: (row[0], row[1]),
        )
    )
    route_id = runtime_plan.rank_configs[0].runtime.router_identity
    plan = IndustrialExecutionPlan(
        runtime_plan=runtime_plan,
        load_plan=load_plan,
        server_launch=server_launch,
        dependency_receipt_sha256s=receipt_sha256s,
        expected_dependency_outputs=expected,
        dependency_artifacts=dependency_artifacts,
        split_artifact=split_artifact,
        sampling_artifact=sampling_artifact,
        model_lock_artifact=model_lock_artifact,
        warmup_requests=_request_routes(load_plan.warmup, route_id=route_id),
        scored_requests=_request_routes(load_plan.scored, route_id=route_id),
        bench_argv=official_bench_argv(
            base_url=server_launch.base_url,
            served_model=runtime_plan.rank_configs[0].model.target,
            request_count=len(load_plan.scored.requests),
            concurrency=runtime_plan.rank_configs[0].runtime.max_running_requests,
            arrival_kind=load_plan.scored.source_kind,
        ),
        startup_timeout_s=startup_timeout_s,
        shutdown_timeout_s=shutdown_timeout_s,
        abort_grace_s=abort_grace_s,
    )
    plan.validate()
    _ = plan.sha256
    return plan


def render_industrial_execution_plan(
    *,
    output_root: str | Path,
    runtime_plan: IndustrialRuntimePlan,
    load_plan: ProductionLoadPlan,
    dependency_receipts: tuple[ExperimentReceipt, ...],
    dependency_artifacts: tuple[ArtifactBinding, ...],
    split_artifact: ArtifactBinding,
    sampling_artifact: ArtifactBinding,
    model_lock_artifact: ArtifactBinding,
    sglang_checkout: str | Path,
    model_roots: Mapping[str, str],
    adaptation_reserve_mb: int,
    mem_fraction_static: float,
    host: str = "127.0.0.1",
) -> IndustrialExecutionPlan:
    """Render one argv-only launch, then bind it to the execution plan."""

    if len(runtime_plan.rank_configs) != 1:
        raise ValueError("server rendering supports the released one-rank topology")
    config = runtime_plan.rank_configs[0]
    target_id = config.model.target
    drafter_id = config.model.drafter
    required_roots = (
        (target_id,) if config.method == "target_only" else (target_id, drafter_id)
    )
    roots = dict(model_roots)
    for model_id in required_roots:
        root = roots.get(model_id)
        if not isinstance(root, str) or not Path(root).is_dir():
            raise ValueError(f"verified local model root is missing: {model_id}")
    adapted = config.method not in {"target_only", "static"}
    if adapted != (adaptation_reserve_mb > 0):
        raise ValueError(
            "adaptation reserve must be positive exactly for adapted methods"
        )
    if not 0 < mem_fraction_static < 1:
        raise ValueError("mem_fraction_static must lie in (0, 1)")
    verified = verify_patched_checkout(sglang_checkout)
    launch = _render_server(
        output=Path(output_root).resolve(),
        method=config.method,
        config=config,
        verified_checkout=verified,
        roots=roots,
        target_id=target_id,
        drafter_id=drafter_id,
        adaptation_reserve_mb=adaptation_reserve_mb,
        mem_fraction_static=mem_fraction_static,
        host=host,
        port=runtime_plan.cell.resources.ports[0],
    )
    plan = build_industrial_execution_plan(
        runtime_plan=runtime_plan,
        load_plan=load_plan,
        server_launch=launch,
        dependency_receipts=dependency_receipts,
        dependency_artifacts=dependency_artifacts,
        split_artifact=split_artifact,
        sampling_artifact=sampling_artifact,
        model_lock_artifact=model_lock_artifact,
    )
    _immutable_json(
        Path(output_root).resolve() / "industrial-execution-plan.json",
        plan.to_dict(),
    )
    return plan


@dataclass(frozen=True)
class RequestExecution:
    request: BoundServingRequest
    outcome: RequestOutcome
    result: BenchServingResult | None

    def timing(self) -> TimingCoverage:
        output_tokens = self.result.output_tokens if self.result is not None else 0
        admitted = self.outcome.admitted_at_us
        offered = (
            self.outcome.offered_at_us
            if self.outcome.offered_at_us is not None
            else self.request.arrival_us
        )
        if admitted is None or self.result is None or not self.result.chunks:
            return TimingCoverage(
                request_id=self.request.request_id,
                output_tokens=output_tokens,
                expected_itl_intervals=max(0, output_tokens - 1),
                supported_itl_intervals=0,
                coalesced_tokens=output_tokens,
                itl_coverage=1.0 if output_tokens <= 1 else 0.0,
                ttft_us=None,
                supported_itls_us=(),
            )
        terminal = self.outcome.terminal_at_us
        absolute = tuple(
            TokenChunkTiming(
                request_id=chunk.request_id,
                first_token_index=chunk.first_token_index,
                token_count=chunk.token_count,
                chunk_observed_at_us=admitted + chunk.chunk_observed_at_us,
                per_token_observed_at_us=(
                    tuple(admitted + value for value in chunk.per_token_observed_at_us)
                    if chunk.per_token_observed_at_us is not None
                    else None
                ),
            )
            for chunk in self.result.chunks
        )
        if terminal is not None and any(
            chunk.chunk_observed_at_us > terminal for chunk in absolute
        ):
            return TimingCoverage(
                request_id=self.request.request_id,
                output_tokens=output_tokens,
                expected_itl_intervals=max(0, output_tokens - 1),
                supported_itl_intervals=0,
                coalesced_tokens=output_tokens,
                itl_coverage=1.0 if output_tokens <= 1 else 0.0,
                ttft_us=None,
                supported_itls_us=(),
            )
        timing = evaluate_token_timing(
            request_id=self.request.request_id,
            request_started_us=offered,
            expected_output_tokens=output_tokens,
            chunks=absolute,
        )
        if self.result.ttft_us is not None:
            observed_ttft = admitted + self.result.ttft_us - offered
            if timing.ttft_us is not None and timing.ttft_us != observed_ttft:
                raise ValueError("bench chunk timing disagrees with its exact TTFT")
            timing = replace(timing, ttft_us=observed_ttft)
        return timing


@dataclass(frozen=True)
class NativeEvidenceBatch:
    """Server-native rows and aggregate fields unavailable to the HTTP client."""

    rounds: tuple[RoundRecord, ...] = ()
    updates: tuple[UpdateRecord, ...] = ()
    performance_overrides: tuple[tuple[str, object], ...] = ()

    def validate(self, *, run_id: str, method: str) -> None:
        if any(row.run_id != run_id for row in (*self.rounds, *self.updates)):
            raise ValueError("native evidence contains a cross-run row")
        if len({(row.request_id, row.round_index) for row in self.rounds}) != len(
            self.rounds
        ):
            raise ValueError("native round evidence contains duplicate identities")
        if len({(row.cohort_sha256, row.update_index) for row in self.updates}) != len(
            self.updates
        ):
            raise ValueError("native update evidence contains duplicate identities")
        keys = tuple(key for key, _ in self.performance_overrides)
        if len(keys) != len(set(keys)):
            raise ValueError("performance override fields must be unique")
        protected = {
            "run_id",
            "prompt_id",
            "method",
            "repetition_block",
            "region",
            "concurrency",
            "generated_bucket_start",
            "generated_bucket_end",
            "at_risk_requests",
            "output_tokens",
            "elapsed_s",
            "decode_goodput_tps",
            "itl_p50_ms",
            "itl_p95_ms",
            "itl_p99_ms",
            "admission_rejections",
            "timeouts",
            "cancellations",
            "offered_requests",
            "admitted_requests",
            "completed_requests",
            "unfinished_requests",
            "evidence_backpressure_events",
            "evidence_dropped_rows",
        }
        available = {field.name for field in fields(PerformanceRecord)} - protected
        if any(key not in available for key in keys):
            raise ValueError(
                "native performance override targets a protected/unknown field"
            )
        for _, value in self.performance_overrides:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("native performance values must be finite")
        if method in {"target_only", "static"} and (self.rounds or self.updates):
            raise ValueError(
                "allocation-free execution cannot contain round or update rows"
            )
        overrides = dict(self.performance_overrides)
        adaptation_fields = {
            "optimizer_bytes",
            "adaptation_memory_ledger",
            "trainable_parameters",
            "training_cuda_ms",
            "optimizer_cuda_ms",
            "merge_cuda_ms",
            "publish_cuda_ms",
            "barrier_cuda_ms",
            "exposed_update_ms",
            "main_side_overlap_ratio",
            "updates_launched",
            "updates_published",
        }
        if method in {"target_only", "static"} and any(
            overrides.get(name) not in {None, 0} for name in adaptation_fields
        ):
            raise ValueError("allocation-free methods cannot report adaptation state")
        if method not in {"target_only", "static"} and (
            not self.rounds or not self.updates
        ):
            raise ValueError("adapted execution requires native round/update evidence")
        if method not in {"target_only", "static"} and (
            not isinstance(overrides.get("updates_launched"), int)
            or not isinstance(overrides.get("updates_published"), int)
            or int(overrides["updates_launched"]) < 1
            or int(overrides["updates_published"]) < 1
        ):
            raise ValueError("adapted execution requires positive update counters")


@dataclass(frozen=True)
class NativeEvidencePreflight:
    """Machine-readable gate for the native speculative evidence boundary."""

    status: str
    reason_code: str | None
    missing_hook: str | None
    required_fields: tuple[str, ...]

    def validate(self) -> None:
        if self.status == "READY":
            if self.reason_code is not None or self.missing_hook is not None:
                raise ValueError("READY native evidence preflight cannot be blocked")
        elif self.status == "BLOCKED":
            if (
                self.reason_code != MISSING_NATIVE_EVIDENCE_REASON
                or self.missing_hook != NATIVE_TERMINAL_EVIDENCE_HOOK
            ):
                raise ValueError("BLOCKED native evidence preflight is ambiguous")
        else:
            raise ValueError("native evidence preflight status is invalid")
        if self.required_fields != NATIVE_TERMINAL_EVIDENCE_FIELDS:
            raise ValueError("native evidence preflight fields differ from the hook")


class NativeEvidenceUnavailableError(RuntimeError):
    """Raised before mutation when the pinned server lacks exact evidence."""

    def __init__(self, preflight: NativeEvidencePreflight) -> None:
        preflight.validate()
        if preflight.status != "BLOCKED":
            raise ValueError("only a BLOCKED preflight can be unavailable")
        self.preflight = preflight
        fields = ",".join(preflight.required_fields)
        super().__init__(
            f"{preflight.reason_code}: missing hook {preflight.missing_hook}; "
            f"required fields={fields}"
        )


class NativeEvidenceProvider(Protocol):
    native_evidence_hook: str
    patched_sglang_tree: str
    supported_methods: frozenset[str]

    async def collect(
        self,
        *,
        run_id: str,
        plan: IndustrialExecutionPlan,
        requests: tuple[RequestExecution, ...],
        accounting: LoadAccounting,
    ) -> NativeEvidenceBatch: ...


def native_evidence_preflight(
    plan: IndustrialExecutionPlan,
    provider: NativeEvidenceProvider | None,
) -> NativeEvidencePreflight:
    """Block speculative execution unless an exact pinned native hook exists."""

    method = plan.runtime_plan.rank_configs[0].method
    # The current patch exposes no trusted terminal provider.  Caller-authored
    # attributes cannot promote an adapted run to READY; a future concrete
    # provider must replace this gate and validate the full terminal envelope.
    ready = method == "target_only"
    value = NativeEvidencePreflight(
        status="READY" if ready else "BLOCKED",
        reason_code=None if ready else MISSING_NATIVE_EVIDENCE_REASON,
        missing_hook=None if ready else NATIVE_TERMINAL_EVIDENCE_HOOK,
        required_fields=NATIVE_TERMINAL_EVIDENCE_FIELDS,
    )
    value.validate()
    return value


class ServerHandle(Protocol):
    async def wait_ready(self, timeout_s: float) -> None: ...

    async def terminate(self, timeout_s: float) -> None: ...


ServerLauncher = Callable[[ServerLaunch], Awaitable[ServerHandle]]


class AsyncSubprocessServerHandle:
    """Opt-in local process handle for the validated launcher argv."""

    def __init__(
        self,
        *,
        process: asyncio.subprocess.Process,
        base_url: str,
        poll_interval_s: float = 0.1,
    ) -> None:
        if process.pid is None or process.pid < 1:
            raise RuntimeError("server subprocess has no process identity")
        if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
            raise ValueError("server poll interval must be finite and positive")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.port is None
        ):
            raise ValueError("server readiness checks require a loopback HTTP URL")
        self._process = process
        self._base_url = base_url.rstrip("/")
        self._poll_interval_s = poll_interval_s

    def _ready(self) -> bool:
        request = Request(self._base_url + "/v1/models", method="GET")
        try:
            with urlopen(request, timeout=2.0) as response:
                return int(response.status) == 200
        except (HTTPError, URLError, TimeoutError, OSError):
            return False

    async def wait_ready(self, timeout_s: float) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("startup timeout must be finite and positive")
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            if self._process.returncode is not None:
                raise RuntimeError(
                    f"SGLang server exited during startup ({self._process.returncode})"
                )
            if await asyncio.to_thread(self._ready):
                return
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("SGLang server did not become ready")
            await asyncio.sleep(min(self._poll_interval_s, remaining))

    async def terminate(self, timeout_s: float) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("shutdown timeout must be finite and positive")
        if self._process.returncode is not None:
            return
        try:
            self._process.terminate()
        except ProcessLookupError:
            await self._process.wait()
            return
        try:
            await asyncio.wait_for(self._process.wait(), timeout=timeout_s)
        except TimeoutError:
            try:
                self._process.kill()
            except ProcessLookupError:
                pass
            await self._process.wait()


async def launch_server_subprocess(launch: ServerLaunch) -> ServerHandle:
    """Explicitly opt into launching the already-validated local server argv."""

    parsed = urlsplit(launch.base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or not launch.argv
        or launch.argv[:3]
        != (sys.executable, "-m", "lightcone_spec.sglang_bridge.launch")
    ):
        raise ValueError("subprocess launch requires the registered loopback launcher")
    config_path = Path(launch.run_config)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        device_identity = config["runtime"]["device_identity"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("subprocess launch lacks a device-bound RunConfig") from error
    if (
        not isinstance(device_identity, str)
        or not device_identity.startswith("GPU-")
        or "\n" in device_identity
        or "," in device_identity
    ):
        raise ValueError("subprocess launch requires exactly one GPU UUID")
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = device_identity
    process = await asyncio.create_subprocess_exec(*launch.argv, env=environment)
    return AsyncSubprocessServerHandle(process=process, base_url=launch.base_url)


@dataclass(frozen=True)
class ExecutionClock:
    monotonic_ns: Callable[[], int] = time.perf_counter_ns
    wall_ns: Callable[[], int] = time.time_ns
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


@dataclass(frozen=True)
class IndustrialExecutionResult:
    run_id: str
    execution_plan_sha256: str
    rank_config_sha256: str
    topology_sha256: str
    resumed: bool
    terminal_receipt: str
    evidence_files: tuple[str, ...]
    accounting: LoadAccounting | None


def industrial_run_id(plan: IndustrialExecutionPlan, run_nonce_sha256: str) -> str:
    if not _is_sha256(run_nonce_sha256):
        raise ValueError("run nonce must be a lowercase SHA-256")
    return (
        "industrial-"
        + content_sha256(
            {
                "schema_version": 1,
                "execution_plan_sha256": plan.sha256,
                "run_nonce_sha256": run_nonce_sha256,
                "rank": 0,
            }
        )[:48]
    )


async def _sleep_until(
    target_us: int,
    *,
    origin_ns: int,
    clock: ExecutionClock,
) -> None:
    remaining = target_us - (clock.monotonic_ns() - origin_ns) // 1000
    if remaining > 0:
        await clock.sleep(remaining / 1_000_000)


def _logical_now_us(*, origin_ns: int, clock: ExecutionClock) -> int:
    return max(0, (clock.monotonic_ns() - origin_ns) // 1000)


async def _execute_request(
    request: BoundServingRequest,
    *,
    deadline_us: int,
    origin_ns: int,
    semaphore: asyncio.Semaphore,
    transport: BenchServingTransport,
    base_url: str,
    served_model: str,
    abort_grace_s: float,
    clock: ExecutionClock,
    offered_at_us: int | None = None,
) -> RequestExecution:
    arrival_us = request.arrival_us if offered_at_us is None else offered_at_us
    await _sleep_until(arrival_us, origin_ns=origin_ns, clock=clock)
    cancel_us = (
        arrival_us + request.cancellation_offset_us
        if request.cancellation_offset_us is not None
        else None
    )
    boundary = min(deadline_us, cancel_us) if cancel_us is not None else deadline_us
    if cancel_us is not None and cancel_us <= arrival_us:
        return RequestExecution(
            request=request,
            outcome=RequestOutcome(
                request_id=request.request_id,
                status="cancelled",
                admitted_at_us=None,
                terminal_at_us=cancel_us,
                code="scheduled_cancellation_before_admission",
                offered_at_us=arrival_us,
            ),
            result=None,
        )

    acquired = False
    acquire_task = asyncio.create_task(semaphore.acquire())
    try:
        remaining = boundary - _logical_now_us(origin_ns=origin_ns, clock=clock)
        if remaining <= 0:
            acquire_task.cancel()
        else:
            done, _ = await asyncio.wait({acquire_task}, timeout=remaining / 1_000_000)
            acquired = acquire_task in done and not acquire_task.cancelled()
        if not acquired:
            acquire_task.cancel()
            await asyncio.gather(acquire_task, return_exceptions=True)
            cancelled = cancel_us is not None and cancel_us <= deadline_us
            return RequestExecution(
                request=request,
                outcome=RequestOutcome(
                    request_id=request.request_id,
                    status="cancelled" if cancelled else "rejected",
                    admitted_at_us=None,
                    terminal_at_us=cancel_us if cancelled else deadline_us,
                    code=(
                        "scheduled_cancellation_before_admission"
                        if cancelled
                        else "admission_deadline"
                    ),
                    offered_at_us=arrival_us,
                ),
                result=None,
            )
        admitted_us = max(
            arrival_us,
            _logical_now_us(origin_ns=origin_ns, clock=clock),
        )
        if admitted_us > boundary:
            cancelled = cancel_us is not None and cancel_us <= deadline_us
            return RequestExecution(
                request=request,
                outcome=RequestOutcome(
                    request_id=request.request_id,
                    status="cancelled" if cancelled else "rejected",
                    admitted_at_us=None,
                    terminal_at_us=cancel_us if cancelled else deadline_us,
                    code=(
                        "scheduled_cancellation_before_admission"
                        if cancelled
                        else "admission_deadline"
                    ),
                    offered_at_us=arrival_us,
                ),
                result=None,
            )
        submit_task = asyncio.create_task(
            transport.submit(
                request,
                base_url=base_url,
                served_model=served_model,
            )
        )
        try:
            remaining = max(
                0,
                boundary - _logical_now_us(origin_ns=origin_ns, clock=clock),
            )
            done, _ = await asyncio.wait({submit_task}, timeout=remaining / 1_000_000)
            if submit_task in done:
                try:
                    result = submit_task.result()
                    result.validate(request)
                except Exception as error:  # noqa: BLE001 - terminal request boundary
                    return RequestExecution(
                        request=request,
                        outcome=RequestOutcome(
                            request_id=request.request_id,
                            status="unfinished",
                            admitted_at_us=admitted_us,
                            terminal_at_us=None,
                            code=f"bench_transport_or_result_error:{type(error).__name__}",
                            offered_at_us=arrival_us,
                        ),
                        result=None,
                    )
                terminal = _logical_now_us(origin_ns=origin_ns, clock=clock)
                if terminal <= boundary:
                    if not result.success:
                        return RequestExecution(
                            request=request,
                            outcome=RequestOutcome(
                                request_id=request.request_id,
                                status="unfinished",
                                admitted_at_us=admitted_us,
                                terminal_at_us=None,
                                code=f"official_bench_error:{result.error_code}",
                                offered_at_us=arrival_us,
                            ),
                            result=result,
                        )
                    return RequestExecution(
                        request=request,
                        outcome=RequestOutcome(
                            request_id=request.request_id,
                            status="completed",
                            admitted_at_us=admitted_us,
                            terminal_at_us=max(admitted_us, terminal),
                            code="completed",
                            offered_at_us=arrival_us,
                        ),
                        result=result,
                    )
            try:
                await transport.abort(request.request_id, base_url=base_url)
            except Exception as error:  # noqa: BLE001 - transport boundary
                submit_task.cancel()
                await asyncio.gather(submit_task, return_exceptions=True)
                return RequestExecution(
                    request=request,
                    outcome=RequestOutcome(
                        request_id=request.request_id,
                        status="unfinished",
                        admitted_at_us=admitted_us,
                        terminal_at_us=None,
                        code=f"abort_transport_error:{type(error).__name__}",
                        offered_at_us=arrival_us,
                    ),
                    result=None,
                )
            try:
                result = await asyncio.wait_for(submit_task, timeout=abort_grace_s)
            except TimeoutError:
                submit_task.cancel()
                await asyncio.gather(submit_task, return_exceptions=True)
                return RequestExecution(
                    request=request,
                    outcome=RequestOutcome(
                        request_id=request.request_id,
                        status="unfinished",
                        admitted_at_us=admitted_us,
                        terminal_at_us=None,
                        code="abort_terminal_evidence_timeout",
                        offered_at_us=arrival_us,
                    ),
                    result=None,
                )
            try:
                result.validate(request)
            except Exception as error:  # noqa: BLE001 - terminal request boundary
                return RequestExecution(
                    request=request,
                    outcome=RequestOutcome(
                        request_id=request.request_id,
                        status="unfinished",
                        admitted_at_us=admitted_us,
                        terminal_at_us=None,
                        code=f"aborted_result_invalid:{type(error).__name__}",
                        offered_at_us=arrival_us,
                    ),
                    result=None,
                )
            cancelled = cancel_us is not None and cancel_us <= deadline_us
            return RequestExecution(
                request=request,
                outcome=RequestOutcome(
                    request_id=request.request_id,
                    status="cancelled" if cancelled else "timed_out",
                    admitted_at_us=admitted_us,
                    terminal_at_us=cancel_us if cancelled else deadline_us,
                    code="scheduled_cancellation" if cancelled else "request_deadline",
                    offered_at_us=arrival_us,
                ),
                result=result,
            )
        except BaseException:
            if not submit_task.done():
                submit_task.cancel()
                await asyncio.gather(submit_task, return_exceptions=True)
            raise
    finally:
        if not acquire_task.done():
            acquire_task.cancel()
            await asyncio.gather(acquire_task, return_exceptions=True)
        elif (
            not acquired
            and not acquire_task.cancelled()
            and acquire_task.exception() is None
            and bool(acquire_task.result())
        ):
            semaphore.release()
        if acquired:
            semaphore.release()


async def _execute_corpus(
    requests: tuple[BoundServingRequest, ...],
    *,
    deadline_for: Callable[[BoundServingRequest], int],
    concurrency: int,
    transport: BenchServingTransport,
    base_url: str,
    served_model: str,
    abort_grace_s: float,
    clock: ExecutionClock,
    on_terminal: Callable[[RequestExecution], Awaitable[None]] | None = None,
) -> tuple[RequestExecution, ...]:
    if not requests:
        return ()
    if len(requests) > MAX_IN_MEMORY_REQUEST_EXECUTIONS:
        raise ValueError("request corpus exceeds the bounded execution capacity")
    origin_ns = clock.monotonic_ns()
    semaphore = asyncio.Semaphore(concurrency)
    completed: dict[str, RequestExecution] = {}
    errors: list[Exception] = []

    async def run_one(request: BoundServingRequest) -> None:
        try:
            execution = await _execute_request(
                request,
                deadline_us=deadline_for(request),
                origin_ns=origin_ns,
                semaphore=semaphore,
                transport=transport,
                base_url=base_url,
                served_model=served_model,
                abort_grace_s=abort_grace_s,
                clock=clock,
            )
            completed[request.request_id] = execution
            if on_terminal is not None:
                await on_terminal(execution)
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(error)

    async with asyncio.TaskGroup() as group:
        for request in requests:
            group.create_task(run_one(request))
    if errors:
        for sibling in errors[1:]:
            errors[0].add_note(f"sibling request also failed: {sibling}")
        raise errors[0]
    if set(completed) != {request.request_id for request in requests}:
        raise RuntimeError("request execution coverage is incomplete")
    return tuple(completed[request.request_id] for request in requests)


async def _execute_closed_loop_corpus(
    requests: tuple[BoundServingRequest, ...],
    *,
    concurrency: int,
    arrival_duration_us: int,
    request_deadline_us: int,
    scored_global_end_us: int,
    transport: BenchServingTransport,
    base_url: str,
    served_model: str,
    abort_grace_s: float,
    clock: ExecutionClock,
    on_terminal: Callable[[RequestExecution], Awaitable[None]],
) -> tuple[RequestExecution, ...]:
    """Issue one zero-think sequential request stream per client lane."""

    if not requests:
        return ()
    if len(requests) > MAX_IN_MEMORY_REQUEST_EXECUTIONS:
        raise ValueError("request corpus exceeds the bounded execution capacity")
    origin_ns = clock.monotonic_ns()
    semaphore = asyncio.Semaphore(concurrency)
    completed: dict[str, RequestExecution] = {}
    exhausted_lanes: set[int] = set()
    errors: list[Exception] = []
    stop_issuing = asyncio.Event()

    async def run_lane(lane: int) -> None:
        next_offer_us = 0
        for request in requests[lane::concurrency]:
            if stop_issuing.is_set() or next_offer_us >= arrival_duration_us:
                return
            try:
                execution = await _execute_request(
                    request,
                    deadline_us=min(
                        next_offer_us + request_deadline_us,
                        scored_global_end_us,
                    ),
                    origin_ns=origin_ns,
                    semaphore=semaphore,
                    transport=transport,
                    base_url=base_url,
                    served_model=served_model,
                    abort_grace_s=abort_grace_s,
                    clock=clock,
                    offered_at_us=next_offer_us,
                )
                completed[request.request_id] = execution
                await on_terminal(execution)
            except (OSError, RuntimeError, ValueError) as error:
                errors.append(error)
                stop_issuing.set()
                return
            if execution.outcome.terminal_at_us is None:
                return
            next_offer_us = execution.outcome.terminal_at_us
        if not stop_issuing.is_set() and next_offer_us < arrival_duration_us:
            exhausted_lanes.add(lane)

    async with asyncio.TaskGroup() as group:
        for lane in range(concurrency):
            group.create_task(run_lane(lane))
    if errors:
        for sibling in errors[1:]:
            errors[0].add_note(f"sibling client lane also failed: {sibling}")
        raise errors[0]
    if exhausted_lanes:
        raise RuntimeError(
            "closed-loop request pool exhausted before the arrival window ended"
        )
    ordered = tuple(
        completed[request.request_id]
        for request in requests
        if request.request_id in completed
    )
    if not ordered:
        raise RuntimeError("closed-loop arrival window offered no requests")
    return ordered


def _timing_timestamps(
    execution: RequestExecution,
    *,
    score_started_ns: int,
) -> tuple[int, ...] | None:
    result = execution.result
    admitted = execution.outcome.admitted_at_us
    if result is None or admitted is None:
        return None
    values: list[int] = []
    for chunk in result.chunks:
        if chunk.per_token_observed_at_us is not None:
            relative = chunk.per_token_observed_at_us
        elif chunk.token_count == 1:
            relative = (chunk.chunk_observed_at_us,)
        else:
            return None
        values.extend(
            score_started_ns + (admitted + value) * 1000 for value in relative
        )
    return tuple(values) if len(values) == result.output_tokens else None


def _request_record(
    execution: RequestExecution,
    *,
    run_id: str,
    method: str,
    block: int,
    concurrency: int,
    score_started_ns: int,
) -> RequestRecord:
    timing = execution.timing()
    outcome = execution.outcome
    offered_at_us = (
        outcome.offered_at_us
        if outcome.offered_at_us is not None
        else execution.request.arrival_us
    )
    result = execution.result
    output_tokens = result.output_tokens if result is not None else 0
    output = result.generated_text if result is not None else ""
    token_ids = (
        result.generated_token_ids
        if result is not None
        else ()
        if execution.outcome.status in {"rejected", "cancelled"}
        else None
    )
    token_ids_body = (
        json.dumps(token_ids, separators=(",", ":")) if token_ids is not None else None
    )
    output_sha256 = (
        hashlib.sha256(token_ids_body.encode("utf-8")).hexdigest()
        if token_ids_body is not None
        else hashlib.sha256(output.encode("utf-8")).hexdigest()
    )
    timestamps = _timing_timestamps(execution, score_started_ns=score_started_ns)
    return RequestRecord(
        run_id=run_id,
        request_id=execution.request.request_id,
        prompt_id=f"{execution.request.namespace}:{execution.request.ordinal}",
        method=method,
        repetition_block=block,
        concurrency=concurrency,
        input_tokens=len(execution.request.input_token_ids),
        output_tokens=output_tokens,
        output_hash_format=OUTPUT_HASH_FORMAT,
        output_sha256=output_sha256,
        ttft_ms=(timing.ttft_us / 1000 if timing.ttft_us is not None else None),
        finished=outcome.status == "completed",
        stop_reason=(
            result.stop_reason
            if outcome.status == "completed" and result
            else outcome.code
        ),
        output_token_ids=token_ids_body,
        output_token_ids_sha256=(output_sha256 if token_ids_body is not None else None),
        outcome_status=outcome.status,
        arrival_ns=score_started_ns + offered_at_us * 1000,
        queue_enter_ns=score_started_ns + offered_at_us * 1000,
        admitted_ns=(
            score_started_ns + outcome.admitted_at_us * 1000
            if outcome.admitted_at_us is not None
            else None
        ),
        first_token_ns=(
            score_started_ns + offered_at_us * 1000 + timing.ttft_us * 1000
            if timing.ttft_us is not None
            else None
        ),
        completed_ns=(
            score_started_ns + outcome.terminal_at_us * 1000
            if outcome.terminal_at_us is not None
            else None
        ),
        token_timestamps_ns=(
            json.dumps(timestamps, separators=(",", ":")) if timestamps else None
        ),
        inter_token_ms=(
            json.dumps(
                [value / 1000 for value in timing.supported_itls_us],
                separators=(",", ":"),
            )
            if timing.full_itl_coverage and timing.supported_itls_us
            else None
        ),
        token_timing_coverage=timing.itl_coverage,
        coalesced_intervals=(
            timing.expected_itl_intervals - timing.supported_itl_intervals
        ),
        admission_code="admitted"
        if outcome.admitted_at_us is not None
        else outcome.code,
        cancellation_code=outcome.code if outcome.status == "cancelled" else None,
        error_code=(
            outcome.code
            if outcome.status in {"rejected", "timed_out", "unfinished"}
            else None
        ),
        cohort_sha256=execution.request.cohort_sha256,
        route_id=execution.request.route_id,
    )


def _percentile(values: Sequence[int], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _performance_record(
    *,
    run_id: str,
    plan: IndustrialExecutionPlan,
    requests: tuple[RequestExecution, ...],
    accounting: LoadAccounting,
    native: NativeEvidenceBatch,
    writer: EvidenceWriter,
    asynchronous_backpressure_events: int = 0,
) -> PerformanceRecord:
    if accounting.elapsed_us <= 0:
        raise RuntimeError("scored load interval must have positive measured duration")
    completed = tuple(row for row in requests if row.outcome.status == "completed")
    output_tokens = sum(
        row.result.output_tokens for row in completed if row.result is not None
    )
    eligible_timings = tuple(
        row.timing()
        for row in completed
        if row.result is not None and row.result.output_tokens > 1
    )
    claimable = bool(eligible_timings) and all(
        timing.full_itl_coverage for timing in eligible_timings
    )
    itls = tuple(
        value for timing in eligible_timings for value in timing.supported_itls_us
    )
    if claimable and not itls:
        claimable = False
    p50 = _percentile(itls, 0.50) / 1000 if claimable else None
    p95 = _percentile(itls, 0.95) / 1000 if claimable else None
    p99 = _percentile(itls, 0.99) / 1000 if claimable else None
    identity = plan.runtime_plan.cell.identity
    allocation_free = identity.method in {"target_only", "static"}
    structurally_non_speculative = identity.method == "target_only"
    base = PerformanceRecord(
        run_id=run_id,
        prompt_id=plan.load_plan.scored.hashes.corpus_sha256,
        method=identity.method,
        repetition_block=identity.block,
        region="scored_window",
        concurrency=int(identity.concurrency),
        generated_bucket_start=0,
        generated_bucket_end=max(
            request.requested_output_tokens for request in plan.scored_requests
        ),
        at_risk_requests=accounting.admitted,
        output_tokens=output_tokens,
        elapsed_s=accounting.elapsed_us / 1_000_000,
        decode_goodput_tps=output_tokens / (accounting.elapsed_us / 1_000_000),
        itl_p50_ms=p50,
        itl_p95_ms=p95,
        itl_p99_ms=p99,
        survival_weighted_accepted_prefix=None,
        accepted_drafts_per_verify=None,
        committed_tokens_per_verify=None,
        verified_drafts_per_verify=None,
        verification_waste=None,
        target_calls_per_output_token=None,
        batch_fill=None,
        queue_occupancy=None,
        gpu_busy=None,
        sm_utilization=None,
        dram_utilization=None,
        target_estimated_mfu=None,
        peak_hbm_bytes=None,
        kv_bytes=None,
        optimizer_bytes=0 if allocation_free else None,
        adaptation_memory_ledger=None,
        trainable_parameters=0 if allocation_free else None,
        training_cuda_ms=None,
        optimizer_cuda_ms=None,
        merge_cuda_ms=None,
        publish_cuda_ms=None,
        barrier_cuda_ms=None,
        exposed_update_ms=None,
        main_side_overlap_ratio=None,
        graph_replay_hit_rate=None,
        updates_launched=0 if allocation_free else None,
        updates_published=0 if allocation_free else None,
        exactness_violations=0 if structurally_non_speculative else None,
        version_mismatches=0 if structurally_non_speculative else None,
        fallbacks=0 if structurally_non_speculative else None,
        nonfinite_updates=0 if structurally_non_speculative else None,
        oom_events=0 if structurally_non_speculative else None,
        retractions=0 if structurally_non_speculative else None,
        communicator_failures=0 if structurally_non_speculative else None,
        admission_rejections=accounting.rejected,
        timeouts=accounting.timed_out,
        cancellations=accounting.cancelled,
        offered_requests=accounting.offered,
        admitted_requests=accounting.admitted,
        completed_requests=accounting.completed,
        unfinished_requests=accounting.unfinished,
        evidence_backpressure_events=(
            writer.backpressure_events + asynchronous_backpressure_events
        ),
        evidence_dropped_rows=writer.dropped_rows,
    )
    return replace(base, **dict(native.performance_overrides))


def _workload_contract(method: str) -> str:
    if method == "target_only":
        return "industrial_target_only"
    if method == "static":
        return "industrial_static"
    return "industrial_adapted"


def _validate_resume(
    *,
    completed: dict[str, Path],
    run_id: str,
    plan: IndustrialExecutionPlan,
    run_nonce_sha256: str,
) -> None:
    run_rows = pq.read_table(completed["run"]).to_pylist()
    request_rows = pq.read_table(completed["request"]).to_pylist()
    performance_rows = pq.read_table(completed["performance"]).to_pylist()
    config = plan.runtime_plan.rank_configs[0]
    expected = {
        "run_id": run_id,
        "manifest_sha256": plan.runtime_plan.registry_sha256,
        "config_sha256": plan.runtime_plan.cell_id,
        "rank_config_sha256": plan.rank_config_sha256,
        "method": config.method,
        "industrial_cell_id": plan.runtime_plan.cell_id,
        "runtime_sha256": plan.sha256,
        "split_sha256": plan.split_artifact.content_sha256,
        "corpus_sha256": plan.load_plan.scored.hashes.corpus_sha256,
        "arrival_trace_sha256": plan.load_plan.scored.hashes.arrivals_sha256,
        "request_ids_sha256": plan.load_plan.scored.hashes.request_ids_sha256,
        "sampling_profile_sha256": plan.sampling_artifact.content_sha256,
        "model_lock_sha256": plan.model_lock_artifact.content_sha256,
        "patched_sglang_tree": plan.patched_sglang_tree,
        "run_nonce_sha256": run_nonce_sha256,
        "topology_sha256": plan.topology_sha256,
        "tensor_parallel_size": config.runtime.tensor_parallel_size,
        "data_parallel_size": config.runtime.data_parallel_size,
        "world_size": 1,
        "rank": 0,
        "expected_request_rows": len(request_rows),
        "expected_performance_rows": 1,
        "workload_contract": _workload_contract(config.method),
        "status": "complete",
    }
    if len(run_rows) != 1 or any(
        run_rows[0].get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("completed industrial receipt has a mismatched run identity")
    for table in ("request", "round", "update", "performance"):
        actual_rows = (
            pq.ParquetFile(completed[table]).metadata.num_rows
            if table in completed
            else 0
        )
        if run_rows[0].get(f"expected_{table}_rows") != actual_rows:
            raise RuntimeError(
                "completed industrial receipt has mismatched table coverage"
            )
    expected_pool = {request.request_id: request for request in plan.scored_requests}
    actual_ids = {str(row.get("request_id")) for row in request_rows}
    if plan.load_plan.scored.source_kind == "closed_loop":
        concurrency = config.runtime.max_running_requests
        if not actual_ids or actual_ids - set(expected_pool):
            raise RuntimeError(
                "completed closed-loop receipt has foreign request identities"
            )
        for lane in range(concurrency):
            seen_gap = False
            for request in plan.scored_requests[lane::concurrency]:
                present = request.request_id in actual_ids
                if seen_gap and present:
                    raise RuntimeError(
                        "completed closed-loop receipt is not a per-client prefix"
                    )
                seen_gap = seen_gap or not present
    elif actual_ids != set(expected_pool):
        raise RuntimeError(
            "completed industrial receipt has mismatched request coverage"
        )
    expected_requests = {
        request_id: expected_pool[request_id] for request_id in actual_ids
    }
    expected_ids = set(expected_requests)
    if (
        len(request_rows) != len(expected_ids)
        or actual_ids != expected_ids
        or any(row.get("method") != config.method for row in request_rows)
    ):
        raise RuntimeError(
            "completed industrial receipt has mismatched request coverage"
        )
    statuses = {
        "rejected",
        "completed",
        "timed_out",
        "cancelled",
        "unfinished",
    }
    status_counts = {status: 0 for status in statuses}
    admitted = 0
    completed_output_tokens = 0
    for row in request_rows:
        request = expected_requests[str(row["request_id"])]
        status = row.get("outcome_status")
        if status not in statuses or row.get("finished") is not (status == "completed"):
            raise RuntimeError(
                "completed industrial receipt has an invalid request outcome"
            )
        status_counts[str(status)] += 1
        if row.get("admitted_ns") is not None:
            admitted += 1
        output_tokens = row.get("output_tokens")
        expected_identity = {
            "prompt_id": f"{request.namespace}:{request.ordinal}",
            "repetition_block": plan.runtime_plan.cell.identity.block,
            "concurrency": config.runtime.max_running_requests,
            "input_tokens": len(request.input_token_ids),
            "cohort_sha256": request.cohort_sha256,
            "route_id": request.route_id,
        }
        if (
            any(row.get(key) != value for key, value in expected_identity.items())
            or isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or not 0 <= output_tokens <= request.requested_output_tokens
            or (status == "completed" and output_tokens < 1)
            or (status == "rejected" and row.get("admitted_ns") is not None)
        ):
            raise RuntimeError(
                "completed industrial receipt has a mismatched request binding"
            )
        if status == "completed":
            completed_output_tokens += output_tokens
    if len(performance_rows) != 1:
        raise RuntimeError(
            "completed industrial receipt lacks exact performance coverage"
        )
    performance = performance_rows[0]
    performance_expected = {
        "run_id": run_id,
        "prompt_id": plan.load_plan.scored.hashes.corpus_sha256,
        "method": config.method,
        "repetition_block": plan.runtime_plan.cell.identity.block,
        "region": "scored_window",
        "concurrency": config.runtime.max_running_requests,
        "at_risk_requests": admitted,
        "output_tokens": completed_output_tokens,
        "offered_requests": len(expected_ids),
        "admitted_requests": admitted,
        "completed_requests": status_counts["completed"],
        "unfinished_requests": status_counts["unfinished"],
        "admission_rejections": status_counts["rejected"],
        "timeouts": status_counts["timed_out"],
        "cancellations": status_counts["cancelled"],
    }
    if any(
        performance.get(key) != value for key, value in performance_expected.items()
    ):
        raise RuntimeError(
            "completed industrial receipt has mismatched load accounting"
        )


async def execute_industrial_plan(
    plan: IndustrialExecutionPlan,
    *,
    output_root: str | Path,
    run_nonce_sha256: str,
    launch_server: ServerLauncher,
    transport: BenchServingTransport,
    native_evidence: NativeEvidenceProvider | None = None,
    clock: ExecutionClock | None = None,
) -> IndustrialExecutionResult:
    """Execute one plan; no process or network action occurs without injection."""

    plan.validate()
    method = plan.runtime_plan.rank_configs[0].method
    if method == "target_only" and native_evidence is not None:
        raise ValueError("Target-only cannot inject native speculative evidence")
    evidence_preflight = native_evidence_preflight(plan, native_evidence)
    if evidence_preflight.status == "BLOCKED":
        raise NativeEvidenceUnavailableError(evidence_preflight)
    if clock is None:
        clock = ExecutionClock()
    run_id = industrial_run_id(plan, run_nonce_sha256)
    root = Path(output_root).resolve()
    registered_root = Path(plan.runtime_plan.cell.resources.evidence_root).resolve()
    if root != registered_root:
        raise ValueError("output_root differs from the registry evidence reservation")
    root.mkdir(parents=True, exist_ok=True)
    completed = load_completed_evidence(root, run_id=run_id, rank=0)
    terminal_receipt = root / f"{run_id}.rank0.complete.json"
    if completed is not None:
        _validate_resume(
            completed=completed,
            run_id=run_id,
            plan=plan,
            run_nonce_sha256=run_nonce_sha256,
        )
        return IndustrialExecutionResult(
            run_id=run_id,
            execution_plan_sha256=plan.sha256,
            rank_config_sha256=plan.rank_config_sha256,
            topology_sha256=plan.topology_sha256,
            resumed=True,
            terminal_receipt=str(terminal_receipt),
            evidence_files=tuple(str(path) for path in completed.values()),
            accounting=None,
        )

    writer = EvidenceWriter(root, run_id=run_id, rank=0)
    sink = _AsyncEvidenceSink(writer)
    handle: ServerHandle | None = None
    try:
        handle = await launch_server(plan.server_launch)
        await handle.wait_ready(plan.startup_timeout_s)
        config = plan.runtime_plan.rank_configs[0]
        concurrency = config.runtime.max_running_requests
        if plan.warmup_requests:
            warmup_end = plan.load_plan.window.warmup_duration_us
            warmup = await _execute_corpus(
                plan.warmup_requests,
                deadline_for=lambda request: min(
                    request.arrival_us + plan.load_plan.window.request_deadline_us,
                    warmup_end,
                ),
                concurrency=concurrency,
                transport=transport,
                base_url=plan.server_launch.base_url,
                served_model=config.model.target,
                abort_grace_s=plan.abort_grace_s,
                clock=clock,
            )
            if any(row.outcome.status != "completed" for row in warmup):
                raise RuntimeError("excluded warm-up did not complete exactly")
        score_started_ns = clock.wall_ns()

        async def record_terminal(execution: RequestExecution) -> None:
            await sink.write(
                _request_record(
                    execution,
                    run_id=run_id,
                    method=config.method,
                    block=plan.runtime_plan.cell.identity.block,
                    concurrency=concurrency,
                    score_started_ns=score_started_ns,
                )
            )

        if plan.load_plan.scored.source_kind == "closed_loop":
            requests = await _execute_closed_loop_corpus(
                plan.scored_requests,
                concurrency=concurrency,
                arrival_duration_us=plan.load_plan.window.arrival_duration_us,
                request_deadline_us=plan.load_plan.window.request_deadline_us,
                scored_global_end_us=plan.load_plan.window.scored_global_end_us,
                transport=transport,
                base_url=plan.server_launch.base_url,
                served_model=config.model.target,
                abort_grace_s=plan.abort_grace_s,
                clock=clock,
                on_terminal=record_terminal,
            )
        else:
            requests = await _execute_corpus(
                plan.scored_requests,
                deadline_for=lambda request: plan.load_plan.window.request_timeout_us(
                    next(
                        value
                        for value in plan.load_plan.scored.requests
                        if value.request_id == request.request_id
                    )
                ),
                concurrency=concurrency,
                transport=transport,
                base_url=plan.server_launch.base_url,
                served_model=config.model.target,
                abort_grace_s=plan.abort_grace_s,
                clock=clock,
                on_terminal=record_terminal,
            )
        accounting = account_scored_requests(
            plan.load_plan,
            tuple(row.outcome for row in requests),
        )
        if native_evidence is None:
            if config.method != "target_only":
                raise RuntimeError(
                    "speculative serving requires trusted native terminal evidence"
                )
            native = NativeEvidenceBatch()
        else:
            native = await native_evidence.collect(
                run_id=run_id,
                plan=plan,
                requests=requests,
                accounting=accounting,
            )
        native.validate(run_id=run_id, method=config.method)
        completed_request_ids = {
            row.request.request_id
            for row in requests
            if row.outcome.status == "completed"
        }
        if native.rounds and not completed_request_ids <= {
            row.request_id for row in native.rounds
        }:
            raise RuntimeError("native round evidence misses a completed request")
        await sink.flush()
        completed_ns = clock.wall_ns()
        for row in native.rounds:
            await sink.write(row)
        for row in native.updates:
            await sink.write(row)
        await sink.write(
            _performance_record(
                run_id=run_id,
                plan=plan,
                requests=requests,
                accounting=accounting,
                native=native,
                writer=writer,
                asynchronous_backpressure_events=sink.backpressure_events,
            )
        )
        await sink.write(
            RunRecord(
                run_id=run_id,
                manifest_sha256=plan.runtime_plan.registry_sha256,
                config_sha256=plan.runtime_plan.cell_id,
                rank_config_sha256=plan.rank_config_sha256,
                method=config.method,
                model_pair=config.model.key,
                repetition_block=plan.runtime_plan.cell.identity.block,
                started_ns=score_started_ns,
                completed_ns=completed_ns,
                status="aborted" if accounting.unfinished else "complete",
                industrial_cell_id=plan.runtime_plan.cell_id,
                runtime_sha256=plan.sha256,
                split_sha256=plan.split_artifact.content_sha256,
                corpus_sha256=plan.load_plan.scored.hashes.corpus_sha256,
                arrival_trace_sha256=plan.load_plan.scored.hashes.arrivals_sha256,
                request_ids_sha256=plan.load_plan.scored.hashes.request_ids_sha256,
                sampling_profile_sha256=plan.sampling_artifact.content_sha256,
                model_lock_sha256=plan.model_lock_artifact.content_sha256,
                patched_sglang_tree=plan.patched_sglang_tree,
                run_nonce_sha256=run_nonce_sha256,
                topology_sha256=plan.topology_sha256,
                tensor_parallel_size=config.runtime.tensor_parallel_size,
                data_parallel_size=config.runtime.data_parallel_size,
                world_size=1,
                rank=0,
                expected_request_rows=len(requests),
                expected_round_rows=len(native.rounds),
                expected_update_rows=len(native.updates),
                expected_performance_rows=1,
                workload_contract=_workload_contract(config.method),
                preflight_attestation_sha256=None,
            )
        )
        await handle.terminate(plan.shutdown_timeout_s)
        handle = None
        await sink.close()
        if accounting.unfinished:
            writer.abort(reason="scored requests contain unfinished outcomes")
            raise RuntimeError(
                "scored requests contain unfinished outcomes; evidence is nonclaimable"
            )
        written = writer.close()
        return IndustrialExecutionResult(
            run_id=run_id,
            execution_plan_sha256=plan.sha256,
            rank_config_sha256=plan.rank_config_sha256,
            topology_sha256=plan.topology_sha256,
            resumed=False,
            terminal_receipt=str(terminal_receipt),
            evidence_files=tuple(str(path) for path in written.values()),
            accounting=accounting,
        )
    except BaseException as error:
        if handle is not None:
            try:
                await handle.terminate(plan.shutdown_timeout_s)
            except (OSError, RuntimeError, TimeoutError) as shutdown_error:
                error.add_note(f"server shutdown also failed: {shutdown_error}")
        try:
            await sink.close()
        except (OSError, RuntimeError, ValueError) as sink_error:
            error.add_note(f"background evidence shutdown also failed: {sink_error}")
        if sink.backpressure_events:
            try:
                writer.register_external_backpressure_events(sink.backpressure_events)
            except (OSError, RuntimeError, ValueError):
                pass
        try:
            writer.abort(reason=f"{type(error).__name__}: {error}")
        except (OSError, RuntimeError):
            pass
        raise
