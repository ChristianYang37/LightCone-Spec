from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
from pathlib import Path

import pytest

from lightcone_spec.orchestration.formal_serving_session_source_chain import (
    FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_EVIDENCE_LEVEL,
    FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_MANIFEST,
    FormalServingResidentSourceChainPublisher,
    revalidate_formal_serving_resident_source_chain,
)
from lightcone_spec.orchestration.formal_terminal_shards import (
    publish_scalable_native_terminal_artifact,
)
from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalProvider,
    NativeTerminalRunBinding,
    canonical_json_bytes,
    canonical_sha256,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding


def _load_fixture(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TERMINAL = _load_fixture(
    "_resident_source_terminal_fixture", "test_native_terminal_provider.py"
)
_SOURCE = _load_fixture(
    "_resident_source_authority_fixture", "test_session_reuse_authority.py"
)


def _sha(label: str) -> str:
    return canonical_sha256(label)


class _ResidentTerminalTransport(_TERMINAL.FakeAdminTransport):
    """Fixture transport with the real cross-trace reset-generation lineage."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._next_begin_generation = 1

    def _begin(self, payload):
        value = super()._begin(payload)
        value["reset_generation"] = self._next_begin_generation
        value.pop("begin_sha256")
        value["begin_sha256"] = canonical_sha256(value)
        return value

    def _reset(self, payload):
        value = super()._reset(payload)
        assert self.begin_receipt is not None
        value["reset_generation"] = self.begin_receipt["reset_generation"] + 1
        value.pop("reset_sha256")
        value["reset_sha256"] = canonical_sha256(value)
        self._next_begin_generation = value["reset_generation"] + 1
        return value


def _terminal(
    *,
    output_dir: Path,
    session_plan_sha256: str,
    execution_plan_sha256: str,
    epoch_index: int,
    previous_run_id: str | None,
    transport=None,
    provider=None,
):
    warmup_id = f"warm-{epoch_index}"
    scored_id = f"score-{epoch_index}"
    binding = NativeTerminalRunBinding(
        run_id=f"run-{epoch_index}",
        run_nonce_sha256=_sha(f"nonce-{epoch_index}"),
        execution_plan_sha256=execution_plan_sha256,
        rank_config_sha256=_sha("rank-config"),
        attempt_id=f"attempt-{epoch_index}",
        session_id=session_plan_sha256,
        session_epoch=epoch_index,
        previous_run_id=previous_run_id,
        challenge_nonce_sha256=_sha(f"challenge-{epoch_index}"),
        method="static",
        warmup_request_ids=(warmup_id,),
        scored_request_ids=(scored_id,),
    )
    warmup = (
        _TERMINAL._server_request(
            warmup_id, inputs=(epoch_index,), outputs=(epoch_index + 1,)
        ),
    )
    scored = (
        _TERMINAL._server_request(
            scored_id, inputs=(epoch_index + 2,), outputs=(epoch_index + 3,)
        ),
    )
    if transport is None:
        transport = _ResidentTerminalTransport(
            binding=binding,
            warmup=warmup,
            scored=scored,
        )
        provider = NativeTerminalProvider(transport)
    else:
        transport.binding = binding
        transport.warmup = warmup
        transport.scored = scored
        assert provider is not None
    _provider, _begin, _reset, terminal = asyncio.run(
        _TERMINAL._run(
            transport,
            provider=provider,
        )
    )
    artifact = publish_scalable_native_terminal_artifact(
        output_path=output_dir / f"terminal-{epoch_index:04d}.json",
        legacy_artifact=terminal.to_artifact(warmup_requests=warmup),
    )
    return terminal, artifact, transport, provider


def _source_rows(
    *,
    session_plan_sha256: str,
    execution_plan_sha256s: tuple[str, ...],
    terminals: tuple[object, ...],
):
    runtime = _SOURCE._FakeSourceRuntime(
        session_plan_sha256=session_plan_sha256,
        traces=execution_plan_sha256s,
        adapted=False,
    )
    runtime.process_identity = "scheduler:1234"

    async def build():
        capability = await runtime.capability(
            session_plan_sha256=session_plan_sha256,
            execution_plan_sha256s=execution_plan_sha256s,
        )
        capability.pop("capability_sha256")
        capability["process_started_ns"] = 1_000_000
        capability = _SOURCE._bind(capability, "capability_sha256")
        initial = await runtime.initial_state(
            capability_sha256=capability["capability_sha256"]
        )
        epochs = []
        previous_plan = None
        for execution_plan_sha256, terminal in zip(
            execution_plan_sha256s, terminals, strict=True
        ):
            reset = await runtime.reset_boundary(
                capability_sha256=capability["capability_sha256"],
                prior_execution_plan_sha256=previous_plan,
                next_execution_plan_sha256=execution_plan_sha256,
            )
            warmup = await runtime.excluded_warmup(
                execution_plan_sha256=execution_plan_sha256
            )
            clock = await runtime.start_scored_clock(
                execution_plan_sha256=execution_plan_sha256
            )
            clock.pop("clock_receipt_sha256")
            clock["native_reset_sha256"] = terminal.reset_receipt.reset_sha256
            clock = _SOURCE._bind(clock, "clock_receipt_sha256")
            runtime.last_clock = clock
            runtime.clock_receipt_sha256s[-1] = clock["clock_receipt_sha256"]
            trace = await runtime.finish_trace(
                execution_plan_sha256=execution_plan_sha256
            )
            trace.pop("trace_receipt_sha256")
            trace["terminal_receipt_sha256"] = terminal.terminal_sha256
            trace = _SOURCE._bind(trace, "trace_receipt_sha256")
            runtime.trace_receipt_sha256s[-1] = trace["trace_receipt_sha256"]
            runtime.terminal_receipt_sha256s[-1] = terminal.terminal_sha256
            epochs.append((reset, warmup, clock, trace))
            previous_plan = execution_plan_sha256
        close = await runtime.close(capability_sha256=capability["capability_sha256"])
        return capability, initial, tuple(epochs), close

    return asyncio.run(build())


def _complete_publication(tmp_path: Path):
    evidence_dir = tmp_path / "source-chain"
    terminal_dir = tmp_path / "terminals"
    evidence_dir.mkdir(mode=0o700)
    terminal_dir.mkdir(mode=0o700)
    os.chmod(evidence_dir, 0o700)
    os.chmod(terminal_dir, 0o700)
    session_plan_sha256 = _sha("resident-session")
    execution_plan_sha256s = (_sha("trace-1"), _sha("trace-2"))
    terminal_rows = []
    terminal_bindings = []
    prior = None
    transport = None
    provider = None
    for index, execution_plan_sha256 in enumerate(execution_plan_sha256s, start=1):
        terminal, terminal_binding, transport, provider = _terminal(
            output_dir=terminal_dir,
            session_plan_sha256=session_plan_sha256,
            execution_plan_sha256=execution_plan_sha256,
            epoch_index=index,
            previous_run_id=prior,
            transport=transport,
            provider=provider,
        )
        terminal_rows.append(terminal)
        terminal_bindings.append(terminal_binding)
        prior = terminal.binding.run_id
    capability, initial, epochs, close = _source_rows(
        session_plan_sha256=session_plan_sha256,
        execution_plan_sha256s=execution_plan_sha256s,
        terminals=tuple(terminal_rows),
    )
    publisher = FormalServingResidentSourceChainPublisher(
        output_dir=evidence_dir,
        session_plan_sha256=session_plan_sha256,
        execution_plan_sha256s=execution_plan_sha256s,
    )
    publisher.record_capability(capability)
    publisher.record_initial_state(initial)
    for epoch, terminal_binding in zip(epochs, terminal_bindings, strict=True):
        reset, warmup, clock, trace = epoch
        publisher.record_reset(reset)
        publisher.record_warmup(warmup)
        publisher.record_scored_clock(clock)
        publisher.record_trace(trace, terminal_artifact=terminal_binding)
    publication = publisher.record_close(close)
    return publication, publisher.manifest_path


def _rewrite_canonical(path: Path, value: object) -> None:
    os.chmod(path, 0o600)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    os.chmod(path, 0o400)


def _rehash_manifest(value: dict[str, object]) -> None:
    value.pop("manifest_sha256", None)
    value["manifest_sha256"] = canonical_sha256(value)


def test_path_only_revalidator_deep_reopens_two_ordered_epochs(tmp_path: Path) -> None:
    publication, manifest_path = _complete_publication(tmp_path)

    signature = inspect.signature(revalidate_formal_serving_resident_source_chain)
    assert tuple(signature.parameters) == ("manifest_path",)
    reopened = revalidate_formal_serving_resident_source_chain(manifest_path)

    assert reopened == publication
    assert reopened.evidence_level == (
        FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_EVIDENCE_LEVEL
    )
    assert reopened.reuse_authorized is False
    assert reopened.formal_measured is False
    assert len(reopened.epochs) == 2
    assert tuple(epoch.epoch_index for epoch in reopened.epochs) == (1, 2)
    assert tuple(
        epoch.terminal.binding.previous_run_id for epoch in reopened.epochs
    ) == (None, "run-1")
    assert all(
        epoch.terminal.authority_kind == "untrusted_raw_terminal"
        for epoch in reopened.epochs
    )
    assert reopened.close.execution_plan_sha256s == reopened.execution_plan_sha256s
    assert manifest_path.name == FORMAL_SERVING_RESIDENT_SOURCE_CHAIN_MANIFEST


def test_path_only_revalidator_rejects_self_bound_epoch_reordering(
    tmp_path: Path,
) -> None:
    _publication, manifest_path = _complete_publication(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["epochs"].reverse()
    _rehash_manifest(manifest)
    _rewrite_canonical(manifest_path, manifest)

    with pytest.raises(ValueError, match="epoch order"):
        revalidate_formal_serving_resident_source_chain(manifest_path)


def test_path_only_revalidator_rejects_rehashed_terminal_digest_substitution(
    tmp_path: Path,
) -> None:
    _publication, manifest_path = _complete_publication(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    first_epoch = manifest["epochs"][0]
    trace_path = Path(first_epoch["trace_receipt"]["absolute_path"])
    trace = json.loads(trace_path.read_text())
    trace.pop("trace_receipt_sha256")
    trace["terminal_receipt_sha256"] = _sha("substituted-terminal")
    trace["trace_receipt_sha256"] = canonical_sha256(trace)
    _rewrite_canonical(trace_path, trace)
    first_epoch["trace_receipt"] = CanonicalJsonProofBinding.bind(trace_path).to_dict()
    _rehash_manifest(manifest)
    _rewrite_canonical(manifest_path, manifest)

    with pytest.raises(ValueError, match="finalized terminal"):
        revalidate_formal_serving_resident_source_chain(manifest_path)
