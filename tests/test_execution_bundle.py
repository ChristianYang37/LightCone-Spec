from __future__ import annotations

import base64
import hashlib
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_trainable_plan_authority import _inputs as _trainable_plan_inputs

from lightcone_spec import PINNED_SGLANG_PATCH_COUNT, PINNED_SGLANG_TREE
from lightcone_spec.cli.main import main as cli_main
from lightcone_spec.config import run_config_sha256
from lightcone_spec.config.schema import ModelPair, RunConfig, RuntimeConfig
from lightcone_spec.execution import ControlledExecutionPolicy
from lightcone_spec.experiments.budget_authority import (
    bind_budget_materialization_authority,
)
from lightcone_spec.experiments.capacity_authority import (
    bind_capacity_authority,
    bind_capacity_raw_json,
    build_capacity_source_manifest,
    build_capacity_verification_payload,
    capacity_source_receipt_sha256_from_paths,
    capacity_verification_receipt_template,
)
from lightcone_spec.experiments.failure_authority import (
    bind_failure_injection_authority,
    release_failure_plan_for_cell,
)
from lightcone_spec.experiments.gpu_pool import (
    DispatchExecutionPhase,
    DispatchScheduleReceipt,
    GpuAvailability,
    GpuDevice,
    GpuDispatchPlanningContext,
    GpuInventory,
)
from lightcone_spec.experiments.inventory import build_serial_interference_envelope
from lightcone_spec.experiments.itl_authority import (
    ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON,
    release_e2_itl_timestamp_plan,
)
from lightcone_spec.experiments.load import (
    FrozenSamplingParameters,
    ProductionLoadPlan,
    ProductionWindow,
    RequestTemplate,
    closed_loop_corpus,
)
from lightcone_spec.experiments.planning import (
    BUDGET_MATERIALIZATION_PROTOCOL_SHA256,
    CELL_CAPACITY_SIZING_PROTOCOL_SHA256,
    ZERO_MILLISECONDS,
    BudgetJobKind,
    BudgetJobPolicy,
    BudgetLoadBinding,
    BudgetPolicy,
    CapacityEnvelope,
    CellCapacityRequirement,
    P99AnchorStatus,
    ScenarioMilliseconds,
    budget_inventory_identity_from_gpu_inventory,
    materialize_industrial_budgets,
)
from lightcone_spec.experiments.planning_artifacts import (
    budget_load_binding_from_dict,
    budget_load_binding_to_dict,
    budget_plan_to_dict,
    budget_policy_from_dict,
    budget_policy_to_dict,
    capacity_envelope_to_dict,
    production_load_plan_to_dict,
)
from lightcone_spec.experiments.registry import (
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.stage_activation import (
    materialize_registry_stage_activation,
)
from lightcone_spec.locking.models import LockedModel, ModelLock
from lightcone_spec.orchestration.execution_bundle import (
    BoundExecutionArtifact,
    BoundJsonSource,
    DispatchAttemptJournal,
    ExecutionBundleBlockedError,
    IndustrialAssignmentExecutionBundle,
    IndustrialExecutionPlanAudit,
    preflight_dispatch_receipt_output,
    prepared_models_to_dict,
    publish_dispatch_schedule_receipt,
    require_release_dispatch_execution_authority,
    server_launch_to_dict,
    topology_receipt_set_to_dict,
)
from lightcone_spec.orchestration.executor import (
    ArtifactBinding,
    industrial_execution_split_contract,
)
from lightcone_spec.orchestration.runtime import ServerLaunch
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    attestation_message,
)
from lightcone_spec.runtime.compile_cache import (
    PINNED_SGLANG_COMPILE_SOURCE_SHA256,
    PINNED_SGLANG_PATCH_MANIFEST_SHA256,
    PINNED_SGLANG_PATCH_SHA256,
    CompileCacheKey,
    CompileCacheLaunchPlan,
)
from lightcone_spec.runtime.distributed import (
    RankTopologyReceipt,
    TopologyIdentity,
    TopologyReceiptSet,
)
from lightcone_spec.telemetry.writer import DEFAULT_EVIDENCE_WRITER_POLICY


def _write_bound(path: Path, value: object) -> Path:
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(body, encoding="utf-8")
    digest = content_sha256(value)
    Path(f"{path}.sha256").write_text(digest + "\n", encoding="utf-8")
    return path


def test_dispatch_attempt_journal_v2_binds_publication_manifest(
    tmp_path: Path,
) -> None:
    inventory = SimpleNamespace(sha256="1" * 64, devices=())
    context = SimpleNamespace(sha256="2" * 64, inventory=inventory)
    plan = SimpleNamespace(sha256="3" * 64, waves=())
    journal_root = tmp_path / "publication-bound-journal"

    journal = DispatchAttemptJournal.open_or_create(
        journal_root,
        plan=plan,
        execution_context=context,
        execution_bundle_manifest_sha256="4" * 64,
    )
    manifest = json.loads((journal_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["execution_bundle_manifest_sha256"] == "4" * 64
    assert (
        journal.manifest_sha256
        == hashlib.sha256((journal_root / "manifest.json").read_bytes()).hexdigest()
    )

    with pytest.raises(
        ExecutionBundleBlockedError,
        match="dispatch_attempt_journal_manifest_identity_mismatch",
    ):
        DispatchAttemptJournal.open_or_create(
            journal_root,
            plan=plan,
            execution_context=context,
            execution_bundle_manifest_sha256="5" * 64,
        )


def _journal_tree_identity(root: Path) -> tuple[tuple[object, ...], ...]:
    paths = (root, *sorted(root.rglob("*"))) if root.exists() else ()
    rows = []
    for path in paths:
        metadata = path.lstat()
        rows.append(
            (
                str(path.relative_to(root.parent)),
                metadata.st_mode,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_nlink,
                metadata.st_uid,
                metadata.st_gid,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
                path.read_bytes() if path.is_file() else None,
            )
        )
    return tuple(rows)


def _journal_test_authority() -> tuple[object, object, str]:
    inventory = SimpleNamespace(sha256="a" * 64, devices=())
    context = SimpleNamespace(sha256="b" * 64, inventory=inventory)
    plan = SimpleNamespace(sha256="c" * 64, waves=())
    return plan, context, "d" * 64


def test_dispatch_attempt_journal_open_existing_is_strictly_read_only(
    tmp_path: Path,
) -> None:
    plan, context, publication_sha256 = _journal_test_authority()
    root = tmp_path / "existing-journal"
    DispatchAttemptJournal.open_or_create(
        root,
        plan=plan,
        execution_context=context,
        execution_bundle_manifest_sha256=publication_sha256,
    )
    before = _journal_tree_identity(root)

    journal = DispatchAttemptJournal.open_existing(
        root,
        plan=plan,
        execution_context=context,
        execution_bundle_manifest_sha256=publication_sha256,
    )

    assert journal.replay().event_sha256s == ()
    assert _journal_tree_identity(root) == before
    with pytest.raises(
        ExecutionBundleBlockedError,
        match="dispatch_attempt_journal_read_only",
    ):
        journal._append_event({})
    assert _journal_tree_identity(root) == before


@pytest.mark.parametrize("missing", ["root", "manifest", "events"])
def test_dispatch_attempt_journal_open_existing_never_completes_missing_state(
    tmp_path: Path,
    missing: str,
) -> None:
    plan, context, publication_sha256 = _journal_test_authority()
    root = tmp_path / f"missing-{missing}"
    if missing != "root":
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        if missing == "manifest":
            (root / "events").mkdir(mode=0o700)
            (root / "events").chmod(0o700)
        else:
            manifest = root / "manifest.json"
            manifest.write_bytes(b"incomplete")
            manifest.chmod(0o400)
    before = _journal_tree_identity(root)

    with pytest.raises(ExecutionBundleBlockedError):
        DispatchAttemptJournal.open_existing(
            root,
            plan=plan,
            execution_context=context,
            execution_bundle_manifest_sha256=publication_sha256,
        )

    assert _journal_tree_identity(root) == before


def test_dispatch_attempt_journal_open_existing_does_not_repair_corruption(
    tmp_path: Path,
) -> None:
    plan, context, publication_sha256 = _journal_test_authority()
    root = tmp_path / "corrupt-existing-journal"
    DispatchAttemptJournal.open_or_create(
        root,
        plan=plan,
        execution_context=context,
        execution_bundle_manifest_sha256=publication_sha256,
    )
    manifest = root / "manifest.json"
    manifest.chmod(0o600)
    manifest.write_bytes(manifest.read_bytes()[:-1])
    manifest.chmod(0o400)
    before = _journal_tree_identity(root)

    with pytest.raises(
        ExecutionBundleBlockedError,
        match="dispatch_attempt_journal_manifest_identity_mismatch",
    ):
        DispatchAttemptJournal.open_existing(
            root,
            plan=plan,
            execution_context=context,
            execution_bundle_manifest_sha256=publication_sha256,
        )

    assert _journal_tree_identity(root) == before


def _write_journal_event(root: Path, sequence: int, body: bytes) -> None:
    digest = hashlib.sha256(body).hexdigest()
    event = root / "events" / f"{sequence:012d}.{digest}.json"
    event.write_bytes(body)
    event.chmod(0o400)


def test_dispatch_attempt_journal_open_existing_bounds_manifest_before_read(
    tmp_path: Path,
) -> None:
    plan, context, publication_sha256 = _journal_test_authority()
    root = tmp_path / "oversized-manifest-journal"
    DispatchAttemptJournal.open_or_create(
        root,
        plan=plan,
        execution_context=context,
        execution_bundle_manifest_sha256=publication_sha256,
    )
    manifest = root / "manifest.json"
    manifest.chmod(0o600)
    manifest.write_bytes(
        b"x" * (DispatchAttemptJournal._READ_ONLY_MANIFEST_MAX_BYTES + 1)
    )
    manifest.chmod(0o400)
    before = _journal_tree_identity(root)

    with pytest.raises(
        ExecutionBundleBlockedError,
        match="dispatch_attempt_journal_manifest_size_limit_exceeded",
    ):
        DispatchAttemptJournal.open_existing(
            root,
            plan=plan,
            execution_context=context,
            execution_bundle_manifest_sha256=publication_sha256,
        )

    assert _journal_tree_identity(root) == before


def test_dispatch_attempt_journal_open_existing_bounds_each_event_before_read(
    tmp_path: Path,
) -> None:
    plan, context, publication_sha256 = _journal_test_authority()
    root = tmp_path / "oversized-event-journal"
    DispatchAttemptJournal.open_or_create(
        root,
        plan=plan,
        execution_context=context,
        execution_bundle_manifest_sha256=publication_sha256,
    )
    _write_journal_event(
        root,
        0,
        b"x" * (DispatchAttemptJournal._READ_ONLY_EVENT_MAX_BYTES + 1),
    )
    before = _journal_tree_identity(root)

    with pytest.raises(
        ExecutionBundleBlockedError,
        match="dispatch_attempt_journal_event_size_limit_exceeded",
    ):
        DispatchAttemptJournal.open_existing(
            root,
            plan=plan,
            execution_context=context,
            execution_bundle_manifest_sha256=publication_sha256,
        )

    assert _journal_tree_identity(root) == before


def test_dispatch_attempt_journal_open_existing_bounds_event_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, context, publication_sha256 = _journal_test_authority()
    root = tmp_path / "too-many-events-journal"
    DispatchAttemptJournal.open_or_create(
        root,
        plan=plan,
        execution_context=context,
        execution_bundle_manifest_sha256=publication_sha256,
    )
    monkeypatch.setattr(DispatchAttemptJournal, "_READ_ONLY_MAX_EVENT_COUNT", 2)
    for sequence in range(3):
        _write_journal_event(root, sequence, b"{}\n")
    before = _journal_tree_identity(root)

    with pytest.raises(
        ExecutionBundleBlockedError,
        match="dispatch_attempt_journal_event_count_limit_exceeded",
    ):
        DispatchAttemptJournal.open_existing(
            root,
            plan=plan,
            execution_context=context,
            execution_bundle_manifest_sha256=publication_sha256,
        )

    assert _journal_tree_identity(root) == before


def test_dispatch_attempt_journal_open_existing_bounds_root_enumeration(
    tmp_path: Path,
) -> None:
    plan, context, publication_sha256 = _journal_test_authority()
    root = tmp_path / "too-many-root-entries-journal"
    DispatchAttemptJournal.open_or_create(
        root,
        plan=plan,
        execution_context=context,
        execution_bundle_manifest_sha256=publication_sha256,
    )
    (root / "unexpected").write_bytes(b"unexpected")
    before = _journal_tree_identity(root)

    with pytest.raises(
        ExecutionBundleBlockedError,
        match="dispatch_attempt_journal_directory_entry_limit_exceeded",
    ):
        DispatchAttemptJournal.open_existing(
            root,
            plan=plan,
            execution_context=context,
            execution_bundle_manifest_sha256=publication_sha256,
        )

    assert _journal_tree_identity(root) == before


def test_dispatch_attempt_journal_open_existing_bounds_cumulative_event_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, context, publication_sha256 = _journal_test_authority()
    root = tmp_path / "cumulative-event-journal"
    DispatchAttemptJournal.open_or_create(
        root,
        plan=plan,
        execution_context=context,
        execution_bundle_manifest_sha256=publication_sha256,
    )
    monkeypatch.setattr(DispatchAttemptJournal, "_READ_ONLY_EVENT_MAX_BYTES", 8)
    monkeypatch.setattr(DispatchAttemptJournal, "_READ_ONLY_EVENTS_MAX_BYTES", 9)
    _write_journal_event(root, 0, b"12345")
    _write_journal_event(root, 1, b"67890")
    before = _journal_tree_identity(root)

    with pytest.raises(
        ExecutionBundleBlockedError,
        match="dispatch_attempt_journal_cumulative_size_limit_exceeded",
    ):
        DispatchAttemptJournal.open_existing(
            root,
            plan=plan,
            execution_context=context,
            execution_bundle_manifest_sha256=publication_sha256,
        )

    assert _journal_tree_identity(root) == before


def _ensure_bound(path: str | Path) -> None:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    Path(f"{source}.sha256").write_text(content_sha256(value) + "\n", encoding="utf-8")


def _source(
    path: str | Path,
    *,
    semantic_sha256: str | None = None,
) -> BoundJsonSource:
    return BoundJsonSource.bind(path, semantic_sha256=semantic_sha256)


def _artifact(binding: ArtifactBinding) -> BoundExecutionArtifact:
    _ensure_bound(binding.path)
    return BoundExecutionArtifact(
        name=binding.name,
        experiment=binding.experiment,
        source=_source(binding.path, semantic_sha256=binding.content_sha256),
    )


def _scenario(value: int) -> ScenarioMilliseconds:
    return ScenarioMilliseconds(value, value, value)


def _budget_policy() -> BudgetPolicy:
    rows: list[BudgetJobPolicy] = []
    for kind in sorted(BudgetJobKind, key=lambda value: value.value):
        rows.append(
            BudgetJobPolicy(
                job_kind=kind,
                startup_model_load=_scenario(1_000),
                compile_jit_graph_prewarm=(
                    _scenario(1) if kind is BudgetJobKind.COMPILE else ZERO_MILLISECONDS
                ),
                reset_finalization=ZERO_MILLISECONDS,
                evidence_flush_shutdown=_scenario(1_000),
                retry=ZERO_MILLISECONDS,
                retry_allowance=0,
                download_compile_reservation=(
                    _scenario(1)
                    if kind is BudgetJobKind.DOWNLOAD
                    else ZERO_MILLISECONDS
                ),
                reserved_gpu_overhead=ZERO_MILLISECONDS,
            )
        )
    return BudgetPolicy(
        schema_version=1,
        policy_name="execution-bundle-test-policy",
        reducer_protocol_sha256=BUDGET_MATERIALIZATION_PROTOCOL_SHA256,
        job_policies=tuple(rows),
    )


def _budget_load_binding(cell, *, execution_cell_id: str) -> BudgetLoadBinding:
    request_count = int(cell.identity.concurrency)
    input_tokens = (
        tuple(range((3 * int(cell.identity.context) + 3) // 4))
        if cell.cell_id == execution_cell_id
        and cell.identity.regime == "long_input_short_output"
        else (1,)
    )
    sampling = FrozenSamplingParameters.from_mapping(
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "sampling_seed": cell.identity.seed,
            "max_new_tokens": 2,
            "ignore_eos": True,
        }
    )
    corpus = closed_loop_corpus(
        tuple(
            RequestTemplate(
                input_token_ids=tuple(token + index for token in input_tokens),
                requested_output_tokens=2,
                sampling=sampling,
            )
            for index in range(request_count)
        ),
        namespace=f"bundle-budget-{cell.cell_id}",
        split="tuning",
        concurrency=request_count,
        cohort_count=cell.identity.cohort_count,
        cohort_popularity="uniform",
        cohort_seed=cell.identity.seed,
    )
    load = ProductionLoadPlan(
        warmup=None,
        scored=corpus,
        window=ProductionWindow(
            warmup_duration_us=0,
            arrival_duration_us=3_000,
            request_deadline_us=100_000,
            drain_duration_us=100_000,
        ),
    )
    return BudgetLoadBinding(
        cell_id=cell.cell_id,
        job_kind=BudgetJobKind.SHORT,
        optimistic_load=load,
        registered_load=load,
        quota_envelope_load=load,
        minimum_completed_requests=1,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
    )


def _raw_capacity_authority(
    root: Path,
    *,
    registry,
    inventory,
    inventory_path: Path,
    inventory_source_receipt_path: Path,
    cell_ids: tuple[str, ...],
):
    root.mkdir()
    budget_inventory = budget_inventory_identity_from_gpu_inventory(inventory)
    collection_nonce = content_sha256({"capacity-collection": str(root.resolve())})
    captured_at_ns = time.time_ns() - 1_000_000
    provider_path = _write_bound(
        root / "provider-quota.json",
        {
            "schema_version": 1,
            "kind": "industrial_provider_quota_receipt",
            "budget_inventory_sha256": budget_inventory.sha256,
            "gpu_inventory_sha256": inventory.sha256,
            "inventory_source_receipt_sha256": inventory.source_receipt_sha256,
            "provider_scope_sha256": content_sha256("bundle-provider-scope"),
            "collection_nonce_sha256": collection_nonce,
            "captured_at_ns": captured_at_ns,
            "total_quota_gpu_ms": 10**18,
            "consumed_gpu_ms": 0,
            "available_gpu_ms": 10**18,
        },
    )
    host_path = _write_bound(
        root / "host-capacity.json",
        {
            "schema_version": 1,
            "kind": "industrial_host_capacity_receipt",
            "budget_inventory_sha256": budget_inventory.sha256,
            "gpu_inventory_sha256": inventory.sha256,
            "inventory_source_receipt_sha256": inventory.source_receipt_sha256,
            "host_sha256": budget_inventory.host_sha256,
            "filesystem_sha256": content_sha256("bundle-capacity-filesystem"),
            "collection_nonce_sha256": collection_nonce,
            "captured_at_ns": captured_at_ns,
            "host_free_bytes": 10**18,
            "host_quota_bytes": 10**18,
        },
    )
    sizing_paths: list[Path] = []
    for index, cell_id in enumerate(cell_ids):
        provenance_paths: dict[str, Path] = {}
        for name, kind in (
            ("evidence", "industrial_evidence_capacity_provenance"),
            ("model", "industrial_model_staging_capacity_provenance"),
            ("compile", "industrial_compile_overlay_capacity_provenance"),
        ):
            provenance_paths[name] = _write_bound(
                root / f"{index:03d}-{name}-provenance.json",
                {
                    "schema_version": 1,
                    "kind": kind,
                    "cell_id": cell_id,
                    "maximum_bytes": 0,
                    "derivation_sha256": content_sha256(
                        {"bundle-capacity": name, "cell_id": cell_id}
                    ),
                },
            )
        sizing_paths.append(
            _write_bound(
                root / f"{index:03d}-cell-sizing.json",
                {
                    "schema_version": 1,
                    "kind": "industrial_cell_capacity_sizing_receipt",
                    "registry_sha256": registry.sha256,
                    "budget_inventory_sha256": budget_inventory.sha256,
                    "cell_id": cell_id,
                    "maximum_evidence_bytes": 0,
                    "model_staging_bytes": 0,
                    "compile_overlay_bytes": 0,
                    "evidence_contract_source": bind_capacity_raw_json(
                        provenance_paths["evidence"]
                    ).to_dict(),
                    "model_staging_source": bind_capacity_raw_json(
                        provenance_paths["model"]
                    ).to_dict(),
                    "compile_overlay_source": bind_capacity_raw_json(
                        provenance_paths["compile"]
                    ).to_dict(),
                    "sizing_protocol_sha256": CELL_CAPACITY_SIZING_PROTOCOL_SHA256,
                },
            )
        )
    source_receipt_sha256 = capacity_source_receipt_sha256_from_paths(
        inventory_source_receipt_path=inventory_source_receipt_path,
        provider_quota_receipt_path=provider_path,
        host_capacity_receipt_path=host_path,
        cell_sizing_receipt_paths=tuple(sizing_paths),
    )
    envelope = CapacityEnvelope(
        schema_version=1,
        budget_inventory_sha256=budget_inventory.sha256,
        provider_quota_gpu_ms=10**18,
        host_free_bytes=10**18,
        host_quota_bytes=10**18,
        cell_requirements=tuple(
            CellCapacityRequirement(
                cell_id=cell_id,
                maximum_evidence_bytes=0,
                model_staging_bytes=0,
                compile_overlay_bytes=0,
            )
            for cell_id in cell_ids
        ),
        source_receipt_sha256=source_receipt_sha256,
    )
    envelope_path = _write_bound(
        root / "capacity-envelope.json",
        capacity_envelope_to_dict(envelope),
    )
    manifest = build_capacity_source_manifest(
        registry_sha256=registry.sha256,
        budget_inventory_sha256=budget_inventory.sha256,
        collection_nonce_sha256=collection_nonce,
        capacity_envelope_path=envelope_path,
        gpu_inventory_path=inventory_path,
        inventory_source_receipt_path=inventory_source_receipt_path,
        provider_quota_receipt_path=provider_path,
        host_capacity_receipt_path=host_path,
        cell_sizing_receipt_paths=tuple(sizing_paths),
    )
    manifest_path = _write_bound(root / "capacity-manifest.json", manifest)
    challenge = AttestationChallenge.issue(
        challenge_id="execution-bundle-capacity",
        subject_sha256=content_sha256(manifest),
        lifetime_s=300.0,
    )
    payload_sha256 = content_sha256(build_capacity_verification_payload(manifest))
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    attestation = SignedAttestation(
        schema_version=1,
        kind="lightcone_signed_attestation",
        algorithm="Ed25519",
        attester_id="untrusted-bundle-test-signer",
        key_id="untrusted-bundle-test-key",
        environment="release",
        public_key_base64=base64.b64encode(public_key).decode(),
        challenge_sha256=challenge.sha256,
        payload_sha256=payload_sha256,
        signature_base64=base64.b64encode(
            private_key.sign(
                attestation_message(challenge, payload_sha256=payload_sha256)
            )
        ).decode(),
    )
    verification = capacity_verification_receipt_template(
        source_manifest=manifest,
        challenge=challenge,
        attestation=attestation,
    )
    verification_path = _write_bound(root / "capacity-verification.json", verification)
    return (
        envelope,
        envelope_path,
        manifest_path,
        verification_path,
        bind_capacity_authority(manifest_path, verification_path),
    )


def _topology(device_id: str) -> TopologyReceiptSet:
    return TopologyReceiptSet(
        (
            RankTopologyReceipt(
                topology=TopologyIdentity(
                    tensor_parallel_size=1,
                    data_parallel_size=1,
                    node_count=1,
                    node_id="bundle-host",
                    node_rank=0,
                    global_rank=0,
                    local_rank=0,
                    tensor_parallel_rank=0,
                    data_parallel_rank=0,
                    device_id=device_id,
                    rendezvous_id="bundle-rendezvous",
                    router_id="bundle-router",
                    clock_id="bundle-clock",
                ),
                process_id="bundle-process",
                observed_world_size=1,
            ),
        )
    )


def _bundle_fixture(
    tmp_path: Path,
) -> tuple[Path, IndustrialAssignmentExecutionBundle]:
    cache_root = tmp_path / "cache"
    evidence_root = tmp_path / "evidence"
    registry = build_industrial_registry(
        gpu_uuids=("GPU-logical-a", "GPU-logical-b"),
        cache_root=str(cache_root),
        evidence_root=str(evidence_root),
        base_port=28_000,
    )
    cell = next(
        row
        for row in registry.cells_for("E3a")
        if row.identity.method == "target_only"
        and row.identity.context == 1024
        and row.identity.concurrency == 1
        and row.identity.regime == "long_input_short_output"
    )
    physical_gpu_uuid = "GPU-physical-bundle"
    driver_version = "580.65.06"
    gpu_model = "bundle-gpu"

    inventory_receipt_content = {
        "schema_version": 1,
        "kind": "gpu_inventory_probe_receipt",
        "challenge_nonce_sha256": content_sha256({"challenge": "bundle"}),
        "host_id": "bundle-host",
        "hostname": "bundle-hostname",
        "machine_id_sha256": content_sha256({"machine": "bundle"}),
        "commands": {
            "gpu": {"argv": ["nvidia-smi"], "stdout": "bundle gpu probe"},
            "processes": {"argv": ["nvidia-smi"], "stdout": ""},
            "topology": {"argv": ["nvidia-smi", "topo"], "stdout": "PHB"},
        },
        "parsed_topology": {"pairs": [], "parse_error": None},
        "pci_locality": [
            {
                "index": 0,
                "uuid": physical_gpu_uuid,
                "pci_bus_id": "0000:01:00.0",
                "pci_root": "bundle-root",
                "numa_node": 0,
            },
            {
                "index": 1,
                "uuid": "GPU-physical-idle",
                "pci_bus_id": "0000:02:00.0",
                "pci_root": "bundle-root",
                "numa_node": 0,
            },
        ],
    }
    inventory_receipt_sha256 = content_sha256(inventory_receipt_content)
    inventory_source_receipt_path = _write_bound(
        tmp_path / "gpu-inventory-source-receipt.json",
        {
            **inventory_receipt_content,
            "receipt_sha256": inventory_receipt_sha256,
        },
    )
    inventory = GpuInventory(
        schema_version=1,
        devices=(
            GpuDevice(
                uuid=physical_gpu_uuid,
                host_id="bundle-host",
                model=gpu_model,
                memory_bytes=80 * 1024**3,
                compute_capability=(9, 0),
                pci_bus_id="0000:01:00.0",
                pci_root="bundle-root",
                numa_node=0,
                interconnects=("pcie",),
                peer_access_class="bundle-peer",
                clock_policy="persistence=Enabled;max_sm_mhz=1500",
                power_limit_watts=700.0,
                thermal_limit_celsius=83.0,
                availability=GpuAvailability.READY,
                reserved_processes=(),
                allowed_topology_groups=(),
            ),
            GpuDevice(
                uuid="GPU-physical-idle",
                host_id="bundle-host",
                model=gpu_model,
                memory_bytes=80 * 1024**3,
                compute_capability=(9, 0),
                pci_bus_id="0000:02:00.0",
                pci_root="bundle-root",
                numa_node=0,
                interconnects=("pcie",),
                peer_access_class="bundle-peer",
                clock_policy="persistence=Enabled;max_sm_mhz=1500",
                power_limit_watts=700.0,
                thermal_limit_celsius=83.0,
                availability=GpuAvailability.RESERVED,
                reserved_processes=(),
                allowed_topology_groups=(),
            ),
        ),
        topology_groups=(),
        source_receipt_sha256=inventory_receipt_sha256,
    )

    sampling_profile = SamplingProfile()
    sampling_path = tmp_path / "sampling.json"
    sampling_profile.write(sampling_path)
    sampling_binding = ArtifactBinding.from_path(
        name="sampling",
        path=sampling_path,
        semantic_sha256=sampling_profile.sha256,
    )
    target_revision = "1" * 40
    drafter_revision = "2" * 40
    drafter_id = "test/drafter"
    model_lock = ModelLock(
        schema_version=2,
        models=tuple(
            sorted(
                (
                    LockedModel(cell.identity.model, target_revision),
                    LockedModel(drafter_id, drafter_revision),
                ),
                key=lambda row: row.model_id,
            )
        ),
    )
    model_lock_path = tmp_path / "model-lock.json"
    model_lock.write(model_lock_path)
    model_lock_binding = ArtifactBinding.from_path(
        name="model-lock",
        path=model_lock_path,
        semantic_sha256=model_lock.sha256,
    )

    load_bindings = tuple(
        sorted(
            (
                _budget_load_binding(row, execution_cell_id=cell.cell_id)
                for row in registry.cells_for("E3a")
                if row.identity.method in {"target_only", "static"}
                and row.status.value == "UNMEASURED"
            ),
            key=lambda row: row.cell_id,
        )
    )
    load = next(
        row.registered_load for row in load_bindings if row.cell_id == cell.cell_id
    )
    split_value = industrial_execution_split_contract(
        registry_sha256=registry.sha256,
        cell=cell,
        load_plan=load,
        sampling_profile_sha256=sampling_profile.sha256,
        model_lock_sha256=model_lock.sha256,
    )
    split_path = _write_bound(tmp_path / "split.json", split_value)
    split_binding = ArtifactBinding.from_path(
        name="split",
        path=split_path,
        semantic_sha256=content_sha256(split_value),
    )

    checkout = tmp_path / "verified-checkout"
    checkout.mkdir()
    target_root = tmp_path / "target-model"
    target_root.mkdir()
    drafter_root = tmp_path / "drafter-model"
    drafter_root.mkdir()
    manifest_sha256 = content_sha256({"runtime-manifest": "bundle"})
    runtime_envelope = {
        "schema_version": 1,
        "status": "PASS",
        "readiness": {
            "status": "PASS",
            "pass_count": 1,
            "fail_count": 0,
            "unknown_count": 0,
        },
        "roots": {
            "project": str(tmp_path.resolve()),
            "patched_sglang": str(checkout.resolve()),
            "distinct": True,
        },
        "runtime_manifest": {
            "valid": True,
            "sha256": manifest_sha256,
            "sidecar_sha256": manifest_sha256,
        },
        "checks": {"fixture_authority": {"status": "PASS"}},
        "source_tree": {
            "path": str(checkout.resolve()),
            "is_git_checkout": True,
            "root_matches_toplevel": True,
            "head": "bundle-head",
            "tree": PINNED_SGLANG_TREE,
            "dirty": False,
            "pinned_ancestor": True,
            "patch_commits": PINNED_SGLANG_PATCH_COUNT,
        },
        "python": {"version": "3.12.11"},
        "gpu": {
            "parsed_inventory": {
                "devices": [
                    {
                        "uuid": device.uuid,
                        "name": device.model,
                        "memory_total_mib": device.memory_bytes // (1024 * 1024),
                        "driver_version": driver_version,
                        "compute_capability": "9.0",
                        "pci_bus_id": device.pci_bus_id,
                    }
                    for device in inventory.devices
                ],
                "parse_error": None,
            },
            "torch": {
                "importable": True,
                "version": "2.11.0+cu130",
                "cuda_build": "13.0",
                "cuda_available": True,
                "device_count": 2,
            },
        },
        "commands": {"nvcc": "Cuda compilation tools, release 13.0, V13.0.88"},
        "packages": {"torch": "2.11.0", "triton": "3.6.0"},
    }
    dependency_artifacts: list[ArtifactBinding] = []
    preflight_outputs: dict[str, str] = {}
    runtime_envelope_binding: ArtifactBinding | None = None
    for name in registry.definition("preflight").locked_outputs:
        value = runtime_envelope if name == "runtime_envelope" else {"output": name}
        path = _write_bound(tmp_path / f"preflight-{name}.json", value)
        binding = ArtifactBinding.from_path(
            name=name,
            path=path,
            semantic_sha256=content_sha256(value),
            experiment="preflight",
        )
        dependency_artifacts.append(binding)
        preflight_outputs[name] = binding.content_sha256
        if name == "runtime_envelope":
            runtime_envelope_binding = binding
    assert runtime_envelope_binding is not None
    dependency_artifacts = sorted(
        dependency_artifacts,
        key=lambda row: (str(row.experiment), row.name),
    )
    receipt = registry.make_receipt(
        "preflight",
        preflight_outputs,
        runtime_sha256=runtime_envelope_binding.content_sha256,
        split_sha256=split_binding.content_sha256,
        completed_cells_sha256=content_sha256({"completed": "preflight"}),
    )
    activation = materialize_registry_stage_activation(
        registry,
        experiment="E3a",
        dependency_receipts=(receipt,),
        runtime_sha256=runtime_envelope_binding.content_sha256,
        split_sha256=split_binding.content_sha256,
    )
    assert tuple(row.cell_id for row in load_bindings) == activation.activated_cell_ids

    registry_path = _write_bound(
        tmp_path / "registry.json",
        {
            "schema_version": 2,
            "generator": (
                "lightcone_spec.experiments.registry.build_industrial_registry:v2"
            ),
            "parameters": {
                "logical_gpu_slots": list(registry.gpu_uuids),
                "base_port": 28_000,
                "cache_root": str(cache_root),
                "evidence_root": str(evidence_root),
                "seed": 20260811,
            },
            "registry_sha256": registry.sha256,
            "registry": registry.to_dict(),
        },
    )
    inventory_path = _write_bound(tmp_path / "inventory.json", inventory.to_dict())
    (
        capacity,
        capacity_path,
        capacity_manifest_path,
        capacity_verification_path,
        capacity_authority,
    ) = _raw_capacity_authority(
        tmp_path / "raw-capacity",
        registry=registry,
        inventory=inventory,
        inventory_path=inventory_path,
        inventory_source_receipt_path=inventory_source_receipt_path,
        cell_ids=activation.activated_cell_ids,
    )
    policy = _budget_policy()
    budget_plan = materialize_industrial_budgets(
        registry,
        activations=(activation,),
        load_bindings=load_bindings,
        policy=policy,
        inventory=budget_inventory_identity_from_gpu_inventory(inventory),
        capacity_envelope=capacity,
        capacity_authority=capacity_authority,
        require_complete=False,
    )
    assert budget_plan.status == "UNRESOLVED"

    interference_envelope, interference_receipt = build_serial_interference_envelope(
        inventory
    )
    context = GpuDispatchPlanningContext(
        registry=registry,
        inventory=inventory,
        interference_envelope=interference_envelope,
        budgets=budget_plan.diagnostic_budgets,
        receipts=(receipt,),
        activation_artifact=activation,
        port_start=31_000,
        port_end=31_999,
        seed=20260811,
    )
    dispatch_plan = context.issue_plan()
    assignment = next(
        assignment
        for wave in dispatch_plan.waves
        for assignment in wave.assignments
        if assignment.work_item.item_id == cell.cell_id
    )
    assert assignment.gpu_uuids == (physical_gpu_uuid,)
    topology = _topology(physical_gpu_uuid)

    config = RunConfig.model_validate(
        RunConfig(
            method="target_only",
            model=ModelPair(
                target=cell.identity.model,
                drafter=drafter_id,
                target_revision=target_revision,
                drafter_revision=drafter_revision,
                algorithm="DFLASH",
                max_context_length=cell.identity.context,
                draft_depth=7,
            ),
            runtime=RuntimeConfig(
                sampling_profile_sha256=sampling_profile.sha256,
                speculation_enabled=False,
                tensor_parallel_size=1,
                data_parallel_size=1,
                tp_rank=0,
                dp_rank=0,
                node_count=1,
                node_rank=0,
                device_identity=physical_gpu_uuid,
                rendezvous_identity="bundle-rendezvous",
                router_identity="bundle-router",
                clock_identity="bundle-clock",
                process_group_backend="nccl",
                distributed_runtime_capability="single_rank",
                distributed_capability_receipt_sha256=None,
                speculative_num_draft_tokens=8,
                speculative_eagle_topk=None,
                use_rejection_sampling=True,
                max_running_requests=1,
                telemetry_detail="headline",
                prefill_decode_disaggregation=False,
                two_batch_overlap=False,
            ),
            adaptation=None,
            online_spec=None,
            tenant_id="bundle-test",
        ).model_dump(mode="json")
    )
    run_config_path = _write_bound(
        tmp_path / "run-config.json", config.model_dump(mode="json")
    )
    compile_key = CompileCacheKey(
        patched_sglang_tree=PINNED_SGLANG_TREE,
        patch_manifest_sha256=PINNED_SGLANG_PATCH_MANIFEST_SHA256,
        patch_sha256=PINNED_SGLANG_PATCH_SHA256,
        source_sha256=PINNED_SGLANG_COMPILE_SOURCE_SHA256,
        python_version="3.12.11",
        torch_version="2.11.0+cu130",
        triton_version="3.6.0",
        cuda_version="13.0",
        driver_version=driver_version,
        sm_architecture="sm_90",
        gpu_model=gpu_model,
        dtype="bfloat16",
        target_revision=target_revision,
        drafter_revision=None,
        tensor_parallel_size=1,
        context_limit=config.runtime.context_length,
        max_running_requests=1,
        graph_buckets=(1,),
        allocator="cuda_malloc_async",
        build_flags=(),
    )
    compile_plan = CompileCacheLaunchPlan.issue(
        key=compile_key,
        cache_root=tmp_path / "compile-cache",
        cache_mode="build",
    )
    compile_plan_path = tmp_path / "compile-plan.json"
    compile_plan.write(compile_plan_path)
    port = assignment.ports[0]
    launch = ServerLaunch(
        method="target_only",
        base_url=f"http://127.0.0.1:{port}",
        exclusive_device=True,
        run_config=str(run_config_path.resolve()),
        adaptation_config=None,
        telemetry_path=None,
        argv=(
            sys.executable,
            "-m",
            "lightcone_spec.sglang_bridge.launch",
            "--checkout",
            str(checkout.resolve()),
            "--compile-cache-plan",
            str(compile_plan_path.resolve()),
            "--compile-cache-plan-sha256",
            compile_plan.sha256,
            "--compile-cache-key-sha256",
            compile_key.sha256,
            "--run-config",
            str(run_config_path.resolve()),
            "--run-config-sha256",
            run_config_sha256(config),
            "--",
            "--model-path",
            str(target_root.resolve()),
            "--max-running-requests",
            "1",
            "--mem-fraction-static",
            "0.8",
            "--tp-size",
            "1",
            "--dtype",
            compile_key.dtype,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--context-length",
            "40960",
            "--random-seed",
            "1",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--disable-overlap-schedule",
            "--speculative-speed-study-metrics",
        ),
        compile_cache_plan=str(compile_plan_path.resolve()),
        compile_cache_plan_sha256=compile_plan.sha256,
        compile_cache_key_sha256=compile_key.sha256,
    )

    budget_plan_path = _write_bound(
        tmp_path / "budget-plan.json", budget_plan_to_dict(budget_plan)
    )
    policy_path = _write_bound(
        tmp_path / "budget-policy.json", budget_policy_to_dict(policy)
    )
    budget_load_sources = tuple(
        _source(
            _write_bound(
                tmp_path / f"budget-load-{index:03d}.json",
                budget_load_binding_to_dict(binding),
            ),
            semantic_sha256=binding.sha256,
        )
        for index, binding in enumerate(load_bindings)
    )
    receipt_path = _write_bound(tmp_path / "receipt-000.json", receipt.to_dict())
    receipt_source = _source(receipt_path, semantic_sha256=receipt.sha256)
    activation_path = _write_bound(
        tmp_path / "activation-manifest.json",
        {
            "schema_version": 1,
            "kind": "industrial_registry_stage_activation_manifest",
            "registry_artifact": str(registry_path.resolve()),
            "experiment": "E3a",
            "runtime_artifact": runtime_envelope_binding.path,
            "split_artifact": split_binding.path,
            "dependency_receipts": [receipt_source.path],
        },
    )
    budget_materialization_authority = bind_budget_materialization_authority(
        activation_manifest_path=activation_path.resolve(),
        policy_path=policy_path.resolve(),
        load_binding_paths=tuple(source.path for source in budget_load_sources),
        capacity_envelope_path=capacity_path.resolve(),
        capacity_authority=capacity_authority,
        declared_plan_path=budget_plan_path.resolve(),
    )
    context_value = context.authority_dict()
    context_value.update(
        {
            "schema_version": 4,
            "kind": "gpu_dispatch_execution_context",
            "interference_calibration_authority_sha256": None,
            "interference_calibration_bootstrap_authority_sha256": None,
            "budget_plan_sha256": budget_plan.sha256,
            "capacity_authority_sha256": capacity_authority.sha256,
            "budget_materialization_authority_sha256": (
                budget_materialization_authority.sha256
            ),
            "completion_authority_sha256s": [],
        }
    )
    context_path = _write_bound(tmp_path / "dispatch-context.json", context_value)
    dispatch_path = _write_bound(tmp_path / "dispatch.json", dispatch_plan.to_dict())
    interference_path = _write_bound(
        tmp_path / "interference.json", interference_envelope.to_dict()
    )
    interference_receipt_path = _write_bound(
        tmp_path / "interference-receipt.json", interference_receipt
    )
    topology_path = _write_bound(
        tmp_path / "topology.json", topology_receipt_set_to_dict(topology)
    )
    load_path = _write_bound(tmp_path / "load.json", production_load_plan_to_dict(load))
    launch_path = _write_bound(
        tmp_path / "server-launch.json", server_launch_to_dict(launch)
    )
    declared_summary = {
        "schema_version": 1,
        "kind": "industrial_execution_plan_declared_unvalidated",
        "assignment_sha256": assignment.assignment_id,
        "dispatch_plan_sha256": dispatch_plan.sha256,
        "budget_plan_sha256": budget_plan.sha256,
    }
    declared_plan_sha256 = content_sha256(declared_summary)
    execution_plan_path = _write_bound(
        tmp_path / "execution-plan-summary.json", declared_summary
    )
    prepared_path = _write_bound(
        tmp_path / "prepared-models.json",
        prepared_models_to_dict(
            model_lock,
            {
                cell.identity.model: str(target_root.resolve()),
                drafter_id: str(drafter_root.resolve()),
            },
        ),
    )
    writer = DEFAULT_EVIDENCE_WRITER_POLICY
    execution_policy_path = _write_bound(
        tmp_path / "execution-policy.json",
        {
            "schema_version": 1,
            "kind": "industrial_assignment_execution_policy",
            "patched_sglang_tree": PINNED_SGLANG_TREE,
            "evidence_writer_policy": writer.to_dict(),
            "evidence_writer_policy_sha256": writer.sha256,
            "controlled_execution_policy": ControlledExecutionPolicy().to_dict(),
            "controlled_execution_policy_sha256": (ControlledExecutionPolicy().sha256),
            "startup_timeout_s": 1.0,
            "shutdown_timeout_s": 1.0,
            "abort_grace_s": 1.0,
        },
    )

    dependency_sources = tuple(_artifact(row) for row in dependency_artifacts)
    runtime_envelope_source = next(
        row for row in dependency_sources if row.name == "runtime_envelope"
    )
    split_artifact = _artifact(split_binding)
    inventory_source_binding = ArtifactBinding.from_path(
        name="gpu_inventory_source_receipt",
        path=inventory_source_receipt_path,
        semantic_sha256=inventory_receipt_sha256,
    )
    bundle = IndustrialAssignmentExecutionBundle(
        schema_version=5,
        kind="industrial_assignment_execution_bundle",
        assignment_sha256=assignment.assignment_id,
        cell_id=cell.cell_id,
        execution_plan_sha256=declared_plan_sha256,
        run_nonce_sha256=content_sha256({"run_nonce": "bundle-test"}),
        output_root=str(Path(cell.resources.evidence_root).resolve()),
        registry=_source(registry_path, semantic_sha256=registry.sha256),
        inventory=_source(inventory_path, semantic_sha256=inventory.sha256),
        interference_envelope=_source(
            interference_path, semantic_sha256=interference_envelope.sha256
        ),
        interference_source_receipt=_source(
            interference_receipt_path,
            semantic_sha256=interference_receipt["receipt_sha256"],
        ),
        interference_calibration_authority=None,
        budget_plan=_source(budget_plan_path, semantic_sha256=budget_plan.sha256),
        budget_policy=_source(policy_path, semantic_sha256=policy.sha256),
        budget_load_bindings=budget_load_sources,
        capacity_envelope=_source(capacity_path, semantic_sha256=capacity.sha256),
        capacity_source_manifest=_source(
            capacity_manifest_path,
            semantic_sha256=capacity_authority.source_manifest.semantic_sha256,
        ),
        capacity_verification_receipt=_source(
            capacity_verification_path,
            semantic_sha256=capacity_authority.verification_receipt.semantic_sha256,
        ),
        dependency_receipts=(receipt_source,),
        activation=_source(activation_path, semantic_sha256=activation.sha256),
        activation_runtime=runtime_envelope_source.source,
        activation_split=split_artifact.source,
        dispatch_context=_source(
            context_path, semantic_sha256=content_sha256(context_value)
        ),
        dispatch_plan=_source(dispatch_path, semantic_sha256=dispatch_plan.sha256),
        topology_receipts=_source(
            topology_path, semantic_sha256=topology.receipt_sha256
        ),
        production_load=_source(load_path, semantic_sha256=load.paired_replay_sha256),
        itl_timestamp_plan=None,
        itl_timestamp_plan_sha256=None,
        itl_timestamp_producer_sha256=None,
        run_config=_source(run_config_path, semantic_sha256=run_config_sha256(config)),
        server_launch=_source(launch_path),
        execution_plan_summary=_source(
            execution_plan_path, semantic_sha256=declared_plan_sha256
        ),
        dependency_artifacts=dependency_sources,
        split_artifact=split_artifact,
        sampling_artifact=_artifact(sampling_binding),
        model_lock_artifact=_artifact(model_lock_binding),
        prepared_models=_source(prepared_path),
        trainable_plan_authority=None,
        failure_injection_authority=None,
        prepared_model_content_release_manifest_sha256=None,
        compile_cache_plan=_source(
            compile_plan_path, semantic_sha256=compile_plan.sha256
        ),
        inventory_source_artifact=_artifact(inventory_source_binding),
        runtime_envelope_artifact=runtime_envelope_source,
        execution_policy=_source(execution_policy_path),
    )
    bundle_path = _write_bound(tmp_path / "execution-bundle.json", bundle.to_dict())
    return bundle_path, bundle


def test_e2_bundle_blocks_empty_release_producer_before_downstream_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightcone_spec.orchestration.execution_bundle as bundle_module

    _, baseline = _bundle_fixture(tmp_path)
    registry = bundle_module._load_registry(baseline.registry.load())
    cell = registry.cells_for("E2")[0]
    plan = release_e2_itl_timestamp_plan(registry, cell)
    plan_path = _write_bound(tmp_path / "e2-itl-plan.json", plan.to_dict())
    bundle = replace(
        baseline,
        cell_id=cell.cell_id,
        itl_timestamp_plan=_source(plan_path, semantic_sha256=plan.sha256),
        itl_timestamp_plan_sha256=plan.sha256,
        itl_timestamp_producer_sha256=None,
    )
    inventory_reads = 0
    original_load = bundle_module.BoundJsonSource.load

    def track_load(source):
        nonlocal inventory_reads
        if source == bundle.inventory:
            inventory_reads += 1
        return original_load(source)

    monkeypatch.setattr(bundle_module.BoundJsonSource, "load", track_load)
    with pytest.raises(
        ExecutionBundleBlockedError,
        match=ITL_TIMESTAMP_PRODUCER_UNAVAILABLE_REASON,
    ):
        bundle.reconstruct_execution_plan()

    assert inventory_reads == 0
    assert not Path(bundle.output_root).exists()


def test_all_wave_preflight_rejects_later_itl_source_tamper_before_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import lightcone_spec.orchestration.execution_bundle as bundle_module
    import lightcone_spec.orchestration.execution_bundle_materializer as materializer

    _, first = _bundle_fixture(tmp_path)
    itl_path = _write_bound(tmp_path / "later-wave-itl-plan.json", {"plan": "bound"})
    itl_source = _source(itl_path)
    later = replace(
        first,
        assignment_sha256="f" * 64,
        itl_timestamp_plan=itl_source,
        itl_timestamp_plan_sha256=itl_source.semantic_sha256,
        itl_timestamp_producer_sha256=None,
    )
    itl_path.write_text('{"plan":"tampered"}\n', encoding="utf-8")
    manifest_path = (tmp_path / "all-wave-manifest.json").resolve()
    publication = SimpleNamespace(
        manifest=SimpleNamespace(sha256="9" * 64, assignments=()),
        bundles=(first, later),
    )
    monkeypatch.setattr(
        bundle_module,
        "require_release_dispatch_execution_authority",
        lambda: None,
    )
    monkeypatch.setattr(
        bundle_module, "preflight_compile_cache_launch", lambda _plan: None
    )
    monkeypatch.setattr(
        materializer,
        "load_materialized_dispatch_execution_bundle_publication",
        lambda _path: publication,
    )
    monkeypatch.setattr(
        bundle_module.IndustrialAssignmentExecutionBundle,
        "reconstruct_execution_plan",
        lambda _self: pytest.fail("assignment reconstruction was reached"),
    )

    with pytest.raises(RuntimeError, match="bound bundle source or sidecar changed"):
        asyncio.run(
            bundle_module.execute_dispatch_wave_bundles(
                manifest_path,
                wave_index=0,
                receipt_output=tmp_path / "must-not-publish.json",
            )
        )

    assert not (tmp_path / "must-not-publish.json").exists()


def test_bundle_audits_raw_components_without_minting_a_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path, expected = _bundle_fixture(tmp_path)
    import lightcone_spec.orchestration.execution_bundle as bundle_module

    loaded = IndustrialAssignmentExecutionBundle.load(bundle_path)
    bundle_module._require_shared_bundle_authority((loaded, expected))
    bundle_module._preflight_bundle_assignment_sources(loaded)
    raw_dispatch = loaded.dispatch_plan.load()
    selected_wave_index = next(
        index
        for index, wave in enumerate(raw_dispatch["waves"])
        if loaded.assignment_sha256 in wave["assignment_sha256"]
    )
    assert loaded.assignment_sha256 in bundle_module._declared_wave_assignment_ids(
        loaded,
        wave_index=selected_wave_index,
    )
    changed_context = replace(
        loaded.dispatch_context,
        semantic_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="shared authority field dispatch_context"):
        bundle_module._require_shared_bundle_authority(
            (loaded, replace(loaded, dispatch_context=changed_context))
        )
    audit = loaded.audit_execution_plan()

    assert loaded == expected
    assert isinstance(audit, IndustrialExecutionPlanAudit)
    assert audit.execution_plan_sha256 is None
    assert audit.execution_plan_status == "NOT_VALIDATED"
    assert audit.exact_dispatch_replay is True
    assert audit.schema_version == 3
    assert audit.execution_semantics_sha256 is None
    assert audit.execution_semantics_authority == "diagnostic_non_authority"
    assert audit.dispatch_plan_sha256 == expected.dispatch_plan.semantic_sha256
    assert audit.cell_id == expected.cell_id
    assert audit.budget_plan_status == "UNRESOLVED"
    with pytest.raises(ValueError, match="audit schema is unsupported"):
        replace(audit, schema_version=3.0)
    assert (
        audit.budget_materialization_authority_sha256
        == expected.dispatch_context.load()["budget_materialization_authority_sha256"]
    )
    with pytest.raises(
        ExecutionBundleBlockedError,
        match=(
            "dependency_completion_manifest_authority_missing|"
            "trusted_capacity_verifier_unavailable"
        ),
    ):
        loaded.reconstruct_execution_plan()

    assert loaded.to_dict()["schema_version"] == 5
    with pytest.raises(ValueError, match="bundle schema is unsupported"):
        replace(loaded, schema_version=5.0)
    assert loaded.trainable_plan_authority is None
    assert loaded.prepared_model_content_release_manifest_sha256 is None
    forged_baseline = replace(
        loaded,
        prepared_model_content_release_manifest_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="must not carry trainable-plan authority"):
        bundle_module._preflight_bundle_trainable_plan_release_trust(forged_baseline)

    adapted_inputs = _trainable_plan_inputs(tmp_path / "adapted-authority")
    adapted_authority = adapted_inputs["binding"]
    adapted_config = adapted_inputs["run_config"]
    adapted_bundle = replace(
        loaded,
        run_config=_source(
            adapted_inputs["run_config_path"],
            semantic_sha256=run_config_sha256(adapted_config),
        ),
        trainable_plan_authority=adapted_authority,
        prepared_model_content_release_manifest_sha256=(
            adapted_authority.prepared_model_content_manifest_sha256
        ),
    )
    adapted_round_trip = IndustrialAssignmentExecutionBundle.from_dict(
        adapted_bundle.to_dict()
    )
    assert adapted_round_trip.trainable_plan_authority == adapted_authority
    with pytest.raises(
        ExecutionBundleBlockedError,
        match="prepared_model_content_release_manifest_pin_unavailable",
    ):
        bundle_module._preflight_bundle_trainable_plan_release_trust(adapted_round_trip)

    # Exercise the future release group path without constructing authority:
    # every assignment has a bundle, but only the requested wave may ask the
    # strict per-assignment plan builder to run.
    import asyncio

    waves = tuple(
        bundle_module.GpuDispatchWave.from_dict(value)
        for value in raw_dispatch["waves"]
    )
    dispatch = SimpleNamespace(waves=waves)
    all_assignments = tuple(
        assignment for wave in waves for assignment in wave.assignments
    )
    paths = tuple(
        str(tmp_path / f"group-bundle-{index:03d}.json")
        for index in range(len(all_assignments))
    )
    group_bundles = tuple(
        replace(
            loaded,
            assignment_sha256=assignment.assignment_id,
            cell_id=assignment.work_item.item_id,
        )
        for assignment in all_assignments
    )
    manifest_path = (tmp_path / "dispatch-execution-bundle-manifest.json").resolve()

    @dataclass(frozen=True)
    class FakeContext:
        budgets: tuple[object, ...] = ()
        resume_terminal_authorities: tuple[object, ...] = ()

        def require_ready_budget_authority(self) -> tuple[object, ...]:
            return ()

    fake_context = FakeContext()
    reconstruct_calls: list[str] = []

    def reconstruct_current_assignment(bundle):
        reconstruct_calls.append(bundle.assignment_sha256)
        return SimpleNamespace(
            dispatch_plan=dispatch,
            dispatch_context=fake_context,
            runtime_plan=SimpleNamespace(
                physical_assignment=SimpleNamespace(
                    assignment_sha256=bundle.assignment_sha256
                )
            ),
            server_launch=SimpleNamespace(
                argv=("python", "-m", "module", "--checkout", str(tmp_path))
            ),
        )

    receipt = SimpleNamespace(
        wave_receipts=(SimpleNamespace(wave_index=0),),
    )

    class FakeSnapshot:
        terminal_bindings = ()
        latest_assignment_receipts = ()

        def __init__(self, *, finished: bool) -> None:
            self.receipt = receipt if finished else None
            self.binding = object() if finished else None
            self.replay_authority = object() if finished else None

        @staticmethod
        def require_complete_cost_authority() -> None:
            return None

    class FakeJournal:
        finished = False

        def replay(self, *, event_count=None):
            assert event_count is None
            return FakeSnapshot(finished=self.finished)

    fake_journal = FakeJournal()

    execute_calls = 0

    async def execute_group_plan(
        plan,
        *,
        execution_context,
        runner,
        resume_receipt,
        attempt_journal,
        attempt_journal_replay,
        stop_after_wave_index,
    ):
        nonlocal execute_calls
        execute_calls += 1
        assert plan is dispatch
        assert execution_context == fake_context
        assert callable(runner)
        assert resume_receipt is None
        assert attempt_journal is fake_journal
        assert attempt_journal_replay is None
        assert stop_after_wave_index == 0
        fake_journal.finished = True
        return receipt

    published: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        bundle_module,
        "require_release_dispatch_execution_authority",
        lambda: None,
    )
    import lightcone_spec.orchestration.execution_bundle_materializer as materializer

    publication_state = {
        "value": SimpleNamespace(
            manifest=SimpleNamespace(
                sha256="9" * 64,
                assignments=tuple(
                    SimpleNamespace(bundle=SimpleNamespace(path=path)) for path in paths
                ),
            ),
            bundles=group_bundles,
        )
    }
    monkeypatch.setattr(
        materializer,
        "load_materialized_dispatch_execution_bundle_publication",
        lambda path: publication_state["value"],
    )
    monkeypatch.setattr(
        bundle_module.IndustrialAssignmentExecutionBundle,
        "reconstruct_execution_plan",
        reconstruct_current_assignment,
    )
    monkeypatch.setattr(
        bundle_module,
        "_preflight_bundle_assignment_sources",
        lambda _bundle: None,
    )
    monkeypatch.setattr(
        bundle_module,
        "preflight_fresh_assignment_trace",
        lambda *args, **kwargs: "fresh-run",
    )
    monkeypatch.setattr(
        bundle_module.PinnedBenchServingTransport,
        "from_checkout",
        staticmethod(lambda _checkout: object()),
    )
    monkeypatch.setattr(bundle_module, "execute_dispatch_plan", execute_group_plan)
    journal_open_kwargs = []

    def open_fake_journal(_cls, *args, **kwargs):
        journal_open_kwargs.append(kwargs)
        return fake_journal

    monkeypatch.setattr(
        bundle_module.DispatchAttemptJournal,
        "open_or_create",
        classmethod(open_fake_journal),
    )
    monkeypatch.setattr(bundle_module, "validate_dispatch_resume", lambda *a, **k: None)
    monkeypatch.setattr(
        bundle_module,
        "publish_dispatch_schedule_receipt",
        lambda path, value, **kwargs: published.append((path, value)),
    )

    result = asyncio.run(
        bundle_module.execute_dispatch_wave_bundles(
            manifest_path,
            wave_index=0,
            receipt_output=tmp_path / "group-receipt.json",
        )
    )
    expected_current_ids = {
        assignment.assignment_id for assignment in waves[0].assignments
    }
    assert result is receipt
    assert set(reconstruct_calls) == expected_current_ids
    assert len(reconstruct_calls) == len(expected_current_ids)
    assert execute_calls == 1
    assert published == [(tmp_path / "group-receipt.json", receipt)]
    assert journal_open_kwargs[-1]["execution_bundle_manifest_sha256"] == "9" * 64

    # Inject a coordinator crash after every FINISH is durable but before the
    # canonical schedule envelope is published.  Re-entering with receipt-only
    # CLI state must recover from the raw journal and must not call the runner.
    fake_journal.finished = False
    crash_target = tmp_path / "crash-after-finish.json"
    crash_once = True

    def crash_before_envelope(path, value, **kwargs):
        nonlocal crash_once
        if path == crash_target and crash_once:
            crash_once = False
            raise RuntimeError("injected crash before schedule envelope")
        published.append((path, value))

    monkeypatch.setattr(
        bundle_module,
        "publish_dispatch_schedule_receipt",
        crash_before_envelope,
    )
    with pytest.raises(
        bundle_module._DispatchOutcomeUnknownError,
        match="dispatch outcome is unknown",
    ):
        asyncio.run(
            bundle_module.execute_dispatch_wave_bundles(
                manifest_path,
                wave_index=0,
                receipt_output=crash_target,
            )
        )
    assert fake_journal.finished
    assert execute_calls == 2
    recovered = asyncio.run(
        bundle_module.execute_dispatch_wave_bundles(
            manifest_path,
            wave_index=0,
            receipt_output=crash_target,
        )
    )
    assert recovered is receipt
    assert execute_calls == 2
    assert published[-1] == (crash_target, receipt)

    publication_state["value"] = SimpleNamespace(
        manifest=publication_state["value"].manifest,
        bundles=group_bundles[:-1],
    )
    with pytest.raises(
        ExecutionBundleBlockedError,
        match="industrial_execution_bundle_coverage_incomplete",
    ):
        asyncio.run(
            bundle_module.execute_dispatch_wave_bundles(
                manifest_path,
                wave_index=0,
                receipt_output=tmp_path / "incomplete-group-receipt.json",
            )
        )
    assert published == [
        (tmp_path / "group-receipt.json", receipt),
        (crash_target, receipt),
    ]


def test_bundle_rejects_failure_authority_on_nonfailure_assignment(
    tmp_path: Path,
) -> None:
    import lightcone_spec.orchestration.execution_bundle as bundle_module

    registry = build_industrial_registry(
        gpu_uuids=("GPU-logical-a", "GPU-logical-b"),
        cache_root=str(tmp_path / "cache"),
        evidence_root=str(tmp_path / "evidence"),
        base_port=28_000,
    )
    failure_cell = next(
        row
        for row in registry.cells_for("E5")
        if row.identity.task == "failure_injection"
    )
    nonfailure_cell = next(
        row
        for row in registry.cells_for("E3a")
        if row.identity.task != "failure_injection"
    )
    plan = release_failure_plan_for_cell(registry, failure_cell)
    plan_path = _write_bound(tmp_path / "failure-plan.json", plan.to_dict())
    binding = bind_failure_injection_authority(plan_path, registry=registry)
    diagnostic_sha256, token = (
        bundle_module._require_bundle_failure_injection_authority(
            registry=registry,
            cell=failure_cell,
            binding=binding,
            diagnostic=True,
        )
    )
    assert diagnostic_sha256 == binding.sha256
    assert token is None
    with pytest.raises(
        ExecutionBundleBlockedError,
        match="failure_injection_first_party_actuator_unavailable",
    ):
        bundle_module._require_bundle_failure_injection_authority(
            registry=registry,
            cell=failure_cell,
            binding=binding,
            diagnostic=False,
        )
    with pytest.raises(
        ExecutionBundleBlockedError,
        match="failure_injection_raw_plan_authority_required",
    ):
        bundle_module._require_bundle_failure_injection_authority(
            registry=registry,
            cell=failure_cell,
            binding=None,
            diagnostic=False,
        )
    with pytest.raises(ValueError, match="non-failure bundle"):
        bundle_module._require_bundle_failure_injection_authority(
            registry=registry,
            cell=nonfailure_cell,
            binding=binding,
            diagnostic=True,
        )


def test_bundle_rejects_summary_and_raw_authority_swaps(tmp_path: Path) -> None:
    bundle_path, expected = _bundle_fixture(tmp_path)
    value = json.loads(bundle_path.read_text(encoding="utf-8"))
    value["execution_plan_sha256"] = "0" * 64
    forged_path = _write_bound(tmp_path / "forged-summary.json", value)
    with pytest.raises(ValueError, match="summary identity"):
        IndustrialAssignmentExecutionBundle.load(forged_path).audit_execution_plan()

    standalone_activation_path = _write_bound(
        tmp_path / "standalone-activation.json",
        {
            "schema_version": 1,
            "kind": "registry_stage_activation_artifact",
            "artifact_sha256": expected.activation.semantic_sha256,
        },
    )
    standalone_activation = replace(
        expected,
        activation=_source(
            standalone_activation_path,
            semantic_sha256=expected.activation.semantic_sha256,
        ),
    )
    with pytest.raises(
        ValueError,
        match="activation manifest|fields differ|unsupported tagged kind",
    ):
        standalone_activation.audit_execution_plan()

    copied_runtime_path = _write_bound(
        tmp_path / "copied-runtime-envelope.json",
        expected.activation_runtime.load(),
    )
    copied_runtime = _source(
        copied_runtime_path,
        semantic_sha256=expected.activation_runtime.semantic_sha256,
    )
    activation_manifest = expected.activation.load()
    activation_manifest["runtime_artifact"] = copied_runtime.path
    copied_runtime_activation_path = _write_bound(
        tmp_path / "copied-runtime-activation-manifest.json",
        activation_manifest,
    )
    copied_runtime_bundle = replace(
        expected,
        activation=_source(
            copied_runtime_activation_path,
            semantic_sha256=expected.activation.semantic_sha256,
        ),
        activation_runtime=copied_runtime,
    )
    with pytest.raises(
        ValueError,
        match=(
            "exact bound runtime-envelope|"
            "dispatch-context summary differs from raw scheduler replay"
        ),
    ):
        copied_runtime_bundle.audit_execution_plan()

    interference_receipt = expected.interference_source_receipt.load()
    interference_receipt["receipt_sha256"] = "2" * 64
    forged_interference_path = _write_bound(
        tmp_path / "forged-interference-receipt.json", interference_receipt
    )
    forged_interference = replace(
        expected,
        interference_source_receipt=_source(
            forged_interference_path,
            semantic_sha256=interference_receipt["receipt_sha256"],
        ),
    )
    with pytest.raises(
        ExecutionBundleBlockedError,
        match="calibrated_interference_raw_authority_required",
    ):
        forged_interference.audit_execution_plan()

    original_policy = budget_policy_from_dict(expected.budget_policy.load())
    forged_policy = replace(
        original_policy,
        policy_name="caller-rewritten-budget-policy",
    )
    forged_policy_path = _write_bound(
        tmp_path / "forged-budget-policy.json",
        budget_policy_to_dict(forged_policy),
    )
    forged_budget = replace(
        expected,
        budget_policy=_source(
            forged_policy_path,
            semantic_sha256=forged_policy.sha256,
        ),
    )
    with pytest.raises(ValueError, match="BudgetPlan differs"):
        forged_budget.audit_execution_plan()

    selected_binding = next(
        budget_load_binding_from_dict(source.load())
        for source in expected.budget_load_bindings
        if budget_load_binding_from_dict(source.load()).cell_id == expected.cell_id
    )
    registered = selected_binding.registered_load
    source_parameters = dict(registered.scored.source_parameters)
    alternate_templates = tuple(
        RequestTemplate(
            input_token_ids=(
                (*request.input_token_ids[:-1], request.input_token_ids[-1] + 1)
                if index == 0
                else request.input_token_ids
            ),
            requested_output_tokens=request.requested_output_tokens,
            sampling=request.sampling,
            cancellation_offset_us=request.cancellation_offset_us,
        )
        for index, request in enumerate(registered.scored.requests)
    )
    alternate_load = replace(
        registered,
        scored=closed_loop_corpus(
            alternate_templates,
            namespace=registered.scored.requests[0].namespace,
            split=registered.scored.split,
            concurrency=int(source_parameters["concurrency"]),
            cohort_count=int(source_parameters["cohort_count"]),
            cohort_popularity=str(source_parameters["cohort_popularity"]),
            cohort_seed=int(source_parameters["cohort_seed"]),
            zipf_exponent=float(source_parameters["zipf_exponent"]),
        ),
    )
    alternate_load_path = _write_bound(
        tmp_path / "alternate-production-load.json",
        production_load_plan_to_dict(alternate_load),
    )
    swapped_load = replace(
        expected,
        production_load=_source(
            alternate_load_path,
            semantic_sha256=alternate_load.paired_replay_sha256,
        ),
    )
    with pytest.raises(ValueError, match="registered budget load"):
        swapped_load.audit_execution_plan()

    split_value = expected.split_artifact.source.load()
    split_value["sampling_profile_sha256"] = "3" * 64
    swapped_split_path = _write_bound(tmp_path / "swapped-split.json", split_value)
    swapped_split = replace(
        expected,
        split_artifact=BoundExecutionArtifact(
            name=expected.split_artifact.name,
            experiment=expected.split_artifact.experiment,
            source=_source(swapped_split_path),
        ),
    )
    with pytest.raises(ValueError, match="activation split"):
        swapped_split.audit_execution_plan()

    inventory_path = Path(expected.inventory.path)
    inventory_value = expected.inventory.load()
    inventory_value["source_receipt_sha256"] = "1" * 64
    inventory_path.write_text(json.dumps(inventory_value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source or sidecar changed"):
        expected.audit_execution_plan()


def test_bundle_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    body = '{"schema_version":1,"schema_version":1}\n'
    path.write_text(body, encoding="utf-8")
    Path(f"{path}.sha256").write_text(
        hashlib.sha256(b"{}").hexdigest() + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        IndustrialAssignmentExecutionBundle.load(path)


def test_bundle_v2_requires_explicit_interference_authority_slot(
    tmp_path: Path,
) -> None:
    bundle_path, _ = _bundle_fixture(tmp_path)
    value = json.loads(bundle_path.read_text(encoding="utf-8"))
    value.pop("interference_calibration_authority")

    forged_path = _write_bound(tmp_path / "missing-calibration-authority.json", value)
    with pytest.raises(ValueError, match="fields differ"):
        IndustrialAssignmentExecutionBundle.load(forged_path)

    value["interference_calibration_authority"] = None
    value["schema_version"] = 1
    legacy_path = _write_bound(tmp_path / "legacy-bundle.json", value)
    with pytest.raises(ValueError, match="schema is unsupported"):
        IndustrialAssignmentExecutionBundle.load(legacy_path)


def test_release_authority_blocks_before_output_root_creation(tmp_path: Path) -> None:
    output_root = tmp_path / "must-not-exist"

    with pytest.raises(
        ExecutionBundleBlockedError,
        match="trusted_hardware_attester_unavailable",
    ):
        require_release_dispatch_execution_authority()

    assert not output_root.exists()


def test_dispatch_receipt_publication_is_immutable_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_parent_output = tmp_path / "missing" / "wave.json"
    with pytest.raises(
        ExecutionBundleBlockedError,
        match="dispatch_receipt_parent_unavailable",
    ):
        preflight_dispatch_receipt_output(missing_parent_output)
    assert not missing_parent_output.parent.exists()

    receipt = DispatchScheduleReceipt(
        plan_sha256=content_sha256({"plan": "empty"}),
        phase=DispatchExecutionPhase.COMPLETE,
        wave_receipts=(),
        inventory_sha256=content_sha256({"inventory": "empty"}),
        fixed_instance_gpu_count=1,
        active_intervals_monotonic_ns=(),
        fixed_instance_actual_billed_gpu_ns=0,
        per_assignment_attributed_gpu_ns=0,
        per_assignment_attributed_fixed_instance_gpu_ns=0,
    )
    receipt_path = tmp_path / "wave.json"
    published = publish_dispatch_schedule_receipt(receipt_path, receipt)
    assert publish_dispatch_schedule_receipt(receipt_path, receipt) == published
    assert published[0].is_file() and published[1].is_file()

    import lightcone_spec.orchestration.execution_bundle as bundle_module

    crash_path = tmp_path / "crash-after-envelope.json"
    original_publish = bundle_module._publish_immutable_file
    calls = 0

    def crash_before_derived_sidecar(path: Path, body: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_publish(path, body)
            return
        raise RuntimeError("injected crash before derived sidecar")

    monkeypatch.setattr(
        bundle_module,
        "_publish_immutable_file",
        crash_before_derived_sidecar,
    )
    with pytest.raises(RuntimeError, match="injected crash"):
        publish_dispatch_schedule_receipt(crash_path, receipt)
    assert crash_path.is_file()
    assert not crash_path.with_name(f"{crash_path.name}.sidecar.json").exists()
    envelope = json.loads(crash_path.read_text(encoding="utf-8"))
    assert envelope["receipt"] == receipt.to_dict()
    assert envelope["sidecar"] == receipt.sidecar().to_dict()

    monkeypatch.setattr(bundle_module, "_publish_immutable_file", original_publish)
    publish_dispatch_schedule_receipt(crash_path, receipt)
    assert crash_path.with_name(f"{crash_path.name}.sidecar.json").is_file()


def test_execute_cli_blocks_before_bundle_read_or_output_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt_output = tmp_path / "missing-parent" / "receipt.json"
    result = cli_main(
        [
            "execute-dispatch-wave",
            "--materialization-manifest",
            str(tmp_path / "missing-manifest.json"),
            "--wave-index",
            "0",
            "--receipt-output",
            str(receipt_output),
        ]
    )

    assert result == 42
    decision = json.loads(capsys.readouterr().out)
    assert decision["status"] == "BLOCKED"
    assert decision["reason_code"] == "trusted_hardware_attester_unavailable"
    assert not receipt_output.parent.exists()


def test_missing_bundle_is_named_before_any_output_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightcone_spec.orchestration.execution_bundle as bundle_module

    monkeypatch.setattr(
        bundle_module,
        "require_release_dispatch_execution_authority",
        lambda: None,
    )
    receipt_output = tmp_path / "receipt.json"
    with pytest.raises(
        RuntimeError,
        match="readable regular file",
    ):
        import asyncio

        asyncio.run(
            bundle_module.execute_dispatch_wave_bundles(
                tmp_path / "missing-manifest.json",
                wave_index=0,
                receipt_output=receipt_output,
            )
        )
    assert not receipt_output.exists()


def test_verified_dispatch_rejects_reopened_publication_mismatch_before_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import lightcone_spec.orchestration.execution_bundle as bundle_module
    import lightcone_spec.orchestration.execution_bundle_materializer as materializer
    from lightcone_spec.orchestration.execution_bundle_materializer import (
        DispatchExecutionBundleManifest,
        MaterializedAssignmentBundleReceipt,
        MaterializedDispatchExecutionBundlePublication,
    )

    bundle_path, bundle = _bundle_fixture(tmp_path)
    bundle_source = BoundJsonSource.bind(
        bundle_path,
        semantic_sha256=bundle.sha256,
    )
    member = MaterializedAssignmentBundleReceipt(
        assignment_sha256=bundle.assignment_sha256,
        cell_id=bundle.cell_id,
        run_nonce_sha256=bundle.run_nonce_sha256,
        execution_plan_sha256=bundle.execution_plan_sha256,
        launch_policy=bundle.run_config,
        run_nonce_receipt=bundle.run_config,
        bundle=bundle_source,
    )
    manifest = DispatchExecutionBundleManifest(
        schema_version=1,
        kind="industrial_dispatch_execution_bundle_manifest",
        bundle_schema_version=5,
        materialization_inputs_sha256="a" * 64,
        request=bundle.registry,
        dispatch_plan=bundle.dispatch_plan,
        assignments=(member,),
    )
    verified = MaterializedDispatchExecutionBundlePublication(
        manifest=manifest,
        bundles=(bundle,),
    )
    reopened = MaterializedDispatchExecutionBundlePublication(
        manifest=replace(manifest, materialization_inputs_sha256="b" * 64),
        bundles=(bundle,),
    )
    assert type(verified) is MaterializedDispatchExecutionBundlePublication
    assert type(reopened) is MaterializedDispatchExecutionBundlePublication
    assert reopened != verified

    loader_calls: list[Path] = []

    def reopen_another_publication(path: Path) -> object:
        loader_calls.append(path)
        return reopened

    monkeypatch.setattr(
        bundle_module,
        "require_release_dispatch_execution_authority",
        lambda: None,
    )
    monkeypatch.setattr(
        materializer,
        "load_materialized_dispatch_execution_bundle_publication",
        reopen_another_publication,
    )
    journal_opened = False
    transport_opened = False

    def forbidden_journal(*args: object, **kwargs: object) -> object:
        nonlocal journal_opened
        journal_opened = True
        raise AssertionError("publication mismatch reached the attempt journal")

    def forbidden_transport(*args: object, **kwargs: object) -> object:
        nonlocal transport_opened
        transport_opened = True
        raise AssertionError("publication mismatch reached serving transport")

    monkeypatch.setattr(
        bundle_module.DispatchAttemptJournal,
        "open_or_create",
        forbidden_journal,
    )
    monkeypatch.setattr(
        bundle_module.PinnedBenchServingTransport,
        "from_checkout",
        forbidden_transport,
    )
    manifest_path = (tmp_path / "dispatch-manifest.json").resolve()

    with pytest.raises(
        ValueError,
        match="verified publication differs from its source membership",
    ):
        asyncio.run(
            bundle_module.execute_dispatch_wave_bundles(
                manifest_path,
                wave_index=0,
                receipt_output=tmp_path / "must-not-publish.json",
                _verified_publication=verified,
                _verified_plans=(object(),),  # type: ignore[arg-type]
            )
        )

    assert loader_calls == [manifest_path]
    assert not journal_opened
    assert not transport_opened
    assert not (tmp_path / "must-not-publish.json").exists()


def test_dispatch_resume_requires_exact_expected_receipt_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import lightcone_spec.orchestration.execution_bundle as bundle_module

    monkeypatch.setattr(
        bundle_module,
        "require_release_dispatch_execution_authority",
        lambda: None,
    )
    resume = (tmp_path / "resume.json").resolve()
    resume.write_text("{}\n", encoding="utf-8")
    receipt_output = (tmp_path / "next.json").resolve()

    with pytest.raises(ValueError, match="require a resume receipt path"):
        asyncio.run(
            bundle_module.execute_dispatch_wave_bundles(
                (tmp_path / "manifest.json").resolve(),
                wave_index=0,
                receipt_output=receipt_output,
                expected_resume_receipt_sha256="a" * 64,
            )
        )

    receipt = SimpleNamespace(sha256="a" * 64)
    envelope_sha256 = "f" * 64
    monkeypatch.setattr(
        bundle_module,
        "_load_dispatch_schedule_envelope",
        lambda *args, **kwargs: (receipt, object(), envelope_sha256),
    )
    # Reach the sole resume decoder without reconstructing the expensive
    # source graph; the identity mismatch must still precede journal reopen.
    assignment = SimpleNamespace(
        assignment_id="c" * 64,
        work_item=SimpleNamespace(
            item_id="cell",
            cell=SimpleNamespace(
                resources=SimpleNamespace(
                    workload_class=bundle_module.WorkloadClass.HEADLINE
                )
            ),
        ),
    )
    wave = SimpleNamespace(assignments=(assignment,))
    fake_plan = SimpleNamespace(waves=(wave,))
    fake_context = SimpleNamespace(
        inventory=SimpleNamespace(sha256="b" * 64),
        budgets=(),
        require_ready_budget_authority=lambda: (),
    )
    bundle = SimpleNamespace(
        assignment_sha256=assignment.assignment_id,
        cell_id="cell",
        reconstruct_execution_plan=lambda: SimpleNamespace(
            dispatch_plan=fake_plan,
            dispatch_context=fake_context,
            runtime_plan=SimpleNamespace(
                physical_assignment=SimpleNamespace(
                    assignment_sha256=assignment.assignment_id
                )
            ),
        ),
    )
    publication = SimpleNamespace(
        manifest=SimpleNamespace(sha256="d" * 64),
        bundles=(bundle,),
    )
    import lightcone_spec.orchestration.execution_bundle_materializer as materializer

    monkeypatch.setattr(
        materializer,
        "load_materialized_dispatch_execution_bundle_publication",
        lambda _path: publication,
    )
    monkeypatch.setattr(
        bundle_module, "_require_shared_bundle_authority", lambda _: None
    )
    monkeypatch.setattr(
        bundle_module, "_preflight_bundle_assignment_sources", lambda _: None
    )
    monkeypatch.setattr(
        bundle_module,
        "_preflight_bundle_trainable_plan_release_trust",
        lambda _: None,
    )
    monkeypatch.setattr(
        bundle_module,
        "_declared_wave_assignment_ids",
        lambda *args, **kwargs: (bundle.assignment_sha256,),
    )
    with pytest.raises(
        ExecutionBundleBlockedError,
        match="dispatch_resume_receipt_content_mismatch",
    ):
        asyncio.run(
            bundle_module.execute_dispatch_wave_bundles(
                (tmp_path / "manifest.json").resolve(),
                wave_index=0,
                receipt_output=receipt_output,
                resume_receipt_path=resume,
                expected_resume_receipt_sha256="e" * 64,
            )
        )
    journal_opened = False

    def forbidden_journal_open(*args, **kwargs):
        nonlocal journal_opened
        journal_opened = True
        raise AssertionError("journal must not open after envelope mismatch")

    monkeypatch.setattr(
        bundle_module.DispatchAttemptJournal,
        "open_or_create",
        forbidden_journal_open,
    )
    with pytest.raises(
        ExecutionBundleBlockedError,
        match="dispatch_resume_receipt_envelope_mismatch",
    ):
        asyncio.run(
            bundle_module.execute_dispatch_wave_bundles(
                (tmp_path / "manifest.json").resolve(),
                wave_index=0,
                receipt_output=receipt_output,
                resume_receipt_path=resume,
                expected_resume_receipt_sha256=receipt.sha256,
                expected_resume_receipt_envelope_sha256="e" * 64,
            )
        )
    assert not journal_opened


def test_formal_dispatch_entry_rejects_raw_bundle_paths_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import lightcone_spec.orchestration.execution_bundle as bundle_module
    import lightcone_spec.orchestration.execution_bundle_materializer as materializer

    monkeypatch.setattr(
        bundle_module,
        "require_release_dispatch_execution_authority",
        lambda: None,
    )
    loaded = False

    def forbidden_loader(path):
        nonlocal loaded
        loaded = True
        raise AssertionError("raw bundle input reached the manifest loader")

    monkeypatch.setattr(
        materializer,
        "load_materialized_dispatch_execution_bundle_publication",
        forbidden_loader,
    )
    with pytest.raises(TypeError, match="one manifest path"):
        asyncio.run(
            bundle_module.execute_dispatch_wave_bundles(
                (str(tmp_path / "raw-bundle.json"),),  # type: ignore[arg-type]
                wave_index=0,
                receipt_output=tmp_path / "receipt.json",
            )
        )
    assert not loaded
    assert not (tmp_path / "receipt.json").exists()
