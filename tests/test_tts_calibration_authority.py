from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_control_attestation import _root_binding
from test_formal_gpu_hour_registry import (
    _bundle as _registry_bundle,
)
from test_formal_gpu_hour_registry import (
    _control as _registry_control,
)
from test_formal_gpu_hour_registry import _deployment as _registry_deployment
from test_formal_gpu_hour_registry import (
    _extend_registry_with_e3a,
    _proof_wrapped_signed_row,
    _registry_receipt,
)
from test_formal_gpu_hour_registry import _sign as _registry_sign
from test_formal_registry_integration import (
    _complete_coverage,
    _synthetic_e3a_artifact,
)
from test_formal_stage_coverage import _runtime_manifest as _coverage_runtime_manifest

from lightcone_spec.experiments.e3a_stage_authority import (
    SignedE3aStagedSelectionReceipt,
    build_e3a_staged_selection_receipt,
)
from lightcone_spec.experiments.e3a_staged_selection_proof import (
    bind_formal_e3a_staged_selection_proof_artifact,
    publish_formal_e3a_staged_selection_proof_artifact,
)
from lightcone_spec.experiments.formal_protocol import (
    TTS_PRIMARY_SOURCE_ID,
    TTS_PRIMARY_SOURCE_VERSION,
    SignedTtsCalibrationSeal,
    TtsCalibrationAuthority,
    TtsCalibrationSeal,
)
from lightcone_spec.experiments.formal_registry import (
    extend_formal_registry_verification_receipt,
    formal_runtime_authority_manifest_to_dict,
    protocol_lock_to_dict,
    stage_materialization_receipt_to_dict,
    tts_calibration_authority_to_dict,
)
from lightcone_spec.experiments.formal_registry_layers import (
    bind_formal_registry_layer_artifact,
    publish_formal_registry_layer_artifact,
    validate_formal_precoverage_registry_state,
)
from lightcone_spec.experiments.formal_stage_coverage import (
    FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256,
    FormalStageCoverageProofArtifact,
    derived_coverage_shards,
    publish_formal_stage_coverage_proof_artifact,
    publish_formal_stage_derived_coverage_shard,
    reduce_tts_calibration_stage_coverage_from_proofs,
    revalidate_formal_stage_coverage_proof_artifact,
)
from lightcone_spec.experiments.formal_stage_coverage_portable import (
    bind_formal_portable_stage_coverage_proof_artifact,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.industrial_analysis import (
    BoundArtifact,
    IndustrialCellEvidence,
    RawTtsCalibrationEvidenceManifest,
    TtsCalibrationPilotEvidence,
    _LoadedCell,
    raw_tts_calibration_manifest_from_dict,
    raw_tts_calibration_manifest_to_dict,
)
from lightcone_spec.experiments.registry import (
    PILOT_BLOCKS,
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    SignedStageCoverageReceipt,
    SignedStageMaterializationReceipt,
    StageCellDisposition,
    StageCoverageReceipt,
    StageMaterializationReceipt,
    _materialize_tts_calibration_diagnostic,
)
from lightcone_spec.experiments.statistics import HardwareEnvelope
from lightcone_spec.experiments.tts_calibration_authority import (
    FormalTtsCalibrationReductionProofArtifact,
    TtsCalibrationReductionReceipt,
    reduce_tts_calibration_from_raw,
    seal_tts_calibration_reduction,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
    attestation_message,
)
from lightcone_spec.runtime.control_attestation import ChallengeReplayStore
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.telemetry.records import OUTPUT_HASH_FORMAT

_SAFETY_COUNTERS = (
    "exactness_violations",
    "version_mismatches",
    "fallbacks",
    "nonfinite_updates",
    "oom_events",
    "retractions",
    "communicator_failures",
    "evidence_dropped_rows",
)


def _sha(label: str) -> str:
    return content_sha256({"tts-calibration-test": label})


def _sign_tts_seal(
    seal: TtsCalibrationSeal,
    *,
    now_ns: int,
) -> tuple[SignedTtsCalibrationSeal, TrustedAttesterPolicy]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_base64 = base64.b64encode(public_key).decode()
    public_sha256 = hashlib.sha256(public_key).hexdigest()
    policy = TrustedAttesterPolicy(
        policy_id="tts-reduction-proof-test",
        trusted_attesters=(("tts-reducer", "tts-key", public_sha256),),
        public_keys=((public_sha256, public_base64),),
    )
    payload_sha256 = content_sha256(seal)
    challenge = AttestationChallenge.issue(
        challenge_id="tts-reduction-proof-challenge",
        subject_sha256=payload_sha256,
        lifetime_s=60,
        now_ns=now_ns,
    )
    signature = private_key.sign(
        attestation_message(challenge, payload_sha256=payload_sha256)
    )
    signed = SignedTtsCalibrationSeal(
        payload=seal,
        payload_sha256=payload_sha256,
        challenge=challenge,
        attestation=SignedAttestation(
            schema_version=1,
            kind="lightcone_signed_attestation",
            algorithm="Ed25519",
            attester_id="tts-reducer",
            key_id="tts-key",
            environment="release",
            public_key_base64=public_base64,
            challenge_sha256=challenge.sha256,
            payload_sha256=payload_sha256,
            signature_base64=base64.b64encode(signature).decode(),
        ),
    )
    return signed, policy


def _authority() -> TtsCalibrationAuthority:
    return TtsCalibrationAuthority(
        schema_version=1,
        authority_id="tts-arxiv-v2-numeric-calibration",
        primary_source_id=TTS_PRIMARY_SOURCE_ID,
        primary_source_version=TTS_PRIMARY_SOURCE_VERSION,
        paper_pdf_sha256=_sha("paper-pdf"),
        paper_source_sha256=_sha("paper-source"),
        tuning_window_sha256=_sha("tuning-window"),
        trainable_plan_sha256=_sha("trainable-plan"),
        drafter_native_loss_recipe_sha256=_sha("native-loss"),
    )


def _unmeasured() -> GpuHourEstimate:
    return GpuHourEstimate(
        status="UNMEASURED",
        source_pilot_receipt_sha256=None,
        compute_gpu_hours=None,
        reserved_gpu_hours=None,
        estimated_wall_hours=None,
        retry_reserve_gpu_hours=None,
        profile_reserve_gpu_hours=None,
        evidence_reserve_gpu_hours=None,
    )


def _inventory() -> GpuInventory:
    device = GpuDevice(
        uuid="GPU-tts-calibration-0",
        host_id="tts-calibration-host",
        model="RTX PRO 6000 Blackwell Server Edition",
        memory_bytes=96 * 1024**3,
        compute_capability=(12, 0),
        pci_bus_id="0000:01:00.0",
        pci_root="root-0",
        numa_node=0,
        interconnects=("PCIe",),
        peer_access_class="P2P",
        clock_policy="locked",
        power_limit_watts=600.0,
        thermal_limit_celsius=83.0,
        availability=GpuAvailability.READY,
        reserved_processes=(),
        allowed_topology_groups=("single",),
    )
    return GpuInventory(
        schema_version=1,
        devices=(device,),
        topology_groups=(
            GpuTopologyGroup(
                group_id="single",
                host_id=device.host_id,
                gpu_uuids=(device.uuid,),
                fabric="PCIe",
                bandwidth_class="local",
            ),
        ),
        source_receipt_sha256=_sha("inventory-source"),
    )


def _hardware() -> HardwareEnvelope:
    return HardwareEnvelope(
        gpu_clock_mhz_min=1.0,
        gpu_clock_mhz_max=4_000.0,
        memory_clock_mhz_min=1.0,
        memory_clock_mhz_max=10_000.0,
        temperature_c_max=90.0,
        power_watts_min=1.0,
        power_watts_max=1_000.0,
        power_state="P0",
    )


def _request_row(*, duration_ns: int) -> dict[str, object]:
    token_ids = (101, 102)
    token_body = json.dumps(list(token_ids), separators=(",", ":"))
    token_sha256 = hashlib.sha256(token_body.encode()).hexdigest()
    return {
        "request_id": "request-0",
        "input_tokens": 32,
        "arrival_ns": 0,
        "admitted_ns": 0,
        "first_token_ns": 1,
        "completed_ns": duration_ns,
        "ttft_ms": 0.000001,
        "inter_token_ms": json.dumps([(duration_ns - 1) / 1_000_000.0]),
        "token_timestamps_ns": json.dumps([1, duration_ns]),
        "token_timing_coverage": 1.0,
        "coalesced_intervals": 0,
        "output_tokens": 2,
        "outcome_status": "completed",
        "finished": True,
        "output_hash_format": OUTPUT_HASH_FORMAT,
        "output_token_ids": token_body,
        "output_token_ids_sha256": token_sha256,
        "output_sha256": token_sha256,
    }


def _raw_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    authority: TtsCalibrationAuthority | None = None,
    protocol_lock_sha256: str | None = None,
    upstream_e3a_receipt_sha256: str | None = None,
    now_ns: int = 10_000_000_000,
):
    from lightcone_spec.experiments import industrial_analysis
    from lightcone_spec.experiments import tts_calibration_authority as calibration

    registry = build_industrial_registry()
    authority = _authority() if authority is None else authority
    materialization = _materialize_tts_calibration_diagnostic(
        protocol_lock_sha256=(
            _sha("protocol-lock")
            if protocol_lock_sha256 is None
            else protocol_lock_sha256
        ),
        upstream_e3a_receipt_sha256=(
            _sha("e3a-coverage")
            if upstream_e3a_receipt_sha256 is None
            else upstream_e3a_receipt_sha256
        ),
        calibration_authority_sha256=authority.sha256,
        gpu_hours=_unmeasured(),
    )
    materialized_by_registry = {
        str(dict(cell.dimensions)["registry_cell_id"]): cell
        for cell in materialization.cells
    }
    registry_cells = registry.cells_for("TTS-Cal")
    terminal_by_registry = {
        cell.cell_id: _sha(f"terminal-{cell.cell_id}") for cell in registry_cells
    }
    coverage = StageCoverageReceipt(
        schema_version=2,
        stage="TTS-Cal",
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(
            sorted(
                (
                    StageCellDisposition(
                        stage="TTS-Cal",
                        cell_id=materialized_by_registry[cell.cell_id].cell_id,
                        status="COMPLETE",
                        reason_code="terminal_complete",
                        terminal_receipt_sha256=terminal_by_registry[cell.cell_id],
                    )
                    for cell in registry_cells
                ),
                key=lambda row: row.cell_id,
            )
        ),
    )
    request_ids_sha256 = content_sha256(["request-0"])
    runtime_sha256 = _sha("runtime")
    split_sha256 = _sha("split")
    loaded: dict[str, _LoadedCell] = {}
    references: dict[str, IndustrialCellEvidence] = {}
    controls: dict[str, BoundArtifact] = {}
    qualification_values: dict[Path, dict[str, object]] = {}
    pilots: list[TtsCalibrationPilotEvidence] = []
    for block in PILOT_BLOCKS:
        block_cells = tuple(
            cell for cell in registry_cells if cell.identity.block == block
        )
        qualification_path = (tmp_path / f"qualification-{block}.json").resolve()
        qualification = BoundArtifact(
            qualification_path, _sha(f"qualification-{block}")
        )
        qualification_values[qualification_path] = {
            "schema_version": 1,
            "kind": "tts_calibration_request_qualification_lock",
            "registry_sha256": registry.sha256,
            "protocol_lock_sha256": materialization.protocol_lock_sha256,
            "materialization_receipt_sha256": materialization.sha256,
            "authority_sha256": authority.sha256,
            "tuning_window_sha256": authority.tuning_window_sha256,
            "runtime_sha256": runtime_sha256,
            "split_sha256": split_sha256,
            "block": block,
            "corpus_sha256": _sha("corpus"),
            "arrival_trace_sha256": _sha(f"arrival-{block}"),
            "request_ids_sha256": request_ids_sha256,
            "sampling_profile_sha256": _sha("sampling"),
            "model_lock_sha256": _sha("model-lock"),
            "rows": [
                {
                    "request_id": "request-0",
                    "prompt_bucket": "short",
                    "eligible": True,
                }
            ],
        }
        block_references: list[IndustrialCellEvidence] = []
        for index, cell in enumerate(block_cells):
            stride = int(cell.identity.variant.removeprefix("tts_calibration:stride="))
            candidate_id = authority.candidate_id(
                learning_rate=float(cell.identity.learning_rate), stride=stride
            )
            winner = candidate_id == authority.candidate_ids[0]
            duration_ns = 10_000_000 if winner else 50_000_000 + index
            terminal = BoundArtifact(
                (tmp_path / f"terminal-{cell.cell_id}.json").resolve(),
                terminal_by_registry[cell.cell_id],
            )
            reference = IndustrialCellEvidence(
                cell_id=cell.cell_id,
                terminal_receipts=(terminal,),
                hardware_receipt=BoundArtifact(
                    (tmp_path / f"hardware-{cell.cell_id}.json").resolve(),
                    _sha(f"hardware-{cell.cell_id}"),
                ),
                budget_observation=BoundArtifact(
                    (tmp_path / f"budget-{cell.cell_id}.json").resolve(),
                    _sha(f"budget-{cell.cell_id}"),
                ),
                completion_contract=BoundArtifact(
                    (tmp_path / f"completion-{cell.cell_id}.json").resolve(),
                    _sha(f"completion-{cell.cell_id}"),
                ),
            )
            references[cell.cell_id] = reference
            controls[cell.cell_id] = BoundArtifact(
                (tmp_path / f"terminal-control-{cell.cell_id}.json").resolve(),
                _sha(f"terminal-control-{cell.cell_id}"),
            )
            block_references.append(reference)
            run = {
                "model_pair": "tts-calibration-model-pair",
                "corpus_sha256": _sha("corpus"),
                "arrival_trace_sha256": _sha(f"arrival-{block}"),
                "request_ids_sha256": request_ids_sha256,
                "sampling_profile_sha256": _sha("sampling"),
                "model_lock_sha256": _sha("model-lock"),
                "patched_sglang_tree": "1" * 40,
                "config_sha256": _sha(f"config-{cell.cell_id}"),
                "rank_config_sha256": _sha(f"rank-config-{cell.cell_id}"),
                "run_id": f"tts-calibration-{block}-{index}",
                "run_nonce_sha256": _sha(f"nonce-{cell.cell_id}"),
                "topology_sha256": _sha(f"topology-{cell.cell_id}"),
                "experiment_budget_sha256": _sha(f"budget-semantic-{cell.cell_id}"),
                "runtime_sha256": _sha(f"execution-{cell.cell_id}"),
                "split_sha256": _sha(f"execution-split-{cell.cell_id}"),
            }
            loaded[cell.cell_id] = _LoadedCell(
                cell=cell,
                observation_source_cell_id=cell.cell_id,
                evidence_alias_reduction_sha256=None,
                run_rows=(run,),
                request_rows=(_request_row(duration_ns=duration_ns),),
                performance_rows_by_rank=(
                    (
                        {
                            "offered_requests": 1,
                            **{counter: 0 for counter in _SAFETY_COUNTERS},
                        },
                    ),
                ),
                update_rows_by_rank=((),),
                terminal_receipt_sha256s=(terminal.sha256,),
                hardware_receipt_sha256=reference.hardware_receipt.sha256,
                physical_gpu_uuids=("GPU-tts-calibration-0",),
                experiment_budget_sha256=run["experiment_budget_sha256"],
                inventory_sha256=_sha("inventory"),
                inventory_source_receipt_sha256=_sha("inventory-source"),
                fixed_instance_gpu_count=1,
                physical_host_id="tts-calibration-host",
                budget_observation_sha256=reference.budget_observation.sha256,
                hardware_validity=((reference.hardware_receipt.sha256, "VALID", ()),),
                itl_timestamp_authority_path=None,
            )
        pilots.append(
            TtsCalibrationPilotEvidence(
                block=block,
                qualification_lock=qualification,
                cells=tuple(sorted(block_references, key=lambda row: row.cell_id)),
                terminal_control_attestations=tuple(
                    controls[row.cell_id]
                    for row in sorted(block_references, key=lambda row: row.cell_id)
                ),
            )
        )
    manifest = RawTtsCalibrationEvidenceManifest(
        schema_version=2,
        tuning_window=BoundArtifact(
            (tmp_path / "tuning-window.json").resolve(),
            authority.tuning_window_sha256,
        ),
        pilots=tuple(pilots),
    )
    monkeypatch.setattr(
        industrial_analysis, "validate_raw_evidence_manifest_sidecars", lambda _: None
    )
    monkeypatch.setattr(
        industrial_analysis,
        "_load_cell",
        lambda reference, **_kwargs: loaded[reference.cell_id],
    )
    monkeypatch.setattr(
        industrial_analysis,
        "_bound_json",
        lambda path, _sha256, **_kwargs: qualification_values[path],
    )

    def fake_terminal_batch(**kwargs):
        replay_reservation = kwargs.get("replay_reservation")
        if replay_reservation is None:
            replay_store = kwargs["replay_store"]
            replay_reservation = replay_store.reserve_verified_content_challenges(
                tuple(
                    sorted(
                        _sha(f"terminal-control-challenge-{cell_id}")
                        for cell_id in loaded
                    )
                ),
                reserved_ns=kwargs["now_ns"],
            )
        reservation = replay_reservation.reservation_sha256
        policy = _sha("terminal-control-policy")
        return tuple(
            calibration._ControlledTtsCalibrationTerminal(
                registry_cell_id=cell_id,
                canonical_raw_sha256=references[cell_id].terminal_receipts[0].sha256,
                terminal_sha256=_sha(f"terminal-semantic-{cell_id}"),
                run_id=str(loaded[cell_id].run_rows[0]["run_id"]),
                run_nonce_sha256=str(loaded[cell_id].run_rows[0]["run_nonce_sha256"]),
                execution_plan_sha256=str(
                    loaded[cell_id].run_rows[0]["runtime_sha256"]
                ),
                rank_config_sha256=str(
                    loaded[cell_id].run_rows[0]["rank_config_sha256"]
                ),
                method="tts",
                control_binding_sha256=_sha(f"control-binding-{cell_id}"),
                control_envelope_sha256=_sha(f"control-envelope-{cell_id}"),
                control_reservation_sha256=reservation,
                control_policy_sha256=policy,
            )
            for cell_id in sorted(loaded)
        )

    monkeypatch.setattr(
        calibration,
        "_validate_tts_calibration_native_terminal_batch",
        fake_terminal_batch,
    )
    replay_root = tmp_path / "terminal-control-replay"
    replay_root.mkdir(parents=True)
    return {
        "registry": registry,
        "authority": authority,
        "materialization": materialization,
        "coverage": coverage,
        "manifest": manifest,
        "hardware_envelope": _hardware(),
        "inventory": _inventory(),
        "runtime_sha256": runtime_sha256,
        "split_sha256": split_sha256,
        "replay_store": ChallengeReplayStore(str(replay_root.resolve())),
        "now_ns": now_ns,
        "loaded": loaded,
    }


def _reduce(bundle: dict[str, object]) -> TtsCalibrationReductionReceipt:
    return reduce_tts_calibration_from_raw(
        **{key: value for key, value in bundle.items() if key != "loaded"}
    )


def _publish_real_tts_coverage_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    lock,
    runtime,
    bundle: dict[str, object],
) -> tuple[Path, StageCoverageReceipt, dict[str, Path]]:
    from lightcone_spec.experiments import formal_stage_coverage as coverage_module

    authority = bundle["authority"]
    materialization = bundle["materialization"]
    manifest = bundle["manifest"]
    inventory = bundle["inventory"]
    assert isinstance(authority, TtsCalibrationAuthority)
    assert isinstance(materialization, StageMaterializationReceipt)
    assert isinstance(manifest, RawTtsCalibrationEvidenceManifest)
    assert isinstance(inventory, GpuInventory)

    monkeypatch.setattr(
        coverage_module,
        "validate_raw_evidence_manifest_sidecars",
        lambda _value: None,
    )
    monkeypatch.setattr(
        coverage_module,
        "_validate_tts_qualification_lock",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        coverage_module,
        "_load_bound_json",
        lambda reference, *, label: {
            "path": str(reference.path),
            "sha256": reference.sha256,
            "label": label,
        },
    )

    class FakeControl:
        def __init__(self, cell_id: str) -> None:
            self.cell_id = cell_id
            self.deployment_policy_authorization = SimpleNamespace(
                root_manifest_sha256=lock.offline_release_trust_root_sha256
            )

    monkeypatch.setattr(
        coverage_module,
        "ControlArtifactAttestation",
        SimpleNamespace(
            from_dict=lambda value: FakeControl(
                Path(value["path"]).stem.removeprefix("terminal-control-")
            )
        ),
    )

    def prepare(value, **_kwargs):
        cell_id = Path(value["path"]).stem.removeprefix("terminal-")
        return SimpleNamespace(
            binding=SimpleNamespace(
                canonical_raw_sha256=value["sha256"],
                sha256=_sha(f"control-binding-{cell_id}"),
            ),
            evidence=SimpleNamespace(
                binding=SimpleNamespace(
                    method="tts",
                    run_id=f"tts-{cell_id}",
                    run_nonce_sha256=_sha(f"run-nonce-{cell_id}"),
                )
            ),
        )

    def verify(control, **_kwargs):
        return SimpleNamespace(
            artifact_sha256=_sha(f"control-binding-{control.cell_id}"),
            challenge_sha256=_sha(f"coverage-challenge-{control.cell_id}"),
            deployment_policy_challenge_sha256=_sha(
                f"coverage-deployment-{control.cell_id}"
            ),
        )

    monkeypatch.setattr(
        coverage_module,
        "prepare_native_terminal_external_control",
        prepare,
    )
    monkeypatch.setattr(
        coverage_module,
        "verify_release_control_artifact_attestation",
        verify,
    )
    coverage = reduce_tts_calibration_stage_coverage_from_proofs(
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime,
        materialization=materialization,
        inventory=inventory,
        authority=authority,
        manifest=manifest,
        now_ns=int(bundle["now_ns"]),
    )
    bundle["coverage"] = coverage

    sources = {
        "protocol": tmp_path / "tts-proof-protocol.json",
        "runtime": tmp_path / "tts-proof-runtime.json",
        "materialization": tmp_path / "tts-proof-materialization.json",
        "inventory": tmp_path / "tts-proof-inventory.json",
        "authority": tmp_path / "tts-proof-authority.json",
        "manifest": tmp_path / "tts-proof-raw-manifest.json",
        "hardware": tmp_path / "tts-proof-hardware.json",
    }
    for name, value in (
        ("protocol", protocol_lock_to_dict(lock)),
        ("runtime", formal_runtime_authority_manifest_to_dict(runtime)),
        ("materialization", stage_materialization_receipt_to_dict(materialization)),
        ("inventory", inventory.to_dict()),
        ("authority", tts_calibration_authority_to_dict(authority)),
        ("manifest", raw_tts_calibration_manifest_to_dict(manifest)),
        ("hardware", asdict(bundle["hardware_envelope"])),
    ):
        publish_canonical_json_no_replace(sources[name], value)
    derived_sources = tuple(
        publish_formal_stage_derived_coverage_shard(
            shard,
            tmp_path / f"tts-derived-coverage-{index:02d}.json",
        )
        for index, shard in enumerate(
            derived_coverage_shards(
                coverage,
                phase="calibration",
                maximum_dispositions_per_shard=128,
            )
        )
    )
    proof = FormalStageCoverageProofArtifact(
        schema_version=1,
        kind="formal_stage_coverage_proof_artifact",
        protocol_sha256=FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256,
        stage="TTS-Cal",
        phase="calibration",
        protocol_lock_sha256=lock.sha256,
        formal_runtime_authority_manifest_sha256=runtime.sha256,
        materialization_receipt_sha256=materialization.sha256,
        inventory_sha256=inventory.sha256,
        coverage_receipt_sha256=coverage.sha256,
        protocol_lock_source=CanonicalJsonProofBinding.bind(sources["protocol"]),
        runtime_authority_source=CanonicalJsonProofBinding.bind(sources["runtime"]),
        materialization_source=CanonicalJsonProofBinding.bind(
            sources["materialization"]
        ),
        inventory_source=CanonicalJsonProofBinding.bind(sources["inventory"]),
        tts_authority_source=CanonicalJsonProofBinding.bind(sources["authority"]),
        raw_tts_evidence_source=CanonicalJsonProofBinding.bind(sources["manifest"]),
        stage_source_rebuild_input_source=None,
        evidence_shard_sources=(),
        execution_rebuild_shard_sources=(),
        candidate_replay_proof_sources=(),
        derived_coverage_shard_sources=derived_sources,
    )
    proof_path = tmp_path / "tts-coverage-proof.json"
    publish_formal_stage_coverage_proof_artifact(proof, proof_path)
    assert (
        revalidate_formal_stage_coverage_proof_artifact(
            str(proof_path),
            now_ns=int(bundle["now_ns"]),
        )
        == coverage
    )
    return proof_path, coverage, sources


def test_exact_288_raw_cells_select_and_seal_one_frozen_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _raw_bundle(tmp_path, monkeypatch)
    reduction = _reduce(bundle)
    authority = bundle["authority"]
    assert isinstance(authority, TtsCalibrationAuthority)
    assert len(reduction.observations) == 288
    assert reduction.selected_candidate_id == authority.candidate_ids[0]
    assert len(reduction.selected_pilot_run_binding_sha256s) == 4
    seal = seal_tts_calibration_reduction(
        reduction,
        authority=authority,
        materialization=bundle["materialization"],
        coverage=bundle["coverage"],
    )
    assert seal.schema_version == 2
    assert seal.reduction_receipt_sha256 == reduction.sha256
    assert seal.selected_pilot_run_binding_sha256s == (
        reduction.selected_pilot_run_binding_sha256s
    )
    seal.validate_against(authority)


def test_tts_reduction_proof_rejects_unrooted_legacy_source_graph() -> None:
    legacy_unrooted = {
        "schema_version": 1,
        "kind": "formal_tts_calibration_reduction_proof_artifact",
        "authority_source": None,
        "materialization_source": None,
        "coverage_proof_source": None,
        "raw_manifest_source": None,
        "hardware_envelope_source": None,
        "inventory_source": None,
        "replay_reservation": None,
        "runtime_sha256": _sha("runtime"),
        "split_sha256": _sha("split"),
        "reduction_payload": {},
        "expected_reduction_sha256": _sha("reduction"),
        "expected_seal_payload_sha256": _sha("seal"),
        "artifact_sha256": _sha("artifact"),
    }
    with pytest.raises(ValueError, match="fields differ"):
        FormalTtsCalibrationReductionProofArtifact.from_dict(legacy_unrooted)


def test_registry_layer_blocks_tts_until_source_owned_exact288_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.runtime import release_trust_root as root_module

    registry_now_ns = 2_000_000_000
    authority = _authority()
    runtime = _coverage_runtime_manifest()
    inventory = _inventory()
    root_private = Ed25519PrivateKey.generate()
    controller_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    monkeypatch.setattr(
        root_module,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    registry_bundle = _registry_bundle(controller_private)
    preflight_root = tmp_path / "preflight-prefix"
    preflight_root.mkdir()
    lock, preflight_receipt, preflight_materialization, preflight_layer = (
        _registry_receipt(
            preflight_root,
            monkeypatch,
            inventory=inventory,
            runtime_manifest=runtime,
            root_private=root_private,
            controller_private=controller_private,
            root_binding=root_binding,
            bundle=registry_bundle,
            tts_calibration_authority_sha256=authority.sha256,
        )
    )
    e3a_root = tmp_path / "e3a-prefix"
    e3a_root.mkdir()
    e3a_receipt, e3a_materialization, e3a_layer = _extend_registry_with_e3a(
        e3a_root,
        monkeypatch,
        lock=lock,
        prior_receipt=preflight_receipt,
        prior_layer_path=preflight_layer,
        preflight_materialization=preflight_materialization,
        inventory=inventory,
        root_private=root_private,
        controller_private=controller_private,
        root_binding=root_binding,
        bundle=registry_bundle,
    )
    e3a_coverage = _complete_coverage(e3a_materialization)
    with pytest.raises(ValueError, match="predecessor prefix differs"):
        validate_formal_precoverage_registry_state(
            e3a_receipt,
            stage="E3a",
            phase="capacity",
            materialization=e3a_materialization,
            immediate_predecessor_prefix_sha256=_sha("foreign-prefix"),
        )
    e3a_artifact = replace(
        _synthetic_e3a_artifact(
            lock=lock,
            materialization=e3a_materialization,
            coverage=e3a_coverage,
        ),
        inventory_sha256=inventory.sha256,
    )
    e3a_selection = build_e3a_staged_selection_receipt(e3a_artifact)
    signed_e3a_selection = SignedE3aStagedSelectionReceipt(
        e3a_selection,
        *_registry_sign(controller_private, e3a_selection, 40),
    )
    e3a_coverage_proof_path = tmp_path / "e3a-coverage-reducer-proof.json"
    publish_canonical_json_no_replace(
        e3a_coverage_proof_path,
        {"schema_version": 1, "kind": "test_e3a_coverage_reducer_proof"},
    )
    from lightcone_spec.experiments import e3a_staged_selection_proof as e3a_proof

    monkeypatch.setattr(
        e3a_proof,
        "FormalStageCoverageProofArtifact",
        SimpleNamespace(
            from_dict=lambda _value: SimpleNamespace(
                stage="E3a",
                phase="capacity",
            )
        ),
    )
    monkeypatch.setattr(
        e3a_proof,
        "rebuild_formal_stage_coverage_context",
        lambda _path, *, now_ns: SimpleNamespace(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime,
            materialization=e3a_materialization,
            coverage=e3a_coverage,
            inventory=inventory,
            execution_bindings=tuple(object() for _ in range(360)),
        ),
    )
    monkeypatch.setattr(e3a_proof, "_coverage_evidence_rows", lambda _proof: ())
    monkeypatch.setattr(
        e3a_proof,
        "E3aStagedEvidenceManifest",
        lambda **_kwargs: SimpleNamespace(sha256=e3a_artifact.evidence_manifest_sha256),
    )
    monkeypatch.setattr(
        e3a_proof,
        "reduce_e3a_staged_selection_from_proofs",
        lambda **_kwargs: e3a_artifact,
    )
    e3a_reduction_proof = bind_formal_e3a_staged_selection_proof_artifact(
        coverage_proof_path=e3a_coverage_proof_path,
        registry_layer_path=e3a_layer,
        now_ns=registry_now_ns,
    )
    e3a_reduction_proof_path = tmp_path / "e3a-selection-reduction-proof.json"
    publish_formal_e3a_staged_selection_proof_artifact(
        e3a_reduction_proof,
        e3a_reduction_proof_path,
    )
    signed_e3a_selection_path = _proof_wrapped_signed_row(
        tmp_path,
        monkeypatch,
        label="e3a-selection",
        artifact_type="e3a-staged-selection",
        signed=signed_e3a_selection,
    )
    raw_root = tmp_path / "tts-raw"
    raw_root.mkdir()
    raw = _raw_bundle(
        raw_root,
        monkeypatch,
        authority=authority,
        protocol_lock_sha256=lock.sha256,
        upstream_e3a_receipt_sha256=signed_e3a_selection.sha256,
        now_ns=registry_now_ns,
    )
    tts_materialization = raw["materialization"]
    assert isinstance(tts_materialization, StageMaterializationReceipt)
    signed_e3a_coverage = SignedStageCoverageReceipt(
        e3a_coverage,
        *_registry_sign(controller_private, e3a_coverage, 41),
    )
    signed_tts_materialization = SignedStageMaterializationReceipt(
        tts_materialization,
        *_registry_sign(controller_private, tts_materialization, 42),
    )
    stage_rows = tuple(
        sorted(
            (
                (signed_e3a_coverage.sha256, "rank_aggregate"),
                (signed_tts_materialization.sha256, "dispatch"),
            )
        )
    )
    stage_lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_registry_control_lineage",
            "protocol_lock_sha256": lock.sha256,
            "registry_sha256": build_industrial_registry().sha256,
            "signed_artifacts": stage_rows,
            "prior_registry_verification_receipt_sha256": e3a_receipt.sha256,
            "signed_source_authorities": (
                (signed_e3a_selection.sha256, "e3a_staged_selection"),
            ),
        }
    )
    stage_authorization = _registry_deployment(
        root_private,
        root_binding=root_binding,
        bundle=registry_bundle,
        inventory_sha256=inventory.sha256,
        nonce=43,
    )
    stage_controls = tuple(
        _registry_control(
            controller_private,
            root_binding=root_binding,
            bundle=registry_bundle,
            authorization=stage_authorization,
            artifact_type=artifact_type,
            artifact_sha256=digest,
            protocol_sha256=lock.sha256,
            lineage_sha256=stage_lineage,
            nonce=44 + index,
        )
        for index, (digest, artifact_type) in enumerate(stage_rows)
    )
    stage_replay = tmp_path / "e3a-to-tts-replay"
    stage_replay.mkdir()
    tts_pending = extend_formal_registry_verification_receipt(
        e3a_receipt,
        appended_signed_materializations=(signed_tts_materialization,),
        appended_signed_coverage=(signed_e3a_coverage,),
        appended_e3a_staged_selection_artifacts=(e3a_artifact,),
        appended_signed_e3a_staged_selections=(signed_e3a_selection,),
        control_attestations=stage_controls,
        replay_store=ChallengeReplayStore(str(stage_replay.resolve())),
        now_ns=registry_now_ns,
    )
    signed_e3a_coverage_path = _proof_wrapped_signed_row(
        tmp_path,
        monkeypatch,
        label="e3a-coverage",
        artifact_type="stage-coverage",
        signed=signed_e3a_coverage,
    )
    signed_tts_materialization_path = _proof_wrapped_signed_row(
        tmp_path,
        monkeypatch,
        label="tts-materialization",
        artifact_type="stage-materialization",
        signed=signed_tts_materialization,
    )
    tts_pending_layer_path = tmp_path / "tts-pending-layer.json"
    publish_formal_registry_layer_artifact(
        bind_formal_registry_layer_artifact(
            tts_pending,
            prior_layer_path=e3a_layer,
            signed_materialization_paths=(signed_tts_materialization_path,),
            signed_coverage_paths=(signed_e3a_coverage_path,),
            formal_stage_prefix_paths=(),
            e3a_staged_selection_proof_paths=(e3a_reduction_proof_path,),
            signed_e3a_staged_selection_paths=(signed_e3a_selection_path,),
        ),
        tts_pending_layer_path,
    )
    with pytest.raises(ValueError, match="current materialization layer"):
        validate_formal_precoverage_registry_state(
            tts_pending,
            stage="E3a",
            phase="capacity",
            materialization=e3a_materialization,
        )

    proof_root = tmp_path / "tts-proof"
    proof_root.mkdir()
    coverage_proof_path, _tts_coverage, _sources = _publish_real_tts_coverage_proof(
        proof_root,
        monkeypatch,
        lock=lock,
        runtime=runtime,
        bundle=raw,
    )
    with pytest.raises(
        ValueError,
        match="exact-288 source-owned execution artifact",
    ):
        bind_formal_portable_stage_coverage_proof_artifact(
            coverage_proof_path,
            registry_layer_path=tts_pending_layer_path,
            now_ns=registry_now_ns,
        )


def test_raw_manifest_codec_requires_288_external_control_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _raw_bundle(tmp_path, monkeypatch)["manifest"]
    assert isinstance(manifest, RawTtsCalibrationEvidenceManifest)
    encoded = raw_tts_calibration_manifest_to_dict(manifest)
    assert raw_tts_calibration_manifest_from_dict(encoded).sha256 == manifest.sha256

    missing = json.loads(json.dumps(encoded))
    missing["pilots"][0].pop("terminal_control_attestations")
    with pytest.raises(ValueError, match="pilot fields differ"):
        raw_tts_calibration_manifest_from_dict(missing)

    legacy = json.loads(json.dumps(encoded))
    legacy["schema_version"] = 1
    with pytest.raises(ValueError, match="identity is invalid"):
        raw_tts_calibration_manifest_from_dict(legacy)


def test_tts_seal_and_reduction_reject_caller_authored_payloads() -> None:
    with pytest.raises(TypeError, match="raw 288-cell reducer"):
        TtsCalibrationSeal(
            schema_version=2,
            authority_sha256=_sha("authority"),
            protocol_lock_sha256=_sha("protocol"),
            materialization_receipt_sha256=_sha("materialization"),
            coverage_receipt_sha256=_sha("coverage"),
            reduction_receipt_sha256=_sha("reduction"),
            raw_manifest_sha256=_sha("manifest"),
            tuning_window_sha256=_sha("window"),
            selected_learning_rate=1e-7,
            selected_stride=1,
            selected_candidate_id=_sha("candidate"),
            selected_pilot_run_binding_sha256s=tuple(
                _sha(f"pilot-{block}") for block in PILOT_BLOCKS
            ),
            _construction_seal=object(),
        )
    with pytest.raises(TypeError, match="raw first-party evidence"):
        TtsCalibrationReductionReceipt(
            schema_version=2,
            protocol_lock_sha256=_sha("protocol"),
            authority_sha256=_sha("authority"),
            materialization_receipt_sha256=_sha("materialization"),
            coverage_receipt_sha256=_sha("coverage"),
            raw_manifest_sha256=_sha("manifest"),
            tuning_window_sha256=_sha("window"),
            registry_sha256=_sha("registry"),
            runtime_sha256=_sha("runtime"),
            split_sha256=_sha("split"),
            inventory_sha256=_sha("inventory"),
            hardware_envelope_sha256=_sha("hardware"),
            terminal_control_reservation_sha256=_sha("control-reservation"),
            terminal_control_policy_sha256=_sha("control-policy"),
            observations=(),
            selected_candidate_id=_sha("candidate"),
            selected_learning_rate=1e-7,
            selected_stride=1,
            selected_mean_slo_goodput_tps=1.0,
            selection_rule="safety_first_then_maximize_slo_goodput",
            _construction_seal=object(),
        )


def test_tts_reducer_rejects_noncomplete_coverage_and_terminal_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _raw_bundle(tmp_path, monkeypatch)
    coverage = bundle["coverage"]
    assert isinstance(coverage, StageCoverageReceipt)
    first = coverage.dispositions[0]
    bundle["coverage"] = replace(
        coverage,
        dispositions=(
            replace(
                first,
                status="BLOCKED",
                reason_code="unsafe",
                terminal_receipt_sha256=None,
            ),
            *coverage.dispositions[1:],
        ),
    )
    with pytest.raises(ValueError, match="exact complete calibration"):
        _reduce(bundle)

    bundle = _raw_bundle(tmp_path / "second", monkeypatch)
    coverage = bundle["coverage"]
    assert isinstance(coverage, StageCoverageReceipt)
    bundle["coverage"] = replace(
        coverage,
        dispositions=(
            replace(
                coverage.dispositions[0],
                terminal_receipt_sha256=_sha("foreign-terminal"),
            ),
            *coverage.dispositions[1:],
        ),
    )
    with pytest.raises(ValueError, match="coverage terminal differs"):
        _reduce(bundle)


def test_unsafe_high_goodput_candidate_is_eliminated_without_skipping_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _raw_bundle(tmp_path, monkeypatch)
    loaded = bundle["loaded"]
    assert isinstance(loaded, dict)
    authority = bundle["authority"]
    assert isinstance(authority, TtsCalibrationAuthority)
    first_id = next(
        cell_id
        for cell_id, row in loaded.items()
        if authority.candidate_id(
            learning_rate=float(row.cell.identity.learning_rate),
            stride=int(
                row.cell.identity.variant.removeprefix("tts_calibration:stride=")
            ),
        )
        == authority.candidate_ids[0]
    )
    first = loaded[first_id]
    performance = dict(first.performance_rows_by_rank[0][0])
    performance["exactness_violations"] = 1
    loaded[first_id] = replace(first, performance_rows_by_rank=((performance,),))
    reduction = _reduce(bundle)
    assert len(reduction.observations) == 288
    assert reduction.selected_candidate_id != authority.candidate_ids[0]
    eliminated = tuple(
        row for row in reduction.observations if row.registry_cell_id == first_id
    )
    assert len(eliminated) == 1
    assert eliminated[0].disposition == "ELIMINATED"
    assert eliminated[0].slo_goodput_tps is None
    assert "safety:exactness_violations" in eliminated[0].reason_codes


def test_no_safe_candidate_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _raw_bundle(tmp_path, monkeypatch)
    loaded = bundle["loaded"]
    assert isinstance(loaded, dict)
    for cell_id, row in tuple(loaded.items()):
        performance = dict(row.performance_rows_by_rank[0][0])
        performance["nonfinite_updates"] = 1
        loaded[cell_id] = replace(row, performance_rows_by_rank=((performance,),))
    with pytest.raises(ValueError, match="no safe SLO-feasible candidate"):
        _reduce(bundle)


def test_equal_feasible_scores_use_candidate_digest_tiebreak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _raw_bundle(tmp_path, monkeypatch)
    loaded = bundle["loaded"]
    assert isinstance(loaded, dict)
    for cell_id, row in tuple(loaded.items()):
        loaded[cell_id] = replace(
            row, request_rows=(_request_row(duration_ns=50_000_000),)
        )
    reduction = _reduce(bundle)
    authority = bundle["authority"]
    assert isinstance(authority, TtsCalibrationAuthority)
    assert reduction.selected_candidate_id == min(authority.candidate_ids)


def test_terminal_controls_prepare_all_288_before_one_atomic_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lightcone_spec import orchestration
    from lightcone_spec.experiments import tts_calibration_authority as calibration

    real_validator = calibration._validate_tts_calibration_native_terminal_batch
    bundle = _raw_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(
        calibration,
        "_validate_tts_calibration_native_terminal_batch",
        real_validator,
    )
    manifest = bundle["manifest"]
    registry = bundle["registry"]
    loaded = bundle["loaded"]
    inventory = bundle["inventory"]
    assert isinstance(manifest, RawTtsCalibrationEvidenceManifest)
    assert isinstance(loaded, dict)
    terminal_digest_to_cell = {
        cell.terminal_receipts[0].sha256: cell.cell_id for cell in manifest.cells
    }

    class _ControlCodec:
        @staticmethod
        def from_dict(_value):
            return SimpleNamespace()

    monkeypatch.setattr(calibration, "ControlArtifactAttestation", _ControlCodec)
    prepared_count = 0

    def fake_prepare(value, **_kwargs):
        nonlocal prepared_count
        prepared_count += 1
        cell_id = terminal_digest_to_cell[value["digest"]]
        row = loaded[cell_id]
        run = row.run_rows[0]
        binding = SimpleNamespace(
            canonical_raw_sha256=value["digest"],
            run_id=run["run_id"],
            run_nonce_sha256=run["run_nonce_sha256"],
            execution_plan_sha256=run["runtime_sha256"],
            rank_config_sha256=run["rank_config_sha256"],
            method="tts",
            scored_request_ids=tuple(
                request["request_id"] for request in row.request_rows
            ),
            warmup_request_ids=(),
        )
        return SimpleNamespace(
            binding=binding,
            evidence=SimpleNamespace(binding=binding),
            cell_id=cell_id,
        )

    batch_calls: list[int] = []

    def fake_batch(prepared, **_kwargs):
        batch_calls.append(len(prepared))
        return tuple(
            SimpleNamespace(
                binding=row.evidence.binding,
                terminal_sha256=_sha(f"terminal-semantic-{row.cell_id}"),
                external_control_binding_sha256=_sha(f"control-binding-{row.cell_id}"),
                external_control_envelope_sha256=_sha(
                    f"control-envelope-{row.cell_id}"
                ),
                external_control_reservation_sha256=_sha("batch-reservation"),
                external_control_trusted_policy_sha256=_sha("batch-policy"),
            )
            for row in prepared
        )

    monkeypatch.setattr(
        orchestration, "prepare_native_terminal_external_control", fake_prepare
    )
    monkeypatch.setattr(
        orchestration,
        "validate_native_terminal_artifacts_with_external_controls",
        fake_batch,
    )
    analysis = SimpleNamespace(
        _bound_json=lambda _path, digest, **_kwargs: {"digest": digest}
    )
    controlled = real_validator(
        analysis=analysis,
        registry=registry,
        manifest=manifest,
        loaded=loaded,
        inventory=inventory,
        replay_store=bundle["replay_store"],
        now_ns=bundle["now_ns"],
    )
    assert prepared_count == 288
    assert batch_calls == [288]
    assert len(controlled) == 288
    assert len({row.control_envelope_sha256 for row in controlled}) == 288
    assert {row.control_reservation_sha256 for row in controlled} == {
        _sha("batch-reservation")
    }

    prepared_count = 0
    batch_calls.clear()

    def fail_last_prepare(value, **kwargs):
        if prepared_count == 287:
            raise ValueError("invalid final control")
        return fake_prepare(value, **kwargs)

    monkeypatch.setattr(
        orchestration,
        "prepare_native_terminal_external_control",
        fail_last_prepare,
    )
    with pytest.raises(ValueError, match="invalid final control"):
        real_validator(
            analysis=analysis,
            registry=registry,
            manifest=manifest,
            loaded=loaded,
            inventory=inventory,
            replay_store=bundle["replay_store"],
            now_ns=bundle["now_ns"],
        )
    assert batch_calls == []
