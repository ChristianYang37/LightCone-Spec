"""Source-owned method authorities consumed by :class:`ProtocolLock`.

The scientific lock must not accept caller supplied digests for method
semantics.  These artifacts retain path-bound source inputs, replay them on
every load, and deterministically rebuild the public TTS and ChronoBelief
authorities.  Publication is canonical JSON and no-replace; a serialized
authority without its reopenable sources is deliberately unusable here.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec import PINNED_SGLANG_COMMIT, PINNED_SGLANG_TREE
from lightcone_spec.adaptation.plan_authority import (
    TrainablePlanAuthorityBinding,
    trainable_plan_authority_binding_from_dict,
)
from lightcone_spec.experiments.formal_protocol import (
    TTS_LEARNING_RATES,
    TTS_PRIMARY_SOURCE_ID,
    TTS_PRIMARY_SOURCE_VERSION,
    TTS_STRIDES,
    ChronoBeliefAuthority,
    TtsCalibrationAuthority,
    content_sha256,
)
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedLockedWorkload,
    TrustedSingleOperatorContentBundle,
    TrustedSingleOperatorContentBundleBinding,
    bind_trusted_locked_workload,
)
from lightcone_spec.experiments.protocol import DFLASH_LOSS_POSITION_DECAY
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.workload_authority import (
    FormalWorkloadAuthority,
    FormalWorkloadSample,
    bind_formal_workload_authority,
    formal_workload_authority_from_cli_artifact,
    revalidate_authorized_formal_workload_authority,
)
from lightcone_spec.runtime.content_authorization import (
    CONTENT_VERIFICATION_PROTOCOL_SHA256,
    TTS_CALIBRATION_TUNING_SELECTOR_NAMESPACE,
    ContentVerificationReceipt,
    TtsCalibrationTuningWindow,
    TtsCalibrationTuningWindowEntry,
    VerifiedReleaseWorkloadSources,
)
from lightcone_spec.runtime.preflight_runner import EvidenceFileBinding
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

TTS_CALIBRATION_SOURCE_AUTHORITY_ID = "tts-primary-source-reconstruction-v4"
TTS_CALIBRATION_CLAIM_SCOPE = (
    "project_preregistered_reconstruction_not_paper_reproduction"
)
TTS_PRIMARY_SOURCE_PDF_SHA256 = (
    "7688b05bab7696f4a47a5987f2fcad13d46f1d84cec9f90caf661fb397f3ee20"
)
TTS_PRIMARY_SOURCE_ARCHIVE_SHA256 = (
    "22c549c0297fc0a2a71af002c3721f71ddfd06d86bc46b2f41592bd6748afe59"
)
TTS_CALIBRATION_SOURCE_ARTIFACT_KIND = (
    "lightcone_tts_calibration_source_authority_artifact"
)
CHRONOBELIEF_SOURCE_AUTHORITY_ID = "lightcone-chronobelief-equations-5.5-5.8-v2"
CHRONOBELIEF_SOURCE_ARTIFACT_KIND = "lightcone_chronobelief_source_authority_artifact"
CHRONOBELIEF_CLAIM_SCOPE = "project_owned_preregistered_optimizer_not_external_paper"
CHRONOBELIEF_PREREG_PDF_SHA256 = (
    "2e79b6d6414d40b38d405f8165d80bb4efd354bf03b2f9ca53df23220435fc7c"
)
CHRONOBELIEF_PREREG_TEX_SHA256 = (
    "941b891e85f7551360133fe13131b88ab0412ecf7f617d3fb959126af43d7d08"
)
TTS_TUNING_WINDOW_SOURCE_KIND = "lightcone_tts_disjoint_tuning_window_source"
TTS_DRAFTER_NATIVE_LOSS_SOURCE_KIND = "lightcone_tts_drafter_native_loss_source"
TTS_TRAINABLE_PLAN_SELECTOR_ID = (
    "tts_calibration_qwen3_8b_dflash_full_all_adam_canonical_slot_v1"
)
TTS_DFLASH_LOSS_PATCH_SHA256 = (
    "091c8a164007691a171a449552a98ff6d68039cf868721bb24480e3ead4018e0"
)

TTS_DRAFTER_NATIVE_LOSS_SOURCE = {
    "schema_version": 3,
    "kind": TTS_DRAFTER_NATIVE_LOSS_SOURCE_KIND,
    "claim_scope": TTS_CALIBRATION_CLAIM_SCOPE,
    "result_classification": (
        "project_calibrated_runtime_baseline_not_paper_reproduction"
    ),
    "runtime_identity": {
        "sglang_upstream_commit": PINNED_SGLANG_COMMIT,
        "sglang_patched_tree": PINNED_SGLANG_TREE,
        "patch_file": (
            "patches/sglang/0001-feat-spec-add-schema-v3-native-online-adaptation.patch"
        ),
        "patch_sha256": TTS_DFLASH_LOSS_PATCH_SHA256,
        "implementation": (
            "python/sglang/srt/speculative/dflash_online_adaptation.py::"
            "DFlashDrafterAdapter._distillation_loss"
        ),
    },
    "loss": {
        "objective": "masked_position_weighted_target_to_draft_forward_kl",
        "target_distribution": "softmax(float32(target_logits))",
        "draft_distribution": "softmax(float32(draft_logits))",
        "temperature": 1.0,
        "accumulation_precision": "float32",
        "position_index_origin": 0,
        "position_weight_formula": "loss_position_decay ** position_index",
        "loss_position_decay": DFLASH_LOSS_POSITION_DECAY,
        "equivalent_one_based_formula": "exp(-(k-1)/7)",
        "valid_mask": "multiplicative_float32_mask",
        "normalization": (
            "sum(masked_position_weighted_kl)/"
            "clamp_min(sum(masked_position_weights),1.0)"
        ),
        "source_point_value_correction": (
            "inference_forward_value_with_differentiable_surrogate_jacobian"
        ),
        "proximal_penalty": "absent",
    },
    "fixed_runtime_semantics": [
        "latest_update_round_teacher_rows",
        "one_optimization_step_per_update",
        "request_local_reset",
        "side_stream_execution",
    ],
}


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _strict_object(
    label: str,
    value: object,
    fields: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


def _raw_binding_from_dict(value: object, *, label: str) -> EvidenceFileBinding:
    return EvidenceFileBinding.from_dict(value, label=label)


def _reopen_raw(
    value: EvidenceFileBinding,
    *,
    label: str,
    expected_raw_sha256: str | None = None,
) -> None:
    if type(value) is not EvidenceFileBinding:
        raise TypeError(f"{label} requires an exact raw source binding")
    value.reopen(label=label)
    if expected_raw_sha256 is not None and value.raw_sha256 != expected_raw_sha256:
        raise ValueError(f"{label} is not the registered primary-source bytes")


def _reopen_json(
    value: CanonicalJsonProofBinding,
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not CanonicalJsonProofBinding:
        raise TypeError(f"{label} requires an exact canonical JSON binding")
    if CanonicalJsonProofBinding.bind(value.absolute_path) != value:
        raise ValueError(f"{label} changed")
    reopened = value.reopen()
    if type(reopened) is not dict:
        raise TypeError(f"{label} must be one canonical JSON object")
    return reopened


def _reopen_trainable_plan(
    value: CanonicalJsonProofBinding,
    *,
    require_full_drafter: bool,
) -> TrainablePlanAuthorityBinding:
    raw = _reopen_json(value, label="trainable-plan authority")
    binding = trainable_plan_authority_binding_from_dict(raw)
    if (
        type(binding) is not TrainablePlanAuthorityBinding
        or binding.sha256 != value.semantic_sha256
    ):
        raise ValueError("trainable-plan source identity differs")
    result = binding.revalidate()
    if result.binding != binding or result.plan.sha256 != binding.trainable_plan_sha256:
        raise ValueError("trainable-plan source did not replay exactly")
    if require_full_drafter:
        _validate_tts_trainable_plan_selection(binding, result.plan)
    return binding


def _canonical_tts_trainable_plan_cell_id() -> str:
    matches = tuple(
        row
        for row in build_industrial_registry().cells_for("TTS-Cal")
        if row.identity.method == "tts"
        and row.identity.model == "Qwen/Qwen3-8B"
        and row.identity.backend == "DFLASH"
        and row.identity.scope == "full_drafter"
        and row.identity.parameterization == "full"
        and row.identity.optimizer == "adam"
        and row.identity.learning_rate == TTS_LEARNING_RATES[0]
        and row.identity.variant == f"tts_calibration:stride={TTS_STRIDES[0]}"
        and row.identity.block == 0
    )
    if len(matches) != 1:
        raise RuntimeError("canonical TTS trainable-plan slot is not unique")
    return matches[0].cell_id


def _validate_tts_trainable_plan_selection(
    binding: TrainablePlanAuthorityBinding,
    plan: object,
) -> None:
    if (
        getattr(plan, "backend", None) != "DFLASH"
        or getattr(plan, "mode", None) != "full"
        or getattr(plan, "scope", None) != "all"
        or binding.method != "tts"
        or binding.backend != "DFLASH"
        or binding.mode != "full"
        or binding.scope != "all"
        or binding.rank is not None
        or binding.lora_alpha is not None
        or binding.optimizer != "adam"
        or binding.target_model_id != "Qwen/Qwen3-8B"
        or binding.drafter_model_id != "z-lab/Qwen3-8B-DFlash-b16"
        or binding.cell_id != _canonical_tts_trainable_plan_cell_id()
    ):
        raise ValueError(
            "TTS source requires the code-owned canonical TTS-Cal trainable plan"
        )


def _validate_tts_plan_against_trusted_content(
    binding: TrainablePlanAuthorityBinding,
    bundle: TrustedSingleOperatorContentBundle,
) -> None:
    """Close model ID, revision, root, and DFlash runtime identity across sources."""

    if type(binding) is not TrainablePlanAuthorityBinding:
        raise TypeError("trusted TTS plan requires an exact plan binding")
    if type(bundle) is not TrustedSingleOperatorContentBundle:
        raise TypeError("trusted TTS plan requires an exact content bundle")
    target_matches = tuple(
        row
        for row in bundle.model_members
        if row.role == "target" and row.model_id == binding.target_model_id
    )
    drafter_matches = tuple(
        row
        for row in bundle.model_members
        if row.role == "drafter" and row.model_id == binding.drafter_model_id
    )
    if len(target_matches) != 1 or len(drafter_matches) != 1:
        raise ValueError("trusted TTS plan models differ from content bundle")
    target = target_matches[0]
    drafter = drafter_matches[0]
    prepared = {
        row.model_id: row
        for row in binding.prepared_model_content_authority.prepared_model_set.snapshots
    }
    prepared_target = prepared.get(binding.target_model_id)
    prepared_drafter = prepared.get(binding.drafter_model_id)
    if (
        binding.target_model_id != "Qwen/Qwen3-8B"
        or binding.drafter_model_id != "z-lab/Qwen3-8B-DFlash-b16"
        or target.revision != binding.target_revision
        or drafter.revision != binding.prepared_drafter_revision
        or "TTS-Cal" not in target.stages
        or "TTS-Cal" not in drafter.stages
        or prepared_target is None
        or prepared_drafter is None
        or prepared_target.revision != target.revision
        or prepared_drafter.revision != drafter.revision
        or prepared_target.root != target.local_snapshot_path
        or prepared_drafter.root != drafter.local_snapshot_path
        or not any(
            row.stage == "preflight"
            and row.target_model_id == binding.target_model_id
            and row.backend == "DFLASH"
            and row.draft_depth == 15
            for row in drafter.runtime_bindings
        )
    ):
        raise ValueError(
            "trusted TTS trainable plan differs from bundle model/runtime identity"
        )


def _ordered_tts_domain_sha256(workload: FormalWorkloadAuthority) -> str:
    return content_sha256(
        [
            {
                "source_problem_id": row.source_row_id,
                "source_sample_id": row.sample_id,
                "prompt_sha256": content_sha256(row.prompt),
            }
            for row in workload.samples
        ]
    )


def _build_tts_calibration_tuning_window(
    workload: FormalWorkloadAuthority,
    *,
    schema_version: Literal[4, 5],
    workload_source_descriptor_sha256: str,
    content_verification_receipt: ContentVerificationReceipt | None = None,
    trusted_content_bundle_sha256: str | None = None,
    trusted_locked_workload_sha256: str | None = None,
) -> TtsCalibrationTuningWindow:
    """Apply the sole registered problem-ID selector to one replayed authority."""

    if type(workload) is not FormalWorkloadAuthority:
        raise TypeError("TTS tuning selector requires an exact workload authority")
    if workload.workload_id != "livecodebench_v6_hard":
        raise ValueError("TTS tuning selector requires LiveCodeBench v6 hard")
    _require_sha256(
        "TTS tuning workload source descriptor",
        workload_source_descriptor_sha256,
    )
    if schema_version == 4:
        if (
            type(content_verification_receipt) is not ContentVerificationReceipt
            or trusted_content_bundle_sha256 is not None
            or trusted_locked_workload_sha256 is not None
        ):
            raise ValueError("release TTS window requires only its content receipt")
    elif schema_version == 5:
        if (
            content_verification_receipt is not None
            or trusted_content_bundle_sha256 is None
            or trusted_locked_workload_sha256 is None
        ):
            raise ValueError("trusted TTS window requires only its content bundle")
    else:  # pragma: no cover - Literal plus callers guard this
        raise ValueError("TTS tuning-window schema is unsupported")
    if len(workload.samples) <= 4:
        raise ValueError("TTS tuning selector requires tuning rows plus four pilots")
    ranked = tuple(
        sorted(
            workload.samples,
            key=lambda row: (
                content_sha256(
                    {
                        "selector_namespace": (
                            TTS_CALIBRATION_TUNING_SELECTOR_NAMESPACE
                        ),
                        "source_problem_id": row.source_row_id,
                    }
                ),
                row.source_row_id,
                row.sample_id,
            ),
        )
    )
    excluded_problem_ids = frozenset(row.source_row_id for row in ranked[:4])
    if len(excluded_problem_ids) != 4:
        raise ValueError("TTS tuning selector problem IDs are not unique")

    def entry(row: FormalWorkloadSample) -> TtsCalibrationTuningWindowEntry:
        return TtsCalibrationTuningWindowEntry(
            workload_id="livecodebench_v6_hard",
            source_problem_id=row.source_row_id,
            source_sample_id=row.sample_id,
            source_descriptor_sha256=workload_source_descriptor_sha256,
            prompt_sha256=content_sha256(row.prompt),
        )

    tuning_entries = tuple(
        sorted(
            (
                entry(row)
                for row in workload.samples
                if row.source_row_id not in excluded_problem_ids
            ),
            key=lambda row: row.entry_id,
        )
    )
    excluded_entries = tuple(
        sorted(
            (
                entry(row)
                for row in workload.samples
                if row.source_row_id in excluded_problem_ids
            ),
            key=lambda row: row.entry_id,
        )
    )
    lane: dict[str, object] = {}
    if schema_version == 4:
        assert content_verification_receipt is not None
        lane = {
            "content_verification_receipt_sha256": (
                content_verification_receipt.sha256
            ),
            "content_verification_verified_ns": (
                content_verification_receipt.verified_ns
            ),
            "content_verification_reservation_sha256": (
                content_verification_receipt.reservation.reservation_sha256
            ),
        }
    else:
        lane = {
            "trusted_content_bundle_sha256": trusted_content_bundle_sha256,
            "trusted_locked_workload_sha256": trusted_locked_workload_sha256,
        }
    return TtsCalibrationTuningWindow(
        schema_version=schema_version,
        kind=TTS_TUNING_WINDOW_SOURCE_KIND,
        selector_namespace=TTS_CALIBRATION_TUNING_SELECTOR_NAMESPACE,
        workload_authority_sha256=workload.sha256,
        ordered_domain_sha256=_ordered_tts_domain_sha256(workload),
        tuning_problem_ids=tuple(
            sorted(row.source_problem_id for row in tuning_entries)
        ),
        excluded_problem_ids=tuple(
            sorted(row.source_problem_id for row in excluded_entries)
        ),
        tuning_entries=tuning_entries,
        excluded_pilot_entries=excluded_entries,
        **lane,
    )


def build_code_owned_tts_calibration_tuning_window(
    workload: FormalWorkloadAuthority,
    *,
    content_verification_receipt: ContentVerificationReceipt,
) -> TtsCalibrationTuningWindow:
    """Build the release-receipt compatibility lane's schema-4 partition."""

    authorization = _verified_tuning_workload_sources(content_verification_receipt)
    workload = revalidate_authorized_formal_workload_authority(
        workload,
        authorization=authorization,
    )
    return _build_tts_calibration_tuning_window(
        workload,
        schema_version=4,
        workload_source_descriptor_sha256=authorization.source(
            "livecodebench_v6_hard"
        ).sha256,
        content_verification_receipt=content_verification_receipt,
    )


def _trusted_tts_sources_from_binding(
    content_source: TrustedSingleOperatorContentBundleBinding,
) -> tuple[
    TrustedSingleOperatorContentBundle,
    TrustedLockedWorkload,
    FormalWorkloadAuthority,
]:
    """Deep-replay the sole exact H=80 LCB member without a signer."""

    if type(content_source) is not TrustedSingleOperatorContentBundleBinding:
        raise TypeError("trusted TTS source requires an exact content binding")
    if content_source.runtime_binding_status != "BOUND":
        raise ValueError("trusted TTS source requires a BOUND content bundle")
    bundle = content_source.reopen()
    matches = tuple(
        row
        for row in bundle.locked_workloads
        if row.workload_id == "livecodebench_v6_hard"
    )
    if len(matches) != 1:
        raise ValueError("trusted TTS source requires one locked LCB hard workload")
    locked = matches[0]
    rebound_locked = bind_trusted_locked_workload(
        "livecodebench_v6_hard",
        locked.raw_source_path,
    )
    if rebound_locked != locked or rebound_locked.sha256 != locked.sha256:
        raise ValueError("trusted TTS locked workload differs from content bundle")
    workload = bind_formal_workload_authority(
        "livecodebench_v6_hard",
        locked.raw_source_path,
    )
    if (
        workload.sha256 != locked.authority_sha256
        or workload.raw_source_path != locked.raw_source_path
        or workload.raw_file_sha256 != locked.raw_file_sha256
        or workload.repository_revision != locked.repository_revision
        or workload.raw_row_count != locked.raw_row_count
        or workload.selected_row_count != locked.selected_row_count
        or workload.selected_rows_sha256 != locked.formal_samples_sha256
        or workload.source_lock_sha256 != locked.source_lock_sha256
        or workload.protocol_sha256 != locked.protocol_sha256
        or tuple(row.source_row_id for row in workload.samples)
        != locked.selected_source_row_ids
        or locked.raw_row_count != 175
        or locked.selected_row_count != 80
        or len(workload.samples) != 80
    ):
        raise ValueError("trusted TTS workload authority differs from exact LCB H=80")
    return bundle, locked, workload


def build_code_owned_trusted_tts_calibration_tuning_window(
    *,
    trusted_content_bundle_path: str | Path,
) -> TtsCalibrationTuningWindow:
    """Build schema 5 directly from one runtime-BOUND trusted content bundle."""

    source = TrustedSingleOperatorContentBundleBinding.bind(trusted_content_bundle_path)
    bundle, locked, workload = _trusted_tts_sources_from_binding(source)
    window = _build_tts_calibration_tuning_window(
        workload,
        schema_version=5,
        workload_source_descriptor_sha256=locked.sha256,
        trusted_content_bundle_sha256=bundle.semantic_sha256,
        trusted_locked_workload_sha256=locked.sha256,
    )
    if (
        TrustedSingleOperatorContentBundleBinding.bind(trusted_content_bundle_path)
        != source
    ):
        raise RuntimeError("trusted TTS content bundle changed during selection")
    return window


def _reopen_tuning_workload(
    value: CanonicalJsonProofBinding,
    *,
    authorization: VerifiedReleaseWorkloadSources,
) -> FormalWorkloadAuthority:
    workload = formal_workload_authority_from_cli_artifact(
        _reopen_json(value, label="TTS tuning workload authority")
    )
    return revalidate_authorized_formal_workload_authority(
        workload,
        authorization=authorization,
    )


def _verified_tuning_workload_sources(
    receipt: ContentVerificationReceipt,
) -> VerifiedReleaseWorkloadSources:
    if type(receipt) is not ContentVerificationReceipt:
        raise TypeError("TTS tuning selector requires an exact content receipt")
    if (
        receipt.schema_version != 2
        or receipt.protocol_sha256 != CONTENT_VERIFICATION_PROTOCOL_SHA256
    ):
        raise ValueError("TTS tuning selector requires a schema-2 content receipt")
    verified_rows = receipt.revalidate_formal_scope(current_ns=time.time_ns())
    matches = tuple(
        row for row in verified_rows if type(row) is VerifiedReleaseWorkloadSources
    )
    if len(verified_rows) != 4 or len(matches) != 1:
        raise ValueError("TTS tuning selector requires the complete master receipt")
    return matches[0]


def _reopen_tuning_content_verification_receipt(
    value: CanonicalJsonProofBinding,
) -> tuple[ContentVerificationReceipt, VerifiedReleaseWorkloadSources]:
    receipt = ContentVerificationReceipt.from_dict(
        _reopen_json(value, label="TTS tuning content-verification receipt")
    )
    if receipt.sha256 != value.semantic_sha256:
        raise ValueError("TTS tuning content-receipt identity differs")
    return receipt, _verified_tuning_workload_sources(receipt)


def _reopen_tuning_window(
    value: CanonicalJsonProofBinding,
    *,
    workload_source: CanonicalJsonProofBinding,
    content_verification_receipt_source: CanonicalJsonProofBinding,
) -> tuple[TtsCalibrationTuningWindow, ContentVerificationReceipt]:
    window = TtsCalibrationTuningWindow.from_dict(
        _reopen_json(value, label="TTS tuning-window source")
    )
    if window.kind != TTS_TUNING_WINDOW_SOURCE_KIND:
        raise ValueError("TTS tuning-window source kind differs")
    receipt, authorization = _reopen_tuning_content_verification_receipt(
        content_verification_receipt_source
    )
    workload = _reopen_tuning_workload(
        workload_source,
        authorization=authorization,
    )
    expected = build_code_owned_tts_calibration_tuning_window(
        workload,
        content_verification_receipt=receipt,
    )
    if window != expected or window.sha256 != expected.sha256:
        raise ValueError("TTS tuning window differs from code-owned selector")
    return window, receipt


def _reopen_trusted_tuning_window(
    value: CanonicalJsonProofBinding,
    *,
    content_source: TrustedSingleOperatorContentBundleBinding,
) -> tuple[
    TtsCalibrationTuningWindow,
    TrustedSingleOperatorContentBundle,
    TrustedLockedWorkload,
]:
    window = TtsCalibrationTuningWindow.from_dict(
        _reopen_json(value, label="trusted TTS tuning-window source")
    )
    if window.schema_version != 5 or window.kind != TTS_TUNING_WINDOW_SOURCE_KIND:
        raise ValueError("trusted TTS tuning-window source schema differs")
    bundle, locked, workload = _trusted_tts_sources_from_binding(content_source)
    expected = _build_tts_calibration_tuning_window(
        workload,
        schema_version=5,
        workload_source_descriptor_sha256=locked.sha256,
        trusted_content_bundle_sha256=bundle.semantic_sha256,
        trusted_locked_workload_sha256=locked.sha256,
    )
    if window != expected or window.sha256 != expected.sha256:
        raise ValueError("trusted TTS tuning window differs from code-owned selector")
    return window, bundle, locked


def _reopen_loss_source(value: CanonicalJsonProofBinding) -> dict[str, object]:
    if _reopen_json(value, label="TTS drafter-native loss source") != (
        TTS_DRAFTER_NATIVE_LOSS_SOURCE
    ):
        raise ValueError("TTS drafter-native loss source differs from protocol")
    return TTS_DRAFTER_NATIVE_LOSS_SOURCE


def publish_code_owned_tts_drafter_native_loss_source(
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish and deep-reopen the sole registered DFlash loss descriptor."""

    publish_canonical_json_no_replace(output_path, TTS_DRAFTER_NATIVE_LOSS_SOURCE)
    binding = CanonicalJsonProofBinding.bind(output_path)
    if _reopen_loss_source(binding) != TTS_DRAFTER_NATIVE_LOSS_SOURCE:
        raise RuntimeError("TTS drafter-native loss source changed during publication")
    if CanonicalJsonProofBinding.bind(output_path) != binding:
        raise RuntimeError("TTS drafter-native loss source changed while reopened")
    return binding


def tts_calibration_authority_to_dict(
    value: TtsCalibrationAuthority,
) -> dict[str, object]:
    if type(value) is not TtsCalibrationAuthority:
        raise TypeError("TTS authority codec requires an exact value")
    value.__post_init__()
    row = asdict(value)
    row["learning_rates"] = list(value.learning_rates)
    row["strides"] = list(value.strides)
    row["excluded_pilot_blocks"] = list(value.excluded_pilot_blocks)
    row["authority_sha256"] = value.sha256
    return row


def tts_calibration_authority_from_dict(value: object) -> TtsCalibrationAuthority:
    row = _strict_object(
        "TTS calibration authority",
        value,
        frozenset(TtsCalibrationAuthority.__dataclass_fields__) | {"authority_sha256"},
    )
    declared = _require_sha256("TTS declared authority", row.pop("authority_sha256"))
    for field in ("learning_rates", "strides", "excluded_pilot_blocks"):
        raw = row[field]
        if type(raw) is not list:
            raise TypeError(f"TTS authority {field} must be an array")
        row[field] = tuple(raw)
    authority = TtsCalibrationAuthority(**row)  # type: ignore[arg-type]
    if authority.sha256 != declared:
        raise ValueError("TTS authority digest differs from content")
    return authority


def chronobelief_authority_to_dict(value: ChronoBeliefAuthority) -> dict[str, object]:
    if type(value) is not ChronoBeliefAuthority:
        raise TypeError("ChronoBelief authority codec requires an exact value")
    value.__post_init__()
    row = asdict(value)
    row["equations"] = list(value.equations)
    row["authority_sha256"] = value.sha256
    return row


def chronobelief_authority_from_dict(value: object) -> ChronoBeliefAuthority:
    row = _strict_object(
        "ChronoBelief authority",
        value,
        frozenset(ChronoBeliefAuthority.__dataclass_fields__) | {"authority_sha256"},
    )
    declared = _require_sha256(
        "ChronoBelief declared authority", row.pop("authority_sha256")
    )
    equations = row["equations"]
    if type(equations) is not list:
        raise TypeError("ChronoBelief equations must be an array")
    row["equations"] = tuple(equations)
    authority = ChronoBeliefAuthority(**row)  # type: ignore[arg-type]
    if authority.sha256 != declared:
        raise ValueError("ChronoBelief authority digest differs from content")
    return authority


@dataclass(frozen=True)
class TtsCalibrationSourceAuthorityArtifact:
    schema_version: Literal[3, 4]
    kind: Literal["lightcone_tts_calibration_source_authority_artifact"]
    claim_scope: Literal["project_preregistered_reconstruction_not_paper_reproduction"]
    paper_pdf_source: EvidenceFileBinding
    paper_source: EvidenceFileBinding
    tuning_workload_authority_source: CanonicalJsonProofBinding | None
    content_verification_receipt_source: CanonicalJsonProofBinding | None
    content_verification_receipt_sha256: str | None
    content_verification_verified_ns: int | None
    content_verification_reservation_sha256: str | None
    trusted_content_bundle_source: TrustedSingleOperatorContentBundleBinding | None
    tuning_window_source: CanonicalJsonProofBinding
    trainable_plan_selector_id: Literal[
        "tts_calibration_qwen3_8b_dflash_full_all_adam_canonical_slot_v1"
    ]
    trainable_plan_authority_source: CanonicalJsonProofBinding
    drafter_native_loss_source: CanonicalJsonProofBinding
    authority: TtsCalibrationAuthority

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {3, 4}
            or self.kind != TTS_CALIBRATION_SOURCE_ARTIFACT_KIND
            or self.claim_scope != TTS_CALIBRATION_CLAIM_SCOPE
            or self.trainable_plan_selector_id != TTS_TRAINABLE_PLAN_SELECTOR_ID
        ):
            raise ValueError("TTS source authority artifact schema is unsupported")
        _reopen_raw(
            self.paper_pdf_source,
            label="TTS paper PDF",
            expected_raw_sha256=TTS_PRIMARY_SOURCE_PDF_SHA256,
        )
        _reopen_raw(
            self.paper_source,
            label="TTS paper source",
            expected_raw_sha256=TTS_PRIMARY_SOURCE_ARCHIVE_SHA256,
        )
        trusted_bundle: TrustedSingleOperatorContentBundle | None = None
        if self.schema_version == 3:
            if (
                type(self.tuning_workload_authority_source)
                is not CanonicalJsonProofBinding
                or type(self.content_verification_receipt_source)
                is not CanonicalJsonProofBinding
                or self.trusted_content_bundle_source is not None
            ):
                raise ValueError(
                    "release TTS source authority requires only receipt sources"
                )
            window, receipt = _reopen_tuning_window(
                self.tuning_window_source,
                workload_source=self.tuning_workload_authority_source,
                content_verification_receipt_source=(
                    self.content_verification_receipt_source
                ),
            )
            if (
                self.content_verification_receipt_sha256 != receipt.sha256
                or self.content_verification_verified_ns != receipt.verified_ns
                or self.content_verification_reservation_sha256
                != receipt.reservation.reservation_sha256
                or self.content_verification_receipt_sha256
                != window.content_verification_receipt_sha256
                or self.content_verification_verified_ns
                != window.content_verification_verified_ns
                or self.content_verification_reservation_sha256
                != window.content_verification_reservation_sha256
            ):
                raise ValueError(
                    "TTS source authority content-receipt identity differs"
                )
        else:
            if (
                self.tuning_workload_authority_source is not None
                or self.content_verification_receipt_source is not None
                or self.content_verification_receipt_sha256 is not None
                or self.content_verification_verified_ns is not None
                or self.content_verification_reservation_sha256 is not None
                or type(self.trusted_content_bundle_source)
                is not TrustedSingleOperatorContentBundleBinding
            ):
                raise ValueError(
                    "trusted TTS source authority requires only its content bundle"
                )
            _window, trusted_bundle, _locked = _reopen_trusted_tuning_window(
                self.tuning_window_source,
                content_source=self.trusted_content_bundle_source,
            )
        _reopen_loss_source(self.drafter_native_loss_source)
        plan = _reopen_trainable_plan(
            self.trainable_plan_authority_source,
            require_full_drafter=True,
        )
        if trusted_bundle is not None:
            _validate_tts_plan_against_trusted_content(plan, trusted_bundle)
        expected = TtsCalibrationAuthority(
            schema_version=2,
            authority_id=TTS_CALIBRATION_SOURCE_AUTHORITY_ID,
            primary_source_id=TTS_PRIMARY_SOURCE_ID,
            primary_source_version=TTS_PRIMARY_SOURCE_VERSION,
            paper_pdf_sha256=TTS_PRIMARY_SOURCE_PDF_SHA256,
            paper_source_sha256=TTS_PRIMARY_SOURCE_ARCHIVE_SHA256,
            tuning_window_sha256=self.tuning_window_source.semantic_sha256,
            trainable_plan_sha256=plan.trainable_plan_sha256,
            drafter_native_loss_recipe_sha256=(
                self.drafter_native_loss_source.semantic_sha256
            ),
        )
        if (
            type(self.authority) is not TtsCalibrationAuthority
            or self.authority != expected
        ):
            raise ValueError("TTS authority differs from source-owned replay")
        if (
            self.trusted_content_bundle_source is not None
            and TrustedSingleOperatorContentBundleBinding.bind(
                self.trusted_content_bundle_source.absolute_path
            )
            != self.trusted_content_bundle_source
        ):
            raise RuntimeError("trusted TTS content bundle changed during replay")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        row: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "claim_scope": self.claim_scope,
            "paper_pdf_source": self.paper_pdf_source.to_dict(),
            "paper_source": self.paper_source.to_dict(),
            "tuning_window_source": self.tuning_window_source.to_dict(),
            "trainable_plan_selector_id": self.trainable_plan_selector_id,
            "trainable_plan_authority_source": (
                self.trainable_plan_authority_source.to_dict()
            ),
            "drafter_native_loss_source": self.drafter_native_loss_source.to_dict(),
            "authority": tts_calibration_authority_to_dict(self.authority),
        }
        if self.schema_version == 3:
            if (
                self.tuning_workload_authority_source is None
                or self.content_verification_receipt_source is None
            ):
                raise ValueError("release TTS source authority fields are incomplete")
            row.update(
                {
                    "tuning_workload_authority_source": (
                        self.tuning_workload_authority_source.to_dict()
                    ),
                    "content_verification_receipt_source": (
                        self.content_verification_receipt_source.to_dict()
                    ),
                    "content_verification_receipt_sha256": (
                        self.content_verification_receipt_sha256
                    ),
                    "content_verification_verified_ns": (
                        self.content_verification_verified_ns
                    ),
                    "content_verification_reservation_sha256": (
                        self.content_verification_reservation_sha256
                    ),
                }
            )
        else:
            if self.trusted_content_bundle_source is None:
                raise ValueError("trusted TTS content bundle source is missing")
            row["trusted_content_bundle_source"] = (
                self.trusted_content_bundle_source.to_dict()
            )
        if include_sha256:
            row["artifact_sha256"] = self.sha256
        return row

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict:
            raise ValueError("TTS source authority artifact fields differ from schema")
        schema_version = value.get("schema_version")
        common_fields = {
            "schema_version",
            "kind",
            "claim_scope",
            "paper_pdf_source",
            "paper_source",
            "tuning_window_source",
            "trainable_plan_selector_id",
            "trainable_plan_authority_source",
            "drafter_native_loss_source",
            "authority",
            "artifact_sha256",
        }
        if schema_version == 3:
            fields = common_fields | {
                "tuning_workload_authority_source",
                "content_verification_receipt_source",
                "content_verification_receipt_sha256",
                "content_verification_verified_ns",
                "content_verification_reservation_sha256",
            }
        elif schema_version == 4:
            fields = common_fields | {"trusted_content_bundle_source"}
        else:
            raise ValueError("TTS source authority artifact schema is unsupported")
        row = _strict_object(
            "TTS source authority artifact",
            value,
            frozenset(fields),
        )
        declared = _require_sha256("TTS source artifact", row.pop("artifact_sha256"))
        workload_source = (
            CanonicalJsonProofBinding.from_dict(row["tuning_workload_authority_source"])
            if schema_version == 3
            else None
        )
        receipt_source = (
            CanonicalJsonProofBinding.from_dict(
                row["content_verification_receipt_source"]
            )
            if schema_version == 3
            else None
        )
        content_source = (
            TrustedSingleOperatorContentBundleBinding.from_dict(
                row["trusted_content_bundle_source"]
            )
            if schema_version == 4
            else None
        )
        artifact = cls(
            schema_version=row["schema_version"],  # type: ignore[arg-type]
            kind=row["kind"],  # type: ignore[arg-type]
            claim_scope=row["claim_scope"],  # type: ignore[arg-type]
            paper_pdf_source=_raw_binding_from_dict(
                row["paper_pdf_source"], label="TTS paper PDF"
            ),
            paper_source=_raw_binding_from_dict(
                row["paper_source"], label="TTS paper source"
            ),
            tuning_workload_authority_source=workload_source,
            content_verification_receipt_source=receipt_source,
            content_verification_receipt_sha256=(
                row["content_verification_receipt_sha256"]
                if schema_version == 3
                else None
            ),  # type: ignore[arg-type]
            content_verification_verified_ns=(
                row["content_verification_verified_ns"] if schema_version == 3 else None
            ),  # type: ignore[arg-type]
            content_verification_reservation_sha256=(
                row["content_verification_reservation_sha256"]
                if schema_version == 3
                else None
            ),  # type: ignore[arg-type]
            trusted_content_bundle_source=content_source,
            tuning_window_source=CanonicalJsonProofBinding.from_dict(
                row["tuning_window_source"]
            ),
            trainable_plan_selector_id=row["trainable_plan_selector_id"],  # type: ignore[arg-type]
            trainable_plan_authority_source=CanonicalJsonProofBinding.from_dict(
                row["trainable_plan_authority_source"]
            ),
            drafter_native_loss_source=CanonicalJsonProofBinding.from_dict(
                row["drafter_native_loss_source"]
            ),
            authority=tts_calibration_authority_from_dict(row["authority"]),
        )
        if artifact.sha256 != declared:
            raise ValueError("TTS source artifact digest differs from content")
        return artifact


@dataclass(frozen=True)
class ChronoBeliefSourceAuthorityArtifact:
    schema_version: Literal[2]
    kind: Literal["lightcone_chronobelief_source_authority_artifact"]
    claim_scope: Literal["project_owned_preregistered_optimizer_not_external_paper"]
    paper_pdf_source: EvidenceFileBinding
    tex_source: EvidenceFileBinding
    authority: ChronoBeliefAuthority

    def __post_init__(self) -> None:
        if (
            self.schema_version != 2
            or self.kind != CHRONOBELIEF_SOURCE_ARTIFACT_KIND
            or self.claim_scope != CHRONOBELIEF_CLAIM_SCOPE
        ):
            raise ValueError(
                "ChronoBelief source authority artifact schema is unsupported"
            )
        _reopen_raw(
            self.paper_pdf_source,
            label="ChronoBelief preregistration PDF",
            expected_raw_sha256=CHRONOBELIEF_PREREG_PDF_SHA256,
        )
        _reopen_raw(
            self.tex_source,
            label="ChronoBelief preregistration TeX",
            expected_raw_sha256=CHRONOBELIEF_PREREG_TEX_SHA256,
        )
        expected = ChronoBeliefAuthority(
            schema_version=1,
            authority_id=CHRONOBELIEF_SOURCE_AUTHORITY_ID,
            paper_pdf_sha256=CHRONOBELIEF_PREREG_PDF_SHA256,
            tex_source_sha256=CHRONOBELIEF_PREREG_TEX_SHA256,
        )
        if (
            type(self.authority) is not ChronoBeliefAuthority
            or self.authority != expected
        ):
            raise ValueError("ChronoBelief authority differs from source-owned replay")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        row: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "claim_scope": self.claim_scope,
            "paper_pdf_source": self.paper_pdf_source.to_dict(),
            "tex_source": self.tex_source.to_dict(),
            "authority": chronobelief_authority_to_dict(self.authority),
        }
        if include_sha256:
            row["artifact_sha256"] = self.sha256
        return row

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_object(
            "ChronoBelief source authority artifact",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "claim_scope",
                    "paper_pdf_source",
                    "tex_source",
                    "authority",
                    "artifact_sha256",
                }
            ),
        )
        declared = _require_sha256(
            "ChronoBelief source artifact", row.pop("artifact_sha256")
        )
        artifact = cls(
            schema_version=row["schema_version"],  # type: ignore[arg-type]
            kind=row["kind"],  # type: ignore[arg-type]
            claim_scope=row["claim_scope"],  # type: ignore[arg-type]
            paper_pdf_source=_raw_binding_from_dict(
                row["paper_pdf_source"], label="ChronoBelief preregistration PDF"
            ),
            tex_source=_raw_binding_from_dict(
                row["tex_source"], label="ChronoBelief preregistration TeX"
            ),
            authority=chronobelief_authority_from_dict(row["authority"]),
        )
        if artifact.sha256 != declared:
            raise ValueError("ChronoBelief source artifact digest differs from content")
        return artifact


def build_source_tts_calibration_authority_artifact(
    *,
    paper_pdf_path: str | Path,
    paper_source_path: str | Path,
    tuning_window_path: str | Path,
    trainable_plan_authority_path: str | Path,
    drafter_native_loss_path: str | Path,
    trusted_content_bundle_path: str | Path | None = None,
    tuning_workload_authority_path: str | Path | None = None,
    content_verification_receipt_path: str | Path | None = None,
) -> TtsCalibrationSourceAuthorityArtifact:
    if trusted_content_bundle_path is not None:
        if (
            tuning_workload_authority_path is not None
            or content_verification_receipt_path is not None
        ):
            raise ValueError(
                "TTS source authority accepts exactly one trusted or release lane"
            )
    elif (
        tuning_workload_authority_path is None
        or content_verification_receipt_path is None
    ):
        raise ValueError(
            "TTS source authority accepts exactly one trusted or release lane"
        )
    pdf = EvidenceFileBinding.bind(Path(paper_pdf_path), label="TTS paper PDF")
    paper = EvidenceFileBinding.bind(Path(paper_source_path), label="TTS paper source")
    _reopen_raw(
        pdf,
        label="TTS paper PDF",
        expected_raw_sha256=TTS_PRIMARY_SOURCE_PDF_SHA256,
    )
    _reopen_raw(
        paper,
        label="TTS paper source",
        expected_raw_sha256=TTS_PRIMARY_SOURCE_ARCHIVE_SHA256,
    )
    tuning = CanonicalJsonProofBinding.bind(tuning_window_path)
    loss = CanonicalJsonProofBinding.bind(drafter_native_loss_path)
    _reopen_loss_source(loss)
    plan_source = CanonicalJsonProofBinding.bind(trainable_plan_authority_path)
    plan = _reopen_trainable_plan(plan_source, require_full_drafter=True)
    workload: CanonicalJsonProofBinding | None = None
    content_verification_receipt: CanonicalJsonProofBinding | None = None
    content_source: TrustedSingleOperatorContentBundleBinding | None = None
    receipt: ContentVerificationReceipt | None = None
    if trusted_content_bundle_path is not None:
        content_source = TrustedSingleOperatorContentBundleBinding.bind(
            trusted_content_bundle_path
        )
        _window, bundle, _locked = _reopen_trusted_tuning_window(
            tuning,
            content_source=content_source,
        )
        _validate_tts_plan_against_trusted_content(plan, bundle)
        schema_version: Literal[3, 4] = 4
    else:
        assert tuning_workload_authority_path is not None
        assert content_verification_receipt_path is not None
        workload = CanonicalJsonProofBinding.bind(tuning_workload_authority_path)
        content_verification_receipt = CanonicalJsonProofBinding.bind(
            content_verification_receipt_path
        )
        _window, receipt = _reopen_tuning_window(
            tuning,
            workload_source=workload,
            content_verification_receipt_source=content_verification_receipt,
        )
        schema_version = 3
    authority = TtsCalibrationAuthority(
        schema_version=2,
        authority_id=TTS_CALIBRATION_SOURCE_AUTHORITY_ID,
        primary_source_id=TTS_PRIMARY_SOURCE_ID,
        primary_source_version=TTS_PRIMARY_SOURCE_VERSION,
        paper_pdf_sha256=TTS_PRIMARY_SOURCE_PDF_SHA256,
        paper_source_sha256=TTS_PRIMARY_SOURCE_ARCHIVE_SHA256,
        tuning_window_sha256=tuning.semantic_sha256,
        trainable_plan_sha256=plan.trainable_plan_sha256,
        drafter_native_loss_recipe_sha256=loss.semantic_sha256,
    )
    return TtsCalibrationSourceAuthorityArtifact(
        schema_version=schema_version,
        kind=TTS_CALIBRATION_SOURCE_ARTIFACT_KIND,
        claim_scope=TTS_CALIBRATION_CLAIM_SCOPE,
        paper_pdf_source=pdf,
        paper_source=paper,
        tuning_workload_authority_source=workload,
        content_verification_receipt_source=content_verification_receipt,
        content_verification_receipt_sha256=(
            None if receipt is None else receipt.sha256
        ),
        content_verification_verified_ns=(
            None if receipt is None else receipt.verified_ns
        ),
        content_verification_reservation_sha256=(
            None if receipt is None else receipt.reservation.reservation_sha256
        ),
        trusted_content_bundle_source=content_source,
        tuning_window_source=tuning,
        trainable_plan_selector_id=TTS_TRAINABLE_PLAN_SELECTOR_ID,
        trainable_plan_authority_source=plan_source,
        drafter_native_loss_source=loss,
        authority=authority,
    )


def build_source_chronobelief_authority_artifact(
    *,
    paper_pdf_path: str | Path,
    tex_source_path: str | Path,
) -> ChronoBeliefSourceAuthorityArtifact:
    pdf = EvidenceFileBinding.bind(
        Path(paper_pdf_path), label="ChronoBelief preregistration PDF"
    )
    tex = EvidenceFileBinding.bind(
        Path(tex_source_path), label="ChronoBelief preregistration TeX"
    )
    _reopen_raw(
        pdf,
        label="ChronoBelief preregistration PDF",
        expected_raw_sha256=CHRONOBELIEF_PREREG_PDF_SHA256,
    )
    _reopen_raw(
        tex,
        label="ChronoBelief preregistration TeX",
        expected_raw_sha256=CHRONOBELIEF_PREREG_TEX_SHA256,
    )
    authority = ChronoBeliefAuthority(
        schema_version=1,
        authority_id=CHRONOBELIEF_SOURCE_AUTHORITY_ID,
        paper_pdf_sha256=CHRONOBELIEF_PREREG_PDF_SHA256,
        tex_source_sha256=CHRONOBELIEF_PREREG_TEX_SHA256,
    )
    return ChronoBeliefSourceAuthorityArtifact(
        schema_version=2,
        kind=CHRONOBELIEF_SOURCE_ARTIFACT_KIND,
        claim_scope=CHRONOBELIEF_CLAIM_SCOPE,
        paper_pdf_source=pdf,
        tex_source=tex,
        authority=authority,
    )


def publish_code_owned_tts_calibration_tuning_window(
    *,
    tuning_workload_authority_path: str | Path,
    content_verification_receipt_path: str | Path,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish the sole code-owned all-hard tuning/excluded partition."""

    workload_source = CanonicalJsonProofBinding.bind(tuning_workload_authority_path)
    content_verification_receipt_source = CanonicalJsonProofBinding.bind(
        content_verification_receipt_path
    )
    receipt, authorization = _reopen_tuning_content_verification_receipt(
        content_verification_receipt_source
    )
    window = build_code_owned_tts_calibration_tuning_window(
        _reopen_tuning_workload(
            workload_source,
            authorization=authorization,
        ),
        content_verification_receipt=receipt,
    )
    publish_canonical_json_no_replace(output_path, window.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    reopened = TtsCalibrationTuningWindow.from_dict(binding.reopen())
    if reopened != window or reopened.sha256 != window.sha256:
        raise RuntimeError("TTS tuning window changed during publication")
    _reopen_tuning_window(
        binding,
        workload_source=workload_source,
        content_verification_receipt_source=content_verification_receipt_source,
    )
    return binding


def publish_code_owned_trusted_tts_calibration_tuning_window(
    *,
    trusted_content_bundle_path: str | Path,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish schema 5 from one path-bound, runtime-BOUND content bundle."""

    content_source = TrustedSingleOperatorContentBundleBinding.bind(
        trusted_content_bundle_path
    )
    window = build_code_owned_trusted_tts_calibration_tuning_window(
        trusted_content_bundle_path=trusted_content_bundle_path,
    )
    publish_canonical_json_no_replace(output_path, window.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    reopened = TtsCalibrationTuningWindow.from_dict(binding.reopen())
    if reopened != window or reopened.sha256 != window.sha256:
        raise RuntimeError("trusted TTS tuning window changed during publication")
    _reopen_trusted_tuning_window(binding, content_source=content_source)
    if (
        TrustedSingleOperatorContentBundleBinding.bind(trusted_content_bundle_path)
        != content_source
    ):
        raise RuntimeError("trusted TTS content bundle changed during publication")
    return binding


def _publish_and_reopen(
    artifact: TtsCalibrationSourceAuthorityArtifact
    | ChronoBeliefSourceAuthorityArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    if isinstance(artifact, TtsCalibrationSourceAuthorityArtifact):
        reopened = TtsCalibrationSourceAuthorityArtifact.from_dict(binding.reopen())
    else:
        reopened = ChronoBeliefSourceAuthorityArtifact.from_dict(binding.reopen())
    if reopened != artifact:
        raise RuntimeError("method authority changed during publication")
    return binding


def publish_tts_calibration_authority_artifact(
    artifact: TtsCalibrationSourceAuthorityArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not TtsCalibrationSourceAuthorityArtifact:
        raise TypeError("TTS publisher requires an exact source artifact")
    artifact.__post_init__()
    return _publish_and_reopen(artifact, output_path)


def publish_chronobelief_authority_artifact(
    artifact: ChronoBeliefSourceAuthorityArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not ChronoBeliefSourceAuthorityArtifact:
        raise TypeError("ChronoBelief publisher requires an exact source artifact")
    artifact.__post_init__()
    return _publish_and_reopen(artifact, output_path)


def load_tts_calibration_authority_artifact(
    path: str | Path,
) -> TtsCalibrationSourceAuthorityArtifact:
    before = CanonicalJsonProofBinding.bind(path)
    artifact = TtsCalibrationSourceAuthorityArtifact.from_dict(before.reopen())
    if CanonicalJsonProofBinding.bind(before.absolute_path) != before:
        raise RuntimeError("TTS source authority changed while loaded")
    return artifact


def load_chronobelief_authority_artifact(
    path: str | Path,
) -> ChronoBeliefSourceAuthorityArtifact:
    before = CanonicalJsonProofBinding.bind(path)
    artifact = ChronoBeliefSourceAuthorityArtifact.from_dict(before.reopen())
    if CanonicalJsonProofBinding.bind(before.absolute_path) != before:
        raise RuntimeError("ChronoBelief source authority changed while loaded")
    return artifact


__all__ = (
    "CHRONOBELIEF_CLAIM_SCOPE",
    "CHRONOBELIEF_PREREG_PDF_SHA256",
    "CHRONOBELIEF_PREREG_TEX_SHA256",
    "CHRONOBELIEF_SOURCE_ARTIFACT_KIND",
    "CHRONOBELIEF_SOURCE_AUTHORITY_ID",
    "TTS_CALIBRATION_CLAIM_SCOPE",
    "TTS_CALIBRATION_SOURCE_ARTIFACT_KIND",
    "TTS_CALIBRATION_SOURCE_AUTHORITY_ID",
    "TTS_DFLASH_LOSS_PATCH_SHA256",
    "TTS_DRAFTER_NATIVE_LOSS_SOURCE",
    "TTS_DRAFTER_NATIVE_LOSS_SOURCE_KIND",
    "TTS_PRIMARY_SOURCE_ARCHIVE_SHA256",
    "TTS_PRIMARY_SOURCE_PDF_SHA256",
    "TTS_TRAINABLE_PLAN_SELECTOR_ID",
    "TTS_TUNING_WINDOW_SOURCE_KIND",
    "ChronoBeliefSourceAuthorityArtifact",
    "TtsCalibrationSourceAuthorityArtifact",
    "build_code_owned_trusted_tts_calibration_tuning_window",
    "build_code_owned_tts_calibration_tuning_window",
    "build_source_chronobelief_authority_artifact",
    "build_source_tts_calibration_authority_artifact",
    "chronobelief_authority_from_dict",
    "chronobelief_authority_to_dict",
    "load_chronobelief_authority_artifact",
    "load_tts_calibration_authority_artifact",
    "publish_chronobelief_authority_artifact",
    "publish_code_owned_trusted_tts_calibration_tuning_window",
    "publish_code_owned_tts_calibration_tuning_window",
    "publish_code_owned_tts_drafter_native_loss_source",
    "publish_tts_calibration_authority_artifact",
    "tts_calibration_authority_from_dict",
    "tts_calibration_authority_to_dict",
)
