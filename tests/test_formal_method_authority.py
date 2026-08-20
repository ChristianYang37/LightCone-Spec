from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_trainable_plan_authority import _inputs

import lightcone_spec.experiments.formal_method_authority as method_module
import lightcone_spec.experiments.formal_stage_execution as e1_module
from lightcone_spec.adaptation.plan_authority import (
    TrainablePlanAuthorityBinding,
    trainable_plan_authority_binding_to_dict,
)
from lightcone_spec.cli import main as cli_module
from lightcone_spec.experiments.formal_method_authority import (
    CHRONOBELIEF_CLAIM_SCOPE,
    CHRONOBELIEF_PREREG_PDF_SHA256,
    CHRONOBELIEF_PREREG_TEX_SHA256,
    TTS_CALIBRATION_CLAIM_SCOPE,
    TTS_DFLASH_LOSS_PATCH_SHA256,
    TTS_DRAFTER_NATIVE_LOSS_SOURCE,
    TTS_PRIMARY_SOURCE_ARCHIVE_SHA256,
    TTS_PRIMARY_SOURCE_PDF_SHA256,
    TtsCalibrationSourceAuthorityArtifact,
    build_source_chronobelief_authority_artifact,
    build_source_tts_calibration_authority_artifact,
    publish_code_owned_tts_drafter_native_loss_source,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedSingleOperatorContentBundle,
    TrustedSingleOperatorContentBundleBinding,
)
from lightcone_spec.experiments.formal_stage_execution import (
    E1_RECIPE_ANCHOR_PLAN_SELECTOR_ID,
    E1RecipeAnchorAuthorityArtifact,
    build_source_e1_recipe_anchor_authority_artifact,
    load_e1_recipe_anchor_authority_artifact,
    publish_e1_recipe_anchor_authority_artifact,
)
from lightcone_spec.experiments.workload_authority import (
    FORMAL_WORKLOAD_PROTOCOLS,
    FormalWorkloadAuthority,
    FormalWorkloadSample,
    formal_workload_samples_sha256,
)
from lightcone_spec.runtime.content_authorization import TtsCalibrationTuningWindow
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _synthetic_tts_workload(tmp_path: Path) -> FormalWorkloadAuthority:
    raw = (tmp_path / "livecodebench-v6-hard.json").resolve()
    raw.write_text("{}\n", encoding="utf-8")
    samples = tuple(
        FormalWorkloadSample(
            source_row_id=f"question-{index:03d}",
            sample_id=f"livecodebench-v6-hard-{index:03d}",
            prompt=f"Solve deterministic problem {index:03d}.",
            seed=index,
        )
        for index in range(80)
    )
    return FormalWorkloadAuthority(
        schema_version=1,
        kind="formal_workload_authority",
        workload_id="livecodebench_v6_hard",
        raw_source_path=str(raw),
        raw_file_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        repository_revision="a" * 40,
        raw_row_count=175,
        selected_row_count=80,
        selected_rows_sha256=formal_workload_samples_sha256(samples),
        source_lock_sha256=_sha("synthetic-lcb-source-lock"),
        protocol_sha256=FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"].sha256,
        samples=samples,
    )


def _trusted_bundle_shell(
    binding: TrainablePlanAuthorityBinding,
    *,
    semantic_sha256: str,
    target_revision: str | None = None,
    drafter_revision: str | None = None,
    target_root: str | None = None,
    drafter_root: str | None = None,
) -> TrustedSingleOperatorContentBundle:
    snapshots = {
        row.model_id: row
        for row in binding.prepared_model_content_authority.prepared_model_set.snapshots
    }
    target_snapshot = snapshots[binding.target_model_id]
    drafter_snapshot = snapshots[binding.drafter_model_id]
    target = SimpleNamespace(
        role="target",
        model_id=binding.target_model_id,
        revision=(
            binding.target_revision if target_revision is None else target_revision
        ),
        local_snapshot_path=(
            target_snapshot.root if target_root is None else target_root
        ),
        stages=("TTS-Cal", "E1"),
        runtime_bindings=(),
    )
    drafter = SimpleNamespace(
        role="drafter",
        model_id=binding.drafter_model_id,
        revision=(
            binding.prepared_drafter_revision
            if drafter_revision is None
            else drafter_revision
        ),
        local_snapshot_path=(
            drafter_snapshot.root if drafter_root is None else drafter_root
        ),
        stages=("TTS-Cal", "E1"),
        runtime_bindings=(
            SimpleNamespace(
                stage="preflight",
                target_model_id=binding.target_model_id,
                backend="DFLASH",
                draft_depth=15,
            ),
        ),
    )
    bundle = object.__new__(TrustedSingleOperatorContentBundle)
    object.__setattr__(bundle, "model_members", (drafter, target))
    object.__setattr__(bundle, "semantic_sha256", semantic_sha256)
    return bundle


def _plan_source(
    tmp_path: Path, *, scope: str
) -> tuple[Path, TrainablePlanAuthorityBinding]:
    values = _inputs(tmp_path / "plan-inputs", mode="full", scope=scope)
    binding = values["binding"]
    assert isinstance(binding, TrainablePlanAuthorityBinding)
    path = (tmp_path / "trainable-plan-authority.json").resolve()
    publish_canonical_json_no_replace(
        path,
        trainable_plan_authority_binding_to_dict(binding),
    )
    return path, binding


def _tts_plan_source(
    tmp_path: Path,
) -> tuple[Path, TrainablePlanAuthorityBinding]:
    values = _inputs(tmp_path / "plan-inputs", tts_calibration=True)
    binding = values["binding"]
    assert isinstance(binding, TrainablePlanAuthorityBinding)
    path = (tmp_path / "trainable-plan-authority.json").resolve()
    publish_canonical_json_no_replace(
        path,
        trainable_plan_authority_binding_to_dict(binding),
    )
    return path, binding


def _allow_test_e1_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        e1_module,
        "_require_publishable_e1_content",
        lambda _bundle: None,
    )


def test_e1_anchor_authority_deep_reopens_plan_and_exact_two_recipes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, binding = _plan_source(tmp_path, scope="last1")
    content_path = (tmp_path / "trusted-content.json").resolve()
    publish_canonical_json_no_replace(content_path, {"kind": "test-content"})
    raw_content = CanonicalJsonProofBinding.bind(content_path)
    content_source = TrustedSingleOperatorContentBundleBinding(
        absolute_path=raw_content.absolute_path,
        size=raw_content.size,
        raw_sha256=raw_content.raw_sha256,
        semantic_sha256=_sha("trusted-e1-content"),
        runtime_binding_status="BOUND",
    )
    bundle = _trusted_bundle_shell(
        binding,
        semantic_sha256=content_source.semantic_sha256,
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "bind",
        classmethod(lambda _cls, _path: content_source),
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: bundle,
    )
    _allow_test_e1_bundle(monkeypatch)
    artifact = build_source_e1_recipe_anchor_authority_artifact(
        trusted_content_bundle_path=content_path,
        trainable_plan_authority_path=plan_path,
    )
    assert isinstance(artifact, E1RecipeAnchorAuthorityArtifact)
    assert artifact.schema_version == 3
    assert artifact.trusted_content_bundle_source == content_source
    assert artifact.trainable_plan_selector_id == E1_RECIPE_ANCHOR_PLAN_SELECTOR_ID
    assert (
        artifact.authority.trainable_plan_selector_id
        == E1_RECIPE_ANCHOR_PLAN_SELECTOR_ID
    )
    assert artifact.authority.trainable_plan_sha256 == binding.trainable_plan_sha256
    assert tuple(row.anchor_name for row in artifact.authority.anchors) == (
        "adamw",
        "sgdm",
    )
    assert artifact.authority.anchor("adamw").optimizer.learning_rate == 1e-4
    assert artifact.authority.anchor("sgdm").optimizer.momentum == 0.9

    output = (tmp_path / "e1-anchor-authority.json").resolve()
    publish_e1_recipe_anchor_authority_artifact(artifact, output)
    assert load_e1_recipe_anchor_authority_artifact(output) == artifact
    with pytest.raises(RuntimeError, match="target already exists"):
        publish_e1_recipe_anchor_authority_artifact(artifact, output)

    def reject_incomplete_bundle(_bundle: object) -> None:
        raise ValueError("formal v03 content coverage is incomplete")

    monkeypatch.setattr(
        e1_module,
        "_require_publishable_e1_content",
        reject_incomplete_bundle,
    )
    with pytest.raises(ValueError, match="coverage is incomplete"):
        publish_e1_recipe_anchor_authority_artifact(
            artifact,
            (tmp_path / "untrusted-e1-anchor-authority.json").resolve(),
        )

    plan_path.chmod(0o600)
    plan_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        load_e1_recipe_anchor_authority_artifact(output)


def test_e1_anchor_rejects_an_arbitrary_valid_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, binding = _plan_source(tmp_path, scope="all")
    content_path = (tmp_path / "trusted-content.json").resolve()
    publish_canonical_json_no_replace(content_path, {"kind": "test-content"})
    raw = CanonicalJsonProofBinding.bind(content_path)
    content_source = TrustedSingleOperatorContentBundleBinding(
        absolute_path=raw.absolute_path,
        size=raw.size,
        raw_sha256=raw.raw_sha256,
        semantic_sha256=_sha("arbitrary-plan-content"),
        runtime_binding_status="BOUND",
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "bind",
        classmethod(lambda _cls, _path: content_source),
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: _trusted_bundle_shell(
            binding,
            semantic_sha256=content_source.semantic_sha256,
        ),
    )
    _allow_test_e1_bundle(monkeypatch)
    with pytest.raises(ValueError, match="code-owned canonical structural plan"):
        build_source_e1_recipe_anchor_authority_artifact(
            trusted_content_bundle_path=content_path,
            trainable_plan_authority_path=plan_path,
        )


def test_e1_plan_rejects_foreign_bundle_revision_and_root(tmp_path: Path) -> None:
    _path, binding = _plan_source(tmp_path, scope="last1")
    e1_module._validate_e1_plan_against_trusted_content(
        binding,
        _trusted_bundle_shell(binding, semantic_sha256=_sha("trusted-e1-bundle")),
    )
    for foreign in (
        _trusted_bundle_shell(
            binding,
            semantic_sha256=_sha("foreign-e1-revision"),
            drafter_revision="f" * 40,
        ),
        _trusted_bundle_shell(
            binding,
            semantic_sha256=_sha("foreign-e1-root"),
            target_root=str((tmp_path / "foreign-target-root").resolve()),
        ),
    ):
        with pytest.raises(
            ValueError,
            match="trainable plan differs from bundle model/runtime identity",
        ):
            e1_module._validate_e1_plan_against_trusted_content(binding, foreign)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("absolute_path", "foreign-content.json"),
        ("size", 1),
        ("raw_sha256", "a" * 64),
        ("semantic_sha256", "b" * 64),
    ),
)
def test_e1_artifact_rejects_content_binding_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str | int,
) -> None:
    plan_path, binding = _plan_source(tmp_path, scope="last1")
    content_path = (tmp_path / "trusted-content.json").resolve()
    publish_canonical_json_no_replace(content_path, {"kind": "test-content"})
    foreign_path = (tmp_path / "foreign-content.json").resolve()
    publish_canonical_json_no_replace(foreign_path, {"kind": "foreign-content"})
    raw = CanonicalJsonProofBinding.bind(content_path)
    content_source = TrustedSingleOperatorContentBundleBinding(
        absolute_path=raw.absolute_path,
        size=raw.size,
        raw_sha256=raw.raw_sha256,
        semantic_sha256=_sha("tamper-test-content"),
        runtime_binding_status="BOUND",
    )
    bundle = _trusted_bundle_shell(
        binding,
        semantic_sha256=content_source.semantic_sha256,
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "bind",
        classmethod(lambda _cls, _path: content_source),
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: bundle,
    )
    _allow_test_e1_bundle(monkeypatch)
    artifact = build_source_e1_recipe_anchor_authority_artifact(
        trusted_content_bundle_path=content_path,
        trainable_plan_authority_path=plan_path,
    )
    encoded = deepcopy(artifact.to_dict())
    source = encoded["trusted_content_bundle_source"]
    assert isinstance(source, dict)
    source[field] = str(foreign_path) if field == "absolute_path" else replacement
    encoded["artifact_sha256"] = content_sha256(
        {key: value for key, value in encoded.items() if key != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="content bundle source changed"):
        E1RecipeAnchorAuthorityArtifact.from_dict(encoded)


def test_legacy_e1_artifact_is_load_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, binding = _plan_source(tmp_path, scope="last1")
    content_path = (tmp_path / "trusted-content.json").resolve()
    publish_canonical_json_no_replace(content_path, {"kind": "test-content"})
    raw = CanonicalJsonProofBinding.bind(content_path)
    content_source = TrustedSingleOperatorContentBundleBinding(
        absolute_path=raw.absolute_path,
        size=raw.size,
        raw_sha256=raw.raw_sha256,
        semantic_sha256=_sha("legacy-test-content"),
        runtime_binding_status="BOUND",
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "bind",
        classmethod(lambda _cls, _path: content_source),
    )
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "reopen",
        lambda _self: _trusted_bundle_shell(
            binding,
            semantic_sha256=content_source.semantic_sha256,
        ),
    )
    _allow_test_e1_bundle(monkeypatch)
    current = build_source_e1_recipe_anchor_authority_artifact(
        trusted_content_bundle_path=content_path,
        trainable_plan_authority_path=plan_path,
    )
    legacy = E1RecipeAnchorAuthorityArtifact(
        schema_version=2,
        kind=current.kind,
        trainable_plan_selector_id=current.trainable_plan_selector_id,
        trainable_plan_authority_source=current.trainable_plan_authority_source,
        trusted_content_bundle_source=None,
        authority=current.authority,
    )
    legacy_path = (tmp_path / "legacy-e1.json").resolve()
    publish_canonical_json_no_replace(legacy_path, legacy.to_dict())
    assert load_e1_recipe_anchor_authority_artifact(legacy_path) == legacy
    with pytest.raises(ValueError, match="requires schema 3"):
        publish_e1_recipe_anchor_authority_artifact(
            legacy,
            (tmp_path / "republished-legacy-e1.json").resolve(),
        )


def test_tts_source_authority_rejects_foreign_primary_source_bytes(
    tmp_path: Path,
) -> None:
    assert TTS_CALIBRATION_CLAIM_SCOPE.endswith("not_paper_reproduction")
    assert TTS_PRIMARY_SOURCE_PDF_SHA256 == (
        "7688b05bab7696f4a47a5987f2fcad13d46f1d84cec9f90caf661fb397f3ee20"
    )
    assert TTS_PRIMARY_SOURCE_ARCHIVE_SHA256 == (
        "22c549c0297fc0a2a71af002c3721f71ddfd06d86bc46b2f41592bd6748afe59"
    )
    paper_pdf = (tmp_path / "tts-v2.pdf").resolve()
    paper_source = (tmp_path / "tts-v2-source.tar.gz").resolve()
    paper_pdf.write_bytes(b"%PDF-1.7\nsource-owned TTS v2 fixture\n")
    paper_source.write_text("TTS source v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registered primary-source bytes"):
        build_source_tts_calibration_authority_artifact(
            paper_pdf_path=paper_pdf,
            paper_source_path=paper_source,
            tuning_workload_authority_path=tmp_path / "unused-workload.json",
            content_verification_receipt_path=tmp_path / "unused-content-receipt.json",
            tuning_window_path=tmp_path / "unused-window.json",
            trainable_plan_authority_path=tmp_path / "unused-plan.json",
            drafter_native_loss_path=tmp_path / "unused-loss.json",
        )


def test_tts_plan_selector_rejects_an_arbitrary_valid_e1_plan(tmp_path: Path) -> None:
    _path, binding = _plan_source(tmp_path, scope="all")
    with pytest.raises(ValueError, match="code-owned canonical TTS-Cal"):
        method_module._validate_tts_trainable_plan_selection(
            binding,
            binding.revalidate().plan,
        )


def test_tts_plan_selector_accepts_only_code_owned_calibration_slot(
    tmp_path: Path,
) -> None:
    _path, binding = _tts_plan_source(tmp_path)
    result = binding.revalidate()
    method_module._validate_tts_trainable_plan_selection(binding, result.plan)
    assert binding.method == "tts"
    assert binding.cell_id == method_module._canonical_tts_trainable_plan_cell_id()


def test_trusted_tts_plan_rejects_foreign_bundle_revision(tmp_path: Path) -> None:
    _path, binding = _tts_plan_source(tmp_path)
    method_module._validate_tts_plan_against_trusted_content(
        binding,
        _trusted_bundle_shell(binding, semantic_sha256=_sha("trusted-bundle")),
    )

    with pytest.raises(
        ValueError,
        match="trainable plan differs from bundle model/runtime identity",
    ):
        method_module._validate_tts_plan_against_trusted_content(
            binding,
            _trusted_bundle_shell(
                binding,
                semantic_sha256=_sha("foreign-bundle"),
                drafter_revision="f" * 40,
            ),
        )


def test_trusted_tts_window_is_exact_h80_and_rejects_foreign_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = _synthetic_tts_workload(tmp_path)
    locked_sha256 = _sha("trusted-locked-lcb-h80")
    bundle_sha256 = _sha("trusted-content-bundle")
    window = method_module._build_tts_calibration_tuning_window(
        workload,
        schema_version=5,
        workload_source_descriptor_sha256=locked_sha256,
        trusted_content_bundle_sha256=bundle_sha256,
        trusted_locked_workload_sha256=locked_sha256,
    )
    assert window.schema_version == 5
    assert len(window.tuning_entries) == 76
    assert len(window.excluded_pilot_entries) == 4
    assert window.trusted_content_bundle_sha256 == bundle_sha256
    assert window.trusted_locked_workload_sha256 == locked_sha256
    assert window.content_verification_receipt_sha256 is None
    assert TtsCalibrationTuningWindow.from_dict(window.to_dict()) == window

    window_path = (tmp_path / "trusted-tts-window.json").resolve()
    publish_canonical_json_no_replace(window_path, window.to_dict())
    window_source = CanonicalJsonProofBinding.bind(window_path)
    content_path = (tmp_path / "trusted-content-bundle.json").resolve()
    publish_canonical_json_no_replace(content_path, {"kind": "test-content"})
    raw_content = CanonicalJsonProofBinding.bind(content_path)
    content_source = TrustedSingleOperatorContentBundleBinding(
        absolute_path=raw_content.absolute_path,
        size=raw_content.size,
        raw_sha256=raw_content.raw_sha256,
        semantic_sha256=bundle_sha256,
        runtime_binding_status="BOUND",
    )
    locked = SimpleNamespace(sha256=locked_sha256)

    def replay(source: TrustedSingleOperatorContentBundleBinding):
        bundle = object.__new__(TrustedSingleOperatorContentBundle)
        object.__setattr__(bundle, "semantic_sha256", source.semantic_sha256)
        return bundle, locked, workload

    monkeypatch.setattr(method_module, "_trusted_tts_sources_from_binding", replay)
    reopened, _bundle, _locked = method_module._reopen_trusted_tuning_window(
        window_source,
        content_source=content_source,
    )
    assert reopened == window

    foreign_source = TrustedSingleOperatorContentBundleBinding(
        absolute_path=raw_content.absolute_path,
        size=raw_content.size,
        raw_sha256=raw_content.raw_sha256,
        semantic_sha256=_sha("foreign-content-bundle"),
        runtime_binding_status="BOUND",
    )
    with pytest.raises(ValueError, match="differs from code-owned selector"):
        method_module._reopen_trusted_tuning_window(
            window_source,
            content_source=foreign_source,
        )

    tampered = window.to_dict()
    tampered["trusted_content_bundle_sha256"] = _sha("tampered-content-bundle")
    tampered_path = (tmp_path / "tampered-trusted-tts-window.json").resolve()
    publish_canonical_json_no_replace(tampered_path, tampered)
    with pytest.raises(ValueError, match="differs from code-owned selector"):
        method_module._reopen_trusted_tuning_window(
            CanonicalJsonProofBinding.bind(tampered_path),
            content_source=content_source,
        )


def test_trusted_tts_source_artifact_embeds_bundle_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan = _tts_plan_source(tmp_path)
    workload = _synthetic_tts_workload(tmp_path)
    bundle_sha256 = _sha("source-artifact-content-bundle")
    locked_sha256 = _sha("source-artifact-locked-workload")
    window = method_module._build_tts_calibration_tuning_window(
        workload,
        schema_version=5,
        workload_source_descriptor_sha256=locked_sha256,
        trusted_content_bundle_sha256=bundle_sha256,
        trusted_locked_workload_sha256=locked_sha256,
    )
    window_path = (tmp_path / "trusted-window.json").resolve()
    publish_canonical_json_no_replace(window_path, window.to_dict())
    loss_path = (tmp_path / "trusted-loss.json").resolve()
    publish_canonical_json_no_replace(loss_path, TTS_DRAFTER_NATIVE_LOSS_SOURCE)
    content_path = (tmp_path / "trusted-content.json").resolve()
    publish_canonical_json_no_replace(content_path, {"kind": "test-content"})
    raw_content = CanonicalJsonProofBinding.bind(content_path)
    content_source = TrustedSingleOperatorContentBundleBinding(
        absolute_path=raw_content.absolute_path,
        size=raw_content.size,
        raw_sha256=raw_content.raw_sha256,
        semantic_sha256=bundle_sha256,
        runtime_binding_status="BOUND",
    )
    locked = SimpleNamespace(sha256=locked_sha256)

    def replay(source: TrustedSingleOperatorContentBundleBinding):
        return (
            _trusted_bundle_shell(
                plan,
                semantic_sha256=source.semantic_sha256,
            ),
            locked,
            workload,
        )

    monkeypatch.setattr(method_module, "_trusted_tts_sources_from_binding", replay)
    monkeypatch.setattr(method_module, "_reopen_raw", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        TrustedSingleOperatorContentBundleBinding,
        "bind",
        classmethod(lambda _cls, _path: content_source),
    )
    paper_pdf = (tmp_path / "paper.pdf").resolve()
    paper_pdf.write_bytes(b"%PDF-1.7\ntest\n")
    paper_source = (tmp_path / "paper.tar.gz").resolve()
    paper_source.write_bytes(b"test source\n")

    artifact = build_source_tts_calibration_authority_artifact(
        paper_pdf_path=paper_pdf,
        paper_source_path=paper_source,
        trusted_content_bundle_path=content_path,
        tuning_window_path=window_path,
        trainable_plan_authority_path=plan_path,
        drafter_native_loss_path=loss_path,
    )
    assert artifact.schema_version == 4
    assert artifact.trusted_content_bundle_source == content_source
    assert artifact.tuning_workload_authority_source is None
    assert artifact.content_verification_receipt_source is None
    encoded = artifact.to_dict()
    assert "trusted_content_bundle_source" in encoded
    assert "content_verification_receipt_source" not in encoded
    assert TtsCalibrationSourceAuthorityArtifact.from_dict(encoded) == artifact

    foreign = deepcopy(encoded)
    foreign_source = foreign["trusted_content_bundle_source"]
    assert isinstance(foreign_source, dict)
    foreign_source["semantic_sha256"] = _sha("foreign-source-artifact-bundle")
    foreign["artifact_sha256"] = content_sha256(
        {key: value for key, value in foreign.items() if key != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="differs from code-owned selector"):
        TtsCalibrationSourceAuthorityArtifact.from_dict(foreign)


def test_trusted_tts_cli_lanes_require_no_release_signer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    window_binding = SimpleNamespace(semantic_sha256=_sha("trusted-cli-window"))

    def publish_window(**kwargs):
        captured["window"] = kwargs
        return window_binding

    monkeypatch.setattr(
        cli_module,
        "publish_code_owned_trusted_tts_calibration_tuning_window",
        publish_window,
    )
    content_path = (tmp_path / "content.json").resolve()
    window_path = (tmp_path / "window.json").resolve()
    assert (
        cli_module.main(
            [
                "publish-tts-calibration-tuning-window",
                "--trusted-content-bundle",
                str(content_path),
                "--output",
                str(window_path),
            ]
        )
        == 0
    )
    assert captured["window"] == {
        "trusted_content_bundle_path": content_path,
        "output_path": window_path,
    }

    source_artifact = SimpleNamespace(
        authority=SimpleNamespace(sha256=_sha("trusted-cli-source"))
    )

    def build_source(**kwargs):
        captured["source"] = kwargs
        return source_artifact

    monkeypatch.setattr(
        cli_module,
        "build_source_tts_calibration_authority_artifact",
        build_source,
    )
    monkeypatch.setattr(
        cli_module,
        "publish_tts_calibration_authority_artifact",
        lambda *_args: None,
    )
    source_path = (tmp_path / "source-authority.json").resolve()
    assert (
        cli_module.main(
            [
                "publish-tts-calibration-source-authority",
                "--paper-pdf",
                str(tmp_path / "paper.pdf"),
                "--paper-source",
                str(tmp_path / "paper.tar.gz"),
                "--trusted-content-bundle",
                str(content_path),
                "--tuning-window",
                str(window_path),
                "--trainable-plan-authority",
                str(tmp_path / "plan.json"),
                "--drafter-native-loss",
                str(tmp_path / "loss.json"),
                "--output",
                str(source_path),
            ]
        )
        == 0
    )
    source_kwargs = captured["source"]
    assert isinstance(source_kwargs, dict)
    assert source_kwargs["trusted_content_bundle_path"] == content_path
    assert source_kwargs["tuning_workload_authority_path"] is None
    assert source_kwargs["content_verification_receipt_path"] is None


def test_tts_loss_source_reopens_exact_pinned_runtime_recipe(tmp_path: Path) -> None:
    loss_path = (tmp_path / "tts-loss.json").resolve()
    publish_canonical_json_no_replace(loss_path, TTS_DRAFTER_NATIVE_LOSS_SOURCE)
    assert (
        method_module._reopen_loss_source(CanonicalJsonProofBinding.bind(loss_path))
        == TTS_DRAFTER_NATIVE_LOSS_SOURCE
    )
    runtime = TTS_DRAFTER_NATIVE_LOSS_SOURCE["runtime_identity"]
    loss = TTS_DRAFTER_NATIVE_LOSS_SOURCE["loss"]
    assert isinstance(runtime, dict)
    assert isinstance(loss, dict)
    assert runtime["patch_sha256"] == TTS_DFLASH_LOSS_PATCH_SHA256
    patch_path = Path(method_module.__file__).resolve().parents[3] / str(
        runtime["patch_file"]
    )
    assert hashlib.sha256(patch_path.read_bytes()).hexdigest() == (
        TTS_DFLASH_LOSS_PATCH_SHA256
    )
    assert loss["objective"] == ("masked_position_weighted_target_to_draft_forward_kl")
    assert loss["equivalent_one_based_formula"] == "exp(-(k-1)/7)"
    assert loss["proximal_penalty"] == "absent"

    tampered = deepcopy(TTS_DRAFTER_NATIVE_LOSS_SOURCE)
    tampered_loss = tampered["loss"]
    assert isinstance(tampered_loss, dict)
    tampered_loss["proximal_penalty"] = "present"
    tampered_path = (tmp_path / "tampered-tts-loss.json").resolve()
    publish_canonical_json_no_replace(tampered_path, tampered)
    with pytest.raises(ValueError, match="differs from protocol"):
        method_module._reopen_loss_source(CanonicalJsonProofBinding.bind(tampered_path))


def test_tts_loss_source_public_producer_is_path_only_and_no_replace(
    tmp_path: Path,
) -> None:
    api_path = (tmp_path / "api-tts-loss.json").resolve()
    binding = publish_code_owned_tts_drafter_native_loss_source(api_path)
    assert binding.reopen() == TTS_DRAFTER_NATIVE_LOSS_SOURCE
    assert method_module._reopen_loss_source(binding) == (
        TTS_DRAFTER_NATIVE_LOSS_SOURCE
    )
    with pytest.raises(RuntimeError, match="target already exists"):
        publish_code_owned_tts_drafter_native_loss_source(api_path)

    cli_path = (tmp_path / "cli-tts-loss.json").resolve()
    assert (
        cli_module.main(
            [
                "publish-tts-drafter-native-loss-source",
                "--output",
                str(cli_path),
            ]
        )
        == 0
    )
    assert CanonicalJsonProofBinding.bind(cli_path).reopen() == (
        TTS_DRAFTER_NATIVE_LOSS_SOURCE
    )


def test_chronobelief_source_authority_rejects_foreign_preregistration_bytes(
    tmp_path: Path,
) -> None:
    assert CHRONOBELIEF_CLAIM_SCOPE.startswith("project_owned_preregistered")
    assert CHRONOBELIEF_PREREG_PDF_SHA256 == (
        "2e79b6d6414d40b38d405f8165d80bb4efd354bf03b2f9ca53df23220435fc7c"
    )
    assert CHRONOBELIEF_PREREG_TEX_SHA256 == (
        "941b891e85f7551360133fe13131b88ab0412ecf7f617d3fb959126af43d7d08"
    )
    paper_pdf = (tmp_path / "paper.pdf").resolve()
    tex_source = (tmp_path / "paper.tex").resolve()
    paper_pdf.write_bytes(b"%PDF-1.7\nChronoBelief fixture\n")
    tex_source.write_text("equations 5.5--5.8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registered primary-source bytes"):
        build_source_chronobelief_authority_artifact(
            paper_pdf_path=paper_pdf,
            tex_source_path=tex_source,
        )
