from __future__ import annotations

import json
from dataclasses import replace

import pytest

from lightcone_spec.experiments.production_output import (
    OutputStatus,
    build_production_output_artifact,
)
from lightcone_spec.experiments.production_visual_plan import (
    ProductionVisualRenderBlocked,
    build_production_visual_plan,
    require_production_visual_renderable,
)


def _blocked_output():
    return build_production_output_artifact(
        e3b_stage=None,
        industrial_reduction=None,
        family_power_reduction=None,
    )


def test_missing_paper_and_sources_emit_one_canonical_planned_block() -> None:
    source = _blocked_output()
    plan = build_production_visual_plan(source)

    assert plan.lifecycle == "planned"
    assert plan.status is OutputStatus.BLOCKED
    assert plan.blocker_codes == (
        "confirmation_family_power_reduction_artifact_missing",
        "e3b_long_context_stage_artifact_missing",
        "industrial_reducer_artifact_missing",
        "paper_main_tex_unregistered",
        "paper_visual_style_unregistered",
        "production_output_blocked",
    )
    assert plan.canonical_json_bytes() == plan.canonical_json_bytes()
    assert json.loads(plan.canonical_json_bytes()) == plan.to_dict()


def test_main_figure_plan_splits_registered_panels_four_plus_four() -> None:
    value = build_production_visual_plan(_blocked_output()).to_dict()
    figures = value["figures"]

    assert [figure["panels"] for figure in figures] == [4, 4]
    assert [figure["placement"] for figure in figures] == ["main", "main"]
    assert [figure["render_mode"] for figure in figures] == [
        "vector-native",
        "vector-native",
    ]
    assert figures[0]["panel_ids"] == [
        "accepted_length:l0_naive:tts",
        "accepted_length:lightcone:l0_naive",
        "accepted_length:lightcone:static",
        "accepted_length:lightcone:tts",
    ]
    assert figures[1]["panel_ids"] == [
        "committed_token_goodput:l0_naive:tts",
        "committed_token_goodput:lightcone:l0_naive",
        "committed_token_goodput:lightcone:static",
        "committed_token_goodput:lightcone:tts",
    ]
    assert all(figure["empirical_curves"] == [] for figure in figures)
    assert all(
        figure["uncertainty"]
        == {
            "status": "REQUIRED_NOT_EMITTED",
            "confidence": "registered_95_percent",
            "method": "paired_hierarchical_block_then_request_percentile",
            "interval_fields": ["estimate", "lower", "upper"],
        }
        for figure in figures
    )


def test_blocked_plan_contains_no_result_or_render_artifact() -> None:
    plan = build_production_visual_plan(_blocked_output())
    value = plan.to_dict()

    assert value["paper"] == {
        "target_visual_manifest_schema_version": 2,
        "main_tex": None,
        "body_font_pt": None,
        "textwidth_pt": None,
        "style_manifest": None,
        "style_sha256": None,
        "legend_location": None,
        "compiled_pdf": None,
        "compiled_pdf_sha256": None,
        "compile_log": None,
        "visual_gate_report": None,
        "review_status": "NOT_RUN",
    }
    assert all(
        figure["renderer_source"] is None
        and figure["pdf"] is None
        and figure["visual_review"]
        == {"status": "NOT_RUN", "png": None, "pdf_sha256": None}
        for figure in value["figures"]
    )
    assert all(
        table["tex"] is None
        and table["rows"] is None
        and table["columns"] is None
        and table["compile_status"] == "NOT_RUN"
        for table in value["tables"]
    )
    assert value["tables"][0]["evidence_role"] == ("preregistered_power_planning_only")
    assert value["tables"][0]["formal_result_eligible"] is False
    assert b'"points"' not in plan.canonical_json_bytes()
    assert b'"scientific_status"' not in plan.canonical_json_bytes()
    assert b'"selected_final_blocks"' not in plan.canonical_json_bytes()
    assert b'"p99_rows"' not in plan.canonical_json_bytes()
    assert b'"primary_rows"' not in plan.canonical_json_bytes()


def test_render_gate_raises_named_blockers_before_any_output() -> None:
    plan = build_production_visual_plan(_blocked_output())

    with pytest.raises(ProductionVisualRenderBlocked) as caught:
        require_production_visual_renderable(plan)
    assert caught.value.blocker_codes == plan.blocker_codes
    assert "paper_main_tex_unregistered" in str(caught.value)

    with pytest.raises(TypeError, match="exact plan artifact"):
        require_production_visual_renderable(plan.to_dict())  # type: ignore[arg-type]


def test_plan_rejects_foreign_or_mutated_source_and_lifecycle_promotion() -> None:
    source = _blocked_output()
    with pytest.raises(TypeError, match="exact output artifact"):
        build_production_visual_plan(source.to_dict())  # type: ignore[arg-type]

    source_sha256 = source.sha256
    source_blocker_codes = source.blocker_codes
    object.__setattr__(source, "blocker_codes", ("forged",))
    with pytest.raises(ValueError, match="blockers differ"):
        build_production_visual_plan(source)
    object.__setattr__(source, "blocker_codes", source_blocker_codes)
    assert source.sha256 == source_sha256

    plan = build_production_visual_plan(_blocked_output())
    payload = plan.to_dict()
    payload["lifecycle"] = "reviewed"
    forged = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    with pytest.raises(ValueError, match="exact source replay"):
        replace(plan, canonical_payload=forged)

    payload = plan.to_dict()
    payload["status"] = OutputStatus.READY.value
    forged = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    with pytest.raises(ValueError, match="exact source replay"):
        replace(plan, canonical_payload=forged)


def test_plan_rejects_coordinated_payload_rehash_with_result_or_output() -> None:
    plan = build_production_visual_plan(_blocked_output())
    payload = plan.to_dict()
    payload["figures"][0]["empirical_curves"] = [{"name": "forged", "estimate": 1.0}]
    forged = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    with pytest.raises(ValueError, match="exact source replay"):
        replace(plan, canonical_payload=forged)

    payload = plan.to_dict()
    payload["tables"][0]["rows"] = [{"power": 0.9}]
    forged = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    with pytest.raises(ValueError, match="exact source replay"):
        replace(plan, canonical_payload=forged)

    payload = plan.to_dict()
    payload["figures"][0]["pdf"] = "forged.pdf"
    payload["figures"][0]["visual_review"] = {
        "status": "pass",
        "png": "forged.png",
        "pdf_sha256": "a" * 64,
    }
    forged = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    with pytest.raises(ValueError, match="exact source replay"):
        replace(plan, canonical_payload=forged)
