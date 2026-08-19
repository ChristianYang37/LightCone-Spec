from __future__ import annotations

import csv
from pathlib import Path

import pytest

from lightcone_spec.experiments import formal_single_operator_loads as load_module
from lightcone_spec.experiments.formal_single_operator_loads import (
    BURSTGPT_V2_ACTIVE_ASSET,
    BURSTGPT_V2_ASSETS,
    BurstGptV2ReleaseVerification,
    BurstGptVerifiedAsset,
    E3aLambdaStar,
    E5ArrivalPlan,
    derive_e5_arrival_plan,
    select_burstgpt_arrival_window,
)


def _lambda_star() -> E3aLambdaStar:
    return E3aLambdaStar(
        numerator_requests_x_1e9=20_000_000_000,
        denominator_window_ns=1_000_000_000,
        source_cell_id="a" * 64,
        source_observation_sha256="b" * 64,
        common_load=8,
        matched_width=16,
        rule=(
            "E3a_Static_context_40928_short_input_long_generation_"
            "matched_width_common_load_completed_requests_per_second"
        ),
    )


def _low_lambda_star() -> E3aLambdaStar:
    return E3aLambdaStar(
        numerator_requests_x_1e9=1_000_000_000,
        denominator_window_ns=1_000_000_000,
        source_cell_id="c" * 64,
        source_observation_sha256="d" * 64,
        common_load=2,
        matched_width=4,
        rule=(
            "E3a_Static_context_40928_short_input_long_generation_"
            "matched_width_common_load_completed_requests_per_second"
        ),
    )


def _verification(*, active_rows: int) -> BurstGptV2ReleaseVerification:
    return BurstGptV2ReleaseVerification(
        schema_version=1,
        kind="burstgpt_v2_release_verification",
        release_tag=load_module.BURSTGPT_V2_RELEASE_TAG,
        release_tag_commit=load_module.BURSTGPT_V2_RELEASE_TAG_COMMIT,
        release_id=load_module.BURSTGPT_V2_RELEASE_ID,
        release_published_at=load_module.BURSTGPT_V2_RELEASE_PUBLISHED_AT,
        release_url=load_module.BURSTGPT_V2_RELEASE_URL,
        active_asset=BURSTGPT_V2_ACTIVE_ASSET,
        assets=tuple(
            BurstGptVerifiedAsset(
                row.name,
                row.size,
                row.sha256,
                active_rows if row.name == BURSTGPT_V2_ACTIVE_ASSET else 1,
            )
            for row in BURSTGPT_V2_ASSETS
        ),
    )


def _active_csv(path: Path, *, rows: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "Timestamp",
                "Session ID",
                "Elapsed time",
                "Model",
                "Request tokens",
                "Response tokens",
                "Total tokens",
                "Log Type",
            )
        )
        for ordinal in range(rows):
            writer.writerow(
                (
                    str(ordinal * 3),
                    f"session-{ordinal}",
                    "1",
                    "ChatGPT",
                    "10",
                    "5",
                    "15",
                    "API log",
                )
            )


def test_burstgpt_window_is_disjoint_stratified_and_digest_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = 2_400
    path = tmp_path / BURSTGPT_V2_ACTIVE_ASSET
    _active_csv(path, rows=rows)
    verification = _verification(active_rows=rows)
    active = next(
        row for row in verification.assets if row.name == BURSTGPT_V2_ACTIVE_ASSET
    )
    monkeypatch.setattr(
        load_module,
        "_hash_file",
        lambda _path: (active.sha256, active.size),
    )

    first = select_burstgpt_arrival_window(
        active_asset_path=path,
        verification=verification,
        block=0,
        request_count=20,
        target_rate=_lambda_star().requests_per_second,
    )
    second = select_burstgpt_arrival_window(
        active_asset_path=path,
        verification=verification,
        block=1,
        request_count=20,
        target_rate=_lambda_star().requests_per_second,
    )
    assert first.source_start_row + first.source_row_count <= second.source_start_row
    assert first.scaled_arrivals_us[0] == 0
    assert first.scaled_arrivals_us[-1] == 950_000
    assert first.sha256 != second.sha256


def test_open_loop_arrivals_are_paired_across_methods() -> None:
    dimensions = {
        "backend_authority": "DFLASH",
        "family_id": "open_loop_0.75",
        "topology": "tp1_dp1",
        "load_factor": 0.75,
    }
    static = derive_e5_arrival_plan(
        cell_id="1" * 64,
        block=4,
        family="open_loop",
        dimensions={**dimensions, "method_role": "Static"},
        lambda_star=_lambda_star(),
    )
    lightcone = derive_e5_arrival_plan(
        cell_id="2" * 64,
        block=4,
        family="open_loop",
        dimensions={
            **dimensions,
            "method_role": "LightCone",
            "recipe_sha256": "f" * 64,
        },
        lambda_star=_lambda_star(),
    )
    assert static.paired_trace_sha256 == lightcone.paired_trace_sha256
    assert static.arrivals_us == lightcone.arrivals_us
    assert static.sha256 != lightcone.sha256
    assert static.arrivals_us[-1] < static.arrival_duration_us
    assert E5ArrivalPlan.from_dict(static.to_dict()) == static


def test_registered_families_and_p99_extension_are_explicit() -> None:
    closed = derive_e5_arrival_plan(
        cell_id="3" * 64,
        block=0,
        family="closed_loop",
        dimensions={"concurrency": 256, "family_id": "closed_loop_c256"},
        lambda_star=_lambda_star(),
    )
    soak = derive_e5_arrival_plan(
        cell_id="4" * 64,
        block=0,
        family="trace_or_soak",
        dimensions={"arrival": "overload_soak"},
        lambda_star=_lambda_star(),
        selected_p99_anchor=True,
    )
    assert len(closed.arrivals_us) == 2_400
    assert set(closed.arrivals_us) == {0}
    assert len(soak.arrivals_us) == 11_000
    assert soak.arrival_duration_us == soak.arrivals_us[-1] + 1
    assert soak.arrival_duration_us > 300_000_000
    assert soak.effective_rate_numerator == 25
    assert soak.effective_rate_denominator == 1
    assert soak.p99_extension_minimum_completed == 10_000
    assert soak.p99_extension_offered_requests == 11_000


@pytest.mark.parametrize(
    ("family", "dimensions", "expected_policy"),
    (
        ("closed_loop", {"concurrency": 1}, "closed_loop"),
        ("topology_cohort", {"cohort_count": 1}, "closed_loop"),
        ("open_loop", {"load_factor": 0.25}, "poisson"),
        ("trace_or_soak", {"arrival": "immediate_burst"}, "immediate_burst"),
        ("trace_or_soak", {"arrival": "moderate_soak"}, "moderate_soak"),
    ),
)
def test_selected_p99_family_has_exact_paired_11k_offer_at_low_lambda(
    family: str,
    dimensions: dict[str, object],
    expected_policy: str,
) -> None:
    shared = {
        "backend_authority": "DFLASH",
        "family_id": f"p99-{family}",
        "topology": "tp1_dp1",
        **dimensions,
    }
    rows = tuple(
        derive_e5_arrival_plan(
            cell_id=f"{index:x}" * 64,
            block=4,
            family=family,  # type: ignore[arg-type]
            dimensions={**shared, "method_role": role},
            lambda_star=_low_lambda_star(),
            selected_p99_anchor=True,
        )
        for index, role in enumerate(
            ("Target-only", "Static", "TTS", "L0-naive", "LightCone"),
            start=1,
        )
    )
    assert {row.arrival_policy for row in rows} == {expected_policy}
    assert {row.paired_trace_sha256 for row in rows} == {rows[0].paired_trace_sha256}
    assert {row.arrivals_us for row in rows} == {rows[0].arrivals_us}
    assert {len(row.arrivals_us) for row in rows} == {11_000}
    assert all(row.arrivals_us[-1] < row.arrival_duration_us for row in rows)
    if expected_policy in {"poisson", "moderate_soak"}:
        assert len(set(rows[0].arrivals_us)) == 11_000
        assert rows[0].arrival_duration_us == rows[0].arrivals_us[-1] + 1


def test_non_anchor_remains_short_and_extension_tamper_is_rejected() -> None:
    dimensions = {
        "backend_authority": "DFLASH",
        "family_id": "open_loop_0.25",
        "topology": "tp1_dp1",
        "load_factor": 0.25,
    }
    ordinary = derive_e5_arrival_plan(
        cell_id="5" * 64,
        block=4,
        family="open_loop",
        dimensions=dimensions,
        lambda_star=_low_lambda_star(),
    )
    selected = derive_e5_arrival_plan(
        cell_id="6" * 64,
        block=4,
        family="open_loop",
        dimensions=dimensions,
        lambda_star=_low_lambda_star(),
        selected_p99_anchor=True,
    )
    assert len(ordinary.arrivals_us) < 11_000
    assert ordinary.p99_extension_offered_requests is None
    assert len(selected.arrivals_us) == 11_000
    assert E5ArrivalPlan.from_dict(selected.to_dict()) == selected
    tampered = selected.to_dict()
    tampered["arrivals_us"] = tampered["arrivals_us"][:-1]
    with pytest.raises(ValueError, match="p99 extension requirement"):
        E5ArrivalPlan.from_dict(tampered)


def test_lambda_star_rejects_token_rate_or_unbound_identity() -> None:
    value = {
        "numerator_requests_x_1e9": 20_000_000_000,
        "denominator_window_ns": 1_000_000_000,
        "source_cell_id": "not-a-digest",
        "source_observation_sha256": "b" * 64,
        "common_load": 8,
        "matched_width": 16,
        "rule": "token_throughput",
    }
    with pytest.raises(ValueError):
        E3aLambdaStar.from_e3a_selection(value)
