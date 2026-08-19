from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lightcone_spec.experiments import e0_stage_authority
from lightcone_spec.experiments import formal_single_operator_downstream as downstream
from lightcone_spec.experiments.e0_stage_authority import (
    E0OnlineSpecSourceAuthority,
)
from lightcone_spec.experiments.formal_registry import (
    e0_compatibility_receipt_to_dict,
    e0_onlinespec_source_authority_to_dict,
)
from lightcone_spec.experiments.onlinespec import (
    ONLINE_SPEC_COMMIT,
    ONLINE_SPEC_SOURCE_AUDIT_SHA256,
    ONLINE_SPEC_TREE,
)
from lightcone_spec.experiments.stage_materialization import (
    E0_BACKENDS,
    E0_MODELS,
    E0_TASKS,
    E1A_FIXED_VERIFICATION_BUDGET,
    E1A_NATIVE_VERIFICATION_BUDGET,
    E0CompatibilityDecision,
    E0CompatibilityReceipt,
    GpuHourEstimate,
    _materialize_e4_profiler_diagnostic,
    content_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@pytest.mark.parametrize(("recovered", "status"), ((True, "PASS"), (False, "FAIL")))
def test_e5_recovery_disposition_is_complete_pass_or_fail(
    recovered: bool,
    status: str,
) -> None:
    actual = SimpleNamespace(
        reducer_payload={
            "diagnostic_status": "PASS" if recovered else "FAIL",
            "recovered": recovered,
        },
        result_identity_sha256=_sha(f"failure-terminal:{recovered}"),
    )
    cell = SimpleNamespace(
        cell_id=_sha("failure-cell"),
        backend="DFLASH",
        dimensions=(
            ("cohort_count", 1),
            ("failure", "queue_saturation"),
            ("topology", "tp1_dp1"),
        ),
    )

    row = downstream._failure_payload(actual, cell)

    assert row["status"] == status
    assert row["recovered"] is recovered


def test_e5_failure_recovery_flag_and_disposition_must_agree() -> None:
    actual = SimpleNamespace(
        reducer_payload={"diagnostic_status": "PASS", "recovered": False},
        result_identity_sha256=_sha("mismatched-failure-terminal"),
    )
    cell = SimpleNamespace(
        cell_id=_sha("failure-cell"),
        backend="DFLASH",
        dimensions=(
            ("cohort_count", 1),
            ("failure", "queue_saturation"),
            ("topology", "tp1_dp1"),
        ),
    )

    with pytest.raises(ValueError, match="flag and disposition differ"):
        downstream._failure_payload(actual, cell)


def test_e5_power_negative_is_sealed_before_p99_anchor_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grouped = {}
    for backend in downstream.E5_BACKENDS:
        for topology in downstream.E5_TOPOLOGIES:
            for ordinal in range(15):
                family = (
                    ("backend_authority", backend),
                    ("family_id", f"family-{ordinal:02d}"),
                    ("topology", topology),
                )
                grouped[family] = {block: {} for block in range(4)}
    assert len(grouped) == 90

    monkeypatch.setattr(downstream, "_serving_observations", lambda *_args: {})
    monkeypatch.setattr(downstream, "_e5_grouped", lambda *_args: grouped)
    monkeypatch.setattr(
        downstream,
        "_family_role_exclusions",
        lambda _blocks: {
            "LightCone": {
                "reason_codes": ["no_published_update"],
                "evidence_cell_ids": [_sha("unsafe-lightcone")],
            }
        },
    )
    monkeypatch.setattr(
        downstream,
        "_paired_role_goodputs",
        lambda *_args, **_kwargs: {
            role: SimpleNamespace(
                goodput_tokens_per_second=100.0,
                sha256=_sha(f"slo:{role}"),
            )
            for role in downstream.FORMAL_METHOD_ROLES
        },
    )
    monkeypatch.setattr(
        downstream,
        "require_paired_primary_goodputs",
        lambda observations: tuple(
            (role, observation.goodput_tokens_per_second)
            for role, observation in observations.items()
        ),
    )
    monkeypatch.setattr(
        downstream,
        "_maximum_request_p99_ns",
        lambda _observation: (_ for _ in ()).throw(
            AssertionError("negative power must not select a p99 anchor")
        ),
    )
    predecessor = SimpleNamespace(
        artifact=SimpleNamespace(node="e1a"),
        decision=SimpleNamespace(
            payload={
                "model": "Qwen/Qwen3-8B",
                "frozen_tts_recipe_sha256": _sha("tts"),
                "source_lightcone_recipe_sha256": _sha("dflash"),
                "selected_dspark_recipe_sha256": _sha("dspark"),
            }
        ),
    )
    materialization = SimpleNamespace(
        stage="E5",
        materialization_rule=("e5_exact_450_headline_rows_x_4_excluded_pilot_blocks"),
        cells=(None,) * 1_800,
        sha256=_sha("e5-pilot-materialization"),
    )

    draft = downstream.reduce_single_operator_e5_pilot(
        predecessor,
        materialization,
        (),
    )

    assert draft.payload["status"] == "POWER_UNRESOLVED"
    assert draft.payload["p99_anchors"] == []


@pytest.mark.parametrize(
    ("kind", "node", "status"),
    (
        ("e5", "e1a", "NO_SAFE_CONFIGURATION"),
        ("e0", "e0_tuning", "NO_SAFE_WINNER"),
    ),
)
def test_downstream_negative_selection_is_a_typed_scientific_stage_block(
    kind: str,
    node: str,
    status: str,
) -> None:
    from lightcone_spec.experiments.formal_single_operator_stages import (
        FormalSingleOperatorStageBlocked,
    )

    predecessor = SimpleNamespace(
        artifact=SimpleNamespace(node=node),
        decision=SimpleNamespace(
            payload={"status": status, "selection_sha256": _sha("selection")}
        ),
    )
    lock = SimpleNamespace(sha256=_sha("lock"))

    with pytest.raises(FormalSingleOperatorStageBlocked, match=status):
        if kind == "e5":
            downstream.materialize_single_operator_e5_pilot(predecessor, lock)
        else:
            downstream.materialize_single_operator_e0_pilot(
                predecessor,
                lock,
                {},
            )

    predecessor.decision.payload["status"] = "UNKNOWN_SELECTION_STATUS"
    with pytest.raises(ValueError, match="status is malformed"):
        if kind == "e5":
            downstream.materialize_single_operator_e5_pilot(predecessor, lock)
        else:
            downstream.materialize_single_operator_e0_pilot(
                predecessor,
                lock,
                {},
            )


def _paired_family() -> dict[str, dict[str, tuple[float, float]]]:
    result = {}
    for name in downstream._CORE_CONTRAST_ROLES:
        result[name] = {
            f"block-{index}": (110.0 + index, 100.0 + (index % 2))
            for index in range(12)
        }
    return result


def test_target_only_slo_is_a_real_deployment_gate() -> None:
    reduction = downstream._resolve_family_contrasts(
        paired=_paired_family(),
        contrast_roles=downstream._CORE_CONTRAST_ROLES,
        exclusions={},
        target_slo_pass=False,
    )

    assert reduction["deployment_confirmed"] is False
    assert reduction["target_only_gate"]["passed"] is False
    assert reduction["target_only_gate"]["target_only_slo_passed"] is False
    assert "target_only_slo_failed" in reduction["reason_codes"]


def test_unsafe_role_is_excluded_while_independent_contrast_continues() -> None:
    reduction = downstream._resolve_family_contrasts(
        paired=_paired_family(),
        contrast_roles=downstream._CORE_CONTRAST_ROLES,
        exclusions={
            "L0-naive": {
                "reason_codes": ["nonfinite_updates"],
                "evidence_cell_ids": [_sha("unsafe-l0")],
            }
        },
        target_slo_pass=True,
    )
    contrasts = reduction["contrast_payloads"]

    assert contrasts["l0_naive_vs_tts"]["status"] == ("EXCLUDED_UNSAFE_OR_INACTIVE")
    assert contrasts["lightcone_vs_l0_naive"]["status"] == (
        "EXCLUDED_UNSAFE_OR_INACTIVE"
    )
    assert contrasts["lightcone_vs_static"]["status"] == "RESOLVED"
    assert reduction["all_registered_contrasts_resolved"] is False
    assert reduction["deployment_confirmed"] is False
    assert "registered_contrast_family_incomplete" in reduction["reason_codes"]


def test_explicitly_unsafe_role_does_not_abort_safe_role_exactness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compared_roles: list[set[str]] = []
    rows = {
        role: (
            SimpleNamespace(cell_id=_sha(f"cell:{role}")),
            {
                "role": role,
                "source_request_pool_sha256": _sha("paired-pool"),
            },
        )
        for role in downstream.FORMAL_METHOD_ROLES
    }

    monkeypatch.setattr(
        downstream,
        "_request_evidence",
        lambda observation: (observation["role"],),
    )
    monkeypatch.setattr(
        downstream,
        "_slo",
        lambda _observation: SimpleNamespace(
            source_request_pool_sha256=_sha("paired-pool")
        ),
    )
    monkeypatch.setattr(
        downstream,
        "require_paired_completed_output_exactness",
        lambda evidence, **_kwargs: compared_roles.append(set(evidence)),
    )
    monkeypatch.setattr(
        downstream,
        "require_paired_primary_goodputs",
        lambda _observations: (),
    )

    downstream._paired_role_goodputs(
        rows,  # type: ignore[arg-type]
        excluded_roles=frozenset({"L0-naive"}),
    )

    assert compared_roles == [set(downstream.FORMAL_METHOD_ROLES) - {"L0-naive"}]


def test_zero_goodput_contrast_is_unresolved_while_other_contrasts_continue() -> None:
    paired = _paired_family()
    paired["l0_naive_vs_tts"] = {
        f"block-{index}": (0.0, 100.0 + index) for index in range(12)
    }

    reduction = downstream._resolve_family_contrasts(
        paired=paired,
        contrast_roles=downstream._CORE_CONTRAST_ROLES,
        exclusions={},
        target_slo_pass=True,
    )
    contrasts = reduction["contrast_payloads"]

    unresolved = contrasts["l0_naive_vs_tts"]
    assert unresolved["status"] == "UNRESOLVED_ZERO_GOODPUT"
    assert "mean_relative_gain" not in unresolved
    assert "ci_lower_relative_gain" not in unresolved
    assert "ci_upper_relative_gain" not in unresolved
    assert contrasts["lightcone_vs_static"]["status"] == "RESOLVED"
    assert reduction["deployment_confirmed"] is False


def test_hierarchical_zero_does_not_mask_a_later_malformed_block() -> None:
    blocks = {
        "a": np.asarray([[0.0, 0.0, 1.0, 1.0, 0.0, 1.0]]),
        "b": np.asarray([[1.0, 2.0]]),
    }

    with pytest.raises(ValueError, match="rows are malformed"):
        downstream._hierarchical_interval_payload("contrast", blocks)

    blocks["b"] = np.asarray([[1.0, 0.0, 1.0, -1.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="tokens are negative"):
        downstream._hierarchical_interval_payload("contrast", blocks)


def test_unsafe_pilot_still_validates_measurement_structure() -> None:
    pilots = tuple(
        downstream.PilotBlock(
            block_id="duplicate",
            static_goodput=100.0,
            tts_goodput=100.0,
            lightcone_goodput=100.0,
        )
        for _ in range(4)
    )
    with pytest.raises(ValueError, match="unique"):
        downstream._pilot_power_resolution(
            pilots,
            {
                "LightCone": {
                    "reason_codes": ["nonfinite_updates"],
                    "evidence_cell_ids": [_sha("unsafe")],
                }
            },
        )


def _predecessor(
    *,
    node: str,
    materialization_sha256: str,
    source_sha256: str,
    payload: dict[str, object],
    predecessor: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        artifact=SimpleNamespace(node=node),
        materialization=SimpleNamespace(sha256=materialization_sha256),
        decision=SimpleNamespace(
            payload=payload,
            next_materialization_source_decision_sha256=source_sha256,
            next_materialization_upstream_receipt_sha256s=(materialization_sha256,),
        ),
        predecessor=predecessor,
    )


def _e0_compatibility(
    *,
    lock_sha256: str,
    e6_materialization_sha256: str,
    valid_count: int,
) -> E0CompatibilityReceipt:
    decisions = []
    for index, (model, backend, task) in enumerate(
        (model, backend, task)
        for model in E0_MODELS
        for backend in E0_BACKENDS
        for task in E0_TASKS
    ):
        valid = index < valid_count
        decisions.append(
            E0CompatibilityDecision(
                model=model,
                backend=backend,
                task=task,
                disposition="VALID" if valid else "N/A",
                reason_code="compatible" if valid else "unsupported_interface",
                interface_sha256=_sha(f"interface:{model}:{backend}:{task}"),
                task_native_workload_sha256=_sha(f"workload:{model}:{backend}:{task}"),
            )
        )
    return E0CompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=lock_sha256,
        upstream_e6_receipt_sha256=e6_materialization_sha256,
        decisions=tuple(sorted(decisions, key=lambda row: row.decision_id)),
    )


def _e0_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> E0OnlineSpecSourceAuthority:
    checkout = (tmp_path / "onlinespec").resolve()
    checkout.mkdir()
    audit = (tmp_path / "onlinespec-audit.json").resolve()
    audit.write_text("{}\n", encoding="utf-8")
    verified = {
        "commit": ONLINE_SPEC_COMMIT,
        "tree": ONLINE_SPEC_TREE,
        "key_files": {"optimizer.py": _sha("optimizer")},
    }
    monkeypatch.setattr(
        e0_stage_authority,
        "verify_onlinespec_source_checkout",
        lambda *_args, **_kwargs: verified,
    )
    return E0OnlineSpecSourceAuthority(
        schema_version=1,
        checkout_path=str(checkout),
        audit_path=str(audit),
        source_audit_sha256=ONLINE_SPEC_SOURCE_AUDIT_SHA256,
        commit=ONLINE_SPEC_COMMIT,
        tree=ONLINE_SPEC_TREE,
        verification_sha256=content_sha256(verified),
    )


def _e0_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    valid_count: int,
) -> tuple[SimpleNamespace, SimpleNamespace, E0CompatibilityReceipt, dict[str, object]]:
    lock = SimpleNamespace(sha256=_sha("e0-lock"))
    e6_materialization_sha256 = _sha("e6-final-materialization")
    confirmation_sha256 = _sha("e6-confirmation")
    e6 = _predecessor(
        node="e6_final",
        materialization_sha256=e6_materialization_sha256,
        source_sha256=confirmation_sha256,
        payload={
            "confirmation_sha256": confirmation_sha256,
            "frozen_tts_recipe_sha256": _sha("e0-tts"),
            "lightcone_recipe_sha256": _sha("e0-lightcone"),
        },
    )
    compatibility = _e0_compatibility(
        lock_sha256=lock.sha256,
        e6_materialization_sha256=e6_materialization_sha256,
        valid_count=valid_count,
    )
    authority = None if valid_count == 0 else _e0_authority(monkeypatch, tmp_path)
    bundle: dict[str, object] = {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_compatibility_bundle",
        "protocol_lock_sha256": lock.sha256,
        "upstream_e6_materialization_sha256": e6_materialization_sha256,
        "upstream_e6_confirmation_sha256": confirmation_sha256,
        "compatibility": e0_compatibility_receipt_to_dict(compatibility),
        "compatibility_sha256": compatibility.sha256,
        "compatibility_evidence_manifest_sha256": _sha("e0-evidence"),
        "onlinespec_source_authority": (
            None
            if authority is None
            else e0_onlinespec_source_authority_to_dict(authority)
        ),
        "onlinespec_source_authority_sha256": (
            None if authority is None else authority.sha256
        ),
        "started_ns": 10,
        "finished_ns": 20,
    }
    bundle["bundle_sha256"] = content_sha256(bundle)
    return e6, lock, compatibility, bundle


def _completion_from_draft(
    *,
    node: str,
    materialization: object,
    draft: object,
    predecessor: SimpleNamespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        artifact=SimpleNamespace(node=node),
        materialization=materialization,
        decision=draft,
        predecessor=predecessor,
    )


def test_e3b_pilot_and_powered_final_materialize_exact_prefixes() -> None:
    profiler_sha = _sha("profiler-materialization")
    completion_sha = _sha("profiler-completion")
    profiler = _predecessor(
        node="e4_profiler",
        materialization_sha256=profiler_sha,
        source_sha256=completion_sha,
        payload={
            "completion_sha256": completion_sha,
            "model": "Qwen/Qwen3-8B",
            "frozen_tts_recipe_sha256": _sha("tts"),
            "lightcone_recipe_sha256": _sha("lightcone"),
        },
    )
    lock = SimpleNamespace(sha256=_sha("lock"))

    pilot = downstream.materialize_single_operator_e3b_pilot(profiler, lock)
    assert pilot.expected_cell_count == 1_920
    assert pilot.materialization_rule == (
        "e3b_exact_480_rows_x_4_excluded_pilot_blocks"
    )
    assert {dict(cell.dimensions)["block"] for cell in pilot.cells} == set(range(4))

    power_sha = _sha("power")
    pilot_completion = _predecessor(
        node="e3b_pilot",
        materialization_sha256=pilot.sha256,
        source_sha256=power_sha,
        payload={
            "status": "READY",
            "power_prefix_sha256": power_sha,
            "selected_final_blocks": 12,
            "model": "Qwen/Qwen3-8B",
            "frozen_tts_recipe_sha256": _sha("tts"),
            "lightcone_recipe_sha256": _sha("lightcone"),
        },
    )
    final = downstream.materialize_single_operator_e3b_final(
        pilot_completion,
        lock,
    )
    assert final.expected_cell_count == 5_760
    assert {dict(cell.dimensions)["block"] for cell in final.cells} == set(range(4, 16))
    assert all(dict(cell.dimensions)["block_phase"] == "final" for cell in final.cells)


def test_e3b_underpowered_cannot_materialize_final() -> None:
    predecessor = _predecessor(
        node="e3b_pilot",
        materialization_sha256=_sha("pilot"),
        source_sha256=_sha("unused"),
        payload={
            "status": "UNDERPOWERED",
            "selected_final_blocks": None,
        },
    )
    with pytest.raises(RuntimeError, match="UNDERPOWERED"):
        downstream.materialize_single_operator_e3b_final(
            predecessor,
            SimpleNamespace(sha256=_sha("lock")),
        )


def test_e1a_materializes_exact_58_by_2_grid() -> None:
    confirmation_sha = _sha("e3b-confirmation")
    predecessor = _predecessor(
        node="e3b_final",
        materialization_sha256=_sha("e3b-final"),
        source_sha256=confirmation_sha,
        payload={
            "confirmation_sha256": confirmation_sha,
            "model": "Qwen/Qwen3-8B",
            "frozen_tts_recipe_sha256": _sha("tts"),
            "lightcone_recipe_sha256": _sha("lightcone"),
        },
    )
    receipt = downstream.materialize_single_operator_e1a(
        predecessor,
        SimpleNamespace(sha256=_sha("lock")),
    )
    assert receipt.expected_cell_count == 116
    assert {dict(cell.dimensions)["verification_mode"] for cell in receipt.cells} == {
        "fixed_verification_budget",
        "native_scheduler",
    }
    assert {
        (
            dict(cell.dimensions)["verification_mode"],
            dict(cell.dimensions)["fixed_verification_budget"],
        )
        for cell in receipt.cells
    } == {
        ("fixed_verification_budget", E1A_FIXED_VERIFICATION_BUDGET),
        ("native_scheduler", E1A_NATIVE_VERIFICATION_BUDGET),
    }
    assert sum(cell.method_role == "Target-only" for cell in receipt.cells) == 2
    assert sum(cell.method_role == "Static" for cell in receipt.cells) == 2
    assert (
        sum(cell.method_role == "LightCone-candidate" for cell in receipt.cells) == 112
    )


def test_e5_pilot_and_final_have_exact_headline_failure_counts() -> None:
    verification_sha = _sha("e1a-verification")
    e1a = _predecessor(
        node="e1a",
        materialization_sha256=_sha("e1a-materialization"),
        source_sha256=verification_sha,
        payload={
            "status": "READY",
            "verification_sha256": verification_sha,
            "model": "Qwen/Qwen3-8B",
            "frozen_tts_recipe_sha256": _sha("tts"),
            "source_lightcone_recipe_sha256": _sha("dflash"),
            "selected_dspark_recipe_sha256": _sha("dspark"),
        },
    )
    lock = SimpleNamespace(sha256=_sha("lock"))
    pilot = downstream.materialize_single_operator_e5_pilot(e1a, lock)
    assert pilot.expected_cell_count == 1_800
    assert all(cell.task == "production_slo_power_prefix" for cell in pilot.cells)

    anchors = [
        {
            "backend": backend,
            "topology": topology,
            "family_id": (
                "closed_loop_c1"
                if topology == "tp1_dp1"
                else f"topology_cohort_{topology}_k1_uniform"
            ),
            "minimum_completions": 10_000,
        }
        for backend in ("DFLASH", "DSPARK")
        for topology in ("tp1_dp1", "tp2_dp1", "tp1_dp2")
    ]
    for row in anchors:
        anchor = downstream.E5SelectedP99Anchor(**row)
        row["anchor_id"] = anchor.anchor_id
    power_sha = _sha("e5-power")
    power = _predecessor(
        node="e5_pilot",
        materialization_sha256=pilot.sha256,
        source_sha256=power_sha,
        payload={
            "status": "READY",
            "power_prefix_sha256": power_sha,
            "selected_final_blocks": 12,
            "model": "Qwen/Qwen3-8B",
            "frozen_tts_recipe_sha256": _sha("tts"),
            "dflash_lightcone_recipe_sha256": _sha("dflash"),
            "dspark_lightcone_recipe_sha256": _sha("dspark"),
            "p99_anchors": anchors,
        },
    )
    final = downstream.materialize_single_operator_e5_final(power, lock)
    headline = [
        cell for cell in final.cells if cell.task == "production_slo_power_prefix"
    ]
    failures = [
        cell for cell in final.cells if cell.task == "deterministic_failure_injection"
    ]
    assert len(headline) == 450 * 12
    assert len(failures) == 264
    assert final.expected_cell_count == 450 * 12 + 264
    assert {dict(cell.dimensions)["block"] for cell in headline} == set(range(4, 16))


def _e5_p99_anchor() -> downstream.E5SelectedP99Anchor:
    return downstream.E5SelectedP99Anchor(
        backend="DFLASH",
        topology="tp1_dp1",
        family_id="closed_loop_c1",
        minimum_completions=10_000,
    )


def _e5_p99_observation(
    *,
    block: int,
    completed: int,
    incomplete: int = 0,
    itl_ns: int,
) -> dict[str, object]:
    rows = []
    for index in range(completed):
        started = index * 10_000_000
        rows.append(
            {
                "request_id": f"request-{block:02d}-{index:05d}",
                "input_token_ids": [1],
                "output_token_ids": [2, 3],
                "request_started_ns": started,
                "request_terminal_ns": started + itl_ns + 2,
                "token_observed_ns": [started + 1, started + itl_ns + 1],
                "terminal_status": "completed",
                "terminal_reason": "FINISH_LENGTH",
                "submitted_to_server": True,
            }
        )
    for offset in range(incomplete):
        index = completed + offset
        started = index * 10_000_000
        rows.append(
            {
                "request_id": f"request-{block:02d}-{index:05d}",
                "input_token_ids": [1],
                "output_token_ids": [],
                "request_started_ns": started,
                "request_terminal_ns": started + 1,
                "token_observed_ns": [],
                "terminal_status": "cancelled",
                "terminal_reason": "client_cancelled",
                "submitted_to_server": True,
            }
        )
    return {
        "requests": rows,
        "performance_counters": {
            "communicator_failures": 0,
            "exactness_violations": 0,
            "fallbacks": 0,
            "nonfinite_updates": 0,
            "oom_events": 0,
            "retractions": 0,
            "version_mismatches": 0,
            "updates_launched": 1,
            "updates_published": 1,
        },
    }


def _e5_p99_block_observations(
    anchor: downstream.E5SelectedP99Anchor,
    *,
    selection_sha256: str,
    counts: dict[int, tuple[int, int, int]],
) -> dict[int, tuple[downstream.MaterializedCell, dict[str, object]]]:
    result = {}
    for block, (completed, incomplete, itl_ns) in counts.items():
        cell = downstream._cell(
            stage="E5",
            method_role="LightCone",
            model="Qwen/Qwen3-8B",
            backend=anchor.backend,
            task="production_slo_power_prefix",
            publication_policy="first_ready",
            recipe_sha256=_sha("lightcone"),
            dimensions={
                "backend_authority": anchor.backend,
                "block": block,
                "family": "closed_loop",
                "family_id": anchor.family_id,
                "topology": anchor.topology,
                "p99_anchor_id": anchor.anchor_id,
                "p99_minimum_completions": anchor.minimum_completions,
                "p99_selection_receipt_sha256": selection_sha256,
            },
        )
        result[block] = (
            cell,
            _e5_p99_observation(
                block=block,
                completed=completed,
                incomplete=incomplete,
                itl_ns=itl_ns,
            ),
        )
    return result


def test_e5_p99_anchor_uses_registered_time_block_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(downstream, "_E5_P99_BOOTSTRAP_REPETITIONS", 100)
    anchor = _e5_p99_anchor()
    selection = _sha("e5-p99-selection")
    blocks = _e5_p99_block_observations(
        anchor,
        selection_sha256=selection,
        counts={
            4: (10_000, 0, 1_000_000),
            5: (10_000, 0, 2_000_000),
        },
    )

    row = downstream._e5_p99_anchor_row(
        anchor,
        blocks,
        expected_blocks={4, 5},
        selection_receipt_sha256=selection,
    )

    assert row["status"] == "CLAIMABLE"
    assert row["block_count"] == row["independent_block_count"] == 2
    assert row["request_count"] == row["offered_request_count"] == 20_000
    assert row["completed_request_count"] == 20_000
    assert row["paired"] is False
    assert row["confidence"] == 0.95
    assert row["bootstrap_repetitions"] == 100
    assert row["independent_units"] == ["time_block"]
    assert row["reducer_method"] == (
        "registered_time_block_bootstrap_native_itl_linear_p99"
    )
    assert row["ci_low"] < row["ci_high"]
    assert row["ci_low"] <= row["point_estimate"] <= row["ci_high"]
    assert row["observed_p99_ms"] == row["point_estimate"]
    assert all(
        block["minimum_completion_gate"] == "PASS" for block in row["block_evidence"]
    )


def test_e5_p99_anchor_rejects_pooled_completion_substitution() -> None:
    anchor = _e5_p99_anchor()
    selection = _sha("e5-p99-selection")
    blocks = _e5_p99_block_observations(
        anchor,
        selection_sha256=selection,
        counts={
            4: (9_999, 1, 1_000_000),
            5: (10_001, 0, 2_000_000),
        },
    )

    row = downstream._e5_p99_anchor_row(
        anchor,
        blocks,
        expected_blocks={4, 5},
        selection_receipt_sha256=selection,
    )

    assert row["completed_request_count"] == 20_000
    assert row["offered_request_count"] == 20_001
    assert row["status"] == "UNRESOLVED"
    assert row["failed_minimum_blocks"] == [4]
    assert row["terminal_status_counts"] == {
        "cancelled": 1,
        "completed": 20_000,
    }
    assert "per_block_minimum_completions_not_met" in row["reason_codes"]
    assert "point_estimate" not in row
    assert "ci_low" not in row
    assert "ci_high" not in row
    assert "reducer_method" not in row


def test_e5_p99_anchor_excludes_unsafe_final_block_without_ci(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(downstream, "_E5_P99_BOOTSTRAP_REPETITIONS", 100)
    anchor = _e5_p99_anchor()
    selection = _sha("e5-p99-selection")
    blocks = _e5_p99_block_observations(
        anchor,
        selection_sha256=selection,
        counts={
            4: (10_000, 0, 1_000_000),
            5: (10_000, 0, 2_000_000),
        },
    )
    unsafe_cell, unsafe_observation = blocks[4]
    unsafe_observation["performance_counters"]["nonfinite_updates"] = 1

    row = downstream._e5_p99_anchor_row(
        anchor,
        blocks,
        expected_blocks={4, 5},
        selection_receipt_sha256=selection,
    )

    assert row["status"] == "EXCLUDED_UNSAFE_OR_INACTIVE"
    assert row["reason_codes"] == ["LightCone:nonfinite_updates"]
    assert row["excluded_roles"] == ["LightCone"]
    assert row["evidence_cell_ids"] == [unsafe_cell.cell_id]
    assert row["observed_p99_ms"] is None
    assert "point_estimate" not in row
    assert "ci_low" not in row
    assert "ci_high" not in row
    assert "reducer_method" not in row


def test_e5_p99_unsafe_block_does_not_mask_later_malformed_safety_evidence() -> None:
    anchor = _e5_p99_anchor()
    selection = _sha("e5-p99-selection")
    blocks = _e5_p99_block_observations(
        anchor,
        selection_sha256=selection,
        counts={
            4: (1, 0, 1_000_000),
            5: (1, 0, 2_000_000),
        },
    )
    blocks[4][1]["performance_counters"]["nonfinite_updates"] = 1
    del blocks[5][1]["performance_counters"]["oom_events"]

    with pytest.raises(ValueError, match="performance counter oom_events is missing"):
        downstream._e5_p99_anchor_row(
            anchor,
            blocks,
            expected_blocks={4, 5},
            selection_receipt_sha256=selection,
        )


def test_e5_p99_anchor_rejects_tampered_cell_binding() -> None:
    anchor = _e5_p99_anchor()
    selection = _sha("e5-p99-selection")
    blocks = _e5_p99_block_observations(
        anchor,
        selection_sha256=selection,
        counts={
            4: (0, 0, 1_000_000),
            5: (0, 0, 2_000_000),
        },
    )
    cell, observation = blocks[4]
    dimensions = dict(cell.dimensions)
    dimensions["p99_minimum_completions"] = 9_999
    blocks[4] = (
        downstream.MaterializedCell(
            stage=cell.stage,
            method_role=cell.method_role,
            model=cell.model,
            backend=cell.backend,
            task=cell.task,
            publication_policy=cell.publication_policy,
            recipe_sha256=cell.recipe_sha256,
            dimensions=tuple(sorted(dimensions.items())),
        ),
        observation,
    )

    with pytest.raises(ValueError, match="cell binding differs"):
        downstream._e5_p99_anchor_row(
            anchor,
            blocks,
            expected_blocks={4, 5},
            selection_receipt_sha256=selection,
        )


def test_profiler_reducer_keeps_descriptive_evidence_out_of_headline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_sha = _sha("local-materialization")
    materialization = _materialize_e4_profiler_diagnostic(
        protocol_lock_sha256=_sha("lock"),
        upstream_local_receipt_sha256=local_sha,
        source_decision_sha256=_sha("selection"),
        selected_configuration_sha256=_sha("configuration"),
        model="Qwen/Qwen3-8B",
        lightcone_recipe_sha256=_sha("lightcone"),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    predecessor = SimpleNamespace(
        artifact=SimpleNamespace(node="e4_local"),
        decision=SimpleNamespace(
            payload={
                "model": "Qwen/Qwen3-8B",
                "selection_sha256": _sha("selection"),
            }
        ),
    )
    monkeypatch.setattr(
        downstream,
        "_frozen_recipes",
        lambda _predecessor: (_sha("tts"), _sha("lightcone")),
    )
    actuals = tuple(
        SimpleNamespace(
            status="COMPLETE",
            result_identity_sha256=_sha(f"terminal:{cell.cell_id}"),
            reducer_payload={
                "profiler_terminal": {
                    "raw_profile_sha256": _sha(f"raw:{cell.cell_id}"),
                    "raw_profile_size_bytes": 1_024,
                }
            },
        )
        for cell in materialization.cells
    )
    decision = downstream.reduce_single_operator_e4_profiler(
        predecessor,
        materialization,
        actuals,
    )
    assert decision.decision_kind == "e4_profiler_actual_3_reduced"
    assert len(decision.payload["profiler_rows"]) == 3
    assert all(
        row["headline_eligible"] is False for row in decision.payload["profiler_rows"]
    )


def test_request_evidence_preserves_noncomplete_denominator() -> None:
    observation = {
        "requests": [
            {
                "request_id": "request-0",
                "input_token_ids": [1],
                "output_token_ids": [],
                "request_started_ns": 1,
                "request_terminal_ns": 2,
                "token_observed_ns": [],
                "terminal_status": "cancelled",
                "terminal_reason": "client_cancelled",
                "submitted_to_server": True,
            }
        ]
    }
    rows = downstream._request_evidence(observation)
    assert len(rows) == 1
    assert rows[0].eligible is True
    assert rows[0].completed is False
    assert rows[0].error is False


def test_e0_tuning_is_exact_108_plus_239v_and_uses_canonical_role_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    e6, lock, compatibility, bundle = _e0_fixture(
        monkeypatch,
        tmp_path,
        valid_count=1,
    )
    materialization = downstream.materialize_single_operator_e0_tuning(
        e6,
        lock,
        bundle,
    )
    compatibility_cells = [
        cell for cell in materialization.cells if cell.task == "compatibility_decision"
    ]
    tuning_cells = [
        cell
        for cell in materialization.cells
        if cell.task == "independent_onlinespec_tuning"
    ]
    assert compatibility.valid_count == 1
    assert materialization.expected_cell_count == 108 + 239
    assert len(compatibility_cells) == 108
    assert len(tuning_cells) == 239
    assert {cell.method_role for cell in compatibility_cells} == {"Compatibility"}
    online_roles = {
        cell.method_role
        for cell in tuning_cells
        if cell.method_role.startswith("OnlineSPEC-")
    }
    assert online_roles == {
        "OnlineSPEC-OGD",
        "OnlineSPEC-OPT",
        "OnlineSPEC-ENS",
    }
    assert "OnlineSPEC-Optimistic-OGD" not in online_roles
    assert "OnlineSPEC-Hedge" not in online_roles
    assert set(dict(compatibility_cells[0].dimensions)) == {
        "compatibility_decision_id",
        "deployment_task",
        "disposition",
        "reason_code",
        "interface_sha256",
        "task_native_workload_sha256",
        "compatibility_receipt_sha256",
        "compatibility_evidence_manifest_sha256",
        "e0_compatibility_bundle_sha256",
    }

    valid = next(row for row in compatibility.decisions if row.disposition == "VALID")
    selected = {
        (valid.decision_id, role): _sha(f"winner:{role}")
        for role in ("OnlineSPEC-OGD", "OnlineSPEC-OPT", "OnlineSPEC-ENS")
    }
    authority_sha256 = bundle["onlinespec_source_authority_sha256"]
    assert type(authority_sha256) is str
    pilots = downstream._e0_serving_cells(
        compatibility=compatibility,
        compatibility_bundle_sha256=bundle["bundle_sha256"],
        compatibility_evidence_sha256=bundle["compatibility_evidence_manifest_sha256"],
        onlinespec_source_authority_sha256=authority_sha256,
        selected_recipes=selected,
        frozen_tts_recipe_sha256=_sha("e0-tts"),
        lightcone_recipe_sha256=_sha("e0-lightcone"),
        block_indices=tuple(range(4)),
    )
    finals = downstream._e0_serving_cells(
        compatibility=compatibility,
        compatibility_bundle_sha256=bundle["bundle_sha256"],
        compatibility_evidence_sha256=bundle["compatibility_evidence_manifest_sha256"],
        onlinespec_source_authority_sha256=authority_sha256,
        selected_recipes=selected,
        frozen_tts_recipe_sha256=_sha("e0-tts"),
        lightcone_recipe_sha256=_sha("e0-lightcone"),
        block_indices=tuple(range(4, 16)),
        final_lineage={
            "pilot_materialization_receipt_sha256": _sha("pilot"),
            "power_prefix_sha256": _sha("power"),
        },
    )
    assert len(pilots) == 16 * 4
    assert len(finals) == 16 * 12


def test_e0_all_na_keeps_108_decisions_and_emits_no_fake_ci(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    e6, lock, _compatibility, bundle = _e0_fixture(
        monkeypatch,
        tmp_path,
        valid_count=0,
    )
    tuning = downstream.materialize_single_operator_e0_tuning(e6, lock, bundle)
    assert tuning.expected_cell_count == 108
    assert all(cell.method_role == "Compatibility" for cell in tuning.cells)
    actuals = []
    for cell in tuning.cells:
        dimensions = dict(cell.dimensions)
        actuals.append(
            SimpleNamespace(
                cell_id=cell.cell_id,
                result_identity_sha256=_sha(f"compatibility:{cell.cell_id}"),
                reducer_payload={
                    "e0_compatibility_decision": {
                        "schema_version": 1,
                        "compatibility_decision_id": dimensions[
                            "compatibility_decision_id"
                        ],
                        "disposition": dimensions["disposition"],
                        "reason_code": dimensions["reason_code"],
                        "interface_sha256": dimensions["interface_sha256"],
                        "task_native_workload_sha256": dimensions[
                            "task_native_workload_sha256"
                        ],
                        "compatibility_evidence_manifest_sha256": dimensions[
                            "compatibility_evidence_manifest_sha256"
                        ],
                    }
                },
            )
        )
    tuning_draft = downstream.reduce_single_operator_e0_tuning(
        e6,
        tuning,
        tuple(actuals),
    )
    assert tuning_draft.payload["status"] == "ALL_NA"
    assert tuning_draft.payload["valid_count"] == 0
    tuning_completion = _completion_from_draft(
        node="e0_tuning",
        materialization=tuning,
        draft=tuning_draft,
        predecessor=e6,
    )
    pilot = downstream.materialize_single_operator_e0_pilot(
        tuning_completion,
        lock,
        bundle,
    )
    assert pilot.expected_cell_count == 0
    pilot_draft = downstream.reduce_single_operator_e0_pilot(
        tuning_completion,
        pilot,
        (),
    )
    pilot_completion = _completion_from_draft(
        node="e0_pilot",
        materialization=pilot,
        draft=pilot_draft,
        predecessor=tuning_completion,
    )
    final = downstream.materialize_single_operator_e0_final(
        pilot_completion,
        lock,
        bundle,
    )
    assert final.expected_cell_count == 0
    final_draft = downstream.reduce_single_operator_e0_final(
        pilot_completion,
        final,
        (),
    )
    payload = final_draft.payload
    assert payload["status"] == "ALL_NA"
    assert len(payload["na_decisions"]) == 108
    assert len(payload["na_hypothesis_exclusions"]) == 108 * 7
    assert all(
        not any(key.startswith("ci_") or key == "point_estimate" for key in row)
        for row in payload["na_hypothesis_exclusions"]
    )
    assert [
        row["tested_hypothesis_count"] for row in payload["breadth_fdr_families"]
    ] == [0, 0]


def test_e0_negative_anchors_do_not_block_independent_onlinespec_ranking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lightcone_spec.experiments import formal_single_operator_stages as stages

    e6, lock, _compatibility, bundle = _e0_fixture(
        monkeypatch,
        tmp_path,
        valid_count=1,
    )
    tuning = downstream.materialize_single_operator_e0_tuning(e6, lock, bundle)
    serving_cells = tuple(
        cell for cell in tuning.cells if cell.task != "compatibility_decision"
    )
    assert len(serving_cells) == 239

    monkeypatch.setattr(
        downstream,
        "_e0_validate_compatibility_actuals",
        lambda *_args: (),
    )
    monkeypatch.setattr(
        downstream,
        "_serving_observations",
        lambda receipt, _actuals: {
            cell.cell_id: {
                "cell_id": cell.cell_id,
                "method_role": cell.method_role,
                "recipe_sha256": cell.recipe_sha256,
            }
            for cell in receipt.cells
        },
    )

    def fake_slo(observation: dict[str, object]) -> SimpleNamespace:
        role = str(observation["method_role"])
        anchor = role in {"Static", "TTS", "L0-naive"}
        cell_id = str(observation["cell_id"])
        recipe = observation["recipe_sha256"]
        return SimpleNamespace(
            status="FAIL" if anchor else "PASS",
            source_request_pool_sha256=_sha("paired-e0-pool"),
            sha256=_sha(f"slo:{cell_id}"),
            goodput_tokens_per_second=(
                1.0 if recipe is None else 1.0 + int(str(recipe)[:8], 16) / 2**32
            ),
            eligible_requests=16,
        )

    monkeypatch.setattr(downstream, "_slo", fake_slo)
    monkeypatch.setattr(
        stages, "_adaptive_safety_reasons", lambda *_args, **_kwargs: ()
    )
    actuals = tuple(
        SimpleNamespace(
            cell_id=cell.cell_id,
            result_identity_sha256=_sha(f"actual:{cell.cell_id}"),
            reducer_payload={},
        )
        for cell in tuning.cells
    )

    draft = downstream.reduce_single_operator_e0_tuning(e6, tuning, actuals)

    assert draft.payload["status"] == "READY"
    assert len(draft.payload["selected_onlinespec_recipes"]) == 3
    assert {
        row["method_role"] for row in draft.payload["selected_onlinespec_recipes"]
    } == {
        "OnlineSPEC-OGD",
        "OnlineSPEC-OPT",
        "OnlineSPEC-ENS",
    }
    assert all(
        row["eligible"] is False and row["reason_codes"] == ["slo_failed"]
        for row in draft.payload["anchor_evaluations"]
    )
    assert all(row["eligible"] for row in draft.payload["candidate_evaluations"])


def test_e0_common_slo_breadth_uses_two_separate_bh_families() -> None:
    contrasts = {
        name: {"raw_p_value": (index + 1) / 100}
        for index, name in enumerate(
            downstream._E0_CORE_BREADTH_CONTRASTS
            + downstream._E0_ONLINE_BREADTH_CONTRASTS
        )
    }
    families = downstream._e0_breadth_fdr(
        [
            {
                "compatibility_decision_id": _sha("breadth-decision"),
                "contrasts": contrasts,
            }
        ]
    )
    assert [row["family_id"] for row in families] == [
        "e0_common_slo_core",
        "e0_common_slo_onlinespec",
    ]
    assert [row["tested_hypothesis_count"] for row in families] == [4, 3]
    assert all(
        decision["procedure"] == "benjamini-hochberg"
        for row in families
        for decision in row["decisions"]
    )


def test_e0_breadth_fdr_rejects_unknown_contrast_status() -> None:
    contrasts = {
        name: {"status": "BOGUS", "reason_codes": ["not_registered"]}
        for name in (
            downstream._E0_CORE_BREADTH_CONTRASTS
            + downstream._E0_ONLINE_BREADTH_CONTRASTS
        )
    }

    with pytest.raises(ValueError, match="contrast status differs"):
        downstream._e0_breadth_fdr(
            [
                {
                    "compatibility_decision_id": _sha("breadth-decision"),
                    "contrasts": contrasts,
                }
            ]
        )

    for contrast in contrasts.values():
        contrast.update(
            {
                "status": "UNRESOLVED_ZERO_VARIANCE",
                "reason_codes": ["UNRESOLVED_ZERO_GOODPUT"],
            }
        )
    with pytest.raises(ValueError, match="reason/status differ"):
        downstream._e0_breadth_fdr(
            [
                {
                    "compatibility_decision_id": _sha("breadth-decision"),
                    "contrasts": contrasts,
                }
            ]
        )


def test_e0_breadth_exclusion_does_not_erase_other_valid_hypotheses() -> None:
    first = {
        name: {"status": "RESOLVED", "raw_p_value": (index + 1) / 100}
        for index, name in enumerate(
            downstream._E0_CORE_BREADTH_CONTRASTS
            + downstream._E0_ONLINE_BREADTH_CONTRASTS
        )
    }
    second = {
        name: {"status": "RESOLVED", "raw_p_value": (index + 2) / 100}
        for index, name in enumerate(
            downstream._E0_CORE_BREADTH_CONTRASTS
            + downstream._E0_ONLINE_BREADTH_CONTRASTS
        )
    }
    second["lightcone_vs_tts"] = {
        "status": "EXCLUDED_UNSAFE_OR_INACTIVE",
        "reason_codes": ["LightCone:oom_events"],
        "excluded_roles": ["LightCone"],
        "evidence_cell_ids": [_sha("unsafe-lightcone")],
    }

    families = downstream._e0_breadth_fdr(
        [
            {
                "compatibility_decision_id": _sha("decision-a"),
                "contrasts": first,
            },
            {
                "compatibility_decision_id": _sha("decision-b"),
                "contrasts": second,
            },
        ]
    )

    core, online = families
    assert core["status"] == "PARTIALLY_RESOLVED"
    assert core["tested_hypothesis_count"] == 7
    assert len(core["decisions"]) == 7
    assert len(core["unresolved_hypotheses"]) == 1
    assert online["status"] == "RESOLVED"
    assert online["tested_hypothesis_count"] == 6
    assert len(online["decisions"]) == 6
