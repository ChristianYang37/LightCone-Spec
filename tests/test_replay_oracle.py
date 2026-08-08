from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightcone_spec.exit_codes import ArtifactValidationFailure
from lightcone_spec.locking.hashing import sha256_file
from lightcone_spec.methods.registry import controller_runtime_identity
from lightcone_spec.replay.real import (
    _controller_runtime_identity,
    _l3_transport_gate,
    _learned_policy_gate,
    _oracle_replay_gate,
    _tts_paired_gate,
    load_real_replay_records,
)
from lightcone_spec.replay.splits import split_of_group
from lightcone_spec.replay.pipeline import _calibration_constant_gate_delays
from lightcone_spec.controller.gate import select_gate_threshold
from lightcone_spec.controller.damping import select_utility_calibrated_radius
from lightcone_spec.trajectory.features import UpdateFeatureRow


def _test_groups(count: int) -> list[str]:
    groups = []
    candidate = 0
    while len(groups) < count:
        group = f"oracle-request-{candidate}"
        if split_of_group(group) == "test":
            groups.append(group)
        candidate += 1
    return groups


def _test_groups_for_seed(count: int, seed: int) -> list[str]:
    groups = []
    candidate = 0
    while len(groups) < count:
        group = f"seed-{seed}-request-{candidate}"
        if split_of_group(group, seed) == "test":
            groups.append(group)
        candidate += 1
    return groups


def _one_group_per_split() -> dict[str, str]:
    groups: dict[str, str] = {}
    candidate = 0
    while len(groups) < 3:
        group = f"schema-request-{candidate}"
        groups.setdefault(split_of_group(group), group)
        candidate += 1
    return groups


def test_constant_apply_hint_uses_only_uniformly_beneficial_calibration_bucket():
    delays = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int64)
    features = np.asarray(
        [[0.0], [1.0], [2.0], [2.0], [3.0], [4.0]], dtype=np.float64
    )
    # Delay 1 is classified apply but its predictor inputs vary, so bypassing
    # the predictor would change semantics. Delay 2 has one exact feature row
    # but contains a harmful utility. Delay 3 is classified discard but also
    # has varying features. None is a valid delay-only fast path.
    utilities = np.asarray([0.2, 0.1, 0.2, -0.1, -0.2, -0.3])
    harm = np.asarray([0.1, 0.2, 0.1, 0.2, 0.9, 0.8])

    constant_features, discards, applies = _calibration_constant_gate_delays(
        delays,
        features,
        utilities,
        harm,
        threshold=0.5,
        discard_all=False,
    )

    assert applies == []
    assert discards == []
    assert set(constant_features) == {2}


def test_constant_gate_hint_requires_same_predictor_input():
    delays = np.asarray([1, 1, 2, 2], dtype=np.int64)
    features = np.asarray([[1.0], [1.0], [2.0], [2.0]], dtype=np.float64)
    utilities = np.asarray([0.2, 0.1, -0.2, -0.3])
    harm = np.asarray([0.1, 0.2, 0.9, 0.8])

    _, discards, applies = _calibration_constant_gate_delays(
        delays,
        features,
        utilities,
        harm,
        threshold=0.5,
        discard_all=False,
    )

    assert applies == [1]
    assert discards == [2]


def test_gate_calibration_prefers_zero_utility_discard_all_to_harm():
    selection = select_gate_threshold(
        harm_probs=np.asarray([0.1, 0.2, 0.9]),
        utilities=np.asarray([-0.4, -0.2, -0.1]),
        unsafe_apply_limit=1.0,
    )

    assert selection.discard_all is True
    assert selection.apply_fraction == 0.0
    assert selection.mean_utility == 0.0


def test_l2_radius_is_selected_from_calibration_kappa_utility():
    grids = [
        {0.0: 0.0, 0.5: 1.0, 1.0: -1.0},
        {0.0: 0.0, 0.5: 0.8, 1.0: -0.8},
    ]

    radius, evidence = select_utility_calibrated_radius(
        np.asarray([1.0, 1.0]), grids
    )

    assert np.exp(-1.0 / radius) == pytest.approx(0.5)
    assert evidence["contract"] == "calibration_same_candidate_kappa_utility_v1"
    assert evidence["mean_calibration_utility"] == pytest.approx(0.9)


def _record(group: str, full: float, l1: float, l2: float):
    return SimpleNamespace(
        row=SimpleNamespace(sequence_id=group),
        full_candidate_utility=full,
        oracle_l1_utility=l1,
        oracle_l2_utility=l2,
        actual_published_utility=full,
        provenance_method="tts",
        candidate_arrival_round=5,
        actual_arrival_round=8,
        paired_tts_barrier=True,
        prefix_feature_exact=True,
    )


def test_oracle_gate_is_a_same_arrival_ceiling_not_a_tts_claim():
    records = [
        _record(group, -0.4, 0.0, 0.2)
        for group in _test_groups(8)
    ]

    gate = _oracle_replay_gate(records, bootstrap_b=200)

    assert gate["complete"] is True
    assert gate["reference"] == "same_arrival_full_candidate_l0"
    assert gate["tts_barrier_comparison_required"] is True
    assert gate["l1_eligible"] is True
    assert gate["l2_eligible"] is True


def test_oracle_gate_fails_closed_when_kappa_trace_is_missing():
    record = _record(_test_groups(1)[0], -0.4, 0.0, 0.2)
    record.oracle_l2_utility = None

    gate = _oracle_replay_gate([record], bootstrap_b=10)

    assert gate["complete"] is False
    assert gate["l1_eligible"] is False
    assert gate["l2_eligible"] is False


def test_oracle_gate_never_claims_from_one_prompt_cluster():
    record = _record(_test_groups(1)[0], -0.4, 0.0, 0.2)

    gate = _oracle_replay_gate(
        [record], bootstrap_b=10, min_test_groups=1
    )

    assert gate["complete"] is False
    assert gate["minimum_test_groups"] == 2


def test_tts_paired_gate_uses_actual_barrier_utility_with_cluster_bca():
    records = [
        _record(group, -0.4, 0.0, 0.2)
        for group in _test_groups(8)
    ]

    gate = _tts_paired_gate(records, bootstrap_b=200)

    assert gate["complete"] is True
    assert gate["reference"] == "same_candidate_actual_tts_barrier"
    assert gate["horizon_alignment"] == "independent_H_from_each_arrival"
    assert gate["l0_gain_vs_tts"] == pytest.approx(0.0)
    assert gate["l0_eligible"] is False
    assert gate["l1_eligible"] is True
    assert gate["l2_eligible"] is True


def test_tts_paired_gate_reports_same_candidate_l0_gain():
    records = [_record(group, 0.3, 0.4, 0.5) for group in _test_groups(8)]
    for record in records:
        record.actual_published_utility = -0.2

    gate = _tts_paired_gate(records, bootstrap_b=200)

    assert gate["complete"] is True
    assert gate["l0_gain_vs_tts"] == pytest.approx(0.5)
    assert gate["l0_ci95"][0] > 0.0
    assert gate["l0_eligible"] is True


def test_tts_paired_gate_uses_the_fitted_split_seed():
    seed = 17
    records = [
        _record(group, 0.3, 0.4, 0.5)
        for group in _test_groups_for_seed(8, seed)
    ]
    for record in records:
        record.actual_published_utility = -0.2

    gate = _tts_paired_gate(records, bootstrap_b=100, seed=seed)

    assert gate["complete"] is True
    assert gate["n_test_groups"] == 8
    assert gate["l0_eligible"] is True


def test_tts_paired_gate_never_claims_from_one_prompt_cluster():
    record = _record(_test_groups(1)[0], 0.3, 0.4, 0.5)
    record.actual_published_utility = -0.2

    gate = _tts_paired_gate(
        [record], bootstrap_b=20, min_test_groups=1
    )

    assert gate["complete"] is False
    assert gate["n_test_groups"] == 1
    assert gate["minimum_test_groups"] == 2
    assert gate["l0_eligible"] is False


def test_tts_paired_gate_fails_closed_on_any_incomplete_pair():
    records = [
        _record(group, -0.4, 0.0, 0.2)
        for group in _test_groups(8)
    ]

    gate = _tts_paired_gate(records, bootstrap_b=20, incomplete_pairs=1)

    assert gate["complete"] is False
    assert gate["incomplete_pairs"] == 1
    assert gate["l1_eligible"] is False
    assert gate["l2_eligible"] is False


@pytest.mark.parametrize("value", [None, float("nan")])
def test_tts_paired_gate_rejects_missing_or_nonfinite_l0_utility(value):
    records = [_record(group, 0.3, 0.4, 0.5) for group in _test_groups(8)]
    records[0].full_candidate_utility = value

    gate = _tts_paired_gate(records, bootstrap_b=20)

    assert gate["complete"] is False
    assert gate["l0_eligible"] is False


def test_tts_paired_gate_ignores_valid_non_tts_training_labels():
    records = [
        _record(group, -0.4, 0.0, 0.2)
        for group in _test_groups(8)
    ]
    naive = _record(_test_groups(9)[-1], 0.1, 0.1, 0.1)
    naive.provenance_method = "naive_async"
    naive.paired_tts_barrier = False
    naive.candidate_arrival_round = naive.actual_arrival_round = 5
    records.append(naive)

    gate = _tts_paired_gate(records, bootstrap_b=100)

    assert gate["complete"] is True
    assert gate["n_test"] == 8


def _l3_gate_records(*, l2_utility: float = 0.5):
    records = []
    map_sha = "a" * 64
    for index, group in enumerate(_test_groups(8)):
        common = {
            "row": SimpleNamespace(
                sequence_id=group,
                source_prefix_len=4096 + index,
            ),
            "delta_g": torch.tensor([1.0 + index]).numpy(),
            "delta_z": torch.tensor([0.0]).numpy(),
            "source_round": 4,
            "candidate_arrival_round": 5,
            "evaluation_pair_id": f"pair-{index}",
            "trace_stage_index": 0,
            "trace_stage_count": 1,
            "trace_capture_sampling": "first",
            "evaluation_concurrency": 4,
        }
        records.append(
            SimpleNamespace(
                **common,
                provenance_method="lc_transport",
                trace_owner_role="phase2_l3",
                trace_evaluation_only=True,
                trace_prompt_offset=136,
                trace_prompt_limit=48,
                transported_candidate_utility=1.0,
                paired_l2_utility=l2_utility,
                actual_published_utility=1.0,
                transport_evaluation_contract=(
                    "joint_fisher_transport_adamw_damping_v1"
                ),
                transport_variant="joint",
                transport_map_sha256=map_sha,
            )
        )
        records.append(
            SimpleNamespace(
                **common,
                provenance_method="tts",
                trace_owner_role="phase2_tts_reference",
                trace_evaluation_only=True,
                trace_prompt_offset=136,
                trace_prompt_limit=48,
                transported_candidate_utility=None,
                paired_l2_utility=None,
                actual_published_utility=0.25,
                transport_evaluation_contract=None,
                transport_variant=None,
                transport_map_sha256=None,
            )
        )
    return records, map_sha


def test_l3_gate_uses_paired_survival_utility_against_tts_and_l2():
    records, map_sha = _l3_gate_records()
    # Staged capture is paired by explicit stage ordinal, not proposal round;
    # different acceptance trajectories need not cross the stage on one round.
    for record in records:
        if record.provenance_method == "tts":
            record.source_round += 2
            record.candidate_arrival_round += 2
    transport = SimpleNamespace(
        state_correction=lambda delta_z: torch.zeros_like(
            torch.as_tensor(delta_z)
        ).numpy()
    )

    gate = _l3_transport_gate(
        records,
        transport,
        expected_transport_map_sha256=map_sha,
        bootstrap_b=200,
    )

    utility = gate["heldout_transported_utility_gate"]
    assert gate["enabled"] is True
    assert utility["complete"] is True
    assert utility["eligible"] is True
    assert utility["gain_vs_tts"] == pytest.approx(0.75)
    assert utility["gain_vs_l2"] == pytest.approx(0.5)
    assert utility["ci95_vs_tts"][0] > 0.0
    assert utility["ci95_vs_l2"][0] > 0.0
    assert utility["pairing_contract"] == (
        "exact_request_seed_concurrency_trace_stage_v1"
    )
    assert utility["max_abs_source_round_difference"] == 2
    assert utility["max_abs_candidate_arrival_round_difference"] == 2
    assert gate["transport_fit_diagnostic"]["used_for_enable"] is False


def test_l3_gate_keeps_same_prompt_c1_c4_as_distinct_pairs():
    records, map_sha = _l3_gate_records()
    expanded = []
    for record in records:
        expanded.append(record)
        expanded.append(
            SimpleNamespace(
                **{
                    **vars(record),
                    "evaluation_concurrency": 1,
                }
            )
        )
    transport = SimpleNamespace(
        state_correction=lambda delta_z: torch.zeros_like(
            torch.as_tensor(delta_z)
        ).numpy()
    )

    gate = _l3_transport_gate(
        expanded,
        transport,
        expected_transport_map_sha256=map_sha,
        bootstrap_b=100,
    )

    utility = gate["heldout_transported_utility_gate"]
    assert gate["enabled"] is True
    assert utility["complete"] is True
    assert utility["n_pairs"] == 16


def test_l3_gate_does_not_pair_across_concurrency():
    records, map_sha = _l3_gate_records()
    for record in records:
        if record.provenance_method == "tts":
            record.evaluation_concurrency = 1
    transport = SimpleNamespace(
        state_correction=lambda delta_z: torch.zeros_like(
            torch.as_tensor(delta_z)
        ).numpy()
    )

    gate = _l3_transport_gate(
        records,
        transport,
        expected_transport_map_sha256=map_sha,
        bootstrap_b=100,
    )

    utility = gate["heldout_transported_utility_gate"]
    assert gate["enabled"] is False
    assert utility["complete"] is False
    assert utility["missing_tts_pairs"] == 8
    assert utility["extra_tts_pairs"] == 8


def test_l3_gate_rejects_phase1_tts_as_heldout_reference():
    records, map_sha = _l3_gate_records()
    for record in records:
        if record.provenance_method == "tts":
            record.trace_owner_role = "phase1_producer"
            record.trace_evaluation_only = False
    transport = SimpleNamespace(state_correction=lambda delta_z: delta_z * 0.0)

    gate = _l3_transport_gate(
        records,
        transport,
        expected_transport_map_sha256=map_sha,
        bootstrap_b=100,
    )

    assert gate["enabled"] is False
    assert "identical held-out window" in gate["disabled_reason"]


def test_l3_gate_requires_one_identical_explicit_prompt_window():
    records, map_sha = _l3_gate_records()
    next(
        record for record in records if record.provenance_method == "tts"
    ).trace_prompt_offset = 137
    transport = SimpleNamespace(state_correction=lambda delta_z: delta_z * 0.0)

    gate = _l3_transport_gate(
        records,
        transport,
        expected_transport_map_sha256=map_sha,
        bootstrap_b=100,
    )

    assert gate["enabled"] is False
    assert "identical held-out window" in gate["disabled_reason"]


def test_l3_gate_requires_both_positive_paired_intervals():
    records, map_sha = _l3_gate_records(l2_utility=1.25)
    transport = SimpleNamespace(state_correction=lambda delta_z: delta_z * 0.0)

    gate = _l3_transport_gate(
        records,
        transport,
        expected_transport_map_sha256=map_sha,
        bootstrap_b=100,
    )

    assert gate["enabled"] is False
    assert gate["evidence_insufficient"] is False
    assert gate["heldout_transported_utility_gate"]["ci95_vs_tts"][0] > 0
    assert gate["heldout_transported_utility_gate"]["ci95_vs_l2"][1] < 0


def test_learned_policy_gate_uses_heldout_decisions_not_oracle_choices():
    records = []
    for index, group in enumerate(_test_groups(8)):
        row = UpdateFeatureRow(
            sequence_id=group,
            update_id=f"u-{index}",
            round_delay=5,
            token_delay=20,
            wall_us=100,
            endpoint_distance=0.1,
            rho_path=0.2,
            parameter_displacement=0.3,
            utility=-0.4,
            relative_gradient_mismatch=1.0,
            harmful=1,
            source_prefix_len=4096 + index,
        )
        records.append(
            SimpleNamespace(
                row=row,
                actual_published_utility=-0.5,
                provenance_method="tts",
                paired_tts_barrier=True,
                prefix_feature_exact=True,
                utility_by_kappa={0.0: 0.0, 0.5: 0.3, 1.0: -0.4},
            )
        )

    class ConstantHarm:
        def probability(self, x):
            return __import__("numpy").ones(len(x))

    class ConstantMismatch:
        def predict(self, x):
            return __import__("numpy").ones(len(x))

    artifact = SimpleNamespace(
        feature_set="path_length",
        harmful_classifier=ConstantHarm(),
        mismatch_predictor=ConstantMismatch(),
        gate_threshold=0.5,
        gate_discard_all=False,
        damping_radius=1.0,
        damping_kernel="exponential",
        extra={
            "gate_constant_discard_delays": [],
            "constant_controller_profiles": {
                "5": {"damping_factor": 0.5},
            },
        },
    )

    gate = _learned_policy_gate(records, artifact, bootstrap_b=200)

    assert gate["complete"] is True
    assert gate["reference"] == (
        "learned_policy_same_candidate_actual_tts_barrier"
    )
    assert gate["l1_apply_fraction"] == 0.0
    assert gate["l2_mean_kappa"] == 0.5
    assert gate["l1_eligible"] is True
    assert gate["l2_eligible"] is True


def test_learned_policy_gate_evaluates_constant_apply_on_heldout_groups():
    records = []
    for index, group in enumerate(_test_groups(8)):
        records.append(
            SimpleNamespace(
                row=UpdateFeatureRow(
                    sequence_id=group,
                    update_id=f"fast-{index}",
                    round_delay=5,
                    token_delay=20,
                    wall_us=100,
                    endpoint_distance=0.1,
                    rho_path=0.2,
                    parameter_displacement=0.3,
                    utility=0.1,
                    relative_gradient_mismatch=0.0,
                    harmful=0,
                    source_prefix_len=4096 + index,
                ),
                actual_published_utility=-0.2,
                provenance_method="tts",
                paired_tts_barrier=True,
                prefix_feature_exact=True,
                utility_by_kappa={0.0: 0.0, 1.0: 0.1},
            )
        )

    artifact = SimpleNamespace(
        feature_set="path_length",
        harmful_classifier=SimpleNamespace(
            probability=lambda x: np.ones(len(x))
        ),
        mismatch_predictor=SimpleNamespace(
            predict=lambda x: np.zeros(len(x))
        ),
        gate_threshold=0.5,
        gate_discard_all=False,
        damping_radius=1.0,
        damping_kernel="exponential",
        extra={
            "gate_constant_discard_delays": [],
            "gate_constant_apply_delays": [5],
            "constant_controller_profiles": {"5": {"damping_factor": 1.0}},
        },
    )

    gate = _learned_policy_gate(records, artifact, bootstrap_b=200)

    assert gate["complete"] is True
    assert gate["l1_apply_fraction"] == 1.0
    assert gate["l1_constant_apply_fastpath_fraction"] == 1.0
    assert gate["l1_predictor_path_fraction"] == 0.0
    assert gate["l1_eligible"] is True
    assert gate["l2_unit_kappa_fastpath_fraction"] == 1.0


def test_learned_policy_gate_fails_closed_without_captured_kappa_grid():
    records = [_record(group, -0.4, 0.0, 0.2) for group in _test_groups(8)]
    artifact = SimpleNamespace()

    gate = _learned_policy_gate(records, artifact, bootstrap_b=10)

    assert gate["complete"] is False
    assert gate["l1_eligible"] is False
    assert gate["l2_eligible"] is False


def _write_real_payload(root, payload: dict) -> None:
    root.mkdir(parents=True)
    path = root / "p1-u.pt"
    torch.save(payload, path)
    (root / "index-p1.jsonl").write_text(
        json.dumps(
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "parameter_layout_sha256": "1" * 64,
            }
        )
        + "\n"
    )


def _real_payload(**overrides) -> dict:
    payload = {
        "schema_version": 2,
        "provenance_method": "lc_damp",
        "controller_label_source": "full_candidate_utility",
        "utility_metric": "survival_weighted_accepted_prefix_v1",
        "sequence_id": "request-0",
        "update_id": "u-0",
        "source_round": 1,
        "arrival_round": 2,
        "candidate_arrival_round": 2,
        "actual_arrival_round": 2,
        "paired_tts_barrier": False,
        "prefix_feature_exact": True,
        "evaluation_pair_id": "pair-request-0",
        "trace_stage_index": 0,
        "trace_stage_count": 1,
        "trace_capture_sampling": "first",
        "evaluation_concurrency": 1,
        "fresh_gradient_scope": "writer_rank_local_v1",
        "round_delay": 1.0,
        "token_delay": 2.0,
        "wall_us": 3.0,
        "endpoint_distance": 0.1,
        "rho_path": 0.2,
        "parameter_displacement": 0.3,
        "source_prefix_len": 4096.0,
        "source_acceptance": 1.5,
        "source_training_loss": 0.75,
        "source_grad_norm": 0.25,
        "actual_published_utility": 1.25,
        "full_candidate_utility": -0.75,
        # Deliberately wrong: the loader must derive harmful from the full raw
        # candidate rather than trusting a policy-dependent stored flag.
        "harmful": 0,
        "relative_gradient_mismatch": 0.4,
        "cosine": 0.5,
        "utility_by_kappa": {"0.0": 0.0, "0.5": 0.25, "1.0": -0.75},
        "delta_g": torch.tensor([0.1, 0.2]),
        "delta_z": torch.tensor([0.3, 0.4]),
        "source_z_raw": torch.tensor([0.1, 0.2]),
        "arrival_z_raw": torch.tensor([0.4, 0.6]),
    }
    payload.update(overrides)
    return payload


def _write_runtime_config(root, pair_id: str) -> None:
    root.mkdir(parents=True)
    (root / "adaptation.runtime.yaml").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "naive_async",
                "optimizer": "adamw",
                "update_stride": 4,
                "async": {"enabled": True, "max_in_flight": 1},
                "trace": {"artifact_root": str(root)},
                "model": {"pair_id": pair_id},
                "dataset": {"adapter": "toy_replay"},
            }
        )
    )


def test_real_replay_pair_filter_uses_owning_runtime_config(tmp_path):
    for pair_id in ("toy_pair_a", "toy_pair_b"):
        run_root = tmp_path / pair_id
        _write_runtime_config(run_root, pair_id)
        split_groups = {}
        candidate = 0
        while len(split_groups) < 3:
            group = f"{pair_id}-request-{candidate}"
            split_groups.setdefault(split_of_group(group), group)
            candidate += 1
        for split, group in split_groups.items():
            _write_real_payload(
                run_root / split,
                _real_payload(
                    sequence_id=group,
                    update_id=f"{pair_id}-{split}",
                ),
            )

    records = load_real_replay_records(
        tmp_path, model_pair_id="toy_pair_a"
    )
    identity, layout = _controller_runtime_identity(
        tmp_path, "toy_pair_a"
    )

    assert len(records) == 3
    assert all(r.row.sequence_id.startswith("toy_pair_a-") for r in records)
    assert identity["model"]["pair_id"] == "toy_pair_a"
    assert layout == "1" * 64


def test_controller_identity_binds_zero_gradient_proximal_contract_not_lambda(
    tmp_path,
):
    from lightcone_spec.config.loader import validate_adaptation_config_dict

    common = {
        "schema_version": 1,
        "optimizer": "adamw",
        "update_stride": 4,
        "async": {"enabled": True, "max_in_flight": 1},
        "trace": {"artifact_root": str(tmp_path)},
        "model": {"pair_id": "toy_pair"},
        "dataset": {"adapter": "toy_replay"},
    }
    tts = validate_adaptation_config_dict(
        {**common, "method": "tts", "lambda_prox": 9.0}
    )
    l0 = validate_adaptation_config_dict(
        {**common, "method": "naive_async", "lambda_prox": 0.0}
    )

    tts_identity = controller_runtime_identity(tts)
    l0_identity = controller_runtime_identity(l0)
    assert tts_identity == l0_identity
    assert "lambda_prox" not in tts_identity["candidate"]
    assert tts_identity["candidate"]["proximal_contract"] == (
        "source_bound_zero_gradient_single_step_v1"
    )


def test_controller_identity_binds_weight_decay_with_zero_default(tmp_path):
    from lightcone_spec.config.loader import validate_adaptation_config_dict

    common = {
        "schema_version": 1,
        "method": "naive_async",
        "optimizer": "adamw",
        "update_stride": 4,
        "async": {"enabled": True, "max_in_flight": 1},
        "trace": {"artifact_root": str(tmp_path)},
        "model": {"pair_id": "toy_pair"},
        "dataset": {"adapter": "toy_replay"},
    }
    implicit_zero = validate_adaptation_config_dict(common)
    explicit_zero = validate_adaptation_config_dict(
        {**common, "weight_decay": 0.0}
    )
    decayed = validate_adaptation_config_dict(
        {**common, "weight_decay": 1e-2}
    )

    implicit_identity = controller_runtime_identity(implicit_zero)
    assert implicit_identity == controller_runtime_identity(explicit_zero)
    assert implicit_identity["schema_version"] == 4
    assert implicit_identity["candidate"][
        "weight_decay"
    ] == 0.0
    assert controller_runtime_identity(decayed) != (
        controller_runtime_identity(explicit_zero)
    )


def test_controller_identity_binds_effective_proposal_depth(tmp_path):
    from lightcone_spec.config.loader import validate_adaptation_config_dict

    common = {
        "schema_version": 1,
        "method": "naive_async",
        "optimizer": "adamw",
        "update_stride": 4,
        "async": {"enabled": True, "max_in_flight": 1},
        "trace": {"artifact_root": str(tmp_path)},
        "model": {"pair_id": "qwen3_4b_dspark7"},
        "dataset": {"adapter": "toy_replay"},
    }
    depth_seven = validate_adaptation_config_dict(
        {
            **common,
            "runtime": {"speculative_num_draft_tokens": 8},
        }
    )
    depth_four = validate_adaptation_config_dict(
        {
            **common,
            "runtime": {"speculative_num_draft_tokens": 5},
        }
    )
    capped_at_seven = validate_adaptation_config_dict(
        {
            **common,
            "runtime": {"speculative_num_draft_tokens": 12},
        }
    )

    identity_seven = controller_runtime_identity(depth_seven)
    identity_four = controller_runtime_identity(depth_four)
    identity_capped = controller_runtime_identity(capped_at_seven)

    assert identity_seven["candidate"]["effective_proposal_depth"] == 7
    assert identity_four["candidate"]["effective_proposal_depth"] == 4
    assert identity_seven != identity_four
    # Bind the semantic proposal depth rather than a larger, inert raw window.
    assert identity_seven == identity_capped


def test_schema_v2_controller_label_uses_full_not_actual_policy_utility(tmp_path):
    root = tmp_path / "v2"
    for split, group in _one_group_per_split().items():
        _write_real_payload(
            root / split,
            _real_payload(sequence_id=group, update_id=f"u-{split}"),
        )

    records = load_real_replay_records(root)

    assert len(records) == 3
    for record in records:
        assert record.row.utility == -0.75
        assert record.row.harmful == 1
        assert record.utilities[8] == -0.75
        assert record.full_candidate_utility == -0.75
        assert record.actual_published_utility == 1.25
        assert record.utility_by_kappa == {0.0: 0.0, 0.5: 0.25, 1.0: -0.75}
        assert record.provenance_method == "lc_damp"


def test_schema_v3_keeps_joint_l3_utility_separate_from_controller_label(
    tmp_path,
):
    root = tmp_path / "v3-l3"
    map_sha = "a" * 64
    for split, group in _one_group_per_split().items():
        _write_real_payload(
            root / split,
            _real_payload(
                schema_version=3,
                sequence_id=group,
                update_id=f"l3-{split}",
                provenance_method="lc_transport",
                evaluation_pair_id=f"pair-{split}",
                actual_published_utility=1.5,
                transported_candidate_utility=1.5,
                paired_l2_utility=0.75,
                transport_evaluation_contract=(
                    "joint_fisher_transport_adamw_damping_v1"
                ),
                transport_variant="joint",
                transport_map_sha256=map_sha,
            ),
        )

    records = load_real_replay_records(root)

    assert len(records) == 3
    for record in records:
        assert record.row.utility == -0.75
        assert record.transported_candidate_utility == 1.5
        assert record.paired_l2_utility == 0.75
        assert record.transport_map_sha256 == map_sha
        assert record.transport_evaluation_contract == (
            "joint_fisher_transport_adamw_damping_v1"
        )


def test_schema_v2_cannot_smuggle_l3_gate_evidence(tmp_path):
    root = tmp_path / "v2-with-l3-names"
    for split, group in _one_group_per_split().items():
        _write_real_payload(
            root / split,
            _real_payload(
                sequence_id=group,
                update_id=f"v2-{split}",
                transported_candidate_utility=9.0,
                paired_l2_utility=-9.0,
            ),
        )

    records = load_real_replay_records(root)

    assert all(record.transported_candidate_utility is None for record in records)
    assert all(record.paired_l2_utility is None for record in records)


def test_schema_v1_naive_async_can_migrate_without_policy_ambiguity(tmp_path):
    root = tmp_path / "v1-naive"
    payload = _real_payload(
        schema_version=1,
        provenance_method="naive_async",
        utility=0.5,
    )
    payload.pop("actual_published_utility")
    payload.pop("full_candidate_utility")
    for split, group in _one_group_per_split().items():
        _write_real_payload(
            root / split,
            {**payload, "sequence_id": group, "update_id": f"u-{split}"},
        )

    records = load_real_replay_records(root)

    assert len(records) == 3
    for record in records:
        assert record.row.utility == 0.5
        assert record.actual_published_utility == 0.5
        assert record.full_candidate_utility == 0.5


@pytest.mark.parametrize("value", [None, False])
def test_real_replay_rejects_missing_or_approximate_prefix_features(
    tmp_path, value
):
    root = tmp_path / f"prefix-{value}"
    payload = _real_payload(prefix_feature_exact=value)
    if value is None:
        payload.pop("prefix_feature_exact")
    _write_real_payload(root, payload)

    with pytest.raises(ArtifactValidationFailure, match="prefix"):
        load_real_replay_records(root)


@pytest.mark.parametrize("field", ["source_acceptance", "rho_path", "delta_z"])
def test_real_replay_schema_v3_rejects_nonfinite_publish_evidence(
    tmp_path, field
):
    root = tmp_path / f"nonfinite-{field}"
    payload = _real_payload(schema_version=3)
    if field == "delta_z":
        payload[field] = torch.tensor([0.0, float("nan")])
    else:
        payload[field] = float("nan")
    _write_real_payload(root, payload)

    with pytest.raises(ArtifactValidationFailure, match="non-finite"):
        load_real_replay_records(root)


def test_real_replay_schema_v3_requires_source_quality_features(tmp_path):
    root = tmp_path / "missing-source-feature"
    payload = _real_payload(schema_version=3)
    payload.pop("source_grad_norm")
    _write_real_payload(root, payload)

    with pytest.raises(ArtifactValidationFailure, match="publish-time"):
        load_real_replay_records(root)


def test_real_replay_tp_rejects_rank_local_fresh_gradient(tmp_path):
    root = tmp_path / "tp-rank-local"
    _write_runtime_config(root, "toy_pair_a")
    config_path = root / "adaptation.runtime.yaml"
    config = json.loads(config_path.read_text())
    config["runtime"] = {"tensor_parallel_size": 2}
    config_path.write_text(json.dumps(config))
    replay = root / "real-replay"
    _write_real_payload(replay, _real_payload(schema_version=3))

    with pytest.raises(ArtifactValidationFailure, match="rank-local fresh"):
        load_real_replay_records(root, model_pair_id="toy_pair_a")


def test_real_replay_rejects_oracle_labels_not_derived_from_kappa_grid(tmp_path):
    root = tmp_path / "forged-oracle"
    payload = _real_payload(
        schema_version=3,
        oracle_l1_utility=9.0,
        oracle_l2_utility=9.0,
        oracle_l2_kappa=0.5,
    )
    _write_real_payload(root, payload)

    with pytest.raises(ArtifactValidationFailure, match="oracle labels"):
        load_real_replay_records(root)


@pytest.mark.parametrize("provenance_method", [None, "lc_gate", "lc_damp"])
def test_schema_v1_policy_contaminated_or_unproven_trace_is_rejected(
    tmp_path, provenance_method
):
    root = tmp_path / f"v1-{provenance_method}"
    payload = _real_payload(schema_version=1, utility=0.5)
    payload.pop("actual_published_utility")
    payload.pop("full_candidate_utility")
    if provenance_method is None:
        payload.pop("provenance_method")
    else:
        payload["provenance_method"] = provenance_method
    _write_real_payload(root, payload)

    with pytest.raises(ArtifactValidationFailure, match="policy-label ambiguous"):
        load_real_replay_records(root)
