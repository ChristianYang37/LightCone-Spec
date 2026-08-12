from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from lightcone_spec.experiments.breadth_fdr_authority import (
    E0_BREADTH_FALSE_DISCOVERY_RATE,
    E0_BREADTH_FDR_PROTOCOL_SHA256,
    E0_BREADTH_FORMAL_SOURCE_UNAVAILABLE_REASON,
    E0_BREADTH_RAW_SOURCE_MISSING_REASON,
    E0_BREADTH_RELEASE_TRUSTED_RAW_SOURCE_SHA256,
    E0BreadthFdrAuthorityBlocked,
    E0BreadthHypothesis,
    bind_e0_breadth_fdr_authority,
    reduce_e0_breadth_fdr,
    registered_e0_breadth_hypotheses,
    require_formal_e0_breadth_fdr_authority,
    revalidate_e0_breadth_fdr_authority,
)
from lightcone_spec.experiments.registry import (
    E0_BACKENDS,
    E0_MODELS,
    E0_TASKS,
    ExperimentRegistry,
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.statistics import benjamini_hochberg


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return build_industrial_registry()


@pytest.fixture(scope="module")
def hypotheses(
    registry: ExperimentRegistry,
) -> tuple[E0BreadthHypothesis, ...]:
    return registered_e0_breadth_hypotheses(registry)


def _raw_row(hypothesis: E0BreadthHypothesis, p_value: float) -> dict[str, object]:
    numerator = content_sha256(
        {"role": "numerator_terminal", "hypothesis_id": hypothesis.hypothesis_id}
    )
    denominator = content_sha256(
        {"role": "denominator_terminal", "hypothesis_id": hypothesis.hypothesis_id}
    )
    artifact = content_sha256(
        {
            "schema_version": 1,
            "hypothesis_id": hypothesis.hypothesis_id,
            "numerator_terminal_sha256": numerator,
            "denominator_terminal_sha256": denominator,
            "raw_p_value": float(p_value),
        }
    )
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "numerator_terminal_sha256": numerator,
        "denominator_terminal_sha256": denominator,
        "contrast_artifact_sha256": artifact,
        "raw_p_value": p_value,
    }


def _raw_document(
    registry: ExperimentRegistry,
    hypotheses: tuple[E0BreadthHypothesis, ...],
    p_value: Callable[[E0BreadthHypothesis, int], float] | None = None,
) -> dict[str, object]:
    value_for = p_value or (lambda _hypothesis, _index: 0.5)
    rows = {
        hypothesis.hypothesis_id: _raw_row(
            hypothesis,
            value_for(hypothesis, index),
        )
        for index, hypothesis in enumerate(hypotheses)
    }
    return {
        "schema_version": 1,
        "kind": "e0_breadth_raw_p_values",
        "registry_sha256": registry.sha256,
        "protocol_sha256": E0_BREADTH_FDR_PROTOCOL_SHA256,
        "false_discovery_rate": E0_BREADTH_FALSE_DISCOVERY_RATE,
        "hypotheses_sha256": content_sha256(
            [hypothesis.to_dict() for hypothesis in hypotheses]
        ),
        "families": [
            {
                "family_id": family_id,
                "p_values": [
                    rows[hypothesis.hypothesis_id]
                    for hypothesis in hypotheses
                    if hypothesis.family_id == family_id
                ],
            }
            for family_id in (
                "e0_core_breadth",
                "e0_isolated_onlinespec_breadth",
            )
        ],
    }


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path.resolve()


def test_registered_universe_is_exact_and_registry_owned(
    registry: ExperimentRegistry,
    hypotheses: tuple[E0BreadthHypothesis, ...],
) -> None:
    assert len(hypotheses) == 540
    assert len({hypothesis.hypothesis_id for hypothesis in hypotheses}) == 540
    assert tuple(hypothesis.hypothesis_id for hypothesis in hypotheses) == tuple(
        sorted(hypothesis.hypothesis_id for hypothesis in hypotheses)
    )
    assert {hypothesis.model for hypothesis in hypotheses} == set(E0_MODELS)
    assert {hypothesis.backend for hypothesis in hypotheses} == set(E0_BACKENDS)
    assert {hypothesis.task for hypothesis in hypotheses} == set(E0_TASKS)

    core = tuple(
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.family_id == "e0_core_breadth"
    )
    online = tuple(
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.family_id == "e0_isolated_onlinespec_breadth"
    )
    assert len(core) == 216
    assert {hypothesis.contrast for hypothesis in core} == {
        "l0_vs_static",
        "l0_vs_tts",
    }
    assert len(online) == 324
    assert {hypothesis.contrast for hypothesis in online} == {
        "onlinespec_ogd_vs_static",
        "onlinespec_opt_vs_static",
        "onlinespec_ens_vs_static",
    }

    cells = {cell.cell_id: cell for cell in registry.cells_for("E0")}
    for hypothesis in hypotheses:
        assert cells[hypothesis.numerator_cell_id].sha256 == (
            hypothesis.numerator_cell_sha256
        )
        assert cells[hypothesis.denominator_cell_id].sha256 == (
            hypothesis.denominator_cell_sha256
        )


def test_complete_source_applies_existing_bh_separately_to_fixed_families(
    tmp_path: Path,
    registry: ExperimentRegistry,
    hypotheses: tuple[E0BreadthHypothesis, ...],
) -> None:
    first_by_family = {
        "e0_core_breadth": next(
            value for value in hypotheses if value.family_id == "e0_core_breadth"
        ).hypothesis_id,
        "e0_isolated_onlinespec_breadth": next(
            value
            for value in hypotheses
            if value.family_id == "e0_isolated_onlinespec_breadth"
        ).hypothesis_id,
    }

    def p_value(hypothesis: E0BreadthHypothesis, _index: int) -> float:
        return 1e-8 if hypothesis.hypothesis_id in first_by_family.values() else 0.5

    path = _write_json(
        tmp_path / "e0-p-values.json",
        _raw_document(registry, hypotheses, p_value),
    )
    authority = bind_e0_breadth_fdr_authority(registry, path)
    reduction = reduce_e0_breadth_fdr(registry, authority)

    assert len(authority.p_values) == len(reduction.decisions) == 540
    assert reduction.primary_family_eligible is False
    assert reduction.formal_execution_authorized is False
    assert reduction.families == (
        "e0_core_breadth",
        "e0_isolated_onlinespec_breadth",
    )
    assert sum(decision.rejected for decision in reduction.decisions) == 2
    decision_by_id = {
        decision.hypothesis_id: decision for decision in reduction.decisions
    }
    raw_by_id = {value.hypothesis_id: value for value in authority.p_values}
    for family_id, expected_size in (
        ("e0_core_breadth", 216),
        ("e0_isolated_onlinespec_breadth", 324),
    ):
        family = tuple(
            hypothesis for hypothesis in hypotheses if hypothesis.family_id == family_id
        )
        expected = benjamini_hochberg(
            {
                hypothesis.hypothesis_id: raw_by_id[
                    hypothesis.hypothesis_id
                ].raw_p_value
                for hypothesis in family
            },
            false_discovery_rate=0.05,
        )
        assert len(expected) == expected_size
        for adjusted in expected:
            decision = decision_by_id[adjusted.name]
            assert decision.q_value == adjusted.adjusted_p_value
            assert decision.rejected == adjusted.rejected
            assert decision.procedure == "benjamini-hochberg"


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda document: document["families"][0]["p_values"].pop(),
            "coverage",
        ),
        (
            lambda document: document["families"].reverse(),
            "reordered or regrouped",
        ),
        (
            lambda document: document["families"].pop(),
            "registered families",
        ),
        (
            lambda document: document.__setitem__("false_discovery_rate", 0.1),
            "registered protocol",
        ),
        (
            lambda document: document["families"][0]["p_values"][0].__setitem__(
                "hypothesis_id", "0" * 64
            ),
            "source identity|coverage",
        ),
        (
            lambda document: document["families"][0]["p_values"][0].__setitem__(
                "raw_p_value", 1.1
            ),
            "finite probability",
        ),
    ],
)
def test_subset_regroup_foreign_and_invalid_values_fail_closed(
    tmp_path: Path,
    registry: ExperimentRegistry,
    hypotheses: tuple[E0BreadthHypothesis, ...],
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    document = _raw_document(registry, hypotheses)
    mutation(document)
    path = _write_json(tmp_path / "mutated.json", document)
    with pytest.raises(ValueError, match=message):
        bind_e0_breadth_fdr_authority(registry, path)


def test_duplicate_json_keys_and_nonfinite_json_fail_closed(
    tmp_path: Path,
    registry: ExperimentRegistry,
    hypotheses: tuple[E0BreadthHypothesis, ...],
) -> None:
    document = _raw_document(registry, hypotheses)
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"))

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        encoded.replace('"families":', '"families":[],"families":', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicates key"):
        bind_e0_breadth_fdr_authority(registry, duplicate.resolve())

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(encoded.replace('"raw_p_value":0.5', '"raw_p_value":NaN', 1))
    with pytest.raises(ValueError, match="non-finite"):
        bind_e0_breadth_fdr_authority(registry, nonfinite.resolve())


def test_source_identities_are_required_and_consistent(
    tmp_path: Path,
    registry: ExperimentRegistry,
    hypotheses: tuple[E0BreadthHypothesis, ...],
) -> None:
    same_terminal = _raw_document(registry, hypotheses)
    row = same_terminal["families"][0]["p_values"][0]
    row["denominator_terminal_sha256"] = row["numerator_terminal_sha256"]
    path = _write_json(tmp_path / "same-terminal.json", same_terminal)
    with pytest.raises(ValueError, match="sources must differ"):
        bind_e0_breadth_fdr_authority(registry, path)

    wrong_artifact = _raw_document(registry, hypotheses)
    wrong_artifact["families"][0]["p_values"][0]["contrast_artifact_sha256"] = "f" * 64
    path = _write_json(tmp_path / "wrong-artifact.json", wrong_artifact)
    with pytest.raises(ValueError, match="source identity"):
        bind_e0_breadth_fdr_authority(registry, path)


def test_missing_symlink_foreign_registry_and_revalidation_tamper_fail_closed(
    tmp_path: Path,
    registry: ExperimentRegistry,
    hypotheses: tuple[E0BreadthHypothesis, ...],
) -> None:
    with pytest.raises(E0BreadthFdrAuthorityBlocked) as missing:
        bind_e0_breadth_fdr_authority(registry, tmp_path / "missing.json")
    assert missing.value.reason == E0_BREADTH_RAW_SOURCE_MISSING_REASON

    path = _write_json(
        tmp_path / "raw.json",
        _raw_document(registry, hypotheses),
    )
    link = tmp_path / "raw-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        bind_e0_breadth_fdr_authority(registry, link.absolute())

    authority = bind_e0_breadth_fdr_authority(registry, path)
    foreign_registry = build_industrial_registry(seed=20260812)
    with pytest.raises(ValueError, match="registered protocol"):
        bind_e0_breadth_fdr_authority(foreign_registry, path)

    document = _raw_document(registry, hypotheses)
    hypothesis = hypotheses[0]
    replacement = _raw_row(hypothesis, 0.25)
    for family in document["families"]:
        for index, row in enumerate(family["p_values"]):
            if row["hypothesis_id"] == hypothesis.hypothesis_id:
                family["p_values"][index] = replacement
    _write_json(path, document)
    with pytest.raises(ValueError, match="changed during revalidation"):
        revalidate_e0_breadth_fdr_authority(registry, authority)


def test_formal_entrypoint_blocks_before_read_while_release_allowlist_is_empty(
    tmp_path: Path,
    registry: ExperimentRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert E0_BREADTH_RELEASE_TRUSTED_RAW_SOURCE_SHA256 == ()
    reads = 0

    def unexpected_read(_path: str | Path) -> tuple[Path, bytes]:
        nonlocal reads
        reads += 1
        raise AssertionError("formal BLOCK must precede raw-source access")

    monkeypatch.setattr(
        "lightcone_spec.experiments.breadth_fdr_authority._read_stable_raw",
        unexpected_read,
    )
    with pytest.raises(E0BreadthFdrAuthorityBlocked) as blocked:
        require_formal_e0_breadth_fdr_authority(
            registry,
            tmp_path / "caller-self-reported.json",
        )
    assert blocked.value.reason == E0_BREADTH_FORMAL_SOURCE_UNAVAILABLE_REASON
    assert reads == 0
