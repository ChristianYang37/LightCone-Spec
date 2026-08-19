from __future__ import annotations

import argparse

import pytest
from test_formal_registry_power_sources import _e3b, _e5, _e6

from lightcone_spec.cli.main import (
    _load_formal_registry_power_sources,
    _parser,
    _write_json,
)
from lightcone_spec.experiments.formal_registry import (
    signed_e3b_power_prefix_to_dict,
    signed_e5_power_and_anchor_to_dict,
    signed_e6_power_prefix_to_dict,
)


def test_formal_registry_cli_exposes_all_typed_power_source_paths() -> None:
    args = _parser().parse_args(
        [
            "extend-formal-registry-verification",
            "--prior-receipt",
            "prior.json",
            "--signed-e3b-power-prefix",
            "e3b.json",
            "--signed-e5-power-and-anchor-prefix",
            "e5.json",
            "--signed-e6-power-prefix",
            "e6.json",
            "--e0-authority-bundle",
            "e0-bundle.json",
            "--control-attestation",
            "control.json",
            "--control-replay-store",
            "replay.sqlite",
            "--now-ns",
            "1",
            "--output",
            "receipt.json",
        ]
    )
    assert args.signed_e3b_power_prefix == ["e3b.json"]
    assert args.signed_e5_power_and_anchor_prefix == ["e5.json"]
    assert args.signed_e6_power_prefix == ["e6.json"]
    assert args.e0_authority_bundle == ["e0-bundle.json"]


def test_formal_registry_cli_strictly_decodes_typed_power_sources(tmp_path) -> None:
    expected_e3b = _e3b()
    expected_e5 = _e5()
    expected_e6 = _e6()
    rows = (
        ("e3b.json", signed_e3b_power_prefix_to_dict(expected_e3b)),
        ("e5.json", signed_e5_power_and_anchor_to_dict(expected_e5)),
        ("e6.json", signed_e6_power_prefix_to_dict(expected_e6)),
    )
    for name, value in rows:
        _write_json(tmp_path / name, value)

    e3b, e5, e6 = _load_formal_registry_power_sources(
        argparse.Namespace(
            signed_e3b_power_prefix=[str(tmp_path / "e3b.json")],
            signed_e5_power_and_anchor_prefix=[str(tmp_path / "e5.json")],
            signed_e6_power_prefix=[str(tmp_path / "e6.json")],
        )
    )
    assert e3b == (expected_e3b,)
    assert e5 == (expected_e5,)
    assert e6 == (expected_e6,)


def test_formal_registry_cli_rejects_noncanonical_power_source(tmp_path) -> None:
    value = signed_e6_power_prefix_to_dict(_e6())
    value["unexpected"] = "digest-fallback"
    path = tmp_path / "foreign-e6.json"
    _write_json(path, value)

    with pytest.raises(ValueError, match="fields"):
        _load_formal_registry_power_sources(
            argparse.Namespace(
                signed_e3b_power_prefix=[],
                signed_e5_power_and_anchor_prefix=[],
                signed_e6_power_prefix=[str(path)],
            )
        )
