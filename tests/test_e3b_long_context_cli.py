from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

cli = importlib.import_module("lightcone_spec.cli.main")


def _sha(value: object) -> str:
    return cli._canonical_sha256(value)


def _write_bound(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    Path(f"{path}.sha256").write_text(f"{_sha(value)}\n", encoding="ascii")


def _outer_manifest(
    tmp_path: Path,
    *,
    family_bindings: list[dict[str, str]],
    repetitions: int = 100,
    seed: int = 17,
    name: str = "e3b-stage.json",
) -> Path:
    path = tmp_path / name
    _write_bound(
        path,
        {
            "schema_version": 1,
            "kind": "industrial_e3b_long_context_analysis_manifest",
            "family_manifests": family_bindings,
            "bootstrap": {"repetitions": repetitions, "seed": seed},
        },
    )
    return path


def _family_bindings(tmp_path: Path, *, count: int = 2) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for index in range(count):
        value = {
            "schema_version": 3,
            "kind": "industrial_analysis_manifest",
            "test_family_index": index,
        }
        path = tmp_path / f"family-{index}.json"
        _write_bound(path, value)
        result.append({"path": path.name, "sha256": _sha(value)})
    return result


def _nested_rows(
    tmp_path: Path,
    bindings: list[dict[str, str]],
    *,
    repetitions: int = 100,
    seed: int = 17,
):
    registry = SimpleNamespace(sha256="a" * 64)
    inventory = object()
    envelope = object()
    rows = {
        (tmp_path / binding["path"]).resolve(): (
            registry,
            object(),
            object(),
            object(),
            inventory,
            envelope,
            (),
            (),
            None,
            None,
            None,
            repetitions,
            seed,
        )
        for binding in bindings
    }
    return registry, inventory, envelope, rows


def _patch_nested_loader(
    monkeypatch: pytest.MonkeyPatch,
    rows: dict[Path, tuple],
) -> list[dict[str, object]]:
    constructed: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli,
        "_load_industrial_analysis_manifest",
        lambda path: rows[Path(path).resolve()],
    )

    def raw_family(**kwargs):
        constructed.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(cli, "E3bLongContextRawFamilyInput", raw_family)
    return constructed


def test_e3b_cli_manifest_reopens_bound_family_paths_and_rejects_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _family_bindings(tmp_path)
    registry, inventory, envelope, rows = _nested_rows(tmp_path, bindings)
    constructed = _patch_nested_loader(monkeypatch, rows)
    manifest = _outer_manifest(tmp_path, family_bindings=bindings)

    loaded = cli._load_e3b_long_context_analysis_manifest(manifest)

    assert loaded[:1] == (registry,)
    assert loaded[2:] == (inventory, envelope, 100, 17)
    assert len(loaded[1]) == len(bindings) == len(constructed)
    assert all(
        set(row)
        == {
            "pilot_activation",
            "final_activation",
            "confirmation_reduction",
            "blocks",
            "evidence_alias_manifests",
            "evidence_dependence_map",
            "gpu_attestation",
            "doctor_report",
        }
        for row in constructed
    )

    (tmp_path / "alias").mkdir()
    duplicate = [
        bindings[0],
        {
            "path": "alias/../family-0.json",
            "sha256": bindings[0]["sha256"],
        },
    ]
    duplicate_manifest = _outer_manifest(
        tmp_path,
        family_bindings=duplicate,
        name="duplicate.json",
    )
    with pytest.raises(ValueError, match="duplicates a raw family path"):
        cli._load_e3b_long_context_analysis_manifest(duplicate_manifest)


def test_e3b_cli_manifest_and_family_sidecars_are_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _family_bindings(tmp_path, count=1)
    _, _, _, rows = _nested_rows(tmp_path, bindings)
    _patch_nested_loader(monkeypatch, rows)

    missing_sidecar = _outer_manifest(
        tmp_path,
        family_bindings=bindings,
        name="missing-sidecar.json",
    )
    Path(f"{missing_sidecar}.sha256").unlink()
    with pytest.raises(ValueError, match="must be a regular bound file"):
        cli._load_e3b_long_context_analysis_manifest(missing_sidecar)

    forged_binding = [dict(bindings[0], sha256="0" * 64)]
    forged_manifest = _outer_manifest(
        tmp_path,
        family_bindings=forged_binding,
        name="forged-binding.json",
    )
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        cli._load_e3b_long_context_analysis_manifest(forged_manifest)

    tampered_manifest = _outer_manifest(
        tmp_path,
        family_bindings=bindings,
        name="tampered-family.json",
    )
    family_path = tmp_path / bindings[0]["path"]
    family_path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar is missing or invalid"):
        cli._load_e3b_long_context_analysis_manifest(tampered_manifest)


@pytest.mark.parametrize(
    ("foreign_repetitions", "foreign_seed"),
    ((101, 17), (100, 18)),
)
def test_e3b_cli_manifest_rejects_foreign_family_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    foreign_repetitions: int,
    foreign_seed: int,
) -> None:
    bindings = _family_bindings(tmp_path, count=2)
    _, _, _, rows = _nested_rows(tmp_path, bindings)
    foreign_path = (tmp_path / bindings[1]["path"]).resolve()
    foreign = list(rows[foreign_path])
    foreign[11] = foreign_repetitions
    foreign[12] = foreign_seed
    rows[foreign_path] = tuple(foreign)
    _patch_nested_loader(monkeypatch, rows)
    manifest = _outer_manifest(tmp_path, family_bindings=bindings)

    with pytest.raises(ValueError, match="differ in .* bootstrap"):
        cli._load_e3b_long_context_analysis_manifest(manifest)


def test_analyze_e3b_cli_writes_canonical_unattested_artifact_and_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = object()
    families = (object(),)
    inventory = object()
    envelope = object()
    monkeypatch.setattr(
        cli,
        "_load_e3b_long_context_analysis_manifest",
        lambda _path: (registry, families, inventory, envelope, 100, 23),
    )
    payload = {
        "schema_version": 1,
        "kind": "e3b_long_context_stage_reducer",
        "status": "UNRESOLVED",
        "evidence_level": "RAW_DIAGNOSTIC_OBSERVED_UNATTESTED",
        "reasons": ["gpu_attestation:missing"],
    }
    artifact = SimpleNamespace(
        sha256=_sha(payload),
        to_dict=lambda: payload,
    )
    received: list[dict[str, object]] = []

    def reduce_probe(**kwargs):
        received.append(kwargs)
        return artifact

    monkeypatch.setattr(cli, "reduce_e3b_long_context_from_raw", reduce_probe)
    output = tmp_path / "e3b-output.json"

    assert (
        cli.main(
            [
                "analyze-e3b-long-context",
                "--manifest",
                str(tmp_path / "bound-input.json"),
                "--output",
                str(output),
            ]
        )
        == 42
    )
    assert received == [
        {
            "registry": registry,
            "families": families,
            "hardware_envelope": envelope,
            "inventory": inventory,
            "bootstrap_repetitions": 100,
            "bootstrap_seed": 23,
        }
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert Path(f"{output}.sha256").read_text(encoding="ascii") == (
        f"{artifact.sha256}\n"
    )
    assert cli._artifact_sha256(output) == artifact.sha256
    assert payload["evidence_level"] == "RAW_DIAGNOSTIC_OBSERVED_UNATTESTED"
