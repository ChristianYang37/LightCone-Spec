from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module_name",
    (
        "lightcone_spec.orchestration.formal_cell_worker",
        "lightcone_spec.orchestration.formal_preflight_exact_ten_group_worker",
    ),
)
def test_formal_worker_module_cold_starts_in_fresh_interpreter(
    module_name: str,
) -> None:
    completed = subprocess.run(
        (sys.executable, "-m", module_name, "--help"),
        cwd=_REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_orchestration_public_exports_resolve_lazily_in_fresh_interpreter() -> None:
    script = """
import importlib
import sys

import lightcone_spec.orchestration as package

assert not any(
    name.startswith("lightcone_spec.orchestration.") for name in sys.modules
)
assert len(package.__all__) == len(set(package.__all__))
for module_name, names in package._LAZY_EXPORT_GROUPS:
    module = importlib.import_module(
        f"lightcone_spec.orchestration.{module_name}"
    )
    for name in names:
        assert getattr(package, name) is getattr(module, name), name
assert set(package.__all__) == {
    name
    for _module_name, names in package._LAZY_EXPORT_GROUPS
    for name in names
}
"""
    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=_REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
