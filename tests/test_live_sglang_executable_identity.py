from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lightcone_spec.orchestration.live_sglang import (
    _require_source_owned_server_executable,
)


def test_live_server_accepts_the_current_venv_launcher_and_real_interpreter() -> None:
    current = Path(sys.executable)

    assert _require_source_owned_server_executable(str(current)) == current
    resolved = current.resolve(strict=True)
    assert _require_source_owned_server_executable(str(resolved)) == resolved


def test_live_server_rejects_a_foreign_venv_launcher(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "foreign-venv").resolve()
    binary = root / "bin"
    binary.mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = foreign\n", encoding="utf-8")
    launcher = binary / "python"
    launcher.symlink_to(Path(sys.executable).resolve(strict=True))

    with pytest.raises(ValueError, match="launcher differs"):
        _require_source_owned_server_executable(str(launcher))
