"""Launch SGLang only from a verified disposable patched checkout."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

from .checkout import verify_patched_checkout


def _bind_interpreter_tools() -> None:
    """Make console tools installed beside this Python visible to SGLang JITs."""
    # Resolve the directory, not the executable: venv Python is commonly a
    # symlink to the system interpreter, while tools such as ninja live beside
    # the symlink in the venv's bin directory.
    interpreter_bin = str(Path(sys.executable).parent.resolve())
    entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    os.environ["PATH"] = os.pathsep.join(
        [interpreter_bin, *(entry for entry in entries if entry != interpreter_bin)]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lightcone-sglang-launch")
    parser.add_argument("--checkout", required=True)
    parser.add_argument("server_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    server_argv = list(args.server_argv)
    if server_argv and server_argv[0] == "--":
        server_argv = server_argv[1:]
    if not server_argv:
        raise ValueError("SGLang server arguments are required after --")
    checkout = verify_patched_checkout(args.checkout)
    python_root = str(checkout / "python")
    if "sglang" in sys.modules:
        raise RuntimeError("sglang was imported before checkout verification")
    _bind_interpreter_tools()
    sys.path.insert(0, python_root)
    sys.argv = ["sglang.launch_server", *server_argv]
    runpy.run_module("sglang.launch_server", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
