"""Launch SGLang only from a verified disposable patched checkout."""

from __future__ import annotations

import argparse
import runpy
import sys

from .checkout import verify_patched_checkout


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
    sys.path.insert(0, python_root)
    sys.argv = ["sglang.launch_server", *server_argv]
    runpy.run_module("sglang.launch_server", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
