#!/usr/bin/env python3
"""Write an immutable runtime-implementation fingerprint bound to a lockfile."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lockfile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    # Import after argv parse so --help works without the runtime on PYTHONPATH.
    from lightcone_spec.locking.lockfile import load_lockfile
    from lightcone_spec.orchestration.runtime_config import (
        runtime_implementation_fingerprint,
    )

    lock = load_lockfile(args.lockfile)
    compiler = lock.environment.compiler_versions or {}
    reference = {
        key: compiler[key]
        for key in (
            "lightcone_runtime_source_sha256",
            "sglang_runtime_source_sha256",
            "sglang_fork_commit",
            "sglang_fork_dirty",
        )
        if key in compiler
    }
    value = runtime_implementation_fingerprint(locked_reference=reference)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output = args.output.expanduser().resolve()
    sidecar = Path(str(output) + ".sha256")
    if output.exists():
        if output.read_text(encoding="utf-8") != text:
            raise SystemExit(f"runtime fingerprint collision: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    if sidecar.exists() and sidecar.read_text(encoding="utf-8").strip() != digest:
        raise SystemExit(f"runtime fingerprint sidecar collision: {sidecar}")
    if not sidecar.exists():
        sidecar.write_text(digest + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
