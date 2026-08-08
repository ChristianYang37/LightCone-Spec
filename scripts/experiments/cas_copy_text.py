#!/usr/bin/env python3
"""CAS-copy a text artifact and ensure its SHA-256 sidecar matches."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source.resolve()
    target = args.target.resolve()
    text = source.read_text(encoding="utf-8")
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise SystemExit(f"final terminal collision: {target}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    sidecar = Path(str(target) + ".sha256")
    if sidecar.exists() and sidecar.read_text(encoding="utf-8").strip() != digest:
        raise SystemExit(f"final terminal sidecar collision: {sidecar}")
    if not sidecar.exists():
        sidecar.write_text(digest + "\n", encoding="utf-8")
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
