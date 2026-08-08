#!/usr/bin/env python3
"""Verify a screen terminal sidecar and evidence hashes without schema pinning.

The screen queue resume validator still requires nested receipts to be schema
v1, but schema-v2 stride-selection receipts are nested under blocked/selected
terminals.  Oracle-scope confirmation resume only needs content integrity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", type=Path, required=True)
    args = parser.parse_args(argv)
    terminal = args.terminal.resolve()
    sidecar = Path(str(terminal) + ".sha256")
    if not terminal.is_file() or not sidecar.is_file():
        raise SystemExit(f"terminal or sidecar missing: {terminal}")
    if sidecar.read_text(encoding="utf-8").strip() != _sha256(terminal):
        raise SystemExit(f"terminal sidecar mismatch: {terminal}")
    payload = json.loads(terminal.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("status")
        not in {"candidate_screen_blocked", "candidate_screen_selected"}
        or payload.get("scope") != "candidate_screen_only_no_claim"
        or not isinstance(payload.get("evidence"), list)
        or not payload["evidence"]
    ):
        raise SystemExit(f"terminal envelope invalid: {terminal}")
    seen: set[Path] = set()
    for index, row in enumerate(payload["evidence"]):
        if not isinstance(row, dict):
            raise SystemExit(f"evidence[{index}] is not an object")
        raw = row.get("path")
        digest = row.get("sha256")
        if not isinstance(raw, str) or not isinstance(digest, str) or len(digest) != 64:
            raise SystemExit(f"evidence[{index}] path/sha256 invalid")
        path = Path(raw).resolve()
        if path in seen or not path.is_file() or _sha256(path) != digest:
            raise SystemExit(f"evidence hash mismatch: {path}")
        seen.add(path)
    print(
        json.dumps(
            {
                "terminal": str(terminal),
                "status": payload["status"],
                "evidence_files": len(seen),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
