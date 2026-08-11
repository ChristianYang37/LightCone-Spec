#!/usr/bin/env python3
"""Scan every reachable Git blob for high-confidence credential signatures.

The detector never prints matched bytes.  A failure identifies only the rule,
blob object, and repository path so CI logs cannot amplify a leaked secret.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_BLOB_BYTES = 5 * 1024 * 1024
RULES = {
    "private-key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "huggingface-token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "github-token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "openai-token": re.compile(rb"\bsk-[A-Za-z0-9_-]{32,}\b"),
    "aws-access-key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "jwt-credential": re.compile(
        rb"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b"
    ),
}


def _git(*arguments: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ("git", "-C", str(ROOT), *arguments),
        input=input_bytes,
        capture_output=True,
        check=True,
    ).stdout


def _reachable_blobs() -> tuple[tuple[str, str], ...]:
    rows: dict[str, str] = {}
    for line in _git("rev-list", "--objects", "--all").decode().splitlines():
        object_id, separator, path = line.partition(" ")
        if separator and object_id not in rows:
            rows[object_id] = path
    return tuple(sorted(rows.items()))


def main() -> int:
    scanned = 0
    for object_id, path in _reachable_blobs():
        object_type = _git("cat-file", "-t", object_id).decode().strip()
        if object_type != "blob":
            continue
        size = int(_git("cat-file", "-s", object_id).decode())
        if size > MAX_BLOB_BYTES:
            raise SystemExit(
                "history-secret check failed: oversized reachable blob "
                f"{object_id} at {path}"
            )
        body = _git("cat-file", "blob", object_id)
        scanned += 1
        for label, pattern in RULES.items():
            if pattern.search(body):
                raise SystemExit(
                    "history-secret check failed: "
                    f"{label} signature in blob {object_id} at {path}"
                )
    print(f"history-secret checks passed for {scanned} reachable blobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
