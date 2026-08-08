#!/usr/bin/env python3
"""Fail closed on publication hygiene, links, and patch metadata."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_PREFIXES = (
    "artifacts/",
    "sglang/",
    "tmp/",
    ".local-backup/",
    ".venv/",
    "build/",
    "dist/",
)
FORBIDDEN_NAMES = {
    "selection_summary.json",
    "selected_optimizer_config.json",
    "SSH_AGENT_HANDOFF_PROMPT_2026-08-04.md",
}
TEXT_SUFFIXES = {
    ".cfg", ".ini", ".json", ".md", ".patch", ".py", ".sh", ".toml",
    ".txt", ".yaml", ".yml",
}


def tracked() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode().split("\0")
    return [ROOT / value for value in output if value]


def fail(message: str) -> None:
    raise SystemExit(f"public-tree check failed: {message}")


def check_paths(files: list[Path]) -> None:
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(FORBIDDEN_PREFIXES) or path.name in FORBIDDEN_NAMES:
            fail(f"forbidden tracked path: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            fail(f"tracked file exceeds 5 MiB: {relative}")


def check_text(files: list[Path]) -> None:
    forbidden = {
        "macOS user path": re.compile(r"/Users/[A-Za-z0-9._-]+/"),
        "root home path": re.compile("/" + "root" + "/"),
        "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
        "JWT-like credential": re.compile(
            r"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b"
        ),
        "private key": re.compile(
            "-----BEGIN " + r"[A-Z ]*" + "PRIVATE KEY-----"
        ),
    }
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        body = path.read_text(encoding="utf-8", errors="strict")
        for label, pattern in forbidden.items():
            if pattern.search(body):
                fail(f"{label} in {path.relative_to(ROOT)}")


def check_markdown_links(files: list[Path]) -> None:
    link = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        for target in link.findall(path.read_text(encoding="utf-8")):
            target = target.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"^(?:https?|mailto):", target):
                continue
            if not (path.parent / target).resolve().exists():
                fail(f"broken link {target!r} in {path.relative_to(ROOT)}")


def check_readme_parity() -> None:
    def structure(name: str) -> list[int]:
        return [
            len(match.group(1))
            for match in re.finditer(
                r"^(#{1,6})\s+", (ROOT / name).read_text(), re.MULTILINE
            )
        ]

    if structure("README.md") != structure("README_zh-CN.md"):
        fail("English and Chinese README heading structures differ")
    en = {path.name for path in (ROOT / "docs/en").glob("*.md")}
    zh = {path.name for path in (ROOT / "docs/zh-CN").glob("*.md")}
    if en != zh:
        fail("English and Chinese documentation file sets differ")


def check_patchset() -> None:
    patch_root = ROOT / "patches/sglang"
    manifest = json.loads((patch_root / "manifest.json").read_text())
    series = [line for line in (patch_root / "series").read_text().splitlines() if line]
    entries = manifest["patches"]
    if series != [entry["file"] for entry in entries]:
        fail("patch series and manifest order differ")
    sums = {}
    for line in (patch_root / "SHA256SUMS").read_text().splitlines():
        digest, file_name = line.split("\t", 1)
        sums[file_name] = digest
    for entry in entries:
        data = (patch_root / entry["file"]).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"] or sums.get(entry["file"]) != digest:
            fail(f"patch identity mismatch: {entry['file']}")


def main() -> int:
    files = tracked()
    check_paths(files)
    check_text(files)
    check_markdown_links(files)
    check_readme_parity()
    check_patchset()
    print(f"public-tree checks passed for {len(files)} tracked files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
