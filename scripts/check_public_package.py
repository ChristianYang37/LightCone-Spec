#!/usr/bin/env python3
"""Check tracked public content only; never scan private diagnostics for publication."""

import re
import subprocess
from pathlib import Path


def main():
    files = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    forbidden_roots = ("diagnostics/", "archives/", "output/", "results/", ".env", "credentials/")
    token = re.compile(r"(?:hf_|gh[pousr]_)[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----")
    errors = []
    for name in filter(None, files):
        if name.startswith(forbidden_roots):
            errors.append(f"unreviewed private artifact tracked: {name}")
        path = Path(name)
        if path.is_file() and path.stat().st_size < 2_000_000:
            if token.search(path.read_bytes().decode("utf-8", errors="replace")):
                errors.append(f"possible credential in {name}; value deliberately not printed")
    for required in ("LICENSE", "LICENSES/Apache-2.0.txt", "NOTICE", "CONTRIBUTING.md"):
        if not Path(required).is_file():
            errors.append(f"missing license/contribution file: {required}")
    if errors:
        raise RuntimeError("\n".join(errors))
    print("PASS: tracked-file artifact/credential scan and license inventory")


if __name__ == "__main__":
    main()
