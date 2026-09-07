"""Mechanical diff formatter; updates hunk counts without changing patch content."""

import re
from pathlib import Path


def recount(text):
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)", line)
        if not match:
            continue
        old = new = 0
        for body in lines[i + 1:]:
            if body.startswith(("@@ ", "diff --git ")):
                break
            old += body.startswith((" ", "-"))
            new += body.startswith((" ", "+"))
        lines[i] = f"@@ -{match[1]},{old} +{match[2]},{new} @@{match[3]}\n"
    return "".join(lines)


if __name__ == "__main__":
    for patch in sorted((Path(__file__).resolve().parents[1] / "patches/sglang").glob("*.diff")):
        patch.write_text(recount(patch.read_text()))
