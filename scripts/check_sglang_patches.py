#!/usr/bin/env python3
"""Apply the five patches to a disposable clean checkout and compile changed Python."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="Disposable upstream 3312645a307453893a00778592f105581e3d1c3d checkout")
    args = parser.parse_args()
    patches = sorted((Path(__file__).resolve().parents[1] / "patches/sglang").glob("*.diff"))
    if len(patches) != 5:
        raise RuntimeError("expected all five cumulative SGLang patches")
    changed = set()
    for patch in patches:
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=args.root, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=args.root, check=True)
        changed.update(line.removeprefix("+++ b/") for line in patch.read_text().splitlines()
                       if line.startswith("+++ b/") and line.endswith(".py"))
    for name in changed:
        compile((args.root / name).read_bytes(), name, "exec")
    print(f"PASS: five ordinary cumulative patches; {len(changed)} Python files compiled")


if __name__ == "__main__":
    main()
