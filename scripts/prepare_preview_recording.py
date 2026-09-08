"""Install pinned recording tools in a separate SERVER-only directory; no GPU work."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise RuntimeError("recording dependencies belong on the SSH Linux x86_64 host")
    root = args.tools.resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = "node-v22.14.0-linux-x64"
    archive = root / (name + ".tar.xz")
    node = root / name
    if not node.exists():
        base = "https://nodejs.org/dist/v22.14.0/"
        with urllib.request.urlopen(base + "SHASUMS256.txt", timeout=60) as response:
            checksums = response.read().decode()
        expected = next(line.split()[0] for line in checksums.splitlines() if line.endswith(archive.name))
        if not archive.exists():
            urllib.request.urlretrieve(base + archive.name, archive)
        with archive.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise RuntimeError("Node archive checksum mismatch")
        with tarfile.open(archive) as source:
            source.extractall(root, filter="data")
    env = {**os.environ, "PATH": str(node / "bin") + os.pathsep + os.environ["PATH"],
           "PLAYWRIGHT_BROWSERS_PATH": str(root / "browsers")}
    subprocess.run([str(node / "bin/npm"), "install", "--prefix", str(root), "--ignore-scripts",
                    "--no-audit", "--no-fund", "playwright@1.55.0"], env=env, check=True)
    subprocess.run([str(node / "bin/node"), str(root / "node_modules/playwright/cli.js"),
                    "install", "chromium"], env=env, check=True)
    if not (root / "python/imageio_ffmpeg").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(root / "python"),
                        "imageio-ffmpeg==0.6.0"], check=True)
    binary = next((root / "python/imageio_ffmpeg/binaries").glob("ffmpeg-linux*"))
    (root / "bin").mkdir(exist_ok=True)
    link = root / "bin/ffmpeg"
    if not link.exists():
        link.symlink_to(binary)
    env.update(PATH=str(root / "bin") + os.pathsep + env["PATH"], NODE_PATH=str(root / "node_modules"))
    test = "const {chromium}=require('playwright');(async()=>{const b=await chromium.launch({headless:true});const p=await b.newPage();await p.setContent('<p>recording environment check</p>');await p.screenshot({path:process.argv[1]});await b.close()})().catch(e=>{console.error(e);process.exit(1)})"
    subprocess.run([str(node / "bin/node"), "-e", test, str(root / "browser-check.png")], env=env, check=True)
    subprocess.run([str(link), "-version"], check=True, stdout=subprocess.DEVNULL)
    (root / "environment.json").write_text(json.dumps({key: env[key] for key in
        ("PATH", "NODE_PATH", "PLAYWRIGHT_BROWSERS_PATH")}, indent=2))
    print("PASS: server-local Chromium, Node and FFmpeg; no GPU jobs started")


if __name__ == "__main__":
    main()
