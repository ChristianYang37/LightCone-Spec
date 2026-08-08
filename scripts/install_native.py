#!/usr/bin/env python3
"""User-local native installer for the pinned LightCone/SGLang workspace.

The default is a dry run.  ``--execute`` is required before any environment
or package is created.  The script never invokes sudo or changes system CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 bootstrap before uv creates 3.12.
    tomllib = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcone_spec.doctor import SUPPORTED_CUDA as SUPPORTED

SGLANG_REPOSITORY = "https://github.com/sgl-project/sglang.git"
SGLANG_UPSTREAM_COMMIT = "3312645a307453893a00778592f105581e3d1c3d"
SGLANG_PATCHED_TREE = "9e32eb1872a963cb3981710079101fe1f4f8fe0a"

PINNED_CU129_WHEELS = {
    "sglang_kernel-0.4.5+cu129-cp310-abi3-manylinux2014_x86_64.whl": {
        "url": "https://github.com/sgl-project/whl/releases/download/v0.4.5/"
        "sglang_kernel-0.4.5+cu129-cp310-abi3-manylinux2014_x86_64.whl",
        "sha256": "aa09af4599121558f813e7089caa9d4ff43e7a31b3794c0e48523085484484f6",
    },
    "sgl_deep_gemm-0.1.5+cu129-py3-none-manylinux2014_x86_64.whl": {
        "url": "https://github.com/sgl-project/whl/releases/download/v0.1.5/"
        "sgl_deep_gemm-0.1.5+cu129-py3-none-manylinux2014_x86_64.whl",
        "sha256": "6ee67b2cc19b3227a376f6cd2b4f8c81bbdbfad1a9c39c984ba4fd6333e8f7c2",
    },
}

# These versions are the non-empty intersection of the CUDA-12 CUTLASS and
# SGLang model-gateway requirements.  Installing dependencies in bounded
# chunks otherwise lets a later unconstrained package downgrade CUTLASS or
# upgrade protobuf past the version accepted by the CUDA-12 libraries.
CU12_COMPATIBILITY_PINS = (
    "cuda-python==12.9.7",
    "nvidia-cutlass-dsl==4.6.0",
    "nvidia-cutlass-dsl-libs-base==4.6.0",
    "numpy==2.3.5",
    "fsspec==2026.6.0",
    "setuptools==81.0.0",
    "protobuf==6.33.5",
    "grpcio==1.82.0",
    "grpcio-health-checking==1.82.0",
    "grpcio-reflection==1.82.0",
)


def _pip_check(pip: str, *, execute: bool, allow_sglang_cuda12: bool) -> None:
    argv = [pip, "-m", "pip", "check"]
    print("+", " ".join(argv))
    if not execute:
        return
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode == 0:
        if result.stdout:
            print(result.stdout, end="")
        print("dependency check passed")
        return
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    allowed_count = 0
    if allow_sglang_cuda12:
        retained = []
        for line in lines:
            allowed = (
                line.startswith("sglang ")
                and "has requirement cuda-python>=13.0" in line
                and "but you have cuda-python 12.9.7" in line
            )
            if allowed:
                allowed_count += 1
            else:
                retained.append(line)
        lines = retained
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if lines or result.stderr.strip() or (result.returncode and allowed_count != 1):
        print("\n".join(lines), file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode or 1, argv)
    print("dependency check passed (audited CUDA-12 metadata transform applied)")


def _run(argv: list[str], *, execute: bool, env=None) -> None:
    print("+", " ".join(argv))
    if execute:
        subprocess.run(argv, check=True, env=env)


def _driver_cuda() -> tuple[int, int]:
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi is unavailable; run doctor before installing")
    output = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, timeout=20, check=True
    ).stdout
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    if not match:
        raise SystemExit("cannot parse maximum CUDA version from nvidia-smi")
    return int(match.group(1)), int(match.group(2))


def _toolkit_cuda() -> tuple[int, int]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise SystemExit(
            "nvcc is unavailable; the pinned SGLang stack needs a local CUDA "
            "toolkit for JIT compilation"
        )
    output = subprocess.run(
        [nvcc, "--version"], capture_output=True, text=True, timeout=20, check=True
    ).stdout
    match = re.search(r"release\s+(\d+)\.(\d+)", output)
    if not match:
        raise SystemExit("cannot parse CUDA toolkit version from nvcc --version")
    return int(match.group(1)), int(match.group(2))


def _select_cuda(requested: str) -> tuple[str, str]:
    driver_maximum = _driver_cuda()
    toolkit_maximum = _toolkit_cuda()
    maximum = min(driver_maximum, toolkit_maximum)
    if requested != "auto":
        wanted = next((x for x in SUPPORTED if x[2] == requested), None)
        if wanted is None:
            raise SystemExit(f"unsupported --cuda {requested}")
        if maximum < wanted[:2]:
            raise SystemExit(
                f"usable CUDA is {maximum[0]}.{maximum[1]} "
                f"(driver {driver_maximum[0]}.{driver_maximum[1]}, toolkit "
                f"{toolkit_maximum[0]}.{toolkit_maximum[1]}), below {requested}"
            )
        return f"{wanted[0]}.{wanted[1]}", wanted[2]
    for major, minor, tag in SUPPORTED:
        if maximum >= (major, minor):
            return f"{major}.{minor}", tag
    raise SystemExit(
        f"usable CUDA {maximum[0]}.{maximum[1]} (driver "
        f"{driver_maximum[0]}.{driver_maximum[1]}, toolkit "
        f"{toolkit_maximum[0]}.{toolkit_maximum[1]}) is below supported CUDA 12.9"
    )


def _transform_dependency(dep: str, cuda_tag: str) -> str | None:
    name = re.split(r"[<>=!~; \[]", dep, maxsplit=1)[0].lower().replace("_", "-")
    if name in {
        "torch",
        "torchvision",
        "torchaudio",
        "sglang-kernel",
        "sgl-deep-gemm",
    }:
        return None
    if cuda_tag.startswith("cu12"):
        dep = dep.replace("cuda-python>=13.0", "cuda-python>=12.8,<13")
        dep = dep.replace("flashinfer_python[cu13]", "flashinfer_python[cu12]")
        # Upstream's CUDA-12 image keeps the base humming package but removes
        # the cu13 extra.  A cu12 extra is not guaranteed to be published.
        dep = dep.replace("humming-kernels[cu13]", "humming-kernels")
        dep = dep.replace("nvidia-cutlass-dsl[cu13]", "nvidia-cutlass-dsl")
    return dep


def _capture(argv: list[str]) -> str:
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=30, check=True
    ).stdout.strip()


def _capture_porcelain(argv: list[str]) -> str:
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=30, check=True
    ).stdout.rstrip("\r\n")


def _resolve_pinned_wheel(
    filename: str, *, wheelhouse: Path | None
) -> tuple[str, dict]:
    """Use an exact local wheel when supplied, otherwise its pinned URL."""
    artifact = PINNED_CU129_WHEELS.get(filename)
    if artifact is None:
        if wheelhouse is not None:
            raise SystemExit(f"no pinned wheel hash for offline artifact {filename}")
        if filename.startswith("sglang_kernel-0.4.5+cu129-"):
            url = "https://github.com/sgl-project/whl/releases/download/v0.4.5/" + filename
        elif filename.startswith("sgl_deep_gemm-0.1.5+cu129-"):
            url = "https://github.com/sgl-project/whl/releases/download/v0.1.5/" + filename
        else:
            raise SystemExit(f"no pinned wheel metadata for {filename}")
        return url, {"filename": filename, "source": url, "sha256": None}
    if wheelhouse is None:
        return artifact["url"], {
            "filename": filename,
            "source": artifact["url"],
            "sha256": artifact["sha256"],
        }
    path = wheelhouse / filename
    if not path.is_file():
        raise SystemExit(f"pinned wheel is missing from --wheelhouse: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != artifact["sha256"]:
        raise SystemExit(
            f"pinned wheel hash drift for {path}: {actual} != {artifact['sha256']}"
        )
    return str(path), {
        "filename": filename,
        "source": str(path),
        "sha256": actual,
    }


def _prepare_sglang_source(
    spec_root: Path, runtime_root: Path, *, execute: bool
) -> tuple[Path, dict]:
    """Create a disposable checkout from the exact upstream plus patchset."""
    apply_script = spec_root / "patches" / "sglang" / "apply.sh"
    manifest_path = spec_root / "patches" / "sglang" / "manifest.json"
    if not apply_script.is_file() or not manifest_path.is_file():
        raise SystemExit("the verified SGLang patchset is missing")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("upstream", {}).get("commit") != SGLANG_UPSTREAM_COMMIT
        or manifest.get("expected_tree") != SGLANG_PATCHED_TREE
    ):
        raise SystemExit("SGLang patch manifest does not match installer pins")
    checkout = runtime_root / "source" / "sglang"
    provenance = {
        "mode": "disposable_patched_checkout",
        "repository": SGLANG_REPOSITORY,
        "upstream_commit": SGLANG_UPSTREAM_COMMIT,
        "tree": SGLANG_PATCHED_TREE,
    }
    if not execute:
        print(
            "+ git clone and checkout exact upstream, then",
            apply_script,
            checkout,
        )
        return checkout, provenance
    if checkout.exists():
        raise SystemExit(
            f"SGLang checkout already exists at {checkout}; choose a fresh runtime root"
        )
    checkout.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--filter=blob:none", SGLANG_REPOSITORY, str(checkout)], execute=True)
    _run(["git", "-C", str(checkout), "checkout", "--detach", SGLANG_UPSTREAM_COMMIT], execute=True)
    _run([str(apply_script), str(checkout)], execute=True)
    actual_tree = _capture(["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"])
    status = _capture_porcelain(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"]
    )
    if actual_tree != SGLANG_PATCHED_TREE or status:
        raise SystemExit("patched SGLang checkout failed final tree/cleanliness verification")
    return checkout, provenance


def _sglang_dependencies(pyproject_path: Path) -> list[str]:
    """Read ``project.dependencies`` without requiring a bootstrap package.

    ``tomllib`` is used on Python 3.11+.  On a minimal Python 3.10 host the
    fallback reads only the line-oriented quoted dependency array present in
    the pinned SGLang pyproject; JSON decoding handles its string escapes.
    """
    body = pyproject_path.read_text()
    if tomllib is not None:
        return list(tomllib.loads(body)["project"]["dependencies"])
    in_dependencies = False
    dependencies: list[str] = []
    decoder = json.JSONDecoder()
    for line in body.splitlines():
        stripped = line.strip()
        if not in_dependencies:
            if stripped == "dependencies = [":
                in_dependencies = True
            continue
        if stripped == "]":
            return dependencies
        if not stripped or stripped.startswith("#"):
            continue
        try:
            value, _ = decoder.raw_decode(stripped)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"cannot parse SGLang dependency line without tomllib: {line!r}"
            ) from exc
        if not isinstance(value, str):
            raise SystemExit(f"non-string SGLang dependency: {value!r}")
        dependencies.append(value)
    raise SystemExit("cannot find a complete project.dependencies array")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--runtime-root", default="~/lightcone-tts-runtime")
    parser.add_argument("--python", default="python3.12")
    parser.add_argument("--cuda", default="auto", choices=("auto", "cu129", "cu130"))
    parser.add_argument(
        "--rust-extensions",
        default="none",
        choices=("none", "all"),
        help=(
            "build optional SGLang Rust gRPC extensions; DSpark/LightCone "
            "does not require them"
        ),
    )
    parser.add_argument(
        "--wheelhouse",
        default=None,
        help="directory containing the exact pinned CUDA wheels for offline hosts",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    spec_root = workspace
    if not (spec_root / "pyproject.toml").is_file():
        raise SystemExit(f"workspace is not a LightCone-Spec source tree: {workspace}")
    for command in ("git", "gcc", "g++", "rustc", "cargo", "nvcc"):
        if shutil.which(command) is None:
            raise SystemExit(
                f"required compiler/tool {command!r} is unavailable; run doctor "
                "and install it outside this script (no sudo is attempted)"
            )
    python = shutil.which(args.python)
    uv = shutil.which("uv")
    if python is None and uv is None:
        raise SystemExit(
            f"{args.python} and uv are unavailable; install one in user space first"
        )
    cuda_version, cuda_tag = _select_cuda(args.cuda)
    runtime_root = Path(args.runtime_root).expanduser().resolve()
    sglang_root, sglang_source = _prepare_sglang_source(
        spec_root, runtime_root, execute=args.execute
    )
    wheelhouse = (
        None if args.wheelhouse is None else Path(args.wheelhouse).expanduser().resolve()
    )
    venv = runtime_root / "venv"
    bin_dir = venv / "bin"
    pip = str(bin_dir / "python")
    if args.execute and venv.exists():
        raise SystemExit(
            f"isolated environment already exists at {venv}; refusing to "
            "mutate an unknown environment (choose a fresh --runtime-root)"
        )
    print(
        json.dumps(
            {
                "workspace": str(workspace),
                "runtime_root": str(runtime_root),
                "cuda_version": cuda_version,
                "cuda_tag": cuda_tag,
                "environment_builder": "uv" if python is None else "venv",
                "execute": args.execute,
                "sglang_source": sglang_source,
            },
            indent=2,
        )
    )
    if args.execute:
        runtime_root.mkdir(parents=True, exist_ok=True)
    else:
        print("Dry run complete; pass --execute to create the isolated checkout and environment.")
        return 0
    if python is None:
        _run(
            [uv, "venv", "--python", "3.12", str(venv)],
            execute=args.execute,
        )
    else:
        _run([python, "-m", "venv", str(venv)], execute=args.execute)
    _run([pip, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], execute=args.execute)
    _run(
        [
            pip,
            "-m",
            "pip",
            "install",
            f"torch==2.11.0+{cuda_tag}",
            f"torchaudio==2.11.0+{cuda_tag}",
            f"torchvision==0.26.0+{cuda_tag}",
            "--index-url",
            f"https://download.pytorch.org/whl/{cuda_tag}",
            "--extra-index-url",
            os.environ.get("PIP_INDEX_URL", "https://pypi.org/simple"),
        ],
        execute=args.execute,
    )
    dependencies = [
        transformed
        for dep in _sglang_dependencies(sglang_root / "python" / "pyproject.toml")
        if (transformed := _transform_dependency(dep, cuda_tag)) is not None
    ]
    # Install in bounded chunks to avoid command-line length limits.
    for start in range(0, len(dependencies), 24):
        _run(
            [pip, "-m", "pip", "install", *dependencies[start : start + 24]],
            execute=args.execute,
        )
    arch = os.uname().machine
    wheel_artifacts: list[dict] = []
    if cuda_tag.startswith("cu12"):
        kernel_name = (
            f"sglang_kernel-0.4.5+cu129-cp310-abi3-manylinux2014_{arch}.whl"
        )
        deep_gemm_name = (
            f"sgl_deep_gemm-0.1.5+cu129-py3-none-manylinux2014_{arch}.whl"
        )
        kernel_source, kernel_artifact = _resolve_pinned_wheel(
            kernel_name, wheelhouse=wheelhouse
        )
        deep_gemm_source, deep_gemm_artifact = _resolve_pinned_wheel(
            deep_gemm_name, wheelhouse=wheelhouse
        )
        wheel_artifacts.extend((kernel_artifact, deep_gemm_artifact))
        _run(
            [
                pip,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                kernel_source,
            ],
            execute=args.execute,
        )
        _run(
            [
                pip,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                deep_gemm_source,
            ],
            execute=args.execute,
        )
    else:
        _run(
            [
                pip,
                "-m",
                "pip",
                "install",
                "sglang-kernel==0.4.5",
                "sgl-deep-gemm==0.1.5",
            ],
            execute=args.execute,
        )
    sglang_build_env = os.environ.copy()
    sglang_build_env["SGLANG_BUILD_RUST_EXTS"] = args.rust_extensions
    _run(
        [pip, "-m", "pip", "install", "--no-deps", "-e", str(sglang_root / "python")],
        execute=args.execute,
        env=sglang_build_env,
    )
    _run(
        [pip, "-m", "pip", "install", "-e", f"{spec_root}[gpu,dev]"],
        execute=args.execute,
    )
    if cuda_tag.startswith("cu12"):
        _run(
            [pip, "-m", "pip", "install", *CU12_COMPATIBILITY_PINS],
            execute=args.execute,
        )
    _pip_check(
        pip,
        execute=args.execute,
        allow_sglang_cuda12=cuda_tag.startswith("cu12"),
    )
    _run(
        [
            pip,
            "-c",
            "from importlib.metadata import version; "
            "import torch,sglang,lightcone_spec; "
            "assert torch.cuda.is_available(); "
            "print(torch.__version__, torch.version.cuda, version('sglang'))",
        ],
        execute=args.execute,
    )
    if args.execute:
        lock_path = runtime_root / "environment.lock.txt"
        with open(lock_path, "w", encoding="utf-8") as fh:
            subprocess.run([pip, "-m", "pip", "freeze"], stdout=fh, check=True)
        script_bytes = Path(__file__).read_bytes()
        provenance = {
            "schema_version": 1,
            "cuda_selection": {"version": cuda_version, "tag": cuda_tag},
            "python": _capture([pip, "--version"]),
            "sglang_upstream_commit": SGLANG_UPSTREAM_COMMIT,
            "sglang_patched_tree": SGLANG_PATCHED_TREE,
            "sglang_source": sglang_source,
            "sglang_rust_extensions": args.rust_extensions,
            "nvcc": _capture(["nvcc", "--version"]) if shutil.which("nvcc") else None,
            "gcc": _capture(["gcc", "--version"]) if shutil.which("gcc") else None,
            "installer_sha256": hashlib.sha256(script_bytes).hexdigest(),
            "environment_lock": str(lock_path),
            "pinned_wheels": wheel_artifacts,
        }
        (runtime_root / "environment.provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        print(f"environment lock written: {lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
