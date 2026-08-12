from __future__ import annotations

import copy
import hashlib
import importlib
import json
import socket
from pathlib import Path

from lightcone_spec import (
    PINNED_SGLANG_COMMIT,
    PINNED_SGLANG_PATCH_COUNT,
    PINNED_SGLANG_TREE,
)
from lightcone_spec.doctor import (
    _disk_report,
    _dns_probe,
    _parse_gpu_inventory,
    _parse_topology,
    doctor_report,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MANIFEST = ROOT / "manifests/runtime/industrial_compatibility_v1.json"


def _tree(path: Path, *, sglang: bool) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path),
        "is_git_checkout": True,
        "root_matches_toplevel": True,
        "head": "a" * 40,
        "tree": PINNED_SGLANG_TREE if sglang else "b" * 40,
        "dirty": False,
    }
    if sglang:
        value.update(
            {
                "pinned_ancestor": True,
                "patch_commits": PINNED_SGLANG_PATCH_COUNT,
            }
        )
    return value


def _dns(host: str, *, resolved: bool = True) -> dict[str, object]:
    return {
        "host": host,
        "resolved": resolved,
        "address_count": 1 if resolved else 0,
        "error": None if resolved else "offline",
    }


def _passing_facts(
    project: Path, sglang: Path, *, gpu_count: int = 2
) -> dict[str, object]:
    devices = [
        {
            "uuid": f"GPU-00000000-0000-0000-0000-00000000000{index}",
            "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "memory_total_mib": 97887,
            "driver_version": "580.65.06",
            "compute_capability": "12.0",
            "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
        }
        for index in range(gpu_count)
    ]
    gpu_rows = [f"GPU{index}" for index in range(gpu_count)]
    topology_pairs = [
        {
            "left": left,
            "right": right,
            "link": "PHB",
            "reciprocal_link": "PHB",
        }
        for left_index, left in enumerate(gpu_rows)
        for right in gpu_rows[left_index + 1 :]
    ]
    return {
        "python": {
            "version": "3.12.11",
            "major_minor": "3.12",
            "detail": "3.12.11 (mock)",
            "executable": "/runtime/bin/python",
        },
        "host": {
            "system": "Linux",
            "platform": "Linux-mock",
            "machine": "x86_64",
            "hostname": "gpu-host",
            "cpu_count": 32,
            "physical_memory_bytes": 256_000_000_000,
            "cpu_detail": "mock CPU",
        },
        "project_source_tree": _tree(project, sglang=False),
        "source_tree": _tree(sglang, sglang=True),
        "disk": {
            "path": str(project),
            "total_bytes": 300_000_000_000,
            "used_bytes": 100_000_000_000,
            "free_bytes": 200_000_000_000,
            "total_inodes": 1_000_000,
            "free_inodes": 900_000,
            "error": None,
        },
        "network": {
            host: _dns(host)
            for host in (
                "huggingface.co",
                "modelscope.cn",
                "pypi.tuna.tsinghua.edu.cn",
            )
        },
        "gpu": {
            "inventory_raw": "mock two-GPU inventory",
            "inventory": {"devices": devices, "parse_error": None},
            "topology_raw": "mock two-GPU topology",
            "topology": {
                "gpu_rows": gpu_rows,
                "pairs": topology_pairs,
                "parse_error": None,
            },
            "torch": {
                "importable": True,
                "import_error": None,
                "version": "2.11.0+cu130",
                "cuda_build": "13.0",
                "cuda_available": True,
                "device_count": gpu_count,
            },
        },
        "commands": {
            "nvidia_smi": "mock two-GPU inventory",
            "nvcc": "Cuda compilation tools, release 13.0, V13.0.88",
            "compiler": "g++ (Ubuntu) 13.2.0",
            "ninja": "1.11.1",
            "git": "git version 2.43.0",
        },
        "packages": {
            "torch": "2.11.0",
            "triton": "3.6.0",
            "sglang": "0.5.6.post2",
            "flashinfer-python": "0.6.15.post1",
            "nvidia-cutlass-dsl-libs-cu13": "4.7.0",
        },
        "sglang_import": {
            "importable": True,
            "origin": str(sglang / "python/sglang/__init__.py"),
        },
    }


def test_dns_probe_records_failure_without_raising(monkeypatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> list[object]:
        raise socket.gaierror("offline")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    assert _dns_probe("example.invalid") == {
        "host": "example.invalid",
        "resolved": False,
        "address_count": 0,
        "error": "gaierror",
    }


def test_disk_report_includes_bytes_inodes_and_probe_status(tmp_path) -> None:
    report = _disk_report(tmp_path)
    assert report["total_bytes"] > report["free_bytes"] > 0
    assert report["used_bytes"] >= 0
    assert report["error"] is None
    assert set(report) == {
        "path",
        "total_bytes",
        "used_bytes",
        "free_bytes",
        "total_inodes",
        "free_inodes",
        "error",
    }


def test_gpu_inventory_and_topology_parsers_preserve_exact_identity() -> None:
    inventory = _parse_gpu_inventory(
        "GPU-a, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, "
        "580.65.06, 12.0, 00000000:01:00.0\n"
        "GPU-b, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, "
        "580.65.06, 12.0, 00000000:02:00.0"
    )
    assert inventory["parse_error"] is None
    assert [device["uuid"] for device in inventory["devices"]] == [
        "GPU-a",
        "GPU-b",
    ]
    assert {device["memory_total_mib"] for device in inventory["devices"]} == {97887}
    topology = _parse_topology(
        "GPU0 GPU1 CPU Affinity\nGPU0 X PHB 0-31\nGPU1 PHB X 0-31\n"
    )
    assert topology == {
        "gpu_rows": ["GPU0", "GPU1"],
        "pairs": [
            {
                "left": "GPU0",
                "right": "GPU1",
                "link": "PHB",
                "reciprocal_link": "PHB",
            }
        ],
        "parse_error": None,
    }


def test_topology_parser_preserves_all_eight_gpu_pairs() -> None:
    gpu_rows = [f"GPU{index}" for index in range(8)]
    header = " ".join((*gpu_rows, "CPU", "Affinity"))
    rows = []
    for left_index, left in enumerate(gpu_rows):
        links = [
            "X" if left_index == right_index else "PHB" for right_index in range(8)
        ]
        rows.append(" ".join((left, *links, "0-31")))
    topology = _parse_topology("\n".join((header, *rows)))
    assert topology["parse_error"] is None
    assert topology["gpu_rows"] == gpu_rows
    assert len(topology["pairs"]) == 28
    assert {pair["link"] for pair in topology["pairs"]} == {"PHB"}


def test_runtime_manifest_is_canonical_and_sidecar_bound() -> None:
    value = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    assert Path(f"{RUNTIME_MANIFEST}.sha256").read_text().strip() == digest
    assert value["packages"] == {
        "flashinfer_python": {
            "cuda_extra": "cu13",
            "cuda_marker_distribution": "nvidia-cutlass-dsl-libs-cu13",
            "distribution": "flashinfer-python",
            "version": "0.6.15.post1",
        },
        "torch": "2.11.0",
        "triton": "3.6.0",
    }
    assert value["system"]["minimum_driver"] == "580.65.06"
    assert value["cuda"]["container_image"].startswith("nvidia/cuda:13.0.1-")
    assert value["sglang"] == {
        "expected_patch_count": PINNED_SGLANG_PATCH_COUNT,
        "expected_tree": PINNED_SGLANG_TREE,
        "patches": [
            {
                "file": ("0001-feat-spec-add-schema-v3-native-online-adaptation.patch"),
                "sha256": (
                    "3d29ea64ac1dc3d9ad2b869ba5ce685e60182f412d9346b5ed9a92ad99baba23"
                ),
            }
        ],
        "repository": "https://github.com/sgl-project/sglang.git",
        "upstream_commit": PINNED_SGLANG_COMMIT,
    }


def test_doctor_passes_only_exact_mocked_industrial_runtime(
    monkeypatch, tmp_path
) -> None:
    sglang = tmp_path / "patched-sglang"
    facts = _passing_facts(ROOT, sglang)
    observed_roots: list[tuple[Path, Path]] = []

    def collect(project_root: Path, sglang_root: Path) -> dict[str, object]:
        observed_roots.append((project_root, sglang_root))
        return facts

    monkeypatch.setattr("lightcone_spec.doctor._collect_facts", collect)
    report = doctor_report(ROOT, sglang)

    assert observed_roots == [(ROOT.resolve(), sglang.resolve())]
    assert report["status"] == "PASS"
    assert report["readiness"] == {
        "status": "PASS",
        "pass_count": len(report["checks"]),
        "fail_count": 0,
        "unknown_count": 0,
    }
    assert {check["status"] for check in report["checks"].values()} == {"PASS"}
    assert report["roots"] == {
        "project": str(ROOT.resolve()),
        "patched_sglang": str(sglang.resolve()),
        "distinct": True,
    }
    assert report["source_tree"]["tree"] == PINNED_SGLANG_TREE
    assert report["project_source_tree"]["tree"] != PINNED_SGLANG_TREE


def test_doctor_accepts_complete_same_host_eight_gpu_inventory(
    monkeypatch, tmp_path
) -> None:
    sglang = tmp_path / "patched-sglang"
    facts = _passing_facts(ROOT, sglang, gpu_count=8)
    monkeypatch.setattr("lightcone_spec.doctor._collect_facts", lambda *_args: facts)
    report = doctor_report(ROOT, sglang)
    assert report["status"] == "PASS"
    assert report["gpu"]["visible_gpu_count"] == 8
    assert report["gpu"]["gpu_pool_visible"] is True
    assert report["gpu"]["two_gpu_visible"] is False
    assert len(report["gpu"]["parsed_topology"]["pairs"]) == 28


def test_doctor_emits_fail_for_incompatible_known_facts(monkeypatch, tmp_path) -> None:
    sglang = tmp_path / "patched-sglang"
    facts = _passing_facts(ROOT, sglang)
    facts["python"]["major_minor"] = "3.11"
    facts["host"]["system"] = "Darwin"
    facts["packages"]["torch"] = "2.10.0"
    facts["commands"]["compiler"] = None
    facts["disk"]["free_bytes"] = 1_000
    facts["network"]["modelscope.cn"] = _dns("modelscope.cn", resolved=False)
    facts["network"]["huggingface.co"] = _dns("huggingface.co", resolved=False)
    facts["gpu"]["inventory"]["devices"][0]["name"] = "different GPU"
    facts["source_tree"]["tree"] = "0" * 40

    monkeypatch.setattr("lightcone_spec.doctor._collect_facts", lambda *_args: facts)
    report = doctor_report(ROOT, sglang)

    assert report["status"] == "FAIL"
    for name in (
        "python",
        "linux_host",
        "torch",
        "compiler",
        "disk",
        "network",
        "gpu_identity",
        "sglang_tree",
    ):
        assert report["checks"][name]["status"] == "FAIL"


def test_doctor_propagates_unknown_without_claiming_readiness(
    monkeypatch, tmp_path
) -> None:
    sglang = tmp_path / "patched-sglang"
    facts = _passing_facts(ROOT, sglang)
    facts["gpu"]["topology_raw"] = None
    facts["gpu"]["topology"] = {
        "gpu_rows": [],
        "pairs": [],
        "parse_error": None,
    }
    facts["disk"].update(
        {
            "total_bytes": None,
            "used_bytes": None,
            "free_bytes": None,
            "error": "OSError",
        }
    )
    monkeypatch.setattr("lightcone_spec.doctor._collect_facts", lambda *_args: facts)

    report = doctor_report(ROOT, sglang)
    assert report["status"] == "UNKNOWN"
    assert report["checks"]["gpu_topology"]["status"] == "UNKNOWN"
    assert report["checks"]["disk"]["status"] == "UNKNOWN"
    assert report["readiness"]["fail_count"] == 0
    assert report["readiness"]["unknown_count"] == 2


def test_local_mac_without_gpu_cannot_report_ready(monkeypatch, tmp_path) -> None:
    sglang = tmp_path / "patched-sglang"
    facts = _passing_facts(ROOT, sglang)
    facts["host"]["system"] = "Darwin"
    facts["gpu"]["inventory"] = {"devices": [], "parse_error": None}
    facts["gpu"]["inventory_raw"] = None
    facts["gpu"]["topology_raw"] = None
    facts["gpu"]["torch"].update(
        {
            "cuda_build": None,
            "cuda_available": False,
            "device_count": 0,
        }
    )
    monkeypatch.setattr("lightcone_spec.doctor._collect_facts", lambda *_args: facts)

    report = doctor_report(ROOT, sglang)
    assert report["status"] == "FAIL"
    assert report["checks"]["linux_host"]["status"] == "FAIL"
    assert report["checks"]["gpu_count"]["status"] == "FAIL"
    assert report["checks"]["driver"]["status"] == "UNKNOWN"
    assert report["checks"]["cuda_build"]["status"] == "FAIL"


def test_unsupported_release_modes_are_explicit(monkeypatch, tmp_path) -> None:
    sglang = tmp_path / "patched-sglang"
    facts = _passing_facts(ROOT, sglang)
    monkeypatch.setattr("lightcone_spec.doctor._collect_facts", lambda *_args: facts)
    report = doctor_report(ROOT, sglang)
    unsupported = report["mode_gates"]["unsupported"]
    assert {
        "multi_node",
        "tp2",
        "dp2",
        "static_speculative",
        "tts",
        "l0",
        "adaptive_dspark",
        "adaptive_eagle",
        "adaptive_eagle3",
        "adaptive_nextn",
        "adaptive_dflash_dspark",
        "quota_shadow_backend_acquisition",
    } == set(unsupported)
    assert {entry["status"] for entry in unsupported.values()} == {"UNSUPPORTED"}


def test_cli_accepts_distinct_project_and_sglang_roots(
    monkeypatch, tmp_path, capsys
) -> None:
    cli = importlib.import_module("lightcone_spec.cli.main")
    project = tmp_path / "project"
    sglang = tmp_path / "sglang"
    calls: list[tuple[str | Path, str | Path | None]] = []

    def format_mock(project_root: str | Path, sglang_root: str | Path | None) -> str:
        calls.append((project_root, sglang_root))
        return json.dumps({"status": "UNKNOWN"})

    monkeypatch.setattr(cli, "format_doctor", format_mock)
    assert (
        cli.main(
            [
                "doctor",
                "--project-root",
                str(project),
                "--sglang-root",
                str(sglang),
            ]
        )
        == 42
    )
    assert calls == [(str(project), str(sglang))]
    assert json.loads(capsys.readouterr().out) == {"status": "UNKNOWN"}


def test_sidecar_bound_but_malformed_manifest_fails_closed(
    monkeypatch, tmp_path
) -> None:
    value = copy.deepcopy(json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8")))
    del value["gpu"]["model"]
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    manifest = tmp_path / "manifests/runtime/industrial_compatibility_v1.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(value), encoding="utf-8")
    Path(f"{manifest}.sha256").write_text(
        f"{hashlib.sha256(canonical).hexdigest()}\n", encoding="utf-8"
    )
    sglang = tmp_path / "patched-sglang"
    facts = _passing_facts(tmp_path, sglang)
    monkeypatch.setattr("lightcone_spec.doctor._collect_facts", lambda *_args: facts)

    report = doctor_report(tmp_path, sglang)
    assert report["status"] == "FAIL"
    assert report["checks"]["compatibility_manifest"]["status"] == "FAIL"
    assert report["runtime_manifest"]["error"] == ("missing_or_invalid_required_field")
    assert report["mode_gates"]["source"] == "fail_closed_builtin"
