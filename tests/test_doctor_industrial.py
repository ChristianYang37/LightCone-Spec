from __future__ import annotations

import copy
import hashlib
import importlib
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

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
from lightcone_spec.experiments.registry import (
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.stage_capacity import (
    STAGE_CAPACITY_GATE_PROTOCOL_SHA256,
    StageCapacityGate,
    StageCapacitySchedule,
)
from lightcone_spec.runtime.control_attestation import ControlArtifactAttestation

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
        "project_runtime_source": {
            "schema_version": 1,
            "kind": "lightcone_project_runtime_source",
            "valid": True,
            "file_count": 1,
            "total_size_bytes": 1,
            "content_sha256": "c" * 64,
        },
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


def test_topology_parser_accepts_redirected_nvidia_smi_sgr_header_only() -> None:
    topology = _parse_topology(
        "\t\x1b[4mGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
        "GPU0\t X \tNODE\t0-51,104-155\t0\t\tN/A\n"
        "GPU1\tNODE\t X \t0-51,104-155\t0\t\tN/A\n"
    )
    assert topology == {
        "gpu_rows": ["GPU0", "GPU1"],
        "pairs": [
            {
                "left": "GPU0",
                "right": "GPU1",
                "link": "NODE",
                "reciprocal_link": "NODE",
            }
        ],
        "parse_error": None,
    }


@pytest.mark.parametrize("control", ("\x00", "\x1b[2J", "\x9b"))
def test_topology_parser_rejects_non_sgr_control_sequences(control: str) -> None:
    topology = _parse_topology(
        f"{control}GPU0 GPU1 CPU Affinity\nGPU0 X NODE 0-31\nGPU1 NODE X 0-31\n"
    )
    assert topology == {
        "gpu_rows": [],
        "pairs": [],
        "parse_error": "unsupported_control_character",
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
                    "091c8a164007691a171a449552a98ff6d68039cf868721bb24480e3ead4018e0"
                ),
            },
            {
                "file": ("0002-feat-spec-add-cpu-native-token-observations.patch"),
                "sha256": (
                    "8b9cf1f19277c765482eb0afe6b31a76f4aa8bd93e6cf29fd0e019a01011ac31"
                ),
            },
            {
                "file": (
                    "0003-feat-session-account-source-owned-http-connections.patch"
                ),
                "sha256": (
                    "fa723d63c031ce01f7354dfeda7b89dd4e7fa06d6fea0ceb81864918fdec1fde"
                ),
            },
            {
                "file": ("0004-feat-session-bind-native-trace-lifecycle.patch"),
                "sha256": (
                    "d640d5fe0ac55cb542d0b885fac51d64dc6a83a142ac52480cbfc99d9b866b6e"
                ),
            },
            {
                "file": ("0005-feat-compile-source-owned-lifecycle.patch"),
                "sha256": (
                    "05ab7ae2074f2e9ffa2387f1897e85ea1527a6daf44e1527dfd908adfb547f12"
                ),
            },
            {
                "file": ("0006-fix-metrics-handle-target-only-draft-width.patch"),
                "sha256": (
                    "8b0d05ba862fb0a9ec02092a35990ed487d56e294eb7b10d210c67ca1e84b163"
                ),
            },
            {
                "file": ("0007-feat-spec-bind-distributed-runtime-readiness.patch"),
                "sha256": (
                    "38b5ec81b9d75950558f8c72c1297bab47badf89d855b3e13dc1ad1c639f7d95"
                ),
            },
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


def _canonical_doctor_source(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def test_doctor_uses_path_bound_stage_capacity_below_100gb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sglang = tmp_path / "patched-sglang"
    facts = _passing_facts(ROOT, sglang)
    facts["disk"].update(
        {
            "used_bytes": 270_000_000_000,
            "free_bytes": 30_000_000_000,
        }
    )
    monkeypatch.setattr("lightcone_spec.doctor._collect_facts", lambda *_args: facts)
    assert doctor_report(ROOT, sglang)["checks"]["disk"]["status"] == "FAIL"

    gate_value = {"kind": "doctor-stage-capacity-fixture", "value": "gate"}
    schedule_value = {
        "kind": "doctor-stage-capacity-fixture",
        "value": "schedule",
    }
    control_value = {"kind": "doctor-stage-capacity-fixture", "value": "control"}
    gate_sha256 = content_sha256(gate_value)
    schedule_sha256 = content_sha256(schedule_value)
    control_sha256 = content_sha256(control_value)
    activation_sha256 = "a" * 64
    inventory_sha256 = "b" * 64
    control_lineage_sha256 = "c" * 64
    gate = SimpleNamespace(
        schema_version=3,
        experiment="preflight",
        schedule_sha256=schedule_sha256,
        mode="SIGNED_STAGE_ENVELOPE",
        status="AVAILABLE",
        reason_code="signed_stage_capacity_verified",
        required_free_bytes=20_000_000_000,
        observed_free_bytes=30_000_000_000,
        retained_evidence_bytes=10_000_000_000,
        maximum_concurrent_transient_bytes=5_000_000_000,
        safety_margin_bytes=5_000_000_000,
        sha256=gate_sha256,
    )
    schedule = SimpleNamespace(
        sha256=schedule_sha256,
        gpu_inventory_sha256=inventory_sha256,
    )
    control = SimpleNamespace(
        sha256=control_sha256,
        subject=SimpleNamespace(
            artifact_type="capacity",
            artifact_sha256=gate_sha256,
            protocol_sha256=STAGE_CAPACITY_GATE_PROTOCOL_SHA256,
            registry_sha256=build_industrial_registry().sha256,
            lineage_sha256=control_lineage_sha256,
        ),
    )
    monkeypatch.setattr(
        StageCapacityGate,
        "from_dict",
        classmethod(lambda _cls, _value: gate),
    )
    monkeypatch.setattr(
        StageCapacitySchedule,
        "from_dict",
        classmethod(lambda _cls, _value: schedule),
    )
    monkeypatch.setattr(
        ControlArtifactAttestation,
        "from_dict",
        classmethod(lambda _cls, _value: control),
    )
    monkeypatch.setattr(
        "lightcone_spec.experiments.stage_capacity.stage_capacity_control_lineage_sha256",
        lambda **_kwargs: control_lineage_sha256,
    )
    monkeypatch.setattr(
        "lightcone_spec.runtime.control_attestation.verify_release_control_artifact_attestation",
        lambda *_args, **_kwargs: SimpleNamespace(challenge_sha256="d" * 64),
    )
    monkeypatch.setattr(
        "lightcone_spec.experiments.stage_capacity.revalidate_stage_capacity_gate_sources",
        lambda *_args, **_kwargs: object(),
    )
    gate_path = _canonical_doctor_source(tmp_path / "capacity-gate.json", gate_value)
    schedule_path = _canonical_doctor_source(
        tmp_path / "capacity-schedule.json", schedule_value
    )
    control_path = _canonical_doctor_source(
        tmp_path / "capacity-control.json", control_value
    )
    report = doctor_report(
        ROOT,
        sglang,
        stage_capacity_gate_path=gate_path,
        stage_capacity_schedule_path=schedule_path,
        stage_capacity_attestation_path=control_path,
        stage_capacity_activation_sha256=activation_sha256,
        stage_capacity_now_ns=1,
    )
    assert report["checks"]["disk"]["status"] == "PASS"
    assert report["checks"]["disk"]["expected"]["minimum_free_bytes"] == (
        20_000_000_000
    )
    assert report["stage_capacity"]["gate_sha256"] == gate_sha256


def test_doctor_never_falls_back_after_stage_capacity_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sglang = tmp_path / "patched-sglang"
    facts = _passing_facts(ROOT, sglang)
    monkeypatch.setattr("lightcone_spec.doctor._collect_facts", lambda *_args: facts)
    gate_value = {"kind": "doctor-stage-capacity-fixture", "value": "gate"}
    schedule_value = {
        "kind": "doctor-stage-capacity-fixture",
        "value": "schedule",
    }
    control_value = {"kind": "doctor-stage-capacity-fixture", "value": "control"}
    gate_sha256 = content_sha256(gate_value)
    schedule_sha256 = content_sha256(schedule_value)
    control_sha256 = content_sha256(control_value)
    gate = SimpleNamespace(
        schema_version=3,
        experiment="preflight",
        schedule_sha256=schedule_sha256,
        mode="SIGNED_STAGE_ENVELOPE",
        status="AVAILABLE",
        reason_code="signed_stage_capacity_verified",
        required_free_bytes=1,
        observed_free_bytes=200_000_000_000,
        retained_evidence_bytes=0,
        maximum_concurrent_transient_bytes=0,
        safety_margin_bytes=1,
        sha256=gate_sha256,
    )
    monkeypatch.setattr(
        StageCapacityGate,
        "from_dict",
        classmethod(lambda _cls, _value: gate),
    )
    monkeypatch.setattr(
        StageCapacitySchedule,
        "from_dict",
        classmethod(
            lambda _cls, _value: SimpleNamespace(
                sha256=schedule_sha256,
                gpu_inventory_sha256="b" * 64,
            )
        ),
    )
    monkeypatch.setattr(
        ControlArtifactAttestation,
        "from_dict",
        classmethod(
            lambda _cls, _value: SimpleNamespace(
                sha256=control_sha256,
                subject=SimpleNamespace(),
            )
        ),
    )

    def tampered(*_args: object, **_kwargs: object) -> object:
        raise ValueError("capacity source changed")

    monkeypatch.setattr(
        "lightcone_spec.experiments.stage_capacity.revalidate_stage_capacity_gate_sources",
        tampered,
    )
    report = doctor_report(
        ROOT,
        sglang,
        stage_capacity_gate_path=_canonical_doctor_source(
            tmp_path / "tampered-gate.json", gate_value
        ),
        stage_capacity_schedule_path=_canonical_doctor_source(
            tmp_path / "tampered-schedule.json", schedule_value
        ),
        stage_capacity_attestation_path=_canonical_doctor_source(
            tmp_path / "tampered-control.json", control_value
        ),
        stage_capacity_activation_sha256="a" * 64,
        stage_capacity_now_ns=1,
    )
    assert report["checks"]["disk"]["status"] == "FAIL"
    assert report["stage_capacity"]["reason_code"] == (
        "stage_capacity_authority_invalid"
    )
    assert report["stage_capacity"]["mode"] != "LEGACY_100GB_FALLBACK"


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
        "adaptive_eagle",
        "adaptive_dflash_dspark",
    } == set(unsupported)
    assert {entry["status"] for entry in unsupported.values()} == {"UNSUPPORTED"}
    pending = report["mode_gates"]["pending_dynamic_gpu_proof"]
    assert {
        "tp2",
        "dp2",
        "adaptive_dspark",
        "adaptive_nextn",
        "adaptive_chronobelief",
        "native_itl",
        "fixed_address_publication_graph",
        "session_reuse",
    } == set(pending)
    assert {entry["status"] for entry in pending.values()} == {"BLOCKED"}
    assert {entry["implementation_status"] for entry in pending.values()} == {
        "IMPLEMENTED_PENDING_DYNAMIC_GPU_PROOF"
    }
    external = report["mode_gates"]["external_authority_required"]
    assert {
        "static_speculative",
        "tts",
        "l0",
        "adaptive_eagle3",
        "quota_shadow_backend_acquisition",
    } == set(external)
    assert report["mode_gates"]["available_after_dynamic_proof"] == {}


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
