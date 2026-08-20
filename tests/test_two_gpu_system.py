from __future__ import annotations

import json
import socket
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.system

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_MANIFEST = _PROJECT_ROOT / "manifests/runtime/industrial_compatibility_v1.json"


def _nccl_all_reduce_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    output_root: str,
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        value = torch.tensor(float(rank + 1), device=f"cuda:{rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(rank)
        properties = torch.cuda.get_device_properties(rank)
        payload = {
            "rank": rank,
            "value": float(value.cpu()),
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "memory_mib": properties.total_memory // (1024 * 1024),
        }
        target = Path(output_root) / f"rank-{rank}.json"
        target.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_declared_two_gpu_host_supports_nccl_collective(tmp_path: Path) -> None:
    """Prove the hardware collective only; TP2 serving remains fail-closed."""

    manifest = json.loads(_RUNTIME_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["gpu"]
    blocked_modes = {
        row["mode"]
        for row in manifest["release_capabilities"][
            "implemented_pending_dynamic_gpu_proof"
        ]
    }
    assert {"tp2", "dp2"}.issubset(blocked_modes)

    assert torch.cuda.is_available(), "the registered system host requires CUDA"
    assert dist.is_nccl_available(), "the registered system host requires NCCL"
    assert torch.cuda.device_count() == expected["count"] == 2

    rendezvous = f"tcp://127.0.0.1:{_free_loopback_port()}"
    mp.spawn(
        _nccl_all_reduce_worker,
        args=(2, rendezvous, str(tmp_path)),
        nprocs=2,
        join=True,
    )

    rows = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert [row["rank"] for row in rows] == [0, 1]
    assert [row["value"] for row in rows] == [3.0, 3.0]
    assert all(row["name"] == expected["model"] for row in rows)
    assert all(
        row["compute_capability"]
        == [int(part) for part in expected["compute_capability"].split(".")]
        for row in rows
    )
    assert all(row["memory_mib"] >= expected["minimum_memory_mib"] for row in rows)
