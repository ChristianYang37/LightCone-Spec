from __future__ import annotations

import math
import multiprocessing
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
import torch

pytestmark = pytest.mark.gpu


def _require_cuda_devices(count: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < count:
        pytest.skip(f"this CUDA contract requires at least {count} GPUs")


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _nccl_rank_worker(rank: int, world_size: int, port: int, queue) -> None:
    try:
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        value = torch.tensor(float(rank + 1), device=f"cuda:{rank}")
        torch.distributed.all_reduce(value)
        torch.cuda.synchronize(rank)
        queue.put((rank, float(value.cpu()), torch.cuda.current_device(), None))
    except (OSError, RuntimeError, ValueError) as error:  # pragma: no cover
        queue.put((rank, None, None, f"{type(error).__name__}: {error}"))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _timed_gemm(
    device_index: int, barrier: threading.Barrier | None
) -> tuple[float, float]:
    with torch.cuda.device(device_index):
        left = torch.randn((512, 512), device=f"cuda:{device_index}")
        right = torch.randn((512, 512), device=f"cuda:{device_index}")
        stream = torch.cuda.Stream(device=device_index)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if barrier is not None:
            barrier.wait(timeout=30)
        with torch.cuda.stream(stream):
            start.record(stream)
            result = left
            for _ in range(8):
                result = result @ right
                result = result / result.norm().clamp_min(1.0)
            end.record(stream)
        end.synchronize()
        return float(start.elapsed_time(end)), float(result.sum().cpu())


def test_real_cuda_reset_restores_rng_and_stabilizes_fixed_allocations() -> None:
    _require_cuda_devices(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    active = torch.zeros((1024, 1024), device=device, dtype=torch.float16)
    candidate = torch.ones_like(active)
    pointer = active.data_ptr()
    rng_state = torch.cuda.get_rng_state(device)
    first_random = torch.rand((4096,), device=device)
    torch.cuda.set_rng_state(rng_state, device)
    replay_random = torch.rand((4096,), device=device)
    torch.testing.assert_close(first_random, replay_random, rtol=0.0, atol=0.0)

    active.copy_(candidate)
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    for value in range(16):
        candidate.fill_(float(value))
        active.copy_(candidate)
    torch.cuda.synchronize(device)

    assert active.data_ptr() == pointer
    assert torch.cuda.memory_allocated(device) == allocated
    assert torch.cuda.memory_reserved(device) == reserved
    assert torch.cuda.max_memory_allocated(device) >= allocated
    torch.testing.assert_close(active, candidate)


def test_real_cuda_graph_replay_preserves_addresses_values_and_hbm() -> None:
    _require_cuda_devices(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    static_input = torch.zeros((4096,), device=device)
    static_output = torch.empty_like(static_input)
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            torch.add(static_input, 2.0, out=static_output)
    torch.cuda.current_stream(device).wait_stream(warmup_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.add(static_input, 2.0, out=static_output)
    graph.replay()
    torch.cuda.synchronize(device)
    input_pointer = static_input.data_ptr()
    output_pointer = static_output.data_ptr()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)

    for value in range(1, 17):
        static_input.fill_(float(value))
        graph.replay()
    torch.cuda.synchronize(device)

    assert static_input.data_ptr() == input_pointer
    assert static_output.data_ptr() == output_pointer
    assert torch.cuda.memory_allocated(device) == allocated
    assert torch.cuda.memory_reserved(device) == reserved
    torch.testing.assert_close(static_output, torch.full_like(static_output, 18.0))


def test_real_two_rank_nccl_gang_is_atomic_across_processes() -> None:
    _require_cuda_devices(2)
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        pytest.skip("NCCL process groups are unavailable")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    port = _free_loopback_port()
    processes = tuple(
        context.Process(target=_nccl_rank_worker, args=(rank, 2, port, queue))
        for rank in range(2)
    )
    for process in processes:
        process.start()
    rows = tuple(queue.get(timeout=90) for _ in processes)
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    assert {row[0] for row in rows} == {0, 1}
    assert all(row[3] is None for row in rows)
    assert {row[2] for row in rows} == {0, 1}
    assert all(row[1] == pytest.approx(3.0) for row in rows)


def test_real_simultaneous_gpu_smoke_records_no_synthetic_interference_pass() -> None:
    """Exercise real co-run timing without inferring a safe concurrency rule."""

    _require_cuda_devices(2)
    isolated = tuple(_timed_gemm(index, None) for index in range(2))
    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(
            executor.submit(_timed_gemm, index, barrier) for index in range(2)
        )
        simultaneous = tuple(future.result(timeout=90) for future in futures)

    for duration_ms, checksum in (*isolated, *simultaneous):
        assert math.isfinite(duration_ms) and duration_ms > 0
        assert math.isfinite(checksum)
    # This smoke deliberately has no performance threshold.  Only a registered
    # raw-evidence reducer may produce an InterferenceEnvelope PASS rule.
