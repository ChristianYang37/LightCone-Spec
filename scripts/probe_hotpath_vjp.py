"""Excluded operator microprobe; synthetic tensors are NOT benchmark/QA evidence."""

import argparse
import ast
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def definitions(path):
    names = {"_DFlashInferenceAttention", "_DFlashInferenceRMS", "_DFlashInferenceRoPE"}
    tree = ast.parse(path.read_text())
    namespace = {"torch": torch, "F": F}
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if len(nodes) != len(names):
        raise ValueError("runtime does not contain all native VJP classes")
    exec(compile(ast.Module(nodes, []), str(path), "exec"), namespace)
    return namespace


def measure(function):
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    wall = time.perf_counter()
    start.record()
    for _ in range(100):
        function()
    end.record()
    end.synchronize()
    return {"cuda_ms": start.elapsed_time(end) / 100,
            "wall_ms": (time.perf_counter() - wall) * 10,
            "incremental_peak_bytes": torch.cuda.max_memory_allocated() - base}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    torch.manual_seed(719)
    old, new = definitions(args.old), definitions(args.new)
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        def tensor(shape):
            return torch.randn(shape, device="cuda", dtype=dtype)
        for kv_heads in (2, 4):
            q, k, v = tensor((1, 4, 16, 128)), tensor((1, kv_heads, 32, 128)), tensor((1, kv_heads, 32, 128))
            mask = torch.ones(1, 1, 16, 32, dtype=torch.bool, device="cuda")
            mask[..., :3] = False
            context = SimpleNamespace(saved_tensors=(q, k, v, mask), scale=128**-.5,
                                      needs_input_grad=(True, True, True, False, False))
            grad = tensor(q.shape)
            functions = [lambda definition=definition: definition["_DFlashInferenceAttention"].backward(context, grad)
                         for definition in (old, new)]
            a, b = (function() for function in functions)
            errors = [float((x.float() - y.float()).abs().max()) for x, y in zip(a[:3], b[:3], strict=True)]
            # Record, do not silently loosen a gate or accept synthetic parity as full QA.
            rows.append({"operator": "attention", "dtype": str(dtype), "kv_heads": kv_heads,
                         "max_absolute_gradient_errors": errors,
                         "old": measure(functions[0]), "new": measure(functions[1])})
        for residual in (False, True):
            h, w, r = tensor((1, 16, 4096)), tensor((4096,)), tensor((1, 16, 4096))
            context = SimpleNamespace(saved_tensors=(h, w, r) if residual else (h, w),
                                      epsilon=1e-6, has_residual=residual,
                                      needs_input_grad=(True, True, False, residual))
            grad, rg = tensor(h.shape), tensor(h.shape) if residual else None
            functions = [lambda definition=definition: definition["_DFlashInferenceRMS"].backward(context, grad, rg)
                         for definition in (old, new)]
            a, b = (function() for function in functions)
            errors = [float((x.float() - y.float()).abs().max()) for x, y in zip(a, b, strict=True) if x is not None]
            rows.append({"operator": "rms", "dtype": str(dtype), "residual": residual,
                         "max_absolute_gradient_errors": errors,
                         "old": measure(functions[0]), "new": measure(functions[1])})
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"measurement_scope": "excluded_synthetic_operator_microprobe", "rank": rank,
               "gpu": torch.cuda.get_device_name(rank), "rows": rows, "formal_acceptance": False}
    (args.output / f"rank-{rank}.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
