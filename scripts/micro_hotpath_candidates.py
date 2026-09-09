"""Excluded two-rank fixed-state microchecks for five optimization directions."""
import argparse
import ast
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightcone_spec.hotpath_candidates import ADAPTER, transform


def function(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module([node], []), name, "exec"), namespace)
    return namespace[name]


def elapsed(fn, repeats=30):
    for _ in range(10):
        fn()
    points = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(repeats):
            fn()
        b.record()
        b.synchronize()
        points.append(a.elapsed_time(b) / repeats)
    return statistics.median(points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    torch.manual_seed(20260910 + rank)
    device = f"cuda:{rank}"
    results = {}
    try:
        # Direction 1: real local Qwen projection geometry, never claim that
        # reassociating matmuls preserves BF16 rounding or the gradient target.
        x = torch.randn(16, 4096, device=device, dtype=torch.bfloat16)
        w = torch.randn(3072, 4096, device=device, dtype=torch.bfloat16) * .02
        a = (torch.randn(8, 4096, device=device) * .02).requires_grad_()
        b = (torch.randn(3072, 8, device=device) * .02).requires_grad_()
        def dense():
            return F.linear(x, (w.float() + b @ a).to(x.dtype))
        def factored():
            return F.linear(x, w) + F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype))
        y, z = dense(), factored()
        cotangent = torch.randn_like(y)
        gy = torch.autograd.grad(y, (a, b), cotangent)
        gz = torch.autograd.grad(z, (a, b), cotangent)
        results["factorized_lora"] = {"forward_exact": torch.equal(y, z),
            "relative_rms": float((y.float()-z.float()).square().mean().sqrt()/y.float().square().mean().sqrt()),
            "gradient_exact": all(torch.equal(v, u) for v, u in zip(gy, gz, strict=True)),
            "old_ms": elapsed(dense), "new_ms": elapsed(factored),
            "deployment": "not_accepted_without_real_state_numerical_review"}
        # Direction 2: NHD gather preserves order, repeated indexes and bytes.
        buffer = torch.randn(40960, 4, 128, device=device, dtype=torch.bfloat16)
        locations = torch.randint(0, 40960, (1, 512), device=device)
        def gather_old():
            return buffer[locations]
        def gather_new():
            return buffer.index_select(0, locations.reshape(-1)).reshape(*locations.shape, *buffer.shape[1:])
        assert torch.equal(gather_old(), gather_new())
        results["kv_index_select"] = {"exact": True, "old_ms": elapsed(gather_old), "new_ms": elapsed(gather_new)}
        # Direction 3: reuse one nonzero result; all gate outputs/decisions
        # must match, including empty supervision and NaN only on padding.
        source = (args.baseline / ADAPTER).read_text()
        old_gate = function(source, "_logit_reconstruction_gate")
        new_gate = function(transform(source, "single_mask_index"), "_logit_reconstruction_gate")
        logits = torch.randn(1, 15, 151936, device=device, dtype=torch.bfloat16)
        replay = logits.clone()
        mask = torch.arange(15, device=device)[None, :] < 11
        cases = []
        for kind in ("same", "padding_nan", "valid_mismatch", "empty"):
            altered = replay.clone()
            selected = mask if kind != "empty" else torch.zeros_like(mask)
            if kind == "padding_nan":
                altered[:, 11:] = float("nan")
            elif kind == "valid_mismatch":
                altered[:, :11] *= 2
            old, new = old_gate(logits, altered, valid_mask=selected), new_gate(logits, altered, valid_mask=selected)
            for u, v in zip(old, new, strict=True):
                torch.testing.assert_close(u, v, rtol=0, atol=0, equal_nan=True)
            assert bool(new[0]) == (kind in ("same", "padding_nan"))
            cases.append(kind)
        results["single_mask_index"] = {"exact": True, "cases": cases,
            "old_ms": elapsed(lambda: old_gate(logits, replay, valid_mask=mask)),
            "new_ms": elapsed(lambda: new_gate(logits, replay, valid_mask=mask))}
        # Direction 4: actual NCCL TP2, preserve rank concatenation and both
        # source/replay tensors while sharing one collective invocation.
        local = torch.randn(1, 15, 75968, device=device, dtype=torch.bfloat16)
        active = local + .125
        def gather(value):
            shards = [torch.empty_like(value) for _ in range(2)]
            dist.all_gather(shards, value.contiguous())
            return torch.cat(shards, dim=-1)
        def pair_old():
            return gather(local), gather(active)
        def pair_new():
            return gather(torch.stack((local, active))).unbind(0)
        for u, v in zip(pair_old(), pair_new(), strict=True):
            assert torch.equal(u, v)
        results["paired_logits_gather"] = {"exact": True, "old_ms": elapsed(pair_old), "new_ms": elapsed(pair_new)}
        # Direction 5: physical lengths need not be copied only to be replaced
        # by the exact existing logical prefix. No new gate or publication rule.
        row = torch.empty(1, 5, device=device, dtype=torch.int64)
        physical = torch.tensor([36870], device=device, dtype=torch.int32)
        committed = torch.tensor(36864, device=device)
        def prefix_old():
            row.fill_(-1)
            row[:, 0].copy_(physical.to(torch.int64))
            row[0, 0].copy_(committed)
            return row
        def prefix_new():
            row.fill_(-1)
            row[0, 0].copy_(committed)
            return row
        assert torch.equal(prefix_old().clone(), prefix_new())
        results["logical_prefix_write"] = {"exact": True, "old_ms": elapsed(prefix_old, 200), "new_ms": elapsed(prefix_new, 200)}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "scope": "excluded_micro_not_throughput", "results": results}, indent=2))
        print(json.dumps({"rank": rank, "results": results}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
