"""Excluded post-reset fingerprints, not a generation determinism claim."""

import torch

from .target_state_diagnostic import parameter_digests, tensor_digest


def reset_snapshot(adapter):
    if adapter.config.weight_update_mode != "lora" or adapter.request_slots is not None:
        raise ValueError("reset diagnostic supports single-session LoRA only")
    runtime, optimizer = adapter.runtime, adapter.optimizer
    if torch.device(adapter.device).type == "cuda":
        torch.cuda.synchronize(adapter.device)

    def digest(tensors):
        return [{"shape": list(t.shape), "dtype": str(t.dtype), "sha256": tensor_digest(t)} for t in tensors]

    state = {
        "initial": digest(adapter.initial_trainable), "master": digest(optimizer.master),
        "first": digest(optimizer.first), "second": digest(optimizer.second),
        "metadata": digest(optimizer.metadata_tensors),
        "active": digest(adapter.inference.active), "staging": digest(adapter.inference.staging),
        "base": digest(tuple(adapter.base.values())),
        "target": parameter_digests(adapter.worker.target_worker.model_runner.model),
    }
    checks = {
        "master_restored": all(torch.equal(x, y) for x, y in zip(optimizer.master, adapter.initial_trainable, strict=True)),
        "moments_zero": all(torch.count_nonzero(t).item() == 0 for t in optimizer.first + optimizer.second),
        "optimizer_step_zero": optimizer.step == 0,
        "active_restored": all(torch.equal(t, adapter.base[name].to(t.dtype))
                               for name, t in zip(adapter.names, adapter.inference.active, strict=True)),
        "staging_matches_active": all(torch.equal(a, b) for a, b in zip(adapter.inference.active, adapter.inference.staging, strict=True)),
        "runtime_empty": runtime.pending is None and runtime.reserved is None
            and not runtime.requests and not runtime.device_commits and not runtime.latest_device_lengths
            and not runtime.update_traces and not runtime.round_traces and not runtime.active_round_rows,
        "runtime_counters_reset": runtime.active_version == 0 and runtime.round == 0
            and runtime.active_request_id is None and all(v == 0 for v in runtime.counters.values()),
    }
    return {"scope": "excluded post-reset exact state; does not prove deterministic async publications",
            "state": state, "checks": checks, "passed": all(checks.values()),
            "epoch": runtime.epoch, "slot_generation": runtime.slot_generation}
