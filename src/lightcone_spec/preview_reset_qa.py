"""Excluded endpoint-only byte/value checks after real adapter reset."""

import functools
import json
from pathlib import Path


def check_reset(adapter):
    import torch
    optimizer = adapter.optimizer
    initial = getattr(adapter, "initial_trainable", None)
    if initial is None:
        initial = adapter.initial
    if len(initial) != len(optimizer.master):
        raise RuntimeError("reset QA parameter layout mismatch")
    master_ok = all(torch.equal(a, b.to(dtype=a.dtype, device=a.device))
                    for a, b in zip(optimizer.master, initial, strict=True))
    if getattr(optimizer, "is_ensemble", False):
        first_ok = all(torch.equal(a, b) for expert in optimizer.experts
                       for a, b in zip(expert, optimizer.master, strict=True))
    else:
        first_ok = all(torch.count_nonzero(t).item() == 0 for t in optimizer.first)
    second_ok = all(torch.count_nonzero(t).item() == 0 for t in optimizer.second)
    if hasattr(adapter, "_effective_parameters"):
        effective = adapter._effective_parameters(optimizer.master)
        expected = tuple(effective[name] for name in adapter.names)
    else:
        expected = initial
    active_ok = all(torch.equal(a, b.to(dtype=a.dtype, device=a.device))
                    for a, b in zip(adapter.inference.active, expected, strict=True))
    result = {"master_equals_initial": master_ok, "optimizer_first_reset": first_ok,
              "optimizer_second_reset": second_ok, "optimizer_step_zero": optimizer.step == 0,
              "inference_bank_matches_initial": active_ok,
              "runtime_version_zero": adapter.runtime.active_version == 0,
              "runtime_round_zero": adapter.runtime.round == 0}
    result["passed"] = all(result.values())
    return result


def install(cls, directory):
    for name in ("reset", "_restore_request_source"):
        original = getattr(cls, name)

        @functools.wraps(original)
        def checked(self, *args, _original=original, _name=name, **kwargs):
            result = _original(self, *args, **kwargs)
            receipt = check_reset(self)
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            receipt.update(method=_name, cohort_epoch=self.runtime.epoch,
                           tp_rank=rank, scope=self.config.reset_scope)
            with (Path(directory) / f"rank-{rank}-reset-qa.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt) + "\n")
            if not receipt["passed"]:
                raise RuntimeError("adapter reset tensor/optimizer QA failed")
            return result

        setattr(cls, name, checked)
