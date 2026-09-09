"""Opt-in, excluded live fixed-state replay. Never a performance measurement.

Three update events duplicate the real KL objective and optimizer proposal.
Synchronous comparisons/I/O are deliberate here, not enabled in timed windows.
"""
import ast
import importlib
import json
import os
from pathlib import Path


def install(module, source, output):
    import torch

    tree = ast.parse(Path(source).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_DFlashInferenceRMS")
    namespace = {"torch": torch}
    exec(compile(ast.Module([cls], []), str(source), "exec"), namespace)
    reference = namespace["_DFlashInferenceRMS"].backward
    candidate = module._DFlashInferenceRMS.backward
    feedback = module.loss_and_grad
    optimizer_module = importlib.import_module("sglang.srt.speculative.online_adaptation_runtime")
    optimizer_type = optimizer_module.ResidentOptimizer
    propose = optimizer_type.propose
    optimizer_reference = None
    optimizer_source = os.environ.get("LIGHTCONE_HOTPATH_OPTIMIZER_REFERENCE")
    if optimizer_source:
        optimizer_tree = ast.parse(Path(optimizer_source).read_text())
        optimizer_cls = next(n for n in optimizer_tree.body
                             if isinstance(n, ast.ClassDef) and n.name == "ResidentOptimizer")
        optimizer_namespace = dict(vars(optimizer_module))
        exec(compile(ast.Module([optimizer_cls], []), optimizer_source, "exec"), optimizer_namespace)
        optimizer_reference = optimizer_namespace["ResidentOptimizer"]
    seen, pending, rows = 0, None, []

    def compare(a, b):
        return {"equal": torch.equal(a, b), "finite": bool(torch.isfinite(a).all()),
                "max_abs": float((a.float()-b.float()).abs().max())}

    def save(row):
        rows.append(row)
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"rank-pid{os.getpid()}.json").write_text(json.dumps({
            "scope": "excluded_live_KL_gradient_proposal_replay; not timing",
            "rows": rows}, indent=2))
        if not row["passed"]:
            raise RuntimeError("fixed-state KL/gradient/proposal mismatch; evidence preserved")

    def loss_and_grad(parameters, objective):
        nonlocal seen, pending
        seen += 1
        if seen not in {1, 100, 300}:
            return feedback(parameters, objective)
        try:
            module._DFlashInferenceRMS.backward = staticmethod(reference)
            old_loss, old_grad = feedback(parameters, objective)
        finally:
            module._DFlashInferenceRMS.backward = staticmethod(candidate)
        new_loss, new_grad = feedback(parameters, objective)
        record = {"event": seen, "loss": compare(new_loss, old_loss),
                  "gradients": [compare(a, b) for a, b in zip(new_grad, old_grad, strict=True)]}
        record["passed"] = all(x["equal"] and x["finite"] for x in [record["loss"], *record["gradients"]])
        if not record["passed"]:
            save(record)
        pending = (old_grad, record)
        return new_loss, new_grad

    def proposal(self, gradients, **kwargs):
        nonlocal pending
        if pending is None:
            return propose(self, gradients, **kwargs)
        old_grad, record = pending
        pending = None
        state = tuple(x.clone() for x in (*self.master, *self.first, *self.second, self._step))
        if optimizer_reference is None:
            old = propose(self, old_grad, **kwargs)
        else:
            # Excluded c1 replay only. Restore even if the reference raises;
            # the live optimizer tensors are never committed by either call.
            candidate_schedule = optimizer_type._scheduled_learning_rate
            try:
                optimizer_type._scheduled_learning_rate = optimizer_reference._scheduled_learning_rate
                old = optimizer_reference.propose(self, old_grad, **kwargs)
            finally:
                optimizer_type._scheduled_learning_rate = candidate_schedule
            record["optimizer_reference"] = optimizer_source
        new = propose(self, gradients, **kwargs)
        def tensors(p):
            return (*p.parameters, *p.first_moments, *p.second_moments,
                    p.step, p.gradient_norms, p.effective_learning_rate, p.schedule_valid)
        record["proposal"] = [compare(a, b) for a, b in zip(tensors(new), tensors(old), strict=True)]
        record["state_unchanged"] = all(torch.equal(a, b) for a, b in zip(
            state, (*self.master, *self.first, *self.second, self._step), strict=True))
        record["versions"] = {k: v for k, v in kwargs.items() if k.endswith("version")}
        record["passed"] &= record["state_unchanged"] and all(x["equal"] and x["finite"] for x in record["proposal"])
        save(record)
        return new

    module.loss_and_grad = loss_and_grad
    optimizer_type.propose = proposal
