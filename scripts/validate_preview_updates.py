"""Excluded preview-v3 update/reset QA; never materializes or accepts formal jobs."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.preview_revision import preview_recipe
from lightcone_spec.runner import _cell_inputs, _speed_metrics
from lightcone_spec.server import ServerProcess, adaptation_payload, apply_runner_affinity
from lightcone_spec.state import StateStore


def update_qa_job(source, case, tp=1):
    if case not in {"s1-long", "ensemble"} or source.backend != "DFLASH":
        raise ValueError("this QA covers 8B DFlash update paths only")
    if type(tp) is not int or tp not in (1, 2):
        raise ValueError("excluded QA supports only registered TP1/TP2")
    method = "lightcone" if case == "s1-long" else "onlinespec_ens"
    recipe = preview_recipe(method, "DFLASH", {"lightcone_recipe": source.parameters["frozen_recipe"]})
    job = replace(source, job_id=f"excluded-preview-v3-{case}", node="excluded-preview-v3-updates",
        method=method, load="c1", gpu_count=tp, block=None,
        parameters={**source.parameters, "preview_revision": 3, "frozen_recipe": recipe,
            "stride": recipe["stride"], "topology": f"tp{tp}_dp1", "excluded_from_analysis": True,
            "execution_request_count": 1, "generation_tokens": 32768 if case == "s1-long" else 512})
    payload = adaptation_payload(job, recipe)
    assert payload["stride"] == (1 if case == "s1-long" else 10)
    assert payload["optimizer"]["name"] == ("chronobelief" if case == "s1-long" else "adam")
    if case == "s1-long":
        assert (payload["rank"], payload["parameter_scope"], payload["optimizer"]["learning_rate"]) == (8, "last1", .001)
    else:
        assert payload["online_spec"]["ensemble_optimizer"] == "adam_preview_v3"
    return job, recipe


def adam_transaction_check(optimizer_class, config, device):
    """Same actual optimizer on CPU/GPU; native Adam reference plus rejected state."""
    import torch

    initial = torch.tensor([.4, -.2], device=device)
    optimizer = optimizer_class((initial,), config)
    assert optimizer.adam_ensemble and len(optimizer.learning_rates) == 3
    reference = [initial.clone().requires_grad_(True) for _ in range(3)]
    native = [torch.optim.Adam([p], lr=lr, betas=(.9, .95), eps=1e-8)
              for p, lr in zip(reference, optimizer.learning_rates, strict=True)]
    losses = torch.tensor([.1, .3, .2], device=device)
    for step in range(3):
        gradients = tuple((torch.tensor([2. + step + i, -3. - i], device=device),) for i in range(3))
        before = tuple(t.clone() for t in optimizer.state_tensors)
        proposal = optimizer.propose_ensemble(losses, gradients)
        optimizer.commit(proposal, valid=torch.tensor(False, device=device))
        assert all(torch.equal(a, b) for a, b in zip(before, optimizer.state_tensors, strict=True))
        for index, (p, opt) in enumerate(zip(reference, native, strict=True)):
            gradient = gradients[index][0]
            p.grad = gradient * torch.clamp(config.optimizer.grad_clip / (gradient.norm() + 1e-12), max=1.)
            opt.step()
        optimizer.commit(proposal, valid=torch.tensor(True, device=device))
        for expected, actual in zip(reference, optimizer.experts, strict=True):
            torch.testing.assert_close(actual[0], expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(optimizer.expert_probabilities,
            torch.softmax(-10 * (step + 1) * losses, dim=0))
    assert all(torch.count_nonzero(t) for t in optimizer.second[1:])
    measured = optimizer.diagnostics()
    optimizer.reset((initial,))
    assert optimizer.step == 0 and all(torch.count_nonzero(t) == 0 for t in optimizer.second)
    assert all(torch.equal(p, initial) for expert in optimizer.experts for p in expert)
    return {"status": "passed", "device": str(device), "before_reset": measured,
            "rejected_state_unchanged": True, "adam_reference_matched": True, "reset_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--case", choices=("s1-long", "ensemble"))
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1, help="Excluded topology QA, not group acceptance")
    parser.add_argument("--reset-diagnostic", type=Path, help="Excluded bounded trace; prior S1 QA directory required")
    parser.add_argument("--optimizer-only", type=Path, help="Subprocess-only GPU transaction config")
    args = parser.parse_args()
    original = ExperimentConfig.load(args.config)
    if args.gpu not in original.gpu_ids:
        raise ValueError("GPU is outside the registered allocation")
    if args.tp == 2 and (len(original.gpu_ids) != 2 or args.gpu != original.gpu_ids[0]):
        raise ValueError("TP2 QA must own both registered GPUs")
    if args.optimizer_only:
        import torch
        sys.path.insert(0, str(original.sglang_root / "python"))
        from sglang.srt.speculative.online_adaptation_config import OnlineAdaptationConfig
        from sglang.srt.speculative.online_adaptation_runtime import OnlineSpecOptimizer
        if args.output.exists():
            raise ValueError("do not overwrite an optimizer QA result")
        config = OnlineAdaptationConfig.load(args.optimizer_only, tp_size=args.tp, dp_size=1)
        result = adam_transaction_check(OnlineSpecOptimizer, config, torch.device(f"cuda:{args.gpu}"))
        args.output.write_text(json.dumps(result, indent=2))
        return
    if args.case is None:
        raise ValueError("QA case is required")
    if args.reset_diagnostic and (args.case != "s1-long" or args.tp != 1):
        raise ValueError("reset fingerprint diagnosis requires TP1 S1 LoRA")
    args.output.mkdir(parents=True, exist_ok=False)
    config = replace(original, results_root=args.output, run_name="excluded")
    config.run_dir.mkdir()
    state = StateStore(original.run_dir)
    source = next(j for j in state.jobs("E3b-preview-v1")
                  if j.method == "lightcone" and j.task == "MATH-500" and j.block == 0)
    job, recipe = update_qa_job(source, args.case, args.tp)
    gpus = original.gpu_ids if args.tp == 2 else (args.gpu,)
    topology = f"tp{args.tp}_dp1"
    payload = adaptation_payload(job, recipe)
    (args.output / "adaptation.json").write_text(json.dumps(payload, indent=2))
    (args.output / "config.json").write_text(json.dumps({"job": job.to_dict(), "execution_gpu_ids": gpus,
        "scope": "excluded: one frozen prompt, forced output pressure, reset and sampling QA; not performance or common-TP acceptance"}, indent=2))
    phases = []
    try:
        if args.case == "ensemble":
            # Exit this tiny GPU process before starting the server: no hidden
            # tensor-test CUDA context remains resident during memory QA.
            subprocess.run([str(config.server.python), str(Path(__file__).resolve()),
                "--config", str(args.config), "--output", str(args.output / "optimizer-transaction.json"),
                "--gpu", str(args.gpu), "--tp", str(args.tp),
                "--optimizer-only", str(args.output / "adaptation.json")], check=True)
        apply_runner_affinity(config.gpu_ids, config.run_dir / "numa-affinity.json")
        server = args.output / "server"
        server.mkdir()
        if args.reset_diagnostic:
            for name in ("reset-a", "reset-b"):
                if not (args.reset_diagnostic / f"{name}-requests.json").is_file():
                    raise ValueError("missing original reset trajectory")
            os.environ["LIGHTCONE_EXCLUDED_VERIFY_TRACE"] = json.dumps({
                "output_directory": str(server.resolve()), "request_suffix": "-00000",
                "start": 150, "end": 190, "reset_audit": True})
            os.environ["PYTHONPATH"] = os.pathsep.join((
                str(Path(__file__).resolve().parent / "preview_verify_trace"), os.environ.get("PYTHONPATH", "")))
        process = ServerProcess(config, job, gpus=gpus, port=config.server.base_port + 50 + args.gpu,
                                output_dir=server, selection=recipe)
        with process as client:
            env = Path(f"/proc/{process.process.pid}/environ").read_bytes().split(b"\0")
            assert ("CUDA_VISIBLE_DEVICES=" + ",".join(map(str, gpus))).encode() in env
            prompts, _, metadata = _cell_inputs(config, state, client, job)
            assert len(prompts) == 1
            (args.output / "inputs.json").write_text(json.dumps({"metadata": metadata, "token_ids": prompts}))
            # Short reset repetitions precede pressure: they cannot substitute
            # for the actual 32K S1 request, and their evidence is separate.
            plan = [("reset-a", 512, 0.), ("reset-b", 512, 0.),
                    ("pressure" if args.case == "s1-long" else "sampled", 32768 if args.case == "s1-long" else 512,
                     0. if args.case == "s1-long" else 1.)]
            if args.reset_diagnostic:
                plan = plan[:2]
            reference = None
            for name, tokens, temperature in plan:
                client.reset()
                before = _speed_metrics(client.server_info(), topology)
                assert len(before["rank_local"]) == args.tp
                assert before["updates_published"] == 0
                if args.case == "ensemble":
                    for rank in before["rank_local"]:
                        online = rank["online_spec_state"]
                        assert online["step"] == 0 and all(loss == 0 for loss in online["cumulative_losses"])
                ids, sequences = [], []
                with (args.output / f"{name}-events.jsonl").open("x") as stream:
                    def observe(event):
                        assert event["sequence"] == len(sequences) + 1
                        sequences.append(event["sequence"])
                        ids.extend(event["token_ids"])
                        stream.write(json.dumps({key: event[key] for key in
                            ("request_id", "sequence", "elapsed_seconds", "token_ids")}) + "\n")
                    client.stream_observer = observe
                    results, duration = client.run_batch(prompts, max_new_tokens=tokens, seed=0,
                        temperature=temperature, ignore_eos=True, timeout_seconds=3600,
                        request_id_prefix=f"qa-v3-{args.case}-{name}")
                client.stream_observer = None
                raw = [r.to_dict() for r in results]
                (args.output / f"{name}-requests.json").write_text(json.dumps(raw))
                after = _speed_metrics(client.server_info(), topology)
                (args.output / f"{name}-metrics.json").write_text(json.dumps({"before": before, "after": after}))
                assert len(raw) == 1 and raw[0]["completion_tokens"] == tokens and ids == raw[0]["output_ids"]
                for key in ("fallbacks", "nonfinite_updates", "exactness_violations", "budget_violations",
                            "version_mismatches", "oom_events", "stale_publications"):
                    assert after[key] == 0, (name, key, after[key])
                assert after["updates_published"] > 0 and after.get("enabled") is True
                if name == "reset-a":
                    reference = raw[0]["output_ids"]
                elif name == "reset-b":
                    if not args.reset_diagnostic:
                        assert raw[0]["output_ids"] == reference, "greedy reset trajectory changed"
                if args.case == "ensemble":
                    assert len(after["rank_local"]) == args.tp
                    assert all(rank["online_spec_state"]["step"] > 0 for rank in after["rank_local"])
                phases.append({"name": name, "tokens": tokens, "updates_published": after["updates_published"],
                               "duration_seconds_excluded": duration, "stream_events": len(sequences)})
                (args.output / "progress.json").write_text(json.dumps(phases, indent=2))
            if args.reset_diagnostic:
                resets = [json.loads(p.read_text()) for p in server.glob("reset-*.json")]
                traces = [json.loads(line) for p in server.glob("verify-*.jsonl") for line in p.read_text().splitlines()]
                if len(resets) != 2 or not traces:
                    raise RuntimeError("missing bounded reset/verification evidence")
                matches = {}
                for name in ("reset-a", "reset-b"):
                    old = json.loads((args.reset_diagnostic / f"{name}-requests.json").read_text())
                    new = json.loads((args.output / f"{name}-requests.json").read_text())
                    matches[name] = old[0]["output_ids"] == new[0]["output_ids"]
                (args.output / "diagnostic.json").write_text(json.dumps({
                    "status": "captured_not_reviewed", "formal_acceptance": False,
                    "reset_state_equal": all(r["passed"] and r["matches_first_reset"] for r in resets),
                    "uninstrumented_trajectory_matches": matches,
                    "trace_rows": len(traces),
                    "commits_match_own_argmax": all(r["committed_matches_verify_argmax"] for r in traces),
                    "warning": "synchronization can change asynchronous publication timing; never performance evidence"}, indent=2))
                return
        (args.output / "result.json").write_text(json.dumps({"status": "passed_excluded_update_qa",
            "formal_acceptance": False, "phases": phases}, indent=2))
    except BaseException as error:
        (args.output / "failure.json").write_text(json.dumps({"type": type(error).__name__,
            "error": str(error), "phases": phases}, indent=2))
        raise


if __name__ == "__main__":
    main()
