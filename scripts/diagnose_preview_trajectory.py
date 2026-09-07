"""Excluded first-divergence capture. Run on the GPU host; never claims formal jobs."""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.runner import _cell_inputs, _speed_metrics
from lightcone_spec.server import ServerProcess, adaptation_payload
from lightcone_spec.state import StateStore


def first_divergence(left, right):
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if a != b:
            return {"position": index, "target_token": a, "method_token": b}
    return None


def diagnostic_logprobs(variant):
    # Native DFlash explicitly rejects return_logprob. Do not change its path
    # to obtain diagnostics: compare committed IDs first, then score their
    # common prefix with target-only (whose top-2 capture is supported).
    return 2 if variant == "target" else 0


def frozen_control(parameters, max_tokens):
    params = dict(parameters)
    recipe = dict(params["frozen_recipe"])
    recipe["stride"] = max_tokens * 8 + 1
    # Job-level execution fields deliberately override selection fields.
    params.update(stride=recipe["stride"], workload="systems_local_factorial",
                  frozen_recipe=recipe)
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--variant", required=True, choices=("target", "static", "frozen", "active"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--verify-trace", action="store_true", help="Excluded synchronized verification trace")
    parser.add_argument("--target-state-audit", action="store_true", help="Excluded exact target/KV fingerprints")
    parser.add_argument("--reference", type=Path, help="Prior successful capture whose output IDs must match")
    args = parser.parse_args()
    if args.target_state_audit and (not args.verify_trace or args.variant != "active"):
        raise ValueError("target state audit requires the active verification trace and reference")
    config = ExperimentConfig.load(args.config)
    if args.gpu not in config.gpu_ids or not 1 <= args.max_tokens <= 32768:
        raise ValueError("invalid GPU or diagnostic bound")
    args.output.mkdir(parents=True, exist_ok=False)
    state = StateStore(config.run_dir)
    original = next(j for j in state.jobs("E3b-preview-v1") if j.task == "MATH-500"
                    and j.method == "lightcone" and j.block == 0)
    method = "target_only" if args.variant == "target" else "static" if args.variant == "static" else "lightcone"
    params = dict(original.parameters)
    recipe = dict(params["frozen_recipe"])
    # A bounded, excluded no-publication control, not a formal stride exception.
    if args.variant == "frozen":
        params = frozen_control(params, args.max_tokens)
        recipe = params["frozen_recipe"]
    params.update(excluded_from_analysis=True, frozen_recipe=recipe, generation_tokens=args.max_tokens)
    job = replace(original, job_id=f"excluded-first-divergence-{args.variant}",
                  node="excluded-preview-trajectory", method=method,
                  backend="NONE" if method == "target_only" else "DFLASH",
                  width=None if method == "target_only" else original.width, parameters=params)
    selection = recipe if method == "lightcone" else None
    if args.variant == "frozen" and adaptation_payload(job, selection)["stride"] != args.max_tokens * 8 + 1:
        raise RuntimeError("frozen control stride was overridden before launch")
    (args.output / "server").mkdir()
    if args.verify_trace:
        if args.variant == "target" or args.reference is None:
            raise ValueError("verification trace requires a DFlash path and a prior output reference")
        os.environ["LIGHTCONE_EXCLUDED_VERIFY_TRACE"] = json.dumps({
            "output_directory": str((args.output / "server").resolve()),
            "request_suffix": "-2-00000", "start": 620, "end": 660,
            "target_state_audit": args.target_state_audit,
        })
        os.environ["PYTHONPATH"] = os.pathsep.join((
            str(Path(__file__).resolve().parent / "preview_verify_trace"),
            os.environ.get("PYTHONPATH", ""),
        ))
    (args.output / "config.json").write_text(json.dumps({"job": job.to_dict(),
        "adaptation": adaptation_payload(job, selection), "excluded": True,
        "capture_scope": "first three original requests, original seed/order, bounded output; not performance evidence",
        "diagnostic_top_logprobs": diagnostic_logprobs(args.variant),
        "verification_trace": args.verify_trace,
        "target_state_audit": args.target_state_audit}, indent=2))
    process = ServerProcess(config, job, gpus=(args.gpu,), port=config.server.base_port + 30 + args.gpu,
                            output_dir=args.output / "server", selection=selection)
    with process as client:
        prompts, _, metadata = _cell_inputs(config, state, client, job)
        client.reset()
        before = _speed_metrics(client.server_info(), "tp1_dp1")
        records = []
        for index, prompt in enumerate(prompts[:3]):
            with (args.output / f"request-{index}-events.jsonl").open("x") as stream:
                client.stream_observer = lambda event: stream.write(json.dumps(event) + "\n")
                result, _ = client.run_batch((prompt,), max_new_tokens=args.max_tokens,
                                            seed=index, temperature=0., ignore_eos=False,
                                            request_id_prefix=f"diag-{args.variant}-{index}",
                                            diagnostic_top_logprobs=diagnostic_logprobs(args.variant))
            client.stream_observer = None
            records.extend(r.to_dict() for r in result)
            (args.output / "requests.json").write_text(json.dumps(records))
        after = _speed_metrics(client.server_info(), "tp1_dp1")
        if args.verify_trace:
            if not list((args.output / "server").glob("verify-*.jsonl")):
                raise RuntimeError("verification trace hook produced no evidence")
            prior = json.loads(args.reference.read_text())
            if [r["output_ids"] for r in records] != [r["output_ids"] for r in prior]:
                raise RuntimeError("verification tracing perturbed the reference trajectory; do not accept")
        if args.target_state_audit:
            audits = list((args.output / "server").glob("target-state-*.json"))
            if not audits or not all(json.loads(p.read_text())["target_parameters_unchanged"]
                                     and json.loads(p.read_text())["committed_prefix_kv_unchanged_during_update"]
                                     for p in audits):
                raise RuntimeError("target state audit missing or failed; do not accept")
        if args.variant == "frozen" and after["updates_published"] != before["updates_published"]:
            raise RuntimeError("frozen control unexpectedly published an update")
        (args.output / "result.json").write_text(json.dumps({"status": "captured_not_reviewed",
            "before": before, "after": after, "input_metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
