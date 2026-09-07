"""Excluded first-divergence capture. Run on the GPU host; never claims formal jobs."""

import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--variant", required=True, choices=("target", "static", "frozen", "active"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
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
        recipe["stride"] = args.max_tokens * 8 + 1
        params["workload"] = "systems_local_factorial"
    params.update(excluded_from_analysis=True, frozen_recipe=recipe, generation_tokens=args.max_tokens)
    job = replace(original, job_id=f"excluded-first-divergence-{args.variant}",
                  node="excluded-preview-trajectory", method=method,
                  backend="NONE" if method == "target_only" else "DFLASH",
                  width=None if method == "target_only" else original.width, parameters=params)
    selection = recipe if method == "lightcone" else None
    (args.output / "server").mkdir()
    (args.output / "config.json").write_text(json.dumps({"job": job.to_dict(),
        "adaptation": adaptation_payload(job, selection), "excluded": True,
        "capture_scope": "first three original requests, original seed/order, bounded output; not performance evidence",
        "diagnostic_top_logprobs": diagnostic_logprobs(args.variant)}, indent=2))
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
        if args.variant == "frozen" and after["updates_published"] != before["updates_published"]:
            raise RuntimeError("frozen control unexpectedly published an update")
        (args.output / "result.json").write_text(json.dumps({"status": "captured_not_reviewed",
            "before": before, "after": after, "input_metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
