"""Opt-in Python tracing for excluded DFlash verification, never performance data."""

import json
import linecache
import os
import sys
import time
from pathlib import Path


def verification_snapshot(values, row):
    import torch

    batch = values["batch"]
    request = batch.reqs[row]
    block = int(values["self"].block_size)
    logits = values["logits_output"].next_token_logits.view(len(batch.reqs), block, -1)[row]
    top = torch.topk(logits.float(), 2, dim=-1)
    count = int(values["commit_lens"][row].item())
    committed = values["out_tokens"][row, :count]
    prediction = values["target_predict"][row]
    adapter = values.get("online_adapter")
    runtime = getattr(adapter, "runtime", None)
    prefix = int(values["prefix_lens"][row].item())
    return {
        "measurement_scope": "excluded_verify_trace; synchronization may perturb timing",
        "rid": request.rid, "output_ids_before": list(request.output_ids),
        "origin_tokens": len(request.origin_input_ids), "prefix_length": prefix,
        "generated_offset": prefix - len(request.origin_input_ids),
        "seq_length": int(batch.seq_lens[row].item()),
        "positions": values["positions"].view(len(batch.reqs), block)[row].tolist(),
        "kv_locations": values["verify_out_cache_loc_2d"][row].tolist(),
        "draft_tokens": values["draft_tokens"][row].tolist(),
        "target_top2_ids": top.indices.tolist(), "target_top2_logits": top.values.tolist(),
        "logit_dtype": str(logits.dtype), "target_argmax": prediction.tolist(),
        "commit_count": count, "committed_ids": committed.tolist(),
        "accept_count": int(values["accept_len"][row].item()),
        "bonus": int(values["bonus"][row].item()),
        "committed_matches_verify_argmax": bool(torch.equal(committed, prediction[:count])),
        "active_version": getattr(runtime, "active_version", None),
        "adaptation_round": getattr(runtime, "round", None),
    }


def install():
    settings = json.loads(os.environ["LIGHTCONE_EXCLUDED_VERIFY_TRACE"])
    output = Path(settings["output_directory"])
    if not output.is_absolute() or not output.is_dir():
        raise ValueError("excluded trace requires an existing absolute output directory")
    if sys.gettrace() is not None:
        raise RuntimeError("do not replace another debugger or trace hook")
    line_numbers = {}
    captured = 0
    state_audit = None
    reset_count = 0
    reset_reference = None

    def local_trace(frame, event, arg):
        nonlocal captured
        if event != "line" or frame.f_lineno != line_numbers[frame.f_code]:
            return local_trace
        values = frame.f_locals
        for row, request in enumerate(values["batch"].reqs):
            if not request.rid.endswith(settings["request_suffix"]):
                continue
            offset = int(values["prefix_lens"][row].item()) - len(request.origin_input_ids)
            if offset > settings["end"] or offset + int(values["self"].block_size) < settings["start"]:
                continue
            if captured >= 64:
                raise RuntimeError("excluded verification trace exceeded its bounded window")
            record = verification_snapshot(values, row)
            with (output / f"verify-{os.getpid()}.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            captured += 1
        return local_trace

    def dispatch(frame, event, arg):
        nonlocal state_audit, reset_count, reset_reference
        code = frame.f_code
        if (event == "call" and settings.get("rank_metrics") and code.co_name == "get_internal_state"
                and code.co_filename.endswith("/managers/scheduler.py")):
            scheduler = frame.f_locals["self"]

            def finish_metrics(frame, event, arg):
                if event == "return" and arg is not None:
                    state = arg.internal_state
                    rank = int(scheduler.ps.tp_rank)
                    record = {"tp_rank": rank, "tp_size": int(scheduler.ps.tp_size),
                              "captured_ns": time.time_ns(),
                              "state": {k: state[k] for k in
                                  ("speed_study_metrics", "speculative_adaptation_info_record") if k in state}}
                    with (output / f"rank-{rank}-metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                return finish_metrics

            return finish_metrics
        if (event == "call" and settings.get("reset_audit") and code.co_name == "reset"
                and code.co_filename.endswith("/speculative/dflash_online_adaptation.py")):
            adapter = frame.f_locals["self"]

            def finish_reset(frame, event, arg):
                nonlocal reset_count, reset_reference
                if event == "return":
                    from lightcone_spec.reset_diagnostic import reset_snapshot

                    reset_count += 1
                    if reset_count > 4:
                        raise RuntimeError("excluded reset audit exceeded its bound")
                    record = reset_snapshot(adapter)
                    if reset_reference is None:
                        reset_reference = record["state"]
                    record["matches_first_reset"] = record["state"] == reset_reference
                    (output / f"reset-{os.getpid()}-{reset_count}.json").write_text(json.dumps(record))
                    if not record["passed"] or not record["matches_first_reset"]:
                        raise RuntimeError("excluded reset fingerprint mismatch")
                return finish_reset

            return finish_reset
        if (event == "call" and settings.get("target_state_audit")
                and code.co_name == "maybe_launch"
                and code.co_filename.endswith("/speculative/dflash_online_adaptation.py")
                and state_audit is not None and not state_audit.done):
            parent = frame.f_back.f_locals
            if parent.get("self") is state_audit.worker and state_audit.adapter.runtime.update_due():
                for row, request in enumerate(parent["batch"].reqs):
                    offset = int(parent["prefix_lens"][row].item()) - len(request.origin_input_ids)
                    if (request.rid.endswith(settings["request_suffix"])
                            and settings["start"] <= offset <= settings["end"]):
                        state_audit.before_update(parent, row)

                        def finish_update(frame, event, arg, values=parent, row=row):
                            if event == "return":
                                state_audit.after_update(values, row)
                            return finish_update

                        return finish_update
        if (not settings.get("trace_verify", True)
                or event != "call" or code.co_name != "forward_batch_generation"
                or not code.co_filename.endswith("/speculative/dflash_worker_v2.py")):
            return None
        if settings.get("target_state_audit") and state_audit is None:
            from lightcone_spec.target_state_diagnostic import TargetStateAudit

            state_audit = TargetStateAudit(frame.f_locals["self"], output / f"target-state-{os.getpid()}.json")
        if code not in line_numbers:
            candidates = [index + 1 for index, line in enumerate(linecache.getlines(code.co_filename))
                          if index + 1 >= code.co_firstlineno
                          and line.strip() == "if self._need_mamba_verify_commit:"]
            if len(candidates) != 1:
                raise RuntimeError("DFlash verification trace location changed; review before capture")
            line_numbers[code] = candidates[0]
        return local_trace

    sys.settrace(dispatch)
