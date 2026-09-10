"""Review explicit v4 lifecycle evidence. No label-only reset acceptance."""

import gzip
import json


def read_rows(path):
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def review_lifecycle(job, request_costs, metrics, reset_receipts):
    adaptive = job.method not in {"static", "target_only"}
    expected = job.parameters["execution_request_count"]
    sequential = job.parameters["preview_panel"] not in {"serving", "burstgpt"}
    scope = job.parameters["preview_state_scope"]
    if adaptive:
        for rank in range(job.gpu_count):
            rows = [r for r in reset_receipts if r.get("tp_rank") == rank]
            if not rows or any(r.get("passed") is not True for r in rows):
                raise RuntimeError("missing/failed real adapter tensor reset receipt")
    if sequential:
        if len(request_costs) != expected:
            raise RuntimeError("missing complete sequential lifecycle trace")
        ids = [(r["source"], r["problem_id"]) for r in job.parameters["preview_prompt_records"]]
        if [(r["source"], r["problem_id"]) for r in request_costs] != ids:
            raise RuntimeError("lifecycle sample order differs from registration")
        previous = None
        for row in request_costs:
            before, after = row["state_before"], row["state_after"]
            if len(before) != job.gpu_count or len(after) != job.gpu_count:
                raise RuntimeError("lifecycle missing TP rank")
            if not adaptive:
                continue
            for snapshot in (before, after):
                for key in ("cohort_epoch", "active_version", "round"):
                    values = [r.get(key) for r in snapshot]
                    if any(type(v) is not int for v in values) or len(set(values)) != 1:
                        raise RuntimeError("lifecycle missing/inconsistent rank versions")
            if previous is not None and before[0]["cohort_epoch"] != previous[0]["cohort_epoch"]:
                raise RuntimeError("unexpected reset between requests")
            if scope == "request":
                if not row["request_scope_release_checked"]:
                    raise RuntimeError("request ownership release not checked")
                if (before[0]["active_version"] != 0 or after[0]["active_version"] != 0
                        or after[0]["cohort_epoch"] != before[0]["cohort_epoch"] + 1
                        or after[0]["round"] != 0):
                    raise RuntimeError("request reset lifecycle mismatch")
            elif (after[0]["cohort_epoch"] != before[0]["cohort_epoch"]
                  or after[0]["active_version"] < before[0]["active_version"]):
                raise RuntimeError("persistent cohort was reset or rolled back")
            previous = after
    elif adaptive:
        before, after = metrics["rank_local_before"], metrics["rank_local_after"]
        if any(a.get("cohort_epoch") is None or a["cohort_epoch"] != b.get("cohort_epoch")
               for a, b in zip(before, after, strict=True)):
            raise RuntimeError("serving window did not preserve cold cohort state")
    return {"state_lifecycle": True, "parameter_optimizer_reset": True,
            "scope": scope, "evidence_kind": "endpoint tensor checks plus request/epoch/rank trace"}
