"""Bounded, calibration-only global-stride selection. No automatic formal mutation."""

import math
import random
import statistics
from copy import deepcopy
from dataclasses import replace

COARSE_STRIDES = (1, 4, 10, 32)
ALL_STRIDES = (1, 2, 4, 8, 10, 16, 32)
SOURCES = {"Code": "APPS", "Math": "OpenR1-Math"}


def calibration_split(records, excluded_records):
    excluded_ids = {str(r["problem_id"]) for r in excluded_records}
    excluded_text = {str(r["prompt"]).strip() for r in excluded_records}
    result = {}
    seen_ids, seen_text = set(), set()
    for domain, source in SOURCES.items():
        available = []
        for row in sorted(records, key=lambda r: str(r["problem_id"])):
            identity, prompt = str(row["problem_id"]), str(row["prompt"]).strip()
            if row.get("source") != source or identity in excluded_ids or prompt in excluded_text:
                continue
            if identity in seen_ids or prompt in seen_text:
                continue
            available.append(deepcopy(row))
            seen_ids.add(identity)
            seen_text.add(prompt)
        random.Random(f"stride-audit-v1:{domain}:0").shuffle(available)
        if len(available) < 12:
            raise ValueError(f"{domain} needs 12 unique non-preview calibration records")
        result[domain] = {"search": available[:4], "confirmation": available[4:12]}
    return result


def audit_job(template, split, *, phase, domain, method, stride, block, implementation, timing_mode="off"):
    if phase not in {"baseline", "coarse", "refine", "confirmation", "optimization"}:
        raise ValueError("unknown audit phase")
    if stride not in ALL_STRIDES or method not in {"static", "lightcone"} or timing_mode not in {"off", "full"}:
        raise ValueError("unregistered audit configuration")
    if template.backend != "DFLASH" or template.model != "Qwen/Qwen3-8B":
        raise ValueError("use the frozen 8B DFlash template")
    params = deepcopy(template.parameters)
    params.pop("preview_revision", None)
    subset = "confirmation" if phase == "confirmation" else "search"
    params.update(stride_audit_v1=True, excluded_from_analysis=True, dataset_key="CalibrationMix",
        preview_prompt_records=deepcopy(split[domain][subset]), execution_request_count=8 if subset=="confirmation" else 4,
        sampling_seed=block, stride=stride, implementation_revision=implementation,
        timing_mode=timing_mode, topology="tp2_dp1", audit_phase=phase, audit_domain=domain,
        pairing_key=f"stride-audit-v1:{phase}:{domain}:b{block}:{implementation}:{timing_mode}")
    recipe = deepcopy(template.parameters["frozen_recipe"])
    if not isinstance(recipe, dict):
        raise ValueError("LightCone template requires a frozen recipe")
    recipe.update(optimizer="chronobelief", parameterization="lora", rank=8, scope="last1",
                  learning_rate=1e-3, lr=1e-3, schedule="constant", stride=stride)
    params["frozen_recipe"] = recipe if method == "lightcone" else None
    params["method_label"] = f"{method} calibration S={stride if method=='lightcone' else 'N/A'}"
    identity = f"stride-audit-v1__{phase}__{domain}__{method}__s{stride if method=='lightcone' else 0}__b{block}__{implementation}__{timing_mode}"
    return replace(template, job_id=identity, node="Stride-audit-v1", task=f"Calibration-{domain}",
                   method=method, load="c1", gpu_count=2, block=block, parameters=params)


def stage_jobs(template, split, *, phase, implementation, strides=None):
    if phase == "coarse":
        strides, blocks, static = COARSE_STRIDES, 2, True
    elif phase == "refine":
        if not strides or len(strides) > 2 or any(s not in ALL_STRIDES for s in strides):
            raise ValueError("at most two registered refinement strides")
        blocks, static = 2, False
    elif phase == "confirmation":
        if not strides or len(strides) != 1:
            raise ValueError("confirmation requires exactly one frozen candidate")
        blocks, static = 4, True
    else:
        raise ValueError("stage not registered")
    jobs = []
    for domain in SOURCES:
        for block in range(blocks):
            unit = [audit_job(template, split, phase=phase, domain=domain, method="lightcone",
                              stride=s, block=block, implementation=implementation) for s in strides]
            if static:
                unit.append(audit_job(template, split, phase=phase, domain=domain, method="static",
                                      stride=1, block=block, implementation=implementation))
            random.Random(f"{phase}:{domain}:{block}").shuffle(unit)
            jobs.extend(unit)
    return tuple(replace(job, ordinal=i) for i, job in enumerate(jobs))


def refinement_candidates(winner, measured=COARSE_STRIDES):
    if winner not in ALL_STRIDES:
        raise ValueError("unregistered winner")
    left = [s for s in ALL_STRIDES if s < winner and s not in measured]
    right = [s for s in ALL_STRIDES if s > winner and s not in measured]
    return tuple(([max(left)] if left else []) + ([min(right)] if right else []))


def rank_candidates(rows):
    """Consume only complete two-domain/two-block off-profiler calibration pairs."""
    keyed = {}
    for row in rows:
        if row["phase"] not in {"coarse", "refine"} or row["timing_mode"] != "off":
            raise ValueError("selection cannot consume confirmation/profiling evidence")
        key = row["domain"], row["block"], row["method"], row["stride"] if row["method"]=="lightcone" else 0
        if key in keyed:
            raise ValueError("duplicate selection evidence")
        keyed[key] = row
    scored = []
    for stride in ALL_STRIDES:
        ratios, costs, itls = [], [], []
        for domain in SOURCES:
            for block in (0, 1):
                a, b = keyed.get((domain, block, "lightcone", stride)), keyed.get((domain, block, "static", 0))
                if not a or not b or not a["hard_feasible"] or not b["hard_feasible"]:
                    continue
                for field in ("implementation", "topology", "budget_policy", "stimulus_ids"):
                    if a[field] != b[field]:
                        raise ValueError(f"mismatched calibration pair: {field}")
                if any(not math.isfinite(r["goodput"]) or r["goodput"]<=0 for r in (a,b)):
                    raise ValueError("nonfinite speed")
                ratios.append(math.log(a["goodput"]/b["goodput"]))
                costs.append(a.get("update_ms_per_1000_tokens"))
                itls.append(a["itl_p99_ms"])
        if len(ratios)==4:
            scored.append({"stride":stride,"score":math.exp(statistics.mean(ratios)),
                           "update_cost":statistics.mean(costs) if all(v is not None and math.isfinite(v) for v in costs) else None,
                           "itl":statistics.mean(itls)})
    if not scored:
        raise ValueError("no complete safe calibration candidate")
    ordered = []
    while scored:
        best = max(r["score"] for r in scored)
        tied = [r for r in scored if r["score"] >= best / 1.01]
        # Missing timing evidence cannot be interpreted as zero cost or silently
        # bypass the registered first tie-break.
        if len(tied)>1 and any(r["update_cost"] is None for r in tied):
            raise ValueError("timing cost required for 1% tie-break")
        winner = min(tied, key=lambda r:(r["update_cost"] or 0, r["itl"], -r["stride"]))
        ordered.append(winner)
        scored.remove(winner)
    return ordered
