"""Fixed 180-cell GitHub preview. Explicit state lifetime, never outcome selection."""

import hashlib
import json
import random
from collections import Counter
from copy import deepcopy
from dataclasses import replace

from .preview_revision import preview_recipe, revision_jobs

NODES = ("Cohort-preview-v4", "E3b-preview-v4", "E5-preview-v4", "Qwen38-preview-v4")
DOMAINS = {"Chat": "Arena-Hard", "Code": "LiveCodeBench", "Math": "MATH-500"}
ORDERS = ("domain16", "mixed", "domain8")


def group_digest(manifest, node):
    if node not in NODES:
        raise ValueError("unknown v4 group")
    rows = [j.to_dict() for j in jobs(manifest) if j.node == node]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def group_accepted(acceptance, manifest, node):
    receipt = acceptance.get("groups", {}).get(node, {})
    required = ("sampling", "parameter_optimizer_reset", "state_lifecycle", "kv_isolation", "tp_ranks", "full_budget")
    return (receipt.get("status") == "accepted" and receipt.get("group_sha256") == group_digest(manifest, node)
            and bool(receipt.get("runtime_commit")) and bool(receipt.get("qa_paths"))
            and all(receipt.get("checks", {}).get(key) is True for key in required))


def video_job(manifest, report, model, method_index, tp, *, cohort=False):
    from .preview import QWEN38_MODEL, QWEN38_VIDEO_METHODS, VIDEO_METHODS
    if tp not in (1, 2) or report.get("version") != 4:
        raise ValueError("v4 video requires matching formal scene ranking and common TP")
    key = "cohort_appendix" if cohort else model
    selection = report["selected_scenes"][key]
    scene = selection["candidates"][0]
    if scene["scene_id"] != selection["selected_scene"] or scene["n_blocks"] != 4:
        raise ValueError("video scene must be the frozen four-block winner")
    rows = jobs(manifest)
    if cohort:
        if method_index not in (0, 1) or model != "Qwen/Qwen3-8B":
            raise ValueError("cohort appendix has exactly Static and persistent LightCone")
        variant = "static" if method_index == 0 else "cohort"
        source = next(j for j in rows if j.node == NODES[0] and j.block == 0
                      and j.parameters["flow_order"] == scene["order"] and j.parameters["state_variant"] == variant)
        if source.gpu_count != tp:
            raise ValueError("appendix must retain the selected full-flow topology")
        params = deepcopy(source.parameters)
    else:
        methods = QWEN38_VIDEO_METHODS if model == QWEN38_MODEL else VIDEO_METHODS
        backend, method, label = methods[method_index]
        source = next(j for j in rows if j.model == model and j.backend == backend and j.method == method
                      and j.node != NODES[0])
        task = scene["task"]
        if task not in DOMAINS.values():
            raise ValueError("selected video scene has no frozen domain pool")
        params = deepcopy(source.parameters)
        params.update(preview_prompt_records=deepcopy(manifest["data"]["video"][task]),
                      execution_request_count=8, input_tokens=16384, generation_tokens=1024,
                      regime="preview_constructed_chat", preview_state_scope="cohort",
                      preview_request_seeds=None, dataset_key=task, method_label=label, respect_eos=True)
        source = replace(source, task=task, load="c8", gpu_count=tp, context=17408)
    params.update(excluded_from_analysis=True, preview_panel="excluded_video",
                  topology=f"tp{tp}_dp1", video_disclosure={
                      "selected_scene": selection["selected_scene"], "selection": selection["disclosure"],
                      "positive_formal_gain": selection["positive_gain"], "cohort_appendix": cohort,
                      "state_scope": params["preview_state_scope"], "take_rule": selection["take_rule"]})
    return replace(source, job_id=f"excluded-video-v4__{key}__{method_index}__tp{tp}",
                   node="excluded-video-v4", ordinal=0, parameters=params)


def record_key(row):
    source, identity = row.get("source"), row.get("problem_id")
    if not source or identity is None or not str(row.get("prompt", "")).strip():
        raise ValueError("preview v4 requires source, problem_id and prompt")
    return str(source), str(identity)


def freeze_data(pools, excluded):
    """Allocate once, seed zero, with ID and exact-text exclusion across all pools."""
    used_ids = {record_key(r) for r in excluded if r.get("source") and r.get("problem_id") is not None}
    used_texts = {r["prompt"].strip() for r in excluded if r.get("prompt")}
    data = {"long8": {}, "long27": {}, "cohort": {}, "video": {}}
    for domain, task in DOMAINS.items():
        rows = sorted(pools[task], key=record_key)
        random.Random(f"preview-v4:0:{task}").shuffle(rows)
        chosen = []
        needed = (0 if domain == "Chat" else 32) + 32 + 64 + 8
        for row in rows:
            key, text = record_key(row), row["prompt"].strip()
            if key in used_ids or text in used_texts:
                continue
            used_ids.add(key)
            used_texts.add(text)
            chosen.append(deepcopy(row))
            if len(chosen) == needed:
                break
        if len(chosen) != needed:
            raise ValueError(f"{task}: need {needed} unseen prompts; have {len(chosen)}")
        cursor = 0
        for name, count in (("long8", 0 if domain == "Chat" else 32),
                            ("long27", 32), ("cohort", 64), ("video", 8)):
            data[name][task] = chosen[cursor:cursor + count]
            cursor += count
    return data


def validate_data(data):
    identities, texts = set(), set()
    for name in ("long8", "long27", "cohort", "video"):
        for domain, task in DOMAINS.items():
            count = {"long8": 0 if domain == "Chat" else 32,
                     "long27": 32, "cohort": 64, "video": 8}[name]
            rows = data[name][task]
            if len(rows) != count:
                raise ValueError(f"{name}/{task}: expected {count} prompts")
            for row in rows:
                identity, text = record_key(row), row["prompt"].strip()
                if identity in identities or text in texts:
                    raise ValueError("preview v4 sample overlap")
                identities.add(identity)
                texts.add(text)


def cohort_records(data, block, order):
    if order not in ORDERS or block not in range(4):
        raise ValueError("invalid cohort block/order")
    domains = list(DOMAINS)
    domains = domains[block % 3:] + domains[:block % 3]
    by_domain = {d: data["cohort"][DOMAINS[d]][block * 16:(block + 1) * 16] for d in domains}
    if order == "domain8":
        rows = [r for half in range(2) for d in domains for r in by_domain[d][half * 8:(half + 1) * 8]]
    else:
        rows = [r for d in domains for r in by_domain[d]]
        if order == "mixed":
            random.Random(f"preview-v4:order:{block}").shuffle(rows)
    return deepcopy(rows)


def request_seed(block, row):
    key = repr((block, record_key(row))).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % (2**31)


def jobs(manifest):
    if manifest.get("version") != 4 or manifest.get("preview_lightcone_stride") != 10:
        raise ValueError("preview v4 requires explicit S10 manifest")
    validate_data(manifest["data"])
    if not manifest.get("qwen38"):
        raise ValueError("preview v4 requires the registered Qwen3.8 group")
    # Reuse frozen serving/trace and model routes, not old attempts or data.
    template = deepcopy(manifest)
    template["version"] = 3
    source = revision_jobs(template)
    output = []

    def append(job, *, node, panel, task, records, scope, state_variant, tp, key, budget):
        p = deepcopy(job.parameters)
        p.pop("replaces_job_id", None)
        p.pop("replacement_reason", None)
        p.pop("input_tokens", None)
        p.update(preview_revision=4, preview_panel=panel, preview_state_scope=scope,
                 state_variant=state_variant, stride=10, preview_lightcone_stride=10,
                 temperature=1.0, enable_thinking=False, pairing_key=key,
                 topology=f"tp{tp}_dp1", context_gate_v1=None,
                 dataset_key=job.task, clean_server_per_cell=True,
                 frozen_recipe=preview_recipe(job.method, job.backend, manifest))
        if p["frozen_recipe"] is not None:
            p["frozen_recipe"].pop("context_gate_v1", None)
        if job.backend == "DSPARK" and job.method == "static" and panel in {"serving", "burstgpt"}:
            p["static_confidence_temperatures"] = preview_recipe("lightcone", "DSPARK", manifest)["confidence_temperatures"]
        if records is not None:
            if panel != "cohort":
                p["dataset_key"] = task
            p.update(preview_prompt_records=records, execution_request_count=len(records),
                     preview_request_seeds=[request_seed(job.block, r) for r in records],
                     regime="mechanism_native_prompt", generation_tokens=budget, respect_eos=True)
        identity = f"{key}__{job.backend}__{state_variant}"
        label = p["method_label"]
        if job.method == "lightcone":
            label += " [request reset]" if scope == "request" else " [persistent cohort]"
        p["method_label"] = label
        output.append(replace(job, job_id=identity, node=node, ordinal=len(output),
                              task=task, context=40928, gpu_count=tp, parameters=p))

    for job in source:
        panel = job.parameters["preview_panel"]
        if panel == "long_generation":
            tp = manifest.get("comparison_topologies", {}).get("dflash_long", 1)
            records = manifest["data"]["long8"][job.task][job.block * 8:(job.block + 1) * 8]
            append(job, node=NODES[1], panel="long_generation", task=job.task, records=records,
                   scope="request", state_variant=job.method, tp=tp, budget=32768,
                   key=f"preview-v4__long8__{job.task}__b{job.block}__tp{tp}")
        elif panel in {"serving", "burstgpt"}:
            tp = manifest.get("comparison_topologies", {}).get("dspark_serving", 1)
            append(job, node=NODES[2], panel=panel, task=job.task, records=None,
                   scope="cohort", state_variant=job.method, tp=tp, budget=None,
                   key=f"preview-v4__{panel}__{job.load}__b{job.block}__tp{tp}")
        elif panel == "qwen38_transfer":
            tp = manifest["qwen38"]["tp"]
            for task in DOMAINS.values():
                records = manifest["data"]["long27"][task][job.block * 8:(job.block + 1) * 8]
                append(job, node=NODES[3], panel=panel, task=task, records=records,
                       scope="request", state_variant=job.method, tp=tp, budget=32768,
                       key=f"preview-v4__long27__{task}__b{job.block}__tp{tp}")
    base = next(j for j in source if j.method == "lightcone" and j.parameters["preview_panel"] == "long_generation")
    tp = manifest.get("comparison_topologies", {}).get("cohort", 1)
    if type(tp) is not int or tp not in (1, 2):
        raise ValueError("cohort requires common TP1 or TP2")
    for block in range(4):
        for order in ORDERS:
            records = cohort_records(manifest["data"], block, order)
            key = f"preview-v4__cohort__{order}__b{block}__tp{tp}"
            for variant, method, scope in (("static", "static", "request"),
                                           ("request", "lightcone", "request"),
                                           ("cohort", "lightcone", "cohort")):
                clone = replace(base, block=block, method=method, load="c1",
                                parameters={**base.parameters, "method_label": "Static DFlash" if method == "static" else "LightCone DFlash S10",
                                            "flow_order": order, "sampling_seed": block})
                append(clone, node=NODES[0], panel="cohort", task="CohortMix", records=records,
                       scope=scope, state_variant=variant, tp=tp, key=key, budget=2048)
    # Freeze method order within whole comparison units, not global method sort.
    units = {}
    for job in output:
        units.setdefault((NODES.index(job.node), job.parameters["pairing_key"]), []).append(job)
    ordered = []
    for key, unit in sorted(units.items()):
        random.Random(str(key)).shuffle(unit)
        ordered.extend(unit)
    ordered = [replace(job, ordinal=i) for i, job in enumerate(ordered)]
    if Counter(j.node for j in ordered) != dict(zip(NODES, (36, 40, 32, 72), strict=True)):
        raise AssertionError("preview v4 matrix changed")
    if len({j.job_id for j in ordered}) != 180:
        raise AssertionError("preview v4 duplicate identity")
    return tuple(ordered)
