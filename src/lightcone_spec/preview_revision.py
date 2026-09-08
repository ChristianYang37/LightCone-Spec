"""Preview v3 only: explicit algorithm transfers, never formal recipe mutation."""

import random
from copy import deepcopy
from dataclasses import replace

PREVIEW_V3_NODES = ("E3b-preview-v3", "E5-preview-v3", "Qwen38-preview-v3")
ENSEMBLE_TRANSFER = {
    "parameterization": "full", "scope": "all", "rank": None,
    "optimizer": "adam", "learning_rate": 1e-4,
    "additional_learning_rates": [2e-4, 4e-4], "beta1": .9, "beta2": .95,
    "epsilon": 1e-8, "grad_clip": .5, "weight_decay": 0.,
    "stride": 10, "schedule": "constant", "hedge_learning_rate": 10.,
    "ensemble_optimizer": "adam_preview_v3",
    "source_backend": "EAGLE3", "source_epochs": 2, "source_chunk_size": 40,
    "source": "https://arxiv.org/html/2603.12617v2#A2.SS10",
    "source_transfer": "in_process_dflash_ensemble_v3",
    "transfer_difference": "one update per S10 event; no chunk/epoch pipeline; cumulative-loss Hedge",
}


def preview_lightcone_stride(manifest):
    """Absent means historical S1; never reinterpret an archived manifest."""
    stride = manifest.get("preview_lightcone_stride", 1)
    if type(stride) is not int or stride not in (1, 10):
        raise ValueError("preview LightCone stride must be 1 (legacy) or 10")
    return stride


def s10_manifest(manifest):
    candidate = deepcopy(manifest)
    candidate["preview_lightcone_stride"] = 10
    return candidate


def preview_recipe(method, backend, manifest):
    if method == "onlinespec_ens":
        return deepcopy(ENSEMBLE_TRANSFER)
    if method != "lightcone":
        return None
    recipe = deepcopy(manifest["dspark_recipe" if backend == "DSPARK" else "lightcone_recipe"])
    recipe.update(optimizer="chronobelief", parameterization="lora", rank=8,
                  scope="last1", lr=1e-3, learning_rate=1e-3, schedule="constant",
                  stride=preview_lightcone_stride(manifest))
    if backend == "DSPARK":
        import math
        temperatures = recipe.get("confidence_temperatures", [])
        if len(temperatures) != 7 or not all(isinstance(t, (int, float)) and math.isfinite(t) and t > 0 for t in temperatures):
            raise ValueError("preview DSpark requires seven frozen STS temperatures")
    return recipe


def revision_jobs(manifest):
    """Reuse frozen input/trace construction, not legacy outcomes or TTS recipes."""
    from .preview import PREVIEW_NODES, preview_jobs

    # Legacy construction is a pure template; these rows are never scheduled.
    template = deepcopy(manifest)
    template.update(version=2, tts_recipe={"lr": 1e-4, "stride": 10})
    template["lightcone_recipe"]["stride"] = 10
    template.pop("comparison_topologies", None)
    if template.get("qwen38"):
        template["qwen38"]["tp"] = manifest["qwen38"]["tp"]
    rows = preview_jobs(template)
    expanded = []
    for job in rows:
        if job.method == "tts":
            job = replace(job, method="onlinespec_ens")
        expanded.append(job)
        if job.parameters["preview_panel"] == "long_generation" and job.method == "static":
            expanded.extend((replace(job, method="target_only", backend="NONE", width=None),
                             replace(job, backend="EAGLE3")))
    units = {}
    for job in expanded:
        units.setdefault(job.parameters["pairing_key"], []).append(job)
    result = []
    for old_key, unit in units.items():
        key = old_key.replace("preview-v1", "preview-v3")
        random.Random(key).shuffle(unit)
        for job in unit:
            panel = job.parameters["preview_panel"]
            group = "dspark_serving" if panel in {"serving", "burstgpt"} else "dflash_long"
            tp = manifest["qwen38"]["tp"] if panel == "qwen38_transfer" else manifest.get("comparison_topologies", {}).get(group, 1)
            if type(tp) is not int or tp not in (1, 2):
                raise ValueError("preview needs common TP1 or TP2")
            params = deepcopy(job.parameters)
            params.update(preview_revision=3, pairing_key=key, topology=f"tp{tp}_dp1",
                          stride=preview_lightcone_stride(manifest) if job.method == "lightcone" else 10,
                          frozen_recipe=preview_recipe(job.method, job.backend, manifest))
            if panel == "burstgpt" and tp == 2:
                anchor = manifest.get("tp2_trace_anchor", {})
                if anchor.get("topology") != "tp2_dp1" or not anchor.get("source_job_ids"):
                    raise ValueError("TP2 preview trace requires measured matched anchor")
                params["preview_anchor"] = deepcopy(anchor)
            if job.method == "onlinespec_ens":
                label = f"OnlineSPEC-Ensemble ({'DFlash2' if panel == 'qwen38_transfer' else 'DFlash'} transfer)"
            elif job.method == "target_only":
                label = "Target-only"
            elif job.method == "lightcone":
                label = f"LightCone ({job.backend}, S={preview_lightcone_stride(manifest)})"
            else:
                label = "Native MTP" if job.backend == "NEXTN" else f"Static {job.backend}"
            params["method_label"] = label
            identity = f"preview-v3__{panel}__{job.task}__{job.load}__b{job.block}__{job.backend}__{job.method}__tp{tp}"
            if job.method == "lightcone" and preview_lightcone_stride(manifest) == 10:
                params["preview_lightcone_stride"] = 10
                params["replaces_job_id"] = identity
                params["replacement_reason"] = "user restored preview LightCone S10; retain legacy S1 raw evidence"
                identity += "__s10"
            result.append(replace(job, job_id=identity, node=PREVIEW_V3_NODES[PREVIEW_NODES.index(job.node)],
                                  ordinal=len(result), gpu_count=tp, parameters=params))
    expected = 96 if manifest.get("qwen38") else 72
    if len(result) != expected or len({j.job_id for j in result}) != expected:
        raise AssertionError("preview-v3 logical identity count changed")
    return tuple(result)
