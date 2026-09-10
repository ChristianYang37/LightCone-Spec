"""Synthetic CPU fixture, never a GPU prompt pool."""

from lightcone_spec.preview import QWEN38_CHECKPOINTS
from lightcone_spec.preview_v4 import DOMAINS, freeze_data


def manifest_v4():
    pools = {task: [{"source": task, "problem_id": str(i), "prompt": f"{task}: {i}"}
                    for i in range(180)] for task in DOMAINS.values()}
    records = pools["LiveCodeBench"][:32]
    return {
        "version": 4, "preview_lightcone_stride": 10,
        "lightcone_recipe": {"stride": 10},
        "dspark_recipe": {"stride": 10, "confidence_temperatures": [1.1] * 7},
        "dflash_width": 16, "dspark_width": 16, "trace_request_count": 16,
        "serving_output_tokens": 256, "trace_offset": 17,
        "trace_anchor": {"source_job_ids": ["fixture-anchor"]},
        "trace": {"arrivals": list(range(16)), "lengths": [[128, 64]] * 16},
        "prompts": {task: records for task in ("MATH-500", "LiveCodeBench")},
        "qwen38": {"tp": 2, "checkpoints": QWEN38_CHECKPOINTS, "prompts": records[:8]},
        "data": freeze_data(pools, records),
    }
