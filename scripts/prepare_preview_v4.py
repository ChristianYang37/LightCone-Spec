"""Freeze v4 confirmation/cohort data without mutating SQLite or powering on GPUs."""

import argparse
import hashlib
import json
from pathlib import Path

from lightcone_spec.data import load_prompt_pool, load_source_prompt_records
from lightcone_spec.preview_v4 import DOMAINS, freeze_data, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path, help="Existing frozen preview model/recipe/serving manifest")
    parser.add_argument("--sources", required=True, type=Path, help="Task -> {path, revision} JSON; local server prompt-only pools")
    parser.add_argument("--exclusions", required=True, type=Path, help="Reviewed {records, covered_categories} JSON for all historical prompt uses")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("never overwrite a frozen manifest")
    source_map = json.loads(args.sources.read_text())
    inventory = json.loads(args.exclusions.read_text())
    if set(inventory["covered_categories"]) != {"tuning", "preview", "diagnostic", "synthetic_benchmark"}:
        raise ValueError("exclusion inventory must cover all four historical usage categories")
    pools, provenance = {}, {}
    for task in DOMAINS.values():
        source = source_map[task]
        if not source.get("revision"):
            raise ValueError(f"missing source revision: {task}")
        path = Path(source["path"])
        records = (load_source_prompt_records(path, max_samples=int(source["source_rows"]), seed=0)
                   if source.get("source_rows") else load_prompt_pool(path))
        pools[task] = [{**r, "source": task} for r in records]
        provenance[task] = {"revision": source["revision"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = json.loads(args.template.read_text())
    manifest.update(version=4, preview_lightcone_stride=10, data=freeze_data(pools, inventory["records"]),
                    data_sources=provenance,
                    exclusion_inventory_sha256=hashlib.sha256(args.exclusions.read_bytes()).hexdigest())
    manifest["data_policy"] = {
        "confirmation": "historical-use-excluded; same prompts across models",
        "cohort": "historical corpus reuse allowed; cold start; four disjoint blocks",
        "video": "selected illustration; historical corpus reuse allowed",
        "within_preview": "confirmation, cohort and video pools disjoint",
    }
    manifest.setdefault("comparison_topologies", {}).update(cohort=1, dflash_long=1, dspark_serving=1)
    manifest["qwen38"]["tp"] = 1  # common minimum tested before group acceptance
    rows = jobs(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {len(rows)} v4 cells; no SQLite/GPU changes. Manifest contains private prompts.")


if __name__ == "__main__":
    main()
