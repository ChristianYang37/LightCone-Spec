"""Frozen synthetic context benchmark; isolated from the formal paper DAG."""

import hashlib
import html
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

VERSION = "preview-context-v1"
DOMAINS = {
    "Chat": {"MT-Bench": 1, "AlpacaEval": 1, "Arena-Hard": 2},
    "Code": {"HumanEval": 1, "MBPP": 1, "LiveCodeBench": 2},
    "Math": {"GSM8K": 1, "MATH-500": 2, "AIME-2025": 1},
}
MODES = ("static", "always_s10", "gated_s10")
OUTPUT_TOKENS = 4096


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"frozen artifact differs: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sample_sets(pools, exclusions):
    """Dataset IDs and prompt text both prevent calibration/evaluation leakage."""
    used_ids = {(r.get("dataset", "*"), str(r["problem_id"])) for r in exclusions if r.get("problem_id") is not None}
    used_text = {str(r["prompt"]).strip() for r in exclusions if r.get("prompt")}
    splits = {"calibration": [], "evaluation": [], "background": []}
    for domain, datasets in DOMAINS.items():
        for dataset, count in datasets.items():
            available = sorted(pools[dataset], key=lambda r: str(r["problem_id"]))
            random.Random(f"{VERSION}:0:{dataset}").shuffle(available)
            clean = []
            for row in available:
                identity, prompt = str(row["problem_id"]), row["prompt"].strip()
                if (dataset, identity) in used_ids or ("*", identity) in used_ids or prompt in used_text:
                    continue
                used_ids.add((dataset, identity))
                used_text.add(prompt)
                clean.append({"dataset": dataset, "problem_id": identity,
                              "domain": domain, "prompt": prompt})
            if len(clean) < 2 * count + 1:
                raise ValueError(f"insufficient disjoint samples: {dataset}")
            splits["calibration"].extend(clean[:count])
            splits["evaluation"].extend(clean[count:2 * count])
            splits["background"].extend(clean[2 * count:])
    return splits


def construct_inputs(tokenizer, splits):
    """Exact IDs including template. Background is prompt-only, never repeated."""
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)
    background = {
        domain: encode("\n\n".join(r["prompt"] for r in splits["background"] if r["domain"] == domain))
        for domain in DOMAINS
    }
    results = []
    for split in ("calibration", "evaluation"):
        for sample, row in enumerate(splits[split]):
            # Split at a sentinel to retain the real chat template and task at the end.
            sentinel = "LIGHTCONE_BACKGROUND_SENTINEL_19"
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": sentinel + "\n\nTask:\n" + row["prompt"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if text.count(sentinel) != 1:
                raise ValueError("chat template did not preserve background sentinel")
            before, after = text.split(sentinel)
            prefix, suffix = encode(before), encode(after)
            short = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if not 0 < len(short) < 4096:
                raise ValueError("frozen task does not fit the short-input reference")
            for bucket in range(10):
                length = bucket * 4096
                if not bucket:
                    tokens = list(short)
                else:
                    needed = length - len(prefix) - len(suffix)
                    if needed < 0 or needed > len(background[row["domain"]]):
                        raise ValueError("insufficient non-repeated background for exact input length")
                    tokens = prefix + background[row["domain"]][:needed] + suffix
                    assert len(tokens) == length
                results.append({**row, "split": split, "sample": sample, "bucket": bucket,
                                "seed": sample, "input_ids": tokens, "input_tokens": len(tokens),
                                "output_tokens": OUTPUT_TOKENS})
    return results


def cases(manifest, phase):
    split = "calibration" if phase == "calibrate" else "evaluation"
    result = []
    for row in sorted(manifest["inputs"], key=lambda r: (r["bucket"], r["sample"])):
        if row["split"] != split:
            continue
        modes = ["static"] if phase == "calibrate" else list(MODES)
        rotation = (row["bucket"] + row["sample"]) % len(modes)
        for mode in modes[rotation:] + modes[:rotation]:
            result.append({**row, "mode": mode,
                "id": f"{VERSION}-{split}-{row['sample']:02d}-{row['bucket']:02d}-{mode}"})
    if len(result) != (120 if phase == "calibrate" else 360):
        raise ValueError("benchmark must contain exactly 120/360 calls")
    return result


def checked_rows(rows, split, modes):
    found = {}
    for row in rows:
        if row["split"] != split or row["mode"] not in modes:
            continue
        key = row["mode"], row["sample"], row["bucket"]
        if key in found or row.get("status") != "completed" or row.get("output_tokens") != 4096:
            raise ValueError("duplicate, failed, or incomplete benchmark evidence")
        if row["domain"] not in DOMAINS:
            raise ValueError("unknown domain")
        for name in ("throughput", "decode_speed", "al"):
            if not isinstance(row[name], (int, float)) or not math.isfinite(row[name]) or row[name] <= 0:
                raise ValueError(f"invalid {name}")
        found[key] = row
    expected = {(m, i, j) for m in modes for i in range(12) for j in range(10)}
    if set(found) != expected:
        raise ValueError("missing benchmark evidence")
    for mode in modes:
        for bucket in range(10):
            if any(sum(found[mode, i, bucket]["domain"] == d for i in range(12)) != 4 for d in DOMAINS):
                raise ValueError("domain balance differs from 4/4/4")
    return found


def calibrate(rows, environment):
    found = checked_rows(rows, "calibration", ("static",))
    critical = float(t.ppf(1 - .05 / 18, 11))
    intervals = []
    for bucket in range(1, 10):
        metrics = {}
        for name in ("decode_speed", "al"):
            ratios = np.array([math.log(found["static", i, bucket][name] /
                                       found["static", i, 0][name]) for i in range(12)])
            mean, sd = float(ratios.mean()), float(ratios.std(ddof=1))
            metrics[name] = {"mean_log_ratio": mean, "sample_std": sd,
                             "upper_bound": mean + critical * sd / math.sqrt(12)}
        intervals.append({"bucket": bucket, "metrics": metrics,
                          "trigger": all(m["upper_bound"] < math.log(.95) for m in metrics.values())})
    threshold = next((a["bucket"] * 4096 for a, b in zip(intervals, intervals[1:])
                      if a["trigger"] and b["trigger"]), None)
    return {"version": VERSION, "environment": environment, "threshold": threshold,
            "max_context": 40960, "status": "candidate" if threshold is not None else "no_trigger_detected",
            "intervals": intervals, "critical_t": critical,
            "interpretation": "acceptance degradation with slowdown; not a causal proof of distribution drift",
            "uncertainty": "12 paired samples; approximate normal log ratios; Bonferroni 18; not run repeats"}


def cached_gate(calibration, environment):
    if not calibration or calibration.get("version") != VERSION or calibration.get("environment") != environment:
        return {"threshold": None, "max_context": 40960}, "uncalibrated"
    if calibration.get("status") not in {"candidate", "no_trigger_detected"}:
        raise ValueError("invalid calibration cache status")
    return {"threshold": calibration["threshold"], "max_context": 40960}, calibration["status"]


def summarize(rows, *, complete=False):
    if complete:
        checked_rows(rows, "evaluation", MODES)
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed" and row["split"] == "evaluation":
            groups[row["mode"], row["domain"], row["bucket"]].append(row)
    tables, scores = [], {}
    for mode in MODES:
        for domain in DOMAINS:
            for bucket in range(10):
                data = groups[mode, domain, bucket]
                item = {"mode": mode, "domain": domain, "bucket": bucket, "n": len(data)}
                for metric in ("throughput", "al", "decode_speed"):
                    values = [r[metric] for r in data]
                    item[metric] = {"mean": float(np.mean(values)) if len(values) == 4 else None,
                                    "sample_variance": float(np.var(values, ddof=1)) if len(values) == 4 else None,
                                    "points": values}
                tables.append(item)
        selected = [r for r in tables if r["mode"] == mode]
        scores[mode] = {k: math.exp(sum(math.log(r[k]["mean"]) for r in selected) / 30)
                       if all(r[k]["mean"] is not None for r in selected) else None
                       for k in ("throughput", "al")}
    effects = {}
    if complete:
        found = checked_rows(rows, "evaluation", MODES)
        for base in ("static", "always_s10"):
            ratios = [np.mean([math.log(found["gated_s10", i, j]["throughput"] /
                                       found[base, i, j]["throughput"]) for j in range(10)]) for i in range(12)]
            mean, se = float(np.mean(ratios)), float(np.std(ratios, ddof=1) / math.sqrt(12))
            half = float(t.ppf(.975, 11)) * se
            effects[base] = {"ratio": math.exp(mean), "ci95": [math.exp(mean-half), math.exp(mean+half)],
                             "paired_log_ratios": list(map(float, ratios)), "independent_samples": 12}
    comparisons = []
    indexed = {(r["mode"], r["domain"], r["bucket"]): r for r in tables}
    for mode in MODES[1:]:
        for metric in ("throughput", "al"):
            ratios = []
            for domain in DOMAINS:
                domain_ratios = []
                for bucket in range(10):
                    candidate = indexed[mode, domain, bucket][metric]["mean"]
                    base = indexed["static", domain, bucket][metric]["mean"]
                    if candidate is not None and base is not None:
                        domain_ratios.append(candidate / base)
                        ratios.append({"domain": domain, "bucket": bucket, "ratio": candidate / base})
                comparisons.append({"mode": mode, "metric": metric, "domain": domain,
                    "ratio": math.exp(np.mean(np.log(domain_ratios))) if len(domain_ratios) == 10 else None})
            comparisons.append({"mode": mode, "metric": metric, "domain": "all",
                "ratio": math.exp(np.mean(np.log([r["ratio"] for r in ratios]))) if len(ratios) == 30 else None,
                "worst_cell": min(ratios, key=lambda r: r["ratio"]) if len(ratios) == 30 else None})
    return {"version": VERSION, "status": "completed" if complete else "partial",
            "tables": tables, "scores": scores, "paired_effects": effects,
            "comparisons_to_static": comparisons,
            "scope": "synthetic ignore-EOS, sample-level variation, not independent-run replication"}


def write_report(directory, rows, provenance, *, complete=False):
    import csv
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = {**summarize(rows, complete=complete), "provenance": provenance, "rows": rows}
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = ["# Synthetic context benchmark", "", report["scope"], ""]
    for mode in MODES:
        for metric in ("throughput", "al"):
            lines.extend([f"## {mode}: {metric}", "", "|Domain|Short→+4K|" + "|".join(f"{j*4}–{(j+1)*4}K" for j in range(1, 10)) + "|",
                          "|---|" + "---|" * 10])
            for domain in DOMAINS:
                values = [r[metric]["mean"] for r in report["tables"] if r["mode"] == mode and r["domain"] == domain]
                lines.append(f"|{domain}|" + "|".join("UNMEASURED" if v is None else f"{v:.3f}" for v in values) + "|")
            lines.append("")
    markdown = "\n".join(lines)
    (directory / "report.md").write_text(markdown)
    data = json.dumps({"tables": report["tables"], "scores": report["scores"]}).replace("<", "\\u003c")
    (directory / "index.html").write_text('''<!doctype html><meta charset="utf-8">
<title>Preview context benchmark</title><style>body{font:16px system-ui;margin:32px;color:#172534}
table{border-collapse:collapse;margin:20px 0;width:100%}td,th{padding:10px;border:1px solid #ccd}
small{color:#456}select{padding:8px}pre{white-space:pre-wrap}</style>
<h1>Qwen3-8B · context benchmark</h1><p>Synthetic ignore-EOS · TP2 · c1 · S10</p>
<label>Metric <select id="metric"><option>throughput</option><option>al</option></select></label>
<label>Method <select id="mode"><option>static</option><option>always_s10</option><option>gated_s10</option></select></label>
<p id="score"></p><div id="table"></div><h2>Ratio to matched Static</h2><div id="delta"></div>
<small>Each cell: four source samples, not four independent runs. Hover to see sample variance.
Short means original input + 4096 output; remaining columns mean exact prefill + 4096 output.</small>
<details><summary>All tables</summary><pre>''' + html.escape(markdown) + '</pre></details><script>const data=' + data + ''';
const metric=document.querySelector('#metric'),mode=document.querySelector('#mode');
function render(){const k=metric.value,m=mode.value;function table(delta){
let h='<table><thead><tr><th>Domain</th>'+Array.from({length:10},(_,j)=>'<th>'+(j?j*4+'–'+(j+1)*4+'K':'Short→+4K')+'</th>').join('')+'</tr></thead><tbody>';
for(const d of ['Chat','Code','Math']){h+='<tr><th>'+d+'</th>';for(let j=0;j<10;j++){
const r=data.tables.find(x=>x.mode===m&&x.domain===d&&x.bucket===j)[k];
const b=data.tables.find(x=>x.mode==='static'&&x.domain===d&&x.bucket===j)[k];
const v=delta?(r.mean===null||b.mean===null?null:r.mean/b.mean):r.mean;
h+='<td title="Sample variance: '+r.sample_variance+'">'+(v===null?'UNMEASURED':v.toFixed(3)+(delta?'×':''))+'</td>';
}h+='</tr>';}return h+'</tbody></table>';}
document.querySelector('#table').innerHTML=table(false);document.querySelector('#delta').innerHTML=table(true);
const s=data.scores[m][k];document.querySelector('#score').textContent='30-cell geometric mean: '+(s===null?'UNMEASURED':s.toFixed(3));}
metric.onchange=mode.onchange=render;render();</script>''')
    with (directory / "tables.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mode", "domain", "bucket", "n", "metric", "mean", "sample_variance", "points"])
        for row in report["tables"]:
            for metric in ("throughput", "al", "decode_speed"):
                writer.writerow([row[k] for k in ("mode", "domain", "bucket", "n")] + [metric,
                    row[metric]["mean"], row[metric]["sample_variance"], json.dumps(row[metric]["points"])])
    return report


def validate_report(report, manifest, *, candidate_commit, environment):
    """Structural/score verifier. Trust of the GPU producer is a separate maintainer gate."""
    provenance = report.get("provenance", {})
    if (report.get("version") != VERSION or report.get("status") != "completed"
            or provenance.get("commit") != candidate_commit or provenance.get("manifest") != digest(manifest)
            or provenance.get("environment") != environment or manifest.get("environment") != environment):
        raise ValueError("report candidate/manifest/environment mismatch")
    checked_rows(report["rows"], "calibration", ("static",))
    checked_rows(report["rows"], "evaluation", MODES)
    for row in report["rows"]:
        if row.get("provenance") != provenance:
            raise ValueError("call has mismatched provenance")
        derived = {"throughput": row["output_tokens"] / row["duration_seconds"],
                   "al": row["delivered_verify_tokens"] / row["target_calls"],
                   "decode_speed": (row["output_tokens"] - 1) * 1e9 /
                       (row["native_last_token_ns"] - row["native_first_token_ns"])}
        if (row["prefill_generated_tokens"] != 1 or row["delivered_verify_tokens"] != 4095
                or any(not math.isclose(row[k], v, rel_tol=1e-12) for k, v in derived.items())):
            raise ValueError("call metrics cannot be reproduced from raw counts/timing")
    expected = summarize(report["rows"], complete=True)
    if any(report.get(key) != expected[key] for key in ("tables", "scores", "paired_effects", "comparisons_to_static")):
        raise ValueError("report scores cannot be reproduced")
    return {"status": "verified", "candidate_commit": candidate_commit, "calibration_calls": 120,
            "evaluation_calls": 360, "trust": "requires maintainer review of GPU origin"}
