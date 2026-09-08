"""Opt-in timing evidence. Inclusive lanes and GPU-rank clocks never sum to wall time."""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path


def interval_union(intervals):
    rows = sorted((float(a), float(b)) for a, b in intervals)
    merged = []
    for start, end in rows:
        if not math.isfinite(start + end) or start < 0 or end < start:
            raise ValueError("invalid clock interval")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def interval_ms(intervals):
    return sum(b - a for a, b in interval_union(intervals))


def overlap_ms(left, right):
    a, b = interval_union(left), interval_union(right)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "total_ms": 0.0, "p50_ms": None, "p95_ms": None, "p99_ms": None}

    def quantile(q):
        index = (len(values) - 1) * q
        low, high = math.floor(index), math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)

    return {"count": len(values), "total_ms": sum(values),
            **{f"p{int(q*100)}_ms": quantile(q) for q in (.5, .95, .99)}}


class TimingRecorder:
    """Bounded CUDA event queue; no per-round wait or silent loss.

    CUDA is injected, making bookkeeping testable without a device. Flush may
    wait at an explicitly excluded cell boundary, never during decoding.
    Request reset is a measured lane, NOT a recorder reset.
    """

    def __init__(self, *, cuda=None, rank=0, max_pending=4096, max_records=500000):
        self.cuda, self.rank = cuda, rank
        self.max_pending, self.max_records = max_pending, max_records
        self.pending = deque()
        self.records = []
        self.dropped = 0
        self.high_water = 0
        self.origin_cpu = time.perf_counter_ns()
        self.origin_gpu = None
        self.inflight = 0

    def append(self, row):
        if len(self.records) >= self.max_records:
            self.dropped += 1
        else:
            self.records.append(row)

    @contextmanager
    def span(self, name, *, gpu=False, lane="main"):
        cpu_start = (time.perf_counter_ns() - self.origin_cpu) / 1e6
        events = None
        if gpu and self.cuda is not None:
            if self.origin_gpu is None:
                self.origin_gpu = self.cuda.Event(enable_timing=True)
                self.origin_gpu.record()
            self.drain()
            if len(self.pending) + self.inflight < self.max_pending:
                start = self.cuda.Event(enable_timing=True)
                end = self.cuda.Event(enable_timing=True)
                start.record()
                events = (start, end)
                self.inflight += 1
            else:
                self.dropped += 1
        try:
            yield
        finally:
            if events:
                self.inflight -= 1
                events[1].record()
                stream = int(self.cuda.current_stream().cuda_stream)
                self.pending.append((name, lane, stream, *events))
                self.high_water = max(self.high_water, len(self.pending))
            self.append({"name": name, "clock": "cpu", "lane": lane, "rank": self.rank,
                         "start_ms": cpu_start,
                         "end_ms": (time.perf_counter_ns() - self.origin_cpu) / 1e6})

    def drain(self, *, boundary=False):
        while self.pending:
            name, lane, stream, start, end = self.pending[0]
            if not end.query():
                if not boundary:
                    break
                end.synchronize()
            self.pending.popleft()
            self.append({"name": name, "clock": "cuda", "lane": lane, "rank": self.rank,
                         "stream": stream, "start_ms": self.origin_gpu.elapsed_time(start),
                         "end_ms": self.origin_gpu.elapsed_time(end)})

    def snapshot(self):
        self.drain(boundary=True)
        return {"schema_version": 1, "measurement_scope": "excluded_inclusive_event_intervals",
                "rank": self.rank, "valid": self.dropped == 0 and not self.pending,
                "dropped_records": self.dropped, "pending_events": len(self.pending),
                "pending_high_water": self.high_water, "records": self.records}


def summarize_timing(snapshots, *, expected_ranks, wall_seconds=None, tokens=None, updates=None):
    ranks = [s["rank"] for s in snapshots]
    if len(set(ranks)) != len(ranks) or set(ranks) != set(expected_ranks):
        raise ValueError("timing report requires each expected rank exactly once")
    if any(not s["valid"] or s["dropped_records"] or s["pending_events"] for s in snapshots):
        raise ValueError("incomplete timing capture; do not accept performance attribution")
    report = {"scope": "inclusive CUDA event intervals, not hardware kernel occupancy",
              "wall_seconds": wall_seconds, "rank_reports": [],
              "main_blocked_ms": None, "unattributed_wall_ms": None,
              "limitation": "events include waits/contention; CPU and rank clocks are independent; use Nsys for kernel attribution"}
    for snapshot in snapshots:
        by_name = defaultdict(list)
        lanes = defaultdict(list)
        for row in snapshot["records"]:
            interval_union([(row["start_ms"], row["end_ms"])])
            if row["rank"] != snapshot["rank"]:
                raise ValueError("mixed rank clocks")
            by_name[(row["clock"], row["name"])].append(row["end_ms"] - row["start_ms"])
            if row["clock"] == "cuda" and row["lane"] in {"main", "side"}:
                lanes[row["lane"]].append((row["start_ms"], row["end_ms"]))
        summaries = []
        for (clock, name), values in sorted(by_name.items()):
            summary = {"clock": clock, "name": name, **distribution(values)}
            summary["ms_per_1000_output_tokens"] = sum(values) * 1000 / tokens if tokens else None
            summary["ms_per_publication"] = sum(values) / updates if updates else None
            summaries.append(summary)
        report["rank_reports"].append({"rank": snapshot["rank"], "stages": summaries,
            "main_interval_union_ms": interval_ms(lanes["main"]),
            "side_interval_union_ms": interval_ms(lanes["side"]),
            "event_interval_overlap_ms": overlap_ms(lanes["main"], lanes["side"]),
            "kernel_overlap_ms": None})
    return report


def legacy_timing_report(evidence):
    """Read-only historical audit: missing samples/other ranks stay unmeasured."""
    rows = []
    for config, metrics in evidence:
        timings = metrics.get("timings_ms", {})
        updates, tokens = metrics.get("updates_published", 0), metrics.get("committed_tokens", 0)
        for lane, elapsed in timings.items():
            rows.append({"job_id": config["job_id"], "task": config["task"], "method": config["method"],
                         "block": config.get("block"), "lane": lane, "total_ms": elapsed,
                         "ms_per_publication": elapsed / updates if updates else None,
                         "ms_per_1000_output_tokens": elapsed * 1000 / tokens if tokens else None,
                         "wall_seconds": metrics.get("duration_seconds"),
                         "rank_scope": "legacy reducer; not all-rank timeline",
                         "window_validated": False, "p50_ms": None, "p95_ms": None, "p99_ms": None})
    return {"rows": rows, "status": "historical counters only; window/reset audit required",
            "measured_kernel_overlap": None}


def write_timing_report(report, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    (output / "timing.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    rows = report.get("rows")
    if rows is None:
        rows = [{"rank": rank["rank"], **row} for rank in report["rank_reports"] for row in rank["stages"]]
    if rows:
        with (output / "timing.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    columns = list(rows[0]) if rows else []
    table = "<tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in columns) + "</tr>"
    for row in rows:
        table += "<tr>" + "".join(f"<td>{html.escape(str(row[k]) if row[k] is not None else 'UNMEASURED')}</td>" for k in columns) + "</tr>"
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>LightCone timing audit</title>'
        '<style>body{font:14px system-ui;margin:28px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}th{background:#eee}</style>'
        '<h1>LightCone timing audit</h1><p>Inclusive intervals may overlap. Do not sum streams/ranks into wall time. Missing measurements are not zero.</p>'
        '<table>' + table + '</table>')


def paired_speed_gain(new, baseline):
    if len(new) != len(baseline) or len(new) < 2 or any(not math.isfinite(v) or v <= 0 for v in (*new, *baseline)):
        raise ValueError("paired finite positive measurements required")
    return math.exp(statistics.mean(math.log(a / b) for a, b in zip(new, baseline, strict=True))) - 1
