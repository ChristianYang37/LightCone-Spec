"""Loss-detecting event sink for excluded REAL streaming recordings.

New contributions: LicenseRef-LightCone-Source-Available-1.0. See LICENSE.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from pathlib import Path


def validate_recording(events, requests, duration, *, expected_requests=8):
    """Compare the complete observer trajectory with native final records."""
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("invalid recording duration")
    rows = {r["request_id"]: r for r in requests}
    if len(requests) != expected_requests or len(rows) != expected_requests:
        raise RuntimeError("recording request count/identity mismatch")
    tokens = {rid: [] for rid in rows}
    finished = set()
    last_time = -1.
    for sequence, event in enumerate(events, 1):
        rid = event["request_id"]
        elapsed = event["elapsed_seconds"]
        if (event["sequence"] != sequence or rid not in rows
                or not math.isfinite(elapsed) or elapsed < last_time or elapsed > duration):
            raise RuntimeError("recording event order/identity mismatch")
        last_time = elapsed
        ids = event["token_ids"]
        if not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise RuntimeError("recording token IDs missing")
        tokens[rid].extend(ids)
        chunk = event["chunk"]
        if chunk.get("output_ids") != tokens[rid]:
            raise RuntimeError("recording trajectory mismatch")
        if chunk.get("meta_info", {}).get("finish_reason") is not None:
            finished.add(rid)
    if finished != set(rows):
        raise RuntimeError("recording missing final events")
    for rid, row in rows.items():
        timestamps = row.get("native_token_timestamps_ns", [])
        if (tokens[rid] != list(row["output_ids"]) or len(tokens[rid]) != row["completion_tokens"]
                or not row.get("stop_reason") or len(timestamps) != len(tokens[rid])
                or any(b < a for a, b in zip(timestamps, timestamps[1:]))):
            raise RuntimeError("recording final/native token count mismatch")
    total = sum(len(ids) for ids in tokens.values())
    return {"event_count": len(events), "committed_tokens": total,
            "aggregate_tok_s": total / duration, "event_accounting": "verified_native_final_records"}


def recording_nvml_peaks(path, gpus):
    peaks = {int(gpu): 0 for gpu in gpus}
    for line in Path(path).read_text().splitlines()[1:]:
        fields = line.split(",")
        gpu = int(fields[1])
        if gpu not in peaks:
            raise RuntimeError("recording NVML sampled an unassigned GPU")
        peaks[gpu] = max(peaks[gpu], int(float(fields[2]) * 1024 * 1024))
    if any(value <= 0 for value in peaks.values()):
        raise RuntimeError("recording NVML window missing a GPU")
    return {"nvml_rank_peak_bytes": peaks, "nvml_peak_hbm_bytes": max(peaks.values()),
            "sum_nvml_rank_peak_bytes": sum(peaks.values()),
            "nvml_peak_scope": "generation-window sampled per-rank peaks; sum is not simultaneous peak"}


class StreamRecording:
    """Never block generation on a disk write; overflow invalidates the recording."""

    def __init__(self, path: Path, capacity: int = 8192):
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.events = []
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.closed = False
        self.count = 0
        self.path = path
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def __call__(self, event: dict) -> None:
        if self.error or self.closed:
            raise RuntimeError(f"recording unavailable: {self.error or 'closed'}")
        try:
            self.queue.put_nowait(event)
        except queue.Full as error:
            self.error = "stream event queue overflow; recording invalid"
            raise RuntimeError(self.error) from error
        self.count += 1

    def _write(self):
        try:
            with self.path.open("x", encoding="utf-8") as stream:
                while True:
                    event = self.queue.get()
                    if event is None:
                        break
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
                    stream.flush()
                    with self.lock:
                        self.events.append(event)
                        self.changed.notify_all()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def snapshot(self, since: int = 0) -> list[dict]:
        with self.lock:
            return self.events[since:]

    def wait_since(self, since: int, timeout: float = 1.0) -> list[dict]:
        """Replay from a cursor; waiting browsers never block generation."""
        with self.changed:
            if since < 0 or since > len(self.events):
                raise ValueError("invalid stream cursor")
            if since == len(self.events) and not self.closed and not self.error:
                self.changed.wait(timeout)
            return self.events[since:]

    def close(self):
        if self.closed:
            return
        self.closed = True
        # A failed writer must not deadlock on a full queue.
        deadline = time.monotonic() + 10
        while self.thread.is_alive() and time.monotonic() < deadline:
            try:
                self.queue.put(None, timeout=.1)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=10)
        if self.thread.is_alive() or self.error or len(self.events) != self.count:
            raise RuntimeError(f"recording incomplete: {self.error or 'writer/count mismatch'}")
