"""Fresh, scheduler-local TP receipts. Never a global Python trace hook."""

import json
import time


def read_rank_receipts(directory, tp, not_before_ns, timeout=5.):
    deadline = time.monotonic() + timeout
    while True:
        rows = []
        for rank in range(tp):
            path = directory / f"rank-{rank}-metrics.jsonl"
            try:
                # Only read the tail. Receipts can grow throughout a long cell.
                with path.open("rb") as stream:
                    stream.seek(0, 2)
                    length = stream.tell()
                    span = min(length, 65536)
                    while True:
                        stream.seek(length - span)
                        data = stream.read(span).splitlines()
                        if len(data) >= 2 or span == length:
                            row = json.loads(data[-1])
                            break
                        span = min(length, span * 2)
            except (FileNotFoundError, IndexError, json.JSONDecodeError):
                break
            if row["tp_rank"] != rank or row["tp_size"] != tp:
                raise RuntimeError("rank receipt topology mismatch")
            if row["captured_ns"] < not_before_ns:
                break
            rows.append(row["state"])
        if len(rows) == tp:
            return {"internal_states": rows}
        if time.monotonic() >= deadline:
            raise RuntimeError("missing fresh TP rank receipts")
        time.sleep(.01)
