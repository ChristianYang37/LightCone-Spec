"""Loss-detecting event sink for excluded REAL streaming recordings.

New contributions: LicenseRef-LightCone-Source-Available-1.0. See LICENSE.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path


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
