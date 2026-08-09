"""CUDA event ordering for side-stream update and fixed-boundary publish."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import torch
from torch import Tensor


class CudaPublicationCoordinator:
    """No host synchronization is introduced by the update protocol."""

    def __init__(self, device: torch.device | str) -> None:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("CUDA publication requires a CUDA device")
        self.device = resolved
        # CUDA clamps out-of-range priorities to the nearest supported value.
        # A large positive value requests the device's lowest-priority stream
        # without relying on a version-specific priority-range API.
        self.side_stream = torch.cuda.Stream(
            device=resolved,
            priority=2**31 - 1,
        )
        self.main_ready = torch.cuda.Event(blocking=False)
        self.side_ready = torch.cuda.Event(blocking=False)
        self.publish_done = torch.cuda.Event(blocking=False)
        self._update_in_flight = False
        self._side_event_recorded = False

    @contextmanager
    def update_window(
        self,
        tensors: Sequence[Tensor],
        *,
        main_stream: torch.cuda.Stream | None = None,
    ) -> Iterator[torch.cuda.Stream]:
        if self._update_in_flight:
            raise RuntimeError("max_in_flight=1 publication window is occupied")
        self._update_in_flight = True
        try:
            main = main_stream or torch.cuda.current_stream(self.device)
            self.main_ready.record(main)
            self.side_stream.wait_event(self.main_ready)
            for tensor in tensors:
                tensor.record_stream(self.side_stream)
            with torch.cuda.stream(self.side_stream):
                yield self.side_stream
                self.side_ready.record(self.side_stream)
                self._side_event_recorded = True
        except BaseException:
            self._update_in_flight = False
            self._side_event_recorded = False
            raise

    def ready(self) -> bool:
        return self._side_event_recorded and self.side_ready.query()

    def discard(self) -> None:
        """Discard an unpublished candidate after a fail-closed diagnosis."""
        self._update_in_flight = False
        self._side_event_recorded = False

    def publish_boundary(
        self,
        *,
        publish: Callable[[], None],
        tensors: Sequence[Tensor],
        main_stream: torch.cuda.Stream | None = None,
    ) -> None:
        if not self._update_in_flight or not self._side_event_recorded:
            raise RuntimeError("no completed side-stream update is publishable")
        main = main_stream or torch.cuda.current_stream(self.device)
        main.wait_event(self.side_ready)
        try:
            with torch.cuda.stream(main):
                publish()
                for tensor in tensors:
                    tensor.record_stream(main)
                self.publish_done.record(main)
        finally:
            self._update_in_flight = False
            self._side_event_recorded = False
