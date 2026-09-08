"""Opt-in failed-candidate evidence; never enabled for performance collection."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def install_reconstruction_capture(adapter):
    directory = os.environ.get("LIGHTCONE_DFLASH_RECONSTRUCTION_DIR")
    if not directory:
        return None
    return FailedReconstructionCapture(adapter, Path(directory))


class FailedReconstructionCapture:
    """Keep one bounded canvas, write only after the host has rejected it.

    No additional device-to-host validity read is introduced before rejection.
    The existing boundary has already waited for the candidate event. Current
    history is also saved to detect writes between proposal and that boundary.
    Checkpoint weights are NOT copied: only the selected active inference bank
    and optimizer state, plus the immutable checkpoint's parameter metadata.
    """

    def __init__(self, adapter, directory):
        self.adapter = adapter
        self.directory = directory
        self.start_round = int(os.environ.get("LIGHTCONE_DFLASH_CAPTURE_FROM_ROUND", "0"))
        if self.start_round < 0:
            raise ValueError("capture start round must be nonnegative")
        self.payload = None
        self.saved = False
        original = adapter.runtime._record_invalid_candidate

        def rejected(trace, **flags):
            fresh = not trace.diagnosed
            original(trace, **flags)  # Preserve rejection even if diagnostic I/O fails.
            if fresh and flags.get("supervision_ok", True) and not flags["reconstruction_ok"]:
                self.save(trace, flags)

        adapter.runtime._record_invalid_candidate = rejected

    def history(self):
        a = self.adapter
        return [
            tuple(t.detach().clone() for t in a._gather_history(
                layer_id=int(layer.self_attn.attn.layer_id),
                locations=a._captured_history.locations,
            ))
            for layer in a.model.layers
        ]

    def stage(self, *, source_round, source_version, draft_hidden, inference_logits,
              replay_logits, valid_mask):
        if self.saved or source_round < self.start_round:
            return
        a = self.adapter
        self.payload = {
            "source_round": source_round, "source_version": source_version,
            "request_ids": a._captured_request_ids,
            "input_embeds": a._captured_input.detach(),
            "positions": a._captured_positions.detach(),
            "prefix_lens": a._captured_prefix_lens.detach(),
            "history_locations": a._captured_history.locations.detach(),
            "history_valid": a._captured_history.valid_mask.detach(),
            "history_at_proposal": self.history(),
            "draft_hidden": draft_hidden.detach(),
            "inference_logits": inference_logits.detach(),
            "replay_logits": replay_logits.detach(), "valid_mask": valid_mask.detach(),
        }

    def save(self, trace, flags):
        if self.saved or self.payload is None:
            return
        if self.payload["source_round"] != trace.source_round:
            raise RuntimeError("reconstruction diagnostic candidate identity mismatch")
        a = self.adapter
        if self.payload["source_version"] != a.runtime.active_version:
            raise RuntimeError("reconstruction diagnostic active version changed")
        payload = {
            **self.payload,
            "measurement_scope": "excluded_failed_reconstruction",
            "flags": flags, "history_at_boundary": self.history(),
            "active_weights": dict(zip(a.names, a.inference.active, strict=True)),
            "master": a.optimizer.master,
            "parameter_metadata": {
                name: {"shape": list(t.shape), "dtype": str(t.dtype)}
                for name, t in a.named.items()
            },
            "tp_world_size": a.tp_group.world_size,
        }

        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [cpu(v) for v in value]
            return value

        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"candidate-{trace.source_round}-pid{os.getpid()}.pt"
        with path.open("xb") as stream:
            torch.save(cpu(payload), stream)
        # A bounded diagnostic supervisor must not mistake a partial torch.save
        # file for a completed snapshot and terminate its writer.
        path.with_suffix(".complete").touch(exist_ok=False)
        self.saved = True
        self.payload = None
