"""Opt-in, excluded GPU diagnostics; never used as benchmark evidence."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import torch


@contextlib.contextmanager
def capture_native_attention(model):
    if not os.environ.get("LIGHTCONE_REPLAY_DIAGNOSTIC_DIR"):
        yield
        return
    handles = []
    model._replay_diagnostic_layers = []
    try:
        for layer in model.layers:
            row = {}
            model._replay_diagnostic_layers.append(row)
            layer.self_attn._replay_diagnostic = row

            def before(module, args, row=row):
                row.update({k: v.detach().clone() for k, v in zip(
                    ("native_q", "native_k", "native_v"), args[:3], strict=True)})

            def after(module, args, output, row=row):
                row["native_attention"] = output.detach().clone()

            handles.append(layer.self_attn.attn.register_forward_pre_hook(before))
            handles.append(layer.self_attn.attn.register_forward_hook(after))
        yield
    finally:
        for handle in handles:
            handle.remove()


def record_replay_attention(attention, query, key, value, history, output):
    row = getattr(attention, "_replay_diagnostic", None)
    if row is not None:
        row.update({k: v.detach() for k, v in zip(
            ("replay_q", "replay_k", "replay_v", "history_k", "history_v",
             "history_valid", "replay_attention"),
            (query, key, value, *history, output), strict=True)})


def finish_reconstruction_diagnostic(model, stats):
    rows = getattr(model, "_replay_diagnostic_layers", None)
    if rows is None:
        return
    model._replay_diagnostic_layers = None
    for layer in model.layers:
        layer.self_attn._replay_diagnostic = None
    if not stats["ok"]:
        directory = Path(os.environ["LIGHTCONE_REPLAY_DIAGNOSTIC_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"gemma-reconstruction-pid{os.getpid()}.pt"
        with path.open("xb") as stream:
            torch.save({"stats": stats, "layers": [
                {key: value.cpu() for key, value in row.items()} for row in rows
            ]}, stream)
        raise RuntimeError(f"Excluded Gemma reconstruction diagnostic saved: {path}")
