"""Excluded import-time instrumentation; disabled unless explicitly installed.

No sys.settrace, no change to model outputs or publication decisions. Existing
runtime timing regions are nested GPU intervals, not additive wall-time pieces.
"""

import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .timing_audit import TimingRecorder

_recorder = None
_settings = None
_window = None
_installed = set()
_profiler = None


def checkpoint(action, job_id):
    settings = os.environ.get("LIGHTCONE_TIMING_AUDIT")
    if not settings:
        return
    root = Path(json.loads(settings)["output_directory"])
    if action not in {"begin", "end"} or not root.is_dir():
        raise ValueError("explicit timing window and existing audit directory required")
    temp = root / "control.tmp"
    temp.write_text(json.dumps({"action": action, "job_id": job_id}))
    temp.replace(root / "control.json")


@contextmanager
def _span(name, *, gpu=True, lane="main"):
    if _recorder is None:
        yield
        return
    nvtx = getattr(_recorder.cuda, "nvtx", None)
    if nvtx is not None:
        nvtx.range_push(f"lightcone-audit/{lane}/{name}")
    try:
        with _recorder.span(name, gpu=gpu, lane=lane):
            yield
    finally:
        if nvtx is not None:
            nvtx.range_pop()


def _wrap(cls, method, name, *, gpu=True, lane="main"):
    if not hasattr(cls, method):
        raise RuntimeError(f"timing hook target missing: {cls.__name__}.{method}")
    original = getattr(cls, method)

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        label = name(self, args, kwargs) if callable(name) else name
        with _span(label, gpu=gpu, lane=lane):
            return original(self, *args, **kwargs)

    setattr(cls, method, wrapped)
    _installed.add(f"{cls.__name__}.{method}")


def _wrap_function(module, function, label, lane):
    original = getattr(module, function)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with _span(label, lane=lane):
            return original(*args, **kwargs)

    setattr(module, function, wrapped)
    _installed.add(f"{module.__name__}.{function}")


def _wrap_autograd(cls, method, label):
    """Keep Function forward/backward static; timing is excluded and opt-in."""
    original = getattr(cls, method)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with _span(label, lane="side"):
            return original(*args, **kwargs)

    setattr(cls, method, staticmethod(wrapped))
    _installed.add(f"{cls.__name__}.{method}")


def _model_phase(self, args, kwargs):
    if self.is_draft_worker:
        return "draft_forward"
    batch = kwargs.get("forward_batch", args[0] if args else None)
    mode = getattr(batch, "forward_mode", None)
    # TARGET_VERIFY is also is_extend() upstream; distinguish it first.
    if mode is not None and mode.is_target_verify():
        return "target_verification"
    if mode is not None and mode.is_extend():
        return "target_prefill"
    return "target_decode"


def _instrument(module):
    name = module.__name__
    if name.endswith(".dflash_online_adaptation") and os.environ.get("LIGHTCONE_HOTPATH_EQUIVALENCE_SOURCE"):
        from .hotpath_equivalence import install
        install(module, os.environ["LIGHTCONE_HOTPATH_EQUIVALENCE_SOURCE"],
                os.environ["LIGHTCONE_HOTPATH_EQUIVALENCE_OUTPUT"])
    if _settings.get("mode", "full") == "off" and not name.endswith(".scheduler"):
        return
    if name.endswith(".model_runner"):
        _wrap(module.ModelRunner, "forward", _model_phase)
        _wrap(module.ModelRunner, "sample", "target_sampling")
    elif name.endswith(".online_adaptation_runtime"):
        for collective in ("all_reduce", "all_gather", "all_gather_into_tensor", "broadcast", "barrier"):
            _wrap_function(module.dist, collective, f"tp_{collective}", "collective")
        cls = module.OnlineCohortRuntime
        original = cls.timing

        @contextmanager
        def timing(self, region):
            lane = "side" if region in {"training", "optimizer", "merge"} else "main"
            with _span(region, lane=lane), original(self, region):
                yield

        cls.timing = timing
        _installed.add("OnlineCohortRuntime.timing")
        for method, label, gpu, lane in (
            ("boundary", "candidate_check_and_publish", True, "inclusive"),
            ("record_device_commit", "commit_record", True, "main"),
            ("reset", "explicit_reset", True, "main"),
            ("_reset_request_scope", "request_reset", True, "main"),
            ("record_confidence", "confidence_telemetry", True, "main"),
            ("submit", "candidate_submit", True, "side"),
        ):
            _wrap(cls, method, label, gpu=gpu, lane=lane)
    elif name.endswith(".dflash_online_adaptation"):
        _wrap_function(module, "_logit_reconstruction_gate", "reconstruction_check", "side")
        for operator, label in (("Attention", "attention"), ("RMS", "rms"), ("RoPE", "rope")):
            for method in ("forward", "backward"):
                _wrap_autograd(getattr(module, f"_DFlashInference{operator}"), method,
                               f"training_{label}_{method}")
        cls = module.DFlashDrafterAdapter
        for method, label, lane in (
            ("begin_round", "context_gate_and_round_setup", "main"),
            ("maybe_launch", "update_schedule_and_prepare", "inclusive"),
            ("_gather_history", "history_kv_gather", "side"),
            ("_surrogate_hidden", "training_replay", "side"),
            ("_distillation_loss", "distillation_loss", "side"),
            ("_full_vocab_logits", "vocabulary_gather", "side"),
        ):
            _wrap(cls, method, label, lane=lane)
    elif name.endswith(".scheduler"):
        cls = module.Scheduler
        original = cls.get_internal_state

        @functools.wraps(original)
        def get_state(self, *args, **kwargs):
            global _recorder, _window, _profiler
            root = Path(_settings["output_directory"])
            control = root / "control.json"
            command = json.loads(control.read_text()) if control.exists() else None
            if command and command["action"] == "end" and _recorder is not None:
                if _profiler is not None:
                    _profiler.__exit__(None, None, None)
                    _profiler.export_chrome_trace(str(root / f"rank-{self.ps.tp_rank}-torch-trace.json"))
                    _profiler = None
                payload = _recorder.snapshot()
                payload.update(job_id=_window, hooks=sorted(_installed))
                (root / f"rank-{self.ps.tp_rank}-timing.json").write_text(json.dumps(payload))
                _recorder = None
            result = original(self, *args, **kwargs)
            if result is not None:
                state = result.internal_state
                record = {"tp_rank": int(self.ps.tp_rank), "tp_size": int(self.ps.tp_size),
                          "captured_ns": time.time_ns(), "state": {k: state[k] for k in
                          ("speed_study_metrics", "speculative_adaptation_info_record") if k in state}}
                with (root / f"rank-{self.ps.tp_rank}-metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
            if command and command["action"] == "begin" and _window != command["job_id"]:
                import torch
                _window = command["job_id"]
                _recorder = TimingRecorder(cuda=torch.cuda if _settings.get("mode", "full") != "off" else None, rank=int(self.ps.tp_rank),
                    max_pending=int(_settings.get("max_pending", 4096)))
                if _settings.get("deep_trace"):
                    _profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=False)
                    _profiler.__enter__()
            return result

        cls.get_internal_state = get_state
        _installed.add("Scheduler.get_internal_state")


TARGETS = {
    "sglang.srt.model_executor.model_runner",
    "sglang.srt.speculative.online_adaptation_runtime",
    "sglang.srt.speculative.dflash_online_adaptation",
    "sglang.srt.managers.scheduler",
}


def install():
    global _settings
    if _settings is not None:
        raise RuntimeError("timing hooks already installed")
    _settings = json.loads(os.environ["LIGHTCONE_TIMING_AUDIT"])
    if not Path(_settings["output_directory"]).is_absolute():
        raise ValueError("audit directory must be absolute")

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in TARGETS:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"missing audited module {fullname}")
            original = spec.loader

            class Loader(importlib.abc.Loader):
                def create_module(self, module_spec):
                    return original.create_module(module_spec)

                def exec_module(self, module):
                    original.exec_module(module)
                    _instrument(module)

            spec.loader = Loader()
            return spec

    if any(name in sys.modules for name in TARGETS):
        raise RuntimeError("install timing hooks before importing SGLang")
    sys.meta_path.insert(0, Finder())
