"""Excluded bounded stream windows. Never a completed-request benchmark."""

import math
import statistics
import threading
import time

from .client import StreamAborted, _native_events


class WindowInterrupted(RuntimeError):
    pass


def bounded_call(function, seconds):
    """A timed-out server must be discarded; never launch another window on it."""
    values, errors = [], []

    def invoke():
        try:
            values.append(function())
        except BaseException as error:
            errors.append(error)

    task = threading.Thread(target=invoke, daemon=True)
    task.start()
    task.join(max(0, seconds))
    if task.is_alive():
        raise TimeoutError("diagnostic cleanup deadline exceeded; discard server")
    if errors:
        raise errors[0]
    return values[0]


class WindowEvidence:
    def __init__(self, seconds, clock=time.perf_counter):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("positive finite window required")
        self.seconds, self.clock = seconds, clock
        self.started = clock()
        self.first = None
        self.events = []
        self.requests = {}

    @property
    def deadline(self):
        return None if self.first is None else self.first + self.seconds

    def observe(self, event):
        now = self.clock()
        rid = event["request_id"]
        row = self.requests.setdefault(rid, {"sequence": 0, "ids": [], "counted": 0, "native": None})
        if event["sequence"] != row["sequence"] + 1:
            raise ValueError("duplicate or missing stream event")
        chunk, new = event["chunk"], event["token_ids"]
        ids = chunk.get("output_ids")
        meta = chunk.get("meta_info", {})
        if meta.get("id") != rid or ids != row["ids"] + new or meta.get("completion_tokens") != len(ids):
            raise ValueError("stream token identity/count mismatch")
        row["sequence"], row["ids"] = event["sequence"], list(ids)
        if new and self.first is None:
            self.first = now
        inside = self.deadline is not None and now <= self.deadline
        if inside:
            row["counted"] += len(new)
        current_native = None
        if "native_token_timestamp_events" in meta:
            current_native = _native_events(meta, ids)
            row["native"] = current_native
        # Preserve deltas, not a quadratic copy of every cumulative SSE payload.
        self.events.append({"request_id": rid, "sequence": event["sequence"],
                            "token_ids": list(new), "completion_tokens": len(ids),
                            "native_timestamps_ns": list(current_native[-len(new):]) if new and current_native is not None else None,
                            "observed_seconds": now - self.started, "inside_window": inside})

    def report(self):
        tokens = sum(r["counted"] for r in self.requests.values())
        speeds = []
        for row in self.requests.values():
            count, stamps = row["counted"], row["native"]
            if count < 2:
                continue
            if stamps is None or len(stamps) < count or stamps[count - 1] <= stamps[0]:
                speeds = None
                break
            speeds.append((count - 1) * 1e9 / (stamps[count - 1] - stamps[0]))
        elapsed = None if self.deadline is None else self.deadline - self.started
        return {"measurement_scope": "excluded_stream_observed_window_v1", "formal_acceptance": False,
                "window_seconds": self.seconds, "observed_committed_tokens": tokens,
                "prefill_seconds": None if self.first is None else self.first - self.started,
                "window_goodput": tokens / elapsed if elapsed else None,
                "decode_window_speed": tokens / self.seconds if self.first is not None else None,
                "per_user_generation_speed": statistics.mean(speeds) if speeds else None,
                "per_user_scope": "observed request prefixes; not full-request benchmark",
                "request_count": len(self.requests), "events": self.events}


def run_window(client, prompts, *, seconds, seed, prefix, reset_and_verify,
               max_new_tokens=32768, temperature=0.0, first_token_timeout=60.0,
               cleanup_seconds=10.0, evidence=None, stop=None):
    """C1 fixed-order EOS-respecting requests; cancellation is separate from failure.

    reset_and_verify receives the absolute cleanup deadline and must preserve
    pre-reset safety evidence, flush, and verify idle/reset before returning.
    A caller retains ``evidence`` even if this function raises.
    """
    prompts = tuple(prompts)
    if not prompts or client.stream_observer is not None:
        raise ValueError("exclusive observer and nonempty frozen pool required")
    evidence = evidence or WindowEvidence(seconds)
    lock, done = threading.Lock(), threading.Event()
    halted = threading.Event()
    active, errors, cancelled = [], [], set()
    outcomes = []

    def worker():
        index = 0
        try:
            while not halted.is_set():
                rid = f"{prefix}-{index:06d}"
                with lock:
                    if halted.is_set():
                        break
                    active[:] = [rid]
                try:
                    client.run_batch((prompts[index % len(prompts)],), max_new_tokens=max_new_tokens,
                        seed=seed + index, temperature=temperature, ignore_eos=False,
                        request_ids=(rid,), timeout_seconds=first_token_timeout + seconds + cleanup_seconds)
                    outcomes.append({"request_id": rid, "status": "completed"})
                except StreamAborted as error:
                    ordinary = error.reason.get("message") in (None, "Aborted", "Aborted by AbortReq.")
                    if rid not in cancelled or error.request_id != rid or not ordinary or error.reason.get("status_code") not in (None, 400):
                        raise
                    outcomes.append({"request_id": rid, "status": "budget_end"})
                finally:
                    with lock:
                        active.clear()
                index += 1
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    client.stream_observer = evidence.observe
    task = threading.Thread(target=worker, daemon=True)
    task.start()
    cause = "budget_end"
    try:
        while not done.wait(.005):
            now = time.perf_counter()
            if stop is not None and stop.is_set():
                cause = "interrupted"
                break
            if evidence.deadline is not None and now >= evidence.deadline:
                break
            if evidence.first is None and now - evidence.started >= first_token_timeout:
                cause = "first_token_timeout"
                break
        halted.set()
        deadline = time.perf_counter() + cleanup_seconds
        with lock:
            targets = tuple(active)
            cancelled.update(targets)
        for rid in targets:
            bounded_call(lambda: client.abort(rid, timeout_seconds=max(.001, deadline-time.perf_counter())),
                         deadline-time.perf_counter())
        if not done.wait(max(0, deadline-time.perf_counter())):
            raise TimeoutError("stream still running after budget; discard server")
        if errors:
            raise errors[0]
        if cause == "interrupted":
            raise WindowInterrupted("quick window interrupted; not a performance sample")
        if cause != "budget_end" or evidence.first is None:
            raise RuntimeError(f"quick window {cause}; not a performance sample")
        cleanup = bounded_call(lambda: reset_and_verify(deadline), deadline-time.perf_counter())
        return {**evidence.report(), "status": cause, "outcomes": outcomes,
                "cleanup_seconds": time.perf_counter() - (deadline-cleanup_seconds), "cleanup": cleanup}
    finally:
        halted.set()
        client.stream_observer = None


def paired_decision(rows):
    """Only paired domains/repeats; never count requests as repetitions."""
    groups = {}
    for row in rows:
        key = row["domain"], row["repeat"], row["variant"]
        if key in groups or row["domain"] not in {"Code", "Math"} or row["variant"] not in {"old", "new"}:
            raise ValueError("invalid/duplicate pair")
        if row.get("status") != "budget_end" or not math.isfinite(row["window_goodput"]) or row["window_goodput"] <= 0:
            raise ValueError("only validated finite windows may be compared")
        groups[key] = row
    repeats = sorted({r["repeat"] for r in rows})
    if repeats not in ([0], [0, 1], [0, 1, 2]):
        raise ValueError("one to three complete ordered repetitions required")
    gains = {}
    for domain in ("Code", "Math"):
        ratios = []
        for repeat in repeats:
            a, b = groups.get((domain, repeat, "new")), groups.get((domain, repeat, "old"))
            if a is None or b is None:
                return {"decision": "await_pairs"}
            if a["comparison_key"] != b["comparison_key"]:
                raise ValueError("mismatched workload/topology/window pair")
            ratios.append(math.log(a["window_goodput"] / b["window_goodput"]))
        gains[domain] = math.exp(statistics.mean(ratios)) - 1
    if all(g < -.03 for g in gains.values()):
        decision = "reject"
    elif len(repeats) == 1 and all(g > .03 for g in gains.values()):
        decision = "repeat_promising"
    elif len(repeats) < 3:
        decision = "repeat"
    else:
        gain = math.sqrt(math.prod(1 + g for g in gains.values())) - 1
        decision = "candidate_for_long_validation" if min(gains.values()) > 0 and gain >= .01 else "keep_old"
    return {"decision": decision, "domain_gains": gains, "repeats": len(repeats), "formal_acceptance": False}
def exact_hotpath_inputs(tokenizer, records, background, length):
    """Non-repeated prompt-only context with the task inside its native template."""
    filler = tokenizer.encode("\n\n".join(r["prompt"] for r in background), add_special_tokens=False)
    output = []
    for row in records:
        marker = "LIGHTCONE_HOTPATH_BACKGROUND_51"
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": marker + "\n\nTask:\n" + row["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if rendered.count(marker) != 1:
            raise ValueError("hotpath template must preserve background marker exactly once")
        before, after = rendered.split(marker)
        prefix = tokenizer.encode(before, add_special_tokens=False)
        suffix = tokenizer.encode(after, add_special_tokens=False)
        needed = length - len(prefix) - len(suffix)
        if not 0 <= needed <= len(filler):
            raise ValueError("insufficient independent background for hotpath input")
        output.append(tuple(prefix + filler[:needed] + suffix))
    return tuple(output)
