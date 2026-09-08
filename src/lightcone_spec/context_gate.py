"""Opt-in request-local update gate. No tensor operations or runtime profiling."""

from dataclasses import dataclass


@dataclass
class ContextGate:
    threshold: int | None
    max_context: int = 40960
    owner: str | None = None
    active: bool = False
    activation_context: int | None = None

    def __post_init__(self):
        if self.threshold is not None and (type(self.threshold) is not int
                                          or not 0 <= self.threshold <= self.max_context):
            raise ValueError("invalid context gate threshold")

    def observe(self, request_id: str, committed_context: int) -> bool:
        if type(committed_context) is not int or not 0 <= committed_context <= self.max_context:
            raise ValueError("context outside calibrated range")
        if self.owner is None:
            self.owner = request_id
        if self.owner != request_id:
            raise RuntimeError("context gate requires request reset before a new owner")
        if not self.active and self.threshold is not None and committed_context >= self.threshold:
            self.active, self.activation_context = True, committed_context
        return self.active

    def reset(self):
        self.owner, self.active, self.activation_context = None, False, None


def validate_gate(config, *, algorithm, method, max_in_flight, reset_scope, dp_size):
    if config is None:
        return
    if (not isinstance(config, dict) or set(config) != {"threshold", "max_context"}
            or config["max_context"] != 40960):
        raise ValueError("invalid context_gate_v1 configuration")
    if (algorithm != "DFLASH" or method != "l0" or max_in_flight != 1
            or reset_scope != "request" or dp_size != 1):
        raise ValueError("context_gate_v1 only supports DFlash LightCone c1 request reset")
    ContextGate(**config)
