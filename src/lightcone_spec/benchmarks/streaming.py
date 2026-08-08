"""Streaming request order (spec 12.4).

Fixed cycle math/science -> code -> chat; 16 consecutive requests per
domain; adapters round-robin inside each domain in the listed order;
request IDs inside each adapter follow the seed-0 permutation. Every
stream owns isolated adapter/optimizer/controller state initialized from
the same offline checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

DOMAIN_POOLS = {
    "math_science": (
        "gsm8k",
        "math500",
        "aime24",
        "aime25",
        "olympiadbench_math",
        "olympiadbench_physics",
        "gpqa_diamond",
        "theoremqa",
    ),
    "code": ("mbpp", "humaneval", "livecodebench"),
    "chat": ("mt_bench", "alpaca", "arena_hard_v2"),
}

DOMAIN_ORDER = ("math_science", "code", "chat")
REQUESTS_PER_DOMAIN = 16


@dataclass(frozen=True)
class StreamSlot:
    position: int
    domain: str
    adapter_key: str
    within_adapter_index: int


def stream_schedule(
    total_requests: int,
    enabled_adapters: tuple[str, ...] | None = None,
) -> list[StreamSlot]:
    """Deterministic stream order. P2 enables only its four main tasks;
    P4 enables the breadth-manifest tasks; disabled adapters are skipped
    inside their domain pool without changing the domain cycle."""
    slots: list[StreamSlot] = []
    adapter_counters: dict[str, int] = {}
    domain_rr: dict[str, int] = {d: 0 for d in DOMAIN_ORDER}
    pos = 0
    domain_idx = 0
    while pos < total_requests:
        domain = DOMAIN_ORDER[domain_idx % len(DOMAIN_ORDER)]
        pool = [
            a
            for a in DOMAIN_POOLS[domain]
            if enabled_adapters is None or a in enabled_adapters
        ]
        if pool:
            for _ in range(REQUESTS_PER_DOMAIN):
                if pos >= total_requests:
                    break
                adapter = pool[domain_rr[domain] % len(pool)]
                domain_rr[domain] += 1
                idx = adapter_counters.get(adapter, 0)
                adapter_counters[adapter] = idx + 1
                slots.append(
                    StreamSlot(
                        position=pos,
                        domain=domain,
                        adapter_key=adapter,
                        within_adapter_index=idx,
                    )
                )
                pos += 1
        domain_idx += 1
        if all(
            not [
                a
                for a in DOMAIN_POOLS[d]
                if enabled_adapters is None or a in enabled_adapters
            ]
            for d in DOMAIN_ORDER
        ):
            raise ValueError("no enabled adapters in any domain")
    return slots
