"""Confirmatory claim gates (spec 14.7, 14.8, 16.3, 1.4).

These functions only *report*; they never delete code paths or hide
negative results. A negative verdict changes the paper narrative, not
the code deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lightcone_spec.statistics.bootstrap import BootstrapResult


@dataclass
class ClaimGateVerdict:
    gates: dict[str, dict] = field(default_factory=dict)

    def add(self, name: str, passed: bool | None, detail: dict) -> None:
        self.gates[name] = {"pass": passed, **detail}

    def to_dict(self) -> dict:
        return {"claim_gates": self.gates}

    def human_readable(self) -> str:
        lines = ["Claim-gate verdict:"]
        for name, g in self.gates.items():
            status = (
                "PASS" if g["pass"] else ("FAIL" if g["pass"] is not None else "N/A")
            )
            detail = {k: v for k, v in g.items() if k != "pass"}
            lines.append(f"  [{status}] {name}: {detail}")
        return "\n".join(lines)


def h1_gate(delay_mae: float, path_mae: float, ci95: tuple[float, float]) -> dict:
    """(MAE_delay - MAE_path) / MAE_delay >= 0.15 with a bootstrap CI
    (spec 14.7)."""
    improvement = (delay_mae - path_mae) / delay_mae if delay_mae > 0 else 0.0
    return {
        "pass": bool(improvement >= 0.15 and ci95[0] > 0.0),
        "delay_mae": delay_mae,
        "path_mae": path_mae,
        "relative_improvement": improvement,
        "ci95": list(ci95),
        "threshold": 0.15,
    }


def harmful_rate_gate(harmful_rate: float) -> dict:
    """If the real harmful stale-update rate is below 1%, the mandated
    conclusion is 'real staleness is essentially harmless' and no
    controller-necessity claim may be made (spec 1.4)."""
    return {
        "pass": None,  # informational; drives narrative not delivery
        "harmful_rate": harmful_rate,
        "below_1pct": bool(harmful_rate < 0.01),
        "mandated_conclusion": (
            "staleness essentially harmless; no controller-necessity claim"
            if harmful_rate < 0.01
            else "harmful updates exist at a reportable rate"
        ),
    }


def h3_gate(
    tps_improvement: BootstrapResult,
    harmful_rate_drop_rel: float,
    controller_overhead_frac: float,
    quality_diff: BootstrapResult,
    d0_throughput_diff_frac: float,
    noninferiority_margin: float = 0.0,
) -> dict:
    """H3 (spec 14.8): in the preregistered stress region,
    - L1 or L2 improves exact decode TPS over L0 by >= 5%, paired CI
      excluding 0;
    - harmful rate drops >= 30% relatively;
    - controller overhead < 1% of decode wall time;
    - quality difference CI contains 0 or stays within the preset
      non-inferiority margin;
    - d = 0 throughput difference <= 2%.
    """
    tps_ok = tps_improvement.estimate >= 0.05 and tps_improvement.ci_low > 0.0
    harm_ok = harmful_rate_drop_rel >= 0.30
    overhead_ok = controller_overhead_frac < 0.01
    quality_ok = (
        (quality_diff.ci_low <= 0.0 <= quality_diff.ci_high)
        or (quality_diff.ci_low >= -abs(noninferiority_margin))
    )
    parity_ok = abs(d0_throughput_diff_frac) <= 0.02
    return {
        "pass": bool(tps_ok and harm_ok and overhead_ok and quality_ok and parity_ok),
        "tps_improvement": tps_improvement.to_dict(),
        "tps_ok": bool(tps_ok),
        "harmful_rate_drop_rel": harmful_rate_drop_rel,
        "harm_ok": bool(harm_ok),
        "controller_overhead_frac": controller_overhead_frac,
        "overhead_ok": bool(overhead_ok),
        "quality_diff": quality_diff.to_dict(),
        "quality_ok": bool(quality_ok),
        "d0_throughput_diff_frac": d0_throughput_diff_frac,
        "parity_ok": bool(parity_ok),
    }


def l3_gate(
    l3_utility: float, l2_utility: float, discard_utility: float
) -> dict:
    """L3 must always be reported; when it does not beat L2 or discard the
    verdict is an explicit negative (spec 1.4, 16.3)."""
    better = l3_utility > max(l2_utility, discard_utility)
    return {
        "pass": bool(better),
        "l3_utility": l3_utility,
        "l2_utility": l2_utility,
        "discard_utility": discard_utility,
        "verdict": (
            "transport beats L2 and discard"
            if better
            else "NEGATIVE: transport not better than L2/discard (cost not justified)"
        ),
    }


def exactness_gate(canary_total: int, deliberate_canary_caught: bool) -> dict:
    """Any exactness canary failure stops all performance and quality
    conclusions; a harness that misses the deliberate canary fails."""
    return {
        "pass": bool(canary_total == 0 and deliberate_canary_caught),
        "canary_total": canary_total,
        "deliberate_canary_caught": deliberate_canary_caught,
    }
