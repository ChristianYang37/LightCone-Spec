"""Coverage matrix generation (spec 16.1).

Dimensions: phase, model, method, task, seed, lifecycle, sampling,
stride/delay/concurrency, trainable scope, transport variant. Each cell
is exactly one of: complete_valid, failed_exactness, failed_runtime,
resource_skip, missing, invalid_artifact. A required unit that is
missing or invalid fails final validation (exit 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lightcone_spec.artifacts.schemas import UNIT_STATUSES
from lightcone_spec.locking.hashing import canonical_json, sha256_bytes

COVERAGE_DIMENSIONS = (
    "phase",
    "model_pair",
    "method",
    "dataset",
    "seed",
    "lifecycle",
    "sampling_profile",
    "stride",
    "logical_delay",
    "concurrency",
    "trainable_scope",
    "transport_variant",
)


@dataclass
class CoverageMatrix:
    cells: dict[str, dict] = field(default_factory=dict)

    def add_expected(self, unit: dict) -> None:
        uid = unit["unit_id"]
        self.cells[uid] = {
            "unit_id": uid,
            "required": unit.get("required", True),
            "allow_resource_skip": unit.get("allow_resource_skip", False),
            "status": "missing",
            **{dim: unit.get(dim) for dim in COVERAGE_DIMENSIONS},
        }

    def record_status(self, unit_id: str, status: str) -> None:
        if status not in UNIT_STATUSES:
            raise ValueError(f"invalid unit status {status!r}")
        if unit_id not in self.cells:
            self.cells[unit_id] = {
                "unit_id": unit_id,
                "required": False,
                "allow_resource_skip": False,
                "status": status,
                **{dim: None for dim in COVERAGE_DIMENSIONS},
            }
        else:
            self.cells[unit_id]["status"] = status

    def missing_required(self) -> list[str]:
        bad = []
        for uid, cell in self.cells.items():
            if not cell["required"]:
                continue
            status = cell["status"]
            if status in (
                "missing",
                "invalid_artifact",
                "failed_exactness",
                "failed_runtime",
            ):
                bad.append(uid)
            elif status == "resource_skip" and not cell["allow_resource_skip"]:
                bad.append(uid)
        return sorted(bad)

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for cell in self.cells.values():
            counts[cell["status"]] = counts.get(cell["status"], 0) + 1
        return {
            "total_units": len(self.cells),
            "by_status": counts,
            "missing_required": self.missing_required(),
            "final_validation_ok": not self.missing_required(),
        }

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = canonical_json({"cells": self.cells, "summary": self.summary()})
        path.write_text(body)
        Path(str(path) + ".sha256").write_text(
            sha256_bytes(body.encode("utf-8")) + "\n"
        )


def build_coverage(
    expected_units: list[dict], unit_status: dict[str, str]
) -> CoverageMatrix:
    cov = CoverageMatrix()
    for unit in expected_units:
        cov.add_expected(unit)
    for uid, status in unit_status.items():
        cov.record_status(uid, status)
    return cov
