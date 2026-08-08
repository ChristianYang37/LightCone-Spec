"""Immutable experiment manifests with unit-level resume (spec 9.2,
13.6).

A manifest is content-addressed (SHA-256 of the canonical JSON) and
lists run units. Execution skips units whose completed run dirs already
carry a matching unit hash, resumes partials from the last complete unit
boundary, and never overwrites immutable artifacts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from lightcone_spec.exit_codes import ConfigError
from lightcone_spec.locking.hashing import canonical_json, sha256_bytes
from lightcone_spec.orchestration.units import RunUnit

MANIFEST_SCHEMA_VERSION = 1


def _reject_duplicate_units(units: list[RunUnit], *, context: str) -> None:
    """Reject identities that collapse after effective canonicalization."""
    by_id: dict[str, list[RunUnit]] = {}
    for unit in units:
        by_id.setdefault(unit.unit_id, []).append(unit)
    duplicates = {
        unit_id: group for unit_id, group in by_id.items() if len(group) > 1
    }
    if duplicates:
        unit_id, group = next(iter(duplicates.items()))
        raise ConfigError(
            f"{context} contains duplicate identity {unit_id} after effective "
            "canonicalization: "
            f"{[u.key_dict() for u in group]}"
        )


@dataclass
class ExperimentManifest:
    name: str
    phase: str
    description: str
    units: list[RunUnit]
    engine_params: dict = field(default_factory=dict)
    lockfile_sha256: str | None = None
    profile: str = "local_1x80gb"

    def to_dict(self) -> dict:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "name": self.name,
            "phase": self.phase,
            "description": self.description,
            "profile": self.profile,
            "lockfile_sha256": self.lockfile_sha256,
            "engine_params": self.engine_params,
            "units": [u.to_manifest_dict() for u in self.units],
        }

    def content_sha256(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict()).encode("utf-8"))

    def write(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = canonical_json(self.to_dict())
        digest = sha256_bytes(body.encode("utf-8"))
        if path.is_file():
            existing = path.read_text()
            if sha256_bytes(existing.encode("utf-8")) != digest:
                raise ConfigError(
                    f"manifest {path} exists with different content; manifests "
                    "are immutable (write a new file instead)"
                )
            return digest
        path.write_text(body)
        Path(str(path) + ".sha256").write_text(digest + "\n")
        return digest

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentManifest":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"manifest not found: {path}")
        d = json.loads(path.read_text())
        if d.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ConfigError(
                f"manifest schema_version {d.get('schema_version')} != "
                f"{MANIFEST_SCHEMA_VERSION}"
            )
        sha_path = Path(str(path) + ".sha256")
        if sha_path.is_file():
            expected = sha_path.read_text().strip()
            # Historical immutable manifests sometimes included a final
            # newline and hashed the exact file bytes.  Accept that provenance
            # convention as well as the canonical-JSON digest; never ignore a
            # mismatch to both representations.
            actual_canonical = sha256_bytes(canonical_json(d).encode("utf-8"))
            actual_source = sha256_bytes(path.read_bytes())
            if expected not in {actual_canonical, actual_source}:
                raise ConfigError(f"manifest hash drift: {path}")
        try:
            units = [RunUnit.from_dict(u) for u in d["units"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"invalid manifest unit in {path}: {exc}") from exc
        _reject_duplicate_units(units, context=f"manifest {path}")
        return cls(
            name=d["name"],
            phase=d["phase"],
            description=d["description"],
            units=units,
            engine_params=d.get("engine_params", {}),
            lockfile_sha256=d.get("lockfile_sha256"),
            profile=d.get("profile", "local_1x80gb"),
        )

    def expected_units(self) -> list[dict]:
        # Engine-parameter overlays (for example learning rate) intentionally
        # do not alter a RunUnit ID.  Bind every expected-unit declaration to
        # the effective outer manifest so two execution overlays cannot
        # silently present an identical, unqualified coverage contract.
        manifest_sha256 = self.content_sha256()
        return [
            {
                **unit.to_manifest_dict(),
                "expected_manifest_sha256": manifest_sha256,
            }
            for unit in self.units
        ]

    def with_methods(self, methods: list[str] | tuple[str, ...] | None) -> "ExperimentManifest":
        """Return a deterministic method subset for staged experiment runs.

        Filtering is an execution overlay, just like ``weight_update_mode``:
        it never edits the source manifest, and the effective manifest hash is
        recomputed from the retained units.  Reject misspellings and an empty
        result before creating artifacts or loading a model.
        """
        if methods is None:
            return self
        requested = tuple(dict.fromkeys(str(method) for method in methods))
        if not requested:
            raise ConfigError("--methods requires at least one method")
        available = {unit.method for unit in self.units}
        unknown = sorted(set(requested) - available)
        if unknown:
            raise ConfigError(
                f"--methods contains values absent from this manifest: {unknown}"
            )
        units = [unit for unit in self.units if unit.method in requested]
        if not units:
            raise ConfigError("--methods removed every manifest unit")
        _reject_duplicate_units(units, context="--methods result")
        return replace(self, units=units)

    def with_lifecycles(
        self, lifecycles: list[str] | tuple[str, ...] | None
    ) -> "ExperimentManifest":
        """Return a deterministic lifecycle subset execution overlay."""
        if lifecycles is None:
            return self
        requested = tuple(
            dict.fromkeys(str(lifecycle) for lifecycle in lifecycles)
        )
        if not requested:
            raise ConfigError("--lifecycles requires at least one lifecycle")
        available = {unit.lifecycle for unit in self.units}
        unknown = sorted(set(requested) - available)
        if unknown:
            raise ConfigError(
                "--lifecycles contains values absent from this manifest: "
                f"{unknown}"
            )
        units = [unit for unit in self.units if unit.lifecycle in requested]
        if not units:
            raise ConfigError("--lifecycles removed every manifest unit")
        _reject_duplicate_units(units, context="--lifecycles result")
        return replace(self, units=units)

    def with_learning_rate(self, learning_rate: float | None) -> "ExperimentManifest":
        """Apply a positive finite learning-rate engine overlay in memory."""
        if learning_rate is None:
            return self
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, (int, float)
        ):
            raise ConfigError("--learning-rate must be a positive finite number")
        resolved = float(learning_rate)
        if not math.isfinite(resolved) or resolved <= 0.0:
            raise ConfigError("--learning-rate must be a positive finite number")
        engine_params = dict(self.engine_params)
        engine_params["lr"] = resolved
        return replace(self, engine_params=engine_params)

    def with_weight_update_mode(self, mode: str | None) -> "ExperimentManifest":
        """Apply a CLI-only mode overlay without mutating the source manifest.

        Static baselines are intentionally untouched.  Unit IDs and the outer
        manifest hash are properties, so replacing the frozen units recomputes
        both.  Converged identities are rejected before any artifact or model
        allocation occurs.
        """
        if mode is None:
            return self
        from lightcone_spec.config.schema import (
            canonical_tail_layout_mode,
            canonical_weight_update_mode,
        )

        canonical = canonical_weight_update_mode(mode)
        units = [
            replace(
                unit,
                trainable_scope=canonical_tail_layout_mode(
                    unit.trainable_scope
                ),
            )
            if unit.method == "static"
            else replace(unit, trainable_scope=canonical_tail_layout_mode(canonical))
            for unit in self.units
        ]
        _reject_duplicate_units(units, context="--weight-update-mode result")
        return replace(self, units=units)
