#!/usr/bin/env python3
"""Fail closed on missing or prohibited installed-package license metadata."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def _license_identity(distribution: importlib.metadata.Distribution) -> tuple[str, ...]:
    metadata = distribution.metadata
    values: list[str] = []
    expression = metadata.get("License-Expression")
    legacy = metadata.get("License")
    if expression and expression.strip():
        values.append(expression.strip())
    values.extend(
        classifier.removeprefix("License :: ").strip()
        for classifier in metadata.get_all("Classifier", [])
        if classifier.startswith("License :: ")
    )
    if not values and legacy and legacy.strip().lower() not in {"", "none", "unknown"}:
        values.append(legacy.strip())
    return tuple(dict.fromkeys(values))


def _prohibited_value(value: str) -> bool:
    joined = value.upper()
    return (
        "AGPL" in joined
        or "GNU AFFERO" in joined
        or ("GPL-" in joined and "LGPL-" not in joined)
        or (
            "GNU GENERAL PUBLIC LICENSE" in joined
            and "LESSER GENERAL PUBLIC LICENSE" not in joined
        )
    )


def _prohibited(values: tuple[str, ...]) -> bool:
    # Multiple classifiers/expressions can name an alternative permissive
    # grant.  Reject only when every declared option is strong copyleft.
    return all(_prohibited_value(value) for value in values)


def inventory() -> dict[str, object]:
    by_identity: dict[tuple[str, str], tuple[str, ...]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            raise SystemExit(
                "license check failed: installed distribution lacks a name"
            )
        licenses = _license_identity(distribution)
        if not licenses:
            raise SystemExit(f"license check failed: {name} has no license metadata")
        if _prohibited(licenses):
            raise SystemExit(
                f"license check failed: {name} has a prohibited strong-copyleft license"
            )
        identity = (name.lower().replace("_", "-"), distribution.version)
        previous = by_identity.setdefault(identity, licenses)
        if previous != licenses:
            raise SystemExit(
                "license check failed: duplicate distribution identity has "
                f"conflicting metadata: {identity[0]}"
            )
    rows = [
        {"name": name, "version": version, "licenses": licenses}
        for (name, version), licenses in by_identity.items()
    ]
    rows.sort(key=lambda row: (str(row["name"]), str(row["version"])))
    payload: dict[str, object] = {"schema_version": 1, "packages": rows}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = inventory()
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(
        "installed-license checks passed for "
        f"{len(value['packages'])} distributions; {value['content_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
