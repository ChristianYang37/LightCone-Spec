#!/usr/bin/env python3
"""Aggregate actual E0 probe terminals into the trusted compatibility bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lightcone_spec.experiments.e0_stage_authority import (
    E0OnlineSpecSourceAuthority,
)
from lightcone_spec.experiments.formal_registry import protocol_lock_from_dict
from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
    E0CompatibilityProbeTerminal,
    E0PreparedModelBackendInterfaceReceipt,
    E0TaskNativeWorkloadAuthority,
    publish_e0_compatibility_probes,
    reduce_e0_compatibility_probes,
)
from lightcone_spec.experiments.formal_single_operator_stages import (
    rebuild_formal_single_operator_stage_completion,
)

_MAX_JSON_BYTES = 64 * 1024 * 1024


def _json(path: str) -> object:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"E0 compatibility input is not a regular file: {source}")
    if source.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"E0 compatibility input is too large: {source}")
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rows(path: str, label: str) -> list[dict[str, object]]:
    value = _json(path)
    if type(value) is not list or any(type(row) is not dict for row in value):
        raise TypeError(f"{label} input must be a list of exact objects")
    return value  # type: ignore[return-value]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument("--e6-completion", required=True)
    parser.add_argument("--interface-receipts", required=True)
    parser.add_argument("--workload-authorities", required=True)
    parser.add_argument("--probe-terminals", required=True)
    parser.add_argument("--onlinespec-checkout")
    parser.add_argument("--onlinespec-audit")
    parser.add_argument("--evidence-manifest-output", required=True)
    parser.add_argument("--bundle-output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if (arguments.onlinespec_checkout is None) != (
            arguments.onlinespec_audit is None
        ):
            raise ValueError("OnlineSPEC checkout and audit must be supplied together")
        lock_value = _json(arguments.protocol_lock)
        protocol_lock = protocol_lock_from_dict(lock_value)
        completion = rebuild_formal_single_operator_stage_completion(
            arguments.e6_completion
        )
        interfaces = tuple(
            E0PreparedModelBackendInterfaceReceipt(**row)
            for row in _rows(arguments.interface_receipts, "interface receipt")
        )
        workloads = tuple(
            E0TaskNativeWorkloadAuthority(**row)
            for row in _rows(arguments.workload_authorities, "workload authority")
        )
        terminals = tuple(
            E0CompatibilityProbeTerminal(**row)
            for row in _rows(arguments.probe_terminals, "probe terminal")
        )
        authority = (
            None
            if arguments.onlinespec_checkout is None
            else E0OnlineSpecSourceAuthority.bind(
                checkout_path=arguments.onlinespec_checkout,
                audit_path=arguments.onlinespec_audit,
            )
        )
        publication = reduce_e0_compatibility_probes(
            protocol_lock=protocol_lock,
            e6_completion=completion,
            interface_receipts=interfaces,
            workload_authorities=workloads,
            probe_terminals=terminals,
            onlinespec_source_authority=authority,
        )
        publish_e0_compatibility_probes(
            publication,
            bundle_output_path=arguments.bundle_output,
            evidence_manifest_output_path=arguments.evidence_manifest_output,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"E0 compatibility publisher: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "bundle_sha256": publication.bundle["bundle_sha256"],
                "compatibility_sha256": publication.compatibility.sha256,
                "evidence_manifest_sha256": publication.evidence_manifest.sha256,
                "valid_count": publication.compatibility.valid_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
