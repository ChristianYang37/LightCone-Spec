#!/usr/bin/env python3
"""Verify a maintainer-provided GPU report. This command does not run a GPU."""

import argparse
import json
from pathlib import Path

from lightcone_spec.preview_benchmark import validate_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--environment", type=Path, required=True)
    args = parser.parse_args()
    result = validate_report(json.loads(args.report.read_text()), json.loads(args.manifest.read_text()),
                             candidate_commit=args.candidate_commit,
                             environment=json.loads(args.environment.read_text()))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
