#!/usr/bin/env python3
"""Path-only CLI for the split-host formal production finalizer."""

from lightcone_spec.orchestration import formal_experiment_cross_host_finalizer

if __name__ == "__main__":
    raise SystemExit(formal_experiment_cross_host_finalizer.main())
