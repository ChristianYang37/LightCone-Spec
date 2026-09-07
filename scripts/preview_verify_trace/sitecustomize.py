"""Loaded only through the excluded diagnostic's explicit PYTHONPATH entry."""

import os

if os.environ.get("LIGHTCONE_EXCLUDED_VERIFY_TRACE"):
    from lightcone_spec.verification_diagnostic import install

    install()
