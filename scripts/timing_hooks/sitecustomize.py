"""Explicitly opted-in excluded GPU timing; no hook in ordinary launches."""
import os

if os.environ.get("LIGHTCONE_TIMING_AUDIT"):
    from lightcone_spec.timing_diagnostic import install

    install()
elif os.environ.get("LIGHTCONE_RANK_RECEIPTS"):
    from lightcone_spec.timing_diagnostic import install

    install({"mode": "off", "output_directory": os.environ["LIGHTCONE_RANK_RECEIPTS"]})
