"""Version-safe asynchronous adaptation for speculative decoding."""

__version__ = "0.1.0"

SPEC_VERSION = "1.0"
SPEC_FREEZE_DATE = "2026-07-31"

# Pinned upstream revisions (spec section 2.1). These are the only
# revisions any lockfile may resolve against.
PINNED_SGLANG_COMMIT = "3312645a307453893a00778592f105581e3d1c3d"
PINNED_DEEPSPEC_COMMIT = "005e03b81cec38b7da6399833d609ee89a2587f2"
PINNED_ONLINESPEC_COMMIT = "e58f82eb3f3adca3a686211236bf4f6e9e7e3a2b"
TTS_ARXIV_ID = "2605.09329v2"
