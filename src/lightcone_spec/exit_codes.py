"""Stable process exit codes used by resumable experiment tooling."""

SUCCESS = 0
SCIENTIFIC_BLOCK = 42
LOCK_BUSY = 75
CONFIG_ERROR = 78


class LightConeError(RuntimeError):
    """Base class for actionable, fail-closed errors."""


class ConfigError(LightConeError):
    """Raised before model allocation when configuration is invalid."""


class ExactnessError(LightConeError):
    """Raised when proposal reconstruction cannot be certified."""
