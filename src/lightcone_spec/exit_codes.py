"""Stable process exit codes (spec section 9.3)."""

SUCCESS = 0
CONFIG_SCHEMA_ERROR = 2
LOCK_HASH_REVISION_ERROR = 3
RESOURCE_SKIP = 4
EXACTNESS_VERSION_FAILURE = 5
NUMERICAL_FAILURE = 6
RUNTIME_GPU_FAILURE = 7
ARTIFACT_VALIDATION_FAILURE = 8
INCOMPLETE_COVERAGE = 9

_MEANINGS = {
    SUCCESS: "success",
    CONFIG_SCHEMA_ERROR: "config/schema error",
    LOCK_HASH_REVISION_ERROR: "lock/hash/revision error",
    RESOURCE_SKIP: "resource skip",
    EXACTNESS_VERSION_FAILURE: "exactness/version failure",
    NUMERICAL_FAILURE: "numerical failure",
    RUNTIME_GPU_FAILURE: "runtime/GPU failure",
    ARTIFACT_VALIDATION_FAILURE: "artifact validation failure",
    INCOMPLETE_COVERAGE: "incomplete coverage",
}


def meaning(code: int) -> str:
    return _MEANINGS.get(code, "unknown exit code")


class LightconeError(Exception):
    """Base error carrying a stable exit code."""

    exit_code = RUNTIME_GPU_FAILURE

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ConfigError(LightconeError):
    exit_code = CONFIG_SCHEMA_ERROR


class LockError(LightconeError):
    exit_code = LOCK_HASH_REVISION_ERROR


class ResourceSkip(LightconeError):
    """Insufficient resources; not success, recorded as `resource_skip`."""

    exit_code = RESOURCE_SKIP


class ExactnessViolation(LightconeError):
    """Any target-exactness / version-consistency violation (spec 3.3)."""

    exit_code = EXACTNESS_VERSION_FAILURE


class NumericalFailure(LightconeError):
    """NaN/Inf/overflow/empty-supervision failures (spec 6.3)."""

    exit_code = NUMERICAL_FAILURE


class RuntimeGpuFailure(LightconeError):
    exit_code = RUNTIME_GPU_FAILURE


class ArtifactValidationFailure(LightconeError):
    exit_code = ARTIFACT_VALIDATION_FAILURE


class IncompleteCoverage(LightconeError):
    exit_code = INCOMPLETE_COVERAGE
