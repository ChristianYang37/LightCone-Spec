"""Credential-safe AutoDL Pro instance lifecycle sampler.

The developer token is read only from an environment variable and is used as
the raw ``Authorization`` header required by AutoDL.  It is never accepted in
argv, persisted in SQLite, or copied into evidence.  Each observation combines
the official status and list endpoints, stores their recursively redacted raw
responses, and emits the exact :class:`ProviderRuntimeSample` consumed by the
whole-instance billed GPU-hour reducer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from lightcone_spec.orchestration.experiment_operator import (
    ExperimentOperatorStore,
    ProviderRuntimeSample,
    SingletonOperatorLock,
)

AUTODL_PRO_API_BASE = "https://api.autodl.com"
AUTODL_STATUS_PATH = "/api/v1/dev/instance/pro/status"
AUTODL_LIST_PATH = "/api/v1/dev/instance/pro/list"
AUTODL_POWER_ON_PATH = "/api/v1/dev/instance/pro/power_on"
AUTODL_POWER_OFF_PATH = "/api/v1/dev/instance/pro/power_off"
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_RFC3339 = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})T(?P<clock>\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?(?P<zone>Z|[+-]\d{2}:\d{2})$"
)
_SECRET_KEY_PARTS = ("authorization", "credential", "password", "secret", "token")


class AutoDlProviderRuntimeError(RuntimeError):
    """Raised when AutoDL lifecycle evidence is incomplete or inconsistent."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class AutoDlPowerOffSafetyProbe:
    """Remote-observed safe boundary required before a provider power-off."""

    schema_version: int
    kind: str
    instance_uuid: str
    run_id: str
    observed_at_ns: int
    observation_window_seconds: int
    scheduler_control_state: str
    running_attempt_count: int
    evidence_writer_process_count: int
    gpu_compute_process_count: int
    open_measurement_port_count: int
    log_growth_bytes: int
    probe_command_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "autodl_power_off_safety_probe"
            or not self.instance_uuid.startswith("pro-")
            or not self.run_id
            or self.observed_at_ns < 1
            or self.observation_window_seconds < 5
            or self.scheduler_control_state != "STOP"
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value != 0
                for value in (
                    self.running_attempt_count,
                    self.evidence_writer_process_count,
                    self.gpu_compute_process_count,
                    self.open_measurement_port_count,
                    self.log_growth_bytes,
                )
            )
            or not _is_sha256(self.probe_command_sha256)
        ):
            raise AutoDlProviderRuntimeError("power-off safety probe differs")


@dataclass(frozen=True)
class AutoDlPowerTransitionReceipt:
    schema_version: int
    kind: str
    operation: Literal["power_on", "power_off"]
    instance_uuid: str
    target_state: Literal["running", "shutdown"]
    provider_request_id: str
    requested_at_ns: int
    confirmed_at_ns: int
    confirmation_attempt_count: int
    provider_sample_id: str
    provider_response_sha256: str
    safety_probe_path: str | None
    safety_probe_sha256: str | None

    def __post_init__(self) -> None:
        expected_state = "running" if self.operation == "power_on" else "shutdown"
        if (
            self.schema_version != 1
            or self.kind != "autodl_power_transition_receipt"
            or self.target_state != expected_state
            or not self.instance_uuid.startswith("pro-")
            or not self.provider_request_id
            or self.requested_at_ns < 1
            or self.confirmed_at_ns < self.requested_at_ns
            or self.confirmation_attempt_count < 1
            or not _is_sha256(self.provider_sample_id)
            or not _is_sha256(self.provider_response_sha256)
            or ((self.safety_probe_path is None) != (self.safety_probe_sha256 is None))
            or (
                self.operation == "power_off"
                and (
                    self.safety_probe_path is None
                    or not _is_sha256(self.safety_probe_sha256)
                )
            )
            or (self.operation == "power_on" and self.safety_probe_path is not None)
        ):
            raise AutoDlProviderRuntimeError("power transition receipt differs")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _absolute_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise AutoDlProviderRuntimeError(f"{label} must be absolute and normalized")
    return path


def _atomic_write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise AutoDlProviderRuntimeError("provider evidence output already exists")
    body = _canonical_bytes(value)
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as error:
        raise AutoDlProviderRuntimeError(
            "provider evidence output became occupied"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _rfc3339_ns(value: object, label: str) -> int:
    if type(value) is not str:
        raise AutoDlProviderRuntimeError(f"{label} must be RFC3339 text")
    match = _RFC3339.fullmatch(value)
    if match is None:
        raise AutoDlProviderRuntimeError(f"{label} is not exact RFC3339")
    zone = "+00:00" if match.group("zone") == "Z" else match.group("zone")
    parsed = datetime.fromisoformat(
        f"{match.group('date')}T{match.group('clock')}{zone}"
    )
    seconds = int(parsed.timestamp())
    fraction = (match.group("fraction") or "").ljust(9, "0")
    return seconds * 1_000_000_000 + int(fraction or "0")


def _redact(value: object) -> object:
    if type(value) is dict:
        output: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            if type(raw_key) is not str:
                raise AutoDlProviderRuntimeError("provider response key is not text")
            lowered = raw_key.lower()
            output[raw_key] = (
                "<redacted>"
                if any(part in lowered for part in _SECRET_KEY_PARTS)
                else _redact(raw_value)
            )
        return output
    if type(value) is list:
        return [_redact(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise AutoDlProviderRuntimeError("provider response contains a non-JSON value")


@dataclass(frozen=True)
class AutoDlApiResponse:
    status_code: int
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            isinstance(self.status_code, bool)
            or not isinstance(self.status_code, int)
            or self.status_code < 100
            or self.status_code > 599
        ):
            raise AutoDlProviderRuntimeError("provider HTTP status is invalid")
        if type(self.payload) is not dict:
            raise AutoDlProviderRuntimeError("provider payload is not one object")


class AutoDlProApiClient:
    """Minimal official Pro API client with an injectable network boundary."""

    def __init__(
        self,
        *,
        token: str,
        base_url: str = AUTODL_PRO_API_BASE,
        timeout_seconds: float = 30.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if type(token) is not str or not token or "\n" in token or "\r" in token:
            raise AutoDlProviderRuntimeError("AutoDL token is empty or malformed")
        if base_url != AUTODL_PRO_API_BASE:
            raise AutoDlProviderRuntimeError("AutoDL API base differs from authority")
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise AutoDlProviderRuntimeError("AutoDL timeout must be positive")
        self._token = token
        self._base_url = base_url
        self._timeout_seconds = float(timeout_seconds)
        self._opener = opener

    def request(
        self,
        method: str,
        path: str,
        body: Mapping[str, object],
    ) -> AutoDlApiResponse:
        if method not in {"GET", "POST"} or path not in {
            AUTODL_STATUS_PATH,
            AUTODL_LIST_PATH,
            AUTODL_POWER_ON_PATH,
            AUTODL_POWER_OFF_PATH,
        }:
            raise AutoDlProviderRuntimeError("AutoDL request route is not registered")
        expected_method = {
            AUTODL_STATUS_PATH: "GET",
            AUTODL_LIST_PATH: "POST",
            AUTODL_POWER_ON_PATH: "POST",
            AUTODL_POWER_OFF_PATH: "POST",
        }[path]
        if method != expected_method:
            raise AutoDlProviderRuntimeError("AutoDL request method differs")
        request_url = self._base_url + path
        encoded: bytes | None
        if method == "GET":
            query = urllib.parse.urlencode(tuple(sorted(body.items())))
            request_url = f"{request_url}?{query}"
            encoded = None
        else:
            encoded = _canonical_bytes(dict(body))
        request = urllib.request.Request(
            request_url,
            data=encoded,
            headers={
                "Authorization": self._token,
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            response = self._opener(request, timeout=self._timeout_seconds)
            with response:
                status_code = int(response.status)
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except (OSError, urllib.error.URLError) as error:
            raise AutoDlProviderRuntimeError("AutoDL API request failed") from error
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise AutoDlProviderRuntimeError("AutoDL response exceeds bound")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AutoDlProviderRuntimeError(
                "AutoDL response is not UTF-8 JSON"
            ) from error
        if type(value) is not dict:
            raise AutoDlProviderRuntimeError("AutoDL response is not one object")
        return AutoDlApiResponse(status_code=status_code, payload=value)


def _success_payload(response: AutoDlApiResponse, label: str) -> dict[str, Any]:
    value = dict(response.payload)
    if response.status_code != 200 or value.get("code") != "Success":
        raise AutoDlProviderRuntimeError(f"{label} did not return code Success")
    if type(value.get("request_id")) is not str or not value["request_id"]:
        raise AutoDlProviderRuntimeError(f"{label} lacks a request ID")
    return value


def _time_field(row: Mapping[str, Any], name: str, *, required: bool) -> int | None:
    raw = row.get(name)
    if type(raw) is not dict or set(raw) != {"Time", "Valid"}:
        raise AutoDlProviderRuntimeError(f"AutoDL {name} wrapper differs")
    if type(raw["Valid"]) is not bool:
        raise AutoDlProviderRuntimeError(f"AutoDL {name} validity is not boolean")
    if not raw["Valid"]:
        if required:
            raise AutoDlProviderRuntimeError(f"AutoDL {name} is not valid")
        return None
    return _rfc3339_ns(raw["Time"], f"AutoDL {name}")


def reduce_autodl_provider_responses(
    *,
    instance_uuid: str,
    observed_at_ns: int,
    status_response: AutoDlApiResponse,
    list_response: AutoDlApiResponse | Sequence[AutoDlApiResponse],
) -> tuple[ProviderRuntimeSample, dict[str, object]]:
    """Cross-check the two official endpoints and build redacted evidence."""

    if type(instance_uuid) is not str or not instance_uuid.startswith("pro-"):
        raise AutoDlProviderRuntimeError("AutoDL instance UUID is malformed")
    if (
        isinstance(observed_at_ns, bool)
        or not isinstance(observed_at_ns, int)
        or observed_at_ns < 1
    ):
        raise AutoDlProviderRuntimeError("provider observation time is invalid")
    status_payload = _success_payload(status_response, "AutoDL status")
    list_responses = (
        (list_response,)
        if type(list_response) is AutoDlApiResponse
        else tuple(list_response)
    )
    if not list_responses or any(
        type(response) is not AutoDlApiResponse for response in list_responses
    ):
        raise AutoDlProviderRuntimeError("AutoDL list response pages are malformed")
    list_payloads = tuple(
        _success_payload(response, "AutoDL list") for response in list_responses
    )
    status = status_payload.get("data")
    if status not in {"running", "shutdown"}:
        raise AutoDlProviderRuntimeError("AutoDL status is not running or shutdown")
    rows: list[object] = []
    expected_pages = len(list_payloads)
    for expected_page, list_payload in enumerate(list_payloads, start=1):
        data = list_payload.get("data")
        page_rows = None if type(data) is not dict else data.get("list")
        if (
            type(data) is not dict
            or type(page_rows) is not list
            or data.get("page_index") != expected_page
            or data.get("max_page") != expected_pages
        ):
            raise AutoDlProviderRuntimeError("AutoDL list pagination differs")
        rows.extend(page_rows)
    matches = tuple(
        row for row in rows if type(row) is dict and row.get("uuid") == instance_uuid
    )
    if len(matches) != 1:
        raise AutoDlProviderRuntimeError("AutoDL list lacks one exact instance")
    row = matches[0]
    if row.get("status") != status:
        raise AutoDlProviderRuntimeError("AutoDL status/list observations disagree")
    gpu_count = row.get("req_gpu_amount")
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 1:
        raise AutoDlProviderRuntimeError("AutoDL requested GPU count is invalid")
    started_at_ns = _time_field(row, "started_at", required=True)
    assert started_at_ns is not None
    stopped_at_ns = _time_field(
        row,
        "stopped_at",
        required=status == "shutdown",
    )
    if status == "running" and stopped_at_ns is not None:
        raise AutoDlProviderRuntimeError("running AutoDL row carries a stop time")
    redacted_raw = {
        "status_response": _redact(status_payload),
        "list_responses": [_redact(payload) for payload in list_payloads],
    }
    response_sha256 = _content_sha256(redacted_raw)
    sample = ProviderRuntimeSample(
        instance_uuid=instance_uuid,
        state=status,
        observed_at_ns=observed_at_ns,
        provider_started_at_ns=started_at_ns,
        provider_stopped_at_ns=stopped_at_ns,
        gpu_count=gpu_count,
        response_sha256=response_sha256,
    )
    evidence = {
        "schema_version": 1,
        "kind": "autodl_pro_provider_runtime_evidence",
        "instance_uuid": instance_uuid,
        "observed_at_ns": observed_at_ns,
        "redacted_raw_responses": redacted_raw,
        "response_sha256": response_sha256,
        "sample": asdict(sample),
        "sample_id": sample.sample_id,
    }
    return sample, evidence


def _fetch_provider_responses(
    client: AutoDlProApiClient,
    *,
    instance_uuid: str,
) -> tuple[AutoDlApiResponse, tuple[AutoDlApiResponse, ...]]:
    status = client.request("GET", AUTODL_STATUS_PATH, {"instance_uuid": instance_uuid})
    first_listing = client.request(
        "POST",
        AUTODL_LIST_PATH,
        {"page_index": 1, "page_size": 100},
    )
    first_payload = _success_payload(first_listing, "AutoDL list")
    first_data = first_payload.get("data")
    max_page = None if type(first_data) is not dict else first_data.get("max_page")
    if (
        isinstance(max_page, bool)
        or not isinstance(max_page, int)
        or max_page < 1
        or max_page > 1_000
    ):
        raise AutoDlProviderRuntimeError("AutoDL list max_page is invalid")
    listings = [first_listing]
    for page_index in range(2, max_page + 1):
        listings.append(
            client.request(
                "POST",
                AUTODL_LIST_PATH,
                {"page_index": page_index, "page_size": 100},
            )
        )
    return status, tuple(listings)


def sample_autodl_provider_runtime(
    *,
    store: ExperimentOperatorStore,
    instance_uuid: str,
    output_path: str | Path,
    token_environment_name: str = "AUTODL_DEVELOPER_TOKEN",
    environment: Mapping[str, str] | None = None,
    client_factory: Callable[[str], AutoDlProApiClient] | None = None,
    clock_ns: Callable[[], int] = time.time_ns,
) -> ProviderRuntimeSample:
    """Fetch, redact, publish, and durably record one lifecycle observation."""

    if type(store) is not ExperimentOperatorStore:
        raise TypeError("provider sampler requires an exact operator store")
    if (
        type(token_environment_name) is not str
        or not token_environment_name
        or "=" in token_environment_name
    ):
        raise AutoDlProviderRuntimeError("token environment name is malformed")
    values = os.environ if environment is None else environment
    token = values.get(token_environment_name)
    if type(token) is not str or not token:
        raise AutoDlProviderRuntimeError(
            f"required token environment variable {token_environment_name} is absent"
        )
    client = (
        AutoDlProApiClient(token=token)
        if client_factory is None
        else client_factory(token)
    )
    observed_at_ns = int(clock_ns())
    status, listings = _fetch_provider_responses(
        client,
        instance_uuid=instance_uuid,
    )
    sample, evidence = reduce_autodl_provider_responses(
        instance_uuid=instance_uuid,
        observed_at_ns=observed_at_ns,
        status_response=status,
        list_response=listings,
    )
    body = _canonical_bytes(evidence)
    if token.encode("utf-8") in body:
        raise AutoDlProviderRuntimeError("provider evidence contains the credential")
    output = _absolute_path(output_path, "provider evidence output")
    _atomic_write_new(output, evidence)
    if (
        hashlib.sha256(_canonical_bytes(evidence)).hexdigest()
        != hashlib.sha256(output.read_bytes()).hexdigest()
    ):
        raise RuntimeError("published provider evidence changed")
    store.record_provider_runtime_sample(sample)
    store.record_watchdog_event(
        event_type="PROVIDER_RUNTIME_EVIDENCE_BOUND",
        severity="INFO",
        payload={
            "evidence_path": str(output),
            "evidence_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "instance_uuid": instance_uuid,
            "sample_id": sample.sample_id,
            "state": sample.state,
        },
        occurred_at_ns=observed_at_ns,
    )
    return sample


def transition_autodl_instance_power(
    *,
    store: ExperimentOperatorStore,
    operation: Literal["power_on", "power_off"],
    instance_uuid: str,
    output_path: str | Path,
    safety_probe_path: str | Path | None = None,
    token_environment_name: str = "AUTODL_DEVELOPER_TOKEN",
    environment: Mapping[str, str] | None = None,
    client_factory: Callable[[str], AutoDlProApiClient] | None = None,
    clock_ns: Callable[[], int] = time.time_ns,
    sleeper: Callable[[float], None] = time.sleep,
    maximum_confirmation_attempts: int = 24,
    confirmation_interval_seconds: float = 5.0,
) -> AutoDlPowerTransitionReceipt:
    """Issue at most one power mutation and prove status/list convergence.

    The mutation response is published to a companion request journal before
    polling.  A restart reopens that journal and therefore never reissues the
    provider mutation merely because confirmation was interrupted.
    """

    if type(store) is not ExperimentOperatorStore:
        raise TypeError("power transition requires an exact operator store")
    if operation not in {"power_on", "power_off"}:
        raise AutoDlProviderRuntimeError("power transition operation differs")
    if not instance_uuid.startswith("pro-"):
        raise AutoDlProviderRuntimeError("AutoDL instance UUID is malformed")
    if (
        isinstance(maximum_confirmation_attempts, bool)
        or not isinstance(maximum_confirmation_attempts, int)
        or maximum_confirmation_attempts < 1
        or maximum_confirmation_attempts > 120
        or isinstance(confirmation_interval_seconds, bool)
        or not isinstance(confirmation_interval_seconds, (int, float))
        or confirmation_interval_seconds < 0
        or confirmation_interval_seconds > 60
    ):
        raise AutoDlProviderRuntimeError("power confirmation policy differs")
    values = os.environ if environment is None else environment
    token = values.get(token_environment_name)
    if type(token) is not str or not token:
        raise AutoDlProviderRuntimeError(
            f"required token environment variable {token_environment_name} is absent"
        )
    output = _absolute_path(output_path, "power transition evidence output")
    request_journal = output.with_name(output.name + ".request.json")
    if output.exists() or output.is_symlink():
        raise AutoDlProviderRuntimeError("power transition output already exists")
    mutation_was_already_journaled = bool(
        request_journal.exists() or request_journal.is_symlink()
    )
    safety_probe, safety_path, safety_sha256 = _power_off_safety_gate(
        store=store,
        operation=operation,
        instance_uuid=instance_uuid,
        safety_probe_path=safety_probe_path,
        now_ns=int(clock_ns()),
        resuming_journaled_mutation=mutation_was_already_journaled,
    )
    client = (
        AutoDlProApiClient(token=token)
        if client_factory is None
        else client_factory(token)
    )
    if request_journal.exists() or request_journal.is_symlink():
        mutation_journal = _load_canonical_object(
            request_journal,
            "power mutation request journal",
        )
        _validate_power_mutation_journal(
            mutation_journal,
            operation=operation,
            instance_uuid=instance_uuid,
            safety_probe_sha256=safety_sha256,
        )
    else:
        requested_at_ns = int(clock_ns())
        path = (
            AUTODL_POWER_ON_PATH if operation == "power_on" else AUTODL_POWER_OFF_PATH
        )
        body: dict[str, object] = {"instance_uuid": instance_uuid}
        if operation == "power_on":
            body["payload"] = "gpu"
        mutation = client.request("POST", path, body)
        mutation_payload = _success_payload(mutation, f"AutoDL {operation}")
        mutation_journal = {
            "schema_version": 1,
            "kind": "autodl_power_mutation_request_journal",
            "operation": operation,
            "instance_uuid": instance_uuid,
            "provider_request_id": mutation_payload["request_id"],
            "requested_at_ns": requested_at_ns,
            "redacted_provider_response": _redact(mutation_payload),
            "safety_probe_sha256": safety_sha256,
        }
        mutation_journal["journal_sha256"] = _content_sha256(mutation_journal)
        _atomic_write_new(request_journal, mutation_journal)

    target_state = "running" if operation == "power_on" else "shutdown"
    confirmations: list[dict[str, object]] = []
    final_sample: ProviderRuntimeSample | None = None
    final_evidence: dict[str, object] | None = None
    last_error: AutoDlProviderRuntimeError | None = None
    for attempt in range(1, maximum_confirmation_attempts + 1):
        observed_at_ns = int(clock_ns())
        try:
            status, listings = _fetch_provider_responses(
                client,
                instance_uuid=instance_uuid,
            )
            sample, evidence = reduce_autodl_provider_responses(
                instance_uuid=instance_uuid,
                observed_at_ns=observed_at_ns,
                status_response=status,
                list_response=listings,
            )
            confirmations.append(
                {
                    "attempt": attempt,
                    "observed_at_ns": observed_at_ns,
                    "state": sample.state,
                    "response_sha256": sample.response_sha256,
                }
            )
            if sample.state == target_state:
                final_sample = sample
                final_evidence = evidence
                break
        except AutoDlProviderRuntimeError as error:
            last_error = error
            confirmations.append(
                {
                    "attempt": attempt,
                    "observed_at_ns": observed_at_ns,
                    "state": "INCONSISTENT",
                    "error_type": type(error).__name__,
                }
            )
        if attempt != maximum_confirmation_attempts:
            sleeper(float(confirmation_interval_seconds))
    if final_sample is None or final_evidence is None:
        raise AutoDlProviderRuntimeError(
            "AutoDL status/list did not converge to the requested state"
        ) from last_error
    receipt = AutoDlPowerTransitionReceipt(
        schema_version=1,
        kind="autodl_power_transition_receipt",
        operation=operation,
        instance_uuid=instance_uuid,
        target_state=target_state,  # type: ignore[arg-type]
        provider_request_id=str(mutation_journal["provider_request_id"]),
        requested_at_ns=int(mutation_journal["requested_at_ns"]),
        confirmed_at_ns=final_sample.observed_at_ns,
        confirmation_attempt_count=len(confirmations),
        provider_sample_id=final_sample.sample_id,
        provider_response_sha256=final_sample.response_sha256,
        safety_probe_path=(None if safety_path is None else str(safety_path)),
        safety_probe_sha256=safety_sha256,
    )
    evidence = {
        "schema_version": 1,
        "kind": "autodl_power_transition_evidence",
        "request_journal_path": str(request_journal),
        "request_journal_sha256": hashlib.sha256(
            request_journal.read_bytes()
        ).hexdigest(),
        "receipt": asdict(receipt),
        "receipt_sha256": _content_sha256(asdict(receipt)),
        "confirmation_history": confirmations,
        "final_provider_evidence": final_evidence,
        "safety_probe": None if safety_probe is None else asdict(safety_probe),
    }
    body = _canonical_bytes(evidence)
    if token.encode("utf-8") in body:
        raise AutoDlProviderRuntimeError("power evidence contains the credential")
    _atomic_write_new(output, evidence)
    store.record_provider_runtime_sample(final_sample)
    store.record_watchdog_event(
        event_type=f"AUTODL_{operation.upper()}_CONFIRMED",
        severity="INFO",
        payload={
            "evidence_path": str(output),
            "evidence_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "instance_uuid": instance_uuid,
            "provider_request_id": receipt.provider_request_id,
            "state": target_state,
        },
        occurred_at_ns=receipt.confirmed_at_ns,
    )
    return receipt


def transition_autodl_instance_power_stateless(
    *,
    operation: Literal["power_off"],
    instance_uuid: str,
    run_id: str,
    shutdown_authority_sha256: str,
    output_path: str | Path,
    safety_probe_path: str | Path,
    token_environment_name: str = "AUTODL_DEVELOPER_TOKEN",
    environment: Mapping[str, str] | None = None,
    client_factory: Callable[[str], AutoDlProApiClient] | None = None,
    clock_ns: Callable[[], int] = time.time_ns,
    sleeper: Callable[[float], None] = time.sleep,
    maximum_confirmation_attempts: int = 24,
    confirmation_interval_seconds: float = 5.0,
) -> AutoDlPowerTransitionReceipt:
    """Power off from an immutable off-host shutdown authority.

    Unlike :func:`transition_autodl_instance_power`, this function never opens
    or writes the remote operator SQLite database.  It is therefore safe to
    run on the local archive host after the remote scientific closure has
    promised that no later database write is required.  The request journal
    retains the same at-most-once crash-restart property.
    """

    if operation != "power_off":
        raise AutoDlProviderRuntimeError("stateless transition only supports power_off")
    if not instance_uuid.startswith("pro-") or not run_id or "\x00" in run_id:
        raise AutoDlProviderRuntimeError("stateless power identity is malformed")
    if not _is_sha256(shutdown_authority_sha256):
        raise AutoDlProviderRuntimeError("shutdown authority digest is malformed")
    if (
        isinstance(maximum_confirmation_attempts, bool)
        or not isinstance(maximum_confirmation_attempts, int)
        or not 1 <= maximum_confirmation_attempts <= 120
        or isinstance(confirmation_interval_seconds, bool)
        or not isinstance(confirmation_interval_seconds, (int, float))
        or not 0 <= confirmation_interval_seconds <= 60
    ):
        raise AutoDlProviderRuntimeError("power confirmation policy differs")
    values = os.environ if environment is None else environment
    token = values.get(token_environment_name)
    if type(token) is not str or not token:
        raise AutoDlProviderRuntimeError(
            f"required token environment variable {token_environment_name} is absent"
        )
    output = _absolute_path(output_path, "stateless power transition output")
    request_journal = output.with_name(output.name + ".request.json")
    intent_journal = output.with_name(output.name + ".intent.json")
    if output.exists() or output.is_symlink():
        raise AutoDlProviderRuntimeError("stateless power output already exists")
    probe_path = _absolute_path(safety_probe_path, "stateless shutdown probe")
    probe_value = _load_canonical_object(probe_path, "stateless shutdown probe")
    try:
        probe = AutoDlPowerOffSafetyProbe(**probe_value)
    except TypeError as error:
        raise AutoDlProviderRuntimeError(
            "stateless shutdown probe fields differ"
        ) from error
    probe_sha256 = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    now_ns = int(clock_ns())
    journal_exists = request_journal.exists() or request_journal.is_symlink()
    intent_exists = intent_journal.exists() or intent_journal.is_symlink()
    if (
        probe.instance_uuid != instance_uuid
        or probe.run_id != run_id
        or probe.observed_at_ns > now_ns
        or (
            not journal_exists
            and not intent_exists
            and now_ns - probe.observed_at_ns > 300 * 1_000_000_000
        )
    ):
        raise AutoDlProviderRuntimeError("stateless shutdown probe lineage differs")
    client = (
        AutoDlProApiClient(token=token)
        if client_factory is None
        else client_factory(token)
    )
    if journal_exists:
        mutation_journal = _load_canonical_object(
            request_journal,
            "stateless power mutation journal",
        )
        unsigned = dict(mutation_journal)
        digest = unsigned.pop("journal_sha256", None)
        if (
            set(mutation_journal)
            != {
                "schema_version",
                "kind",
                "operation",
                "instance_uuid",
                "run_id",
                "shutdown_authority_sha256",
                "provider_request_id",
                "requested_at_ns",
                "redacted_provider_response",
                "safety_probe_sha256",
                "journal_sha256",
            }
            or mutation_journal.get("schema_version") != 1
            or mutation_journal.get("kind")
            != "autodl_stateless_power_mutation_request_journal"
            or mutation_journal.get("operation") != "power_off"
            or mutation_journal.get("instance_uuid") != instance_uuid
            or mutation_journal.get("run_id") != run_id
            or mutation_journal.get("shutdown_authority_sha256")
            != shutdown_authority_sha256
            or mutation_journal.get("safety_probe_sha256") != probe_sha256
            or digest != _content_sha256(unsigned)
        ):
            raise AutoDlProviderRuntimeError("stateless power mutation journal differs")
    else:
        requested_at_ns = int(clock_ns())
        intent = {
            "schema_version": 1,
            "kind": "autodl_stateless_power_mutation_intent",
            "operation": "power_off",
            "instance_uuid": instance_uuid,
            "run_id": run_id,
            "shutdown_authority_sha256": shutdown_authority_sha256,
            "requested_at_ns": requested_at_ns,
            "safety_probe_sha256": probe_sha256,
        }
        intent["intent_sha256"] = _content_sha256(intent)
        if intent_exists:
            existing_intent = _load_canonical_object(
                intent_journal,
                "stateless power mutation intent",
            )
            unsigned = dict(existing_intent)
            digest = unsigned.pop("intent_sha256", None)
            if (
                set(existing_intent) != set(intent)
                or existing_intent.get("schema_version") != 1
                or existing_intent.get("kind")
                != "autodl_stateless_power_mutation_intent"
                or existing_intent.get("operation") != "power_off"
                or existing_intent.get("instance_uuid") != instance_uuid
                or existing_intent.get("run_id") != run_id
                or existing_intent.get("shutdown_authority_sha256")
                != shutdown_authority_sha256
                or existing_intent.get("safety_probe_sha256") != probe_sha256
                or digest != _content_sha256(unsigned)
            ):
                raise AutoDlProviderRuntimeError(
                    "stateless power mutation intent differs"
                )
            raise AutoDlProviderRuntimeError(
                "power_off intent is indeterminate; refusing a second mutation"
            )
        _atomic_write_new(intent_journal, intent)
        mutation = client.request(
            "POST",
            AUTODL_POWER_OFF_PATH,
            {"instance_uuid": instance_uuid},
        )
        mutation_payload = _success_payload(mutation, "AutoDL power_off")
        mutation_journal = {
            "schema_version": 1,
            "kind": "autodl_stateless_power_mutation_request_journal",
            "operation": "power_off",
            "instance_uuid": instance_uuid,
            "run_id": run_id,
            "shutdown_authority_sha256": shutdown_authority_sha256,
            "provider_request_id": mutation_payload["request_id"],
            "requested_at_ns": requested_at_ns,
            "redacted_provider_response": _redact(mutation_payload),
            "safety_probe_sha256": probe_sha256,
        }
        mutation_journal["journal_sha256"] = _content_sha256(mutation_journal)
        _atomic_write_new(request_journal, mutation_journal)

    confirmations: list[dict[str, object]] = []
    final_sample: ProviderRuntimeSample | None = None
    final_evidence: dict[str, object] | None = None
    last_error: AutoDlProviderRuntimeError | None = None
    for attempt in range(1, maximum_confirmation_attempts + 1):
        observed_at_ns = int(clock_ns())
        try:
            status, listings = _fetch_provider_responses(
                client,
                instance_uuid=instance_uuid,
            )
            sample, evidence = reduce_autodl_provider_responses(
                instance_uuid=instance_uuid,
                observed_at_ns=observed_at_ns,
                status_response=status,
                list_response=listings,
            )
            confirmations.append(
                {
                    "attempt": attempt,
                    "observed_at_ns": observed_at_ns,
                    "state": sample.state,
                    "response_sha256": sample.response_sha256,
                }
            )
            if sample.state == "shutdown":
                final_sample = sample
                final_evidence = evidence
                break
        except AutoDlProviderRuntimeError as error:
            last_error = error
            confirmations.append(
                {
                    "attempt": attempt,
                    "observed_at_ns": observed_at_ns,
                    "state": "INCONSISTENT",
                    "error_type": type(error).__name__,
                }
            )
        if attempt != maximum_confirmation_attempts:
            sleeper(float(confirmation_interval_seconds))
    if final_sample is None or final_evidence is None:
        raise AutoDlProviderRuntimeError(
            "AutoDL status/list did not converge to shutdown"
        ) from last_error
    receipt = AutoDlPowerTransitionReceipt(
        schema_version=1,
        kind="autodl_power_transition_receipt",
        operation="power_off",
        instance_uuid=instance_uuid,
        target_state="shutdown",
        provider_request_id=str(mutation_journal["provider_request_id"]),
        requested_at_ns=int(mutation_journal["requested_at_ns"]),
        confirmed_at_ns=final_sample.observed_at_ns,
        confirmation_attempt_count=len(confirmations),
        provider_sample_id=final_sample.sample_id,
        provider_response_sha256=final_sample.response_sha256,
        safety_probe_path=str(probe_path),
        safety_probe_sha256=probe_sha256,
    )
    evidence = {
        "schema_version": 1,
        "kind": "autodl_stateless_power_off_transition_evidence",
        "run_id": run_id,
        "shutdown_authority_sha256": shutdown_authority_sha256,
        "request_journal_path": str(request_journal),
        "request_journal_sha256": hashlib.sha256(
            request_journal.read_bytes()
        ).hexdigest(),
        "intent_journal_path": str(intent_journal),
        "intent_journal_sha256": hashlib.sha256(
            intent_journal.read_bytes()
        ).hexdigest(),
        "receipt": asdict(receipt),
        "receipt_sha256": _content_sha256(asdict(receipt)),
        "confirmation_history": confirmations,
        "final_provider_evidence": final_evidence,
        "safety_probe": asdict(probe),
    }
    body = _canonical_bytes(evidence)
    if token.encode("utf-8") in body:
        raise AutoDlProviderRuntimeError(
            "stateless power evidence contains the credential"
        )
    _atomic_write_new(output, evidence)
    return receipt


def _power_off_safety_gate(
    *,
    store: ExperimentOperatorStore,
    operation: str,
    instance_uuid: str,
    safety_probe_path: str | Path | None,
    now_ns: int,
    resuming_journaled_mutation: bool = False,
) -> tuple[AutoDlPowerOffSafetyProbe | None, Path | None, str | None]:
    if operation == "power_on":
        if safety_probe_path is not None:
            raise AutoDlProviderRuntimeError("power-on cannot carry a shutdown probe")
        return None, None, None
    if safety_probe_path is None:
        raise AutoDlProviderRuntimeError("power-off requires a safety probe")
    path = _absolute_path(safety_probe_path, "power-off safety probe")
    value = _load_canonical_object(path, "power-off safety probe")
    try:
        probe = AutoDlPowerOffSafetyProbe(**value)
    except TypeError as error:
        raise AutoDlProviderRuntimeError(
            "power-off safety probe fields differ"
        ) from error
    if (
        probe.instance_uuid != instance_uuid
        or probe.run_id != store.run_id
        or probe.observed_at_ns > now_ns
        or (
            not resuming_journaled_mutation
            and now_ns - probe.observed_at_ns > 300 * 1_000_000_000
        )
    ):
        raise AutoDlProviderRuntimeError("power-off safety probe lineage differs")
    dispatch_state, _reason = store.dispatch_control()
    snapshot = store.snapshot()
    if dispatch_state != "STOP" or any(
        row["status"] == "RUNNING" for row in snapshot["attempts"]
    ):
        raise AutoDlProviderRuntimeError("operator is not safe for power-off")
    return probe, path, hashlib.sha256(path.read_bytes()).hexdigest()


def _load_canonical_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AutoDlProviderRuntimeError(f"{label} is not a regular file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AutoDlProviderRuntimeError(f"{label} is not JSON") from error
    if type(value) is not dict or payload != _canonical_bytes(value):
        raise AutoDlProviderRuntimeError(f"{label} is not canonical")
    return value


def _validate_power_mutation_journal(
    value: Mapping[str, Any],
    *,
    operation: str,
    instance_uuid: str,
    safety_probe_sha256: str | None,
) -> None:
    expected_fields = {
        "schema_version",
        "kind",
        "operation",
        "instance_uuid",
        "provider_request_id",
        "requested_at_ns",
        "redacted_provider_response",
        "safety_probe_sha256",
        "journal_sha256",
    }
    unsigned = dict(value)
    digest = unsigned.pop("journal_sha256", None)
    if (
        set(value) != expected_fields
        or value.get("schema_version") != 1
        or value.get("kind") != "autodl_power_mutation_request_journal"
        or value.get("operation") != operation
        or value.get("instance_uuid") != instance_uuid
        or type(value.get("provider_request_id")) is not str
        or not value.get("provider_request_id")
        or isinstance(value.get("requested_at_ns"), bool)
        or not isinstance(value.get("requested_at_ns"), int)
        or int(value["requested_at_ns"]) < 1
        or value.get("safety_probe_sha256") != safety_probe_sha256
        or digest != _content_sha256(unsigned)
    ):
        raise AutoDlProviderRuntimeError("power mutation request journal differs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--lock")
    parser.add_argument("--instance-uuid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--token-env", default="AUTODL_DEVELOPER_TOKEN")
    parser.add_argument("--power-operation", choices=("power_on", "power_off"))
    parser.add_argument("--safety-probe")
    parser.add_argument("--maximum-confirmation-attempts", type=int, default=24)
    parser.add_argument("--confirmation-interval-seconds", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    database = Path(arguments.db)
    lock = (
        Path(arguments.lock)
        if arguments.lock
        else database.with_name(f"{database.name}.operator.lock")
    )
    with SingletonOperatorLock(lock), ExperimentOperatorStore(database) as store:
        if arguments.power_operation is None:
            if arguments.safety_probe is not None:
                raise AutoDlProviderRuntimeError(
                    "provider sampling cannot carry a safety probe"
                )
            result: ProviderRuntimeSample | AutoDlPowerTransitionReceipt = (
                sample_autodl_provider_runtime(
                    store=store,
                    instance_uuid=arguments.instance_uuid,
                    output_path=arguments.output,
                    token_environment_name=arguments.token_env,
                )
            )
        else:
            result = transition_autodl_instance_power(
                store=store,
                operation=arguments.power_operation,
                instance_uuid=arguments.instance_uuid,
                output_path=arguments.output,
                safety_probe_path=arguments.safety_probe,
                token_environment_name=arguments.token_env,
                maximum_confirmation_attempts=(arguments.maximum_confirmation_attempts),
                confirmation_interval_seconds=(arguments.confirmation_interval_seconds),
            )
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTODL_LIST_PATH",
    "AUTODL_POWER_OFF_PATH",
    "AUTODL_POWER_ON_PATH",
    "AUTODL_PRO_API_BASE",
    "AUTODL_STATUS_PATH",
    "AutoDlApiResponse",
    "AutoDlPowerOffSafetyProbe",
    "AutoDlPowerTransitionReceipt",
    "AutoDlProApiClient",
    "AutoDlProviderRuntimeError",
    "reduce_autodl_provider_responses",
    "sample_autodl_provider_runtime",
    "transition_autodl_instance_power",
    "transition_autodl_instance_power_stateless",
]
