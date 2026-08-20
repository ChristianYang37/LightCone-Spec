from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightcone_spec.orchestration.autodl_provider_runtime import (
    AUTODL_LIST_PATH,
    AUTODL_POWER_OFF_PATH,
    AUTODL_STATUS_PATH,
    AutoDlApiResponse,
    AutoDlProApiClient,
    AutoDlProviderRuntimeError,
    reduce_autodl_provider_responses,
    sample_autodl_provider_runtime,
    transition_autodl_instance_power,
)
from lightcone_spec.orchestration.experiment_operator import (
    ExperimentOperatorStore,
)

INSTANCE = "pro-test-instance-0001"
START = "2026-08-19T04:16:21.123456789+08:00"
STOP = "2026-08-19T05:43:08.000000001+08:00"


def _status(state: str) -> AutoDlApiResponse:
    return AutoDlApiResponse(
        200,
        {
            "code": "Success",
            "data": state,
            "msg": "",
            "request_id": f"status-{state}",
        },
    )


def _listing(state: str, *, password: str | None = None) -> AutoDlApiResponse:
    row = {
        "uuid": INSTANCE,
        "status": state,
        "req_gpu_amount": 2,
        "started_at": {"Time": START, "Valid": True},
        "stopped_at": {
            "Time": STOP if state == "shutdown" else "0001-01-01T00:00:00Z",
            "Valid": state == "shutdown",
        },
    }
    if password is not None:
        row["root_password"] = password
    return AutoDlApiResponse(
        200,
        {
            "code": "Success",
            "data": {
                "list": [row],
                "page_index": 1,
                "page_size": 100,
                "max_page": 1,
            },
            "msg": "",
            "request_id": f"list-{state}",
        },
    )


def test_reduce_cross_checks_status_and_preserves_nanoseconds() -> None:
    observed = 1_800_000_000_000_000_000
    running, evidence = reduce_autodl_provider_responses(
        instance_uuid=INSTANCE,
        observed_at_ns=observed,
        status_response=_status("running"),
        list_response=_listing("running", password="do-not-store"),
    )
    assert running.state == "running"
    assert running.gpu_count == 2
    assert running.provider_started_at_ns % 1_000_000_000 == 123_456_789
    assert running.provider_stopped_at_ns is None
    encoded = json.dumps(evidence, sort_keys=True)
    assert "do-not-store" not in encoded
    assert "<redacted>" in encoded

    shutdown, _ = reduce_autodl_provider_responses(
        instance_uuid=INSTANCE,
        observed_at_ns=observed,
        status_response=_status("shutdown"),
        list_response=_listing("shutdown"),
    )
    assert shutdown.provider_stopped_at_ns is not None
    assert shutdown.provider_stopped_at_ns % 1_000_000_000 == 1


def test_reduce_rejects_disagreeing_status() -> None:
    with pytest.raises(AutoDlProviderRuntimeError, match="disagree"):
        reduce_autodl_provider_responses(
            instance_uuid=INSTANCE,
            observed_at_ns=1_800_000_000_000_000_000,
            status_response=_status("running"),
            list_response=_listing("shutdown"),
        )


def test_sampler_never_persists_token_and_records_billed_interval(
    tmp_path: Path,
) -> None:
    token = "private-autodl-token"
    responses = {
        ("GET", AUTODL_STATUS_PATH): _status("running"),
        ("POST", AUTODL_LIST_PATH): _listing("running", password=token),
    }

    class Client:
        def request(self, method, path, body):
            assert body == (
                {"instance_uuid": INSTANCE}
                if method == "GET"
                else {"page_index": 1, "page_size": 100}
            )
            return responses[(method, path)]

    database = tmp_path / "operator.sqlite"
    evidence = tmp_path / "running.json"
    observed = 1_800_000_000_000_000_000
    with ExperimentOperatorStore(database, run_id="run-v03") as store:
        sample = sample_autodl_provider_runtime(
            store=store,
            instance_uuid=INSTANCE,
            output_path=evidence,
            environment={"AUTODL_DEVELOPER_TOKEN": token},
            client_factory=lambda observed_token: (
                Client()
                if observed_token == token
                else pytest.fail("token changed before transport")
            ),
            clock_ns=lambda: observed,
        )
        assert sample.state == "running"
        assert store.whole_instance_billed_gpu_seconds(require_complete=False) > 0
        assert token.encode() not in evidence.read_bytes()
        snapshot = store.snapshot()
        assert (
            sample.response_sha256
            in snapshot["provider_billing_intervals"][0]["response_sha256s"]
        )
        store.export_progress(tmp_path / "progress", exported_at_ns=observed + 1)
        event = next(
            json.loads(line)
            for line in (tmp_path / "progress" / "watchdog_events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if "PROVIDER_RUNTIME_EVIDENCE_BOUND" in line
        )
        assert event["payload"]["evidence_path"] == str(evidence)


def test_official_client_uses_raw_authorization_header() -> None:
    captured = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_error):
            return None

        def read(self, _maximum):
            return json.dumps(
                {
                    "code": "Success",
                    "data": "running",
                    "msg": "",
                    "request_id": "request-1",
                }
            ).encode()

    def opener(request, *, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["method"] = request.get_method()
        captured["url"] = request.full_url
        captured["body"] = request.data
        captured["timeout"] = timeout
        return Response()

    client = AutoDlProApiClient(token="raw-token", opener=opener)
    response = client.request("GET", AUTODL_STATUS_PATH, {"instance_uuid": INSTANCE})
    assert response.payload["code"] == "Success"
    assert captured == {
        "authorization": "raw-token",
        "method": "GET",
        "url": (
            "https://api.autodl.com/api/v1/dev/instance/pro/status"
            f"?instance_uuid={INSTANCE}"
        ),
        "body": None,
        "timeout": 30.0,
    }


def test_power_off_is_safe_resumable_and_dually_confirmed(tmp_path: Path) -> None:
    database = tmp_path / "operator.sqlite"
    probe = tmp_path / "shutdown-probe.json"
    probe_observed = 1_800_000_000_000_000_000
    probe.write_bytes(
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "autodl_power_off_safety_probe",
                    "instance_uuid": INSTANCE,
                    "run_id": "run-v03",
                    "observed_at_ns": probe_observed,
                    "observation_window_seconds": 5,
                    "scheduler_control_state": "STOP",
                    "running_attempt_count": 0,
                    "evidence_writer_process_count": 0,
                    "gpu_compute_process_count": 0,
                    "open_measurement_port_count": 0,
                    "log_growth_bytes": 0,
                    "probe_command_sha256": "a" * 64,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    )
    calls: list[tuple[str, str, dict[str, object]]] = []

    class Client:
        def __init__(self, state: str, *, permit_mutation: bool) -> None:
            self.state = state
            self.permit_mutation = permit_mutation

        def request(self, method, path, body):
            calls.append((method, path, body))
            if path == AUTODL_POWER_OFF_PATH:
                assert self.permit_mutation
                return AutoDlApiResponse(
                    200,
                    {
                        "code": "Success",
                        "data": None,
                        "msg": "",
                        "request_id": "power-off-request-1",
                    },
                )
            if path == AUTODL_STATUS_PATH:
                return _status(self.state)
            assert path == AUTODL_LIST_PATH
            return _listing(self.state)

    output = (tmp_path / "power-off.json").resolve()
    with ExperimentOperatorStore(database, run_id="run-v03") as store:
        store.set_dispatch_stop("safe_shutdown", stopped_at_ns=probe_observed)
        with pytest.raises(AutoDlProviderRuntimeError, match="did not converge"):
            transition_autodl_instance_power(
                store=store,
                operation="power_off",
                instance_uuid=INSTANCE,
                output_path=output,
                safety_probe_path=probe.resolve(),
                environment={"AUTODL_DEVELOPER_TOKEN": "private-token"},
                client_factory=lambda token: (
                    Client("running", permit_mutation=True)
                    if token == "private-token"
                    else pytest.fail("token changed")
                ),
                clock_ns=lambda: probe_observed + 1,
                maximum_confirmation_attempts=1,
                confirmation_interval_seconds=0,
            )
        assert output.with_name(output.name + ".request.json").is_file()

        receipt = transition_autodl_instance_power(
            store=store,
            operation="power_off",
            instance_uuid=INSTANCE,
            output_path=output,
            safety_probe_path=probe.resolve(),
            environment={"AUTODL_DEVELOPER_TOKEN": "private-token"},
            client_factory=lambda _token: Client("shutdown", permit_mutation=False),
            clock_ns=lambda: probe_observed + 2,
            confirmation_interval_seconds=0,
        )
        assert receipt.target_state == "shutdown"
        assert receipt.provider_request_id == "power-off-request-1"
        assert store.whole_instance_billed_gpu_seconds(require_complete=True) >= 0
    assert sum(path == AUTODL_POWER_OFF_PATH for _method, path, _body in calls) == 1
    evidence = json.loads(output.read_bytes())
    assert evidence["receipt"]["safety_probe_sha256"]
    assert b"private-token" not in output.read_bytes()


def test_power_off_rejects_running_scheduler_before_provider_call(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "probe.json"
    probe.write_bytes(b"{}\n")
    with (
        ExperimentOperatorStore(tmp_path / "unsafe.sqlite", run_id="run-v03") as store,
        pytest.raises(AutoDlProviderRuntimeError),
    ):
        transition_autodl_instance_power(
            store=store,
            operation="power_off",
            instance_uuid=INSTANCE,
            output_path=(tmp_path / "never.json").resolve(),
            safety_probe_path=probe.resolve(),
            environment={"AUTODL_DEVELOPER_TOKEN": "private-token"},
            client_factory=lambda _token: pytest.fail("provider must not be called"),
            clock_ns=lambda: 1_800_000_000_000_000_000,
        )
