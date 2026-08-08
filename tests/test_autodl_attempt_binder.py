from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


def _binder_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "providers"
        / "autodl"
        / "autodl_attempt_binder.py"
    )
    spec = importlib.util.spec_from_file_location("autodl_attempt_binder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_process(proc_root: Path, pid: int, *, start_ticks: int, state: str = "S"):
    process = proc_root / str(pid)
    process.mkdir(parents=True, exist_ok=True)
    # remainder[0] is state (proc field 3); remainder[19] is starttime (field 22).
    remainder = [state, *("0" for _ in range(18)), str(start_ticks)]
    process.joinpath("stat").write_text(
        f"{pid} (launcher with ) parens) {' '.join(remainder)}\n",
        encoding="utf-8",
    )
    boot = proc_root / "sys" / "kernel" / "random"
    boot.mkdir(parents=True, exist_ok=True)
    boot.joinpath("boot_id").write_text("boot-a\n", encoding="utf-8")


def _receipt(path: Path, payload: dict, *, mtime: float = 100.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    Path(str(path) + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8"
    )
    os.utime(path, (mtime, mtime))


def _foundation_receipt(
    path: Path,
    *,
    status: str,
    passed: bool,
    mtime: float = 100.0,
) -> Path:
    evidence = path.parent / "foundation-input.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text('{"locked":true}\n', encoding="utf-8")
    _receipt(
        path,
        {
            "schema_version": 2,
            "status": status,
            "scope": "tts_0_40k_foundation",
            "formal_acceptance_foundation_pass": passed,
            "evidence": [
                {
                    "path": str(evidence.resolve()),
                    "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                }
            ],
        },
        mtime=mtime,
    )
    return evidence


def test_binder_process_identity_rejects_pid_reuse_and_zombie(tmp_path):
    binder = _binder_module()
    proc = tmp_path / "proc"
    _fake_process(proc, 123, start_ticks=77)

    identity = binder.capture_process_identity(123, proc_root=proc)
    assert identity == binder.ProcessIdentity(123, "boot-a", 77)
    assert binder.process_identity_matches(identity, proc_root=proc)
    proc.joinpath("uptime").write_text("100.0 1.0\n", encoding="utf-8")
    expected_start = 900.0 + 77 / os.sysconf("SC_CLK_TCK")
    assert binder.process_start_epoch(
        identity, proc_root=proc, now_epoch=1000.0
    ) == pytest.approx(expected_start)

    _fake_process(proc, 123, start_ticks=78)
    assert not binder.process_identity_matches(identity, proc_root=proc)
    _fake_process(proc, 123, start_ticks=77, state="Z")
    assert binder.capture_process_identity(123, proc_root=proc) is None


def test_binder_publication_is_attempt_safe(tmp_path):
    binder = _binder_module()
    first = binder.ProcessIdentity(101, "boot-a", 10)
    second = binder.ProcessIdentity(202, "boot-a", 20)

    attempt = binder.publish_attempt_binding(
        tmp_path, "proof-a", 1, first, launcher_start_epoch=100.0
    )
    assert attempt.joinpath("queue.pid").read_text().strip() == "proof-a\t1\t101"
    identity = json.loads(attempt.joinpath("launcher.identity.json").read_text())
    assert identity["launcher_start_epoch"] == 100.0
    assert binder.publish_heartbeat(
        tmp_path, "proof-a", 1, first, now_epoch=123.0
    )
    with pytest.raises(RuntimeError, match="already bound to another launcher"):
        binder.publish_attempt_binding(tmp_path, "proof-a", 1, second)

    with pytest.raises(RuntimeError, match="another attempt is published"):
        binder.publish_attempt_binding(tmp_path, "proof-b", 1, second)

    binder.publish_attempt_binding(
        tmp_path, "proof-b", 1, second, supersede_current=True
    )
    assert not binder.publish_heartbeat(
        tmp_path, "proof-a", 1, first, now_epoch=124.0
    )
    assert not binder.publish_terminal(tmp_path, "proof-a", 1, "complete")
    assert binder.publish_terminal(
        tmp_path, "proof-b", 1, "failed", reason="test_failure"
    )
    assert (tmp_path / "QUEUE_FAILED").read_text().strip() == "proof-b\t1"
    terminal = json.loads(
        (tmp_path / "sessions/proof-b/attempts/1/binder-terminal.json").read_text()
    )
    assert terminal["reason"] == "test_failure"


def test_binder_classifies_scientific_block_as_complete(tmp_path):
    binder = _binder_module()
    screen = tmp_path / "screen"
    confirmation = tmp_path / "confirmation"
    controller = tmp_path / "controller"
    headline = tmp_path / "headline"

    _receipt(
        confirmation / "formal-acceptance-comparison.json",
        {"formal_acceptance_claim_pass": False},
    )
    assert binder.proof_chain_terminal_state(
        screen_root=screen,
        confirmation_root=confirmation,
        controller_root=controller,
        headline_root=headline,
        not_before_epoch=50.0,
    ) == ("complete", "formal_confirmation_scientifically_blocked")

    # A terminal from an older attempt cannot make a new failed attempt look
    # successful.
    assert binder.proof_chain_terminal_state(
        screen_root=screen,
        confirmation_root=confirmation,
        controller_root=controller,
        headline_root=headline,
        not_before_epoch=150.0,
    )[0] == "failed"


def test_binder_classifies_new_tts_foundation_block_as_complete(tmp_path):
    binder = _binder_module()
    foundation = (
        tmp_path
        / "confirmation/tts-0-40k-foundation/TTS_0_40K_FOUNDATION.json"
    )
    _foundation_receipt(foundation, status="BLOCKED", passed=False)

    assert binder.proof_chain_terminal_state(
        screen_root=tmp_path / "screen",
        confirmation_root=tmp_path / "confirmation",
        controller_root=tmp_path / "controller",
        headline_root=tmp_path / "headline",
        not_before_epoch=50.0,
    ) == ("complete", "tts_0_40k_foundation_scientifically_blocked")


def test_binder_does_not_treat_confirmed_foundation_as_full_completion(tmp_path):
    binder = _binder_module()
    confirmation = tmp_path / "confirmation"
    foundation = (
        confirmation
        / "tts-0-40k-foundation/TTS_0_40K_FOUNDATION.json"
    )
    _foundation_receipt(
        foundation,
        status="TTS_0_40K_CONFIRMED",
        passed=True,
    )
    # A legacy negative terminal must not override the current foundation.
    _receipt(
        confirmation / "formal-acceptance-comparison.json",
        {"formal_acceptance_claim_pass": False},
    )

    assert binder.proof_chain_terminal_state(
        screen_root=tmp_path / "screen",
        confirmation_root=confirmation,
        controller_root=tmp_path / "controller",
        headline_root=tmp_path / "headline",
        not_before_epoch=50.0,
    ) == ("failed", "launcher_exited_without_fresh_proof_terminal")


@pytest.mark.parametrize("tamper", ["evidence", "sidecar", "stale_sidecar"])
def test_binder_rejects_invalid_new_tts_foundation_tree(tmp_path, tamper):
    binder = _binder_module()
    foundation = (
        tmp_path
        / "confirmation/tts-0-40k-foundation/TTS_0_40K_FOUNDATION.json"
    )
    evidence = _foundation_receipt(foundation, status="BLOCKED", passed=False)
    if tamper == "evidence":
        evidence.write_text('{"locked":false}\n', encoding="utf-8")
    elif tamper == "sidecar":
        Path(str(foundation) + ".sha256").write_text("0" * 64 + "\n")
    else:
        os.utime(Path(str(foundation) + ".sha256"), (25.0, 25.0))

    state, reason = binder.proof_chain_terminal_state(
        screen_root=tmp_path / "screen",
        confirmation_root=tmp_path / "confirmation",
        controller_root=tmp_path / "controller",
        headline_root=tmp_path / "headline",
        not_before_epoch=50.0,
    )
    assert state == "failed"
    assert reason == "tts_foundation_terminal_invalid"


@pytest.mark.parametrize(
    ("name", "status"),
    [
        ("CONTROLLER_SELECTED.json", "matched_controller_selected"),
        ("CONTROLLER_BLOCKED.json", "matched_controller_blocked"),
    ],
)
def test_binder_requires_controller_terminal_and_completed_v11_headline(
    tmp_path, name, status
):
    binder = _binder_module()
    screen = tmp_path / "screen"
    confirmation = tmp_path / "confirmation"
    controller = tmp_path / "controller"
    headline = tmp_path / "headline"
    _receipt(controller / name, {"status": status})

    assert binder.proof_chain_terminal_state(
        screen_root=screen,
        confirmation_root=confirmation,
        controller_root=controller,
        headline_root=headline,
        not_before_epoch=50.0,
    )[0] == "failed"

    controller.joinpath("queue-state.jsonl").write_text(
        json.dumps({"phase": "headline", "status": "complete"}) + "\n",
        encoding="utf-8",
    )
    os.utime(controller / "queue-state.jsonl", (100, 100))
    for coverage in (
        headline / "algorithmic-coverage.json",
        headline / "mfu-coverage.json",
    ):
        coverage.parent.mkdir(parents=True, exist_ok=True)
        coverage.write_text("{}\n", encoding="utf-8")
        os.utime(coverage, (100, 100))

    assert binder.proof_chain_terminal_state(
        screen_root=screen,
        confirmation_root=confirmation,
        controller_root=controller,
        headline_root=headline,
        not_before_epoch=50.0,
    ) == ("complete", "controller_terminal_and_headline_complete")


def test_binder_rejects_tampered_terminal_sidecar(tmp_path):
    binder = _binder_module()
    blocked = tmp_path / "screen" / "runs" / "CANDIDATE_SCREEN_BLOCKED.json"
    _receipt(blocked, {"status": "candidate_screen_blocked"})
    blocked.write_text('{"status":"candidate_screen_blocked","changed":true}\n')

    assert binder.proof_chain_terminal_state(
        screen_root=tmp_path / "screen",
        confirmation_root=tmp_path / "confirmation",
        controller_root=tmp_path / "controller",
        headline_root=tmp_path / "headline",
        not_before_epoch=50.0,
    )[0] == "failed"
