from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_industrial_executor import _execution_fixture
from test_telemetry_durability import (
    performance_record,
    request_record,
    run_record,
)

from lightcone_spec.telemetry import (
    DEFAULT_EVIDENCE_WRITER_POLICY,
    EvidenceWriter,
    EvidenceWriterPolicy,
    evidence_writer_policy_from_receipt,
    load_completed_evidence,
)


def test_registered_writer_policy_is_strict_and_changes_plan_identity(
    tmp_path: Path,
) -> None:
    policy = DEFAULT_EVIDENCE_WRITER_POLICY
    assert EvidenceWriterPolicy.from_dict(policy.to_dict()) == policy
    assert len(policy.sha256) == 64
    with pytest.raises(ValueError, match="fields differ"):
        EvidenceWriterPolicy.from_dict({**policy.to_dict(), "extra": 1})

    plan = _execution_fixture(tmp_path, request_count=1).plan
    changed = replace(policy, async_batch_rows=64)
    changed_plan = replace(plan, evidence_writer_policy=changed)
    changed_plan.validate()
    assert changed_plan.sha256 != plan.sha256
    assert changed_plan.to_dict()["evidence_writer_policy_sha256"] == changed.sha256


def test_terminal_receipt_binds_writer_policy_and_checkpoint(
    tmp_path: Path,
) -> None:
    policy = DEFAULT_EVIDENCE_WRITER_POLICY
    writer = EvidenceWriter(
        tmp_path,
        run_id="policy-run",
        rank=0,
        max_queued_rows=policy.writer_queue_rows,
        row_group_rows=policy.parquet_row_group_rows,
        checkpoint_interval_s=policy.checkpoint_interval_ms / 1000,
        overflow_policy=policy.overflow_policy,
        registered_policy=policy,
    )
    writer.write(run_record("policy-run", "target_only"))
    writer.write(request_record("policy-run", "target_only"))
    writer.write(performance_record("policy-run", "target_only"))
    written = writer.close()
    assert set(written) == {"performance", "request", "run"}

    receipt = tmp_path / "policy-run.rank0.complete.json"
    assert evidence_writer_policy_from_receipt(receipt) == policy
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["writer_policy"] == policy.to_dict()
    assert value["writer_policy_sha256"] == policy.sha256
    checkpoint = tmp_path / value["checkpoint"]["name"]
    checkpoint_value = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_value["writer_policy"] == policy.to_dict()
    assert checkpoint_value["writer_policy_sha256"] == policy.sha256
    assert load_completed_evidence(tmp_path, run_id="policy-run", rank=0)


def test_writer_rejects_settings_that_differ_from_registered_policy(
    tmp_path: Path,
) -> None:
    policy = DEFAULT_EVIDENCE_WRITER_POLICY
    with pytest.raises(ValueError, match="differ from the registered policy"):
        EvidenceWriter(
            tmp_path,
            run_id="policy-mismatch",
            rank=0,
            max_queued_rows=policy.writer_queue_rows,
            row_group_rows=policy.parquet_row_group_rows + 1,
            checkpoint_interval_s=policy.checkpoint_interval_ms / 1000,
            registered_policy=policy,
        )
