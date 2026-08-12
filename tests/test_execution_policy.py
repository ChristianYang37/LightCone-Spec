from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from lightcone_spec.config.schema import ModelPair, RunConfig, RuntimeConfig
from lightcone_spec.execution import ControlledExecutionPolicy
from lightcone_spec.orchestration.executor import (
    _require_live_controlled_execution_policy,
)
from lightcone_spec.orchestration.runtime import _execution_argv

POLICY_SHA256 = "231ca57941f96b2cd1593f360137aa005bccc8145296c2f06d2a13cd23c02d2b"


def test_registered_policy_identity_and_tracked_bytes() -> None:
    policy = ControlledExecutionPolicy()
    source = Path("manifests/speed-study/execution_policy_v2.json").resolve()
    assert policy.sha256 == POLICY_SHA256
    assert ControlledExecutionPolicy.load(source) == policy
    assert (
        source.with_name(f"{source.name}.sha256").read_text().strip() == policy.sha256
    )


def test_role_policy_differs_only_at_overlap_schedule() -> None:
    policy = ControlledExecutionPolicy()
    target = policy.server_info_fields(role="target_reference")
    speculative = policy.server_info_fields(role="speculative")
    assert {name for name in target if target[name] != speculative[name]} == {
        "disable_overlap_schedule"
    }
    assert target["disable_overlap_schedule"] is True
    assert speculative["disable_overlap_schedule"] is False


def test_policy_publication_is_immutable_and_rejects_leaf_aliases(
    tmp_path: Path,
) -> None:
    policy = ControlledExecutionPolicy()
    path = tmp_path / "execution-policy.json"
    policy.write(path)
    policy.write(path)
    alias = tmp_path / "execution-policy-alias.json"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="regular file"):
        ControlledExecutionPolicy.load(alias)
    with pytest.raises(ValueError, match="regular file"):
        policy.write(alias)

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        policy.write(path)


def test_policy_loader_rejects_noncanonical_or_mismatched_sidecar(
    tmp_path: Path,
) -> None:
    policy = ControlledExecutionPolicy()
    path = tmp_path / "execution-policy.json"
    path.write_text(json.dumps(policy.to_dict(), indent=2) + "\n", encoding="utf-8")
    Path(f"{path}.sha256").write_text(f"{policy.sha256}\n", encoding="ascii")
    with pytest.raises(ValueError, match="not canonical"):
        ControlledExecutionPolicy.load(path)

    canonical = tmp_path / "canonical.json"
    policy.write(canonical)
    Path(f"{canonical}.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="sidecar"):
        ControlledExecutionPolicy.load(canonical)


def test_runtime_schema_and_role_argv_are_exact() -> None:
    runtime = RuntimeConfig(sampling_profile_sha256="a" * 64)
    common = [
        "--context-length",
        "40960",
        "--random-seed",
        "1",
        "--disable-radix-cache",
        "--disable-cuda-graph",
    ]
    assert _execution_argv(runtime, role="speculative") == common
    assert _execution_argv(runtime, role="target_reference") == [
        *common,
        "--disable-overlap-schedule",
    ]
    with pytest.raises(ValidationError, match="execution-policy identity"):
        RuntimeConfig(
            sampling_profile_sha256="a" * 64,
            execution_policy_sha256="b" * 64,
        )


def test_live_policy_rejects_role_swap_and_missing_controls() -> None:
    policy = ControlledExecutionPolicy()
    target = policy.server_info_fields(role="target_reference")
    with pytest.raises(ValueError, match="disable_overlap_schedule"):
        policy.validate_server_info(target, role="speculative")
    for field in ("disable_radix_cache", "disable_cuda_graph", "random_seed"):
        mismatched = dict(target)
        mismatched.pop(field)
        with pytest.raises(ValueError, match=field):
            policy.validate_server_info(mismatched, role="target_reference")


def test_live_policy_uses_server_info_and_rejects_role_swap() -> None:
    config = RunConfig(
        method="target_only",
        model=ModelPair(
            target="test/target",
            drafter="test/drafter",
            target_revision="a" * 40,
            drafter_revision="b" * 40,
            algorithm="DFLASH",
            max_context_length=1024,
            draft_depth=7,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256="c" * 64,
            speculation_enabled=False,
        ),
        adaptation=None,
        online_spec=None,
        tenant_id="execution-policy-test",
    )

    class Transport:
        def __init__(self, value: dict[str, int | bool]) -> None:
            self.value = value
            self.paths: list[str] = []

        async def get_json(self, path: str) -> object:
            self.paths.append(path)
            return self.value

    policy = ControlledExecutionPolicy()
    transport = Transport(policy.server_info_fields(role="target_reference"))
    assert (
        asyncio.run(
            _require_live_controlled_execution_policy(
                transport=transport,  # type: ignore[arg-type]
                config=config,
            )
        )
        == policy.sha256
    )
    assert transport.paths == ["/server_info"]

    swapped = Transport(policy.server_info_fields(role="speculative"))
    with pytest.raises(RuntimeError, match="registered controlled execution policy"):
        asyncio.run(
            _require_live_controlled_execution_policy(
                transport=swapped,  # type: ignore[arg-type]
                config=config,
            )
        )


def test_policy_default_identity_does_not_hide_unregistered_variant() -> None:
    with pytest.raises(ValueError, match="CUDA graph"):
        replace(ControlledExecutionPolicy(), disable_cuda_graph=False).validate()
    with pytest.raises(ValueError, match="overlap"):
        replace(
            ControlledExecutionPolicy(),
            speculative_disable_overlap_schedule=True,
        ).validate()
