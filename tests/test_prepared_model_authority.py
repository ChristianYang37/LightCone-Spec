from __future__ import annotations

from pathlib import Path

import pytest

from lightcone_spec.locking import (
    ModelLock,
    PreparedModelSet,
    bind_prepared_models,
    revalidate_prepared_models,
)
from lightcone_spec.locking.models import LockedModel

TARGET = "1" * 40
DRAFTER = "2" * 40


def _fixture(tmp_path: Path) -> tuple[ModelLock, dict[str, str]]:
    lock = ModelLock(
        schema_version=2,
        models=(
            LockedModel(model_id="target", revision=TARGET),
            LockedModel(model_id="drafter", revision=DRAFTER),
        ),
    )
    roots: dict[str, str] = {}
    for model_id, revision in (("target", TARGET), ("drafter", DRAFTER)):
        root = tmp_path / model_id / "snapshots" / revision
        root.mkdir(parents=True)
        roots[model_id] = str(root.resolve())
    return lock, roots


def test_prepared_models_bind_exact_revision_snapshots(tmp_path: Path) -> None:
    lock, roots = _fixture(tmp_path)
    prepared = bind_prepared_models(lock, roots)

    assert PreparedModelSet.from_dict(prepared.to_dict()) == prepared
    assert revalidate_prepared_models(lock, prepared) == roots


def test_prepared_models_reject_swapped_or_non_snapshot_roots(tmp_path: Path) -> None:
    lock, roots = _fixture(tmp_path)

    with pytest.raises(ValueError, match="locked revision snapshot"):
        bind_prepared_models(
            lock,
            {"target": roots["drafter"], "drafter": roots["target"]},
        )

    arbitrary = tmp_path / "arbitrary"
    arbitrary.mkdir()
    with pytest.raises(ValueError, match="locked revision snapshot"):
        bind_prepared_models(lock, {"target": arbitrary, "drafter": roots["drafter"]})


def test_prepared_models_reject_symlink_and_coordinated_rehash(tmp_path: Path) -> None:
    lock, roots = _fixture(tmp_path)
    prepared = bind_prepared_models(lock, roots)
    alias = tmp_path / "target-alias"
    alias.symlink_to(Path(roots["target"]), target_is_directory=True)

    with pytest.raises(ValueError, match="locked revision snapshot"):
        bind_prepared_models(lock, {"target": alias, "drafter": roots["drafter"]})

    forged_lock = ModelLock(
        schema_version=2,
        models=(
            LockedModel(model_id="target", revision="3" * 40),
            LockedModel(model_id="drafter", revision=DRAFTER),
        ),
    )
    forged = PreparedModelSet(
        schema_version=prepared.schema_version,
        kind=prepared.kind,
        model_lock_sha256=forged_lock.sha256,
        snapshots=prepared.snapshots,
        protocol_sha256=prepared.protocol_sha256,
    )
    with pytest.raises(ValueError, match="revisions differ"):
        revalidate_prepared_models(forged_lock, forged)
