"""Sequence-grouped deterministic splits (spec 13.2, 14.6).

train/calibration/test = 60/20/20 by deterministic hash of the base
prompt (sequence group), seed 0. Multiple sampled trajectories of one
base prompt always stay in the same group/cluster; leave-one-task-out
splits reuse the same grouping.
"""

from __future__ import annotations

from lightcone_spec.locking.hashing import stable_hash_int

SPLIT_NAMES = ("train", "calibration", "test")


def split_of_group(group_id: str, seed: int = 0) -> str:
    """Deterministic 60/20/20 assignment by hash bucket."""
    bucket = stable_hash_int(f"split/{seed}/{group_id}") % 100
    if bucket < 60:
        return "train"
    if bucket < 80:
        return "calibration"
    return "test"


def split_groups(
    group_ids: list[str], seed: int = 0
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for gid in group_ids:
        out[split_of_group(gid, seed)].append(gid)
    return out


def assert_no_group_leakage(assignments: dict[str, str]) -> None:
    """assignments: item_id -> split, where item_id = f\"{group}::{sub}\".
    Every item of a group must share the group's split."""
    seen: dict[str, str] = {}
    for item_id, split in assignments.items():
        group = item_id.split("::", 1)[0]
        if group in seen and seen[group] != split:
            raise AssertionError(
                f"group {group} leaks across splits: {seen[group]} vs {split}"
            )
        seen[group] = split


def leave_one_task_out(
    tasks: list[str], held_out: str
) -> tuple[list[str], list[str]]:
    train = [t for t in tasks if t != held_out]
    if held_out not in tasks:
        raise KeyError(f"held-out task {held_out!r} not in {tasks}")
    return train, [held_out]
