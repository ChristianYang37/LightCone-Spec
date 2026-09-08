"""Scoped preview acceptance and exclusive, resumable server-local handoff."""

import fcntl
import json
import os
import sqlite3
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from .preview import preview_jobs
from .preview_revision import PREVIEW_V3_NODES


def group_rows(manifest, node):
    if node not in PREVIEW_V3_NODES:
        raise ValueError("unknown preview group")
    return [j.to_dict() for j in preview_jobs(manifest) if j.node == node]


def group_accepted(acceptance, manifest, node):
    """Legacy full-manifest evidence remains valid only for identical group rows."""
    if acceptance.get("trajectory_diagnosis") != "reviewed":
        return False
    scoped = acceptance.get("groups", {}).get(node)
    if scoped is not None:
        return scoped.get("status") == "accepted" and scoped.get("jobs") == group_rows(manifest, node)
    previous = acceptance.get("manifest")
    return (acceptance.get("nodes", {}).get(node) == "accepted"
            and isinstance(previous, dict)
            and group_rows(previous, node) == group_rows(manifest, node))


def replace_unstarted_group(state, manifest, candidate, node):
    """Only a not-yet-materialized group's topology/anchor may change."""
    if state.jobs(node):
        if group_rows(manifest, node) != group_rows(candidate, node):
            raise RuntimeError("cannot remap materialized preview group")
    for other in PREVIEW_V3_NODES:
        if other != node and group_rows(manifest, other) != group_rows(candidate, other):
            raise RuntimeError("candidate changed an unrelated preview group")
    state.set_selection("formal_preview_manifest_v3", deepcopy(candidate))


def read_state(run_dir):
    """Monitoring never constructs StateStore or recovers active attempts."""
    with sqlite3.connect(f"file:{Path(run_dir) / 'state.sqlite'}?mode=ro", uri=True) as db:
        selections = {name: json.loads(value) for name, value in
                      db.execute("select name,value_json from selections")}
        counts = {(node, status): count for node, status, count in db.execute(
            "select node,status,count(*) from jobs group by node,status")}
        active = db.execute("select count(*) from jobs where status='running'").fetchone()[0]
    return selections, counts, active


def first_group_complete(counts):
    node = PREVIEW_V3_NODES[0]
    return counts.get((node, "completed"), 0) == 40 and not any(
        count for (stage, status), count in counts.items() if stage == node and status != "completed")


@contextmanager
def continuation_lock(path):
    """A second controller fails immediately; children retain the parent's lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("preview continuation already owns the GPU window") from error
        try:
            yield lock.fileno()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def gpu_lease(run_dir):
    path = Path(run_dir) / "preview-continuation.lock"
    inherited = os.environ.get("LIGHTCONE_PREVIEW_LEASE_FD")
    if inherited is None:
        with continuation_lock(path):
            yield
    else:
        fd = int(inherited)
        if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (path.stat().st_dev, path.stat().st_ino):
            raise RuntimeError("invalid inherited preview GPU lease")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield  # the parent retains ownership; do not unlock its shared description
