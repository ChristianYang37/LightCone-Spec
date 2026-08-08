"""Run directory layout and immutability (spec 11.1).

run_id/
  manifest.json manifest.sha256 environment.json lock-reference.json
  stdout.log stderr.log exit.json
  rounds.parquet updates.parquet decisions.parquet
  system_samples.parquet request_summary.parquet
  hashes.json

Once a run is marked complete it is immutable; re-runs use a fresh
run_id.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lightcone_spec.artifacts.schemas import TABLES
from lightcone_spec.exit_codes import ArtifactValidationFailure
from lightcone_spec.locking.hashing import canonical_json, sha256_bytes, sha256_file

TABLE_FILES = {name: f"{name}.parquet" for name in TABLES}
REQUIRED_FILES = (
    "manifest.json",
    "manifest.sha256",
    "environment.json",
    "lock-reference.json",
    "stdout.log",
    "stderr.log",
    "exit.json",
    "rounds.parquet",
    "updates.parquet",
    "decisions.parquet",
    "system_samples.parquet",
    "request_summary.parquet",
    "hashes.json",
)


class RunDirectory:
    def __init__(self, root: str | Path, run_id: str):
        self.run_id = run_id
        self.path = Path(root) / run_id
        self._complete_marker = self.path / "hashes.json"

    @property
    def is_complete(self) -> bool:
        """Whether the final completion ledger was published completely.

        Older writers briefly exposed ``hashes.json`` before adding conditional
        runtime/checkpoint hashes.  A crash in that window must be treated like
        an incomplete attempt, not like immutable completion evidence.  We only
        inspect ledger structure here; content/hash drift remains the
        validator's responsibility.
        """
        if not self._complete_marker.is_file():
            return False
        try:
            hashes = json.loads(self._complete_marker.read_text())
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(hashes, dict):
            return False
        normative = set(REQUIRED_FILES) - {self._complete_marker.name}
        if not normative.issubset(hashes):
            return False
        if not all(relative in hashes for relative in self._auxiliary_files()):
            return False
        return all(
            isinstance(relative, str)
            and isinstance(entry, dict)
            and isinstance(entry.get("sha256"), str)
            and len(entry["sha256"]) == 64
            and isinstance(entry.get("bytes"), int)
            and entry["bytes"] >= 0
            for relative, entry in hashes.items()
        )

    def _auxiliary_files(self) -> list[str]:
        paths = sorted((self.path / "runtime").glob("*.jsonl"))
        checkpoint = self.path / "prefix-checkpoints.json"
        if checkpoint.is_file():
            paths.append(checkpoint)
        return [path.relative_to(self.path).as_posix() for path in paths]

    def _require_mutable(self, action: str) -> None:
        # Existence is the publication boundary.  Even a torn legacy marker is
        # preserved as evidence and retried under a fresh run_id; no writer is
        # allowed to repair it in place.
        if self._complete_marker.exists():
            raise ArtifactValidationFailure(
                f"run {self.run_id} is immutable; cannot {action}"
            )

    def create(
        self,
        manifest: dict,
        lock_reference: dict | None = None,
        environment: dict | None = None,
    ) -> None:
        self._require_mutable("create; re-runs need a new run_id")
        self.path.mkdir(parents=True, exist_ok=True)
        body = canonical_json(manifest)
        (self.path / "manifest.json").write_text(body)
        (self.path / "manifest.sha256").write_text(
            sha256_bytes(body.encode("utf-8")) + "\n"
        )
        env = environment or {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        (self.path / "environment.json").write_text(json.dumps(env, indent=2))
        (self.path / "lock-reference.json").write_text(
            json.dumps(lock_reference or {"lockfile_sha256": None}, indent=2)
        )
        for name in ("stdout.log", "stderr.log"):
            (self.path / name).touch()

    def append_log(self, stream: str, text: str) -> None:
        self._require_mutable(f"append {stream}.log")
        with open(self.path / f"{stream}.log", "a") as f:
            f.write(text)

    def write_table(self, name: str, rows: list[dict]) -> None:
        self._require_mutable(f"write {name}")
        schema = TABLES[name]
        columns: dict[str, list] = {f.name: [] for f in schema}
        for row in rows:
            for f in schema:
                columns[f.name].append(row.get(f.name))
        table = pa.Table.from_pydict(columns, schema=schema)
        pq.write_table(table, self.path / TABLE_FILES[name])

    def finalize(self, exit_code: int, status: str, extra: dict | None = None) -> None:
        """Durably publish one immutable completion transaction.

        ``hashes.json`` is the final and only completion marker.  Every
        normative and conditional provenance file is closed/fsynced and bound
        before a temporary ledger is atomically renamed into place.  No run
        file is modified after that rename.
        """
        self._require_mutable("finalize")
        exit_path = self.path / "exit.json"
        with open(exit_path, "w") as handle:
            handle.write(
                json.dumps(
                    {"exit_code": exit_code, "status": status, **(extra or {})},
                    indent=2,
                )
            )
            handle.flush()
            os.fsync(handle.fileno())

        relative_paths = [
            name for name in REQUIRED_FILES if name != self._complete_marker.name
        ]
        relative_paths.extend(self._auxiliary_files())
        missing = [name for name in relative_paths if not (self.path / name).is_file()]
        if missing:
            raise ArtifactValidationFailure(
                f"run {self.run_id} cannot finalize; missing files: {missing}"
            )

        hashes = {}
        for name in relative_paths:
            path = self.path / name
            with open(path, "rb") as handle:
                os.fsync(handle.fileno())
            hashes[name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }

        # Persist all preceding file/directory entries before publishing the
        # ledger that makes them visible as a completed transaction.
        runtime_dir = self.path / "runtime"
        if runtime_dir.is_dir():
            _fsync_directory(runtime_dir)
        _fsync_directory(self.path)

        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path,
                prefix=".hashes.json.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(json.dumps(hashes, indent=2, sort_keys=True))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self._complete_marker)
            temporary_name = None
            _fsync_directory(self.path)
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass

    def read_table(self, name: str) -> pa.Table:
        return pq.read_table(self.path / TABLE_FILES[name])

    def read_manifest(self) -> dict:
        return json.loads((self.path / "manifest.json").read_text())

    def read_exit(self) -> dict:
        return json.loads((self.path / "exit.json").read_text())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
