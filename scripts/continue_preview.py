"""Server-local boundary controller: remaining QA, formal groups, real videos, DAG.

Prestage from a CI-green detached checkout. Never edits an active runner's checkout.
All children run synchronously and inherit the exclusive controller lock.
"""

import argparse
import gzip
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import tarfile
import time
import urllib.request
from copy import deepcopy
from pathlib import Path

from validate_remaining_preview import remaining_cases, review_case

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.preview import QWEN38_MODEL
from lightcone_spec.preview_continuation import (
    continuation_lock,
    first_group_complete,
    group_accepted,
    group_rows,
    read_state,
    replace_unstarted_group,
)
from lightcone_spec.state import StateStore


def gpu_owners():
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = path.read_bytes().split(b"\0")
        except OSError:
            continue
        if b"sglang.launch_server" in args or any(
            Path(arg.decode(errors="replace")).name in {
                "lightcone-spec", "validate_preview_group.py", "validate_preview_updates.py",
                "validate_remaining_preview.py", "record_preview.py"} for arg in args if arg
        ):
            found.append(int(path.parent.name))
    return found


class Controller:
    def __init__(self, args, lock_fd):
        self.args, self.lock_fd = args, lock_fd
        self.config = ExperimentConfig.load(args.config)
        self.out = args.output
        self.env = {**os.environ, "PYTHONPATH": str(args.repo / "src"), "LIGHTCONE_NUMA_ISOLATION": "1",
                    "LIGHTCONE_PREVIEW_LEASE_FD": str(lock_fd)}
        if args.recording_tools:
            environment = json.loads((args.recording_tools / "environment.json").read_text())
            self.env.update({key: environment[key] for key in ("PATH", "NODE_PATH", "PLAYWRIGHT_BROWSERS_PATH")})

    def status(self, phase, **details):
        path = self.out / "status.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({"phase": phase, "time": time.time(), "pid": os.getpid(), **details}, indent=2))
        temp.replace(path)

    def check_boundary(self):
        _, _, active = read_state(self.config.run_dir)
        if active or gpu_owners():
            raise RuntimeError("GPU window is not exclusive")
        compute = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        if compute:
            raise RuntimeError("unowned GPU compute process remains")
        disk = os.statvfs(self.config.run_dir)
        if disk.f_bavail * disk.f_frsize < 12884901888:
            raise RuntimeError("disk below 12 GiB; archive before new work")
        with sqlite3.connect(f"file:{self.config.run_dir / 'state.sqlite'}?mode=ro", uri=True) as db:
            if db.execute("pragma integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("formal SQLite integrity failure")

    def backup(self, label):
        backup = self.out / f"{label}.sqlite"
        if backup.exists():
            return
        with sqlite3.connect(f"file:{self.config.run_dir / 'state.sqlite'}?mode=ro", uri=True) as src, sqlite3.connect(backup) as dst:
            src.backup(dst)
            if dst.execute("pragma integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("backup integrity failure")
            latest = src.execute("select output_dir from attempts order by started_at desc limit 3").fetchall()
        with tarfile.open(self.out / f"{label}.tar.gz", "x:gz", compresslevel=1) as archive:
            archive.add(backup, arcname=backup.name)
            for i, (directory,) in enumerate(latest):
                path = Path(directory)
                if path.is_dir() and path.is_relative_to(self.config.run_dir):
                    archive.add(path, arcname=f"boundary-attempt-{i}")
            for i, path in enumerate(sorted(self.out.glob("*.log"))[-5:]):
                archive.add(path, arcname=f"controller-log-{i}.log")

    def safe_power_off(self, token):
        if not token or not self.args.power_off_instance:
            return
        # Never power off a healthy sibling cell. Existing workers drain first.
        while gpu_owners():
            time.sleep(5)
        self.backup(f"error-boundary-{time.time_ns()}")
        request = urllib.request.Request("https://api.autodl.com/api/v1/dev/instance/pro/power_off",
            data=json.dumps({"instance_uuid": self.args.power_off_instance}).encode(),
            headers={"Authorization": token, "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            code = json.load(response).get("code")
        if code != "Success":
            raise RuntimeError("AutoDL power-off did not confirm Success")

    def child(self, command, name, *, capacity_result=None):
        self.check_boundary()
        self.status("running", work=name)
        log = self.out / f"{name}-{time.time_ns()}.log"
        with log.open("x") as stream:
            child = subprocess.Popen(command, cwd=self.args.repo, env=self.env,
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     pass_fds=(self.lock_fd,))
            try:
                code = child.wait()
            except BaseException:
                child.send_signal(signal.SIGINT)
                child.wait()  # the runner finishes its active cell; never SIGKILL
                raise
        if code:
            # Failed processes are never silently retried. Only allocator/KV
            # admission evidence permits the registered common-TP2 candidate.
            message = log.read_text(errors="replace")
            if capacity_result is not None and re.search(
                r"CUDA out of memory|torch\.OutOfMemoryError|leave no GPU memory for the KV cache", message
            ):
                self.check_boundary()
                capacity_result.write_text(json.dumps({"status": "capacity_infeasible", "log": str(log)}))
                return False
            raise RuntimeError(f"child failed ({code}): {name}; evidence in {log}")
        self.check_boundary()
        return True

    def qa(self, manifest, node, index, tp, anchor=None):
        path = self.out / f"qa-{node}-tp{tp}-{index}{'-'+anchor if anchor else ''}"
        frozen = self.out / f"manifest-{node}-tp{tp}.json"
        if frozen.exists() and json.loads(frozen.read_text()) != manifest:
            raise RuntimeError("QA manifest changed; use a new audited controller output")
        if not frozen.exists():
            frozen.write_text(json.dumps(manifest, indent=2))
        if not path.exists():
            command = [sys.executable, str(self.args.repo / "scripts/validate_remaining_preview.py"),
                       "--config", str(self.args.config), "--manifest", str(frozen), "--node", node,
                       "--case", str(index), "--output", str(path)]
            if anchor:
                command.extend(("--anchor", anchor))
            self.child(command, path.name)
        result = json.loads((path / "result.json").read_text())
        # Re-read raw metrics on resume, not only a status label.
        if result["status"] == "passed":
            from lightcone_spec.protocol import Job
            metric_path = Path(result["metrics_path"])
            metrics = json.loads(metric_path.read_text())
            with gzip.open(metric_path.with_name("requests.jsonl.gz"), "rt") as stream:
                requests = [json.loads(line) for line in stream if line.strip()]
            expected = remaining_cases(manifest, node)[index]
            if not anchor and result["job"] != expected.to_dict():
                raise RuntimeError("QA evidence/config mismatch")
            if review_case(metrics, requests, Job(**result["job"])) != "passed":
                raise RuntimeError("QA raw evidence no longer passes")
        elif result["status"] != "capacity_infeasible":
            raise RuntimeError("unreviewed QA failure")
        return result

    def tp2_anchor(self, manifest):
        results = []
        for i, job in enumerate(remaining_cases(manifest, "E5-preview-v3")):
            if job.method != "static" or not job.load.startswith("closed_loop_c"):
                continue
            for method in ("target_only", "static"):
                results.append(self.qa(manifest, "E5-preview-v3", i, 1, anchor=method))
        feasible = [r for r in results if r["status"] == "passed"]
        if {r["job"]["method"] for r in feasible} != {"target_only", "static"}:
            raise RuntimeError("TP2 trace lacks feasible measured Target/Static anchors")
        scored = [(json.loads(Path(r["metrics_path"]).read_text())["request_rate"], r) for r in feasible]
        rate, best = max(scored, key=lambda pair: pair[0])
        return {"topology": "tp2_dp1", "request_rate": rate,
                "concurrency": int(best["job"]["load"].removeprefix("closed_loop_c")),
                "source_job_ids": [r["job"]["job_id"] for r in feasible],
                "evidence": feasible, "scope": "supplemental matched TP2 anchor; outside 96 cells"}

    def group(self, state, node):
        manifest = state.selection("formal_preview_manifest_v3")
        acceptance = state.selection("formal_preview_acceptance_v3", {})
        if not group_accepted(acceptance, manifest, node):
            for tp in (1, 2):
                candidate = deepcopy(manifest)
                if node == "E5-preview-v3":
                    if tp == 2:
                        candidate["tp2_trace_anchor"] = self.tp2_anchor(manifest)
                    candidate.setdefault("comparison_topologies", {})["dspark_serving"] = tp
                else:
                    candidate["qwen38"]["tp"] = tp
                proof = []
                for i, _ in enumerate(remaining_cases(candidate, node)):
                    row = self.qa(candidate, node, i, tp)
                    proof.append(row)
                    if row["status"] == "capacity_infeasible":
                        break
                if len(proof) == len(remaining_cases(candidate, node)) and all(r["status"] == "passed" for r in proof):
                    replace_unstarted_group(state, manifest, candidate, node)
                    acceptance = deepcopy(acceptance)
                    acceptance.setdefault("groups", {})[node] = {
                        "status": "accepted", "jobs": group_rows(candidate, node), "evidence": proof,
                        "commit": self.args.commit, "runtime_marker": self.args.marker}
                    state.set_selection("formal_preview_acceptance_v3", acceptance)
                    break
            else:
                self.status("capacity_unavailable", node=node)
                state.set_selection(f"preview_capacity_{node}", {"status": "unavailable", "evidence_directory": str(self.out)})
                return False
        self.child([str(Path(sys.executable).with_name("lightcone-spec")), "run", "--config", str(self.args.config)], node)
        counts = state.status_counts(node)
        if counts.get("completed", 0) != len(group_rows(state.selection("formal_preview_manifest_v3"), node)):
            raise RuntimeError(f"group did not finish: {node}: {counts}")
        return True

    def videos(self, state):
        for model_index, model in enumerate(("Qwen/Qwen3-8B", QWEN38_MODEL)):
            accepted = None
            for tp in (1, 2):
                proofs = []
                for i in range(6):
                    path = self.out / f"video-qa-{model_index}-tp{tp}-{i}"
                    capacity = self.out / f"video-qa-{model_index}-tp{tp}-{i}-capacity.json"
                    if capacity.exists():
                        break
                    if not path.exists():
                        if not self.child(self.record_command(model, tp, i, path) + ["--qa"], path.name,
                                          capacity_result=capacity):
                            break
                    row = json.loads((path / "recording.json").read_text())
                    if row.get("status") != "completed" or row.get("event_accounting") != "verified_native_final_records":
                        raise RuntimeError("video QA failure; keep failed take and diagnose")
                    proofs.append(row)
                if len(proofs) != 6:
                    continue
                accepted = {"status": "accepted", "tp": tp, "evidence": proofs,
                            "manifest": state.selection("formal_preview_manifest_v3")}
                break
            if accepted is None:
                raise RuntimeError("no common video topology")
            all_acceptance = state.selection("formal_preview_video_acceptance_v3", {})
            all_acceptance[model] = accepted
            state.set_selection("formal_preview_video_acceptance_v3", all_acceptance)
            takes = []
            for i in range(6):
                path = self.out / f"video-{model_index}-{i}"
                capture = self.out / f"capture-{model_index}-{i}"
                if not capture.exists():
                    self.check_boundary()
                    with (self.out / f"video-{model_index}-{i}.log").open("x") as log:
                        server = subprocess.Popen(self.record_command(model, accepted["tp"], i, path),
                            cwd=self.args.repo, env=self.env, stdout=log, stderr=subprocess.STDOUT,
                            pass_fds=(self.lock_fd,))
                        try:
                            self.wait_recording(server)
                            subprocess.run(["node", str(self.args.repo / "scripts/capture_preview.cjs"),
                                            "http://127.0.0.1:8765", str(capture)], check=True,
                                           cwd=self.args.repo, env=self.env, pass_fds=(self.lock_fd,))
                        finally:
                            server.send_signal(signal.SIGINT)
                            server.wait()
                    self.check_boundary()
                if json.loads((capture / "capture.json").read_text())["status"] != "completed":
                    raise RuntimeError("failed video take retained; no automatic cherry-picked re-recording")
                takes.append(str(capture / "original.mp4"))
            combined = self.out / f"combined-{model_index}.mp4"
            if not combined.exists():
                self.child([sys.executable, str(self.args.repo / "scripts/compose_preview.py"),
                            "--output", str(combined), "--inputs", *takes], f"compose-{model_index}")
        value = state.selection("formal_preview_continuation_v1")
        state.set_selection("formal_preview_continuation_v1", {**value, "videos": "completed"})

    def record_command(self, model, tp, index, output):
        return [sys.executable, str(self.args.repo / "scripts/record_preview.py"), "--config", str(self.args.config),
                "--model", model, "--tp", str(tp), "--gpu", str(self.config.gpu_ids[0]),
                "--method-index", str(index), "--output", str(output)]

    def wait_recording(self, server):
        deadline = time.monotonic() + self.config.server.startup_timeout_seconds + 300
        while server.poll() is None and time.monotonic() < deadline:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8765/status", timeout=2) as response:
                    if json.load(response)["status"] == "ready":
                        return
            except OSError:
                pass
            time.sleep(1)
        raise RuntimeError("recording service failed to become ready")

    def run(self):
        while True:
            selections, counts, active = read_state(self.config.run_dir)
            from lightcone_spec.preview_revision import preview_lightcone_stride
            if preview_lightcone_stride(selections.get("formal_preview_manifest_v3", {})) != 10:
                raise RuntimeError("restore preview S10 at idle boundary and revalidate first40 before continuation")
            if counts.get(("E3b-preview-v3", "failed"), 0):
                raise RuntimeError("first forty failed; diagnose without replacing active work")
            if first_group_complete(counts) and not active and not gpu_owners():
                break
            self.status("waiting_first40_boundary", completed=counts.get(("E3b-preview-v3", "completed"), 0))
            time.sleep(5)
        self.check_boundary()
        self.backup("before-continuation")
        if (self.config.sglang_root / ".lightcone-spec-patched").read_text().strip() != self.args.marker:
            raise RuntimeError("runtime marker changed")
        subprocess.run(["git", "merge", "--ff-only", self.args.commit], cwd=self.args.repo, check=True)
        state = StateStore(self.config.run_dir)
        previous = state.selection("formal_preview_continuation_v1", {})
        state.set_selection("formal_preview_continuation_v1", {**previous, "enabled": True, "commit": self.args.commit})
        ready = [self.group(state, node) for node in ("E5-preview-v3", "Qwen38-preview-v3")]
        if not all(ready):
            raise RuntimeError("remaining group capacity unavailable; review retained evidence")
        self.videos(state)
        self.status("resuming_complete_dag")
        self.child([str(Path(sys.executable).with_name("lightcone-spec")), "run", "--config", str(self.args.config)], "full-dag")
        self.status("completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "repo", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--recording-tools", type=Path)
    parser.add_argument("--power-off-instance", help="Explicitly authorized AutoDL instance; token supplied once over stdin, never persisted")
    args = parser.parse_args()
    token = sys.stdin.readline().strip() if args.power_off_instance else None
    if args.power_off_instance and not token:
        raise RuntimeError("missing in-memory power-off authorization")
    args.output.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig.load(args.config)
    with continuation_lock(config.run_dir / "preview-continuation.lock") as fd:
        controller = Controller(args, fd)
        try:
            controller.run()
        except BaseException as error:
            controller.status("stopped_error", error=f"{type(error).__name__}: {error}")
            try:
                controller.safe_power_off(token)
            except Exception:
                controller.status("power_off_unconfirmed", original_error=type(error).__name__)
            raise


if __name__ == "__main__":
    main()
