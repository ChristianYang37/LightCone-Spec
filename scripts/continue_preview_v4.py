"""Exclusive server-local v4 controller. Never resumes the complete DAG."""

import argparse
import json
import signal
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

from continue_preview import Controller
from validate_remaining_preview import remaining_cases

from lightcone_spec.config import ExperimentConfig
from lightcone_spec.preview import QWEN38_MODEL
from lightcone_spec.preview_continuation import continuation_lock
from lightcone_spec.preview_v4 import NODES, group_accepted, group_digest, jobs, video_job
from lightcone_spec.state import StateStore


class V4Controller(Controller):
    def set_topology(self, state, node, tp):
        old = state.selection("formal_preview_manifest_v4")
        candidate = deepcopy(old)
        if node == NODES[3]:
            candidate["qwen38"]["tp"] = tp
        else:
            key = {NODES[0]: "cohort", NODES[1]: "dflash_long", NODES[2]: "dspark_serving"}[node]
            if node == NODES[2] and tp == 2 and not candidate.get("tp2_trace_anchor", {}).get("preview_v4_verified"):
                candidate["tp2_trace_anchor"] = self.tp2_anchor_v4(old)
            candidate.setdefault("comparison_topologies", {})[key] = tp
        # DFlash/DSpark/27B may only change their own not-yet-started comparison.
        for other in NODES:
            if (other != node or state.jobs(other)) and group_digest(old, other) != group_digest(candidate, other):
                raise RuntimeError("topology change would alter an unrelated/started comparison")
        state.set_selection("formal_preview_manifest_v4", candidate)
        return candidate

    def tp2_anchor_v4(self, manifest):
        results = []
        for index, job in enumerate(remaining_cases(manifest, NODES[2])):
            if job.method == "static" and job.load.startswith("closed_loop_c"):
                for method in ("target_only", "static"):
                    results.append(self.qa(manifest, NODES[2], index, 1, anchor=method))
        feasible = [r for r in results if r["status"] == "passed"]
        if {r["job"]["method"] for r in feasible} != {"target_only", "static"}:
            raise RuntimeError("no feasible matching TP2 anchors; capacity evidence needs review")
        rate, best = max((json.loads(Path(r["metrics_path"]).read_text())["request_rate"], i)
                         for i, r in enumerate(feasible))
        return {"topology": "tp2_dp1", "request_rate": rate,
                "concurrency": int(feasible[best]["job"]["load"].removeprefix("closed_loop_c")),
                "source_job_ids": [r["job"]["job_id"] for r in feasible], "evidence": feasible,
                "preview_v4_verified": True, "marker": self.args.marker,
                "scope": "supplementary TP2 anchors; excluded from 180; arrivals unchanged"}

    def group(self, state, node):
        manifest = state.selection("formal_preview_manifest_v4")
        acceptance = state.selection("formal_preview_acceptance_v4", {})
        receipt = acceptance.get("groups", {}).get(node, {})
        terminal_capacity = (receipt.get("status") == "capacity_infeasible"
                             and receipt.get("group_sha256") == group_digest(manifest, node)
                             and bool(receipt.get("qa_paths")))
        if group_accepted(acceptance, manifest, node) or terminal_capacity:
            if receipt.get("runtime_commit") != self.args.commit or receipt.get("marker") != self.args.marker:
                raise RuntimeError("accepted group runtime differs; review reuse before execution")
            return
        paths, accepted = [], False
        for tp in (1, 2):
            try:
                manifest = self.set_topology(state, node, tp)
            except ValueError as error:
                # Missing matched trace anchors are dependencies, never fabricated.
                raise RuntimeError(f"{node}: missing common-topology dependency: {error}") from error
            frozen = self.out / f"{node}-tp{tp}-manifest.json"
            if frozen.exists() and json.loads(frozen.read_text()) != manifest:
                raise RuntimeError("QA manifest changed since the previous controller attempt")
            if not frozen.exists():
                frozen.write_text(json.dumps(manifest, indent=2))
            passed = True
            cases = remaining_cases(manifest, node)
            self.status("qa_budget", node=node, tp=tp, full_condition_cells=len(cases), excluded=True)
            for index, source in enumerate(cases):
                path = self.out / f"qa-{node}-tp{tp}-{index}"
                if not path.exists():
                    self.child([sys.executable, str(self.args.repo / "scripts/validate_remaining_preview.py"),
                        "--config", str(self.args.config), "--manifest", str(frozen), "--node", node,
                        "--case", str(index), "--output", str(path)], path.name)
                result_path = path / "result.json"
                if not result_path.is_file():
                    raise RuntimeError("interrupted/failed QA retained; diagnosis required before retry")
                result = json.loads(result_path.read_text())
                if result.get("job") != source.to_dict():
                    raise RuntimeError("QA result is from another frozen configuration")
                paths.append(str(result_path))
                if result["status"] == "capacity_infeasible":
                    passed = False
                    break
                if (result.get("status") != "passed" or not result.get("gpu_binding_verified")
                        or not result.get("full_workload") or not result.get("lifecycle", {}).get("state_lifecycle")
                        or not result.get("lifecycle", {}).get("parameter_optimizer_reset")):
                    raise RuntimeError("v4 QA lacks full-condition/lifecycle/tensor evidence")
            if passed:
                accepted = True
                break
        receipt = {"status": "accepted" if accepted else "capacity_infeasible",
                   "group_sha256": group_digest(manifest, node), "runtime_commit": self.args.commit,
                   "marker": self.args.marker, "qa_paths": paths,
                   "checks": {key: accepted for key in ("sampling", "parameter_optimizer_reset", "state_lifecycle",
                                                          "kv_isolation", "tp_ranks", "full_budget")},
                   "coverage_note": "sampled complete requests, native trajectory counts and inherited exactness gates; endpoint adapter tensor reset and distinct request ownership; not a distributional equivalence proof"}
        acceptance.setdefault("groups", {})[node] = receipt
        state.set_selection("formal_preview_acceptance_v4", acceptance)

    def record_command_v4(self, model, tp, index, output, cohort):
        return [*self.record_command(model, tp, index, output), "--preview-version", "4", *(["--cohort"] if cohort else [])]

    def videos(self, state):
        if not self.args.recording_tools:
            raise RuntimeError("recording environment must be prepared before the GPU window")
        manifest = state.selection("formal_preview_manifest_v4")
        report = json.loads((self.config.run_dir / "stages/preview-v4/preview.json").read_text())
        for number, (key, model, count, cohort) in enumerate((
            ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B", 6, False),
            (QWEN38_MODEL, QWEN38_MODEL, 6, False),
            ("cohort_appendix", "Qwen/Qwen3-8B", 2, True))):
            if key not in report["selected_scenes"]:
                self.status("video_unavailable", panel=key, reason="no complete safe four-block scene")
                continue
            accepted = None
            topologies = (int(report["selected_scenes"][key]["candidates"][0]["topology"][2]),) if cohort else (1, 2)
            for tp in topologies:
                paths = []
                for index in range(count):
                    path = self.out / f"video-qa-{number}-tp{tp}-{index}"
                    capacity = self.out / f"video-qa-{number}-tp{tp}-{index}-capacity.json"
                    if capacity.exists():
                        break
                    if not path.exists() and not self.child(
                        self.record_command_v4(model, tp, index, path, cohort) + ["--qa"], path.name,
                        capacity_result=capacity):
                        break
                    row = json.loads((path / "recording.json").read_text())
                    if row.get("status") != "completed" or row.get("event_accounting") != "verified_native_final_records":
                        raise RuntimeError("video QA failed; retain this technical failure")
                    paths.append(str(path))
                if len(paths) == count:
                    accepted = {"status": "accepted", "tp": tp, "manifest": manifest, "qa_paths": paths,
                                "jobs": [video_job(manifest, report, model, i, tp, cohort=cohort).to_dict() for i in range(count)]}
                    break
            if accepted is None:
                self.status("video_unavailable", panel=key, reason="no common accepted display topology")
                continue
            video_acceptance = state.selection("formal_preview_video_acceptance_v4", {})
            video_acceptance[key] = accepted
            state.set_selection("formal_preview_video_acceptance_v4", video_acceptance)
            takes = []
            for index in range(count):
                path, capture = self.out / f"video-{number}-{index}", self.out / f"capture-{number}-{index}"
                if not capture.exists():
                    if path.exists():
                        raise RuntimeError("previous recording without capture receipt; preserve and review technical retry")
                    self.check_boundary()
                    with (self.out / f"video-{number}-{index}.log").open("x") as log:
                        server = subprocess.Popen(self.record_command_v4(model, accepted["tp"], index, path, cohort),
                            cwd=self.args.repo, env=self.env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(self.lock_fd,))
                        try:
                            self.wait_recording(server)
                            subprocess.run(["node", str(self.args.repo / "scripts/capture_preview.cjs"),
                                "http://127.0.0.1:8765", str(capture)], check=True, env=self.env,
                                cwd=self.args.repo, pass_fds=(self.lock_fd,))
                        finally:
                            if server.poll() is None:
                                server.send_signal(signal.SIGINT)
                            server.wait()
                if json.loads((capture / "capture.json").read_text()).get("status") != "completed":
                    raise RuntimeError("failed take retained; performance is never a retry reason")
                takes.append(str(capture / "original.mp4"))
            combined = self.out / f"combined-{number}.mp4"
            if not combined.exists():
                self.child([sys.executable, str(self.args.repo / "scripts/compose_preview.py"),
                    "--output", str(combined), "--inputs", *takes], f"compose-{number}")

    def run(self):
        self.check_boundary()
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.args.repo, text=True).strip() != self.args.commit:
            raise RuntimeError("deploy the CI-green exact commit before starting this controller")
        if (self.config.sglang_root / ".lightcone-spec-patched").read_text().strip() != self.args.marker:
            raise RuntimeError("runtime marker differs from deployed acceptance target")
        state = StateStore(self.config.run_dir)
        context = {"commit": self.args.commit, "marker": self.args.marker, "config": self.config.normalized()}
        context_path = self.out / "controller-context.json"
        if context_path.exists() and json.loads(context_path.read_text()) != context:
            raise RuntimeError("controller context changed; do not reuse incompatible QA")
        if not context_path.exists():
            context_path.write_text(json.dumps(context, indent=2))
        self.backup("before-v4")
        candidate = json.loads(self.args.manifest.read_text())
        jobs(candidate)
        existing = state.selection("formal_preview_manifest_v4", None)
        if existing is None:
            state.set_selection("formal_preview_manifest_v4", candidate)
        else:
            # Only controller-selected common topology and its matching anchor
            # can diverge from the immutable preparation manifest on resume.
            for item in (existing := deepcopy(existing), candidate := deepcopy(candidate)):
                item.pop("comparison_topologies", None)
                item.pop("tp2_trace_anchor", None)
                item.get("qwen38", {}).pop("tp", None)
            if existing != candidate:
                raise RuntimeError("frozen v4 manifest changed beyond accepted topology")
        state.set_selection("formal_preview_v4", {"enabled": True, "status": "awaiting_acceptance"})
        for node in NODES:
            self.group(state, node)
            self.child([str(Path(sys.executable).with_name("lightcone-spec")), "run", "--config", str(self.args.config)], node)
        self.child([sys.executable, str(self.args.repo / "scripts/render_preview_v4.py"),
                    str(self.config.run_dir / "stages/preview-v4")], "render-v4-results")
        self.videos(state)
        self.check_boundary()
        self.backup(f"v4-complete-{time.time_ns()}")
        self.status("completed_preview_only", complete_dag_resumed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "repo", "output", "manifest", "recording-tools"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("commit", "marker", "power-off-instance"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    token = sys.stdin.readline().strip()
    if not token:
        raise RuntimeError("supply authorized power-off token via stdin; never persist it")
    args.output.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig.load(args.config)
    with continuation_lock(config.run_dir / "preview-continuation.lock") as fd:
        controller = V4Controller(args, fd)
        try:
            controller.run()
        except BaseException as error:
            controller.status("stopped_error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            controller.safe_power_off(token)


if __name__ == "__main__":
    main()
