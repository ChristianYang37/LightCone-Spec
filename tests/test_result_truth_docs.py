from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README_PATHS = (ROOT / "README.md", ROOT / "README_zh-CN.md")
TRUTH_BEGIN = "<!-- RESULT_TRUTH_GATE_BEGIN -->"
TRUTH_END = "<!-- RESULT_TRUTH_GATE_END -->"
TRUTH_ROW = re.compile(
    r"^\| `(?P<field>[a-z0-9_]+)` \| `(?P<value>[^`]+)` \|",
    re.MULTILINE,
)
GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")

STATUS_FIELDS = {
    "formal_industrial_gpu_evidence": "UNMEASURED",
    "formal_industrial_execution": "BLOCKED",
    "historical_snapshot_evidence": "PRELIMINARY_NON_FORMAL",
    "historical_snapshot_host_at_archive": "POWERED_OFF_NOT_RELEASED",
}
IDENTITY_FIELDS = {
    "current_sglang_upstream_commit",
    "current_patched_sglang_tree",
    "current_patch_payload_sha256",
    "current_patch_manifest_sha256",
    "historical_main_code_prefix",
    "historical_patched_tree_prefix",
}
HISTORICAL_RESULT_TOKENS = (
    "1,342.0 tok/s",
    "2,497.9 tok/s",
    "2,519.5 tok/s",
    "654,042",
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _truth_block(content: str) -> str:
    assert content.count(TRUTH_BEGIN) == 1
    assert content.count(TRUTH_END) == 1
    start = content.index(TRUTH_BEGIN) + len(TRUTH_BEGIN)
    end = content.index(TRUTH_END, start)
    return content[start:end]


def _truth_fields(content: str) -> dict[str, str]:
    rows = TRUTH_ROW.findall(_truth_block(content))
    fields = dict(rows)
    assert len(fields) == len(rows), "truth gate contains a duplicate field"
    return fields


def _release_identities() -> dict[str, str]:
    manifest_path = ROOT / "patches" / "sglang" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patches = manifest["patches"]
    assert patches, "README truth schema requires a patch payload"
    patch_path = manifest_path.parent / patches[-1]["file"]
    patch_sha256 = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    assert patch_sha256 == patches[-1]["sha256"]
    return {
        "current_sglang_upstream_commit": manifest["upstream"]["commit"],
        "current_patched_sglang_tree": manifest["expected_tree"],
        "current_patch_payload_sha256": patch_sha256,
        "current_patch_manifest_sha256": _canonical_sha256(manifest),
        "historical_main_code_prefix": "0db2ff4",
        "historical_patched_tree_prefix": "e795ecc",
    }


def test_readmes_publish_the_same_fail_closed_result_truth() -> None:
    expected = STATUS_FIELDS | _release_identities()
    parsed: list[dict[str, str]] = []

    for path in README_PATHS:
        content = path.read_text(encoding="utf-8")
        fields = _truth_fields(content)
        parsed.append(fields)
        assert STATUS_FIELDS.items() <= fields.items()
        assert IDENTITY_FIELDS <= fields.keys()
        assert {field: fields[field] for field in expected} == expected

        block = _truth_block(content)
        assert re.search(r"(?<!UN)MEASURED", block) is None
        assert "`PASS`" not in block
        assert fields["historical_snapshot_host_at_archive"] not in {
            fields["formal_industrial_gpu_evidence"],
            fields["formal_industrial_execution"],
        }

    assert parsed[0] == parsed[1]


def test_truth_gate_keeps_identity_domains_distinct() -> None:
    fields = _truth_fields(README_PATHS[0].read_text(encoding="utf-8"))
    commit = fields["current_sglang_upstream_commit"]
    tree = fields["current_patched_sglang_tree"]
    patch_sha256 = fields["current_patch_payload_sha256"]
    manifest_sha256 = fields["current_patch_manifest_sha256"]
    historical = {
        fields["historical_main_code_prefix"],
        fields["historical_patched_tree_prefix"],
    }

    assert GIT_OBJECT.fullmatch(commit)
    assert GIT_OBJECT.fullmatch(tree)
    assert SHA256.fullmatch(patch_sha256)
    assert SHA256.fullmatch(manifest_sha256)
    assert len({commit, tree, patch_sha256, manifest_sha256}) == 4
    assert all(re.fullmatch(r"[0-9a-f]{7}", value) for value in historical)
    assert all(not commit.startswith(value) for value in historical)
    assert all(not tree.startswith(value) for value in historical)


def test_preliminary_numbers_remain_inside_the_historical_snapshot() -> None:
    headings = {
        "README.md": "## Historical preliminary mechanism snapshot",
        "README_zh-CN.md": "## 历史 preliminary 机制 snapshot",
    }
    required_disclaimers = {
        "README.md": "retained only as preliminary engineering evidence",
        "README_zh-CN.md": "仅保留为 preliminary 工程证据",
    }

    for path in README_PATHS:
        content = path.read_text(encoding="utf-8")
        before, heading, snapshot = content.partition(headings[path.name])
        assert heading
        assert required_disclaimers[path.name] in snapshot
        assert "`n=1`" in snapshot
        assert "`failed_resumable`" in snapshot
        for token in HISTORICAL_RESULT_TOKENS:
            assert token not in before
            assert token in snapshot
