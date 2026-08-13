from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest

from lightcone_spec.orchestration.native_terminal import canonical_json_bytes
from lightcone_spec.orchestration.session_live_evidence import (
    SESSION_LIVE_CLOSE_MANIFEST,
    SESSION_LIVE_STEP_PREFIX,
    IncrementalSessionLiveEvidenceSink,
    publish_session_live_evidence,
    reopen_session_live_evidence,
)

_LIVE_FIXTURE_PATH = Path(__file__).with_name("test_session_live_runtime.py")
_LIVE_SPEC = importlib.util.spec_from_file_location(
    "_durable_live_fixture", _LIVE_FIXTURE_PATH
)
assert _LIVE_SPEC is not None and _LIVE_SPEC.loader is not None
_LIVE = importlib.util.module_from_spec(_LIVE_SPEC)
_LIVE_SPEC.loader.exec_module(_LIVE)


def _complete_result():
    result, _owner, _backend, _events = _LIVE._run(_LIVE._resources())
    return result


def _replace(path: Path, body: bytes) -> None:
    path.unlink()
    path.write_bytes(body)
    path.chmod(0o400)


def _rewrite_manifest_digest(value: dict[str, object]) -> bytes:
    value.pop("manifest_sha256", None)
    import hashlib

    value["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return canonical_json_bytes(value)


def test_publish_and_reopen_requires_exact_typed_live_result(tmp_path: Path) -> None:
    result = _complete_result()

    publication = publish_session_live_evidence(
        output_dir=tmp_path.resolve(),
        result=result,
    )
    reopened = reopen_session_live_evidence(
        output_dir=tmp_path.resolve(),
        expected_result=result,
    )

    assert publication.committed and reopened.committed
    assert publication == reopened
    assert publication.reuse_authorized is False
    assert publication.evidence_level == "CPU_CONTRACT_ONLY"
    assert publication.gpu_reset_semantics == "PENDING"
    assert len(publication.trace_artifacts) == 1
    paths = [
        tmp_path / publication.trace_artifacts[0].filename,
        tmp_path / SESSION_LIVE_CLOSE_MANIFEST,
    ]
    for path in paths:
        metadata = path.stat()
        assert metadata.st_nlink == 1
        assert stat.S_IMODE(metadata.st_mode) == 0o400
    manifest = json.loads((tmp_path / SESSION_LIVE_CLOSE_MANIFEST).read_bytes())
    assert [step["step"] for step in manifest["global_steps"]] == [
        "session_capability",
        "session_initial_state",
        "session_close_terminal",
    ]
    assert manifest["close_receipt_sha256"] == result.audit.close_receipt_sha256


def test_failed_session_retains_partial_trace_without_commit_marker(
    tmp_path: Path,
) -> None:
    result, _owner, _backend, _events = _LIVE._run(
        _LIVE._resources(malformed_clock=True)
    )

    publication = publish_session_live_evidence(
        output_dir=tmp_path.resolve(),
        result=result,
    )

    assert publication.status == "FRESH_PROCESS_REQUIRED"
    assert not publication.committed
    assert publication.reuse_authorized is False
    assert publication.trace_artifacts
    assert not publication.trace_artifacts[-1].complete
    assert not (tmp_path / SESSION_LIVE_CLOSE_MANIFEST).exists()
    reopened = reopen_session_live_evidence(
        output_dir=tmp_path.resolve(),
        expected_result=result,
    )
    assert reopened == publication
    assert not reopened.committed

    (tmp_path / SESSION_LIVE_CLOSE_MANIFEST).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forged a close marker"):
        reopen_session_live_evidence(
            output_dir=tmp_path.resolve(),
            expected_result=result,
        )


def test_crash_before_manifest_leaves_trace_uncommitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lightcone_spec.orchestration import session_live_evidence as evidence

    result = _complete_result()
    original = evidence._publish_exclusive_at

    def fail_manifest(directory_fd, *, filename, body, label):
        if filename == SESSION_LIVE_CLOSE_MANIFEST:
            raise OSError("injected close-manifest crash")
        return original(
            directory_fd,
            filename=filename,
            body=body,
            label=label,
        )

    monkeypatch.setattr(evidence, "_publish_exclusive_at", fail_manifest)
    with pytest.raises(OSError, match="injected"):
        publish_session_live_evidence(
            output_dir=tmp_path.resolve(),
            result=result,
        )

    assert any(tmp_path.glob("trace-*.json"))
    assert not (tmp_path / SESSION_LIVE_CLOSE_MANIFEST).exists()


@pytest.mark.parametrize(
    "body,error",
    [
        (b'{"schema_version":1,"schema_version":1}', "duplicate"),
        (b'{"schema_version":NaN}', "non-finite"),
        (b'{"schema_version": 1}', "canonical"),
    ],
)
def test_reopen_rejects_duplicate_nonfinite_and_noncanonical_manifest(
    tmp_path: Path,
    body: bytes,
    error: str,
) -> None:
    result = _complete_result()
    publish_session_live_evidence(output_dir=tmp_path.resolve(), result=result)
    _replace(tmp_path / SESSION_LIVE_CLOSE_MANIFEST, body)

    with pytest.raises(ValueError, match=error):
        reopen_session_live_evidence(
            output_dir=tmp_path.resolve(),
            expected_result=result,
        )


def test_reopen_rejects_symlink_hardlink_delete_and_path_escape(
    tmp_path: Path,
) -> None:
    result = _complete_result()
    root = (tmp_path / "evidence").resolve()
    root.mkdir()
    publication = publish_session_live_evidence(output_dir=root, result=result)
    manifest = root / SESSION_LIVE_CLOSE_MANIFEST

    alias = tmp_path / "manifest-hardlink.json"
    os.link(manifest, alias)
    with pytest.raises(RuntimeError, match="single-link"):
        reopen_session_live_evidence(output_dir=root, expected_result=result)
    alias.unlink()

    trace_path = root / publication.trace_artifacts[0].filename
    trace_path.unlink()
    with pytest.raises(FileNotFoundError):
        reopen_session_live_evidence(output_dir=root, expected_result=result)

    trace_path.symlink_to(manifest)
    with pytest.raises(OSError):
        reopen_session_live_evidence(output_dir=root, expected_result=result)
    trace_path.unlink()

    symlink = tmp_path / "evidence-link"
    symlink.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="resolved.*symlink-free"):
        reopen_session_live_evidence(output_dir=symlink, expected_result=result)
    with pytest.raises(ValueError, match="absolute"):
        reopen_session_live_evidence(
            output_dir=Path("relative"), expected_result=result
        )


def test_replace_and_rehash_cannot_reorder_or_self_authorize(
    tmp_path: Path,
) -> None:
    result = _complete_result()
    publication = publish_session_live_evidence(
        output_dir=tmp_path.resolve(),
        result=result,
    )
    trace_binding = publication.trace_artifacts[0]
    trace_path = tmp_path / trace_binding.filename
    trace = json.loads(trace_path.read_bytes())
    trace["steps"] = list(reversed(trace["steps"]))
    replacement = canonical_json_bytes(trace)
    _replace(trace_path, replacement)

    import hashlib

    manifest_path = tmp_path / SESSION_LIVE_CLOSE_MANIFEST
    manifest = json.loads(manifest_path.read_bytes())
    manifest["trace_artifacts"][0]["size"] = len(replacement)
    manifest["trace_artifacts"][0]["raw_sha256"] = hashlib.sha256(
        replacement
    ).hexdigest()
    _replace(manifest_path, _rewrite_manifest_digest(manifest))

    with pytest.raises(RuntimeError, match="trace artifact changed"):
        reopen_session_live_evidence(
            output_dir=tmp_path.resolve(),
            expected_result=result,
        )


def test_incremental_sink_persists_each_step_and_binds_final_manifest(
    tmp_path: Path,
) -> None:
    sink = IncrementalSessionLiveEvidenceSink(tmp_path.resolve())
    resources = _LIVE._resources(evidence_sink=sink)
    result, _owner, _backend, _events = _LIVE._run(resources)

    publication = sink.publication
    assert publication is not None and publication.committed
    assert len(sink.step_artifacts) == len(result.steps)
    step_paths = sorted(tmp_path.glob(f"{SESSION_LIVE_STEP_PREFIX}*.json"))
    assert len(step_paths) == len(result.steps)
    manifest = json.loads((tmp_path / SESSION_LIVE_CLOSE_MANIFEST).read_bytes())
    assert len(manifest["incremental_step_artifacts"]) == len(result.steps)
    assert reopen_session_live_evidence(
        output_dir=tmp_path.resolve(),
        expected_result=result,
    ) == publication


def test_incremental_sink_failure_retains_steps_without_close_marker(
    tmp_path: Path,
) -> None:
    sink = IncrementalSessionLiveEvidenceSink(tmp_path.resolve())
    resources = _LIVE._resources(malformed_clock=True, evidence_sink=sink)

    result, _owner, _backend, _events = _LIVE._run(resources)

    assert result.audit.status == "FRESH_PROCESS_REQUIRED"
    assert sink.publication is not None
    assert not sink.publication.committed
    assert sink.step_artifacts
    assert list(tmp_path.glob(f"{SESSION_LIVE_STEP_PREFIX}*.json"))
    assert not (tmp_path / SESSION_LIVE_CLOSE_MANIFEST).exists()
    with pytest.raises(RuntimeError, match="closed"):
        sink.record_step(sink.step_artifacts[0])
