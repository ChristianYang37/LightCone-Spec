from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README_PATHS = (ROOT / "README.md", ROOT / "README_zh-CN.md")
CLI_PATHS = (ROOT / "docs/en/cli.md", ROOT / "docs/zh-CN/cli.md")
PROTOCOL_PATHS = (
    ROOT / "docs/en/experiment-protocol.md",
    ROOT / "docs/zh-CN/experiment-protocol.md",
)

EN_TRUSTED_HEADING = "## Trusted single-operator formal workflow"
ZH_TRUSTED_HEADING = "## 单可信操作者正式实验链"

EXPECTED_EN_SUBHEADINGS = (
    "### Claim boundary",
    "### Publish immutable inputs",
    "### Configure and cross the first GPU boundary",
    "### Automated 21-node progression",
    "### Rolling archive and restoration",
    "### Cross-host archive, shutdown, and billing",
)
EXPECTED_ZH_SUBHEADINGS = (
    "### 结论边界",
    "### 发布不可变输入",
    "### 配置并到达第一个 GPU 边界",
    "### 自动推进 21 节点",
    "### 滚动归档与恢复",
    "### 跨主机归档、关机与计费",
)

REQUIRED_COMMANDS = (
    "publish-formal-runtime-authority-manifest",
    "publish-tts-calibration-source-authority",
    "publish-chronobelief-source-authority",
    "publish-e1-recipe-anchor-authority",
    "publish-onlinespec-source-authority",
    "publish-trusted-content",
    "publish-preflight-workload",
    "build-trusted-protocol-lock",
    "write-dag-driver-config",
    "write-bootstrap-config",
    "bootstrap-once",
    "bootstrap-run",
    "formal_rolling_archive_companion",
    "formal_experiment_production_finalizer.py",
)


def _trusted_section(path: Path, heading: str) -> str:
    content = path.read_text(encoding="utf-8")
    before, marker, section = content.partition(heading)
    assert before and marker
    return section


def _bash_blocks(content: str) -> tuple[str, ...]:
    return tuple(re.findall(r"```bash\n(.*?)```", content, flags=re.DOTALL))


def test_trusted_v03_cli_docs_have_parallel_structure_and_public_commands() -> None:
    english = _trusted_section(CLI_PATHS[0], EN_TRUSTED_HEADING)
    chinese = _trusted_section(CLI_PATHS[1], ZH_TRUSTED_HEADING)

    assert tuple(re.findall(r"^### .+$", english, flags=re.MULTILINE)) == (
        EXPECTED_EN_SUBHEADINGS
    )
    assert tuple(re.findall(r"^### .+$", chinese, flags=re.MULTILINE)) == (
        EXPECTED_ZH_SUBHEADINGS
    )
    for command in REQUIRED_COMMANDS:
        assert command in english
        assert command in chinese

    for section in (english, chinese):
        assert "readiness_scope=code_capability_only" in section
        assert "operator.sqlite3" in section
        assert "formal-dag-driver.lock" in section
        assert "PENDING" in section
        assert "11,000" in section
        assert "10,000" in section
        assert 'code=="Success"' in section
        assert "formal_measured=false" in section
        assert "trusted_single_operator_empirical_no_signature" in section


def test_trusted_v03_examples_use_only_placeholder_absolute_paths() -> None:
    for path, heading in zip(
        CLI_PATHS,
        (EN_TRUSTED_HEADING, ZH_TRUSTED_HEADING),
        strict=True,
    ):
        section = _trusted_section(path, heading)
        blocks = _bash_blocks(section)
        assert blocks
        path_literals = tuple(
            match.group(0)
            for block in blocks
            for match in re.finditer(r"(?<![A-Za-z0-9_.-])/[A-Za-z0-9_./-]+", block)
        )
        assert path_literals
        assert all(value.startswith("/absolute/") for value in path_literals)
        assert not any(
            token in section
            for token in (
                "/Users/",
                "/" + "root/",
                "root@",
                "connect.",
                "Bearer ",
                "ssh -p ",
            )
        )


def test_readmes_and_protocol_keep_the_unsigned_empirical_boundary() -> None:
    for path in (*README_PATHS, *PROTOCOL_PATHS):
        content = path.read_text(encoding="utf-8")
        assert "trusted_single_operator_empirical_no_signature" in content
        assert "UNMEASURED" in content
        assert "MEASURED" in content

    for path in PROTOCOL_PATHS:
        content = path.read_text(encoding="utf-8")
        assert "formal_measured=false" in content
        assert "11,000" in content
        assert "10,000" in content
        assert "mtp.*" in content
        assert "108" in content


def test_readme_trusted_v03_summaries_match_the_implemented_boundaries() -> None:
    for path in README_PATHS:
        content = path.read_text(encoding="utf-8")
        assert "21" in content
        assert "bootstrap-run" in content
        assert "bootstrap-once" in content
        assert "PENDING" in content
        assert "mtp.*" in content
        assert "12" in content
        assert "108" in content
