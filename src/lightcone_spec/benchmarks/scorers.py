"""Scorers (spec 12.2, 14.3).

- exact-match / accuracy for math and knowledge tasks (boxed-answer
  extraction plus numeric equivalence);
- pass@1 for code tasks via sandboxed subprocess execution;
- locked-judge scorers for MT-Bench / Alpaca / Arena-Hard-v2, which
  require the pinned judge model at GPU runtime and fail closed without
  it.

Invalid outputs score 0 and are counted, never dropped.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_LAST_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def extract_final_answer(text: str) -> Optional[str]:
    m = _BOXED.findall(text)
    if m:
        return m[-1].strip()
    hash_split = text.rsplit("####", 1)
    if len(hash_split) == 2:
        return hash_split[1].strip().splitlines()[0].strip()
    nums = _LAST_NUMBER.findall(text)
    if nums:
        return nums[-1]
    return None


def _to_number(s: str) -> Optional[Fraction]:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            return Fraction(int(num.strip()), int(den.strip()))
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def answers_equivalent(pred: str, gold: str) -> bool:
    if pred is None:
        return False
    p, g = pred.strip(), gold.strip()
    if p == g:
        return True
    pn, gn = _to_number(p), _to_number(g)
    if pn is not None and gn is not None:
        return pn == gn
    return p.lower().replace(" ", "") == g.lower().replace(" ", "")


def exact_match_score(output: str, gold_answer: str) -> float:
    pred = extract_final_answer(output)
    if pred is None:
        return 0.0
    return 1.0 if answers_equivalent(pred, gold_answer) else 0.0


# ---------------------------------------------------------------------------
# pass@1 code execution
# ---------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = _CODE_BLOCK.findall(text)
    if blocks:
        return blocks[-1]
    return text


def pass_at_1_score(
    output: str, test_code: str, timeout_s: float = 20.0
) -> float:
    """Run extracted solution + locked test code in a subprocess with a
    hard timeout; nonzero exit or timeout scores 0."""
    code = extract_code(output)
    program = f"{code}\n\n{test_code}\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(program)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True,
                timeout=timeout_s,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return 0.0
        return 1.0 if proc.returncode == 0 else 0.0


# ---------------------------------------------------------------------------
# locked judge
# ---------------------------------------------------------------------------


@dataclass
class LockedJudgeScorer:
    """Judge-scored tasks: judge model, revision and prompt are locked in
    the dataset manifest; scoring requires a judge callable bound at GPU
    runtime. Without one, scoring fails closed (never a silent 0)."""

    judge_model: str
    judge_revision: str
    judge_prompt_sha256: str
    judge_fn: Optional[Callable[[str, str], float]] = None

    def score(self, prompt: str, output: str) -> float:
        if self.judge_fn is None:
            raise RuntimeError(
                f"locked judge {self.judge_model}@{self.judge_revision} not "
                "bound; judge-scored tasks cannot be scored offline"
            )
        return float(self.judge_fn(prompt, output))
