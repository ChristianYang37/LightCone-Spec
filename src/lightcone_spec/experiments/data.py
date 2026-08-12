"""Deterministic, copyright-independent controlled long-continuation data."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import ClassVar

# The checkpoint accepts at most 40,960 prompt-plus-generated tokens. DFlash
# block-16 verification needs two blocks of request-to-token/KV headroom at a
# decode boundary, so formal measurements stop before those reserved slots.
DFLASH_MODEL_CONTEXT_LIMIT = 40960
DFLASH_SPECULATIVE_HEADROOM = 2 * 16
DFLASH_SAFE_CONTEXT_LIMIT = DFLASH_MODEL_CONTEXT_LIMIT - DFLASH_SPECULATIVE_HEADROOM


@dataclass(frozen=True)
class ControlledWindow:
    name: str
    offset: int
    count: int

    @property
    def stop(self) -> int:
        return self.offset + self.count


@dataclass(frozen=True)
class PromptSample:
    sample_id: str
    prompt: str
    seed: int


def sample_set_sha256(samples: tuple[PromptSample, ...]) -> str:
    rows = [
        {
            "sample_id": sample.sample_id,
            "prompt": sample.prompt,
            "seed": sample.seed,
        }
        for sample in samples
    ]
    body = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


class LongContinuationAdapter:
    """Generate stable prompts from a documented finite-state construction."""

    WINDOWS: ClassVar[dict[str, ControlledWindow]] = {
        "load": ControlledWindow("load", 0, 8),
        "tune": ControlledWindow("tune", 8, 16),
        "confirm": ControlledWindow("confirm", 24, 32),
    }

    def __init__(self, *, namespace: str = "lightcone-speed-v2") -> None:
        self.namespace = namespace

    def sample(self, index: int) -> PromptSample:
        if index < 0:
            raise ValueError("sample index must be non-negative")
        digest = hashlib.sha256(f"{self.namespace}:{index}".encode()).hexdigest()
        seed = int(digest[:8], 16)
        vocabulary = (
            "amber",
            "birch",
            "cobalt",
            "delta",
            "ember",
            "fjord",
            "garnet",
            "harbor",
            "indigo",
            "juniper",
            "kelp",
            "linen",
            "maple",
            "nectar",
            "onyx",
            "pebble",
            "quartz",
            "reed",
            "saffron",
            "thistle",
        )
        ordered = tuple(random.Random(seed).sample(vocabulary, 6))
        prompt = (
            "Produce one space-separated stream only. Repeatedly emit the "
            "following six-word cycle without headings, commentary, lists, "
            "abbreviation, or an end marker: "
            + " ".join(ordered)
            + ". Continue until the generation limit is reached."
        )
        return PromptSample(
            sample_id=f"controlled-{digest[:16]}",
            prompt=prompt,
            seed=seed,
        )

    def window(self, name: str) -> tuple[PromptSample, ...]:
        try:
            window = self.WINDOWS[name]
        except KeyError as exc:
            raise ValueError(f"unknown controlled window {name!r}") from exc
        return tuple(self.sample(index) for index in range(window.offset, window.stop))

    def window_sha256(self, name: str) -> str:
        return sample_set_sha256(self.window(name))

    @classmethod
    def default_hashes(cls) -> dict[str, str]:
        adapter = cls()
        return {
            name: adapter.window_sha256(name) for name in ("load", "tune", "confirm")
        }

    @classmethod
    def assert_disjoint(cls) -> None:
        windows = tuple(cls.WINDOWS.values())
        for index, left in enumerate(windows):
            left_ids = set(range(left.offset, left.stop))
            for right in windows[index + 1 :]:
                if left_ids.intersection(range(right.offset, right.stop)):
                    raise AssertionError(
                        f"controlled windows {left.name} and {right.name} overlap"
                    )
        adapter = cls()
        sample_sets = {
            name: {
                hashlib.sha256(sample.prompt.encode()).hexdigest()
                for sample in adapter.window(name)
            }
            for name in cls.WINDOWS
        }
        for index, left in enumerate(sample_sets):
            for right in tuple(sample_sets)[index + 1 :]:
                if sample_sets[left] & sample_sets[right]:
                    raise AssertionError(
                        f"controlled prompt content leaks across {left}/{right}"
                    )


GENERATED_TOKEN_BUCKETS = (
    (0, 2048),
    (2048, 4096),
    (4096, 8192),
    (8192, 16384),
    (16384, 24576),
    (24576, 32768),
    (32768, DFLASH_SAFE_CONTEXT_LIMIT),
)


def load_natural_prompts(
    dataset_name: str,
    *,
    revision: str,
    split: str,
    limit: int = 32,
) -> tuple[PromptSample, ...]:
    """Load the legacy 32-row preliminary side table.

    This network-capable helper is retained only for historical schema-v2
    diagnostics.  It is not a formal workload authority.  Formal experiments
    use :mod:`lightcone_spec.experiments.workload_authority`, which is local,
    path-bound, release-allowlisted, and selects every exact protocol match.
    """
    specifications = {
        "livecodebench": (
            "livecodebench/code_generation_lite",
            ("prompt", "question_content"),
        ),
        "math500": ("HuggingFaceH4/MATH-500", ("problem", "prompt")),
    }
    if dataset_name not in specifications:
        raise ValueError("unsupported natural side table")
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("a locked 40-character dataset revision is required")
    if not split:
        raise ValueError("dataset split must be non-empty")
    from datasets import load_dataset

    repository, prompt_fields = specifications[dataset_name]
    dataset = load_dataset(
        repository,
        split=split,
        revision=revision,
        streaming=True,
    )
    samples: list[PromptSample] = []
    for index, row in enumerate(dataset):
        if len(samples) >= limit:
            break
        prompt = next(
            (
                row.get(field)
                for field in prompt_fields
                if isinstance(row.get(field), str) and row.get(field).strip()
            ),
            None,
        )
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        digest = hashlib.sha256(
            f"{repository}:{revision}:{split}:{index}".encode()
        ).hexdigest()
        samples.append(
            PromptSample(
                sample_id=f"{dataset_name}-{digest[:16]}",
                prompt=prompt,
                seed=int(digest[:8], 16),
            )
        )
    if len(samples) != limit:
        raise ValueError(
            f"{dataset_name} produced {len(samples)} prompts, expected {limit}"
        )
    identifiers = [sample.sample_id for sample in samples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("natural side-table sample IDs are not unique")
    return tuple(samples)
