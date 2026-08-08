"""The 14 unique benchmark adapters (spec 12.1).

DeepSpec-group tasks reuse the pinned DeepSpec data/evaluator semantics;
TTS-supplement tasks are locked to the standard splits listed in the TTS
paper. After the first `lock`, none of them follow upstream changes.
Overlapping datasets (e.g. AIME25 appearing in both papers) get exactly
one adapter, one locked revision and one scorer.
"""

from __future__ import annotations

from lightcone_spec.benchmarks.base import BenchmarkAdapter

_MATH_TEMPLATE = (
    "Solve the following problem. Put your final answer in \\boxed{{}}.\n\n"
    "{question}"
)
_CODE_TEMPLATE = (
    "Write a complete Python solution for the following task. Answer with a "
    "single fenced python code block.\n\n{question}"
)
_CHAT_TEMPLATE = "{question}"

_JUDGE_MODEL = "Qwen/Qwen3-32B-Judge"
_JUDGE_REVISION = "locked"


def _adapter(**kw) -> BenchmarkAdapter:
    field_maps = kw.pop("field_maps", {})
    a = BenchmarkAdapter(**kw)
    a._field_maps = field_maps
    return a


BENCHMARK_ADAPTERS: dict[str, BenchmarkAdapter] = {
    # ---- DeepSpec group -------------------------------------------------
    "gsm8k": _adapter(
        key="gsm8k",
        source_group="deepspec",
        hf_path="openai/gsm8k",
        hf_config="main",
        split="test",
        quality_metric="exact_match",
        output_cap=4096,
        task_type="math",
        prompt_template=_MATH_TEMPLATE,
        stop_strings=("</s>",),
        scorer_kind="exact_match",
        timeout_s=0.0,
        license_note="MIT (openai/gsm8k)",
        field_maps={"prompt_field": "question", "answer_field": "answer"},
    ),
    "math500": _adapter(
        key="math500",
        source_group="deepspec",
        hf_path="HuggingFaceH4/MATH-500",
        hf_config=None,
        split="test",
        quality_metric="exact_match",
        output_cap=32768,
        task_type="math",
        prompt_template=_MATH_TEMPLATE,
        stop_strings=("</s>",),
        scorer_kind="exact_match",
        timeout_s=0.0,
        license_note="MIT (MATH-500 subset of hendrycks MATH)",
        data_file="test.jsonl",
        field_maps={
            "prompt_field": "problem",
            "answer_field": "answer",
            "id_field": "unique_id",
        },
    ),
    "aime25": _adapter(
        key="aime25",
        source_group="deepspec",
        hf_path="opencompass/AIME2025",
        hf_config="AIME2025-I",
        split="test",
        quality_metric="accuracy",
        output_cap=32768,
        task_type="math",
        prompt_template=_MATH_TEMPLATE,
        stop_strings=("</s>",),
        scorer_kind="exact_match",
        timeout_s=0.0,
        license_note="AIME 2025 competition problems (fair-use eval set)",
        field_maps={"prompt_field": "question", "answer_field": "answer"},
    ),
    "mbpp": _adapter(
        key="mbpp",
        source_group="deepspec",
        hf_path="google-research-datasets/mbpp",
        hf_config="sanitized",
        split="test",
        quality_metric="pass@1",
        output_cap=4096,
        task_type="code",
        prompt_template=_CODE_TEMPLATE,
        stop_strings=("</s>",),
        scorer_kind="pass_at_1",
        timeout_s=20.0,
        license_note="CC-BY-4.0 (MBPP)",
        field_maps={
            "prompt_field": "prompt",
            "test_field": "test_list",
            "id_field": "task_id",
        },
    ),
    "humaneval": _adapter(
        key="humaneval",
        source_group="deepspec",
        hf_path="openai/openai_humaneval",
        hf_config=None,
        split="test",
        quality_metric="pass@1",
        output_cap=4096,
        task_type="code",
        prompt_template=_CODE_TEMPLATE,
        stop_strings=("</s>",),
        scorer_kind="pass_at_1",
        timeout_s=20.0,
        license_note="MIT (HumanEval)",
        field_maps={
            "prompt_field": "prompt",
            "test_field": "test",
            "id_field": "task_id",
        },
    ),
    "livecodebench": _adapter(
        key="livecodebench",
        source_group="deepspec",
        hf_path="livecodebench/code_generation_lite",
        hf_config="release_v6",
        split="test",
        quality_metric="pass@1",
        output_cap=32768,
        task_type="code",
        prompt_template=_CODE_TEMPLATE,
        stop_strings=("</s>",),
        scorer_kind="pass_at_1",
        timeout_s=30.0,
        license_note="LiveCodeBench license; problems from public judges",
        data_file=tuple(
            "test.jsonl" if index == 1 else f"test{index}.jsonl"
            for index in range(1, 7)
        ),
        field_maps={
            "prompt_field": "question_content",
            "test_field": "public_test_cases",
            "id_field": "question_id",
        },
    ),
    "mt_bench": _adapter(
        key="mt_bench",
        source_group="deepspec",
        hf_path="HuggingFaceH4/mt_bench_prompts",
        hf_config=None,
        split="train",
        quality_metric="judge_score",
        output_cap=4096,
        task_type="chat",
        prompt_template=_CHAT_TEMPLATE,
        stop_strings=("</s>",),
        scorer_kind="judge",
        timeout_s=0.0,
        license_note="Apache-2.0 (MT-Bench prompts)",
        judge_model=_JUDGE_MODEL,
        judge_revision=_JUDGE_REVISION,
        data_file="data/train-00000-of-00001-67c6c9fef07685a3.parquet",
        # P5 uses the standard first-turn MT-Bench prompt as the task suffix.
        # The second turn depends on the model's first answer and therefore is
        # not an identical paired prefix checkpoint across methods.
        field_maps={
            "prompt_field": "prompt",
            "prompt_index": 0,
            "id_field": "prompt_id",
        },
    ),
    "alpaca": _adapter(
        key="alpaca",
        source_group="deepspec",
        hf_path="tatsu-lab/alpaca_eval",
        hf_config=None,
        split="eval",
        quality_metric="judge_score",
        output_cap=4096,
        task_type="chat",
        prompt_template=_CHAT_TEMPLATE,
        stop_strings=("</s>",),
        scorer_kind="judge",
        timeout_s=0.0,
        license_note="CC-BY-NC-4.0 (AlpacaEval); locked evaluator",
        judge_model=_JUDGE_MODEL,
        judge_revision=_JUDGE_REVISION,
        data_file="alpaca_eval.json",
        field_maps={"prompt_field": "instruction", "id_field": "dataset_index"},
    ),
    "arena_hard_v2": _adapter(
        key="arena_hard_v2",
        source_group="deepspec",
        hf_path="lmarena-ai/arena-hard-v2.0",
        hf_config=None,
        split="test",
        quality_metric="judge_score",
        output_cap=4096,
        task_type="chat",
        prompt_template=_CHAT_TEMPLATE,
        stop_strings=("</s>",),
        scorer_kind="judge",
        timeout_s=0.0,
        license_note="Apache-2.0 (Arena-Hard v2)",
        judge_model=_JUDGE_MODEL,
        judge_revision=_JUDGE_REVISION,
        field_maps={"prompt_field": "prompt", "id_field": "uid"},
    ),
    # ---- TTS supplement -------------------------------------------------
    "aime24": _adapter(
        key="aime24",
        source_group="tts",
        hf_path="HuggingFaceH4/aime_2024",
        hf_config=None,
        split="train",
        quality_metric="accuracy",
        output_cap=32768,
        task_type="math",
        prompt_template=_MATH_TEMPLATE,
        stop_strings=("</s>",),
        scorer_kind="exact_match",
        timeout_s=0.0,
        license_note="AIME 2024 competition problems (fair-use eval set)",
        field_maps={"prompt_field": "problem", "answer_field": "answer", "id_field": "id"},
    ),
    "olympiadbench_math": _adapter(
        key="olympiadbench_math",
        source_group="tts",
        hf_path="Hothan/OlympiadBench",
        hf_config="OE_TO_maths_en_COMP",
        split="train",
        quality_metric="accuracy",
        output_cap=32768,
        task_type="math",
        prompt_template=_MATH_TEMPLATE,
        stop_strings=("</s>",),
        scorer_kind="exact_match",
        timeout_s=0.0,
        license_note="Apache-2.0 (OlympiadBench)",
        field_maps={"prompt_field": "question", "answer_field": "final_answer", "id_field": "id"},
    ),
    "olympiadbench_physics": _adapter(
        key="olympiadbench_physics",
        source_group="tts",
        hf_path="Hothan/OlympiadBench",
        hf_config="OE_TO_physics_en_COMP",
        split="train",
        quality_metric="accuracy",
        output_cap=32768,
        task_type="science",
        prompt_template=_MATH_TEMPLATE,
        stop_strings=("</s>",),
        scorer_kind="exact_match",
        timeout_s=0.0,
        license_note="Apache-2.0 (OlympiadBench)",
        field_maps={"prompt_field": "question", "answer_field": "final_answer", "id_field": "id"},
    ),
    "gpqa_diamond": _adapter(
        key="gpqa_diamond",
        source_group="tts",
        hf_path="Idavidrein/gpqa",
        hf_config="gpqa_diamond",
        split="train",
        quality_metric="accuracy",
        output_cap=32768,
        task_type="science",
        prompt_template=_MATH_TEMPLATE,
        stop_strings=("</s>",),
        scorer_kind="exact_match",
        timeout_s=0.0,
        license_note="CC-BY-4.0 (GPQA); do not redistribute answers",
        field_maps={
            "prompt_field": "Question",
            "answer_field": "Correct Answer",
            "id_field": "Record ID",
        },
    ),
    "theoremqa": _adapter(
        key="theoremqa",
        source_group="tts",
        hf_path="TIGER-Lab/TheoremQA",
        hf_config=None,
        split="test",
        quality_metric="accuracy",
        output_cap=32768,
        task_type="math",
        prompt_template=_MATH_TEMPLATE,
        stop_strings=("</s>",),
        scorer_kind="exact_match",
        timeout_s=0.0,
        license_note="MIT (TheoremQA)",
        field_maps={"prompt_field": "Question", "answer_field": "Answer", "id_field": "id"},
    ),
}

assert len(BENCHMARK_ADAPTERS) == 14


def get_adapter(key: str) -> BenchmarkAdapter:
    if key not in BENCHMARK_ADAPTERS:
        raise KeyError(f"unknown benchmark adapter {key!r}")
    return BENCHMARK_ADAPTERS[key]
