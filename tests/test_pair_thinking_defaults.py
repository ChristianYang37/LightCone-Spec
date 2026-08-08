"""Thinking-mode defaults for Qwen3 / Gemma model pairs."""

from __future__ import annotations

from lightcone_spec.config.schema import MODEL_PAIRS, pair_thinking_config
from lightcone_spec.sglang_bridge.client import (
    _encode_prompt_token_ids,
    _pair_server_args,
)


def test_qwen3_and_gemma_default_thinking_on():
    for pair_id in (
        "qwen3_4b_dflash16",
        "qwen3_8b_dflash16",
        "qwen3_4b_dspark7",
        "qwen3_8b_dspark7",
        "gemma4_12b_dspark7",
    ):
        cfg = pair_thinking_config(MODEL_PAIRS[pair_id])
        assert cfg["enable_thinking"] is True
        assert cfg["chat_template_kwargs"] == {"enable_thinking": True}
        assert cfg["reasoning_parser"] in {"qwen3", "gemma4"}


def test_llama2_default_thinking_off():
    cfg = pair_thinking_config(MODEL_PAIRS["llama2_7b_eagle"])
    assert cfg["enable_thinking"] is False
    assert cfg["reasoning_parser"] is None
    assert cfg["chat_template_kwargs"] is None


def test_pair_server_args_sets_reasoning_parser_for_qwen3():
    pair = MODEL_PAIRS["qwen3_4b_dflash16"]
    args = _pair_server_args(
        pair,
        target_path="/tmp/target",
        drafter_path="/tmp/draft",
        adaptation_config_path=None,
        num_draft_tokens=16,
        tensor_parallel_size=1,
        random_seed=0,
    )
    assert args["reasoning_parser"] == "qwen3"
    assert args["default_chat_template_kwargs"] == {"enable_thinking": True}


def test_encode_prompt_uses_chat_template_when_thinking_enabled():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs.get("enable_thinking") is True or (
                kwargs.get("chat_template_kwargs") or {}
            ).get("enable_thinking")
            assert messages == [{"role": "user", "content": "hello"}]
            return [9, 8, 7]

    assert _encode_prompt_token_ids(
        Tokenizer(), "hello", enable_thinking=True
    ) == [9, 8, 7]
    assert _encode_prompt_token_ids(
        Tokenizer(), "hello", enable_thinking=False
    ) == [1, 2, 3]
