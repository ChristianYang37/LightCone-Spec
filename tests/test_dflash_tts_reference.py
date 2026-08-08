from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn


SCRIPT = Path(__file__).parents[1] / "scripts" / "experiments" / "dflash_tts_reference.py"
SPEC = importlib.util.spec_from_file_location("dflash_tts_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference
SPEC.loader.exec_module(reference)

PROVENANCE = {
    "reference_revision": "ref",
    "reference_source_sha256": "ref-sha",
    "target_declared_revision": "target",
    "draft_declared_revision": "draft",
    "dataset_declared_revision": "data",
    "dataset_sha256": "data-sha",
    "harness_source_sha256": "harness-sha",
}


def test_reference_api_and_official_position_recipe_contract():
    module = ModuleType("fake_reference")

    class DFlashDraftModel(nn.Module):
        pass

    def dflash_generate(
        model,
        target,
        input_ids,
        max_new_tokens,
        stop_token_ids,
        temperature,
        block_size=None,
        mask_token_id=None,
        return_stats=False,
    ):
        del (
            model,
            target,
            input_ids,
            max_new_tokens,
            stop_token_ids,
            temperature,
            block_size,
            mask_token_id,
            return_stats,
        )

    module.DFlashDraftModel = DFlashDraftModel
    module.dflash_generate = dflash_generate
    module.extract_context_feature = lambda hidden, layers: (hidden, layers)
    reference.validate_reference_api(module)

    weights = reference.build_position_weights(15, "exponential", 7.0)
    expected = torch.exp(-torch.arange(15, dtype=torch.float32) / 7.0)
    torch.testing.assert_close(weights, expected)
    assert weights[0] == 1.0


def test_static_and_full_drafter_scopes_are_disjoint():
    static = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    assert reference.configure_trainable_scope(static, "static") == ()
    assert not static.training
    assert all(not parameter.requires_grad for parameter in static.parameters())

    full = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    parameters = reference.configure_trainable_scope(full, "full-drafter")
    assert not full.training
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in full.parameters()
    }
    assert all(parameter.requires_grad for parameter in full.parameters())


def test_turns_records_use_chat_template_instead_of_stringifying_list():
    class RecordingTokenizer:
        def __init__(self):
            self.messages = None
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return torch.tensor([[7, 8]])

    tokenizer = RecordingTokenizer()
    ids = reference._tokenize_record(
        tokenizer,
        {"turns": ["Prove the claim."]},
        prompt_field="prompt",
        messages_field="messages",
        turns_field="turns",
        enable_thinking=True,
    )

    assert ids.tolist() == [[7, 8]]
    assert tokenizer.messages == [
        {"role": "user", "content": "Prove the claim."}
    ]
    assert tokenizer.kwargs["enable_thinking"] is True


def test_plain_prompt_enable_thinking_uses_user_chat_template():
    class RecordingTokenizer:
        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is True
            self.messages = messages
            return torch.tensor([[9]])

    tokenizer = RecordingTokenizer()
    ids = reference._tokenize_record(
        tokenizer,
        {"prompt": "Reason carefully."},
        prompt_field="prompt",
        messages_field="messages",
        turns_field="turns",
        enable_thinking=True,
    )

    assert ids.tolist() == [[9]]
    assert tokenizer.messages == [
        {"role": "user", "content": "Reason carefully."}
    ]


def test_full_drafter_step_owns_and_updates_every_parameter():
    torch.manual_seed(0)
    draft = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 5))
    parameters = reference.configure_trainable_scope(draft, "full-drafter")
    optimizer = torch.optim.Adam(parameters, lr=0.05)
    inputs = torch.randn(1, 2, 3)
    proposal_logits = draft(inputs)
    target_logits = torch.zeros_like(proposal_logits)
    target_logits[..., 1] = 3.0
    before = {
        name: parameter.detach().clone()
        for name, parameter in draft.named_parameters()
    }

    evidence = reference.apply_full_drafter_adam_step(
        draft_model=draft,
        optimizer=optimizer,
        proposal_logits=proposal_logits,
        target_logits=target_logits,
        position_weights=torch.ones(2),
        proximal_lambda=0.7,
        loss_reduction="weighted-mean",
        optimizer_step=1,
    )

    assert evidence.applied
    assert evidence.optimizer_step == 1
    assert evidence.grad_norm is not None and evidence.grad_norm > 0.0
    assert evidence.parameters_with_grad == len(tuple(draft.named_parameters()))
    assert evidence.parameters_without_grad == ()
    for name, parameter in draft.named_parameters():
        assert not torch.equal(parameter.detach(), before[name]), name


class FakeCache:
    def __init__(self):
        self.length = 0
        self.layers = [SimpleNamespace(keys=None, values=None)]

    def get_seq_length(self):
        return self.length

    def crop(self, length):
        self.length = min(self.length, int(length))
        for layer in self.layers:
            if layer.keys is not None:
                layer.keys = layer.keys[:, : self.length]
                layer.values = layer.values[:, : self.length]


def test_dynamic_cache_detach_preserves_values_and_drops_graph():
    class Layer:
        pass

    cache = FakeCache()
    cache.length = 2
    cache.layers = [Layer()]
    source = torch.randn(1, 2, requires_grad=True)
    cache.layers[0].keys = source * 2.0
    cache.layers[0].values = source + 1.0
    keys_before = cache.layers[0].keys.clone()
    values_before = cache.layers[0].values.clone()

    assert reference._detach_dynamic_cache(cache) == 2
    torch.testing.assert_close(cache.layers[0].keys, keys_before)
    torch.testing.assert_close(cache.layers[0].values, values_before)
    assert not cache.layers[0].keys.requires_grad
    assert not cache.layers[0].values.requires_grad


class FakeTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(4, 2)
        self.lm_head = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [-1.0, 0.0],
                        [0.0, -1.0],
                    ]
                )
            )
            self.lm_head.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [2.0, 0.0],
                        [0.0, 2.0],
                        [-1.0, -1.0],
                    ]
                )
            )

    def forward(
        self,
        input_ids,
        *,
        past_key_values,
        output_hidden_states,
        logits_to_keep=None,
        **_kwargs,
    ):
        assert output_hidden_states
        past_key_values.length += int(input_ids.shape[1])
        hidden = self.model.embed_tokens(input_ids)
        length = 1 if logits_to_keep == 1 else int(input_ids.shape[1])
        logits = torch.full((1, length, 4), -4.0)
        logits[..., 1] = 4.0
        return SimpleNamespace(logits=logits, hidden_states=(hidden, hidden))


class FakeDraft(nn.Module):
    target_layer_ids = [0]
    block_size = 3
    mask_token_id = 0

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(2))

    def forward(
        self,
        *,
        target_hidden,
        noise_embedding,
        past_key_values,
        **_kwargs,
    ):
        layer = past_key_values.layers[0]
        context = target_hidden.mean(dim=1, keepdim=True)
        cached_context = 0.0
        if layer.keys is not None:
            cached_context = 0.1 * layer.keys.mean(dim=1, keepdim=True)
        output = (noise_embedding + context + cached_context) @ self.weight
        cache_piece = output
        if layer.keys is None:
            layer.keys = cache_piece
            layer.values = cache_piece + 0.0
        else:
            layer.keys = torch.cat((layer.keys, cache_piece), dim=1)
            layer.values = torch.cat((layer.values, cache_piece + 0.0), dim=1)
        past_key_values.length += int(target_hidden.shape[1]) + int(
            noise_embedding.shape[1]
        )
        return output


class FakeLoRADraft(FakeDraft):
    def __init__(self):
        nn.Module.__init__(self)
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(2))

    def forward(
        self,
        *,
        target_hidden,
        noise_embedding,
        past_key_values,
        **_kwargs,
    ):
        layer = past_key_values.layers[0]
        context = target_hidden.mean(dim=1, keepdim=True)
        cached_context = 0.0
        if layer.keys is not None:
            cached_context = 0.1 * layer.keys.mean(dim=1, keepdim=True)
        output = self.fc(noise_embedding + context + cached_context)
        if layer.keys is None:
            layer.keys = output
            layer.values = output + 0.0
        else:
            layer.keys = torch.cat((layer.keys, output), dim=1)
            layer.values = torch.cat((layer.values, output + 0.0), dim=1)
        past_key_values.length += int(target_hidden.shape[1]) + int(
            noise_embedding.shape[1]
        )
        return output


def _extract_context(hidden_states, layer_ids):
    return torch.cat([hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1)


@pytest.mark.parametrize("cache_policy", ["stale", "rebuild"])
def test_round_harness_records_full_drafter_update(cache_policy):
    target = FakeTarget()
    draft = FakeDraft()
    reference.freeze_target(target)
    parameters = reference.configure_trainable_scope(draft, "full-drafter")
    optimizer = torch.optim.Adam(parameters, lr=0.05)
    initial_weight = draft.weight.detach().clone()

    output_ids, rows, summary = reference.run_reference_sequence(
        draft_model=draft,
        target=target,
        input_ids=torch.tensor([[2, 3]]),
        max_new_tokens=2,
        stop_token_ids=None,
        temperature=0.0,
        block_size=3,
        mask_token_id=0,
        mode="full-drafter",
        update_stride=1,
        position_weighting="exponential",
        position_decay_gamma=7.0,
        loss_reduction="weighted-mean",
        proximal_lambda=0.5,
        optimizer=optimizer,
        draft_cache_policy=cache_policy,
        cache_factory=FakeCache,
        extract_context_feature=_extract_context,
        seed=7,
        sample_id="toy",
        provenance=PROVENANCE,
    )

    assert output_ids.shape == (1, 4)
    assert summary["num_output_tokens"] == 2
    assert summary["optimizer_steps"] == len(rows) == 2
    assert all("prefix_len_before" in row for row in rows)
    assert all("prefix_length_before" not in row for row in rows)
    assert all(row["hbm_bytes"] is None for row in rows)
    assert not torch.equal(draft.weight.detach(), initial_weight)
    assert all(row["trainable_scope"] == "full_drafter_all_parameters" for row in rows)
    assert all(row["draft_module_training"] is False for row in rows)
    assert all(row["provenance"] == PROVENANCE for row in rows)
    expected_gradient_history = (
        "detached_truncated"
        if cache_policy == "stale"
        else "full_prefix_recomputed_current_parameters"
    )
    assert all(
        row["gradient_history"] == expected_gradient_history for row in rows
    )
    assert all(row["update"]["applied"] for row in rows)
    assert all(row["update"]["loss"] is not None for row in rows)
    assert all(row["update"]["grad_norm"] > 0.0 for row in rows)
    assert all(row["update"]["backward_cuda_us"] is None for row in rows)
    assert all(row["update"]["optimizer_cuda_us"] is None for row in rows)
    assert all(row["update"]["update_cuda_us"] is None for row in rows)
    assert all(0 <= row["accepted_draft_tokens"] < 3 for row in rows)
    assert all(
        len(row["committed_token_ids"]) == row["acceptance_length"]
        for row in rows
    )
    assert all(len(row["draft_block_token_ids"]) == 3 for row in rows)
    assert all(len(row["target_posterior_token_ids"]) == 3 for row in rows)
    assert all(row["bonus_token_id"] == 1 for row in rows)
    expected_detached = 2 if cache_policy == "stale" else 0
    assert all(
        row["cache_lengths"]["draft_tensors_detached_after_update"]
        == expected_detached
        for row in rows
    )
    assert all(parameter.grad is None for parameter in target.parameters())


def test_first_online_update_cannot_change_greedy_target_output_across_boundaries():
    """An update may change proposal chunking, never the greedy target result.

    Real target kernels can have small query-chunk-dependent numeric drift.  The
    mock makes that drift large and deterministic: a token verified at block
    offset zero has target id 2, while the same logical prefix evaluated later
    in a block has id 1.  Static rejects every draft and therefore always uses
    offset zero.  After the first update, the adaptive drafter proposes id 2
    and changes round boundaries.  A correct reference verifier must still
    commit the canonical one-token greedy trajectory.
    """

    class BoundarySensitiveTarget(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(3, 1)
            self.lm_head = nn.Linear(1, 3, bias=False)
            with torch.no_grad():
                self.model.embed_tokens.weight.fill_(1.0)
                self.lm_head.weight.copy_(torch.tensor([[0.0], [1.0], [-1.0]]))

        def forward(
            self,
            input_ids,
            *,
            past_key_values,
            output_hidden_states,
            logits_to_keep=None,
            **_kwargs,
        ):
            assert output_hidden_states
            past_key_values.length += int(input_ids.shape[1])
            hidden = torch.ones((*input_ids.shape, 1), dtype=torch.float32)
            length = 1 if logits_to_keep == 1 else int(input_ids.shape[1])
            logits = torch.full((1, length, 3), -4.0)
            logits[..., 2] = 4.0
            if length > 1:
                logits[:, 1:, 2] = -4.0
                logits[:, 1:, 1] = 4.0
            return SimpleNamespace(logits=logits, hidden_states=(hidden, hidden))

    class UpdatingDraft(nn.Module):
        target_layer_ids = [0]

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0]))

        def forward(
            self,
            *,
            target_hidden,
            noise_embedding,
            past_key_values,
            **_kwargs,
        ):
            past_key_values.length += int(target_hidden.shape[1]) + int(
                noise_embedding.shape[1]
            )
            return torch.ones_like(noise_embedding) * self.weight

    def run(mode, *, canonical_greedy_verifier=True):
        target = BoundarySensitiveTarget()
        draft = UpdatingDraft()
        reference.freeze_target(target)
        parameters = reference.configure_trainable_scope(draft, mode)
        optimizer = (
            None
            if mode == "static"
            else torch.optim.SGD(parameters, lr=5.0)
        )
        return reference.run_reference_sequence(
            draft_model=draft,
            target=target,
            input_ids=torch.tensor([[0, 0]]),
            max_new_tokens=8,
            stop_token_ids=None,
            temperature=0.0,
            block_size=3,
            mask_token_id=0,
            mode=mode,
            update_stride=2,
            position_weighting="uniform",
            position_decay_gamma=None,
            loss_reduction="weighted-mean",
            proximal_lambda=0.0,
            optimizer=optimizer,
            draft_cache_policy="rebuild",
            cache_factory=FakeCache,
            extract_context_feature=_extract_context,
            seed=7,
            sample_id=f"boundary-{mode}",
            provenance=PROVENANCE,
            canonical_greedy_verifier=canonical_greedy_verifier,
        )

    static_output, static_rows, static_summary = run("static")
    adaptive_output, adaptive_rows, adaptive_summary = run("full-drafter")
    diagnostic_output, _, diagnostic_summary = run(
        "full-drafter", canonical_greedy_verifier=False
    )

    assert adaptive_rows[1]["update"]["applied"] is True
    assert adaptive_rows[0]["draft_block_token_ids"] == (
        static_rows[0]["draft_block_token_ids"]
    )
    assert any(
        row["draft_block_token_ids"] != static_rows[0]["draft_block_token_ids"]
        for row in adaptive_rows[2:]
    )
    torch.testing.assert_close(adaptive_output, static_output, rtol=0, atol=0)
    assert not torch.equal(diagnostic_output, static_output)
    assert diagnostic_summary["exactness"]["selection_eligible"] is False
    assert static_summary["exactness"]["selection_eligible"] is True
    assert adaptive_summary["target_calls"][
        "canonical_commit_verify_decode"
    ] > 0
    assert adaptive_summary["target_calls"]["physical_total"] > (
        adaptive_summary["target_calls"]["block_verify_decode"] + 1
    )
    assert adaptive_rows[1]["proposal_parameter_version"] == 0
    assert adaptive_rows[1]["parameter_version_after_update"] == 1
    assert adaptive_rows[2]["proposal_parameter_version"] == 1
    assert all(
        row["commit_verifier"] == "canonical_greedy_q_len_1"
        and row["canonical_verifier_token_ids"]
        for row in adaptive_rows
    )
    assert any(
        row["accepted_draft_tokens"]
        != row["block_verifier_accepted_draft_tokens"]
        for row in adaptive_rows[2:]
    )


def test_canonical_greedy_verifier_is_default_and_stochastic_is_diagnostic_only():
    parser = reference.build_parser()
    action = next(
        item
        for item in parser._actions
        if item.dest == "canonical_greedy_verifier"
    )
    assert action.default is True

    args = SimpleNamespace(
        lr=1e-4,
        proximal_lambda=0.0,
        update_stride=1,
        max_new_tokens=32,
        temperature=0.7,
        canonical_greedy_verifier=True,
        weight_decay=0.0,
        rank=16,
        mode="tail-lora",
        projection_artifact=None,
        position_weighting="uniform",
        position_decay_gamma=None,
    )
    with pytest.raises(ValueError, match="temperature=0 greedy"):
        reference._validate_args(args)
    args.canonical_greedy_verifier = False
    reference._validate_args(args)


def test_official_parity_reconstructs_block_path_not_canonical_commit(monkeypatch):
    draft = torch.nn.Linear(1, 1, bias=False)
    target = torch.nn.Linear(1, 1, bias=False)
    official = SimpleNamespace(
        output_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        acceptance_lengths=[2],
    )
    module = SimpleNamespace(
        dflash_generate=lambda *args, **kwargs: official,
        extract_context_feature=lambda *args, **kwargs: None,
    )
    observed = []

    def fake_reference_sequence(**kwargs):
        observed.append(kwargs["canonical_greedy_verifier"])
        return official.output_ids.clone(), [], {"acceptance_lengths": [2]}

    monkeypatch.setattr(reference, "run_reference_sequence", fake_reference_sequence)
    result = reference._assert_official_parity(
        module=module,
        draft_model=draft,
        target=target,
        input_ids=torch.tensor([[7]], dtype=torch.long),
        max_new_tokens=2,
        stop_token_ids=None,
        temperature=0.0,
        block_size=2,
        mask_token_id=0,
        cache_factory=lambda: object(),
        seed=0,
        provenance=PROVENANCE,
    )

    assert observed == [False]
    assert result["classification"] == (
        "official_stale_cache_block_verifier_reconstruction"
    )
    assert result["official_policy"] == "stale"
    assert set(result["policies"]) == {"stale"}
    assert all(
        policy["output_ids_match"] and policy["acceptance_lengths_match"]
        for policy in result["policies"].values()
    )


def test_source_point_proximal_gradient_is_zero():
    logits = torch.randn(1, 3, 5, requires_grad=True)
    total, distillation, proximal = reference.tts_kl_objective(
        logits,
        logits.detach(),
        logits.detach(),
        torch.ones(3),
        proximal_lambda=2.0,
        reduction="sum",
    )
    total.backward()
    assert abs(float(distillation.detach())) < 1e-6
    assert abs(float(proximal.detach())) < 1e-6
    assert torch.linalg.vector_norm(logits.grad).item() < 1e-6
    assert math.isfinite(float(total.detach()))


class ToyDrafterWithProjectionFamilies(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(4, 4, bias=False),
                "o_proj": nn.Linear(4, 4, bias=False),
            }
        )
        self.mlp = nn.ModuleDict(
            {
                "up_proj": nn.Linear(4, 8, bias=False),
                "down_proj": nn.Linear(8, 4, bias=False),
            }
        )
        self.unrelated = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)

    def forward(self, value):
        value = self.fc(value)
        value = self.self_attn["o_proj"](self.self_attn["q_proj"](value))
        value = self.mlp["down_proj"](self.mlp["up_proj"](value))
        return self.unrelated(self.norm(value))


def test_drafter_lora_is_zero_effect_and_covers_only_declared_linear_families():
    torch.manual_seed(19)
    draft = ToyDrafterWithProjectionFamilies().to(torch.bfloat16)
    value = torch.randn(2, 4, dtype=torch.bfloat16)
    expected = draft(value).detach().clone()

    parameters = reference.configure_trainable_scope(
        draft, "drafter-lora", rank=2, adapter_seed=37
    )
    actual = draft(value)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    adapters = {
        name: module
        for name, module in draft.named_modules()
        if isinstance(module, reference.DrafterLoRALinear)
    }
    assert set(adapters) == {
        "fc",
        "self_attn.q_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
    assert isinstance(draft.unrelated, nn.Linear)
    assert isinstance(draft.norm, nn.LayerNorm)
    assert len(parameters) == 2 * len(adapters)
    assert all(parameter.requires_grad for parameter in parameters)
    assert all(parameter.dtype == torch.bfloat16 for parameter in parameters)
    for adapter in adapters.values():
        assert adapter.scaling == 1.0
        assert torch.count_nonzero(adapter.lora_a) > 0
        assert torch.count_nonzero(adapter.lora_b) == 0
        assert not adapter.base.weight.requires_grad

    same_seed = ToyDrafterWithProjectionFamilies().to(torch.bfloat16)
    reference.configure_trainable_scope(
        same_seed, "drafter-lora", rank=2, adapter_seed=37
    )
    for left, right in zip(parameters, reference._trainable_parameters(same_seed)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode", ["output-residual", "tail-lora", "full-rank-tail"]
)
def test_tail_modes_are_exact_zero_effect_and_match_declared_layout(mode):
    torch.manual_seed(23)
    head = nn.Linear(128, 131, bias=False).to(torch.bfloat16)
    adapter = reference.TailAdapter.from_target_head(
        mode=mode,
        target_head=head,
        rank=3,
        adapter_seed=0,
    )
    hidden = torch.randn(2, 4, 128, dtype=torch.bfloat16)
    base = head(hidden)

    actual = adapter(hidden, base, head)

    torch.testing.assert_close(actual, base, rtol=0, atol=0)
    layout = adapter.layout()
    assert layout["mode"] == mode
    assert layout["hidden_size"] == 128
    assert layout["vocab_size"] == 131
    assert layout["rank"] == (None if mode == "full-rank-tail" else 3)
    assert layout["has_markov"] is False
    assert layout["has_confidence"] is False
    if mode == "output-residual":
        from lightcone_spec.adapters.projections import build_hidden_projection

        assert tuple(adapter.a_h.shape) == (3, 128)
        assert torch.count_nonzero(adapter.a_h) == 0
        assert tuple(adapter.hidden_projection.shape) == (128, 128)
        assert tuple(adapter.output_basis.shape) == (131, 3)
        torch.testing.assert_close(
            adapter.hidden_projection.cpu(),
            torch.from_numpy(build_hidden_projection(128, 128, seed=0)).to(
                torch.bfloat16
            ),
            rtol=0,
            atol=0,
        )
        projection_identity = layout["projection_identity"]
        assert projection_identity["hidden_projection_seed"] == 0
        assert len(projection_identity["hidden_projection_sha256"]) == 64
        assert len(projection_identity["output_basis_sha256"]) == 64
    elif mode == "tail-lora":
        assert tuple(adapter.a_h.shape) == (128, 3)
        assert tuple(adapter.b_h.shape) == (3, 128)
        assert torch.count_nonzero(adapter.a_h) > 0
        assert torch.count_nonzero(adapter.b_h) == 0
    else:
        assert tuple(adapter.d_h.shape) == (128, 128)
        assert torch.count_nonzero(adapter.d_h) == 0


@pytest.mark.parametrize(
    "mode", ["output-residual", "tail-lora", "full-rank-tail"]
)
def test_tail_update_changes_logits_without_training_drafter(mode):
    torch.manual_seed(29)
    head = nn.Linear(128, 137, bias=False).to(torch.bfloat16)
    adapter = reference.TailAdapter.from_target_head(
        mode=mode,
        target_head=head,
        rank=2,
        adapter_seed=0,
    )
    hidden = torch.randn(1, 3, 128, dtype=torch.bfloat16)
    logits = adapter(hidden, head(hidden), head)
    target_logits = torch.zeros_like(logits)
    target_logits[..., 4] = 4
    named = tuple(adapter.named_parameters())
    optimizer = reference.FP32MasterOptimizer(
        named,
        optimizer_name="adamw",
        lr=0.03,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    before = [parameter.detach().clone() for _, parameter in named]

    evidence = reference.apply_tts_optimizer_step(
        trainable_named_parameters=named,
        optimizer=optimizer,
        proposal_logits=logits,
        target_logits=target_logits,
        position_weights=torch.ones(3),
        proximal_lambda=0.0,
        loss_reduction="weighted-mean",
        optimizer_step=1,
    )

    assert evidence.applied
    assert evidence.parameters_with_grad == len(named)
    assert any(
        not torch.equal(parameter.detach(), old)
        for (_name, parameter), old in zip(named, before)
    )
    accounting = optimizer.memory_accounting()
    assert accounting["master_parameter_bytes"] == sum(
        parameter.numel() * 4 for _, parameter in named
    )
    assert accounting["master_gradient_bytes"] == accounting[
        "master_parameter_bytes"
    ]
    assert accounting["optimizer_moment_bytes"] == 2 * accounting[
        "master_parameter_bytes"
    ]
    assert accounting["forward_gradient_bytes"] == accounting[
        "forward_parameter_bytes"
    ]
    assert accounting["persistent_bytes"] == (
        accounting["forward_parameter_bytes"]
        + accounting["master_parameter_bytes"]
        + accounting["optimizer_moment_bytes"]
    )
    assert accounting["estimated_update_peak_bytes"] == (
        accounting["persistent_bytes"]
        + accounting["forward_gradient_bytes"]
        + accounting["master_gradient_bytes"]
    )
    assert accounting["total_bytes"] == accounting[
        "estimated_update_peak_bytes"
    ]
    assert accounting["parameter_audit_cpu_snapshot_bytes"] == 0


def test_fp32_master_adamw_zero_decay_matches_adam_for_one_step():
    initial = torch.tensor([1.0, -2.0, 0.5], dtype=torch.bfloat16)
    adam_parameter = nn.Parameter(initial.clone())
    adamw_parameter = nn.Parameter(initial.clone())
    adam = reference.FP32MasterOptimizer(
        (("value", adam_parameter),),
        optimizer_name="adam",
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    adamw = reference.FP32MasterOptimizer(
        (("value", adamw_parameter),),
        optimizer_name="adamw",
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    gradient = torch.tensor([0.2, -0.3, 0.4], dtype=torch.bfloat16)
    adam_parameter.grad = gradient.clone()
    adamw_parameter.grad = gradient.clone()

    adam.step()
    adamw.step()

    torch.testing.assert_close(adam_parameter, adamw_parameter, rtol=0, atol=0)
    assert all(
        master.dtype == torch.float32 for master in adam.master_parameters
    )
    assert all(
        state_value.dtype == torch.float32
        for state in adam.inner_optimizer.state.values()
        for key, state_value in state.items()
        if key in {"exp_avg", "exp_avg_sq"}
    )


def test_fp32_master_adamw_applies_nonzero_decay_to_master_and_forward():
    forward = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    optimizer = reference.FP32MasterOptimizer(
        (("value", forward),),
        optimizer_name="adamw",
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1,
    )
    forward.grad = torch.zeros_like(forward)

    optimizer.step()

    torch.testing.assert_close(
        optimizer.master_parameters[0],
        torch.tensor([0.99, -1.98], dtype=torch.float32),
        rtol=1e-6,
        atol=0,
    )
    torch.testing.assert_close(
        forward,
        optimizer.master_parameters[0].to(dtype=torch.bfloat16),
        rtol=0,
        atol=0,
    )


def test_parameter_audit_is_opt_in_and_reports_sample_interval():
    parameter = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    optimizer = reference.FP32MasterOptimizer(
        (("value", parameter),),
        optimizer_name="adam",
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        parameter_audit_enabled=True,
    )
    assert optimizer.memory_accounting()["parameter_audit_cpu_snapshot_bytes"] == 16

    parameter.grad = torch.tensor([0.2, -0.3], dtype=torch.bfloat16)
    optimizer.step()
    first = optimizer.audit_parameters(optimizer_step=1)
    assert first["parameter_audit_interval_steps"] == 1
    assert first["parameter_delta_l2"] > 0.0
    assert first["parameter_displacement_l2"] == pytest.approx(
        first["parameter_delta_l2"]
    )
    assert first["relative_parameter_delta"] > 0.0

    parameter.grad = torch.tensor([-0.1, 0.4], dtype=torch.bfloat16)
    optimizer.step()
    parameter.grad = torch.tensor([0.3, 0.1], dtype=torch.bfloat16)
    optimizer.step()
    third = optimizer.audit_parameters(optimizer_step=3)
    assert third["parameter_audit_interval_steps"] == 2
    assert third["parameter_delta_l2"] > 0.0
    assert third["parameter_displacement_l2"] > 0.0


def test_parameter_audit_disabled_does_not_retain_cpu_snapshots():
    parameter = nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    optimizer = reference.FP32MasterOptimizer(
        (("value", parameter),),
        optimizer_name="adam",
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    assert optimizer.memory_accounting()["parameter_audit_cpu_snapshot_bytes"] == 0
    with pytest.raises(RuntimeError, match="parameter audit was not enabled"):
        optimizer.audit_parameters(optimizer_step=1)


def test_update_evidence_contains_opt_in_parameter_stats():
    layer = nn.Linear(2, 3, bias=False).to(torch.float32)
    named = tuple(layer.named_parameters())
    optimizer = reference.FP32MasterOptimizer(
        named,
        optimizer_name="adam",
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        parameter_audit_enabled=True,
    )
    proposal = layer(torch.tensor([[[1.0, -0.5]]]))
    target = torch.tensor([[[0.0, 2.0, -1.0]]])

    evidence = reference.apply_tts_optimizer_step(
        trainable_named_parameters=named,
        optimizer=optimizer,
        proposal_logits=proposal,
        target_logits=target,
        position_weights=torch.ones(1),
        proximal_lambda=0.0,
        loss_reduction="weighted-mean",
        optimizer_step=1,
        audit_cuda_timing=True,
        audit_parameter_stats=True,
    )

    assert evidence.parameter_delta_l2 is not None
    assert evidence.parameter_delta_l2 > 0.0
    assert evidence.parameter_displacement_l2 == pytest.approx(
        evidence.parameter_delta_l2
    )
    assert evidence.parameter_l2 is not None
    assert evidence.relative_parameter_delta is not None
    assert evidence.parameter_audit_interval_steps == 1
    assert evidence.update_cuda_us is None


def test_total_context_limit_includes_the_pending_dflash_block():
    assert reference.validate_total_context_limit(
        input_tokens=100,
        max_new_tokens=20,
        block_size=16,
        checkpoint_limit=135,
    ) == 135
    with pytest.raises(ValueError, match="prefix plus pending DFlash block"):
        reference.validate_total_context_limit(
            input_tokens=100,
            max_new_tokens=20,
            block_size=16,
            checkpoint_limit=134,
        )


def test_cli_exposes_all_adaptation_modes_and_optimizer_controls():
    mode_action = next(
        action for action in reference.build_parser()._actions if action.dest == "mode"
    )
    assert set(mode_action.choices) == {
        "static",
        "full-drafter",
        "drafter-lora",
        "full-rank-tail",
        "tail-lora",
        "output-residual",
    }
    parser = reference.build_parser()
    optimizer_action = next(
        action for action in parser._actions if action.dest == "optimizer"
    )
    assert set(optimizer_action.choices) == {"adam", "adamw"}
    assert optimizer_action.default == "adam"
    assert next(
        action for action in parser._actions if action.dest == "weight_decay"
    ).default == 0.0
    assert next(
        action for action in parser._actions if action.dest == "audit_cuda_timing"
    ).default is False
    assert next(
        action
        for action in parser._actions
        if action.dest == "parameter_audit_stride"
    ).default == 0
    assert reference.SCHEMA_VERSION == 3


@pytest.mark.parametrize("mode", ["tail-lora", "full-rank-tail"])
def test_round_harness_integrates_cache_safe_tail_without_detaching_kv(mode):
    target = FakeTarget().to(torch.float32)
    draft = FakeDraft().to(torch.float32)
    reference.freeze_target(target)
    reference.configure_trainable_scope(draft, mode, rank=1, adapter_seed=0)
    tail = reference.TailAdapter.from_target_head(
        mode=mode,
        target_head=target.lm_head,
        rank=1,
        adapter_seed=0,
    )
    named = reference._trainable_named_parameters(draft, tail)
    optimizer = reference.FP32MasterOptimizer(
        named,
        optimizer_name="adamw",
        lr=0.05,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    _output_ids, rows, summary = reference.run_reference_sequence(
        draft_model=draft,
        target=target,
        input_ids=torch.tensor([[2, 3]]),
        max_new_tokens=2,
        stop_token_ids=None,
        temperature=0.0,
        block_size=3,
        mask_token_id=0,
        mode=mode,
        update_stride=1,
        position_weighting="uniform",
        position_decay_gamma=None,
        loss_reduction="weighted-mean",
        proximal_lambda=0.0,
        optimizer=optimizer,
        tail_adapter=tail,
        draft_cache_policy="stale",
        cache_factory=FakeCache,
        extract_context_feature=_extract_context,
        seed=7,
        sample_id="tail-toy",
        provenance=PROVENANCE,
    )

    assert summary["mode"] == mode
    assert summary["optimizer"] == "adamw"
    assert summary["optimizer_steps"] == len(rows) == 2
    assert all(
        row["proposal_cache_version"] == "cache_safe_frozen_drafter_history"
        for row in rows
    )
    assert all(
        row["cache_lengths"]["draft_tensors_detached_after_update"] == 0
        for row in rows
    )


def test_round_harness_integrates_drafter_lora_with_frozen_old_kv():
    target = FakeTarget()
    draft = FakeLoRADraft()
    reference.freeze_target(target)
    reference.configure_trainable_scope(
        draft, "drafter-lora", rank=1, adapter_seed=0
    )
    assert isinstance(draft.fc, reference.DrafterLoRALinear)
    frozen_weight = draft.fc.base.weight.detach().clone()
    named = reference._trainable_named_parameters(draft)
    optimizer = reference.FP32MasterOptimizer(
        named,
        optimizer_name="adamw",
        lr=0.05,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    _output_ids, rows, summary = reference.run_reference_sequence(
        draft_model=draft,
        target=target,
        input_ids=torch.tensor([[2, 3]]),
        max_new_tokens=2,
        stop_token_ids=None,
        temperature=0.0,
        block_size=3,
        mask_token_id=0,
        mode="drafter-lora",
        update_stride=1,
        position_weighting="uniform",
        position_decay_gamma=None,
        loss_reduction="weighted-mean",
        proximal_lambda=0.0,
        optimizer=optimizer,
        draft_cache_policy="stale",
        cache_factory=FakeCache,
        extract_context_feature=_extract_context,
        seed=7,
        sample_id="drafter-lora-toy",
        provenance=PROVENANCE,
    )

    assert summary["optimizer_steps"] == len(rows) == 2
    assert torch.equal(draft.fc.base.weight, frozen_weight)
    assert torch.count_nonzero(draft.fc.lora_b) > 0
    assert all(
        row["proposal_cache_version"]
        == "hybrid_pre_update_history_plus_current_round"
        for row in rows
    )
    assert all(
        row["cache_lengths"]["draft_tensors_detached_after_update"] == 2
        for row in rows
    )


def test_output_residual_projection_artifact_build_then_load_is_identical(
    tmp_path, monkeypatch
):
    torch.manual_seed(43)
    head = nn.Linear(128, 131, bias=False).to(torch.bfloat16)
    artifact_path = tmp_path / "qwen-dflash-output-residual.npz"
    binding = {
        "target_revision": "target-rev",
        "draft_revision": "draft-rev",
        "target_head_artifact": {
            "config_sha256": "abc",
            "weight_files": [
                {"name": "model.safetensors", "bytes": 7, "sha256": "a" * 64}
            ],
        },
    }
    built = reference.TailAdapter.from_target_head(
        mode="output-residual",
        target_head=head,
        rank=3,
        adapter_seed=11,
        projection_artifact=artifact_path,
        projection_binding=binding,
    )
    assert artifact_path.is_file()
    assert Path(str(artifact_path) + ".meta.json").is_file()

    def forbidden_rebuild(*_args, **_kwargs):
        raise AssertionError("existing projection artifact must skip the SVD")

    monkeypatch.setattr(reference, "_build_output_basis", forbidden_rebuild)
    loaded = reference.TailAdapter.from_target_head(
        mode="output-residual",
        target_head=head,
        rank=3,
        adapter_seed=11,
        projection_artifact=artifact_path,
        projection_binding=binding,
    )

    torch.testing.assert_close(
        loaded.hidden_projection, built.hidden_projection, rtol=0, atol=0
    )
    torch.testing.assert_close(
        loaded.output_basis, built.output_basis, rtol=0, atol=0
    )
    assert loaded.layout() == built.layout()
    assert loaded.layout()["projection_identity"]["storage"] == "artifact"


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"rank": 4}, "rank"),
        ({"adapter_seed": 12}, "seed"),
        ({"target_revision": "other-target"}, "binding"),
    ],
)
def test_output_residual_projection_artifact_mismatch_fails_closed(
    tmp_path, changed, message
):
    torch.manual_seed(47)
    head = nn.Linear(128, 131, bias=False).to(torch.bfloat16)
    artifact_path = tmp_path / "projection.npz"
    binding = {
        "target_revision": "target-rev",
        "draft_revision": "draft-rev",
        "target_head_artifact": {
            "config_sha256": "abc",
            "weight_files": [
                {"name": "model.safetensors", "bytes": 7, "sha256": "a" * 64}
            ],
        },
    }
    reference.TailAdapter.from_target_head(
        mode="output-residual",
        target_head=head,
        rank=3,
        adapter_seed=11,
        projection_artifact=artifact_path,
        projection_binding=binding,
    )
    rank = changed.get("rank", 3)
    adapter_seed = changed.get("adapter_seed", 11)
    changed_binding = dict(binding)
    if "target_revision" in changed:
        changed_binding["target_revision"] = changed["target_revision"]

    with pytest.raises(ValueError, match=message):
        reference.TailAdapter.from_target_head(
            mode="output-residual",
            target_head=head,
            rank=rank,
            adapter_seed=adapter_seed,
            projection_artifact=artifact_path,
            projection_binding=changed_binding,
        )


def test_irrelevant_projection_artifact_flag_is_rejected():
    args = SimpleNamespace(
        lr=1e-4,
        proximal_lambda=0.0,
        update_stride=1,
        max_new_tokens=32,
        temperature=0.0,
        weight_decay=0.0,
        rank=16,
        mode="tail-lora",
        projection_artifact="unused.npz",
        position_weighting="uniform",
        position_decay_gamma=None,
    )
    with pytest.raises(ValueError, match="only valid with --mode output-residual"):
        reference._validate_args(args)


def test_model_and_tokenizer_content_identities_change_on_same_size_edit(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    shard = model / "model.safetensors"
    shard.write_bytes(b"abcd")
    tokenizer = model / "tokenizer.json"
    tokenizer.write_bytes(b"wxyz")

    model_before = reference._model_artifact_identity(model)
    tokenizer_before = reference._tokenizer_artifact_identity(model)
    shard.write_bytes(b"abce")
    tokenizer.write_bytes(b"wxy0")
    model_after = reference._model_artifact_identity(model)
    tokenizer_after = reference._tokenizer_artifact_identity(model)

    assert model_before["weight_files"][0]["bytes"] == 4
    assert model_after["weight_files"][0]["bytes"] == 4
    assert model_before["weight_files"][0]["sha256"] != (
        model_after["weight_files"][0]["sha256"]
    )
    assert tokenizer_before["content_identity_sha256"] != (
        tokenizer_after["content_identity_sha256"]
    )


def test_rendered_input_identity_binds_exact_token_ids():
    left = reference._rendered_input_token_ids_identity(
        torch.tensor([[1, 2, 3]], dtype=torch.long)
    )
    right = reference._rendered_input_token_ids_identity(
        torch.tensor([[1, 2, 4]], dtype=torch.long)
    )
    assert left["shape"] == [1, 3]
    assert left["serialization"] == "int64_le_c_order_v1"
    assert left["sha256"] != right["sha256"]


def test_runtime_fingerprint_is_complete_on_cpu_without_cuda_sync(monkeypatch):
    def forbidden_sync(*_args, **_kwargs):
        raise AssertionError("runtime fingerprint must not synchronize CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden_sync)
    fingerprint = reference.build_runtime_fingerprint(
        device=torch.device("cpu"),
        dtype="float32",
        attention_implementation="sdpa",
    )
    assert fingerprint["schema_version"] == 1
    assert fingerprint["gpu"] is None
    assert fingerprint["cuda_driver_version"] is None
    assert fingerprint["dtype"] == "float32"
    assert fingerprint["device"] == "cpu"
    assert set(fingerprint["allocator_config"]) == {
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
    }


def test_deterministic_contract_is_default_and_fingerprinted(monkeypatch):
    parser = reference.build_parser()
    deterministic_action = next(
        action for action in parser._actions if action.dest == "deterministic"
    )
    assert deterministic_action.default is True

    prior_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    prior_algorithms = torch.are_deterministic_algorithms_enabled()
    warn_only_query = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None
    )
    prior_warn_only = (
        bool(warn_only_query()) if callable(warn_only_query) else False
    )
    prior_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prior_benchmark = torch.backends.cudnn.benchmark
    prior_cudnn_deterministic = torch.backends.cudnn.deterministic
    prior_precision = torch.get_float32_matmul_precision()
    prior_sdpa = {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
    }
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    try:
        contract = reference.configure_determinism(True)
        assert contract == reference.determinism_contract(True)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        fingerprint = reference.build_runtime_fingerprint(
            device=torch.device("cpu"),
            dtype="float32",
            attention_implementation="sdpa",
            deterministic=True,
        )
        assert fingerprint["determinism_contract"] == contract
        assert fingerprint["cudnn_deterministic"] is True
        assert fingerprint["allow_tf32"] == {"matmul": False, "cudnn": False}
        assert fingerprint["sdpa_backends"] == {
            "flash": False,
            "memory_efficient": False,
            "math": True,
            "cudnn": False,
        }
    finally:
        if prior_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prior_workspace
        torch.use_deterministic_algorithms(
            prior_algorithms, warn_only=prior_warn_only
        )
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32
        torch.backends.cudnn.benchmark = prior_benchmark
        torch.backends.cudnn.deterministic = prior_cudnn_deterministic
        torch.set_float32_matmul_precision(prior_precision)
        torch.backends.cuda.enable_flash_sdp(prior_sdpa["flash"])
        torch.backends.cuda.enable_mem_efficient_sdp(
            prior_sdpa["memory_efficient"]
        )
        torch.backends.cuda.enable_math_sdp(prior_sdpa["math"])
        torch.backends.cuda.enable_cudnn_sdp(prior_sdpa["cudnn"])


def test_deterministic_direct_harness_always_rejects_initialized_cuda(
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        reference.configure_determinism(True)


def test_command_attestation_binds_ordered_harness_argv():
    identity_sha256 = "1" * 64
    unsigned = [
        "--mode",
        "static",
        "--run-identity-sha256",
        identity_sha256,
    ]
    command_sha256 = reference._argv_sha256(
        [str(SCRIPT.resolve()), *unsigned]
    )
    signed = [*unsigned, "--command-sha256", command_sha256]
    assert reference._command_attestation(
        signed,
        run_identity_sha256=identity_sha256,
        command_sha256=command_sha256,
    ) == {
        "status": "runner_bound",
        "scheme": reference.COMMAND_SHA256_SCHEME,
        "run_identity_sha256": identity_sha256,
        "command_sha256": command_sha256,
    }

    tampered = ["--mode", "full-drafter", *signed[2:]]
    with pytest.raises(ValueError, match="command sha256 mismatch"):
        reference._command_attestation(
            tampered,
            run_identity_sha256=identity_sha256,
            command_sha256=command_sha256,
        )


def test_projection_artifact_without_target_shard_hashes_fails_closed(tmp_path):
    head = nn.Linear(128, 131, bias=False).to(torch.bfloat16)
    with pytest.raises(ValueError, match="target LM-head weight shards"):
        reference.TailAdapter.from_target_head(
            mode="output-residual",
            target_head=head,
            rank=3,
            adapter_seed=11,
            projection_artifact=tmp_path / "projection.npz",
            projection_binding={
                "target_revision": "target-rev",
                "target_head_artifact": {"weight_files": []},
            },
        )
