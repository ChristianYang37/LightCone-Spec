"""Opt-in, bounded long-generation diagnostics; never retain vocabulary logits."""

from __future__ import annotations

import torch


class MechanismRecorder:
    def __init__(self, bin_tokens: int = 2048):
        if bin_tokens <= 0:
            raise ValueError("mechanism bin size must be positive")
        self.bin_tokens = bin_tokens
        self.reset()

    def reset(self):
        self.rows: dict[tuple[str, int], dict] = {}
        self.endings: dict[str, str] = {}
        self.prompt_lengths: dict[str, int] = {}

    def finish(self, rid: str, *, natural_stop: bool):
        self.endings[rid] = "natural_stop" if natural_stop else "length_or_abort"

    @torch.no_grad()
    def record(
        self,
        *,
        request_ids,
        generated_offsets,
        teacher_logits,
        draft_logits,
        valid_mask,
        accepted_drafts,
        committed_tokens,
        prompt_lengths,
    ):
        if (
            teacher_logits.shape != draft_logits.shape
            or teacher_logits.shape[:2] != valid_mask.shape
        ):
            raise ValueError("mechanism teacher/draft/owned-mask alignment mismatch")
        batch, width = valid_mask.shape
        if len(request_ids) != batch or len(generated_offsets) != batch:
            raise ValueError("mechanism request ownership mismatch")
        if batch != 1:
            raise ValueError("registered mechanism diagnostics require matched c1")
        # Compute one request at a time. Only scalar sums/counts survive this call.
        for index, rid in enumerate(request_ids):
            self.prompt_lengths[rid] = int(prompt_lengths[index])
            positions = torch.nonzero(valid_mask[index], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            teacher = teacher_logits[index, positions].float()
            draft = draft_logits[index, positions].float()
            if not bool((torch.isfinite(teacher).all() & torch.isfinite(draft).all()).item()):
                raise ValueError("nonfinite logits in owned mechanism positions")
            logp = teacher.log_softmax(-1)
            entropy = -(logp.exp() * logp).sum(-1)
            ce = -draft.log_softmax(-1).gather(-1, teacher.argmax(-1, keepdim=True)).squeeze(-1)
            accepted = int(accepted_drafts[index])
            if not 0 <= accepted <= width:
                raise ValueError("mechanism accepted count exceeds draft width")
            offset = int(generated_offsets[index])
            for pos, ent, loss in zip(
                positions.tolist(), entropy.tolist(), ce.tolist(), strict=True
            ):
                bucket = (offset + pos) // self.bin_tokens
                row = self.rows.setdefault(
                    (rid, bucket),
                    {
                        "request_id": rid,
                        "position_start": bucket * self.bin_tokens,
                        "position_end": (bucket + 1) * self.bin_tokens,
                        "exposed_draft_positions": 0,
                        "target_entropy_sum": 0.0,
                        "draft_top1_ce_sum": 0.0,
                        "target_calls": 0,
                        "accepted_drafts": 0,
                        "committed_tokens": 0,
                        "updates_published": 0,
                        "positions": {},
                    },
                )
                row["exposed_draft_positions"] += 1
                row["target_entropy_sum"] += ent
                row["draft_top1_ce_sum"] += loss
                counts = row["positions"].setdefault(
                    str(pos + 1), {"exposed": 0, "reached": 0, "accepted": 0}
                )
                counts["exposed"] += 1
                counts["reached"] += int(accepted >= pos)
                counts["accepted"] += int(accepted > pos)
            # Whole verification rounds are attributed to their starting bin.
            row = self.rows[(rid, (offset + int(positions[0])) // self.bin_tokens)]
            row["target_calls"] += 1
            row["accepted_drafts"] += accepted
            row["committed_tokens"] += int(committed_tokens[index])

    def snapshot(self, updates=()):
        # Publication attribution uses the source-prefix position, not a claimed
        # wall-clock publication position. Native traces retain both round IDs.
        publications = {}
        for update in updates:
            if update.get("status") != "published":
                continue
            for rid, prefix in zip(update["request_ids"], update["prefix_len_before"], strict=True):
                if rid in self.prompt_lengths:
                    key = (rid, max(0, prefix - self.prompt_lengths[rid]) // self.bin_tokens)
                    publications[key] = publications.get(key, 0) + 1
        return {
            "measurement_scope": "long_generation_mechanism_diagnostic",
            "mechanism_bin_tokens": self.bin_tokens,
            "mechanism_bins": [
                {
                    **row,
                    "request_ending": self.endings.get(rid, "in_progress"),
                    "updates_published": publications.get((rid, bucket), 0),
                    "publication_position_basis": "source_prefix",
                }
                for (rid, bucket), row in sorted(self.rows.items())
            ],
        }
