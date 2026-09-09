"""Excluded, single-change runtime candidates. Never installed by formal runners.

Each transformation is pinned to a reviewed source block and refuses a changed
baseline. A successful microcheck is not end-to-end performance acceptance.
"""

CANDIDATES = ("kv_index_select", "single_mask_index", "paired_logits_gather", "logical_prefix_write")
ADAPTER = "python/sglang/srt/speculative/dflash_online_adaptation.py"
RUNTIME = "python/sglang/srt/speculative/online_adaptation_runtime.py"


def transform(source, candidate):
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"{candidate}: reviewed source block not unique")
        source = source.replace(old, new, 1)

    if candidate == "kv_index_select":
        replace("return buffer[locations]", "return torch.index_select(buffer, 0, locations.reshape(-1)).reshape(\n"
                "                    *locations.shape, *buffer.shape[1:]\n                )")
    elif candidate == "single_mask_index":
        replace("        inference = inference[selected]\n        reconstructed = reconstructed[selected]",
                "        indexes = selected.reshape(-1).nonzero(as_tuple=True)[0]\n"
                "        inference = inference.reshape(-1, inference.shape[-1]).index_select(0, indexes)\n"
                "        reconstructed = reconstructed.reshape(-1, reconstructed.shape[-1]).index_select(0, indexes)")
    elif candidate == "paired_logits_gather":
        replace("                    inference_logits = self._full_vocab_logits(\n"
                "                        local_inference_logits, target.shape[-1]\n                    )\n", "")
        replace("                    active_logits = self._full_vocab_logits(\n"
                "                        local_active_logits, target.shape[-1]\n                    )",
                "                    inference_logits, active_logits = self._full_vocab_logits(\n"
                "                        torch.stack((local_inference_logits, local_active_logits)),\n"
                "                        target.shape[-1],\n                    ).unbind(0)")
    elif candidate == "logical_prefix_write":
        replace("        row[:, 0].copy_(prefix_lens.detach().to(torch.int64))\n"
                "        for index, committed_prefix in enumerate(committed_prefixes):\n"
                "            if committed_prefix is not None:\n"
                "                row[index, 0].copy_(committed_prefix)",
                "        if all(value is None for value in committed_prefixes):\n"
                "            row[:, 0].copy_(prefix_lens.detach().to(torch.int64))\n"
                "        else:\n"
                "            for index, committed_prefix in enumerate(committed_prefixes):\n"
                "                row[index, 0].copy_(\n"
                "                    prefix_lens[index].detach() if committed_prefix is None\n"
                "                    else committed_prefix\n                )")
    else:
        raise ValueError("unknown excluded candidate")
    compile(source, candidate, "exec")
    return source
