"""Excluded state fingerprints; copies stay on the server and timings are invalid."""

import hashlib
import json

import torch


def tensor_digest(tensor):
    """Read exact bytes in bounded host chunks, including BF16; never retain weights."""
    if not tensor.is_contiguous():
        raise ValueError("excluded digest requires reviewed contiguous tensor layout")
    raw = tensor.detach().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for start in range(0, raw.numel(), 16 * 1024 * 1024):
        digest.update(raw[start:start + 16 * 1024 * 1024].cpu().numpy().tobytes())
    return digest.hexdigest()


def parameter_digests(model):
    rows = {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                   "sha256": tensor_digest(tensor)}
            for name, tensor in model.named_parameters()}
    if not rows:
        raise ValueError("target exposes no parameters")
    return rows


def assert_disjoint_target_storage(model, adapter):
    """Address ranges, not just data_ptr equality: reject offset views as well."""
    def span(tensor):
        storage = tensor.untyped_storage()
        return str(tensor.device), storage.data_ptr(), storage.data_ptr() + storage.nbytes()

    targets = [span(tensor) for tensor in model.parameters()]

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from tensors(item)

    # Include publication destinations and optimizer-owned nested tensor state.
    mutable = (adapter.inference.active, adapter.inference.staging, vars(adapter.optimizer))
    count = 0
    for tensor in tensors(mutable):
        device, start, end = span(tensor)
        if any(device == d and start < hi and lo < end for d, lo, hi in targets):
            raise RuntimeError("adaptation tensor aliases target parameter storage")
        count += 1
    if not count:
        raise ValueError("adaptation exposes no auditable tensors")
    return count


def target_prefix_kv(values, row):
    """Only existing committed prefix slots, across all target layers (Qwen MHA)."""
    worker, batch = values["self"], values["batch"]
    pool = worker.model_runner.token_to_kv_pool
    if pool.__class__.__name__ != "MHATokenToKVPool":
        raise ValueError("excluded target KV audit supports unquantized MHA only")
    if pool.is_quantized_kv_cache or pool.layer_transfer_counter is not None:
        raise ValueError("quantized/offloaded KV needs its own reviewed audit")
    length = int(values["prefix_lens"][row].item())
    if not 0 < length <= 2048:
        raise ValueError("excluded target KV audit prefix exceeds its bound")
    index = int(batch.req_pool_indices[row].item())
    locations = worker.model_runner.req_to_token_pool.req_to_token[index, :length].clone().long()
    result = {"locations": locations.cpu().tolist(), "layers": {}}
    # ModelRunner may pass an exclusive end_layer, while KVCache's default is
    # inclusive. The actual allocated layer count is the unambiguous authority.
    for layer in range(pool.start_layer, pool.start_layer + pool.layer_num):
        pair = pool.get_kv_buffer(layer)
        if any(buffer.ndim != 3 for buffer in pair):
            raise ValueError("excluded target KV audit requires token-major NHD")
        result["layers"][str(layer)] = [tensor_digest(buffer[locations].contiguous()) for buffer in pair]
    return result


class TargetStateAudit:
    def __init__(self, worker, output):
        self.worker, self.output = worker, output
        self.model = worker.target_worker.model_runner.model
        self.adapter = worker._online_drafter_adapter
        self.done = False
        self.mutable_tensors = assert_disjoint_target_storage(self.model, self.adapter)
        torch.cuda.synchronize(worker.device)
        self.initial = parameter_digests(self.model)

    def before_update(self, values, row):
        torch.cuda.synchronize(self.worker.device)
        self.before_kv = target_prefix_kv(values, row)
        self.round = self.adapter.runtime.round

    def after_update(self, values, row):
        torch.cuda.synchronize(self.worker.device)
        final = parameter_digests(self.model)
        after_kv = target_prefix_kv(values, row)
        unchanged = final == self.initial
        kv_unchanged = after_kv == self.before_kv
        record = {"measurement_scope": "excluded_target_state_audit",
                  "target_parameters": self.initial, "final_target_parameters": final,
                  "target_parameters_unchanged": unchanged,
                  "committed_prefix_kv_unchanged_during_update": kv_unchanged,
                  "prefix_kv_before": self.before_kv, "prefix_kv_after": after_kv,
                  "round": self.round, "mutable_tensors_disjoint": self.mutable_tensors,
                  "scope": "target params since first decode; one due update's committed prefix KV"}
        self.output.write_text(json.dumps(record, indent=2))
        self.done = True
        if not unchanged or not kv_unchanged:
            raise RuntimeError("excluded audit observed a target parameter or committed KV mutation")
