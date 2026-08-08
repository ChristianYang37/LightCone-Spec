# Experiment protocol

[中文](../zh-CN/experiment-protocol.md) · [Home](../../README.md)

Experiments compare methods on identical prompt IDs, prefix checkpoints,
sampling profiles, seeds, context buckets, and load cells. Context length is
the true prefix length immediately before each proposal. Prompt-level grouping
prevents a request family from crossing controller train, calibration, and test
splits.

The primary acceptance quantity is the survival-weighted accepted prefix, not
the ratio of accepted to verified drafts. Reports must also retain committed
tokens per verification, verification waste, target calls per output token,
decode goodput, streaming latency, CUDA component time, peak device memory,
retractions, version mismatches, and exactness violations.

Detailed profiling and headline throughput are separate runs. The hot path must
not synchronize once per round. A method is never described as faster from an
acceptance-only change; algorithmic and engineering gates are reported
separately. Result artifacts are intentionally not included in this repository.
