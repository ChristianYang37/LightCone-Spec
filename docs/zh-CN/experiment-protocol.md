# 实验协议

[English](../en/experiment-protocol.md) · [首页](../../README_zh-CN.md)

实验在相同 prompt ID、prefix checkpoint、sampling profile、seed、context bucket 和
load cell 上配对比较。context length 定义为每次 proposal 前的真实 prefix length。
按 prompt 分组，防止同一请求族跨越 controller 的 train、calibration 与 test split。

主要 acceptance 指标是 survival-weighted accepted prefix，而非 accepted/verified
draft 比率。同时必须保留 committed tokens/verification、verification waste、target
calls/output token、decode goodput、streaming latency、各 CUDA 组件耗时、peak device
memory、retraction、version mismatch 与 exactness violation。

详细 profiling 与 headline throughput 分开运行；hot path 不得逐轮 synchronize。
不能把单独的 acceptance 改善表述为加速，algorithmic 与 engineering gate 分开报告。
本仓库有意不包含结果 artifact。
