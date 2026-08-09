# 故障排查

[English](../en/troubleshooting.md) · [首页](../../README_zh-CN.md)

## Patch 或 runtime 身份失败

- 核对完整 upstream commit、detached clean 状态、patch digest 和
  `patches/sglang/manifest.json` 中的 expected tree；
- 开始负载筛选或调参后不得编辑 runtime。源码、包、模型、sampling 或启动参数发生
  drift 都需要新的 selection 和 evidence root；
- 测试不得隐式引用工作区 `sglang/`；必须显式传入 clean upstream 或一次性 patched
  checkout。

## 模型或配置被拒绝

- Upstream revision 改变时重新生成 model lock。Model-roots 文件必须绑定该精确 lock，
  且选择的每个本地目录都必须存在；
- DFlash adaptation 拒绝大于一的 TP/DP、量化 drafter path、block/canvas 不匹配、
  未支持的 speculative 选项和未显式保留足够 HBM 的配置；
- 正式 Static/TTS/L0 endpoint 必须显式启用 speed-study metric 与 exact rejection
  sampling。若 exact kernel 不可用，应修复环境，不能改成 greedy DFlash fallback；
- Static 不能含 adaptation object。TTS 与 L0 的 adaptation 字段和选择的 concurrency
  必须完全相同。

## 显存压力

Adaptation state 在 KV pool 之前计算。它常驻 GPU、不可驱逐、不自动 offload，也不会
静默从 Full 改成 LoRA 或 tail。应降低 admission、context 或显式选择的参数布局；由于
runtime 身份已经改变，随后必须重新执行负载筛选。

OOM 或 retraction 是正式实验的安全事件。不得删除计数、单纯增加 timeout，或把失败请求
从分母移除。

## Confirmation 中断

对同一不可变 root 重跑完全相同的 `run-confirmation` 命令。已完成的 hash-bound receipt
会先验证再跳过。缺少 terminal receipt 的 Parquet shard 属于中断 attempt，不会进入
统计。如果 receipt 损坏、重复或绑定另一身份，保留目录用于审计并创建新的 evidence
root，不要手工编辑。

## 意外的 `UNMEASURED` 或 `BLOCKED`

`UNMEASURED` 表示没有提供有效 GPU attestation，即使诊断算术为正也不例外。
`BLOCKED` 表示存在 attested GPU evidence，但至少一个注册的速度、区间、安全、覆盖或
发布条件失败。两者都是合法结果，不得重新命名为成功。

报告 bug 时请提供脱敏命令、包与平台版本、精确 upstream/patched tree ID 和最小复现
配置。不得公开 token、密码、私有 prompt、实例地址或机器专用模型路径。
