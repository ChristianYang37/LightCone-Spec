# 配置

[English](../en/configuration.md) · [首页](../../README_zh-CN.md)

## Schema 身份

所有 run config 使用 schema version 2，并拒绝未知字段。正式方法只有 `static`、`tts`
和 `naive_async`。三个隔离的 OnlineSpec 标识只用于外部 baseline 核验，不能替代正式
速度实验中的方法。

每个真实 run 都绑定不可变的 target/drafter revision、固定 SGLang commit、sampling
profile digest、tenant 与 runtime load。旧 schema 或已删除的方法名会在模型加载前作为
未知输入拒绝。

## Adaptation 配置

TTS 与 L0 共享完全相同的 adaptation object：

| 字段 | 允许的合同 |
|---|---|
| `weight_update_mode` | `residual`、`lora` 或 `full` |
| `parameter_scope` | `tail` 或 `drafter` |
| `kv_history_policy` | 只能是 `frozen` |
| `adaptation_scope` | 只能是 `cohort` |
| `adaptation_group_id` | 显式、非空的实验组身份 |
| `optimizer.name` | TTS/L0 使用 `adam` 或 `adamw` |
| `rank` | residual/LoRA 必须显式给出；full 为 `null` |
| `stride` | 正整数 |
| `max_in_flight` | 只能为一 |

Residual 仅允许 tail。Drafter Full/LoRA 当前只认证未量化 DFlash、TP=DP=1，且 canvas
宽度必须等于 speculative block size。Target embedding、target LM head 与 target model
始终冻结。

Static 必须使用 `adaptation: null`。这是 fast path 的语义要求：不得创建 optimizer、
gradient、trace、candidate 或 adaptation-reserve allocation。

正式 launcher 会为三个 endpoint 显式加入 `--speculative-speed-study-metrics` 和 exact
rejection sampling。没有该实验开关时，原生 SGLang 不分配 LightCone 指标状态；启用后则
必须使用 DFlash、rejection sampling，并将两个 acceptance threshold 设为一。缺少 exact
kernel 时直接报错，绝不退回 greedy decoding。

## Cohort 身份

只有 target/drafter revision、算法、sampling profile、tenant、实验组、参数布局与
optimizer 配置全部相同的请求才能共享更新。每个活跃请求只贡献最近一次合法监督。
Cancellation、epoch rollover、slot reuse 或 source-version conflict 都会使 candidate 失效。

## Runtime 渲染

`render-runtime` 消费锁定的 selection artifact，输出三份匹配的 run config 和仅含 argv
的启动计划。Adaptation reserve 与 Static KV memory fraction 必须由硬件预检显式提供，
源码不设置默认值。生成的 runtime 文件和绝对模型路径应位于 ignored artifact 目录，
不得提交。

除 `method` 字段外，TTS 与 L0 配置必须逐项一致。修改超参数、sampling profile、load、
模型 revision 或 runtime tree 后，必须创建新的 selection 和 evidence root。
