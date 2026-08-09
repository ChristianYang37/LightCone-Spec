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
| `optimizer.name` | `adam`、`adamw`、`sgdm`、`nag`、`muon` 或 `lion` |
| `rank` | residual/LoRA 必须显式给出；full 为 `null` |
| `stride` | 正整数 |
| `max_in_flight` | 只能为一 |

Residual 仅允许 tail。Drafter Full/LoRA 仅用于 DFlash，要求未量化、TP=DP=1，且 canvas
宽度等于 speculative block size。DSpark 和 EAGLE/EAGLE3 只接受 tail scope。Adapted
DSpark 必须使用 verify-all；adapted EAGLE/EAGLE3 必须 one layer、fixed proposal
depth、top-k one、无 token-map remapping，并使用 exact full-vocabulary rejection
sampling。Target embedding、target LM head 与 target model 始终冻结。

Static 必须使用 `adaptation: null`。这是 fast path 的语义要求：不得创建 optimizer、
gradient、trace、candidate 或 adaptation-reserve allocation。

正式 launcher 会为三个 endpoint 显式加入 `--speculative-speed-study-metrics` 和 exact
rejection sampling。没有该实验开关时，原生 SGLang 不分配 LightCone 指标状态；启用后则
必须使用 DFlash、rejection sampling，并将两个 acceptance threshold 设为一。缺少 exact
kernel 时直接报错，绝不退回 greedy decoding。

## Optimizer 合同

所有在线 optimizer 都是 functional 的：side stream 在不修改 active optimizer 的
前提下生成 candidate parameter 与 candidate state，只有 TTS 或 L0 的发布策略提交后
才成为 active state。配置严格拒绝没有实际作用的字段：

| Optimizer | 必需字段 | Decay 与 resident state |
|---|---|---|
| `adam` | learning rate、betas、epsilon | 不支持 weight decay；FP32 一阶与二阶动量 |
| `adamw` | learning rate、betas、epsilon | decoupled decay；FP32 一阶与二阶动量 |
| `sgdm` | learning rate、`momentum` | coupled weight decay；一个 FP32 momentum buffer |
| `nag` | learning rate、`momentum` | PyTorch-style Nesterov 与 coupled weight decay；一个 FP32 momentum buffer |
| `lion` | learning rate、betas | decoupled decay；一个 FP32 momentum buffer |
| `muon` | learning rate、`momentum`、`muon_ns_steps`、辅助 AdamW learning rate 与 decay | 二维 tensor 使用 Muon；非 matrix tensor 使用辅助 AdamW 与两个 moment |

`grad_clip` 对整个 candidate parameter list 做全局裁剪。未使用 momentum 的 optimizer
会拒绝该字段；Muon 专属字段在其他 optimizer 中也会被拒绝。Adam 未使用的 weight
decay 与 Lion 未使用的 epsilon 变体同样不能形成伪 tuning identity。Static 不创建
optimizer。Plain `sgd` 只保留给隔离的 OnlineSpec baseline，不是 TTS/L0 的可选项。

HBM 账本包含 FP32 master、所有真实分配的 moment 和 device step scalar。空 moment 不
分配：SGDm、NAG 与 Lion 不承担第二个 state tensor；Muon 只对交给辅助 AdamW 的非
matrix 参数保留两个 moment。

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
