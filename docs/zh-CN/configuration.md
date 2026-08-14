# 配置

[English](../en/configuration.md) · [首页](../../README_zh-CN.md)

## Schema 身份

Run config 使用 schema version 3，验证后不可变，并拒绝未知字段。Runtime method 为
`target_only`、`static`、`tts` 与 `l0`。Recipe authority 从中派生五种科学角色：Target-only、
Static、TTS（`tts` 加 frozen TTS authority）、L0-naive（`l0` 加同一个 authority）与
LightCone（`l0` 加准确 sealed E2 recipe receipt）。E1/E2 search 中的 `l0` declaration 是
LC-candidate。隔离的 OnlineSPEC 对比另有
`onlinespec_ogd`、`onlinespec_opt` 与 `onlinespec_ens`；它使用独立 selection 与证据
身份，不能替换核心方法。

每个 run 绑定准确 target/drafter revision、backend algorithm、context/draft depth、固定
SGLang commit、sampling-profile SHA-256、tenant 与 runtime topology。Target-only 要求
`speculation_enabled=false`；其他方法必须启用 speculation，且 verification width 等于
draft depth 加一。未知 schema 与已退役 adaptation 字段会在加载模型前失败。

这些名称包含目标 protocol vocabulary，不承诺每个合法 scientific declaration 都可执行。
当前端到端 industrial executor 只接受 TP1/DP1 Target-only。Static/TTS/L0-naive/LightCone 会在任何
mutation 前被阻止，因为固定 native begin/reset/finalize hook 没有配置 trusted hardware
signer。底层 adaptive patch path 与 terminal schema 仅限 TP1/DP1 DFlash；adaptive runtime 会执行
已注册 constant、按 published update 的 inverse-square-root 与有限 horizon cosine schedule，
也会执行非负 logical publication delay。`quota_shadow` 可以声明，但 DFlash 当前只提供
update-round teacher acquisition，因此真实缺 row 时会 fail closed。

每个 serving run 还会绑定 schema-v2 controlled execution policy。注册策略固定 context
length 40,960 与 server seed 1，关闭 radix cache 和 CUDA Graph，并要求
non-incremental output。Target-reference role 关闭 overlap scheduling，speculative role
保留 overlap。Native begin/reset/finalize 会重验同一策略；reset 必须保持 seed 1，不能根据
run nonce 派生新 seed。这些值是 preliminary GPU snapshot 实际使用的实验身份，不表示其已被
证明吞吐最优。

## 方法与 Disabled Path 合同

Target-only 与 Static schema config 都要求 `adaptation: null` 和 `online_spec: null`。
Target-only 启动无 speculation 的 target path；Static 描述原生 speculative decoding。
二者都不能分配 optimizer、gradient、master、candidate 或 cohort-adaptation state。当前
只有 Target-only 可端到端执行。Static 必须保持 round/update 详细 trace 零分配，同时仍需
content-bound request、performance 与汇总 speculative safety evidence，所以无法通过
trusted-signer release preflight。

Recipe 与 publication identity 正交。TTS 使用 frozen、primary-source-bound recipe 与 fixed
barrier；L0-naive 使用同一个 frozen recipe authority 与 first-ready safe-boundary
publication。LC-candidate 与 LightCone 都使用 L0 policy，但 recipe 分别是已注册 E1/E2
search recipe 与准确 sealed E2 winner。共享 reconstruction、candidate lifecycle 或 evidence
代码不代表 live config、candidate、optimizer state 或 row 相同。只有受控 replay 的
source-state 与 proposal-evidence digest 都相同时才比较 candidate byte。必须使用准确
full-vocabulary rejection sampling；注册 kernel 不存在时必须报错，不能静默退回 greedy。

## Adaptation 对象

| 字段 | Schema-v3 合同 |
|---|---|
| `weight_update_mode` | `full` 或 `lora` |
| `parameter_scope` | `last1`、`last3`、`last5`、`all`；DSpark 另允许三个 `*_native_heads` scope |
| `kv_history_policy` | 只能为 `frozen` |
| `adaptation_scope` | 只能为 `cohort` |
| `adaptation_group_id` | 显式非空 cohort 身份 |
| `rank`、`lora_alpha` | Full 时均为 `null`；LoRA 时为相同注册 rank，使 `alpha/r=1` |
| `lora_matrix_policy` | 只能为 `registered_matrices_v1` |
| `native_head_policy` | layer-only 为 `frozen`；DSpark hybrid 为 `full` |
| `stride` | 正整数 |
| `max_in_flight` | 只能为一 |
| `canvas_tokens` | 等于 speculative verification width |
| `teacher_row_policy` | `update_round` 或已注册 `quota_shadow` |

LoRA rank 恰为 1、2、4、8、16、32、64。LoRA plan 只选择已注册二维 native matrix，初始
functional delta 为零。Full 选择命名 native layer scope 中全部合格浮点参数。借用的 target
embedding、target LM head 与 target model 始终冻结。量化或不属于 backend 的可训练
coordinate 会在 preflight 失败。

Trainable-plan digest 覆盖 selected/frozen parameter、shape、dtype、parameterization、LoRA
rank/alpha 以及 sharded/replicated ownership。改变任一值都会产生新的 config、memory plan、
selection 与证据身份。

## Backend 专属字段

本节描述目标 contract。当前可执行 schema 的 adaptive config 必须使用 DFlash；DSpark、
EAGLE、EAGLE3 与 NEXTN 会在 model loading 前被拒绝。DFlash 的目标 contract 使用 native
differentiable-canvas evidence。EAGLE/EAGLE3 adaptation 将要求
`speculative_eagle_topk=1`，并钉住一个 proposal source version。NEXTN 将要求单独 preflight
的 native interface digest；registry E6 还以双 rank memory-fit receipt 为目标。这些
prerequisite 不会让 cell 在当前 schema 中变得可执行。

DSpark layer-only scope 冻结 W1、W2 与 acceptance/confidence state。Hybrid scope
`last1_native_heads`、`last3_native_heads`、`last5_native_heads` 选择命名 backbone scope，
同时把 W1、W2 与 scalar native acceptance/confidence parameter 作为 Full replicated state
训练。Hybrid 要求非空 `confidence_loss_weight`；layer-only plan 则拒绝该字段。

只有 `fixed_verification_budget` 存在时，`verification_mode` 才能是 `fixed_budget`；否则为
`native_scheduler`。Fixed-budget phase 是 tuning control，confirmation 使用 native
scheduler。Proposal cross-entropy 与 proper confidence loss 使用真实 native Markov
feature 和实际 sampled predecessor，配置不能用重建 placeholder feature 替代。

E1a registry 恰有 56 个 adaptive configuration：四种 layer-only scope 与三种 hybrid
scope，各自交叉 Full 加七个 LoRA rank。

## Optimizer 合同

E1/E2 **LC-candidate** optimizer registry 包含 `adam`、`adamw`、`sgdm`、`nag`、`muon` 与
`lion`。它们生成 functional parameter/state proposal，只有 commit 时才修改 active state。
TTS 不搜索该 registry：paper-bound authority 指定 Adam 与一次 step，但 numeric learning
rate/schedule、beta、epsilon、optimizer-state reset、weight decay/clip、position weight、
proximal coefficient、loss normalization/precision/temperature、准确 trainable-parameter
manifest、stride-selection rule 与官方 implementation commit 尚未解析。因此
`TTS-paper-reconstruction` 与 L0-naive
保持 `BLOCKED`；二者不能继承 schema default、历史 AdamW recipe 或 E1/E2 selection。Plain
`sgd` 只用于 OnlineSPEC。

| Optimizer | 必需身份 | 常驻 moment 规则 |
|---|---|---|
| Adam | learning rate、beta、epsilon | FP32 一阶与二阶 moment；无 decay |
| AdamW | learning rate、beta、epsilon、decay | FP32 一阶与二阶 moment；decoupled decay |
| SGDm | learning rate、momentum、可选 decay | 一个 FP32 momentum；coupled decay |
| NAG | learning rate、momentum、可选 decay | 一个 FP32 momentum；coupled decay |
| Lion | learning rate、beta、可选 decay | 一个 FP32 moment；decoupled decay |
| Muon | learning rate、momentum、Newton--Schulz step、辅助 AdamW 字段 | matrix momentum，加 non-matrix 的两个辅助 moment |

全局 `grad_clip` 必须为正并进入证据身份；未使用 optimizer 字段会被拒绝。目标 schedule
vocabulary 包含 `constant`、`inverse_sqrt_published_update` 与 `cosine_to_zero`，按已发布
update 数而不是 attempted work 前进。当前 adaptive `RunConfig` 只接受 `constant`；另外两种
schedule 保持已注册但不可执行。Optimizer 与 schedule identity 进入 cohort、plan、
selection 与 evidence digest。

## Runtime Topology

目标 registry/coordinator vocabulary 描述一台 node、最多两个 rank：

| Shape | 必需字段 |
|---|---|
| TP1/DP1 | `distributed_runtime_capability=single_rank`，无 capability receipt |
| TP2/DP1 | `patched_two_gpu_v1`、capability receipt、不同 rank/device identity |
| TP1/DP2 | 相同 capability receipt，另需显式 sticky `router_identity` |

Rank 字段必须位于 TP/DP 维度内。Device、rendezvous、router、clock、process-group backend
与 capability receipt 都属于目标 runtime identity。当前 release 只接受 TP1/DP1，并会在
model loading 前拒绝全部 TP2/DP2 `RunConfig`；caller 自己填写 receipt 也不能启用。

`process_group_backend=gloo` 只适用于 CPU collective 合同，不能认证 NCCL/CUDA 行为，
也不能充当 GPU capability receipt。CPU contract 只保留未来 receipt vocabulary。
Production TP2/DP2 工作会保持 `BLOCKED`，直到新的固定 runtime 实现并发出 GPU receipt
与 all-rank publication evidence。

Prefill/decode disaggregation 与 two-batch overlap 仍关闭。Multi-node 和超过两个 rank 的
配置会验证失败；不存在 Kubernetes、elastic 或 automatic-failover 设置。

Runtime topology 与 pool capacity 是不同 identity。Scientific registry 使用两个稳定
logical rank slot；严格 `GpuInventory` 可以包含任意数量 device。唯一 scheduler 对
1/2/4/8/16 GPU 有明确 regression coverage，并在 `IndustrialPhysicalAssignment` 中冻结
physical UUID、rank layout、port、topology 与 whole-instance size。这不会启用 TP2/DP2：
release `RunConfig` validation 仍只接受 TP1/DP1。

## HBM 与 Cohort 策略

Runtime renderer 从 preflight 接收显式 adaptation reserve 与 model/KV memory fraction；
不存在通用源码默认值。Admission 在计入 model、KV、optimizer、candidate、activation、
graph、telemetry 与 safety margin 后，由 headroom 最小的 rank 决定。

每个 materialized cell 还需要 immutable `ExperimentBudget`；short、p99-anchor、soak、
failure、profiler、compile 与 download job 不共享 hidden duration。Startup/load、compile/
prewarm、excluded warm-up 与 request pool、scored arrival、deadline、drain、reset/finalization、
evidence close、retry、token、minimum completion、topology、reserved GPU time 与
whole-instance billed time 都是显式值。Execution copy 保持 `measured_gpu_ms=null`；独立且与
terminal 绑定的 observation 记录所有 phase 与准确 delta。Missing、N/A 与零互不等价。

固定 cohort slab 按 tenant 与 replica 设置 quota。可选 cold offload 必须显式配置，且只
适用于 inactive cohort。Memory pressure 不会静默改变 Full/LoRA、precision、optimizer 或
scope；任何改变都需要新 config、load screen、selection 与 evidence root。

## OnlineSPEC 对比

隔离的 OnlineSPEC protocol 要求额外 `online_spec` object，其 optimizer 必须为 plain
SGD；其 declaration 在独立注册的 tuning protocol 中保持 TP1/DP1。这个单独的
schema/runtime surface 不会提供 release-trusted hardware signer，也不会让 industrial
speculative cell 变得可运行。

| 字段 | 合同 |
|---|---|
| `projection_radius` | 可选正 Euclidean radius |
| `additional_learning_rates` | 唯一递增 expert rate，仅 ensemble |
| `hedge_learning_rate` | 正 meta rate，仅 ensemble |

OGD 与 optimistic OGD 拒绝 ensemble 字段。Ensemble 至少需要两个有序 expert rate。Full
与 LoRA 是不同 decision-coordinate class；对 LoRA factor 求平均不能改称对 dense update
求平均。OnlineSPEC 使用相同已注册 native layer scope、frozen historical KV、准确 proposal
distribution、cohort isolation 与单 candidate 上限，但其 manifest、selection、evidence 与
attestation 永不进入核心 gate。
