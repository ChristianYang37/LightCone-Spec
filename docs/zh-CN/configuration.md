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

这些名称表示源码 capability，不承诺某个 declaration 在当前 session 已可执行。Formal
mutation 还必须具有准确的 root-authorized deployment/hardware policy、prepared content、
workload、compile、qualification、terminal 与 capacity authority。源码只固定 public
Ed25519 root，不携带猜测的 hardware allowlist 或可复用 GPU proof。DFlash、DSpark、NEXTN
以及 `tp1_dp1`、`tp2_dp1`、`tp1_dp2` topology vocabulary 已在源码实现；其依赖硬件的路径
在 matching fresh proof 被验证前保持 `implemented_pending_dynamic_gpu_proof`。Adaptive
runtime 会执行已注册 constant、按 published update 的 inverse-square-root 与有限 horizon
cosine schedule，也会执行非负 logical publication delay。`quota_shadow` 可声明，但缺少
独立授权的 backend acquisition path 时会 fail closed。

每个 serving run 还会绑定 schema-v2 controlled execution policy。注册策略固定 context
length 40,960 与 server seed 1，关闭 radix cache 和 CUDA Graph，并要求
non-incremental output。Target-reference role 关闭 overlap scheduling，speculative role
保留 overlap。Native begin/reset/finalize 会重验同一策略；reset 必须保持 seed 1，不能根据
run nonce 派生新 seed。这些值是 preliminary GPU snapshot 实际使用的实验身份，不表示其已被
证明吞吐最优。

## 方法与 Disabled Path 合同

Target-only 与 Static schema config 都要求 `adaptation: null` 和 `online_spec: null`。
Target-only 启动无 speculation 的 target path；Static 描述原生 speculative decoding。
二者都不能分配 optimizer、gradient、master、candidate 或 cohort-adaptation state，且都不能
仅凭 config 获得执行权限。在 release-attested lane，Static 需要 content-bound request、
performance、汇总 speculative safety 与 signed terminal evidence。Trusted
`formal_single_operator_v1` 不要求 external signer，但同样必须通过实质 source/content/runtime、
fresh GPU qualification、capacity、terminal 与 coverage gate；其结果保持 unsigned，且固定为
`trusted_single_operator_empirical_no_signature`、`formal_measured=false`、`UNMEASURED`。

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

Adaptive schema 可使用 DFlash、DSpark 或 NEXTN。DFlash 使用 native
differentiable-canvas evidence；DSpark 要求实际 sampled predecessor、W1/W2、native
confidence head 与准确的 56-candidate selector；NEXTN 要求 MTP hidden/interface、teacher
row、valid mask、source version，并在 TP2 下要求 target/drafter shard authority。这些都是
源码 contract；execution 仍必须提供 backend 专属 GPU qualification。EAGLE 不受支持。
EAGLE3 只允许 post-probe compatible model/selector 组合；release-attested lane 要求独立 signed
official decision，trusted single-operator lane 则保留其 unsigned compatibility authority。它
不能作为 generic fallback。

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

E1/E2 **LC-candidate** optimizer registry 包含 `adam`、`adamw`、`sgdm`、`nag`、`muon`、
`lion` 与项目自有 `chronobelief`。它们生成 functional parameter/state proposal，只有 commit
时才修改 active state。TTS 不搜索该 registry；TTS-Cal reconstruction 注册 Adam、一步、
`(beta1=.9, beta2=.999, epsilon=1e-8)`、zero decay、无 clipping、全 drafter/
latest-update-round-only update、request reset、side stream、注册的 learning-rate grid 与八个
stride。其 pinned DFlash loss 在 temperature 1 下计算 float32 target-to-draft forward KL，使用
valid-row mask、`exp(-(k-1)/7)` position weight 与 masked weighted normalization。
source-point value correction 不是独立 proximal term，因此 `lambda` 既不是输入也不是搜索轴。
TTS 与 L0-naive 不得继承 schema default、历史 AdamW recipe 或 E1/E2 selection。Plain `sgd`
只用于 OnlineSPEC。

Code-owned TTS split 是 post-master artifact。新 content-verification producer 发布 schema 2，并在
master ceremony 中拒绝 tuning window；window publisher 随后强制使用该准确 durable
master receipt，发布绑定 receipt SHA-256、verification time 与 replay-reservation SHA-256 的
schema-4 window。Schema-1 receipt 仍可解码以重放历史路径，但不能授权新 TTS window。

| Optimizer | 必需身份 | 常驻 moment 规则 |
|---|---|---|
| Adam | learning rate、beta、epsilon | FP32 一阶与二阶 moment；无 decay |
| AdamW | learning rate、beta、epsilon、decay | FP32 一阶与二阶 moment；decoupled decay |
| SGDm | learning rate、momentum、可选 decay | 一个 FP32 momentum；coupled decay |
| NAG | learning rate、momentum、可选 decay | 一个 FP32 momentum；coupled decay |
| Lion | learning rate、beta、可选 decay | 一个 FP32 moment；decoupled decay |
| Muon | learning rate、momentum、Newton--Schulz step、辅助 AdamW 字段 | matrix momentum，加 non-matrix 的两个辅助 moment |
| ChronoBelief | learning rate、beta、epsilon、decoupled decay、source/safe-boundary version | FP32 一阶与 centered 二阶 moment；标准 bias correction；从版本准确派生 $d_r$ age |

使用 clip 的 recipe，其全局 `grad_clip` 必须为正并进入证据身份；TTS 明确为 no-clip。未使用
optimizer 字段会被拒绝。目标 schedule vocabulary 包含 `constant`、
`inverse_sqrt_published_update` 与 `cosine_to_zero`，按已发布 update 数而不是 attempted work
前进；三者都由正式 numeric authority 精确绑定。Optimizer 与 schedule identity 进入
cohort、plan、selection 与 evidence digest。ChronoBelief 的 skipped/aborted proposal 不推进
moment、update count 或 safe boundary；其准确 GPU parity suite 仍是执行前 dynamic gate。

## Runtime Topology

目标 registry/coordinator vocabulary 描述一台 node、最多两个 rank：

| Shape | 必需字段 |
|---|---|
| TP1/DP1 | `distributed_runtime_capability=single_rank`，无 capability receipt |
| TP2/DP1 | `patched_two_gpu_v1`、capability receipt、不同 rank/device identity |
| TP1/DP2 | 相同 capability receipt，另需显式 sticky `router_identity` |

Rank 字段必须位于 TP/DP 维度内。Device、rendezvous、router、clock、process-group backend
与 capability receipt 都属于 runtime identity。Configuration 只在准确的 source-owned
`patched_two_gpu_v1` capability identity 与 runtime receipt claim 存在时接受 distributed
row；formal dispatch 随后深验对应 dynamic GPU proof。Caller 自己填写 receipt 不能启用。

`process_group_backend=gloo` 只适用于 CPU collective 合同，不能认证 NCCL/CUDA 行为，
也不能充当 GPU capability receipt。CPU contract 只验证状态机。Production TP2/DP2
在固定 runtime 发出、且本地控制验证准确 GPU qualification、all-rank publication 与
terminal evidence 前保持 `BLOCKED`。

Prefill/decode disaggregation 与 two-batch overlap 仍关闭。Multi-node 和超过两个 rank 的
配置会验证失败；不存在 Kubernetes、elastic 或 automatic-failover 设置。

Runtime topology 与 pool capacity 是不同 identity。Scientific registry 使用两个稳定
logical rank slot；严格 `GpuInventory` 可以包含任意数量 device。唯一 scheduler 对
1/2/4/8/16 GPU 有明确 regression coverage，并在 `IndustrialPhysicalAssignment` 中冻结
physical UUID、rank layout、port、topology 与 whole-instance size。这不会自动启用
TP2/DP2：准确 dynamic qualification 仍是强制门禁。

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
schema/runtime surface 不会提供 release-trusted hardware signer，因此不能把 OnlineSPEC 结果
提升为 release-level `MEASURED`。Trusted single-operator 的 E0 comparison 可在 audited source
authority 与准确 content/runtime、GPU qualification、capacity、terminal、coverage gate 通过后
无 signer 运行，但始终保持 `trusted_single_operator_empirical_no_signature`、
`formal_measured=false` 与 `UNMEASURED`。

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
