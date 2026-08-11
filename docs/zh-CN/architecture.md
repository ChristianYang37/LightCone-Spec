# 架构

[English](../en/architecture.md) · [README](../../README_zh-CN.md)

## 边界与状态

LightCone-Spec 分离 Python 协议、面向一次性 SGLang checkout 的 semantic mail-patch
集成，以及外部模型/数据/run artifact。唯一 runtime 源码身份是固定 upstream commit、
完整 patch series 与 expected final Git tree。工作区 checkout 永远不是隐式依赖。

CPU 测试只验证合同，不验证 GPU 速度。所有新 GPU cell 都是 `UNMEASURED`。当前端到端
industrial executor 只支持 TP1/DP1 Target-only。固定 patch 现已实现准确的
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v1`
begin/reset/finalize hook，host 会验证其内容，而不信任 provider attribute。但本 release
没有 allowlist 内的 out-of-band 硬件 signer，因此 Static/TTS/L0 仍在任何 mutation 前
fail closed。实证 Stage B 还会因 immutable model/data/trace 输入、provider access、已注册
硬件与 interference evidence，以及 trusted signer 尚不可用而 `BLOCKED`。历史 v2
evidence 仅可用于 regression/debugging，不能支撑新结论。

## 统一 Decode 与 Candidate 生命周期

Target-only 关闭 speculation，是 industrial executor 当前唯一可完成的 path。Static 是
零 adaptation 分配的原生 speculative 目标 path，但 release preflight 会在 trusted
terminal attestation 可用前阻止它。两者的 schema contract 都不导入 adaptation state，也不分配
optimizer、gradient、candidate 或 adaptation trace。

目标 TTS/L0 contract 共用一个 candidate 生命周期；固定 patch 只为 TP1/DP1 DFlash 实现
底层路径与 native terminal lifecycle；没有 trusted signer 时仍不构成 release-executable
support：

1. Verification 为实际被采样的 proposal 生成一个 `ProposalEvidence` envelope；
2. Cohort 只保留每个请求最新的合法 supervision row，并把 batch 绑定到 cohort、epoch、
   slot/buffer generation、source round 与 source adapter version；
3. Side device stream 重建 backend-native 可微 proposal，应用已选择 trainable plan，
   计算 loss，并在不修改 active parameter 或 optimizer state 的情况下生成 functional
   optimizer candidate；
4. Device check 覆盖 finiteness、reconstruction 与 supervision validity。三 byte receipt
   会异步复制到 pinned host memory，并且只在 nonblocking ready-event query 确认完成后、
   measured update timing 之外读取；ready event 身份与预留 scratch byte 属于 candidate；
5. TTS 等待下一个固定更新边界；L0 在同一个 candidate ready 后的首个合法 graph 边界发布；
6. Commit 把完整 candidate 恰好一次复制到固定地址 inference storage。Stale、duplicate、
   cancelled、generation mismatch、non-finite 或 conflicting candidate 都带终止原因被丢弃。

Candidate 创建、commit、discard、cancellation、reset 与 disable 各自只有一个终止转移。
每个 cohort 最多一个 in-flight candidate，因此 side-stream 工作与 scratch memory 有界。

## 公共 Backend 证据与重建

`ProposalEvidence` 刻意保持小而 backend-neutral，包含：

- adapter-free 与 deployed proposal logits；
- 采样实际使用的 normalized corrected distribution；
- semantic valid mask 与 target teacher row；
- sampled predecessor token ID 与 embedding；
- 可选 confidence row；
- 唯一 request ID、cohort digest、source version 与 typed backend payload。

目标 contract 要求构造阶段检查 shape、dtype、device、唯一性与身份，同时把数值检查保留
为 device predicate；还要求 backend validator 重建 `proposal_logits`、
`corrected_distribution` 与可选 confidence，并在 native inference 已施加 adapter 时拒绝
额外 delta。当前 patch 只为 TP1/DP1 DFlash 实现该底层 adaptive reconstruction。

目标 envelope 声明、但本 release 不执行下列 cross-backend 语义：

- DFlash 绑定 deployed differentiable canvas 与 sampling-time proposal correction；只有该
  backend 具有底层 adaptive patch path；
- DSpark 绑定真实 inference-native Markov W1/W2 feature、实际 sampled predecessor、
  scheduler mode、proposal distribution 与 confidence state；
- EAGLE/EAGLE3 绑定 tree state、top-k-one execution，以及贯穿整个 proposal/verification
  chain 的一个 source version；
- NEXTN 绑定 native MTP hidden state 与不可变 upstream interface digest。其目标 contract
  还将要求 interface 与 memory-fit preflight；当前 schema 会独立拒绝它。

DSpark、EAGLE、EAGLE3 与 NEXTN adaptation 会在 model loading 前保持 `BLOCKED`。在目标
contract 中，历史 drafter KV 被 detach、不可变且带版本；合格 update path 可把它当作
state 使用，但绝不能重建或对其求导，未来 KV 记录最新 published version。

## Trainable Plan

目标 adaptation contract 只有 Full 与 LoRA 两种 parameterization。每个 plan 都列出 selected/frozen name、
shape、dtype、Full/LoRA coordinate 身份及 sharded/replicated ownership；其 digest 进入配置
和证据。Full 为所选 native parameter 保留 FP32 master。LoRA 只选择已注册二维 matrix，
初始 functional delta 为零，rank 为 1/2/4/8/16/32/64，并固定 `alpha/r=1`。

DFlash、EAGLE/EAGLE3 与 NEXTN 使用 `last1`、`last3`、`last5` 或 `all` native layer
scope。借用的 target embedding、LM head 与 target model 始终冻结。

DSpark 使用相同四种 layer-only scope，此时全部 native W1/W2/acceptance 参数冻结。
另有 `last1_native_heads`、`last3_native_heads` 与 `last5_native_heads`：所选 backbone
matrix 按 Full 或 LoRA 处理，W1、W2 与 scalar acceptance/confidence 参数则始终为 Full
且 replicated。已注册 E1a 集合恰好为：

```text
4 layer-only scopes x (Full + 7 LoRA ranks) = 32
3 hybrid scopes     x (Full + 7 LoRA ranks) = 24
total                                             56
```

Plan 按实际 coordinate 与 optimizer state 计算显存，而不是使用 model size 的固定倍数。

## Topology 与发布

目标 schema 与 CPU coordinator vocabulary 描述单主机 TP1/DP1、TP2/DP1 与 TP1/DP2
identity。当前 `RunConfig` 只接受 TP1/DP1；由于本 release 无法签发内容绑定的
`patched_two_gpu_v1` capability receipt，它会在 model loading 前拒绝全部 TP2/DP2 config。
目标 multi-rank receipt 会绑定 global/local/TP/DP rank、device、node、process、rendezvous、
router、clock、runtime 与 model identity。

在该不可执行目标 coordinator contract 中，TP trainable state 跟随 inference ownership。Sharded parameter 保持 local；replicated
parameter 只在所属 TP replica 内 reduce。DP 使用 sticky cohort routing：一个 cohort 始终
留在 replica-local，replica 之间绝不平均 adaptation gradient。

目标 distributed publication 分两阶段。每个 rank 对同一个 retry-stable update/candidate 身份
prepare，并报告 source version、epoch、buffer/optimizer generation、readiness、finiteness、
memory reservation、safe boundary 与 process-group health。单一 all-rank decision 要么全部
commit，要么全部 abort；post-copy receipt 会拒绝 partial application。Collective 或 split
decision 失败会令 service unready、停止新 admission，并要求同 topology 显式 restart。

`GlooPublicationTransport` 是该状态机的真实双进程 CPU harness。它刻意拒绝 NCCL，
不能验证 CUDA event、graph capture、固定地址 device copy、GPU 数值、性能或资源竞争。

## HBM 与 Cohort 治理

HBM ledger 分别计费 active/base state、FP32 master、gradient、实际 optimizer moment、
metadata、candidate/staging buffer、training activation、KV gather scratch、graph buffer、
telemetry 与 safety margin。Admission 由 headroom 最小的 rank 决定；只在一个 rank 可容纳
而另一个不可容纳的总量，会在昂贵分配前被拒绝。

显存压力按固定顺序处理：保留 immutable/runtime correctness，保留 active KV 与 published
state，只驱逐 native inactive prefix，取消 pending adaptation 并释放 temporary state，
最后 queue 或拒绝新工作。可选 cold-cohort offload 排在末尾，必须显式启用与计时，默认
关闭。任何路径都不会静默改变 parameterization、precision 或 optimizer。

Cohort manager 使用有界固定大小 slab 与 tenant quota。Key 包含 tenant、cohort digest 与
replica；slot/optimizer generation 阻止 stale reuse。只有 inactive cohort 可被回收或 cold
offload；transfer/reclamation receipt 保留 slab 与 byte 身份。

## Trace 与 Durable Telemetry

Load trace 为 request content、arrival、timeout/cancellation 与 warmup/scored window 绑定
相互独立的身份。Synthetic Poisson 与 immediate-burst generator 按 seed 确定，并明确标为
synthetic。没有不可变外部 corpus digest 时，BurstGPT-shaped trace 不会被表示成真实数据集。
配对方法必须绑定相同 trace digest，并把每个 offered request 恰好计为 rejected、completed、
timed out、cancelled 或 unfinished。

Evidence schema 可以表示每个 process/rank 的 run、request、round、update 与 performance
record，并通过有界 queue 写入。
Durable index 拒绝重复 primary identity。周期或容量触发的 flush 会创建 fsynced Parquet WAL
segment 与 checkpoint；backpressure 与任何显式 drop 都有计数。Native terminal lifecycle
把 capability、begin、reset 与 finalize receipt 绑定到同一个 process/session/run/nonce/plan/
rank identity 及准确 ordered token ID。Static 保持 round/update 详细 trace 零分配，只生成
所需汇总 speculative safety/accounting；TTS/L0 还绑定 request、round、update、KV version、
publication、performance 与 safety row。signer 不可用时，release verifier 仍在 mutation 前
阻止这三种方法。

仓库为未来相邻且兼容的 trace 定义了 immutable session key 及 reset/finalize receipt 数据
契约；完整 key 包含 source/capability、RunConfig、model/drafter revision、method/backend、
topology 与 physical GPU UUID、memory/graph/telemetry 配置、compile-cache identity 和 port。
但当前 release **不会执行或声明** shared server session：尚无 release-owned trusted boundary
实现、session receipt 的 durable terminal binding，以及从 launch 到 terminate 的连续
whole-inventory 计费。因此所有 shared-session mutation 入口都会在 launch、network、reset
及 evidence-root 创建之前以
`shared_session_trusted_durable_boundary_and_continuous_accounting_unavailable` 失败。
当前唯一可声明的 Target-only 路径仍是 single-trace execution。

成功 close 时，WAL segment 先通过 coverage 检查，再组装为 process-unique Parquet shard
与 durable prepared receipt。executor 随后完成 native terminal lifecycle，并发布与 prepared
receipt 精确字节绑定的强制 budget observation；两者都存在后，才发布一个不同的
exclusive terminal envelope。该 envelope 保留 prepared receipt 的 schema、row-group
coverage、counter、file size、schema digest 与 SHA-256，并额外绑定 prepared receipt 的
raw digest/size，以及 budget observation 的 semantic digest、raw receipt/sidecar digest、
size 与安全路径。如果在 observation 与 terminal 发布之间崩溃，可从已验证的 prepared
receipt 构造最终 envelope 而不重跑；只有 prepared receipt 而没有 observation 时不得形成
claim。中断或 aborted WAL 保持可审计，但不能进入分析。Completed receipt 不可覆盖，也
不能与竞争 attempt 合并。

## Industrial Registry 边界

不可变 registry 声明顺序
`preflight -> E3a -> E1 -> E2 -> E4 -> E3b -> E1a -> E5 -> E6 -> E0`。
每个 cell 绑定 axis、seed、status/reason、两个 logical rank slot、port、cache/evidence root
与 workload 隔离。Physical UUID 永远不是 registry identity：content-bound inventory 与
确定性 frozen assignment 会在 launch 前绑定 UUID、rank layout、topology、port、完整的
per-cell `ExperimentBudget` 以及 whole-instance billing。

单一 same-host `GpuPoolScheduler` 是唯一 scheduler。它接受任意 inventory，并对 1、2、4、
8、16 GPU 有明确 regression coverage；gang placement 是 atomic 且 topology-aware，所有
headline wave 都在执行前冻结。并发上限由准确 `InterferenceEnvelope` 决定，而不是由空闲
device 数量决定。Resource、port、cache writer 与 evidence root 不能重叠，receipt-only
resume 也绝不从目录存在推断完成。

Reducer-owned activation artifact 只 materialize E1 的一个 130-cell slice 与 E2 每个
successive-halving round，并为每个未激活 template 记录 immutable disposition。Confirmation
planning 以 family 为局部单位：恰好四个 excluded pilot 会在 confirmation 可见前选择
`POWERED` 的 12--20-block final prefix，或选择 `UNDERPOWERED`。合法 Target-only 复用必须
使用 byte-equivalent、content-bound evidence alias；analysis 会保留其 dependence unit，
不会把 alias 当作独立样本。Self-described non-singleton alias 会保持 blocked，直至
execution evidence 重新计算其 equivalence。

每个 launch cell 都绑定准确 budget，覆盖 startup、compile、excluded warm-up、scored
arrival、deadline、drain、reset、evidence close、special-job duration、retry、token、p99
status 与 GPU accounting。强制 observation receipt 会记录所有 observed component、measured
GPU time 与 whole-instance billed GPU time；缺失 component 不能视为零。Immutable
compile-cache base 使用每个 process 的 private overlay，official serving client 复用一个
caller-owned HTTP pool，evidence writer 批量写 durable WAL row group，同时保留 terminal
fsync 与 coverage check。

Registry 与 pool planning 都不分配 device state，也不产生实证结论。其 target declaration
不会绕过 release preflight：Static/TTS/L0 因 trusted signer 保持 `BLOCKED`；全部 TP2/DP2
和不受支持的 adaptive backend 仍受各自 implementation gate 阻止。

本 release 不声明 speculative industrial execution、multi-rank execution、multi-node
execution、Kubernetes scheduling、elastic membership、remote evidence storage 或
automatic failover，也不包含任何新 GPU 结果。
