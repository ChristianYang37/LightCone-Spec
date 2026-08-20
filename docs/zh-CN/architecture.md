# 架构

[English](../en/architecture.md) · [README](../../README_zh-CN.md)

## 边界与状态

LightCone-Spec 分离 Python 协议、面向一次性 SGLang checkout 的 semantic mail-patch
集成，以及外部模型/数据/run artifact。唯一 runtime 源码身份是固定 upstream commit、
完整 patch series 与 expected final Git tree。工作区 checkout 永远不是隐式依赖。

CPU 测试只验证合同，不验证 GPU 速度。所有新 GPU cell 都是 `UNMEASURED`。Release-attested
executor 当前唯一可用于 claim 的 execution 是 TP1/DP1 Target-only。固定 patch 现已实现准确的
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v2`
begin/reset/finalize hook，host 会验证其内容，而不信任 provider attribute。但本 release
没有 allowlist 内的 out-of-band 硬件 signer，因此 Static/TTS/L0-naive/LightCone 仍会在
release-attested lane 的任何 mutation 前 fail closed。Trusted `formal_single_operator_v1` lane
则无需 external signer：immutable source/content/runtime 输入、provider access、fresh GPU
qualification/interference evidence、capacity、terminal 与 coverage gate 通过后，它可以运行完整
empirical DAG。Closure 固定为 `trusted_single_operator_empirical_no_signature`、
`formal_measured=false` 与 `UNMEASURED`；只有 release attestation 才能提升为 `MEASURED`。
历史 v2 evidence 仅可用于 regression/debugging，不能支撑新结论。

## 统一 Decode 与 Candidate 生命周期

Target-only 关闭 speculation，是当前唯一可用于 release claim 的 path。Static 是
零 adaptation 分配的原生 speculative 目标 path，但 release preflight 会在 trusted
terminal attestation 可用前阻止它。两者的 schema contract 都不导入 adaptation state，也不分配
optimizer、gradient、candidate 或 adaptation trace。

TTS、L0-naive 与 LC-candidate 复用一个有界 candidate-lifecycle 实现。固定 patch 还包含
注册的 DSpark、NEXTN、compatible EAGLE3、TP2 与 sticky DP2 source path。Recipe authority
与 publication policy 保持正交，共享 machinery 不会合并 live candidate、optimizer state、
config 或 evidence；每个准确 path 都必须通过 dynamic GPU qualification 与 trusted
external-control proof 才可用于 claim：

1. Verification 为实际被采样的 proposal 生成一个 `ProposalEvidence` envelope；
2. Cohort 只保留每个请求最新的合法 supervision row，并把 batch 绑定到 cohort、epoch、
   slot/buffer generation、source round 与 source adapter version；
3. Side device stream 重建 backend-native 可微 proposal，应用已选择 trainable plan，
   计算 loss，并在不修改 active parameter 或 optimizer state 的情况下生成 functional
   optimizer candidate；
4. Device check 覆盖 finiteness、reconstruction 与 supervision validity。三 byte receipt
   会异步复制到 pinned host memory，并且只在 nonblocking ready-event query 确认完成后、
   measured update timing 之外读取；ready event 身份与预留 scratch byte 属于 candidate；
5. TTS 等待固定 update barrier；L0-naive、LC-candidate 与 LightCone 在 ready 后首个合法
   safe graph boundary 发布；
6. Commit 把完整 candidate 恰好一次复制到固定地址 inference storage。Stale、duplicate、
   cancelled、generation mismatch、non-finite 或 conflicting candidate 都带终止原因被丢弃。

Candidate 创建、commit、discard、cancellation、reset 与 disable 各自只有一个终止转移。
每个 cohort 最多一个 in-flight candidate，因此 side-stream 工作与 scratch memory 有界。
Candidate equality 只在 source-state 与 proposal-evidence digest 完全相同的受控 replay 中
检查；TTS 与 L0-naive 的 publication decision 分叉后，live history 可以继续分叉。

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
额外 delta。固定 patch 已为 DFlash、DSpark、NEXTN 以及受官方 selector 门控的 EAGLE3
实现 reconstruction。这些是源码能力而非硬件结论；每条路径在其准确、全新的 GPU
qualification 被验证前都保持 fail closed。

Cross-backend 语义如下：

- DFlash 绑定 deployed differentiable canvas 与 sampling-time proposal correction；只有该
  backend 具有底层 adaptive patch path；
- DSpark 绑定真实 inference-native Markov W1/W2 feature、实际 sampled predecessor、
  scheduler mode、proposal distribution 与 confidence state；
- EAGLE3 绑定 tree state、top-k-one execution，以及贯穿整个 proposal/verification chain
  的一个 source version，但只适用于 prepared-model selector 判定为 compatible 的组合；
  generic EAGLE 仍不受支持；
- NEXTN 绑定 native MTP hidden state 与不可变 upstream interface digest，并要求
  interface 与 memory-fit preflight。

DSpark、NEXTN 与 compatible EAGLE3 的状态为
`implemented_pending_dynamic_gpu_proof`；proof 缺失或不匹配时会在 worker allocation 前
拒绝。EAGLE 不受支持。历史 drafter KV 被 detach、不可变且带版本；合格 update path
可把它当作 state 使用，但绝不能重建或对其求导，未来 KV 记录最新 published version。

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

Schema 与 coordinator 描述单主机 TP1/DP1、TP2/DP1 与 TP1/DP2。`RunConfig` 只在携带
准确的 source-owned `patched_two_gpu_v1` capability identity 时接受双 rank 配置；formal
dispatch 还必须验证全新的 root-authorized dynamic GPU proof。Caller 自填 receipt 不能
启用它。Multi-rank identity 绑定 global/local/TP/DP rank、device、node、process、
rendezvous、router、clock、runtime 与 model identity。

在该 coordinator contract 中，TP trainable state 跟随 inference ownership。Sharded parameter 保持 local；replicated
parameter 只在所属 TP replica 内 reduce。DP 使用 sticky cohort routing：一个 cohort 始终
留在 replica-local，replica 之间绝不平均 adaptation gradient。

Distributed publication 分两阶段。每个 rank 对同一个 retry-stable update/candidate 身份
prepare，并报告 source version、epoch、buffer/optimizer generation、readiness、finiteness、
memory reservation、safe boundary 与 process-group health。单一 all-rank decision 要么全部
commit，要么全部 abort；post-copy receipt 会拒绝 partial application。Collective 或 split
decision 失败会令 service unready、停止新 admission，并要求同 topology 显式 restart。

`GlooPublicationTransport` 是该状态机的真实双进程 CPU harness。它刻意拒绝 NCCL，
不能验证 CUDA event、graph capture、固定地址 device copy、GPU 数值、性能或资源竞争。

## Fleet Control Plane 与 Release 状态

`GpuFleetScheduler` 只负责 host affinity 与 serial partition 的 control-plane 工作；每个
child placement 都委托给唯一的 same-host `GpuPoolScheduler`，因此 fleet composition 不会
引入第二套 placement algorithm。`HostInventoryBinding` 把准确一台主机的 content-bound
`GpuInventory` 与 `InterferenceEnvelope` 结合；`GpuFleetInventory` 对这些 binding 排序并去重，
每个 `HostExecutionBinding` 分配的 port、cache、evidence 与 contention namespace 必须在该
host 内无冲突，不同 host 可以重复 literal value。Remote execution binding 再固定 host-local
materialization manifest。Independent cell 在 eligible host 间均衡；matched-geometry baseline anchor 与完整
confirmation block 始终留在同一 host/GPU；异构 hardware envelope 绝不混入同一个
statistical family。

Gang placement 仍只在一台主机内 topology-aware、atomic。若请求 shape 只能通过组合多台主机
满足，会在 dispatch 前固定以 `cross_host_collectives_unvalidated` 失败。这是硬 capability
边界：multi-host inventory 与 SSH dispatch 不构成 cross-host NCCL、TP/DP rendezvous 或
model-parallel execution 的证据。

Coordinator 向 `execute-dispatch-wave --host-request-stdin` 发送 canonical、content-bound host
request。`SshHostRoute` 只存在于本地且不可序列化；OpenSSH 使用 agent、batch public-key
authentication、基于固定文件的 strict host-key checking、禁用 forwarding，并限制 stdout、
stderr 与运行时间。Worker 重开 host-local materialization manifest，返回不含 routing data 或
raw log 的 content-bound response。若 request 可能已到达 worker 却没有权威 response 返回，
outcome 必须是不可直接重试的 `REMOTE_OUTCOME_UNKNOWN`。Reconcile 通过准确原始
destination、port 与 known-host-key authority 独立取回准确 receipt envelope 与 evidence
bytes，在严格 size bound 下于内存重算全部 content identity，最终只发布不含 path 的 digest
projection。Endpoint value、host-key bytes 与 credential 都不会持久化到 request、receipt、
evidence 或 log。已完成 receipt 保留在 fleet receipt 中，in-flight attempt 不能用同一
identity 迁移到其他主机；有界 semaphore 限制 fleet transport concurrency。

架构使用三个不可互换的标签。`CPU_READY` 只覆盖非 device contract validation；
`GPU_SMOKE_READY` 只表示有界 device check 已准备；`MEASURED` 必须取得已注册 workload、完整
terminal evidence、hardware envelope 与 release-owned attestation。这些标签不授权 device
或 performance claim；每条 execution path 都必须分别闭合自己的 runtime、evidence 与
attestation gate。

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
配对方法必须绑定相同的完整 registered source-pool digest；closed-loop 中各 method 实际消费的
offered prefix 与 observed output token 不会重新定义配对。Completed output 由独立 exactness
gate 检查；每个 offered request 恰好计为 rejected、completed、timed out、cancelled 或
unfinished。

Evidence schema 可以表示每个 process/rank 的 run、request、round、update 与 performance
record，并通过有界 queue 写入。
Durable index 拒绝重复 primary identity。周期或容量触发的 flush 会创建 fsynced Parquet WAL
segment 与 checkpoint；backpressure 与任何显式 drop 都有计数。Native terminal lifecycle
把 capability、begin、reset 与 finalize receipt 绑定到同一个 process/session/run/nonce/plan/
rank identity 及准确 ordered token ID。Static 保持 round/update 详细 trace 零分配，只生成
所需汇总 speculative safety/accounting；adaptive scientific role 还绑定 request、round、update、KV version、
publication、performance 与 safety row。signer 不可用时，release verifier 仍在 mutation 前
阻止这三种方法；该 release-verifier decision 不会阻止上述 trusted single-operator empirical
lane。

仓库为相邻且兼容的 trace 定义了 immutable session key 及 reset/finalize receipt 数据
契约；完整 key 包含 source/capability、RunConfig、model/drafter revision、method/backend、
topology 与 physical GPU UUID、memory/graph/telemetry 配置、compile-cache identity 和 port。
固定 SGLang source 现已实现 all-reset capability/state/receipt producer。在支持的 single-
tokenizer HTTP/1.1 uvicorn 路径上，它会在真实 transport lifecycle 上绑定累计 HTTP process/
generation/created/closed/current；Granian HTTP/2 与 multiple-tokenizer HTTP-process 路径会在
生成该 capability 前 fail closed。其 GPU semantics 仍为 `PENDING`，且仍缺少 session receipt
的 durable terminal binding 与从 launch 到 terminate 的连续 whole-inventory 计费。这只关闭
reset-state accounting slice；patch 尚不生成 native warm-up、trace 或 close receipt。
因此当前 release **不会执行或声明** shared server session；
所有 shared-session mutation 入口都会在 launch、network、reset
及 evidence-root 创建之前以
`shared_session_gpu_reset_and_durable_terminal_authority_unavailable` 失败。
高层 block executor 会先验证全部成员，再为每条 trace 使用独立 clean process、HTTP pool、
native lifecycle、terminal receipt 与 budget observation；该路径不声明 reuse。当前唯一可
声明的 Target-only 路径仍是 single-trace execution。

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
resume 也绝不从目录存在推断完成。Dispatch 会在每个 runner 前后写入 path-bound
WAVE/INTENT/FINISH hash chain；partial sibling failure 会保留成功 raw terminal authority，
未完成 intent 则阻断，而不是伪造 retry cost。

Reducer-owned signed materialization 消费 sealed upstream receipt，恰好生成一个 68-cell
E1 stage：Target-only、Static、frozen TTS、frozen L0-naive，以及 32 个 geometry、每个两个
optimizer anchor 所形成的 64 个 LightCone candidate。L0-naive 只是 mechanism anchor，不参与
候选排名。对 `g` 个 E1 survivor geometry，E2 首轮物化 `n0 = 105g` 个 recipe（七种
optimizer × 三种 schedule × 五个 learning rate），后三轮满足
`n(k+1) = max(ceil(nk/4), 21)`，且每轮另带四个 fixed anchor；不存在 eager sentinel
matrix。只有准确 sealed E2 final-recipe receipt 才能产生 LightCone role。Confirmation
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

Registry 与 pool planning 都不分配 device state，也不产生实证结论。其 declaration 不会
绕过 release preflight：每个正式角色仍须具备签名 content、capacity、compile、execution、
terminal 与 qualification 全链。TP2/DP2 和 backend 专属路径在准确 dynamic proof 出现前
仍保持 blocked。

本 release 不声明已完成 industrial measurement、跨主机 collective、Kubernetes scheduling、
elastic membership、remote evidence storage 或 automatic failover。Multi-host control plane
只分发 independent host-local work；源码实现不能替代全新的 GPU 结果。
