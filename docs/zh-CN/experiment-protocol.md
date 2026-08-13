# 实验协议

[English](../en/experiment-protocol.md) · [首页](../../README_zh-CN.md)

## 问题与当前状态

工业级研究考察在线 drafter adaptation 在何处有效、为何有效，以及其成本或操作风险何时
超过节省的 target work。Target-only、Static、TTS 与 L0 始终分开；TTS 与 L0 使用相同
candidate，只在发布时间上不同。

所有 formal industrial GPU 结果都是 `UNMEASURED`。代码、CPU 测试与 registry 建立目标 protocol 与
coordinator contract，不是完整可运行 speculative surface 或 benchmark 结果。Industrial
executor 当前只运行 TP1/DP1 Target-only。Static/TTS/L0 会在任何 mutation 前因缺少
trusted hardware signer 而 `BLOCKED`；固定 native
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v1`
begin/reset/finalize hook 已经实现。Stage B 还因 immutable model/data/trace lock、provider
credential、已注册硬件、GPU smoke 与准确 interference envelope 尚不可用而 blocked。历史
v2 evidence 只用于 regression/debugging，不得进入 schema-v3 selection、power sizing、
confirmation 或结论。

## 不可变 Dependency DAG

Registry 固定以下顺序：

```text
preflight -> E3a -> E1 -> E2 -> E4 -> E3b -> E1a -> E5 -> E6 -> E0
```

每个目标 definition 命名 dependency、locked output 与科学 axis。每个 cell 再绑定完整身份：
experiment/model/backend/task/method、parameterization/native scope、optimizer/schedule、
context/regime/width/arrival/SLO、cohort/topology、seed/block、两个 logical rank slot、logical
port/cache/evidence claim、workload class 及真实 status/reason。独立 frozen physical
assignment 会绑定 inventory UUID、rank layout、concrete port、topology、准确 per-cell budget
与 whole-instance billing；更换 host 不会重写 scientific registry。

只有准确 dependency receipt 验证通过且 cell 通过 executable-release preflight 后，stage
才可能 dispatch。Receipt 绑定 registry、
runtime、split、dependency-output 与 locked-output SHA-256，并声明 selection 已在下游
unblinding 前封存。编辑 registry artifact 或以另一 digest 重新序列化 dependency 都会
fail closed。

同 host interference calibration 使用成对的 isolated 与 simultaneous Static block。
committed-token goodput 和原始 within-request p99 ITL 的成对平均相对差都必须不超过
1%，且各自的 paired BCa 95% interval 必须包含零。缺少逐 token 时间戳时保持
UNRESOLVED；request latency 与对 SSE gap 的均摊都不能替代 ITL。通过的 rule 只授权实测
cardinality，并且仍必须通过 release-owned trusted-attester chain；缺少该信任链时 headline
scheduling 保持 serial，正式测量继续 blocked。

## Stage 与 Locked Decision

| Stage | 目的 | 下游使用前锁定的输出 |
|---|---|---|
| Preflight | source/runtime/model/data identity、exactness、HBM、telemetry、inventory/topology、仅审计用 session-reset schema、cache/HTTP/writer 与 interference calibration | runtime envelope |
| E3a | Target-only/Static context、regime、concurrency 与 draft-width capacity | reference load、matched width、crossover 与 drift witness |
| E1 | DFlash layer scope 与 Full/LoRA geometry，使用 AdamW/SGDm anchor | safe Pareto set 与 common load |
| E2 | optimizer、log learning-rate grid、schedule 与 successive halving | 一个 DFlash recipe |
| E4 | 累积 systems mechanism 与隔离 profiling | mechanism gate |
| E3b | 配对 long-context Target-only/Static/TTS/L0 confirmation | long-context confirmation |
| E1a | 原生 DSpark transfer 与 retuning | 一个 DSpark recipe |
| E5 | production arrival、cohort、topology、SLO 与 failure | production/topology surface |
| E6 | 原生 NEXTN interface 与双 rank fit，随后 transfer | native MTP transfer surface |
| E0 | model/backend/task breadth，包括隔离 OnlineSPEC | breadth surface |

E1 将四种 native layer scope 与 Full 加七个 LoRA rank、两个 optimizer anchor 交叉，在下游
optimizer 搜索前恰有 64 个 geometry cell。E2 将 optimizer 专属字段和 `constant`、按
published update 的 inverse-square-root、cosine-to-zero schedule 保持为不同身份。
ChronoBelief declaration 显式为 `BLOCKED`：当前未注册 authoritative update equation 或
source identity，禁止用另一 optimizer 替代。

E1a 恰有 56 个 adaptive configuration。32 个 layer-only cell 将
`last1/last3/last5/all` 与 Full 加七个 LoRA rank 交叉，并冻结 DSpark native head。24 个
hybrid cell 将 `last1/last3/last5` 与相同八种 backbone parameterization 交叉，同时把原生
W1、W2 与 scalar acceptance/confidence state 作为 Full 训练。Fixed verification budget
只用于 tuning control；转移后的 candidate 还必须通过 native scheduler。

这些是已注册 scientific envelope，不是执行每个 template 的指令，也不是当前 release
support。E1 activation 消费 sealed E3a selection，只 materialize 一个 130-cell width/load
slice，并为其他 E1 template 记录 immutable disposition。E2 每次只 materialize 一个
successive-halving round，保留 matched TTS/L0 pair 与 family floor，且下一 round 只能从 prior
sealed survivor receipt 派生。Formal E2 reducer 要求真实的 per-token observation time。
官方 SGLang SSE client 可能在
一个 chunk 中合并多个 token ID，因此 adapter 会把这些 token timestamp 记为 unavailable，
而不是虚构等间隔 ITL。只有绑定 native per-token timestamp hook，或由固定 runtime 证明
one-token-per-chunk delivery，E2 才能从 `BLOCKED` 解锁。Confirmation materialization 以
family 为局部单位：四个 excluded pilot 会在 confirmation 可见前归约，随后只激活 sealed
12--20-block final prefix。
Family 仍是增量 scheduling/power 单位，但 stage dependency receipt 只能由独立 exact-coverage
aggregate 签发：所有 family 按 SHA 排序，并从 raw pilot/final completion authority 现场重放。
E5 的 264 个非 family failure-injection cells 使用 deterministic auxiliary activation/completion；
family 与 auxiliary disposition 必须构成该 stage 的准确不交并集。
固定 patch 仍会拒绝 DSpark/EAGLE/EAGLE3/NEXTN adaptation 与全部 multi-rank execution；
TP1/DP1 DFlash 已实现注册 E2 schedule 与 logical delay，真实 quota-shadow acquisition 需求
仍由具名 backend capability gate 阻止。该路径没有 out-of-band trusted signer 时不能产生结论。

## 数据、Context 与 Trace

Controlled prompt window 内容不重叠并绑定 digest；selection 数据不能进入 confirmation。
Model/tokenizer revision、prompt compiler、task split、max context、generation budget 与安全
speculative headroom 都是不可变输入。Natural dataset 要求准确外部 revision，并保持在
仓库之外。

Long-context axis 为 1K、2K、4K、8K、16K、24K、32K 与 40,928 token，覆盖
long-input/short-output、short-input/long-generation、multi-turn shared-prefix regime。
DFlash draft width 为 4、8、16。E3b 分别报告 matched-width 与 deployment-optimal-width
panel；看到 confirmation 后再改变 width 属于禁止行为。

Controlled profile 使用 greedy。Exactness 诊断会给每条完整、有序 output-token-ID 轨迹
加显式格式标识后计算 hash；decoded text 不能作为 exactness witness。Legacy formal
collection 还会绑定一份在相同 load 与安全 context 上限捕获的 32-prompt Target-only greedy
reference，并要求每种 method 的每个 block 都与之匹配；仅 cross-method agreement 不充分。
该 reference 在当前 release 中仍只是 `UNMEASURED` 诊断，不能绕过 speculative execution
或 attestation blocker。

Production trace 分离 content identity 与 arrival identity。Open-loop Poisson、immediate
burst、BurstGPT-shaped 与 soak trace 绑定准确 arrival offset。Closed-loop run 改为绑定最大
request pool、population 与每个 client 的顺序；每次实际 offer 必须晚于该 client 上一次
terminal event。各 method 可以消费不同的连续 prefix；固定 arrival window 结束前耗尽任一
client pool 会使 run 不可用于结论。没有不可变外部 corpus digest 时，synthetic
BurstGPT-shaped trace 绝不标成真实 dataset。

配对 open-loop method 消费相同 trace byte；配对 closed-loop method 消费相同 pool 与 client
顺序。每个实际 offered request 恰好记为 rejected、completed、timed out、cancelled 或
unfinished；unfinished 工作按注册 timeout boundary 留在 denominator。缺失 row 不补零。

## GPU Pool Staging

Registry 使用两个 logical rank slot；physical device 只来自 content-bound `GpuInventory`
与 frozen assignment。唯一 deterministic same-host scheduler 支持任意 inventory size，并对
1、2、4、8、16 GPU 有明确测试。它会在一个合法 topology group 上 atomic 分配 k-GPU
TP/DP shape，拒绝 partial gang 与 cross-host placement，在 eligible UUID 之间轮换 independent
block，并阻止 GPU、port、cache writer、evidence root 与 exclusive resource 重叠。

`GpuFleetInventory` 在该 same-host scheduler 之上组合多台 independently verified host。
它在 eligible host 间均衡 independent cell 与完整 confirmation block，同时保留 host-local
inventory、interference、port、cache、evidence 与 materialization identity。Paired TTS/L0
work 留在同一 host/GPU。每台主机的 headline concurrency 只由该主机已校准 envelope 限制；
异构 hardware envelope 不进入同一 family。跨主机 gang 以
`cross_host_collectives_unvalidated` 拒绝。

并行工作前，preflight 记录 clock、temperature、power state、background process、driver/
runtime identity、topology、per-rank HBM 与准确 `InterferenceEnvelope`。Envelope 按 hardware、
workload、co-run signature/count、gang shape、thermal/power/load state 与 host contention
索引。如果八 GPU host 只校准过 two-way concurrency，frozen headline wave 仍最多 two-way；
runtime completion 不能创造 result-dependent co-tenancy。Profiler、download 与 compile
工作是 exclusive-host，不能与 headline timing 竞争。

Exclusive-host 分类不等于 execution authority。当前 release 不会 dispatch COMPILE 或
DOWNLOAD cell：compile 缺少准确的 release-owned prewarm/finalization manifest 与原子
cache-result pointer；download 缺少 first-party terminal receipt contract。

未来的 compile contract 必须绑定 assignment、budget、inventory 与 GPU UUID、compile
plan/key 与 model revision、TP、context、concurrency、graph bucket、deterministic prewarm
payload，以及 graceful-shutdown ACK。原子发布的 result pointer 必须给出 manifest、attempt
receipt、final cache receipt 与 immutable cache object 的路径和 hash；resume 必须重开全部
原始文件，不能信任 pointer 中的 serialized summary。

每个 dispatch plan 为每个 assignment 绑定准确 `ExperimentBudget` digest。Wall time、requested
gang compute GPU time、reserved GPU time 与 fixed-instance billed GPU time 必须分开；two-GPU
gang 消耗两倍 wall time，而 whole-instance billing 对 observed wall interval 计费整个 frozen
inventory。Queue 是数据，不是同时启动全部 assignment 的指令。

TP2 与 sticky-replica DP2 是目标 coordinator contract。未来 release 将要求 verified
patched-runtime capability receipt，并由全部 rank 为同一 publication identity 提供
prepare/decide/apply/receipt evidence。当前 `RunConfig` 会在 model loading 前拒绝全部
TP2/DP2 cell；CPU `gloo` harness 只是状态机测试，不能启用这些 cell。

Multi-host control 通过 content-bound request/receipt 分发 independent host-local work。
任何 stage 都不声明跨主机 collective、可执行 TP2/DP2 path、Kubernetes scheduling、elastic
membership 或 automatic failover。Fleet SSH transport concurrency 必须显式有界。一台主机
失败时保留其他主机已完成 receipt。任何 post-dispatch authority 丢失都必须标记为
`REMOTE_OUTCOME_UNKNOWN`，并通过准确原始 destination、port 与 known-host-key authority
独立、内容寻址地 fetch/reconcile；endpoint value、key bytes 与 credential 不会持久化。只有
terminal-negative outcome 才能创建新的同主机 receipt-bound attempt，不能静默迁移
in-flight attempt。

## Production 指标与安全

每个 system point 报告 offered/admitted/terminal request accounting、throughput/decode
goodput、按 prompt bucket 的 TTFT、within-request ITL、completion/error rate、target
call/work、accepted/verified/committed draft、update/candidate/publication count、exposed/
overlapped update time、queue/batch occupancy、HBM category、每个 output token 能耗与
hardware envelope validity。

已注册 production SLO 要求 short/medium/long prompt 的 TTFT 不超过 2/5/10 秒，
within-request p99 ITL 不超过 100 ms，qualification 至少 99%，error 至多 0.1%，completion
至少 99.9%。完成请求少于 10,000 时，p99 结论是 `UNRESOLVED`；不能用小样本估计后表示为
qualified。

Exactness violation、重复/混合身份、non-finite candidate、partial publication、fallback、
OOM、retraction、evidence drop、不完整 terminal accounting 或超出 hardware envelope 的
观测都会使对应 block 无效。Profiler 与带同步的诊断必须独立运行，并禁止进入 headline
evidence。

## Power 与统计推断

每个准确 `ConfirmationFamilyIdentity` 都使用恰好四个 paired pilot block，只估计
L0--Static 与 L0--TTS 的 log-effect variance。该 identity 绑定 experiment/model/backend/task、
context/regime/load/arrival、width panel、topology、cohort/method family、runtime/split/trace/
sampling 与 hardware envelope；一个 family 的 pilot 不能为另一个 family 计算 power。Pilot
ID 永久排除在 confirmation 外。Family alpha 0.05、第一 Holm threshold、3% 最小相对效应与
80% target power 全部预先固定。Reducer 从 12 至 20 选择能同时满足两个 contrast 的最小
common final-block prefix，并在 confirmation 可见前封存；若没有合格数量，状态为
`UNDERPOWERED`，confirmation 不能开始。

最终 goodput effect 是 paired log ratio，并在独立 repetition block 上计算 95% BCa 区间。
Primary family 恰好包含 L0--Static 与 L0--TTS，使用 Holm family-wise adjustment。
Secondary breadth hypothesis 必须显式分组并使用 Benjamini--Hochberg FDR；看到结果后不能
把它升级到 primary family。

Long-context request-level summary 使用 hierarchical bootstrap：先重采样独立 block，再
在每个 sampled block 内重采样 request。Production arrival 实验重采样整个 time block，以
保留 block 内 tail dependence。二者均使用固定 95% 区间与已注册 seed/repetition count。
共享一个 service interval 的 request 不能视为独立 system replicate。

合法 evidence alias 共享一个 byte-equivalent Target-only observation，而不是复制 row。
Alias 验证 model/runtime/tree、sampling/seed、corpus/trace、limit、hardware/topology/rank、
method/server configuration、schema、output-token trajectory 与 timing contract。Static 不会
自动 alias；TTS/L0 永远不能 alias。Dependence map 会让所有 aliased consumer 在 resampling/
covariance 中共用一个 unit。当前 formal reducer 会对 non-singleton unit fail closed，除非
execution plan 与 terminal evidence 重新计算该 equivalence；structurally valid 的
self-described alias 不是 claim evidence。

Power、clock、temperature、throttling reason 与 background process 作为 per-block hardware
envelope 锁定。缺失或越界观测会使 block 无效，不能在事后选择为 covariate。

## 证据、恢复与结论

Evidence 通过有界 single-writer queue 增量写入 batched Parquet WAL row group。已注册 row/
time checkpoint 与 terminal boundary 保留 WAL fsync、directory sync、uniqueness、negative-row
durability 与 zero-drop coverage；queue 暂时清空不是隐式 fsync boundary。中断和 aborted
attempt 保持可审计，但其 row 被排除。Resume 只跳过一个完整身份及 file digest 都验证通过
的 receipt；出现竞争 completed attempt 属于错误。
每个 execution plan 还绑定 producer queue/batch 上限、writer queue 与 Parquet row-group
上限、checkpoint interval、overflow mode、SQLite WAL/FULL 设置，以及 file/checkpoint/
directory fsync gate。checkpoint、prepared receipt、terminal receipt 与 resume 必须认可同一
policy digest。

固定 source 现在会生成 content-bound all-reset capability、initial-state 与 boundary receipt。
该 receipt 在 CPU contract boundary 证明已注册的 drain、KV/prefix、RNG/counter、scheduler/
telemetry、adaptation state、allocator/HBM 与 completion-event predicate。其 GPU reset
semantics 仍为 `PENDING`。在支持的 single-tokenizer HTTP/1.1 uvicorn 路径上，连续 HTTP
accounting 来自真实 protocol connection create/close event，并以同一 HTTP process/
generation、单调且守恒的累计值绑定；request counter 与 client field 不能替代。Granian
HTTP/2 与 multiple-tokenizer HTTP-process 路径会在生成该 capability 前 fail closed。receipt
尚未 durable
绑定进 terminal envelope，连续 launch-to-terminate inventory accounting 也未完成。Source
patch 只覆盖 reset-state accounting，尚不生成 native warm-up、trace 或 close receipt。因此
所有 live GPU reuse 入口继续阻止，并为每条 trace
回退到独立 clean process 与 HTTP pool；该路径不声明 reset 或 startup saving。每个支持的
single-trace 路径中，submit/abort 共用 official HTTP pool，timeout 绑定 registered request
deadline 与 abort grace。Immutable
compile-cache base 仍按内容寻址并验证，每个
process 都有 private writable overlay，CUDA Graph 绝不跨 process 或 GPU。当前 release 的
fault-injection cell 始终使用 fresh process。

Native hook 现会绑定 capability、begin、reset、finalize、准确 terminal request coverage 与
ordered token ID，以及 Static aggregate safety 或 TTS/L0 request/round/update/KV/performance
evidence。Provider object attribute 不是 trust evidence。每个成功 serving run 还必须发布与
terminal 绑定的 `BudgetObservationReceipt`，覆盖所有注册 phase、measured gang GPU time、
whole-instance billed time 与准确 delta。缺失 component 保持 missing 或显式 N/A，绝不能为零。

Empirical gate 还要求 attestation 绑定 registry/manifest、selection/stage receipt、准确
runtime/capability 与 patched tree、model/tokenizer/data/trace identity、Target-only
reference、hardware/power report 及每个 final Parquet digest。缺少它时，即使本地算术为正，
状态仍为 `UNMEASURED`；有效 attested evidence 未通过注册标准时为 `BLOCKED`。仓库只包含
协议代码，不包含性能结论或 result artifact；本 release 也不包含任何 GPU 结果。
本 release 未配置 trusted hardware-attester identity，因此内容自洽但由 caller 编写的
attestation file 不能把 industrial 或 legacy analyzer 提升为 `MEASURED`；即使 hook 已实现，
Static/TTS/L0 仍在 mutation 前保持 blocked。

外部 `TrustedAttesterPolicyBundle` format 把 public verification key、nonce freshness/replay
policy、hardware-envelope allowlist 与 validity 绑定到另行 provision 的 anchor。Private signing
material 绝不能进入仓库或 experiment artifact。当前 source-release anchor 未配置，因此加载
operator bundle 本身不能授权 formal DAG。

协议把纯代码收口标为 `CPU_READY`，把已准备但尚未执行的 device check 标为
`GPU_SMOKE_READY`，只有合格 attested terminal evidence 才能标为 `MEASURED`。这些标签不是
递进结论：CPU 或 diagnostic smoke 成功仍然 non-measured。准确顺序与当前 blocker 见
[工程就绪矩阵及 SSH runbook](engineering-readiness.md)。
