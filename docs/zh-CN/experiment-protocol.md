# 实验协议

[English](../en/experiment-protocol.md) · [首页](../../README_zh-CN.md)

## 问题与当前状态

工业级研究考察在线 drafter adaptation 在何处有效、为何有效，以及其成本或操作风险何时
超过节省的 target work。Formal role 是 Target-only、Static、TTS、L0-naive 与 LightCone。
Recipe authority 与 publication policy 正交：TTS 使用 frozen primary-source recipe 与 fixed
barrier；L0-naive 使用同一个 authority 与 first-ready publication；LC-candidate 与 E2-sealed
LightCone winner 分别使用 L0 policy 加 search recipe 与 sealed recipe。

所有 formal industrial GPU 结果都是 `UNMEASURED`。Staged registry、CPU 测试与 runtime
contract 本身不能授权 benchmark 结果。单可信操作者路径可以从一个 clean Git HEAD/tree、
完整 semantic SGLang patch series、schema-v5 trusted ProtocolLock、准确 external-source lock
与 fresh dynamic GPU qualification 采集完整 empirical evidence；但缺少当前 root-authorized
attestation 时，这些证据只能标记为 `trusted_single_operator_empirical_no_signature`，且
`formal_measured=false`、状态保持 `UNMEASURED`。Release-level `MEASURED` 结果还必须具备
仓库 Ed25519 public root chain。相应 run-specific receipt 未就绪时，每个 assignment 都会在
分配资源前 fail closed。CPU state-machine test、caller-authored digest 与旧 smoke 结果都不能
把 TP2/DP2、native ITL、DSpark、NEXTN 或 EAGLE3 提升为 formal support。历史
使用 shared tuned AdamW 的历史 row 只是 matched-recipe publication-policy diagnostic，不是
TTS-paper reproduction；它们只用于 regression/debugging，不得进入 schema-v3 selection、
power sizing、confirmation 或结论。

## 不可变 Dependency DAG

Registry 固定以下顺序：

```text
preflight -> E3a -> TTS-Cal -> E1 -> E2 -> E4 -> E3b -> E1a -> E5 -> E6 -> E0
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
| TTS-Cal | 288 个不相交 tuning-only row：9 个 learning rate x 8 个 stride x 4 个 excluded pilot | 一个 frozen TTS recipe |
| E1 | E3a/TTS-Cal 封存后恰好 68 行 | safe LC Pareto set 与 common load |
| E2 | 四轮 successive materialization；`n0=105g`、`n(k+1)=max(ceil(nk/4),21)`，每轮另加四个 anchor | 一个 sealed LightCone recipe |
| E4 | 48 个 strength-2 screen、96 个 local-factorial 与 3 个隔离 profiler row | mechanism gate |
| E3b | `480B` 个 long-context row，其中 `B=4+N`，`N` 为封存的 12--20 final prefix | long-context confirmation |
| E1a | 58 个 configuration x 2 个 verification mode = 116 行 | 一个 DSpark recipe |
| E5 | `450B+264` 行：powered headline family，加上 11 类故障 x 2 backend x 3 topology x 4 cohort-count 的 one-shot diagnostic 矩阵 | production/topology surface |
| E6 | 只覆盖两个 NEXTN target model 的 `2+60B` 行 | native MTP transfer surface |
| E0 | 先签署 108 个 compatibility decision，再为 `V` 个有效 model/backend/task 组合 materialize `16VB` 行 | OnlineSPEC breadth surface |

E1 恰好 68 行：Target-only、Static、frozen TTS、frozen L0-naive，以及 32 个 Full/LoRA
LightCone geometry 各自的两个 optimizer anchor。L0-naive 只作 mechanism anchor，不能进入
candidate ranking。E2 只调优 LightCone candidate。每个 E1 survivor geometry 注册七个
optimizer（`Adam`、`AdamW`、`SGDM`、`NAG`、`Muon`、`Lion` 与项目自有
`ChronoBelief`）、三个 schedule 与五个 learning rate，因此第零轮每 geometry 有 105 个
recipe。后续每轮必须是前一 sealed round 的准确子集，candidate 不得重新进入。每轮另加
四个 fixed-role anchor，所以 E2 总数为 `16 + sum(n0..n3)`，不再展开 eager sentinel matrix。

TTS reconstruction authority 固定 Adam `(beta1=.9, beta2=.999,
epsilon=1e-8, weight_decay=0)`、一次 update step、full-drafter training、无 clipping、
latest-round-only supervision、drafter-native position weight、source-point proximal term、
request reset 与 side stream。TTS-Cal 将 learning rate
`1e-7, 3e-7, ..., 1e-3`、stride `1,5,10,15,20,30,40,50` 与四个 excluded pilot
交叉；先安全、后最大化 SLO-goodput，并在 E1 前签名封存。TTS 使用 fixed barrier；
L0-naive 消费相同的 frozen recipe/candidate byte，只把 publication 改为 first-ready；两者都
不得继承 E2 winner。

ChronoBelief 是项目自有的 preregistered optimizer。其身份绑定四个更新方程、PDF/TeX
digest、标准 update-count bias correction、decoupled weight decay 与 safe-boundary age
`d_r`。Skip 与 abort 必须让 moments 和 update counter 保持 byte-identical；只有 committed
proposal 可以推进 optimizer state。

E1a 恰有 58 个 configuration：56 个 adaptive configuration 加两个 fixed reference。32 个
layer-only adaptive configuration 将
`last1/last3/last5/all` 与 Full 加七个 LoRA rank 交叉，并冻结 DSpark native head。24 个
hybrid cell 将 `last1/last3/last5` 与相同八种 backbone parameterization 交叉，同时把原生
W1、W2 与 scalar acceptance/confidence state 作为 Full 训练。每个 configuration 分别运行
fixed verification budget 与 native scheduler，总计 116 行。

这些是 signed staged materialization，不是展开未来矩阵的指令。E1 消费 sealed E3a 与
TTS-Cal receipt，只 materialize 一个 68-row width/load slice。E2 每次只 materialize 一个
successive-halving round，把每个 LC-candidate 与 fixed Static、frozen-TTS reference 比较，
保留 family floor，且下一 round 只能从 prior sealed survivor receipt 派生。只有 tuning-only
Formal E2 reducer 要求 native per-token production/commit timestamp。若 SSE chunk 合并
多个 token，也绝不把时间间隔虚构均分；native timestamp coverage 缺失或不完整会直接阻止
reducer。Confirmation materialization 以
family 为局部单位：四个 excluded pilot 会在 confirmation 可见前归约，随后只激活 sealed
12--20-block final prefix。
Family 仍是增量 scheduling/power 单位，但 stage dependency receipt 只能由独立 exact-coverage
aggregate 签发：所有 family 按 SHA 排序，并从 raw pilot/final completion authority 现场重放。
E5 的 264 个 deterministic failure injection 是 one-shot diagnostic pass，不乘 confirmation
block。Selected p99 anchor 至少需要 10,000 个 completed request，但该要求附着于既有
headline cell。Pilot reducer 会在 final unblinding 前封存每个选中的 backend/topology
family；该既有 family 的五个 paired method cell 在每个 included block 中都消费同一份准确
11,000-offer extension pool。Extension 不新增 materialized row。Headline family 与 264 个
failure diagnostic 共同组成准确的 signed materialization。

E3b 每个 block materialize 480 行。E5 每个 block materialize 450 个 headline row，另加
264 个 one-shot failure diagnostic；sealed selected-p99-anchor set 只为匹配的 headline
cell 增加执行要求。E6 对每个 target model 只运行一次、全局去重的 immutable-interface
preflight，再每 block materialize 60 行；target model 恰好为
`Qwen/Qwen3.6-35B-A3B` 与 `Qwen/Qwen3.5-122B-A10B-FP8`。Source-owned producer 只从
两个 frozen target snapshot 自身内置的 `mtp.*` component 派生准确两条 TP2 launch，禁止
独立 draft-model path。TTS 始终使用 frozen external recipe；LightCone 使用 sealed E1/E2
winner，不在 E6 retune。

E0 先为四个 model 与三个 backend 发布准确 12 个 source-owned pre-probe interface；其中
不预装 task proof。随后 physical campaign 运行 12 个 launch group，每组覆盖九个 task-native
probe，最终产生准确 108 个 model/backend/task compatibility terminal。Task-specific EAGLE3
proof row 只能在对应 one-request core evidence 成功后发布。只有结果中 `V` 个有效 decision
才 materialize serving row：八个 role、task-native request、concurrency one 与 common SLO
load，共 `16VB` 行。若全部 N/A，compatibility receipt 可以合法地产生零个 timing row。

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
顺序。配对绑定该完整 registered source pool 的 digest，而不是各 method 实际消费的 offered
prefix 或 response token。Observed output 仍完整保存在证据中；两个 method 都 completed 的
同一 request 由独立 exactness gate 比较。每个实际 offered request 恰好记为 rejected、
completed、timed out、cancelled 或 unfinished；unfinished 工作按注册 timeout boundary 留在
denominator。缺失 row 不补零。

## GPU Pool Staging

Registry 使用两个 logical rank slot；physical device 只来自 content-bound `GpuInventory`
与 frozen assignment。唯一 deterministic same-host scheduler 支持任意 inventory size，并对
1、2、4、8、16 GPU 有明确测试。它会在一个合法 topology group 上 atomic 分配 k-GPU
TP/DP shape，拒绝 partial gang 与 cross-host placement，在 eligible UUID 之间轮换 independent
block，并阻止 GPU、port、cache writer、evidence root 与 exclusive resource 重叠。

`GpuFleetInventory` 在该 same-host scheduler 之上组合多台 independently verified host。
它在 eligible host 间均衡 independent cell 与完整 confirmation block，同时保留 host-local
inventory、interference、port、cache、evidence 与 materialization identity。Matched baseline
anchor 与完整 confirmation block 留在同一 host/GPU。每台主机的 headline concurrency 只由该主机已校准 envelope 限制；
异构 hardware envelope 不进入同一 family。跨主机 gang 以
`cross_host_collectives_unvalidated` 拒绝。

并行工作前，preflight 记录 clock、temperature、power state、background process、driver/
runtime identity、topology、per-rank HBM 与准确 `InterferenceEnvelope`。Envelope 按 hardware、
workload、co-run signature/count、gang shape、thermal/power/load state 与 host contention
索引。如果八 GPU host 只校准过 two-way concurrency，frozen headline wave 仍最多 two-way；
runtime completion 不能创造 result-dependent co-tenancy。Profiler、download 与 compile
工作是 exclusive-host，不能与 headline timing 竞争。

Disk admission 使用 signed stage-specific capacity envelope，将静态 input sizing、
compile/cache high-water、evidence high-water、retry reserve 与 safety margin 绑定到准确
dispatch schedule。只有 authority 缺失或 unavailable 时才使用 legacy 100 GB free-space
规则；签名错误或 envelope tamper 绝不 fallback。扩容本身不会自动通过，也不会为 admission
删除用户数据。

Exclusive-host 分类不等于 execution authority。COMPILE 与 DOWNLOAD 使用不同的 first-party
runner；typed assignment 绑定 budget、inventory/GPU UUID、source checkout、prepared
model/tokenizer content、RunConfig、launch argv/port、deterministic prewarm 或 download plan，
以及 graceful shutdown。每个 runner 都发布 immutable terminal 与 atomic no-replace result
pointer；resume 必须重开每个 raw file 与 sidecar。缺少 dynamic control、terminal coverage
不完整或 cache pointer 不一致都会阻止 activation。

每个 dispatch plan 为每个 assignment 绑定准确 `ExperimentBudget` digest。Wall time、requested
gang compute GPU time、reserved GPU time 与 fixed-instance billed GPU time 必须分开；two-GPU
gang 消耗两倍 wall time，而 whole-instance billing 对 observed wall interval 计费整个 frozen
inventory。Queue 是数据，不是同时启动全部 assignment 的指令。

`RunConfig` 只支持单机、最多两 rank 的 `tp1_dp1`、`tp2_dp1` 与 `tp1_dp2`。TP2 要求全部
rank 为同一 publication identity 提供 prepare/decide/apply/receipt evidence，并证明任一
rank abort 时 partial publication 为零。DP2 是 sticky cohort router 后的两个 TP1 replica，
禁止跨 replica gradient averaging。CPU state-machine coverage 不能启用这些 topology；必须
由 fresh root-authorized mode-specific GPU qualification receipt 绑定准确 patched tree 与双卡
inventory。

Multi-host control 通过 content-bound request/receipt 分发 independent host-local work。
任何 stage 都不声明跨主机 collective、world size 大于二、Kubernetes scheduling、elastic
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
LightCone--TTS 与 LightCone--Static 的 log-effect variance。该 identity 绑定 experiment/model/backend/task、
context/regime/load/arrival、width panel、topology、cohort/method family、runtime/split/trace/
sampling 与 hardware envelope；一个 family 的 pilot 不能为另一个 family 计算 power。Pilot
ID 永久排除在 confirmation 外。Family alpha 0.05、第一 Holm threshold、3% 最小相对效应与
80% target power 全部预先固定。Reducer 从 12 至 20 选择能同时满足两个 contrast 的最小
common final-block prefix，并在 confirmation 可见前封存；若没有合格数量，状态为
`UNDERPOWERED`，confirmation 不能开始。

最终 goodput effect 是 paired log ratio，并在独立 repetition block 上计算 95% BCa 区间。
Primary family 恰好包含 LightCone--TTS 与 LightCone--Static，使用 Holm family-wise
adjustment。预注册 secondary decomposition 是 L0-naive--TTS 与 LightCone--L0-naive。
其他 secondary breadth hypothesis 必须显式分组并使用 Benjamini--Hochberg FDR；看到结果后
不能把它升级到 primary family。已注册的报告目标包含 method-by-model、method-by-context 与
method-by-load interaction；不预设 interaction 或 large-model effect 为正，也不根据观察到的
LightCone gain 选择 model。当前 `CrossFamilyInteractionReducerArtifact` 只记录一个内容绑定、
structural、non-formal 的 `UNRESOLVED` contract；它不能证明 GPU coverage 已完成、interval
有效或存在 formal interaction effect。只有已注册 statistical reducer 消费完整 attested final
evidence 后，才可能形成这类结论。

Long-context request-level summary 使用 hierarchical bootstrap：先重采样独立 block，再
在每个 sampled block 内重采样 request。Production arrival 实验重采样整个 time block，以
保留 block 内 tail dependence。二者均使用固定 95% 区间与已注册 seed/repetition count。
共享一个 service interval 的 request 不能视为独立 system replicate。

合法 evidence alias 共享一个 byte-equivalent Target-only observation，而不是复制 row。
Alias 验证 model/runtime/tree、sampling/seed、corpus/trace、limit、hardware/topology/rank、
method/server configuration、schema、output-token trajectory 与 timing contract。Static 不会
自动 alias；adaptive scientific role 永远不能 alias。Dependence map 会让所有 aliased consumer 在 resampling/
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

固定 source 会生成 content-bound all-reset capability、initial-state 与 boundary receipt，
覆盖 drain、KV/prefix、RNG/counter、scheduler/telemetry、adaptation state、allocator/HBM 与
completion event。只有准确 identity 通过 reset/address/value/HBM qualification 后才能启用
GPU reuse；verified proof 只授权实测 identity 与 topology。在支持的 single-tokenizer
HTTP/1.1 uvicorn 路径上，连续 HTTP
accounting 来自真实 protocol connection create/close event，并以同一 HTTP process/
generation、单调且守恒的累计值绑定；request counter 与 client field 不能替代。Granian
HTTP/2 与 multiple-tokenizer HTTP-process 路径会在生成该 capability 前 fail closed。Terminal
envelope 绑定 warm-up、trace、reset、close 与连续 launch-to-terminate inventory accounting。
若任一 receipt 缺失或 dynamic proof 不匹配，executor 会为每条 trace 回退到独立 clean
process 与 HTTP pool，并且不记录 startup-saving claim。Immutable
compile-cache base 仍按内容寻址并验证，每个
process 都有 private writable overlay，CUDA Graph 绝不跨 process 或 GPU。当前 release 的
fault-injection cell 始终使用 fresh process。

Native hook 现会绑定 capability、begin、reset、finalize、准确 terminal request coverage 与
ordered token ID，以及 Static aggregate safety 或 adaptive-role request/round/update/KV/performance
evidence。Provider object attribute 不是 trust evidence。每个成功 serving run 还必须发布与
terminal 绑定的 `BudgetObservationReceipt`，覆盖所有注册 phase、measured gang GPU time、
whole-instance billed time 与准确 delta。缺失 component 保持 missing 或显式 N/A，绝不能为零。

单可信操作者 final audit 可以在无签名时收口完整且可复现的 empirical evidence，但必须把
closure 标记为 `trusted_single_operator_empirical_no_signature` 并保持
`formal_measured=false`。要提升为 `MEASURED`，还必须取得 attestation，绑定 registry/
manifest、selection/stage receipt、准确 runtime/capability 与 patched tree、model/tokenizer/
data/trace identity、Target-only reference、hardware/power report 及每个 final Parquet digest。
缺少它时，即使 empirical campaign 和本地算术都完整，状态仍为 `UNMEASURED`；有效 attested
evidence 未通过注册标准时为 `BLOCKED`。仓库只包含协议代码，不包含性能结论或 result
artifact；本 release 也不包含任何 GPU 结果。
仓库只保存一个 offline Ed25519 public root 及其 fingerprint。Private signing material 绝不能
进入仓库、远端实例、argv、environment、log 或 experiment artifact。Root-authorized
deployment policy 绑定 control type、nonce freshness/replay、准确 inventory/hardware、validity
与 source lineage。Dispatch、compile、non-serving terminal、capacity、interference 与 rank
aggregate control 共用该 policy 与同一个 atomic replay reservation。内容自洽但由 caller
自建 key 或 bundle 仍不能授权 formal DAG。

协议把纯代码收口标为 `CPU_READY`，把已准备但尚未执行的 device check 标为
`GPU_SMOKE_READY`，只有合格 attested terminal evidence 才能标为 `MEASURED`。这些标签不是
递进结论：CPU 或 diagnostic smoke 成功仍然 non-measured。
