# 实验协议

[English](../en/experiment-protocol.md) · [首页](../../README_zh-CN.md)

## 问题与当前状态

工业级研究考察在线 drafter adaptation 在何处有效、为何有效，以及其成本或操作风险何时
超过节省的 target work。Target-only、Static、TTS 与 L0 始终分开；TTS 与 L0 使用相同
candidate，只在发布时间上不同。

所有新 GPU 结果都是 `UNMEASURED`。代码、CPU 测试与 registry 建立目标 protocol 与
coordinator contract，不是完整可运行 speculative surface 或 benchmark 结果。Industrial
executor 当前只运行 TP1/DP1 Target-only。Static/TTS/L0 会在任何 mutation 前因缺少
`sglang.schema_v3.content_bound_terminal_speculative_evidence.v1` 而 `BLOCKED`；固定
integration 没有实现该 hook。Stage B 还因缺少 provider credential 与已注册硬件而 blocked。
历史 v2 evidence 只用于 regression/debugging，不得进入 schema-v3 selection、power sizing、
confirmation 或结论。

## 不可变 Dependency DAG

Registry 固定以下顺序：

```text
preflight -> E3a -> E1 -> E2 -> E4 -> E3b -> E1a -> E5 -> E6 -> E0
```

每个目标 definition 命名 dependency、locked output 与科学 axis。每个 cell 再绑定完整身份：
experiment/model/backend/task/method、parameterization/native scope、optimizer/schedule、
context/regime/width/arrival/SLO、cohort/topology、seed/block、GPU UUID、port、cache/evidence
root、workload class 及真实 status/reason。

只有准确 dependency receipt 验证通过且 cell 通过 executable-release preflight 后，stage
才可能 dispatch。Receipt 绑定 registry、
runtime、split、dependency-output 与 locked-output SHA-256，并声明 selection 已在下游
unblinding 前封存。编辑 registry artifact 或以另一 digest 重新序列化 dependency 都会
fail closed。

## Stage 与 Locked Decision

| Stage | 目的 | 下游使用前锁定的输出 |
|---|---|---|
| Preflight | source/runtime/model/data identity、exactness、HBM、telemetry、双 GPU interference | runtime envelope |
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

这些是已注册 scientific grid，不是当前 executor support。固定 patch 会拒绝
DSpark/EAGLE/EAGLE3/NEXTN adaptation、非 constant schedule 与全部 multi-rank execution。
即使其底层 TP1/DP1 DFlash path，也必须先实现 native terminal provider 才能生成可声明的
industrial evidence。

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

## 双 GPU Staging

Registry 要求两个显式 GPU UUID。并行工作前，preflight 记录 clock、temperature、power
state、background process、driver/runtime identity、topology receipt、per-rank HBM 与
interference receipt。缺少该 receipt 时，确定性 scheduler 会串行全部单 GPU cell。

双 GPU、headline、profiler、download 与 compile cell 都是 exclusive。只有 interference
gate 通过后，GPU UUID、port、cache root 与 evidence root 都不相交的两个单 GPU cell 才能
共享 dispatch wave。Queue 是数据，不是同时启动全部 argv 的指令。

TP2 与 sticky-replica DP2 是目标 coordinator contract。未来 release 将要求 verified
patched-runtime capability receipt，并由全部 rank 为同一 publication identity 提供
prepare/decide/apply/receipt evidence。当前 `RunConfig` 会在 model loading 前拒绝全部
TP2/DP2 cell；CPU `gloo` harness 只是状态机测试，不能启用这些 cell。

任何 stage 都不声明 multi-node、超过两个 rank、Kubernetes scheduling、elastic membership
或 automatic failover。

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

恰好四个 paired pilot block 只用于估计 L0--Static 与 L0--TTS 的 log-effect variance；
pilot ID 永久排除在 confirmation 外。Family alpha 0.05、第一 Holm threshold、3% 最小相对
效应与 80% target power 全部预先固定。Power grid 从 12 至 20 选择能同时满足两个 contrast
的最小 common final-block count；若没有合格数量，状态为 `UNDERPOWERED`，confirmation
不能开始。

最终 goodput effect 是 paired log ratio，并在独立 repetition block 上计算 95% BCa 区间。
Primary family 恰好包含 L0--Static 与 L0--TTS，使用 Holm family-wise adjustment。
Secondary breadth hypothesis 必须显式分组并使用 Benjamini--Hochberg FDR；看到结果后不能
把它升级到 primary family。

Long-context request-level summary 使用 hierarchical bootstrap：先重采样独立 block，再
在每个 sampled block 内重采样 request。Production arrival 实验重采样整个 time block，以
保留 block 内 tail dependence。二者均使用固定 95% 区间与已注册 seed/repetition count。
共享一个 service interval 的 request 不能视为独立 system replicate。

Power、clock、temperature、throttling reason 与 background process 作为 per-block hardware
envelope 锁定。缺失或越界观测会使 block 无效，不能在事后选择为 covariate。

## 证据、恢复与结论

Evidence 逐步写入有界、fsynced Parquet WAL segment。只有 table coverage 完整且没有不允许
的 drop，才能组装 final shard 并获得 exclusive content-bound receipt。中断和 aborted
attempt 保持可审计，但其 row 被排除。Resume 只跳过一个完整身份及 file digest 都验证通过
的 receipt；出现竞争 completed attempt 属于错误。

该端到端 evidence path 当前只会关闭 Target-only run。所有 speculative method 都需要准确
固定 native terminal hook 来绑定 request/round/update/performance row；缺少 hook 是 mutation
前的 `BLOCKED` outcome，不代表可以合成或省略 evidence。

Empirical gate 还要求 attestation 绑定 registry/manifest、selection/stage receipt、准确
runtime/capability 与 patched tree、model/tokenizer/data/trace identity、Target-only
reference、hardware/power report 及每个 final Parquet digest。缺少它时，即使本地算术为正，
状态仍为 `UNMEASURED`；有效 attested evidence 未通过注册标准时为 `BLOCKED`。仓库只包含
协议代码，不包含性能结论或 result artifact；本 release 也不包含任何 GPU 结果。
本 release 未配置 trusted hardware-attester identity，因此内容自洽但由 caller 编写的
attestation file 不能把 industrial 或 legacy analyzer 提升为 `MEASURED`。
