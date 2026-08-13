# CLI

[English](../en/cli.md) · [首页](../../README_zh-CN.md)

## 命令 Surface

准确参数以 `lightcone-spec --help` 与 `lightcone-spec COMMAND --help` 为准。Schema-v3 与
industrial 命令包括：

| 命令 | 用途 |
|---|---|
| `doctor` | 只读 host、Python、CUDA 与 source identity 报告 |
| `validate-config` | 验证一份严格 schema-v3 run config 与 sidecar |
| `bind-formal-workload-authority` | 绑定 release allowlist 中的本地 LiveCodeBench v6 Hard 或 MATH-500 Level 5 raw source |
| `revalidate-formal-workload-authority` | 重开诊断性 workload binding，并重放其 path、revision、raw bytes 与完整筛选结果 |
| `build-industrial-registry` | 绑定一个或更多稳定 logical rank slot 并生成不可变实验 DAG |
| `collect-gpu-inventory` | 收集 nonce-bound physical GPU/topology inventory 与 raw probe receipt |
| `build-interference-envelope` | 派生当前 serial interference envelope 及其 inventory-bound raw receipt |
| `materialize-interference-calibration-bootstrap` | 从 raw preflight activation 与准确 inventory 派生仅供校准的 two-way execution envelope |
| `reduce-interference-calibration` | 重开 path-bound execution bundle 与 terminal authority，把 raw isolated/simultaneous evidence 归约为准确 cardinality rule |
| `reduce-e1-activation` | 从 sealed E3a evidence 派生唯一 130-cell E1 slice |
| `reduce-e2-activation` | 从 E1 Pareto artifact 或 prior survivor materialize 一个 E2 round |
| `reduce-e2-successive-halving` | 把准确 E2 stage evidence 归约为 sealed survivor receipt |
| `materialize-confirmation-pilots` | 为一个 confirmation family 激活恰好四个 excluded pilot |
| `reduce-confirmation-family-power` | 把一个 family 的准确四个 pilot block 归约为 sealed power plan |
| `materialize-confirmation-prefix` | 只激活 powered family 已封存的 12--20-block final prefix |
| `validate-evidence-alias` | 用 registry、hardware、inventory 与 terminal authority 重放 raw alias manifest |
| `build-evidence-dependence-map` | 从 reducer 签发的 alias artifact 保留 shared-observation dependence |
| `materialize-stage-activation` | 从 raw registry、lineage、runtime 与 split authority 重放 generic stage dispatchability |
| `materialize-industrial-budgets` | 从 reducer、load、inventory、policy 与 capacity authority 派生 fail-closed `BudgetPlan` |
| `bind-industrial-budget-authority` | 把声明的 `BudgetPlan` 绑定到完整 tagged raw activation/load/capacity 闭包 |
| `estimate-industrial-budget` | 在准确 physical inventory 与 interference envelope 上重放 ready `BudgetPlan` |
| `plan-industrial-dispatch` | 冻结确定性、topology-aware GPU-pool wave 与 physical assignment |
| `materialize-dispatch-execution-bundles` | 绑定一份 path-only raw input graph，并在全 assignment 预检后发布完整 schema-v5 assignment-bundle set |
| `execute-dispatch-wave` | 重开一份已提交的 materialization manifest，并在 release authority 完整时执行一个 receipt-bounded frozen wave |
| `seal-industrial-stage` | 绑定 activated completion、disposition、budget、runtime、split、dependency 与 locked output |
| `analyze-industrial` | 验证 schema-v3 terminal、budget、family-power 与 hardware evidence |
| `analyze-e3b-long-context` | 验证并归约隔离的 E3b long-context evidence family |
| `build-preliminary-speed-study` | 生成历史 `PRELIMINARY_DIAGNOSTIC_ONLY` 协议；绝不构成 industrial authority |
| `lock-models` | 把 model ID 解析为不可变 revision |
| `prepare-models` | 下载或 offline 验证 locked snapshot |
| `list-preliminary-tuning-candidates` | 写入历史 Full/LoRA 诊断 grid |
| `render-preliminary-target-only-runtime` | 渲染诊断性、关闭 speculation 的 Target-only endpoint |
| `render-preliminary-static-load-runtime` | 渲染诊断性、零 adaptation 分配的 Static endpoint |
| `render-preliminary-tuning-runtime` | 渲染 matched preliminary TTS/L0 endpoint |
| `run-preliminary-controlled-slice` | 测量一个 preliminary controlled slice |
| `collect-preliminary-static-load-screen` | 验证 preliminary Static load coverage |
| `advance-preliminary-tuning-stage` | 验证 preliminary halving stage |
| `select-preliminary-speed-config` | 应用历史 tuning-only rule |
| `select-preliminary-anchor-config` | 锁定 preliminary anchor，但不形成 claim |
| `render-preliminary-runtime` | 生成 matched preliminary config 与 launch argv |
| `build-preliminary-confirmation-queue` | 生成 clean-server 诊断 job |
| `run-preliminary-confirmation` | 执行一个 preliminary method/block slice |
| `run-preliminary-target-reference` | 捕获 preliminary Target-only token-ID 轨迹 |
| `collect-preliminary-speed-study` | 派生 preliminary 诊断 table |
| `render-preliminary-replication-runtime` | 渲染 preliminary natural-task 或 profiler slice |
| `run-preliminary-natural-slice` | 运行一个 preliminary natural-task slice |
| `build-preliminary-profiler-plan` | 生成隔离的诊断 profile plan |
| `attest-preliminary-speed-study` | 输出确定性的 non-authority decision（退出 42） |
| `analyze-preliminary-speed-study` | 只生成 preliminary 统计（退出 42） |

隔离的 OnlineSPEC command family 为：

| 命令 | 用途 |
|---|---|
| `build-onlinespec-study` | 生成 provenance-bound comparison protocol |
| `verify-onlinespec-source` | 按 source audit 验证外部 clean checkout |
| `list-onlinespec-candidates` | 写 OGD、optimistic 与 ensemble candidate |
| `render-onlinespec-tuning-runtime` | 渲染 paired Static/candidate tuning endpoint |
| `run-onlinespec-tuning-slice` | 测量一个 learner tuning slice |
| `advance-onlinespec-tuning-stage` | 各 learner 独立 halving |
| `select-onlinespec-config` | 每个 learner 选择一个 safe terminal candidate |
| `select-onlinespec-anchor-config` | 锁定三个 terminal anchor，不声明 grid optimum |
| `render-onlinespec-runtime` | 渲染 sequential Static 与 learner endpoint |
| `build-onlinespec-queue` | 生成 randomized clean-server comparison job |
| `run-onlinespec-confirmation` | 执行一个 learner/method block |
| `collect-onlinespec-study` | 派生隔离 comparison table |
| `attest-onlinespec-study` | 输出确定性的 non-authority decision（退出 42） |
| `analyze-onlinespec-study` | 只生成 learner-versus-Static 诊断区间（退出 42） |

历史 speed-study manifest/API/CLI surface 被严格固定为
`PRELIMINARY_DIAGNOSTIC_ONLY`。它不能消费 industrial registry、activation、budget、
completion authority 或 materialized bundle；其 table 与 receipt 也绝不会被
`analyze-industrial` 接受。历史 schema-v2 manifest 仍可在强制 preliminary scope 下读取，
但不能改写或重新标成 formal。唯一 formal execution 命令是 `execute-dispatch-wave`；它会
重开 materialized bundle 并进入 industrial executor。不存在 generic method override，
也不存在把无 receipt 目录转换为 completed evidence 的命令。

Bootstrap envelope 只授权生成八个已注册 Static calibration observation，不能授权
headline co-tenancy 或更大 cardinality。`reduce-interference-calibration` 会在打开 caller
path 前先检查 release trust root；因此当前无 signer 的 release 只写入 `BLOCKED` decision，
不会生成 calibrated envelope。Preflight stage 只有在重开同一份 raw execution authority
后才能封存 `runtime_envelope=PATH`，绝不接受手写 rule list 或 bare digest。

## Formal external workload authority

这些命令绝不会下载 formal benchmark 数据。Bind 命令只接受
`livecodebench_v6_hard` 或 `math500_level5`；准确 repository revision、raw-file
SHA-256、row count、筛选协议与完整 selected-row SHA-256 全部由源码拥有。它按 raw 顺序
选择全部准确匹配项，绝不截取前 32 条。

```bash
lightcone-spec bind-formal-workload-authority \
  --workload math500_level5 \
  --source /absolute/path/to/math500-locked.json \
  --output artifacts/industrial/math500-workload-authority.json

lightcone-spec revalidate-formal-workload-authority \
  --authority artifacts/industrial/math500-workload-authority.json
```

本 release 的 formal-workload source allowlist 为空。因此 bind 会在检查 source path 或创建
output directory 之前返回 42，并报告 `status=BLOCKED` 与
`reason_code=formal_workload_source_allowlist_empty`。未来 allowlisted source 可以生成
`BOUND_DIAGNOSTIC` artifact，但该 artifact 永久携带
`formal_execution_authorized=false`；它只是 workload input identity，不是 GPU dispatch、
attestation 或 formal result。

## Industrial Registry 工作流

Registry 只包含 scientific identity 与一个或更多 logical rank slot，不包含 host-specific GPU UUID：

```bash
lightcone-spec build-industrial-registry \
  --logical-gpu-slot logical-rank-slot-0 logical-rank-slot-1 \
  --base-port 24000 \
  --cache-root runtime-cache/industrial \
  --evidence-root artifacts/industrial \
  --seed 20260811 \
  --output artifacts/industrial/registry.json
```

输出嵌入 generator identity、输入 parameter、完整 declaration 与 registry SHA-256。加载时
会重新生成 registry 并比较准确内容，因此手工编辑 cell 会被拒绝。独立 content-bound
inventory 提供 physical UUID、model/memory/capability、PCI/NUMA/interconnect topology、
availability 与 topology group。

没有 E1/E2 或 confirmation-family 专用 reducer 的 stage 使用 first-party generic reducer。
它的 raw manifest 只能准确包含 registry、stage name、runtime、split 与完整有序 dependency
receipt path；不接受 caller cell list 或 bare activation plan。Preflight 的 receipt list 为空，
canonical genesis authority 会由准确 registry 派生：

```json
{
  "schema_version": 1,
  "kind": "industrial_registry_stage_activation_manifest",
  "registry_artifact": "artifacts/industrial/registry.json",
  "experiment": "preflight",
  "runtime_artifact": "artifacts/industrial/runtime.json",
  "split_artifact": "artifacts/industrial/preflight-split.json",
  "dependency_receipts": []
}
```

为 manifest 写相邻 `.sha256` 后，可生成 canonical artifact 供检查：

```bash
lightcone-spec materialize-stage-activation \
  --manifest artifacts/industrial/preflight-activation-manifest.json \
  --output artifacts/industrial/preflight-activation.json
```

Consumer 通过 `--activation-plan` 接收 raw manifest path 并现场重放 reducer；单独传序列化
output 会被拒绝。Reducer 从 registry status 与 release dispatch predicate 为每个 declared
stage cell 生成 disposition。当前 release 的全部 preflight cell 都被阻止：Target-only
compile cell 缺少 release-owned 的准确 prewarm manifest、graceful-finalization ACK 与原子
attempt/final-cache result pointer；Static/TTS/L0 也没有受支持的 execution contract。
因此 command 写出 canonical `BLOCKED` artifact 并返回 42。E6 download cell 由于没有
first-party download terminal contract，也会被独立阻止。

Budget materialization 会 fail closed。它消费 reducer-generated stage/family activation、完整
`BudgetPolicy`、每个 activated serving cell 各一份 `BudgetLoadBinding`、独立来源的
`CapacityEnvelope`，以及完整 physical inventory。Formal capacity authority 还必须成对传入
`--capacity-manifest` 与 `--capacity-verification-receipt` raw artifact。Verifier 会重新打开
每个 path-bound inventory/source、provider quota、host capacity 与逐 cell sizing input，且
只接受 release-owned trust root。只有 load coverage、provider GPU quota 与 maximum-attempt
disk capacity 全部闭合时才写 `READY`；否则保留真实派生 budget，写入不可变的诊断性
`UNRESOLVED` plan 并返回 42。Bare envelope 或 SHA-256 绝不构成 execution authority。
每个 activated serving cell 都要重复一次 `--budget-load-binding PATH`：

```bash
lightcone-spec materialize-industrial-budgets \
  --registry artifacts/industrial/registry.json \
  --activation-plan artifacts/industrial/preflight-activation-manifest.json \
  --inventory artifacts/industrial/inventory.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --capacity-manifest artifacts/industrial/capacity-source-manifest.json \
  --capacity-verification-receipt artifacts/industrial/capacity-verification.json \
  --output artifacts/industrial/budget-plan.json
```

不存在 caller-selected cell 或 duration-sequence fallback。Preflight 现在由 generic reducer
闭合；若省略 bound raw manifest，仍会得到明确 unresolved authority error，caller-authored
activation 不能绕过。Serving activation 还要为每个 activated serving cell 重复传入一次
`--budget-load-binding PATH`。

Formal consumer 不接受单独的 `BudgetPlan`。Materialization 后，必须用 tagged raw
activation manifest（不是 serialized activation summary）发布其 path-bound 原始闭包：

```bash
lightcone-spec bind-industrial-budget-authority \
  --activation-manifest artifacts/industrial/preflight-activation-manifest.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --capacity-manifest artifacts/industrial/capacity-source-manifest.json \
  --capacity-verification-receipt artifacts/industrial/capacity-verification.json \
  --budget-plan artifacts/industrial/budget-plan.json \
  --output artifacts/industrial/budget-authority.json
```

每个 activated cell 还要重复 `--budget-load-binding PATH`。该命令会重开所有原始来源、
现场重跑注册 reducer 并准确比较 declared plan，只写出 binding；它绝不会把
`UNRESOLVED` plan 变成 execution permission。

Planning 会重新加载 `BudgetPlan`，用相同 raw input 现场重跑 materialization，并在调度前
要求两者完全相等；同时还强制要求成对的 interference authority：

```bash
lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
  --activation-plan artifacts/industrial/preflight-activation-manifest.json \
  --inventory artifacts/industrial/inventory.json \
  --interference-envelope artifacts/industrial/interference-envelope.json \
  --budget-plan artifacts/industrial/budget-plan.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --capacity-manifest artifacts/industrial/capacity-source-manifest.json \
  --capacity-verification-receipt artifacts/industrial/capacity-verification.json \
  --output artifacts/industrial/dispatch.json
```

Frozen plan 会绑定准确 inventory/interference envelope、每个 cell 的 budget digest、physical
UUID/rank/port/topology assignment、wave membership 与 scheduler identity。Scheduler 接受任意
same-host inventory，并对 1/2/4/8/16 GPU 有 regression test。一个 wave 绝不超过已校准的
co-run class，因此八块空闲 GPU 不表示可以执行八路 headline concurrency。Envelope 通过
structural loading 不能证明 calibration；formal timing 仍要求 raw isolated/co-run
evidence 与 trusted hardware binding。

`materialize-dispatch-execution-bundles` 闭合 frozen planner output 与 formal execution 之间的
结构边界。它的 schema-v1 request 只包含 absolute resolved raw-artifact path、对应 source role，
以及 dispatch 中每个 cell 各一组 assignment-runtime path；刻意不接受 caller-supplied
assignment SHA、execution-plan SHA/summary、output root 或 semantic hash。Reducer 会重开 request
及每一份 sidecar，从 frozen dispatch plan 现场派生 assignment，并要求准确的一一 assignment
覆盖，之后才能构造 schema-v5 bundle。

创建 fresh publication directory 或任何 renderer artifact 之前，命令会预检全部 assignment
的 raw activation/completion、budget、capacity、interference、topology、runtime、sampling、
model、compile、nonce、launch-policy，以及可选 trainable/failure authority。Tagged generic、E1、
E2、confirmation-pilot、confirmation-final 与 stage-aggregate activation authority 都会从完整
path closure 现场重放。Final prefix 只使用 schema-v4/native completion authority；bare
completed-cell ID 会被拒绝。E3b/E5 family aggregate 仍保持独立；E5 还要求为其 264 个非 family
failure-injection cell 提供 deterministic auxiliary activation/completion 准确覆盖。Formal
reconstruction 还必须取得 live `READY` `BudgetPlan` 及其 path-bound materialization authority；
自洽的 planner 或 execution summary 不能自我授权。

Output path 必须为 absolute、resolved、尚不存在，并位于稳定且 owner-private 的 parent 之下。
全 assignment 预检通过后，命令创建 private `0700` directory，以 exclusive write 写入每个
schema-v5 bundle 与相邻 sidecar，最后写入 `dispatch-execution-bundle-manifest.json` 及其
sidecar 作为 commit marker。中断留下的 partial directory 会被保留，但因没有可消费 manifest
而不构成 publication；重试必须使用另一 fresh directory，不能通过删除 evidence 恢复。

```bash
lightcone-spec materialize-dispatch-execution-bundles \
  --request /absolute/path/to/dispatch-bundle-request.json \
  --output-directory /absolute/private/path/dispatch-bundles-attempt-0001
```

`execute-dispatch-wave` 绝不会把 planner JSON、
`IndustrialExecutionPlan.to_dict()` summary 或 raw bundle path 当作 launch authority。它只接受
一份必需的 `--materialization-manifest`。通过 release dispatch trust gate 后，loader 会重开
manifest、对应 request 与 dispatch plan、每个 schema-v5 member，以及完整 path-bound
construction graph；input byte 变化、member 被替换、assignment coverage 不完整或 bundle
位于 manifest directory 外都会被拒绝。之后 execution 才会重放共享 scheduler authority，
并构造所请求 wave 的 physical plan（及 resume receipt 实际引用的历史 plan）。Compile/download
和 unsupported serving assignment 仍明确 `BLOCKED`，每个可能真实 launch 的 plan 也仍必须
通过完整 public validation boundary。

每次命令只执行一个 `--wave-index`。Wave 0 不带 resume input；成功 prefix 的下一波必须传
上一轮 `--resume-receipt`，failed wave 则在同一 index resume。任何 sibling runner 启动前，
命令都会先向 private `RECEIPT.attempt-journal` durable append 一个 wave intent 与所有
per-assignment intent；每次 terminal 或 failure 返回后再 append finish、准确 retry identity
与累计 monotonic cost。因此 partial wave 会保留成功 sibling，只重跑失败 sibling。若 intent
没有 finish，成本无法确认，会以
`dispatch_attempt_intent_without_finish_cost_unresolved` 阻断。

Formal manifest-based dispatch 使用 schema-v2 attempt-journal manifest，并绑定准确的
materialization-manifest SHA-256。因此，即使 dispatch plan 没有变化，使用另一份 bundle
publication 重开同一个 journal 也会产生 identity mismatch。

`--receipt-output` 是 schema-v2 canonical immutable envelope，同时内嵌 receipt、structured
sidecar 与准确 journal manifest/head/event-count prefix。Resume 只把 caller receipt 当作
anchor：授权来自 raw journal 重放及每个成功 `AssignmentTerminalAuthority` 的现场重开。
若所有 finish 已落盘、但 coordinator 在 envelope 前崩溃，用相同 wave/output 重跑命令即可
恢复，不会再次启动 runner。`RECEIPT.sidecar.json` 只是 derived convenience copy，因此在
该 copy 写入前崩溃不会丢失 resume authority；parent directory 必须已经存在。

相对于已发布 envelope prefix，hash chain 能检测删除、替换、symlink 与 event 联合重哈希；
它不是外部签名。能够同时替换整个 unsigned journal 与全部 anchor 的 actor 超出这个纯软件
恢复威胁模型。Formal GPU claim 仍必须使用下述 release-owned trusted attester。

```bash
lightcone-spec execute-dispatch-wave \
  --materialization-manifest \
    /absolute/private/path/dispatch-bundles-attempt-0001/dispatch-execution-bundle-manifest.json \
  --wave-index 0 \
  --receipt-output artifacts/industrial/dispatch-wave-0.json
```

当前 source release 的 release-owned trust policy 没有 trusted hardware attester。即使
caller/test signer 的签名在密码学上有效，也不能解锁 formal execution；bare
`CapacityEnvelope` 同样不能授予 execution authority。因此，即使 request 在其他方面完整，
materialization 也会在全 assignment 的 release preflight 处返回 `BLOCKED`/42，且不会创建
publication directory。Formal execution 则会独立地在 entry trust gate 返回 `BLOCKED`/42，
发生在读取 materialization manifest、创建 receipt parent/evidence root、导入 serving client
或启动 process 之前。Bundle publication 只是 structural authority，不是 GPU attestation 或
execution permission。Fresh execution 会拒绝未被 journal 授权的已有 per-plan trace file；
resume 必须同时重放 raw append-only attempt chain 并现场重验 structured terminal binding。
Bare terminal digest 或 caller 联合重哈希的 schedule JSON 都不能跳过 work。若 coordinator
在 durable intent 后、durable finish 前崩溃，则保持 `BLOCKED`，绝不伪造 monotonic cost。

Dispatch 前要对同一个 activated set 估算。每个 budget field 都是显式值；report 会分别列出
optimistic、registered 与 quota-envelope scenario 的 wall、compute、reserved 与
whole-instance billed GPU time：

```bash
lightcone-spec estimate-industrial-budget \
  --registry artifacts/industrial/registry.json \
  --activation-plan artifacts/industrial/preflight-activation-manifest.json \
  --inventory artifacts/industrial/inventory.json \
  --interference-envelope artifacts/industrial/interference-envelope.json \
  --budget-plan artifacts/industrial/budget-plan.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --capacity-manifest artifacts/industrial/capacity-source-manifest.json \
  --capacity-verification-receipt artifacts/industrial/capacity-verification.json \
  --output artifacts/industrial/budget-report.json
```

Estimator 会执行同样的 raw-input rematerialization 与现场 capacity revalidation。Report
同时绑定 scheduler inventory SHA-256 与 interference-envelope SHA-256；若准确 scheduler
replay 不可执行，它仍保留真实派生的 GPU-hour diagnostics，记录具名 unresolved assumption
并返回 42；该 report 不能用作 launch authority。

Stage 的 activated cell 与 disposition durable 后再封存。`--inventory PATH` 是 completed
evidence 所用 physical GPU identity 与 topology 的 authority。每次重复的
`--locked-output` 只能使用 `NAME=PATH`：name 必须唯一，且 `PATH` 必须指向
content-bound artifact；SHA-256 literal 不能替代该 artifact。Runtime/split 同样必须是 bound
file，dependency 必须传 receipt file，而不是复制 hash string：

```bash
lightcone-spec seal-industrial-stage \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/inventory.json \
  --experiment preflight \
  --runtime-artifact artifacts/industrial/runtime.json \
  --split-artifact artifacts/industrial/preflight-split.json \
  --activation-plan artifacts/industrial/preflight-activation-manifest.json \
  --completed-cells artifacts/industrial/completed-cells.json \
  --locked-output runtime_envelope=artifacts/industrial/runtime-envelope.json \
  --output artifacts/industrial/receipts/preflight.json
```

封存 E2 时还强制要求额外 authority：`--e2-final-stage-manifest PATH`。它必须是 raw
`halving_3` manifest；command 会现场重跑 first-party raw reducer，并要求结果为
`FINAL_RECIPE`。随后，`dflash_recipe=PATH` artifact 必须精确绑定该 final candidate、
registry、runtime、split 与 final-stage reduction。用 digest 替代 `PATH`、caller-authored
summary 或绑定错误 candidate 的 recipe 都会被拒绝：

```bash
lightcone-spec seal-industrial-stage \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/inventory.json \
  --experiment E2 \
  --runtime-artifact artifacts/industrial/runtime.json \
  --split-artifact artifacts/industrial/e2-split.json \
  --completed-cells artifacts/industrial/e2-completed-cells.json \
  --e2-final-stage-manifest artifacts/industrial/e2-halving_3-manifest.json \
  --dependency-receipt artifacts/industrial/receipts/E1.json \
  --locked-output dflash_recipe=artifacts/industrial/e2-dflash-recipe.json \
  --output artifacts/industrial/receipts/E2.json
```

这些检查定义的是注册 trusted hardware attester 后的 claim-bearing 路径。当前 release 的
seal 仍会在签发 receipt 之前返回 `BLOCKED`；该 raw-authority contract 不会让 final
execution 或 performance claim 变为可达。

下游 stage 对其准确 declared dependency 重复 `--dependency-receipt`，随后把所有 completed
receipt 交给 planner：

```bash
lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/inventory.json \
  --interference-envelope artifacts/industrial/interference-envelope.json \
  --budget-plan artifacts/industrial/budget-plan.json \
  --budget-policy artifacts/industrial/budget-policy.json \
  --budget-load-binding artifacts/industrial/load-cell-000.json \
  --capacity-envelope artifacts/industrial/capacity-envelope.json \
  --receipt artifacts/industrial/receipts/preflight.json \
  --completed-cells artifacts/industrial/completed-cells.json \
  --activation-plan artifacts/industrial/next-stage-activation-manifest.json \
  --output artifacts/industrial/next-dispatch.json
```

Formal completed-cell row 会绑定 frozen physical assignment、准确 budget、terminal receipt、
强制 `BudgetObservationReceipt` 与三方 budget/terminal digest closure。缺失 phase timing、
compile/download N/A 理由或 GPU accounting 都不能替换为零。Resume 只接受完整 receipt，
绝不会只看目录存在。

Dispatch plan 是目标 protocol 数据，不证明 cell 可执行。Library industrial executor 会在
launch 前验证 provider state。固定 tree 已实现准确 native terminal begin/reset/finalize
hook，但没有配置 trusted hardware signer。Generic activation 只记录 canonical blocked
preflight disposition；它不会创造 compile runner、execution authority 或 performance claim。
Static/TTS/L0 在缺少 validated native capability 与 trusted signer 时仍被阻止。CLI 不会
静默 provision hardware，也不会启动 GPU。

## 身份与 Topology 链

最小 industrial identity chain 为：

```text
source + patched tree + model/data/trace locks
                         |
                         v
          registry logical slots + activation reducers
                         |
                         v
 inventory + interference + budgets -> frozen physical waves
                         |
                         v
       terminal + reset + budget-observation receipts
                         |
                         v
 family power + dependence-aware statistics + trusted attestation
```

TP2 与 sticky DP2 字段是目标 registry/CPU-coordinator vocabulary。当前 `RunConfig` 会在
model loading 前拒绝全部 TP2/DP2 value；CPU `gloo` contract 或 caller 自己填写 receipt 都
不能启用。未来实现必须准确绑定 rank、UUID、rendezvous、router、clock、process-group、
ownership 与 receipt identity。

当 command contract 要求时，generated JSON 使用相邻 `.sha256` sidecar。Loader 验证
canonical content，不只检查 filename。

## 核心 Tuning 与 Confirmation

Target-only 与 Static 渲染时不含 adaptation object 或 reserve。TTS 与 L0 的目标 declaration
从已注册 native layer scope 共享一个 Full/LoRA candidate。Rendering 只是 pure planning；
当前 release 只有 Target-only 可以进入 industrial execution。Static/TTS/L0 会在 endpoint
启动前因 trusted-terminal-attester preflight 失败。

Industrial E1/E2 command 由 reducer 控制。E1 消费 sealed E3a output，只激活一个 width/load
slice。E2 materialize stage zero，随后每个 successive-halving round 都要求 prior sealed
survivor receipt，同时保留 matched TTS/L0 pair 与 family floor。Confirmation planning 为
每个准确 family 激活四个 excluded pilot，在 confirmation data 可见前封存 `POWERED` 的
12--20 final block 或 `UNDERPOWERED`，并只 materialize sealed prefix。

Family power 不接受 bare score 或 hand-authored count：

```bash
lightcone-spec reduce-confirmation-family-power \
  --manifest artifacts/industrial/family-power-manifest.json \
  --output artifacts/industrial/family-power-plan.json
```

准确 `industrial_family_power_manifest` 绑定 registry artifact、pilot activation、hardware
envelope 与四个 pilot block。每个 pilot cell 都提供 terminal receipt、hardware receipt 与
budget observation。Reducer 保持 confirmation hidden，并拒绝 missing、extra、cross-family 或
unsafe evidence。

`validate-evidence-alias` 不会从 label 创造 scientific equivalence，也不会 round-trip
caller-authored receipt。它会将 bound raw manifest 对照当前 registry、hardware envelope、
GPU inventory 及其 source receipt、execution/load/config/split/sampling/model/budget artifact、
source terminal evidence 与 native terminal artifact 完整重放：

```bash
lightcone-spec validate-evidence-alias \
  --manifest artifacts/industrial/raw-alias-manifest.json \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/gpu-inventory.json \
  --hardware-envelope artifacts/industrial/hardware-envelope.json \
  --output artifacts/industrial/alias-reduction.json

lightcone-spec build-evidence-dependence-map \
  --direct-map artifacts/industrial/direct-dependence-map.json \
  --alias-reduction artifacts/industrial/alias-reduction.json \
  --output artifacts/industrial/evidence-dependence-map.json
```

Dependence map 只接受 first-party reduction artifact，让 shared control 在 covariance/
bootstrap 中保持一个 observation。Static 是 backend-specific；TTS/L0 永远不能 alias。
schema-v3 industrial analysis manifest 会列出 raw alias manifests；formal reducer 会逐一重放，
并拒绝与重放结果不同的 serialized map。旧 alias receipt、caller-authored reduction summary
以及旧 `--alias` flags 都会被拒绝。

每个 completed slice 都以 content-bound evidence 结束。只有 manifest/config/method/block/
data/trace 与全部 shard digest 验证通过，resume 才跳过它。中断 Parquet WAL segment 没有
terminal receipt，会被排除。Profiler output 带 `headline_evidence_forbidden=true`，不能合并
进 measured timing table。

## OnlineSPEC 工作流

OnlineSPEC 复用 model lock、controlled data 与 paired Static reference，但绝不复用核心
tuning/confirmation row。`verify-onlinespec-source` 检查外部 checkout 的准确 commit/tree、
clean status、audited key-file hash 与 license inventory；checkout 保持在仓库外。

Successive halving 在 OGD、optimistic OGD 与 ensemble 内独立运行。Selection 还绑定核心
reference load，但不能改变该 load 或影响核心 gate。Anchor selection 标注
`optimized_grid_claim=false`。OnlineSPEC 保持 TP1/DP1 且只能用于诊断；其 attestation
命令只能输出 non-authority decision。

## 状态、Resume 与 Exit Code

`analyze-preliminary-speed-study` 始终写入 `PRELIMINARY_DIAGNOSTIC_ONLY` 并退出
42。即使未来配置 trusted attester，它也会拒绝全部 attestation。
`attest-preliminary-speed-study` 与 `attest-onlinespec-study` 同样只输出确定性的
non-authority decision 并退出 42。Identity、schema、I/O、receipt 或 runtime 错误是普通
nonzero failure，不是科学结果。
当前 release 没有 trusted hardware-attester identity，因此 supplied legacy attestation
会被拒绝；即使 diagnostic JSON 内部一致，`analyze-industrial` 也不能生成 `MEASURED`，
并会退出 42。

Industrial registry 可能把目标 cell 声明为 `UNMEASURED`，但 declaration status 不等于
executable readiness。Native hook 已存在；由于 trusted signer 缺失，executor preflight
会把 Static/TTS/L0 解析为 `BLOCKED`。全部 TP2/DP2 与
DSpark/EAGLE/EAGLE3/NEXTN adaptive cell 同样受当前 release 的其他 gate 阻止。历史 v2
artifact 仅用于 regression，不能作为 schema-v3 stage receipt。

### Target-output Reference 诊断

针对一个单独锁定的 Target-only server，`run-preliminary-target-reference` 记录每个 prompt 的 token
数量，以及完整有序 output-token-ID 数组带格式标识的 SHA-256；它绝不会用 decoded-text
digest 替代。Legacy collector 要求 `--target-reference`，任何 method/block 的轨迹不匹配
reference 都会被拒绝；仅 speculative method 彼此一致不能证明 exactness。无论 hardware
是否可用，该结果始终是 `PRELIMINARY_DIAGNOSTIC_ONLY`。

## Credential 与 Output Root

Artifact、model root、cache、trace、provider state、profile、selection、attestation 与
handoff file 必须位于 ignored external root。通过临时 `HF_TOKEN` 环境变量或其他安全渠道
传递模型权限。不要把 token、password、provider API key、private prompt、instance address
或 machine-specific path 放入 argument、manifest、log、文档或 Git。
