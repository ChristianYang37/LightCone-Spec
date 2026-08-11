# CLI

[English](../en/cli.md) · [首页](../../README_zh-CN.md)

## 命令 Surface

准确参数以 `lightcone-spec --help` 与 `lightcone-spec COMMAND --help` 为准。Schema-v3 与
industrial 命令包括：

| 命令 | 用途 |
|---|---|
| `doctor` | 只读 host、Python、CUDA 与 source identity 报告 |
| `validate-config` | 验证一份严格 schema-v3 run config 与 sidecar |
| `build-industrial-registry` | 绑定两个稳定 logical rank slot 并生成不可变实验 DAG |
| `reduce-e1-activation` | 从 sealed E3a evidence 派生唯一 130-cell E1 slice |
| `reduce-e2-activation` | 从 E1 Pareto artifact 或 prior survivor materialize 一个 E2 round |
| `reduce-e2-successive-halving` | 把准确 E2 stage evidence 归约为 sealed survivor receipt |
| `materialize-confirmation-pilots` | 为一个 confirmation family 激活恰好四个 excluded pilot |
| `reduce-confirmation-family-power` | 把一个 family 的准确四个 pilot block 归约为 sealed power plan |
| `materialize-confirmation-prefix` | 只激活 powered family 已封存的 12--20-block final prefix |
| `validate-evidence-alias` | 严格加载并重写 content-bound evidence alias receipt |
| `build-evidence-dependence-map` | 保留合法 alias 带来的 shared-observation dependence |
| `estimate-industrial-budget` | 为指定 inventory 与 activation bundle 归约准确 per-cell budget |
| `plan-industrial-dispatch` | 冻结确定性、topology-aware GPU-pool wave 与 physical assignment |
| `seal-industrial-stage` | 绑定 activated completion、disposition、budget、runtime、split、dependency 与 locked output |
| `analyze-industrial` | 验证 schema-v3 terminal、budget、family-power 与 hardware evidence |
| `build-speed-study` | 生成较小的核心源协议 |
| `lock-models` | 把 model ID 解析为不可变 revision |
| `prepare-models` | 下载或 offline 验证 locked snapshot |
| `list-tuning-candidates` | 写入已注册 Full/LoRA tuning grid |
| `render-target-only-runtime` | 渲染关闭 speculation 的 Target-only endpoint |
| `render-static-load-runtime` | 渲染零 adaptation 分配的 Static endpoint |
| `render-tuning-runtime` | 渲染 matched TTS/L0 tuning endpoint |
| `run-controlled-slice` | 测量一个已注册 controlled slice |
| `collect-static-load-screen` | 验证 Static load coverage 并选择 reference load |
| `advance-tuning-stage` | 验证 halving stage 并封存 survivor |
| `select-speed-config` | 应用 tuning-only 注册 selection rule |
| `select-anchor-config` | 锁定 terminal registered anchor，但不声明 optimum |
| `render-runtime` | 生成 matched sequential 核心 config 与 launch argv |
| `build-confirmation-queue` | 生成 clean-server confirmation job |
| `run-confirmation` | 执行一个 method/block confirmation slice |
| `run-target-reference` | 捕获锁定的 Target-only greedy token-ID 轨迹诊断 |
| `collect-speed-study` | 从 completed receipt-bound evidence 派生 table |
| `render-replication-runtime` | 渲染 natural-task 或 profiler-only slice |
| `run-natural-slice` | 运行一个 locked natural-task slice |
| `build-profiler-plan` | 生成隔离 profile plan，并禁止 headline evidence |
| `attest-speed-study` | 绑定 GPU、runtime、model、selection、trace 与 evidence identity |
| `analyze-speed-study` | 评估已注册 paired gate |

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
| `attest-onlinespec-study` | 绑定 comparison evidence 与 source identity |
| `analyze-onlinespec-study` | 生成 learner-versus-Static 诊断区间 |

不存在 generic method override，也不存在把无 receipt 目录转换为 completed evidence 的命令。

## Industrial Registry 工作流

Registry 只包含 scientific identity 与两个 logical rank slot，不包含 host-specific GPU UUID：

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

Planning 会 fail closed：它需要 inventory、准确 per-cell budget sequence 与 interference
envelope。Template stage 还需要 reducer-generated stage/family activation；手写 cell list
不会被接受。

```bash
lightcone-spec plan-industrial-dispatch \
  --registry artifacts/industrial/registry.json \
  --inventory artifacts/industrial/inventory.json \
  --interference-envelope artifacts/industrial/interference-envelope.json \
  --budget-plan artifacts/industrial/budgets.json \
  --output artifacts/industrial/preflight-dispatch.json
```

Frozen plan 会绑定准确 inventory/interference envelope、每个 cell 的 budget digest、physical
UUID/rank/port/topology assignment、wave membership 与 scheduler identity。Scheduler 接受任意
same-host inventory，并对 1/2/4/8/16 GPU 有 regression test。一个 wave 绝不超过已校准的
co-run class，因此八块空闲 GPU 不表示可以执行八路 headline concurrency。Envelope 通过
structural loading 不能证明 calibration；formal timing 仍要求 raw isolated/co-run
evidence 与 trusted hardware binding。目前没有 `execute-dispatch-wave` CLI command；
structured library executor 要求 first-party、content-bound launch adapter，不会从 planner
JSON 推断 launch 配置。

Dispatch 前要对同一个 activated set 估算。每个 budget field 都是显式值；report 会分别列出
optimistic、registered 与 quota-envelope scenario 的 wall、compute、reserved 与
whole-instance billed GPU time：

```bash
lightcone-spec estimate-industrial-budget \
  --registry artifacts/industrial/registry.json \
  --activation-plan artifacts/industrial/activation.json \
  --inventory artifacts/industrial/inventory.json \
  --budgets artifacts/industrial/budgets.json \
  --output artifacts/industrial/budget-report.json
```

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
  --budget-plan artifacts/industrial/budgets.json \
  --receipt artifacts/industrial/receipts/preflight.json \
  --completed-cells artifacts/industrial/completed-cells.json \
  --activation-plan artifacts/industrial/activation.json \
  --output artifacts/industrial/next-dispatch.json
```

Formal completed-cell row 会绑定 frozen physical assignment、准确 budget、terminal receipt、
强制 `BudgetObservationReceipt` 与三方 budget/terminal digest closure。缺失 phase timing、
compile/download N/A 理由或 GPU accounting 都不能替换为零。Resume 只接受完整 receipt，
绝不会只看目录存在。

Dispatch plan 是目标 protocol 数据，不证明 cell 可执行。Library industrial executor 会在
launch 前验证 provider state。固定 tree 已实现准确 native terminal begin/reset/finalize
hook，但没有配置 trusted hardware signer。因此 release preflight 只运行 TP1/DP1
Target-only，并在 process、filesystem、root 或 network mutation 前阻止 Static/TTS/L0。
CLI 不会静默 provision hardware。

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

`validate-evidence-alias` 不会从 label 创造 scientific equivalence；它只接受完全
content-bound、byte-equivalent 的 receipt。Dependence map 让 shared control 在 covariance/
bootstrap 中保持一个 observation。Static 是 backend-specific；TTS/L0 永远不能 alias。本
release 的 formal reducer 会拒绝 non-singleton dependence unit，除非 alias 是从 execution
plan 与 terminal evidence 重新计算得出；self-described alias 可以通过 structural validation，
但不能进入 claim。

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
`optimized_grid_claim=false`。OnlineSPEC 保持 TP1/DP1，并使用自己的 GPU attestation。

## 状态、Resume 与 Exit Code

缺少 attestation 时，`analyze-speed-study` 可输出诊断，但状态为 `UNMEASURED` 且退出 42。
提供有效 content-bound GPU evidence 后，只有完整注册 pass 才退出零；有效 evidence 未满足
任一标准时为 `BLOCKED`，同样退出 42。Identity、schema、I/O、receipt 或 runtime 错误是
普通 nonzero failure，不是科学结果。
当前 release 没有 trusted hardware-attester identity，因此 supplied legacy attestation
会被拒绝；即使 diagnostic JSON 内部一致，`analyze-industrial` 也不能生成 `MEASURED`，
并会退出 42。

Industrial registry 可能把目标 cell 声明为 `UNMEASURED`，但 declaration status 不等于
executable readiness。Native hook 已存在；由于 trusted signer 缺失，executor preflight
会把 Static/TTS/L0 解析为 `BLOCKED`。全部 TP2/DP2 与
DSpark/EAGLE/EAGLE3/NEXTN adaptive cell 同样受当前 release 的其他 gate 阻止。历史 v2
artifact 仅用于 regression，不能作为 schema-v3 stage receipt。

### Target-output Reference 诊断

针对一个单独锁定的 Target-only server，`run-target-reference` 记录每个 prompt 的 token
数量，以及完整有序 output-token-ID 数组带格式标识的 SHA-256；它绝不会用 decoded-text
digest 替代。Legacy collector 要求 `--target-reference`，任何 method/block 的轨迹不匹配
reference 都会被拒绝；仅 speculative method 彼此一致不能证明 exactness。在本 release
中，该结果仍只是 `UNMEASURED` 诊断，因为 Static/TTS/L0 execution 与 trusted hardware
attestation 都处于 blocked 状态。

## Credential 与 Output Root

Artifact、model root、cache、trace、provider state、profile、selection、attestation 与
handoff file 必须位于 ignored external root。通过临时 `HF_TOKEN` 环境变量或其他安全渠道
传递模型权限。不要把 token、password、provider API key、private prompt、instance address
或 machine-specific path 放入 argument、manifest、log、文档或 Git。
